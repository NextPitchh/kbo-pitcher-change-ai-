"""B1 v3 추천 API. 경기 상태 및 후보의 시점 안전 이력은 호출자가 제공합니다."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictInt, StrictStr

from .b1_chronology_v3 import load_model
from .catboost_schema import PaRequest
from .catboost_service_schema import ServicePaRequest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "outputs/bandit/serving/b1_final_serving_v3"
logger = logging.getLogger(__name__)
_pa_lock = threading.Lock()
_pa_adapter_lock = threading.Lock()
_service_pa_lock = threading.Lock()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GameState(StrictModel):
    inning: FiniteFloat
    out_count: FiniteFloat
    score_diff_defense: FiniteFloat
    is_base1: FiniteFloat
    is_base2: FiniteFloat
    is_base3: FiniteFloat
    is_home_defense: FiniteFloat
    bullpen_available_ratio: FiniteFloat
    n_pitchers_used: FiniteFloat
    # 이미 TRAIN league baseline으로 정규화된 LI. 모델 내부에서 TRAIN decision z-score 적용.
    leverage_index: FiniteFloat


class Candidate(StrictModel):
    pitcher_id: StrictStr | StrictInt
    prior_n_apps: FiniteFloat | None
    prior_pitcher_reward_mean: FiniteFloat | None
    prior_wpa_std: FiniteFloat | None
    prior_bb_rate: FiniteFloat | None
    prior_so_rate: FiniteFloat | None
    prior_hr_rate: FiniteFloat | None
    prior_avg_pa_per_app: FiniteFloat | None


class RecommendationRequest(StrictModel):
    game_state: GameState
    candidates: list[Candidate]


class RankedPitcher(BaseModel):
    pitcher_id: StrictStr | StrictInt
    predicted_reward: FiniteFloat
    model_rank: int


class RecommendationResponse(BaseModel):
    model_version: str
    rankings: list[RankedPitcher]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.load_error = None
    app.state.pa_model = None
    app.state.pa_adapter = None
    app.state.service_pa_model = None
    artifact = Path(os.environ.get("KBO_B1_ARTIFACT_DIR", str(DEFAULT_ARTIFACT)))
    try:
        app.state.model = load_model(artifact)
    except Exception as exc:
        app.state.load_error = type(exc).__name__
        logger.exception("B1 모델 로드 실패")
    yield
    app.state.model = None
    app.state.pa_model = None
    app.state.pa_adapter = None
    app.state.service_pa_model = None


app = FastAPI(title="K-Moneyball B1", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    # 유효하지 않은 NaN/Infinity 원본을 JSON 오류 응답에 다시 직렬화하지 않는다.
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request, response: Response):
    model = request.app.state.model
    if model is None:
        response.status_code = 503
        return {"status": "unavailable", "model_loaded": False, "model_version": None}
    return {"status": "ok", "model_loaded": True, "model_version": model.metadata["model_version"]}


@app.post("/v1/bandit/recommend", response_model=RecommendationResponse)
def recommend(body: RecommendationRequest, request: Request, response: Response):
    start = time.perf_counter()
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        rows = model.score_candidates(body.game_state.model_dump(), [c.model_dump() for c in body.candidates])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("B1 추천 계산 실패")
        raise HTTPException(status_code=500, detail="Recommendation failed") from exc
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.3f}"
    return {"model_version": model.metadata["model_version"], "rankings": rows}


def get_pa_model(request: Request):
    if request.app.state.pa_model is None:
        with _pa_lock:
            if request.app.state.pa_model is None:
                try:
                    from .catboost_inference import DEFAULT_ARTIFACT, FrozenPaModel
                    artifact = Path(os.environ.get("KBO_CATBOOST_ARTIFACT_DIR", str(DEFAULT_ARTIFACT)))
                    request.app.state.pa_model = FrozenPaModel(artifact)
                except Exception as exc:
                    logger.exception("CatBoost 모델 로드 실패")
                    raise HTTPException(status_code=503, detail="PA model is not available") from exc
    return request.app.state.pa_model


@app.post("/v1/pa/predict")
def predict_pa(body: PaRequest, request: Request, response: Response):
    start = time.perf_counter()
    model = get_pa_model(request)
    try:
        probabilities = model.predict(body.features.model_dump())
    except Exception as exc:
        logger.exception("CatBoost 추론 실패")
        raise HTTPException(status_code=500, detail="PA prediction failed") from exc
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.3f}"
    return {"model_version": model.manifest["model_version"], "probabilities": probabilities,
            "class_contract": "native_v5_OUT_and_SF_not_GO_or_FO"}


@app.post("/v1/pa/predict-web")
def predict_pa_web(body: PaRequest, request: Request, response: Response):
    """Return native probabilities and a separately labelled empirical projection.

    The caller is responsible for validating feature provenance and time safety.
    """
    start = time.perf_counter()
    model = get_pa_model(request)
    if request.app.state.pa_adapter is None:
        with _pa_adapter_lock:
            if request.app.state.pa_adapter is None:
                try:
                    from .catboost_web_adapter import DEFAULT_MAPPING, WebOutcomeAdapter
                    path = Path(os.environ.get("KBO_CATBOOST_WEB_MAPPING", str(DEFAULT_MAPPING)))
                    adapter = WebOutcomeAdapter(path)
                    if adapter.metadata["model_version"] != model.manifest["model_version"]:
                        raise ValueError("Mapping is for a different model")
                    request.app.state.pa_adapter = adapter
                except Exception as exc:
                    logger.exception("CatBoost 웹 매핑 로드 실패")
                    raise HTTPException(status_code=503, detail="PA web mapping is not available") from exc
    try:
        native = model.predict(body.features.model_dump())
        adapter = request.app.state.pa_adapter
        web = adapter.convert(native, out_count=body.features.out_count, base_state=body.features.base_state)
    except Exception as exc:
        logger.exception("CatBoost 웹 매핑 실패")
        raise HTTPException(status_code=500, detail="PA web prediction failed") from exc
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.3f}"
    return {"model_version": model.manifest["model_version"], "mapping_version": adapter.metadata["version"],
            "native_probabilities": native, "web_probabilities": web,
            "probability_provenance": "empirical_OUT_split_and_Spring_transition_policy",
            "feature_time_safety": "not_validated_by_api"}


@app.post("/v1/pa/predict-service")
def predict_pa_service(body: ServicePaRequest, request: Request, response: Response):
    start = time.perf_counter()
    if request.app.state.service_pa_model is None:
        with _service_pa_lock:
            if request.app.state.service_pa_model is None:
                try:
                    from .catboost_service_v6 import DEFAULT_ARTIFACT, ServicePaModel
                    artifact = Path(os.environ.get("KBO_CATBOOST_SERVICE_ARTIFACT_DIR", str(DEFAULT_ARTIFACT)))
                    request.app.state.service_pa_model = ServicePaModel(artifact)
                except Exception as exc:
                    logger.exception("서비스 PA 모델 로드 실패")
                    raise HTTPException(status_code=503, detail="Service PA model is not available") from exc
    try:
        result = request.app.state.service_pa_model.predict(body.state.model_dump())
    except Exception as exc:
        logger.exception("서비스 PA 추론 실패")
        raise HTTPException(status_code=500, detail="Service PA prediction failed") from exc
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-start)*1000:.3f}"
    return result
