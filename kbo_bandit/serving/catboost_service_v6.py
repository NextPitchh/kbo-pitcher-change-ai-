"""Time-bounded pre-pitch feature builder and independent v6 model serving."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from catboost import CatBoostClassifier
from .catboost_service_schema import ServicePaState

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "outputs/serving/catboost_service_v6"
CLASSES = ("1B", "2B", "3B", "BB", "FO", "GDP", "GO", "HR", "SO")
NUM = ("inning", "score_diff_attacker", "out_count", "total_pitch_count_before", "inning_pitch_count_before")
HIST = ("batter_prior_ab", "batter_prior_hits", "matchup_ab", "matchup_hits")
CAT = ("half", "base_state", "pitcher_id", "batter_id")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


class ServicePaModel:
    def __init__(self, artifact: Path = DEFAULT_ARTIFACT):
        self.artifact = Path(artifact)
        meta = json.loads((self.artifact / "selection.json").read_text(encoding="utf8"))
        expected = NUM + (HIST if meta["selected"] == "B" else ()) + CAT
        if meta["selected"] not in ("A", "B") or tuple(meta["features"]) != expected or tuple(meta["classes"]) != CLASSES:
            raise ValueError("Unsupported service model schema")
        if digest(self.artifact / "model.cbm") != meta["model_sha256"]:
            raise ValueError("Service model hash mismatch")
        model = CatBoostClassifier()
        model.load_model(str(self.artifact / "model.cbm"))
        if tuple(model.feature_names_) != expected or tuple(model.classes_) != CLASSES:
            raise ValueError("Model feature or class order mismatch")
        if model.get_cat_feature_indices() != [expected.index(c) for c in CAT]:
            raise ValueError("Categorical feature order mismatch")
        self.history = None
        if meta["selected"] == "B":
            path = self.artifact / "train_history_2023.json"
            if digest(path) != meta["train_history_sha256"]: raise ValueError("History hash mismatch")
            self.history = json.loads(path.read_text(encoding="utf8"))
            if self.history.get("as_of_exclusive") != "2024-01-01": raise ValueError("Invalid history cutoff")
        self.model, self.metadata = model, meta

    def build_features(self, state: dict) -> dict:
        """Validate the caller's pre-pitch state; never read future PA outcomes."""
        state = ServicePaState(**state).model_dump()
        pitcher, batter = str(state["pitcher_id"]), str(state["batter_id"])
        score = state["away_score"] - state["home_score"] if state["half"] == "top" else state["home_score"] - state["away_score"]
        features = {"inning": state["inning"], "score_diff_attacker": score, "out_count": state["out_count"],
                    "total_pitch_count_before": state["total_pitch_count_before"],
                    "inning_pitch_count_before": state["inning_pitch_count_before"], "half":state["half"],
                    "base_state":state["base_state"], "pitcher_id":pitcher, "batter_id":batter}
        if self.history is not None:
            ba, bh = self.history["batter"].get(batter, (0, 0))
            ma, mh = self.history["matchup"].get(f"{pitcher}:{batter}", (0, 0))
            features.update({"batter_prior_ab":ba, "batter_prior_hits":bh, "matchup_ab":ma, "matchup_hits":mh})
        return {key: features[key] for key in self.metadata["features"]}

    def predict(self, state: dict) -> dict:
        features = self.build_features(state)
        pred = self.model.predict_proba([list(features.values())], thread_count=1)[0]
        if len(pred) != len(CLASSES) or any(not math.isfinite(float(x)) or not 0 <= x <= 1 for x in pred):
            raise ValueError("Invalid service model probability")
        if not math.isclose(float(sum(pred)), 1, rel_tol=0, abs_tol=1e-12): raise ValueError("Probability sum")
        raw = {key: float(value) for key, value in zip(CLASSES, pred)}
        web = raw.copy()
        if state["out_count"] == 2 or "1" not in state["base_state"]:
            web["GO"] += web["GDP"]
            web["GDP"] = 0.0
        if not math.isclose(sum(web.values()), 1.0, rel_tol=0, abs_tol=1e-12): raise ValueError("Web probability sum")
        return {"model_version":self.metadata["model_version"], "feature_schema_version":self.metadata["feature_schema_version"],
                "mapping_version":"direct-web9-spring-gdp-v1", "probability_provenance":"direct_web9_catboost_spring_gdp_policy",
                "features":features, "native_probabilities":raw, "web_probabilities":web,
                "history_cutoff":"2024-01-01" if self.history is not None else None,
                "input_provenance":"caller_supplied_pre_pitch_state_not_db_verified"}
