"""NumPy-only, float64 inference for the corrected final 88-dimensional B1.

Historical nulls must be explicit ``None``; non-finite numbers are errors.
``prior_n_apps=None`` becomes zero before log1p, as in the original pipeline.
The other six historical nulls use fitted TRAIN medians. No training imports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "b1-li-88-v1"
MODEL_VERSION = "b1-optionb-final-v1"
SITUATION_FEATURES = (
    "inning", "out_count", "score_diff_defense", "is_base1", "is_base2",
    "is_base3", "is_home_defense", "bullpen_available_ratio", "n_pitchers_used",
)
PITCHER_FEATURES = (
    "prior_n_apps_log", "prior_pitcher_reward_mean", "prior_wpa_std",
    "prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_avg_pa_per_app",
)
BASE_FEATURES = SITUATION_FEATURES + PITCHER_FEATURES
HISTORICAL_INPUTS = ("prior_n_apps",) + PITCHER_FEATURES[1:]
GAME_INPUTS = SITUATION_FEATURES + ("leverage_index",)
FEATURE_ORDER = (
    ("intercept",) + BASE_FEATURES
    + tuple(f"{s}__X__{p}" for s in SITUATION_FEATURES for p in PITCHER_FEATURES)
    + ("leverage_index",)
    + tuple(f"leverage_index__X__{p}" for p in PITCHER_FEATURES)
)
STATE_SHAPES = {
    "medians": (16,), "base_mean": (16,), "base_scale": (16,), "base_var": (16,),
    "interaction_mean": (63,), "interaction_scale": (63,), "interaction_var": (63,),
    "li_mean": (), "li_std": (),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema() -> dict:
    return {
        "feature_schema_version": SCHEMA_VERSION,
        "feature_dim": 88,
        "dtype": "float64",
        "feature_order": list(FEATURE_ORDER),
        "game_inputs": list(GAME_INPUTS),
        "historical_inputs": list(HISTORICAL_INPUTS),
        "transforms": [
            "prior_n_apps: log1p(0 if null else raw_count)",
            "six historical nulls: fitted TRAIN median",
            "16 bases: (imputed_value - base_mean) / base_scale",
            "63 situation-major products of standardized bases, then interaction scaling",
            "LI: (already_train_normalized_LI - li_mean) / li_std",
            "7 LI products: standardized_LI * standardized_pitcher; no second scaling",
            "prepend intercept; no feature clipping; alpha=0",
        ],
        "ranking": "descending reward, then lexicographic str(pitcher_id); preserve input ID",
    }


def _number(value, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number" + (" or null" if nullable else ""))
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be represented as float64") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite (NaN/Infinity are not missing values)")
    return number


def _required(mapping: Mapping, names: tuple, label: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(set(names) - mapping.keys())
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


@dataclass(frozen=True)
class B1Model:
    theta: np.ndarray
    state: Mapping[str, np.ndarray]
    metadata: Mapping

    def transform_candidates(self, game_state: Mapping, candidates: Sequence[Mapping]) -> np.ndarray:
        """Build the original ordered 88-vector, with strict raw-input validation."""
        _required(game_state, GAME_INPUTS, "game_state")
        game = {name: _number(game_state[name], name) for name in GAME_INPUTS}
        for name in ("is_base1", "is_base2", "is_base3", "is_home_defense"):
            if game[name] not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        for name, minimum in (("inning", 1), ("out_count", 0), ("n_pitchers_used", 0)):
            if game[name] < minimum or not game[name].is_integer():
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if game["out_count"] > 2:
            raise ValueError("out_count must be 0, 1 or 2")
        if not 0 <= game["bullpen_available_ratio"] <= 1 or game["leverage_index"] < 0:
            raise ValueError("invalid bullpen_available_ratio or leverage_index")
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValueError("candidates must be a list or sequence")
        rows, seen = [], set()
        for candidate in candidates:
            _required(candidate, ("pitcher_id",) + HISTORICAL_INPUTS, "candidate")
            pid = candidate["pitcher_id"]
            if isinstance(pid, bool) or not isinstance(pid, (str, Integral)):
                raise ValueError("pitcher_id must be a nonempty string or integer (float IDs rejected)")
            if isinstance(pid, str) and (not pid or pid != pid.strip()):
                raise ValueError("pitcher_id must be nonempty with no surrounding whitespace")
            key = str(pid)
            if key in seen:
                raise ValueError(f"duplicate pitcher_id: {key}")
            seen.add(key)
            raw_count = _number(candidate["prior_n_apps"], "prior_n_apps", nullable=True)
            raw_count = 0.0 if raw_count is None else raw_count
            if raw_count < 0 or not raw_count.is_integer():
                raise ValueError("prior_n_apps must be a nonnegative integer or null")
            historical = [float(np.log1p(raw_count))]
            for index, name in enumerate(PITCHER_FEATURES[1:], start=10):
                value = _number(candidate[name], name, nullable=True)
                historical.append(float(self.state["medians"][index]) if value is None else value)
            rows.append([game[name] for name in SITUATION_FEATURES] + historical)
        if not rows:
            return np.empty((0, 88), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                base = (np.asarray(rows, dtype=np.float64) - self.state["base_mean"]) / self.state["base_scale"]
                products = (base[:, :9, None] * base[:, None, 9:]).reshape(len(rows), 63)
                products = (products - self.state["interaction_mean"]) / self.state["interaction_scale"]
                li = (game["leverage_index"] - self.state["li_mean"]) / self.state["li_std"]
                features = np.column_stack((np.ones(len(rows)), base, products,
                                            np.full(len(rows), li), li * base[:, 9:]))
            except FloatingPointError as exc:
                raise ValueError("non-finite feature calculation") from exc
        if not np.isfinite(features).all():
            raise ValueError("non-finite features")
        return features

    def score_candidates(self, game_state: Mapping, candidates: Sequence[Mapping]) -> list[dict]:
        """Return ranked pitcher_id / predicted_reward / model_rank; [] for no candidates."""
        features = self.transform_candidates(game_state, candidates)
        with np.errstate(over="raise", invalid="raise"):
            try:
                # Row-wise reduction keeps each candidate bitwise independent of batch ordering/size.
                scores = np.sum(features * self.theta, axis=1)
            except FloatingPointError as exc:
                raise ValueError("non-finite predicted reward") from exc
        if not np.isfinite(scores).all():
            raise ValueError("non-finite predicted reward")
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), str(candidates[i]["pitcher_id"])))
        return [
            {"pitcher_id": candidates[i]["pitcher_id"], "predicted_reward": float(scores[i]), "model_rank": rank}
            for rank, i in enumerate(order, start=1)
        ]


def load_model(artifact_dir: str | Path) -> B1Model:
    """Load validated JSON/NPY/NPZ only; never pickle, fit, or read source datasets."""
    directory = Path(artifact_dir)
    if not (directory / "metadata.json").exists() and (directory / "lineage_audit.json").exists():
        raise ValueError("B1 freeze blocked by lineage/as-of audit; no deployable artifact was exported")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in (("feature_dim", 88), ("alpha", 0), ("ridge_lambda", 1.0),
                          ("feature_schema_version", SCHEMA_VERSION), ("model_version", MODEL_VERSION),
                          ("feature_order", list(FEATURE_ORDER))):
        if metadata.get(key) != expected:
            raise ValueError(f"incompatible metadata: {key}")
    hashes = metadata.get("artifact_hashes", {})
    required = {"theta.npy", "preprocessing.npz", "feature_schema.json"}
    if not required.issubset(hashes):
        raise ValueError("required artifact hashes missing")
    for name in required:
        if sha256(directory / name) != hashes[name]:
            raise ValueError(f"artifact integrity failure: {name}")
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    if schema != feature_schema():
        raise ValueError("feature schema/order differs from the supported builder")
    theta = np.load(directory / "theta.npy", allow_pickle=False)
    if theta.shape != (88,) or theta.dtype != np.float64 or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite float64 (88,) vector")
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(STATE_SHAPES):
            raise ValueError("unexpected preprocessing state keys")
        state = {key: archive[key].copy() for key in archive.files}
    for key, shape in STATE_SHAPES.items():
        value = state[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"invalid preprocessing shape/dtype/value: {key}")
        if (key.endswith("scale") or key == "li_std") and np.any(value <= 0):
            raise ValueError(f"nonpositive scale: {key}")
        if key.endswith("var") and np.any(value < 0):
            raise ValueError(f"negative variance: {key}")
        value.setflags(write=False)
    theta.setflags(write=False)
    return B1Model(theta=theta, state=state, metadata=metadata)
