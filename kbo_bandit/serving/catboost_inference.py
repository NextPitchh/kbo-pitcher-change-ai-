"""Independent inference for the available v5 PA classifier; no training data needed."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "outputs/serving/catboost_final"
NUMERIC = ("inning", "score_diff_attacker", "out_count", "total_pitch_count", "inning_pitch_count", "pitcher_vs_batter_avg", "batter_recent_avg")
CATEGORICAL = ("half", "base_state", "batter_hit_type", "pitcher_id", "batter_id")
FEATURES = NUMERIC + CATEGORICAL
CLASSES = ("1B", "2B", "3B", "BB", "GDP", "HR", "OUT", "SF", "SO")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_row(features: dict) -> list:
    if set(features) != set(FEATURES):
        raise ValueError("Exactly the 12 documented features are required")
    row = []
    for name in NUMERIC:
        value = features[name]
        if value is None:
            value = 0.25 if name in ("pitcher_vs_batter_avg", "batter_recent_avg") else 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number or null")
        row.append(float(value))
    for name in CATEGORICAL:
        value = features[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            raise ValueError(f"{name} must be a string, integer, or null")
        row.append("__MISSING__" if value is None else str(value))
    return row


class FrozenPaModel:
    def __init__(self, artifact: Path = DEFAULT_ARTIFACT):
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        model_path = artifact / "model.cbm"
        if sha256(model_path) != manifest["model_sha256"]:
            raise ValueError("CatBoost model hash mismatch")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        if (tuple(model.feature_names_) != FEATURES or tuple(model.classes_) != CLASSES
                or model.get_cat_feature_indices() != list(range(7, 12))
                or manifest["features"] != list(FEATURES) or manifest["classes"] != list(CLASSES)):
            raise ValueError("CatBoost model schema mismatch")
        self.model, self.manifest = model, manifest

    def predict(self, features: dict) -> dict[str, float]:
        probabilities = self.model.predict_proba([feature_row(features)])[0]
        if (len(probabilities) != len(CLASSES) or any(not math.isfinite(float(p)) or p < 0 or p > 1 for p in probabilities)
                or not math.isclose(float(sum(probabilities)), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Invalid model probabilities")
        return {name: float(p) for name, p in zip(CLASSES, probabilities)}
