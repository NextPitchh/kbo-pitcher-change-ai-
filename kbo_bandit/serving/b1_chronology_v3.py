"""정정 데이터로 재학습한 B1 v3 전용 로더. 학습 코드나 parquet를 읽지 않습니다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .b1_inference import B1Model, FEATURE_ORDER, SCHEMA_VERSION, STATE_SHAPES, feature_schema, sha256

MODEL_VERSION = "b1-optionb-chronology67360-v3"


def load_model(artifact_dir: str | Path) -> B1Model:
    directory = Path(artifact_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected = {"feature_dim": 88, "alpha": 0, "ridge_lambda": 1.0,
                "feature_schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
                "feature_order": list(FEATURE_ORDER), "data_validation_pass": True}
    for key, value in expected.items():
        if metadata.get(key) != value:
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
        raise ValueError("feature schema/order differs from supported builder")
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
