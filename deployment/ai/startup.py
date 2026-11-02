"""Fail startup if either frozen model is missing, corrupt, or unexpected."""
import hashlib
import json
import os
from pathlib import Path

from kbo_bandit.serving.b1_chronology_v3 import load_model
from kbo_bandit.serving.catboost_service_v6 import ServicePaModel

root = Path(__file__).resolve().parent
for item in json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8")):
    path = root / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise RuntimeError("Artifact hash mismatch: " + item["path"])
b1 = load_model(os.environ.get("KBO_B1_ARTIFACT_DIR", root / "outputs/bandit/serving/b1_final_serving_v3"))
v6 = ServicePaModel(Path(os.environ.get("KBO_CATBOOST_SERVICE_ARTIFACT_DIR", root / "outputs/serving/catboost_service_v6")))
if b1.metadata["model_version"] != "b1-optionb-chronology67360-v3" or v6.metadata["model_version"] != "catboost-pa-service-v6-b-7e1e4a64634c":
    raise RuntimeError("Unexpected model version")
del b1, v6

import uvicorn
uvicorn.run("kbo_bandit.serving.api:app", host="0.0.0.0", port=int(os.environ.get("KBO_PORT", "8000")))
