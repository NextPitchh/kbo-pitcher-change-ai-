"""Read-only frozen-model and actual localhost HTTP checks; no reports are overwritten."""
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import shutil

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kbo_bandit.serving.b1_chronology_v3 import load_model
from kbo_bandit.serving.catboost_service_v6 import ServicePaModel

b1dir = ROOT / "outputs/bandit/serving/b1_final_serving_v3"
v6dir = ROOT / "outputs/serving/catboost_service_v6"
payload = json.loads((b1dir / "example_request.json").read_text(encoding="utf-8"))
expected = json.loads((b1dir / "example_output.json").read_text(encoding="utf-8"))
state = {"game_id": "20250322HHKT02025", "inning": 5, "half": "top", "out_count": 0,
         "base_state": "0", "home_score": 2, "away_score": 1, "pitcher_id": 54354,
         "batter_id": 50704, "total_pitch_count_before": 63, "inning_pitch_count_before": 0}
b1 = load_model(b1dir)
assert b1.score_candidates(**payload) == expected["rankings"]
cases = json.loads((b1dir / "builder_cases.json").read_text(encoding="utf-8"))
with np.load(b1dir / "builder_reference.npz", allow_pickle=False) as reference:
    for case in cases:
        assert np.allclose(b1.transform_candidates(case["game_state"], case["candidates"]),
                           reference[case["name"]], rtol=0, atol=1e-13)
v6 = ServicePaModel(v6dir)
ref = json.loads((v6dir / "original_golden_2024.json").read_text(encoding="utf-8"))
def state_from_features(row):
    delta = int(row["score_diff_attacker"])
    home_diff = delta if row["half"] == "bot" else -delta
    return {"game_id": "20250322HHKT02025", "inning": int(row["inning"]), "half": row["half"],
            "out_count": int(row["out_count"]), "base_state": row["base_state"],
            "home_score": max(home_diff, 0), "away_score": max(-home_diff, 0),
            "pitcher_id": row["pitcher_id"], "batter_id": row["batter_id"],
            "total_pitch_count_before": int(row["total_pitch_count_before"]),
            "inning_pitch_count_before": int(row["inning_pitch_count_before"])}
for row, probabilities in zip(ref["features"], ref["probabilities"], strict=True):
    case = state_from_features(row)
    assert v6.build_features(case) == row
    assert np.allclose(list(v6.predict(case)["native_probabilities"].values()), probabilities, rtol=0, atol=1e-12)

# Work only on disposable copies; never mutate a frozen original.
with tempfile.TemporaryDirectory() as directory:
    tmp = Path(directory)
    for source, names, loader, corrupt in (
        (b1dir, ("metadata.json", "theta.npy", "preprocessing.npz", "feature_schema.json"), load_model, "theta.npy"),
        (v6dir, ("selection.json", "model.cbm", "train_history_2023.json"), ServicePaModel, "model.cbm"),
    ):
        target = tmp / source.name
        target.mkdir()
        for name in names:
            shutil.copyfile(source / name, target / name)
        (target / corrupt).unlink()
        try:
            loader(target)
            raise AssertionError("missing artifact was accepted")
        except FileNotFoundError:
            pass
        shutil.copyfile(source / corrupt, target / corrupt)
        with (target / corrupt).open("ab") as f:
            f.write(b"corrupt")
        try:
            loader(target)
            raise AssertionError("corrupt artifact was accepted")
        except ValueError:
            pass

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "kbo_bandit.serving.api:app", "--host", "127.0.0.1",
                         "--port", str(port), "--log-level", "error"], cwd=ROOT)
try:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
        for attempt in range(150):
            try:
                health = client.get("/health")
                break
            except httpx.HTTPError:
                if proc.poll() is not None:
                    raise RuntimeError("server exited")
                time.sleep(.1)
        else:
            raise RuntimeError("server startup timeout")
        assert health.status_code == 200
        a = client.post("/v1/bandit/recommend", json=payload)
        b = client.post("/v1/pa/predict-service", json={"state": state})
        assert a.status_code == 200 and a.json() == expected
        assert b.status_code == 200 and b.json() == v6.predict(state)
        assert client.post("/v1/pa/predict-service", json={"state": dict(state, out_count=3)}).status_code == 422
        assert client.post("/v1/bandit/recommend", json={}).status_code == 422
        assert client.post("/v1/pa/predict-service", json={"state": state}).json() == b.json()
        print(json.dumps({"health": health.json(), "b1": a.json(), "v6": b.json(),
                          "b1_builder_cases": len(cases), "v6_golden_cases": len(ref["features"]),
                          "validation_and_repeat": "PASS", "missing_and_corrupt_models": "PASS"}, ensure_ascii=False, indent=2))
finally:
    proc.terminate()
    proc.wait(timeout=10)
