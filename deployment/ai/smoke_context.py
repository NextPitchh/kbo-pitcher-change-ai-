"""Run a build context in a separate process while denying parquet access."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[2]
context = Path(sys.argv[1]).resolve()
request = json.loads((root / "outputs/bandit/serving/b1_final_serving_v3/example_request.json").read_text(encoding="utf-8"))
expected = json.loads((root / "outputs/bandit/serving/b1_final_serving_v3/example_output.json").read_text(encoding="utf-8"))
state = {"game_id": "20250322HHKT02025", "inning": 5, "half": "top", "out_count": 0,
         "base_state": "0", "home_score": 2, "away_score": 1, "pitcher_id": 54354,
         "batter_id": 50704, "total_pitch_count_before": 63, "inning_pitch_count_before": 0}
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
env = os.environ.copy()
env["KBO_PORT"] = str(port)
env.pop("KBO_B1_ARTIFACT_DIR", None)
env.pop("KBO_CATBOOST_SERVICE_ARTIFACT_DIR", None)
script = """import runpy,sys
def audit(event,args):
 if event == 'open' and isinstance(args[0], (str,bytes)) and '.parquet' in str(args[0]).lower():
  raise RuntimeError('parquet access forbidden')
sys.addaudithook(audit)
runpy.run_path('startup.py',run_name='__main__')
"""
proc = subprocess.Popen([sys.executable, "-c", script], cwd=context, env=env)
try:
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
        for attempt in range(300):
            try:
                health = client.get("/health")
                break
            except httpx.HTTPError:
                if proc.poll() is not None:
                    raise RuntimeError("context startup failed")
                time.sleep(.1)
        else:
            raise RuntimeError("context startup timeout")
        assert health.status_code == 200
        assert client.post("/v1/bandit/recommend", json=request).json() == expected
        result = client.post("/v1/pa/predict-service", json={"state": state})
        assert result.status_code == 200
        assert result.json()["model_version"] == "catboost-pa-service-v6-b-7e1e4a64634c"
        assert abs(sum(result.json()["web_probabilities"].values()) - 1) < 1e-12
        print("PASS: isolated context startup, B1, v6 and parquet access guard")
finally:
    proc.terminate()
    proc.wait(timeout=10)
