"""Create a new, allowlisted Docker build context from verified frozen files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B1 = "outputs/bandit/serving/b1_final_serving_v3/"
V6 = "outputs/serving/catboost_service_v6/"
FILES = {
    B1 + "metadata.json": "537e4e323b0957418d9f233ec908c7abfde7b6ea592a0cc946f8ed9d2583e1df",
    B1 + "theta.npy": "00de12f81085ab68874d28dabe5336afd7e37ecc73ed9db8ae7b82df2e4d225f",
    B1 + "preprocessing.npz": "3529d69912fd8f8643d1eea2c412f210bccfeee70b3a5fefc8f1e4ef467b4fc3",
    B1 + "feature_schema.json": "14de38e662f8417780d8fcdb1e4631040119f7d872781cd9a8dc0be23f7c6f7e",
    V6 + "model.cbm": "7e1e4a64634cdcacbf580453e0413cf3b9948adb5692cd402e97dba000030a6d",
    V6 + "selection.json": "5a6f7d47157117ff73857b73bd2a19d062ffa21cf259f98dbcbd2c02d49163f4",
    V6 + "train_history_2023.json": "63010d94ab60031d5940f8dc1f6688ad5e0088e8986607cd581223071c4584bd",
}
SOURCE = ["__init__.py", "api.py", "b1_chronology_v3.py", "b1_inference.py",
          "catboost_schema.py", "catboost_service_schema.py", "catboost_service_v6.py",
          "catboost_inference.py", "catboost_web_adapter.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path, help="new, nonexistent context directory")
    dest = parser.parse_args().destination.resolve()
    if dest.exists():
        raise SystemExit("destination already exists; choose a new directory")
    for name, expected in FILES.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise SystemExit(f"missing or mismatched frozen artifact: {name}")
    dest.mkdir(parents=True)
    entries = []
    for name, expected in FILES.items():
        source = ROOT / name
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != expected:
            raise SystemExit(f"copy verification failed: {name}")
        entries.append({"path": name, "sha256": expected, "bytes": target.stat().st_size})
    (dest / "artifact-manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (dest / "kbo_bandit/serving").mkdir(parents=True)
    shutil.copyfile(ROOT / "kbo_bandit/__init__.py", dest / "kbo_bandit/__init__.py")
    for name in SOURCE:
        shutil.copyfile(ROOT / "kbo_bandit/serving" / name, dest / "kbo_bandit/serving" / name)
    for name in ("Dockerfile", "requirements.txt", "startup.py"):
        shutil.copyfile(ROOT / "deployment/ai" / name, dest / name)
    print(f"Created verified context: {dest}")


if __name__ == "__main__":
    main()
