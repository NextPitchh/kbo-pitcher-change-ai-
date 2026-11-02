"""Versioned conditional OUT split, expressed as estimates rather than model output."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from .catboost_inference import CLASSES, ROOT

DEFAULT_MAPPING = ROOT / "outputs/serving/catboost_web_audit/out_mapping_2024_v1.json"
MAPPING_SHA256 = "872518250777251e3729b9a554fb182f15d8d93ddebe08881ea498e3486802cb"
WEB_CLASSES = ("1B", "2B", "3B", "BB", "GDP", "HR", "GO", "FO", "SO")
BASES = ("0", "1", "2", "3", "12", "13", "23", "123")


class WebOutcomeAdapter:
    def __init__(self, path: Path = DEFAULT_MAPPING):
        contents = Path(path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != MAPPING_SHA256:
            raise ValueError("Web mapping hash mismatch")
        self.metadata = json.loads(contents)
        if self.metadata.get("version") != "v5-out-2024-state-jeffreys-v1":
            raise ValueError("Unexpected mapping version")
        states = self.metadata["states"]
        if set(states) != {f"{out}:{base}" for out in range(3) for base in BASES}:
            raise ValueError("Incomplete state mapping")
        for value in states.values():
            counts, weights = value["counts"], value["probabilities"]
            if set(counts) != {"GO", "FO", "SO"} or set(weights) != set(counts):
                raise ValueError("Invalid mapped classes")
            if any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError("Invalid counts")
            n = sum(counts.values())
            for key in counts:
                expected = (counts[key] + .5) / (n + 1.5)
                if not math.isfinite(weights[key]) or not math.isclose(weights[key], expected, abs_tol=1e-14):
                    raise ValueError("Mapping probabilities do not match source counts")

    def convert(self, native: dict[str, float], *, out_count: int, base_state: str) -> dict[str, float]:
        if set(native) != set(CLASSES) or any(not math.isfinite(p) or p < 0 or p > 1 for p in native.values()):
            raise ValueError("Invalid native probability vector")
        if not math.isclose(sum(native.values()), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Native probabilities must sum to one")
        if type(out_count) is not int or out_count not in (0, 1, 2) or base_state not in BASES:
            raise ValueError("Invalid game state")
        w = self.metadata["states"][f"{out_count}:{base_state}"]["probabilities"]
        result = {k: native[k] for k in WEB_CLASSES if k not in ("GO", "FO", "SO")}
        result["GO"] = native["OUT"] * w["GO"]
        result["FO"] = native["OUT"] * w["FO"] + native["SF"]
        result["SO"] = native["OUT"] * w["SO"] + native["SO"]
        if out_count == 2 or "1" not in base_state:
            # Spring RunnerAdvancement.doublePlay deterministically degrades it to GO.
            result["GO"] += result["GDP"]
            result["GDP"] = 0.0
        if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Converted probabilities must sum to one")
        return {k: result[k] for k in WEB_CLASSES}
