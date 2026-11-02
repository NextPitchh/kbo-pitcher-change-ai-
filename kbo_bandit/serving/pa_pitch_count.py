"""Sample a PA pitch count from a frozen train-only empirical distribution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


class PaPitchCountSampler:
    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != "2d42b53987ba6837a09bb74e62f15036b44ecbec84ab2c6f474601a10a3ca8be":
            raise ValueError("Pitch count policy hash mismatch")
        self.metadata = json.loads(data)
        if self.metadata.get("version") != "pa-pitches-by-web-outcome-train2016-2023-v1":
            raise ValueError("Unsupported pitch count policy")

    def sample(self, outcome: str, rng: np.random.Generator) -> int:
        counts = self.metadata["outcomes"][outcome]["counts"]
        values = np.array([int(x) for x in counts], dtype=int)
        weights = np.array(list(counts.values()), dtype=float)
        if np.any(values < 1) or np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("Invalid pitch count distribution")
        return int(rng.choice(values, p=weights/weights.sum()))
