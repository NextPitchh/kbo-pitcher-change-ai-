"""
투수 품질(quality_p) 불펜 프로덕션 기준값 산출.

build_quality_baseline.py(선발용)와 동일한 방식이지만, 공식과 shrinkage
단위가 다르다 (Task 6-9에서 확정):

파이프라인 순서:
  1. raw = prior_wpa_per_pa (= -prior_wpa_sum/prior_n_pa)   (Task 7)
  2. shrunk = shrink(raw, prior_n_pa, k=25, league_mean)      (Task 9)
  3. quality_p = 2 * percentile_rank(shrunk, 이 기준분포)      (선발과 동일 스케일링 방식)

기준분포는 불펜(is_starter=False, prior_n_apps>=1) 모집단의 shrunk 값을
0.1%포인트 간격(1001개 지점)으로 뽑아 저장한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OUT_PATH = Path("data_analysis/results/quality_baseline_bullpen.json")

SHRINKAGE_K = 25  # Task 9에서 확정
N_GRID_POINTS = 1001  # 0.0%, 0.1%, ..., 100.0%


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = hist[(~hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df["prior_wpa_sum"] = df["prior_wpa_mean"] * df["prior_n_apps"]
    df["prior_wpa_per_pa"] = -df["prior_wpa_sum"] / df["prior_n_pa"]
    df = df.dropna(subset=["prior_wpa_per_pa"])

    league_mean = float(df["prior_wpa_per_pa"].mean())

    w = df["prior_n_pa"] / (df["prior_n_pa"] + SHRINKAGE_K)
    shrunk = w * df["prior_wpa_per_pa"] + (1 - w) * league_mean

    grid_pct = np.linspace(0, 100, N_GRID_POINTS)
    grid_values = np.percentile(shrunk, grid_pct)

    output = {
        "role": "bullpen",
        "n_fit": int(len(df)),
        "league_mean_wpa_per_pa": league_mean,
        "shrinkage_k": SHRINKAGE_K,
        "shrinkage_unit": "prior_n_pa",
        "raw_formula": "prior_wpa_per_pa",
        "scaling": "2 * percentile_rank(shrunk_wpa_per_pa)",
        "percentile_grid_pct": grid_pct.tolist(),
        "percentile_grid_values": grid_values.tolist(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {OUT_PATH} 저장 완료")
    print(f"  n_fit = {len(df)}, league_mean_wpa_per_pa = {league_mean:.6f}")
    print(f"  grid points = {N_GRID_POINTS}")


if __name__ == "__main__":
    main()
