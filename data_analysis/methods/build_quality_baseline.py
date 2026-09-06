"""
투수 품질(quality_p) 프로덕션 기준값 산출 — Task 5 사전 작업.

pitcher_quality.py가 실시간으로 "이 투수 지금 quality_p 얼마?"를 계산하려면,
매번 전체 데이터를 다시 훑어서 백분위를 계산할 수는 없다. 그래서 기준
분포(quality_baseline.json)를 미리 만들어두고, 새 값이 들어오면 이 기준
분포에서 "몇 번째 백분위에 해당하는지"만 보간(interpolation)으로 빠르게
찾는다 (leverage_index.py가 league_pa_distribution.json을 쓰는 것과 같은
방식).

파이프라인 순서 (Task 3/4에서 확정된 순서 그대로):
  1. raw = prior_so_rate - prior_bb_rate - 2*prior_hr_rate   (Layer 1, Task 1-2)
  2. shrunk = shrink(raw, n_apps, k=5, league_mean)            (Task 4)
  3. quality_p = 2 * percentile_rank(shrunk, 기준분포)          (Task 3)

기준분포는 선발투수(is_starter=True, prior_n_apps>=1) 모집단의 shrunk 값을
0.1%포인트 간격(1001개 지점)으로 뽑아 저장한다 — 원본 14,137개를 다
저장할 필요 없이 이 정도 해상도면 보간 오차가 무시할 만하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OUT_PATH = Path("data_analysis/results/quality_baseline.json")

SHRINKAGE_K = 5  # Task 4에서 확정
N_GRID_POINTS = 1001  # 0.0%, 0.1%, ..., 100.0%


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = hist[(hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df["prior_composite"] = (
        df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]
    )

    league_mean = float(df["prior_composite"].mean())

    w = df["prior_n_apps"] / (df["prior_n_apps"] + SHRINKAGE_K)
    shrunk = w * df["prior_composite"] + (1 - w) * league_mean

    grid_pct = np.linspace(0, 100, N_GRID_POINTS)
    grid_values = np.percentile(shrunk, grid_pct)

    output = {
        "role": "starter",
        "n_fit": int(len(df)),
        "league_mean_composite": league_mean,
        "shrinkage_k": SHRINKAGE_K,
        "scaling": "2 * percentile_rank(shrunk_composite)",
        "percentile_grid_pct": grid_pct.tolist(),
        "percentile_grid_values": grid_values.tolist(),
        "note": (
            "불펜(reliever)용 기준분포는 아직 없음 — Task 2에서 예측력 검증을 "
            "통과한 지표가 없어 팀 논의 후 별도 확정 예정."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {OUT_PATH} 저장 완료")
    print(f"  n_fit = {len(df)}, league_mean_composite = {league_mean:.6f}")
    print(f"  grid points = {N_GRID_POINTS}")


if __name__ == "__main__":
    main()
