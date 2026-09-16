"""
투수 품질(quality_p) 프로덕션 기준값 산출 — train-frozen (Task 5, Task 14).

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

**Task 14 변경**: 이전 버전은 전체 기간(2016~2025)으로 기준분포를 만든
잠정본이었다. Bandit v2가 `bandit_game_split.csv`(game_id 단위 70/15/15
분할)로 train/validation/test를 나누므로, 리그 평균/백분위 분포도
**train 게임만으로** 다시 만든다 — validation/test 기간 정보가 feature
정규화에 새어 들어가는 걸 막기 위함(LI의 train-frozen 재계산과 동일한
이유). validation/test 게임의 quality_p도 이 train-frozen 기준분포를
그대로 적용해서 계산한다(기준분포 자체는 train만, 적용은 전체).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
SPLIT_PATH = Path("bandit_game_split.csv")
OUT_PATH = Path("data_analysis/results/quality_baseline.json")

SHRINKAGE_K = 5  # Task 4에서 확정
N_GRID_POINTS = 1001  # 0.0%, 0.1%, ..., 100.0%


def load_train_game_ids() -> set[str]:
    split = pd.read_csv(SPLIT_PATH)
    return set(split.loc[split["split"] == "train", "game_id"])


def main() -> None:
    train_game_ids = load_train_game_ids()
    print(f"train 게임 수: {len(train_game_ids)} (bandit_game_split.csv 기준)")

    hist = pd.read_parquet(HIST_PATH)
    df = hist[(hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    n_before_split_filter = len(df)
    df = df[df["game_id"].isin(train_game_ids)]
    print(f"선발 등판 {n_before_split_filter}건 중 train 게임 소속: {len(df)}건")

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
        "fit_split": "train",
        "fit_split_source": "bandit_game_split.csv",
        "league_mean_composite": league_mean,
        "shrinkage_k": SHRINKAGE_K,
        "scaling": "2 * percentile_rank(shrunk_composite)",
        "percentile_grid_pct": grid_pct.tolist(),
        "percentile_grid_values": grid_values.tolist(),
        "note": "불펜(reliever)용 기준분포는 build_quality_baseline_bullpen.py 참고.",
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {OUT_PATH} 저장 완료")
    print(f"  n_fit = {len(df)}, league_mean_composite = {league_mean:.6f}")
    print(f"  grid points = {N_GRID_POINTS}")


if __name__ == "__main__":
    main()
