"""
투수 품질(quality_p) 불펜 프로덕션 기준값 산출 — Option B + train-frozen (Task 15).

build_quality_baseline.py(선발용)와 동일한 방식이지만, 공식과 shrinkage
단위가 다르다 (Task 6-9에서 확정):

파이프라인 순서:
  1. raw = prior_wpa_per_pa (= -prior_wpa_sum/prior_n_pa)   (Task 7)
  2. shrunk = shrink(raw, prior_n_pa, k=25, league_mean)      (Task 9)
  3. quality_p = 2 * percentile_rank(shrunk, 이 기준분포)      (선발과 동일 스케일링 방식)

기준분포는 불펜(is_starter=False, prior_n_apps>=1) 모집단의 shrunk 값을
0.1%포인트 간격(1001개 지점)으로 뽑아 저장한다.

**Task 15 변경 (2026-09-22)**: `prior_wpa_per_pa`의 원재료였던
`all_pitcher_history.prior_wpa_mean`이 Option A(아웃카운트 보정 없는 낡은
WE 계산)로 만들어진 값임이 확인됨 — 같은 파일의 `app_wpa`를 Option B
기준(`all_pa_with_li.reward_wpa_computed`)과 직접 비교했을 때 값이 유의미하게
다름을 검증(부호가 뒤집히는 사례도 존재). `build_prior_wpa_optionb.py`로
재계산한 Option B 버전(`prior_wpa_mean_optionb`)을 사용하도록 교체.

재검증 결과(재료만 교체, 방법론은 동일):
  - Task 7 재검증: r = 0.1487 -> 0.1284 (여전히 PASS, 기준 0.1 초과)
  - Task 8 재검증(LI 착시): 0.9% -> 0.1% 감소 (여전히 착시 아님, 더 견고해짐)
  - Task 9 재검증(k): k=25 여전히 근접 최적 (패턴 동일)
  - league_mean_wpa_per_pa: -0.009747(Option A) -> 재계산 필요(Option B, train-frozen)

**Task 14의 train-frozen 로직은 그대로 유지** — `bandit_game_split.csv`의
train 게임만으로 기준분포를 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OPTIONB_PATH = Path("data_analysis/results/prior_wpa_optionb.parquet")
SPLIT_PATH = Path("bandit_game_split.csv")
OUT_PATH = Path("data_analysis/results/quality_baseline_bullpen.json")

SHRINKAGE_K = 25  # Task 9에서 확정 (Task 15 재검증에서도 근접 최적 재확인)
N_GRID_POINTS = 1001  # 0.0%, 0.1%, ..., 100.0%


def load_train_game_ids() -> set[str]:
    split = pd.read_csv(SPLIT_PATH)
    return set(split.loc[split["split"] == "train", "game_id"])


def main() -> None:
    train_game_ids = load_train_game_ids()
    print(f"train 게임 수: {len(train_game_ids)} (bandit_game_split.csv 기준)")

    hist = pd.read_parquet(HIST_PATH)
    opb = pd.read_parquet(OPTIONB_PATH)

    df = hist[["pitcher_id", "game_id", "half", "appearance_order", "is_starter", "prior_n_pa"]].merge(
        opb[["pitcher_id", "game_id", "half", "appearance_order", "prior_wpa_mean_optionb"]],
        on=["pitcher_id", "game_id", "half", "appearance_order"],
        how="inner",
    )
    df = df[(~df["is_starter"]) & df["prior_wpa_mean_optionb"].notna()].copy()

    n_before_split_filter = len(df)
    df = df[df["game_id"].isin(train_game_ids)]
    print(f"불펜 등판 {n_before_split_filter}건 중 train 게임 소속: {len(df)}건")

    # prior_n_apps: hist와 동일한 방식(등판 순번, Option A/B 무관 — 그냥 개수)
    df = df.sort_values(["pitcher_id", "game_id"]).reset_index(drop=True)
    hist_prior_n_apps = hist.set_index(["pitcher_id", "game_id", "half", "appearance_order"])["prior_n_apps"]
    df = df.set_index(["pitcher_id", "game_id", "half", "appearance_order"])
    df["prior_n_apps"] = hist_prior_n_apps
    df = df.reset_index()

    df["prior_wpa_sum"] = df["prior_wpa_mean_optionb"] * df["prior_n_apps"]
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
        "fit_split": "train",
        "fit_split_source": "bandit_game_split.csv",
        "wpa_convention": "Option B (reward_wpa_computed 기반, Task 15 수정)",
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
