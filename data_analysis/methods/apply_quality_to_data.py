"""
전체 PA에 quality_p 컬럼 적용 (Task 13 핸드오프용).

quality_p는 PA 단위가 아니라 "그 시점에 등판 중인 투수" 단위의 값이다
(all_pitcher_history.parquet의 prior_* 컬럼 기준, 등판 1회당 값 1개).
apply_li_to_data.py가 leverage_index를 PA 레벨에 직접 계산해 붙인 것과
달리, quality_p는:

  1. all_pitcher_history.parquet에서 등판(pitcher_id, game_id) 단위로
     quality_p를 계산하고 (선발/불펜 공식 자동 분기)
  2. all_pa_with_li.parquet에 (pitcher_id, game_id) 키로 병합(broadcast)한다
     — 같은 등판 안의 모든 PA는 동일한 quality_p 값을 공유한다(등판 시작
     시점 기준 값이라 등판 도중에는 바뀌지 않으므로).

**Task 14 (2026-09-16)**: quality_baseline.json / quality_baseline_bullpen.json은
이제 `bandit_game_split.csv`의 train 게임(5,030경기, 70%)만으로 만든
train-frozen 기준분포다 — validation/test 게임 정보가 feature 정규화에
새어 들어가지 않도록(LI의 train-frozen 재계산과 동일한 이유). 이 스크립트는
그 기준분포를 **전체 게임**(train+validation+test)에 적용해 quality_p를
계산한다 — 기준분포는 train만, 적용은 전체가 맞다(정규화 기준은 고정,
평가 대상은 전체 분할이어야 validation/test 성능이 진짜 out-of-sample
평가가 됨).

**Task 15 (2026-09-22)**: 불펜 quality_p의 원재료였던 `prior_wpa_mean`이
Option A(낡은 WE 계산)로 만들어진 값임이 확인되어, `prior_wpa_optionb.parquet`
(Option B로 재계산된 `prior_wpa_mean_optionb`)로 교체. 선발 공식(so/bb/hr
비율 기반)은 WPA를 안 써서 이 문제와 무관 — 그대로 유지.

prior_n_apps < 1인 행(그 투수의 첫 등판, 이전 기록 없음)은 quality_p를
정의할 수 없으므로 NaN으로 남긴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.methods.pitcher_quality import (
    BASELINE_PATH_BULLPEN,
    BASELINE_PATH_STARTER,
    apply_shrinkage,
    compute_raw_composite,
    load_quality_baseline,
)

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OPTIONB_PATH = Path("data_analysis/results/prior_wpa_optionb.parquet")
PA_PATH = Path("data_analysis/results/all_pa_with_li.parquet")
OUT_PATH = Path("data_analysis/results/all_pa_with_li.parquet")  # 같은 파일에 컬럼 추가


def _percentile_lookup_vec(values: np.ndarray, grid_pct: list[float], grid_values: list[float]) -> np.ndarray:
    return np.interp(values, grid_values, grid_pct) / 100.0


def compute_outing_quality_p(hist: pd.DataFrame) -> pd.DataFrame:
    """등판(pitcher_id, game_id) 단위로 quality_p 계산. prior_n_apps<1은 NaN."""
    df = hist.copy()
    df["quality_p"] = np.nan

    has_prior = df["prior_n_apps"] >= 1

    # ── 선발 ──
    starter_mask = has_prior & df["is_starter"]
    if starter_mask.any():
        baseline = load_quality_baseline(BASELINE_PATH_STARTER)
        sub = df.loc[starter_mask]
        raw = compute_raw_composite(sub["prior_so_rate"], sub["prior_bb_rate"], sub["prior_hr_rate"])
        shrunk = apply_shrinkage(
            raw, sub["prior_n_apps"], baseline["league_mean_composite"], baseline["shrinkage_k"]
        )
        pct = _percentile_lookup_vec(
            shrunk.to_numpy(), baseline["percentile_grid_pct"], baseline["percentile_grid_values"]
        )
        df.loc[starter_mask, "quality_p"] = 2.0 * pct

    # ── 불펜 ──
    bullpen_mask = has_prior & (~df["is_starter"]) & (df["prior_n_pa"] >= 1) & df["prior_wpa_mean_optionb"].notna()
    if bullpen_mask.any():
        baseline = load_quality_baseline(BASELINE_PATH_BULLPEN)
        sub = df.loc[bullpen_mask]
        prior_wpa_sum = sub["prior_wpa_mean_optionb"] * sub["prior_n_apps"]
        prior_wpa_per_pa = -prior_wpa_sum / sub["prior_n_pa"]
        shrunk = apply_shrinkage(
            prior_wpa_per_pa, sub["prior_n_pa"], baseline["league_mean_wpa_per_pa"], baseline["shrinkage_k"]
        )
        pct = _percentile_lookup_vec(
            shrunk.to_numpy(), baseline["percentile_grid_pct"], baseline["percentile_grid_values"]
        )
        df.loc[bullpen_mask, "quality_p"] = 2.0 * pct

    return df[["pitcher_id", "game_id", "is_starter", "quality_p"]]


def main() -> None:
    print("[1/3] all_pitcher_history.parquet 로드 및 등판별 quality_p 계산...")
    hist = pd.read_parquet(HIST_PATH)

    opb = pd.read_parquet(OPTIONB_PATH)
    hist = hist.merge(
        opb[["pitcher_id", "game_id", "half", "appearance_order", "prior_wpa_mean_optionb"]],
        on=["pitcher_id", "game_id", "half", "appearance_order"],
        how="left",
    )

    outing_quality = compute_outing_quality_p(hist)

    hist_quality = outing_quality

    n_total = len(outing_quality)
    n_missing = outing_quality["quality_p"].isna().sum()
    print(f"      등판 {n_total}건 중 quality_p 계산됨: {n_total - n_missing}건, "
          f"NaN(첫 등판, 이전 기록 없음): {n_missing}건")

    dup_counts = outing_quality.groupby(["pitcher_id", "game_id"]).size()
    dup_keys = dup_counts[dup_counts > 1]
    if len(dup_keys):
        print(
            f"[경고] (pitcher_id, game_id) 중복 {len(dup_keys)}건 발견 — "
            "한 경기에 같은 투수가 두 번 등장하는 데이터 이상치(예: half가 "
            "top/bot 둘 다 기록됨). quality_p는 평균으로 합쳐 대표값 1개로 축소함."
        )
        for (pid, gid), cnt in dup_keys.items():
            print(f"         pitcher_id={pid}, game_id={gid} ({cnt}건)")
    outing_quality = (
        outing_quality.groupby(["pitcher_id", "game_id"], as_index=False)["quality_p"].mean()
    )

    print("[2/3] all_pa_with_li.parquet에 (pitcher_id, game_id) 기준 병합...")
    pa = pd.read_parquet(PA_PATH)
    n_pa_before = len(pa)
    if "quality_p" in pa.columns:
        pa = pa.drop(columns=["quality_p"])
    pa = pa.merge(
        outing_quality[["pitcher_id", "game_id", "quality_p"]],
        on=["pitcher_id", "game_id"],
        how="left",
    )
    assert len(pa) == n_pa_before, "병합 후 행 수가 달라짐 — 조인 키 중복 의심"

    n_pa_missing = pa["quality_p"].isna().sum()
    print(f"      PA {len(pa)}행 중 quality_p 채워짐: {len(pa) - n_pa_missing}행 "
          f"({(len(pa) - n_pa_missing) / len(pa):.1%}), 결측: {n_pa_missing}행 "
          f"(해당 투수의 그 경기가 첫 등판 기록인 경우)")

    print("[3/3] 저장...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pa.to_parquet(OUT_PATH, index=False)
    print(f"      저장 완료: {OUT_PATH} ({pa.shape[0]}행 x {pa.shape[1]}열)")

    print("\n[요약] 역할별 quality_p 분포 (등판 단위, 중복 제거 전)")
    print(hist_quality.groupby("is_starter")["quality_p"].describe()[["count", "mean", "std", "min", "max"]])

    print(
        "\n[OK] 이 quality_p는 bandit_game_split.csv의 train 게임(5,030건)만으로 만든 "
        "train-frozen baseline(Task 14)을 전체 게임에 적용하고, 불펜 WPA 재료를 "
        "Option B로 교체(Task 15)한 최종 버전입니다."
    )


if __name__ == "__main__":
    main()
