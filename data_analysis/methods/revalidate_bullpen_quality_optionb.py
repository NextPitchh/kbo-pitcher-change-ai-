"""
불펜 quality_p 재검증 — Option B 재료로 Task 7 다시 확인 (Task 15).

Task 7(validate_quality_predictive_bullpen_smoothed.py)은 X(prior_wpa_per_pa)와
y(전방 5등판 평균 실제 성과) 둘 다 all_pitcher_history.parquet의 app_wpa/
prior_wpa_mean(Option A, 낡음)으로 계산했었다. build_prior_wpa_optionb.py로
만든 Option B 버전(app_wpa_optionb, prior_wpa_mean_optionb)으로 X, y를 모두
다시 만들어 같은 방식(전방 5등판 평균, Spearman r>0.1 기준)으로 재검증한다.

목적: 재료를 Option A -> B로 바꿔도 "이 지표가 기준을 통과한다"는 결론이
그대로 유지되는지 확인. 유지 안 되면 quality_p 공식 자체를 다시 논의해야 함.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OPTIONB_PATH = Path("data_analysis/results/prior_wpa_optionb.parquet")

WINDOW_N = 5


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    opb = pd.read_parquet(OPTIONB_PATH)

    df = hist[["pitcher_id", "game_id", "half", "appearance_order", "is_starter", "prior_n_pa", "date"]].merge(
        opb[["pitcher_id", "game_id", "half", "appearance_order", "app_wpa_optionb", "n_pa_optionb",
             "prior_wpa_mean_optionb"]],
        on=["pitcher_id", "game_id", "half", "appearance_order"],
        how="inner",
    )

    df = df[(~df["is_starter"]) & df["prior_wpa_mean_optionb"].notna()].copy()
    df = df.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    # prior_wpa_per_pa_optionb 유도 (기존 공식과 동일한 방식, 재료만 교체)
    df["prior_n_apps"] = df.groupby("pitcher_id").cumcount()  # hist와 정확히 일치 확인된 방식
    df["prior_wpa_sum_optionb"] = df["prior_wpa_mean_optionb"] * df["prior_n_apps"]
    df["prior_wpa_per_pa_optionb"] = -df["prior_wpa_sum_optionb"] / df["prior_n_pa"]

    def forward_window_mean(g: pd.DataFrame) -> pd.Series:
        wpa = g["app_wpa_optionb"].to_numpy()
        npa = g["n_pa_optionb"].to_numpy()
        n = len(g)
        out = np.full(n, np.nan)
        for i in range(n - WINDOW_N + 1):
            w = npa[i: i + WINDOW_N]
            out[i] = -wpa[i: i + WINDOW_N].sum() / w.sum()
        return pd.Series(out, index=g.index)

    df["y_window_mean_optionb"] = df.groupby("pitcher_id", group_keys=False).apply(forward_window_mean)

    sub = df.dropna(subset=["prior_wpa_per_pa_optionb", "y_window_mean_optionb"])
    print(f"검증 대상 행: {len(sub)}")

    r, p = spearmanr(sub["prior_wpa_per_pa_optionb"], sub["y_window_mean_optionb"])
    print(f"\n[Option B 재검증] prior_wpa_per_pa_optionb vs y_window_mean_optionb")
    print(f"  Spearman r = {r:.4f}   p = {p:.4g}")
    print(f"  기준(r>0.1, p<0.05): {'PASS' if r > 0.1 and p < 0.05 else 'FAIL'}")

    # 참고: 기존 Option A 결과와 비교
    r_old = 0.1487
    print(f"\n[참고] 기존 Option A 결과: r = {r_old:.4f}")
    print(f"  차이: {r - r_old:+.4f}")


if __name__ == "__main__":
    main()
