"""
불펜 quality_p LI 착시 재검증 — Option B 재료 (Task 15).

check_bandit_wpa_li_confound.py를 Option B 재료(prior_wpa_per_pa_optionb,
y_window_mean_optionb)로 다시 실행. LI 자체는 Option A/B와 무관하므로
outing_li_mean 계산 로직은 그대로 재사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OPTIONB_PATH = Path("data_analysis/results/prior_wpa_optionb.parquet")
PA_PATH = Path("data_analysis/results/all_pa_with_li.parquet")

WINDOW_N = 5


def build_outing_li(pa: pd.DataFrame) -> pd.DataFrame:
    return pa.groupby(["pitcher_id", "game_id"]).agg(
        outing_li_mean=("leverage_index", "mean"),
    ).reset_index()


def partial_corr(x: pd.Series, y: pd.Series, cx: pd.Series, cy: pd.Series) -> tuple[float, float]:
    resid_x = x.to_numpy() - LinearRegression().fit(
        cx.to_numpy().reshape(-1, 1), x.to_numpy()
    ).predict(cx.to_numpy().reshape(-1, 1))
    resid_y = y.to_numpy() - LinearRegression().fit(
        cy.to_numpy().reshape(-1, 1), y.to_numpy()
    ).predict(cy.to_numpy().reshape(-1, 1))
    return spearmanr(resid_x, resid_y)


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    opb = pd.read_parquet(OPTIONB_PATH)
    pa = pd.read_parquet(PA_PATH)

    df = hist[["pitcher_id", "game_id", "half", "appearance_order", "is_starter", "prior_n_pa", "date"]].merge(
        opb[["pitcher_id", "game_id", "half", "appearance_order", "app_wpa_optionb", "n_pa_optionb",
             "prior_wpa_mean_optionb"]],
        on=["pitcher_id", "game_id", "half", "appearance_order"], how="inner",
    )
    df = df[(~df["is_starter"]) & df["prior_wpa_mean_optionb"].notna()].copy()
    df = df.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)
    df["prior_n_apps"] = df.groupby("pitcher_id").cumcount()
    df["prior_wpa_sum_optionb"] = df["prior_wpa_mean_optionb"] * df["prior_n_apps"]
    df["prior_wpa_per_pa_optionb"] = -df["prior_wpa_sum_optionb"] / df["prior_n_pa"]

    outing_li = build_outing_li(pa)
    df = df.merge(outing_li, on=["pitcher_id", "game_id"], how="left")

    def per_pitcher(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        wpa = g["app_wpa_optionb"].to_numpy()
        npa = g["n_pa_optionb"].to_numpy()
        li = g["outing_li_mean"].to_numpy()
        n = len(g)

        prior_li = np.full(n, np.nan)
        cum_li_pa, cum_pa = 0.0, 0.0
        for i in range(n):
            if cum_pa > 0:
                prior_li[i] = cum_li_pa / cum_pa
            if not np.isnan(li[i]):
                cum_li_pa += li[i] * npa[i]
                cum_pa += npa[i]

        y_win = np.full(n, np.nan)
        li_win = np.full(n, np.nan)
        for i in range(n - WINDOW_N + 1):
            w_pa = npa[i:i + WINDOW_N]
            y_win[i] = -wpa[i:i + WINDOW_N].sum() / w_pa.sum()
            w_li = li[i:i + WINDOW_N]
            if not np.isnan(w_li).any():
                li_win[i] = float(np.sum(w_li * w_pa) / w_pa.sum())

        g["prior_li_mean"] = prior_li
        g["y_window_mean_optionb"] = y_win
        g["li_window_mean"] = li_win
        return g

    df = df.groupby("pitcher_id", group_keys=False).apply(per_pitcher)

    cols = ["prior_wpa_per_pa_optionb", "y_window_mean_optionb", "prior_li_mean", "li_window_mean"]
    sub = df[cols].dropna()
    print(f"분석 대상 행: {len(sub)}\n")

    r_raw, p_raw = spearmanr(sub["prior_wpa_per_pa_optionb"], sub["y_window_mean_optionb"])
    print(f"[LI 통제 전] r = {r_raw:.4f}  p = {p_raw:.4g}")

    r_partial, p_partial = partial_corr(
        sub["prior_wpa_per_pa_optionb"], sub["y_window_mean_optionb"],
        sub["prior_li_mean"], sub["li_window_mean"],
    )
    print(f"[LI 통제 후] r = {r_partial:.4f}  p = {p_partial:.4g}")

    drop_pct = (r_raw - r_partial) / r_raw * 100 if r_raw else float("nan")
    print(f"\nLI 통제로 인한 감소: {drop_pct:.1f}%")
    passed = r_partial > 0.1 and p_partial < 0.05
    print(f"판정: {'PASS (착시 아님)' if passed else 'FAIL (착시였을 가능성)'}")


if __name__ == "__main__":
    main()
