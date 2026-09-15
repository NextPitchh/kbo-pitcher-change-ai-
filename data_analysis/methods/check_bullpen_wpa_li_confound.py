"""
불펜 prior_wpa_per_pa <-> y_window_mean 상관관계가 LI(상황 긴장도) 착시인지 확인.

validate_quality_predictive_bullpen_smoothed.py에서 prior_wpa_per_pa가
불펜의 "다음 5등판 평균 성과"를 잘 예측하는 것으로 나왔다(r=0.1487, PASS).
그런데 WPA는 그 상황의 leverage_index가 클수록 같은 결과라도 값이 커지는
지표라, 이 상관관계가 "투수 실력" 때문이 아니라 "이 투수가 원래 감독한테
신뢰받아 높은 LI 상황에 반복적으로 기용된다"는 역할/기용 패턴 때문일 수
있다. 이 스크립트는 그 의심을 직접 검증한다.

방법: 각 불펜 등판(pitcher_id, game_id)의 평균 leverage_index를
all_pa_with_li.parquet에서 계산해 붙인다.

  prior_li_mean   = 이 시점 이전 등판들의 (n_pa 가중) 누적 평균 LI
  li_window_mean  = y_window_mean과 동일한 전방 5등판 구간의 (n_pa 가중) 평균 LI

그 다음 편상관(partial correlation)을 계산한다:
  1) prior_wpa_per_pa ~ prior_li_mean 으로 회귀 -> 잔차 A
     (prior_wpa_per_pa에서 "과거 LI 수준으로 설명되는 부분"을 제거)
  2) y_window_mean ~ li_window_mean 으로 회귀 -> 잔차 B
     (y_window_mean에서 "미래 LI 수준으로 설명되는 부분"을 제거)
  3) 잔차 A와 잔차 B의 Spearman 상관 = "LI 수준을 통제한 뒤에도 남는 진짜 상관"

이게 원래 r=0.1487보다 많이 줄어들면(특히 0.1 밑으로 떨어지면) 착시였다는
뜻이고, 비슷하게 유지되면 LI와 무관한 진짜 실력 신호일 가능성이 높다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
PA_PATH = Path("data_analysis/results/all_pa_with_li.parquet")

WINDOW_N = 5


def build_outing_li(pa: pd.DataFrame) -> pd.DataFrame:
    g = pa.groupby(["pitcher_id", "game_id"]).agg(
        outing_li_mean=("leverage_index", "mean"),
        outing_li_n=("leverage_index", "size"),
    ).reset_index()
    return g


def _prep(hist: pd.DataFrame, outing_li: pd.DataFrame) -> pd.DataFrame:
    df = hist[(~hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df = df.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    df["y_this_outing"] = -df["app_wpa"] / df["n_pa"]
    df["prior_wpa_sum"] = df["prior_wpa_mean"] * df["prior_n_apps"]
    df["prior_wpa_per_pa"] = -df["prior_wpa_sum"] / df["prior_n_pa"]

    df = df.merge(outing_li, on=["pitcher_id", "game_id"], how="left")
    n_missing = df["outing_li_mean"].isna().sum()
    if n_missing:
        print(f"[경고] LI 매칭 안 된 행 {n_missing}개 (game_id/pitcher_id 불일치) — 제외")

    def per_pitcher(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        wpa = g["app_wpa"].to_numpy()
        npa = g["n_pa"].to_numpy()
        li = g["outing_li_mean"].to_numpy()
        n = len(g)

        # prior_li_mean: 이 행 "이전" 등판들만의 n_pa 가중 누적평균 (leakage 없음)
        prior_li = np.full(n, np.nan)
        cum_li_pa = 0.0
        cum_pa = 0.0
        for i in range(n):
            if cum_pa > 0:
                prior_li[i] = cum_li_pa / cum_pa
            if not np.isnan(li[i]):
                cum_li_pa += li[i] * npa[i]
                cum_pa += npa[i]

        # y_window_mean / li_window_mean: 이 행부터 N개(이 행 포함) 전방 구간
        y_win = np.full(n, np.nan)
        li_win = np.full(n, np.nan)
        for i in range(n - WINDOW_N + 1):
            w_pa = npa[i : i + WINDOW_N]
            y_win[i] = -wpa[i : i + WINDOW_N].sum() / w_pa.sum()
            w_li = li[i : i + WINDOW_N]
            if not np.isnan(w_li).any():
                li_win[i] = float(np.sum(w_li * w_pa) / w_pa.sum())

        g["prior_li_mean"] = prior_li
        g["y_window_mean"] = y_win
        g["li_window_mean"] = li_win
        return g

    df = df.groupby("pitcher_id", group_keys=False).apply(per_pitcher)
    return df


def partial_corr(x: pd.Series, y: pd.Series, cx: pd.Series, cy: pd.Series) -> tuple[float, float]:
    resid_x = x.to_numpy() - LinearRegression().fit(cx.to_numpy().reshape(-1, 1), x.to_numpy()).predict(
        cx.to_numpy().reshape(-1, 1)
    )
    resid_y = y.to_numpy() - LinearRegression().fit(cy.to_numpy().reshape(-1, 1), y.to_numpy()).predict(
        cy.to_numpy().reshape(-1, 1)
    )
    r, p = spearmanr(resid_x, resid_y)
    return float(r), float(p)


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    pa = pd.read_parquet(PA_PATH)
    outing_li = build_outing_li(pa)
    df = _prep(hist, outing_li)

    cols = ["prior_wpa_per_pa", "y_window_mean", "prior_li_mean", "li_window_mean"]
    sub = df[cols].dropna()
    print(f"분석 대상 행: {len(sub)} (원본 불펜 행 {len(df)}개 중)\n")

    r_raw, p_raw = spearmanr(sub["prior_wpa_per_pa"], sub["y_window_mean"])
    print(f"[원래 상관, LI 통제 전]      r = {r_raw:.4f}  p = {p_raw:.4g}")

    r_partial, p_partial = partial_corr(
        sub["prior_wpa_per_pa"], sub["y_window_mean"],
        sub["prior_li_mean"], sub["li_window_mean"],
    )
    print(f"[편상관, LI 통제 후]         r = {r_partial:.4f}  p = {p_partial:.4g}")

    print(f"\n참고 — LI 자체가 두 변수를 얼마나 설명하는지:")
    r1, p1 = spearmanr(sub["prior_wpa_per_pa"], sub["prior_li_mean"])
    r2, p2 = spearmanr(sub["y_window_mean"], sub["li_window_mean"])
    print(f"  prior_wpa_per_pa  vs prior_li_mean : r = {r1:.4f}  p = {p1:.4g}")
    print(f"  y_window_mean     vs li_window_mean: r = {r2:.4f}  p = {p2:.4g}")

    drop_pct = (r_raw - r_partial) / r_raw * 100 if r_raw else float("nan")
    print(f"\nLI 통제로 인한 상관 감소: {drop_pct:.1f}%")
    print("판정 기준: r>0.1 & p<0.05 (Task 2와 동일)")
    passed = r_partial > 0.1 and p_partial < 0.05
    print(f"LI 통제 후에도 기준 통과: {'PASS (착시 아님, 실력 신호로 볼 근거 있음)' if passed else 'FAIL (착시였을 가능성 높음)'}")


if __name__ == "__main__":
    main()
