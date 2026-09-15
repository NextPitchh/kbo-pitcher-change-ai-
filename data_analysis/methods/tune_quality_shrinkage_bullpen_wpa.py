"""
불펜 quality_p shrinkage 계수 k 탐색 — WPA 기반 공식판.

check_bullpen_wpa_li_confound.py에서 prior_wpa_per_pa가 불펜의 "다음 5등판
평균 성과"(y_window_mean)를 예측하는 진짜 신호(LI 착시 아님, r=0.147)임을
확인했다. 이 스크립트는 그 raw 지표에 shrinkage를 적용해 더 나아지는지,
그리고 최적 k가 뭔지 찾는다.

    w = prior_n_pa / (prior_n_pa + k)
    shrunk = w * prior_wpa_per_pa + (1 - w) * league_mean_wpa_per_pa

표본 크기 단위는 등판 수(prior_n_apps)가 아니라 누적 상대 타석 수
(prior_n_pa)를 쓴다 — 2026-09 회의에서 나온 "불펜은 등판당 타자 수
편차가 커서 등판 횟수가 표본 크기를 잘 대변 못한다"는 지적을 반영.

평가 목표(y)는 validate_quality_predictive_bullpen_smoothed.py와 동일한
전방 5등판 평균 실제 성과(y_window_mean) — 이게 실제로 신호가 검증된
목표이므로 shrinkage 튜닝도 같은 목표로 해야 일관성이 있다.

기준: Spearman r > 0.1, p < 0.05 (Task 2/4와 동일).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OUT_PATH = Path("data_analysis/results/quality_shrinkage_tuning_bullpen_wpa.json")

WINDOW_N = 5
K_CANDIDATES = [25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]
RNG_SEED = 42


def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    r, p = spearmanr(X.ravel(), y)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    return {"spearman_r": float(r), "p_value": float(p),
            "cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}


def _prep(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist[(~hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df = df.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    df["prior_wpa_sum"] = df["prior_wpa_mean"] * df["prior_n_apps"]
    df["prior_wpa_per_pa"] = -df["prior_wpa_sum"] / df["prior_n_pa"]

    def forward_window_mean(g: pd.DataFrame) -> pd.Series:
        wpa = g["app_wpa"].to_numpy()
        npa = g["n_pa"].to_numpy()
        n = len(g)
        out = np.full(n, np.nan)
        for i in range(n - WINDOW_N + 1):
            w = npa[i : i + WINDOW_N]
            out[i] = -wpa[i : i + WINDOW_N].sum() / w.sum()
        return pd.Series(out, index=g.index)

    df["y_window_mean"] = df.groupby("pitcher_id", group_keys=False).apply(forward_window_mean)
    return df


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = _prep(hist)
    df = df.dropna(subset=["prior_wpa_per_pa", "y_window_mean"])

    league_mean = df["prior_wpa_per_pa"].mean()
    print(f"n = {len(df)}")
    print(f"league_mean_wpa_per_pa(bullpen) = {league_mean:.6f}\n")

    baseline = evaluate(df[["prior_wpa_per_pa"]].to_numpy(), df["y_window_mean"].to_numpy())

    results = {
        "league_mean_wpa_per_pa": float(league_mean),
        "n": int(len(df)),
        "shrinkage_unit": "prior_n_pa",
        "target": "y_window_mean (forward 5-outing avg)",
        "no_shrinkage_baseline": baseline,
        "k_results": {},
    }

    print(f"{'k':>6s} {'Spearman r':>11s} {'p-value':>10s} {'CV R2':>9s}")
    print("-" * 42)
    print(f"{'없음':>6s} {baseline['spearman_r']:>11.4f} {baseline['p_value']:>10.4g} "
          f"{baseline['cv_r2_mean']:>9.4f}   (대조군: shrinkage 미적용)")

    best_k, best_r = None, -np.inf
    for k in K_CANDIDATES:
        w = df["prior_n_pa"] / (df["prior_n_pa"] + k)
        shrunk = w * df["prior_wpa_per_pa"] + (1 - w) * league_mean
        res = evaluate(shrunk.to_numpy().reshape(-1, 1), df["y_window_mean"].to_numpy())
        results["k_results"][str(k)] = res
        print(f"{k:>6d} {res['spearman_r']:>11.4f} {res['p_value']:>10.4g} {res['cv_r2_mean']:>9.4f}")
        if res["spearman_r"] > best_r:
            best_r, best_k = res["spearman_r"], k

    results["best_k"] = best_k
    print(f"\n최적 k = {best_k} (Spearman r = {best_r:.4f})")
    print("판정 기준: r>0.1 & p<0.05")
    passed = best_r > 0.1 and results["k_results"][str(best_k)]["p_value"] < 0.05
    print(f"최적 k 기준 통과 여부: {'PASS' if passed else 'FAIL'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
