"""
투수 품질(quality_p) shrinkage 계수 k 탐색 — 불펜판 (Task 4-B).

선발용 tune_quality_shrinkage.py는 표본 크기를 등판 수(prior_n_apps)로
잰다. 불펜은 등판당 상대 타자 수 편차가 커서(1타자 원포인트 ~ 롱릴리프
10타자 이상) 등판 횟수가 실제 표본 크기를 잘 대변하지 못한다는 문제가
2026-09 팀 회의에서 지적됨. 이 스크립트는 표본 크기를 등판 수 대신
누적 상대 타석 수(prior_n_pa)로 바꿔서 같은 방식(Spearman, 5-fold CV R²,
y=이 등판 자체의 실제 wpa_per_pa)으로 k를 재탐색한다.

    w = prior_n_pa / (prior_n_pa + k)
    quality_shrunk = w * prior_composite + (1 - w) * league_mean_composite

prior_n_pa는 prior_n_apps보다 스케일이 약 5~6배 크므로(median 473 vs 86,
data_analysis/all_pitcher_history.parquet 기준) 선발의 k=5를 그대로
재사용할 수 없다 — 후보군을 그 스케일에 맞춰 새로 잡는다.

주의: 이 스크립트는 shrinkage 단위 교체 효과만 검증한다. Task 2에서
불펜이 예측력 기준(r>0.1, p<0.05)을 raw 기준으로 통과하지 못했던 것과는
별개 질문 — shrink한 값이 그 기준을 통과하는지는 여기 결과로 별도 판단.
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
OUT_PATH = Path("data_analysis/results/quality_shrinkage_tuning_bullpen.json")

K_CANDIDATES = [25, 50, 100, 200, 300, 500]
RNG_SEED = 42


def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    r, p = spearmanr(X.ravel(), y)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    return {"spearman_r": float(r), "p_value": float(p),
            "cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = hist[(~hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df["prior_composite"] = (
        df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]
    )
    df["y_wpa_per_pa"] = -df["app_wpa"] / df["n_pa"]

    league_mean = df["prior_composite"].mean()
    print(f"n = {len(df)}")
    print(f"league_mean_composite(bullpen) = {league_mean:.6f}\n")

    # 대조군: shrinkage 없이(원값 그대로) — Task 2 raw 결과 재확인용
    baseline = evaluate(df[["prior_composite"]].to_numpy(), df["y_wpa_per_pa"].to_numpy())

    results = {
        "league_mean_composite": float(league_mean),
        "n": int(len(df)),
        "shrinkage_unit": "prior_n_pa",
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
        shrunk = w * df["prior_composite"] + (1 - w) * league_mean
        res = evaluate(shrunk.to_numpy().reshape(-1, 1), df["y_wpa_per_pa"].to_numpy())
        results["k_results"][str(k)] = res
        print(f"{k:>6d} {res['spearman_r']:>11.4f} {res['p_value']:>10.4g} {res['cv_r2_mean']:>9.4f}")
        if res["spearman_r"] > best_r:
            best_r, best_k = res["spearman_r"], k

    results["best_k"] = best_k
    print(f"\n최적 k = {best_k} (Spearman r = {best_r:.4f})")
    print("\n판정 기준(Task 2와 동일): r>0.1(방향 반영) & p<0.05")
    passed = best_r > 0.1 and results["k_results"][str(best_k)]["p_value"] < 0.05
    print(f"최적 k 기준 통과 여부: {'PASS' if passed else 'FAIL'}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
