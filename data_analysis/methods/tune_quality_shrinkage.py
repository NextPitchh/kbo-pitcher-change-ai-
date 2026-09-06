"""
투수 품질(quality_p) shrinkage 계수 k 탐색 — Task 4.

등판 수(prior_n_apps)가 적은 투수의 prior_composite는 신뢰하기 어렵다
(Task 3에서 확인한 극단치 사례: 등판 1번에 볼넷만 내준 경우 raw=-1.0).
이걸 리그 평균 쪽으로 당겨서(shrinkage) 완화한다.

    w = prior_n_apps / (prior_n_apps + k)
    quality_shrunk = w * prior_composite + (1 - w) * league_mean_composite

k가 클수록 리그 평균 쪽으로 더 세게 당겨진다(=적은 표본을 더 안 믿음).
k=5,10,15,20 각각에 대해 Task 2와 동일한 방식(Spearman, 5-fold CV R²,
y=다음 등판 실제 wpa_per_pa)으로 예측력을 재서, 가장 좋은 k를 찾는다.

적용 순서: 원값에 shrinkage를 먼저 적용한 뒤, Task 3에서 정한 스케일링
(백분위 x 2)을 그 위에 적용한다. shrinkage는 "값 자체를 더 믿을 만하게
다듬는" 단계이고 스케일링은 "다듬어진 값의 크기를 맞추는" 단계라, 순서가
바뀌면 안 된다(스케일링 후 값을 다시 shrink하면 백분위의 '평균=0.5'라는
성질 때문에 의미가 달라진다).

league_mean_composite는 시점별로 바뀌는 값이 아니라 전체 모집단 평균
(고정 상수)이다 — li_league_avg, handedness_baseline 등 프로젝트 전반에서
쓰인 것과 동일한 방식이라 새로운 종류의 leakage를 만들지 않는다.
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
OUT_PATH = Path("data_analysis/results/quality_shrinkage_tuning.json")

K_CANDIDATES = [5, 10, 15, 20]
RNG_SEED = 42


def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    r, p = spearmanr(X.ravel(), y)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    return {"spearman_r": float(r), "p_value": float(p),
            "cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std())}


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = hist[(hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df["prior_composite"] = (
        df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]
    )
    df["y_wpa_per_pa"] = -df["app_wpa"] / df["n_pa"]

    league_mean = df["prior_composite"].mean()
    print(f"league_mean_composite = {league_mean:.6f}\n")

    # 대조군: shrinkage 없이(원값 그대로)
    baseline = evaluate(df[["prior_composite"]].to_numpy(), df["y_wpa_per_pa"].to_numpy())

    results = {
        "league_mean_composite": float(league_mean),
        "n": int(len(df)),
        "no_shrinkage_baseline": baseline,
        "k_results": {},
    }

    print(f"{'k':>6s} {'Spearman r':>11s} {'p-value':>10s} {'CV R2':>9s}")
    print("-" * 42)
    print(f"{'없음':>6s} {baseline['spearman_r']:>11.4f} {baseline['p_value']:>10.4g} "
          f"{baseline['cv_r2_mean']:>9.4f}   (대조군: shrinkage 미적용)")

    best_k, best_r = None, -np.inf
    for k in K_CANDIDATES:
        w = df["prior_n_apps"] / (df["prior_n_apps"] + k)
        shrunk = w * df["prior_composite"] + (1 - w) * league_mean
        res = evaluate(shrunk.to_numpy().reshape(-1, 1), df["y_wpa_per_pa"].to_numpy())
        results["k_results"][str(k)] = res
        print(f"{k:>6d} {res['spearman_r']:>11.4f} {res['p_value']:>10.4g} {res['cv_r2_mean']:>9.4f}")
        if res["spearman_r"] > best_r:
            best_r, best_k = res["spearman_r"], k

    results["best_k"] = best_k
    print(f"\n최적 k = {best_k} (Spearman r = {best_r:.4f})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
