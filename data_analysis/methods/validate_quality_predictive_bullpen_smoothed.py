"""
투수 품질(quality_p) 불펜 후보 — 목표를 "다음 등판 평균"으로 바꿔 재검증.

validate_quality_predictive.py(Task 2)는 y=바로 다음 등판 1번의 실제
wpa_per_pa로 뒀다. 불펜은 등판 1번당 평균 4.56타자(선발은 22.9타자)라,
등판 1번의 결과가 실력보다 그날의 표본노이즈(운)에 훨씬 크게 좌우된다.
이건 원래 지표 문제가 아니라 "등판 1번"이라는 목표 단위 자체가 불펜한테는
너무 작다는 문제일 수 있다 — 실제로 22.9/4.56 ≈ 5.02이므로, 선발 1경기와
비슷한 표본이 되려면 불펜은 등판 5번을 봐야 한다.

그래서 y를 "다음 등판 1번" 대신 "이 시점부터 등판 N=5회 평균 wpa_per_pa"로
바꿔서 같은 방식(Spearman, 5-fold CV R², 동일 candidates)으로 재검증한다.
X(prior_*)는 이전과 동일 — 그 시점까지의 누적 기록.

주의: 이건 "통과할 때까지 기준을 만지는" 게 아니라 "목표 단위를 등판
사이즈에 맞게 재정의"하는 것이지만, 이래도 기준(r>0.1, p<0.05) 미달이면
"불펜 지표에 원래 신호가 없다"는 결론을 강화하는 진짜 실패로 취급한다.

기준: Spearman r > 0.1(부호 반영), p < 0.05 — Task 2와 동일.
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
OUT_PATH = Path("data_analysis/results/quality_candidate_predictive_bullpen_smoothed.json")

WINDOW_N = 5  # 불펜 평균 4.56타자/등판 x 5 ≈ 선발 평균 22.9타자/등판
RNG_SEED = 42

CANDIDATES: list[tuple[str, int]] = [
    ("prior_so_rate", +1),
    ("prior_bb_rate", -1),
    ("prior_hr_rate", -1),
    ("prior_composite", +1),
    ("prior_wpa_per_pa", +1),
]


def _prep(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist[(~hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df = df.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    df["y_this_outing"] = -df["app_wpa"] / df["n_pa"]
    df["prior_wpa_sum"] = df["prior_wpa_mean"] * df["prior_n_apps"]
    df["prior_wpa_per_pa"] = -df["prior_wpa_sum"] / df["prior_n_pa"]
    df["prior_composite"] = df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]

    # 투수별로 "이 등판부터 N번(이 등판 포함) 평균 실제 성과" 계산.
    # 표본 가중(타자 수 다른 등판을 그냥 평균 내면 안 됨) — n_pa로 가중평균.
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


def evaluate(df: pd.DataFrame, col: str) -> dict:
    sub = df[[col, "y_window_mean"]].dropna()
    if len(sub) < 30:
        return {"n": len(sub), "spearman_r": None, "p_value": None, "cv_r2_mean": None}

    r, p = spearmanr(sub[col], sub["y_window_mean"])

    X = sub[[col]].to_numpy()
    y = sub["y_window_mean"].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")

    return {
        "n": int(len(sub)),
        "spearman_r": float(r),
        "p_value": float(p),
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
    }


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = _prep(hist)
    n_with_window = df["y_window_mean"].notna().sum()
    print(f"불펜 행 전체: {len(df)}  /  전방 {WINDOW_N}등판 윈도우 확보된 행: {n_with_window}\n")

    results: dict = {
        "criteria": "Spearman r > 0.1 (signed) and p < 0.05",
        "window_n_outings": WINDOW_N,
        "window_rationale": "선발 평균 22.9타자/등판 ≈ 불펜 평균 4.56타자/등판 x 5",
        "candidates": {},
    }

    print(f"{'후보':22s} {'n':>7s} {'Spearman r':>11s} {'p-value':>10s} {'CV R2':>9s}  판정")
    print("-" * 80)
    for col, expected_sign in CANDIDATES:
        res = evaluate(df, col)
        results["candidates"][col] = res
        if res["spearman_r"] is None:
            print(f"{col:22s} {'N/A':>7s}")
            continue
        r = res["spearman_r"]
        signed_r = r * expected_sign if expected_sign != 0 else abs(r)
        passed = signed_r > 0.1 and res["p_value"] < 0.05
        mark = "PASS" if passed else "fail"
        print(f"{col:22s} {res['n']:>7d} {r:>11.4f} {res['p_value']:>10.4g} "
              f"{res['cv_r2_mean']:>9.4f}  {mark}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
