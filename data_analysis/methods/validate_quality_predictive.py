"""
투수 품질(quality_p) 후보의 실제 성과 예측력 검증 — Task 2.

Task 1은 "같은 지표가 시간이 지나도 일관되는가"(자기상관)를 봤다.
Task 2는 한 단계 더 나아가 "그 시점까지의 prior_* 기록으로, 바로 다음
등판의 실제 성과를 예측할 수 있는가"를 직접 검증한다.

all_pitcher_history.parquet의 각 행은 이미 "이 등판 이전까지의 누적 기록"
(prior_*, leakage-free)과 "이 등판 자체의 실제 결과"(app_wpa, n_pa)를
같이 갖고 있다. 즉 X=prior_*(해당 행), y=이 행 자신의 실제 성과로 두면
그게 곧 "과거 기록으로 다음 등판을 예측"하는 것과 동일하다(prior_*가
정의상 이 행보다 이전 등판만 반영하므로).

y = -app_wpa / n_pa  (공격팀 관점 값을 투수 관점으로 부호 반전, 등판
     길이로 정규화 — HANDOFF_FULL.md 함정 1 방지)

기준: Spearman r > 0.1, p < 0.05.
주의: 회귀 R²는 낮게 나올 수 있다. 순위 상관이 유의하면 사용 가능
(추천 시스템엔 순위만 맞으면 충분하기 때문 — 정확한 값 예측이 목적이 아님).

prior_wpa_per_pa는 새로 원본 데이터를 재계산하는 게 아니라, 이미 존재하는
leakage-free 컬럼(prior_wpa_mean, prior_n_apps, prior_n_pa)을 대수적으로
조합해 유도한다: prior_wpa_sum = prior_wpa_mean * prior_n_apps,
prior_wpa_per_pa = -prior_wpa_sum / prior_n_pa. prior_* 컬럼 자체를
재계산하는 게 아니므로 금지사항("prior_* 재계산 금지")에 해당하지 않는다.
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
OUT_PATH = Path("data_analysis/results/quality_candidate_predictive.json")

RNG_SEED = 42


def _prep(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist[hist["prior_n_apps"] >= 1].copy()
    df["y_wpa_per_pa"] = -df["app_wpa"] / df["n_pa"]  # 투수 관점, 이 등판 자체의 실제 성과
    df["prior_wpa_sum"] = df["prior_wpa_mean"] * df["prior_n_apps"]
    df["prior_wpa_per_pa"] = -df["prior_wpa_sum"] / df["prior_n_pa"]
    df["prior_so_minus_bb"] = df["prior_so_rate"] - df["prior_bb_rate"]
    df["prior_composite"] = df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]
    return df


# (컬럼명, 기대 부호) — "높을수록 나쁜" 지표(볼넷/홈런/원본WPA)는 기대 부호가 음수다.
# 이걸 반영 안 하면 진짜 신호를 "부호가 반대"라는 이유로 fail 처리하는 오판이 생긴다.
CANDIDATES: list[tuple[str, int]] = [
    ("prior_so_rate", +1),
    ("prior_bb_rate", -1),        # 볼넷 많을수록 나쁨 → 음의 상관이 정상
    ("prior_hr_rate", -1),        # 피홈런 많을수록 나쁨 → 음의 상관이 정상
    ("prior_so_minus_bb", +1),
    ("prior_composite", +1),
    ("prior_wpa_mean", -1),       # 공격팀 관점 원본값(부호 반전 안 됨) → 음의 상관이 정상
    ("prior_wpa_per_pa", +1),     # 이미 투수 관점으로 부호 반전된 파생값
    ("prior_n_apps", 0),          # 방향성 가설 없음(볼륨 지표) — 절대값 기준으로만 참고
]


def evaluate(df: pd.DataFrame, col: str) -> dict:
    sub = df[[col, "y_wpa_per_pa"]].dropna()
    if len(sub) < 30:
        return {"n": len(sub), "spearman_r": None, "p_value": None, "cv_r2_mean": None}

    r, p = spearmanr(sub[col], sub["y_wpa_per_pa"])

    X = sub[[col]].to_numpy()
    y = sub["y_wpa_per_pa"].to_numpy()
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

    results: dict = {"criteria": "Spearman r > 0.1 and p < 0.05", "groups": {}}

    groups = {
        "전체(pooled, 참고용)": df,
        "선발만": df[df["is_starter"]],
        "불펜만": df[~df["is_starter"]],
    }

    for gname, gdf in groups.items():
        print(f"\n=== {gname} (n={len(gdf)}) ===")
        print(f"{'후보':22s} {'n':>7s} {'Spearman r':>11s} {'p-value':>10s} {'CV R2':>9s}  {'기준통과(부호반영)'}")
        print("-" * 85)
        results["groups"][gname] = {}
        for col, expected_sign in CANDIDATES:
            res = evaluate(gdf, col)
            results["groups"][gname][col] = res
            if res["spearman_r"] is None:
                print(f"{col:22s} {'N/A':>7s}")
                continue
            r = res["spearman_r"]
            signed_r = r * expected_sign if expected_sign != 0 else abs(r)
            passed = signed_r > 0.1 and res["p_value"] < 0.05
            mark = "PASS" if passed else "fail"
            print(
                f"{col:22s} {res['n']:>7d} {r:>11.4f} "
                f"{res['p_value']:>10.4g} {res['cv_r2_mean']:>9.4f}  {mark}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
