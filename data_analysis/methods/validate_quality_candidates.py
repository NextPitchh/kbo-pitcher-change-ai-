"""
투수 품질(quality_p) 후보 지표 자기상관 재검증 — Task 1.

prior_wpa_mean 탈락 배경: 자기상관 r=0.83으로 높게 보였으나, 이는 "선발은
계속 선발로, 불펜은 계속 불펜으로 나온다"는 역할 고정 효과를 실력으로
착각한 것이었다. 역할(선발/불펜)을 통제(분리)하면 r이 불펜 0.11 / 선발
-0.07로 사라졌다.

방법: 투수별 등판을 시간순으로 정렬해 전반부/후반부로 절반 분할하고,
전반부 값과 후반부 값의 Spearman 상관(투수 간 비교)을 잰다. 상관이 높다는
건 "잘하는 투수는 계속 잘한다"는 진짜 실력 신호라는 뜻이고, 낮다는 건
노이즈이거나 다른 교란요인(역할)의 산물이라는 뜻이다.

반드시 is_starter로 나눠서(선발끼리만, 불펜끼리만) 이 상관을 재는 것이
핵심이다 (금지사항: "역할 통제 없이 상관 계산").

app_wpa는 등판 길이에 비례하는 합계값이라 그대로 쓰면 안 된다
(HANDOFF_FULL.md 함정 1). wpa_per_pa = -app_wpa.sum()/n_pa.sum()로 정규화한
값(투수 관점, 부호 반전)을 "ERA 대용" 후보로 쓴다.

대조군: wpa_mean_raw(=mean(app_wpa), 등판당 평균 — 정규화 없이 원래
prior_wpa_mean과 동일한 정의)를 역할 통제 없이/있이 둘 다 계산해서, 탈락이
실제로 재현되는지 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OUT_PATH = Path("data_analysis/results/quality_candidate_autocorrelation.json")

MIN_APPS = 10  # 절반 분할 시 최소 5+5


def _split_half(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = g.sort_values(["date", "game_id"])
    half = len(g) // 2
    return g.iloc[:half], g.iloc[half:]


# ── 후보 지표 계산 함수들 (등판 묶음 -> 스칼라값) ───────────────────────────

def m_so_rate(g: pd.DataFrame) -> float:
    return g["n_so"].sum() / g["n_pa"].sum()


def m_bb_rate(g: pd.DataFrame) -> float:
    return g["n_bb"].sum() / g["n_pa"].sum()


def m_hr_rate(g: pd.DataFrame) -> float:
    return g["n_hr"].sum() / g["n_pa"].sum()


def m_so_minus_bb(g: pd.DataFrame) -> float:
    return m_so_rate(g) - m_bb_rate(g)


def m_wpa_mean_raw(g: pd.DataFrame) -> float:
    """원래 prior_wpa_mean과 동일한 정의: 등판당 app_wpa 평균 (정규화 없음)."""
    return g["app_wpa"].mean()


def m_wpa_per_pa(g: pd.DataFrame) -> float:
    """투수 관점, 타석수로 정규화 (ERA 대용)."""
    return -g["app_wpa"].sum() / g["n_pa"].sum()


def m_composite(g: pd.DataFrame) -> float:
    """탐색적 가중합: 삼진은 +1, 볼넷은 -1, 피홈런은 -2 (선형가중치 근사, 미세조정 아님)."""
    return m_so_rate(g) - m_bb_rate(g) - 2 * m_hr_rate(g)


def m_n_apps(g: pd.DataFrame) -> float:
    return float(len(g))


CANDIDATES = {
    "prior_so_rate": m_so_rate,
    "prior_bb_rate": m_bb_rate,
    "prior_hr_rate": m_hr_rate,
    "so_rate_minus_bb_rate": m_so_minus_bb,
    "wpa_mean_raw(=구 prior_wpa_mean 정의)": m_wpa_mean_raw,
    "wpa_per_pa(ERA 대용)": m_wpa_per_pa,
    "composite(so-bb-2*hr)": m_composite,
    "n_apps_단독": m_n_apps,
}


def split_half_correlation(df: pd.DataFrame, metric_fn, min_apps: int = MIN_APPS):
    v1s, v2s, ns = [], [], []
    for pid, g in df.groupby("pitcher_id"):
        if len(g) < min_apps:
            continue
        first, second = _split_half(g)
        v1, v2 = metric_fn(first), metric_fn(second)
        if v1 is None or v2 is None or np.isnan(v1) or np.isnan(v2):
            continue
        v1s.append(v1)
        v2s.append(v2)
        ns.append(len(g))
    if len(v1s) < 5:
        return {"n_pitchers": len(v1s), "spearman_r": None, "p_value": None}
    r, p = spearmanr(v1s, v2s)
    return {"n_pitchers": len(v1s), "spearman_r": float(r), "p_value": float(p)}


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    hist = hist.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    results: dict = {"min_apps_threshold": MIN_APPS, "candidates": {}}

    starters = hist[hist["is_starter"]]
    relievers = hist[~hist["is_starter"]]

    print(f"{'후보 지표':35s} {'선발 r':>10s} {'선발 p':>10s} {'선발 n':>8s}   "
          f"{'불펜 r':>10s} {'불펜 p':>10s} {'불펜 n':>8s}")
    print("-" * 105)

    for name, fn in CANDIDATES.items():
        r_start = split_half_correlation(starters, fn)
        r_relief = split_half_correlation(relievers, fn)
        results["candidates"][name] = {"starter": r_start, "reliever": r_relief}

        def fmt(d):
            if d["spearman_r"] is None:
                return "N/A", "N/A", str(d["n_pitchers"])
            return f"{d['spearman_r']:.3f}", f"{d['p_value']:.4f}", str(d["n_pitchers"])

        sr, sp, sn = fmt(r_start)
        rr, rp, rn = fmt(r_relief)
        print(f"{name:35s} {sr:>10s} {sp:>10s} {sn:>8s}   {rr:>10s} {rp:>10s} {rn:>8s}")

    # 대조군: 역할 통제 없이(pooled) wpa_mean_raw — 원래 탈락 전 r=0.83 재현 확인
    pooled = hist  # starters + relievers 섞음
    pooled_r = split_half_correlation(pooled, m_wpa_mean_raw)
    results["pooled_wpa_mean_raw_no_role_control"] = pooled_r
    print("\n[대조군] wpa_mean_raw, 역할 통제 없이(선발+불펜 혼합):")
    print(f"  r={pooled_r['spearman_r']:.3f}, p={pooled_r['p_value']:.4g}, n={pooled_r['n_pitchers']}")
    print("  (참고: 과거 발견된 자기상관 0.83과 유사한 수준이 재현되는지 확인용)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
