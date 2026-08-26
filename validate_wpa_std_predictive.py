"""
KBO WPA Pipeline — Task 1: prior_wpa_std 예측력 검증 (역할 분리 제안의 전제 조건)

v6 피처 확장이 시뮬레이션 ratio를 개선하지 못했다(0.049→0.039, 후퇴). 새 제안은
E(기대값)는 시뮬레이션, Std(변동성)는 prior_wpa_std(과거 기록)로 역할을 분리하자는
것이다. 이 제안 전체는 "prior_wpa_std가 실제 미래 변동성을 예측하는가"에 달려
있다 — 이걸 확인하지 않고 Task2~4로 넘어가면 안 된다(사용자 명시 금지 사항).

방법: 각 투수의 등판을 시간순으로 반씩 나눠 전반부 app_wpa std(=prior_wpa_std
역할) vs 후반부 app_wpa std(=실제 미래 변동성)의 상관계수를 측정한다.

1-1) 전체 모집단 (등판 30회 이상)
1-2) 조건 통제 모집단 (불펜·3~7타석·6~8회·|점수차|<=3, 20회 이상) — 등판 길이 효과 제거

기준: 상관 0.3 이상이면 예측력 있음. 0에 가까우면 prior_wpa_std는 노이즈이므로
전체 제안이 무효.

절대 prior_*를 재계산하지 않는다 — 이미 leakage-free로 검증된 값을 그대로 쓴다.
여기서 재구성하는 "전반부/후반부 std"는 prior_wpa_std 자체가 아니라 그 예측력을
독립적으로 검증하기 위한 것(같은 방식이 아니라 out-of-sample 검증).

실행: python validate_wpa_std_predictive.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Task 1: prior_wpa_std 예측력 검증 — 역할 분리 제안의 전제 조건")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)
df["_date"] = df["game_id"].str[:8].astype(int)

first_pa = df[df["current_pitcher_pa_in_app"] == 1].copy()
first_pa["abs_score_diff_start"] = first_pa["score_diff_attacker"].abs()

report = {}


def split_half_correlation(apps_df: pd.DataFrame, min_appearances: int, label: str) -> dict:
    counts = apps_df.groupby("pitcher_id").size()
    eligible = counts[counts >= min_appearances].index
    sub = apps_df[apps_df["pitcher_id"].isin(eligible)].copy()

    rows = []
    for pid, g in sub.groupby("pitcher_id"):
        g = g.sort_values("_date")
        n = len(g)
        mid = n // 2
        first_half = g.iloc[:mid]
        second_half = g.iloc[mid:]
        if len(first_half) < 2 or len(second_half) < 2:
            continue
        std_first = first_half["app_wpa"].std(ddof=1)
        std_second = second_half["app_wpa"].std(ddof=1)
        if pd.isna(std_first) or pd.isna(std_second):
            continue
        rows.append({
            "pitcher_id": pid, "n_total": n,
            "n_first": len(first_half), "n_second": len(second_half),
            "std_first_half": float(std_first), "std_second_half": float(std_second),
        })

    result_df = pd.DataFrame(rows)
    n_pitchers = len(result_df)

    pearson_r, pearson_p = stats.pearsonr(result_df["std_first_half"], result_df["std_second_half"])
    spearman_r, spearman_p = stats.spearmanr(result_df["std_first_half"], result_df["std_second_half"])

    print(f"\n[{label}] 투수 {n_pitchers}명 (등판 {min_appearances}회 이상)")
    print(f"  전반부 std 분포: min={result_df['std_first_half'].min():.4f} "
          f"25%={result_df['std_first_half'].quantile(.25):.4f} "
          f"median={result_df['std_first_half'].median():.4f} "
          f"75%={result_df['std_first_half'].quantile(.75):.4f} "
          f"max={result_df['std_first_half'].max():.4f}")
    print(f"  후반부 std 분포: min={result_df['std_second_half'].min():.4f} "
          f"25%={result_df['std_second_half'].quantile(.25):.4f} "
          f"median={result_df['std_second_half'].median():.4f} "
          f"75%={result_df['std_second_half'].quantile(.75):.4f} "
          f"max={result_df['std_second_half'].max():.4f}")
    print(f"  Pearson  r={pearson_r:.4f}  p={pearson_p:.2e}")
    print(f"  Spearman r={spearman_r:.4f}  p={spearman_p:.2e}")

    verdict = "예측력 있음 (r>=0.3)" if pearson_r >= 0.3 else "예측력 부족 (r<0.3) — 노이즈에 가까움"
    print(f"  ▶ {verdict}")

    return {
        "label": label, "min_appearances": min_appearances, "n_pitchers": n_pitchers,
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "std_first_half_stats": result_df["std_first_half"].describe().to_dict(),
        "std_second_half_stats": result_df["std_second_half"].describe().to_dict(),
        "predictive": bool(pearson_r >= 0.3),
        "per_pitcher": result_df.to_dict(orient="records"),
    }


# ──────────────────────────────────────────────
# 1-1) 전체 모집단
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("1-1) 전체 모집단 (등판 30회 이상)")
print("=" * 70)
r1 = split_half_correlation(first_pa, min_appearances=30, label="전체 모집단")
report["1-1_full_population"] = r1

# ──────────────────────────────────────────────
# 1-2) 조건 통제 모집단 (불펜·3~7타석·6~8회·|점수차|<=3, 20회 이상)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("1-2) 조건 통제 모집단 (불펜·3~7타석·6~8회·|점수차|<=3, 20회 이상)")
print("=" * 70)

cond_all = (
    (first_pa["is_starter"] == False) &
    (first_pa["n_pa"] >= 3) & (first_pa["n_pa"] <= 7) &
    (first_pa["start_inning"] >= 6) & (first_pa["start_inning"] <= 8) &
    (first_pa["abs_score_diff_start"] <= 3)
)
subset_condition = first_pa[cond_all]
r2 = split_half_correlation(subset_condition, min_appearances=20, label="조건 통제 모집단")
report["1-2_condition_controlled"] = r2

# ──────────────────────────────────────────────
# 결론 — 게이트
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("게이트 판정")
print("=" * 70)

gate_pass = r1["predictive"] and r2["predictive"]
print(f"\n1-1 (전체): r={r1['pearson_r']:.4f}  {'PASS' if r1['predictive'] else 'FAIL'}")
print(f"1-2 (조건통제): r={r2['pearson_r']:.4f}  {'PASS' if r2['predictive'] else 'FAIL'}")
print(f"\n▶ 게이트 {'통과 — Task2~4 진행 가능' if gate_pass else '실패 — Task2~4 진행 중단, 역할 분리 제안 무효'}")

report["gate"] = {
    "pass": gate_pass,
    "rule": "Pearson r >= 0.3 in both 1-1 and 1-2",
}

with open(OUTPUT_DIR / "wpa_std_predictive_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 완료! {OUTPUT_DIR / 'wpa_std_predictive_report.json'}")
