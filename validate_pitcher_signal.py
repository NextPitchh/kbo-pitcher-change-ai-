"""
KBO WPA Pipeline — 투수 간 신호가 왜 작은가 (between/within ≈ 0.06 원인 규명)

validate_horizon.py에서 horizon을 좁혀도 투수 간 격차(between/within)가
개선되지 않았다 (h5_1=0.072, h10_2=0.056, full=0.057). horizon 문제가
아니라면 원인은 (1) 모델이 투수를 구별 못함, (2) 표본 투수 8명이 우연히
비슷한 유형, (3) 야구 자체의 본질적 특성 중 하나다. 세 가지를 각각 검증한다.

절대 placeholder ID를 쓰지 않는다 — all_pa_enriched.parquet의 실제
pitcher_id/batter_id만 사용한다 (CLAUDE.md, HANDOFF_FULL.md 컨벤션).

실행: python validate_pitcher_signal.py
요구사항: pip install catboost pandas numpy pyarrow
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monte_carlo_simulate import MonteCarloSimulator, GameState

OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42
MIN_APPEARANCES = 30

report = {}

print("=" * 70)
print("투수 간 신호 규명: 모델 한계 vs 표본 편향 vs 야구의 본질")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)  # groupby/loc 키를 GameState의 str(pitcher_id)와 일치시킴

apps = df.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"]).copy()
app_counts = apps.groupby("pitcher_id").size()
eligible_ids = app_counts[app_counts >= MIN_APPEARANCES].index
apps_elig = apps[apps["pitcher_id"].isin(eligible_ids)].copy()

print(f"\n등판 {MIN_APPEARANCES}회 이상 투수: {len(eligible_ids)}명 "
      f"(등판 레코드 {len(apps_elig)}건)")


# ──────────────────────────────────────────────
# Task 1: CatBoost 확률 벡터의 투수별 분화 측정
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 1: 확률 벡터의 투수별 분화 (고정 상황: 7회말 1아웃 1루 -1점)")
print("=" * 70)

sim = MonteCarloSimulator(n_simulations=1, rng_seed=RNG_SEED)

rng = np.random.default_rng(RNG_SEED)
pitchers_30 = rng.choice(eligible_ids.values, size=30, replace=False)

# 타자 문맥 고정: 데이터에서 가장 흔한 실제 배터 ID + 중앙값 근처 매치업 수치
FIXED_BATTER_ID = str(int(df["batter_id"].value_counts().index[0]))
FIXED_BATTER_HIT_TYPE = "R"
FIXED_PVB = 0.250
FIXED_BRA = 0.317

fixed_kwargs = dict(
    inning=7, half="bot", out_count=1, base_state="1", score_diff_attacker=-1,
    batter_id=FIXED_BATTER_ID, batter_hit_type=FIXED_BATTER_HIT_TYPE,
    pitcher_vs_batter_avg=FIXED_PVB, batter_recent_avg=FIXED_BRA,
)

states_30 = [GameState(pitcher_id=str(int(pid)), **fixed_kwargs) for pid in pitchers_30]
batch_rows = [
    [str(feat.get(c, 0)) if c in sim.cat_features else float(feat.get(c, 0))
     for c in sim.all_features]
    for feat in (s.to_feature_row() for s in states_30)
]
proba_matrix = sim.model.predict_proba(batch_rows)  # (30, 9)
proba_matrix = proba_matrix / proba_matrix.sum(axis=1, keepdims=True)

proba_df = pd.DataFrame(proba_matrix, columns=sim.classes,
                         index=[str(int(p)) for p in pitchers_30])

class_std = proba_df.std(axis=0).sort_values(ascending=False)
print("\n클래스별 확률의 투수 간 표준편차 (30명, 내림차순):")
for cls, s in class_std.items():
    print(f"  {cls:5s}: std={s:.5f}  mean={proba_df[cls].mean():.5f}")

max_class, max_std = class_std.index[0], float(class_std.iloc[0])
min_class, min_std = class_std.index[-1], float(class_std.iloc[-1])
all_below_001 = bool((class_std < 0.01).all())

# 쌍별 L1 거리 / KL divergence (평균)
eps = 1e-9
P = proba_matrix
n_p = P.shape[0]
l1_sum, kl_sum, n_pairs = 0.0, 0.0, 0
for i in range(n_p):
    for j in range(i + 1, n_p):
        p, q = P[i] + eps, P[j] + eps
        l1_sum += np.abs(p - q).sum()
        kl_sum += 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
        n_pairs += 1
mean_l1 = l1_sum / n_pairs
mean_kl_sym = kl_sum / n_pairs

print(f"\n평균 쌍별 L1 거리: {mean_l1:.5f}")
print(f"평균 쌍별 대칭 KL divergence: {mean_kl_sym:.5f}")
print(f"\n가장 차이 큰 클래스: {max_class} (std={max_std:.5f})")
print(f"가장 차이 작은 클래스: {min_class} (std={min_std:.5f})")
print(f"모든 클래스 std < 0.01: {'예 — 모델이 투수를 사실상 구별 못함' if all_below_001 else '아니오'}")

report["task1_probability_divergence"] = {
    "n_pitchers": 30,
    "fixed_situation": fixed_kwargs,
    "class_std": {k: float(v) for k, v in class_std.items()},
    "class_mean": {k: float(proba_df[k].mean()) for k in sim.classes},
    "mean_pairwise_l1": float(mean_l1),
    "mean_pairwise_kl_symmetric": float(mean_kl_sym),
    "max_std_class": max_class, "max_std_value": max_std,
    "min_std_class": min_class, "min_std_value": min_std,
    "all_classes_std_below_0.01": all_below_001,
}


# ──────────────────────────────────────────────
# Task 2: 투수 선정 편향 확인
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 2: 투수 선정 편향 — validate_horizon.py의 8명은 비슷한 유형인가")
print("=" * 70)

# 투수별 실제 성적(등판 기록 평균) + prior_wpa_std 평균
prior_cols = ["prior_wpa_std", "prior_bb_rate", "prior_so_rate", "prior_hr_rate"]
pitcher_stats = apps_elig.groupby("pitcher_id")[prior_cols].mean()

def load_veteran_pitchers(df_: pd.DataFrame, n: int = 8, min_appearances: int = 30,
                           seed: int = 42) -> list[dict]:
    apps_ = df_.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"])
    counts_ = apps_.groupby("pitcher_id").size()
    eligible_ = counts_[counts_ >= min_appearances].index
    avg_by_pitcher_ = (
        df_[df_["pitcher_id"].isin(eligible_)]
        .groupby("pitcher_id")["pitcher_vs_batter_avg"].mean()
    )
    rng_ = np.random.default_rng(seed)
    chosen_ = rng_.choice(avg_by_pitcher_.index.values, size=n, replace=False)
    return [
        {"pitcher_id": str(int(pid)), "pitcher_vs_batter_avg": float(avg_by_pitcher_.loc[pid]),
         "n_appearances": int(counts_.loc[pid])}
        for pid in chosen_
    ]

pitchers_8 = load_veteran_pitchers(df, n=8, min_appearances=MIN_APPEARANCES, seed=RNG_SEED)
pid8 = [p["pitcher_id"] for p in pitchers_8]

print(f"\n(참고) validate_horizon.py 3-2에서 쓴 8명: {pid8}")
print(f"\n{'pitcher_id':>10s} {'wpa_std':>9s} {'wpa_std%ile':>11s} "
      f"{'bb_rate':>9s} {'bb%ile':>7s} {'so_rate':>9s} {'so%ile':>7s} "
      f"{'hr_rate':>9s} {'hr%ile':>7s}")
pop_pct = {c: pitcher_stats[c].rank(pct=True) for c in prior_cols}
pitcher8_detail = []
for pid in pid8:
    row = pitcher_stats.loc[pid]
    pct = {c: float(pop_pct[c].loc[pid]) * 100 for c in prior_cols}
    print(f"{pid:>10s} {row['prior_wpa_std']:9.4f} {pct['prior_wpa_std']:10.1f}% "
          f"{row['prior_bb_rate']:9.4f} {pct['prior_bb_rate']:6.1f}% "
          f"{row['prior_so_rate']:9.4f} {pct['prior_so_rate']:6.1f}% "
          f"{row['prior_hr_rate']:9.4f} {pct['prior_hr_rate']:6.1f}%")
    pitcher8_detail.append({"pitcher_id": pid, **row.to_dict(),
                             **{f"{c}_pctile": pct[c] for c in prior_cols}})

pctile_spread = {c: float(pop_pct[c].loc[pid8].max() - pop_pct[c].loc[pid8].min()) * 100
                  for c in prior_cols}
print(f"\n8명의 백분위 스프레드(최대-최소): "
      + ", ".join(f"{c}={v:.1f}%p" for c, v in pctile_spread.items()))
narrow_types = bool(all(v < 50 for v in pctile_spread.values()))
print(f"모든 지표에서 스프레드 < 50%p: {'예 — 비슷한 유형일 가능성' if narrow_types else '아니오'}")

report["task2_original_8_profile"] = {
    "pitcher_ids": pid8, "detail": pitcher8_detail,
    "pctile_spread": pctile_spread, "narrow_types": narrow_types,
}

# ── 극단 투수 9명 선정: bb_rate 상위3, so_rate 상위3(제외), wpa_std 하위3(제외) ──
bb_top3 = pitcher_stats["prior_bb_rate"].nlargest(3)
remaining = pitcher_stats.drop(bb_top3.index)
so_top3 = remaining["prior_so_rate"].nlargest(3)
remaining2 = remaining.drop(so_top3.index)
wpa_std_bottom3 = remaining2["prior_wpa_std"].nsmallest(3)

extreme_groups = {
    "bb_rate_top3 (제구 불안형)": bb_top3.index.tolist(),
    "so_rate_top3 (파워형)": so_top3.index.tolist(),
    "wpa_std_bottom3 (안정형)": wpa_std_bottom3.index.tolist(),
}
print("\n극단 투수 9명 선정:")
for label, ids in extreme_groups.items():
    print(f"  {label}: {ids}")
    for pid in ids:
        row = pitcher_stats.loc[pid]
        print(f"     {pid}: wpa_std={row['prior_wpa_std']:.4f} bb={row['prior_bb_rate']:.4f} "
              f"so={row['prior_so_rate']:.4f} hr={row['prior_hr_rate']:.4f}")

extreme_ids = bb_top3.index.tolist() + so_top3.index.tolist() + wpa_std_bottom3.index.tolist()
pvb_by_pitcher = apps_elig.groupby("pitcher_id")["pitcher_vs_batter_avg"].mean()
pitchers_extreme9 = [
    {"pitcher_id": pid, "pitcher_vs_batter_avg": float(pvb_by_pitcher.loc[pid])}
    for pid in extreme_ids
]

# 3-2와 동일한 방식으로 situations_10 재현 (validate_horizon.py와 동일 시드/함수)
def _row_to_state(row) -> GameState:
    return GameState(
        inning=int(row["inning"]), half=str(row["half"]),
        out_count=int(row["out_count"]), base_state=str(row["base_state"]),
        score_diff_attacker=int(row["score_diff_attacker"]),
        pitcher_id=str(int(row["pitcher_id"])), batter_id=str(int(row["batter_id"])),
        batter_hit_type=str(row["batter_hit_type"]),
        pitcher_vs_batter_avg=float(row["pitcher_vs_batter_avg"]) if pd.notna(row["pitcher_vs_batter_avg"]) else 0.25,
        batter_recent_avg=float(row["batter_recent_avg"]) if pd.notna(row["batter_recent_avg"]) else 0.25,
    )

def load_situations(df_: pd.DataFrame, n: int, seed: int) -> list[GameState]:
    sample = df_.sample(n=n, random_state=seed)
    return [_row_to_state(row) for _, row in sample.iterrows()]

def with_pitcher(situation: GameState, pitcher: dict) -> GameState:
    return GameState(
        inning=situation.inning, half=situation.half,
        out_count=situation.out_count, base_state=situation.base_state,
        score_diff_attacker=situation.score_diff_attacker,
        pitcher_id=pitcher["pitcher_id"], batter_id=situation.batter_id,
        batter_hit_type=situation.batter_hit_type,
        pitcher_vs_batter_avg=pitcher["pitcher_vs_batter_avg"],
        batter_recent_avg=situation.batter_recent_avg,
    )

situations_30 = load_situations(df, n=30, seed=RNG_SEED)
situations_10 = situations_30[:10]

sim_500 = MonteCarloSimulator(n_simulations=500, rng_seed=RNG_SEED)

def between_within_ratio(pitchers: list[dict], situations: list[GameState],
                          horizon_pa=5, horizon_innings=1.0) -> dict:
    between_list, within_list = [], []
    for s in situations:
        e_per_pitcher, std_per_pitcher = [], []
        for p in pitchers:
            state = with_pitcher(s, p)
            r = sim_500.simulate(state, horizon_pa=horizon_pa, horizon_innings=horizon_innings)
            e_per_pitcher.append(-r["e_wpa"])  # 투수(수비) 관점
            std_per_pitcher.append(r["std_wpa"])
        between_list.append(float(np.std(e_per_pitcher)))
        within_list.append(float(np.mean(std_per_pitcher)))
    between = float(np.mean(between_list))
    within = float(np.mean(within_list))
    ratio = between / within if within > 0 else float("nan")
    return {"between": between, "within": within, "ratio": ratio}

print("\n재측정: 원본 8명 vs 극단 9명 (horizon=(5,1.0), n=500, 동일 situations_10)")
orig8_result = between_within_ratio(pitchers_8, situations_10)
extreme9_result = between_within_ratio(pitchers_extreme9, situations_10)
print(f"  원본 8명   : between={orig8_result['between']:.4f} within={orig8_result['within']:.4f} "
      f"ratio={orig8_result['ratio']:.4f}")
print(f"  극단 9명   : between={extreme9_result['between']:.4f} within={extreme9_result['within']:.4f} "
      f"ratio={extreme9_result['ratio']:.4f}")

ratio_improvement = extreme9_result["ratio"] / orig8_result["ratio"] if orig8_result["ratio"] > 0 else float("nan")
print(f"  ratio 배율(극단/원본): {ratio_improvement:.2f}x")

report["task2_extreme_selection"] = {
    "extreme_groups": extreme_groups,
    "original_8": orig8_result,
    "extreme_9": extreme9_result,
    "ratio_improvement_factor": ratio_improvement,
}


# ──────────────────────────────────────────────
# Task 3: 실제 데이터의 투수 간 차이 (baseline)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 3: 실제 등판 기록 기준 투수 간 신호 (baseline, 상한선)")
print("=" * 70)

app_wpa_by_pitcher = apps_elig.groupby("pitcher_id")["app_wpa"]
mean_app_wpa = app_wpa_by_pitcher.mean()
std_app_wpa = app_wpa_by_pitcher.std(ddof=1).dropna()

between_actual = float(mean_app_wpa.std(ddof=0))
within_actual = float(std_app_wpa.mean())
ratio_actual = between_actual / within_actual if within_actual > 0 else float("nan")

print(f"\n실제 데이터 (등판 {MIN_APPEARANCES}회 이상 투수 {len(eligible_ids)}명, app_wpa 기준):")
print(f"  between (투수별 app_wpa 평균의 표준편차) = {between_actual:.4f}")
print(f"  within  (투수 내 app_wpa 표준편차의 평균) = {within_actual:.4f}")
print(f"  ratio = {ratio_actual:.4f}")

sim_ratios = {
    "h5_1 (시뮬레이션, 원본 8명)": orig8_result["ratio"],
    "h5_1 (시뮬레이션, 극단 9명)": extreme9_result["ratio"],
}
print("\n시뮬레이션 ratio 대비:")
for label, v in sim_ratios.items():
    print(f"  {label}: {v:.4f}  (실제 대비 {v/ratio_actual*100:.1f}%)" if ratio_actual > 0 else f"  {label}: {v:.4f}")

report["task3_actual_baseline"] = {
    "n_pitchers": int(len(eligible_ids)),
    "between_actual": between_actual,
    "within_actual": within_actual,
    "ratio_actual": ratio_actual,
    "sim_ratio_comparison": sim_ratios,
}


# ──────────────────────────────────────────────
# 결론
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("결론")
print("=" * 70)

model_cannot_discriminate = all_below_001
selection_bias_explains_it = ratio_improvement > 2.0 if not np.isnan(ratio_improvement) else False
sim_close_to_actual = (ratio_actual > 0 and 0.5 <= (orig8_result["ratio"] / ratio_actual) <= 2.0)

conclusion_lines = []
conclusion_lines.append(
    f"Task1: 클래스별 확률 표준편차 최대={max_std:.5f}({max_class}), 최소={min_std:.5f}({min_class}). "
    + ("모든 클래스 std<0.01 — 모델이 투수를 거의 구별하지 못함."
       if model_cannot_discriminate else
       "일부 클래스는 투수 간 차이를 반영하나 전반적으로 미미함." if max_std < 0.03 else
       "모델이 투수 간 차이를 어느 정도 반영함.")
)
conclusion_lines.append(
    f"Task2: 원본 8명 ratio={orig8_result['ratio']:.4f} → 극단 9명 ratio={extreme9_result['ratio']:.4f} "
    f"({ratio_improvement:.2f}배). "
    + ("표본 편향이 주요 원인 중 하나로 보임." if selection_bias_explains_it else
       "극단 투수를 뽑아도 ratio가 크게 개선되지 않음 — 표본 편향이 주된 원인은 아님.")
)
conclusion_lines.append(
    f"Task3: 실제 데이터 ratio={ratio_actual:.4f} vs 시뮬레이션(원본8) ratio={orig8_result['ratio']:.4f}. "
    + ("두 값이 비슷한 수준 — 낮은 ratio는 야구의 본질적 특성(상황 변동성이 투수 실력차보다 훨씬 큼)일 가능성이 높음."
       if sim_close_to_actual else
       "시뮬레이션 ratio가 실제 데이터보다 뚜렷이 낮음 — 모델이 실제 존재하는 투수 신호 중 일부를 놓치고 있을 가능성.")
)

final_verdict = (
    "야구의 본질 (상황 변동성 >> 투수 실력차) 이 지배적 원인으로 보임"
    if (sim_close_to_actual and not model_cannot_discriminate)
    else "모델이 실제 신호 대비 투수 구별력을 상당 부분 놓치고 있을 가능성"
    if not sim_close_to_actual
    else "모델의 근본적 한계(확률 벡터가 투수별로 거의 분화되지 않음)로 보임"
)

for line in conclusion_lines:
    print(f"\n- {line}")
print(f"\n▶ 최종 판단: {final_verdict}")

report["conclusion"] = {
    "lines": conclusion_lines,
    "final_verdict": final_verdict,
    "model_cannot_discriminate": model_cannot_discriminate,
    "selection_bias_explains_it": selection_bias_explains_it,
    "sim_close_to_actual": sim_close_to_actual,
}

with open(OUTPUT_DIR / "pitcher_signal_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"✅ 완료! 리포트: {OUTPUT_DIR / 'pitcher_signal_report.json'}")
print("=" * 70)
