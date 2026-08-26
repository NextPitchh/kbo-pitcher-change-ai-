"""
KBO WPA Pipeline — 실제 baseline ratio(0.79) vs 시뮬레이션 ratio(0.07~0.12) 격차의
원인이 "교란 요인(보직/등판 길이/선택 편향)"인지 검증한다.

가설: 실제 app_wpa의 between 분산에는 투수 실력 외에 (1) 보직 배치(마무리 9회
리드 vs 추격조), (2) 등판 길이(합계 지표라 이닝 수에 비례), (3) 선택 편향(감독이
컨디션 보고 길이 조절)이 섞여 있다. 시뮬레이션은 이 세 요인을 전부 통제하므로
0.79와 0.07~0.12는 애초에 다른 것을 재고 있었을 수 있다.

Task 1: 조건(불펜/3~7타석/6~8회 시작/|점수차|<=3, 투수당 20회↑) 통제 후 실제 baseline 재계산
Task 2: 조건을 하나씩 추가하며 ratio 변화 추적 → 요인별 기여도
Task 3: wpa_per_pa = app_wpa/n_pa 로 정규화 → 길이 효과 크기 확인
Task 4: 보직(마무리/셋업/롱릴리프) 그룹 "내부"에서 ratio 재계산
Task 5: Task1 조건과 동일한 상황·동일 투수 풀로 시뮬레이션 실행 → 직접 비교

절대 placeholder ID 사용 안 함. horizon 기본값(5PA/1이닝) 변경 안 함.
모델 변경·데이터 증강 시도 안 함 — 순수 측정/비교.

실행: python investigations/validate_confounds.py (레포 루트 기준)
요구사항: pip install catboost pandas numpy pyarrow
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo_simulate import MonteCarloSimulator, GameState

OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42
report = {}

print("=" * 70)
print("실제-시뮬레이션 ratio 격차: 교란 요인(보직/길이/선택편향) 검증")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)

# 등판(appearance)의 시작 시점 스냅샷 — app_wpa/n_pa/start_inning/is_starter는
# 등판 내 모든 PA행에 동일하게 복제돼 있음(검증됨). score_diff류만 PA마다 바뀌므로
# current_pitcher_pa_in_app==1인 행에서 "시작 시점" 값을 뽑는다.
first_pa = df[df["current_pitcher_pa_in_app"] == 1].copy()
first_pa["abs_score_diff_start"] = first_pa["score_diff_attacker"].abs()
print(f"\n전체 등판(첫 타석 기준) 수: {len(first_pa)}")


def between_within_ratio_actual(apps_df: pd.DataFrame, min_appearances: int,
                                 value_col: str = "app_wpa") -> dict:
    """등판 레코드(1행=1등판)에서 투수별 between/within/ratio 계산."""
    counts = apps_df.groupby("pitcher_id").size()
    eligible = counts[counts >= min_appearances].index
    sub = apps_df[apps_df["pitcher_id"].isin(eligible)]
    means = sub.groupby("pitcher_id")[value_col].mean()
    stds = sub.groupby("pitcher_id")[value_col].std(ddof=1).dropna()
    between = float(means.std(ddof=0))
    within = float(stds.mean())
    ratio = between / within if within > 0 else float("nan")
    return {"n_pitchers": int(len(eligible)), "n_appearances": int(len(sub)),
            "between": between, "within": within, "ratio": ratio}


# ──────────────────────────────────────────────
# Task 1: 조건 통제 baseline
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 1: 조건 통제 baseline (불펜, 3~7타석, 6~8회 시작, |점수차|<=3, 투수당 20회↑)")
print("=" * 70)

cond_all = (
    (first_pa["is_starter"] == False) &
    (first_pa["n_pa"] >= 3) & (first_pa["n_pa"] <= 7) &
    (first_pa["start_inning"] >= 6) & (first_pa["start_inning"] <= 8) &
    (first_pa["abs_score_diff_start"] <= 3)
)
subset_task1 = first_pa[cond_all]
result_task1 = between_within_ratio_actual(subset_task1, min_appearances=20)

# 대조군: 전체(무조건, 투수당 30회↑) — 이전 조사의 0.79 재확인
baseline_full = between_within_ratio_actual(first_pa, min_appearances=30)

print(f"\n  조건 없음(투수당 30회↑, 이전 baseline 재확인): "
      f"n_pitchers={baseline_full['n_pitchers']} ratio={baseline_full['ratio']:.4f}")
print(f"  조건 통제(투수당 20회↑): n_pitchers={result_task1['n_pitchers']} "
      f"n_appearances={result_task1['n_appearances']} "
      f"between={result_task1['between']:.5f} within={result_task1['within']:.5f} "
      f"ratio={result_task1['ratio']:.4f}")
print(f"\n  0.79 → {result_task1['ratio']:.4f}  "
      f"({'유지' if result_task1['ratio'] >= 0.5*baseline_full['ratio'] else '큰 폭 하락'})")

report["task1_condition_controlled"] = {"baseline_full_thresh30": baseline_full,
                                         "condition_controlled_thresh20": result_task1}


# ──────────────────────────────────────────────
# Task 2: 요인별 기여도 분해 (단계적 조건 추가, 모두 min_appearances=20으로 통일)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 2: 요인별 기여도 (단계적 조건 추가, 투수당 20회↑ 통일)")
print("=" * 70)

MIN_APP_STEP = 20
steps = []

c0 = pd.Series(True, index=first_pa.index)
steps.append(("조건0_전체", c0))

c1 = c0 & (first_pa["is_starter"] == False)
steps.append(("조건1_+불펜만", c1))

c2 = c1 & (first_pa["n_pa"] >= 3) & (first_pa["n_pa"] <= 7)
steps.append(("조건2_+등판길이3~7타석", c2))

c3 = c2 & (first_pa["start_inning"] >= 6) & (first_pa["start_inning"] <= 8)
steps.append(("조건3_+이닝6~8시작", c3))

c4 = c3 & (first_pa["abs_score_diff_start"] <= 3)
steps.append(("조건4_+점수차|diff|<=3", c4))

step_rows = []
prev_ratio = None
for label, mask in steps:
    r = between_within_ratio_actual(first_pa[mask], min_appearances=MIN_APP_STEP)
    drop_pct = (None if prev_ratio is None or prev_ratio == 0
                else (r["ratio"] - prev_ratio) / prev_ratio * 100)
    step_rows.append({"step": label, **r, "ratio_change_pct_vs_prev": drop_pct})
    print(f"  {label:24s} n_pitchers={r['n_pitchers']:4d} n_app={r['n_appearances']:6d} "
          f"ratio={r['ratio']:.4f}"
          + (f"  ({drop_pct:+.1f}% vs 이전 단계)" if drop_pct is not None else ""))
    prev_ratio = r["ratio"]

report["task2_stepwise_factor_contribution"] = step_rows

biggest_drop = min(
    (row for row in step_rows if row["ratio_change_pct_vs_prev"] is not None),
    key=lambda r: r["ratio_change_pct_vs_prev"]
)
print(f"\n  가장 크게 ratio를 떨어뜨린 단계: {biggest_drop['step']} "
      f"({biggest_drop['ratio_change_pct_vs_prev']:+.1f}%)")
report["task2_biggest_drop_step"] = biggest_drop["step"]


# ──────────────────────────────────────────────
# Task 3: 등판 길이 정규화 (wpa_per_pa)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 3: 등판 길이 정규화 (wpa_per_pa = app_wpa / n_pa)")
print("=" * 70)

first_pa["wpa_per_pa"] = first_pa["app_wpa"] / first_pa["n_pa"]

result_raw = between_within_ratio_actual(first_pa, min_appearances=30, value_col="app_wpa")
result_norm = between_within_ratio_actual(first_pa, min_appearances=30, value_col="wpa_per_pa")

print(f"\n  합계 지표(app_wpa)   : between={result_raw['between']:.5f} "
      f"within={result_raw['within']:.5f} ratio={result_raw['ratio']:.4f}")
print(f"  정규화 지표(wpa/PA) : between={result_norm['between']:.5f} "
      f"within={result_norm['within']:.5f} ratio={result_norm['ratio']:.4f}")

length_effect_pct = (result_norm["ratio"] - result_raw["ratio"]) / result_raw["ratio"] * 100
print(f"\n  정규화로 인한 ratio 변화: {length_effect_pct:+.1f}%")

report["task3_length_normalization"] = {
    "raw_app_wpa": result_raw, "normalized_wpa_per_pa": result_norm,
    "length_effect_pct": length_effect_pct,
}


# ──────────────────────────────────────────────
# Task 4: 보직 효과 분리
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 4: 보직 그룹 내부 ratio (불펜 등판만 대상)")
print("=" * 70)

relief = first_pa[first_pa["is_starter"] == False].copy()
per_pitcher = relief.groupby("pitcher_id").agg(
    n_relief_app=("app_wpa", "size"),
    frac_9th=("start_inning", lambda s: (s == 9).mean()),
    frac_78=("start_inning", lambda s: s.isin([7, 8]).mean()),
    mean_n_pa=("n_pa", "mean"),
)
MIN_RELIEF_APP = 20
per_pitcher = per_pitcher[per_pitcher["n_relief_app"] >= MIN_RELIEF_APP]


def classify(row) -> str:
    if row["frac_9th"] >= 0.5:
        return "마무리형"
    if row["frac_78"] >= 0.5:
        return "셋업형"
    if row["mean_n_pa"] >= 8:
        return "롱릴리프형"
    return "기타"


per_pitcher["role"] = per_pitcher.apply(classify, axis=1)
role_counts = per_pitcher["role"].value_counts()
print(f"\n등판 {MIN_RELIEF_APP}회 이상 불펜 투수 {len(per_pitcher)}명, 보직 분포:")
for role, n in role_counts.items():
    print(f"  {role}: {n}명")

role_rows = []
for role in ["마무리형", "셋업형", "롱릴리프형", "기타"]:
    pids = per_pitcher[per_pitcher["role"] == role].index
    if len(pids) < 3:
        print(f"\n  {role}: 표본 부족(n={len(pids)}) — 생략")
        continue
    sub = relief[relief["pitcher_id"].isin(pids)]
    r = between_within_ratio_actual(sub, min_appearances=MIN_RELIEF_APP)
    role_rows.append({"role": role, **r})
    print(f"\n  {role} (n_pitchers={r['n_pitchers']}): between={r['between']:.5f} "
          f"within={r['within']:.5f} ratio={r['ratio']:.4f}")

overall_relief = between_within_ratio_actual(relief, min_appearances=MIN_RELIEF_APP)
print(f"\n  (대조) 불펜 전체 섞어서 계산: ratio={overall_relief['ratio']:.4f}")

role_ratios_lower = all(r["ratio"] < overall_relief["ratio"] for r in role_rows) if role_rows else False
print(f"\n  모든 그룹 내부 ratio < 전체 섞은 ratio: "
      f"{'예 — 보직이 주요 교란 요인' if role_ratios_lower else '아니오 — 보직만으로는 설명 부족'}")

report["task4_role_groups"] = {
    "role_counts": role_counts.to_dict(), "role_rows": role_rows,
    "overall_relief_mixed": overall_relief, "role_ratios_all_lower": role_ratios_lower,
}


# ──────────────────────────────────────────────
# Task 5: 시뮬레이션과 공정 비교 (Task1과 동일 조건·동일 투수 풀)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 5: 시뮬레이션 공정 비교 (6~8회, |점수차|<=3, horizon=5PA/1이닝 기본값)")
print("=" * 70)

qualifying_pids = subset_task1.groupby("pitcher_id").size()
qualifying_pids = qualifying_pids[qualifying_pids >= 20].index.tolist()
print(f"\nTask1 조건을 만족하는 투수 풀: {len(qualifying_pids)}명")

pvb_by_pitcher = df[df["pitcher_id"].isin(qualifying_pids)].groupby("pitcher_id")["pitcher_vs_batter_avg"].mean()
pitchers_pool = [{"pitcher_id": pid, "pitcher_vs_batter_avg": float(pvb_by_pitcher.loc[pid])}
                  for pid in qualifying_pids]

# 시뮬레이션 상황: 실제 PA 레코드 중 6~8회, |score_diff_attacker|<=3 인 것만 샘플링
situation_pool = df[(df["inning"] >= 6) & (df["inning"] <= 8) &
                     (df["score_diff_attacker"].abs() <= 3)]
print(f"조건을 만족하는 실제 PA 레코드: {len(situation_pool)} (여기서 상황 샘플링)")


def _row_to_state(row) -> GameState:
    return GameState(
        inning=int(row["inning"]), half=str(row["half"]),
        out_count=int(row["out_count"]), base_state=str(row["base_state"]),
        score_diff_attacker=int(row["score_diff_attacker"]),
        pitcher_id=str(row["pitcher_id"]), batter_id=str(int(row["batter_id"])),
        batter_hit_type=str(row["batter_hit_type"]),
        pitcher_vs_batter_avg=float(row["pitcher_vs_batter_avg"]) if pd.notna(row["pitcher_vs_batter_avg"]) else 0.25,
        batter_recent_avg=float(row["batter_recent_avg"]) if pd.notna(row["batter_recent_avg"]) else 0.25,
    )


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


N_SITUATIONS = 15
situations_sample = situation_pool.sample(n=N_SITUATIONS, random_state=RNG_SEED)
situations = [_row_to_state(row) for _, row in situations_sample.iterrows()]

sim = MonteCarloSimulator(n_simulations=500, rng_seed=RNG_SEED)

print(f"\n{len(situations)}개 상황 × {len(pitchers_pool)}명 투수 시뮬레이션 시작 "
      f"(horizon_pa=5, horizon_innings=1.0 기본값)...")
t0 = time.time()
per_pitcher_e = {p["pitcher_id"]: [] for p in pitchers_pool}
per_pitcher_std = {p["pitcher_id"]: [] for p in pitchers_pool}
for s in situations:
    for p in pitchers_pool:
        state = with_pitcher(s, p)
        r = sim.simulate(state, horizon_pa=5, horizon_innings=1.0)
        per_pitcher_e[p["pitcher_id"]].append(-r["e_wpa"])
        per_pitcher_std[p["pitcher_id"]].append(r["std_wpa"])
elapsed = time.time() - t0
print(f"완료 ({elapsed:.1f}s, {len(situations)*len(pitchers_pool)}회 호출)")

pitcher_means_sim = {pid: float(np.mean(v)) for pid, v in per_pitcher_e.items()}
pitcher_stds_within_sim = {pid: float(np.mean(v)) for pid, v in per_pitcher_std.items()}

sim_between = float(np.std(list(pitcher_means_sim.values()), ddof=0))
sim_within = float(np.mean(list(pitcher_stds_within_sim.values())))
sim_ratio = sim_between / sim_within if sim_within > 0 else float("nan")

print(f"\n  시뮬레이션(조건 통제, n_pitchers={len(pitchers_pool)}, n_situations={N_SITUATIONS}): "
      f"between={sim_between:.5f} within={sim_within:.5f} ratio={sim_ratio:.4f}")
print(f"  실제(Task1 조건 통제, n_pitchers={result_task1['n_pitchers']}): "
      f"between={result_task1['between']:.5f} within={result_task1['within']:.5f} "
      f"ratio={result_task1['ratio']:.4f}")

fair_gap_pct = sim_ratio / result_task1["ratio"] * 100 if result_task1["ratio"] > 0 else float("nan")
print(f"\n  공정 비교 격차: 시뮬레이션이 실제의 {fair_gap_pct:.1f}%")

report["task5_fair_comparison"] = {
    "n_pitchers_pool": len(pitchers_pool), "n_situations": N_SITUATIONS,
    "sim_between": sim_between, "sim_within": sim_within, "sim_ratio": sim_ratio,
    "actual_condition_controlled": result_task1,
    "fair_gap_pct": fair_gap_pct, "elapsed_sec": elapsed,
}


# ──────────────────────────────────────────────
# 결론
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("결론")
print("=" * 70)

condition_control_closes_gap = result_task1["ratio"] < 0.5 * baseline_full["ratio"]
length_dominant = abs(length_effect_pct) > 30
role_dominant = role_ratios_lower and role_rows and \
    (overall_relief["ratio"] - np.mean([r["ratio"] for r in role_rows])) / overall_relief["ratio"] > 0.3
sim_still_far_after_control = fair_gap_pct < 50

lines = [
    f"Task1: 조건 통제 후 실제 ratio {baseline_full['ratio']:.4f} → {result_task1['ratio']:.4f}"
    + (" (조건만으로도 크게 떨어짐 — 확실히 교란 요인이 baseline을 부풀리고 있었음)"
       if condition_control_closes_gap else
       " (조건 통제해도 크게 안 떨어짐 — 원래 baseline이 교란 요인만의 산물은 아님)"),
    f"Task2: 가장 큰 하락 단계 = {biggest_drop['step']} ({biggest_drop['ratio_change_pct_vs_prev']:+.1f}%)",
    f"Task3: 길이 정규화(wpa_per_pa) 효과 = {length_effect_pct:+.1f}%"
    + (" (길이가 주요 교란 요인)" if length_dominant else " (길이 효과는 제한적)"),
    f"Task4: 보직 그룹 내부 ratio가 전체보다 {'낮음(보직이 교란 요인)' if role_ratios_lower else '낮지 않음(보직만으론 설명 부족)'}"
    + (f" — 그룹별: " + ", ".join(f"{r['role']}={r['ratio']:.4f}" for r in role_rows) if role_rows else ""),
    f"Task5: 조건·투수 풀을 시뮬레이션과 실제 모두 동일하게 맞춘 공정 비교 — "
    f"시뮬레이션 ratio={sim_ratio:.4f} vs 실제(조건통제) ratio={result_task1['ratio']:.4f} "
    f"(시뮬레이션이 실제의 {fair_gap_pct:.1f}%)",
]

final_verdict = (
    "교란 요인(보직/길이/선택편향)이 baseline 0.79를 상당 부분 부풀렸던 것은 맞지만, "
    "조건을 공정하게 맞춘 뒤에도 시뮬레이션은 여전히 실제 신호에 크게 못 미친다 — "
    "즉 비교 기준 문제와 모델의 실제 한계가 둘 다 존재하며, 격차의 상당 부분은 "
    "교란 요인 제거만으로는 사라지지 않는다."
    if sim_still_far_after_control else
    "조건을 공정하게 맞추자 시뮬레이션과 실제의 격차가 크게 좁혀졌다 — "
    "원래의 '모델이 신호를 놓치고 있다'는 결론은 상당 부분 비교 기준(교란 요인 미통제) "
    "때문이었을 가능성이 높다."
)

for line in lines:
    print(f"\n- {line}")
print(f"\n▶ 최종 판단: {final_verdict}")

report["conclusion"] = {
    "lines": lines, "final_verdict": final_verdict,
    "condition_control_closes_gap": condition_control_closes_gap,
    "length_dominant": length_dominant, "role_dominant": bool(role_dominant),
    "sim_still_far_after_control": sim_still_far_after_control,
}

with open(OUTPUT_DIR / "confounds_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"✅ 완료! 리포트: {OUTPUT_DIR / 'confounds_validation_report.json'}")
print("=" * 70)
