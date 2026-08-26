"""
KBO WPA Pipeline — Task 3: 통합 효용 함수 프로토타입 검증

U(p) = [E_sit(situation) + skill_adj(p)] - λ * prior_wpa_std(p)

핵심 구조적 관찰: skill_adj(p)와 prior_wpa_std(p)는 후보 투수 p의 고유 속성이고
E_sit(situation)은 "어떤 후보를 고르든 동일하게 적용되는" 상황 상수다(회귀
보정항이 situation을 입력으로 받지 않으므로). 따라서 같은 상황에서 여러 후보를
비교할 때 E_sit은 모든 후보에게 똑같이 더해지는 상수라 순위(argmax)에 전혀
영향을 주지 못한다 — U(p)의 순위는 skill_adj(p) - λ*prior_wpa_std(p) 만으로
완전히 결정된다. 이는 시뮬레이션을 다시 돌리지 않고도 분석적으로 계산 가능함을
의미하며, 동시에 이 설계가 "상황별로 다른 투수가 유리해지는" 효과를 표현할 수
없다는 구조적 한계를 뜻한다(3-3에서 실측으로 재확인).

Task2에서 확인된 값: skill_adj_relative std=0.00084 (R² CV=-0.0008, 사실상 노이즈),
prior_wpa_std 범위 0.03~0.35 수준. E 항이 거의 기여하지 못하므로 λ>0인 순간
사실상 prior_wpa_std만으로 순위가 결정될 것으로 예상됨 — 아래에서 직접 검증한다.

실행: python investigations/task3_utility_prototype.py (레포 루트 기준)
사전 조건: task2_skill_regression.py 실행 완료 (동일 회귀 로직을 재사용)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo_simulate import MonteCarloSimulator, GameState

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)
RNG_SEED = 42
FEATURES = ["prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_n_apps"]

print("=" * 70)
print("Task 3: 통합 효용 함수 프로토타입 (U = E_sit + skill_adj - λ*prior_wpa_std)")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)
df["_date"] = df["game_id"].str[:8].astype(int)

first_pa = df[df["current_pitcher_pa_in_app"] == 1].copy()
first_pa["abs_score_diff_start"] = first_pa["score_diff_attacker"].abs()
cond_all = (
    (first_pa["is_starter"] == False) &
    (first_pa["n_pa"] >= 3) & (first_pa["n_pa"] <= 7) &
    (first_pa["start_inning"] >= 6) & (first_pa["start_inning"] <= 8) &
    (first_pa["abs_score_diff_start"] <= 3)
)
subset = first_pa[cond_all].copy()
subset["wpa_per_pa"] = subset["app_wpa"] / subset["n_pa"]

# ── 회귀 재적합 (Task2와 동일 로직) ──
reg_data = subset.dropna(subset=FEATURES + ["wpa_per_pa"]).copy()
X = reg_data[FEATURES].values
y = reg_data["wpa_per_pa"].values
skill_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RNG_SEED))
skill_model.fit(X, y)

counts = subset.groupby("pitcher_id").size()
eligible = counts[counts >= 20].index
last_rows = (
    df[df["pitcher_id"].isin(eligible)]
    .sort_values("_date")
    .drop_duplicates(subset=["pitcher_id"], keep="last")
    .set_index("pitcher_id")
).dropna(subset=FEATURES + ["prior_wpa_std"])

skill_adj = skill_model.predict(last_rows[FEATURES].values)
league_mean = float(np.mean(skill_adj))
pool = pd.DataFrame({
    "pitcher_id": last_rows.index,
    "skill_adj_relative": skill_adj - league_mean,
    "prior_wpa_std": last_rows["prior_wpa_std"].values,
})
print(f"\n풀 크기: {len(pool)}명")

report = {}

# ──────────────────────────────────────────────
# 3-1) between vs 실제 데이터, 순위 상관
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-1) between(E_pitcher) vs 실제 데이터, 순위 상관 (Spearman)")
print("=" * 70)

between_pred = float(np.var(pool["skill_adj_relative"], ddof=0))
print(f"\nVar(E_pitcher, λ=0 즉 skill_adj만) = {between_pred:.8f}  (std={np.sqrt(between_pred):.5f})")

actual_mean_wpa_per_pa = subset[subset["pitcher_id"].isin(pool["pitcher_id"])].groupby("pitcher_id")["wpa_per_pa"].mean()
actual_mean_wpa_per_pa = actual_mean_wpa_per_pa.reindex(pool["pitcher_id"]).values
between_actual = float(np.var(actual_mean_wpa_per_pa, ddof=0))
print(f"Var(실제 투수별 평균 wpa_per_pa) = {between_actual:.8f}  (std={np.sqrt(between_actual):.5f})")
print(f"비율(예측/실제) = {between_pred/between_actual:.4f}")

spearman_r, spearman_p = stats.spearmanr(pool["skill_adj_relative"], actual_mean_wpa_per_pa)
print(f"\nSpearman(예측순위 skill_adj vs 실제순위 wpa_per_pa) = {spearman_r:.4f}  p={spearman_p:.3e}")
rank_verdict = "약한 양의 상관" if 0.1 <= spearman_r < 0.3 else ("의미있는 상관" if spearman_r >= 0.3 else "상관 없음/노이즈")
print(f"▶ {rank_verdict}")

report["3-1"] = {
    "n_pitchers": len(pool), "var_e_pitcher_pred": between_pred, "var_actual": between_actual,
    "ratio_pred_over_actual": between_pred / between_actual,
    "spearman_r": float(spearman_r), "spearman_p": float(spearman_p), "verdict": rank_verdict,
}

# ──────────────────────────────────────────────
# 3-2) λ* 분포
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-2) λ* = (E_A - E_B) / (Std_A - Std_B) 분포 (skill_adj, prior_wpa_std 기준)")
print("=" * 70)

rng = np.random.default_rng(RNG_SEED)
sample_idx = rng.choice(len(pool), size=min(60, len(pool)), replace=False)
sample_pool = pool.iloc[sample_idx].reset_index(drop=True)

lambda_stars = []
for i in range(len(sample_pool)):
    for j in range(i + 1, len(sample_pool)):
        std_a, std_b = sample_pool["prior_wpa_std"].iloc[i], sample_pool["prior_wpa_std"].iloc[j]
        e_a, e_b = sample_pool["skill_adj_relative"].iloc[i], sample_pool["skill_adj_relative"].iloc[j]
        if abs(std_a - std_b) < 1e-6:
            continue
        lam = (e_a - e_b) / (std_a - std_b)
        if lam > 0:
            lambda_stars.append(lam)

lambda_stars = np.array(lambda_stars)
n_valid = len(lambda_stars)
n_total_pairs = len(sample_pool) * (len(sample_pool) - 1) // 2
print(f"\n유효 λ* 표본: {n_valid} / {n_total_pairs} ({n_valid/n_total_pairs*100:.1f}%)")

if n_valid > 0:
    pcts = [5, 25, 50, 75, 95]
    pct_vals = {p: float(np.percentile(lambda_stars, p)) for p in pcts}
    print("λ* 분위수:")
    for p in pcts:
        print(f"  {p:2d}%ile: {pct_vals[p]:.5f}")
    q95 = pct_vals[95]
    in_target_range = 0.5 <= q95 <= 3
    print(f"\nλ_max(95%ile)={q95:.5f}  목표범위(0.5~3) {'내' if in_target_range else '밖'}")
    report["3-2"] = {"n_valid": n_valid, "n_total_pairs": n_total_pairs,
                      "percentiles": pct_vals, "lambda_max": q95,
                      "in_target_range_0.5_3": in_target_range}
else:
    q95 = None
    report["3-2"] = {"n_valid": 0}
    print("⚠️  유효 λ* 없음")

# ──────────────────────────────────────────────
# 3-3) λ=0 vs λ=λ_max 추천 변화 (구조적으로 상황-불변임을 실측 확인)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-3) λ=0 vs λ=λ_max 추천 변화 — 15개 실제 상황에서 실측")
print("=" * 70)

situation_pool = df[(df["inning"] >= 6) & (df["inning"] <= 8) &
                     (df["score_diff_attacker"].abs() <= 3)]
situations_sample = situation_pool.sample(n=15, random_state=RNG_SEED)

candidate_ids = rng.choice(pool["pitcher_id"].values, size=8, replace=False)
cand_pool = pool[pool["pitcher_id"].isin(candidate_ids)].reset_index(drop=True)

sim = MonteCarloSimulator(n_simulations=300, rng_seed=RNG_SEED)


def _row_to_state(row, pid) -> GameState:
    return GameState(
        inning=int(row["inning"]), half=str(row["half"]),
        out_count=int(row["out_count"]), base_state=str(row["base_state"]),
        score_diff_attacker=int(row["score_diff_attacker"]),
        pitcher_id=str(pid), batter_id=str(int(row["batter_id"])),
        batter_hit_type=str(row["batter_hit_type"]),
        pitcher_vs_batter_avg=float(row["pitcher_vs_batter_avg"]) if pd.notna(row["pitcher_vs_batter_avg"]) else 0.25,
        batter_recent_avg=float(row["batter_recent_avg"]) if pd.notna(row["batter_recent_avg"]) else 0.25,
    )


lam_max = q95 if q95 is not None else 1.0
n_changed = 0
detail = []
for si, (_, srow) in enumerate(situations_sample.iterrows()):
    # E_sit: "평균 투수"로 시뮬레이션 (여기서는 후보 풀 전체 평균을 근사하기 위해
    # 후보 8명의 평균으로 대체 — 어차피 아래에서 모든 후보에 동일하게 더해지므로
    # argmax에는 영향 없음, 값 자체의 스케일 확인용)
    e_sit_vals = []
    for pid in candidate_ids[:3]:
        state = _row_to_state(srow, pid)
        r = sim.simulate(state, horizon_pa=5, horizon_innings=1.0)
        e_sit_vals.append(-r["e_wpa"])
    e_sit = float(np.mean(e_sit_vals))

    u0 = e_sit + cand_pool["skill_adj_relative"].values - 0.0 * cand_pool["prior_wpa_std"].values
    u_max = e_sit + cand_pool["skill_adj_relative"].values - lam_max * cand_pool["prior_wpa_std"].values
    best0 = cand_pool["pitcher_id"].iloc[int(np.argmax(u0))]
    best_max = cand_pool["pitcher_id"].iloc[int(np.argmax(u_max))]
    changed = best0 != best_max
    n_changed += changed
    detail.append({"situation_idx": si, "e_sit": e_sit,
                    "best_at_lambda0": best0, "best_at_lambda_max": best_max, "changed": bool(changed)})
    print(f"  상황{si:2d}: e_sit={e_sit:+.4f}  λ=0→{best0}  λ_max→{best_max}  "
          f"{'✓변화' if changed else ' 동일'}")

print(f"\nλ=0 vs λ_max({lam_max:.4f}) 추천 변화: {n_changed}/15")
print("※ 구조적으로 e_sit은 모든 후보에 동일하게 더해지므로, 위 15개 결과는 전부 "
      "동일하거나(0/15) 전부 달라야(15/15) 정상이다 — 상황에 따라 갈리지 않는 것은 "
      "구현 오류가 아니라 이 설계(E가 후보 간 상대순위에 영향 못 줌)의 필연적 결과다.")

report["3-3"] = {"lambda_max_used": lam_max, "n_changed": n_changed, "n_total": 15, "detail": detail,
                  "note": "e_sit이 모든 후보에 동일 상수로 더해지므로 결과는 상황에 무관하게 전부 동일해야 정상"}

with open(OUTPUT_DIR / "task3_utility_prototype_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 완료! {OUTPUT_DIR / 'task3_utility_prototype_report.json'}")
