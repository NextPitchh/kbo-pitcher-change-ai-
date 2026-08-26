"""
KBO WPA Pipeline — Task 2: 투수 능력 보정항 회귀 (역할 분리 제안)

Task 1 게이트(prior_wpa_std 예측력 r=0.59/0.37, PASS)를 통과했으므로 진행한다.
E(기대값)는 시뮬레이션(상황 가치)으로, 투수별 편차는 별도 회귀 보정항으로
분리한다: y=wpa_per_pa(등판별 타석당 정규화), X=prior_bb_rate/so_rate/hr_rate/n_apps.

조건 통제 데이터(불펜·3~7타석·6~8회·|점수차|<=3)로 학습 — 등판 길이 효과를
정규화(wpa_per_pa)로 이미 제거했으므로 이중 통제.

prior_*는 이미 leakage-free로 검증된 값 그대로 사용 — 재계산 금지.

실행: python investigations/task2_skill_regression.py (레포 루트 기준)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo_simulate import MonteCarloSimulator, GameState

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)
RNG_SEED = 42

print("=" * 70)
print("Task 2: 투수 능력 보정항 회귀 (Ridge)")
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

FEATURES = ["prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_n_apps"]
reg_data = subset.dropna(subset=FEATURES + ["wpa_per_pa"]).copy()
print(f"\n조건 통제 등판 {len(subset)}건 중 prior_* 결측 제외 후 회귀 표본: {len(reg_data)}건")

X = reg_data[FEATURES].values
y = reg_data["wpa_per_pa"].values

# ──────────────────────────────────────────────
# 2-1) Ridge 회귀
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2-1) Ridge 회귀 (y=wpa_per_pa, X=prior_bb/so/hr_rate, prior_n_apps)")
print("=" * 70)

skill_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RNG_SEED))
skill_model.fit(X, y)

r2_in_sample = skill_model.score(X, y)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
cv_scores = cross_val_score(skill_model, X, y, cv=kf, scoring="r2")
r2_cv_mean = float(np.mean(cv_scores))
r2_cv_std = float(np.std(cv_scores))

ridge = skill_model.named_steps["ridge"]
scaler = skill_model.named_steps["standardscaler"]
std_coefs = dict(zip(FEATURES, ridge.coef_))
raw_coefs = dict(zip(FEATURES, ridge.coef_ / scaler.scale_))

print(f"\nR² (in-sample) = {r2_in_sample:.4f}")
print(f"R² (5-fold CV) = {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
print(f"\n표준화 계수 (상대적 중요도):")
for f, c in sorted(std_coefs.items(), key=lambda x: -abs(x[1])):
    print(f"  {f:16s}: {c:+.5f}")
print(f"\n원 단위 계수 (skill_adj = intercept + Σ raw_coef * X):")
intercept_raw = ridge.intercept_ - sum(raw_coefs[f] * reg_data[f].mean() for f in FEATURES) \
    if False else None  # (아래에서 predict()로 직접 계산하므로 별도 절편 역산 불필요)
for f, c in raw_coefs.items():
    print(f"  {f:16s}: {c:+.6f}  per unit")

sign_bb = "양수(기대와 일치)" if std_coefs["prior_bb_rate"] > 0 else "음수(기대와 반대)"
sign_so = "음수(기대와 일치)" if std_coefs["prior_so_rate"] < 0 else "양수(기대와 반대)"
print(f"\nprior_bb_rate 부호: {sign_bb}")
print(f"prior_so_rate 부호: {sign_so}")

r2_useful = r2_cv_mean >= 0.05
print(f"\n▶ R²(CV) {'>= 0.05 — 쓸 만한 설명력' if r2_useful else '< 0.05 — 설명력 부족'}")

report = {
    "n_samples": len(reg_data),
    "r2_in_sample": float(r2_in_sample),
    "r2_cv_mean": r2_cv_mean, "r2_cv_std": r2_cv_std, "cv_scores": cv_scores.tolist(),
    "standardized_coefs": {k: float(v) for k, v in std_coefs.items()},
    "raw_coefs": {k: float(v) for k, v in raw_coefs.items()},
    "intercept_standardized_space": float(ridge.intercept_),
    "sign_check": {"prior_bb_rate": sign_bb, "prior_so_rate": sign_so},
    "r2_useful": bool(r2_useful),
}

# ──────────────────────────────────────────────
# 2-2) 보정 크기 산정
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("2-2) 보정 크기 산정 (조건 통제 218명 풀, 최신 prior_* 스냅샷)")
print("=" * 70)

counts = subset.groupby("pitcher_id").size()
eligible = counts[counts >= 20].index
last_rows = (
    df[df["pitcher_id"].isin(eligible)]
    .sort_values("_date")
    .drop_duplicates(subset=["pitcher_id"], keep="last")
    .set_index("pitcher_id")
)
last_rows = last_rows.dropna(subset=FEATURES)
print(f"\n최신 prior_* 스냅샷 확보 투수: {len(last_rows)}명")

X_pool = last_rows[FEATURES].values
skill_adj = skill_model.predict(X_pool)
league_mean = float(np.mean(skill_adj))
skill_adj_relative = skill_adj - league_mean

pcts = [0, 25, 50, 75, 100]
dist = {p: float(np.percentile(skill_adj_relative, p)) for p in pcts}
print(f"\nskill_adj_relative 분포 (리그 평균 대비 편차, {len(last_rows)}명):")
for p in pcts:
    label = {0: "min", 25: "25%", 50: "median", 75: "75%", 100: "max"}[p]
    print(f"  {label:6s}: {dist[p]:+.5f}")
print(f"  std   : {float(np.std(skill_adj_relative)):.5f}")

report["skill_adj_distribution"] = {
    "n_pitchers": len(last_rows), "league_mean_raw": league_mean,
    "percentiles": dist, "std": float(np.std(skill_adj_relative)),
}

# ──────────────────────────────────────────────
# 스케일 비교용 참고: 상황 가치(E_situation)의 전형적 크기
# (v5 모델로 조건 통제 조건과 유사한 몇 상황을 빠르게 시뮬레이션)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("참고: E_situation 스케일 (v5 모델, horizon=5PA/1이닝 기본값)")
print("=" * 70)

situation_pool = df[(df["inning"] >= 6) & (df["inning"] <= 8) &
                     (df["score_diff_attacker"].abs() <= 3)]
situations_sample = situation_pool.sample(n=10, random_state=RNG_SEED)

median_pid = last_rows.iloc[(last_rows["prior_bb_rate"] - last_rows["prior_bb_rate"].median()).abs().argsort()[:3]].index.tolist()
print(f"중앙값 근접 '평균 투수' 대표 3명: {median_pid}")

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


e_situation_vals = []
for _, srow in situations_sample.iterrows():
    for pid in median_pid:
        state = _row_to_state(srow, pid)
        r = sim.simulate(state, horizon_pa=5, horizon_innings=1.0)
        e_situation_vals.append(-r["e_wpa"])

e_sit_mean = float(np.mean(e_situation_vals))
e_sit_std = float(np.std(e_situation_vals))
print(f"\nE_situation (평균 투수, 10상황×3명): mean={e_sit_mean:+.4f}  std={e_sit_std:.4f}")
print(f"skill_adj_relative 범위(min~max): {dist[0]:+.4f} ~ {dist[100]:+.4f}")
print(f"skill_adj std({float(np.std(skill_adj_relative)):.4f}) / E_situation std({e_sit_std:.4f}) "
      f"= {float(np.std(skill_adj_relative))/e_sit_std:.2f}배")

report["e_situation_reference"] = {"mean": e_sit_mean, "std": e_sit_std,
                                    "representative_pitchers": median_pid}

with open(OUTPUT_DIR / "task2_skill_regression_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 완료! {OUTPUT_DIR / 'task2_skill_regression_report.json'}")
