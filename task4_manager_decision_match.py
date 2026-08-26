"""
KBO WPA Pipeline — Task 4: 실제 감독 결정과의 일치율

U(p) = skill_adj(p) - λ * prior_wpa_std(p)  (Task3에서 확인했듯 E_sit은 상수라
후보 간 순위에 기여하지 않으므로 생략 — 순위는 이 식만으로 완전히 결정된다)

조건 통제 구간(불펜·3~7타석·6~8회·|점수차|<=3)의 실제 투수 교체 이벤트마다,
"그 시점에 감독이 실제로 쓸 수 있었을 법한 후보군"을 (같은 팀·같은 시즌에
등판한 적 있는 218-pool 투수) 프록시로 구성하고, 우리 U가 최댓값을 주는 투수가
실제로 기용된 투수와 일치하는 비율을 λ를 바꿔가며 측정한다.

주의: game_id에 인코딩된 팀 코드(홈 4~5번째, 원정 6~7번째 문자)로 소속팀을
역산하고, 후보군은 실측 로스터가 아니라 "같은 팀·같은 시즌에 등판 기록이 있는
투수" 프록시다 — 부상/콜업 등으로 실제 그 순간의 가용 로스터와는 다를 수 있음을
전제로 해석해야 한다.

실행: python task4_manager_decision_match.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)
RNG_SEED = 42
FEATURES = ["prior_bb_rate", "prior_so_rate", "prior_hr_rate", "prior_n_apps"]

print("=" * 70)
print("Task 4: 실제 감독 결정과의 일치율")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)
df["_date"] = df["game_id"].str[:8].astype(int)
df["season"] = df["_date"] // 10000

# game_id 팀 코드 순서 검증: half='top'(원정 공격, 투수는 홈팀 소속)일 때
# 실제 소속팀과 일치하는 조합은 game_id[10:12](경험적으로 검증 — 투수 77637이
# 전 시즌 'HT' 하나로 일관되게 나옴; 반대 조합은 매 시즌 팀이 무작위로 흩어짐)
df["team_code"] = np.where(df["half"] == "top", df["game_id"].str.slice(10, 12),
                            df["game_id"].str.slice(8, 10))

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
skill_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RNG_SEED))
skill_model.fit(reg_data[FEATURES].values, reg_data["wpa_per_pa"].values)

counts = subset.groupby("pitcher_id").size()
eligible = set(counts[counts >= 20].index)
print(f"\n218-pool(등판 20회 이상 조건통제): {len(eligible)}명")

# ── 시즌 단위 스냅샷: 투수별·시즌별 prior_* 평균 (그 시점 근방 '컨디션' 프록시) ──
season_stats = (
    df.groupby(["pitcher_id", "season"])[FEATURES + ["prior_wpa_std"]]
    .mean()
    .dropna()
)
skill_adj_by_season = skill_model.predict(season_stats[FEATURES].values)
league_mean = float(np.mean(skill_adj_by_season))
season_stats["skill_adj_relative"] = skill_adj_by_season - league_mean
print(f"시즌×투수 스냅샷: {len(season_stats)}건")

# ── 팀·시즌별 로스터 프록시: 그 팀·시즌에 등판 기록이 있는 218-pool 투수 ──
team_season_roster = (
    df[df["pitcher_id"].isin(eligible)]
    .groupby(["team_code", "season"])["pitcher_id"]
    .apply(lambda s: sorted(set(s)))
)
print(f"팀×시즌 로스터 그룹: {len(team_season_roster)}개")

# ── 실제 교체 이벤트: eligible 투수가 실제로 기용된 조건 통제 이벤트 ──
events = subset[subset["pitcher_id"].isin(eligible)].copy()
print(f"\n분석 대상 실제 교체 이벤트: {len(events)}건 (eligible 투수가 실제 기용된 경우)")

rng = np.random.default_rng(RNG_SEED)
N_EVENTS = 800
sample_events = events.sample(n=min(N_EVENTS, len(events)), random_state=RNG_SEED)

usable = []
for _, ev in sample_events.iterrows():
    team, season, actual_pid = ev["team_code"], ev["season"], ev["pitcher_id"]
    roster = team_season_roster.get((team, season), [])
    candidates = [p for p in roster if p != actual_pid]
    if not candidates:
        continue
    candidates.append(actual_pid)
    # 후보들의 시즌 스냅샷이 존재해야 함
    valid = [p for p in candidates if (p, season) in season_stats.index]
    if actual_pid not in valid or len(valid) < 2:
        continue
    usable.append({"team": team, "season": season, "actual_pid": actual_pid, "candidates": valid})

print(f"후보군(2명 이상) 구성 가능 이벤트: {len(usable)}건")

LAMBDA_GRID = np.round(np.arange(0.0, 1.01, 0.02), 3)
match_rates = []
for lam in LAMBDA_GRID:
    n_match = 0
    for ev in usable:
        rows = season_stats.loc[[(p, ev["season"]) for p in ev["candidates"]]]
        u = rows["skill_adj_relative"].values - lam * rows["prior_wpa_std"].values
        best_idx = int(np.argmax(u))
        best_pid = ev["candidates"][best_idx]
        if best_pid == ev["actual_pid"]:
            n_match += 1
    rate = n_match / len(usable) if usable else float("nan")
    match_rates.append(rate)

match_df = pd.DataFrame({"lambda": LAMBDA_GRID, "match_rate": match_rates})
best_row = match_df.loc[match_df["match_rate"].idxmax()]
print(f"\nλ별 일치율 (일부):")
for lam, rate in zip(LAMBDA_GRID[::5], match_rates[::5]):
    print(f"  λ={lam:.2f}: {rate:.4f}")
print(f"\n▶ 최대 일치율: λ={best_row['lambda']:.3f}에서 {best_row['match_rate']:.4f}")

# baseline: 무작위 선택 시 기대 일치율 (후보 수 역수의 평균)
random_baseline = float(np.mean([1 / len(ev["candidates"]) for ev in usable])) if usable else float("nan")
print(f"무작위 baseline 일치율: {random_baseline:.4f}")

# baseline: 항상 prior_n_apps(경험) 최댓값 선택
n_apps_match = 0
for ev in usable:
    rows = season_stats.loc[[(p, ev["season"]) for p in ev["candidates"]]]
    best_idx = int(np.argmax(rows["prior_n_apps"].values))
    if ev["candidates"][best_idx] == ev["actual_pid"]:
        n_apps_match += 1
n_apps_baseline = n_apps_match / len(usable) if usable else float("nan")
print(f"'경험 최다 투수 선택' baseline 일치율: {n_apps_baseline:.4f}")

report = {
    "n_events_sampled": len(sample_events), "n_usable": len(usable),
    "lambda_grid": LAMBDA_GRID.tolist(), "match_rates": match_rates,
    "best_lambda": float(best_row["lambda"]), "best_match_rate": float(best_row["match_rate"]),
    "random_baseline": random_baseline, "n_apps_baseline": n_apps_baseline,
    "beats_random": bool(best_row["match_rate"] > random_baseline),
}

with open(OUTPUT_DIR / "task4_manager_match_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 완료! {OUTPUT_DIR / 'task4_manager_match_report.json'}")
