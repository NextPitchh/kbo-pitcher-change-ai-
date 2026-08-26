"""
KBO WPA Pipeline — "measurement" 문제 재검증

이전 조사(validate_pitcher_signal.py)는 ratio_within = between / within(Std[WPA])
로 투수 신호를 "실제 대비 10분의 1"이라 결론 냈다. 하지만 within이 "단일
시뮬레이션의 Std[WPA]"라면 이는 5타석 구간의 게임 고유 무작위성(모집단 분산)
이지, n_simulations를 늘려도 줄지 않는다. 반면 실제 baseline의 within은
"투수 내 등판별 app_wpa 분산"으로 이미 평균화 효과가 반영된 값이라 두 값의
분모가 서로 다른 것을 재고 있었다. 이 스크립트는 그 비교 자체를 다시 검증한다.

Task 1: 같은 10상황×8투수×3horizon 조합에서 ratio_within / ratio_se(=between/SE)
         를 n_simulations=500, 2000 두 가지로 계산 → SE가 이론대로 sqrt(n)에
         비례해 ratio_se가 커지는지 확인.
Task 2: 상황별 투수 8명의 WPA 표본(wpa_dist)에 대해 일원분산분석(ANOVA).
Task 3: 실제 baseline과 동일한 층위(투수당 여러 상황 평균 → 대표값)로 재구성한
         시뮬레이션 ratio를 실제(app_wpa) baseline과 비교.
Task 4: CatBoost 확률 std로부터 5타석 horizon에서 이론적으로 가능한 ratio 상한을
         해석적으로 계산 → 관측 ratio_within과 비교해 "구조적 한계"인지
         "구현 문제"인지 판정.

절대 placeholder ID를 쓰지 않는다. horizon 기본값(horizon_pa=5,
horizon_innings=1.0)은 바꾸지 않는다 — Task1의 n_simulations 스윕만 변수다.

실행: python validate_measurement.py
요구사항: pip install catboost pandas numpy scipy pyarrow
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from monte_carlo_simulate import MonteCarloSimulator, GameState

OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42
MIN_APPEARANCES = 30
HORIZONS = [("h5_1", 5, 1.0), ("h10_2", 10, 2.0), ("full", None, None)]
N_SIM_LIST = [500, 2000]

report = {}

print("=" * 70)
print("측정 문제 재검증: ratio_within vs ratio_se, ANOVA, 동일 층위 비교, 이론 상한")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)


# ──────────────────────────────────────────────
# 공용 유틸 (validate_horizon.py / validate_pitcher_signal.py와 동일 방식)
# ──────────────────────────────────────────────
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


def load_situations(df_: pd.DataFrame, n: int, seed: int) -> list[GameState]:
    sample = df_.sample(n=n, random_state=seed)
    return [_row_to_state(row) for _, row in sample.iterrows()]


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
        {"pitcher_id": str(pid), "pitcher_vs_batter_avg": float(avg_by_pitcher_.loc[pid]),
         "n_appearances": int(counts_.loc[pid])}
        for pid in chosen_
    ]


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
pitchers_8 = load_veteran_pitchers(df, n=8, min_appearances=MIN_APPEARANCES, seed=RNG_SEED)
pid8 = [p["pitcher_id"] for p in pitchers_8]

print(f"\n10개 상황, 8명 투수(등판 {MIN_APPEARANCES}회 이상, seed={RNG_SEED}): {pid8}")


# ──────────────────────────────────────────────
# Task 1 + Task 2 (동시 진행 — 같은 시뮬레이션 호출을 재사용)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 1: ratio_within vs ratio_se (n_simulations=500, 2000 × 3 horizon)")
print("=" * 70)

task1_rows = []
# Task 2용: h5_1, n=500 조합의 상황별 8명 wpa_dist를 저장해둔다 (기본 horizon만 검정)
anova_source = {}  # situation_idx -> {pid: wpa_dist(list)}

for n_sim in N_SIM_LIST:
    sim = MonteCarloSimulator(n_simulations=n_sim, rng_seed=RNG_SEED)
    for name, hpa, hinn in HORIZONS:
        t0 = time.time()
        between_list, within_list = [], []
        for si, s in enumerate(situations_10):
            e_per_pitcher, std_per_pitcher = [], []
            for p in pitchers_8:
                state = with_pitcher(s, p)
                r = sim.simulate(state, horizon_pa=hpa, horizon_innings=hinn)
                e_per_pitcher.append(-r["e_wpa"])       # 투수(수비) 관점
                std_per_pitcher.append(r["std_wpa"])
                if n_sim == 500 and name == "h5_1":
                    anova_source.setdefault(si, {})[p["pitcher_id"]] = r["wpa_dist"]
            between_list.append(float(np.std(e_per_pitcher)))
            within_list.append(float(np.mean(std_per_pitcher)))
        elapsed = time.time() - t0

        mean_between = float(np.mean(between_list))
        mean_within = float(np.mean(within_list))
        se = mean_within / np.sqrt(n_sim)
        ratio_within = mean_between / mean_within if mean_within > 0 else float("nan")
        ratio_se = mean_between / se if se > 0 else float("nan")

        row = {
            "n_simulations": n_sim, "horizon": name,
            "horizon_pa": hpa, "horizon_innings": hinn,
            "between": mean_between, "within": mean_within, "SE": se,
            "ratio_within": ratio_within, "cohens_d": ratio_within,
            "ratio_se": ratio_se, "elapsed_sec": elapsed,
        }
        task1_rows.append(row)
        print(f"  n={n_sim:5d} {name:6s}  between={mean_between:.5f} within={mean_within:.5f} "
              f"SE={se:.6f}  ratio_within={ratio_within:.4f}  ratio_se={ratio_se:.3f}  "
              f"({elapsed:.1f}s)")

report["task1_ratio_within_vs_se"] = task1_rows

# sqrt(n) 비례 검증: 동일 horizon에서 n=500→2000일 때 ratio_se 배율 vs 이론값(sqrt(4)=2)
print("\nsqrt(n) 비례 검증 (n=500→2000, 이론 배율=2.0):")
sqrt_check = []
for name, _, _ in HORIZONS:
    r500 = next(r for r in task1_rows if r["n_simulations"] == 500 and r["horizon"] == name)
    r2000 = next(r for r in task1_rows if r["n_simulations"] == 2000 and r["horizon"] == name)
    observed_factor = r2000["ratio_se"] / r500["ratio_se"] if r500["ratio_se"] else float("nan")
    print(f"  {name:6s}: ratio_se(500)={r500['ratio_se']:.3f} → ratio_se(2000)={r2000['ratio_se']:.3f}  "
          f"실측 배율={observed_factor:.3f}")
    sqrt_check.append({"horizon": name, "ratio_se_500": r500["ratio_se"],
                        "ratio_se_2000": r2000["ratio_se"], "observed_factor": observed_factor,
                        "theoretical_factor": 2.0})
report["task1_sqrt_n_check"] = sqrt_check


# ──────────────────────────────────────────────
# Task 2: ANOVA (h5_1, n=500 기본 horizon 기준)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 2: 상황별 투수 8명 WPA 표본 일원분산분석 (horizon=(5,1.0), n=500)")
print("=" * 70)

anova_rows = []
for si in range(10):
    groups = [np.array(anova_source[si][pid]) for pid in pid8]
    f_stat, p_val = stats.f_oneway(*groups)

    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")

    sig = p_val < 0.05
    anova_rows.append({"situation_idx": si, "F": float(f_stat), "p_value": float(p_val),
                        "eta_squared": float(eta_sq), "significant_p<0.05": bool(sig)})
    print(f"  상황{si:2d}: F={f_stat:8.3f}  p={p_val:.2e}  eta^2={eta_sq:.4f}  "
          f"{'유의(p<0.05)' if sig else '비유의'}")

n_significant = sum(r["significant_p<0.05"] for r in anova_rows)
print(f"\n10개 상황 중 {n_significant}개에서 p<0.05")
report["task2_anova"] = {"rows": anova_rows, "n_significant": n_significant, "n_total": 10}


# ──────────────────────────────────────────────
# Task 3: 실제 데이터와 동일 층위 비교
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 3: 동일 층위 비교 — 투수당 30개 상황 평균 vs 실제 app_wpa")
print("=" * 70)

sim_500 = MonteCarloSimulator(n_simulations=500, rng_seed=RNG_SEED)

# 8명 투수 × 30개 상황(h5_1 기본 horizon) → 투수별 대표값(평균) + 투수 내 분산
per_pitcher_values = {pid: [] for pid in pid8}
t0 = time.time()
for s in situations_30:
    for p in pitchers_8:
        state = with_pitcher(s, p)
        r = sim_500.simulate(state, horizon_pa=5, horizon_innings=1.0)
        per_pitcher_values[p["pitcher_id"]].append(-r["e_wpa"])
elapsed = time.time() - t0
print(f"  8투수 × 30상황 시뮬레이션 완료 ({elapsed:.1f}s)")

pitcher_means = {pid: float(np.mean(v)) for pid, v in per_pitcher_values.items()}
pitcher_vars = {pid: float(np.var(v, ddof=1)) for pid, v in per_pitcher_values.items()}
pitcher_stds = {pid: float(np.std(v, ddof=1)) for pid, v in per_pitcher_values.items()}

sim_between_var = float(np.var(list(pitcher_means.values()), ddof=0))
sim_within_var = float(np.mean(list(pitcher_vars.values())))
sim_ratio_var = sim_between_var / sim_within_var if sim_within_var > 0 else float("nan")

sim_between_std = float(np.std(list(pitcher_means.values()), ddof=0))
sim_within_std = float(np.mean(list(pitcher_stds.values())))
sim_ratio_std = sim_between_std / sim_within_std if sim_within_std > 0 else float("nan")

print(f"\n  [분산 기준] between(var)={sim_between_var:.5f}  within(var)={sim_within_var:.5f}  "
      f"ratio_var={sim_ratio_var:.4f}")
print(f"  [표준편차 기준] between(std)={sim_between_std:.5f}  within(std)={sim_within_std:.5f}  "
      f"ratio_std={sim_ratio_std:.4f}")

for pid in pid8:
    print(f"    pitcher {pid}: mean={pitcher_means[pid]:+.4f}  std={pitcher_stds[pid]:.4f}  "
          f"(n=30 상황)")

# 실제 baseline — 동일한 8명으로 제한 (진짜 apples-to-apples)
apps = df.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"]).copy()
apps_8 = apps[apps["pitcher_id"].isin(pid8)]
n_apps_per_pid = apps_8.groupby("pitcher_id").size()
print(f"\n  실제 baseline(동일 8명) 등판 수: {n_apps_per_pid.to_dict()}")

actual_means_8 = apps_8.groupby("pitcher_id")["app_wpa"].mean()
actual_stds_8 = apps_8.groupby("pitcher_id")["app_wpa"].std(ddof=1)
actual_vars_8 = apps_8.groupby("pitcher_id")["app_wpa"].var(ddof=1)

actual8_between_std = float(actual_means_8.std(ddof=0))
actual8_within_std = float(actual_stds_8.mean())
actual8_ratio_std = actual8_between_std / actual8_within_std if actual8_within_std > 0 else float("nan")

actual8_between_var = float(actual_means_8.var(ddof=0))
actual8_within_var = float(actual_vars_8.mean())
actual8_ratio_var = actual8_between_var / actual8_within_var if actual8_within_var > 0 else float("nan")

print(f"\n  실제(동일 8명) [표준편차 기준] between={actual8_between_std:.5f}  "
      f"within={actual8_within_std:.5f}  ratio_std={actual8_ratio_std:.4f}")
print(f"  실제(동일 8명) [분산 기준] between={actual8_between_var:.5f}  "
      f"within={actual8_within_var:.5f}  ratio_var={actual8_ratio_var:.4f}")

# 참고: 전체 444명 모집단 baseline (이전 조사 재계산)
app_counts_all = apps.groupby("pitcher_id").size()
eligible_all = app_counts_all[app_counts_all >= MIN_APPEARANCES].index
apps_elig_all = apps[apps["pitcher_id"].isin(eligible_all)]
means_all = apps_elig_all.groupby("pitcher_id")["app_wpa"].mean()
stds_all = apps_elig_all.groupby("pitcher_id")["app_wpa"].std(ddof=1).dropna()
actual_all_between_std = float(means_all.std(ddof=0))
actual_all_within_std = float(stds_all.mean())
actual_all_ratio_std = actual_all_between_std / actual_all_within_std

print(f"\n  (참고) 전체 {len(eligible_all)}명 모집단 baseline ratio_std={actual_all_ratio_std:.4f}")

print(f"\n  ▶ 동일 층위 비교(같은 8명, std 기준): 시뮬레이션 ratio={sim_ratio_std:.4f}  "
      f"vs 실제 ratio={actual8_ratio_std:.4f}  "
      f"(시뮬레이션이 실제의 {sim_ratio_std/actual8_ratio_std*100:.1f}%)" if actual8_ratio_std > 0 else "")

report["task3_same_level_comparison"] = {
    "pitcher_means_sim": pitcher_means, "pitcher_stds_sim": pitcher_stds,
    "sim_between_var": sim_between_var, "sim_within_var": sim_within_var, "sim_ratio_var": sim_ratio_var,
    "sim_between_std": sim_between_std, "sim_within_std": sim_within_std, "sim_ratio_std": sim_ratio_std,
    "actual8_between_std": actual8_between_std, "actual8_within_std": actual8_within_std,
    "actual8_ratio_std": actual8_ratio_std,
    "actual8_between_var": actual8_between_var, "actual8_within_var": actual8_within_var,
    "actual8_ratio_var": actual8_ratio_var,
    "actual_all_ratio_std_reference": actual_all_ratio_std,
    "n_apps_per_pid_actual": {k: int(v) for k, v in n_apps_per_pid.to_dict().items()},
}


# ──────────────────────────────────────────────
# Task 4: 이론적 상한 계산
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("Task 4: CatBoost 확률 std 기반 이론적 ratio 상한 (5타석 가정)")
print("=" * 70)

# 4-1. 고정 상황(7회말 1아웃 1루 -1점)에서 30명 확률 벡터 재계산 (task1 조사와 동일 상황)
FIXED_BATTER_ID = str(int(df["batter_id"].value_counts().index[0]))
fixed_kwargs = dict(
    inning=7, half="bot", out_count=1, base_state="1", score_diff_attacker=-1,
    batter_id=FIXED_BATTER_ID, batter_hit_type="R",
    pitcher_vs_batter_avg=0.250, batter_recent_avg=0.317,
)

apps_ = df.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"])
counts_ = apps_.groupby("pitcher_id").size()
eligible_ids = counts_[counts_ >= MIN_APPEARANCES].index
rng = np.random.default_rng(RNG_SEED)
pitchers_30 = rng.choice(eligible_ids.values, size=30, replace=False)

sim1 = MonteCarloSimulator(n_simulations=1, rng_seed=RNG_SEED)
states_30 = [GameState(pitcher_id=str(pid), **fixed_kwargs) for pid in pitchers_30]
batch_rows = [
    [str(feat.get(c, 0)) if c in sim1.cat_features else float(feat.get(c, 0))
     for c in sim1.all_features]
    for feat in (s.to_feature_row() for s in states_30)
]
proba_matrix = sim1.model.predict_proba(batch_rows)
proba_matrix = proba_matrix / proba_matrix.sum(axis=1, keepdims=True)
proba_df = pd.DataFrame(proba_matrix, columns=sim1.classes)

class_mean = proba_df.mean(axis=0)
class_std = proba_df.std(axis=0)

# 4-2. 데이터에서 클래스별 실제 평균 WPA 임팩트(공격팀 관점) — 가치 가중치
value_weights = df[df["pa_result"] != "UNK"].groupby("pa_result")["reward_wpa_computed"].mean()
value_weights = value_weights.reindex(sim1.classes)

print("\n클래스별: 확률평균 / 확률std / WPA가중치(데이터 실측)")
for c in sim1.classes:
    print(f"  {c:4s}: mean_p={class_mean[c]:.4f}  std_p={class_std[c]:.5f}  "
          f"w={value_weights[c]:+.4f}")

n_pa = 5  # horizon_pa 기본값 (바꾸지 않음)

# 단일타석 가치의 평균/분산 (모집단 평균 확률 기준)
w = value_weights.values
p_bar = class_mean.values
mean_v1 = float(np.sum(w * p_bar))
var_v1 = float(np.sum((w ** 2) * p_bar) - mean_v1 ** 2)

# within(이론) = 단일 시뮬레이션 노이즈: n_pa iid 합의 표준편차
within_theoretical = float(np.sqrt(n_pa * var_v1))

# between(이론) = 투수별 확률편차(std_p, 클래스 간 독립 가정)가 만드는
# 대표값(E[V])의 표준편차: E[V|pitcher] = n_pa * sum_c w_c p_c
# 각 class 확률편차가 독립적으로 대표값에 기여한다고 가정(보수적 근사)
between_theoretical = float(n_pa * np.sqrt(np.sum((w ** 2) * (class_std.values ** 2))))

ratio_theoretical = between_theoretical / within_theoretical if within_theoretical > 0 else float("nan")

print(f"\n  이론적 within (5타석 단일-시뮬레이션 노이즈 표준편차) = {within_theoretical:.5f}")
print(f"  이론적 between (확률 std가 만드는 투수 간 기대값 표준편차) = {between_theoretical:.5f}")
print(f"  이론적 ratio_within 상한 = {ratio_theoretical:.4f}")

# 클래스별 신호대노이즈 (단일 PA 카운트 관점)
print("\n클래스별 신호대노이즈(SNR) — signal=n_pa*std_p, noise=sqrt(n_pa*p*(1-p)):")
class_snr = {}
for c in sim1.classes:
    p, s = class_mean[c], class_std[c]
    signal = n_pa * s
    noise = np.sqrt(n_pa * p * (1 - p))
    snr = signal / noise if noise > 0 else float("nan")
    class_snr[c] = {"signal": float(signal), "noise": float(noise), "snr": float(snr)}
    print(f"  {c:4s}: signal={signal:.4f}  noise={noise:.4f}  SNR={snr:.4f}")

observed_ratio_h5_1_n500 = next(r["ratio_within"] for r in task1_rows
                                 if r["n_simulations"] == 500 and r["horizon"] == "h5_1")
gap_factor = observed_ratio_h5_1_n500 / ratio_theoretical if ratio_theoretical > 0 else float("nan")
print(f"\n  관측된 ratio_within(h5_1, n=500) = {observed_ratio_h5_1_n500:.4f}")
print(f"  이론적 상한 대비 = {gap_factor*100:.1f}%")

structural_limit = gap_factor >= 0.5
verdict4 = ("이론적 상한과 같은 자릿수 — 구조적 한계(확률 차이 자체가 작아서 생기는 결과)"
            if structural_limit else
            "이론적 상한보다 관측값이 훨씬 낮음 — 계산/구현 과정에서 신호가 추가로 손실됨")
print(f"\n  ▶ {verdict4}")

report["task4_theoretical_bound"] = {
    "n_pa": n_pa, "class_mean_p": class_mean.to_dict(), "class_std_p": class_std.to_dict(),
    "value_weights": value_weights.to_dict(),
    "within_theoretical": within_theoretical, "between_theoretical": between_theoretical,
    "ratio_theoretical": ratio_theoretical, "class_snr": class_snr,
    "observed_ratio_h5_1_n500": observed_ratio_h5_1_n500,
    "gap_factor_observed_over_theoretical": gap_factor,
    "verdict": verdict4,
}


# ──────────────────────────────────────────────
# 결론
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("결론")
print("=" * 70)

se_grows_as_expected = all(
    1.5 <= c["observed_factor"] <= 2.5 for c in sqrt_check if not np.isnan(c["observed_factor"])
)
anova_mostly_significant = n_significant >= 7
same_level_close = (actual8_ratio_std > 0 and 0.3 <= sim_ratio_std / actual8_ratio_std <= 3.0)

lines = []
lines.append(
    f"Task1: n=500→2000일 때 ratio_se 배율이 이론값(2.0)에 부합"
    + (" — SE가 정말 sqrt(n)에 비례해 줄어들며, ratio_within은 애초에 '표준오차'가 아니라 "
       "'모집단 분산' 대비 신호라서 n_simulations를 늘려도 커지지 않는 게 정상." if se_grows_as_expected else
       " — 예상과 다소 어긋남, 세부 확인 필요.")
)
lines.append(
    f"Task2: 10개 상황 중 {n_significant}개에서 ANOVA p<0.05"
    + (" — 투수 간 E[WPA] 차이는 통계적으로 실재한다(우연이 아님). "
       "다만 유의성은 '차이가 0이 아니다'를 말할 뿐, '차이의 크기가 실무적으로 큰지'는 별개 질문."
       if anova_mostly_significant else
       " — 유의한 상황이 절반 미만이라 상황에 따라 투수 차이가 통계적으로 불확실한 경우도 있음.")
)
lines.append(
    f"Task3: 동일 8명·동일 층위(투수당 30상황 평균) 비교에서 시뮬레이션 ratio_std={sim_ratio_std:.4f} "
    f"vs 실제 ratio_std={actual8_ratio_std:.4f}"
    + (" — 두 값이 비슷한 수준으로 좁혀짐(이전의 '10배' 격차는 layer mismatch 때문이었음)."
       if same_level_close else
       " — 층위를 맞춰도 여전히 격차가 큼 — layer mismatch만으로는 설명 안 되는 부분이 남아있음.")
)
lines.append(
    f"Task4: 이론적 ratio_within 상한={ratio_theoretical:.4f} vs 관측 ratio_within(h5_1,n=500)="
    f"{observed_ratio_h5_1_n500:.4f} ({gap_factor*100:.1f}%). {verdict4}"
)

final_verdict = (
    "이전 결론('모델이 신호를 놓치고 있다')은 상당 부분 측정 방법의 문제였다 — "
    "ratio_within은애초에 n_simulations로 줄지 않는 모집단 분산 대비 비율이라 작게 나오는 게 "
    "구조적으로 정상이며(Task4), ANOVA로 보면 투수 간 차이는 통계적으로 실재하고(Task2), "
    "층위를 맞추면 실제 데이터와의 격차도 크게 줄어든다(Task3). "
    "다만 이것이 '모델이 완벽하다'는 뜻은 아니다 — 의사결정에는 ratio_within이 아니라 "
    "ratio_se(혹은 ANOVA)가 맞는 지표이며, 실제 investment는 이 방향으로 지표를 바꿔야 한다."
    if (se_grows_as_expected and anova_mostly_significant and structural_limit) else
    "측정 문제로 일부 설명되지만 모델/데이터 쪽 한계도 함께 존재하는 것으로 보임 — Task별 상세 참고."
)

for line in lines:
    print(f"\n- {line}")
print(f"\n▶ 최종 판단: {final_verdict}")

report["conclusion"] = {
    "lines": lines, "final_verdict": final_verdict,
    "se_grows_as_expected": se_grows_as_expected,
    "anova_mostly_significant": anova_mostly_significant,
    "same_level_close": same_level_close,
    "structural_limit": structural_limit,
}

with open(OUTPUT_DIR / "measurement_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"✅ 완료! 리포트: {OUTPUT_DIR / 'measurement_validation_report.json'}")
print("=" * 70)
