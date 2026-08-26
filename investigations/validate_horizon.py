"""
KBO WPA Pipeline — horizon 파라미터 검증

monte_carlo_simulate.py의 simulate()에 추가된 horizon_pa / horizon_innings
(측정 구간을 "이번 등판"으로 제한하는 기능)가 의도대로 동작하는지 검증한다.

절대 placeholder ID(99999/88888)를 쓰지 않는다 — 학습 데이터에 없는 ID라
모델이 비정상 확률을 출력하므로, 전부 all_pa_enriched.parquet의 실제
pitcher_id/batter_id를 사용한다.

실행: python investigations/validate_horizon.py (레포 루트 기준)
요구사항: pip install catboost pandas numpy pyarrow matplotlib
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo_simulate import MonteCarloSimulator, GameState

OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42

HORIZONS = [
    ("h5_1",   5,    1.0),
    ("h10_2",  10,   2.0),
    ("full",   None, None),
]

report = {}


# ──────────────────────────────────────────────
# 0. 데이터 로드 유틸 — 전부 실제 ID 사용
# ──────────────────────────────────────────────
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


def load_situations(df: pd.DataFrame, n: int, seed: int) -> list[GameState]:
    sample = df.sample(n=n, random_state=seed)
    return [_row_to_state(row) for _, row in sample.iterrows()]


def load_veteran_pitchers(df: pd.DataFrame, n: int = 8, min_appearances: int = 30, seed: int = 42) -> list[dict]:
    apps = df.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"])
    counts = apps.groupby("pitcher_id").size()
    eligible = counts[counts >= min_appearances].index
    avg_by_pitcher = (
        df[df["pitcher_id"].isin(eligible)]
        .groupby("pitcher_id")["pitcher_vs_batter_avg"].mean()
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(avg_by_pitcher.index.values, size=n, replace=False)
    return [
        {
            "pitcher_id": str(int(pid)),
            "pitcher_vs_batter_avg": float(avg_by_pitcher.loc[pid]),
            "n_appearances": int(counts.loc[pid]),
        }
        for pid in chosen
    ]


def with_pitcher(situation: GameState, pitcher: dict) -> GameState:
    """situation의 타석 문맥은 유지하되 투수만 교체."""
    return GameState(
        inning=situation.inning, half=situation.half,
        out_count=situation.out_count, base_state=situation.base_state,
        score_diff_attacker=situation.score_diff_attacker,
        pitcher_id=pitcher["pitcher_id"], batter_id=situation.batter_id,
        batter_hit_type=situation.batter_hit_type,
        pitcher_vs_batter_avg=pitcher["pitcher_vs_batter_avg"],
        batter_recent_avg=situation.batter_recent_avg,
    )


print("=" * 70)
print("horizon 파라미터 검증")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()

situations_30 = load_situations(df, n=30, seed=RNG_SEED)
situations_10 = situations_30[:10]
pitchers_8 = load_veteran_pitchers(df, n=8, min_appearances=30, seed=RNG_SEED)

print(f"\n대표 상황 30개 로드 완료 (앞 10개를 3-1/3-2에 재사용)")
print(f"등판 30회 이상 투수 8명:")
for p in pitchers_8:
    print(f"  pitcher_id={p['pitcher_id']:8s} n_appearances={p['n_appearances']:4d} "
          f"vs타율(평균)={p['pitcher_vs_batter_avg']:.3f}")


# ──────────────────────────────────────────────
# 3-1) horizon별 비교 (10개 상황 × 3 horizon, n=1000)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-1) horizon별 비교 (10개 실제 상황, n=1000)")
print("=" * 70)

sim_1000 = MonteCarloSimulator(n_simulations=1000, rng_seed=RNG_SEED)

horizon_compare_rows = []
for name, hpa, hinn in HORIZONS:
    e_list, var_list, std_list, pa_list, path_agg = [], [], [], [], {}
    t0 = time.time()
    for s in situations_10:
        r = sim_1000.simulate(s, horizon_pa=hpa, horizon_innings=hinn)
        e_list.append(r["e_wpa"]); var_list.append(r["var_wpa"]); std_list.append(r["std_wpa"])
        pa_list.append(r["avg_pa_simulated"])
        for k, v in r["termination_reason"].items():
            path_agg[k] = path_agg.get(k, 0) + v
    elapsed = time.time() - t0

    row = {
        "horizon": name,
        "horizon_pa": hpa, "horizon_innings": hinn,
        "mean_E": float(np.mean(e_list)),
        "mean_Var": float(np.mean(var_list)),
        "mean_Std": float(np.mean(std_list)),
        "mean_avg_pa_simulated": float(np.mean(pa_list)),
        "elapsed_sec_10situations": elapsed,
        "elapsed_sec_per_call": elapsed / 10,
        "termination_reason_pooled": path_agg,
    }
    horizon_compare_rows.append(row)
    print(f"\n[{name}] horizon_pa={hpa}, horizon_innings={hinn}")
    print(f"  mean E[WPA]={row['mean_E']:+.4f}  mean Var={row['mean_Var']:.5f}  mean Std={row['mean_Std']:.4f}")
    print(f"  평균 소화 타석={row['mean_avg_pa_simulated']:.2f}  소요={elapsed:.3f}s (10상황, {elapsed/10*1000:.1f}ms/call)")
    print(f"  종료 경로(10상황 풀링): {path_agg}")

report["3-1_horizon_compare"] = horizon_compare_rows


# ──────────────────────────────────────────────
# 3-2) 투수 간 격차 (핵심 지표)
#      10개 상황 × 8명 투수 × 3 horizon
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-2) 투수 간 격차 측정 (핵심 지표) — n=500")
print("=" * 70)

sim_500 = MonteCarloSimulator(n_simulations=500, rng_seed=RNG_SEED)

# situation × pitcher × horizon → (E_pitcher_perspective, Std) 캐시
# (3-3에서 horizon=(5,1.0) 결과를 재사용하기 위해 별도 dict로 보관)
cache_5_1 = {}  # (situation_idx, pitcher_id) -> (E, Std)  — 30상황 전체용, 아래 3-3에서 채움

gap_rows = []
for name, hpa, hinn in HORIZONS:
    between_list, within_list = [], []
    for si, s in enumerate(situations_10):
        e_per_pitcher, std_per_pitcher = [], []
        for p in pitchers_8:
            state = with_pitcher(s, p)
            r = sim_500.simulate(state, horizon_pa=hpa, horizon_innings=hinn)
            e_pitcher = -r["e_wpa"]  # 투수(수비) 관점
            e_per_pitcher.append(e_pitcher)
            std_per_pitcher.append(r["std_wpa"])
            if name == "h5_1":
                cache_5_1[(si, p["pitcher_id"])] = (e_pitcher, r["std_wpa"])
        between_list.append(float(np.std(e_per_pitcher)))   # 상황 내 투수 간 E 표준편차
        within_list.append(float(np.mean(std_per_pitcher)))  # 상황 내 투수별 Std 평균

    between = float(np.mean(between_list))
    within  = float(np.mean(within_list))
    ratio   = between / within if within > 0 else float("nan")
    gap_rows.append({"horizon": name, "horizon_pa": hpa, "horizon_innings": hinn,
                      "between": between, "within": within, "ratio": ratio})
    print(f"  {name:8s} (pa={str(hpa):5s} inn={str(hinn):5s})  "
          f"between={between:.4f}  within={within:.4f}  ratio={ratio:.4f}")

report["3-2_pitcher_gap"] = gap_rows

print("\n요약 표:")
print(f"{'horizon':10s} {'between':>10s} {'within':>10s} {'ratio':>10s}")
for row in gap_rows:
    print(f"{row['horizon']:10s} {row['between']:10.4f} {row['within']:10.4f} {row['ratio']:10.4f}")


# ──────────────────────────────────────────────
# 3-3) λ 임계값 분포 — 30개 상황 × 8명 투수, horizon=(5,1.0)만
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-3) λ 임계값 분포 (horizon=(5,1.0), n=500)")
print("=" * 70)

# situations_10(0~9)은 3-2에서 이미 계산된 cache_5_1 재사용, 나머지 20개만 추가 계산
for si in range(10, 30):
    s = situations_30[si]
    for p in pitchers_8:
        state = with_pitcher(s, p)
        r = sim_500.simulate(state, horizon_pa=5, horizon_innings=1.0)
        cache_5_1[(si, p["pitcher_id"])] = (-r["e_wpa"], r["std_wpa"])

print(f"situation×pitcher 캐시 크기: {len(cache_5_1)} (기대: 30*8=240)")

lambda_stars = []
for si in range(30):
    for i in range(len(pitchers_8)):
        for j in range(i + 1, len(pitchers_8)):
            pid_a, pid_b = pitchers_8[i]["pitcher_id"], pitchers_8[j]["pitcher_id"]
            e_a, std_a = cache_5_1[(si, pid_a)]
            e_b, std_b = cache_5_1[(si, pid_b)]
            if abs(std_a - std_b) < 1e-6:
                continue
            lam = (e_a - e_b) / (std_a - std_b)
            if lam > 0:
                lambda_stars.append(lam)

lambda_stars = np.array(lambda_stars)
n_valid = len(lambda_stars)
n_total_pairs = 30 * (8 * 7 // 2)
print(f"\n유효 λ* 표본: {n_valid} / 전체 쌍 {n_total_pairs} ({n_valid/n_total_pairs*100:.1f}%)")

if n_valid > 0:
    pcts = [5, 25, 50, 75, 95]
    pct_vals = {p: float(np.percentile(lambda_stars, p)) for p in pcts}
    print("λ* 분위수:")
    for p in pcts:
        print(f"  {p:2d}%ile: {pct_vals[p]:.4f}")

    q25, q50, q75, q90 = (pct_vals[25], pct_vals[50], pct_vals[75],
                           float(np.percentile(lambda_stars, 90)))
    q95 = pct_vals[95]

    recommended_slider = [0.0, q95]
    recommended_discrete = [0.0, q25, q50, q90]
    region_bounds = [q25, q75]

    print(f"\n권장 슬라이더 범위: [0.0, {q95:.4f}]")
    print(f"권장 이산 λ 4개: {[round(x,4) for x in recommended_discrete]}")
    print(f"구간 라벨 경계: q25={q25:.4f}, q75={q75:.4f}")

    # 히스토그램
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lambda_stars, bins=40, color="#4C72B0", edgecolor="white")
    for p in pcts:
        ax.axvline(pct_vals[p], color="#C44E52", linestyle="--", linewidth=1)
        ax.text(pct_vals[p], ax.get_ylim()[1]*0.95, f"{p}%", rotation=90,
                va="top", ha="right", fontsize=8, color="#C44E52")
    ax.set_xlabel("λ* = (E_A - E_B) / (Std_A - Std_B)")
    ax.set_ylabel("빈도")
    ax.set_title(f"λ 임계값 분포 (horizon=(5,1.0), n_valid={n_valid})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lambda_threshold_dist.png", dpi=120)
    plt.close(fig)
    print(f"\n히스토그램 저장: {OUTPUT_DIR / 'lambda_threshold_dist.png'}")

    # 검증: λ_max(q95)에서 λ=0 대비 추천이 달라지는지 (대표 상황 10개)
    n_changed = 0
    change_detail = []
    for si in range(10):
        e_std = [cache_5_1[(si, p["pitcher_id"])] for p in pitchers_8]
        e_vals = np.array([x[0] for x in e_std])
        std_vals = np.array([x[1] for x in e_std])

        u0 = e_vals - 0.0 * std_vals
        u_max = e_vals - q95 * std_vals
        best0 = int(np.argmax(u0))
        best_max = int(np.argmax(u_max))
        changed = best0 != best_max
        n_changed += changed
        change_detail.append({
            "situation_idx": si,
            "best_at_lambda0": pitchers_8[best0]["pitcher_id"],
            "best_at_lambda_max": pitchers_8[best_max]["pitcher_id"],
            "changed": changed,
        })

    print(f"\nλ=0 vs λ_max({q95:.4f}) 추천 변화: {n_changed}/10 "
          f"({'PASS' if n_changed >= 7 else 'FAIL'}, 기준 7/10 이상)")
    for d in change_detail:
        mark = "✓변화" if d["changed"] else " 동일"
        print(f"  상황{d['situation_idx']:2d}: {mark}  "
              f"λ=0→{d['best_at_lambda0']:8s}  λ_max→{d['best_at_lambda_max']:8s}")

    report["3-3_lambda_threshold"] = {
        "n_valid": n_valid, "n_total_pairs": n_total_pairs,
        "percentiles": pct_vals,
        "recommended_slider": recommended_slider,
        "recommended_discrete": recommended_discrete,
        "region_bounds": region_bounds,
        "lambda_max_recommendation_check": {
            "n_changed": n_changed, "n_total": 10, "pass": n_changed >= 7,
            "detail": change_detail,
        },
    }
else:
    print("⚠️  유효한 λ* 표본이 없습니다.")
    report["3-3_lambda_threshold"] = {"n_valid": 0}


# ──────────────────────────────────────────────
# 3-4) 도메인 정합성 (horizon=(5,1.0), 실제 ID 사용)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-4) 도메인 정합성 (horizon=(5,1.0), n=1000)")
print("=" * 70)

REAL_PID, REAL_BID = "70839", "71752"
REAL_PVB, REAL_BRA = 0.0, 0.167  # 3-5 회귀 테스트와 동일 실제 매치업 재사용

def mk(inning, half, out_count, base_state, score_diff_attacker):
    return GameState(
        inning=inning, half=half, out_count=out_count, base_state=base_state,
        score_diff_attacker=score_diff_attacker,
        pitcher_id=REAL_PID, batter_id=REAL_BID, batter_hit_type="R",
        pitcher_vs_batter_avg=REAL_PVB, batter_recent_avg=REAL_BRA,
    )

domain_checks = []

s_a = mk(9, "bot", 1, "123", 0)
r_a = sim_1000.simulate(s_a, horizon_pa=5, horizon_innings=1.0)
ok_a = r_a["e_wpa"] > 0
domain_checks.append(("9회말 동점 1아웃 만루 → E[WPA] 양수", r_a["e_wpa"], ok_a))

s_b = mk(1, "top", 0, "0", 0)
r_b = sim_1000.simulate(s_b, horizon_pa=5, horizon_innings=1.0)
ok_b = abs(r_b["e_wpa"]) < 0.05
domain_checks.append(("1회초 0-0 무사 주자없음 → E[WPA]≈0 (|E|<0.05)", r_b["e_wpa"], ok_b))

s_c = mk(7, "bot", 2, "0", 3)
r_c = sim_1000.simulate(s_c, horizon_pa=5, horizon_innings=1.0)
ok_c = abs(r_c["e_wpa"]) < 0.05
domain_checks.append(("7회말 3점리드 2아웃 → E[WPA] 작은 값 (|E|<0.05)", r_c["e_wpa"], ok_c))

s_d_top = mk(7, "top", 1, "1", -1)
s_d_bot = mk(7, "bot", 1, "1", -1)
r_d_top = sim_1000.simulate(s_d_top, horizon_pa=5, horizon_innings=1.0)
r_d_bot = sim_1000.simulate(s_d_bot, horizon_pa=5, horizon_innings=1.0)
diff_d = abs(r_d_top["e_wpa"] - r_d_bot["e_wpa"])
ok_d = diff_d < 0.05
domain_checks.append((f"대칭성: 7회 1아웃1루 -1점 top({r_d_top['e_wpa']:+.4f}) vs "
                       f"bot({r_d_bot['e_wpa']:+.4f}) 차이<0.05", diff_d, ok_d))

for label, val, ok in domain_checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (값={val:+.4f})")

report["3-4_domain_checks"] = [
    {"label": label, "value": float(val), "pass": bool(ok)} for label, val, ok in domain_checks
]


# ──────────────────────────────────────────────
# 3-5) 회귀 테스트 (horizon=None,None)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3-5) 회귀 테스트 (horizon=None,None, 관점 통일 수정 직후 기준값)")
print("=" * 70)

REF_E, REF_VAR = -0.09842100000000002, 0.16471519675899995

sim_regr = MonteCarloSimulator(n_simulations=2000, rng_seed=42)
s_regr = GameState(inning=7, half="bot", out_count=1, base_state="1", score_diff_attacker=-1,
    pitcher_id=REAL_PID, batter_id=REAL_BID, batter_hit_type="R",
    pitcher_vs_batter_avg=REAL_PVB, batter_recent_avg=REAL_BRA)
r_regr = sim_regr.simulate(s_regr, horizon_pa=None, horizon_innings=None)

match_e = abs(r_regr["e_wpa"] - REF_E) < 1e-9
match_var = abs(r_regr["var_wpa"] - REF_VAR) < 1e-9
print(f"  E[WPA]  = {r_regr['e_wpa']:.10f}  (기준 {REF_E:.10f})  {'일치' if match_e else '불일치'}")
print(f"  Var[WPA]= {r_regr['var_wpa']:.10f}  (기준 {REF_VAR:.10f})  {'일치' if match_var else '불일치'}")
print(f"  [{'PASS' if (match_e and match_var) else 'FAIL'}] horizon=None,None 회귀 테스트")

report["3-5_regression"] = {
    "e_wpa": r_regr["e_wpa"], "ref_e_wpa": REF_E, "match_e": match_e,
    "var_wpa": r_regr["var_wpa"], "ref_var_wpa": REF_VAR, "match_var": match_var,
    "pass": bool(match_e and match_var),
}


# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────
with open(OUTPUT_DIR / "horizon_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"✅ 완료! 리포트: {OUTPUT_DIR / 'horizon_validation_report.json'}")
print(f"   히스토그램: {OUTPUT_DIR / 'lambda_threshold_dist.png'}")
print("=" * 70)
