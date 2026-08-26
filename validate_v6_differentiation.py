"""
KBO WPA Pipeline — v6 피처셋(A/B/C/D)별 투수 분화도 측정 (Task 3)

이번 작업의 성패 지표. validate_pitcher_signal.py Task1과 동일한 고정 상황
(7회말 1아웃 1루 -1점)·동일 선정 방식(등판 30회 이상 투수 30명, seed=42)으로
각 피처셋 모델의 확률 벡터를 비교한다.

기준값(v5/A_baseline 재현): BB std=0.0404, SO std=0.0234.
prior_bb_rate/prior_so_rate 추가로 이 값이 커지는지가 핵심 검증 포인트.

각 후보 투수에게는 실제 데이터에서 뽑은 그 투수의 최신 prior_* 스냅샷과
실제 pitcher_throws를 붙인다(가짜 값 사용 금지 — 모델이 실제로 그 투수를
어떻게 보는지 측정해야 하므로).

실행: python validate_v6_differentiation.py
사전 조건: catboost_feature_compare.py 실행 완료 (catboost_output/compare_v6/*)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monte_carlo_simulate import MonteCarloSimulator, GameState

COMPARE_DIR = Path("catboost_output/compare_v6")
OUTPUT_DIR = Path("validation_output")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42
MIN_APPEARANCES = 30
VARIANTS = ["A_baseline", "B_prior", "C_hand", "D_no_pvb"]

print("=" * 70)
print("Task 3: v6 피처셋별 투수 분화도 (BB/SO std 변화가 핵심)")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
df["pitcher_id"] = df["pitcher_id"].astype(str)

apps = df.drop_duplicates(subset=["pitcher_id", "game_id", "appearance_order"]).copy()
app_counts = apps.groupby("pitcher_id").size()
eligible_ids = app_counts[app_counts >= MIN_APPEARANCES].index

rng = np.random.default_rng(RNG_SEED)
pitchers_30 = rng.choice(eligible_ids.values, size=30, replace=False)
print(f"\n등판 {MIN_APPEARANCES}회 이상 투수 {len(eligible_ids)}명 중 30명 샘플 (seed={RNG_SEED})")

# 각 투수의 "최신" prior_* 스냅샷 + 실제 throws — 등판 순서(appearance_order)가
# 아니라 date 기준으로 가장 최근 것을 사용 (현재 시점 그 투수를 모델이 어떻게
# 보는지가 목적이므로).
df["_date"] = df["game_id"].str[:8].astype(int)
last_rows = (
    df[df["pitcher_id"].isin(pitchers_30)]
    .sort_values("_date")
    .drop_duplicates(subset=["pitcher_id"], keep="last")
    .set_index("pitcher_id")
)

FIXED_BATTER_ID = str(int(df["batter_id"].value_counts().index[0]))
FIXED_KWARGS_BASE = dict(
    inning=7, half="bot", out_count=1, base_state="1", score_diff_attacker=-1,
    batter_id=FIXED_BATTER_ID, batter_hit_type="R",
    pitcher_vs_batter_avg=0.250, batter_recent_avg=0.317,
)


def build_state(pid: str) -> GameState:
    row = last_rows.loc[pid]
    kwargs = dict(FIXED_KWARGS_BASE)
    kwargs["pitcher_id"] = pid
    kwargs["pitcher_throws"] = str(row["pitcher_throws"]) if pd.notna(row["pitcher_throws"]) else "UNK"
    for c in ["prior_bb_rate", "prior_so_rate", "prior_hr_rate",
              "prior_wpa_std", "prior_n_apps", "prior_avg_pa_per_app"]:
        val = row[c]
        kwargs[c] = float(val) if pd.notna(val) else float("nan")
    return GameState(**kwargs)


states_30 = [build_state(str(pid)) for pid in pitchers_30]

report = {}
summary_rows = []

for variant in VARIANTS:
    model_path = COMPARE_DIR / f"catboost_{variant}.cbm"
    meta_path = COMPARE_DIR / f"model_meta_{variant}.json"
    if not model_path.exists():
        print(f"\n⚠️  {variant}: 모델 파일 없음 ({model_path}) — 건너뜀")
        continue

    print("\n" + "=" * 70)
    print(f"모델: {variant}")
    print("=" * 70)
    sim = MonteCarloSimulator(n_simulations=1, model_path=str(model_path),
                               meta_path=str(meta_path), rng_seed=RNG_SEED)

    batch_rows = [
        [str(feat.get(c, 0)) if c in sim.cat_features else float(feat.get(c, 0))
         for c in sim.all_features]
        for feat in (s.to_feature_row() for s in states_30)
    ]
    proba_matrix = sim.model.predict_proba(batch_rows)
    proba_matrix = proba_matrix / proba_matrix.sum(axis=1, keepdims=True)
    proba_df = pd.DataFrame(proba_matrix, columns=sim.classes,
                             index=[str(pid) for pid in pitchers_30])

    class_std = proba_df.std(axis=0).sort_values(ascending=False)
    class_mean = proba_df.mean(axis=0)

    eps = 1e-9
    P = proba_matrix
    n_p = P.shape[0]
    l1_sum, n_pairs = 0.0, 0
    for i in range(n_p):
        for j in range(i + 1, n_p):
            l1_sum += np.abs((P[i] + eps) - (P[j] + eps)).sum()
            n_pairs += 1
    mean_l1 = l1_sum / n_pairs

    bb_std = float(class_std.get("BB", float("nan")))
    so_std = float(class_std.get("SO", float("nan")))

    print(f"클래스별 확률 std (내림차순):")
    for cls, s in class_std.items():
        print(f"  {cls:4s}: std={s:.5f}  mean={class_mean[cls]:.5f}")
    print(f"평균 쌍별 L1 거리: {mean_l1:.5f}")
    print(f"\nBB std={bb_std:.5f}  SO std={so_std:.5f}")

    report[variant] = {
        "class_std": {k: float(v) for k, v in class_std.items()},
        "class_mean": {k: float(v) for k, v in class_mean.items()},
        "mean_pairwise_l1": float(mean_l1),
        "bb_std": bb_std, "so_std": so_std,
    }
    summary_rows.append({"variant": variant, "BB_std": bb_std, "SO_std": so_std,
                          "mean_pairwise_l1": mean_l1,
                          "GDP_std": float(class_std.get("GDP", float("nan"))),
                          "OUT_std": float(class_std.get("OUT", float("nan")))})

print("\n" + "=" * 70)
print("요약: 기준(v5) BB std=0.0404, SO std=0.0234")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

if len(summary_df) > 0:
    base_bb = summary_df.loc[summary_df["variant"] == "A_baseline", "BB_std"]
    base_so = summary_df.loc[summary_df["variant"] == "A_baseline", "SO_std"]
    if len(base_bb) and len(base_so):
        for _, row in summary_df.iterrows():
            bb_chg = (row["BB_std"] / base_bb.iloc[0] - 1) * 100
            so_chg = (row["SO_std"] / base_so.iloc[0] - 1) * 100
            print(f"  {row['variant']:12s}: BB std {bb_chg:+.1f}%  SO std {so_chg:+.1f}%  (vs A_baseline)")

report["summary_table"] = summary_rows

with open(OUTPUT_DIR / "v6_differentiation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 완료! {OUTPUT_DIR / 'v6_differentiation_report.json'}")
