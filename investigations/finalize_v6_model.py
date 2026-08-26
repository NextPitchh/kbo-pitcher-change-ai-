"""
KBO WPA Pipeline — v6 최종 모델 선정 및 시뮬레이션 ratio 재측정 (Task 4 + Task 5)

Task 2(성능 비교) + Task 3(투수 분화도)를 검토한 뒤 결정한 MODEL_VARIANT를
아래에서 지정하고 실행한다. 이 스크립트는:

1. 선정된 변형의 .cbm/meta를 catboost_pa_classifier_v6.cbm / model_meta_v6.json
   으로 저장한다 (v5 파일은 절대 건드리지 않음 — 별도 파일명).
2. validate_confounds.py Task5와 동일한 공정 조건(불펜, 6~8회 시작,
   |점수차|<=3, horizon=5PA/1이닝 기본값, 동일 투수 풀 선정 로직)으로
   시뮬레이션 between/within ratio를 재계산한다. 목표: 0.049 → 0.10 이상.
   v6 모델은 prior_*/pitcher_throws 피처가 필요하므로, 각 후보 투수에게
   실제 데이터에서 뽑은 그 투수의 최신 prior_* 스냅샷을 붙인다.

실행: python investigations/finalize_v6_model.py (레포 루트 기준)
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monte_carlo_simulate import MonteCarloSimulator, GameState

# ──────────────────────────────────────────────
# 선정 결과 — Task2/Task3 리뷰 후 값 지정
# ──────────────────────────────────────────────
MODEL_VARIANT = "C_hand"

COMPARE_DIR = Path("catboost_output/compare_v6")
OUTPUT_DIR = Path("catboost_output")
VALIDATION_DIR = Path("validation_output")
VALIDATION_DIR.mkdir(exist_ok=True)

DATA_PATH = "data_analysis/results/all_pa_enriched.parquet"
RNG_SEED = 42


def finalize_model_files() -> Path:
    src_cbm = COMPARE_DIR / f"catboost_{MODEL_VARIANT}.cbm"
    src_meta = COMPARE_DIR / f"model_meta_{MODEL_VARIANT}.json"
    if not src_cbm.exists():
        raise FileNotFoundError(f"{src_cbm} 없음 — catboost_feature_compare.py를 먼저 실행하세요")

    dst_cbm = OUTPUT_DIR / "catboost_pa_classifier_v6.cbm"
    dst_meta = OUTPUT_DIR / "model_meta_v6.json"

    v5_cbm = OUTPUT_DIR / "catboost_pa_classifier.cbm"
    if not v5_cbm.exists():
        raise RuntimeError("v5 모델(catboost_pa_classifier.cbm)이 없습니다 — 절대 이 상태로 진행 금지")

    shutil.copy(src_cbm, dst_cbm)
    with open(src_meta, encoding="utf-8") as f:
        meta = json.load(f)
    meta["version"] = "v6"
    meta["selected_variant"] = MODEL_VARIANT
    meta["v5_preserved_at"] = str(v5_cbm)
    with open(dst_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장: {dst_cbm}")
    print(f"✅ 저장: {dst_meta}")
    print(f"✅ v5 보존 확인: {v5_cbm} (건드리지 않음)")
    return dst_cbm


def run_fair_comparison(model_path: Path):
    print("\n" + "=" * 70)
    print("Task 4/5: v6 모델 공정 비교 재측정 (불펜, 6~8회, |점수차|<=3, horizon=5PA/1이닝)")
    print("=" * 70)

    df = pd.read_parquet(DATA_PATH)
    df = df[df["data_quality_flag"] == ""].copy()
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    first_pa = df[df["current_pitcher_pa_in_app"] == 1].copy()
    first_pa["abs_score_diff_start"] = first_pa["score_diff_attacker"].abs()

    cond_all = (
        (first_pa["is_starter"] == False) &
        (first_pa["n_pa"] >= 3) & (first_pa["n_pa"] <= 7) &
        (first_pa["start_inning"] >= 6) & (first_pa["start_inning"] <= 8) &
        (first_pa["abs_score_diff_start"] <= 3)
    )
    subset_task1 = first_pa[cond_all]
    counts = subset_task1.groupby("pitcher_id").size()
    eligible = counts[counts >= 20].index

    means = subset_task1[subset_task1["pitcher_id"].isin(eligible)].groupby("pitcher_id")["app_wpa"].mean()
    stds = subset_task1[subset_task1["pitcher_id"].isin(eligible)].groupby("pitcher_id")["app_wpa"].std(ddof=1).dropna()
    actual_between = float(means.std(ddof=0))
    actual_within = float(stds.mean())
    actual_ratio = actual_between / actual_within
    print(f"\n실제(조건 통제) baseline: n_pitchers={len(eligible)} between={actual_between:.5f} "
          f"within={actual_within:.5f} ratio={actual_ratio:.4f}  (validate_confounds.py Task1과 동일해야 함)")

    qualifying_pids = list(eligible)

    # 각 후보 투수의 최신 prior_*/pitcher_throws 스냅샷 (v6 피처용)
    df["_date"] = df["game_id"].str[:8].astype(int)
    last_rows = (
        df[df["pitcher_id"].isin(qualifying_pids)]
        .sort_values("_date")
        .drop_duplicates(subset=["pitcher_id"], keep="last")
        .set_index("pitcher_id")
    )
    pvb_by_pitcher = df[df["pitcher_id"].isin(qualifying_pids)].groupby("pitcher_id")["pitcher_vs_batter_avg"].mean()

    def make_pitcher_dict(pid: str) -> dict:
        row = last_rows.loc[pid]
        d = {"pitcher_id": pid, "pitcher_vs_batter_avg": float(pvb_by_pitcher.loc[pid]),
             "pitcher_throws": str(row["pitcher_throws"]) if pd.notna(row["pitcher_throws"]) else "UNK"}
        for c in ["prior_bb_rate", "prior_so_rate", "prior_hr_rate",
                  "prior_wpa_std", "prior_n_apps", "prior_avg_pa_per_app"]:
            val = row[c]
            d[c] = float(val) if pd.notna(val) else float("nan")
        return d

    pitchers_pool = [make_pitcher_dict(pid) for pid in qualifying_pids]
    print(f"조건 통제 투수 풀: {len(pitchers_pool)}명")

    situation_pool = df[(df["inning"] >= 6) & (df["inning"] <= 8) &
                         (df["score_diff_attacker"].abs() <= 3)]

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
            pitcher_throws=pitcher["pitcher_throws"],
            prior_bb_rate=pitcher["prior_bb_rate"], prior_so_rate=pitcher["prior_so_rate"],
            prior_hr_rate=pitcher["prior_hr_rate"], prior_wpa_std=pitcher["prior_wpa_std"],
            prior_n_apps=pitcher["prior_n_apps"], prior_avg_pa_per_app=pitcher["prior_avg_pa_per_app"],
        )

    N_SITUATIONS = 15
    situations_sample = situation_pool.sample(n=N_SITUATIONS, random_state=RNG_SEED)
    situations = [_row_to_state(row) for _, row in situations_sample.iterrows()]

    meta_path = OUTPUT_DIR / "model_meta_v6.json"
    sim = MonteCarloSimulator(n_simulations=500, model_path=str(model_path),
                               meta_path=str(meta_path), rng_seed=RNG_SEED)

    print(f"\n{len(situations)}개 상황 × {len(pitchers_pool)}명 투수 시뮬레이션 시작 "
          f"(horizon_pa=5, horizon_innings=1.0 기본값, v6 모델)...")
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

    print(f"\nv6({MODEL_VARIANT}) 시뮬레이션: between={sim_between:.5f} within={sim_within:.5f} ratio={sim_ratio:.4f}")
    print(f"실제(조건 통제) baseline ratio={actual_ratio:.4f}")
    print(f"v5 이전 결과: sim_ratio=0.0488 (목표: 0.10 이상, 실제 0.200에 근접)")

    goal_met = sim_ratio >= 0.10
    improvement_factor = sim_ratio / 0.0488 if sim_ratio else float("nan")
    print(f"\n목표(0.10↑) 달성: {'예' if goal_met else '아니오'}  (v5 대비 {improvement_factor:.2f}배)")
    print(f"실제 대비: {sim_ratio/actual_ratio*100:.1f}%")

    result = {
        "model_variant": MODEL_VARIANT, "model_path": str(model_path),
        "n_pitchers_pool": len(pitchers_pool), "n_situations": N_SITUATIONS,
        "sim_between": sim_between, "sim_within": sim_within, "sim_ratio": sim_ratio,
        "actual_ratio_condition_controlled": actual_ratio,
        "v5_sim_ratio_reference": 0.0488, "improvement_factor_vs_v5": improvement_factor,
        "goal_0.10_met": goal_met, "pct_of_actual": sim_ratio / actual_ratio * 100,
        "elapsed_sec": elapsed,
    }
    with open(VALIDATION_DIR / "v6_final_ratio_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 완료! {VALIDATION_DIR / 'v6_final_ratio_report.json'}")
    return result


if __name__ == "__main__":
    if MODEL_VARIANT == "__SET_AFTER_REVIEW__":
        raise SystemExit("MODEL_VARIANT를 Task2/Task3 결과 검토 후 지정하세요.")
    model_path = finalize_model_files()
    run_fair_comparison(model_path)
