"""
KBO WPA Pipeline — CatBoost v6 피처 확장 비교 (Task 1 + Task 2)

배경: 투수 간 신호 격차 진단(validate_pitcher_signal.py, validate_measurement.py,
validate_confounds.py)에서 CatBoost가 학습한 확률 차이가 실제 투수 간 신호에
비해 구조적으로 작다는 결론에 도달했다. 현재 투수 관련 피처는 pitcher_id
(범주형, target encoding shrinkage로 개별 효과가 억제됨) 하나뿐이며, 시점
안전(leakage-free)한 prior_* 연속형 피처들(prior_bb_rate 등, BB의 SNR=0.26로
가장 강한 투수 신호)이 데이터에 이미 존재하는데도 사용되지 않고 있었다.

Task 0(별도 조사)에서 pitcher_vs_batter_avg는 이름과 달리 pitcher_id와
무관하게 (batter_id, game_id) 단위로 고정된 값(99.4%)임을 확인 — 개별
투수 매칭업 정보로 기능하지 않는 오염된 피처. Variant D는 이를 제거했을 때
오히려 개선되는지 검증한다.

4개 피처셋을 동일 분할·동일 하이퍼파라미터로 비교:
  A. baseline   : v5 재현 (12피처)
  B. +prior     : A + prior_bb_rate/so_rate/hr_rate/wpa_std/n_apps/avg_pa_per_app
  C. +hand      : B + pitcher_throws + handedness_matchup(파생)
  D. -pvb       : C에서 pitcher_vs_batter_avg 제거

prior_*는 NaN을 유지한다 (CatBoost가 자체 결측 처리; 0이나 리그평균으로
채우면 "신인/첫등판" 신호가 사라짐 — 절대 금지 사항).

분할 방식(시간순 70/15/15)과 클래스 정의는 v5와 동일하게 유지 — 변경 금지.

실행: python catboost_feature_compare.py
요구사항: pip install catboost scikit-learn pandas numpy pyarrow
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, top_k_accuracy_score,
)

DATA_PATH = Path("data_analysis/results/all_pa_enriched.parquet")
OUTPUT_DIR = Path("catboost_output/compare_v6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CatBoost v6 피처 확장 비교 — A(baseline) / B(+prior) / C(+hand) / D(-pvb)")
print("=" * 70)

# ──────────────────────────────────────────────
# 1. 데이터 로드 · 클래스 정리 (v5와 동일)
# ──────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"원본: {len(df):,}행")
df = df[df["data_quality_flag"] == ""].copy()
print(f"품질 플래그 제외 후: {len(df):,}행")

TARGET = "pa_result"
EXPECTED_CLASSES = {"1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF"}
LEGACY_MERGE_MAP = {"IBB": "BB", "HBP": "BB", "ROE": "1B"}

unexpected = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
if unexpected:
    df[TARGET] = df[TARGET].replace(LEGACY_MERGE_MAP)
    still = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
    if still:
        raise ValueError(f"병합 후에도 미확인 라벨 존재: {sorted(still)}")

unk_cnt = (df[TARGET] == "UNK").sum()
df = df[df[TARGET] != "UNK"].copy()
print(f"UNK 제거: {unk_cnt}건")

# ──────────────────────────────────────────────
# 2. 피처 정의
# ──────────────────────────────────────────────
BASE_CAT = ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]
BASE_NUM = [
    "inning", "score_diff_attacker", "out_count",
    "total_pitch_count", "inning_pitch_count",
    "pitcher_vs_batter_avg", "batter_recent_avg",
]
PRIOR_NUM = [
    "prior_bb_rate", "prior_so_rate", "prior_hr_rate",
    "prior_wpa_std", "prior_n_apps", "prior_avg_pa_per_app",
]

df["pitcher_vs_batter_avg"] = df["pitcher_vs_batter_avg"].fillna(0.250)
df["batter_recent_avg"] = df["batter_recent_avg"].fillna(0.250)
for col in BASE_NUM:
    if col not in ("pitcher_vs_batter_avg", "batter_recent_avg"):
        df[col] = df[col].fillna(0)
for col in BASE_CAT:
    df[col] = df[col].astype(str)
# prior_* : NaN 유지 (절대 채우지 않음) — CatBoost 네이티브 결측 처리에 위임

# pitcher_throws: 결측 → "UNK" (범주형이라 명시적 문자열 필요, prior_*와는 다른 취급)
df["pitcher_throws"] = df["pitcher_throws"].fillna("UNK").astype(str)

# handedness_matchup 파생: 둘 중 하나라도 원래 결측이면 "UNK"
_pt_missing = df["pitcher_throws"] == "UNK"
_bh_missing = df["batter_hit_type"].isna()
df["batter_hit_type"] = df["batter_hit_type"].astype(str)
df["handedness_matchup"] = np.where(
    _pt_missing | _bh_missing, "UNK",
    df["pitcher_throws"] + "_" + df["batter_hit_type"]
)

print(f"\nhandedness_matchup 분포:\n{df['handedness_matchup'].value_counts()}")

FEATURE_SETS = {
    "A_baseline": {"num": BASE_NUM, "cat": BASE_CAT},
    "B_prior": {"num": BASE_NUM + PRIOR_NUM, "cat": BASE_CAT},
    "C_hand": {"num": BASE_NUM + PRIOR_NUM,
               "cat": BASE_CAT + ["pitcher_throws", "handedness_matchup"]},
    "D_no_pvb": {"num": [c for c in BASE_NUM if c != "pitcher_vs_batter_avg"] + PRIOR_NUM,
                 "cat": BASE_CAT + ["pitcher_throws", "handedness_matchup"]},
}

# ──────────────────────────────────────────────
# 3. 시간순 분할 (v5와 완전히 동일한 방식 — 변경 금지)
# ──────────────────────────────────────────────
df["_date"] = df["game_id"].str[:8].astype(int)
df = df.sort_values("_date").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
print(f"\nTrain : {train_end:,}  Val : {val_end - train_end:,}  Test : {n - val_end:,}")

y_train_full = df[TARGET].iloc[:train_end]
y_val_full = df[TARGET].iloc[train_end:val_end]
y_test_full = df[TARGET].iloc[val_end:]

counts = y_train_full.value_counts()
total = len(y_train_full)
class_weights = {cls: float(np.sqrt(total / counts[cls])) for cls in sorted(y_train_full.unique())}
print("\n클래스 가중치 (sqrt 역빈도, 전체 변형 공통):")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1]):
    print(f"  {cls:4s}: {w:5.2f}x")

CB_PARAMS = dict(
    iterations=2000, depth=7, learning_rate=0.04,
    loss_function="MultiClass", eval_metric="MultiClass",
    l2_leaf_reg=4.0, random_strength=1.0,
    bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
    early_stopping_rounds=100, use_best_model=True,
    random_seed=42, thread_count=-1, verbose=200,
)

# ──────────────────────────────────────────────
# 4. 변형별 학습 · 평가
# ──────────────────────────────────────────────
results = {}
BB_SO_TRACK = ["BB", "SO"]

for variant, spec in FEATURE_SETS.items():
    print("\n" + "=" * 70)
    print(f"변형 {variant}: num={spec['num']}\n         cat={spec['cat']}")
    print("=" * 70)
    all_features = spec["num"] + spec["cat"]

    X_train = df[all_features].iloc[:train_end]
    X_val = df[all_features].iloc[train_end:val_end]
    X_test = df[all_features].iloc[val_end:]

    train_pool = Pool(X_train, y_train_full, cat_features=spec["cat"])
    val_pool = Pool(X_val, y_val_full, cat_features=spec["cat"])
    test_pool = Pool(X_test, y_test_full, cat_features=spec["cat"])

    model = CatBoostClassifier(class_weights=class_weights, **CB_PARAMS)
    t0 = time.time()
    model.fit(train_pool, eval_set=val_pool)
    elapsed = time.time() - t0
    print(f"학습 완료 ({elapsed:.0f}s), best_iteration={model.best_iteration_}")

    classes = model.classes_
    y_pred_proba = model.predict_proba(test_pool)
    y_pred_label = model.predict(test_pool).ravel()

    acc = accuracy_score(y_test_full, y_pred_label)
    ll = log_loss(y_test_full, y_pred_proba, labels=classes)
    macro_f1 = f1_score(y_test_full, y_pred_label, average="macro", labels=classes, zero_division=0)
    try:
        top3 = float(top_k_accuracy_score(y_test_full, y_pred_proba, k=3, labels=classes))
    except Exception:
        top3 = None

    per_class_f1 = dict(zip(classes, f1_score(y_test_full, y_pred_label, average=None,
                                               labels=classes, zero_division=0)))

    fi = pd.Series(model.get_feature_importance(), index=all_features).sort_values(ascending=False)
    top15 = fi.head(15)

    print(f"Accuracy={acc:.4f}  Log-loss={ll:.4f}  Macro-F1={macro_f1:.4f}  Top3={top3}")
    print(f"BB F1={per_class_f1.get('BB', float('nan')):.4f}  SO F1={per_class_f1.get('SO', float('nan')):.4f}")
    print("피처 중요도 top15:")
    print(top15.to_string())

    model_path = OUTPUT_DIR / f"catboost_{variant}.cbm"
    model.save_model(str(model_path))

    results[variant] = {
        "num_features": spec["num"], "cat_features": spec["cat"],
        "all_features": all_features, "classes": list(classes),
        "best_iteration": int(model.best_iteration_), "train_elapsed_sec": elapsed,
        "test_accuracy": float(acc), "test_log_loss": float(ll),
        "test_macro_f1": float(macro_f1), "test_top3": top3,
        "per_class_f1": {k: float(v) for k, v in per_class_f1.items()},
        "feature_importance_top15": top15.to_dict(),
        "model_path": str(model_path),
    }

    # 변형별 meta도 개별 저장 (Task3/Task4용 — MonteCarloSimulator가 이 meta로 로드)
    variant_meta = {
        "version": f"v6_{variant}",
        "data_path": str(DATA_PATH),
        "classes": list(classes),
        "cat_features": spec["cat"],
        "num_features": spec["num"],
        "all_features": all_features,
        "best_iteration": int(model.best_iteration_),
        "test_accuracy": float(acc), "test_log_loss": float(ll), "test_macro_f1": float(macro_f1),
        "merge_map": {}, "merge_applied": False, "removed": ["UNK"],
        "note": f"catboost_feature_compare.py variant={variant}",
        "wpa_class_mapping": {
            "1B": {"out_delta": 0, "batter_base": 1}, "2B": {"out_delta": 0, "batter_base": 2},
            "3B": {"out_delta": 0, "batter_base": 3}, "HR": {"out_delta": 0, "batter_base": 4},
            "BB": {"out_delta": 0, "batter_base": 1, "force_only": True},
            "SO": {"out_delta": 1, "batter_base": None}, "OUT": {"out_delta": 1, "batter_base": None},
            "GDP": {"out_delta": 2, "batter_base": None, "note": "병살"},
            "SF": {"out_delta": 1, "batter_base": None, "note": "희생플라이"},
        },
    }
    with open(OUTPUT_DIR / f"model_meta_{variant}.json", "w", encoding="utf-8") as f:
        json.dump(variant_meta, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 5. 비교표
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("비교표")
print("=" * 70)
cmp_rows = []
for variant, r in results.items():
    cmp_rows.append({
        "variant": variant, "n_features": len(r["all_features"]),
        "accuracy": r["test_accuracy"], "log_loss": r["test_log_loss"],
        "macro_f1": r["test_macro_f1"], "top3": r["test_top3"],
        "BB_F1": r["per_class_f1"].get("BB"), "SO_F1": r["per_class_f1"].get("SO"),
    })
cmp_df = pd.DataFrame(cmp_rows)
print(cmp_df.to_string(index=False))

with open(OUTPUT_DIR / "comparison_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

cmp_df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

print(f"\n✅ 완료! {OUTPUT_DIR}/comparison_report.json, comparison_table.csv")
