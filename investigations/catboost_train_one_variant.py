"""
catboost_feature_compare.py의 단일 변형만 학습 (재시도 복원력 확보용).
사용: python catboost_train_one_variant.py <variant>
  variant in {A_baseline, B_prior, C_hand, D_no_pvb}
로직·하이퍼파라미터는 원본과 완전히 동일. 이미 저장된 변형은 건너뛴다.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score

VARIANT = sys.argv[1] if len(sys.argv) > 1 else None
if VARIANT not in ("A_baseline", "B_prior", "C_hand", "D_no_pvb"):
    raise SystemExit(f"사용법: python catboost_train_one_variant.py <A_baseline|B_prior|C_hand|D_no_pvb>")

DATA_PATH = Path("data_analysis/results/all_pa_enriched.parquet")
OUTPUT_DIR = Path("catboost_output/compare_v6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_path = OUTPUT_DIR / f"catboost_{VARIANT}.cbm"
if model_path.exists():
    print(f"이미 완료됨: {model_path} — 건너뜀")
    sys.exit(0)

print("=" * 70, flush=True)
print(f"변형 {VARIANT} 단독 학습", flush=True)
print("=" * 70, flush=True)

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()

TARGET = "pa_result"
EXPECTED_CLASSES = {"1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF"}
LEGACY_MERGE_MAP = {"IBB": "BB", "HBP": "BB", "ROE": "1B"}
unexpected = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
if unexpected:
    df[TARGET] = df[TARGET].replace(LEGACY_MERGE_MAP)
df = df[df[TARGET] != "UNK"].copy()

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

df["pitcher_throws"] = df["pitcher_throws"].fillna("UNK").astype(str)
_pt_missing = df["pitcher_throws"] == "UNK"
_bh_missing = df["batter_hit_type"].isna()
df["batter_hit_type"] = df["batter_hit_type"].astype(str)
df["handedness_matchup"] = np.where(
    _pt_missing | _bh_missing, "UNK",
    df["pitcher_throws"] + "_" + df["batter_hit_type"]
)

FEATURE_SETS = {
    "A_baseline": {"num": BASE_NUM, "cat": BASE_CAT},
    "B_prior": {"num": BASE_NUM + PRIOR_NUM, "cat": BASE_CAT},
    "C_hand": {"num": BASE_NUM + PRIOR_NUM,
               "cat": BASE_CAT + ["pitcher_throws", "handedness_matchup"]},
    "D_no_pvb": {"num": [c for c in BASE_NUM if c != "pitcher_vs_batter_avg"] + PRIOR_NUM,
                 "cat": BASE_CAT + ["pitcher_throws", "handedness_matchup"]},
}
spec = FEATURE_SETS[VARIANT]

df["_date"] = df["game_id"].str[:8].astype(int)
df = df.sort_values("_date").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
print(f"Train : {train_end:,}  Val : {val_end - train_end:,}  Test : {n - val_end:,}", flush=True)

y_train_full = df[TARGET].iloc[:train_end]
y_val_full = df[TARGET].iloc[train_end:val_end]
y_test_full = df[TARGET].iloc[val_end:]

counts = y_train_full.value_counts()
total = len(y_train_full)
class_weights = {cls: float(np.sqrt(total / counts[cls])) for cls in sorted(y_train_full.unique())}

snapshot_file = str(OUTPUT_DIR / f"snapshot_{VARIANT}.bin")
CB_PARAMS = dict(
    iterations=2000, depth=7, learning_rate=0.04,
    loss_function="MultiClass", eval_metric="MultiClass",
    l2_leaf_reg=4.0, random_strength=1.0,
    bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
    early_stopping_rounds=100, use_best_model=True,
    random_seed=42, thread_count=-1, verbose=100,
    save_snapshot=True, snapshot_file=snapshot_file, snapshot_interval=120,
)

all_features = spec["num"] + spec["cat"]
print(f"num={spec['num']}\ncat={spec['cat']}", flush=True)

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
print(f"학습 완료 ({elapsed:.0f}s), best_iteration={model.best_iteration_}", flush=True)

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

print(f"Accuracy={acc:.4f}  Log-loss={ll:.4f}  Macro-F1={macro_f1:.4f}  Top3={top3}", flush=True)
print(f"BB F1={per_class_f1.get('BB', float('nan')):.4f}  SO F1={per_class_f1.get('SO', float('nan')):.4f}", flush=True)
print(top15.to_string(), flush=True)

model.save_model(str(model_path))

variant_meta = {
    "version": f"v6_{VARIANT}", "data_path": str(DATA_PATH), "classes": list(classes),
    "cat_features": spec["cat"], "num_features": spec["num"], "all_features": all_features,
    "best_iteration": int(model.best_iteration_),
    "test_accuracy": float(acc), "test_log_loss": float(ll), "test_macro_f1": float(macro_f1),
    "test_top3": top3, "per_class_f1": {k: float(v) for k, v in per_class_f1.items()},
    "feature_importance_top15": top15.to_dict(),
    "merge_map": {}, "merge_applied": False, "removed": ["UNK"],
    "note": f"catboost_train_one_variant.py variant={VARIANT}",
    "wpa_class_mapping": {
        "1B": {"out_delta": 0, "batter_base": 1}, "2B": {"out_delta": 0, "batter_base": 2},
        "3B": {"out_delta": 0, "batter_base": 3}, "HR": {"out_delta": 0, "batter_base": 4},
        "BB": {"out_delta": 0, "batter_base": 1, "force_only": True},
        "SO": {"out_delta": 1, "batter_base": None}, "OUT": {"out_delta": 1, "batter_base": None},
        "GDP": {"out_delta": 2, "batter_base": None, "note": "병살"},
        "SF": {"out_delta": 1, "batter_base": None, "note": "희생플라이"},
    },
}
with open(OUTPUT_DIR / f"model_meta_{VARIANT}.json", "w", encoding="utf-8") as f:
    json.dump(variant_meta, f, ensure_ascii=False, indent=2)
print(f"✅ 저장: {model_path}", flush=True)
