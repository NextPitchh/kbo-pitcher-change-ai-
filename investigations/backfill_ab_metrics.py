"""A_baseline/B_prior .cbm은 이미 저장돼 있으므로 재학습 없이 테스트셋 추론만
다시 돌려 per_class_f1/top3를 meta json에 채워넣는다."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, top_k_accuracy_score

DATA_PATH = Path("data_analysis/results/all_pa_enriched.parquet")
COMPARE_DIR = Path("catboost_output/compare_v6")

df = pd.read_parquet(DATA_PATH)
df = df[df["data_quality_flag"] == ""].copy()
TARGET = "pa_result"
EXPECTED_CLASSES = {"1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF"}
LEGACY_MERGE_MAP = {"IBB": "BB", "HBP": "BB", "ROE": "1B"}
unexpected = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
if unexpected:
    df[TARGET] = df[TARGET].replace(LEGACY_MERGE_MAP)
df = df[df[TARGET] != "UNK"].copy()

df["pitcher_vs_batter_avg"] = df["pitcher_vs_batter_avg"].fillna(0.250)
df["batter_recent_avg"] = df["batter_recent_avg"].fillna(0.250)
for col in ["inning", "score_diff_attacker", "out_count", "total_pitch_count", "inning_pitch_count"]:
    df[col] = df[col].fillna(0)
for col in ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]:
    df[col] = df[col].astype(str)

df["_date"] = df["game_id"].str[:8].astype(int)
df = df.sort_values("_date").reset_index(drop=True)
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
y_test_full = df[TARGET].iloc[val_end:]

for variant in ["A_baseline", "B_prior"]:
    meta_path = COMPARE_DIR / f"model_meta_{variant}.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    all_features = meta["all_features"]
    cat_features = meta["cat_features"]

    X_test = df[all_features].iloc[val_end:]
    test_pool = Pool(X_test, y_test_full, cat_features=cat_features)

    model = CatBoostClassifier()
    model.load_model(str(COMPARE_DIR / f"catboost_{variant}.cbm"))
    classes = model.classes_
    y_pred_proba = model.predict_proba(test_pool)
    y_pred_label = model.predict(test_pool).ravel()

    per_class_f1 = dict(zip(classes, f1_score(y_test_full, y_pred_label, average=None,
                                               labels=classes, zero_division=0)))
    try:
        top3 = float(top_k_accuracy_score(y_test_full, y_pred_proba, k=3, labels=classes))
    except Exception:
        top3 = None

    meta["per_class_f1"] = {k: float(v) for k, v in per_class_f1.items()}
    meta["test_top3"] = top3
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"{variant}: top3={top3}  BB_F1={per_class_f1.get('BB'):.4f}  SO_F1={per_class_f1.get('SO'):.4f}")
    print(f"  per_class_f1={ {k: round(v,4) for k,v in per_class_f1.items()} }")

print("✅ 완료")
