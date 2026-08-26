"""
KBO WPA Pipeline — CatBoost 다중 분류 (v5, 전 구단 10년 확장 데이터)

데이터: data_analysis/results/all_pa_enriched.parquet
        (558,064행 · 7,197경기 · 2016~2025 · 전 구단). 컨벤션은 HANDOFF_FULL.md 참조.

클래스 9개: 1B / 2B / 3B / HR / BB / SO / OUT / GDP / SF
  - UNK 제거 (모호한 이벤트, 학습 노이즈)
  - IBB→BB, HBP→BB, ROE→1B 는 데이터에 이미 병합되어 제공됨
    (미병합 라벨이 남아있으면 안전장치가 감지해 하위 호환 병합)
  - 3B / GDP / SF 유지 (WPA 계산 시 상태 전이 달라서 합치면 안 됨)

가중치: sqrt 역빈도
분할: game_id 시간순 70 / 15 / 15

실행:
    python catboost_train.py

요구사항:
    pip install catboost scikit-learn pandas pyarrow
"""


# 다음 타석에 뭐가 나올지 맞히는 AI를 훈련시키는 곳

import json
import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)

# ──────────────────────────────────────────────
# 0. 경로
# ──────────────────────────────────────────────
DATA_PATH  = Path("data_analysis/results/all_pa_enriched.parquet")
OUTPUT_DIR = Path("catboost_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_parquet(DATA_PATH)
print(f"원본: {len(df):,}행")

df = df[df["data_quality_flag"] == ""].copy()
print(f"품질 플래그 제외 후: {len(df):,}행")

# ──────────────────────────────────────────────
# 2. 클래스 정리
#    all_pa_enriched.parquet은 이미 9클래스로 병합되어 제공됨
#    (IBB→BB, HBP→BB, ROE→1B). 원본 세분 라벨은 pa_result_raw 컬럼.
#    안전장치: 혹시 미병합 라벨이 남아있으면 경고 후 하위 호환 병합.
#    3B / GDP / SF 는 WPA 계산상 유지
# ──────────────────────────────────────────────
#타석 결과를 딱 9가지로만 정리해. "고의사구"랑 "몸에 맞는 볼"은 "볼넷"이랑 사실상 같은 거라 합치고, 
# "실책으로 출루"는 "1루타"랑 같은 취급으로 합쳐. 
# 재밌는 부분: 이번 새 데이터는 이미 합쳐진 상태로 제공되는데, 코드는 "혹시 안 합쳐진 게 남아있으면 자동으로 감지해서 알아서 합쳐준다"는 안전장치까지 만들어놨어
TARGET = "pa_result"
EXPECTED_CLASSES = {"1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF"}
LEGACY_MERGE_MAP = {"IBB": "BB", "HBP": "BB", "ROE": "1B"}

merge_applied = False
unexpected = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
if unexpected:
    print(f"⚠️  예상 외 라벨 발견: {sorted(unexpected)} — 하위 호환 병합 수행")
    df[TARGET] = df[TARGET].replace(LEGACY_MERGE_MAP)
    merge_applied = True
    still_unexpected = set(df[TARGET].unique()) - EXPECTED_CLASSES - {"UNK"}
    if still_unexpected:
        raise ValueError(f"병합 후에도 미확인 라벨 존재: {sorted(still_unexpected)}")
else:
    print("✅ pa_result 9클래스 + UNK 검증 통과 (이미 병합된 데이터)")

unk_cnt = (df[TARGET] == "UNK").sum()
df = df[df[TARGET] != "UNK"].copy()
print(f"UNK 제거: {unk_cnt}건")

print("\n클래스 분포:")
dist = df[TARGET].value_counts()
for cls, n in dist.items():
    bar = "█" * int(n / len(df) * 50)
    print(f"  {cls:4s}: {n:5,} ({n/len(df)*100:5.1f}%)  {bar}")

# ──────────────────────────────────────────────
# 3. 피처
# ──────────────────────────────────────────────
# AI가 판단할 때 참고할 정보 12가지를 정하는 거야. "몇 회인지, 초/말인지, 몇 아웃인지, 점수차, 주자 상황, 투수/타자가 누군지, 상대전적, 최근 타율, 피로도" 등.
CAT_FEATURES = ["half", "base_state", "batter_hit_type", "pitcher_id", "batter_id"]
NUM_FEATURES = [
    "inning",
    "score_diff_attacker",
    "out_count",
    "total_pitch_count",
    "inning_pitch_count",
    "pitcher_vs_batter_avg",
    "batter_recent_avg",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

df["pitcher_vs_batter_avg"] = df["pitcher_vs_batter_avg"].fillna(0.250)
df["batter_recent_avg"]     = df["batter_recent_avg"].fillna(0.250)
for col in NUM_FEATURES:
    df[col] = df[col].fillna(0)
for col in CAT_FEATURES:
    # pandas의 Arrow 문자열 dtype에서는 astype(str)만으로 NaN이
    # 문자열 "nan"으로 안 바뀌고 결측으로 남는 경우가 있어 fillna를 먼저 적용
    df[col] = df[col].fillna("UNK").astype(str)

classes_all = sorted(df[TARGET].unique())
print(f"\n최종 클래스 {len(classes_all)}개: {classes_all}")

# ──────────────────────────────────────────────
# 4. 시간순 분할 (game_id 앞 8자리 = 날짜)
# ──────────────────────────────────────────────
# 날짜 순서대로 정렬해서, 앞쪽 70%로 공부시키고, 그다음 15%로 "언제 그만 공부할지" 확인하고, 제일 마지막 15%(제일 최근 경기들)로 진짜 실력 시험을 봐. 
# 왜 무작위로 안 섞고 날짜순으로 나누냐면: 실전에서는 "과거 데이터로 배워서 미래 경기를 맞혀야" 하니까, 연습도 똑같은 방식으로 해야 정직한 실력 측정이 되기 때문이야.
print("\n" + "=" * 60)
print("4. 시간순 분할 (70 / 15 / 15)")
print("=" * 60)

df["_date"] = df["game_id"].str[:8].astype(int)
df = df.sort_values("_date").reset_index(drop=True)

n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = df[ALL_FEATURES].iloc[:train_end]
y_train = df[TARGET].iloc[:train_end]
X_val   = df[ALL_FEATURES].iloc[train_end:val_end]
y_val   = df[TARGET].iloc[train_end:val_end]
X_test  = df[ALL_FEATURES].iloc[val_end:]
y_test  = df[TARGET].iloc[val_end:]

print(f"Train : {len(X_train):,}  Val : {len(X_val):,}  Test : {len(X_test):,}")

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
val_pool   = Pool(X_val,   y_val,   cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

# ──────────────────────────────────────────────
# 5. 클래스 가중치 — sqrt 역빈도
#    v2 순수역빈도(너무 강함) vs v1 log(너무 약함) 사이
# ──────────────────────────────────────────────
#"아웃"은 흔해서 안 가르쳐도 잘 맞히는데, "3루타"는 워낙 드물어서 AI가 신경도 안 쓰려고 해. 
#그래서 드문 결과일수록 "이거 맞히면 점수 더 줄게"라고 가산점을 주는 거야. 너무 세게 주면 반대로 문제 생기니까(제곱근 처리로) 적당한 세기로 조절해놨어.
counts  = y_train.value_counts()
total   = len(y_train)
classes_sorted = sorted(y_train.unique())

class_weights = {cls: np.sqrt(total / counts[cls]) for cls in classes_sorted}

print("\n클래스 가중치 (sqrt 역빈도):")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1]):
    print(f"  {cls:4s}: {w:5.2f}x  (train n={counts[cls]:,})")

# ──────────────────────────────────────────────
# 6. 학습
# ──────────────────────────────────────────────
# iterations=2000은 "최대 2000번 반복해서 점점 더 똑똑하게 만들어라", depth=7은 "판단 기준을 얼마나 복잡하게 쪼갤지"(나무의 깊이), 
# early_stopping_rounds=100(6단계 위쪽에 있음)은 "100번 연속으로 안 좋아지면 그만 멈춰라"는 뜻이야 — 과적합 막는 안전장치야.
print("\n" + "=" * 60)
print("6. CatBoost 학습")
print("=" * 60)

model = CatBoostClassifier(
    iterations=2000,
    depth=7,
    learning_rate=0.04,

    loss_function="MultiClass",
    eval_metric="MultiClass",

    class_weights=class_weights,

    l2_leaf_reg=4.0,
    random_strength=1.0,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    rsm=0.8,

    early_stopping_rounds=100,
    use_best_model=True,

    random_seed=42,
    thread_count=-1,
    verbose=200,
)

model.fit(train_pool, eval_set=val_pool)
print(f"\n최적 iteration: {model.best_iteration_}")

# ──────────────────────────────────────────────
# 7. 평가
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 평가")
print("=" * 60)

classes      = model.classes_
y_pred_proba = model.predict_proba(test_pool)
y_pred_label = model.predict(test_pool).ravel()

acc      = accuracy_score(y_test, y_pred_label) #정확도
ll       = log_loss(y_test, y_pred_proba, labels=classes)  # 확률 예측이 얼마나 자신있게 맞았는지
macro_f1 = f1_score(y_test, y_pred_label, average="macro",
                    labels=classes, zero_division=0)  # 소수 결과까지 골고루 잘 맞혔는지
# 정확도만 보면 안 되는 이유: "아웃"만 계속 찍어도 정확도는 꽤 높게 나오거든. Macro F1은 "9가지 결과를 골고루 잘 맞혔는지"를 보여줘서 더 정직한 지표야.

print(f"Accuracy : {acc:.4f}")
print(f"Log-loss : {ll:.4f}   (낮을수록 좋음, 목표 1.35↓)")
print(f"Macro F1 : {macro_f1:.4f}   (소수 클래스 균형 핵심 지표)")

try:
    top3 = top_k_accuracy_score(y_test, y_pred_proba, k=3, labels=classes)
    print(f"Top-3 Acc: {top3:.4f}")
except Exception:
    pass

print("\n분류 리포트:")
print(classification_report(y_test, y_pred_label,
                             target_names=classes, zero_division=0))

fi = (pd.Series(model.get_feature_importance(), index=ALL_FEATURES)
        .sort_values(ascending=False))
print("피처 중요도:")
print(fi.to_string())

# ──────────────────────────────────────────────
# 8. 저장
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. 저장")
print("=" * 60)

model.save_model(str(OUTPUT_DIR / "catboost_pa_classifier.cbm"))  # AI 모델 자체 저장

meta = {  # 클래스 목록, 성적, 피처 목록 등 설명서 저장( 다음 파일(Monte Carlo)이 이 AI를 불러다 쓸 때, "이 AI가 뭘 입력받고 뭘 출력하는지"를 이 설명서를 보고 알아내거든.)
    "version": "v5",
    "data_path": str(DATA_PATH),
    "classes": list(classes),
    "cat_features": CAT_FEATURES,
    "num_features": NUM_FEATURES,
    "all_features": ALL_FEATURES,
    "best_iteration": int(model.best_iteration_),
    "test_accuracy":  float(acc),
    "test_log_loss":  float(ll),
    "test_macro_f1":  float(macro_f1),
    "merge_map": LEGACY_MERGE_MAP if merge_applied else {},
    "merge_applied": merge_applied,
    "removed": ["UNK"],
    "note": (
        "3B/GDP/SF 유지 (WPA 상태 전이 다름). "
        "all_pa_enriched.parquet은 IBB→BB, HBP→BB, ROE→1B 병합 완료 상태로 제공됨. "
        "UNK 제거. sqrt 역빈도 가중치."
    ),
    # Monte Carlo / DQN 연동용 WPA 처리 정보
    "wpa_class_mapping": {
        "1B":  {"out_delta": 0, "batter_base": 1},
        "2B":  {"out_delta": 0, "batter_base": 2},
        "3B":  {"out_delta": 0, "batter_base": 3},
        "HR":  {"out_delta": 0, "batter_base": 4},
        "BB":  {"out_delta": 0, "batter_base": 1, "force_only": True},
        "SO":  {"out_delta": 1, "batter_base": None},
        "OUT": {"out_delta": 1, "batter_base": None},
        "GDP": {"out_delta": 2, "batter_base": None,
                "note": "병살: 아웃 +2, 1루 주자 제거"},
        "SF":  {"out_delta": 1, "batter_base": None,
                "note": "희생플라이: 아웃 +1, 3루 주자 홈"},
    },
}

with open(OUTPUT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

fi.reset_index().rename(
    columns={"index": "feature", 0: "importance"}
).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

print(f"모델  : {OUTPUT_DIR}/catboost_pa_classifier.cbm")
print(f"메타  : {OUTPUT_DIR}/model_meta.json")
print(f"중요도: {OUTPUT_DIR}/feature_importance.csv")

# ──────────────────────────────────────────────
# 9. 추론 예시
# ──────────────────────────────────────────────
# 저장 다 하고 나서, 가상의 상황(7회말, 1루, 1아웃, -1점) 하나를 던져줘서 "이 AI가 확률을 잘 뱉는지" 눈으로 확인해보는 부분이야.
print("\n" + "=" * 60)
print("9. 추론 예시")
print("=" * 60)

sample = pd.DataFrame([{
    "inning": 7, "score_diff_attacker": -1, "out_count": 1,
    "total_pitch_count": 65, "inning_pitch_count": 12,
    "pitcher_vs_batter_avg": 0.280, "batter_recent_avg": 0.310,
    "half": "bot", "base_state": "1",
    "batter_hit_type": "R", "pitcher_id": "99999", "batter_id": "88888",
}])
for col in CAT_FEATURES:
    sample[col] = sample[col].astype(str)

proba     = model.predict_proba(sample)[0]
prob_dict = dict(zip(classes, proba))

print("7회말, 1루, 1아웃, 공격팀 -1점")
for cls, p in sorted(prob_dict.items(), key=lambda x: -x[1]):
    bar = "█" * int(p * 50)
    print(f"  {cls:4s}: {p:.4f}  {bar}")

print("\n✅ 완료!")
print("   다음: python monte_carlo_simulate.py")