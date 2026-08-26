# KBO 투수 교체 의사결정 AI

KBO(한국프로야구) 데이터를 기반으로 "지금 투수를 계속 쓸지, 누구로 교체할지"를
정량적으로 판단하는 파이프라인입니다. 타석 결과를 예측하는 분류 모델 →
그 모델로 경기를 몬테카를로 시뮬레이션해 승리확률기여도(WPA)를 추정 →
그 추정치를 보상으로 강화학습 에이전트가 교체 여부/대상을 추천하는 3단계로
구성됩니다.

## 파이프라인

```
catboost_train.py  →  monte_carlo_simulate.py  →  dqn_agent.py  →  dqn_recommend.py
 (타석 결과 예측)        (WPA 분포 추정)              (교체 정책 학습)     (실전 추천)
```

1. **`catboost_train.py`** — 게임 상태(이닝/점수차/주자/투수·타자 정보)로부터
   타석 결과(`1B/2B/3B/HR/BB/SO/OUT/GDP/SF`, 9클래스)를 예측하는 CatBoost 분류기를 학습합니다.
2. **`monte_carlo_simulate.py`** — 학습된 분류기로 임의의 게임 상태에서 경기를
   N회 병렬로 시뮬레이션해, 특정 투수를 계속 기용했을 때의 `E[WPA]`/`Var[WPA]`/95% 신뢰구간을 추정합니다.
   단독 실행 시 데모로도 동작합니다.
3. **`dqn_agent.py`** / **`dqn_recommend.py`** — "현재 투수 유지" vs "불펜 후보 3명"
   중 하나를 고르는 DQN 에이전트를 위험회피도(λ) 여러 값에 대해 각각 학습하고,
   실전 게임 상태에 대한 추천을 생성합니다.

## 설치

스크립트별로 필요한 패키지가 다릅니다(파일 상단 docstring 참고):

```bash
pip install catboost scikit-learn pandas pyarrow scipy matplotlib   # 학습/시뮬레이션/진단
pip install torch                                                    # DQN
```

## 실행

```bash
python catboost_train.py           # → catboost_output/ (모델 + 메타 + 피처 중요도)
python monte_carlo_simulate.py     # 시뮬레이터 단독 데모 (catboost_output/ 필요)
python dqn_agent.py                # λ별 에이전트 학습 → dqn_output/
python dqn_recommend.py            # 학습된 에이전트로 추천 테이블 출력
```

`catboost_output/`, `dqn_output/`, `catboost_info/`는 로컬에서 생성되는 학습
산출물이며 `.gitignore`에 포함되어 커밋되지 않습니다 — 위 순서대로 직접
재현해야 합니다.

## 디렉토리 구조

```
├── catboost_train.py            # 파이프라인 1단계
├── monte_carlo_simulate.py      # 파이프라인 2단계 (GameState, MonteCarloSimulator)
├── dqn_agent.py                 # 파이프라인 3단계 (학습)
├── dqn_recommend.py             # 파이프라인 3단계 (추론)
│
├── tests/
│   └── test_state_transition.py # apply_pa_result() 상태 전이 단위 테스트
│
├── investigations/              # 일회성 진단/실험 스크립트 (아래 "진행 상황" 참고)
│   ├── validate_horizon.py
│   ├── validate_pitcher_signal.py
│   ├── validate_measurement.py
│   ├── validate_confounds.py
│   ├── validate_wpa_std_predictive.py
│   ├── validate_v6_differentiation.py
│   ├── catboost_feature_compare*.py / catboost_train_one_variant.py
│   ├── finalize_v6_model.py / backfill_ab_metrics.py
│   └── task2_skill_regression.py / task3_utility_prototype.py / task4_manager_decision_match.py
│
├── validation_output/           # 위 스크립트들이 남긴 JSON 리포트 · 그래프
│
├── data_analysis/
│   ├── methods/we_re_lookup.py  # RE/WE(득점·승리기댓값) 조회 테이블 — 공용 정답 소스
│   └── results/*.parquet        # 데이터셋 (아래 "데이터" 참고)
│
├── CLAUDE.md                    # 저장소 컨벤션 (AI 어시스턴트용 작업 지침)
└── HANDOFF_FULL.md              # 데이터 명세 — parquet를 건드리기 전 필독
```

`investigations/`, `tests/`의 스크립트는 모두 **레포 루트에서** 실행해야
합니다(상대경로로 `data_analysis/`, `catboost_output/`, `validation_output/`를
참조하므로):

```bash
python investigations/validate_horizon.py
python tests/test_state_transition.py
```

## 데이터

`data_analysis/results/all_pa_enriched.parquet`(558,064행, 2016~2025 전 구단)가
메인 데이터셋입니다. 컬럼 의미·부호 규칙·정책(연장이닝 처리, terminal PA 보정
등)은 반드시 **`HANDOFF_FULL.md`**를 먼저 읽고 확인하세요 — 이 문서의 경고대로,
컨벤션을 데이터에서 역추론하다가 과거에 여러 번 결론이 뒤집힌 이력이 있습니다.
`hsk_pa_with_wpa.parquet`(구버전, 153경기)는 롤백 지점으로만 보존됩니다.

## 진행 상황 — 투수 교체 신호 진단

핵심 질문은 "이 시뮬레이터가 투수를 얼마나 잘 구별하는가"였습니다.
`investigations/`의 스크립트들이 시간순으로 이 질문을 파고든 기록이며,
현재까지의 결론은 다음과 같습니다.

- **측정 방법 자체의 문제**: horizon 제한, 표본 편향은 원인이 아니었고, 실제
  ratio(0.79)와 시뮬레이션 ratio(0.07)의 격차 대부분은 baseline이 선발·불펜을
  섞어 비교한 탓(교란 요인)이었습니다. 조건을 공정하게 맞추면 격차는
  0.20 vs 0.05로 좁혀지지만(→ 4배), 완전히 사라지지는 않습니다.
- **CatBoost v6 피처 확장 시도**(`prior_bb_rate`/`prior_so_rate`/`prior_hr_rate`/
  `pitcher_throws`/좌우 매치업 추가)는 실패했습니다 — 로그손실 등 집계 지표는
  개선됐지만, 목표였던 시뮬레이션 ratio는 오히려 후퇴(0.049→0.039)했습니다.
  **현재 운영 모델은 v5 그대로입니다** (`catboost_output/catboost_pa_classifier.cbm`).
- **"E는 시뮬레이션, σ는 과거 기록(prior_wpa_std)으로 분리"** 제안을 검증한
  결과: **σ 쪽은 실제로 유효한 신호**(미래 변동성 예측 상관 r=0.37~0.59)였지만,
  **E 쪽 보정 회귀는 사실상 무효**(R²≈0, 실제 순위와 상관 없음)였고, 실제
  감독 결정과의 일치율도 변동성 페널티를 넣을수록 오히려 나빠졌습니다.

즉 현재 시뮬레이션 기반 접근은 "5타석 구간에서 투수 개인차를 구별하는" 과제
자체에 구조적 한계가 있다는 것이 여러 각도(측정법 교정, 피처 확장, 역할 분리)
로 반복 확인된 상태입니다. 각 조사의 상세 수치와 방법론은 `investigations/`의
스크립트 docstring과 `validation_output/*.json`에 남아 있습니다.

## 참고

- **`CLAUDE.md`** — 이 저장소에서 작업할 때 지켜야 할 컨벤션(클래스 정의,
  시간순 분할, 부호 규칙 등). Claude Code가 자동으로 읽습니다.
- **`HANDOFF_FULL.md`** — 데이터 명세 원문. parquet 컬럼을 다루기 전 필독.
