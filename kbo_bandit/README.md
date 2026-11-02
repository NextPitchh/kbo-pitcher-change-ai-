# KBO Contextual Bandit

## Overview

이 패키지는 KBO 투수 교체 의사결정 중 “교체 상황에서 실제 경기 불펜 후보들 중 어떤 투수를 추천할 것인가”를 평가하기 위한 Contextual Bandit 실험 코드 모음입니다.

현재 reward는 pitcher perspective normalized WPA/PA이며, Option B 기반입니다. 현재 frozen baseline은 B0입니다.

## Policy Lineage

### B0

- 현재 frozen Contextual Bandit baseline
- actual game-day bullpen candidate set
- leakage-safe pitcher as-of profile
- context × pitcher interactions
- normalized WPA/PA reward
- alpha = 0
- current main baseline

### B1

- B0 + leverage_index + LI × pitcher interactions
- reward prediction / manager agreement는 개선
- Test OPE policy-value improvement는 robust하지 않았음
- final policy로 채택하지 않고 ablation 결과로 보존

### B2

- B0 + quality_p
- pending / next experiment

### B3

- B0 + LI + quality_p
- pending

### B4

- B0 + LI + quality_p + LI × quality_p interaction
- pending

이후에는 λ 기반 rule-based utility 및 quality_p × LI - λ × cost 계열 실험을 예정하고 있습니다.

## Directory Structure

- `core/`: shared contextual bandit implementation
- `diagnostics/`: candidate / reward / profile / alpha / segment diagnostics
- `ope/`: behavior policy, DM / IPS / SNIPS / DR, weight clipping robustness, paired game bootstrap
- `leverage/`: Train-frozen LI, B1 LI ablation, segment analysis, LI OPE, Common-Q sensitivity
- `quality/`: B2/B3/B4 예정
- `utils/`: game split export 등의 helper

## Evaluation

현재 주요 검증 항목은 다음과 같습니다.

- Agreement
- Top-3 / Top-5
- actual pitcher rank
- reward prediction
- behavior propensity
- DM
- IPS
- SNIPS
- DR
- ESS
- weight clipping robustness
- paired game bootstrap
- Common-Q sensitivity

Agreement는 manager imitation metric이며 policy quality 자체가 아닙니다. OPE는 estimated policy value로 해석하며 인과적 진실이라고 표현하지 않습니다.

## Important Data / Split

Chronological split:

- Train: 34,138 relief decisions
- Validation: 7,904
- Test: 8,135

기간:

- Train: 2016-04-01 ~ 2023-04-01
- Validation: 2023-04-01 ~ 2024-06-20
- Test: 2024-06-20 ~ 2025-10-04

game split은 날짜 문자열만으로 자르지 않고 game 단위 chronological split을 사용합니다. 정확한 split은 `data_analysis/results/bandit_game_split.csv`를 참고하세요.

## Running

프로젝트 루트 기준 예시입니다.

```bash
python -m kbo_bandit.leverage.bandit_v3_li_ablation
python -m kbo_bandit.ope.bandit_v2_ope
python -m kbo_bandit.diagnostics.analyze_bandit_v2_segments
python -m kbo_bandit.utils.export_bandit_game_split
```

## Outputs

실험 결과는 `outputs/bandit/` 하위에 저장합니다.

후속 모델 입력으로 재사용하는 가공 데이터는 `data_analysis/results/`에 유지합니다.
