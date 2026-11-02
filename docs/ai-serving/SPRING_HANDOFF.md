# Spring 인계: 현재 상태와 남은 작업

현재 B1 `b1-optionb-chronology67360-v3`는 교체 후보의 보상을 평가하고, CatBoost `catboost-pa-service-v6-b-7e1e4a64634c`는 타석 결과 9종의 확률을 제공합니다. B1 점수는 승리확률이 아닙니다. v6는 Log5보다 낫다고 주장할 수 없습니다. 2025 독립 테스트 log loss는 v6 1.788382, Log5 재구성 1.782328입니다. Log5는 별도 비교 및 대체 경로로 유지하고, 실패 시 자동 전환 여부는 팀이 명시적으로 정해야 합니다.

요청·응답, 입력 의미, 2025 및 1~9회 제한은 [API 계약](API_CONTRACT.md)을 확인하세요. 로컬·Docker 실행과 아티팩트는 [배포 메모](DEPLOYMENT.md)에 있습니다. 실제 AWS 배포는 수행하지 않았습니다.

BE `develop`에는 CatBoost v6 동결 이력 ETL, `CatBoostV6FeatureAssembler`, HTTP 클라이언트와 Log5·CatBoost를 선택하는 독립 확률 공급 계층이 반영되어 있습니다. 자세한 사용 방법은 [BE 인계 문서](https://github.com/NextPitchh/BE/blob/develop/docs/catboost-v6-handoff.md)를 참고하세요. 이 계층은 아직 `simulate-pa` 엔진에 연결되지 않았습니다. `PitcherDay`는 이닝별 완료 구수를 보유하지 않으며 기존 분기 후 평균 구수 방식은 v6 입력 의미와 다르므로 팀원 엔진의 상태 관리 변경이 필요합니다. B1 추천 및 B1 전용 ETL은 별도 후속 작업입니다.

필요한 후속 작업:

1. B1용 2025 경기별 투수 snapshot 및 TRAIN LI를 별도 테이블로 적재하고 키·행 수·선수/경기 관계를 로컬 DB에서 검증합니다. 기존 화면 LI와 별개로 조회하고 `load.sh`의 전체 TRUNCATE를 실행하지 않습니다.
2. 세션에서 수비팀, 현 투수 및 사용 투수, 미사용 불펜과 경기 상태를 구성합니다. 후보 투입 직후의 사용 수·가용 비율을 계산하고 B1을 호출합니다. 빈 후보와 누락 snapshot을 명시적으로 처리합니다.
3. 각 PA 직전 상태 및 해당 투수의 완료 누적·이닝별 구수를 v6에 전달합니다. 타순 변경 시 타자를, 투수 교체 시 해당 투수의 카운터를 갱신합니다. 결과별 투구 수 동결 분포를 샘플링하고 이후 카운터를 갱신합니다. 기존 `RunnerAdvancement`와 Dice를 재사용합니다.
4. Python은 고정 이력을 직접 사용합니다. BE의 독립 CatBoost 공급 계층은 별도 적재한 동결 이력과 Python 응답의 Feature를 대조합니다. BE 쪽에서 PostgreSQL 18 격리 테스트 DB에 656/95,448행을 적재해 조회·Cold-start·실제 FastAPI 연동을 검증했습니다. 운영 DB 적재 및 전체 엔진 연결 검증은 아직 수행하지 않았습니다.
5. 전체 경기 반복 `/api/v1/sessions/{id}/repeat`는 기존 Spring 엔진에서 독립 seed 100개, 사용자 선택 재적용과 통계 집계가 필요합니다. 구현 담당자는 아직 확정되지 않았습니다. Python 연구용 단기 Monte Carlo는 그대로 보존하며 전체 경기 API가 아닙니다.

원본 B1 상세 계약: `outputs/bandit/serving/b1_service_features/DATA_CONTRACT.md`. v6 상세 계약 및 샘플 정책: `outputs/serving/catboost_service_v6/SERVICE_CONTRACT_ko.md`, `pitch_count_policy.json`, `kbo_bandit/serving/pa_pitch_count.py`. 이 파일들은 `outputs/`가 Git 무시 대상이므로 전달 경로가 정해져야 팀원이 접근할 수 있습니다.
