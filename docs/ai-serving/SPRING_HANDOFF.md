# Spring 인계: 현재 상태와 남은 작업

현재 B1 `b1-optionb-chronology67360-v3`는 교체 후보의 보상을 평가하고, CatBoost `catboost-pa-service-v6-b-7e1e4a64634c`는 타석 결과 9종의 확률을 제공합니다. B1 점수는 승리확률이 아닙니다. v6는 Log5보다 낫다고 주장할 수 없습니다. 2025 독립 테스트 log loss는 v6 1.788382, Log5 재구성 1.782328입니다. Log5는 별도 비교 및 대체 경로로 유지하고, 실패 시 자동 전환 여부는 팀이 명시적으로 정해야 합니다.

요청·응답, 입력 의미, 2025 및 1~9회 제한은 [API 계약](API_CONTRACT.md)을 확인하세요. 로컬·Docker 실행과 아티팩트는 [배포 메모](DEPLOYMENT.md)에 있습니다. 실제 AWS 배포는 수행하지 않았습니다.

BE `develop`에는 B1 추천과 AI ETL이 아직 구현되지 않았습니다. 별도 로컬 브랜치 `codex/catboost-v6-client`에는 `CatBoostV6State`와 `CatBoostV6Client`가 있으며 실제 로컬 FastAPI 호출을 테스트했습니다. 이 브랜치는 아직 PR이나 엔진 연결이 아닙니다. 원격 `feat/#25-simulate-pa` 브랜치가 존재하며 해당 팀원 코드는 변경하지 않았습니다. 현재 `SimState`의 `PitcherDay`는 전체 구수만 보유하고 분기 이후 타석당 평균을 더하도록 설계되어 있어, v6의 완료된 이닝별·전체 구수 의미를 충족하려면 팀원 코드 변경이 필요합니다. 실제 엔진에 CatBoost가 연결된 상태가 아닙니다.

필요한 후속 작업:

1. B1용 2025 경기별 투수 snapshot 및 TRAIN LI를 별도 테이블로 적재하고 키·행 수·선수/경기 관계를 로컬 DB에서 검증합니다. 기존 화면 LI와 별개로 조회하고 `load.sh`의 전체 TRUNCATE를 실행하지 않습니다.
2. 세션에서 수비팀, 현 투수 및 사용 투수, 미사용 불펜과 경기 상태를 구성합니다. 후보 투입 직후의 사용 수·가용 비율을 계산하고 B1을 호출합니다. 빈 후보와 누락 snapshot을 명시적으로 처리합니다.
3. 각 PA 직전 상태 및 해당 투수의 완료 누적·이닝별 구수를 v6에 전달합니다. 타순 변경 시 타자를, 투수 교체 시 해당 투수의 카운터를 갱신합니다. 결과별 투구 수 동결 분포를 샘플링하고 이후 카운터를 갱신합니다. 기존 `RunnerAdvancement`와 Dice를 재사용합니다.
4. Python이 고정 이력을 직접 사용하므로 같은 이력을 DB에 중복 적재하는 것은 필수는 아닙니다. 별도 적재 시 완전한 버전과 656/95,448행을 검증한 뒤 누락 키를 0으로 해석합니다. 실제 PostgreSQL 검증은 아직 하지 않았습니다.
5. 전체 경기 반복 `/api/v1/sessions/{id}/repeat`는 기존 Spring 엔진에서 독립 seed 100개, 사용자 선택 재적용과 통계 집계가 필요합니다. 구현 담당자는 아직 확정되지 않았습니다. Python 연구용 단기 Monte Carlo는 그대로 보존하며 전체 경기 API가 아닙니다.

원본 B1 상세 계약: `outputs/bandit/serving/b1_service_features/DATA_CONTRACT.md`. v6 상세 계약 및 샘플 정책: `outputs/serving/catboost_service_v6/SERVICE_CONTRACT_ko.md`, `pitch_count_policy.json`, `kbo_bandit/serving/pa_pitch_count.py`. 이 파일들은 `outputs/`가 Git 무시 대상이므로 전달 경로가 정해져야 팀원이 접근할 수 있습니다.
