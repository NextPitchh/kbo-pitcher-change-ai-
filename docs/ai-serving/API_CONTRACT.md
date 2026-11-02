# AI 추론 API 계약 (2025)

개발 주소는 `http://localhost:8000`, 같은 Docker 네트워크에서는 서비스 이름을 설정한 `http://nemosdream-ai:8000`을 사용할 수 있습니다. 실제 배포 주소는 아직 정해지지 않았습니다. 두 API는 JSON POST이며 `/health`는 현재 B1 로드 상태만 나타냅니다. v6의 정상 여부는 별도 예측 요청으로 확인해야 합니다.

## B1: POST /v1/bandit/recommend

모델 `b1-optionb-chronology67360-v3`. 점수는 후보 교체 투수의 예측 보상이며 승리확률이 아닙니다. 현재 투수 유지(STAY)에 점수를 부여하지 않습니다. 호출자가 현재 투수 및 사용한 투수를 제외한 불펜 후보의 경기 시작 전 스냅샷을 제공합니다. 추가 피로·부상 정책은 별도 책임입니다. prior 값이 없으면 필드를 생략하지 않고 null을 보냅니다. 투수 ID는 문자열 또는 정수입니다.

`score_diff_defense`는 수비팀 점수 − 공격팀 점수, `is_base1/2/3`과 `is_home_defense`는 각각 0 또는 1입니다. 후보 투입 직후 기준으로 `n_pitchers_used`는 기존 사용 투수 수 + 1, `bullpen_available_ratio`는 (현재 미사용 불펜 수 − 1) / 전체 불펜 명단 수입니다. `leverage_index`는 B1 TRAIN-frozen lookup 값이고 화면용 LI와 다릅니다. lookup 키는 이닝, 초말, 아웃, 주자, 공격팀 점수차이며 점수차를 -4~4로 제한합니다. lookup은 1~9회만 지원합니다. 모델 내부에서 추가 표준화하므로 리그 baseline으로 다시 나누지 않습니다.

요청 예제는 원본 동결 `outputs/bandit/serving/b1_final_serving_v3/example_request.json`, 실제 기대 응답은 같은 폴더의 `example_output.json`입니다. 예제 두 파일은 배포 이미지에 포함되지 않습니다. 요청 형태:

```json
{"game_state":{"inning":5.0,"out_count":0.0,"score_diff_defense":-1.0,"is_base1":0.0,"is_base2":0.0,"is_base3":0.0,"is_home_defense":1.0,"bullpen_available_ratio":0.9090909090909091,"n_pitchers_used":2.0,"leverage_index":0.8064482681933727},"candidates":[{"pitcher_id":"52204","prior_n_apps":9.0,"prior_pitcher_reward_mean":0.003665111111111111,"prior_wpa_std":0.02870109206739547,"prior_bb_rate":0.2857142857142857,"prior_so_rate":0.14285714285714285,"prior_hr_rate":0.0,"prior_avg_pa_per_app":3.111111111111111},{"pitcher_id":"52206","prior_n_apps":2.0,"prior_pitcher_reward_mean":-0.014227500000000004,"prior_wpa_std":0.08103514423075954,"prior_bb_rate":0.18181818181818182,"prior_so_rate":0.18181818181818182,"prior_hr_rate":0.0,"prior_avg_pa_per_app":5.5}]}
```

```json
{"model_version":"b1-optionb-chronology67360-v3","rankings":[{"pitcher_id":"52204","predicted_reward":-0.010522805277505758,"model_rank":1},{"pitcher_id":"52206","predicted_reward":-0.011436264376065089,"model_rank":2}]}
```

## CatBoost: POST /v1/pa/predict-service

모델 `catboost-pa-service-v6-b-7e1e4a64634c`, 스키마 `pa-service-v6-b-before-pitch-v1`. 첫 투구 전 상태를 보냅니다. 현재 스키마는 2025 경기 ID와 1~9회만 허용합니다. `base_state`는 `0`, `1`, `2`, `3`, `12`, `13`, `23`, `123` 중 하나입니다. ID는 선행 0이 없는 양의 십진 ID입니다. 점수는 양 팀 점수이고 모델은 공격팀 관점 점수차로 변환합니다.

```json
{"state":{"game_id":"20250322HHKT02025","inning":5,"half":"top","out_count":0,"base_state":"0","home_score":2,"away_score":1,"pitcher_id":54354,"batter_id":50704,"total_pitch_count_before":63,"inning_pitch_count_before":0}}
```

실제 응답에는 `model_version`, `feature_schema_version`, `mapping_version`, `probability_provenance`, `features`, `native_probabilities`, `web_probabilities`, `history_cutoff`, `input_provenance`가 있습니다. 위 요청의 동결 검증 응답은 `outputs/serving/catboost_service_v6/serving_validation.json`의 `cases.A.response`에 있습니다. `web_probabilities` 예시는 다음과 같습니다.

```json
{"1B":0.18248802117378662,"2B":0.03882699704290742,"3B":0.003631345497043199,"BB":0.08610284252985136,"FO":0.23161711265051602,"GDP":0.0,"GO":0.23930343699973367,"HR":0.022588003316212607,"SO":0.19544224078994898}
```

`native_probabilities`는 9종 직접 예측이며, `web_probabilities`는 2아웃 또는 1루가 비었을 때 GDP 확률을 GO에 합친 값입니다. 각 합은 1입니다. v6는 2023년 말까지 고정된 타자·투수 대 타자 이력을 Python에서 조회합니다. 누락 키는 0이며 2024·2025 이력으로 갱신하지 않습니다. v5의 `vsHra`/`todayHra`는 넣지 않습니다.

경기 중 이닝, 초말, 아웃, 주자, 점수, 타순과 투수를 현재 가상 상태에서 갱신합니다. 투수 교체 시 해당 투수의 이번 경기 완료 누적 구수와 이닝 구수를 사용합니다. 새 투수는 0/0, 이닝이 바뀌면 이닝 구수는 0입니다. PA 완료 후 `pitch_count_policy.json`의 결과별 분포로 구수를 샘플링해 두 카운터에 더합니다. 미래 실제 PA의 구수를 복사하지 않습니다.

## 오류와 책임

잘못된 요청은 422 (`detail`), 모델 누락 또는 로드 실패는 503, 추론 실패는 500입니다. 호출자는 연결·읽기 timeout과 실패 정책을 명시해야 합니다. API는 전달된 경기 상태와 명단이 실제 DB에 존재하는지 검증하지 않습니다. v5 `/v1/pa/predict`, `/v1/pa/predict-web`은 연구용이며 별도의 v5 artifact가 필요합니다. 제공된 최소 이미지에는 그 artifact를 포함하지 않습니다.
