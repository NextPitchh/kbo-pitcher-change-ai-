# 투수 품질 지표(quality_p) 검증 리포트

작성일: 2026-09-06 (불펜 섹션[Task 6~10] 추가: 2026-09-15)
관련 코드: `data_analysis/methods/pitcher_quality.py`,
`build_quality_baseline.py`, `validate_quality_candidates.py`,
`validate_quality_predictive.py`, `decide_quality_scaling.py`,
`tune_quality_shrinkage.py`,
`tune_quality_shrinkage_bullpen.py`, `validate_quality_predictive_bullpen_smoothed.py`,
`check_bullpen_wpa_li_confound.py`, `tune_quality_shrinkage_bullpen_wpa.py`,
`build_quality_baseline_bullpen.py`

## 배경

`prior_wpa_mean`(최근 등판 승리확률기여 평균)을 실력 지표로 쓰려 했으나,
자기상관 r=0.83이 "선발은 계속 선발로, 불펜은 계속 불펜으로 나온다"는
역할 고정 효과를 실력으로 착각한 착시였음이 밝혀졌다 (역할 통제 시 불펜
0.11 / 선발 -0.07로 붕괴). DIPS 이론(삼진·볼넷·피홈런 비율은 수비/운의
영향을 덜 받아 투수 개인 실력을 더 순수하게 반영한다)에 따라 대안
지표를 재검증했다.

---

## Task 1: 후보 지표 자기상관 재검증

방법: 투수별 등판을 시간순으로 정렬해 전반부/후반부로 분할, 선발/불펜을
분리해 Spearman 상관(전반부 값 vs 후반부 값) 측정. 등판 10회 이상만 포함
(선발 297명, 불펜 463명).

| 후보 | 선발 r | 불펜 r |
| --- | --- | --- |
| prior_so_rate | 0.705 | 0.463 |
| prior_bb_rate | 0.656 | 0.456 |
| prior_hr_rate | 0.198 | 0.138 |
| so_rate − bb_rate | **0.734** | 0.408 |
| composite(so−bb−2·hr) | **0.740** | 0.371 |
| wpa_per_pa(정규화) | 0.317 | 0.309 |
| wpa_mean_raw(구 prior_wpa_mean) | 0.112 (p=0.055, 유의하지 않음) | 0.263 |
| n_apps 단독 | 0.997* | 1.000* |

\* n_apps는 절반분할 방식 자체가 앞/뒤 개수를 거의 동일하게 만드는
구조적 산물이라 참고하지 않음.

**대조군**: 역할 분리 없이(선발+불펜 혼합) wpa_mean_raw 계산 시 r=0.726 —
과거 알려진 0.83과 유사한 착시가 재현됨을 확인.

**결론**: so_rate, bb_rate(및 조합)는 역할 통제 후에도 강한 자기상관
유지 → 실력 신호로 채택. wpa_mean_raw는 역할 통제 시 붕괴 → 탈락 재확인.

상세: `quality_candidate_autocorrelation.json`

---

## Task 2: 실제 성과 예측력 검증

방법: X=prior_*(해당 시점 누적값), y=이 등판 자체의 실제 wpa_per_pa
(투수 관점, 부호 반전). Spearman + 5-fold CV R². 지표별 기대 부호(볼넷·
홈런·구WPA는 "높을수록 나쁨"이라 음의 상관이 정상)를 반영해 판정.

| 후보 | 선발 r | 불펜 r |
| --- | --- | --- |
| composite(so−bb−2·hr) | **0.153** ✅ | 0.028 |
| so_rate − bb_rate | **0.147** ✅ | 0.029 |
| prior_so_rate | 0.110 ✅ | 0.024 |
| prior_bb_rate | -0.110 ✅ (방향 정상) | -0.025 |
| prior_hr_rate | -0.063 | -0.001 |
| wpa_per_pa | 0.013 (n.s.) | **0.105** ✅ (간신히) |
| wpa_mean_raw | 0.085 (방향도 불안정) | -0.087 |

기준: r>0.1(방향 반영) & p<0.05.

**선발**: composite가 최고 성능(r=0.153)으로 채택. so_rate/bb_rate
단독도 통과하나 조합이 더 우수.

**불펜**: 사실상 전 지표 기준 미달(wpa_per_pa만 간신히 0.105로 통과선
근처). 원인 추정: 불펜 등판은 타자 2~3명 수준으로 짧아, 개별 등판의
실제 결과가 실력보다 표본노이즈에 더 좌우됨(Task 1의 자기상관은 여러
등판에 걸쳐 평균화되어 안정적이었던 것과 대비됨).

**결정 (2026-09-06, 사용자 확인)**: 불펜 quality_p 공식은 팀 논의 후
별도 확정. 이 리포트는 **선발 전용** 최종 공식만 다룬다.

상세: `quality_candidate_predictive.json`

---

## Task 3: 스케일링 결정

quality_p가 `U = quality_p × LI − λ·cost·(...)`에서 LI와 곱해지므로,
LI(평균 1.0, p25 0.51 / p50 0.84 / p75 1.30 / p95 2.30 / 최대 9.79)와
자릿수를 맞춰야 한다.

| 방식 | 평균 | p25 | p50 | p75 | p95 | 최대 |
| --- | --- | --- | --- | --- | --- | --- |
| (a) 원값 | 0.022 | -0.012 | 0.024 | 0.059 | 0.120 | 0.500 |
| (b) z-score | 0.000 | -0.517 | 0.026 | 0.555 | 1.484 | 7.208 |
| (c) 백분위 | 0.500 | 0.250 | 0.500 | 0.750 | 0.950 | 1.000 |

(a)는 LI 대비 자릿수가 너무 작고, (b)는 평균이 0이라 평균급 투수의 U
기여가 0이 되는 문제 및 소표본 극단치(-15.4) 노출, (c)는 범위가 좁음.

**결정: 백분위 × 2** (사용자 확인 후 채택). 평균 1.0, 범위 0~2, 25~75%
구간(0.5~1.5)이 LI(0.51~1.30)와 자릿수 일치. 순위 기반이라 소표본
극단치에도 안정적.

**한계**: 백분위는 실력 격차의 "크기" 정보를 압축한다(최댓값이 LI처럼
9.8까지 못 벌어지고 2.0에서 막힘). 실제 추천 결과에 문제가 되는지는
추후 U 공식 통합 시 재확인 필요.

상세: `quality_scaling_comparison.json`

---

## Task 4: Shrinkage 계수 k 확정

`quality_shrunk = w·raw + (1-w)·league_mean`, `w = n_apps/(n_apps+k)`.
k=5/10/15/20을 Task 2와 동일 방식(선발, composite)으로 재검증.

| k | Spearman r | CV R² |
| --- | --- | --- |
| (미적용) | 0.1527 | 0.0223 |
| **5** | **0.1568** | **0.0260** |
| 10 | 0.1566 | 0.0257 |
| 15 | 0.1557 | 0.0252 |
| 20 | 0.1547 | 0.0247 |

**결정: k=5.** k가 커질수록(5→20) 예측력이 지속적으로 하락하는 추세
확인 — k<5 구간의 추가 탐색 여지가 있으나 이번 범위(지시서 지정) 내에서는
5가 최적.

상세: `quality_shrinkage_tuning.json`

---

## 최종 확정 공식 (선발 전용)

```
raw    = prior_so_rate - prior_bb_rate - 2 * prior_hr_rate
shrunk = w * raw + (1-w) * league_mean_composite(=0.0220),  w = n_apps/(n_apps+5)
quality_p = 2 * percentile_rank(shrunk, 기준분포)   # 평균 1.0, 범위 0~2
```

구현: `data_analysis/methods/pitcher_quality.py` (`compute_quality_p`)
기준분포: `data_analysis/results/quality_baseline.json`
(선발 14,137건 기준, 1001포인트 백분위 그리드)

**불펜(is_starter=False)은 이 시점(2026-09-06)에는 미구현이었음.**
2026-09-15에 Task 6~10으로 별도 확정됨 — 아래 섹션 참고.

## 검증된 동작 (선발, pitcher_quality.py 자체 테스트)

- 리그 평균급 투수(so=0.166, bb=0.099, hr=0.023) → quality_p ≈ 0.95~0.97
- quality_p는 항상 [0, 2] 범위 유지
- 같은 raw 값이라도 등판 수 적은 투수(n=1)는 등판 많은 투수(n=300)보다
  quality_p가 낮게 나옴(shrinkage 정상 작동)

---

# 불펜(is_starter=False) 공식 확정 — Task 6~10 (2026-09-15)

## 배경

선발 공식(Task 1~5)이 확정된 시점엔 불펜용 지표가 Task 2 예측력 검증을
통과하지 못해(최고 wpa_per_pa r=0.105로 간신히 걸침) `compute_quality_p`가
`is_starter=False` 입력에 `NotImplementedError`를 내도록 의도적으로
막아뒀다. 2026-09 팀 회의에서 "불펜은 등판당 상대 타자 수 편차가 커서
(1~10명+), 표본 크기를 등판 횟수(`prior_n_apps`)가 아니라 누적 상대
타석 수(`prior_n_pa`)로 재야 하는 것 아니냐"는 문제 제기가 있었고, 이를
검증하는 과정에서 더 근본적인 원인과 해법을 찾았다.

## Task 6: shrinkage 표본 단위 교체 시도 (실패)

기존 선발 공식(composite = `so_rate - bb_rate - 2*hr_rate`)은 그대로 두고,
shrinkage의 표본 크기만 `prior_n_apps` → `prior_n_pa`로 교체해 재검증.
목표(y)는 Task 2와 동일하게 "바로 다음 등판 1번의 실제 wpa_per_pa".

| k (prior_n_pa 단위) | Spearman r | p-value |
| --- | --- | --- |
| 없음 | 0.0276 | 7.9e-10 |
| 500 | 0.0386 | 7.6e-18 |
| **2000~3000 (그리드 밖 추가 탐색 결과 실제 최적점)** | **0.0399** | ~7e-19 |

k를 아무리 키워도(최적점 k≈2000~3000, n=49,605) r이 0.04 근처에서
수렴 — 기준(r>0.1, p<0.05) 미달. **결론: 표본 재는 단위(등판 수 vs
타석 수) 자체는 문제의 핵심이 아니었음.** 표본 자체는 충분히 컸으므로
(n=49,605), 통계적 유의성(p<0.05)은 항상 확보되지만 효과 크기(r)가
근본적으로 작다는 게 문제.

상세: `quality_shrinkage_tuning_bullpen.json`

## Task 7: 목표(y) 재정의 재검증 — prior_wpa_per_pa 발견

**원인 진단**: 불펜은 등판 1회 평균 4.56타자 상대, 선발은 22.9타자
(비율 5.02배). "바로 다음 등판 1번"의 결과를 맞히는 것 자체가 불펜한테는
표본이 너무 작아 실력보다 그날의 노이즈(운)에 결과가 좌우되는 구조적
문제로 판단.

**대응**: 목표를 "다음 등판 1번" → **"다음 5등판 평균"**으로 재정의
(5 = 22.9÷4.56 ≈ 선발 1경기와 동일한 타자 수가 되도록 역산 — 임의
선택 아님). X(prior_*)는 그대로 유지, y만 전방 5등판 구간의 실제
wpa_per_pa 가중평균(n_pa 가중)으로 교체.

| 후보 지표 | Spearman r | p-value | 판정 |
| --- | --- | --- | --- |
| prior_so_rate | -0.0292 | 2.0e-10 | fail |
| prior_bb_rate | -0.0036 | 0.44 | fail |
| prior_hr_rate | 0.0114 | 0.013 | fail |
| prior_composite (기존 선발 공식) | **-0.0296** | 1.3e-10 | **fail (부호 반전)** |
| **prior_wpa_per_pa** | **0.1487** | 1.1e-231 | **PASS** |

기존 DIPS 기반 composite는 목표를 바꿔도 여전히 실패(오히려 부호까지
반대로 뒤집힘 — "삼진 잘 잡을수록 다음 5등판이 나쁘다"는 반직관적
결과가 통계적으로 유의하게 나옴, 단 효과크기는 미미). 대신 Task 1~2
당시 선발/불펜 혼합 착시로 탈락했던 `prior_wpa_per_pa`가 **순수 불펜
집단에서는** 기준을 명확히 통과(r=0.1487, CV R²=0.0243 — 선발 최종
성능과 비등함). **결론: 불펜은 선발과 다른 공식이 필요함.**

상세: `quality_candidate_predictive_bullpen_smoothed.json`

## Task 8: WPA 지표의 LI(레버리지) 착시 여부 검증

**우려**: WPA는 상황 긴장도(leverage_index)가 높을수록 같은 결과라도
값이 커진다. 감독이 신뢰하는 투수를 접전 상황에 반복 기용해서
`prior_wpa_per_pa`가 높게 나온 것일 뿐, 실력 차이가 아닐 가능성 —
이 프로젝트가 이미 "선발/불펜 혼합 역할 착시"로 지표 하나를 탈락시킨
전례가 있어(prior_wpa_mean, r=0.83→0.11/0.07) 반드시 확인 필요.

**방법**: 편상관(partial correlation). `prior_li_mean`(그 투수의 과거
등판들의 n_pa 가중 평균 LI, leakage 없는 방향으로 시점 이전만 반영)과
`li_window_mean`(y와 동일한 전방 5등판 구간의 n_pa 가중 평균 LI)을
각각 통제한 뒤에도 `prior_wpa_per_pa` ↔ `y_window_mean` 상관이
남는지 확인.

| 구분 | Spearman r |
| --- | --- |
| LI 통제 전 | 0.1471 |
| LI 통제 후 (편상관) | 0.1458 |
| 감소율 | 0.9% |

**결론: 착시 아님.** LI를 통계적으로 걷어내도 상관관계가 거의 그대로
유지됨 → 진짜 실력 신호로 판정. (참고: `prior_li_mean`과 `prior_wpa_per_pa`는
오히려 약한 음의 상관(-0.1251, `y_window_mean`↔`li_window_mean`은 -0.2409)으로,
우려했던 방향(높은 LI 기용 → 높은 WPA)과 반대로 나옴 — 고LI 상황은
상대 타자 수준도 높고 실패 시 WPA 손실 폭도 커서 평균이 눌리는 것으로
추정되나, 원인 규명은 이 리포트 범위 밖이며 최종 결론에는 영향 없음.)

상세: `check_bullpen_wpa_li_confound.py` 실행 로그 (별도 JSON 저장 없음)

## Task 9: shrinkage 계수 k 확정 (WPA 기반 공식)

Task 7에서 확정된 `prior_wpa_per_pa`에 `prior_n_pa` 기준 shrinkage를
적용해 k를 재탐색. 목표는 Task 7과 동일한 `y_window_mean`(전방 5등판
평균).

| k | Spearman r | CV R² |
| --- | --- | --- |
| 없음 | 0.1487 | 0.0243 |
| 5 | 0.1488 | 0.0266 |
| **25** | 0.1476 | **0.0275** |
| 50 | 0.1465 | 0.0275 |
| 100 | 0.1451 | 0.0271 |
| 1000+ | 계속 하락 | 계속 하락 |

선발과 달리 shrinkage 효과가 크지 않음 — 순위 정확도(r) 기준으로는
k=1~5 근처가 최고치와 사실상 동일(원래 지표 자체가 이미 안정적이라는
뜻), 예측 안정성(CV R²) 기준으로는 k=25~50 근처가 최선. **k=25 채택**
— 순위 손실을 사실상 감수하지 않으면서(0.1488→0.1476) 안정성 이득은
확실히 얻는 균형점(0.0243→0.0275, +13%).

상세: `quality_shrinkage_tuning_bullpen_wpa.json`

## Task 10: 불펜 기준분포(baseline) 구축

Task 9까지 확정된 공식으로 불펜 모집단(is_starter=False,
prior_n_apps≥1, 49,605건)의 shrunk 값을 0.1%포인트 간격(1001개 지점)
백분위 그리드로 저장 — 선발용 `quality_baseline.json`과 동일한 방식,
분포 자체가 다르므로 별도 파일로 분리.

- `league_mean_wpa_per_pa = -0.009747`
- 구현: `data_analysis/methods/build_quality_baseline_bullpen.py`
- 저장: `data_analysis/results/quality_baseline_bullpen.json`

## 최종 확정 공식 (불펜 전용)

```
raw    = prior_wpa_per_pa   (= -prior_wpa_sum / prior_n_pa, 기존 leakage-free
                              컬럼들을 대수적으로 조합 — prior_* 재계산 아님)
shrunk = w * raw + (1-w) * league_mean_wpa_per_pa(=-0.009747),  w = prior_n_pa/(prior_n_pa+25)
quality_p = 2 * percentile_rank(shrunk, 불펜 기준분포)   # 평균 1.0, 범위 0~2
```

구현: `data_analysis/methods/pitcher_quality.py` (`compute_quality_p`,
`is_starter=False` 분기)
기준분포: `data_analysis/results/quality_baseline_bullpen.json`

**선발과 불펜은 원인 진단부터 공식, shrinkage 단위까지 전부 다르다** —
단순히 k 값만 다른 게 아니라 두 개의 독립된 공식임에 유의.

| 구분 | 선발 | 불펜 |
| --- | --- | --- |
| raw 공식 | `so_rate - bb_rate - 2*hr_rate` | `prior_wpa_per_pa` |
| shrinkage 단위 | `prior_n_apps` (등판 수) | `prior_n_pa` (누적 타석 수) |
| k | 5 | 25 |
| league_mean | 0.0220 | -0.009747 |
| 검증 r (목표: 이 등판) | 0.153 | — (미검증, 이 목표에선 실패) |
| 검증 r (목표: 전방 5등판 평균) | — (미검증) | 0.1487(raw)/0.1476(k=25) |

## 검증된 동작 (불펜, pitcher_quality.py 자체 테스트)

- 리그 평균급 불펜(raw=league_mean) → quality_p ≈ 1.12 (정확히 1.0이
  아닌 이유: 불펜 분포가 비대칭이라 리그 평균값이 50th가 아닌 55.8th
  백분위에 위치 — 버그 아니고 Task 3에서 이미 문서화한 백분위 스케일링의
  알려진 한계와 같은 종류)
- quality_p는 항상 [0, 2] 범위 유지
- 같은 raw 값이라도 누적 타석 수 적은 투수(n_pa=5)는 많은 투수(n_pa=3000)
  보다 quality_p가 낮게 나옴(shrinkage 정상 작동)
- 더 이상 `NotImplementedError` 발생하지 않음(2026-09-15부로 해제)

---

# Task 15 — 불펜 WPA 재료 Option A → B 교정 (2026-09-22)

## 배경

Bandit 팀(`kbo_bandit/`)이 독립적으로 자신들의 reward 계산을 검증하던 중
`all_pitcher_history.app_wpa`가 Option A(아웃카운트 보정 없는 낡은 WE
계산)로 만들어졌음을 확인하고, 자체 reward 계산에서는 이를 우회해
PA 레벨 `reward_wpa_computed`(Option B)로 직접 재계산했다는 사실이
확인됨. 불펜 quality_p의 `prior_wpa_per_pa`가 바로 이 `app_wpa`를 누적한
`prior_wpa_mean`에서 유도된 값이라, 같은 문제가 우리 쪽에도 있는지 점검함.

**직접 검증**: 같은 투수·같은 경기에 대해 `all_pitcher_history.app_wpa`와
`all_pa_with_li.reward_wpa_computed`를 독립적으로 합산한 값을 비교 — 다수
사례에서 유의미하게 다름을 확인(부호가 반대로 나온 사례 포함). Option A
오염이 실재함을 확인.

**영향 범위**: 불펜 공식(`prior_wpa_per_pa`)만 영향 있음. 선발 공식
(`so_rate - bb_rate - 2*hr_rate`)은 WPA를 전혀 안 써서 무관.

## 재료 재생성

`build_prior_wpa_optionb.py`: `all_pa_with_li.reward_wpa_computed`를
(game_id, half, pitcher_id, appearance_order) 단위로 합산해 등판별 Option B
WPA를 만들고, 투수별 시간순으로 leakage-free 누적평균(`prior_wpa_mean_optionb`)
계산. 기존 `all_pitcher_history.prior_n_apps`와 완전히 동일하게 세어지는지
검증(불일치 0건, 64,571건 전부 일치) — **표준 품질필터(data_quality_flag/
pa_result)를 적용하면 오히려 불일치 3,763건 발생** → 원본 파일 자체가
필터 없이 집계됐던 것으로 확인, 필터 미적용으로 확정.

기존(Option A) vs 신규(Option B) `prior_wpa_mean` 상관계수: 0.9866
(전체적으로는 비슷하지만 개별 사례 차이는 실재).

## 재검증 — Task 7~9를 동일 방법론으로 재확인

재료만 교체하고 방법론(목표 정의, 편상관 통제, shrinkage 튜닝)은
동일하게 유지해 다시 검증:

| 항목 | Option A (기존) | Option B (교정) | 판정 |
| --- | --- | --- | --- |
| Task 7 (예측력, r) | 0.1487 | 0.1284 | 여전히 PASS (기준 0.1) |
| Task 8 (LI 통제 후 감소율) | 0.9% | 0.1% | 여전히 착시 아님 (더 견고) |
| Task 9 (최적 k) | k=25 | k=25 | 동일 (재확인) |
| league_mean_wpa_per_pa (train-frozen) | -0.009477 | -0.004568 | 값 자체는 이동 |

**결론**: 공식·k·방법론은 그대로 유지, 재료(원본 WPA 값)만 Option B로
교체. 결론(PASS, 착시 아님)은 안 바뀌었지만 수치는 유의미하게 달라졌으므로
"재료 바꿔도 결론 같으니 재검증 생략" 하지 않고 매번 실제로 다시 돌려
확인함.

## 반영

`build_quality_baseline_bullpen.py`, `apply_quality_to_data.py`를
`prior_wpa_optionb.parquet` 사용하도록 수정, `quality_baseline_bullpen.json`
재생성, `all_pa_with_li.parquet`의 `quality_p` 컬럼 재계산 완료.

## 다음 단계 (이 리포트 범위 밖)

- `leverage_index`와 결합한 `U = quality_p × LI − λ·cost·(...)` 통합 검증
  (선발/불펜 quality_p 모두 준비 완료, 마지막 단계)
- 선발도 shrinkage 표본 단위를 `prior_n_apps` → `prior_n_pa`로 통일할지
  검토 (이론적으론 세이버메트릭스 안정화 지점 관례상 더 정확하나, 선발은
  이미 검증 통과 상태라 급하지 않음 — 여유 있을 때 재검증)
- k<5(선발) 추가 탐색 여부 검토
- Task 8에서 발견한 `prior_li_mean` ↔ `wpa_per_pa` 음의 상관의 원인 규명
  (결론에 영향은 없으나 설명되지 않은 채로 남아있음)
- **(팀 공유 필요) Bandit B0 자체의 `prior_pitcher_reward_mean`/
  `prior_wpa_std` feature도 동일한 Option A 오염이 있음이 확인됨
  (`kbo_bandit/core/bandit_v2_actual_bullpen.py`) — 이건 우리 쪽에서
  임의로 고칠 수 없고 팀/팀원과 논의 필요**
