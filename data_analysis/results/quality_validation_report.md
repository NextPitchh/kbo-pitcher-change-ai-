# 투수 품질 지표(quality_p) 검증 리포트

작성일: 2026-09-06
관련 코드: `data_analysis/methods/pitcher_quality.py`,
`build_quality_baseline.py`, `validate_quality_candidates.py`,
`validate_quality_predictive.py`, `decide_quality_scaling.py`,
`tune_quality_shrinkage.py`

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

**불펜(is_starter=False)은 아직 미구현** — `compute_quality_p` 호출 시
`NotImplementedError` 발생. 검증 안 된 값을 검증된 것처럼 반환하지
않기 위한 의도적 설계. 팀 논의 후 별도 공식/기준분포 확정 필요.

## 검증된 동작 (pitcher_quality.py 자체 테스트)

- 리그 평균급 투수(so=0.166, bb=0.099, hr=0.023) → quality_p ≈ 0.95~0.97
- quality_p는 항상 [0, 2] 범위 유지
- 같은 raw 값이라도 등판 수 적은 투수(n=1)는 등판 많은 투수(n=300)보다
  quality_p가 낮게 나옴(shrinkage 정상 작동)
- 불펜 입력 시 명시적으로 차단됨

## 다음 단계 (이 리포트 범위 밖)

- 불펜 quality_p 설계 방향 팀 논의
- leverage_index와 결합한 `U = quality_p × LI − λ·cost·(...)` 통합 검증
- k<5 추가 탐색 여부 검토
