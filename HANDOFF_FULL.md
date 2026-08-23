# K-Moneyball 데이터 핸드오프 — 전 구단 확장판

**대상**: CatBoost / Monte Carlo / DQN 담당 팀원
**작성일**: 2026-05-31
**데이터 범위**: KBO 2016~2025 정규시즌 전 구단

---

## 0. 먼저 읽어야 할 것

이 문서를 읽지 않고 데이터를 사용하면 안 됩니다. 특히 **3장(컨벤션)**과
**4장(적용된 정책)**은 값의 해석 자체를 바꾸는 내용입니다.

과거 이 프로젝트에서 컨벤션을 데이터 내부 추론으로 재판정하려다
다섯 번 결론이 뒤집힌 이력이 있습니다. 코드나 데이터를 보고
"이게 맞나?" 싶으면 추론하지 말고 이 문서를 확인하거나 문의하십시오.

---

## 1. 최종 산출물

### 메인 데이터

| 파일 | shape | 용도 |
|------|-------|------|
| **`data_analysis/results/all_pa_enriched.parquet`** | **558,064 × 68** | **학습 메인** |

- 총 PA: **558,064건**
- 대상 경기: **7,197경기** (2016~2025 정규시즌, 연장 제외분 반영)
- 투수: 829명 (10등판 이상 626명)

### 부속 산출물

| 파일 | shape | 내용 |
|------|-------|------|
| `all_pa_with_wpa.parquet` | 558,064 × 31 | enrichment 이전 병합본 |
| `all_pitcher_appearances.parquet` | 64,571 × 22 | 등판 단위 집계 |
| `all_pitcher_history.parquet` | 64,571 × 28 | 시점 안전 투수 이력 |
| `all_pa_bullpen_state.parquet` | 558,064 × 14 | PA별 불펜 소모 상태 |
| `all_game_lineup.parquet` | 12,466행 | 선발투수 + 시즌 스탯 |
| `all_game_bullpen.parquet` | 155,911행 | 불펜 명단 |
| `all_league_baseline.json` | — | 리그 평균 통계 |
| `game_index.csv` | — | 경기 메타 (팀 코드, 경기 상태) |
| `staging/pa_states_{year}.parquet` | ×10 | 연도별 원본 (롤백 지점) |

### 참조용 (사용 금지)

| 파일 | 이유 |
|------|------|
| `hsk_pa_enriched_STALE.parquet` | **절대 사용 금지.** 153경기 기준 구버전이며 이번 라운드 보정(정규식·터미널PA·연장정책) 미반영 |
| `hsk_pa_with_wpa.parquet` | 구 153경기 원본. 롤백 지점으로만 보존 |
| `hsk_pa_with_wpa_corrected.parquet` | 위 데이터에 신규 정책 적용본. 비교용 |

---

## 2. 데이터 생성 이력

### 확장 전 (153경기)

한화(HH) vs SK/SSG(SK) 맞대결만 수집. 11,984 PA.
WPA 자체 산출(문형우 외 2016 마르코프 RE/WE 테이블 이식)까지 완료.

### 확장 (7,200경기 → 최종 558,064 PA)

| 단계 | 내용 | 결과 |
|------|------|------|
| Stage 0 | 파일럿 200경기 검증 | 팀 코드·스키마·WPA 정합성 확인 |
| Stage 1 | 이닝 완전성 보강 | gap 4건 복구, 기존 153경기 안전 확인 |
| Stage 2 | concurrency 실측 | 8/12/16 전부 통과 → 16 채택 |
| Stage 3 | 성능 격차 규명 | 파일럿 스로틀링 확정 |
| Stage 4 | 중단 경기 정책 | suspended 0건, called 38건 보정 |
| Stage 5 | 전 연도 크롤링 | 2025→2016 역순, 실패 0건 |
| Stage 6 | UNK 정규식 수정 | 4.1% → 0.31% |
| Stage 7 | 터미널 PA 보정 | 3,380경기(52%) 수정 |
| Stage 8 | 연장 정책 적용 | 505경기 연장분 제외 |
| Stage 9 | 병합 · enrichment | 558,064 × 68 최종 통합 |

### 확장 과정에서 발견·수정된 버그

**① UNK 정규식 커버리지 누락**

네이버 중계 텍스트의 표현 변형을 정규식이 놓쳐 정상 타석이 UNK로 분류됨.

| 실제 유형 | 놓친 표현 | 처리 |
|---|---|---|
| HBP | "몸에 맞는 볼" (패턴은 "공/사구") | → BB |
| 내야안타 | "OO수 왼쪽/앞 내야안타" | → 1B |
| 실책 출루 | "~실책으로 출루" | → 1B (ROE) |
| IBB | "고의4구" (패턴은 "고의사구") | → BB |
| 번트안타 | "번트안타" | → 1B |
| 야수선택 | "야수선택으로 출루" | → 1B |

야수선택 선행주자 아웃률 0.7%(145건 중 1건) 실측 후 1B 병합 결정.
원본 라벨은 `pa_result_raw`에 보존.

**② 터미널 PA 득점 누락**

경기 마지막 PA에서 `_estimate_runs_from_pa_result()`가 HR/SF/만루BB 외에는
항상 runs=0을 반환. 314건(4.9%)에서 득점 누락, 그중 306건이 실제 끝내기.
288건은 보정 전 WPA가 음수였음(결승타인데 승률이 떨어지는 모순).

schedule API 최종 스코어를 ground truth로 사용해 확정 처리.

**③ 연장 끝내기 미처리**

기존 로직이 9회말 끝내기만 특례 처리해, 연장(10~12회) 끝내기 140건은
애초에 잡히지 않았음. 터미널 PA 보정으로 함께 해결.

---

## 3. 데이터 컨벤션 (변경 금지)

이 값들은 박스스코어·선발 라인업 등 **외부 진실 대조로 확정**된 것입니다.
데이터 내부 추론으로 재판정하지 마십시오.

| 항목 | 규칙 |
|------|------|
| `home_or_away` | **0 = 원정 공격(초/top)**, **1 = 홈 공격(말/bot)** |
| `half` | `"top"` = 초, `"bot"` = 말 |
| `score_diff` | **home − away** (H1 컨벤션) |
| `score_diff_attacker` | **이미 공격팀 관점으로 변환 완료.** 추가 부호 반전 금지 |
| 수비팀 식별 | `half="top"` → 수비는 홈팀 / `half="bot"` → 수비는 원정팀 |
| 득점 반영 | score_diff는 **1 PA 지연**. 득점이 다음 타석부터 반영됨 (의도된 설계) |
| `reward_wpa_computed` | **공격팀 관점** ΔWE. 투수 관점은 부호 반전(`-값`) |

### 팀 코드 (10개, 2016~2025 불변)

```
HH(한화)  HT(KIA)  KT(kt)   LG(LG)   LT(롯데)
NC(NC)    OB(두산)  SK(SK/SSG)  SS(삼성)  WO(넥센/키움)
```

**주의**: API가 반환하는 팀 **이름** 필드는 과거 경기에도 현재 명칭을
소급 적용합니다(2016년 SK 경기도 "SSG"로 반환). **이름이 아닌 코드로만
식별**하십시오.

### game_id 형식

```
YYYYMMDD{away}{home}{seq}{year}
예: 20160317SKHH02016 → 2016-03-17, 원정 SK, 홈 HH, 1차전
```

**주의**: `game_id` 내장 날짜와 schedule API의 date 필드가
하루 어긋나는 경기가 있습니다. **`game_id` 기준을 신뢰**하십시오.
더블헤더는 `seq` 마커로 구분됩니다.

---

## 4. 적용된 정책

### 4-1. 정규시즌만 포함

시범경기 제외. 판별 기준은 `config/season_opening_dates.json`의
연도별 개막일.

```
2016: 04-01   2017: 03-31   2018: 03-24   2019: 03-23   2020: 05-05*
2021: 04-03   2022: 04-02   2023: 04-01   2024: 03-23   2025: 03-22
```
\* 2020년은 코로나로 5월 개막

**제외 이유**: 시범경기는 불펜 명단이 40~45명(정규시즌 12명 대비 3배)이라
`bullpen_available_ratio`가 구조적으로 왜곡되며, 투수 운용 목적 자체가 다름.

**검증**: 개막일 기준과 불펜 인원 기준(>20명)이 127경기에서 100% 일치.

포스트시즌·올스타전은 game_id가 특수 코드(3333/4444/9999 등)를 써서
자동 제외됩니다.

### 4-2. 연장 경기 처리

**9회말까지는 정교하게, 연장 진입은 무승부로 종료.**

| 상황 | 처리 |
|------|------|
| 9회말 끝내기 (홈팀 역전승) | `we_after = 1.0` (공격팀 기준) |
| 9회말 종료 시점 동점 | `we_after = 0.5`로 확정, 경기 종료 처리 |
| 10회 이후 PA | **데이터셋에서 제외** (`is_extra_innings` 플래그) |

**근거**: 연장 진입은 정의상 동점이므로 승률 0.5는 근사가 아닌 정확한 값.

**보존한 것**: 9회말 동점 상황의 WE는 논문 테이블 값을 그대로 사용합니다.
1사 만루(0.729)와 2사 무주자(0.56)의 차이가 살아있습니다.
이 구간의 |ΔWE| 평균이 전체의 **1.85배**로 가장 중요한 승부처이기 때문입니다.

**규모**: 9개년 505경기(7.8%), 8,464 PA 제외 + 2016년 61경기 880 PA

### 4-3. 콜드게임 (called) 처리

5회 이상 진행 후 우천 등으로 성립된 경기. 총 38건.

마지막 PA의 `we_after`를 경기 확정 결과로 강제:
- 승패 확정 → 1.0 또는 0.0
- 무승부 → 0.5

**보정 전 문제**: 6:5 접전 콜드게임에서 오차 0.293,
일부는 `reward_wpa_computed`가 음수(결승 플레이가 승률을 깎는 모순)

### 4-4. 터미널 PA 확정

경기 마지막 PA의 득점과 `we_after`를 schedule API 최종 스코어로 확정.
추정 로직은 fallback으로만 사용.

**적용 규모**: 6,477경기 중 3,380경기(52%)에서 실질 변경

### 4-5. 시점 안전(leakage-free) 통계

`prior_*` 컬럼은 `expanding().shift(1)`로 계산되어 **현재 등판을 제외한
과거만** 반영합니다.

```
정렬 키: (pitcher_id, date, game_id)
date = game_id 앞 8자리
```

**절대 재계산하지 마십시오.** 시퀀스가 바뀌면 leakage가 발생합니다.

preview API의 시즌 통계도 `generate_date`가 경기 **전날** 기준임을
전 경기에서 확인했습니다(256/256, 정확히 1일 차이).

---

## 5. 컬럼 명세 (68열)

### 5-1. 식별자 · 게임 상황

| 컬럼 | 타입 | 범위/값 | 설명 |
|------|------|---------|------|
| `game_id` | str | — | 경기 식별자 |
| `_orig_idx` | int | — | 경기 내 PA 순서. **병합 시 전역 재채번됨** (아래 7장 참고) |
| `inning` | int | 1~9 | 이닝 (연장 제외) |
| `half` | str | top/bot | 초/말 |
| `home_or_away` | int | 0/1 | **0=원정공격, 1=홈공격** |
| `out_count` | int | 0~2 | 아웃 카운트 |
| `is_base1` / `is_base2` / `is_base3` | bool | — | 주자 유무 (개별) |
| `base_state` | str | 0,1,2,3,12,13,23,123 | 주자 상태 (통합) |
| `score_diff` | int | — | **home − away** |
| `score_diff_attacker` | int | — | **공격팀 관점. 재변환 금지** |

### 5-2. 투수 · 타자

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pitcher_id` | str | 투수 ID (네이버 playerCode) |
| `batter_id` | str | 타자 ID |
| `pitcher_throws` | str | L/R/U. **결측 13.66%** (preview 없는 구간) |
| `batter_hit_type` | str | 타자 좌우 |
| `pitcher_vs_batter_avg` | float | 투타 상대 타율 |
| `batter_recent_avg` | float | 타자 최근 타율 |
| `total_pitch_count` | int | 투수 누적 투구 수 |
| `inning_pitch_count` | int | 이닝 내 투구 수 |
| `pitches_per_pa` | int | 해당 타석 투구 수 |
| `pa_avg_pitch_speed` | float | 타석 평균 구속 |

### 5-3. 타석 결과 · WPA

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pa_result` | str | **9클래스** (1B/2B/3B/HR/BB/SO/OUT/GDP/SF) + 잔여 UNK 0.3% |
| `pa_result_raw` | str | 병합 전 세분 라벨 (HBP, IBB, ROE 등) |
| `runs_scored` | int | 해당 타석 득점 |
| `we_before` / `we_after` | float | 타석 전/후 승리확률 |
| `inning_ended` | bool | 이닝 종료 여부 |
| **`reward_wpa_computed`** | float | **ΔWE (공격팀 관점) — 학습 타겟** |
| `reward_wpa` | float | 네이버 원본. **결측 79.97%, 사용 금지** |
| `data_quality_flag` | str | 품질 플래그 (빈 문자열 = 정상) |
| `is_extra_innings` | bool | **전량 False** (연장 PA는 이미 제외. 감사 추적용) |

### 5-4. 등판 이력 (28열)

**해당 등판 자체의 집계** (현재 등판 포함):

| 컬럼 | 설명 |
|------|------|
| `date` | 경기일 (game_id 앞 8자리) |
| `app_wpa` | 등판 WPA 합계 (공격팀 관점) |
| `n_pa` | 등판 타석 수 |
| `n_bb`, `n_so`, `n_hr`, `n_out`, `n_1b`, `n_2b`, `n_3b`, `n_gdp`, `n_sf` | 결과별 건수 |
| `outs_recorded` | 기록한 아웃 수 |
| `innings_pitched` | 소화 이닝 (outs/3) |
| `total_pitches` | 등판 총 투구 수 |
| `start_inning`, `end_inning` | 등판 시작·종료 이닝 |
| `appearance_order` | 그 경기 내 등판 순서 (1부터) |
| `is_starter` | 선발 여부 (appearance_order == 1) |

**시점 안전(leakage-free) 누적 통계** (현재 등판 **제외**):

| 컬럼 | 설명 | 결측 조건 |
|------|------|-----------|
| `prior_n_apps` | 이전 등판 수 | 없음 (0 가능) |
| `prior_wpa_mean` | 이전 등판 WPA 평균 | n_apps = 0 |
| `prior_wpa_std` | **이전 등판 WPA 표준편차** (변동성 지표) | n_apps < 2 |
| `prior_bb_rate` | 누적 볼넷율 | n_apps = 0 |
| `prior_so_rate` | 누적 삼진율 | n_apps = 0 |
| `prior_hr_rate` | 누적 피홈런율 | n_apps = 0 |
| `prior_n_pa` | 누적 타석 수 | n_apps = 0 |
| `prior_innings` | 누적 이닝 | n_apps = 0 |
| `prior_avg_pa_per_app` | 등판당 평균 타석 | n_apps = 0 |

**표본 규모** (전 구단 확장 효과):
```
prior_n_apps 중앙값  5 →  52  (10.4배)
평균              8.08 → 80.34
10등판 이상 투수   41명 → 626명  (15.3배)
```

### 5-5. 불펜 자원 (8열)

| 컬럼 | 설명 |
|------|------|
| `n_pitchers_used` | 그 시점까지 등판한 투수 수 |
| `current_pitcher_pa_in_app` | 현재 투수의 이번 등판 소화 타석 |
| `is_pitcher_change` | 직전 PA 대비 투수 교체 여부 |
| `bullpen_listed` | 명단 등재 투수 수 |
| `bullpen_used` | 명단 중 등판한 수 |
| `bullpen_available` | 잔여 불펜 수 |
| `bullpen_available_ratio` | 잔여 비율 [0,1] |
| `bullpen_source` | **`preview`(86.3%) / `estimated`(13.7%)** |

`estimated` 구간 = 2016년 전체 + 2017-05-30 이전.
`pitcher_throws` 결측률(13.66%)과 정확히 일치하므로 두 컬럼은 함께 필터링 가능.

## 6. 사용 가이드

### 로드

```python
import pandas as pd

df = pd.read_parquet("data_analysis/results/all_pa_enriched.parquet")
# 558,064 × 68
```

### 학습 데이터 필터링 권장안

```python
# 품질 플래그 제외
df = df[df["data_quality_flag"] == ""]

# UNK 제외 (잔여 0.3%)
df = df[df["pa_result"] != "UNK"]

# 연장은 이미 제외됨 (is_extra_innings 전량 False, 별도 처리 불필요)

# preview 없는 구간만 쓰거나 빼고 싶을 때
df_preview = df[df["bullpen_source"] == "preview"]   # 86.3%
```

### 클래스 병합 방침 (기존 유지)

```python
# pa_result는 이미 9클래스로 병합되어 있음
# 원본 세분 라벨이 필요하면 pa_result_raw 사용
df["pa_result_raw"].value_counts()   # HBP, IBB, ROE 등 확인 가능
```

| 원본 | 병합 후 | 근거 |
|------|---------|------|
| HBP, IBB | BB | 출루 이벤트 동일 |
| 내야안타, 번트안타 | 1B | 타자 1루 도달 동일 |
| 실책출루, 야수선택, 타격방해 (ROE) | 1B | 선행주자 아웃률 0.7% 실측 후 병합 |

### 시간순 분할 (필수)

미래 데이터 누수를 막기 위해 **랜덤 분할 금지**. 반드시 시간순.

```python
df["_date"] = df["game_id"].str[:8].astype(int)
df = df.sort_values("_date")

n = len(df)
train = df.iloc[:int(n*0.70)]
val   = df.iloc[int(n*0.70):int(n*0.85)]
test  = df.iloc[int(n*0.85):]
```

### 투수 관점 보상 변환

```python
# reward_wpa_computed는 공격팀 관점
# 투수(수비팀) 관점은 부호 반전
pitcher_reward = -df["reward_wpa_computed"]
```

### WE 조회 (Monte Carlo / DQN)

```python
from data_analysis.methods.we_re_lookup import get_we_with_boundary

we = get_we_with_boundary(
    inning=7, half="bot", score_diff=2,
    out_count=1, base_state="13"
)
```

---

## 7. 알려진 한계 및 주의사항

| # | 항목 | 영향 | 대응 |
|---|------|------|------|
| 1 | **WE 테이블 Option A** — 논문이 1사만 제공해 0/1/2사를 모두 1사 값으로 처리 | 주자 있는 아웃에서 WE 변화가 약하게 평가됨 | RE 차이 기반 보정 (향후) |
| 2 | **이닝 보간** — 3회·7회 값에서 선형 보간 | 극단 상황 반영 부족 | 자체 데이터로 전 이닝 추정 (향후) |
| 3 | **preview 미제공 구간** — 2016 전체 + 2017-05-30 이전 (13.7%) | `bullpen_source=estimated`, `pitcher_throws=NaN` | 플래그로 구분 학습 |
| 4 | **연장 제외** — 10회 이후 PA 없음 | 연장 상황 학습 불가 | 정책적 결정 (4-2 참고) |
| 5 | **잔여 UNK 0.3%** | 미미 | 필요 시 패턴 추가 |
| 6 | **`reward_wpa` 결측 79.97%** | 네이버 원본은 2024~2025만 존재 | **사용 금지.** `reward_wpa_computed` 사용 |

### `_orig_idx` 주의 (중요)

`staging/pa_states_{year}.parquet`를 단순 concat하면 안 됩니다.
연도별 파일이 각각 0부터 시작하므로 **500,158건이 충돌**합니다.

`all_pa_enriched.parquet`은 `(game_id, _orig_idx)` 정렬 후 **전역 재채번**된
상태입니다. 향후 연도를 추가할 때도 동일하게 재채번해야 합니다.

### `is_extra_innings` 컬럼

최종 데이터에서 **전량 False**입니다. 연장 PA가 이미 제외됐기 때문입니다.
컬럼을 남긴 이유는 "이 정책이 적용됐다"는 감사 추적 표식입니다.

## 8. 검증 결과

### 8-1. 최종 검증 8항목 (전부 PASS)

| # | 항목 | 결과 |
|---|------|------|
| 1 | 행 수 일치 | 558,064 → 558,064 |
| 2 | 기존 31컬럼 전수 일치 | `assert_frame_equal` 통과 |
| 3 | leakage 없음 | 투수 30명 / 등판 2,226건 독립 재계산 일치 |
| 4 | `prior_n_apps==0` → `prior_wpa_std` NaN | 위반 0건 (829명) |
| 5 | `n_pitchers_used` 단조 증가 | 위반 0건 (14,394 그룹 전수) |
| 6 | 비율 컬럼 [0,1] 또는 NaN | 전부 정상 |
| 7 | pa_result 도메인 순서 (Tier 1) | 경고 0건 (Tier 2 포함) |
| 8 | **Telescoping** | **평균오차 5×10⁻⁶, 오차≥0.1 게임 0건** |

### 8-2. Telescoping 검증에 대해

WPA는 telescoping 성질을 가지므로, 한 경기의 모든 ΔWE 합은
이론적으로 `WE(최종) − WE(시작)`과 같아야 합니다.

**주의**: 단순 합산으로는 검증되지 않습니다. WE는 매 반이닝 공격팀이
바뀌며 관점이 전환되므로, **홈팀 관점으로 통일**한 뒤 계산해야 합니다.

이 실수로 1차 검증이 잘못된 신호를 냈던 이력이 있습니다:
```
잘못된 공식 (단순 합산):  오차 1.079 → 1.046  (개선 없어 보임)
올바른 공식 (관점 통일):  오차 0.077 → 0.000005
```

7,197경기 전체에서 오차 0.1 이상인 경기가 **0건**입니다.
WPA 계산이 이론적 항등식을 정확히 만족한다는 강한 증거입니다.

### 8-3. league_baseline (전 구단 기준)

| 지표 | 153경기 | 전 구단 10년 |
|------|---------|--------------|
| `league_wpa_std` | 0.1399 | **0.1851** |
| `league_bb_rate` | 0.0907 (sd 0.0451) | **0.1011** (sd 0.0432) |
| `league_so_rate` | 0.1755 (sd 0.0486) | **0.1654** (sd 0.0415) |
| `league_hr_rate` | 0.0255 (sd 0.0143) | **0.0228** (sd 0.0106) |
| 표본 투수 | 41명 | **626명** |
| 등판 수 | 1,330 | **64,571** |

세 비율 지표 모두 `sd < mean` 정상 범위. 표본이 48배 커지며 sd 안정화.

### 8-4. 크롤링 과정에서 수정된 버그

| 버그 | 규모 | 조치 |
|------|------|------|
| UNK 정규식 커버리지 누락 | 4.1% → 0.31% | 7개 표현 패턴 추가 |
| 터미널 PA 득점 누락 | 314건 (288건이 음수 WPA) | schedule API ground truth 확정 |
| 연장 끝내기 미처리 | 140건 | 터미널 PA 보정으로 해결 |
| 콜드게임 we_after 부정확 | 38건 (최대 오차 0.293) | 경기 결과로 확정 |
| `_orig_idx` 병합 충돌 | 500,158건 | 전역 재채번 |

## 9. 문의

컨벤션 관련 질문은 **3장을 먼저 확인**하십시오.
데이터를 보고 컨벤션을 역추론하지 마십시오 — 과거 그 방식으로
다섯 번 결론이 뒤집힌 이력이 있습니다.

정책 변경이 필요하면 이 문서를 함께 갱신하십시오.
