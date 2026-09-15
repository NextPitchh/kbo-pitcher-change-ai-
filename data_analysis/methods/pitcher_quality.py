"""
투수 품질 지표(quality_p) 계산 모듈 — 선발/불펜 모두 확정 (Task 5, Task 9).

배경: prior_wpa_mean(옛 실력 지표)은 자기상관 r=0.83으로 높아 보였으나,
선발/불펜 역할 고정 효과를 실력으로 착각한 착시였다(역할 통제 시 불펜
0.11 / 선발 -0.07로 붕괴). DIPS 이론에 따라 삼진/볼넷/피홈런 비율을
대안으로 재검증했다.

선발과 불펜은 원인 진단부터 공식, shrinkage 단위까지 전부 다르다 —
같은 파이프라인에 k만 다르게 넣는 게 아니라 두 개의 독립된 공식이다.

**선발** (Task 1~5):
    1. raw = prior_so_rate - prior_bb_rate - 2*prior_hr_rate      [Task 1-2]
       Spearman 자기상관(역할분리) so_rate 0.71/0.46, bb_rate 0.66/0.46,
       composite 선발 0.740(최고), 예측력(다음 등판 실제성과) 선발 r=0.153
    2. shrunk = w*raw + (1-w)*league_mean_composite,               [Task 4]
       w = prior_n_apps/(prior_n_apps+k), k=5
    3. quality_p = 2 * percentile_rank(shrunk, 선발 기준분포)        [Task 3]

**불펜** (Task 6~9): Task 2에서 composite가 예측력 기준(r>0.1, p<0.05)을
통과 못했다(불펜 등판은 평균 4.56타자로 선발 22.9타자보다 훨씬 짧아,
"다음 등판 1번"이라는 목표 자체가 표본이 너무 작았음이 원인으로 진단됨).
목표를 "전방 5등판 평균 실제 성과"로 재정의해 재검증한 결과 prior_wpa_per_pa가
r=0.1487로 통과(Task 7). 이 상관관계가 LI(상황 긴장도)에 따른 기용 착시가
아님을 편상관으로 확인(Task 8, 0.1471→0.1458, 0.9%만 감소). shrinkage
표본 단위도 등판 수 대신 누적 타석 수(prior_n_pa)를 쓴다 — 불펜은 등판당
상대 타자 수 편차가 크기 때문(Task 6에서 이 단위 교체만으로는 부족함을
확인했지만, Task 7의 목표 재정의와 결합하면 유효함).
    1. raw = prior_wpa_per_pa (= -prior_wpa_sum/prior_n_pa)         [Task 7]
    2. shrunk = w*raw + (1-w)*league_mean_wpa_per_pa,               [Task 9]
       w = prior_n_pa/(prior_n_pa+k), k=25
    3. quality_p = 2 * percentile_rank(shrunk, 불펜 기준분포)

두 역할 모두 quality_p는 평균 1.0, 범위 대략 0~2 — leverage_index(평균 1.0)와
자릿수를 맞추기 위함. 한계: 백분위 기반이라 실력 격차의 '크기' 정보가
압축된다(LI처럼 최댓값이 9.8까지 벌어지지 못하고 2.0에서 막힘).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASELINE_PATH_STARTER = Path("data_analysis/results/quality_baseline.json")
BASELINE_PATH_BULLPEN = Path("data_analysis/results/quality_baseline_bullpen.json")

_baseline_cache: dict[str, dict] = {}


def load_quality_baseline(path: Path = BASELINE_PATH_STARTER) -> dict:
    key = str(path)
    if key not in _baseline_cache:
        with open(path, encoding="utf-8") as f:
            _baseline_cache[key] = json.load(f)
    return _baseline_cache[key]


def compute_raw_composite(
    prior_so_rate: float, prior_bb_rate: float, prior_hr_rate: float
) -> float:
    """Layer 1: DIPS 기반 실력 원값. 삼진 +1, 볼넷 -1, 피홈런 -2 가중."""
    return prior_so_rate - prior_bb_rate - 2 * prior_hr_rate


def apply_shrinkage(
    raw: float, prior_n_apps: float, league_mean: float, k: int
) -> float:
    """Layer 2: 표본 크기에 따라 리그 평균 쪽으로 당김. w = n/(n+k)."""
    w = prior_n_apps / (prior_n_apps + k)
    return w * raw + (1 - w) * league_mean


def _percentile_lookup(value: float, grid_pct: list[float], grid_values: list[float]) -> float:
    """
    value가 기준분포에서 몇 백분위에 해당하는지 선형보간으로 조회.
    grid_values는 오름차순 정렬되어 있음(np.percentile 결과이므로 보장됨).
    """
    return float(np.interp(value, grid_values, grid_pct)) / 100.0


def compute_quality_p(
    prior_so_rate: float | None = None,
    prior_bb_rate: float | None = None,
    prior_hr_rate: float | None = None,
    prior_n_apps: float | None = None,
    is_starter: bool = True,
    baseline: dict | None = None,
    prior_wpa_per_pa: float | None = None,
    prior_n_pa: float | None = None,
) -> float:
    """
    투수 품질 지표 quality_p 계산 (평균 1.0, 범위 대략 0~2).

    선발과 불펜은 서로 다른 인자를 쓴다 — 공식 자체가 다르기 때문(모듈
    docstring 참고).

    Parameters
    ----------
    prior_so_rate, prior_bb_rate, prior_hr_rate : 선발용. 시점 안전 누적 비율
        (all_pitcher_history.parquet의 prior_* 컬럼, 재계산 금지 — 그대로 사용)
    prior_n_apps : 선발용. 이전 등판 수 (shrinkage 표본 크기).
    prior_wpa_per_pa : 불펜용. -prior_wpa_sum/prior_n_pa로 유도된 값
        (all_pitcher_history.parquet의 prior_wpa_mean/prior_n_apps/prior_n_pa를
        대수적으로 조합 — prior_* 재계산 아님).
    prior_n_pa : 불펜용. 누적 상대 타석 수 (shrinkage 표본 크기).
    is_starter : 선발 여부에 따라 어떤 인자 세트를 쓸지, 어떤 기준분포를
        로드할지가 갈린다.
    baseline : load_quality_baseline() 결과. None이면 is_starter에 맞는
        기준분포(BASELINE_PATH_STARTER/BULLPEN)를 자동 로드.
    """
    if is_starter:
        if prior_n_apps is None or prior_n_apps < 1:
            raise ValueError("prior_n_apps < 1: 이전 등판 기록이 없는 투수는 quality_p를 정의할 수 없다.")
        baseline = baseline or load_quality_baseline(BASELINE_PATH_STARTER)
        raw = compute_raw_composite(prior_so_rate, prior_bb_rate, prior_hr_rate)
        shrunk = apply_shrinkage(
            raw, prior_n_apps, baseline["league_mean_composite"], baseline["shrinkage_k"]
        )
    else:
        if prior_wpa_per_pa is None or prior_n_pa is None:
            raise ValueError("불펜 quality_p는 prior_wpa_per_pa와 prior_n_pa가 모두 필요하다.")
        if prior_n_pa < 1:
            raise ValueError("prior_n_pa < 1: 이전 상대 타석 기록이 없는 투수는 quality_p를 정의할 수 없다.")
        baseline = baseline or load_quality_baseline(BASELINE_PATH_BULLPEN)
        shrunk = apply_shrinkage(
            prior_wpa_per_pa, prior_n_pa, baseline["league_mean_wpa_per_pa"], baseline["shrinkage_k"]
        )

    pct = _percentile_lookup(shrunk, baseline["percentile_grid_pct"], baseline["percentile_grid_values"])
    return 2.0 * pct


# ─────────────────────────────────────────────────────────────────────────────
#  단위 테스트 (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def _test_average_pitcher_near_one() -> None:
    """리그 평균 수준(so_rate/bb_rate/hr_rate가 딱 평균인) 베테랑 투수는 quality_p≈1.0."""
    baseline = load_quality_baseline()
    # 리그 평균 composite가 나오는 임의의 조합 하나를 역산
    league_mean = baseline["league_mean_composite"]
    so, bb, hr = 0.18, 0.08, (0.18 - 0.08 - league_mean) / 2  # so-bb-2hr = league_mean
    q = compute_quality_p(so, bb, hr, prior_n_apps=200, is_starter=True, baseline=baseline)
    assert 0.9 <= q <= 1.1, f"평균급 베테랑 quality_p={q:.3f}, 1.0 근처여야 함"
    print(f"[PASS] test_average_pitcher_near_one — quality_p={q:.3f}")


def _test_range_bounds() -> None:
    """quality_p는 항상 [0, 2] 범위."""
    baseline = load_quality_baseline()
    for so, bb, hr, n in [(0.35, 0.02, 0.0, 300), (0.05, 0.20, 0.05, 300), (0.20, 0.08, 0.02, 1)]:
        q = compute_quality_p(so, bb, hr, n, is_starter=True, baseline=baseline)
        assert 0.0 <= q <= 2.0, f"범위 위반: quality_p={q}"
    print("[PASS] test_range_bounds — 극단 입력에도 [0,2] 범위 유지")


def _test_shrinkage_pulls_small_sample_toward_average() -> None:
    """같은 raw 값이라도 등판 수가 적으면 quality_p가 리그 평균(1.0)에 더 가까워야 함."""
    baseline = load_quality_baseline()
    so, bb, hr = 0.30, 0.03, 0.0  # 매우 좋은 raw composite
    q_small_n = compute_quality_p(so, bb, hr, prior_n_apps=1, is_starter=True, baseline=baseline)
    q_large_n = compute_quality_p(so, bb, hr, prior_n_apps=300, is_starter=True, baseline=baseline)
    assert q_small_n < q_large_n, (
        f"등판 1번({q_small_n:.3f})이 등판 300번({q_large_n:.3f})보다 quality_p가 낮아야 함 "
        "(표본 적으면 평균 쪽으로 당겨지므로)"
    )
    print(
        f"[PASS] test_shrinkage_pulls_small_sample_toward_average — "
        f"n=1: {q_small_n:.3f} < n=300: {q_large_n:.3f}"
    )


def _test_bullpen_average_near_one() -> None:
    """
    불펜도 raw가 리그 평균이면 quality_p가 1.0 근처여야 함.

    단, 정확히 1.0은 아닐 수 있다 — quality_p는 평균이 아니라 백분위
    기반이라(Task 3 결정), 분포가 비대칭이면 "평균값"과 "중앙값(50th
    백분위)"이 다른 지점에 위치한다. 불펜 분포에서는 리그 평균이 약
    55.8th 백분위에 위치해 quality_p≈1.12가 나온다 — 이는 백분위
    스케일링의 알려진 한계(Task 3 문서화)이지 버그가 아니다.
    """
    baseline = load_quality_baseline(BASELINE_PATH_BULLPEN)
    league_mean = baseline["league_mean_wpa_per_pa"]
    q = compute_quality_p(
        prior_wpa_per_pa=league_mean, prior_n_pa=500, is_starter=False, baseline=baseline
    )
    assert 0.8 <= q <= 1.2, f"평균급 불펜 quality_p={q:.3f}, 1.0 근처여야 함"
    print(f"[PASS] test_bullpen_average_near_one — quality_p={q:.3f}")


def _test_bullpen_range_bounds() -> None:
    """불펜 quality_p도 항상 [0, 2] 범위."""
    baseline = load_quality_baseline(BASELINE_PATH_BULLPEN)
    for wpa_pp, n in [(0.05, 1000), (-0.05, 1000), (0.10, 5)]:
        q = compute_quality_p(prior_wpa_per_pa=wpa_pp, prior_n_pa=n, is_starter=False, baseline=baseline)
        assert 0.0 <= q <= 2.0, f"범위 위반: quality_p={q}"
    print("[PASS] test_bullpen_range_bounds — 극단 입력에도 [0,2] 범위 유지")


def _test_bullpen_shrinkage_pulls_small_sample_toward_average() -> None:
    """불펜도 표본(타석 수)이 적으면 quality_p가 리그 평균(1.0)에 더 가까워야 함."""
    baseline = load_quality_baseline(BASELINE_PATH_BULLPEN)
    wpa_pp = 0.08  # 매우 좋은 raw 값
    q_small_n = compute_quality_p(prior_wpa_per_pa=wpa_pp, prior_n_pa=5, is_starter=False, baseline=baseline)
    q_large_n = compute_quality_p(prior_wpa_per_pa=wpa_pp, prior_n_pa=3000, is_starter=False, baseline=baseline)
    assert q_small_n < q_large_n, (
        f"타석 5개({q_small_n:.3f})가 타석 3000개({q_large_n:.3f})보다 quality_p가 낮아야 함 "
        "(표본 적으면 평균 쪽으로 당겨지므로)"
    )
    print(
        f"[PASS] test_bullpen_shrinkage_pulls_small_sample_toward_average — "
        f"n_pa=5: {q_small_n:.3f} < n_pa=3000: {q_large_n:.3f}"
    )


if __name__ == "__main__":
    print("=" * 55)
    print("pitcher_quality 모듈 sanity check")
    print("=" * 55)

    _test_average_pitcher_near_one()
    _test_range_bounds()
    _test_shrinkage_pulls_small_sample_toward_average()
    _test_bullpen_average_near_one()
    _test_bullpen_range_bounds()
    _test_bullpen_shrinkage_pulls_small_sample_toward_average()

    print("=" * 55)
    print("모든 테스트 통과!")
    print("=" * 55)

    starter_baseline = load_quality_baseline(BASELINE_PATH_STARTER)
    print("\n[quality_p 대표값] (선발, 등판 100회 기준)")
    # 리그 평균: so_rate 0.166, bb_rate 0.099, hr_rate 0.023 (all_pitcher_history 실측)
    starter_samples = [
        (0.30, 0.05, 0.01, "에이스급 (삼진多/볼넷少/피홈런少)"),
        (0.166, 0.099, 0.023, "리그 평균급"),
        (0.10, 0.15, 0.04, "부진한 투수 (삼진少/볼넷多/피홈런多)"),
    ]
    for so, bb, hr, label in starter_samples:
        q = compute_quality_p(so, bb, hr, prior_n_apps=100, is_starter=True, baseline=starter_baseline)
        print(f"  {label:35s} quality_p = {q:.3f}")

    bullpen_baseline = load_quality_baseline(BASELINE_PATH_BULLPEN)
    print("\n[quality_p 대표값] (불펜, 누적 타석 500 기준)")
    # 리그 평균 wpa_per_pa ≈ -0.0097 (all_pitcher_history 실측)
    bullpen_samples = [
        (0.008, "필승조급 (99th 백분위 근처)"),
        (-0.0097, "리그 평균급"),
        (-0.020, "부진한 불펜 (1st 백분위 근처)"),
    ]
    for wpa_pp, label in bullpen_samples:
        q = compute_quality_p(
            prior_wpa_per_pa=wpa_pp, prior_n_pa=500, is_starter=False, baseline=bullpen_baseline
        )
        print(f"  {label:35s} quality_p = {q:.3f}")
