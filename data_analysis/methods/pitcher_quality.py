"""
투수 품질 지표(quality_p) 계산 모듈 — 최종 확정 (Task 5).

배경: prior_wpa_mean(옛 실력 지표)은 자기상관 r=0.83으로 높아 보였으나,
선발/불펜 역할 고정 효과를 실력으로 착각한 착시였다(역할 통제 시 불펜
0.11 / 선발 -0.07로 붕괴). DIPS 이론에 따라 삼진/볼넷/피홈런 비율을
대안으로 재검증했다.

3층 구조:
    Layer 1 (실력)   = prior_so_rate - prior_bb_rate - 2*prior_hr_rate
    Layer 2 (신뢰도) = w = prior_n_apps / (prior_n_apps + k),  k=5
    Layer 3 (역할)   = quality_p 계산에 포함하지 않음. 선발/불펜을
                       완전히 분리된 두 개의 기준분포로 다룬다.

파이프라인 (Task 1~4에서 각각 확정된 결정을 그대로 이어붙인 것):
    1. raw = prior_so_rate - prior_bb_rate - 2*prior_hr_rate      [Task 1-2]
       Spearman 자기상관(역할분리) so_rate 0.71/0.46, bb_rate 0.66/0.46,
       composite 선발 0.740(최고), 예측력(다음 등판 실제성과) 선발 r=0.153
    2. shrunk = w*raw + (1-w)*league_mean_composite, k=5           [Task 4]
       k=5,10,15,20 중 최적 (r 0.1527→0.1568)
    3. quality_p = 2 * percentile_rank(shrunk, 기준분포)            [Task 3]
       평균 1.0, 범위 대략 0~2 — leverage_index(평균 1.0)와 자릿수 일치.
       한계: 백분위 기반이라 실력 격차의 '크기' 정보가 압축된다
       (LI처럼 최댓값이 9.8까지 벌어지지 못하고 2.0에서 막힘).

**중요: 이 모듈은 선발(is_starter=True)에만 검증되어 있다.**
불펜은 Task 2에서 어떤 후보도 예측력 기준(r>0.1, p<0.05)을 명확히
통과하지 못했다(최고 wpa_per_pa r=0.105로 간신히 걸침). 원인은 불펜
등판이 타자 2~3명 수준으로 짧아 개별 등판 결과의 노이즈 비중이 크기
때문으로 추정된다. 불펜 quality_p 공식은 팀 논의 후 별도 확정 예정이며,
그 전까지 compute_quality_p()는 불펜 입력에 대해 NotImplementedError를
낸다 — 검증 안 된 숫자를 검증된 것처럼 반환하지 않기 위함이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASELINE_PATH = Path("data_analysis/results/quality_baseline.json")

_baseline_cache: dict | None = None


def load_quality_baseline(path: Path = BASELINE_PATH) -> dict:
    global _baseline_cache
    if _baseline_cache is None:
        with open(path, encoding="utf-8") as f:
            _baseline_cache = json.load(f)
    return _baseline_cache


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
    prior_so_rate: float,
    prior_bb_rate: float,
    prior_hr_rate: float,
    prior_n_apps: float,
    is_starter: bool,
    baseline: dict | None = None,
) -> float:
    """
    투수 품질 지표 quality_p 계산 (평균 1.0, 범위 대략 0~2).

    Parameters
    ----------
    prior_so_rate, prior_bb_rate, prior_hr_rate : 시점 안전 누적 비율
        (all_pitcher_history.parquet의 prior_* 컬럼, 재계산 금지 — 그대로 사용)
    prior_n_apps : 이전 등판 수
    is_starter : 선발 여부. False(불펜)면 NotImplementedError.
    baseline : load_quality_baseline() 결과. None이면 자동 로드.

    Raises
    ------
    NotImplementedError
        불펜(is_starter=False) 입력 — 아직 검증된 공식이 없음.
    """
    if not is_starter:
        raise NotImplementedError(
            "불펜 quality_p는 아직 확정되지 않았다 (Task 2 예측력 검증 미통과). "
            "팀 논의 후 별도 기준분포/공식을 확정할 것."
        )
    if prior_n_apps < 1:
        raise ValueError("prior_n_apps < 1: 이전 등판 기록이 없는 투수는 quality_p를 정의할 수 없다.")

    baseline = baseline or load_quality_baseline()

    raw = compute_raw_composite(prior_so_rate, prior_bb_rate, prior_hr_rate)
    shrunk = apply_shrinkage(
        raw, prior_n_apps, baseline["league_mean_composite"], baseline["shrinkage_k"]
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


def _test_bullpen_not_implemented() -> None:
    """불펜 입력은 명시적으로 막혀 있어야 함 (검증 안 된 숫자 반환 금지)."""
    try:
        compute_quality_p(0.2, 0.08, 0.02, 50, is_starter=False)
        raise AssertionError("불펜 입력인데 예외가 안 났음")
    except NotImplementedError:
        print("[PASS] test_bullpen_not_implemented — 불펜은 의도대로 차단됨")


if __name__ == "__main__":
    print("=" * 55)
    print("pitcher_quality 모듈 sanity check")
    print("=" * 55)

    _test_average_pitcher_near_one()
    _test_range_bounds()
    _test_shrinkage_pulls_small_sample_toward_average()
    _test_bullpen_not_implemented()

    print("=" * 55)
    print("모든 테스트 통과!")
    print("=" * 55)

    baseline = load_quality_baseline()
    print("\n[quality_p 대표값] (선발, 등판 100회 기준)")
    # 리그 평균: so_rate 0.166, bb_rate 0.099, hr_rate 0.023 (all_pitcher_history 실측)
    samples = [
        (0.30, 0.05, 0.01, "에이스급 (삼진多/볼넷少/피홈런少)"),
        (0.166, 0.099, 0.023, "리그 평균급"),
        (0.10, 0.15, 0.04, "부진한 투수 (삼진少/볼넷多/피홈런多)"),
    ]
    for so, bb, hr, label in samples:
        q = compute_quality_p(so, bb, hr, prior_n_apps=100, is_starter=True, baseline=baseline)
        print(f"  {label:35s} quality_p = {q:.3f}")
