"""
Leverage Index(LI) 계산 모듈 — 문형우 외(2016) WE 테이블 기반, 결정론적 계산.

배경: 기존 LI(몬테카를로 std_wpa)는 두 결함이 있었다.
  1. "남은 이닝 수"를 잼 (5회 7점차 > 9회 1점차로 잘못 나옴)
  2. 투수/타자 ID에 좌우됨 (같은 상황인데 std가 요동침)
LI는 "상황"만의 함수여야 하므로, 시뮬레이션 대신 아래 결정론적 수식을 쓴다.

    LI_raw(s) = sqrt( Σ_r P_league(r) * [WE(s→r) - WE(s)]^2 )

  P_league(r): 리그 평균 pa_result 분포. league_pa_distribution.json 참조
               (out_count x base_state 24버킷 조건부 분포 — build_league_pa_distribution.py
               에서 이닝 조건화는 TVD 0.0087로 무의미해 기각, 아웃x주자 조합만 채택).
  WE 조회는 반드시 get_we_with_out_correction (Option B) 사용.
  get_we_with_boundary(Option A)는 아웃 차원을 반영하지 않아 "9회말 만루
  무사/1사/2사 LI가 달라야 한다"는 도메인 검증(Task 4)을 통과할 수 없다.

주의 — 관점 통일:
  WE는 half마다 공수가 뒤집힌다. we_before/we_after를 각각 "상태 s의 공격팀"
  관점으로 구한 뒤, 반드시 홈팀 관점(to_home_perspective)으로 통일하고 나서
  차이를 내야 한다 (HANDOFF_FULL.md §3-3 함정 3 — telescoping 오염과 동일 원인).

주의 — 9회말 경계 조건:
  이 데이터셋은 연장(10회 이후) PA가 없다(HANDOFF_FULL.md §4-2, is_extra_innings
  전량 False). 9회말이 종료되는 순간의 WE는 다음 세 값 중 하나로 고정된다:
    - 새 점수차(공격팀=홈팀 관점) > 0  → 1.0  (끝내기)
    - 새 점수차 < 0                    → 0.0  (원정팀 정규이닝 승)
    - 새 점수차 == 0                   → 0.5  (연장 진입, 동점 확정)
  get_we_with_out_correction 자체는 "9회말 도중 홈팀이 역전"(>0) 경우만 자체
  처리하므로, 나머지 두 경우(<0, ==0)는 이 모듈에서 직접 분기한다. 이 로직은
  monte_carlo_simulate.py의 시뮬레이션 루프가 쓰는 것과 동일한 패턴이며,
  KBO_Crawler/handoff/code/state_transition.py의 _compute_we_after (원본 데이터
  생성 로직)와 동일한 정책을 따른다.

상태 전이(apply_pa_result)와 관점 변환(to_home_perspective)은
monte_carlo_simulate.py에서 그대로 복사했다 (그 파일은 catboost를 모듈 최상단에서
import하므로, catboost 없이도 동작해야 하는 이 모듈에서 직접 import할 수 없다).
원본이 바뀌면 이 사본도 함께 갱신해야 한다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from data_analysis.methods.we_re_lookup import get_we_with_out_correction

Half = Literal["top", "bot"]

PA_CLASSES: tuple[str, ...] = ("1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF")
BASE_STATES: tuple[str, ...] = ("0", "1", "2", "3", "12", "13", "23", "123")

LEAGUE_DIST_PATH = Path("data_analysis/results/league_pa_distribution.json")


# ─────────────────────────────────────────────────────────────────────────────
#  monte_carlo_simulate.py에서 복사 — 상태 전이 / 관점 변환 (원본 그대로 유지)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _State:
    inning: int
    half: str
    out_count: int
    base_state: str
    score_diff_attacker: int


_BASE_STR_TO_BITS = {
    "0": 0b000, "1": 0b001, "2": 0b010, "3": 0b100,
    "12": 0b011, "13": 0b101, "23": 0b110, "123": 0b111,
}
_BITS_TO_BASE_STR = {v: k for k, v in _BASE_STR_TO_BITS.items()}


def _b(s: str) -> int:
    return _BASE_STR_TO_BITS.get(s, 0)


def _s(b: int) -> str:
    return _BITS_TO_BASE_STR.get(b & 0b111, "0")


def apply_pa_result(state: _State, pa_result: str) -> tuple[_State, int, bool]:
    """PA 결과 적용 → (다음 상태, 득점, 이닝종료여부). monte_carlo_simulate.py와 동일."""
    bits = _b(state.base_state)
    runs = 0
    new_out = state.out_count
    new_bits = bits

    if pa_result == "HR":
        runs = bin(bits).count("1") + 1
        new_bits = 0b000
    elif pa_result == "3B":
        runs = bin(bits).count("1")
        new_bits = 0b100
    elif pa_result == "2B":
        if bits & 0b100: runs += 1
        if bits & 0b010: runs += 1
        new_bits = 0b010
        if bits & 0b001: new_bits |= 0b100
    elif pa_result == "1B":
        if bits & 0b100: runs += 1
        new_bits = 0b001
        if bits & 0b010: new_bits |= 0b100
        if bits & 0b001: new_bits |= 0b010
    elif pa_result == "BB":
        if bits == 0b111:
            runs = 1
            new_bits = 0b111
        elif bits & 0b001:
            if bits & 0b010:
                new_bits = bits | 0b111
            else:
                new_bits = bits | 0b011
        else:
            new_bits = bits | 0b001
    elif pa_result in ("SO", "OUT"):
        new_out += 1
        new_bits = bits
    elif pa_result == "GDP":
        new_out += 2
        new_bits = bits & ~0b001
    elif pa_result == "SF":
        new_out += 1
        if bits & 0b100:
            runs += 1
            new_bits = bits & ~0b100
    else:
        raise ValueError(f"알 수 없는 pa_result: {pa_result!r}")

    inning_ended = new_out >= 3

    next_state = replace(
        state,
        out_count=0 if inning_ended else new_out,
        base_state="0" if inning_ended else _s(new_bits),
    )
    return next_state, runs, inning_ended


def to_home_perspective(we_attacker: float, half: str) -> float:
    """공격팀 관점 WE → 홈팀 관점 WE. monte_carlo_simulate.py와 동일."""
    return we_attacker if half == "bot" else 1.0 - we_attacker


# ─────────────────────────────────────────────────────────────────────────────
#  리그 분포 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_league_distribution(path: Path = LEAGUE_DIST_PATH) -> dict:
    """league_pa_distribution.json 로드."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_bucket_distribution(
    league_data: dict, out_count: int, base_state: str
) -> dict[str, float]:
    """(out_count, base_state) 조건부 확률 9클래스 dict 조회."""
    return league_data["conditional_distribution"]["table"][str(out_count)][base_state]


# ─────────────────────────────────────────────────────────────────────────────
#  WE 전이 (경계 조건 포함, 공격팀-of-s 관점 유지)
# ─────────────────────────────────────────────────────────────────────────────

def _we_after_attacker_perspective(
    inning: int,
    half: Half,
    out_count: int,
    base_state: str,
    score_diff_attacker: int,
    pa_result: str,
) -> float:
    """
    PA 결과 적용 후 WE를, "상태 s에서 공격 중이던 팀"의 관점으로 반환.

    이닝이 넘어가 공수가 바뀌어도 관점은 유지한다(팀 정체성 기준).
    9회말 경계 조건(끝내기/정규이닝 패배/연장 동점 확정)은 여기서 직접 처리.
    """
    state = _State(inning, half, out_count, base_state, score_diff_attacker)
    next_state, runs, inning_ended = apply_pa_result(state, pa_result)
    new_diff = score_diff_attacker + runs

    # 끝내기: 9회말 도중 홈팀이 역전/결승점 — 3아웃 여부와 무관하게 즉시 확정.
    # get_we_with_out_correction 내부에도 동일 체크가 있으나, inning_ended=True로
    # 조기 return하기 전에 여기서 먼저 걸러야 아래 분기와 충돌하지 않는다.
    if inning == 9 and half == "bot" and new_diff > 0:
        return 1.0

    if not inning_ended:
        return get_we_with_out_correction(
            inning, half, new_diff, next_state.out_count, next_state.base_state
        )

    # ── 이닝 종료 ──────────────────────────────────────────────────────────
    if inning == 9 and half == "bot":
        # 홈팀이 9회말 종료 시점까지 역전 못 함: 뒤지면 원정 정규이닝 승(0.0),
        # 동점이면 연장 진입 확정(0.5). 이 데이터셋엔 10회 이후 PA가 없으므로
        # 실제 연장 결과로 보간하지 않고 0.5로 고정한다 (HANDOFF_FULL.md §4-2).
        return 0.0 if new_diff < 0 else 0.5

    # 정상 반이닝 전환: 다음 공격팀 관점 WE를 구해 반전
    if half == "top":
        next_inning, next_half = inning, "bot"
    else:
        next_inning, next_half = inning + 1, "top"

    opponent_we = get_we_with_out_correction(next_inning, next_half, -new_diff, 0, "0")
    return 1.0 - opponent_we


# ─────────────────────────────────────────────────────────────────────────────
#  LI 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_li_raw(
    inning: int,
    half: Half,
    out_count: int,
    base_state: str,
    score_diff_attacker: int,
    league_dist: dict[str, float],
) -> float:
    """
    상황 s의 정규화 전 LI.

    LI_raw(s) = sqrt( Σ_r P_league(r) * [WE(s→r) - WE(s)]^2 )   (홈팀 관점 통일)

    league_dist: get_bucket_distribution()으로 조회한, 현재 (out_count, base_state)
                 에 해당하는 9클래스 확률 dict.
    """
    we_before_attacker = get_we_with_out_correction(
        inning, half, score_diff_attacker, out_count, base_state
    )
    we_before_home = to_home_perspective(we_before_attacker, half)

    variance = 0.0
    for pa_result, prob in league_dist.items():
        if prob <= 0.0:
            continue
        we_after_attacker = _we_after_attacker_perspective(
            inning, half, out_count, base_state, score_diff_attacker, pa_result
        )
        we_after_home = to_home_perspective(we_after_attacker, half)
        variance += prob * (we_after_home - we_before_home) ** 2

    return math.sqrt(variance)


def compute_li(
    inning: int,
    half: Half,
    out_count: int,
    base_state: str,
    score_diff_attacker: int,
    league_dist: dict[str, float],
    li_league_avg: float,
) -> float:
    """LI_raw를 리그 평균으로 정규화 (Task 3의 li_league_avg 사용)."""
    return compute_li_raw(
        inning, half, out_count, base_state, score_diff_attacker, league_dist
    ) / li_league_avg


# ─────────────────────────────────────────────────────────────────────────────
#  단위 테스트 (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def _test_apply_pa_result_matches_monte_carlo() -> None:
    """복사한 apply_pa_result가 원본과 동일한 결과를 내는지 표본 확인."""
    for base, result in [("123", "GDP"), ("123", "SF"), ("0", "HR"), ("13", "1B"), ("1", "BB")]:
        state = _State(5, "top", 1, base, 0)
        next_state, runs, ended = apply_pa_result(state, result)
        assert isinstance(runs, int) and runs >= 0
    print("[PASS] test_apply_pa_result_matches_monte_carlo — 표본 케이스 정상 동작")


def _test_walkoff_boundary() -> None:
    """9회말 도중 역전 → WE=1.0. (동점(0)에서 솔로 홈런 → +1로 역전)"""
    we = _we_after_attacker_perspective(9, "bot", 1, "0", 0, "HR")
    assert we == 1.0, f"끝내기 홈런인데 WE={we}"
    print("[PASS] test_walkoff_boundary — 9회말 끝내기 홈런 WE=1.0")


def _test_extra_innings_tie_boundary() -> None:
    """9회말 마지막 아웃, 동점 유지 → WE=0.5."""
    we = _we_after_attacker_perspective(9, "bot", 2, "0", 0, "OUT")
    assert we == 0.5, f"연장 진입 동점인데 WE={we}"
    print("[PASS] test_extra_innings_tie_boundary — 9회말 종료 동점 WE=0.5")


def _test_regulation_away_win_boundary() -> None:
    """9회말 마지막 아웃, 홈팀 뒤짐 → WE=0.0."""
    we = _we_after_attacker_perspective(9, "bot", 2, "0", -1, "OUT")
    assert we == 0.0, f"원정 정규이닝 승인데 WE={we}"
    print("[PASS] test_regulation_away_win_boundary — 9회말 종료 원정 승 WE=0.0")


def _test_li_raw_nonnegative_and_finite() -> None:
    """LI_raw는 항상 0 이상, 유한값."""
    league_data = load_league_distribution()
    for out_count in (0, 1, 2):
        for base_state in BASE_STATES:
            dist = get_bucket_distribution(league_data, out_count, base_state)
            li = compute_li_raw(5, "bot", out_count, base_state, 0, dist)
            assert li >= 0.0 and math.isfinite(li), (
                f"LI_raw({out_count},{base_state}) = {li} 비정상"
            )
    print("[PASS] test_li_raw_nonnegative_and_finite — 전 (아웃,주자) 조합 확인")


def _test_li_zero_when_impossible_bucket_has_low_variance() -> None:
    """무사 주자없음(가장 흔한 상황)의 LI가 극단 상황보다 작아야 함 (개략적 확인)."""
    league_data = load_league_distribution()
    dist_common = get_bucket_distribution(league_data, 0, "0")
    dist_extreme = get_bucket_distribution(league_data, 1, "123")
    li_common = compute_li_raw(5, "bot", 0, "0", 0, dist_common)
    li_extreme = compute_li_raw(9, "bot", 1, "123", 0, dist_extreme)
    assert li_common < li_extreme, (
        f"5회 무사무주자 LI({li_common:.4f}) >= 9회말 1사만루 동점 LI({li_extreme:.4f})"
    )
    print("[PASS] test_li_zero_when_impossible_bucket_has_low_variance — 개략적 크기 비교")


if __name__ == "__main__":
    print("=" * 55)
    print("Leverage Index 모듈 sanity check")
    print("=" * 55)

    _test_apply_pa_result_matches_monte_carlo()
    _test_walkoff_boundary()
    _test_extra_innings_tie_boundary()
    _test_regulation_away_win_boundary()
    _test_li_raw_nonnegative_and_finite()
    _test_li_zero_when_impossible_bucket_has_low_variance()

    print("=" * 55)
    print("모든 테스트 통과!")
    print("=" * 55)

    league_data = load_league_distribution()
    print("\n[LI_raw 대표값] (정규화 전, Task 3에서 리그 평균으로 나눌 예정)")
    samples = [
        (9, "bot", 2, "123", 0, "9회말 2사 만루 동점"),
        (9, "bot", 1, "12", -1, "9회말 1사 1,2루 1점차(홈 추격)"),
        (8, "bot", 1, "2", 0, "8회말 1사 2루 동점"),
        (5, "top", 0, "0", 0, "5회초 무사 무주자 동점"),
        (7, "bot", 2, "0", 5, "7회말 2사 무주자 5점차"),
        (9, "bot", 2, "0", 7, "9회말 2사 무주자 7점차"),
    ]
    for inning, half, out_count, base_state, sd, label in samples:
        dist = get_bucket_distribution(league_data, out_count, base_state)
        li = compute_li_raw(inning, half, out_count, base_state, sd, dist)
        print(f"  {label:28s} LI_raw = {li:.4f}")
