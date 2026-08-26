"""
apply_pa_result() 상태 전이 단위 테스트

실행: python tests/test_state_transition.py (레포 루트 기준)
요구사항: (monte_carlo_simulate.py와 동일) pip install catboost pandas numpy pyarrow

Task 2 요구사항: BB 8케이스 전수 검증 + 다른 클래스 주요 케이스.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monte_carlo_simulate import apply_pa_result, GameState

BASE_KWARGS = dict(inning=5, half="top", out_count=0, score_diff_attacker=0)


def make_state(base_state: str, out_count: int = 0) -> GameState:
    kwargs = dict(BASE_KWARGS)
    kwargs["out_count"] = out_count
    kwargs["base_state"] = base_state
    return GameState(**kwargs)


def check(label: str, state: GameState, pa_result: str,
          expect_base: str, expect_runs: int, expect_out_count: int | None = None):
    next_state, runs, inning_ended = apply_pa_result(state, pa_result)
    exp_ended = (expect_out_count is not None and expect_out_count >= 3)
    actual_base = "0" if exp_ended != inning_ended else next_state.base_state
    # 이닝 종료 시 base_state/out_count는 "0"/0으로 리셋되므로 별도 비교
    ok = True
    problems = []

    if inning_ended != exp_ended:
        ok = False
        problems.append(f"inning_ended={inning_ended} (기대={exp_ended})")

    if not inning_ended:
        if next_state.base_state != expect_base:
            ok = False
            problems.append(f"base={next_state.base_state} (기대={expect_base})")
        if expect_out_count is not None and next_state.out_count != expect_out_count:
            ok = False
            problems.append(f"out_count={next_state.out_count} (기대={expect_out_count})")

    if runs != expect_runs:
        ok = False
        problems.append(f"runs={runs} (기대={expect_runs})")

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}")
    if not ok:
        raise AssertionError(f"{label}: " + "; ".join(problems))


# ──────────────────────────────────────────────
# BB — 8가지 주자 상태 전수 검증 (Task 2 핵심 요구사항)
# 원칙: 볼넷은 연결된 주자만 밀려난다. 1루가 비어있으면
#       2루/3루 주자는 움직이지 않는다.
# ──────────────────────────────────────────────
def test_bb_all_cases():
    check("BB 주자없음(0) → 1",     make_state("0"),   "BB", "1",   0)
    check("BB 1루(1) → 12",        make_state("1"),   "BB", "12",  0)
    check("BB 2루(2) → 12",        make_state("2"),   "BB", "12",  0)
    check("BB 3루(3) → 13",        make_state("3"),   "BB", "13",  0)
    check("BB 1·2루(12) → 123",    make_state("12"),  "BB", "123", 0)
    check("BB 1·3루(13) → 123",    make_state("13"),  "BB", "123", 0)
    check("BB 2·3루(23) → 123",    make_state("23"),  "BB", "123", 0)
    check("BB 만루(123) → 123 밀어내기1점", make_state("123"), "BB", "123", 1)


# ──────────────────────────────────────────────
# 1B / 2B / 3B / HR — 주요 케이스
# ──────────────────────────────────────────────
def test_1b():
    check("1B 주자없음 → 1",        make_state("0"),  "1B", "1",   0)
    check("1B 1루만 → 12",         make_state("1"),  "1B", "12",  0)
    check("1B 3루만 → 1, 1점",      make_state("3"),  "1B", "1",   1)
    check("1B 만루 → 123, 1점",     make_state("123"), "1B", "123", 1)


def test_2b():
    check("2B 주자없음 → 2",        make_state("0"),  "2B", "2",   0)
    check("2B 1루만 → 23",         make_state("1"),  "2B", "23",  0)
    check("2B 2·3루 → 2, 2점",      make_state("23"), "2B", "2",   2)
    check("2B 만루 → 23, 2점",      make_state("123"), "2B", "23",  2)


def test_3b():
    check("3B 주자없음 → 3",        make_state("0"),  "3B", "3",   0)
    check("3B 1루만 → 3, 1점",      make_state("1"),  "3B", "3",   1)
    check("3B 만루 → 3, 3점",       make_state("123"), "3B", "3",   3)


def test_hr():
    check("HR 주자없음 → 0, 1점",    make_state("0"),  "HR", "0",   1)
    check("HR 1루만 → 0, 2점",      make_state("1"),  "HR", "0",   2)
    check("HR 만루(그랜드슬램) → 0, 4점", make_state("123"), "HR", "0", 4)


# ──────────────────────────────────────────────
# SO / OUT — 주자 변화 없음, 아웃만 증가
# ──────────────────────────────────────────────
def test_so_out():
    check("SO 1루 유지, 아웃+1",     make_state("1", out_count=0), "SO",  "1", 0, expect_out_count=1)
    check("OUT 만루 유지, 아웃+1",   make_state("123", out_count=1), "OUT", "123", 0, expect_out_count=2)
    check("SO 2아웃 → 이닝종료",     make_state("1", out_count=2), "SO",  "0", 0, expect_out_count=3)


# ──────────────────────────────────────────────
# GDP — 병살: 아웃 +2, 1루 주자 제거
# ──────────────────────────────────────────────
def test_gdp():
    check("GDP 1루만 → 주자없음, 아웃+2", make_state("1", out_count=0), "GDP", "0", 0, expect_out_count=2)
    check("GDP 1·3루 → 3루만, 아웃+2",   make_state("13", out_count=0), "GDP", "3", 0, expect_out_count=2)
    check("GDP 1아웃 만루 → 2·3루, 아웃+2 → 이닝종료", make_state("123", out_count=1), "GDP", "0", 0, expect_out_count=3)


# ──────────────────────────────────────────────
# SF — 희생플라이: 아웃 +1, 3루 주자만 득점
# ──────────────────────────────────────────────
def test_sf():
    check("SF 3루만 → 주자없음, 1점, 아웃+1", make_state("3", out_count=0), "SF", "0", 1, expect_out_count=1)
    check("SF 1루만(3루無) → 1루 유지, 0점, 아웃+1", make_state("1", out_count=0), "SF", "1", 0, expect_out_count=1)
    check("SF 1·3루 → 1루만, 1점, 아웃+1", make_state("13", out_count=0), "SF", "1", 1, expect_out_count=1)
    check("SF 2아웃 3루 → 이닝종료, 1점 인정", make_state("3", out_count=2), "SF", "0", 1, expect_out_count=3)


if __name__ == "__main__":
    tests = [
        test_bb_all_cases,
        test_1b,
        test_2b,
        test_3b,
        test_hr,
        test_so_out,
        test_gdp,
        test_sf,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()

    print("\n" + "=" * 50)
    print("✅ 전체 통과!")
    print("=" * 50)
