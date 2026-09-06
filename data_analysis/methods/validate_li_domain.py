"""
Leverage Index 도메인 검증 — Task 4.

지시서 표 6항목(절대 기준) + Option B 아웃차원 효과 확인 + 점수차 단조성 확인.
하나라도 크게 어긋나면 여기서 멈추고 원인을 봐야 한다 (다음 단계 진행 금지).

"9회말 1점차 1사 1,2루"처럼 어느 팀이 앞서는지 표기가 없는 항목은, 실제로
일어날 수 있는 상황(홈팀이 뒤지며 추격 중)으로 해석한다. 반대(홈팀이 이미
앞선 채 9회말을 진행)는 정의상 이미 경기가 끝났어야 하는 모순 상태라
Task 2 개발 중 실제로 확인된 바 있다 (score_diff>0 이면 즉시 WE=1.0 확정).
"""

from __future__ import annotations

import json
from pathlib import Path

from data_analysis.methods.leverage_index import (
    BASE_STATES,
    compute_li_raw,
    get_bucket_distribution,
    load_league_distribution,
)
from data_analysis.methods.we_re_lookup import get_we_with_boundary, get_we_with_out_correction

LI_BASELINE_PATH = Path("data_analysis/results/li_baseline.json")
REPORT_PATH = Path("data_analysis/results/li_domain_validation_report.md")


def _li_normalized(inning, half, out_count, base_state, sd, league_dist, li_league_avg) -> float:
    return compute_li_raw(inning, half, out_count, base_state, sd, league_dist) / li_league_avg


# ─────────────────────────────────────────────────────────────────────────────
#  Option A vs B 비교 — "9회말 만루에서 아웃별 WE가 같게/다르게 나오는가"
#
#  주의: 이건 LI 전체값이 아니라 WE(상태) 값 자체를 비교한다. LI 전체값은
#  Option A를 쓰더라도 (1) league_dist가 아웃별로 이미 다르고, (2) 병살타가
#  이닝을 끝내는지 여부가 아웃카운트에 따라 구조적으로 다르므로, WE 자체가
#  아웃 무관이어도 LI 전체는 아웃별로 달라질 수 있다 (첫 시도에서 이 점을
#  놓치고 LI 전체를 비교해 잘못된 FAIL을 냈었음 — 검증 스크립트 실수였음).
#  "Option A는 아웃 무관"이라는 HANDOFF_FULL.md의 주장은 WE(상태) 값 자체에
#  대한 것이므로, 그 수준에서 검증하는 것이 올바른 조작화다.
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    league_data = load_league_distribution()
    with LI_BASELINE_PATH.open(encoding="utf-8") as f:
        baseline = json.load(f)
    li_league_avg = baseline["li_league_avg"]

    lines: list[str] = []
    all_passed = True

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("# LI 도메인 검증 리포트 (Task 4)\n")
    log(f"li_league_avg = {li_league_avg:.6f}\n")

    # ── 1) 절대 기준 6항목 ───────────────────────────────────────────────
    log("## 1. 절대 기준 6항목\n")
    log("| 상황 | 기대 | 실제 LI | 판정 |")
    log("| --- | --- | --- | --- |")

    def dist(out_count, base_state):
        return get_bucket_distribution(league_data, out_count, base_state)

    # note: 근사 허용 사유는 표 아래에 별도로 설명 (사용자 확인 후 반영, 2026-09-06)
    cases = [
        ("9회말 동점 2사 만루", 9, "bot", 2, "123", 0, ">= 3.0", lambda v: v >= 3.0, None),
        ("9회말 1점차 1사 1,2루(홈 추격)", 9, "bot", 1, "12", -1, ">= 2.5", lambda v: v >= 2.5, None),
        ("8회말 동점 1사 2루", 8, "bot", 1, "2", 0, "2.0 내외 (1.5~2.5)", lambda v: 1.5 <= v <= 2.5, None),
        ("5회말 무사 주자없음 동점", 5, "bot", 0, "0", 0, "0.8~1.2", lambda v: 0.8 <= v <= 1.3, "approx1"),
        ("7회말 2사 무주자 5점차", 7, "bot", 2, "0", 5, "<= 0.3", lambda v: v <= 0.32, "approx2"),
        ("9회말 2사 무주자 7점차", 9, "bot", 2, "0", 7, "<= 0.1", lambda v: v <= 0.1, None),
    ]

    for label, inning, half, out_count, base_state, sd, expected_label, check, note_tag in cases:
        li = _li_normalized(inning, half, out_count, base_state, sd, dist(out_count, base_state), li_league_avg)
        ok = check(li)
        all_passed &= ok
        if ok and note_tag:
            mark = "PASS (근사 허용*)"
        else:
            mark = "PASS" if ok else "**FAIL**"
        log(f"| {label} | {expected_label} | {li:.3f} | {mark} |")

    log(
        "\n*근사 허용 사유 (2026-09-06 사용자 확인 후 허용범위 조정):\n"
        "- 5회 무사 무주자 동점(1.257): 이닝별로 찍어보면 1~3회 0.92 → 5회 1.26 → "
        "9회 2.10으로 매끄럽게 증가한다. 이는 '같은 상황이라도 후반으로 갈수록 "
        "만회 기회가 줄어 레버리지가 오른다'는 실제 세이버메트릭스 현상과 일치. "
        "li_league_avg는 1~9회 전 구간 평균이라 특정 이닝(5회) 하나가 정확히 "
        "1.0에 오라는 보장이 없으므로, 지시서의 '5회≈평균'은 근사 앵커로 해석.\n"
        "- 7회 5점차(0.315): 리드/추격 방향 모두 낮게 나오고(0.315/0.370), "
        "4~6점차가 모두 동일값(WE 테이블이 ±4점차 초과를 4점차로 취급하는 "
        "설계상 특성)으로 나와 '큰 점수차=저LI' 방향성 자체는 정상. 경계값(0.3) "
        "대비 5% 초과는 앵커 근사 오차로 판단."
    )

    # ── 2) Option B 아웃차원 효과 — LI 전체값이 아웃별로 갈리는가 ─────────
    log("\n## 2. Option B 효과 — 9회말 만루 동점, 아웃별 LI 구분\n")
    b_vals = {}
    for out_count in (0, 1, 2):
        li = _li_normalized(9, "bot", out_count, "123", 0, dist(out_count, "123"), li_league_avg)
        b_vals[out_count] = li
        log(f"- Option B, {out_count}아웃: LI = {li:.4f}")
    b_distinct = len(set(round(v, 6) for v in b_vals.values())) == 3
    all_passed &= b_distinct
    log(f"\n판정: {'PASS' if b_distinct else '**FAIL**'} (0/1/2사 LI 전체값이 서로 달라야 함)")

    # ── 2-2) WE(상태) 값 자체의 아웃 반응성 — Option A vs B ──────────────
    log("\n### WE(상태) 값 자체 비교 (같은 상태, 아웃카운트만 변경)\n")
    log("| 아웃 | Option A WE | Option B WE |")
    log("| --- | --- | --- |")
    we_a_vals, we_b_vals = {}, {}
    for out_count in (0, 1, 2):
        we_a = get_we_with_boundary(9, "bot", 0, out_count, "123")
        we_b = get_we_with_out_correction(9, "bot", 0, out_count, "123")
        we_a_vals[out_count] = we_a
        we_b_vals[out_count] = we_b
        log(f"| {out_count} | {we_a:.4f} | {we_b:.4f} |")

    a_identical = len(set(round(v, 6) for v in we_a_vals.values())) == 1
    b_we_distinct = len(set(round(v, 6) for v in we_b_vals.values())) == 3
    all_passed &= a_identical and b_we_distinct
    log(
        f"\n판정: {'PASS' if a_identical else '**FAIL**'} "
        "(Option A는 WE(상태) 값 자체가 아웃 무관하게 완전히 동일해야 함) / "
        f"{'PASS' if b_we_distinct else '**FAIL**'} (Option B는 아웃별로 달라야 함 — "
        "이게 Option B 도입 근거)"
    )

    # ── 3) 점수차 단조성 — 점수차 클수록 LI가 커지면 안 됨 ─────────────
    log("\n## 3. 점수차 단조성 (7회말 1사 무주자, 점수차 0~4)\n")
    prev = None
    monotonic_ok = True
    row = []
    for sd in range(0, 5):
        li = _li_normalized(7, "bot", 1, "0", sd, dist(1, "0"), li_league_avg)
        row.append((sd, li))
        if prev is not None and li > prev + 1e-9:
            monotonic_ok = False
        prev = li
    for sd, li in row:
        log(f"- 점수차 {sd}: LI = {li:.4f}")
    all_passed &= monotonic_ok
    log(f"\n판정: {'PASS' if monotonic_ok else '**FAIL**'} (점수차 커질수록 LI가 단조 비증가해야 함)")

    # ── 결론 ───────────────────────────────────────────────────────────
    log("\n## 결론\n")
    log(f"전체 판정: {'**PASS — Task 5 진행 가능**' if all_passed else '**FAIL — 중단, Task 2 재검토 필요**'}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] 리포트 저장: {REPORT_PATH}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
