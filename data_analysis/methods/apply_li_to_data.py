"""
전체 PA에 LI(leverage_index) 컬럼 적용 — Task 5.

LI는 (inning, half, out_count, base_state, score_diff_attacker)라는 "상태"만의
함수이고 pa_result나 data_quality_flag에 의존하지 않는다. 따라서 학습용
표준필터(품질플래그/UNK 제외) 없이 all_pa_enriched_v3.parquet의 전체
558,064행 모두에 대해 값을 계산해 붙인다 (검증 결과 상태 필드에 결측 없음
확인 완료).

정규화 분모(li_league_avg)는 Task 3에서 이미 확정한 값(li_baseline.json)을
그대로 재사용한다 — 여기서 새로 평균을 내면 안 된다. 새로 내면 "평균 상황
=1.0"이라는 기준점이 산출물마다 달라져 버려서, 이전 Task들과 일관성이
깨진다.

패턴 확인(이닝/점수차/주자/아웃별 평균)은 학습용 표준필터가 적용된 부분
집합(554,146행)으로 계산한다 — Task 1/3/4와 동일 기준이라야 비교 가능하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.methods.build_li_baseline import build_lookup_table
from data_analysis.methods.leverage_index import load_league_distribution

PARQUET_PATH = Path("data_analysis/results/all_pa_enriched_v3.parquet")
LI_BASELINE_PATH = Path("data_analysis/results/li_baseline.json")
OUT_PARQUET_PATH = Path("data_analysis/results/all_pa_with_li.parquet")
REPORT_PATH = Path("data_analysis/results/li_pattern_report.md")


def main() -> None:
    league_data = load_league_distribution()
    with LI_BASELINE_PATH.open(encoding="utf-8") as f:
        li_league_avg = json.load(f)["li_league_avg"]

    print("[1/4] LI_raw 룩업 테이블 계산 중...")
    lookup = build_lookup_table(league_data)

    print("[2/4] 전체 PA 데이터 로드 (94열 전체, 필터링 없음)...")
    df = pd.read_parquet(PARQUET_PATH)
    n_total = len(df)
    n_cols_before = df.shape[1]
    print(f"      {n_total}행 x {n_cols_before}열")

    print("[3/4] leverage_index 컬럼 계산 및 부착...")
    score_diff_clipped = np.clip(df["score_diff_attacker"], -4, 4)
    li_raw = [
        lookup[(int(inn), half, int(out), base, int(sd))]
        for inn, half, out, base, sd in zip(
            df["inning"], df["half"], df["out_count"], df["base_state"], score_diff_clipped
        )
    ]
    df["leverage_index"] = np.array(li_raw) / li_league_avg

    OUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET_PATH, index=False)
    print(f"      저장 완료: {OUT_PARQUET_PATH} ({df.shape[0]}행 x {df.shape[1]}열)")

    print("[4/4] 패턴 분석 (표준필터 적용 부분집합 기준)...")
    dfa = df[(df["data_quality_flag"] == "") & (df["pa_result"] != "UNK")].copy()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("# LI 실제 데이터 패턴 리포트 (Task 5)\n")
    log(f"전체 저장 행수: {n_total} (94열 + leverage_index = {df.shape[1]}열)")
    log(f"패턴 분석 대상(표준필터 적용): {len(dfa)}행\n")

    log("## 이닝별 평균 leverage_index (후반으로 갈수록 상승 기대)\n")
    by_inning = dfa.groupby("inning")["leverage_index"].mean()
    for inn, v in by_inning.items():
        log(f"  {inn}회: {v:.4f}")
    inning_monotonic = all(
        by_inning.iloc[i] <= by_inning.iloc[i + 1] + 1e-6 for i in range(len(by_inning) - 1)
    )
    log(f"\n판정: {'PASS' if inning_monotonic else 'NOTE — 완전 단조는 아님, 세부 추세 확인 필요'} (단조 비감소 여부)")

    log("\n## 점수차별 평균 leverage_index (접전일수록 높음 기대)\n")
    dfa["abs_score_diff_clipped"] = np.clip(dfa["score_diff_attacker"].abs(), 0, 4)
    by_sd = dfa.groupby("abs_score_diff_clipped")["leverage_index"].mean()
    for sd, v in by_sd.items():
        log(f"  |점수차|={sd}: {v:.4f}")
    sd_monotonic = all(by_sd.iloc[i] >= by_sd.iloc[i + 1] - 1e-6 for i in range(len(by_sd) - 1))
    log(f"\n판정: {'PASS' if sd_monotonic else 'NOTE — 완전 단조는 아님'} (점수차 클수록 비증가)")

    log("\n## 주자 상태별 평균 leverage_index (득점권 있을 때 높음 기대)\n")
    by_base = dfa.groupby("base_state")["leverage_index"].mean().sort_values(ascending=False)
    for base, v in by_base.items():
        log(f"  주자={base:>4s}: {v:.4f}")
    scoring_position = {"2", "3", "23", "123", "12", "13"}
    no_scoring = {"0", "1"}
    avg_scoring = dfa[dfa["base_state"].isin(scoring_position)]["leverage_index"].mean()
    avg_no_scoring = dfa[dfa["base_state"].isin(no_scoring)]["leverage_index"].mean()
    log(f"\n  득점권 주자 평균: {avg_scoring:.4f}  vs  비득점권 평균: {avg_no_scoring:.4f}")
    log(f"판정: {'PASS' if avg_scoring > avg_no_scoring else '**FAIL**'} (득점권이 더 높아야 함)")

    log("\n## 아웃 카운트별 평균 leverage_index (Option B 효과 확인)\n")
    by_out = dfa.groupby("out_count")["leverage_index"].mean()
    for out, v in by_out.items():
        log(f"  {out}아웃: {v:.4f}")
    log(
        "\n참고: Option A였다면 이 경향이 훨씬 완만했을 것 (WE 자체가 아웃무관 고정). "
        "Task 4에서 확인한 대로 2아웃일수록 병살/희생플라이 가능성이 사라지고 "
        "이닝 종료 위험이 커지는 구조적 효과가 아웃별 평균에 그대로 반영됨."
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] 패턴 리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
