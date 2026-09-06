"""
LI 정규화 기준(li_league_avg) 산출 — Task 3.

전체 PA에 compute_li_raw를 직접 55만 번 돌리는 대신, LI_raw가 실제로
의존하는 변수의 조합이 유한하다는 점을 이용해 룩업 테이블을 미리 계산한다.

LI_raw(inning, half, out_count, base_state, score_diff_attacker)는:
  - inning: 1~9 (연장 제외, all_pa_enriched_v3 정책상 이미 보장됨)
  - half: top/bot
  - out_count: 0~2
  - base_state: 8종
  - score_diff_attacker: we_re_lookup.get_we가 내부적으로 항상 ±4로 clip
    하므로, ±4를 벗어나는 실제 값(예: 27점차)도 clip된 값과 완전히 동일한
    LI_raw를 낸다 (부호도 clip 과정에서 절대 안 바뀜 — 0을 넘어 반대 부호로
    넘어가는 clip은 없기 때문).

따라서 조합 수는 9 x 2 x 3 x 8 x 9 = 3,888개로 유한하고, 이 조합에 대해서만
compute_li_raw를 계산한 뒤 전체 데이터프레임에 매핑(join)한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.methods.leverage_index import (
    BASE_STATES,
    compute_li_raw,
    get_bucket_distribution,
    load_league_distribution,
)

PARQUET_PATH = Path("data_analysis/results/all_pa_enriched_v3.parquet")
OUT_PATH = Path("data_analysis/results/li_baseline.json")

INNINGS = range(1, 10)
HALVES = ("top", "bot")
OUT_COUNTS = (0, 1, 2)
SCORE_DIFFS = range(-4, 5)


def build_lookup_table(league_data: dict) -> dict[tuple, float]:
    """(inning, half, out_count, base_state, clipped_score_diff) -> LI_raw."""
    table: dict[tuple, float] = {}
    for inning in INNINGS:
        for half in HALVES:
            for out_count in OUT_COUNTS:
                for base_state in BASE_STATES:
                    dist = get_bucket_distribution(league_data, out_count, base_state)
                    for sd in SCORE_DIFFS:
                        li = compute_li_raw(inning, half, out_count, base_state, sd, dist)
                        table[(inning, half, out_count, base_state, sd)] = li
    return table


def main() -> None:
    league_data = load_league_distribution()
    print(f"[1/4] LI_raw 룩업 테이블 계산 중 (9x2x3x8x9 = {9*2*3*8*9}개 조합)...")
    lookup = build_lookup_table(league_data)
    print(f"      완료: {len(lookup)}개 조합")

    print("[2/4] 전체 PA 데이터 로드 및 필터링...")
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["inning", "half", "out_count", "base_state", "score_diff_attacker",
                  "data_quality_flag", "pa_result"],
    )
    n_raw = len(df)
    df = df[(df["data_quality_flag"] == "") & (df["pa_result"] != "UNK")].copy()
    n_used = len(df)
    print(f"      n = {n_used} / {n_raw}")

    print("[3/4] LI_raw 매핑 (score_diff는 WE 테이블과 동일하게 ±4 clip)...")
    df["score_diff_clipped"] = np.clip(df["score_diff_attacker"], -4, 4)
    df["li_raw"] = df.apply(
        lambda r: lookup[(int(r["inning"]), r["half"], int(r["out_count"]),
                           r["base_state"], int(r["score_diff_clipped"]))],
        axis=1,
    )

    li_league_avg = float(df["li_raw"].mean())
    df["li"] = df["li_raw"] / li_league_avg

    print("[4/4] 분포 통계 계산 및 저장...")
    percentiles = [0, 25, 50, 75, 95, 100]
    raw_stats = {f"p{p}": float(np.percentile(df["li_raw"], p)) for p in percentiles}
    norm_stats = {f"p{p}": float(np.percentile(df["li"], p)) for p in percentiles}

    output = {
        "source": str(PARQUET_PATH),
        "n_pa_used": n_used,
        "li_league_avg": li_league_avg,
        "normalization": "li = li_raw / li_league_avg  (평균 상황 = 1.0)",
        "li_raw_distribution": raw_stats,
        "li_normalized_distribution": norm_stats,
        "lookup_table_combinations": len(lookup),
        "note": (
            "score_diff_attacker는 get_we 내부 clip 규칙과 동일하게 ±4로 clip 후 "
            "룩업. clip은 부호를 바꾸지 않으므로(0을 가로지르지 않으므로) 9회말 "
            "경계조건(끝내기/원정승/연장동점) 판정에는 영향 없음."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {OUT_PATH} 저장 완료")
    print(f"  li_league_avg = {li_league_avg:.6f}")
    print("\n[LI_raw 분포]")
    for k, v in raw_stats.items():
        print(f"  {k}: {v:.4f}")
    print("\n[LI 정규화 분포] (평균=1.0 기준)")
    for k, v in norm_stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
