"""
리그 평균 타석 결과 분포 집계 — LI(Leverage Index) 계산용 P_league(r) 산출.

LI_raw(s) = sqrt( Σ_r P_league(r) * [WE(s->r) - WE(s)]^2 ) 에 쓰이는
P_league(r)을 (a) 전체 평균, (b) 상황별 조건부 두 버전으로 만들어 비교하고,
차이가 유의미하면 (b)를 채택한다 (작업 지시서 Task 1).

조건부 후보로 이닝대/아웃/주자 구간을 각각 단독으로 총변동거리(TVD)를 재본 결과:
  - inning_bucket 단독:        weighted TVD = 0.0087  (거의 무의미)
  - out_count 단독:            weighted TVD = 0.0288
  - base_state 단독:           weighted TVD = 0.0472
  - out_count x base_state:    weighted TVD = 0.0580, 최대 0.234 (1아웃, 만루)

이닝은 타석 결과 확률 자체에는 거의 영향이 없다 (투수 피로/전략 변화가 있어도
9클래스 분포 수준에서는 안 잡힘 - WE 테이블이 이미 이닝에 따른 "가치" 차이를
반영하므로 "확률"까지 이닝별로 쪼갤 필요는 낮다).

반면 out_count x base_state는 구조적으로 중요하다:
  - SF(희생플라이)는 3루 주자가 있어야만 발생 가능
  - GDP(병살)는 주자가 있고 아웃 2개 이하일 때만 발생 가능
  전체 평균(a)을 모든 상황에 그대로 쓰면, 주자 없는 상황에도 SF/GDP 확률이
  실리게 되어 LI_raw가 왜곡된다.

따라서 (out_count, base_state) 24개 버킷의 조건부 분포를 채택한다.
버킷 최소 표본 816건으로 안정적 (3루타처럼 희귀한 클래스도 버킷당 최소
관측 3건 이상, 전체 비율과 크게 어긋나지 않음).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PARQUET_PATH = Path("data_analysis/results/all_pa_enriched_v3.parquet")
OUT_PATH = Path("data_analysis/results/league_pa_distribution.json")

CLASSES: list[str] = ["1B", "2B", "3B", "HR", "BB", "SO", "OUT", "GDP", "SF"]
BASE_STATES: list[str] = ["0", "1", "2", "3", "12", "13", "23", "123"]
OUT_COUNTS: list[int] = [0, 1, 2]


def _inning_bucket(inning: int) -> str:
    if inning <= 3:
        return "1-3"
    if inning <= 6:
        return "4-6"
    return "7-9+"


def _weighted_tvd(df: pd.DataFrame, group_cols: list[str], overall: pd.Series) -> float:
    g = (
        df.groupby(group_cols)["pa_result"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=CLASSES, fill_value=0.0)
    )
    sizes = df.groupby(group_cols).size()
    tvd = (g.subtract(overall, axis=1)).abs().sum(axis=1) / 2
    return float((tvd * sizes).sum() / sizes.sum())


def main() -> None:
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["pa_result", "inning", "out_count", "base_state", "data_quality_flag"],
    )
    n_raw = len(df)

    # 6-2장 표준 필터: 품질 플래그 정상 + UNK 제외
    df = df[df["data_quality_flag"] == ""]
    df = df[df["pa_result"] != "UNK"]
    n_used = len(df)

    assert set(df["pa_result"].unique()) == set(CLASSES), "pa_result 9클래스 불일치"

    overall = df["pa_result"].value_counts(normalize=True).reindex(CLASSES).fillna(0.0)

    df = df.copy()
    df["inning_bucket"] = df["inning"].apply(_inning_bucket)

    diag = {
        "weighted_tvd_inning_bucket_only": _weighted_tvd(df, ["inning_bucket"], overall),
        "weighted_tvd_out_count_only": _weighted_tvd(df, ["out_count"], overall),
        "weighted_tvd_base_state_only": _weighted_tvd(df, ["base_state"], overall),
        "weighted_tvd_out_x_base": _weighted_tvd(df, ["out_count", "base_state"], overall),
    }

    # 채택: (out_count, base_state) 조건부
    grouped = (
        df.groupby(["out_count", "base_state"])["pa_result"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=CLASSES, fill_value=0.0)
    )
    sizes = df.groupby(["out_count", "base_state"]).size()

    conditional_table: dict[str, dict[str, dict[str, float]]] = {}
    for out_count in OUT_COUNTS:
        conditional_table[str(out_count)] = {}
        for base_state in BASE_STATES:
            row = grouped.loc[(out_count, base_state)]
            conditional_table[str(out_count)][base_state] = {
                c: round(float(row[c]), 6) for c in CLASSES
            }

    output = {
        "source": str(PARQUET_PATH),
        "n_pa_raw": n_raw,
        "n_pa_used": n_used,
        "filters_applied": [
            "data_quality_flag == ''",
            "pa_result != 'UNK'",
        ],
        "classes": CLASSES,
        "decision": "conditional_b",
        "decision_rationale": (
            "이닝 단독 조건부는 TVD 0.0087로 무의미하여 제외. "
            "out_count x base_state 조건부는 TVD 0.058(최대 0.234, 1아웃 만루)로 "
            "구조적 차이(SF는 3루주자 필수, GDP는 주자+2아웃미만 필수)를 반영하므로 채택. "
            "최소 버킷 표본 816건으로 24개 버킷 모두 안정적."
        ),
        "diagnostics": diag,
        "overall_distribution": {c: round(float(overall[c]), 6) for c in CLASSES},
        "conditional_distribution": {
            "conditioning_vars": ["out_count", "base_state"],
            "bucket_sizes": {
                f"{o}_{b}": int(sizes.loc[(o, b)]) for o in OUT_COUNTS for b in BASE_STATES
            },
            "min_bucket_size": int(sizes.min()),
            "table": conditional_table,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {OUT_PATH} 저장 완료")
    print(f"  n_pa_used = {n_used} / {n_raw}")
    print(f"  결정: {output['decision']} — {output['decision_rationale']}")
    print("\n[진단] TVD 비교")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}")
    print("\n[전체 평균 P_league(r)]")
    for c in CLASSES:
        print(f"  {c}: {overall[c]:.4f}")


if __name__ == "__main__":
    main()
