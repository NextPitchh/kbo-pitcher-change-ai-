"""
투수 품질(quality_p) 스케일링 결정 — Task 3.

Task 2에서 검증 통과한 선발용 지표(prior_composite = prior_so_rate -
prior_bb_rate - 2*prior_hr_rate, Spearman r=0.153)를 대상으로, 세 가지
스케일링 방식을 비교한다.

(a) 원값: 범위 -1.0~+0.5 (대부분 -0.08~0.12) — LI(0.1~9.8)에 비해
    자릿수가 너무 작아 U = quality_p * LI 공식에서 존재감이 사라진다.
(b) z-score: 평균 0, 범위 -15.4~+7.2 — 평균적인 투수가 곱해지면 U 기여가
    0이 되어버리는 문제, 게다가 표본이 적은 투수의 극단치(-15.4)가 그대로
    노출된다.
(c) 백분위(0~1): 범위 안정적이지만 실력 차이의 "크기" 정보가 사라진다.

결정 (2026-09-06 사용자 확인): 백분위 x 2 채택.
  - 평균이 정확히 1.0이 되어 LI의 "평균 상황=1.0" 컨벤션과 맞아떨어진다.
  - 순위 기반이라 표본이 적은 투수의 극단값(예: 등판 1번에 볼넷만 내준
    경우 raw=-1.0, z=-15.4)에 영향받지 않는다 — 이런 극단값은 원래
    Task 4(shrinkage)가 해결할 문제인데, 백분위는 그 전 단계에서도
    안정적이라는 부수적 장점이 있다.
  - 25~75% 구간이 0.5~1.5로, LI의 25~75% 구간(0.51~1.30)과 자릿수가
    비슷하다.

한계로 인지해야 할 것: 백분위는 실력 차이의 "크기"를 압축한다(1등과 100등
차이가 1등과 2등 차이와 비슷하게 취급될 수 있음). Task 5 최종 확정 시
이 한계가 실제 추천 결과에 문제를 일으키는지 재확인이 필요하다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
LI_BASELINE_PATH = Path("data_analysis/results/li_baseline.json")
OUT_PATH = Path("data_analysis/results/quality_scaling_comparison.json")


def _stats(s: pd.Series) -> dict:
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    return {f"p{p}": float(np.percentile(s, p)) for p in percentiles} | {
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def main() -> None:
    hist = pd.read_parquet(HIST_PATH)
    df = hist[(hist["is_starter"]) & (hist["prior_n_apps"] >= 1)].copy()
    df["prior_composite"] = (
        df["prior_so_rate"] - df["prior_bb_rate"] - 2 * df["prior_hr_rate"]
    )
    raw = df["prior_composite"]

    options = {
        "a_원값": raw,
        "b_zscore": (raw - raw.mean()) / raw.std(),
        "c_백분위": raw.rank(pct=True),
        "채택안_백분위x2": raw.rank(pct=True) * 2,
    }

    with LI_BASELINE_PATH.open(encoding="utf-8") as f:
        li_dist = json.load(f)["li_normalized_distribution"]

    result = {
        "candidate_metric": "prior_composite (선발용, Task 2에서 검증 통과, r=0.153)",
        "n": int(len(df)),
        "decision": "백분위 x 2",
        "decision_date": "2026-09-06",
        "options": {name: _stats(s) for name, s in options.items()},
        "li_reference_distribution": li_dist,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"n = {len(df)}\n")
    print(f"{'방식':20s} {'평균':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p95':>8s} {'p100':>8s}")
    print("-" * 70)
    for name, s in options.items():
        st = _stats(s)
        print(
            f"{name:20s} {st['mean']:>8.3f} {st['p25']:>8.3f} {st['p50']:>8.3f} "
            f"{st['p75']:>8.3f} {st['p95']:>8.3f} {st['p100']:>8.3f}"
        )
    li_row = li_dist
    print(
        f"{'(참고)LI 목표분포':20s} {'1.000':>8s} {li_row['p25']:>8.3f} {li_row['p50']:>8.3f} "
        f"{li_row['p75']:>8.3f} {li_row['p95']:>8.3f} {li_row['p100']:>8.3f}"
    )
    print(f"\n[OK] 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
