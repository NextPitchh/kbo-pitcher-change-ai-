"""
불펜 quality_p 원재료(prior_wpa_mean) Option A -> B 재계산 — Task 15.

all_pitcher_history.parquet의 prior_wpa_mean은 그 파일 자체의 app_wpa
컬럼(Option A, 아웃카운트 보정 없는 낡은 WE 계산)을 누적평균낸 값이다.
팀원(Bandit)이 별도로 발견/확인한 사실과 동일하게, 이 프로젝트에서 실제로
쓰기로 한 정답 컬럼은 all_pa_enriched_v3/all_pa_with_li의 reward_wpa_computed
(Option B, 아웃카운트 보정 적용)다.

이 스크립트는 prior_wpa_mean과 "같은 방식"(leakage 없이, 이 시점 이전
등판만, 역할 구분 없이 그 투수의 전체 등판을 순서대로 누적평균)으로,
재료만 Option B로 바꿔서 다시 만든다. leakage-free 로직 자체는 원본과
동일하게 유지 — 재료만 교체하는 것이지 새로운 종류의 계산을 만드는 게
아니다.

1. all_pa_with_li.parquet에서 (game_id, half, pitcher_id, appearance_order)
   단위로 reward_wpa_computed를 합산 -> 등판별 Option B WPA(app_wpa_optionb)
   **표준필터(data_quality_flag/pa_result) 적용 안 함** — 검증 결과
   all_pitcher_history.parquet 자체가 필터 없이 집계된 것으로 확인됨
   (필터 적용 시 n_pa 불일치 3,763건, 필터 미적용 시 불일치 0건 — 완전 일치).
   기존 집계 관례와 다르게 하면 prior_n_apps 기준이 어긋나므로 그대로 따름.
2. 투수별로 시간순 정렬 후, "이 등판 이전까지"만 평균(자기 자신 제외)
   -> prior_wpa_mean_optionb
3. all_pitcher_history.parquet의 기존 prior_n_apps와 일치하는지 검증
   (같은 방식으로 세었다면 등판 순번이 정확히 일치해야 함)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PA_PATH = Path("data_analysis/results/all_pa_with_li.parquet")
HIST_PATH = Path("data_analysis/all_pitcher_history.parquet")
OUT_PATH = Path("data_analysis/results/prior_wpa_optionb.parquet")


def build_appearance_level_optionb(pa: pd.DataFrame) -> pd.DataFrame:
    df = pa.copy()

    keys = ["game_id", "half", "pitcher_id", "appearance_order"]
    app = df.groupby(keys, as_index=False).agg(
        app_wpa_optionb=("reward_wpa_computed", "sum"),
        n_pa_optionb=("reward_wpa_computed", "size"),
    )

    date_map = df.drop_duplicates("game_id")[["game_id", "date"]]
    app = app.merge(date_map, on="game_id", how="left")
    return app


def add_leakage_free_prior_mean(app: pd.DataFrame) -> pd.DataFrame:
    app = app.sort_values(["pitcher_id", "date", "game_id"]).reset_index(drop=True)

    def expanding_prior_mean(g: pd.DataFrame) -> pd.Series:
        wpa = g["app_wpa_optionb"].to_numpy()
        n = len(wpa)
        out = np.full(n, np.nan)
        cum_sum = 0.0
        for i in range(n):
            if i > 0:
                out[i] = cum_sum / i
            cum_sum += wpa[i]
        return pd.Series(out, index=g.index)

    app["prior_wpa_mean_optionb"] = app.groupby("pitcher_id", group_keys=False).apply(
        expanding_prior_mean
    )
    app["prior_n_apps_optionb"] = app.groupby("pitcher_id").cumcount()
    return app


def main() -> None:
    pa = pd.read_parquet(PA_PATH)
    print(f"[1/3] all_pa_with_li.parquet 로드: {len(pa)}행")

    app = build_appearance_level_optionb(pa)
    print(f"      등판 단위 집계(Option B): {len(app)}건")

    dup = app.groupby(["pitcher_id", "game_id"]).size()
    dup = dup[dup > 1]
    if len(dup):
        print(f"[경고] (pitcher_id, game_id) 중복 {len(dup)}건 — half가 둘 다 기록된 이상치와 동일 패턴")

    app = add_leakage_free_prior_mean(app)
    print("[2/3] leakage-free 누적평균(prior_wpa_mean_optionb) 계산 완료")

    print("[3/3] 기존 all_pitcher_history.prior_n_apps와 검증...")
    hist = pd.read_parquet(HIST_PATH)
    check = hist[["pitcher_id", "game_id", "half", "appearance_order", "prior_n_apps", "prior_wpa_mean"]].merge(
        app[["pitcher_id", "game_id", "half", "appearance_order", "prior_n_apps_optionb", "prior_wpa_mean_optionb"]],
        on=["pitcher_id", "game_id", "half", "appearance_order"],
        how="left",
    )
    n_matched = check["prior_n_apps_optionb"].notna().sum()
    n_count_mismatch = (check["prior_n_apps"] != check["prior_n_apps_optionb"]).sum()
    print(f"      매칭된 행: {n_matched} / {len(hist)}")
    print(f"      prior_n_apps 불일치 행: {n_count_mismatch} (0에 가까울수록 같은 방식으로 센 것)")

    corr = check[["prior_wpa_mean", "prior_wpa_mean_optionb"]].dropna().corr().iloc[0, 1]
    print(f"      기존(Option A) vs 신규(Option B) prior_wpa_mean 상관계수: {corr:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    app.to_parquet(OUT_PATH, index=False)
    print(f"\n[OK] 저장: {OUT_PATH} ({len(app)}행)")


if __name__ == "__main__":
    main()
