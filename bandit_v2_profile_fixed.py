import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

# Step 4의 코드 재사용
import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path(
    "bandit_output_v2_profile_fixed"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 1. 전체 pitcher history를 profile snapshot으로 준비
# ============================================================

def prepare_full_history(history):
    """
    all_pitcher_history 전체를 사용한다.

    중요:
    prior_*는 이미 leakage-free로 만들어진 값이므로
    여기서 새로 재계산하지 않는다.

    단지 기존 prior_*를 후보 투수 feature 형식으로 변환한다.
    """

    history = history.copy()

    history["game_id"] = (
        history["game_id"]
        .astype(str)
    )

    history["pitcher_id"] = (
        history["pitcher_id"]
        .apply(base.normalize_id)
    )

    history["appearance_order"] = (
        pd.to_numeric(
            history["appearance_order"],
            errors="coerce",
        )
    )

    history = (
        history.dropna(
            subset=[
                "appearance_order"
            ]
        )
        .copy()
    )

    history["appearance_order"] = (
        history["appearance_order"]
        .astype(int)
    )

    # 기존 Bandit feature와 동일
    history[
        "prior_pitcher_reward_mean"
    ] = (
        -history[
            "prior_wpa_mean"
        ]
    )

    history[
        "prior_n_apps_log"
    ] = np.log1p(
        history[
            "prior_n_apps"
        ].fillna(0)
    )

    history = (
        history.sort_values(
            [
                "pitcher_id",
                "game_id",
                "appearance_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return history


# ============================================================
# 2. pitcher별 history index 생성
# ============================================================

def build_profile_index(history):
    """
    pitcher_id별로 모든 appearance snapshot을 보관한다.

    각 entry:
        time_key = (game_id, appearance_order)
        profile = 그 appearance 직전 prior_* 값
    """

    base.print_section(
        "BUILD FULL PITCHER PROFILE INDEX"
    )

    profile_index = {}

    for pid, group in history.groupby(
        "pitcher_id",
        sort=False,
    ):

        keys = []
        profiles = []

        group = group.sort_values(
            [
                "game_id",
                "appearance_order",
            ]
        )

        for _, row in group.iterrows():

            key = (
                str(
                    row["game_id"]
                ),
                int(
                    row[
                        "appearance_order"
                    ]
                ),
            )

            profile = {
                "pitcher_id":
                    str(pid),

                "prior_n_apps":
                    row[
                        "prior_n_apps"
                    ],

                "prior_n_apps_log":
                    row[
                        "prior_n_apps_log"
                    ],

                "prior_pitcher_reward_mean":
                    row[
                        "prior_pitcher_reward_mean"
                    ],

                "prior_wpa_std":
                    row[
                        "prior_wpa_std"
                    ],

                "prior_bb_rate":
                    row[
                        "prior_bb_rate"
                    ],

                "prior_so_rate":
                    row[
                        "prior_so_rate"
                    ],

                "prior_hr_rate":
                    row[
                        "prior_hr_rate"
                    ],

                "prior_avg_pa_per_app":
                    row[
                        "prior_avg_pa_per_app"
                    ],
            }

            keys.append(
                key
            )

            profiles.append(
                profile
            )

        profile_index[
            str(pid)
        ] = {
            "keys":
                keys,

            "profiles":
                profiles,
        }

    print(
        "Indexed pitchers : "
        f"{len(profile_index):,}"
    )

    print(
        "History rows     : "
        f"{len(history):,}"
    )

    return profile_index


# ============================================================
# 3. 현재 시점 candidate profile 조회
# ============================================================

def get_asof_profile(
    decision,
    pitcher_id,
    profile_index,
):
    """
    현재 decision 시점의 후보 투수 profile을 반환한다.

    실제 선택 투수:
        decision row 자체의 prior_* 사용.
        이것이 해당 시점의 정확한 leakage-free snapshot.

    counterfactual 후보:
        1) 현재 시점 이후 그 투수의 '다음 appearance'가 있다면
           그 row의 prior_*를 사용한다.

           이유:
           prior_*는 해당 appearance 자체를 shift(1)로 제외한다.
           그리고 그 row는 현재 시점 이후의 '첫 appearance'이므로
           현재 시점과 그 appearance 사이에 해당 투수의
           다른 appearance가 존재하지 않는다.

           따라서 그 prior_*는 현재 시점까지 알려진
           과거 기록과 동일하다.

        2) 이후 appearance가 없다면
           가장 최근 과거 snapshot을 fallback으로 사용한다.
           이 경우 마지막 appearance 결과가 반영되지 않은
           한-등판 stale 가능성이 있으므로 별도 source로 기록.

        3) 투수 history 자체가 없으면 blank.
    """

    pid = str(
        pitcher_id
    )

    actual = str(
        decision[
            "pitcher_id"
        ]
    )

    # --------------------------------------------------------
    # 실제 감독 선택 투수
    # current decision의 prior_*가 가장 정확
    # --------------------------------------------------------

    if pid == actual:

        return (
            base.profile_from_row(
                decision
            ),
            "actual_current",
        )

    # --------------------------------------------------------
    # history 자체가 없음
    # --------------------------------------------------------

    if pid not in profile_index:

        return (
            base.blank_profile(
                pid
            ),
            "blank_no_history",
        )

    item = (
        profile_index[
            pid
        ]
    )

    keys = (
        item["keys"]
    )

    profiles = (
        item["profiles"]
    )

    current_key = (
        str(
            decision[
                "game_id"
            ]
        ),
        int(
            decision[
                "appearance_order"
            ]
        ),
    )

    # --------------------------------------------------------
    # current_key보다 큰 첫 appearance 탐색
    # --------------------------------------------------------

    pos = bisect_right(
        keys,
        current_key,
    )

    # 다음 appearance가 있다면
    # 그 appearance 직전 prior_* = 현재 시점 as-of snapshot
    if pos < len(keys):

        return (
            profiles[pos],
            "next_snapshot",
        )

    # --------------------------------------------------------
    # 앞으로 appearance가 없는 경우
    # 가장 최근 과거 snapshot fallback
    # --------------------------------------------------------

    past_pos = (
        pos - 1
    )

    if past_pos >= 0:

        return (
            profiles[
                past_pos
            ],
            "past_snapshot_fallback",
        )

    return (
        base.blank_profile(
            pid
        ),
        "blank_no_snapshot",
    )


# ============================================================
# 4. Evaluation
# ============================================================

def evaluate_profile_fixed(
    name,
    model,
    processor,
    eval_df,
    bullpen_map,
    used_before_map,
    profile_index,
):

    base.print_section(
        f"EVALUATE {name} - PROFILE FIXED"
    )

    # 실제 당일 명단이 존재하는 구간만
    eval_df = (
        eval_df[
            eval_df[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    eval_df = (
        eval_df.sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
    )

    rows = []
    candidate_rows = []

    total_decisions = (
        len(eval_df)
    )

    actual_candidate_hits = 0
    skipped_small = 0

    source_counts = {}

    # ========================================================
    # decision loop
    # ========================================================

    for _, decision in (
        eval_df.iterrows()
    ):

        game_id = str(
            decision[
                "game_id"
            ]
        )

        team = str(
            decision[
                "def_team"
            ]
        )

        actual = str(
            decision[
                "pitcher_id"
            ]
        )

        appearance_order = int(
            decision[
                "appearance_order"
            ]
        )

        # ----------------------------------------------------
        # 실제 당일 bullpen
        # ----------------------------------------------------

        listed = set(
            bullpen_map.get(
                (
                    game_id,
                    team,
                ),
                set(),
            )
        )

        # ----------------------------------------------------
        # 이미 사용된 투수
        # ----------------------------------------------------

        used = set(
            used_before_map.get(
                (
                    game_id,
                    team,
                    appearance_order,
                ),
                set(),
            )
        )

        available_ids = (
            listed
            - used
        )

        actual_in_candidates = (
            actual
            in available_ids
        )

        if actual_in_candidates:
            actual_candidate_hits += 1

        # 실제 투수가 없으면 agreement 계산 불가
        if not actual_in_candidates:
            continue

        if len(
            available_ids
        ) < 2:

            skipped_small += 1
            continue

        scored = []

        # ====================================================
        # 후보별 score
        # ====================================================

        for pid in sorted(
            available_ids
        ):

            (
                profile,
                profile_source,
            ) = get_asof_profile(
                decision,
                pid,
                profile_index,
            )

            source_counts[
                profile_source
            ] = (
                source_counts.get(
                    profile_source,
                    0,
                )
                + 1
            )

            values = (
                base.candidate_feature_values(
                    decision,
                    profile,
                )
            )

            x = (
                processor.transform_values(
                    values
                )
            )

            (
                ucb_score,
                pred_mean,
                uncertainty,
            ) = model.score(x)

            prior_n_apps = (
                profile[
                    "prior_n_apps"
                ]
            )

            if pd.isna(
                prior_n_apps
            ):
                prior_n_apps = 0.0

            candidate_item = {
                "pitcher_id":
                    str(pid),

                "profile_source":
                    profile_source,

                "prior_n_apps":
                    float(
                        prior_n_apps
                    ),

                "pred_mean":
                    float(
                        pred_mean
                    ),

                "uncertainty":
                    float(
                        uncertainty
                    ),

                "ucb_score":
                    float(
                        ucb_score
                    ),

                "is_actual":
                    (
                        str(pid)
                        == actual
                    ),
            }

            scored.append(
                candidate_item
            )

            candidate_rows.append(
                {
                    "game_id":
                        game_id,

                    "date":
                        decision[
                            "date"
                        ],

                    "def_team":
                        team,

                    "appearance_order":
                        appearance_order,

                    **candidate_item,
                }
            )

        # ----------------------------------------------------
        # UCB 추천
        # ----------------------------------------------------

        ucb_best = max(
            scored,
            key=lambda x:
                x[
                    "ucb_score"
                ],
        )

        # ----------------------------------------------------
        # Mean-only 진단
        # ----------------------------------------------------

        mean_best = max(
            scored,
            key=lambda x:
                x[
                    "pred_mean"
                ],
        )

        # ----------------------------------------------------
        # Heuristic
        # ----------------------------------------------------

        heuristic = max(
            scored,
            key=lambda x:
                x[
                    "prior_n_apps"
                ],
        )

        actual_item = next(
            x
            for x in scored
            if x[
                "pitcher_id"
            ]
            == actual
        )

        actual_ucb_rank = (
            1
            + sum(
                x[
                    "ucb_score"
                ]
                >
                actual_item[
                    "ucb_score"
                ]
                + 1e-12
                for x in scored
            )
        )

        actual_mean_rank = (
            1
            + sum(
                x[
                    "pred_mean"
                ]
                >
                actual_item[
                    "pred_mean"
                ]
                + 1e-12
                for x in scored
            )
        )

        rows.append(
            {
                "game_id":
                    game_id,

                "date":
                    decision[
                        "date"
                    ],

                "def_team":
                    team,

                "inning":
                    decision[
                        "inning"
                    ],

                "appearance_order":
                    appearance_order,

                "actual_pitcher":
                    actual,

                "n_candidates":
                    len(scored),

                "ucb_pitcher":
                    ucb_best[
                        "pitcher_id"
                    ],

                "ucb_source":
                    ucb_best[
                        "profile_source"
                    ],

                "ucb_match":
                    (
                        ucb_best[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                "mean_pitcher":
                    mean_best[
                        "pitcher_id"
                    ],

                "mean_source":
                    mean_best[
                        "profile_source"
                    ],

                "mean_match":
                    (
                        mean_best[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                "heuristic_pitcher":
                    heuristic[
                        "pitcher_id"
                    ],

                "heuristic_match":
                    (
                        heuristic[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                "actual_source":
                    actual_item[
                        "profile_source"
                    ],

                "actual_ucb_rank":
                    actual_ucb_rank,

                "actual_mean_rank":
                    actual_mean_rank,

                "random_expected_match":
                    (
                        1.0
                        /
                        len(scored)
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    if result.empty:

        print(
            "No evaluable decisions."
        )

        return (
            result,
            candidates,
            {},
        )

    # ========================================================
    # Summary
    # ========================================================

    actual_coverage = (
        actual_candidate_hits
        /
        total_decisions
        if total_decisions
        else 0.0
    )

    random_agreement = (
        result[
            "random_expected_match"
        ]
        .mean()
    )

    heuristic_agreement = (
        result[
            "heuristic_match"
        ]
        .mean()
    )

    ucb_agreement = (
        result[
            "ucb_match"
        ]
        .mean()
    )

    mean_agreement = (
        result[
            "mean_match"
        ]
        .mean()
    )

    avg_candidates = (
        result[
            "n_candidates"
        ]
        .mean()
    )

    print(
        f"Total preview decisions    : "
        f"{total_decisions:,}"
    )

    print(
        f"Actual candidate coverage  : "
        f"{actual_coverage * 100:.2f}%"
    )

    print(
        f"Evaluated decisions        : "
        f"{len(result):,}"
    )

    print(
        f"Avg candidates             : "
        f"{avg_candidates:.2f}"
    )

    print()

    print(
        f"Random agreement           : "
        f"{random_agreement * 100:.2f}%"
    )

    print(
        f"Heuristic agreement        : "
        f"{heuristic_agreement * 100:.2f}%"
    )

    print(
        f"UCB agreement              : "
        f"{ucb_agreement * 100:.2f}%"
    )

    print(
        f"Mean-only agreement        : "
        f"{mean_agreement * 100:.2f}%"
    )

    # ========================================================
    # Profile source distribution
    # ========================================================

    print()
    print(
        "Candidate profile source:"
    )

    source_series = (
        candidates[
            "profile_source"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print(
        source_series
    )

    print()

    print(
        "UCB recommendation source:"
    )

    print(
        result[
            "ucb_source"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print()

    print(
        "MEAN recommendation source:"
    )

    print(
        result[
            "mean_source"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    # ========================================================
    # Rank
    # ========================================================

    print()

    print(
        f"Actual UCB rank mean       : "
        f"{result['actual_ucb_rank'].mean():.2f}"
    )

    print(
        f"Actual UCB rank median     : "
        f"{result['actual_ucb_rank'].median():.2f}"
    )

    print(
        f"Actual UCB Top-3           : "
        f"{(result['actual_ucb_rank'] <= 3).mean() * 100:.2f}%"
    )

    print(
        f"Actual UCB Top-5           : "
        f"{(result['actual_ucb_rank'] <= 5).mean() * 100:.2f}%"
    )

    print()

    print(
        f"Actual MEAN rank mean      : "
        f"{result['actual_mean_rank'].mean():.2f}"
    )

    print(
        f"Actual MEAN rank median    : "
        f"{result['actual_mean_rank'].median():.2f}"
    )

    print(
        f"Actual MEAN Top-3          : "
        f"{(result['actual_mean_rank'] <= 3).mean() * 100:.2f}%"
    )

    print(
        f"Actual MEAN Top-5          : "
        f"{(result['actual_mean_rank'] <= 5).mean() * 100:.2f}%"
    )

    # ========================================================
    # Summary dict
    # ========================================================

    summary = {
        "split":
            name,

        "total_preview_decisions":
            int(
                total_decisions
            ),

        "evaluated_decisions":
            int(
                len(result)
            ),

        "actual_candidate_coverage":
            float(
                actual_coverage
            ),

        "avg_candidates":
            float(
                avg_candidates
            ),

        "random_agreement":
            float(
                random_agreement
            ),

        "heuristic_agreement":
            float(
                heuristic_agreement
            ),

        "ucb_agreement":
            float(
                ucb_agreement
            ),

        "mean_only_agreement":
            float(
                mean_agreement
            ),

        "skipped_small":
            int(
                skipped_small
            ),
    }

    return (
        result,
        candidates,
        summary,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- PROFILE FIX"
    )

    print(
        f"REWARD MODE : "
        f"{base.REWARD_MODE}"
    )

    print(
        f"ALPHA       : "
        f"{base.ALPHA}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 동일 데이터 로드
    # ========================================================

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    (
        decisions,
        first_app,
    ) = (
        base.build_decision_dataset(
            pa,
            history,
        )
    )

    bullpen_map = (
        base.build_bullpen_map(
            bullpen
        )
    )

    used_before_map = (
        base.build_used_before_map(
            first_app
        )
    )

    (
        train,
        val,
        test,
    ) = (
        base.chronological_split(
            decisions
        )
    )

    # ========================================================
    # LinUCB도 기존과 완전히 동일
    # ========================================================

    processor = (
        base.FeatureProcessor()
    )

    model = (
        base.train_bandit(
            train,
            processor,
        )
    )

    # ========================================================
    # 바뀌는 부분:
    # all_pitcher_history 전체 profile index
    # ========================================================

    full_history = (
        prepare_full_history(
            history
        )
    )

    profile_index = (
        build_profile_index(
            full_history
        )
    )

    # ========================================================
    # Validation
    # ========================================================

    (
        val_result,
        val_candidates,
        val_summary,
    ) = evaluate_profile_fixed(
        "VALIDATION",
        model,
        processor,
        eval_df=val,
        bullpen_map=(
            bullpen_map
        ),
        used_before_map=(
            used_before_map
        ),
        profile_index=(
            profile_index
        ),
    )

    # ========================================================
    # Test
    # ========================================================

    (
        test_result,
        test_candidates,
        test_summary,
    ) = evaluate_profile_fixed(
        "TEST",
        model,
        processor,
        eval_df=test,
        bullpen_map=(
            bullpen_map
        ),
        used_before_map=(
            used_before_map
        ),
        profile_index=(
            profile_index
        ),
    )

    # ========================================================
    # Save
    # ========================================================

    val_result.to_csv(
        OUTPUT_DIR
        / "validation_decisions.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_decisions.csv",
        index=False,
    )

    val_candidates.to_csv(
        OUTPUT_DIR
        / "validation_candidates.csv",
        index=False,
    )

    test_candidates.to_csv(
        OUTPUT_DIR
        / "test_candidates.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "reward_mode":
                    base.REWARD_MODE,

                "alpha":
                    base.ALPHA,

                "profile_method":
                    (
                        "global_full_history_"
                        "asof_snapshot"
                    ),

                "validation":
                    val_summary,

                "test":
                    test_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    base.print_section(
        "DONE"
    )

    print("Saved:")

    print(
        " ",
        OUTPUT_DIR
        / "validation_decisions.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_decisions.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "summary.json"
    )


if __name__ == "__main__":
    main()