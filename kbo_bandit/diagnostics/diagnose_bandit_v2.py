from pathlib import Path

import numpy as np
import pandas as pd

# Step 4에서 만든 실제 Bandit 코드를 그대로 사용한다.
from kbo_bandit.core import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path(
    "outputs/bandit/diagnostics/v2_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helper
# ============================================================

def get_candidate_profile(
    decision,
    pid,
    profiles,
):
    """
    현재 Bandit v2의 후보 profile 생성 규칙을 그대로 재현하면서
    어디서 profile을 가져왔는지 source도 기록한다.

    source:
    - known_profile:
        과거에 이 팀에서 관측된 profile이 있음

    - actual_current_prior:
        과거 profile은 없지만 실제 감독 선택 투수이므로
        현재 decision row의 leakage-free prior_* 사용

    - blank:
        counterfactual 후보인데 과거 profile이 없어
        blank_profile + median imputation 사용
    """

    team = str(
        decision["def_team"]
    )

    actual = str(
        decision["pitcher_id"]
    )

    pid = str(pid)

    if pid in profiles[team]:

        profile = (
            profiles[
                team
            ][pid]
        )

        source = (
            "known_profile"
        )

    elif pid == actual:

        profile = (
            base.profile_from_row(
                decision
            )
        )

        source = (
            "actual_current_prior"
        )

    else:

        profile = (
            base.blank_profile(
                pid
            )
        )

        source = (
            "blank"
        )

    return (
        profile,
        source,
    )


# ============================================================
# Diagnostic Evaluation
# ============================================================

def diagnose_split(
    name,
    model,
    processor,
    seed_df,
    eval_df,
    bullpen_map,
    used_before_map,
):

    base.print_section(
        f"DIAGNOSE {name}"
    )

    # --------------------------------------------------------
    # 평가 시작 전까지 알고 있는 투수 profile
    # --------------------------------------------------------

    profiles = (
        base.seed_profiles(
            seed_df
        )
    )

    # 실제 명단이 존재하는 preview만
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

    decision_rows = []
    candidate_rows = []

    total_decisions = 0
    skipped_no_actual = 0
    skipped_small_set = 0

    # ========================================================
    # Decision Loop
    # ========================================================

    for _, decision in (
        eval_df.iterrows()
    ):

        total_decisions += 1

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
        # 실제 당일 명단
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
        # 현재 시점 이전 이미 사용된 투수
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

        # ----------------------------------------------------
        # 실제 Action Set
        # ----------------------------------------------------

        available_ids = (
            listed
            - used
        )

        if (
            actual
            not in available_ids
        ):

            skipped_no_actual += 1

            # profile 시간 진행은 유지
            profiles[
                team
            ][actual] = (
                base.profile_from_row(
                    decision
                )
            )

            continue

        if len(
            available_ids
        ) < 2:

            skipped_small_set += 1

            profiles[
                team
            ][actual] = (
                base.profile_from_row(
                    decision
                )
            )

            continue

        # ====================================================
        # 모든 후보 scoring
        # ====================================================

        scored = []

        for pid in sorted(
            available_ids
        ):

            (
                profile,
                profile_source,
            ) = get_candidate_profile(
                decision,
                pid,
                profiles,
            )

            values = (
                base.candidate_feature_values(
                    decision,
                    profile,
                )
            )

            x = (
                processor
                .transform_values(
                    values
                )
            )

            (
                score,
                pred_mean,
                uncertainty,
            ) = model.score(x)

            exploration_bonus = (
                base.ALPHA
                * uncertainty
            )

            prior_n_apps = (
                profile[
                    "prior_n_apps"
                ]
            )

            if pd.isna(
                prior_n_apps
            ):
                prior_n_apps = 0.0

            row = {
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

                "pitcher_id":
                    str(pid),

                "actual_pitcher":
                    actual,

                "is_actual":
                    (
                        str(pid)
                        == actual
                    ),

                "profile_source":
                    profile_source,

                "is_blank":
                    (
                        profile_source
                        == "blank"
                    ),

                "prior_n_apps":
                    float(
                        prior_n_apps
                    ),

                "is_zero_history":
                    (
                        float(
                            prior_n_apps
                        )
                        <= 0
                    ),

                "pred_mean":
                    float(
                        pred_mean
                    ),

                "uncertainty":
                    float(
                        uncertainty
                    ),

                "exploration_bonus":
                    float(
                        exploration_bonus
                    ),

                "ucb_score":
                    float(
                        score
                    ),
            }

            scored.append(
                row
            )

            candidate_rows.append(
                row.copy()
            )

        # ====================================================
        # UCB ranking
        # ====================================================

        scored_ucb = sorted(
            scored,
            key=lambda x:
                x["ucb_score"],
            reverse=True,
        )

        ucb_recommended = (
            scored_ucb[0]
        )

        # ====================================================
        # Mean-only ranking
        #
        # 같은 모델의 theta*x만 사용.
        # exploration bonus를 제거한 진단용 정책.
        # ====================================================

        scored_mean = sorted(
            scored,
            key=lambda x:
                x["pred_mean"],
            reverse=True,
        )

        mean_recommended = (
            scored_mean[0]
        )

        # ----------------------------------------------------
        # 실제 투수 정보
        # ----------------------------------------------------

        actual_item = next(
            x
            for x in scored
            if x[
                "pitcher_id"
            ] == actual
        )

        # ----------------------------------------------------
        # 실제 투수 UCB rank
        #
        # 1등 = Bandit 최우선 추천
        # ----------------------------------------------------

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

        # 실제 투수 mean rank
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

        blank_count = sum(
            x[
                "is_blank"
            ]
            for x in scored
        )

        zero_history_count = sum(
            x[
                "is_zero_history"
            ]
            for x in scored
        )

        # ----------------------------------------------------
        # 기존 Heuristic
        # ----------------------------------------------------

        heuristic_item = max(
            scored,
            key=lambda x:
                x[
                    "prior_n_apps"
                ],
        )

        # ====================================================
        # Decision-level 결과
        # ====================================================

        decision_rows.append(
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
                    len(
                        scored
                    ),

                "blank_candidates":
                    blank_count,

                "blank_candidate_ratio":
                    (
                        blank_count
                        /
                        len(scored)
                    ),

                "zero_history_candidates":
                    zero_history_count,

                "zero_history_candidate_ratio":
                    (
                        zero_history_count
                        /
                        len(scored)
                    ),

                # ------------------------
                # UCB 추천
                # ------------------------

                "ucb_pitcher":
                    ucb_recommended[
                        "pitcher_id"
                    ],

                "ucb_match":
                    (
                        ucb_recommended[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                "ucb_profile_source":
                    ucb_recommended[
                        "profile_source"
                    ],

                "ucb_is_blank":
                    ucb_recommended[
                        "is_blank"
                    ],

                "ucb_prior_n_apps":
                    ucb_recommended[
                        "prior_n_apps"
                    ],

                "ucb_pred_mean":
                    ucb_recommended[
                        "pred_mean"
                    ],

                "ucb_uncertainty":
                    ucb_recommended[
                        "uncertainty"
                    ],

                "ucb_exploration_bonus":
                    ucb_recommended[
                        "exploration_bonus"
                    ],

                "ucb_score":
                    ucb_recommended[
                        "ucb_score"
                    ],

                # ------------------------
                # Mean-only 추천
                # ------------------------

                "mean_pitcher":
                    mean_recommended[
                        "pitcher_id"
                    ],

                "mean_match":
                    (
                        mean_recommended[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                "mean_profile_source":
                    mean_recommended[
                        "profile_source"
                    ],

                "mean_is_blank":
                    mean_recommended[
                        "is_blank"
                    ],

                # ------------------------
                # Heuristic
                # ------------------------

                "heuristic_pitcher":
                    heuristic_item[
                        "pitcher_id"
                    ],

                "heuristic_match":
                    (
                        heuristic_item[
                            "pitcher_id"
                        ]
                        == actual
                    ),

                # ------------------------
                # 실제 투수
                # ------------------------

                "actual_profile_source":
                    actual_item[
                        "profile_source"
                    ],

                "actual_is_blank":
                    actual_item[
                        "is_blank"
                    ],

                "actual_zero_history":
                    actual_item[
                        "is_zero_history"
                    ],

                "actual_prior_n_apps":
                    actual_item[
                        "prior_n_apps"
                    ],

                "actual_pred_mean":
                    actual_item[
                        "pred_mean"
                    ],

                "actual_uncertainty":
                    actual_item[
                        "uncertainty"
                    ],

                "actual_exploration_bonus":
                    actual_item[
                        "exploration_bonus"
                    ],

                "actual_ucb_score":
                    actual_item[
                        "ucb_score"
                    ],

                "actual_ucb_rank":
                    actual_ucb_rank,

                "actual_mean_rank":
                    actual_mean_rank,

                # Random baseline
                "random_expected_match":
                    (
                        1.0
                        /
                        len(scored)
                    ),
            }
        )

        # ====================================================
        # 시간 진행
        # ====================================================

        profiles[
            team
        ][actual] = (
            base.profile_from_row(
                decision
            )
        )

    # ========================================================
    # DataFrames
    # ========================================================

    decisions_df = pd.DataFrame(
        decision_rows
    )

    candidates_df = pd.DataFrame(
        candidate_rows
    )

    if decisions_df.empty:

        print(
            "진단 가능한 decision이 없습니다."
        )

        return (
            decisions_df,
            candidates_df,
        )

    # ========================================================
    # Summary
    # ========================================================

    print(
        f"Total preview decisions     : "
        f"{total_decisions:,}"
    )

    print(
        f"Evaluated decisions         : "
        f"{len(decisions_df):,}"
    )

    print(
        f"Skipped - actual missing    : "
        f"{skipped_no_actual:,}"
    )

    print(
        f"Skipped - candidates < 2    : "
        f"{skipped_small_set:,}"
    )

    print()

    print(
        f"Avg candidates              : "
        f"{decisions_df['n_candidates'].mean():.2f}"
    )

    print(
        f"Random expected agreement   : "
        f"{decisions_df['random_expected_match'].mean() * 100:.2f}%"
    )

    print(
        f"Heuristic agreement         : "
        f"{decisions_df['heuristic_match'].mean() * 100:.2f}%"
    )

    print()

    print(
        "------------------------------------------------------------"
    )

    print(
        "1. UCB vs MEAN-ONLY"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"UCB agreement               : "
        f"{decisions_df['ucb_match'].mean() * 100:.2f}%"
    )

    print(
        f"Mean-only agreement         : "
        f"{decisions_df['mean_match'].mean() * 100:.2f}%"
    )

    print()

    # ========================================================
    # Blank profile
    # ========================================================

    print(
        "------------------------------------------------------------"
    )

    print(
        "2. BLANK PROFILE"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Blank among all candidates  : "
        f"{candidates_df['is_blank'].mean() * 100:.2f}%"
    )

    print(
        f"Blank among UCB recommends  : "
        f"{decisions_df['ucb_is_blank'].mean() * 100:.2f}%"
    )

    print(
        f"Blank among mean recommends : "
        f"{decisions_df['mean_is_blank'].mean() * 100:.2f}%"
    )

    print(
        f"Actual zero-history rate    : "
        f"{decisions_df['actual_zero_history'].mean() * 100:.2f}%"
    )

    print()

    print(
        "UCB recommendation profile source:"
    )

    print(
        decisions_df[
            "ucb_profile_source"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    print()

    print(
        "Actual pitcher profile source:"
    )

    print(
        decisions_df[
            "actual_profile_source"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    # ========================================================
    # Uncertainty
    # ========================================================

    print()

    print(
        "------------------------------------------------------------"
    )

    print(
        "3. UNCERTAINTY / EXPLORATION"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Recommended avg uncertainty : "
        f"{decisions_df['ucb_uncertainty'].mean():.6f}"
    )

    print(
        f"Actual avg uncertainty      : "
        f"{decisions_df['actual_uncertainty'].mean():.6f}"
    )

    print()

    print(
        f"Recommended avg bonus       : "
        f"{decisions_df['ucb_exploration_bonus'].mean():.6f}"
    )

    print(
        f"Actual avg bonus            : "
        f"{decisions_df['actual_exploration_bonus'].mean():.6f}"
    )

    # Blank vs known uncertainty
    blank_candidates = (
        candidates_df[
            candidates_df[
                "is_blank"
            ]
        ]
    )

    known_candidates = (
        candidates_df[
            ~candidates_df[
                "is_blank"
            ]
        ]
    )

    print()

    if len(
        blank_candidates
    ) > 0:

        print(
            f"Blank avg uncertainty       : "
            f"{blank_candidates['uncertainty'].mean():.6f}"
        )

        print(
            f"Blank avg pred_mean         : "
            f"{blank_candidates['pred_mean'].mean():.6f}"
        )

        print(
            f"Blank avg UCB score         : "
            f"{blank_candidates['ucb_score'].mean():.6f}"
        )

    if len(
        known_candidates
    ) > 0:

        print(
            f"Known avg uncertainty       : "
            f"{known_candidates['uncertainty'].mean():.6f}"
        )

        print(
            f"Known avg pred_mean         : "
            f"{known_candidates['pred_mean'].mean():.6f}"
        )

        print(
            f"Known avg UCB score         : "
            f"{known_candidates['ucb_score'].mean():.6f}"
        )

    # ========================================================
    # Actual rank
    # ========================================================

    print()

    print(
        "------------------------------------------------------------"
    )

    print(
        "4. ACTUAL PITCHER RANK"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Actual UCB rank mean        : "
        f"{decisions_df['actual_ucb_rank'].mean():.2f}"
    )

    print(
        f"Actual UCB rank median      : "
        f"{decisions_df['actual_ucb_rank'].median():.2f}"
    )

    print(
        f"Actual mean rank mean       : "
        f"{decisions_df['actual_mean_rank'].mean():.2f}"
    )

    print(
        f"Actual mean rank median     : "
        f"{decisions_df['actual_mean_rank'].median():.2f}"
    )

    print()

    print(
        f"Actual UCB Top-1            : "
        f"{(decisions_df['actual_ucb_rank'] <= 1).mean() * 100:.2f}%"
    )

    print(
        f"Actual UCB Top-3            : "
        f"{(decisions_df['actual_ucb_rank'] <= 3).mean() * 100:.2f}%"
    )

    print(
        f"Actual UCB Top-5            : "
        f"{(decisions_df['actual_ucb_rank'] <= 5).mean() * 100:.2f}%"
    )

    print()

    print(
        f"Actual MEAN Top-1           : "
        f"{(decisions_df['actual_mean_rank'] <= 1).mean() * 100:.2f}%"
    )

    print(
        f"Actual MEAN Top-3           : "
        f"{(decisions_df['actual_mean_rank'] <= 3).mean() * 100:.2f}%"
    )

    print(
        f"Actual MEAN Top-5           : "
        f"{(decisions_df['actual_mean_rank'] <= 5).mean() * 100:.2f}%"
    )

    # ========================================================
    # Save
    # ========================================================

    split_name = (
        name.lower()
    )

    decisions_df.to_csv(
        OUTPUT_DIR
        / (
            f"{split_name}_"
            "diagnostic_decisions.csv"
        ),
        index=False,
    )

    candidates_df.to_csv(
        OUTPUT_DIR
        / (
            f"{split_name}_"
            "diagnostic_candidates.csv"
        ),
        index=False,
    )

    print()

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / (
            f"{split_name}_"
            "diagnostic_decisions.csv"
        ),
    )

    print(
        " ",
        OUTPUT_DIR
        / (
            f"{split_name}_"
            "diagnostic_candidates.csv"
        ),
    )

    return (
        decisions_df,
        candidates_df,
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
        "- FAILURE DIAGNOSTICS"
    )

    print(
        f"ALPHA       : "
        f"{base.ALPHA}"
    )

    print(
        f"REWARD MODE : "
        f"{base.REWARD_MODE}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 기존 Step 4와 똑같은 데이터 / 모델 준비
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
    # Validation
    # ========================================================

    diagnose_split(
        "VALIDATION",
        model,
        processor,
        seed_df=train,
        eval_df=val,
        bullpen_map=(
            bullpen_map
        ),
        used_before_map=(
            used_before_map
        ),
    )

    # ========================================================
    # Test
    # ========================================================

    test_seed = (
        pd.concat(
            [
                train,
                val,
            ],
            ignore_index=True,
        )
    )

    diagnose_split(
        "TEST",
        model,
        processor,
        seed_df=test_seed,
        eval_df=test,
        bullpen_map=(
            bullpen_map
        ),
        used_before_map=(
            used_before_map
        ),
    )

    base.print_section(
        "DONE"
    )


if __name__ == "__main__":
    main()
