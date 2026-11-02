from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.quality import bandit_v3_li_quality_ope as common
from kbo_bandit.quality import (
    bandit_v3_li_quality_interaction_ope as b4_ope
)


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(
    "outputs/bandit/quality/"
    "v3_li_quality_interaction_common_q_ope"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


POLICIES = {
    "B0": "q_b0",
    "B1_LI": "q_b1_li",
    "B2_QUALITY": "q_b2_quality",
    "B3_LI_QUALITY": "q_b3_li_quality",
    "B4_LI_X_QUALITY": "q_b4_li_x_quality",
}


COMMON_Q_MODELS = {
    "B0_Q": "q_b0",
    "B1_Q": "q_b1_li",
    "B2_Q": "q_b2_quality",
    "B3_Q": "q_b3_li_quality",
    "B4_Q": "q_b4_li_x_quality",
}


COMPARISONS = [
    ("B4_LI_X_QUALITY", "B3_LI_QUALITY"),
    ("B4_LI_X_QUALITY", "B1_LI"),
    ("B4_LI_X_QUALITY", "B2_QUALITY"),
    ("B4_LI_X_QUALITY", "B0"),
]


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


# =============================================================================
# COMMON-Q POLICY DECISION TABLE
# =============================================================================

def build_common_q_policy_decisions(
    candidates: pd.DataFrame,
    policy_name: str,
    common_q_name: str,
) -> pd.DataFrame:

    policy_score_col = (
        POLICIES[
            policy_name
        ]
    )

    common_q_col = (
        COMMON_Q_MODELS[
            common_q_name
        ]
    )

    # -------------------------------------------------------------------------
    # Target policy가 실제로 선택하는 pitcher
    # -------------------------------------------------------------------------

    ordered = candidates.sort_values(
        [
            "decision_id",
            policy_score_col,
            "pitcher_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    selected = (
        ordered.groupby(
            "decision_id",
            as_index=False,
        )
        .first()
    )

    # -------------------------------------------------------------------------
    # DM:
    # target policy가 선택한 action을
    # common Q로 평가
    # -------------------------------------------------------------------------

    selected = (
        selected[
            [
                "decision_id",
                "pitcher_id",
                common_q_col,
            ]
        ]
        .rename(
            columns={
                "pitcher_id":
                    "selected_pitcher",

                common_q_col:
                    "dm_i",
            }
        )
    )

    # -------------------------------------------------------------------------
    # Logged actual action
    # -------------------------------------------------------------------------

    actual = (
        candidates[
            candidates[
                "is_actual"
            ]
            ==
            1
        ]
        .copy()
    )

    actual = actual.merge(
        selected,
        on="decision_id",
        how="left",
        validate="one_to_one",
    )

    actual[
        "pi_actual"
    ] = (
        actual[
            "pitcher_id"
        ]
        .astype(str)
        ==
        actual[
            "selected_pitcher"
        ]
        .astype(str)
    ).astype(float)

    actual[
        "mu_actual"
    ] = (
        actual[
            "behavior_prob"
        ]
        .clip(
            lower=1e-12
        )
    )

    # -------------------------------------------------------------------------
    # Actual action 역시 동일 common Q로 평가
    # -------------------------------------------------------------------------

    actual[
        "q_actual"
    ] = (
        actual[
            common_q_col
        ]
    )

    actual[
        "weight"
    ] = (
        actual[
            "pi_actual"
        ]
        /
        actual[
            "mu_actual"
        ]
    )

    actual[
        "residual"
    ] = (
        actual[
            "logged_reward"
        ]
        -
        actual[
            "q_actual"
        ]
    )

    actual[
        "policy"
    ] = (
        policy_name
    )

    actual[
        "common_q"
    ] = (
        common_q_name
    )

    return actual


# =============================================================================
# COMMON-Q SPLIT EVALUATION
# =============================================================================

def evaluate_split(
    split: str,
    candidates: pd.DataFrame,
):

    section(
        f"COMMON-Q OPE - {split.upper()}"
    )

    policy_rows = []
    comparison_rows = []

    # =========================================================================
    # Common reward model loop
    # =========================================================================

    for (
        common_q_name,
        common_q_col,
    ) in COMMON_Q_MODELS.items():

        print()
        print(
            f"[ Common Q = {common_q_name} ]"
        )

        base_policy_tables = {
            policy_name:
                build_common_q_policy_decisions(
                    candidates,
                    policy_name,
                    common_q_name,
                )
            for policy_name
            in POLICIES
        }

        # =====================================================================
        # Clipping sensitivity
        # =====================================================================

        for threshold in common.CLIP_THRESHOLDS:

            threshold_name = (
                common.clip_label(
                    threshold
                )
            )

            evaluated = {}

            # -----------------------------------------------------------------
            # Individual target policies
            # -----------------------------------------------------------------

            for (
                policy_name,
                policy_table,
            ) in base_policy_tables.items():

                (
                    evaluated_table,
                    metrics,
                ) = common.compute_ope(
                    policy_table,
                    threshold,
                )

                evaluated[
                    policy_name
                ] = evaluated_table

                policy_rows.append(
                    {
                        "split":
                            split,

                        "common_q":
                            common_q_name,

                        "threshold":
                            threshold_name,

                        "policy":
                            policy_name,

                        **metrics,
                    }
                )

            # -----------------------------------------------------------------
            # Paired comparisons
            # -----------------------------------------------------------------

            for (
                policy_a,
                policy_b,
            ) in COMPARISONS:

                paired = (
                    common.paired_game_bootstrap(
                        evaluated[
                            policy_a
                        ],
                        evaluated[
                            policy_b
                        ],
                    )
                )

                comparison_rows.append(
                    {
                        "split":
                            split,

                        "common_q":
                            common_q_name,

                        "threshold":
                            threshold_name,

                        "policy_a":
                            policy_a,

                        "policy_b":
                            policy_b,

                        "comparison":
                            (
                                f"{policy_a} - "
                                f"{policy_b}"
                            ),

                        **paired,
                    }
                )

    policy_result = pd.DataFrame(
        policy_rows
    )

    comparison_result = pd.DataFrame(
        comparison_rows
    )

    return (
        policy_result,
        comparison_result,
    )


# =============================================================================
# PRINT B4-B3 FOCUS
# =============================================================================

def print_b4_b3_focus(
    comparison_result: pd.DataFrame,
) -> None:

    section(
        "B4 - B3 COMMON-Q FOCUS"
    )

    focus = (
        comparison_result[
            comparison_result[
                "comparison"
            ]
            ==
            (
                "B4_LI_X_QUALITY - "
                "B3_LI_QUALITY"
            )
        ]
        .copy()
    )

    printable = (
        focus.copy()
    )

    printable[
        "prob_a_better"
    ] *= 100

    print(
        printable[
            [
                "split",
                "common_q",
                "threshold",
                "delta_dr",
                "ci_low",
                "ci_high",
                "prob_a_better",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "delta_dr":
                    lambda x:
                    f"{x:+.6f}",

                "ci_low":
                    lambda x:
                    f"{x:+.6f}",

                "ci_high":
                    lambda x:
                    f"{x:+.6f}",

                "prob_a_better":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )


# =============================================================================
# UNCLIPPED SUMMARY
# =============================================================================

def print_unclipped_summary(
    comparison_result: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "UNCLIPPED COMMON-Q SUMMARY"
    )

    summary = (
        comparison_result[
            (
                comparison_result[
                    "threshold"
                ]
                ==
                "unclipped"
            )
            &
            (
                comparison_result[
                    "comparison"
                ]
                ==
                (
                    "B4_LI_X_QUALITY - "
                    "B3_LI_QUALITY"
                )
            )
        ]
        .copy()
    )

    printable = (
        summary.copy()
    )

    printable[
        "prob_a_better"
    ] *= 100

    print(
        printable[
            [
                "split",
                "common_q",
                "delta_dr",
                "ci_low",
                "ci_high",
                "prob_a_better",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "delta_dr":
                    lambda x:
                    f"{x:+.6f}",

                "ci_low":
                    lambda x:
                    f"{x:+.6f}",

                "ci_high":
                    lambda x:
                    f"{x:+.6f}",

                "prob_a_better":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    return summary


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check_own_q(
    split: str,
    policy_result: pd.DataFrame,
) -> None:

    section(
        f"SANITY CHECK - {split.upper()}"
    )

    expected_pairs = [
        (
            "B0",
            "B0_Q",
        ),

        (
            "B1_LI",
            "B1_Q",
        ),

        (
            "B2_QUALITY",
            "B2_Q",
        ),

        (
            "B3_LI_QUALITY",
            "B3_Q",
        ),

        (
            "B4_LI_X_QUALITY",
            "B4_Q",
        ),
    ]

    for (
        policy,
        common_q,
    ) in expected_pairs:

        row = (
            policy_result[
                (
                    policy_result[
                        "split"
                    ]
                    ==
                    split
                )
                &
                (
                    policy_result[
                        "policy"
                    ]
                    ==
                    policy
                )
                &
                (
                    policy_result[
                        "common_q"
                    ]
                    ==
                    common_q
                )
                &
                (
                    policy_result[
                        "threshold"
                    ]
                    ==
                    "unclipped"
                )
            ]
        )

        if len(
            row
        ) != 1:
            raise ValueError(
                f"Missing sanity row: "
                f"{split} / {policy} / {common_q}"
            )

        value = float(
            row.iloc[
                0
            ][
                "dr"
            ]
        )

        print(
            f"{policy:18s} "
            f"under {common_q:6s} "
            f"DR={value:+.6f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 112
    )

    print(
        "STEP 5-2 - B4 COMMON-Q ROBUSTNESS"
    )

    print()

    print(
        "Primary question:"
    )

    print(
        "  Does B4 - B3 remain positive "
        "when both policies are evaluated "
        "with the same Q model?"
    )

    print()

    print(
        "Common Q models:"
    )

    print(
        "  B0_Q / B1_Q / B2_Q / B3_Q / B4_Q"
    )

    print()

    print(
        "Target policies still choose actions "
        "using their own learned scores."
    )

    print(
        "=" * 112
    )

    # =========================================================================
    # Logged reward lookup
    # =========================================================================

    reward_lookup = (
        common.build_logged_reward_lookup()
    )

    all_policy_results = []
    all_comparison_results = []

    # =========================================================================
    # Validation / Test
    # =========================================================================

    for split in [
        "validation",
        "test",
    ]:

        candidates = (
            b4_ope.build_ope_candidates(
                split,
                reward_lookup[
                    split
                ],
            )
        )

        (
            policy_result,
            comparison_result,
        ) = evaluate_split(
            split,
            candidates,
        )

        all_policy_results.append(
            policy_result
        )

        all_comparison_results.append(
            comparison_result
        )

    # =========================================================================
    # Combine
    # =========================================================================

    policy_summary = pd.concat(
        all_policy_results,
        ignore_index=True,
    )

    comparison_summary = pd.concat(
        all_comparison_results,
        ignore_index=True,
    )

    # =========================================================================
    # Sanity
    # =========================================================================

    sanity_check_own_q(
        "validation",
        policy_summary,
    )

    sanity_check_own_q(
        "test",
        policy_summary,
    )

    # =========================================================================
    # Focus
    # =========================================================================

    print_b4_b3_focus(
        comparison_summary
    )

    unclipped_summary = (
        print_unclipped_summary(
            comparison_summary
        )
    )

    # =========================================================================
    # Save
    # =========================================================================

    section(
        "SAVE"
    )

    policy_summary.to_csv(
        OUTPUT_DIR
        /
        "common_q_policy_values.csv",
        index=False,
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        /
        "common_q_paired_comparisons.csv",
        index=False,
    )

    unclipped_summary.to_csv(
        OUTPUT_DIR
        /
        "b4_b3_unclipped_summary.csv",
        index=False,
    )

    print(
        f"Saved to:\n  {OUTPUT_DIR}"
    )

    # =========================================================================
    # Final
    # =========================================================================

    section(
        "FINAL"
    )

    print(
        "Interpretation rule:"
    )

    print(
        "  Strong robustness:"
    )

    print(
        "    B4-B3 positive under most/all common-Q models"
    )

    print(
        "    Validation and Test directions agree"
    )

    print(
        "    clipping does not reverse the conclusion"
    )

    print()

    print(
        "  Weak / Q-sensitive:"
    )

    print(
        "    sign changes depending on common Q"
    )

    print(
        "    Validation/Test directions disagree"
    )

    print()

    print(
        "Do not choose the final policy "
        "before reviewing this table."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()