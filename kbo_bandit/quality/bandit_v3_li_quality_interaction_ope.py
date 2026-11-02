from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.quality import bandit_v3_li_quality_ope as b3_ope


# =============================================================================
# CONFIG
# =============================================================================

ABLATION_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_interaction"
)

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_interaction_ope"
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


PAIRWISE_COMPARISONS = [
    ("B4_LI_X_QUALITY", "B3_LI_QUALITY"),
    ("B4_LI_X_QUALITY", "B0"),
    ("B4_LI_X_QUALITY", "B1_LI"),
    ("B4_LI_X_QUALITY", "B2_QUALITY"),
]


# Step 4-1에서 이미 확정된 값
EXPECTED_UNCLIPPED = {
    "validation": {
        "B0": 0.005778,
        "B1_LI": 0.006927,
        "B2_QUALITY": 0.006392,
        "B3_LI_QUALITY": 0.006866,
    },

    "test": {
        "B0": 0.003222,
        "B1_LI": 0.003267,
        "B2_QUALITY": 0.003783,
        "B3_LI_QUALITY": 0.003666,
    },
}


REPRO_TOLERANCE = 0.00005


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


# =============================================================================
# 1. LOAD B4 CANDIDATES
# =============================================================================

def load_b4_candidates(
    split: str,
) -> pd.DataFrame:

    path = (
        ABLATION_DIR
        /
        f"{split}_candidates.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find B4 ablation output:\n  {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "def_team": str,
            "pitcher_id": str,
            "actual_pitcher": str,
        },
    )

    required = [
        "decision_id",
        "game_id",
        "def_team",
        "appearance_order",
        "pitcher_id",
        "actual_pitcher",
        "is_actual",
        "q_b0",
        "q_b1_li",
        "q_b2_quality",
        "q_b3_li_quality",
        "q_b4_li_x_quality",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    df["game_id"] = (
        df["game_id"]
        .astype(str)
    )

    df["def_team"] = (
        df["def_team"]
        .astype(str)
    )

    df["pitcher_id"] = (
        df["pitcher_id"]
        .apply(
            b3_ope.normalize_id
        )
    )

    df["actual_pitcher"] = (
        df["actual_pitcher"]
        .apply(
            b3_ope.normalize_id
        )
    )

    df["appearance_order"] = (
        pd.to_numeric(
            df["appearance_order"],
            errors="raise",
        )
        .astype(int)
    )

    df["decision_id"] = (
        pd.to_numeric(
            df["decision_id"],
            errors="raise",
        )
        .astype(int)
    )

    df["is_actual"] = (
        pd.to_numeric(
            df["is_actual"],
            errors="raise",
        )
        .astype(int)
    )

    for column in POLICIES.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


# =============================================================================
# 2. BUILD OPE CANDIDATES
# =============================================================================

def build_ope_candidates(
    split: str,
    reward_lookup: pd.DataFrame,
) -> pd.DataFrame:

    section(
        f"2. BUILD OPE CANDIDATES - {split.upper()}"
    )

    candidates = (
        load_b4_candidates(
            split
        )
    )

    behavior = (
        b3_ope.load_behavior_candidates(
            split
        )
    )

    print(
        f"B4 candidate rows       : "
        f"{len(candidates):,}"
    )

    print(
        f"Behavior candidate rows : "
        f"{len(behavior):,}"
    )

    merged = candidates.merge(
        behavior,
        on=b3_ope.KEYS,
        how="left",
        validate="one_to_one",
    )

    missing_behavior = int(
        merged[
            "behavior_prob"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing behavior prob   : "
        f"{missing_behavior:,}"
    )

    if missing_behavior > 0:
        raise ValueError(
            "Behavior propensity join failed."
        )

    merged = merged.merge(
        reward_lookup,
        on=b3_ope.KEYS,
        how="left",
        validate="many_to_one",
    )

    actual = (
        merged[
            merged["is_actual"]
            ==
            1
        ]
    )

    missing_reward = int(
        actual[
            "logged_reward"
        ]
        .isna()
        .sum()
    )

    print(
        f"Decisions               : "
        f"{len(actual):,}"
    )

    print(
        f"Missing logged reward   : "
        f"{missing_reward:,}"
    )

    if missing_reward > 0:
        raise ValueError(
            "Logged reward join failed."
        )

    probability_sum = (
        merged.groupby(
            "decision_id"
        )[
            "behavior_prob"
        ]
        .sum()
    )

    max_error = float(
        (
            probability_sum
            -
            1.0
        )
        .abs()
        .max()
    )

    print(
        f"Behavior prob max error : "
        f"{max_error:.12f}"
    )

    if max_error > 1e-5:
        raise ValueError(
            "Behavior probabilities do not sum to 1."
        )

    return merged


# =============================================================================
# 3. BUILD POLICY DECISIONS
# =============================================================================

def build_policy_decisions(
    candidates: pd.DataFrame,
    policy_name: str,
) -> pd.DataFrame:

    score_col = (
        POLICIES[
            policy_name
        ]
    )

    ordered = candidates.sort_values(
        [
            "decision_id",
            score_col,
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

    selected = (
        selected[
            [
                "decision_id",
                "pitcher_id",
                score_col,
            ]
        ]
        .rename(
            columns={
                "pitcher_id":
                    "selected_pitcher",

                score_col:
                    "dm_i",
            }
        )
    )

    actual = (
        candidates[
            candidates["is_actual"]
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

    actual["pi_actual"] = (
        actual["pitcher_id"]
        .astype(str)
        ==
        actual["selected_pitcher"]
        .astype(str)
    ).astype(float)

    actual["mu_actual"] = (
        actual["behavior_prob"]
        .clip(
            lower=1e-12
        )
    )

    actual["q_actual"] = (
        actual[
            score_col
        ]
    )

    actual["weight"] = (
        actual["pi_actual"]
        /
        actual["mu_actual"]
    )

    actual["residual"] = (
        actual["logged_reward"]
        -
        actual["q_actual"]
    )

    actual["policy"] = (
        policy_name
    )

    return actual


# =============================================================================
# 4. SPLIT EVALUATION
# =============================================================================

def evaluate_split(
    split: str,
    candidates: pd.DataFrame,
):

    section(
        f"3. B0 / B1 / B2 / B3 / B4 OPE - {split.upper()}"
    )

    policy_base = {
        policy:
            build_policy_decisions(
                candidates,
                policy,
            )
        for policy
        in POLICIES
    }

    policy_rows = []
    pair_rows = []

    evaluated_by_threshold = {}

    for threshold in b3_ope.CLIP_THRESHOLDS:

        threshold_name = (
            b3_ope.clip_label(
                threshold
            )
        )

        evaluated_by_threshold[
            threshold_name
        ] = {}

        # ---------------------------------------------------------------------
        # Individual policy OPE
        # ---------------------------------------------------------------------

        for (
            policy_name,
            decisions,
        ) in policy_base.items():

            (
                evaluated,
                metrics,
            ) = (
                b3_ope.compute_ope(
                    decisions,
                    threshold,
                )
            )

            evaluated_by_threshold[
                threshold_name
            ][
                policy_name
            ] = evaluated

            policy_rows.append(
                {
                    "split":
                        split,

                    "threshold":
                        threshold_name,

                    "policy":
                        policy_name,

                    **metrics,
                }
            )

        # ---------------------------------------------------------------------
        # Paired comparisons
        # ---------------------------------------------------------------------

        for (
            policy_a,
            policy_b,
        ) in PAIRWISE_COMPARISONS:

            comparison = (
                b3_ope.paired_game_bootstrap(
                    evaluated_by_threshold[
                        threshold_name
                    ][
                        policy_a
                    ],
                    evaluated_by_threshold[
                        threshold_name
                    ][
                        policy_b
                    ],
                )
            )

            pair_rows.append(
                {
                    "split":
                        split,

                    "threshold":
                        threshold_name,

                    "policy_a":
                        policy_a,

                    "policy_b":
                        policy_b,

                    "comparison":
                        f"{policy_a} - {policy_b}",

                    **comparison,
                }
            )

    policy_result = pd.DataFrame(
        policy_rows
    )

    pair_result = pd.DataFrame(
        pair_rows
    )

    # =========================================================================
    # Print policy value
    # =========================================================================

    print()
    print(
        "POLICY VALUE"
    )

    pivot = (
        policy_result.pivot(
            index="threshold",
            columns="policy",
            values="dr",
        )
        .reset_index()
    )

    print(
        pivot.to_string(
            index=False,
            formatters={
                policy:
                    lambda x:
                    f"{x:+.6f}"
                for policy
                in POLICIES
            },
        )
    )

    # =========================================================================
    # Print paired
    # =========================================================================

    print()
    print(
        "PAIRED POLICY DIFFERENCES"
    )

    printable = (
        pair_result.copy()
    )

    printable[
        "prob_a_better"
    ] *= 100

    print(
        printable[
            [
                "threshold",
                "comparison",
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

    # =========================================================================
    # ESS
    # =========================================================================

    print()
    print(
        "ESS"
    )

    ess = (
        policy_result.pivot(
            index="threshold",
            columns="policy",
            values="ess",
        )
        .reset_index()
    )

    print(
        ess.to_string(
            index=False,
            formatters={
                policy:
                    lambda x:
                    f"{x:.1f}"
                for policy
                in POLICIES
            },
        )
    )

    return (
        policy_result,
        pair_result,
        policy_base,
    )


# =============================================================================
# 5. REPRODUCTION CHECK
# =============================================================================

def check_reproduction(
    split: str,
    policy_result: pd.DataFrame,
) -> bool:

    section(
        f"4. REPRODUCTION CHECK - {split.upper()}"
    )

    expected = (
        EXPECTED_UNCLIPPED[
            split
        ]
    )

    passed = True

    for (
        policy,
        expected_value,
    ) in expected.items():

        row = (
            policy_result[
                (
                    policy_result[
                        "threshold"
                    ]
                    ==
                    "unclipped"
                )
                &
                (
                    policy_result[
                        "policy"
                    ]
                    ==
                    policy
                )
            ]
        )

        if len(row) != 1:
            raise ValueError(
                f"Cannot find {split} {policy} unclipped row."
            )

        current = float(
            row.iloc[
                0
            ][
                "dr"
            ]
        )

        difference = abs(
            current
            -
            expected_value
        )

        ok = (
            difference
            <=
            REPRO_TOLERANCE
        )

        print(
            f"{policy:16s} | "
            f"expected={expected_value:+.6f} | "
            f"current={current:+.6f} | "
            f"diff={difference:.6f} | "
            f"{'PASS' if ok else 'WARNING'}"
        )

        if not ok:
            passed = False

    return passed


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 108
    )

    print(
        "STEP 5-1 - B4 LI x QUALITY_P INTERACTION OPE"
    )

    print()

    print(
        "Primary question:"
    )

    print(
        "  Does B4 improve B3?"
    )

    print()

    print(
        "B3:"
    )

    print(
        "  B0 + LI + LI x pitcher + quality_p"
    )

    print()

    print(
        "B4:"
    )

    print(
        "  B3 + LI x quality_p"
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        "  DM / IPS / SNIPS / DR"
    )

    print(
        "  clipping robustness"
    )

    print(
        "  paired game bootstrap"
    )

    print(
        "=" * 108
    )

    # =========================================================================
    # Logged rewards
    # =========================================================================

    reward_lookup = (
        b3_ope.build_logged_reward_lookup()
    )

    all_policy_results = []
    all_pair_results = []

    reproduction_status = {}

    # =========================================================================
    # Validation / Test
    # =========================================================================

    for split in [
        "validation",
        "test",
    ]:

        candidates = (
            build_ope_candidates(
                split,
                reward_lookup[
                    split
                ],
            )
        )

        (
            policy_result,
            pair_result,
            policy_base,
        ) = evaluate_split(
            split,
            candidates,
        )

        reproduction_status[
            split
        ] = (
            check_reproduction(
                split,
                policy_result,
            )
        )

        all_policy_results.append(
            policy_result
        )

        all_pair_results.append(
            pair_result
        )

        # Decision-level policy tables
        for (
            policy_name,
            decisions,
        ) in policy_base.items():

            decisions.to_csv(
                OUTPUT_DIR
                /
                f"{split}_{policy_name.lower()}_decisions.csv",
                index=False,
            )

    # =========================================================================
    # Combine
    # =========================================================================

    policy_summary = pd.concat(
        all_policy_results,
        ignore_index=True,
    )

    pair_summary = pd.concat(
        all_pair_results,
        ignore_index=True,
    )

    # =========================================================================
    # Focus on B4
    # =========================================================================

    section(
        "FINAL - B4 POLICY VALUE FOCUS"
    )

    focus = (
        pair_summary[
            pair_summary[
                "comparison"
            ]
            .isin(
                [
                    "B4_LI_X_QUALITY - B3_LI_QUALITY",
                    "B4_LI_X_QUALITY - B0",
                    "B4_LI_X_QUALITY - B1_LI",
                    "B4_LI_X_QUALITY - B2_QUALITY",
                ]
            )
        ]
        .copy()
    )

    focus[
        "prob_a_better"
    ] *= 100

    print(
        focus[
            [
                "split",
                "threshold",
                "comparison",
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

    # =========================================================================
    # Save
    # =========================================================================

    section(
        "SAVE"
    )

    policy_summary.to_csv(
        OUTPUT_DIR
        /
        "policy_value_summary.csv",
        index=False,
    )

    pair_summary.to_csv(
        OUTPUT_DIR
        /
        "paired_policy_comparisons.csv",
        index=False,
    )

    focus.to_csv(
        OUTPUT_DIR
        /
        "b4_focus_comparisons.csv",
        index=False,
    )

    print(
        f"Saved to:\n  {OUTPUT_DIR}"
    )

    # =========================================================================
    # Final status
    # =========================================================================

    section(
        "FINAL STATUS"
    )

    for split in [
        "validation",
        "test",
    ]:

        print(
            f"{split:10s} reproduction : "
            +
            (
                "PASS"
                if reproduction_status[
                    split
                ]
                else "WARNING"
            )
        )

    print()
    print(
        "Primary B4 decision rule:"
    )

    print(
        "  B4 - B3 paired DR"
    )

    print(
        "  Validation/Test direction consistency"
    )

    print(
        "  95% CI"
    )

    print(
        "  clipping robustness"
    )

    print(
        "  ESS"
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()