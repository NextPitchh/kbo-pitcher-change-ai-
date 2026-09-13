from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

INPUT_DIR = Path(
    "bandit_output_v3_li_ope"
)

OUTPUT_DIR = Path(
    "bandit_output_v3_li_segment_ope"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


CLIP_THRESHOLDS = [
    10.0,
    20.0,
    30.0,
    50.0,
    100.0,
    None,
]


LI_BINS = [
    -np.inf,
    0.5,
    1.5,
    2.5,
    np.inf,
]

LI_LABELS = [
    "low_<0.5",
    "normal_0.5-1.5",
    "high_1.5-2.5",
    "critical_>=2.5",
]


BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42


# ============================================================
# Helper
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def clip_label(threshold):

    if threshold is None:
        return "unclipped"

    return f"clip_{int(threshold)}"


# ============================================================
# Load decision-level OPE data
# ============================================================

def load_policy_decisions(
    split,
    model,
):

    path = (
        INPUT_DIR
        / f"{split}_{model}_decisions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "pitcher_id": str,
            "selected_pitcher": str,
        },
    )

    required = [
        "decision_id",
        "game_id",
        "leverage_index",
        "logged_reward",
        "dm_i",
        "q_actual",
        "mu_actual",
        "pi_actual",
        "weight",
        "residual",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    numeric = [
        "decision_id",
        "leverage_index",
        "logged_reward",
        "dm_i",
        "q_actual",
        "mu_actual",
        "pi_actual",
        "weight",
        "residual",
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    df["decision_id"] = (
        df["decision_id"]
        .astype(int)
    )

    return df


# ============================================================
# Verify B0 / B1 alignment
# ============================================================

def verify_alignment(
    b0,
    b1,
    split,
):

    section(
        f"VERIFY B0 / B1 ALIGNMENT - {split.upper()}"
    )

    print(
        f"B0 rows : {len(b0):,}"
    )

    print(
        f"B1 rows : {len(b1):,}"
    )

    if len(b0) != len(b1):

        raise ValueError(
            "B0/B1 row counts differ."
        )

    left = (
        b0[
            [
                "decision_id",
                "game_id",
                "leverage_index",
                "logged_reward",
            ]
        ]
        .rename(
            columns={
                "leverage_index":
                    "li_b0",

                "logged_reward":
                    "reward_b0",
            }
        )
    )

    right = (
        b1[
            [
                "decision_id",
                "game_id",
                "leverage_index",
                "logged_reward",
            ]
        ]
        .rename(
            columns={
                "leverage_index":
                    "li_b1",

                "logged_reward":
                    "reward_b1",
            }
        )
    )

    merged = left.merge(
        right,
        on=[
            "decision_id",
            "game_id",
        ],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    bad_merge = int(
        (
            merged["_merge"]
            != "both"
        ).sum()
    )

    print(
        f"Unmatched decisions : {bad_merge:,}"
    )

    if bad_merge > 0:

        raise ValueError(
            "B0/B1 decision identity differs."
        )

    li_match = np.allclose(
        merged["li_b0"],
        merged["li_b1"],
        rtol=0,
        atol=1e-12,
    )

    reward_match = np.allclose(
        merged["reward_b0"],
        merged["reward_b1"],
        rtol=0,
        atol=1e-12,
    )

    print(
        f"LI identical     : {li_match}"
    )

    print(
        f"Reward identical : {reward_match}"
    )

    if not li_match:

        raise ValueError(
            "B0/B1 LI differs."
        )

    if not reward_match:

        raise ValueError(
            "B0/B1 logged reward differs."
        )


# ============================================================
# Add LI segment
# ============================================================

def add_segment(df):

    work = df.copy()

    work["li_bucket"] = pd.cut(
        work["leverage_index"],
        bins=LI_BINS,
        labels=LI_LABELS,
        right=False,
    )

    if (
        work["li_bucket"]
        .isna()
        .any()
    ):

        raise ValueError(
            "LI bucket assignment failed."
        )

    return work


# ============================================================
# Compute DR contribution for one clipping threshold
# ============================================================

def compute_dr(
    df,
    threshold,
):

    work = df.copy()

    if threshold is None:

        work["weight_used"] = (
            work["weight"]
        )

    else:

        work["weight_used"] = (
            work["weight"]
            .clip(
                upper=threshold
            )
        )

    work["dr_i"] = (
        work["dm_i"]
        +
        work["weight_used"]
        *
        work["residual"]
    )

    work["ips_i"] = (
        work["weight_used"]
        *
        work["logged_reward"]
    )

    weights = (
        work["weight_used"]
        .to_numpy(
            dtype=float
        )
    )

    denominator = float(
        np.sum(
            weights ** 2
        )
    )

    if denominator > 0:

        ess = float(
            weights.sum() ** 2
            /
            denominator
        )

    else:

        ess = 0.0

    weight_sum = float(
        work["weight_used"]
        .sum()
    )

    if weight_sum > 0:

        snips = float(
            work["ips_i"]
            .sum()
            /
            weight_sum
        )

    else:

        snips = np.nan

    metrics = {
        "n":
            int(
                len(work)
            ),

        "agreement":
            float(
                work["pi_actual"]
                .mean()
            ),

        "dm":
            float(
                work["dm_i"]
                .mean()
            ),

        "ips":
            float(
                work["ips_i"]
                .mean()
            ),

        "snips":
            snips,

        "dr":
            float(
                work["dr_i"]
                .mean()
            ),

        "ess":
            ess,

        "ess_fraction":
            float(
                ess
                /
                len(work)
            ),
    }

    return (
        work,
        metrics,
    )


# ============================================================
# Paired game bootstrap
# ============================================================

def paired_bootstrap(
    b0,
    b1,
    seed_offset,
):

    left = (
        b0[
            [
                "decision_id",
                "game_id",
                "dr_i",
            ]
        ]
        .rename(
            columns={
                "dr_i":
                    "dr_b0"
            }
        )
    )

    right = (
        b1[
            [
                "decision_id",
                "game_id",
                "dr_i",
            ]
        ]
        .rename(
            columns={
                "dr_i":
                    "dr_b1"
            }
        )
    )

    paired = left.merge(
        right,
        on=[
            "decision_id",
            "game_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if (
        len(paired) != len(b0)
        or
        len(paired) != len(b1)
    ):

        raise ValueError(
            "Paired B0/B1 rows do not align."
        )

    paired["delta_dr"] = (
        paired["dr_b1"]
        -
        paired["dr_b0"]
    )

    point = float(
        paired["delta_dr"]
        .mean()
    )

    # --------------------------------------------------------
    # game-level cluster bootstrap
    # --------------------------------------------------------

    games = (
        paired.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            n=(
                "decision_id",
                "size",
            ),

            delta_sum=(
                "delta_dr",
                "sum",
            ),
        )
    )

    n_games = len(
        games
    )

    n_values = (
        games["n"]
        .to_numpy(
            dtype=float
        )
    )

    delta_values = (
        games["delta_sum"]
        .to_numpy(
            dtype=float
        )
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        +
        seed_offset
    )

    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_REPS
    ):

        idx = rng.integers(
            0,
            n_games,
            size=n_games,
        )

        boot[i] = (
            delta_values[idx]
            .sum()
            /
            n_values[idx]
            .sum()
        )

    return {
        "delta_dr":
            point,

        "ci_low":
            float(
                np.quantile(
                    boot,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    boot,
                    0.975,
                )
            ),

        "prob_b1_better":
            float(
                (
                    boot > 0
                )
                .mean()
            ),

        "n_games":
            int(
                n_games
            ),
    }


# ============================================================
# Evaluate one split
# ============================================================

def evaluate_split(
    split,
    b0,
    b1,
):

    section(
        f"LI-SEGMENT OPE - {split.upper()}"
    )

    b0 = add_segment(
        b0
    )

    b1 = add_segment(
        b1
    )

    rows = []

    for bucket_index, bucket in enumerate(
        LI_LABELS
    ):

        b0_segment = (
            b0[
                b0["li_bucket"]
                == bucket
            ]
            .copy()
        )

        b1_segment = (
            b1[
                b1["li_bucket"]
                == bucket
            ]
            .copy()
        )

        if len(b0_segment) == 0:

            continue

        if (
            len(b0_segment)
            !=
            len(b1_segment)
        ):

            raise ValueError(
                f"Segment size differs: {bucket}"
            )

        for clip_index, threshold in enumerate(
            CLIP_THRESHOLDS
        ):

            (
                b0_eval,
                m0,
            ) = compute_dr(
                b0_segment,
                threshold,
            )

            (
                b1_eval,
                m1,
            ) = compute_dr(
                b1_segment,
                threshold,
            )

            paired = paired_bootstrap(
                b0_eval,
                b1_eval,
                seed_offset=(
                    bucket_index
                    * 100
                    +
                    clip_index
                ),
            )

            rows.append(
                {
                    "split":
                        split,

                    "li_bucket":
                        bucket,

                    "threshold":
                        clip_label(
                            threshold
                        ),

                    "n":
                        int(
                            len(
                                b0_segment
                            )
                        ),

                    "avg_li":
                        float(
                            b0_segment[
                                "leverage_index"
                            ]
                            .mean()
                        ),

                    "b0_agreement":
                        m0[
                            "agreement"
                        ],

                    "b1_agreement":
                        m1[
                            "agreement"
                        ],

                    "b0_dm":
                        m0[
                            "dm"
                        ],

                    "b1_dm":
                        m1[
                            "dm"
                        ],

                    "b0_ips":
                        m0[
                            "ips"
                        ],

                    "b1_ips":
                        m1[
                            "ips"
                        ],

                    "b0_snips":
                        m0[
                            "snips"
                        ],

                    "b1_snips":
                        m1[
                            "snips"
                        ],

                    "b0_dr":
                        m0[
                            "dr"
                        ],

                    "b1_dr":
                        m1[
                            "dr"
                        ],

                    "delta_dr":
                        paired[
                            "delta_dr"
                        ],

                    "ci_low":
                        paired[
                            "ci_low"
                        ],

                    "ci_high":
                        paired[
                            "ci_high"
                        ],

                    "prob_b1_better":
                        paired[
                            "prob_b1_better"
                        ],

                    "b0_ess":
                        m0[
                            "ess"
                        ],

                    "b1_ess":
                        m1[
                            "ess"
                        ],

                    "b0_ess_fraction":
                        m0[
                            "ess_fraction"
                        ],

                    "b1_ess_fraction":
                        m1[
                            "ess_fraction"
                        ],

                    "n_games":
                        paired[
                            "n_games"
                        ],
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# Print result
# ============================================================

def print_result(
    split,
    result,
):

    section(
        f"RESULT - {split.upper()}"
    )

    focus = (
        result[
            [
                "li_bucket",
                "threshold",
                "n",

                "b0_dr",
                "b1_dr",
                "delta_dr",

                "ci_low",
                "ci_high",

                "prob_b1_better",

                "b0_ess",
                "b1_ess",
            ]
        ]
        .copy()
    )

    focus[
        "prob_b1_better"
    ] *= 100

    print(
        focus.to_string(
            index=False,
            formatters={
                "b0_dr":
                    lambda x:
                    f"{x:+.6f}",

                "b1_dr":
                    lambda x:
                    f"{x:+.6f}",

                "delta_dr":
                    lambda x:
                    f"{x:+.6f}",

                "ci_low":
                    lambda x:
                    f"{x:+.6f}",

                "ci_high":
                    lambda x:
                    f"{x:+.6f}",

                "prob_b1_better":
                    lambda x:
                    f"{x:.2f}%",

                "b0_ess":
                    lambda x:
                    f"{x:.1f}",

                "b1_ess":
                    lambda x:
                    f"{x:.1f}",
            },
        )
    )


# ============================================================
# High / Critical focus
# ============================================================

def print_high_leverage_focus(
    validation,
    test,
):

    section(
        "HIGH / CRITICAL LEVERAGE OPE FOCUS"
    )

    combined = pd.concat(
        [
            validation,
            test,
        ],
        ignore_index=True,
    )

    focus = combined[
        (
            combined["li_bucket"]
            .isin(
                [
                    "high_1.5-2.5",
                    "critical_>=2.5",
                ]
            )
        )
        &
        (
            combined["threshold"]
            .isin(
                [
                    "clip_10",
                    "clip_50",
                    "unclipped",
                ]
            )
        )
    ].copy()

    focus[
        "prob_b1_better"
    ] *= 100

    print(
        focus[
            [
                "split",
                "li_bucket",
                "threshold",
                "n",

                "delta_dr",
                "ci_low",
                "ci_high",

                "prob_b1_better",

                "b0_ess",
                "b1_ess",
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

                "prob_b1_better":
                    lambda x:
                    f"{x:.2f}%",

                "b0_ess":
                    lambda x:
                    f"{x:.1f}",

                "b1_ess":
                    lambda x:
                    f"{x:.1f}",
            },
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "STEP 13-6 - LI-SEGMENT B0 VS B1 OFFLINE POLICY EVALUATION"
    )

    print()

    print(
        "Goal:"
    )

    print(
        "Determine whether LI improves estimated policy value"
    )

    print(
        "specifically in Low / Normal / High / Critical leverage situations."
    )

    print()

    print(
        "No model is retrained in this step."
    )

    print(
        "Step 13-5 B0/B1 decisions are reused."
    )

    print(
        "=" * 100
    )

    # ========================================================
    # Load
    # ========================================================

    validation_b0 = (
        load_policy_decisions(
            "validation",
            "b0",
        )
    )

    validation_b1 = (
        load_policy_decisions(
            "validation",
            "b1",
        )
    )

    test_b0 = (
        load_policy_decisions(
            "test",
            "b0",
        )
    )

    test_b1 = (
        load_policy_decisions(
            "test",
            "b1",
        )
    )

    # ========================================================
    # Alignment
    # ========================================================

    verify_alignment(
        validation_b0,
        validation_b1,
        "validation",
    )

    verify_alignment(
        test_b0,
        test_b1,
        "test",
    )

    # ========================================================
    # Evaluate
    # ========================================================

    validation_result = (
        evaluate_split(
            "validation",
            validation_b0,
            validation_b1,
        )
    )

    test_result = (
        evaluate_split(
            "test",
            test_b0,
            test_b1,
        )
    )

    # ========================================================
    # Print
    # ========================================================

    print_result(
        "validation",
        validation_result,
    )

    print_result(
        "test",
        test_result,
    )

    print_high_leverage_focus(
        validation_result,
        test_result,
    )

    # ========================================================
    # Save
    # ========================================================

    section(
        "SAVE"
    )

    validation_result.to_csv(
        OUTPUT_DIR
        / "validation_li_segment_ope.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_li_segment_ope.csv",
        index=False,
    )

    combined = pd.concat(
        [
            validation_result,
            test_result,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "li_segment_ope_all.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_li_segment_ope.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_li_segment_ope.csv"
    )

    # ========================================================
    # Done
    # ========================================================

    section(
        "DONE"
    )

    print(
        "Interpretation rule:"
    )

    print(
        "Do not design gated B2 from Agreement alone."
    )

    print(
        "Use LI-segment DR / paired CI to decide whether gating is justified."
    )


if __name__ == "__main__":
    main()