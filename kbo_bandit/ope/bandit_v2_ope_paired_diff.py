from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

INPUT_DIR = Path(
    "outputs/bandit/ope/v2_ope"
)

OUTPUT_DIR = Path(
    "outputs/bandit/ope/v2_ope_paired_diff"
)

OUTPUT_DIR.mkdir(
    parents=True,
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


BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42


POLICY_PAIRS = [
    (
        "reward_policy",
        "heuristic",
    ),
    (
        "reward_policy",
        "random",
    ),
    (
        "heuristic",
        "random",
    ),
]


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def threshold_label(
    threshold,
):

    if threshold is None:
        return "unclipped"

    return (
        f"clip_{int(threshold)}"
    )


# ============================================================
# 1. Load OPE decision contributions
# ============================================================

def load_ope_decisions(
    split,
):

    path = (
        INPUT_DIR
        / f"{split}_ope_decisions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "game_id":
                str,

            "policy":
                str,
        },
    )

    required = [
        "game_id",
        "decision_id",
        "policy",

        "logged_reward",
        "q_actual",
        "dm_i",
        "residual",

        "pi_actual",
        "mu_actual",
        "weight",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    numeric_cols = [
        "decision_id",
        "logged_reward",
        "q_actual",
        "dm_i",
        "residual",
        "pi_actual",
        "mu_actual",
        "weight",
    ]

    for col in numeric_cols:

        df[col] = (
            pd.to_numeric(
                df[col],
                errors="raise",
            )
        )

    print(
        f"{split:10s}: "
        f"{len(df):,} policy-decision rows"
    )

    print(
        "Policies   :",
        sorted(
            df[
                "policy"
            ]
            .unique()
            .tolist()
        ),
    )

    return df


# ============================================================
# 2. DR contribution under clipping
# ============================================================

def compute_dr_contribution(
    df,
    threshold,
):

    out = (
        df.copy()
    )

    if threshold is None:

        out[
            "weight_used"
        ] = (
            out[
                "weight"
            ]
        )

    else:

        out[
            "weight_used"
        ] = (
            out[
                "weight"
            ]
            .clip(
                upper=threshold
            )
        )

    # --------------------------------------------------------
    # DR_i
    #
    # Q(s, pi(s))
    # +
    # w * (r - Q(s, logged action))
    # --------------------------------------------------------

    out[
        "dr_used"
    ] = (
        out[
            "dm_i"
        ]
        +
        out[
            "weight_used"
        ]
        *
        out[
            "residual"
        ]
    )

    return out


# ============================================================
# 3. Align two policies decision-by-decision
# ============================================================

def build_paired_decisions(
    df,
    policy_a,
    policy_b,
    threshold,
):

    work = (
        compute_dr_contribution(
            df,
            threshold,
        )
    )

    a = (
        work[
            work[
                "policy"
            ]
            == policy_a
        ][
            [
                "game_id",
                "decision_id",
                "dr_used",
            ]
        ]
        .copy()
    )

    a = (
        a.rename(
            columns={
                "dr_used":
                    "dr_a"
            }
        )
    )

    b = (
        work[
            work[
                "policy"
            ]
            == policy_b
        ][
            [
                "game_id",
                "decision_id",
                "dr_used",
            ]
        ]
        .copy()
    )

    b = (
        b.rename(
            columns={
                "dr_used":
                    "dr_b"
            }
        )
    )

    paired = (
        a.merge(
            b,
            on=[
                "game_id",
                "decision_id",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(paired)
        != len(a)
        or len(paired)
        != len(b)
    ):

        raise ValueError(
            "Policy decision sets are not aligned."
        )

    paired[
        "delta_dr"
    ] = (
        paired[
            "dr_a"
        ]
        -
        paired[
            "dr_b"
        ]
    )

    return paired


# ============================================================
# 4. Paired game bootstrap
# ============================================================

def paired_game_bootstrap(
    paired,
):

    # --------------------------------------------------------
    # 경기 내부 decision은 독립이라고 보기 어려우므로
    # game 단위로 묶는다.
    # --------------------------------------------------------

    game_agg = (
        paired.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            n_decisions=(
                "decision_id",
                "size",
            ),

            delta_sum=(
                "delta_dr",
                "sum",
            ),
        )
    )

    n_games = (
        len(
            game_agg
        )
    )

    n_arr = (
        game_agg[
            "n_decisions"
        ]
        .to_numpy(
            dtype=float
        )
    )

    delta_arr = (
        game_agg[
            "delta_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rng = (
        np.random.default_rng(
            BOOTSTRAP_SEED
        )
    )

    estimates = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_REPS
    ):

        sampled = (
            rng.integers(
                0,
                n_games,
                size=n_games,
            )
        )

        total_n = float(
            n_arr[
                sampled
            ]
            .sum()
        )

        total_delta = float(
            delta_arr[
                sampled
            ]
            .sum()
        )

        estimates[i] = (
            total_delta
            /
            total_n
        )

    ci_low = float(
        np.quantile(
            estimates,
            0.025,
        )
    )

    ci_high = float(
        np.quantile(
            estimates,
            0.975,
        )
    )

    # --------------------------------------------------------
    # Bootstrap probability
    # --------------------------------------------------------

    prob_positive = float(
        (
            estimates
            >
            0
        )
        .mean()
    )

    prob_negative = float(
        (
            estimates
            <
            0
        )
        .mean()
    )

    bootstrap_mean = float(
        estimates.mean()
    )

    bootstrap_std = float(
        estimates.std(
            ddof=1
        )
    )

    return {
        "n_games":
            int(
                n_games
            ),

        "bootstrap_reps":
            int(
                BOOTSTRAP_REPS
            ),

        "bootstrap_mean":
            bootstrap_mean,

        "bootstrap_std":
            bootstrap_std,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "prob_positive":
            prob_positive,

        "prob_negative":
            prob_negative,
    }


# ============================================================
# 5. Evaluate one policy pair
# ============================================================

def evaluate_pair(
    split,
    df,
    policy_a,
    policy_b,
    threshold,
):

    paired = (
        build_paired_decisions(
            df,
            policy_a,
            policy_b,
            threshold,
        )
    )

    dr_a = float(
        paired[
            "dr_a"
        ]
        .mean()
    )

    dr_b = float(
        paired[
            "dr_b"
        ]
        .mean()
    )

    delta = float(
        paired[
            "delta_dr"
        ]
        .mean()
    )

    bootstrap = (
        paired_game_bootstrap(
            paired
        )
    )

    return {
        "split":
            split,

        "policy_a":
            policy_a,

        "policy_b":
            policy_b,

        "comparison":
            (
                f"{policy_a}"
                "_minus_"
                f"{policy_b}"
            ),

        "threshold_label":
            threshold_label(
                threshold
            ),

        "threshold":
            (
                np.inf
                if threshold is None
                else float(
                    threshold
                )
            ),

        "n_decisions":
            int(
                len(paired)
            ),

        "dr_a":
            dr_a,

        "dr_b":
            dr_b,

        "delta_dr":
            delta,

        **bootstrap,
    }


# ============================================================
# 6. Evaluate split
# ============================================================

def evaluate_split(
    split,
    df,
):

    print_section(
        f"PAIRED POLICY DIFFERENCE - {split.upper()}"
    )

    rows = []

    for (
        policy_a,
        policy_b,
    ) in POLICY_PAIRS:

        print()
        print(
            "-" * 92
        )

        print(
            f"{policy_a} - {policy_b}"
        )

        print(
            "-" * 92
        )

        pair_rows = []

        for threshold in (
            CLIP_THRESHOLDS
        ):

            result = (
                evaluate_pair(
                    split,
                    df,
                    policy_a,
                    policy_b,
                    threshold,
                )
            )

            rows.append(
                result
            )

            pair_rows.append(
                result
            )

        display = pd.DataFrame(
            pair_rows
        )

        display[
            "prob_positive"
        ] *= 100

        print(
            display[
                [
                    "threshold_label",
                    "dr_a",
                    "dr_b",
                    "delta_dr",
                    "ci_low",
                    "ci_high",
                    "prob_positive",
                    "n_games",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "dr_a":
                        lambda x:
                        f"{x:.6f}",

                    "dr_b":
                        lambda x:
                        f"{x:.6f}",

                    "delta_dr":
                        lambda x:
                        f"{x:+.6f}",

                    "ci_low":
                        lambda x:
                        f"{x:+.6f}",

                    "ci_high":
                        lambda x:
                        f"{x:+.6f}",

                    "prob_positive":
                        lambda x:
                        f"{x:.2f}%",
                },
            )
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. Focus on Reward Policy
# ============================================================

def print_reward_policy_focus(
    combined,
):

    print_section(
        "REWARD POLICY - PAIRED DIFFERENCE FOCUS"
    )

    focus = (
        combined[
            combined[
                "policy_a"
            ]
            == "reward_policy"
        ]
        .copy()
    )

    focus[
        "prob_positive"
    ] *= 100

    print(
        focus[
            [
                "split",
                "comparison",
                "threshold_label",
                "delta_dr",
                "ci_low",
                "ci_high",
                "prob_positive",
                "n_decisions",
                "n_games",
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

                "prob_positive":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )


# ============================================================
# 8. Robustness summary
# ============================================================

def build_summary(
    combined,
):

    print_section(
        "PAIRED DIFFERENCE ROBUSTNESS SUMMARY"
    )

    rows = []

    for (
        split,
        comparison,
    ), g in (
        combined.groupby(
            [
                "split",
                "comparison",
            ]
        )
    ):

        all_positive = bool(
            (
                g[
                    "delta_dr"
                ]
                >
                0
            )
            .all()
        )

        all_negative = bool(
            (
                g[
                    "delta_dr"
                ]
                <
                0
            )
            .all()
        )

        ci_positive_count = int(
            (
                g[
                    "ci_low"
                ]
                >
                0
            )
            .sum()
        )

        ci_negative_count = int(
            (
                g[
                    "ci_high"
                ]
                <
                0
            )
            .sum()
        )

        rows.append(
            {
                "split":
                    split,

                "comparison":
                    comparison,

                "delta_min":
                    float(
                        g[
                            "delta_dr"
                        ]
                        .min()
                    ),

                "delta_max":
                    float(
                        g[
                            "delta_dr"
                        ]
                        .max()
                    ),

                "all_delta_positive":
                    all_positive,

                "all_delta_negative":
                    all_negative,

                "positive_ci_count":
                    ci_positive_count,

                "negative_ci_count":
                    ci_negative_count,

                "n_thresholds":
                    int(
                        len(g)
                    ),

                "min_prob_positive":
                    float(
                        g[
                            "prob_positive"
                        ]
                        .min()
                    ),

                "max_prob_positive":
                    float(
                        g[
                            "prob_positive"
                        ]
                        .max()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    display = (
        summary.copy()
    )

    display[
        "min_prob_positive"
    ] *= 100

    display[
        "max_prob_positive"
    ] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "delta_min":
                    lambda x:
                    f"{x:+.6f}",

                "delta_max":
                    lambda x:
                    f"{x:+.6f}",

                "min_prob_positive":
                    lambda x:
                    f"{x:.2f}%",

                "max_prob_positive":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 92
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- PAIRED POLICY VALUE DIFFERENCE"
    )

    print()

    print(
        "Goal:"
    )

    print(
        "Estimate DR(policy A) - DR(policy B)"
    )

    print(
        "with paired game-level bootstrap."
    )

    print()

    print(
        "Comparisons:"
    )

    print(
        "  reward_policy - heuristic"
    )

    print(
        "  reward_policy - random"
    )

    print(
        "  heuristic - random"
    )

    print()

    print(
        "Clipping:"
    )

    print(
        "  10 / 20 / 30 / 50 / 100 / unclipped"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This remains model-based offline evidence."
    )

    print(
        "Estimated behavior propensities are not "
        "known randomized logging probabilities."
    )

    print(
        "=" * 92
    )

    # ========================================================
    # Load
    # ========================================================

    validation = (
        load_ope_decisions(
            "validation"
        )
    )

    test = (
        load_ope_decisions(
            "test"
        )
    )

    # ========================================================
    # Evaluate
    # ========================================================

    validation_result = (
        evaluate_split(
            "validation",
            validation,
        )
    )

    test_result = (
        evaluate_split(
            "test",
            test,
        )
    )

    combined = (
        pd.concat(
            [
                validation_result,
                test_result,
            ],
            ignore_index=True,
        )
    )

    # ========================================================
    # Summaries
    # ========================================================

    print_reward_policy_focus(
        combined
    )

    summary = (
        build_summary(
            combined
        )
    )

    # ========================================================
    # Save
    # ========================================================

    validation_result.to_csv(
        OUTPUT_DIR
        / "validation_paired_diff.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_paired_diff.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "paired_diff_all.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "paired_diff_summary.csv",
        index=False,
    )

    # ========================================================
    # Done
    # ========================================================

    print_section(
        "DONE"
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_paired_diff.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_paired_diff.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "paired_diff_all.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "paired_diff_summary.csv"
    )


if __name__ == "__main__":
    main()
