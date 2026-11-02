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
    "outputs/bandit/ope/v2_ope_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# None = clipping 없음
CLIP_THRESHOLDS = [
    10.0,
    20.0,
    30.0,
    50.0,
    100.0,
    None,
]


BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def threshold_label(
    threshold,
):

    if threshold is None:

        return "unclipped"

    return (
        f"clip_{int(threshold)}"
    )


# ============================================================
# 1. Load existing OPE decision contributions
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

        "pi_actual",
        "mu_actual",

        "logged_reward",
        "q_actual",
        "dm_i",
        "residual",

        "weight",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    numeric_cols = [
        "decision_id",
        "pi_actual",
        "mu_actual",
        "logged_reward",
        "q_actual",
        "dm_i",
        "residual",
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
# 2. Apply one clipping threshold
# ============================================================

def apply_clipping(
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

        out[
            "was_clipped"
        ] = (
            False
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

        out[
            "was_clipped"
        ] = (
            out[
                "weight"
            ]
            >
            threshold
        )

    # --------------------------------------------------------
    # IPS contribution
    # --------------------------------------------------------

    out[
        "ips_i"
    ] = (
        out[
            "weight_used"
        ]
        *
        out[
            "logged_reward"
        ]
    )

    # --------------------------------------------------------
    # DR contribution
    #
    # DR_i =
    # DM_i
    # +
    # weight_used
    # * (r - Q_logged)
    # --------------------------------------------------------

    out[
        "dr_i_used"
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
# 3. Effective Sample Size
# ============================================================

def effective_sample_size(
    weights,
):

    weights = np.asarray(
        weights,
        dtype=float,
    )

    numerator = (
        weights.sum()
        ** 2
    )

    denominator = (
        np.sum(
            weights ** 2
        )
    )

    if denominator <= 0:

        return 0.0

    return float(
        numerator
        /
        denominator
    )


# ============================================================
# 4. Metrics for one policy / threshold
# ============================================================

def compute_metrics(
    df,
    threshold,
):

    n = (
        len(df)
    )

    weights = (
        df[
            "weight_used"
        ]
        .to_numpy(
            dtype=float
        )
    )

    raw_weights = (
        df[
            "weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Direct Method
    # clipping과 무관
    # --------------------------------------------------------

    dm = float(
        df[
            "dm_i"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # IPS
    # --------------------------------------------------------

    ips = float(
        df[
            "ips_i"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # SNIPS
    # --------------------------------------------------------

    weight_sum = float(
        weights.sum()
    )

    if weight_sum > 0:

        snips = float(
            df[
                "ips_i"
            ]
            .sum()
            /
            weight_sum
        )

    else:

        snips = np.nan

    # --------------------------------------------------------
    # DR
    # --------------------------------------------------------

    dr = float(
        df[
            "dr_i_used"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # ESS
    # --------------------------------------------------------

    ess = (
        effective_sample_size(
            weights
        )
    )

    # --------------------------------------------------------
    # Support
    #
    # deterministic target policy:
    # actual action이 policy 선택과 같으면 pi > 0
    #
    # random:
    # 모든 action에서 pi > 0
    # --------------------------------------------------------

    supported = (
        df[
            "pi_actual"
        ]
        >
        0
    )

    support_rate = float(
        supported.mean()
    )

    support_n = int(
        supported.sum()
    )

    # --------------------------------------------------------
    # Clipping diagnostics
    # --------------------------------------------------------

    clipped_fraction = float(
        df[
            "was_clipped"
        ]
        .mean()
    )

    clipped_n = int(
        df[
            "was_clipped"
        ]
        .sum()
    )

    nonzero_raw = (
        raw_weights[
            raw_weights
            >
            0
        ]
    )

    nonzero_used = (
        weights[
            weights
            >
            0
        ]
    )

    if len(
        nonzero_raw
    ) > 0:

        raw_p95 = float(
            np.quantile(
                nonzero_raw,
                0.95,
            )
        )

        raw_p99 = float(
            np.quantile(
                nonzero_raw,
                0.99,
            )
        )

        raw_max = float(
            nonzero_raw.max()
        )

    else:

        raw_p95 = np.nan
        raw_p99 = np.nan
        raw_max = np.nan

    if len(
        nonzero_used
    ) > 0:

        used_max = float(
            nonzero_used.max()
        )

    else:

        used_max = np.nan

    return {
        "n":
            int(n),

        "threshold":
            (
                np.inf
                if threshold is None
                else float(
                    threshold
                )
            ),

        "threshold_label":
            threshold_label(
                threshold
            ),

        "support_n":
            support_n,

        "support_rate":
            support_rate,

        "dm":
            dm,

        "ips":
            ips,

        "snips":
            snips,

        "dr":
            dr,

        "ess":
            ess,

        "ess_fraction":
            float(
                ess
                /
                n
            ),

        "weight_sum":
            weight_sum,

        "mean_weight":
            float(
                weights.mean()
            ),

        "raw_weight_p95":
            raw_p95,

        "raw_weight_p99":
            raw_p99,

        "raw_weight_max":
            raw_max,

        "used_weight_max":
            used_max,

        "clipped_n":
            clipped_n,

        "clipped_fraction":
            clipped_fraction,
    }


# ============================================================
# 5. Game-level bootstrap
# ============================================================

def bootstrap_ci(
    df,
):

    # --------------------------------------------------------
    # 하나의 경기에서 여러 교체 의사결정이 나오므로
    # decision-level bootstrap 대신
    # game-level cluster bootstrap
    # --------------------------------------------------------

    game_df = (
        df.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            n=(
                "decision_id",
                "size",
            ),

            ips_sum=(
                "ips_i",
                "sum",
            ),

            weight_sum=(
                "weight_used",
                "sum",
            ),

            dr_sum=(
                "dr_i_used",
                "sum",
            ),
        )
    )

    n_games = (
        len(
            game_df
        )
    )

    n_arr = (
        game_df[
            "n"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ips_arr = (
        game_df[
            "ips_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    weight_arr = (
        game_df[
            "weight_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    dr_arr = (
        game_df[
            "dr_sum"
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

    ips_values = []
    snips_values = []
    dr_values = []

    for _ in range(
        BOOTSTRAP_REPS
    ):

        sample_idx = (
            rng.integers(
                0,
                n_games,
                size=n_games,
            )
        )

        total_n = float(
            n_arr[
                sample_idx
            ]
            .sum()
        )

        total_ips = float(
            ips_arr[
                sample_idx
            ]
            .sum()
        )

        total_weight = float(
            weight_arr[
                sample_idx
            ]
            .sum()
        )

        total_dr = float(
            dr_arr[
                sample_idx
            ]
            .sum()
        )

        ips_values.append(
            total_ips
            /
            total_n
        )

        if total_weight > 0:

            snips_values.append(
                total_ips
                /
                total_weight
            )

        else:

            snips_values.append(
                np.nan
            )

        dr_values.append(
            total_dr
            /
            total_n
        )

    def ci(values):

        arr = np.asarray(
            values,
            dtype=float,
        )

        arr = (
            arr[
                np.isfinite(
                    arr
                )
            ]
        )

        return (
            float(
                np.quantile(
                    arr,
                    0.025,
                )
            ),
            float(
                np.quantile(
                    arr,
                    0.975,
                )
            ),
        )

    (
        ips_low,
        ips_high,
    ) = ci(
        ips_values
    )

    (
        snips_low,
        snips_high,
    ) = ci(
        snips_values
    )

    (
        dr_low,
        dr_high,
    ) = ci(
        dr_values
    )

    return {
        "bootstrap_games":
            int(
                n_games
            ),

        "bootstrap_reps":
            int(
                BOOTSTRAP_REPS
            ),

        "ips_ci_low":
            ips_low,

        "ips_ci_high":
            ips_high,

        "snips_ci_low":
            snips_low,

        "snips_ci_high":
            snips_high,

        "dr_ci_low":
            dr_low,

        "dr_ci_high":
            dr_high,
    }


# ============================================================
# 6. Run robustness for one split
# ============================================================

def evaluate_split(
    split,
    ope_df,
):

    print_section(
        f"OPE ROBUSTNESS - {split.upper()}"
    )

    rows = []

    policies = [
        "random",
        "heuristic",
        "reward_policy",
    ]

    for policy in policies:

        policy_df = (
            ope_df[
                ope_df[
                    "policy"
                ]
                == policy
            ]
            .copy()
        )

        if policy_df.empty:

            raise ValueError(
                f"No rows for policy: {policy}"
            )

        print()
        print(
            "-" * 84
        )

        print(
            f"POLICY: {policy}"
        )

        print(
            "-" * 84
        )

        policy_rows = []

        for threshold in (
            CLIP_THRESHOLDS
        ):

            clipped = (
                apply_clipping(
                    policy_df,
                    threshold,
                )
            )

            metrics = (
                compute_metrics(
                    clipped,
                    threshold,
                )
            )

            ci = (
                bootstrap_ci(
                    clipped
                )
            )

            row = {
                "split":
                    split,

                "policy":
                    policy,

                **metrics,
                **ci,
            }

            rows.append(
                row
            )

            policy_rows.append(
                row
            )

        display = pd.DataFrame(
            policy_rows
        )

        display[
            "support_rate"
        ] *= 100

        display[
            "ess_fraction"
        ] *= 100

        display[
            "clipped_fraction"
        ] *= 100

        print(
            display[
                [
                    "threshold_label",
                    "support_rate",
                    "ips",
                    "snips",
                    "dr",
                    "dr_ci_low",
                    "dr_ci_high",
                    "ess",
                    "ess_fraction",
                    "clipped_fraction",
                    "used_weight_max",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "support_rate":
                        lambda x:
                        f"{x:.2f}%",

                    "ips":
                        lambda x:
                        f"{x:.6f}",

                    "snips":
                        lambda x:
                        f"{x:.6f}",

                    "dr":
                        lambda x:
                        f"{x:.6f}",

                    "dr_ci_low":
                        lambda x:
                        f"{x:.6f}",

                    "dr_ci_high":
                        lambda x:
                        f"{x:.6f}",

                    "ess":
                        lambda x:
                        f"{x:.1f}",

                    "ess_fraction":
                        lambda x:
                        f"{x:.2f}%",

                    "clipped_fraction":
                        lambda x:
                        f"{x:.3f}%",

                    "used_weight_max":
                        lambda x:
                        f"{x:.2f}",
                },
            )
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# 7. Summarize stability
# ============================================================

def robustness_summary(
    result,
):

    print_section(
        "ROBUSTNESS SUMMARY"
    )

    rows = []

    for (
        split,
        policy,
    ), g in (
        result.groupby(
            [
                "split",
                "policy",
            ]
        )
    ):

        finite_clip = (
            g[
                g[
                    "threshold_label"
                ]
                != "unclipped"
            ]
            .copy()
        )

        dr_min = float(
            g[
                "dr"
            ]
            .min()
        )

        dr_max = float(
            g[
                "dr"
            ]
            .max()
        )

        snips_min = float(
            g[
                "snips"
            ]
            .min()
        )

        snips_max = float(
            g[
                "snips"
            ]
            .max()
        )

        all_dr_positive = bool(
            (
                g[
                    "dr"
                ]
                >
                0
            )
            .all()
        )

        all_dr_negative = bool(
            (
                g[
                    "dr"
                ]
                <
                0
            )
            .all()
        )

        positive_ci_count = int(
            (
                g[
                    "dr_ci_low"
                ]
                >
                0
            )
            .sum()
        )

        negative_ci_count = int(
            (
                g[
                    "dr_ci_high"
                ]
                <
                0
            )
            .sum()
        )

        total_thresholds = int(
            len(g)
        )

        best_ess = float(
            finite_clip[
                "ess"
            ]
            .max()
        )

        worst_ess = float(
            g[
                "ess"
            ]
            .min()
        )

        rows.append(
            {
                "split":
                    split,

                "policy":
                    policy,

                "dr_min":
                    dr_min,

                "dr_max":
                    dr_max,

                "snips_min":
                    snips_min,

                "snips_max":
                    snips_max,

                "all_dr_positive":
                    all_dr_positive,

                "all_dr_negative":
                    all_dr_negative,

                "positive_dr_ci_count":
                    positive_ci_count,

                "negative_dr_ci_count":
                    negative_ci_count,

                "n_thresholds":
                    total_thresholds,

                "best_ess":
                    best_ess,

                "worst_ess":
                    worst_ess,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "dr_min":
                    lambda x:
                    f"{x:.6f}",

                "dr_max":
                    lambda x:
                    f"{x:.6f}",

                "snips_min":
                    lambda x:
                    f"{x:.6f}",

                "snips_max":
                    lambda x:
                    f"{x:.6f}",

                "best_ess":
                    lambda x:
                    f"{x:.1f}",

                "worst_ess":
                    lambda x:
                    f"{x:.1f}",
            },
        )
    )

    return summary


# ============================================================
# 8. Focus table for Reward Policy
# ============================================================

def print_reward_policy_focus(
    result,
):

    print_section(
        "REWARD POLICY - MAIN ROBUSTNESS CHECK"
    )

    reward = (
        result[
            result[
                "policy"
            ]
            == "reward_policy"
        ]
        .copy()
    )

    display = (
        reward[
            [
                "split",
                "threshold_label",
                "ips",
                "snips",
                "dr",
                "dr_ci_low",
                "dr_ci_high",
                "ess",
                "ess_fraction",
                "clipped_fraction",
                "raw_weight_max",
                "used_weight_max",
            ]
        ]
        .copy()
    )

    display[
        "ess_fraction"
    ] *= 100

    display[
        "clipped_fraction"
    ] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "ips":
                    lambda x:
                    f"{x:.6f}",

                "snips":
                    lambda x:
                    f"{x:.6f}",

                "dr":
                    lambda x:
                    f"{x:.6f}",

                "dr_ci_low":
                    lambda x:
                    f"{x:.6f}",

                "dr_ci_high":
                    lambda x:
                    f"{x:.6f}",

                "ess":
                    lambda x:
                    f"{x:.1f}",

                "ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "clipped_fraction":
                    lambda x:
                    f"{x:.3f}%",

                "raw_weight_max":
                    lambda x:
                    f"{x:.2f}",

                "used_weight_max":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 84
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- OPE ROBUSTNESS CHECK"
    )

    print()

    print(
        "Goal:"
    )

    print(
        "Check whether IPS / SNIPS / DR "
        "depend heavily on extreme importance weights."
    )

    print()

    print(
        "Clipping thresholds:"
    )

    print(
        "10 / 20 / 30 / 50 / 100 / unclipped"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Clipping reduces variance but introduces bias."
    )

    print(
        "Stable conclusions across thresholds are "
        "more important than one best-looking value."
    )

    print(
        "=" * 84
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
    # Summary
    # ========================================================

    summary = (
        robustness_summary(
            combined
        )
    )

    print_reward_policy_focus(
        combined
    )

    # ========================================================
    # Save
    # ========================================================

    validation_result.to_csv(
        OUTPUT_DIR
        / "validation_robustness.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_robustness.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "ope_robustness_all.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "robustness_summary.csv",
        index=False,
    )

    print_section(
        "DONE"
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_robustness.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_robustness.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "ope_robustness_all.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "robustness_summary.csv"
    )


if __name__ == "__main__":
    main()
