from pathlib import Path

import numpy as np
import pandas as pd

import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

ABLATION_DIR = Path(
    "bandit_output_v3_li_ablation"
)

BEHAVIOR_DIR = Path(
    "bandit_output_v2_behavior_policy"
)

OUTPUT_DIR = Path(
    "bandit_output_v3_li_common_q_ope"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


CLIPS = [
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


KEYS = [
    "game_id",
    "def_team",
    "appearance_order",
    "pitcher_id",
]


# ============================================================
# Helpers
# ============================================================

def section(title):

    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def normalize_id(x):

    if pd.isna(x):
        return "__MISSING__"

    s = str(x).strip()

    if (
        s.endswith(".0")
        and s[:-2].isdigit()
    ):
        s = s[:-2]

    return s


def clip_name(clip):

    if clip is None:
        return "unclipped"

    return f"clip_{int(clip)}"


# ============================================================
# Logged reward
# ============================================================

def build_reward_lookup():

    section(
        "1. BUILD LOGGED REWARD LOOKUP"
    )

    base.REWARD_MODE = "wpa_per_pa"
    base.ALPHA = 0.0

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    (
        decisions,
        first_app,
    ) = base.build_decision_dataset(
        pa,
        history,
    )

    (
        train,
        val,
        test,
    ) = base.chronological_split(
        decisions
    )

    result = {}

    for split, df in [
        (
            "validation",
            val,
        ),
        (
            "test",
            test,
        ),
    ]:

        work = (
            df[
                df["bullpen_source"]
                == "preview"
            ]
            .copy()
        )

        work["game_id"] = (
            work["game_id"]
            .astype(str)
        )

        work["def_team"] = (
            work["def_team"]
            .astype(str)
        )

        work["pitcher_id"] = (
            work["pitcher_id"]
            .apply(normalize_id)
        )

        work["appearance_order"] = (
            pd.to_numeric(
                work["appearance_order"],
                errors="raise",
            )
            .astype(int)
        )

        work["reward"] = (
            pd.to_numeric(
                work["reward"],
                errors="raise",
            )
        )

        lookup = (
            work[
                KEYS
                + [
                    "reward",
                ]
            ]
            .rename(
                columns={
                    "reward":
                        "logged_reward"
                }
            )
        )

        if (
            lookup[
                KEYS
            ]
            .duplicated()
            .any()
        ):

            raise ValueError(
                "Duplicate logged reward keys."
            )

        result[split] = (
            lookup
        )

        print(
            f"{split:10s}: "
            f"{len(lookup):,}"
        )

    return result


# ============================================================
# Candidate data
# ============================================================

def load_candidates(
    split,
    reward_lookup,
):

    section(
        f"2. LOAD CANDIDATES - {split.upper()}"
    )

    candidate_path = (
        ABLATION_DIR
        / f"{split}_candidates.csv"
    )

    behavior_path = (
        BEHAVIOR_DIR
        / f"{split}_behavior_candidates.csv"
    )

    candidates = pd.read_csv(
        candidate_path,
        dtype={
            "game_id": str,
            "def_team": str,
            "pitcher_id": str,
            "actual_pitcher": str,
        },
    )

    behavior = pd.read_csv(
        behavior_path,
        dtype={
            "game_id": str,
            "def_team": str,
            "pitcher_id": str,
        },
    )

    for df in [
        candidates,
        behavior,
    ]:

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
            .apply(normalize_id)
        )

        df["appearance_order"] = (
            pd.to_numeric(
                df["appearance_order"],
                errors="raise",
            )
            .astype(int)
        )

    candidates[
        "actual_pitcher"
    ] = (
        candidates[
            "actual_pitcher"
        ]
        .apply(normalize_id)
    )

    for col in [
        "decision_id",
        "is_actual",
        "leverage_index",
        "q_b0",
        "q_b1_li",
    ]:

        candidates[col] = (
            pd.to_numeric(
                candidates[col],
                errors="raise",
            )
        )

    candidates[
        "decision_id"
    ] = (
        candidates[
            "decision_id"
        ]
        .astype(int)
    )

    candidates[
        "is_actual"
    ] = (
        candidates[
            "is_actual"
        ]
        .astype(int)
    )

    behavior[
        "behavior_prob"
    ] = (
        pd.to_numeric(
            behavior[
                "behavior_prob"
            ],
            errors="raise",
        )
    )

    merged = (
        candidates.merge(
            behavior[
                KEYS
                + [
                    "behavior_prob"
                ]
            ],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    )

    missing_mu = int(
        merged[
            "behavior_prob"
        ]
        .isna()
        .sum()
    )

    print(
        f"Candidate rows          : "
        f"{len(merged):,}"
    )

    print(
        f"Missing behavior prob   : "
        f"{missing_mu:,}"
    )

    if missing_mu:

        raise ValueError(
            "Behavior join failed."
        )

    # --------------------------------------------------------
    # Logged reward
    # --------------------------------------------------------

    merged = (
        merged.merge(
            reward_lookup,
            on=KEYS,
            how="left",
            validate="many_to_one",
        )
    )

    actual = (
        merged[
            merged[
                "is_actual"
            ]
            == 1
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

    if missing_reward:

        raise ValueError(
            "Reward join failed."
        )

    # --------------------------------------------------------
    # Propensity sum check
    # --------------------------------------------------------

    sums = (
        merged.groupby(
            "decision_id"
        )[
            "behavior_prob"
        ]
        .sum()
    )

    max_error = float(
        (
            sums - 1.0
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


# ============================================================
# Target policy
# ============================================================

def choose_policy(
    candidates,
    policy,
):

    if policy == "B0":

        score_col = "q_b0"

    elif policy == "B1":

        score_col = "q_b1_li"

    else:

        raise ValueError(
            policy
        )

    ordered = (
        candidates.sort_values(
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
            ]
        ]
        .rename(
            columns={
                "pitcher_id":
                    "selected_pitcher"
            }
        )
    )

    return selected


# ============================================================
# Common-Q decision data
# ============================================================

def build_common_q_decisions(
    candidates,
    policy,
    q_eval,
):

    if q_eval == "B0_Q":

        q_col = "q_b0"

    elif q_eval == "B1_Q":

        q_col = "q_b1_li"

    else:

        raise ValueError(
            q_eval
        )

    selected = (
        choose_policy(
            candidates,
            policy,
        )
    )

    # --------------------------------------------------------
    # Q(selected action)
    # --------------------------------------------------------

    selected_q = (
        selected.merge(
            candidates[
                [
                    "decision_id",
                    "pitcher_id",
                    q_col,
                ]
            ],
            left_on=[
                "decision_id",
                "selected_pitcher",
            ],
            right_on=[
                "decision_id",
                "pitcher_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    selected_q = (
        selected_q[
            [
                "decision_id",
                "selected_pitcher",
                q_col,
            ]
        ]
        .rename(
            columns={
                q_col:
                    "dm_i"
            }
        )
    )

    # --------------------------------------------------------
    # Actual logged action
    # --------------------------------------------------------

    actual = (
        candidates[
            candidates[
                "is_actual"
            ]
            == 1
        ]
        .copy()
    )

    actual = (
        actual.merge(
            selected_q,
            on="decision_id",
            how="left",
            validate="one_to_one",
        )
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

    actual[
        "q_actual"
    ] = (
        actual[
            q_col
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
    ] = policy

    actual[
        "q_eval"
    ] = q_eval

    actual[
        "li_bucket"
    ] = pd.cut(
        actual[
            "leverage_index"
        ],
        bins=LI_BINS,
        labels=LI_LABELS,
        right=False,
    )

    return actual


# ============================================================
# Apply clipping / DR
# ============================================================

def compute_dr(
    decisions,
    clip,
):

    df = (
        decisions.copy()
    )

    if clip is None:

        df[
            "weight_used"
        ] = (
            df[
                "weight"
            ]
        )

    else:

        df[
            "weight_used"
        ] = (
            df[
                "weight"
            ]
            .clip(
                upper=clip
            )
        )

    df[
        "dr_i"
    ] = (
        df[
            "dm_i"
        ]
        +
        df[
            "weight_used"
        ]
        *
        df[
            "residual"
        ]
    )

    weights = (
        df[
            "weight_used"
        ]
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

    return (
        df,
        {
            "dr":
                float(
                    df[
                        "dr_i"
                    ]
                    .mean()
                ),

            "agreement":
                float(
                    df[
                        "pi_actual"
                    ]
                    .mean()
                ),

            "ess":
                ess,
        }
    )


# ============================================================
# Paired game bootstrap
# ============================================================

def paired_bootstrap(
    b0,
    b1,
    seed,
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
                    "b0_dr_i"
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
                    "b1_dr_i"
            }
        )
    )

    paired = (
        left.merge(
            right,
            on=[
                "decision_id",
                "game_id",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(paired)
        != len(b0)
        or
        len(paired)
        != len(b1)
    ):

        raise ValueError(
            "Paired decisions mismatch."
        )

    paired[
        "delta"
    ] = (
        paired[
            "b1_dr_i"
        ]
        -
        paired[
            "b0_dr_i"
        ]
    )

    point = float(
        paired[
            "delta"
        ]
        .mean()
    )

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
                "delta",
                "sum",
            ),
        )
    )

    n_games = len(
        games
    )

    ns = (
        games[
            "n"
        ]
        .to_numpy(
            dtype=float
        )
    )

    sums = (
        games[
            "delta_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rng = (
        np.random.default_rng(
            seed
        )
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

        boot[
            i
        ] = (
            sums[
                idx
            ]
            .sum()
            /
            ns[
                idx
            ]
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
    }


# ============================================================
# Evaluate
# ============================================================

def evaluate(
    split,
    candidates,
):

    section(
        f"3. COMMON-Q OPE - {split.upper()}"
    )

    rows = []

    for q_index, q_eval in enumerate(
        [
            "B0_Q",
            "B1_Q",
        ]
    ):

        b0_base = (
            build_common_q_decisions(
                candidates,
                "B0",
                q_eval,
            )
        )

        b1_base = (
            build_common_q_decisions(
                candidates,
                "B1",
                q_eval,
            )
        )

        # ====================================================
        # Overall + LI segments
        # ====================================================

        segments = [
            "overall",
            *LI_LABELS,
        ]

        for segment_index, segment in enumerate(
            segments
        ):

            if segment == "overall":

                b0_segment = (
                    b0_base.copy()
                )

                b1_segment = (
                    b1_base.copy()
                )

            else:

                b0_segment = (
                    b0_base[
                        b0_base[
                            "li_bucket"
                        ]
                        == segment
                    ]
                    .copy()
                )

                b1_segment = (
                    b1_base[
                        b1_base[
                            "li_bucket"
                        ]
                        == segment
                    ]
                    .copy()
                )

            if len(
                b0_segment
            ) == 0:

                continue

            for clip_index, clip in enumerate(
                CLIPS
            ):

                (
                    b0_eval,
                    m0,
                ) = compute_dr(
                    b0_segment,
                    clip,
                )

                (
                    b1_eval,
                    m1,
                ) = compute_dr(
                    b1_segment,
                    clip,
                )

                bootstrap = (
                    paired_bootstrap(
                        b0_eval,
                        b1_eval,
                        seed=(
                            BOOTSTRAP_SEED
                            +
                            q_index
                            * 10000
                            +
                            segment_index
                            * 100
                            +
                            clip_index
                        ),
                    )
                )

                rows.append(
                    {
                        "split":
                            split,

                        "q_eval":
                            q_eval,

                        "segment":
                            segment,

                        "threshold":
                            clip_name(
                                clip
                            ),

                        "n":
                            int(
                                len(
                                    b0_segment
                                )
                            ),

                        "b0_dr":
                            m0[
                                "dr"
                            ],

                        "b1_dr":
                            m1[
                                "dr"
                            ],

                        "delta_dr":
                            bootstrap[
                                "delta_dr"
                            ],

                        "ci_low":
                            bootstrap[
                                "ci_low"
                            ],

                        "ci_high":
                            bootstrap[
                                "ci_high"
                            ],

                        "prob_b1_better":
                            bootstrap[
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

                        "b0_agreement":
                            m0[
                                "agreement"
                            ],

                        "b1_agreement":
                            m1[
                                "agreement"
                            ],
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Print
# ============================================================

def print_focus(
    split,
    result,
):

    section(
        f"COMMON-Q FOCUS - {split.upper()}"
    )

    focus = (
        result[
            (
                result[
                    "threshold"
                ]
                .isin(
                    [
                        "clip_10",
                        "clip_50",
                        "unclipped",
                    ]
                )
            )
        ]
        .copy()
    )

    focus[
        "prob_b1_better"
    ] *= 100

    print(
        focus[
            [
                "q_eval",
                "segment",
                "threshold",
                "n",

                "b0_dr",
                "b1_dr",
                "delta_dr",

                "ci_low",
                "ci_high",

                "prob_b1_better",
            ]
        ]
        .to_string(
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
            },
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 104
    )

    print(
        "STEP 13-7 - COMMON-Q OPE SENSITIVITY"
    )

    print()

    print(
        "Purpose:"
    )

    print(
        "Compare B0 and B1 policies while holding "
        "the DR reward model Q fixed."
    )

    print()

    print(
        "Evaluation A:"
    )

    print(
        "  B0 policy and B1 policy both evaluated with B0 Q"
    )

    print()

    print(
        "Evaluation B:"
    )

    print(
        "  B0 policy and B1 policy both evaluated with B1 Q"
    )

    print(
        "=" * 104
    )

    reward = (
        build_reward_lookup()
    )

    all_results = []

    for split in [
        "validation",
        "test",
    ]:

        candidates = (
            load_candidates(
                split,
                reward[
                    split
                ],
            )
        )

        result = (
            evaluate(
                split,
                candidates,
            )
        )

        all_results.append(
            result
        )

        print_focus(
            split,
            result,
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    section(
        "SAVE"
    )

    combined.to_csv(
        OUTPUT_DIR
        / "common_q_ope.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "common_q_ope.csv"
    )

    section(
        "DONE"
    )

    print(
        "Decision rule:"
    )

    print(
        "Only treat LI-segment OPE patterns as robust "
        "if their direction remains similar under B0_Q and B1_Q."
    )


if __name__ == "__main__":
    main()