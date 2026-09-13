from pathlib import Path

import numpy as np
import pandas as pd

import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

LI_ABLATION_DIR = Path(
    "bandit_output_v3_li_ablation"
)

BEHAVIOR_DIR = Path(
    "bandit_output_v2_behavior_policy"
)

OUTPUT_DIR = Path(
    "bandit_output_v3_li_ope"
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


BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42


KEYS = [
    "game_id",
    "def_team",
    "appearance_order",
    "pitcher_id",
]


# 이전 Step 12의 frozen B0 OPE 결과
EXPECTED_B0_UNCLIPPED = {
    "validation": 0.005778,
    "test": 0.003222,
}


# ============================================================
# Helper
# ============================================================

def section(title):

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


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


def clip_label(threshold):

    if threshold is None:
        return "unclipped"

    return f"clip_{int(threshold)}"


# ============================================================
# 1. Load B0 / B1 candidate predictions
# ============================================================

def load_li_candidates(split):

    path = (
        LI_ABLATION_DIR
        / f"{split}_candidates.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
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

    df["game_id"] = (
        df["game_id"]
        .astype(str)
    )

    df["pitcher_id"] = (
        df["pitcher_id"]
        .apply(normalize_id)
    )

    df["actual_pitcher"] = (
        df["actual_pitcher"]
        .apply(normalize_id)
    )

    df["appearance_order"] = (
        pd.to_numeric(
            df["appearance_order"],
            errors="raise",
        )
        .astype(int)
    )

    for col in [
        "q_b0",
        "q_b1_li",
        "leverage_index",
    ]:

        df[col] = (
            pd.to_numeric(
                df[col],
                errors="raise",
            )
        )

    df["is_actual"] = (
        pd.to_numeric(
            df["is_actual"],
            errors="raise",
        )
        .astype(int)
    )

    return df


# ============================================================
# 2. Load frozen behavior propensity
# ============================================================

def load_behavior_candidates(split):

    path = (
        BEHAVIOR_DIR
        / f"{split}_behavior_candidates.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "def_team": str,
            "pitcher_id": str,
        },
    )

    df["game_id"] = (
        df["game_id"]
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

    df["behavior_prob"] = (
        pd.to_numeric(
            df["behavior_prob"],
            errors="raise",
        )
    )

    return (
        df[
            KEYS
            + [
                "behavior_prob",
            ]
        ]
        .copy()
    )


# ============================================================
# 3. Load logged reward
# ============================================================

def build_logged_reward_lookup():

    section(
        "1. BUILD LOGGED WPA_PER_PA REWARD LOOKUP"
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

    for split_name, df in [
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
                df[
                    "bullpen_source"
                ]
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
            .copy()
        )

        duplicate = int(
            lookup[
                KEYS
            ]
            .duplicated()
            .sum()
        )

        if duplicate > 0:

            raise ValueError(
                f"Duplicate reward keys: {duplicate}"
            )

        result[
            split_name
        ] = lookup

        print(
            f"{split_name:10s}: "
            f"{len(lookup):,} logged decisions"
        )

    return result


# ============================================================
# 4. Build OPE candidate table
# ============================================================

def build_ope_candidates(
    split,
    reward_lookup,
):

    section(
        f"2. BUILD OPE CANDIDATES - {split.upper()}"
    )

    li = (
        load_li_candidates(
            split
        )
    )

    behavior = (
        load_behavior_candidates(
            split
        )
    )

    print(
        f"LI candidate rows       : "
        f"{len(li):,}"
    )

    print(
        f"Behavior candidate rows : "
        f"{len(behavior):,}"
    )

    merged = (
        li.merge(
            behavior,
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
        f"Missing behavior prob   : "
        f"{missing_mu:,}"
    )

    if missing_mu > 0:

        raise ValueError(
            "Behavior propensity join failed."
        )

    # --------------------------------------------------------
    # Logged reward only exists for actual action
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

    if missing_reward > 0:

        raise ValueError(
            "Logged reward join failed."
        )

    # --------------------------------------------------------
    # Behavior probabilities should sum to 1
    # --------------------------------------------------------

    prob_sum = (
        merged.groupby(
            "decision_id"
        )[
            "behavior_prob"
        ]
        .sum()
    )

    max_error = float(
        (
            prob_sum
            -
            1.0
        )
        .abs()
        .max()
    )

    print(
        f"Behavior prob max error : "
        f"{max_error:.10f}"
    )

    if max_error > 1e-5:

        raise ValueError(
            "Behavior probabilities do not sum to 1."
        )

    return merged


# ============================================================
# 5. Select deterministic policy
# ============================================================

def select_policy(
    candidates,
    model_name,
):

    if model_name == "B0":

        score_col = "q_b0"

    elif model_name == "B1_LI":

        score_col = "q_b1_li"

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
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

    return (
        score_col,
        selected,
    )


# ============================================================
# 6. Build decision-level OPE contributions
# ============================================================

def build_policy_decisions(
    candidates,
    model_name,
):

    score_col, selected = (
        select_policy(
            candidates,
            model_name,
        )
    )

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
            selected,
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
            score_col
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
        "model"
    ] = (
        model_name
    )

    return actual


# ============================================================
# 7. Metrics with clipping
# ============================================================

def compute_ope_metrics(
    decisions,
    threshold,
):

    df = (
        decisions.copy()
    )

    if threshold is None:

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
                upper=threshold
            )
        )

    df[
        "ips_i"
    ] = (
        df[
            "weight_used"
        ]
        *
        df[
            "logged_reward"
        ]
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

    n = (
        len(
            df
        )
    )

    dm = float(
        df[
            "dm_i"
        ]
        .mean()
    )

    ips = float(
        df[
            "ips_i"
        ]
        .mean()
    )

    weight_sum = float(
        df[
            "weight_used"
        ]
        .sum()
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

    dr = float(
        df[
            "dr_i"
        ]
        .mean()
    )

    agreement = float(
        df[
            "pi_actual"
        ]
        .mean()
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
            weights
            ** 2
        )
    )

    if denominator > 0:

        ess = float(
            (
                weights.sum()
                ** 2
            )
            /
            denominator
        )

    else:

        ess = 0.0

    nonzero = (
        weights[
            weights
            >
            0
        ]
    )

    if len(
        nonzero
    ) > 0:

        max_weight = float(
            nonzero.max()
        )

        p95_weight = float(
            np.quantile(
                nonzero,
                0.95,
            )
        )

        p99_weight = float(
            np.quantile(
                nonzero,
                0.99,
            )
        )

    else:

        max_weight = np.nan
        p95_weight = np.nan
        p99_weight = np.nan

    if threshold is None:

        clipped_fraction = 0.0

    else:

        clipped_fraction = float(
            (
                df[
                    "weight"
                ]
                >
                threshold
            )
            .mean()
        )

    return (
        df,
        {
            "n":
                int(n),

            "agreement":
                agreement,

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

            "max_weight":
                max_weight,

            "p95_weight":
                p95_weight,

            "p99_weight":
                p99_weight,

            "clipped_fraction":
                clipped_fraction,
        }
    )


# ============================================================
# 8. Paired game bootstrap B1 - B0
# ============================================================

def paired_bootstrap(
    b0,
    b1,
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
            "B0/B1 decision sets are not aligned."
        )

    paired[
        "delta_dr"
    ] = (
        paired[
            "dr_b1"
        ]
        -
        paired[
            "dr_b0"
        ]
    )

    point_delta = float(
        paired[
            "delta_dr"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Cluster by game
    # --------------------------------------------------------

    game = (
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

    n_games = (
        len(
            game
        )
    )

    n_arr = (
        game[
            "n"
        ]
        .to_numpy(
            dtype=float
        )
    )

    delta_arr = (
        game[
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

        idx = (
            rng.integers(
                0,
                n_games,
                size=n_games,
            )
        )

        estimates[
            i
        ] = (
            delta_arr[
                idx
            ]
            .sum()
            /
            n_arr[
                idx
            ]
            .sum()
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

    prob_positive = float(
        (
            estimates
            >
            0
        )
        .mean()
    )

    return {
        "delta_dr":
            point_delta,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "prob_positive":
            prob_positive,

        "n_games":
            int(
                n_games
            ),
    }


# ============================================================
# 9. Evaluate split
# ============================================================

def evaluate_split(
    split,
    candidates,
):

    section(
        f"3. B0 VS B1 OPE - {split.upper()}"
    )

    b0_base = (
        build_policy_decisions(
            candidates,
            "B0",
        )
    )

    b1_base = (
        build_policy_decisions(
            candidates,
            "B1_LI",
        )
    )

    rows = []

    for threshold in (
        CLIP_THRESHOLDS
    ):

        label = (
            clip_label(
                threshold
            )
        )

        (
            b0,
            m0,
        ) = (
            compute_ope_metrics(
                b0_base,
                threshold,
            )
        )

        (
            b1,
            m1,
        ) = (
            compute_ope_metrics(
                b1_base,
                threshold,
            )
        )

        paired = (
            paired_bootstrap(
                b0,
                b1,
            )
        )

        rows.append(
            {
                "split":
                    split,

                "threshold":
                    label,

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

                "delta_ci_low":
                    paired[
                        "ci_low"
                    ],

                "delta_ci_high":
                    paired[
                        "ci_high"
                    ],

                "prob_b1_better":
                    paired[
                        "prob_positive"
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

                "b0_max_weight":
                    m0[
                        "max_weight"
                    ],

                "b1_max_weight":
                    m1[
                        "max_weight"
                    ],

                "b0_clipped_fraction":
                    m0[
                        "clipped_fraction"
                    ],

                "b1_clipped_fraction":
                    m1[
                        "clipped_fraction"
                    ],
            }
        )

    result = pd.DataFrame(
        rows
    )

    display = (
        result.copy()
    )

    for col in [
        "b0_agreement",
        "b1_agreement",
        "prob_b1_better",
        "b0_ess_fraction",
        "b1_ess_fraction",
        "b0_clipped_fraction",
        "b1_clipped_fraction",
    ]:

        display[
            col
        ] *= 100

    print()
    print(
        "Policy Value"
    )

    print(
        display[
            [
                "threshold",

                "b0_dr",
                "b1_dr",
                "delta_dr",

                "delta_ci_low",
                "delta_ci_high",

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

                "delta_ci_low":
                    lambda x:
                    f"{x:+.6f}",

                "delta_ci_high":
                    lambda x:
                    f"{x:+.6f}",

                "prob_b1_better":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    print()
    print(
        "IPS / SNIPS"
    )

    print(
        display[
            [
                "threshold",
                "b0_ips",
                "b1_ips",
                "b0_snips",
                "b1_snips",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "b0_ips":
                    lambda x:
                    f"{x:+.6f}",

                "b1_ips":
                    lambda x:
                    f"{x:+.6f}",

                "b0_snips":
                    lambda x:
                    f"{x:+.6f}",

                "b1_snips":
                    lambda x:
                    f"{x:+.6f}",
            },
        )
    )

    print()
    print(
        "Agreement / ESS"
    )

    print(
        display[
            [
                "threshold",

                "b0_agreement",
                "b1_agreement",

                "b0_ess",
                "b1_ess",

                "b0_ess_fraction",
                "b1_ess_fraction",

                "b0_max_weight",
                "b1_max_weight",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "b0_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "b1_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "b0_ess":
                    lambda x:
                    f"{x:.1f}",

                "b1_ess":
                    lambda x:
                    f"{x:.1f}",

                "b0_ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "b1_ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "b0_max_weight":
                    lambda x:
                    f"{x:.2f}",

                "b1_max_weight":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    return (
        result,
        b0_base,
        b1_base,
    )


# ============================================================
# 10. B0 OPE reproduction
# ============================================================

def check_b0_ope_reproduction(
    split,
    result,
):

    section(
        f"4. B0 OPE REPRODUCTION - {split.upper()}"
    )

    row = (
        result[
            result[
                "threshold"
            ]
            == "unclipped"
        ]
        .iloc[
            0
        ]
    )

    expected = (
        EXPECTED_B0_UNCLIPPED[
            split
        ]
    )

    actual = float(
        row[
            "b0_dr"
        ]
    )

    diff = abs(
        actual
        -
        expected
    )

    print(
        f"Expected B0 DR : "
        f"{expected:+.6f}"
    )

    print(
        f"Current B0 DR  : "
        f"{actual:+.6f}"
    )

    print(
        f"Difference     : "
        f"{diff:.6f}"
    )

    if diff <= 0.00005:

        print(
            "B0 OPE reproduction : PASS"
        )

        return True

    print(
        "B0 OPE reproduction : WARNING"
    )

    return False


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 92
    )

    print(
        "STEP 13-5 - B0 VS B1(+LI) OFFLINE POLICY EVALUATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen pre-LI reward policy"
    )

    print()

    print(
        "B1:"
    )

    print(
        "  B0 + leverage_index + LI x pitcher interactions"
    )

    print()

    print(
        "Behavior policy:"
    )

    print(
        "  Frozen Step 12 manager propensity model"
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        "  DM / IPS / SNIPS / DR"
    )

    print(
        "  Weight clipping sensitivity"
    )

    print(
        "  Paired game-bootstrap B1 - B0"
    )

    print(
        "=" * 92
    )

    # ========================================================
    # Logged rewards
    # ========================================================

    reward_lookup = (
        build_logged_reward_lookup()
    )

    # ========================================================
    # Validation
    # ========================================================

    validation_candidates = (
        build_ope_candidates(
            "validation",
            reward_lookup[
                "validation"
            ],
        )
    )

    (
        validation_result,
        validation_b0,
        validation_b1,
    ) = (
        evaluate_split(
            "validation",
            validation_candidates,
        )
    )

    validation_reproduction = (
        check_b0_ope_reproduction(
            "validation",
            validation_result,
        )
    )

    # ========================================================
    # Test
    # ========================================================

    test_candidates = (
        build_ope_candidates(
            "test",
            reward_lookup[
                "test"
            ],
        )
    )

    (
        test_result,
        test_b0,
        test_b1,
    ) = (
        evaluate_split(
            "test",
            test_candidates,
        )
    )

    test_reproduction = (
        check_b0_ope_reproduction(
            "test",
            test_result,
        )
    )

    # ========================================================
    # Combined
    # ========================================================

    combined = pd.concat(
        [
            validation_result,
            test_result,
        ],
        ignore_index=True,
    )

    # ========================================================
    # Final focus
    # ========================================================

    section(
        "FINAL B1 - B0 POLICY VALUE COMPARISON"
    )

    focus = (
        combined[
            [
                "split",
                "threshold",

                "b0_dr",
                "b1_dr",

                "delta_dr",
                "delta_ci_low",
                "delta_ci_high",

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

                "delta_ci_low":
                    lambda x:
                    f"{x:+.6f}",

                "delta_ci_high":
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

    # ========================================================
    # Save
    # ========================================================

    section(
        "SAVE"
    )

    validation_result.to_csv(
        OUTPUT_DIR
        / "validation_ope.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_ope.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "b0_b1_li_ope_all.csv",
        index=False,
    )

    validation_b0.to_csv(
        OUTPUT_DIR
        / "validation_b0_decisions.csv",
        index=False,
    )

    validation_b1.to_csv(
        OUTPUT_DIR
        / "validation_b1_decisions.csv",
        index=False,
    )

    test_b0.to_csv(
        OUTPUT_DIR
        / "test_b0_decisions.csv",
        index=False,
    )

    test_b1.to_csv(
        OUTPUT_DIR
        / "test_b1_decisions.csv",
        index=False,
    )

    # ========================================================
    # Final status
    # ========================================================

    section(
        "FINAL STATUS"
    )

    print(
        "B0 Agreement reproduction : PASS"
    )

    print(
        "Validation B0 OPE reproduction :",
        (
            "PASS"
            if validation_reproduction
            else "WARNING"
        ),
    )

    print(
        "Test B0 OPE reproduction       :",
        (
            "PASS"
            if test_reproduction
            else "WARNING"
        ),
    )

    print()

    print(
        "If both B0 OPE reproduction checks pass,"
    )

    print(
        "interpret B1 - B0 delta DR and its paired CI."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()