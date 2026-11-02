from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base


# =============================================================================
# CONFIG
# =============================================================================

ABLATION_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_ablation"
)

BEHAVIOR_DIR = Path(
    "outputs/bandit/ope/v2_behavior_policy"
)

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_ope"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


POLICIES = {
    "B0": "q_b0",
    "B1_LI": "q_b1_li",
    "B2_QUALITY": "q_b2_quality",
    "B3_LI_QUALITY": "q_b3_li_quality",
}


PAIRWISE_COMPARISONS = [
    ("B1_LI", "B0"),
    ("B2_QUALITY", "B0"),
    ("B3_LI_QUALITY", "B0"),
    ("B3_LI_QUALITY", "B1_LI"),
    ("B3_LI_QUALITY", "B2_QUALITY"),
]


# 기존 결과 reproduction check
EXPECTED_UNCLIPPED = {
    "validation": {
        "B0": 0.005778,

        # 이전 LI OPE
        "B1_minus_B0": 0.001149,

        # 이전 quality OPE
        "B2_minus_B0": 0.000614,
    },

    "test": {
        "B0": 0.003222,

        # 이전 LI OPE
        "B1_minus_B0": 0.000045,

        # 이전 quality OPE
        "B2_minus_B0": 0.000561,
    },
}


REPRO_TOLERANCE = 0.00005


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 105)
    print(title)
    print("=" * 105)


def normalize_id(value) -> str:
    if pd.isna(value):
        return "__MISSING__"

    value = str(value).strip()

    if (
        value.endswith(".0")
        and
        value[:-2].isdigit()
    ):
        value = value[:-2]

    return value


def clip_label(threshold) -> str:
    if threshold is None:
        return "unclipped"

    return f"clip_{int(threshold)}"


# =============================================================================
# 1. LOAD STEP 4 CANDIDATES
# =============================================================================

def load_ablation_candidates(
    split: str,
) -> pd.DataFrame:

    path = (
        ABLATION_DIR
        /
        f"{split}_candidates.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find Step 4 candidate output:\n  {path}"
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
# 2. LOAD FROZEN BEHAVIOR POLICY
# =============================================================================

def load_behavior_candidates(
    split: str,
) -> pd.DataFrame:

    path = (
        BEHAVIOR_DIR
        /
        f"{split}_behavior_candidates.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find frozen behavior output:\n  {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "def_team": str,
            "pitcher_id": str,
        },
    )

    required = (
        KEYS
        +
        [
            "behavior_prob",
        ]
    )

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
        .apply(normalize_id)
    )

    df["appearance_order"] = (
        pd.to_numeric(
            df["appearance_order"],
            errors="raise",
        )
        .astype(int)
    )

    df["behavior_prob"] = pd.to_numeric(
        df["behavior_prob"],
        errors="raise",
    )

    return (
        df[
            KEYS
            +
            [
                "behavior_prob",
            ]
        ]
        .copy()
    )


# =============================================================================
# 3. LOGGED REWARD LOOKUP
# =============================================================================

def build_logged_reward_lookup() -> dict:

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
        validation,
        test,
    ) = base.chronological_split(
        decisions
    )

    result = {}

    for split_name, frame in [
        (
            "validation",
            validation,
        ),
        (
            "test",
            test,
        ),
    ]:

        work = (
            frame[
                frame["bullpen_source"]
                ==
                "preview"
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

        work["reward"] = pd.to_numeric(
            work["reward"],
            errors="raise",
        )

        lookup = (
            work[
                KEYS
                +
                [
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

        if lookup[KEYS].duplicated().any():
            raise ValueError(
                f"{split_name}: duplicate logged reward keys"
            )

        result[
            split_name
        ] = lookup

        print(
            f"{split_name:10s}: "
            f"{len(lookup):,} logged decisions"
        )

    return result


# =============================================================================
# 4. BUILD OPE CANDIDATE TABLE
# =============================================================================

def build_ope_candidates(
    split: str,
    reward_lookup: pd.DataFrame,
) -> pd.DataFrame:

    section(
        f"2. BUILD OPE CANDIDATES - {split.upper()}"
    )

    candidates = (
        load_ablation_candidates(
            split
        )
    )

    behavior = (
        load_behavior_candidates(
            split
        )
    )

    print(
        f"Step 4 candidate rows   : "
        f"{len(candidates):,}"
    )

    print(
        f"Behavior candidate rows : "
        f"{len(behavior):,}"
    )

    merged = candidates.merge(
        behavior,
        on=KEYS,
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
        on=KEYS,
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
            "Behavior probability does not sum to 1."
        )

    return merged


# =============================================================================
# 5. BUILD TARGET POLICY DECISIONS
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
# 6. OPE METRICS
# =============================================================================

def compute_ope(
    decisions: pd.DataFrame,
    threshold,
):

    frame = (
        decisions.copy()
    )

    if threshold is None:
        frame[
            "weight_used"
        ] = frame[
            "weight"
        ]

    else:
        frame[
            "weight_used"
        ] = (
            frame[
                "weight"
            ]
            .clip(
                upper=threshold
            )
        )

    frame[
        "ips_i"
    ] = (
        frame[
            "weight_used"
        ]
        *
        frame[
            "logged_reward"
        ]
    )

    frame[
        "dr_i"
    ] = (
        frame[
            "dm_i"
        ]
        +
        frame[
            "weight_used"
        ]
        *
        frame[
            "residual"
        ]
    )

    n = len(
        frame
    )

    dm = float(
        frame[
            "dm_i"
        ]
        .mean()
    )

    ips = float(
        frame[
            "ips_i"
        ]
        .mean()
    )

    weight_sum = float(
        frame[
            "weight_used"
        ]
        .sum()
    )

    if weight_sum > 0:
        snips = float(
            frame[
                "ips_i"
            ]
            .sum()
            /
            weight_sum
        )
    else:
        snips = np.nan

    dr = float(
        frame[
            "dr_i"
        ]
        .mean()
    )

    agreement = float(
        frame[
            "pi_actual"
        ]
        .mean()
    )

    weights = (
        frame[
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
            (
                weights.sum()
                ** 2
            )
            /
            denominator
        )
    else:
        ess = 0.0

    positive_weights = (
        weights[
            weights > 0
        ]
    )

    if len(
        positive_weights
    ) > 0:
        max_weight = float(
            positive_weights.max()
        )
    else:
        max_weight = np.nan

    return (
        frame,
        {
            "n":
                int(
                    n
                ),

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
        }
    )


# =============================================================================
# 7. PAIRED GAME BOOTSTRAP
# =============================================================================

def paired_game_bootstrap(
    policy_a: pd.DataFrame,
    policy_b: pd.DataFrame,
) -> dict:

    left = (
        policy_a[
            [
                "decision_id",
                "game_id",
                "dr_i",
            ]
        ]
        .rename(
            columns={
                "dr_i":
                    "dr_a"
            }
        )
    )

    right = (
        policy_b[
            [
                "decision_id",
                "game_id",
                "dr_i",
            ]
        ]
        .rename(
            columns={
                "dr_i":
                    "dr_b"
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
        len(paired)
        !=
        len(policy_a)
        or
        len(paired)
        !=
        len(policy_b)
    ):
        raise ValueError(
            "Policy decision alignment failed."
        )

    paired[
        "delta"
    ] = (
        paired[
            "dr_a"
        ]
        -
        paired[
            "dr_b"
        ]
    )

    point_delta = float(
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

    n_values = (
        games[
            "n"
        ]
        .to_numpy(
            dtype=float
        )
    )

    delta_values = (
        games[
            "delta_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    bootstrap = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )

    for i in range(
        BOOTSTRAP_REPS
    ):

        sample_index = rng.integers(
            0,
            n_games,
            size=n_games,
        )

        bootstrap[
            i
        ] = (
            delta_values[
                sample_index
            ]
            .sum()
            /
            n_values[
                sample_index
            ]
            .sum()
        )

    ci_low = float(
        np.quantile(
            bootstrap,
            0.025,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap,
            0.975,
        )
    )

    probability_positive = float(
        (
            bootstrap > 0
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

        "prob_a_better":
            probability_positive,

        "n_games":
            int(
                n_games
            ),
    }


# =============================================================================
# 8. SPLIT EVALUATION
# =============================================================================

def evaluate_split(
    split: str,
    candidates: pd.DataFrame,
):

    section(
        f"3. B0 / B1 / B2 / B3 OPE - {split.upper()}"
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

    clipped_policy_frames = {}

    for threshold in CLIP_THRESHOLDS:

        threshold_name = (
            clip_label(
                threshold
            )
        )

        clipped_policy_frames[
            threshold_name
        ] = {}

        for (
            policy_name,
            decisions,
        ) in policy_base.items():

            (
                evaluated,
                metrics,
            ) = compute_ope(
                decisions,
                threshold,
            )

            clipped_policy_frames[
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

        for (
            policy_a,
            policy_b,
        ) in PAIRWISE_COMPARISONS:

            comparison = (
                paired_game_bootstrap(
                    clipped_policy_frames[
                        threshold_name
                    ][
                        policy_a
                    ],
                    clipped_policy_frames[
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

    policy_result = (
        pd.DataFrame(
            policy_rows
        )
    )

    pair_result = (
        pd.DataFrame(
            pair_rows
        )
    )

    # -------------------------------------------------------------------------
    # Print policy values
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Print paired comparisons
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # ESS
    # -------------------------------------------------------------------------

    print()
    print(
        "ESS"
    )

    ess_pivot = (
        policy_result.pivot(
            index="threshold",
            columns="policy",
            values="ess",
        )
        .reset_index()
    )

    print(
        ess_pivot.to_string(
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
# 9. REPRODUCTION CHECK
# =============================================================================

def find_pair_delta(
    pair_result: pd.DataFrame,
    comparison: str,
) -> float:

    row = (
        pair_result[
            (
                pair_result[
                    "threshold"
                ]
                ==
                "unclipped"
            )
            &
            (
                pair_result[
                    "comparison"
                ]
                ==
                comparison
            )
        ]
    )

    if len(
        row
    ) != 1:
        raise ValueError(
            f"Could not find unique comparison: {comparison}"
        )

    return float(
        row.iloc[
            0
        ][
            "delta_dr"
        ]
    )


def check_reproduction(
    split: str,
    policy_result: pd.DataFrame,
    pair_result: pd.DataFrame,
) -> bool:

    section(
        f"4. REPRODUCTION CHECK - {split.upper()}"
    )

    expected = (
        EXPECTED_UNCLIPPED[
            split
        ]
    )

    b0_row = (
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
                "B0"
            )
        ]
        .iloc[
            0
        ]
    )

    current_b0 = float(
        b0_row[
            "dr"
        ]
    )

    current_b1_delta = (
        find_pair_delta(
            pair_result,
            "B1_LI - B0",
        )
    )

    current_b2_delta = (
        find_pair_delta(
            pair_result,
            "B2_QUALITY - B0",
        )
    )

    checks = [
        (
            "B0 DR",
            expected[
                "B0"
            ],
            current_b0,
        ),

        (
            "B1-B0",
            expected[
                "B1_minus_B0"
            ],
            current_b1_delta,
        ),

        (
            "B2-B0",
            expected[
                "B2_minus_B0"
            ],
            current_b2_delta,
        ),
    ]

    passed = True

    for (
        name,
        expected_value,
        current_value,
    ) in checks:

        difference = abs(
            current_value
            -
            expected_value
        )

        ok = (
            difference
            <=
            REPRO_TOLERANCE
        )

        print(
            f"{name:10s} | "
            f"expected={expected_value:+.6f} | "
            f"current={current_value:+.6f} | "
            f"diff={difference:.6f} | "
            f"{'PASS' if ok else 'WARNING'}"
        )

        if not ok:
            passed = False

    return passed


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 105
    )

    print(
        "STEP 4-1 - B0 / B1 / B2 / B3 OFFLINE POLICY EVALUATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen baseline"
    )

    print()

    print(
        "B1:"
    )

    print(
        "  B0 + LI + LI x pitcher interactions"
    )

    print()

    print(
        "B2:"
    )

    print(
        "  B0 + quality_p"
    )

    print()

    print(
        "B3:"
    )

    print(
        "  B0 + LI + LI x pitcher interactions + quality_p"
    )

    print()

    print(
        "Behavior:"
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
        "  clipping 10 / 20 / 30 / 50 / 100 / unclipped"
    )

    print(
        "  paired game bootstrap"
    )

    print(
        "=" * 105
    )

    # =========================================================================
    # Logged reward
    # =========================================================================

    reward_lookup = (
        build_logged_reward_lookup()
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
                pair_result,
            )
        )

        all_policy_results.append(
            policy_result
        )

        all_pair_results.append(
            pair_result
        )

        # Decision-level base tables
        for (
            policy_name,
            decisions,
        ) in policy_base.items():

            safe_name = (
                policy_name.lower()
            )

            decisions.to_csv(
                OUTPUT_DIR
                /
                f"{split}_{safe_name}_decisions.csv",
                index=False,
            )

    policy_summary = pd.concat(
        all_policy_results,
        ignore_index=True,
    )

    pair_summary = pd.concat(
        all_pair_results,
        ignore_index=True,
    )

    # =========================================================================
    # Final focus
    # =========================================================================

    section(
        "FINAL - B3 POLICY VALUE FOCUS"
    )

    focus = (
        pair_summary[
            pair_summary[
                "comparison"
            ]
            .isin(
                [
                    "B3_LI_QUALITY - B0",
                    "B3_LI_QUALITY - B1_LI",
                    "B3_LI_QUALITY - B2_QUALITY",
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
        "b3_focus_comparisons.csv",
        index=False,
    )

    print(
        f"Saved to:\n  {OUTPUT_DIR}"
    )

    # =========================================================================
    # Status
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
        "Primary B3 questions:"
    )

    print(
        "  1. Is B3 - B0 positive?"
    )

    print(
        "  2. Is B3 - B1 positive?"
    )

    print(
        "  3. Does the paired 95% CI exclude zero?"
    )

    print(
        "  4. Is the direction stable across clipping?"
    )

    print(
        "  5. Is ESS comparable?"
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()