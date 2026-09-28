from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base


# =============================================================================
# CONFIG
# =============================================================================

QUALITY_ABLATION_DIR = Path(
    "outputs/bandit/quality/v3_quality_ablation"
)

BEHAVIOR_DIR = Path(
    "outputs/bandit/ope/v2_behavior_policy"
)

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_quality_ope"
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


# 기존 Step 12에서 확정된 B0 OPE 값
EXPECTED_B0_UNCLIPPED = {
    "validation": 0.005778,
    "test": 0.003222,
}


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def normalize_id(value) -> str:
    if pd.isna(value):
        return "__MISSING__"

    value = str(value).strip()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    return value


def clip_label(threshold) -> str:
    if threshold is None:
        return "unclipped"

    return f"clip_{int(threshold)}"


# =============================================================================
# 1. QUALITY ABLATION CANDIDATES
# =============================================================================

def load_quality_candidates(
    split: str,
) -> pd.DataFrame:

    path = (
        QUALITY_ABLATION_DIR
        /
        f"{split}_candidates.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find quality ablation output:\n  {path}"
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
        "quality_p",
        "q_b0",
        "q_b2_quality",
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

    for col in [
        "quality_p",
        "q_b0",
        "q_b2_quality",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    return df


# =============================================================================
# 2. FROZEN BEHAVIOR POLICY
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
            f"Cannot find frozen behavior policy output:\n  {path}"
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

    df["behavior_prob"] = (
        pd.to_numeric(
            df["behavior_prob"],
            errors="raise",
        )
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

        work["reward"] = (
            pd.to_numeric(
                work["reward"],
                errors="raise",
            )
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

        duplicates = int(
            lookup[
                KEYS
            ]
            .duplicated()
            .sum()
        )

        if duplicates > 0:
            raise ValueError(
                f"Duplicate reward keys: {duplicates}"
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
# 4. OPE CANDIDATE TABLE
# =============================================================================

def build_ope_candidates(
    split: str,
    reward_lookup: pd.DataFrame,
) -> pd.DataFrame:

    section(
        f"2. BUILD OPE CANDIDATES - {split.upper()}"
    )

    quality = (
        load_quality_candidates(
            split
        )
    )

    behavior = (
        load_behavior_candidates(
            split
        )
    )

    print(
        f"Quality candidate rows  : "
        f"{len(quality):,}"
    )

    print(
        f"Behavior candidate rows : "
        f"{len(behavior):,}"
    )

    merged = quality.merge(
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

    # -------------------------------------------------------------------------
    # 각 decision에서 behavior probability 합 = 1 확인
    # -------------------------------------------------------------------------

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
# 5. TARGET POLICY
# =============================================================================

def select_policy(
    candidates: pd.DataFrame,
    model_name: str,
):

    if model_name == "B0":
        score_col = "q_b0"

    elif model_name == "B2_QUALITY":
        score_col = "q_b2_quality"

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
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

    return (
        score_col,
        selected,
    )


# =============================================================================
# 6. DECISION-LEVEL OPE DATA
# =============================================================================

def build_policy_decisions(
    candidates: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    (
        score_col,
        selected,
    ) = select_policy(
        candidates,
        model_name,
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

    actual["model"] = (
        model_name
    )

    return actual


# =============================================================================
# 7. OPE METRICS
# =============================================================================

def compute_ope_metrics(
    decisions: pd.DataFrame,
    threshold,
):

    frame = (
        decisions.copy()
    )

    if threshold is None:
        frame["weight_used"] = (
            frame["weight"]
        )

    else:
        frame["weight_used"] = (
            frame["weight"]
            .clip(
                upper=threshold
            )
        )

    frame["ips_i"] = (
        frame["weight_used"]
        *
        frame["logged_reward"]
    )

    frame["dr_i"] = (
        frame["dm_i"]
        +
        frame["weight_used"]
        *
        frame["residual"]
    )

    n = len(
        frame
    )

    dm = float(
        frame["dm_i"]
        .mean()
    )

    ips = float(
        frame["ips_i"]
        .mean()
    )

    weight_sum = float(
        frame["weight_used"]
        .sum()
    )

    if weight_sum > 0:
        snips = float(
            frame["ips_i"]
            .sum()
            /
            weight_sum
        )
    else:
        snips = np.nan

    dr = float(
        frame["dr_i"]
        .mean()
    )

    agreement = float(
        frame["pi_actual"]
        .mean()
    )

    weights = (
        frame["weight_used"]
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

    nonzero = (
        weights[
            weights > 0
        ]
    )

    if len(nonzero) > 0:
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
                frame["weight"]
                >
                threshold
            )
            .mean()
        )

    return (
        frame,
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


# =============================================================================
# 8. PAIRED GAME BOOTSTRAP
# =============================================================================

def paired_bootstrap(
    b0: pd.DataFrame,
    b2: pd.DataFrame,
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
        b2[
            [
                "decision_id",
                "game_id",
                "dr_i",
            ]
        ]
        .rename(
            columns={
                "dr_i":
                    "dr_b2"
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
        len(paired) != len(b2)
    ):
        raise ValueError(
            "B0/B2 decision sets are not aligned."
        )

    paired["delta_dr"] = (
        paired["dr_b2"]
        -
        paired["dr_b0"]
    )

    point_delta = float(
        paired["delta_dr"]
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

        bootstrap[i] = (
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

        "prob_b2_better":
            probability_positive,

        "n_games":
            int(
                n_games
            ),
    }


# =============================================================================
# 9. SPLIT EVALUATION
# =============================================================================

def evaluate_split(
    split: str,
    candidates: pd.DataFrame,
):

    section(
        f"3. B0 VS B2 QUALITY OPE - {split.upper()}"
    )

    b0_base = build_policy_decisions(
        candidates,
        "B0",
    )

    b2_base = build_policy_decisions(
        candidates,
        "B2_QUALITY",
    )

    rows = []

    for threshold in CLIP_THRESHOLDS:
        label = clip_label(
            threshold
        )

        (
            b0,
            metrics_b0,
        ) = compute_ope_metrics(
            b0_base,
            threshold,
        )

        (
            b2,
            metrics_b2,
        ) = compute_ope_metrics(
            b2_base,
            threshold,
        )

        paired = paired_bootstrap(
            b0,
            b2,
        )

        rows.append(
            {
                "split":
                    split,

                "threshold":
                    label,

                "b0_agreement":
                    metrics_b0[
                        "agreement"
                    ],

                "b2_agreement":
                    metrics_b2[
                        "agreement"
                    ],

                "b0_dm":
                    metrics_b0[
                        "dm"
                    ],

                "b2_dm":
                    metrics_b2[
                        "dm"
                    ],

                "b0_ips":
                    metrics_b0[
                        "ips"
                    ],

                "b2_ips":
                    metrics_b2[
                        "ips"
                    ],

                "b0_snips":
                    metrics_b0[
                        "snips"
                    ],

                "b2_snips":
                    metrics_b2[
                        "snips"
                    ],

                "b0_dr":
                    metrics_b0[
                        "dr"
                    ],

                "b2_dr":
                    metrics_b2[
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

                "prob_b2_better":
                    paired[
                        "prob_b2_better"
                    ],

                "b0_ess":
                    metrics_b0[
                        "ess"
                    ],

                "b2_ess":
                    metrics_b2[
                        "ess"
                    ],

                "b0_ess_fraction":
                    metrics_b0[
                        "ess_fraction"
                    ],

                "b2_ess_fraction":
                    metrics_b2[
                        "ess_fraction"
                    ],

                "b0_max_weight":
                    metrics_b0[
                        "max_weight"
                    ],

                "b2_max_weight":
                    metrics_b2[
                        "max_weight"
                    ],

                "b0_clipped_fraction":
                    metrics_b0[
                        "clipped_fraction"
                    ],

                "b2_clipped_fraction":
                    metrics_b2[
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

    display[
        "prob_b2_better"
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
                "b2_dr",
                "delta_dr",
                "delta_ci_low",
                "delta_ci_high",
                "prob_b2_better",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "b0_dr":
                    lambda x:
                    f"{x:+.6f}",

                "b2_dr":
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

                "prob_b2_better":
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
                "b2_ips",
                "b0_snips",
                "b2_snips",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "b0_ips":
                    lambda x:
                    f"{x:+.6f}",

                "b2_ips":
                    lambda x:
                    f"{x:+.6f}",

                "b0_snips":
                    lambda x:
                    f"{x:+.6f}",

                "b2_snips":
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
                "b2_agreement",
                "b0_ess",
                "b2_ess",
                "b0_ess_fraction",
                "b2_ess_fraction",
                "b0_max_weight",
                "b2_max_weight",
            ]
        ]
        .assign(
            b0_agreement=lambda x:
                x["b0_agreement"] * 100,

            b2_agreement=lambda x:
                x["b2_agreement"] * 100,

            b0_ess_fraction=lambda x:
                x["b0_ess_fraction"] * 100,

            b2_ess_fraction=lambda x:
                x["b2_ess_fraction"] * 100,
        )
        .to_string(
            index=False,
            formatters={
                "b0_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "b2_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "b0_ess":
                    lambda x:
                    f"{x:.1f}",

                "b2_ess":
                    lambda x:
                    f"{x:.1f}",

                "b0_ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "b2_ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "b0_max_weight":
                    lambda x:
                    f"{x:.2f}",

                "b2_max_weight":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    return (
        result,
        b0_base,
        b2_base,
    )


# =============================================================================
# 10. B0 REPRODUCTION
# =============================================================================

def check_b0_ope_reproduction(
    split: str,
    result: pd.DataFrame,
) -> bool:

    section(
        f"4. B0 OPE REPRODUCTION - {split.upper()}"
    )

    row = (
        result[
            result["threshold"]
            ==
            "unclipped"
        ]
        .iloc[0]
    )

    expected = (
        EXPECTED_B0_UNCLIPPED[
            split
        ]
    )

    actual = float(
        row["b0_dr"]
    )

    difference = abs(
        actual
        -
        expected
    )

    print(
        f"Expected B0 DR : {expected:+.6f}"
    )

    print(
        f"Current B0 DR  : {actual:+.6f}"
    )

    print(
        f"Difference     : {difference:.6f}"
    )

    passed = (
        difference
        <=
        0.00005
    )

    print()
    print(
        "B0 OPE reproduction : "
        +
        (
            "PASS"
            if passed
            else "WARNING"
        )
    )

    return passed


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print(
        "=" * 96
    )

    print(
        "STEP 3-1 - B0 VS B2(+QUALITY_P) OFFLINE POLICY EVALUATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen pre-quality reward policy"
    )

    print()

    print(
        "B2:"
    )

    print(
        "  B0 + train-frozen quality_p"
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
        "  Paired game-bootstrap B2 - B0"
    )

    print(
        "=" * 96
    )

    # =========================================================================
    # Logged reward
    # =========================================================================

    reward_lookup = (
        build_logged_reward_lookup()
    )

    # =========================================================================
    # Validation
    # =========================================================================

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
        validation_b2,
    ) = evaluate_split(
        "validation",
        validation_candidates,
    )

    validation_reproduction = (
        check_b0_ope_reproduction(
            "validation",
            validation_result,
        )
    )

    # =========================================================================
    # Test
    # =========================================================================

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
        test_b2,
    ) = evaluate_split(
        "test",
        test_candidates,
    )

    test_reproduction = (
        check_b0_ope_reproduction(
            "test",
            test_result,
        )
    )

    # =========================================================================
    # Combined
    # =========================================================================

    combined = pd.concat(
        [
            validation_result,
            test_result,
        ],
        ignore_index=True,
    )

    # =========================================================================
    # Final comparison
    # =========================================================================

    section(
        "FINAL B2 - B0 POLICY VALUE COMPARISON"
    )

    focus = (
        combined[
            [
                "split",
                "threshold",
                "b0_dr",
                "b2_dr",
                "delta_dr",
                "delta_ci_low",
                "delta_ci_high",
                "prob_b2_better",
                "b0_ess",
                "b2_ess",
            ]
        ]
        .copy()
    )

    focus[
        "prob_b2_better"
    ] *= 100

    print(
        focus.to_string(
            index=False,
            formatters={
                "b0_dr":
                    lambda x:
                    f"{x:+.6f}",

                "b2_dr":
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

                "prob_b2_better":
                    lambda x:
                    f"{x:.2f}%",

                "b0_ess":
                    lambda x:
                    f"{x:.1f}",

                "b2_ess":
                    lambda x:
                    f"{x:.1f}",
            },
        )
    )

    # =========================================================================
    # Save
    # =========================================================================

    section(
        "SAVE"
    )

    validation_result.to_csv(
        OUTPUT_DIR
        /
        "validation_ope.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        /
        "test_ope.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT_DIR
        /
        "b0_b2_quality_ope_all.csv",
        index=False,
    )

    validation_b0.to_csv(
        OUTPUT_DIR
        /
        "validation_b0_decisions.csv",
        index=False,
    )

    validation_b2.to_csv(
        OUTPUT_DIR
        /
        "validation_b2_decisions.csv",
        index=False,
    )

    test_b0.to_csv(
        OUTPUT_DIR
        /
        "test_b0_decisions.csv",
        index=False,
    )

    test_b2.to_csv(
        OUTPUT_DIR
        /
        "test_b2_decisions.csv",
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

    print(
        "Validation B0 OPE reproduction : "
        +
        (
            "PASS"
            if validation_reproduction
            else "WARNING"
        )
    )

    print(
        "Test B0 OPE reproduction       : "
        +
        (
            "PASS"
            if test_reproduction
            else "WARNING"
        )
    )

    print()
    print(
        "Interpret B2 only after checking:"
    )

    print(
        "  1. paired delta DR"
    )

    print(
        "  2. paired 95% CI"
    )

    print(
        "  3. clipping robustness"
    )

    print(
        "  4. ESS"
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()