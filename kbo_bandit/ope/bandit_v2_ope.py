from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

BEHAVIOR_DIR = Path(
    "outputs/bandit/ope/v2_behavior_policy"
)

REWARD_DIR = Path(
    "outputs/bandit/diagnostics/v2_reward_recheck"
    "/wpa_per_pa"
)

OUTPUT_DIR = Path(
    "outputs/bandit/ope/v2_ope"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Weight clipping은 main 결과가 아니라
# sensitivity check 용도
WEIGHT_CLIP = 50.0

BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42


KEYS = [
    "game_id",
    "def_team",
    "appearance_order",
    "pitcher_id",
]


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def normalize_bool(series):

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# 1. Load behavior candidates
# ============================================================

def load_behavior_candidates(split):

    path = (
        BEHAVIOR_DIR
        / f"{split}_behavior_candidates.csv"
    )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "pitcher_id": str,
            "actual_pitcher": str,
            "def_team": str,
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

    df["is_actual"] = (
        normalize_bool(
            df["is_actual"]
        )
    )

    df["behavior_prob"] = (
        pd.to_numeric(
            df["behavior_prob"],
            errors="raise",
        )
    )

    df["prior_n_apps"] = (
        pd.to_numeric(
            df["prior_n_apps"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return df


# ============================================================
# 2. Load reward-model candidate scores
# ============================================================

def load_reward_candidates(split):

    path = (
        REWARD_DIR
        / f"{split}_candidates.csv"
    )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "pitcher_id": str,
            "def_team": str,
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

    df["pred_mean"] = (
        pd.to_numeric(
            df["pred_mean"],
            errors="raise",
        )
    )

    return (
        df[
            KEYS
            + [
                "pred_mean",
            ]
        ]
        .rename(
            columns={
                "pred_mean":
                    "q_hat"
            }
        )
    )


# ============================================================
# 3. Logged reward lookup
# ============================================================

def load_logged_rewards():

    print_section(
        "LOAD LOGGED WPA_PER_PA REWARDS"
    )

    base.REWARD_MODE = (
        "wpa_per_pa"
    )

    base.ALPHA = (
        0.0
    )

    (
        pa,
        history,
        bullpen,
    ) = (
        base.load_data()
    )

    (
        decisions,
        first_app,
    ) = (
        base.build_decision_dataset(
            pa,
            history,
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

    lookups = {}

    for name, df in [
        (
            "validation",
            val,
        ),
        (
            "test",
            test,
        ),
    ]:

        df = (
            df[
                df[
                    "bullpen_source"
                ]
                == "preview"
            ]
            .copy()
        )

        df["game_id"] = (
            df["game_id"]
            .astype(str)
        )

        df["pitcher_id"] = (
            df["pitcher_id"]
            .apply(normalize_id)
        )

        df["def_team"] = (
            df["def_team"]
            .astype(str)
        )

        df["appearance_order"] = (
            df["appearance_order"]
            .astype(int)
        )

        lookup = (
            df[
                [
                    "game_id",
                    "def_team",
                    "appearance_order",
                    "pitcher_id",
                    "reward",
                ]
            ]
            .copy()
        )

        lookup = (
            lookup.rename(
                columns={
                    "reward":
                        "logged_reward"
                }
            )
        )

        lookups[name] = (
            lookup
        )

        print(
            f"{name:10s}: "
            f"{len(lookup):,} logged decisions"
        )

    return lookups


# ============================================================
# 4. Merge all OPE inputs
# ============================================================

def build_ope_candidates(
    split,
    reward_lookup,
):

    print_section(
        f"BUILD OPE DATASET - {split.upper()}"
    )

    behavior = (
        load_behavior_candidates(
            split
        )
    )

    q_scores = (
        load_reward_candidates(
            split
        )
    )

    # --------------------------------------------------------
    # Behavior candidate set
    # + reward model Q(s,a)
    # --------------------------------------------------------

    merged = (
        behavior.merge(
            q_scores,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    )

    missing_q = (
        merged[
            "q_hat"
        ]
        .isna()
        .sum()
    )

    print(
        f"Behavior candidate rows : "
        f"{len(behavior):,}"
    )

    print(
        f"Merged candidate rows   : "
        f"{len(merged):,}"
    )

    print(
        f"Missing Q(s,a)          : "
        f"{missing_q:,}"
    )

    if missing_q > 0:

        raise ValueError(
            "Reward candidate score join failed."
        )

    # --------------------------------------------------------
    # logged reward는 actual action row에만 붙인다.
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
        ]
    )

    missing_reward = (
        actual[
            "logged_reward"
        ]
        .isna()
        .sum()
    )

    print(
        f"Decisions               : "
        f"{actual['decision_id'].nunique():,}"
    )

    print(
        f"Missing logged reward   : "
        f"{missing_reward:,}"
    )

    if missing_reward > 0:

        raise ValueError(
            "Actual logged reward join failed."
        )

    # --------------------------------------------------------
    # behavior probability sanity
    # --------------------------------------------------------

    prob_sum = (
        merged.groupby(
            "decision_id"
        )[
            "behavior_prob"
        ]
        .sum()
    )

    max_prob_sum_error = float(
        (
            prob_sum
            - 1.0
        )
        .abs()
        .max()
    )

    print(
        f"Max behavior prob error : "
        f"{max_prob_sum_error:.8f}"
    )

    if (
        max_prob_sum_error
        > 1e-5
    ):

        raise ValueError(
            "Behavior probabilities do not "
            "sum to 1 inside a decision."
        )

    return merged


# ============================================================
# 5. Target policies
# ============================================================

def assign_target_policy(
    candidates,
    policy_name,
):

    df = (
        candidates.copy()
    )

    df[
        "target_prob"
    ] = (
        0.0
    )

    # ========================================================
    # Random policy
    # ========================================================

    if (
        policy_name
        == "random"
    ):

        counts = (
            df.groupby(
                "decision_id"
            )[
                "pitcher_id"
            ]
            .transform(
                "size"
            )
        )

        df[
            "target_prob"
        ] = (
            1.0
            /
            counts
        )

    # ========================================================
    # prior_n_apps heuristic
    # ========================================================

    elif (
        policy_name
        == "heuristic"
    ):

        ordered = (
            df.sort_values(
                [
                    "decision_id",
                    "prior_n_apps",
                    "pitcher_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
        )

        selected_idx = (
            ordered.groupby(
                "decision_id"
            )
            .head(1)
            .index
        )

        df.loc[
            selected_idx,
            "target_prob",
        ] = (
            1.0
        )

    # ========================================================
    # Our reward policy
    #
    # argmax predicted wpa_per_pa reward
    # ========================================================

    elif (
        policy_name
        == "reward_policy"
    ):

        ordered = (
            df.sort_values(
                [
                    "decision_id",
                    "q_hat",
                    "pitcher_id",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
        )

        selected_idx = (
            ordered.groupby(
                "decision_id"
            )
            .head(1)
            .index
        )

        df.loc[
            selected_idx,
            "target_prob",
        ] = (
            1.0
        )

    else:

        raise ValueError(
            f"Unknown policy: {policy_name}"
        )

    # --------------------------------------------------------
    # sanity check
    # --------------------------------------------------------

    target_sum = (
        df.groupby(
            "decision_id"
        )[
            "target_prob"
        ]
        .sum()
    )

    max_error = float(
        (
            target_sum
            - 1.0
        )
        .abs()
        .max()
    )

    if (
        max_error
        > 1e-10
    ):

        raise ValueError(
            "Target probabilities "
            "do not sum to 1."
        )

    return df


# ============================================================
# 6. OPE decision table
# ============================================================

def build_decision_contributions(
    candidates,
    policy_name,
):

    df = (
        assign_target_policy(
            candidates,
            policy_name,
        )
    )

    # --------------------------------------------------------
    # Direct Method expectation
    #
    # DM_i = sum_a pi(a|s) Q_hat(s,a)
    # --------------------------------------------------------

    df[
        "pi_q"
    ] = (
        df[
            "target_prob"
        ]
        *
        df[
            "q_hat"
        ]
    )

    dm = (
        df.groupby(
            "decision_id"
        )[
            "pi_q"
        ]
        .sum()
        .rename(
            "dm_i"
        )
    )

    # --------------------------------------------------------
    # Logged action rows
    # --------------------------------------------------------

    actual = (
        df[
            df[
                "is_actual"
            ]
        ]
        .copy()
    )

    actual = (
        actual.rename(
            columns={
                "behavior_prob":
                    "mu_actual",

                "target_prob":
                    "pi_actual",

                "q_hat":
                    "q_actual",
            }
        )
    )

    actual = (
        actual.merge(
            dm,
            left_on=(
                "decision_id"
            ),
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Importance weight
    #
    # w = pi(a_logged|s) / mu(a_logged|s)
    # --------------------------------------------------------

    eps = (
        1e-12
    )

    actual[
        "mu_actual"
    ] = (
        actual[
            "mu_actual"
        ]
        .clip(
            lower=eps
        )
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
        "weight_clipped"
    ] = (
        actual[
            "weight"
        ]
        .clip(
            upper=(
                WEIGHT_CLIP
            )
        )
    )

    # --------------------------------------------------------
    # Logged residual
    #
    # r - Q_hat(s, a_logged)
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IPS contribution
    # --------------------------------------------------------

    actual[
        "ips_num_i"
    ] = (
        actual[
            "weight"
        ]
        *
        actual[
            "logged_reward"
        ]
    )

    actual[
        "ips_num_clipped_i"
    ] = (
        actual[
            "weight_clipped"
        ]
        *
        actual[
            "logged_reward"
        ]
    )

    # --------------------------------------------------------
    # Doubly Robust
    #
    # DR_i =
    # DM_i
    # +
    # w_i * (r_i - Q_actual)
    # --------------------------------------------------------

    actual[
        "dr_i"
    ] = (
        actual[
            "dm_i"
        ]
        +
        actual[
            "weight"
        ]
        *
        actual[
            "residual"
        ]
    )

    actual[
        "dr_clipped_i"
    ] = (
        actual[
            "dm_i"
        ]
        +
        actual[
            "weight_clipped"
        ]
        *
        actual[
            "residual"
        ]
    )

    actual[
        "policy"
    ] = (
        policy_name
    )

    return actual


# ============================================================
# 7. Compute OPE metrics
# ============================================================

def compute_metrics(
    decision_df,
):

    n = (
        len(
            decision_df
        )
    )

    reward_mean = float(
        decision_df[
            "logged_reward"
        ]
        .mean()
    )

    dm = float(
        decision_df[
            "dm_i"
        ]
        .mean()
    )

    ips = float(
        decision_df[
            "ips_num_i"
        ]
        .mean()
    )

    weight_sum = float(
        decision_df[
            "weight"
        ]
        .sum()
    )

    if (
        weight_sum
        > 0
    ):

        snips = float(
            decision_df[
                "ips_num_i"
            ]
            .sum()
            /
            weight_sum
        )

    else:

        snips = (
            np.nan
        )

    dr = float(
        decision_df[
            "dr_i"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Clipped sensitivity
    # --------------------------------------------------------

    ips_clip = float(
        decision_df[
            "ips_num_clipped_i"
        ]
        .mean()
    )

    clipped_weight_sum = float(
        decision_df[
            "weight_clipped"
        ]
        .sum()
    )

    if (
        clipped_weight_sum
        > 0
    ):

        snips_clip = float(
            decision_df[
                "ips_num_clipped_i"
            ]
            .sum()
            /
            clipped_weight_sum
        )

    else:

        snips_clip = (
            np.nan
        )

    dr_clip = float(
        decision_df[
            "dr_clipped_i"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # ESS
    #
    # Effective Sample Size
    # --------------------------------------------------------

    w = (
        decision_df[
            "weight"
        ]
        .to_numpy(
            dtype=float
        )
    )

    w2_sum = float(
        np.sum(
            w ** 2
        )
    )

    if (
        w2_sum
        > 0
    ):

        ess = float(
            (
                np.sum(w) ** 2
            )
            /
            w2_sum
        )

    else:

        ess = (
            0.0
        )

    nonzero = (
        decision_df[
            decision_df[
                "weight"
            ]
            > 0
        ]
        .copy()
    )

    if (
        len(nonzero)
        > 0
    ):

        max_weight = float(
            nonzero[
                "weight"
            ]
            .max()
        )

        p95_weight = float(
            nonzero[
                "weight"
            ]
            .quantile(
                0.95
            )
        )

        p99_weight = float(
            nonzero[
                "weight"
            ]
            .quantile(
                0.99
            )
        )

    else:

        max_weight = (
            np.nan
        )

        p95_weight = (
            np.nan
        )

        p99_weight = (
            np.nan
        )

    # deterministic policy에서는
    # mean(pi_actual) = Agreement
    #
    # random에서는 expected agreement
    agreement = float(
        decision_df[
            "pi_actual"
        ]
        .mean()
    )

    match_count = int(
        (
            decision_df[
                "pi_actual"
            ]
            > 0
        )
        .sum()
    )

    clipped_fraction = float(
        (
            decision_df[
                "weight"
            ]
            >
            WEIGHT_CLIP
        )
        .mean()
    )

    return {
        "n":
            int(n),

        "logged_manager_reward":
            reward_mean,

        "agreement_or_expected":
            agreement,

        "matched_or_supported_n":
            match_count,

        "dm":
            dm,

        "ips":
            ips,

        "snips":
            snips,

        "dr":
            dr,

        "ips_clip50":
            ips_clip,

        "snips_clip50":
            snips_clip,

        "dr_clip50":
            dr_clip,

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

        "p95_nonzero_weight":
            p95_weight,

        "p99_nonzero_weight":
            p99_weight,

        "weight_clipped_fraction":
            clipped_fraction,
    }


# ============================================================
# 8. Cluster bootstrap by game
# ============================================================

def bootstrap_ci(
    decision_df,
):

    # --------------------------------------------------------
    # 같은 게임 내 여러 교체 decision은
    # 서로 독립이라고 보기 어렵다.
    #
    # 따라서 decision이 아니라 game_id 단위 bootstrap.
    # --------------------------------------------------------

    game_agg = (
        decision_df.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            n=(
                "decision_id",
                "size",
            ),

            dm_sum=(
                "dm_i",
                "sum",
            ),

            ips_num_sum=(
                "ips_num_i",
                "sum",
            ),

            weight_sum=(
                "weight",
                "sum",
            ),

            dr_sum=(
                "dr_i",
                "sum",
            ),

            ips_num_clip_sum=(
                "ips_num_clipped_i",
                "sum",
            ),

            weight_clip_sum=(
                "weight_clipped",
                "sum",
            ),

            dr_clip_sum=(
                "dr_clipped_i",
                "sum",
            ),
        )
    )

    n_games = (
        len(
            game_agg
        )
    )

    rng = (
        np.random.default_rng(
            BOOTSTRAP_SEED
        )
    )

    estimates = {
        "dm": [],
        "ips": [],
        "snips": [],
        "dr": [],
        "ips_clip50": [],
        "snips_clip50": [],
        "dr_clip50": [],
    }

    # numpy arrays for speed
    n_arr = (
        game_agg[
            "n"
        ]
        .to_numpy(
            dtype=float
        )
    )

    dm_arr = (
        game_agg[
            "dm_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ips_arr = (
        game_agg[
            "ips_num_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    w_arr = (
        game_agg[
            "weight_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    dr_arr = (
        game_agg[
            "dr_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ips_c_arr = (
        game_agg[
            "ips_num_clip_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    w_c_arr = (
        game_agg[
            "weight_clip_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    dr_c_arr = (
        game_agg[
            "dr_clip_sum"
        ]
        .to_numpy(
            dtype=float
        )
    )

    for _ in range(
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

        total_dm = float(
            dm_arr[
                sampled
            ]
            .sum()
        )

        total_ips = float(
            ips_arr[
                sampled
            ]
            .sum()
        )

        total_w = float(
            w_arr[
                sampled
            ]
            .sum()
        )

        total_dr = float(
            dr_arr[
                sampled
            ]
            .sum()
        )

        total_ips_c = float(
            ips_c_arr[
                sampled
            ]
            .sum()
        )

        total_w_c = float(
            w_c_arr[
                sampled
            ]
            .sum()
        )

        total_dr_c = float(
            dr_c_arr[
                sampled
            ]
            .sum()
        )

        estimates[
            "dm"
        ].append(
            total_dm
            /
            total_n
        )

        estimates[
            "ips"
        ].append(
            total_ips
            /
            total_n
        )

        if (
            total_w
            > 0
        ):

            estimates[
                "snips"
            ].append(
                total_ips
                /
                total_w
            )

        else:

            estimates[
                "snips"
            ].append(
                np.nan
            )

        estimates[
            "dr"
        ].append(
            total_dr
            /
            total_n
        )

        estimates[
            "ips_clip50"
        ].append(
            total_ips_c
            /
            total_n
        )

        if (
            total_w_c
            > 0
        ):

            estimates[
                "snips_clip50"
            ].append(
                total_ips_c
                /
                total_w_c
            )

        else:

            estimates[
                "snips_clip50"
            ].append(
                np.nan
            )

        estimates[
            "dr_clip50"
        ].append(
            total_dr_c
            /
            total_n
        )

    ci = {}

    for key, values in (
        estimates.items()
    ):

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

        if (
            len(arr)
            == 0
        ):

            ci[
                f"{key}_ci_low"
            ] = np.nan

            ci[
                f"{key}_ci_high"
            ] = np.nan

        else:

            ci[
                f"{key}_ci_low"
            ] = float(
                np.quantile(
                    arr,
                    0.025,
                )
            )

            ci[
                f"{key}_ci_high"
            ] = float(
                np.quantile(
                    arr,
                    0.975,
                )
            )

    ci[
        "bootstrap_games"
    ] = int(
        n_games
    )

    ci[
        "bootstrap_reps"
    ] = int(
        BOOTSTRAP_REPS
    )

    return ci


# ============================================================
# 9. Run one split
# ============================================================

def evaluate_split(
    split,
    candidates,
):

    print_section(
        f"OPE - {split.upper()}"
    )

    all_decisions = []

    summary_rows = []

    for policy in [
        "random",
        "heuristic",
        "reward_policy",
    ]:

        print()
        print(
            "-" * 78
        )

        print(
            f"POLICY: {policy}"
        )

        print(
            "-" * 78
        )

        decision_df = (
            build_decision_contributions(
                candidates,
                policy,
            )
        )

        metrics = (
            compute_metrics(
                decision_df
            )
        )

        ci = (
            bootstrap_ci(
                decision_df
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

        summary_rows.append(
            row
        )

        all_decisions.append(
            decision_df
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"N decisions          : "
            f"{metrics['n']:,}"
        )

        print(
            f"Agreement/Expected   : "
            f"{metrics['agreement_or_expected'] * 100:.2f}%"
        )

        print(
            f"Logged manager reward: "
            f"{metrics['logged_manager_reward']:.6f}"
        )

        print()

        print(
            "Policy value estimates "
            "(higher is better):"
        )

        print(
            f"  DM     : "
            f"{metrics['dm']:.6f}"
        )

        print(
            f"  IPS    : "
            f"{metrics['ips']:.6f}"
        )

        print(
            f"  SNIPS  : "
            f"{metrics['snips']:.6f}"
        )

        print(
            f"  DR     : "
            f"{metrics['dr']:.6f}"
        )

        print()

        print(
            f"  DR 95% game-bootstrap CI: "
            f"[{ci['dr_ci_low']:.6f}, "
            f"{ci['dr_ci_high']:.6f}]"
        )

        print()

        print(
            f"Clip {WEIGHT_CLIP:.0f} sensitivity:"
        )

        print(
            f"  IPS clip   : "
            f"{metrics['ips_clip50']:.6f}"
        )

        print(
            f"  SNIPS clip : "
            f"{metrics['snips_clip50']:.6f}"
        )

        print(
            f"  DR clip    : "
            f"{metrics['dr_clip50']:.6f}"
        )

        print()

        print(
            f"ESS                  : "
            f"{metrics['ess']:.1f} "
            f"({metrics['ess_fraction'] * 100:.2f}% of N)"
        )

        print(
            f"Max weight           : "
            f"{metrics['max_weight']:.2f}"
        )

        print(
            f"p95 nonzero weight   : "
            f"{metrics['p95_nonzero_weight']:.2f}"
        )

        print(
            f"p99 nonzero weight   : "
            f"{metrics['p99_nonzero_weight']:.2f}"
        )

        print(
            f"Weight > {WEIGHT_CLIP:.0f}      : "
            f"{metrics['weight_clipped_fraction'] * 100:.2f}%"
        )

    summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    decisions = (
        pd.concat(
            all_decisions,
            ignore_index=True,
        )
    )

    return (
        summary,
        decisions,
    )


# ============================================================
# 10. Pretty summary
# ============================================================

def print_final_summary(
    name,
    summary,
):

    print_section(
        f"FINAL {name.upper()} POLICY COMPARISON"
    )

    table = (
        summary[
            [
                "policy",
                "agreement_or_expected",
                "dm",
                "ips",
                "snips",
                "dr",
                "dr_ci_low",
                "dr_ci_high",
                "dr_clip50",
                "ess",
                "ess_fraction",
                "max_weight",
            ]
        ]
        .copy()
    )

    table[
        "agreement_or_expected"
    ] *= (
        100
    )

    table[
        "ess_fraction"
    ] *= (
        100
    )

    print(
        table.to_string(
            index=False,
            formatters={
                "agreement_or_expected":
                    lambda x:
                    f"{x:.2f}%",

                "dm":
                    lambda x:
                    f"{x:.6f}",

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

                "dr_clip50":
                    lambda x:
                    f"{x:.6f}",

                "ess":
                    lambda x:
                    f"{x:.1f}",

                "ess_fraction":
                    lambda x:
                    f"{x:.2f}%",

                "max_weight":
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
        "=" * 78
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- OFFLINE POLICY EVALUATION"
    )

    print()

    print(
        "Reward:"
    )

    print(
        "  -Option B WPA / n_pa"
    )

    print()

    print(
        "Policies:"
    )

    print(
        "  random"
    )

    print(
        "  prior_n_apps heuristic"
    )

    print(
        "  reward-model policy"
    )

    print()

    print(
        "Estimators:"
    )

    print(
        "  DM / IPS / SNIPS / DR"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Behavior propensities are estimated, "
        "not known randomized logging probabilities."
    )

    print(
        "Therefore OPE is supplementary evidence, "
        "not an unbiased causal policy-value estimate."
    )

    print(
        "=" * 78
    )

    # ========================================================
    # Logged rewards
    # ========================================================

    reward_lookups = (
        load_logged_rewards()
    )

    # ========================================================
    # Validation
    # ========================================================

    val_candidates = (
        build_ope_candidates(
            "validation",
            reward_lookups[
                "validation"
            ],
        )
    )

    (
        val_summary,
        val_decisions,
    ) = (
        evaluate_split(
            "validation",
            val_candidates,
        )
    )

    # ========================================================
    # Test
    # ========================================================

    test_candidates = (
        build_ope_candidates(
            "test",
            reward_lookups[
                "test"
            ],
        )
    )

    (
        test_summary,
        test_decisions,
    ) = (
        evaluate_split(
            "test",
            test_candidates,
        )
    )

    # ========================================================
    # Print final
    # ========================================================

    print_final_summary(
        "validation",
        val_summary,
    )

    print_final_summary(
        "test",
        test_summary,
    )

    # ========================================================
    # Save
    # ========================================================

    val_summary.to_csv(
        OUTPUT_DIR
        / "validation_ope_summary.csv",
        index=False,
    )

    test_summary.to_csv(
        OUTPUT_DIR
        / "test_ope_summary.csv",
        index=False,
    )

    val_decisions.to_csv(
        OUTPUT_DIR
        / "validation_ope_decisions.csv",
        index=False,
    )

    test_decisions.to_csv(
        OUTPUT_DIR
        / "test_ope_decisions.csv",
        index=False,
    )

    # combined summary
    combined = (
        pd.concat(
            [
                val_summary,
                test_summary,
            ],
            ignore_index=True,
        )
    )

    combined.to_csv(
        OUTPUT_DIR
        / "ope_summary_all.csv",
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
        / "validation_ope_summary.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_ope_summary.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "ope_summary_all.csv"
    )


if __name__ == "__main__":
    main()
