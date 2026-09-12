from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import bandit_v2_actual_bullpen as base
import bandit_v2_profile_fixed as profile_fix
import bandit_v2_interactions as interactions


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path(
    "bandit_output_v2_behavior_policy"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

L2 = 1e-3
MAX_ITER = 200


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 1. Candidate dataset
# ============================================================

def build_candidate_dataset(
    name,
    decisions,
    bullpen_map,
    used_before_map,
    profile_index,
):

    print_section(
        f"BUILD CANDIDATES - {name}"
    )

    # --------------------------------------------------------
    # 실제 game-day bullpen 명단이 존재하는 preview 구간만 사용
    # --------------------------------------------------------

    decisions = (
        decisions[
            decisions[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    decisions = (
        decisions.sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
    )

    rows = []

    total = 0
    evaluated = 0
    missing_actual = 0
    small_sets = 0

    decision_id = 0

    # ========================================================
    # Decision loop
    # ========================================================

    for _, decision in (
        decisions.iterrows()
    ):

        total += 1

        game_id = str(
            decision[
                "game_id"
            ]
        )

        team = str(
            decision[
                "def_team"
            ]
        )

        actual = str(
            decision[
                "pitcher_id"
            ]
        )

        appearance_order = int(
            decision[
                "appearance_order"
            ]
        )

        # ----------------------------------------------------
        # 실제 당일 bullpen 명단
        # ----------------------------------------------------

        listed = set(
            bullpen_map.get(
                (
                    game_id,
                    team,
                ),
                set(),
            )
        )

        # ----------------------------------------------------
        # 현재 시점 이전에 이미 등판한 투수
        # ----------------------------------------------------

        used = set(
            used_before_map.get(
                (
                    game_id,
                    team,
                    appearance_order,
                ),
                set(),
            )
        )

        # ----------------------------------------------------
        # 실제 가능한 후보군
        # ----------------------------------------------------

        available = (
            listed
            - used
        )

        # 실제 감독 선택 투수가 후보군에 없으면
        # behavior choice likelihood 계산 불가능
        if actual not in available:

            missing_actual += 1
            continue

        if len(available) < 2:

            small_sets += 1
            continue

        # ====================================================
        # 후보별 feature 생성
        # ====================================================

        for pid in sorted(
            available
        ):

            (
                profile,
                profile_source,
            ) = (
                profile_fix
                .get_asof_profile(
                    decision,
                    pid,
                    profile_index,
                )
            )

            # 기존 Bandit과 동일한 feature
            values = (
                base
                .candidate_feature_values(
                    decision,
                    profile,
                )
            )

            # ------------------------------------------------
            # prior_n_apps 원본값
            #
            # 모델에는 prior_n_apps_log가 들어가지만
            # 기존 heuristic baseline은 prior_n_apps 원값 사용
            # ------------------------------------------------

            raw_prior_n_apps = (
                profile.get(
                    "prior_n_apps",
                    0.0,
                )
            )

            if pd.isna(
                raw_prior_n_apps
            ):

                raw_prior_n_apps = (
                    0.0
                )

            # ------------------------------------------------
            # Candidate row
            # ------------------------------------------------

            row = {
                "decision_id":
                    decision_id,

                "game_id":
                    game_id,

                "date":
                    decision[
                        "date"
                    ],

                "def_team":
                    team,

                "appearance_order":
                    appearance_order,

                "pitcher_id":
                    str(pid),

                "actual_pitcher":
                    actual,

                "is_actual":
                    int(
                        str(pid)
                        == actual
                    ),

                "profile_source":
                    profile_source,

                # heuristic baseline용
                "prior_n_apps":
                    float(
                        raw_prior_n_apps
                    ),
            }

            # 모델 feature 추가
            row.update(
                values
            )

            rows.append(
                row
            )

        evaluated += 1
        decision_id += 1

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    print(
        f"Preview decisions      : "
        f"{total:,}"
    )

    print(
        f"Evaluated decisions    : "
        f"{evaluated:,}"
    )

    print(
        f"Missing actual         : "
        f"{missing_actual:,}"
    )

    print(
        f"Candidate sets < 2     : "
        f"{small_sets:,}"
    )

    print(
        f"Candidate rows         : "
        f"{len(df):,}"
    )

    if evaluated > 0:

        print(
            f"Avg candidates         : "
            f"{len(df) / evaluated:.2f}"
        )

    # --------------------------------------------------------
    # 검증:
    # decision당 실제 감독 선택 투수는 정확히 하나
    # --------------------------------------------------------

    actual_counts = (
        df.groupby(
            "decision_id"
        )[
            "is_actual"
        ]
        .sum()
    )

    if not (
        actual_counts
        == 1
    ).all():

        raise ValueError(
            "A decision must contain exactly "
            "one actual pitcher."
        )

    return df


# ============================================================
# 2. Behavior feature index
# ============================================================

def behavior_feature_indices():
    """
    Conditional Logit에서는 한 decision 안에서
    situation-only feature가 모든 후보에게 동일하다.

    예:
        inning = 8
        score_diff = 1
        outs = 0

    는 모든 후보 A/B/C에게 동일하므로
    softmax utility 차이를 만들지 못한다.

    따라서 behavior choice utility에는:

        1) pitcher feature
        2) context x pitcher interaction

    만 사용한다.
    """

    base_features = (
        interactions
        .BASE_FEATURES
    )

    pitcher_features = (
        interactions
        .PITCHER_FEATURES
    )

    indices = []
    names = []

    # --------------------------------------------------------
    # transform_frame output 구조:
    #
    # [0] intercept
    #
    # [1 ... len(BASE_FEATURES)]
    # base standardized features
    #
    # 그 이후
    # context × pitcher interactions
    # --------------------------------------------------------

    for feature in pitcher_features:

        idx = (
            1
            +
            base_features.index(
                feature
            )
        )

        indices.append(
            idx
        )

        names.append(
            feature
        )

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    interaction_names = (
        interactions
        .interaction_names()
    )

    interaction_start = (
        1
        +
        len(
            base_features
        )
    )

    for i, feature in enumerate(
        interaction_names
    ):

        indices.append(
            interaction_start
            + i
        )

        names.append(
            feature
        )

    return (
        np.asarray(
            indices,
            dtype=int,
        ),
        names,
    )


# ============================================================
# 3. Candidate transform
# ============================================================

def transform_candidates(
    df,
    processor,
):

    # 전체 interaction feature 생성
    X_full = (
        processor
        .transform_frame(
            df
        )
    )

    (
        indices,
        feature_names,
    ) = (
        behavior_feature_indices()
    )

    # Behavior model에 필요한 부분만 사용
    X = (
        X_full[
            :,
            indices,
        ]
    )

    return (
        X,
        feature_names,
    )


# ============================================================
# 4. Conditional Softmax
# ============================================================

def group_softmax(
    scores,
    group_ids,
    n_groups,
):
    """
    decision별 candidate softmax.

        mu(a | s)
        =
        exp(score_a)
        /
        sum_j exp(score_j)

    numerical stability를 위해
    각 decision의 max score를 먼저 뺀다.
    """

    group_max = np.full(
        n_groups,
        -np.inf,
        dtype=float,
    )

    np.maximum.at(
        group_max,
        group_ids,
        scores,
    )

    shifted = (
        scores
        -
        group_max[
            group_ids
        ]
    )

    exp_scores = np.exp(
        shifted
    )

    group_sum = np.bincount(
        group_ids,
        weights=exp_scores,
        minlength=n_groups,
    )

    probabilities = (
        exp_scores
        /
        group_sum[
            group_ids
        ]
    )

    log_partition = (
        group_max
        +
        np.log(
            group_sum
        )
    )

    return (
        probabilities,
        log_partition,
    )


# ============================================================
# 5. Conditional Logit training
# ============================================================

def fit_behavior_model(
    X,
    group_ids,
    y,
):

    print_section(
        "FIT MANAGER BEHAVIOR POLICY"
    )

    n_groups = int(
        group_ids.max()
        + 1
    )

    dim = (
        X.shape[1]
    )

    print(
        f"Decisions   : "
        f"{n_groups:,}"
    )

    print(
        f"Candidates  : "
        f"{len(X):,}"
    )

    print(
        f"Feature dim : "
        f"{dim}"
    )

    print(
        f"L2          : "
        f"{L2}"
    )

    # ========================================================
    # Objective
    # ========================================================

    def objective(w):

        scores = (
            X
            @ w
        )

        (
            probabilities,
            log_partition,
        ) = (
            group_softmax(
                scores,
                group_ids,
                n_groups,
            )
        )

        # ----------------------------------------------------
        # Conditional negative log likelihood
        #
        # -sum chosen score
        # +sum log(sum exp(score))
        # ----------------------------------------------------

        chosen_score_sum = (
            scores[
                y == 1
            ]
            .sum()
        )

        nll = (
            -chosen_score_sum
            +
            log_partition.sum()
        )

        # decision 평균 loss
        nll = (
            nll
            /
            n_groups
        )

        # ----------------------------------------------------
        # L2 regularization
        # ----------------------------------------------------

        reg = (
            0.5
            *
            L2
            *
            np.dot(
                w,
                w,
            )
        )

        loss = (
            nll
            +
            reg
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        residual = (
            probabilities
            -
            y
        )

        grad = (
            X.T
            @ residual
        )

        grad = (
            grad
            /
            n_groups
        )

        grad = (
            grad
            +
            L2
            *
            w
        )

        return (
            float(loss),
            grad,
        )

    # ========================================================
    # Optimization
    # ========================================================

    initial = np.zeros(
        dim,
        dtype=float,
    )

    result = minimize(
        fun=objective,
        x0=initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter":
                MAX_ITER,

            "ftol":
                1e-9,
        },
    )

    print()

    print(
        "Optimization success :",
        result.success,
    )

    print(
        "Message              :",
        result.message,
    )

    print(
        "Iterations           :",
        result.nit,
    )

    print(
        "Final loss           :",
        f"{result.fun:.6f}",
    )

    return (
        result.x,
        result,
    )


# ============================================================
# 6. Behavior propensity prediction
# ============================================================

def predict_behavior(
    df,
    X,
    weights,
):

    out = (
        df.copy()
    )

    group_ids = (
        out[
            "decision_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    n_groups = int(
        group_ids.max()
        + 1
    )

    scores = (
        X
        @ weights
    )

    (
        probabilities,
        _
    ) = (
        group_softmax(
            scores,
            group_ids,
            n_groups,
        )
    )

    out[
        "behavior_score"
    ] = (
        scores
    )

    out[
        "behavior_prob"
    ] = (
        probabilities
    )

    return out


# ============================================================
# 7. Behavior model evaluation
# ============================================================

def evaluate_behavior(
    name,
    df,
):

    print_section(
        f"BEHAVIOR POLICY EVALUATION - {name}"
    )

    # ========================================================
    # Actual choice
    # ========================================================

    actual = (
        df[
            df[
                "is_actual"
            ]
            == 1
        ]
        .copy()
    )

    eps = (
        1e-12
    )

    # --------------------------------------------------------
    # Log loss
    # --------------------------------------------------------

    log_loss = float(
        -np.log(
            np.clip(
                actual[
                    "behavior_prob"
                ],
                eps,
                1.0,
            )
        )
        .mean()
    )

    # ========================================================
    # Behavior Top-1
    # ========================================================

    ordered = (
        df.sort_values(
            [
                "decision_id",
                "behavior_prob",
                "pitcher_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

    top1 = (
        ordered.groupby(
            "decision_id",
            as_index=False,
        )
        .first()
    )

    top1_agreement = float(
        top1[
            "is_actual"
        ]
        .mean()
    )

    # ========================================================
    # Actual pitcher rank
    # ========================================================

    ranked = (
        df.copy()
    )

    ranked[
        "behavior_rank"
    ] = (
        ranked.groupby(
            "decision_id"
        )[
            "behavior_prob"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    actual_rank = (
        ranked[
            ranked[
                "is_actual"
            ]
            == 1
        ]
        .copy()
    )

    top3 = float(
        (
            actual_rank[
                "behavior_rank"
            ]
            <= 3
        )
        .mean()
    )

    top5 = float(
        (
            actual_rank[
                "behavior_rank"
            ]
            <= 5
        )
        .mean()
    )

    mean_rank = float(
        actual_rank[
            "behavior_rank"
        ]
        .mean()
    )

    # ========================================================
    # Uniform baseline
    # ========================================================

    candidate_counts = (
        df.groupby(
            "decision_id"
        )
        .size()
    )

    uniform_log_loss = float(
        np.log(
            candidate_counts
        )
        .mean()
    )

    random_agreement = float(
        (
            1.0
            /
            candidate_counts
        )
        .mean()
    )

    # ========================================================
    # prior_n_apps heuristic
    # ========================================================

    heuristic = (
        df.copy()
    )

    heuristic[
        "prior_n_apps"
    ] = (
        pd.to_numeric(
            heuristic[
                "prior_n_apps"
            ],
            errors="coerce",
        )
        .fillna(-1)
    )

    heuristic = (
        heuristic.sort_values(
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

    heuristic_top = (
        heuristic.groupby(
            "decision_id",
            as_index=False,
        )
        .first()
    )

    heuristic_agreement = float(
        heuristic_top[
            "is_actual"
        ]
        .mean()
    )

    # ========================================================
    # Propensity support diagnostics
    # ========================================================

    actual_probs = (
        actual[
            "behavior_prob"
        ]
        .to_numpy(
            dtype=float
        )
    )

    quantile_levels = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    quantiles = np.quantile(
        actual_probs,
        quantile_levels,
    )

    below_001 = float(
        (
            actual_probs
            < 0.01
        )
        .mean()
    )

    below_005 = float(
        (
            actual_probs
            < 0.05
        )
        .mean()
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        f"Decisions              : "
        f"{len(actual):,}"
    )

    print(
        f"Avg candidates         : "
        f"{candidate_counts.mean():.2f}"
    )

    print()

    print(
        f"Random Top-1           : "
        f"{random_agreement * 100:.2f}%"
    )

    print(
        f"prior_n_apps heuristic : "
        f"{heuristic_agreement * 100:.2f}%"
    )

    print(
        f"Behavior model Top-1   : "
        f"{top1_agreement * 100:.2f}%"
    )

    print()

    print(
        f"Behavior Top-3         : "
        f"{top3 * 100:.2f}%"
    )

    print(
        f"Behavior Top-5         : "
        f"{top5 * 100:.2f}%"
    )

    print(
        f"Actual mean rank       : "
        f"{mean_rank:.2f}"
    )

    print()

    print(
        f"Uniform log loss       : "
        f"{uniform_log_loss:.4f}"
    )

    print(
        f"Behavior log loss      : "
        f"{log_loss:.4f}"
    )

    print()

    print(
        "Actual-choice propensity quantiles:"
    )

    for q, value in zip(
        [
            1,
            5,
            10,
            25,
            50,
            75,
            90,
            95,
            99,
        ],
        quantiles,
    ):

        print(
            f"  p{q:02d}: "
            f"{value:.6f}"
        )

    print()

    print(
        f"P(mu_actual < 0.01)    : "
        f"{below_001 * 100:.2f}%"
    )

    print(
        f"P(mu_actual < 0.05)    : "
        f"{below_005 * 100:.2f}%"
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "split":
            name,

        "n_decisions":
            int(
                len(actual)
            ),

        "avg_candidates":
            float(
                candidate_counts.mean()
            ),

        "random_top1":
            random_agreement,

        "heuristic_top1":
            heuristic_agreement,

        "behavior_top1":
            top1_agreement,

        "behavior_top3":
            top3,

        "behavior_top5":
            top5,

        "actual_mean_rank":
            mean_rank,

        "uniform_log_loss":
            uniform_log_loss,

        "behavior_log_loss":
            log_loss,

        "actual_propensity_p01":
            float(
                quantiles[0]
            ),

        "actual_propensity_p05":
            float(
                quantiles[1]
            ),

        "actual_propensity_p10":
            float(
                quantiles[2]
            ),

        "actual_propensity_p25":
            float(
                quantiles[3]
            ),

        "actual_propensity_p50":
            float(
                quantiles[4]
            ),

        "actual_propensity_p75":
            float(
                quantiles[5]
            ),

        "actual_propensity_p90":
            float(
                quantiles[6]
            ),

        "actual_propensity_p95":
            float(
                quantiles[7]
            ),

        "actual_propensity_p99":
            float(
                quantiles[8]
            ),

        "actual_propensity_below_001":
            below_001,

        "actual_propensity_below_005":
            below_005,
    }

    return (
        ranked,
        summary,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- MANAGER BEHAVIOR POLICY"
    )

    print(
        "Purpose:"
    )

    print(
        "Estimate manager choice propensity "
        "mu(a|s) for future OPE."
    )

    print(
        "This is NOT the recommendation policy."
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 1. Data
    # ========================================================

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

    bullpen_map = (
        base.build_bullpen_map(
            bullpen
        )
    )

    used_before_map = (
        base.build_used_before_map(
            first_app
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

    # ========================================================
    # 2. Full-history profile
    # ========================================================

    full_history = (
        profile_fix
        .prepare_full_history(
            history
        )
    )

    profile_index = (
        profile_fix
        .build_profile_index(
            full_history
        )
    )

    # ========================================================
    # 3. Feature processor
    #
    # median/scaler는 train에서만 fit
    # ========================================================

    processor = (
        interactions
        .InteractionFeatureProcessor()
    )

    processor.fit(
        train
    )

    # ========================================================
    # 4. Candidate datasets
    # ========================================================

    train_candidates = (
        build_candidate_dataset(
            "TRAIN",
            train,
            bullpen_map,
            used_before_map,
            profile_index,
        )
    )

    val_candidates = (
        build_candidate_dataset(
            "VALIDATION",
            val,
            bullpen_map,
            used_before_map,
            profile_index,
        )
    )

    test_candidates = (
        build_candidate_dataset(
            "TEST",
            test,
            bullpen_map,
            used_before_map,
            profile_index,
        )
    )

    # ========================================================
    # 5. Feature transform
    # ========================================================

    (
        X_train,
        feature_names,
    ) = (
        transform_candidates(
            train_candidates,
            processor,
        )
    )

    (
        X_val,
        _,
    ) = (
        transform_candidates(
            val_candidates,
            processor,
        )
    )

    (
        X_test,
        _,
    ) = (
        transform_candidates(
            test_candidates,
            processor,
        )
    )

    print_section(
        "BEHAVIOR FEATURE SET"
    )

    print(
        f"Pitcher features       : "
        f"{len(interactions.PITCHER_FEATURES)}"
    )

    print(
        f"Interaction features   : "
        f"{len(interactions.INTERACTION_PAIRS)}"
    )

    print(
        f"Total behavior dim     : "
        f"{len(feature_names)}"
    )

    # ========================================================
    # 6. Train behavior model
    # ========================================================

    train_groups = (
        train_candidates[
            "decision_id"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_train = (
        train_candidates[
            "is_actual"
        ]
        .to_numpy(
            dtype=float
        )
    )

    (
        weights,
        optimization_result,
    ) = (
        fit_behavior_model(
            X_train,
            train_groups,
            y_train,
        )
    )

    # ========================================================
    # 7. Behavior propensity prediction
    # ========================================================

    train_pred = (
        predict_behavior(
            train_candidates,
            X_train,
            weights,
        )
    )

    val_pred = (
        predict_behavior(
            val_candidates,
            X_val,
            weights,
        )
    )

    test_pred = (
        predict_behavior(
            test_candidates,
            X_test,
            weights,
        )
    )

    # ========================================================
    # 8. Evaluation
    # ========================================================

    (
        train_ranked,
        train_summary,
    ) = (
        evaluate_behavior(
            "TRAIN",
            train_pred,
        )
    )

    (
        val_ranked,
        val_summary,
    ) = (
        evaluate_behavior(
            "VALIDATION",
            val_pred,
        )
    )

    (
        test_ranked,
        test_summary,
    ) = (
        evaluate_behavior(
            "TEST",
            test_pred,
        )
    )

    # ========================================================
    # 9. Behavior coefficients
    # ========================================================

    coef = pd.DataFrame(
        {
            "feature":
                feature_names,

            "coefficient":
                weights,
        }
    )

    coef[
        "abs_coefficient"
    ] = (
        coef[
            "coefficient"
        ]
        .abs()
    )

    coef = (
        coef.sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print_section(
        "TOP 20 BEHAVIOR COEFFICIENTS"
    )

    print(
        coef[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 10. Save
    # ========================================================

    train_ranked.to_csv(
        OUTPUT_DIR
        / "train_behavior_candidates.csv",
        index=False,
    )

    val_ranked.to_csv(
        OUTPUT_DIR
        / "validation_behavior_candidates.csv",
        index=False,
    )

    test_ranked.to_csv(
        OUTPUT_DIR
        / "test_behavior_candidates.csv",
        index=False,
    )

    coef.to_csv(
        OUTPUT_DIR
        / "behavior_coefficients.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            train_summary,
            val_summary,
            test_summary,
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "behavior_summary.csv",
        index=False,
    )

    # numpy weight도 저장
    np.save(
        OUTPUT_DIR
        / "behavior_weights.npy",
        weights,
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
        / "behavior_summary.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "behavior_coefficients.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "behavior_weights.npy"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_behavior_candidates.csv"
    )


if __name__ == "__main__":
    main()