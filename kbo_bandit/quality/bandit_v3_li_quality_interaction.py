from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.quality import bandit_v3_li_quality_ablation as b3_mod


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_interaction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RIDGE = 1.0


EXPECTED = {
    "validation": {
        "B0": 0.0982,
        "B1": 0.1116,
        "B2": 0.0966,
        "B3": 0.1113,
    },

    "test": {
        "B0": 0.1059,
        "B1": 0.1203,
        "B2": 0.1036,
        "B3": 0.1167,
    },
}


REPRO_TOLERANCE = 0.002


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


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


# =============================================================================
# B4 FEATURE EXTENSION
# =============================================================================

class LIQualityInteractionFeatureExtension(
    b3_mod.LIQualityFeatureExtension
):
    """
    B4:

    B0
    + leverage_index
    + leverage_index x 7 pitcher features
    + quality_p
    + leverage_index x quality_p

    모든 LI / quality 값은 Train statistics로 standardized 된다.
    """

    def transform(
        self,
        frame: pd.DataFrame,
    ):

        (
            X0,
            X1,
            X2,
            X3,
        ) = super().transform(
            frame
        )

        li = pd.to_numeric(
            frame[
                "leverage_index_model"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        quality = pd.to_numeric(
            frame[
                "quality_p_model"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        z_li = (
            li
            -
            self.li_mean
        ) / self.li_std

        z_quality = (
            quality
            -
            self.quality_mean
        ) / self.quality_std

        # ---------------------------------------------------------------------
        # 새로운 B4 feature
        #
        # 동일 decision에서는 LI가 모든 candidate에게 동일하지만
        # quality_p는 candidate마다 다르므로,
        # 이 interaction은 leverage 수준에 따라 quality의 ranking 영향력을
        # 다르게 만들 수 있다.
        # ---------------------------------------------------------------------

        li_quality_interaction = (
            z_li
            *
            z_quality
        )

        X4 = np.column_stack(
            [
                X3,
                li_quality_interaction,
            ]
        )

        return (
            X0,
            X1,
            X2,
            X3,
            X4,
        )


# =============================================================================
# RIDGE
# =============================================================================

def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    dim = X.shape[1]

    A = (
        X.T
        @ X
        +
        RIDGE
        *
        np.eye(
            dim
        )
    )

    b = X.T @ y

    return np.linalg.solve(
        A,
        b,
    )


# =============================================================================
# FIT B0 ~ B4
# =============================================================================

def fit_models(
    train: pd.DataFrame,
    extension: LIQualityInteractionFeatureExtension,
):

    section(
        "9. FIT B0 / B1 / B2 / B3 / B4 REWARD MODELS"
    )

    (
        X0,
        X1,
        X2,
        X3,
        X4,
    ) = extension.transform(
        train
    )

    y = pd.to_numeric(
        train[
            "reward"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    theta0 = fit_ridge(
        X0,
        y,
    )

    theta1 = fit_ridge(
        X1,
        y,
    )

    theta2 = fit_ridge(
        X2,
        y,
    )

    theta3 = fit_ridge(
        X3,
        y,
    )

    theta4 = fit_ridge(
        X4,
        y,
    )

    print(
        f"Ridge         : {RIDGE}"
    )

    print(
        f"B0 dimensions : {X0.shape[1]}"
    )

    print(
        f"B1 dimensions : {X1.shape[1]}"
    )

    print(
        f"B2 dimensions : {X2.shape[1]}"
    )

    print(
        f"B3 dimensions : {X3.shape[1]}"
    )

    print(
        f"B4 dimensions : {X4.shape[1]}"
    )

    print()
    print(
        "Expected:"
    )

    print(
        "  B0 = 80"
    )

    print(
        "  B1 = 88"
    )

    print(
        "  B2 = 81"
    )

    print(
        "  B3 = 89"
    )

    print(
        "  B4 = 90"
    )

    if (
        X0.shape[1] != 80
        or
        X1.shape[1] != 88
        or
        X2.shape[1] != 81
        or
        X3.shape[1] != 89
        or
        X4.shape[1] != 90
    ):
        raise ValueError(
            "Unexpected feature dimensions."
        )

    print()
    print(
        f"B3 quality coefficient       : "
        f"{theta3[-1]:+.6f}"
    )

    print(
        f"B4 quality coefficient       : "
        f"{theta4[-2]:+.6f}"
    )

    print(
        f"B4 LI x quality coefficient  : "
        f"{theta4[-1]:+.6f}"
    )

    return (
        theta0,
        theta1,
        theta2,
        theta3,
        theta4,
    )


# =============================================================================
# REWARD PREDICTION
# =============================================================================

def evaluate_reward_prediction(
    split_name: str,
    frame: pd.DataFrame,
    extension: LIQualityInteractionFeatureExtension,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
    theta4: np.ndarray,
):

    (
        X0,
        X1,
        X2,
        X3,
        X4,
    ) = extension.transform(
        frame
    )

    y = pd.to_numeric(
        frame[
            "reward"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    predictions = {
        "B0":
            X0 @ theta0,

        "B1_LI":
            X1 @ theta1,

        "B2_QUALITY":
            X2 @ theta2,

        "B3_LI_QUALITY":
            X3 @ theta3,

        "B4_LI_X_QUALITY":
            X4 @ theta4,
    }

    metrics = {
        model:
            b3_mod.prediction_metrics(
                y,
                pred,
            )
        for model, pred
        in predictions.items()
    }

    print()
    print(
        f"{split_name} reward prediction"
    )

    print(
        f"{'Metric':14s}"
        f"{'B0':>11s}"
        f"{'B1(LI)':>11s}"
        f"{'B2(Q)':>11s}"
        f"{'B3(LI+Q)':>12s}"
        f"{'B4(+LIxQ)':>13s}"
    )

    for metric in [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    ]:

        print(
            f"{metric:14s}"
            f"{metrics['B0'][metric]:11.6f}"
            f"{metrics['B1_LI'][metric]:11.6f}"
            f"{metrics['B2_QUALITY'][metric]:11.6f}"
            f"{metrics['B3_LI_QUALITY'][metric]:12.6f}"
            f"{metrics['B4_LI_X_QUALITY'][metric]:13.6f}"
        )

    rows = []

    for (
        model,
        model_metrics,
    ) in metrics.items():

        rows.append(
            {
                "split":
                    split_name.lower(),

                "model":
                    model,

                **model_metrics,
            }
        )

    return rows


# =============================================================================
# CANDIDATE RANKING
# =============================================================================

def evaluate_actual_bullpen(
    split_name: str,
    frame: pd.DataFrame,
    bullpen_map,
    used_before_map,
    profile_index,
    quality_index,
    quality_fallback: float,
    extension: LIQualityInteractionFeatureExtension,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
    theta4: np.ndarray,
):

    section(
        f"10. ACTUAL BULLPEN EVALUATION - {split_name}"
    )

    work = (
        frame[
            frame[
                "bullpen_source"
            ]
            ==
            "preview"
        ]
        .copy()
    )

    work = work.sort_values(
        [
            "date",
            "game_id",
            "_orig_idx",
        ]
    )

    decision_rows = []
    candidate_rows = []

    quality_source_counter = Counter()

    missing_actual = 0
    decision_id = 0

    for _, decision in work.iterrows():

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

        appearance_order = int(
            decision[
                "appearance_order"
            ]
        )

        actual_pitcher = normalize_id(
            decision[
                "pitcher_id"
            ]
        )

        listed = set(
            bullpen_map.get(
                (
                    game_id,
                    team,
                ),
                set(),
            )
        )

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

        candidates = sorted(
            listed
            -
            used
        )

        candidate_ids = [
            normalize_id(
                pitcher
            )
            for pitcher in candidates
        ]

        if (
            actual_pitcher
            not in candidate_ids
        ):
            missing_actual += 1
            continue

        if len(
            candidate_ids
        ) < 2:
            continue

        (
            candidate_frame,
            metadata,
        ) = b3_mod.build_candidate_frame(
            decision,
            candidates,
            profile_index,
            quality_index,
            quality_fallback,
        )

        for source in metadata[
            "quality_source"
        ]:
            quality_source_counter[
                source
            ] += 1

        (
            X0,
            X1,
            X2,
            X3,
            X4,
        ) = extension.transform(
            candidate_frame
        )

        scores = {
            "b0":
                X0 @ theta0,

            "b1":
                X1 @ theta1,

            "b2":
                X2 @ theta2,

            "b3":
                X3 @ theta3,

            "b4":
                X4 @ theta4,
        }

        recommendations = {}
        rankings = {}

        for (
            model_name,
            model_scores,
        ) in scores.items():

            (
                recommendation,
                ranking,
            ) = b3_mod.rank_candidates(
                candidate_ids,
                model_scores,
            )

            recommendations[
                model_name
            ] = recommendation

            rankings[
                model_name
            ] = ranking

        decision_rows.append(
            {
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

                "actual_pitcher":
                    actual_pitcher,

                "leverage_index":
                    float(
                        decision[
                            "leverage_index_model"
                        ]
                    ),

                "n_candidates":
                    len(
                        candidate_ids
                    ),

                "b0_rec":
                    recommendations[
                        "b0"
                    ],

                "b1_rec":
                    recommendations[
                        "b1"
                    ],

                "b2_rec":
                    recommendations[
                        "b2"
                    ],

                "b3_rec":
                    recommendations[
                        "b3"
                    ],

                "b4_rec":
                    recommendations[
                        "b4"
                    ],

                "b0_agreement":
                    int(
                        recommendations[
                            "b0"
                        ]
                        ==
                        actual_pitcher
                    ),

                "b1_agreement":
                    int(
                        recommendations[
                            "b1"
                        ]
                        ==
                        actual_pitcher
                    ),

                "b2_agreement":
                    int(
                        recommendations[
                            "b2"
                        ]
                        ==
                        actual_pitcher
                    ),

                "b3_agreement":
                    int(
                        recommendations[
                            "b3"
                        ]
                        ==
                        actual_pitcher
                    ),

                "b4_agreement":
                    int(
                        recommendations[
                            "b4"
                        ]
                        ==
                        actual_pitcher
                    ),

                "b0_rank":
                    int(
                        rankings[
                            "b0"
                        ][
                            actual_pitcher
                        ]
                    ),

                "b1_rank":
                    int(
                        rankings[
                            "b1"
                        ][
                            actual_pitcher
                        ]
                    ),

                "b2_rank":
                    int(
                        rankings[
                            "b2"
                        ][
                            actual_pitcher
                        ]
                    ),

                "b3_rank":
                    int(
                        rankings[
                            "b3"
                        ][
                            actual_pitcher
                        ]
                    ),

                "b4_rank":
                    int(
                        rankings[
                            "b4"
                        ][
                            actual_pitcher
                        ]
                    ),
            }
        )

        for index, pitcher_id in enumerate(
            candidate_ids
        ):

            candidate_rows.append(
                {
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
                        pitcher_id,

                    "actual_pitcher":
                        actual_pitcher,

                    "is_actual":
                        int(
                            pitcher_id
                            ==
                            actual_pitcher
                        ),

                    "n_candidates":
                        len(
                            candidate_ids
                        ),

                    "leverage_index":
                        float(
                            metadata.iloc[
                                index
                            ][
                                "leverage_index"
                            ]
                        ),

                    "quality_p":
                        float(
                            metadata.iloc[
                                index
                            ][
                                "quality_p"
                            ]
                        ),

                    "quality_source":
                        metadata.iloc[
                            index
                        ][
                            "quality_source"
                        ],

                    "profile_source":
                        metadata.iloc[
                            index
                        ][
                            "profile_source"
                        ],

                    "q_b0":
                        float(
                            scores[
                                "b0"
                            ][
                                index
                            ]
                        ),

                    "q_b1_li":
                        float(
                            scores[
                                "b1"
                            ][
                                index
                            ]
                        ),

                    "q_b2_quality":
                        float(
                            scores[
                                "b2"
                            ][
                                index
                            ]
                        ),

                    "q_b3_li_quality":
                        float(
                            scores[
                                "b3"
                            ][
                                index
                            ]
                        ),

                    "q_b4_li_x_quality":
                        float(
                            scores[
                                "b4"
                            ][
                                index
                            ]
                        ),
                }
            )

        decision_id += 1

    decision_result = pd.DataFrame(
        decision_rows
    )

    candidate_result = pd.DataFrame(
        candidate_rows
    )

    print(
        f"Preview decisions : "
        f"{len(work):,}"
    )

    print(
        f"Evaluated         : "
        f"{len(decision_result):,}"
    )

    print(
        f"Missing actual    : "
        f"{missing_actual:,}"
    )

    print(
        f"Avg candidates    : "
        f"{decision_result['n_candidates'].mean():.2f}"
    )

    print()
    print(
        "Candidate quality source:"
    )

    total = sum(
        quality_source_counter.values()
    )

    for (
        source,
        count,
    ) in quality_source_counter.most_common():

        print(
            f"  {source:34s}"
            f"{count:8,d} "
            f"({count / total * 100:6.2f}%)"
        )

    return (
        decision_result,
        candidate_result,
        quality_source_counter,
    )


# =============================================================================
# RANKING SUMMARY
# =============================================================================

def print_ranking_summary(
    split_name: str,
    frame: pd.DataFrame,
):

    model_info = [
        (
            "B0",
            "b0",
        ),

        (
            "B1(+LI)",
            "b1",
        ),

        (
            "B2(+Q)",
            "b2",
        ),

        (
            "B3(+LI+Q)",
            "b3",
        ),

        (
            "B4(+LIxQ)",
            "b4",
        ),
    ]

    results = {
        label:
            b3_mod.ranking_metrics(
                frame,
                prefix,
            )
        for (
            label,
            prefix,
        )
        in model_info
    }

    print()
    print(
        split_name
    )

    print(
        f"{'Metric':16s}"
        f"{'B0':>11s}"
        f"{'B1(+LI)':>11s}"
        f"{'B2(+Q)':>11s}"
        f"{'B3(+LI+Q)':>13s}"
        f"{'B4(+LIxQ)':>13s}"
    )

    for metric in [
        "agreement",
        "top3",
        "top5",
    ]:

        print(
            f"{metric:16s}"
            f"{results['B0'][metric] * 100:10.2f}%"
            f"{results['B1(+LI)'][metric] * 100:10.2f}%"
            f"{results['B2(+Q)'][metric] * 100:10.2f}%"
            f"{results['B3(+LI+Q)'][metric] * 100:12.2f}%"
            f"{results['B4(+LIxQ)'][metric] * 100:12.2f}%"
        )

    for metric in [
        "mean_rank",
        "median_rank",
    ]:

        print(
            f"{metric:16s}"
            f"{results['B0'][metric]:11.2f}"
            f"{results['B1(+LI)'][metric]:11.2f}"
            f"{results['B2(+Q)'][metric]:11.2f}"
            f"{results['B3(+LI+Q)'][metric]:13.2f}"
            f"{results['B4(+LIxQ)'][metric]:13.2f}"
        )

    return results


# =============================================================================
# REPRODUCTION CHECK
# =============================================================================

def check_reproduction(
    split: str,
    results: dict,
) -> bool:

    section(
        f"11. REPRODUCTION CHECK - {split.upper()}"
    )

    mapping = {
        "B0":
            "B0",

        "B1":
            "B1(+LI)",

        "B2":
            "B2(+Q)",

        "B3":
            "B3(+LI+Q)",
    }

    passed = True

    for (
        expected_name,
        result_name,
    ) in mapping.items():

        expected = (
            EXPECTED[
                split
            ][
                expected_name
            ]
        )

        current = (
            results[
                result_name
            ][
                "agreement"
            ]
        )

        difference = abs(
            current
            -
            expected
        )

        ok = (
            difference
            <=
            REPRO_TOLERANCE
        )

        print(
            f"{expected_name:3s} | "
            f"expected="
            f"{expected * 100:6.2f}% | "
            f"current="
            f"{current * 100:6.2f}% | "
            f"diff="
            f"{difference * 100:.3f}p | "
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
        "=" * 104
    )

    print(
        "STEP 5 - LI x QUALITY_P INTERACTION ABLATION"
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
        "  B0 + LI + LI x pitcher"
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
        "Everything else remains frozen:"
    )

    print(
        "  data / reward / candidate set / profile / split / ridge / alpha"
    )

    print(
        "=" * 104
    )

    # =========================================================================
    # Load exactly the same data pipeline as Step 4
    # =========================================================================

    li_pa = (
        b3_mod.load_li_data()
    )

    quality_pa = (
        b3_mod.load_quality_data()
    )

    (
        pa,
        history,
        bullpen,
        decisions,
        first_app,
    ) = b3_mod.load_b0_data()

    # =========================================================================
    # Alignment
    # =========================================================================

    section(
        "4. VERIFY DATA ALIGNMENT"
    )

    b3_mod.verify_alignment(
        pa,
        li_pa,
        "LI",
    )

    b3_mod.verify_alignment(
        pa,
        quality_pa,
        "QUALITY",
    )

    # =========================================================================
    # Logged decisions
    # =========================================================================

    section(
        "5. ATTACH LI / QUALITY TO LOGGED DECISIONS"
    )

    decisions = b3_mod.attach_li(
        decisions,
        li_pa,
    )

    decisions = b3_mod.attach_quality(
        decisions,
        quality_pa,
    )

    (
        train,
        validation,
        test,
        quality_fallback,
    ) = b3_mod.split_data(
        decisions
    )

    # =========================================================================
    # Candidate as-of info
    # =========================================================================

    quality_index = (
        b3_mod.build_quality_index(
            quality_pa,
            pa,
        )
    )

    profile_index = (
        b3_mod.build_profile_index(
            history
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

    # =========================================================================
    # B0 processor
    # =========================================================================

    b0_processor = (
        b3_mod.fit_b0_processor(
            train
        )
    )

    # =========================================================================
    # B4 extension
    # =========================================================================

    section(
        "9. FIT LI + QUALITY + LIxQUALITY FEATURE EXTENSION"
    )

    extension = (
        LIQualityInteractionFeatureExtension(
            b0_processor
        )
    )

    extension.fit(
        train
    )

    (
        theta0,
        theta1,
        theta2,
        theta3,
        theta4,
    ) = fit_models(
        train,
        extension,
    )

    # =========================================================================
    # Reward prediction
    # =========================================================================

    section(
        "REWARD PREDICTION DIAGNOSTICS"
    )

    reward_rows = []

    for (
        split_name,
        split_frame,
    ) in [
        (
            "TRAIN",
            train,
        ),

        (
            "VALIDATION",
            validation,
        ),

        (
            "TEST",
            test,
        ),
    ]:

        reward_rows.extend(
            evaluate_reward_prediction(
                split_name,
                split_frame,
                extension,
                theta0,
                theta1,
                theta2,
                theta3,
                theta4,
            )
        )

    reward_summary = pd.DataFrame(
        reward_rows
    )

    # =========================================================================
    # Candidate ranking
    # =========================================================================

    (
        validation_decisions,
        validation_candidates,
        validation_sources,
    ) = evaluate_actual_bullpen(
        "VALIDATION",
        validation,
        bullpen_map,
        used_before_map,
        profile_index,
        quality_index,
        quality_fallback,
        extension,
        theta0,
        theta1,
        theta2,
        theta3,
        theta4,
    )

    (
        test_decisions,
        test_candidates,
        test_sources,
    ) = evaluate_actual_bullpen(
        "TEST",
        test,
        bullpen_map,
        used_before_map,
        profile_index,
        quality_index,
        quality_fallback,
        extension,
        theta0,
        theta1,
        theta2,
        theta3,
        theta4,
    )

    # =========================================================================
    # Ranking summary
    # =========================================================================

    section(
        "B0 / B1 / B2 / B3 / B4 RANKING SUMMARY"
    )

    validation_results = (
        print_ranking_summary(
            "VALIDATION",
            validation_decisions,
        )
    )

    test_results = (
        print_ranking_summary(
            "TEST",
            test_decisions,
        )
    )

    # =========================================================================
    # Reproduction
    # =========================================================================

    validation_repro = (
        check_reproduction(
            "validation",
            validation_results,
        )
    )

    test_repro = (
        check_reproduction(
            "test",
            test_results,
        )
    )

    all_reproduced = (
        validation_repro
        and
        test_repro
    )

    print()
    print(
        "Overall reproduction : "
        +
        (
            "PASS"
            if all_reproduced
            else "WARNING"
        )
    )

    # =========================================================================
    # Ranking CSV
    # =========================================================================

    ranking_rows = []

    for (
        split,
        results,
    ) in [
        (
            "validation",
            validation_results,
        ),

        (
            "test",
            test_results,
        ),
    ]:

        for (
            model,
            metrics,
        ) in results.items():

            ranking_rows.append(
                {
                    "split":
                        split,

                    "model":
                        model,

                    **metrics,
                }
            )

    ranking_summary = pd.DataFrame(
        ranking_rows
    )

    # =========================================================================
    # Quality source summary
    # =========================================================================

    source_rows = []

    for (
        split,
        counter,
    ) in [
        (
            "validation",
            validation_sources,
        ),

        (
            "test",
            test_sources,
        ),
    ]:

        total = sum(
            counter.values()
        )

        for (
            source,
            count,
        ) in counter.items():

            source_rows.append(
                {
                    "split":
                        split,

                    "quality_source":
                        source,

                    "count":
                        count,

                    "ratio":
                        (
                            count
                            /
                            total
                            if total
                            else np.nan
                        ),
                }
            )

    quality_source_summary = (
        pd.DataFrame(
            source_rows
        )
    )

    # =========================================================================
    # Feature stats
    # =========================================================================

    feature_stats = {
        "li_mean":
            extension.li_mean,

        "li_std":
            extension.li_std,

        "quality_mean":
            extension.quality_mean,

        "quality_std":
            extension.quality_std,

        "ridge":
            RIDGE,

        "b4_interaction":
            "standardized_li_x_standardized_quality",

        "dimensions": {
            "B0":
                int(
                    len(theta0)
                ),

            "B1":
                int(
                    len(theta1)
                ),

            "B2":
                int(
                    len(theta2)
                ),

            "B3":
                int(
                    len(theta3)
                ),

            "B4":
                int(
                    len(theta4)
                ),
        },

        "b3_quality_coefficient":
            float(
                theta3[-1]
            ),

        "b4_quality_coefficient":
            float(
                theta4[-2]
            ),

        "b4_li_quality_interaction_coefficient":
            float(
                theta4[-1]
            ),
    }

    # =========================================================================
    # Save
    # =========================================================================

    section(
        "12. SAVE"
    )

    ranking_summary.to_csv(
        OUTPUT_DIR
        /
        "ranking_summary.csv",
        index=False,
    )

    reward_summary.to_csv(
        OUTPUT_DIR
        /
        "reward_prediction_summary.csv",
        index=False,
    )

    validation_decisions.to_csv(
        OUTPUT_DIR
        /
        "validation_decisions.csv",
        index=False,
    )

    test_decisions.to_csv(
        OUTPUT_DIR
        /
        "test_decisions.csv",
        index=False,
    )

    validation_candidates.to_csv(
        OUTPUT_DIR
        /
        "validation_candidates.csv",
        index=False,
    )

    test_candidates.to_csv(
        OUTPUT_DIR
        /
        "test_candidates.csv",
        index=False,
    )

    quality_source_summary.to_csv(
        OUTPUT_DIR
        /
        "quality_source_summary.csv",
        index=False,
    )

    with (
        OUTPUT_DIR
        /
        "feature_extension_stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(
        OUTPUT_DIR
        /
        "theta_b0.npy",
        theta0,
    )

    np.save(
        OUTPUT_DIR
        /
        "theta_b1_li.npy",
        theta1,
    )

    np.save(
        OUTPUT_DIR
        /
        "theta_b2_quality.npy",
        theta2,
    )

    np.save(
        OUTPUT_DIR
        /
        "theta_b3_li_quality.npy",
        theta3,
    )

    np.save(
        OUTPUT_DIR
        /
        "theta_b4_li_x_quality.npy",
        theta4,
    )

    print(
        f"Saved to:\n  {OUTPUT_DIR}"
    )

    # =========================================================================
    # Final
    # =========================================================================

    section(
        "FINAL"
    )

    print(
        "B0/B1/B2/B3 reproduction : "
        +
        (
            "PASS"
            if all_reproduced
            else "WARNING"
        )
    )

    print()
    print(
        "TEST"
    )

    for model in [
        "B0",
        "B1(+LI)",
        "B2(+Q)",
        "B3(+LI+Q)",
        "B4(+LIxQ)",
    ]:

        metrics = (
            test_results[
                model
            ]
        )

        print(
            f"{model:12s} | "
            f"Agreement "
            f"{metrics['agreement'] * 100:6.2f}% | "
            f"Top3 "
            f"{metrics['top3'] * 100:6.2f}% | "
            f"Top5 "
            f"{metrics['top5'] * 100:6.2f}% | "
            f"Mean Rank "
            f"{metrics['mean_rank']:.2f}"
        )

    print()
    print(
        "Primary question:"
    )

    print(
        "Does adding LI x quality_p improve B3?"
    )

    print()
    print(
        "Do not select B4 from Agreement alone."
    )

    print(
        "If reproduction passes, evaluate B4 with paired OPE next."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()