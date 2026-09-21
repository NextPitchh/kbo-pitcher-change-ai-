import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.core import bandit_v2_profile_fixed as profile_fix


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path(
    "outputs/bandit/core/v2_interactions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Feature 정의
# ============================================================

SITUATION_FEATURES = list(
    base.SITUATION_FEATURES
)

PITCHER_FEATURES = list(
    base.PITCHER_FEATURES
)

BASE_FEATURES = (
    SITUATION_FEATURES
    + PITCHER_FEATURES
)


# 모든 상황 × 투수 interaction
INTERACTION_PAIRS = [
    (situation, pitcher)
    for situation in SITUATION_FEATURES
    for pitcher in PITCHER_FEATURES
]


# ============================================================
# Interaction Feature Processor
# ============================================================

class InteractionFeatureProcessor:
    """
    1. 기존 base feature median imputation
    2. base feature standardization
    3. standardized situation × pitcher interaction 생성
    4. interaction도 다시 standardization

    최종:
        intercept
        + base z-score
        + interaction z-score
    """

    def __init__(self):

        self.medians = {}

        self.base_scaler = (
            StandardScaler()
        )

        self.interaction_scaler = (
            StandardScaler()
        )

        self.feature_to_idx = {
            feature: i
            for i, feature
            in enumerate(
                BASE_FEATURES
            )
        }


    # ========================================================
    # 내부 helper
    # ========================================================

    def _fill_frame(
        self,
        df,
    ):

        x = (
            df[
                BASE_FEATURES
            ]
            .copy()
        )

        for col in BASE_FEATURES:

            x[col] = (
                x[col]
                .fillna(
                    self.medians[col]
                )
            )

        return x


    def _interaction_matrix(
        self,
        z_base,
    ):

        columns = []

        for (
            situation_feature,
            pitcher_feature,
        ) in INTERACTION_PAIRS:

            s_idx = (
                self.feature_to_idx[
                    situation_feature
                ]
            )

            p_idx = (
                self.feature_to_idx[
                    pitcher_feature
                ]
            )

            interaction = (
                z_base[:, s_idx]
                *
                z_base[:, p_idx]
            )

            columns.append(
                interaction
            )

        return np.column_stack(
            columns
        )


    # ========================================================
    # fit
    # ========================================================

    def fit(
        self,
        train,
    ):

        x = (
            train[
                BASE_FEATURES
            ]
            .copy()
        )

        # median은 train에서만 계산
        for col in BASE_FEATURES:

            med = (
                x[col]
                .median()
            )

            if pd.isna(med):
                med = 0.0

            self.medians[col] = (
                float(med)
            )

            x[col] = (
                x[col]
                .fillna(med)
            )

        # base standardization
        z_base = (
            self.base_scaler
            .fit_transform(x)
        )

        # interaction
        interactions = (
            self._interaction_matrix(
                z_base
            )
        )

        # interaction 자체도 scale 통일
        self.interaction_scaler.fit(
            interactions
        )


    # ========================================================
    # DataFrame transform
    # ========================================================

    def transform_frame(
        self,
        df,
    ):

        x = self._fill_frame(
            df
        )

        z_base = (
            self.base_scaler
            .transform(x)
        )

        interactions = (
            self._interaction_matrix(
                z_base
            )
        )

        z_interactions = (
            self.interaction_scaler
            .transform(
                interactions
            )
        )

        # intercept
        return np.column_stack(
            [
                np.ones(
                    len(df)
                ),
                z_base,
                z_interactions,
            ]
        )


    # ========================================================
    # 단일 후보 transform
    # ========================================================

    def transform_values(
        self,
        values,
    ):

        arr = np.asarray(
            [
                (
                    self.medians[col]
                    if pd.isna(
                        values[col]
                    )
                    else values[col]
                )
                for col
                in BASE_FEATURES
            ],
            dtype=float,
        )

        # base z-score
        z_base = (
            (
                arr
                - self.base_scaler.mean_
            )
            /
            self.base_scaler.scale_
        )

        # interaction
        raw_interactions = []

        for (
            situation_feature,
            pitcher_feature,
        ) in INTERACTION_PAIRS:

            s_idx = (
                self.feature_to_idx[
                    situation_feature
                ]
            )

            p_idx = (
                self.feature_to_idx[
                    pitcher_feature
                ]
            )

            value = (
                z_base[s_idx]
                *
                z_base[p_idx]
            )

            raw_interactions.append(
                value
            )

        raw_interactions = (
            np.asarray(
                raw_interactions,
                dtype=float,
            )
        )

        z_interactions = (
            (
                raw_interactions
                -
                self.interaction_scaler.mean_
            )
            /
            self.interaction_scaler.scale_
        )

        return np.concatenate(
            [
                np.array(
                    [1.0]
                ),
                z_base,
                z_interactions,
            ]
        )


# ============================================================
# Train
# ============================================================

def train_interaction_bandit(
    train,
    processor,
):

    base.print_section(
        "TRAIN LINEAR UCB + INTERACTIONS"
    )

    processor.fit(
        train
    )

    X = (
        processor
        .transform_frame(
            train
        )
    )

    y = (
        train[
            "reward"
        ]
        .to_numpy(
            dtype=float
        )
    )

    model = base.LinearUCB(
        dim=X.shape[1],
        alpha=base.ALPHA,
        ridge=base.RIDGE,
    )

    model.fit(
        X,
        y,
    )

    print(
        f"Training samples     : "
        f"{len(train):,}"
    )

    print(
        f"Base features        : "
        f"{len(BASE_FEATURES)}"
    )

    print(
        f"Interaction features : "
        f"{len(INTERACTION_PAIRS)}"
    )

    print(
        f"Feature dim          : "
        f"{X.shape[1]}"
    )

    print(
        f"Alpha                : "
        f"{base.ALPHA}"
    )

    print(
        f"Reward               : "
        f"{base.REWARD_MODE}"
    )

    return model


# ============================================================
# Interaction 이름
# ============================================================

def interaction_names():

    return [
        (
            f"{situation}"
            "__X__"
            f"{pitcher}"
        )
        for (
            situation,
            pitcher,
        )
        in INTERACTION_PAIRS
    ]


# ============================================================
# 계수 확인용
# ============================================================

def save_coefficients(
    model,
):

    names = (
        ["intercept"]
        + BASE_FEATURES
        + interaction_names()
    )

    coef_df = pd.DataFrame(
        {
            "feature":
                names,

            "coefficient":
                model.theta,
        }
    )

    coef_df[
        "abs_coefficient"
    ] = (
        coef_df[
            "coefficient"
        ]
        .abs()
    )

    coef_df = (
        coef_df.sort_values(
            "abs_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    coef_df.to_csv(
        OUTPUT_DIR
        / "coefficients.csv",
        index=False,
    )

    return coef_df


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- CONTEXT x PITCHER INTERACTIONS"
    )

    print(
        f"REWARD MODE : "
        f"{base.REWARD_MODE}"
    )

    print(
        f"ALPHA       : "
        f"{base.ALPHA}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 1. 데이터
    # ========================================================

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    (
        decisions,
        first_app,
    ) = (
        base.build_decision_dataset(
            pa,
            history,
        )
    )

    # ========================================================
    # 2. 실제 bullpen
    # ========================================================

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

    # ========================================================
    # 3. Split
    # ========================================================

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
    # 4. Profile fix
    # ========================================================

    full_history = (
        profile_fix.prepare_full_history(
            history
        )
    )

    profile_index = (
        profile_fix.build_profile_index(
            full_history
        )
    )

    # ========================================================
    # 5. Interaction model
    # ========================================================

    processor = (
        InteractionFeatureProcessor()
    )

    model = (
        train_interaction_bandit(
            train,
            processor,
        )
    )

    # ========================================================
    # 6. 계수 저장
    # ========================================================

    coef_df = (
        save_coefficients(
            model
        )
    )

    print()
    print(
        "Top 15 coefficients:"
    )

    print(
        coef_df[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 7. Validation
    # ========================================================

    (
        val_result,
        val_candidates,
        val_summary,
    ) = (
        profile_fix
        .evaluate_profile_fixed(
            "VALIDATION",
            model,
            processor,
            eval_df=val,
            bullpen_map=(
                bullpen_map
            ),
            used_before_map=(
                used_before_map
            ),
            profile_index=(
                profile_index
            ),
        )
    )

    # ========================================================
    # 8. Test
    # ========================================================

    (
        test_result,
        test_candidates,
        test_summary,
    ) = (
        profile_fix
        .evaluate_profile_fixed(
            "TEST",
            model,
            processor,
            eval_df=test,
            bullpen_map=(
                bullpen_map
            ),
            used_before_map=(
                used_before_map
            ),
            profile_index=(
                profile_index
            ),
        )
    )

    # ========================================================
    # Save
    # ========================================================

    val_result.to_csv(
        OUTPUT_DIR
        / "validation_decisions.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_decisions.csv",
        index=False,
    )

    val_candidates.to_csv(
        OUTPUT_DIR
        / "validation_candidates.csv",
        index=False,
    )

    test_candidates.to_csv(
        OUTPUT_DIR
        / "test_candidates.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "reward_mode":
                    base.REWARD_MODE,

                "alpha":
                    base.ALPHA,

                "ridge":
                    base.RIDGE,

                "base_feature_count":
                    len(
                        BASE_FEATURES
                    ),

                "interaction_count":
                    len(
                        INTERACTION_PAIRS
                    ),

                "total_feature_dim":
                    (
                        1
                        + len(
                            BASE_FEATURES
                        )
                        + len(
                            INTERACTION_PAIRS
                        )
                    ),

                "interaction_method":
                    (
                        "all_situation_x_"
                        "pitcher_features"
                    ),

                "profile_method":
                    (
                        "global_full_history_"
                        "asof_snapshot"
                    ),

                "validation":
                    val_summary,

                "test":
                    test_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    base.print_section(
        "DONE"
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_decisions.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_decisions.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "coefficients.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "summary.json"
    )


if __name__ == "__main__":
    main()
