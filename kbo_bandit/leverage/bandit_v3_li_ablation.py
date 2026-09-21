from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.core import bandit_v2_profile_fixed as profile_fix
from kbo_bandit.core import bandit_v2_interactions as interactions


# ============================================================
# Config
# ============================================================

LI_PA_PATH = Path(
    "data_analysis/results/"
    "all_pa_with_li_train_frozen.parquet"
)

OUTPUT_DIR = Path(
    "outputs/bandit/leverage/v3_li_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# 기존 Bandit과 동일한 ridge 사용
RIDGE = getattr(
    base,
    "RIDGE",
    getattr(
        interactions,
        "RIDGE",
        1.0,
    ),
)


EXPECTED_B0 = {
    "validation_agreement": 0.0982,
    "test_agreement": 0.1059,
}


# B0 재현 허용 오차
# 0.2 percentage point
BASELINE_TOLERANCE = 0.002


# ============================================================
# Helper
# ============================================================

def section(title):

    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


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


# ============================================================
# 1. Load data + attach LI
# ============================================================

def load_data():

    section(
        "1. LOAD BANDIT + TRAIN-FROZEN LI DATA"
    )

    base.REWARD_MODE = "wpa_per_pa"
    base.ALPHA = 0.0

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    print(
        "Original PA :",
        pa.shape,
    )

    if not LI_PA_PATH.exists():

        raise FileNotFoundError(
            f"Cannot find: {LI_PA_PATH}"
        )

    li_pa = pd.read_parquet(
        LI_PA_PATH
    )

    print(
        "LI PA       :",
        li_pa.shape,
    )

    li_lookup = (
        li_pa[
            [
                "_orig_idx",
                "leverage_index",
            ]
        ]
        .copy()
    )

    li_lookup[
        "_orig_idx"
    ] = (
        pd.to_numeric(
            li_lookup[
                "_orig_idx"
            ],
            errors="raise",
        )
        .astype(int)
    )

    if (
        li_lookup[
            "_orig_idx"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate _orig_idx in LI data."
        )

    # ========================================================
    # 기존 Bandit decision 생성
    # ========================================================

    (
        decisions,
        first_app,
    ) = base.build_decision_dataset(
        pa,
        history,
    )

    decisions[
        "_orig_idx"
    ] = (
        pd.to_numeric(
            decisions[
                "_orig_idx"
            ],
            errors="raise",
        )
        .astype(int)
    )

    # ========================================================
    # LI는 immutable _orig_idx로만 붙인다.
    # ========================================================

    decisions = decisions.merge(
        li_lookup,
        on="_orig_idx",
        how="left",
        validate="many_to_one",
    )

    missing_li = int(
        decisions[
            "leverage_index"
        ]
        .isna()
        .sum()
    )

    print(
        f"Relief decisions : "
        f"{len(decisions):,}"
    )

    print(
        f"Missing LI       : "
        f"{missing_li:,}"
    )

    if missing_li > 0:

        raise ValueError(
            "Missing decision leverage_index."
        )

    # ========================================================
    # Candidate 관련 map
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
    # Chronological split
    # ========================================================

    (
        train,
        val,
        test,
    ) = base.chronological_split(
        decisions
    )

    return (
        pa,
        history,
        bullpen,
        decisions,
        train,
        val,
        test,
        bullpen_map,
        used_before_map,
    )


# ============================================================
# 2. Build leakage-safe pitcher profile
# ============================================================

def build_profile_index(
    history,
):

    section(
        "2. BUILD FULL PITCHER PROFILE INDEX"
    )

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

    return profile_index


# ============================================================
# 3. Fit original B0 feature processor
# ============================================================

def fit_b0_processor(
    train,
):

    section(
        "3. FIT B0 FEATURE PROCESSOR"
    )

    processor = (
        interactions
        .InteractionFeatureProcessor()
    )

    processor.fit(
        train
    )

    X0_train = (
        processor
        .transform_frame(
            train
        )
    )

    print(
        f"B0 feature dim : "
        f"{X0_train.shape[1]}"
    )

    return (
        processor,
        X0_train,
    )


# ============================================================
# 4. LI extension
# ============================================================

class LIFeatureExtension:

    def __init__(
        self,
        b0_processor,
    ):

        self.b0_processor = (
            b0_processor
        )

        self.li_mean = None
        self.li_std = None

        self.pitcher_indices = []

        for feature in (
            interactions
            .PITCHER_FEATURES
        ):

            base_index = (
                interactions
                .BASE_FEATURES
                .index(
                    feature
                )
            )

            # transform_frame 구조:
            # 0 = intercept
            # 1... = standardized base feature
            self.pitcher_indices.append(
                1
                +
                base_index
            )

    def fit(
        self,
        train,
    ):

        li = (
            pd.to_numeric(
                train[
                    "leverage_index"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=float
            )
        )

        self.li_mean = float(
            li.mean()
        )

        self.li_std = float(
            li.std(
                ddof=0
            )
        )

        if (
            not np.isfinite(
                self.li_std
            )
            or
            self.li_std
            <= 0
        ):

            raise ValueError(
                "Invalid LI std."
            )

        print(
            f"Train decision LI mean : "
            f"{self.li_mean:.6f}"
        )

        print(
            f"Train decision LI std  : "
            f"{self.li_std:.6f}"
        )

    def transform(
        self,
        frame,
    ):

        # ----------------------------------------------------
        # B0 features
        # ----------------------------------------------------

        X0 = (
            self.b0_processor
            .transform_frame(
                frame
            )
        )

        # ----------------------------------------------------
        # Standardized LI
        # ----------------------------------------------------

        li = (
            pd.to_numeric(
                frame[
                    "leverage_index"
                ],
                errors="raise",
            )
            .to_numpy(
                dtype=float
            )
        )

        z_li = (
            (
                li
                -
                self.li_mean
            )
            /
            self.li_std
        )

        # ----------------------------------------------------
        # Candidate-specific pitcher standardized features
        # ----------------------------------------------------

        pitcher_z = (
            X0[
                :,
                self.pitcher_indices,
            ]
        )

        # ----------------------------------------------------
        # LI × pitcher interactions
        # ----------------------------------------------------

        li_interactions = (
            z_li[
                :,
                None
            ]
            *
            pitcher_z
        )

        # ----------------------------------------------------
        # B1
        #
        # B0 features
        # + LI
        # + LI × 7 pitcher features
        # ----------------------------------------------------

        X1 = np.column_stack(
            [
                X0,
                z_li,
                li_interactions,
            ]
        )

        return (
            X0,
            X1,
        )

    def new_feature_names(
        self,
    ):

        names = [
            "leverage_index",
        ]

        for feature in (
            interactions
            .PITCHER_FEATURES
        ):

            names.append(
                "leverage_index"
                "__X__"
                f"{feature}"
            )

        return names


# ============================================================
# 5. Ridge reward model
# ============================================================

def fit_ridge(
    X,
    y,
    ridge,
):

    dim = (
        X.shape[1]
    )

    A = (
        X.T
        @ X
    )

    A = (
        A
        +
        ridge
        *
        np.eye(
            dim
        )
    )

    b = (
        X.T
        @ y
    )

    theta = np.linalg.solve(
        A,
        b,
    )

    return theta


# ============================================================
# 6. Fit B0 / B1
# ============================================================

def fit_models(
    train,
    b0_processor,
    li_extension,
):

    section(
        "4. FIT B0 AND B1 REWARD MODELS"
    )

    (
        X0_train,
        X1_train,
    ) = (
        li_extension
        .transform(
            train
        )
    )

    y_train = (
        pd.to_numeric(
            train[
                "reward"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=float
        )
    )

    theta0 = fit_ridge(
        X0_train,
        y_train,
        RIDGE,
    )

    theta1 = fit_ridge(
        X1_train,
        y_train,
        RIDGE,
    )

    print(
        f"Ridge          : "
        f"{RIDGE}"
    )

    print(
        f"B0 dimensions  : "
        f"{X0_train.shape[1]}"
    )

    print(
        f"B1 dimensions  : "
        f"{X1_train.shape[1]}"
    )

    print(
        f"Added features : "
        f"{X1_train.shape[1] - X0_train.shape[1]}"
    )

    print()

    print(
        "B1 added features:"
    )

    for name in (
        li_extension
        .new_feature_names()
    ):

        print(
            " ",
            name,
        )

    return (
        theta0,
        theta1,
    )


# ============================================================
# 7. Logged-action reward prediction diagnostics
# ============================================================

def prediction_metrics(
    y,
    pred,
):

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    mae = float(
        np.mean(
            np.abs(
                y
                -
                pred
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    y
                    -
                    pred
                )
                ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                y
                -
                y.mean()
            )
            ** 2
        )
    )

    if denominator > 0:

        r2 = float(
            1.0
            -
            (
                np.sum(
                    (
                        y
                        -
                        pred
                    )
                    ** 2
                )
                /
                denominator
            )
        )

    else:

        r2 = np.nan

    pearson = float(
        pd.Series(
            y
        )
        .corr(
            pd.Series(
                pred
            ),
            method="pearson",
        )
    )

    spearman = float(
        pd.Series(
            y
        )
        .corr(
            pd.Series(
                pred
            ),
            method="spearman",
        )
    )

    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "pearson":
            pearson,

        "spearman":
            spearman,
    }


def reward_prediction_check(
    name,
    df,
    li_extension,
    theta0,
    theta1,
):

    (
        X0,
        X1,
    ) = (
        li_extension
        .transform(
            df
        )
    )

    y = (
        pd.to_numeric(
            df[
                "reward"
            ],
            errors="raise",
        )
        .to_numpy(
            dtype=float
        )
    )

    pred0 = (
        X0
        @ theta0
    )

    pred1 = (
        X1
        @ theta1
    )

    m0 = prediction_metrics(
        y,
        pred0,
    )

    m1 = prediction_metrics(
        y,
        pred1,
    )

    print()
    print(
        f"{name} reward prediction"
    )

    print(
        f"{'Metric':12s}"
        f"{'B0':>12s}"
        f"{'B1(+LI)':>12s}"
        f"{'Delta':>12s}"
    )

    for key in [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    ]:

        delta = (
            m1[
                key
            ]
            -
            m0[
                key
            ]
        )

        print(
            f"{key:12s}"
            f"{m0[key]:12.6f}"
            f"{m1[key]:12.6f}"
            f"{delta:+12.6f}"
        )

    rows = []

    for model_name, metrics in [
        (
            "B0",
            m0,
        ),
        (
            "B1_LI",
            m1,
        ),
    ]:

        rows.append(
            {
                "split":
                    name,

                "model":
                    model_name,

                **metrics,
            }
        )

    return rows


# ============================================================
# 8. Candidate feature frame
# ============================================================

def make_candidate_frame(
    decision,
    candidates,
    profile_index,
):

    rows = []

    meta = []

    for pid in candidates:

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

        values = (
            base
            .candidate_feature_values(
                decision,
                profile,
            )
        )

        row = dict(
            values
        )

        # 같은 decision의 모든 후보가 동일한 LI
        row[
            "leverage_index"
        ] = float(
            decision[
                "leverage_index"
            ]
        )

        rows.append(
            row
        )

        meta.append(
            {
                "pitcher_id":
                    str(
                        pid
                    ),

                "profile_source":
                    profile_source,
            }
        )

    frame = pd.DataFrame(
        rows
    )

    meta = pd.DataFrame(
        meta
    )

    return (
        frame,
        meta,
    )


# ============================================================
# 9. Candidate rank helper
# ============================================================

def rank_pitchers(
    pitcher_ids,
    scores,
):

    order = sorted(
        range(
            len(
                pitcher_ids
            )
        ),
        key=lambda i: (
            -float(
                scores[
                    i
                ]
            ),
            str(
                pitcher_ids[
                    i
                ]
            ),
        ),
    )

    ranks = {}

    for rank, idx in enumerate(
        order,
        start=1,
    ):

        ranks[
            str(
                pitcher_ids[
                    idx
                ]
            )
        ] = rank

    top_pitcher = str(
        pitcher_ids[
            order[
                0
            ]
        ]
    )

    return (
        top_pitcher,
        ranks,
    )


# ============================================================
# 10. Evaluate candidate ranking
# ============================================================

def evaluate_candidate_ranking(
    split_name,
    df,
    bullpen_map,
    used_before_map,
    profile_index,
    li_extension,
    theta0,
    theta1,
):

    section(
        f"5. ACTUAL BULLPEN EVALUATION - {split_name}"
    )

    # 실제 game-day bullpen이 있는 구간만 사용
    eval_df = (
        df[
            df[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    eval_df = (
        eval_df.sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
    )

    total = (
        len(
            eval_df
        )
    )

    rows = []
    candidate_rows = []

    missing_actual = 0

    decision_counter = 0

    for _, decision in (
        eval_df.iterrows()
    ):

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

        actual = normalize_id(
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
                x
            )
            for x in candidates
        ]

        if (
            actual
            not in candidate_ids
        ):

            missing_actual += 1
            continue

        if (
            len(
                candidates
            )
            < 2
        ):

            continue

        (
            candidate_frame,
            meta,
        ) = (
            make_candidate_frame(
                decision,
                candidates,
                profile_index,
            )
        )

        (
            X0,
            X1,
        ) = (
            li_extension
            .transform(
                candidate_frame
            )
        )

        q0 = (
            X0
            @ theta0
        )

        q1 = (
            X1
            @ theta1
        )

        (
            rec0,
            ranks0,
        ) = (
            rank_pitchers(
                candidate_ids,
                q0,
            )
        )

        (
            rec1,
            ranks1,
        ) = (
            rank_pitchers(
                candidate_ids,
                q1,
            )
        )

        rank0 = int(
            ranks0[
                actual
            ]
        )

        rank1 = int(
            ranks1[
                actual
            ]
        )

        rows.append(
            {
                "decision_id":
                    decision_counter,

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
                    actual,

                "n_candidates":
                    len(
                        candidates
                    ),

                "leverage_index":
                    float(
                        decision[
                            "leverage_index"
                        ]
                    ),

                "b0_rec":
                    rec0,

                "b1_rec":
                    rec1,

                "b0_agreement":
                    int(
                        rec0
                        ==
                        actual
                    ),

                "b1_agreement":
                    int(
                        rec1
                        ==
                        actual
                    ),

                "b0_rank":
                    rank0,

                "b1_rank":
                    rank1,
            }
        )

        # ----------------------------------------------------
        # Candidate-level 결과 저장
        # 다음 LI segment / OPE에서 재사용
        # ----------------------------------------------------

        for i, pid in enumerate(
            candidate_ids
        ):

            candidate_rows.append(
                {
                    "decision_id":
                        decision_counter,

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
                        pid,

                    "actual_pitcher":
                        actual,

                    "is_actual":
                        int(
                            pid
                            ==
                            actual
                        ),

                    "n_candidates":
                        len(
                            candidates
                        ),

                    "leverage_index":
                        float(
                            decision[
                                "leverage_index"
                            ]
                        ),

                    "profile_source":
                        meta.iloc[
                            i
                        ][
                            "profile_source"
                        ],

                    "q_b0":
                        float(
                            q0[
                                i
                            ]
                        ),

                    "q_b1_li":
                        float(
                            q1[
                                i
                            ]
                        ),
                }
            )

        decision_counter += 1

    result = pd.DataFrame(
        rows
    )

    candidates_out = pd.DataFrame(
        candidate_rows
    )

    print(
        f"Preview decisions : "
        f"{total:,}"
    )

    print(
        f"Evaluated         : "
        f"{len(result):,}"
    )

    print(
        f"Missing actual    : "
        f"{missing_actual:,}"
    )

    print(
        f"Avg candidates    : "
        f"{result['n_candidates'].mean():.2f}"
    )

    return (
        result,
        candidates_out,
    )


# ============================================================
# 11. Ranking metrics
# ============================================================

def ranking_metrics(
    result,
    prefix,
):

    rank = (
        result[
            f"{prefix}_rank"
        ]
    )

    agreement = float(
        result[
            f"{prefix}_agreement"
        ]
        .mean()
    )

    top3 = float(
        (
            rank
            <= 3
        )
        .mean()
    )

    top5 = float(
        (
            rank
            <= 5
        )
        .mean()
    )

    mean_rank = float(
        rank.mean()
    )

    median_rank = float(
        rank.median()
    )

    return {
        "agreement":
            agreement,

        "top3":
            top3,

        "top5":
            top5,

        "mean_rank":
            mean_rank,

        "median_rank":
            median_rank,
    }


def print_ranking_comparison(
    split_name,
    result,
):

    m0 = ranking_metrics(
        result,
        "b0",
    )

    m1 = ranking_metrics(
        result,
        "b1",
    )

    print()
    print(
        f"{split_name} B0 vs B1"
    )

    print(
        f"{'Metric':15s}"
        f"{'B0':>12s}"
        f"{'B1(+LI)':>12s}"
        f"{'Delta':>12s}"
    )

    for key in [
        "agreement",
        "top3",
        "top5",
    ]:

        print(
            f"{key:15s}"
            f"{m0[key] * 100:11.2f}%"
            f"{m1[key] * 100:11.2f}%"
            f"{(m1[key] - m0[key]) * 100:+11.2f}p"
        )

    for key in [
        "mean_rank",
        "median_rank",
    ]:

        print(
            f"{key:15s}"
            f"{m0[key]:12.2f}"
            f"{m1[key]:12.2f}"
            f"{m1[key] - m0[key]:+12.2f}"
        )

    return (
        m0,
        m1,
    )


# ============================================================
# 12. B0 reproduction check
# ============================================================

def check_b0_reproduction(
    val_metrics,
    test_metrics,
):

    section(
        "6. B0 REPRODUCTION CHECK"
    )

    val_diff = abs(
        val_metrics[
            "agreement"
        ]
        -
        EXPECTED_B0[
            "validation_agreement"
        ]
    )

    test_diff = abs(
        test_metrics[
            "agreement"
        ]
        -
        EXPECTED_B0[
            "test_agreement"
        ]
    )

    print(
        f"Expected Validation B0 : "
        f"{EXPECTED_B0['validation_agreement'] * 100:.2f}%"
    )

    print(
        f"Current Validation B0  : "
        f"{val_metrics['agreement'] * 100:.2f}%"
    )

    print()

    print(
        f"Expected Test B0       : "
        f"{EXPECTED_B0['test_agreement'] * 100:.2f}%"
    )

    print(
        f"Current Test B0        : "
        f"{test_metrics['agreement'] * 100:.2f}%"
    )

    print()

    if (
        val_diff
        <= BASELINE_TOLERANCE
        and
        test_diff
        <= BASELINE_TOLERANCE
    ):

        print(
            "B0 reproduction : PASS"
        )

        return True

    print(
        "B0 reproduction : WARNING"
    )

    print(
        "Current B0 differs from the frozen baseline."
    )

    print(
        "Do not interpret B1 improvement until "
        "the baseline difference is explained."
    )

    return False


# ============================================================
# 13. Added LI coefficients
# ============================================================

def print_li_coefficients(
    li_extension,
    theta1,
):

    section(
        "7. B1 ADDED LI COEFFICIENTS"
    )

    names = (
        li_extension
        .new_feature_names()
    )

    added = (
        theta1[
            -len(
                names
            ):
        ]
    )

    coef = pd.DataFrame(
        {
            "feature":
                names,

            "coefficient":
                added,
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
    )

    print(
        coef[
            [
                "feature",
                "coefficient",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "coefficient":
                    lambda x:
                    f"{x:+.6f}"
            },
        )
    )

    return coef


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 84
    )

    print(
        "STEP 13-3 - LEVERAGE INDEX BANDIT ABLATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen pre-LI Bandit v2"
    )

    print()

    print(
        "B1:"
    )

    print(
        "  B0 + leverage_index"
    )

    print(
        "  B0 + leverage_index x pitcher interactions"
    )

    print()

    print(
        "Everything else remains unchanged:"
    )

    print(
        "  reward / candidate set / pitcher profile / split / alpha"
    )

    print(
        "=" * 84
    )

    # ========================================================
    # Load
    # ========================================================

    (
        pa,
        history,
        bullpen,
        decisions,
        train,
        val,
        test,
        bullpen_map,
        used_before_map,
    ) = load_data()

    # ========================================================
    # Profiles
    # ========================================================

    profile_index = (
        build_profile_index(
            history
        )
    )

    # ========================================================
    # B0 processor
    # ========================================================

    (
        b0_processor,
        X0_train,
    ) = (
        fit_b0_processor(
            train
        )
    )

    # ========================================================
    # LI extension
    # ========================================================

    section(
        "4. FIT LI FEATURE EXTENSION"
    )

    li_extension = (
        LIFeatureExtension(
            b0_processor
        )
    )

    li_extension.fit(
        train
    )

    # ========================================================
    # Models
    # ========================================================

    (
        theta0,
        theta1,
    ) = (
        fit_models(
            train,
            b0_processor,
            li_extension,
        )
    )

    # ========================================================
    # Reward prediction
    # ========================================================

    section(
        "REWARD PREDICTION DIAGNOSTICS"
    )

    reward_rows = []

    for name, split_df in [
        (
            "train",
            train,
        ),
        (
            "validation",
            val,
        ),
        (
            "test",
            test,
        ),
    ]:

        reward_rows.extend(
            reward_prediction_check(
                name,
                split_df,
                li_extension,
                theta0,
                theta1,
            )
        )

    reward_summary = pd.DataFrame(
        reward_rows
    )

    # ========================================================
    # Actual bullpen ranking
    # ========================================================

    (
        val_result,
        val_candidates,
    ) = (
        evaluate_candidate_ranking(
            "VALIDATION",
            val,
            bullpen_map,
            used_before_map,
            profile_index,
            li_extension,
            theta0,
            theta1,
        )
    )

    (
        test_result,
        test_candidates,
    ) = (
        evaluate_candidate_ranking(
            "TEST",
            test,
            bullpen_map,
            used_before_map,
            profile_index,
            li_extension,
            theta0,
            theta1,
        )
    )

    # ========================================================
    # Ranking summaries
    # ========================================================

    section(
        "B0 VS B1 RANKING SUMMARY"
    )

    (
        val_b0,
        val_b1,
    ) = (
        print_ranking_comparison(
            "VALIDATION",
            val_result,
        )
    )

    print()

    (
        test_b0,
        test_b1,
    ) = (
        print_ranking_comparison(
            "TEST",
            test_result,
        )
    )

    # ========================================================
    # B0 reproduction
    # ========================================================

    baseline_ok = (
        check_b0_reproduction(
            val_b0,
            test_b0,
        )
    )

    # ========================================================
    # LI coefficients
    # ========================================================

    li_coef = (
        print_li_coefficients(
            li_extension,
            theta1,
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    for (
        split_name,
        model_name,
        metrics,
    ) in [
        (
            "validation",
            "B0",
            val_b0,
        ),
        (
            "validation",
            "B1_LI",
            val_b1,
        ),
        (
            "test",
            "B0",
            test_b0,
        ),
        (
            "test",
            "B1_LI",
            test_b1,
        ),
    ]:

        summary_rows.append(
            {
                "split":
                    split_name,

                "model":
                    model_name,

                **metrics,
            }
        )

    ranking_summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Save
    # ========================================================

    section(
        "8. SAVE"
    )

    ranking_summary.to_csv(
        OUTPUT_DIR
        / "ranking_summary.csv",
        index=False,
    )

    reward_summary.to_csv(
        OUTPUT_DIR
        / "reward_prediction_summary.csv",
        index=False,
    )

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

    li_coef.to_csv(
        OUTPUT_DIR
        / "li_coefficients.csv",
        index=False,
    )

    np.save(
        OUTPUT_DIR
        / "theta_b0.npy",
        theta0,
    )

    np.save(
        OUTPUT_DIR
        / "theta_b1_li.npy",
        theta1,
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "ranking_summary.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "reward_prediction_summary.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_candidates.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_candidates.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "li_coefficients.csv"
    )

    # ========================================================
    # Final
    # ========================================================

    section(
        "FINAL"
    )

    print(
        f"B0 reproduction : "
        f"{'PASS' if baseline_ok else 'WARNING'}"
    )

    print()

    print(
        "TEST"
    )

    print(
        f"B0 Agreement    : "
        f"{test_b0['agreement'] * 100:.2f}%"
    )

    print(
        f"B1 Agreement    : "
        f"{test_b1['agreement'] * 100:.2f}%"
    )

    print(
        f"Delta           : "
        f"{(test_b1['agreement'] - test_b0['agreement']) * 100:+.2f}p"
    )

    print()

    print(
        f"B0 Top-3        : "
        f"{test_b0['top3'] * 100:.2f}%"
    )

    print(
        f"B1 Top-3        : "
        f"{test_b1['top3'] * 100:.2f}%"
    )

    print()

    print(
        f"B0 Mean Rank    : "
        f"{test_b0['mean_rank']:.2f}"
    )

    print(
        f"B1 Mean Rank    : "
        f"{test_b1['mean_rank']:.2f}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Agreement alone does not determine policy quality."
    )

    print(
        "If B0 reproduction passes, the next step is"
    )

    print(
        "LI-segment analysis followed by B0 vs B1 OPE."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()
