from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.core import bandit_v2_profile_fixed as profile_fix
from kbo_bandit.core import bandit_v2_interactions as interactions
from kbo_bandit.quality import bandit_v3_quality_ablation as quality_mod


# =============================================================================
# CONFIG
# =============================================================================

LI_PATH = Path(
    "data_analysis/results/all_pa_with_li_train_frozen.parquet"
)

QUALITY_PATH = Path(
    "data_analysis/results/all_pa_with_quality_train_frozen.parquet"
)

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_li_quality_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REWARD_MODE = "wpa_per_pa"
ALPHA = 0.0
RIDGE = 1.0


# -----------------------------------------------------------------------------
# Frozen baseline results
# -----------------------------------------------------------------------------

EXPECTED = {
    "validation": {
        "B0": 0.0982,
        "B1": 0.1116,
        "B2": 0.0966,
    },
    "test": {
        "B0": 0.1059,
        "B1": 0.1203,
        "B2": 0.1036,
    },
}

REPRO_TOLERANCE = 0.002


# -----------------------------------------------------------------------------
# B0 feature layout
#
# Previous frozen B0:
#   intercept          = 1
#   situation features = 9
#   pitcher features   = 7
#   interactions       = 63
#
# total = 80
#
# InteractionFeatureProcessor의 출력이
# [intercept, standardized base features, interactions]
# 순서라는 기존 B0 구조를 사용한다.
# -----------------------------------------------------------------------------

N_SITUATION_FEATURES = 9
N_PITCHER_FEATURES = 7

PITCHER_BASE_START = (
    1
    +
    N_SITUATION_FEATURES
)

PITCHER_BASE_END = (
    PITCHER_BASE_START
    +
    N_PITCHER_FEATURES
)


LI_ADDED_FEATURE_NAMES = [
    "leverage_index",
    "leverage_index__X__prior_n_apps_log",
    "leverage_index__X__prior_pitcher_reward_mean",
    "leverage_index__X__prior_wpa_std",
    "leverage_index__X__prior_bb_rate",
    "leverage_index__X__prior_so_rate",
    "leverage_index__X__prior_hr_rate",
    "leverage_index__X__prior_avg_pa_per_app",
]


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


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
# 1. LOAD LI DATA
# =============================================================================

def load_li_data() -> pd.DataFrame:
    section(
        "1. LOAD TRAIN-FROZEN LEVERAGE INDEX DATA"
    )

    if not LI_PATH.exists():
        raise FileNotFoundError(
            f"Train-frozen LI parquet not found:\n  {LI_PATH}"
        )

    li_pa = pd.read_parquet(
        LI_PATH
    )

    required = [
        "_orig_idx",
        "game_id",
        "pitcher_id",
        "appearance_order",
        "leverage_index",
    ]

    missing = [
        col
        for col in required
        if col not in li_pa.columns
    ]

    if missing:
        raise ValueError(
            f"LI parquet missing columns: {missing}"
        )

    li = pd.to_numeric(
        li_pa["leverage_index"],
        errors="coerce",
    )

    print(
        f"LI input   : {LI_PATH}"
    )

    print(
        f"Shape      : {li_pa.shape}"
    )

    print(
        f"LI missing : {li.isna().sum():,}"
    )

    print(
        f"LI finite  : "
        f"{np.isfinite(li.dropna()).sum():,}"
        f" / {li.notna().sum():,}"
    )

    if li.isna().any():
        raise ValueError(
            "Train-frozen LI contains missing values."
        )

    if not np.isfinite(li).all():
        raise ValueError(
            "Train-frozen LI contains inf/-inf."
        )

    return li_pa


# =============================================================================
# 2. LOAD QUALITY DATA
# =============================================================================

def load_quality_data() -> pd.DataFrame:
    section(
        "2. LOAD TRAIN-FROZEN QUALITY DATA"
    )

    if not QUALITY_PATH.exists():
        raise FileNotFoundError(
            f"Train-frozen quality parquet not found:\n  {QUALITY_PATH}"
        )

    quality_pa = pd.read_parquet(
        QUALITY_PATH
    )

    required = [
        "_orig_idx",
        "game_id",
        "pitcher_id",
        "appearance_order",
        "quality_p",
    ]

    missing = [
        col
        for col in required
        if col not in quality_pa.columns
    ]

    if missing:
        raise ValueError(
            f"Quality parquet missing columns: {missing}"
        )

    quality = pd.to_numeric(
        quality_pa["quality_p"],
        errors="coerce",
    )

    print(
        f"Quality input : {QUALITY_PATH}"
    )

    print(
        f"Shape         : {quality_pa.shape}"
    )

    print(
        f"quality non-null : "
        f"{quality.notna().sum():,}"
        f" / {len(quality):,}"
    )

    print(
        f"quality null     : "
        f"{quality.isna().sum():,}"
    )

    finite = quality.dropna()

    if finite.empty:
        raise ValueError(
            "quality_p is completely empty."
        )

    if not np.isfinite(finite).all():
        raise ValueError(
            "quality_p contains inf/-inf."
        )

    print(
        f"quality mean   : {finite.mean():.6f}"
    )

    print(
        f"quality median : {finite.median():.6f}"
    )

    print(
        f"quality min    : {finite.min():.6f}"
    )

    print(
        f"quality max    : {finite.max():.6f}"
    )

    return quality_pa


# =============================================================================
# 3. LOAD B0 DATA
# =============================================================================

def load_b0_data():
    section(
        "3. LOAD FROZEN B0 BANDIT DATA"
    )

    base.REWARD_MODE = REWARD_MODE
    base.ALPHA = ALPHA

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

    print(
        f"Original PA       : {pa.shape}"
    )

    print(
        f"Pitcher history   : {history.shape}"
    )

    print(
        f"Game bullpen      : {bullpen.shape}"
    )

    print(
        f"Relief decisions  : {len(decisions):,}"
    )

    print(
        f"Decision games    : "
        f"{decisions['game_id'].nunique():,}"
    )

    return (
        pa,
        history,
        bullpen,
        decisions,
        first_app,
    )


# =============================================================================
# 4. ALIGNMENT
# =============================================================================

def verify_alignment(
    original_pa: pd.DataFrame,
    enriched_pa: pd.DataFrame,
    name: str,
) -> None:

    print()
    print(
        f"[{name}]"
    )

    print(
        f"Original rows : {len(original_pa):,}"
    )

    print(
        f"{name} rows : {len(enriched_pa):,}"
    )

    if len(original_pa) != len(enriched_pa):
        raise ValueError(
            f"{name}: row count mismatch"
        )

    original_idx = pd.to_numeric(
        original_pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    enriched_idx = pd.to_numeric(
        enriched_pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    same_idx = np.array_equal(
        original_idx,
        enriched_idx,
    )

    print(
        f"Same _orig_idx order   : {same_idx}"
    )

    if not same_idx:
        raise ValueError(
            f"{name}: _orig_idx mismatch"
        )

    for col in [
        "game_id",
        "pitcher_id",
        "appearance_order",
    ]:
        if col == "pitcher_id":
            left = (
                original_pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

            right = (
                enriched_pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

        else:
            left = (
                original_pa[col]
                .fillna("__NA__")
                .astype(str)
                .to_numpy()
            )

            right = (
                enriched_pa[col]
                .fillna("__NA__")
                .astype(str)
                .to_numpy()
            )

        same = np.array_equal(
            left,
            right,
        )

        print(
            f"Same {col:18s}: {same}"
        )

        if not same:
            raise ValueError(
                f"{name}: {col} mismatch"
            )


# =============================================================================
# 5. ATTACH LI / QUALITY TO LOGGED DECISIONS
# =============================================================================

def attach_li(
    decisions: pd.DataFrame,
    li_pa: pd.DataFrame,
) -> pd.DataFrame:

    lookup = (
        li_pa[
            [
                "_orig_idx",
                "leverage_index",
            ]
        ]
        .copy()
    )

    lookup["_orig_idx"] = pd.to_numeric(
        lookup["_orig_idx"],
        errors="raise",
    ).astype(int)

    lookup["leverage_index"] = pd.to_numeric(
        lookup["leverage_index"],
        errors="raise",
    )

    if lookup["_orig_idx"].duplicated().any():
        raise ValueError(
            "Duplicate _orig_idx in LI lookup."
        )

    result = decisions.copy()

    result["_orig_idx"] = pd.to_numeric(
        result["_orig_idx"],
        errors="raise",
    ).astype(int)

    before = len(
        result
    )

    result = result.merge(
        lookup,
        on="_orig_idx",
        how="left",
        validate="many_to_one",
    )

    if len(result) != before:
        raise ValueError(
            "LI merge changed decision count."
        )

    missing = int(
        result["leverage_index"]
        .isna()
        .sum()
    )

    print(
        f"Missing decision LI : {missing:,}"
    )

    if missing > 0:
        raise ValueError(
            "Decision LI missing."
        )

    result = result.rename(
        columns={
            "leverage_index":
                "leverage_index_model"
        }
    )

    return result


def attach_quality(
    decisions: pd.DataFrame,
    quality_pa: pd.DataFrame,
) -> pd.DataFrame:

    lookup = (
        quality_pa[
            [
                "_orig_idx",
                "quality_p",
            ]
        ]
        .copy()
    )

    lookup["_orig_idx"] = pd.to_numeric(
        lookup["_orig_idx"],
        errors="raise",
    ).astype(int)

    lookup["quality_p"] = pd.to_numeric(
        lookup["quality_p"],
        errors="coerce",
    )

    if lookup["_orig_idx"].duplicated().any():
        raise ValueError(
            "Duplicate _orig_idx in quality lookup."
        )

    result = decisions.copy()

    result["_orig_idx"] = pd.to_numeric(
        result["_orig_idx"],
        errors="raise",
    ).astype(int)

    before = len(
        result
    )

    result = result.merge(
        lookup,
        on="_orig_idx",
        how="left",
        validate="many_to_one",
    )

    if len(result) != before:
        raise ValueError(
            "Quality merge changed decision count."
        )

    result = result.rename(
        columns={
            "quality_p":
                "quality_p_logged"
        }
    )

    print(
        f"Missing decision quality : "
        f"{result['quality_p_logged'].isna().sum():,}"
    )

    return result


# =============================================================================
# 6. SPLIT + TRAIN-ONLY QUALITY FALLBACK
# =============================================================================

def split_data(
    decisions: pd.DataFrame,
):
    section(
        "6. CHRONOLOGICAL SPLIT"
    )

    (
        train,
        validation,
        test,
    ) = base.chronological_split(
        decisions
    )

    train_quality = pd.to_numeric(
        train["quality_p_logged"],
        errors="coerce",
    )

    valid_train_quality = (
        train_quality.dropna()
    )

    if valid_train_quality.empty:
        raise ValueError(
            "No valid Train quality values."
        )

    quality_fallback = float(
        valid_train_quality.median()
    )

    print()
    print(
        f"Train-only quality fallback : "
        f"{quality_fallback:.6f}"
    )

    result = []

    for name, frame in [
        (
            "Train",
            train,
        ),
        (
            "Validation",
            validation,
        ),
        (
            "Test",
            test,
        ),
    ]:
        current = frame.copy()

        quality = pd.to_numeric(
            current["quality_p_logged"],
            errors="coerce",
        )

        missing = int(
            quality.isna().sum()
        )

        current[
            "quality_p_model"
        ] = quality.fillna(
            quality_fallback
        )

        current[
            "leverage_index_model"
        ] = pd.to_numeric(
            current["leverage_index_model"],
            errors="raise",
        )

        print(
            f"{name:10s}: "
            f"{len(current):,} decisions | "
            f"quality missing={missing:,} | "
            f"LI mean="
            f"{current['leverage_index_model'].mean():.4f} | "
            f"quality mean="
            f"{current['quality_p_model'].mean():.4f}"
        )

        result.append(
            current
        )

    return (
        result[0],
        result[1],
        result[2],
        quality_fallback,
    )


# =============================================================================
# 7. QUALITY SNAPSHOT INDEX
# =============================================================================

def build_quality_index(
    quality_pa: pd.DataFrame,
    original_pa: pd.DataFrame,
):
    return (
        quality_mod
        .build_quality_snapshot_index(
            quality_pa,
            original_pa,
        )
    )


# =============================================================================
# 8. PROFILE INDEX
# =============================================================================

def build_profile_index(
    history: pd.DataFrame,
):
    section(
        "7. BUILD FULL PITCHER PROFILE INDEX"
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

    print(
        f"Indexed pitchers : "
        f"{len(profile_index):,}"
    )

    print(
        f"History rows     : "
        f"{len(full_history):,}"
    )

    return profile_index


# =============================================================================
# 9. B0 PROCESSOR
# =============================================================================

def fit_b0_processor(
    train: pd.DataFrame,
):
    section(
        "8. FIT FROZEN B0 FEATURE PROCESSOR"
    )

    processor = (
        interactions
        .InteractionFeatureProcessor()
    )

    processor.fit(
        train
    )

    X0 = processor.transform_frame(
        train
    )

    print(
        f"B0 feature dim : {X0.shape[1]}"
    )

    if X0.shape[1] != 80:
        raise ValueError(
            "Frozen B0 dimension is not 80."
        )

    if (
        PITCHER_BASE_END
        >
        X0.shape[1]
    ):
        raise ValueError(
            "Invalid pitcher feature slice."
        )

    return processor


# =============================================================================
# 10. FEATURE EXTENSION
# =============================================================================

class LIQualityFeatureExtension:

    def __init__(
        self,
        b0_processor,
    ):
        self.b0_processor = (
            b0_processor
        )

        self.li_mean = None
        self.li_std = None

        self.quality_mean = None
        self.quality_std = None

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:

        li = pd.to_numeric(
            train["leverage_index_model"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        quality = pd.to_numeric(
            train["quality_p_model"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        self.li_mean = float(
            li.mean()
        )

        self.li_std = float(
            li.std(
                ddof=0
            )
        )

        self.quality_mean = float(
            quality.mean()
        )

        self.quality_std = float(
            quality.std(
                ddof=0
            )
        )

        if (
            not np.isfinite(
                self.li_std
            )
            or
            self.li_std <= 0
        ):
            raise ValueError(
                "Invalid Train LI std."
            )

        if (
            not np.isfinite(
                self.quality_std
            )
            or
            self.quality_std <= 0
        ):
            raise ValueError(
                "Invalid Train quality std."
            )

        print(
            f"Train LI mean      : "
            f"{self.li_mean:.6f}"
        )

        print(
            f"Train LI std       : "
            f"{self.li_std:.6f}"
        )

        print(
            f"Train quality mean : "
            f"{self.quality_mean:.6f}"
        )

        print(
            f"Train quality std  : "
            f"{self.quality_std:.6f}"
        )

    def transform(
        self,
        frame: pd.DataFrame,
    ):
        X0 = (
            self.b0_processor
            .transform_frame(
                frame
            )
        )

        li = pd.to_numeric(
            frame["leverage_index_model"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        quality = pd.to_numeric(
            frame["quality_p_model"],
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
        # Frozen B0의 standardized pitcher base features.
        #
        # X0:
        # [intercept,
        #  9 standardized situation features,
        #  7 standardized pitcher features,
        #  63 context x pitcher interactions]
        # ---------------------------------------------------------------------

        pitcher_z = X0[
            :,
            PITCHER_BASE_START:
            PITCHER_BASE_END
        ]

        if (
            pitcher_z.shape[1]
            !=
            N_PITCHER_FEATURES
        ):
            raise ValueError(
                "Could not resolve 7 standardized pitcher features."
            )

        li_pitcher_interactions = (
            z_li.reshape(
                -1,
                1,
            )
            *
            pitcher_z
        )

        li_block = np.column_stack(
            [
                z_li,
                li_pitcher_interactions,
            ]
        )

        # B1 = B0 + LI + LI x pitcher
        X1 = np.column_stack(
            [
                X0,
                li_block,
            ]
        )

        # B2 = B0 + quality
        X2 = np.column_stack(
            [
                X0,
                z_quality,
            ]
        )

        # B3 = B0 + LI block + quality
        X3 = np.column_stack(
            [
                X0,
                li_block,
                z_quality,
            ]
        )

        return (
            X0,
            X1,
            X2,
            X3,
        )


# =============================================================================
# 11. RIDGE
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

    b = (
        X.T
        @ y
    )

    return np.linalg.solve(
        A,
        b,
    )


def fit_models(
    train: pd.DataFrame,
    extension: LIQualityFeatureExtension,
):
    section(
        "9. FIT B0 / B1 / B2 / B3 REWARD MODELS"
    )

    (
        X0,
        X1,
        X2,
        X3,
    ) = extension.transform(
        train
    )

    y = pd.to_numeric(
        train["reward"],
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

    if (
        X0.shape[1] != 80
        or
        X1.shape[1] != 88
        or
        X2.shape[1] != 81
        or
        X3.shape[1] != 89
    ):
        raise ValueError(
            "Unexpected feature dimensions."
        )

    print()
    print(
        "B1 LI block coefficients:"
    )

    for name, coefficient in zip(
        LI_ADDED_FEATURE_NAMES,
        theta1[-8:],
    ):
        print(
            f"  {name:55s} "
            f"{coefficient:+.6f}"
        )

    print()
    print(
        f"B2 quality coefficient : "
        f"{theta2[-1]:+.6f}"
    )

    print(
        f"B3 quality coefficient : "
        f"{theta3[-1]:+.6f}"
    )

    return (
        theta0,
        theta1,
        theta2,
        theta3,
    )


# =============================================================================
# 12. REWARD PREDICTION
# =============================================================================

def prediction_metrics(
    y: np.ndarray,
    pred: np.ndarray,
) -> dict:

    y = np.asarray(
        y,
        dtype=float,
    )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    error = (
        y
        -
        pred
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
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
            np.sum(
                error ** 2
            )
            /
            denominator
        )
    else:
        r2 = np.nan

    pearson = float(
        pd.Series(
            y
        ).corr(
            pd.Series(
                pred
            ),
            method="pearson",
        )
    )

    spearman = float(
        pd.Series(
            y
        ).corr(
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


def evaluate_reward_prediction(
    split_name: str,
    frame: pd.DataFrame,
    extension: LIQualityFeatureExtension,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
):

    (
        X0,
        X1,
        X2,
        X3,
    ) = extension.transform(
        frame
    )

    y = pd.to_numeric(
        frame["reward"],
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
    }

    metrics = {
        model:
            prediction_metrics(
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
        f"{'B0':>12s}"
        f"{'B1(LI)':>12s}"
        f"{'B2(Q)':>12s}"
        f"{'B3(LI+Q)':>12s}"
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
            f"{metrics['B0'][metric]:12.6f}"
            f"{metrics['B1_LI'][metric]:12.6f}"
            f"{metrics['B2_QUALITY'][metric]:12.6f}"
            f"{metrics['B3_LI_QUALITY'][metric]:12.6f}"
        )

    rows = []

    for model, model_metrics in metrics.items():
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
# 13. CANDIDATE FRAME
# =============================================================================

def build_candidate_frame(
    decision: pd.Series,
    candidates,
    profile_index,
    quality_index,
    quality_fallback: float,
):

    feature_rows = []
    metadata = []

    decision_li = float(
        decision[
            "leverage_index_model"
        ]
    )

    for pitcher_id in candidates:

        (
            profile,
            profile_source,
        ) = (
            profile_fix.get_asof_profile(
                decision,
                pitcher_id,
                profile_index,
            )
        )

        values = (
            base.candidate_feature_values(
                decision,
                profile,
            )
        )

        (
            quality,
            quality_source,
        ) = (
            quality_mod.get_candidate_quality(
                decision,
                pitcher_id,
                quality_index,
                quality_fallback,
            )
        )

        feature = dict(
            values
        )

        feature[
            "leverage_index_model"
        ] = decision_li

        feature[
            "quality_p_model"
        ] = quality

        feature_rows.append(
            feature
        )

        metadata.append(
            {
                "pitcher_id":
                    normalize_id(
                        pitcher_id
                    ),

                "profile_source":
                    profile_source,

                "quality_source":
                    quality_source,

                "quality_p":
                    quality,

                "leverage_index":
                    decision_li,
            }
        )

    return (
        pd.DataFrame(
            feature_rows
        ),
        pd.DataFrame(
            metadata
        ),
    )


# =============================================================================
# 14. RANKING
# =============================================================================

def rank_candidates(
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
                scores[i]
            ),
            str(
                pitcher_ids[i]
            ),
        ),
    )

    ranking = {}

    for rank, index in enumerate(
        order,
        start=1,
    ):
        ranking[
            str(
                pitcher_ids[index]
            )
        ] = rank

    recommendation = str(
        pitcher_ids[
            order[0]
        ]
    )

    return (
        recommendation,
        ranking,
    )


# =============================================================================
# 15. ACTUAL BULLPEN EVALUATION
# =============================================================================

def evaluate_actual_bullpen(
    split_name: str,
    frame: pd.DataFrame,
    bullpen_map,
    used_before_map,
    profile_index,
    quality_index,
    quality_fallback: float,
    extension: LIQualityFeatureExtension,
    theta0: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
):

    section(
        f"10. ACTUAL BULLPEN EVALUATION - {split_name}"
    )

    work = (
        frame[
            frame["bullpen_source"]
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

    missing_actual = 0
    decision_id = 0

    quality_source_counter = Counter()

    for _, decision in work.iterrows():

        game_id = str(
            decision["game_id"]
        )

        team = str(
            decision["def_team"]
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

        bullpen_pitchers = set(
            bullpen_map.get(
                (
                    game_id,
                    team,
                ),
                set(),
            )
        )

        used_pitchers = set(
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
            bullpen_pitchers
            -
            used_pitchers
        )

        candidate_ids = [
            normalize_id(
                pitcher_id
            )
            for pitcher_id in candidates
        ]

        if (
            actual_pitcher
            not in candidate_ids
        ):
            missing_actual += 1
            continue

        if len(candidate_ids) < 2:
            continue

        (
            candidate_frame,
            metadata,
        ) = build_candidate_frame(
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
        }

        recommendations = {}
        rankings = {}

        for model_name, model_scores in scores.items():
            (
                recommendation,
                ranking,
            ) = rank_candidates(
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
                    decision["date"],

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
                    recommendations["b0"],

                "b1_rec":
                    recommendations["b1"],

                "b2_rec":
                    recommendations["b2"],

                "b3_rec":
                    recommendations["b3"],

                "b0_agreement":
                    int(
                        recommendations["b0"]
                        ==
                        actual_pitcher
                    ),

                "b1_agreement":
                    int(
                        recommendations["b1"]
                        ==
                        actual_pitcher
                    ),

                "b2_agreement":
                    int(
                        recommendations["b2"]
                        ==
                        actual_pitcher
                    ),

                "b3_agreement":
                    int(
                        recommendations["b3"]
                        ==
                        actual_pitcher
                    ),

                "b0_rank":
                    int(
                        rankings["b0"][
                            actual_pitcher
                        ]
                    ),

                "b1_rank":
                    int(
                        rankings["b1"][
                            actual_pitcher
                        ]
                    ),

                "b2_rank":
                    int(
                        rankings["b2"][
                            actual_pitcher
                        ]
                    ),

                "b3_rank":
                    int(
                        rankings["b3"][
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
                        decision["date"],

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

                    "profile_source":
                        metadata.iloc[
                            index
                        ][
                            "profile_source"
                        ],

                    "quality_source":
                        metadata.iloc[
                            index
                        ][
                            "quality_source"
                        ],

                    "q_b0":
                        float(
                            scores["b0"][
                                index
                            ]
                        ),

                    "q_b1_li":
                        float(
                            scores["b1"][
                                index
                            ]
                        ),

                    "q_b2_quality":
                        float(
                            scores["b2"][
                                index
                            ]
                        ),

                    "q_b3_li_quality":
                        float(
                            scores["b3"][
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
        f"Preview decisions : {len(work):,}"
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
# 16. RANKING METRICS
# =============================================================================

def ranking_metrics(
    frame: pd.DataFrame,
    prefix: str,
) -> dict:

    rank = frame[
        f"{prefix}_rank"
    ]

    return {
        "agreement":
            float(
                frame[
                    f"{prefix}_agreement"
                ]
                .mean()
            ),

        "top3":
            float(
                (
                    rank <= 3
                )
                .mean()
            ),

        "top5":
            float(
                (
                    rank <= 5
                )
                .mean()
            ),

        "mean_rank":
            float(
                rank.mean()
            ),

        "median_rank":
            float(
                rank.median()
            ),
    }


def print_ranking_summary(
    split_name: str,
    frame: pd.DataFrame,
):

    models = [
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
    ]

    results = {
        label:
            ranking_metrics(
                frame,
                prefix,
            )
        for label, prefix
        in models
    }

    print()
    print(
        split_name
    )

    print(
        f"{'Metric':16s}"
        f"{'B0':>12s}"
        f"{'B1(+LI)':>12s}"
        f"{'B2(+Q)':>12s}"
        f"{'B3(+LI+Q)':>14s}"
    )

    for metric in [
        "agreement",
        "top3",
        "top5",
    ]:
        print(
            f"{metric:16s}"
            f"{results['B0'][metric] * 100:11.2f}%"
            f"{results['B1(+LI)'][metric] * 100:11.2f}%"
            f"{results['B2(+Q)'][metric] * 100:11.2f}%"
            f"{results['B3(+LI+Q)'][metric] * 100:13.2f}%"
        )

    for metric in [
        "mean_rank",
        "median_rank",
    ]:
        print(
            f"{metric:16s}"
            f"{results['B0'][metric]:12.2f}"
            f"{results['B1(+LI)'][metric]:12.2f}"
            f"{results['B2(+Q)'][metric]:12.2f}"
            f"{results['B3(+LI+Q)'][metric]:14.2f}"
        )

    return results


# =============================================================================
# 17. REPRODUCTION CHECK
# =============================================================================

def check_reproduction(
    split: str,
    results: dict,
) -> bool:

    print()
    print(
        f"{split.upper()} reproduction"
    )

    passed = True

    mapping = {
        "B0":
            "B0",

        "B1":
            "B1(+LI)",

        "B2":
            "B2(+Q)",
    }

    for expected_key, result_key in mapping.items():

        expected = (
            EXPECTED[
                split
            ][
                expected_key
            ]
        )

        current = (
            results[
                result_key
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
            f"{expected_key:3s} "
            f"expected={expected * 100:6.2f}% | "
            f"current={current * 100:6.2f}% | "
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
        "=" * 100
    )

    print(
        "STEP 4 - B0 + LI + QUALITY_P ABLATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen contextual bandit baseline"
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
        "  B0 + leverage_index + LI x pitcher interactions + quality_p"
    )

    print()

    print(
        "NOT INCLUDED YET:"
    )

    print(
        "  leverage_index x quality_p"
    )

    print()

    print(
        "Everything else remains frozen:"
    )

    print(
        "  reward / candidate set / profile / split / alpha / ridge"
    )

    print(
        "=" * 100
    )

    # =========================================================================
    # Data
    # =========================================================================

    li_pa = (
        load_li_data()
    )

    quality_pa = (
        load_quality_data()
    )

    (
        pa,
        history,
        bullpen,
        decisions,
        first_app,
    ) = (
        load_b0_data()
    )

    section(
        "4. VERIFY DATA ALIGNMENT"
    )

    verify_alignment(
        pa,
        li_pa,
        "LI",
    )

    verify_alignment(
        pa,
        quality_pa,
        "QUALITY",
    )

    # =========================================================================
    # Decision-level features
    # =========================================================================

    section(
        "5. ATTACH LI / QUALITY TO LOGGED DECISIONS"
    )

    decisions = attach_li(
        decisions,
        li_pa,
    )

    decisions = attach_quality(
        decisions,
        quality_pa,
    )

    (
        train,
        validation,
        test,
        quality_fallback,
    ) = split_data(
        decisions
    )

    # =========================================================================
    # As-of candidate information
    # =========================================================================

    quality_index = (
        build_quality_index(
            quality_pa,
            pa,
        )
    )

    profile_index = (
        build_profile_index(
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
    # Features
    # =========================================================================

    b0_processor = (
        fit_b0_processor(
            train
        )
    )

    section(
        "9. FIT LI + QUALITY FEATURE EXTENSION"
    )

    extension = (
        LIQualityFeatureExtension(
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
            )
        )

    reward_summary = pd.DataFrame(
        reward_rows
    )

    # =========================================================================
    # Ranking
    # =========================================================================

    (
        validation_decisions,
        validation_candidates,
        validation_sources,
    ) = (
        evaluate_actual_bullpen(
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
        )
    )

    (
        test_decisions,
        test_candidates,
        test_sources,
    ) = (
        evaluate_actual_bullpen(
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
        )
    )

    # =========================================================================
    # Ranking summary
    # =========================================================================

    section(
        "B0 / B1 / B2 / B3 RANKING SUMMARY"
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

    section(
        "11. B0 / B1 / B2 REPRODUCTION CHECK"
    )

    validation_reproduction = (
        check_reproduction(
            "validation",
            validation_results,
        )
    )

    test_reproduction = (
        check_reproduction(
            "test",
            test_results,
        )
    )

    all_reproduced = (
        validation_reproduction
        and
        test_reproduction
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
    # Ranking summary CSV
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

        "dimensions": {
            "B0":
                int(
                    len(
                        theta0
                    )
                ),

            "B1":
                int(
                    len(
                        theta1
                    )
                ),

            "B2":
                int(
                    len(
                        theta2
                    )
                ),

            "B3":
                int(
                    len(
                        theta3
                    )
                ),
        },
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
        "B0/B1/B2 reproduction : "
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
        "IMPORTANT:"
    )

    print(
        "B3 does NOT contain LI x quality_p yet."
    )

    print(
        "If B0/B1/B2 reproduction passes, "
        "the next step is B0/B1/B2/B3 OPE comparison."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()