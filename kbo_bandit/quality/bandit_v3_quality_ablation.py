from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.core import bandit_v2_profile_fixed as profile_fix
from kbo_bandit.core import bandit_v2_interactions as interactions


# =============================================================================
# CONFIG
# =============================================================================

QUALITY_PATH = Path(
    "data_analysis/results/all_pa_with_quality_train_frozen.parquet"
)

OUTPUT_DIR = Path(
    "outputs/bandit/quality/v3_quality_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# B0에서 확정된 값
RIDGE = 1.0
ALPHA = 0.0
REWARD_MODE = "wpa_per_pa"


# 이전 frozen B0 결과
EXPECTED_B0_VALIDATION_AGREEMENT = 0.0982
EXPECTED_B0_TEST_AGREEMENT = 0.1059

B0_REPRO_TOLERANCE = 0.002


# =============================================================================
# COMMON
# =============================================================================

def section(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def normalize_id(value) -> str:
    if pd.isna(value):
        return "__MISSING__"

    value = str(value).strip()

    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]

    return value


# =============================================================================
# 1. QUALITY DATA
# =============================================================================

def load_quality_data() -> pd.DataFrame:
    section(
        "1. LOAD TRAIN-FROZEN QUALITY DATA"
    )

    if not QUALITY_PATH.exists():
        raise FileNotFoundError(
            "\nTrain-frozen quality parquet을 찾을 수 없습니다.\n"
            f"Expected:\n  {QUALITY_PATH}\n\n"
            "팀원에게 받은 quality parquet을 위 경로에 두거나 "
            "QUALITY_PATH를 실제 파일명으로 수정해주세요."
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

    print(
        f"Quality input : {QUALITY_PATH}"
    )

    print(
        f"Shape         : {quality_pa.shape}"
    )

    quality = pd.to_numeric(
        quality_pa["quality_p"],
        errors="coerce",
    )

    print(
        f"quality_p non-null : "
        f"{quality.notna().sum():,}"
        f" / {len(quality):,}"
    )

    print(
        f"quality_p null     : "
        f"{quality.isna().sum():,}"
    )

    if quality.notna().sum() == 0:
        raise ValueError(
            "quality_p가 전부 NaN입니다."
        )

    finite = quality.dropna()

    if not np.isfinite(finite).all():
        raise ValueError(
            "quality_p에 inf/-inf가 존재합니다."
        )

    print(
        f"quality_p mean     : {finite.mean():.6f}"
    )

    print(
        f"quality_p median   : {finite.median():.6f}"
    )

    print(
        f"quality_p min      : {finite.min():.6f}"
    )

    print(
        f"quality_p max      : {finite.max():.6f}"
    )

    return quality_pa


# =============================================================================
# 2. LOAD FROZEN B0
# =============================================================================

def load_b0_data():
    section(
        "2. LOAD FROZEN B0 DATA"
    )

    base.REWARD_MODE = REWARD_MODE
    base.ALPHA = ALPHA

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    print(
        f"Original PA          : {pa.shape}"
    )

    print(
        f"Pitcher history      : {history.shape}"
    )

    print(
        f"Game bullpen         : {bullpen.shape}"
    )

    (
        decisions,
        first_app,
    ) = base.build_decision_dataset(
        pa,
        history,
    )

    print(
        f"Relief decisions     : {len(decisions):,}"
    )

    print(
        f"Decision games       : "
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
# 3. PA / QUALITY ALIGNMENT
# =============================================================================

def verify_quality_alignment(
    pa: pd.DataFrame,
    quality_pa: pd.DataFrame,
) -> None:

    section(
        "3. VERIFY ORIGINAL PA / QUALITY PA ALIGNMENT"
    )

    print(
        f"Original PA rows : {len(pa):,}"
    )

    print(
        f"Quality PA rows  : {len(quality_pa):,}"
    )

    if len(pa) != len(quality_pa):
        raise ValueError(
            "Original PA와 Quality PA row 수가 다릅니다."
        )

    if "_orig_idx" not in pa.columns:
        raise ValueError(
            "Original PA에 _orig_idx가 없습니다."
        )

    original_idx = pd.to_numeric(
        pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    quality_idx = pd.to_numeric(
        quality_pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    same_idx = np.array_equal(
        original_idx,
        quality_idx,
    )

    print(
        f"Same _orig_idx row order : {same_idx}"
    )

    if not same_idx:
        raise ValueError(
            "_orig_idx row alignment mismatch"
        )

    checks = [
        "game_id",
        "pitcher_id",
        "appearance_order",
    ]

    for col in checks:
        if col not in pa.columns:
            raise ValueError(
                f"Original PA missing column: {col}"
            )

        if col == "pitcher_id":
            left = (
                pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

            right = (
                quality_pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

        else:
            left = (
                pa[col]
                .fillna("__NA__")
                .astype(str)
                .to_numpy()
            )

            right = (
                quality_pa[col]
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
                f"PA identity mismatch: {col}"
            )

    print()
    print(
        "PA / quality alignment : PASS"
    )


# =============================================================================
# 4. ATTACH LOGGED QUALITY
# =============================================================================

def attach_logged_quality(
    decisions: pd.DataFrame,
    quality_pa: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "4. ATTACH QUALITY_P TO LOGGED DECISIONS"
    )

    lookup = quality_pa[
        [
            "_orig_idx",
            "quality_p",
        ]
    ].copy()

    lookup["_orig_idx"] = pd.to_numeric(
        lookup["_orig_idx"],
        errors="raise",
    ).astype(int)

    lookup["quality_p"] = pd.to_numeric(
        lookup["quality_p"],
        errors="coerce",
    )

    duplicate_count = int(
        lookup["_orig_idx"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate _orig_idx : {duplicate_count:,}"
    )

    if duplicate_count > 0:
        raise ValueError(
            "Quality lookup에 duplicate _orig_idx가 있습니다."
        )

    result = decisions.copy()

    result["_orig_idx"] = pd.to_numeric(
        result["_orig_idx"],
        errors="raise",
    ).astype(int)

    before = len(result)

    result = result.merge(
        lookup,
        on="_orig_idx",
        how="left",
        validate="many_to_one",
    )

    after = len(result)

    result = result.rename(
        columns={
            "quality_p":
                "quality_p_logged"
        }
    )

    missing = int(
        result["quality_p_logged"]
        .isna()
        .sum()
    )

    print(
        f"Decisions before merge : {before:,}"
    )

    print(
        f"Decisions after merge  : {after:,}"
    )

    print(
        f"Missing quality_p      : {missing:,}"
    )

    if before != after:
        raise ValueError(
            "Quality merge 후 decision 수가 변경되었습니다."
        )

    return result


# =============================================================================
# 5. SPLIT
# =============================================================================

def split_and_prepare_quality(
    decisions: pd.DataFrame,
):
    section(
        "5. CHRONOLOGICAL SPLIT"
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

    available = train_quality.dropna()

    if available.empty:
        raise ValueError(
            "Train quality_p가 전부 missing입니다."
        )

    # 첫 등판 등 quality가 정의되지 않은 경우를 위한
    # Train-only neutral fallback.
    quality_fallback = float(
        available.median()
    )

    print()
    print(
        f"Train-only quality fallback "
        f"(median): {quality_fallback:.6f}"
    )

    prepared = []

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

        current["quality_p_model"] = (
            quality.fillna(
                quality_fallback
            )
        )

        print(
            f"{name:10s}: "
            f"{len(current):,} decisions | "
            f"quality missing "
            f"{missing:,} "
            f"({missing / len(current) * 100:.2f}%)"
        )

        prepared.append(
            current
        )

    return (
        prepared[0],
        prepared[1],
        prepared[2],
        quality_fallback,
    )


# =============================================================================
# 6. QUALITY SNAPSHOT INDEX
# =============================================================================

def build_quality_snapshot_index(
    quality_pa: pd.DataFrame,
    original_pa: pd.DataFrame,
):
    section(
        "6. BUILD QUALITY AS-OF SNAPSHOT INDEX"
    )

    work = quality_pa.copy()

    # quality 파일에 is_starter가 없으면
    # original PA에서 _orig_idx 기준으로 가져온다.
    if "is_starter" not in work.columns:
        starter_lookup = original_pa[
            [
                "_orig_idx",
                "is_starter",
            ]
        ].copy()

        work = work.merge(
            starter_lookup,
            on="_orig_idx",
            how="left",
            validate="one_to_one",
        )

    required = [
        "_orig_idx",
        "game_id",
        "pitcher_id",
        "appearance_order",
        "is_starter",
        "quality_p",
    ]

    missing = [
        col
        for col in required
        if col not in work.columns
    ]

    if missing:
        raise ValueError(
            f"Quality snapshot missing columns: {missing}"
        )

    work["_orig_idx"] = pd.to_numeric(
        work["_orig_idx"],
        errors="raise",
    ).astype(int)

    work["appearance_order"] = pd.to_numeric(
        work["appearance_order"],
        errors="coerce",
    )

    work["quality_p"] = pd.to_numeric(
        work["quality_p"],
        errors="coerce",
    )

    work["pitcher_key"] = (
        work["pitcher_id"]
        .apply(normalize_id)
    )

    work = work.dropna(
        subset=[
            "appearance_order",
        ]
    ).copy()

    work["appearance_order"] = (
        work["appearance_order"]
        .astype(int)
    )

    # PA 단위 데이터를 appearance 단위 snapshot으로 변환
    appearance = (
        work.sort_values(
            "_orig_idx"
        )
        .groupby(
            [
                "game_id",
                "pitcher_key",
                "appearance_order",
            ],
            as_index=False,
        )
        .agg(
            first_orig_idx=(
                "_orig_idx",
                "min",
            ),
            is_starter=(
                "is_starter",
                "first",
            ),
            quality_p=(
                "quality_p",
                "first",
            ),
        )
    )

    appearance = appearance.sort_values(
        [
            "pitcher_key",
            "first_orig_idx",
        ]
    ).reset_index(
        drop=True
    )

    index = {}

    for pitcher_id, group in appearance.groupby(
        "pitcher_key",
        sort=False,
    ):
        index[str(pitcher_id)] = {
            "orig_idx":
                group["first_orig_idx"]
                .to_numpy(
                    dtype=np.int64
                ),

            "quality":
                group["quality_p"]
                .to_numpy(
                    dtype=float
                ),

            "is_starter":
                group["is_starter"]
                .astype(bool)
                .to_numpy(),
        }

    print(
        f"Appearance snapshots : {len(appearance):,}"
    )

    print(
        f"Indexed pitchers     : {len(index):,}"
    )

    return index


# =============================================================================
# 7. CANDIDATE QUALITY
# =============================================================================

def get_candidate_quality(
    decision: pd.Series,
    pitcher_id,
    quality_index,
    fallback: float,
):
    """
    Candidate 투수가 실제 해당 경기에서 등판하지 않았더라도
    decision 시점에서 사용할 수 있는 quality_p를 구한다.

    quality_p는 각 appearance 시작 직전의 prior 기록으로 계산된 값이라고
    가정한다.

    next appearance quality snapshot을 사용할 수 있는 이유:
    현재 decision과 그 투수의 next appearance 사이에 그 투수의 다른
    appearance가 없기 때문에, pitching-history 기반 quality 입력은
    현재 decision 시점과 동일하다.

    단 next appearance가 starter라면 starter용 quality일 수 있으므로
    사용하지 않고 과거 relief snapshot으로 fallback한다.
    """

    pid = normalize_id(
        pitcher_id
    )

    info = quality_index.get(
        pid
    )

    if info is None:
        return (
            fallback,
            "train_fallback_no_history",
        )

    decision_idx = int(
        decision["_orig_idx"]
    )

    indices = info["orig_idx"]
    qualities = info["quality"]
    starters = info["is_starter"]

    pos = int(
        np.searchsorted(
            indices,
            decision_idx,
            side="left",
        )
    )

    # -------------------------------------------------------------------------
    # 현재 appearance 또는 next appearance에 저장된 prior-quality 이용
    # -------------------------------------------------------------------------

    if pos < len(indices):
        q = qualities[pos]

        if (
            not starters[pos]
            and
            np.isfinite(q)
        ):
            if indices[pos] == decision_idx:
                source = (
                    "exact_current_relief"
                )
            else:
                source = (
                    "next_relief_prior_snapshot"
                )

            return (
                float(q),
                source,
            )

    # -------------------------------------------------------------------------
    # next snapshot을 사용할 수 없으면
    # 가장 최근의 과거 relief quality 사용
    # -------------------------------------------------------------------------

    previous = min(
        pos - 1,
        len(indices) - 1,
    )

    while previous >= 0:
        q = qualities[previous]

        if (
            not starters[previous]
            and
            np.isfinite(q)
        ):
            return (
                float(q),
                "past_relief_snapshot",
            )

        previous -= 1

    return (
        fallback,
        "train_fallback_no_relief_quality",
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
        f"Indexed pitchers : {len(profile_index):,}"
    )

    print(
        f"History rows     : {len(full_history):,}"
    )

    return profile_index


# =============================================================================
# 9. B0 FEATURE PROCESSOR
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

    return processor


# =============================================================================
# 10. B2 QUALITY EXTENSION
# =============================================================================

class QualityFeatureExtension:
    """
    B2 = B0 + standardized quality_p

    이번 Step에서는 quality_p main effect 하나만 추가한다.

    LI 없음.
    LI x quality interaction 없음.
    quality x context interaction 없음.
    """

    def __init__(
        self,
        b0_processor,
    ):
        self.b0_processor = (
            b0_processor
        )

        self.quality_mean = None
        self.quality_std = None

    def fit(
        self,
        train: pd.DataFrame,
    ) -> None:

        quality = pd.to_numeric(
            train["quality_p_model"],
            errors="raise",
        ).to_numpy(
            dtype=float
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
                self.quality_std
            )
            or
            self.quality_std <= 0
        ):
            raise ValueError(
                "Invalid quality_p Train std."
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
        X0 = self.b0_processor.transform_frame(
            frame
        )

        quality = pd.to_numeric(
            frame["quality_p_model"],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        quality_z = (
            quality
            -
            self.quality_mean
        ) / self.quality_std

        X2 = np.column_stack(
            [
                X0,
                quality_z,
            ]
        )

        return (
            X0,
            X2,
        )


# =============================================================================
# 11. RIDGE MODEL
# =============================================================================

def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    dim = X.shape[1]

    A = (
        X.T @ X
        +
        RIDGE
        *
        np.eye(dim)
    )

    b = X.T @ y

    return np.linalg.solve(
        A,
        b,
    )


def fit_b0_b2(
    train: pd.DataFrame,
    extension: QualityFeatureExtension,
):
    section(
        "9. FIT B0 AND B2(+QUALITY_P)"
    )

    (
        X0,
        X2,
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

    theta2 = fit_ridge(
        X2,
        y,
    )

    print(
        f"Ridge          : {RIDGE}"
    )

    print(
        f"B0 dimensions  : {X0.shape[1]}"
    )

    print(
        f"B2 dimensions  : {X2.shape[1]}"
    )

    print(
        f"Added features : "
        f"{X2.shape[1] - X0.shape[1]}"
    )

    print()
    print(
        "B2 added feature:"
    )

    print(
        "  quality_p"
    )

    print()
    print(
        f"quality_p coefficient : "
        f"{theta2[-1]:+.6f}"
    )

    return (
        theta0,
        theta2,
    )


# =============================================================================
# 12. REWARD PREDICTION
# =============================================================================

def prediction_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict:

    y = np.asarray(
        y,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = (
        y
        -
        prediction
    )

    mae = float(
        np.mean(
            np.abs(error)
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
        pd.Series(y).corr(
            pd.Series(prediction),
            method="pearson",
        )
    )

    spearman = float(
        pd.Series(y).corr(
            pd.Series(prediction),
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
    extension: QualityFeatureExtension,
    theta0: np.ndarray,
    theta2: np.ndarray,
):
    (
        X0,
        X2,
    ) = extension.transform(
        frame
    )

    y = pd.to_numeric(
        frame["reward"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    pred0 = X0 @ theta0
    pred2 = X2 @ theta2

    b0 = prediction_metrics(
        y,
        pred0,
    )

    b2 = prediction_metrics(
        y,
        pred2,
    )

    print()
    print(
        f"{split_name} reward prediction"
    )

    print(
        f"{'Metric':14s}"
        f"{'B0':>12s}"
        f"{'B2(+Q)':>12s}"
        f"{'Delta':>12s}"
    )

    for metric in [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    ]:
        delta = (
            b2[metric]
            -
            b0[metric]
        )

        print(
            f"{metric:14s}"
            f"{b0[metric]:12.6f}"
            f"{b2[metric]:12.6f}"
            f"{delta:+12.6f}"
        )

    result = []

    for model, values in [
        (
            "B0",
            b0,
        ),
        (
            "B2_quality",
            b2,
        ),
    ]:
        result.append(
            {
                "split":
                    split_name.lower(),

                "model":
                    model,

                **values,
            }
        )

    return result


# =============================================================================
# 13. CANDIDATE FEATURES
# =============================================================================

def build_candidate_frame(
    decision: pd.Series,
    candidate_ids,
    profile_index,
    quality_index,
    quality_fallback: float,
):
    feature_rows = []
    metadata = []

    for pitcher_id in candidate_ids:
        (
            profile,
            profile_source,
        ) = profile_fix.get_asof_profile(
            decision,
            pitcher_id,
            profile_index,
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
        ) = get_candidate_quality(
            decision,
            pitcher_id,
            quality_index,
            quality_fallback,
        )

        feature = dict(
            values
        )

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
            len(pitcher_ids)
        ),
        key=lambda i: (
            -float(scores[i]),
            str(pitcher_ids[i]),
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
# 15. ACTUAL BULLPEN RANKING
# =============================================================================

def evaluate_actual_bullpen(
    split_name: str,
    frame: pd.DataFrame,
    bullpen_map,
    used_before_map,
    profile_index,
    quality_index,
    quality_fallback: float,
    extension: QualityFeatureExtension,
    theta0: np.ndarray,
    theta2: np.ndarray,
):
    section(
        f"10. ACTUAL BULLPEN EVALUATION - {split_name}"
    )

    work = frame[
        frame["bullpen_source"]
        ==
        "preview"
    ].copy()

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

    source_counter = Counter()

    decision_id = 0

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
            decision["pitcher_id"]
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
            source_counter[
                source
            ] += 1

        (
            X0,
            X2,
        ) = extension.transform(
            candidate_frame
        )

        q0 = X0 @ theta0
        q2 = X2 @ theta2

        (
            rec0,
            ranks0,
        ) = rank_candidates(
            candidate_ids,
            q0,
        )

        (
            rec2,
            ranks2,
        ) = rank_candidates(
            candidate_ids,
            q2,
        )

        actual_rank0 = int(
            ranks0[
                actual_pitcher
            ]
        )

        actual_rank2 = int(
            ranks2[
                actual_pitcher
            ]
        )

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

                "n_candidates":
                    len(
                        candidate_ids
                    ),

                "b0_rec":
                    rec0,

                "b2_rec":
                    rec2,

                "b0_agreement":
                    int(
                        rec0
                        ==
                        actual_pitcher
                    ),

                "b2_agreement":
                    int(
                        rec2
                        ==
                        actual_pitcher
                    ),

                "b0_rank":
                    actual_rank0,

                "b2_rank":
                    actual_rank2,
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

                    "profile_source":
                        metadata.iloc[index][
                            "profile_source"
                        ],

                    "quality_source":
                        metadata.iloc[index][
                            "quality_source"
                        ],

                    "quality_p":
                        float(
                            metadata.iloc[index][
                                "quality_p"
                            ]
                        ),

                    "q_b0":
                        float(
                            q0[index]
                        ),

                    "q_b2_quality":
                        float(
                            q2[index]
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
        source_counter.values()
    )

    for (
        source,
        count,
    ) in source_counter.most_common():

        ratio = (
            count
            /
            total
            *
            100
        )

        print(
            f"  {source:34s}"
            f"{count:8,d} "
            f"({ratio:6.2f}%)"
        )

    return (
        decision_result,
        candidate_result,
        source_counter,
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


def print_ranking_comparison(
    split_name: str,
    frame: pd.DataFrame,
):
    b0 = ranking_metrics(
        frame,
        "b0",
    )

    b2 = ranking_metrics(
        frame,
        "b2",
    )

    print()
    print(
        f"{split_name} B0 vs B2"
    )

    print(
        f"{'Metric':18s}"
        f"{'B0':>12s}"
        f"{'B2(+Q)':>12s}"
        f"{'Delta':>12s}"
    )

    for metric in [
        "agreement",
        "top3",
        "top5",
    ]:
        print(
            f"{metric:18s}"
            f"{b0[metric] * 100:11.2f}%"
            f"{b2[metric] * 100:11.2f}%"
            f"{(b2[metric] - b0[metric]) * 100:+11.2f}p"
        )

    for metric in [
        "mean_rank",
        "median_rank",
    ]:
        print(
            f"{metric:18s}"
            f"{b0[metric]:12.2f}"
            f"{b2[metric]:12.2f}"
            f"{b2[metric] - b0[metric]:+12.2f}"
        )

    return (
        b0,
        b2,
    )


# =============================================================================
# 17. B0 REPRODUCTION
# =============================================================================

def check_b0_reproduction(
    validation_b0: dict,
    test_b0: dict,
) -> bool:

    section(
        "11. B0 REPRODUCTION CHECK"
    )

    validation_actual = (
        validation_b0[
            "agreement"
        ]
    )

    test_actual = (
        test_b0[
            "agreement"
        ]
    )

    validation_diff = abs(
        validation_actual
        -
        EXPECTED_B0_VALIDATION_AGREEMENT
    )

    test_diff = abs(
        test_actual
        -
        EXPECTED_B0_TEST_AGREEMENT
    )

    print(
        f"Expected Validation B0 : "
        f"{EXPECTED_B0_VALIDATION_AGREEMENT * 100:.2f}%"
    )

    print(
        f"Current Validation B0  : "
        f"{validation_actual * 100:.2f}%"
    )

    print()
    print(
        f"Expected Test B0       : "
        f"{EXPECTED_B0_TEST_AGREEMENT * 100:.2f}%"
    )

    print(
        f"Current Test B0        : "
        f"{test_actual * 100:.2f}%"
    )

    passed = (
        validation_diff
        <=
        B0_REPRO_TOLERANCE
        and
        test_diff
        <=
        B0_REPRO_TOLERANCE
    )

    print()
    print(
        "B0 reproduction : "
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
        "=" * 92
    )

    print(
        "STEP 3 - B0 + QUALITY_P ABLATION"
    )

    print()

    print(
        "B0:"
    )

    print(
        "  Frozen pre-quality Contextual Bandit"
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
        "This step DOES NOT use:"
    )

    print(
        "  leverage_index"
    )

    print(
        "  LI x quality_p interaction"
    )

    print()

    print(
        "Everything else remains unchanged:"
    )

    print(
        "  reward / candidate set / pitcher profile / split / alpha / ridge"
    )

    print(
        "=" * 92
    )

    # =========================================================================
    # Data
    # =========================================================================

    quality_pa = (
        load_quality_data()
    )

    (
        pa,
        history,
        bullpen,
        decisions,
        first_app,
    ) = load_b0_data()

    verify_quality_alignment(
        pa,
        quality_pa,
    )

    decisions = (
        attach_logged_quality(
            decisions,
            quality_pa,
        )
    )

    (
        train,
        validation,
        test,
        quality_fallback,
    ) = split_and_prepare_quality(
        decisions
    )

    # =========================================================================
    # Candidate support
    # =========================================================================

    quality_index = (
        build_quality_snapshot_index(
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
    # B0 / B2
    # =========================================================================

    b0_processor = (
        fit_b0_processor(
            train
        )
    )

    section(
        "9. FIT QUALITY FEATURE EXTENSION"
    )

    extension = QualityFeatureExtension(
        b0_processor
    )

    extension.fit(
        train
    )

    (
        theta0,
        theta2,
    ) = fit_b0_b2(
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
                theta2,
            )
        )

    reward_summary = pd.DataFrame(
        reward_rows
    )

    # =========================================================================
    # Candidate ranking
    # =========================================================================

    (
        validation_result,
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
        theta2,
    )

    (
        test_result,
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
        theta2,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    section(
        "B0 VS B2 RANKING SUMMARY"
    )

    (
        validation_b0,
        validation_b2,
    ) = print_ranking_comparison(
        "VALIDATION",
        validation_result,
    )

    (
        test_b0,
        test_b2,
    ) = print_ranking_comparison(
        "TEST",
        test_result,
    )

    b0_reproduction = (
        check_b0_reproduction(
            validation_b0,
            test_b0,
        )
    )

    ranking_rows = []

    for (
        split,
        model,
        metrics,
    ) in [
        (
            "validation",
            "B0",
            validation_b0,
        ),
        (
            "validation",
            "B2_quality",
            validation_b2,
        ),
        (
            "test",
            "B0",
            test_b0,
        ),
        (
            "test",
            "B2_quality",
            test_b2,
        ),
    ]:
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
        split_name,
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
                        split_name,

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

    validation_result.to_csv(
        OUTPUT_DIR
        /
        "validation_decisions.csv",
        index=False,
    )

    test_result.to_csv(
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

    np.save(
        OUTPUT_DIR
        /
        "theta_b0.npy",
        theta0,
    )

    np.save(
        OUTPUT_DIR
        /
        "theta_b2_quality.npy",
        theta2,
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
        "B0 reproduction : "
        +
        (
            "PASS"
            if b0_reproduction
            else "WARNING"
        )
    )

    print()

    print(
        "TEST"
    )

    print(
        f"B0 Agreement : "
        f"{test_b0['agreement'] * 100:.2f}%"
    )

    print(
        f"B2 Agreement : "
        f"{test_b2['agreement'] * 100:.2f}%"
    )

    print(
        f"Delta        : "
        f"{(test_b2['agreement'] - test_b0['agreement']) * 100:+.2f}p"
    )

    print()

    print(
        f"B0 Top-3     : "
        f"{test_b0['top3'] * 100:.2f}%"
    )

    print(
        f"B2 Top-3     : "
        f"{test_b2['top3'] * 100:.2f}%"
    )

    print()

    print(
        f"B0 Top-5     : "
        f"{test_b0['top5'] * 100:.2f}%"
    )

    print(
        f"B2 Top-5     : "
        f"{test_b2['top5'] * 100:.2f}%"
    )

    print()

    print(
        f"B0 Mean Rank : "
        f"{test_b0['mean_rank']:.2f}"
    )

    print(
        f"B2 Mean Rank : "
        f"{test_b2['mean_rank']:.2f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Agreement alone does not determine policy quality."
    )

    print(
        "If B0 reproduction passes, "
        "the next step is B0 vs B2 OPE."
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()