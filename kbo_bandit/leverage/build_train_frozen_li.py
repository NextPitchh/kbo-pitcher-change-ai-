from pathlib import Path
import json

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as bandit_base

from data_analysis.methods.leverage_index import (
    compute_li_raw,
)


# ============================================================
# Config
# ============================================================

INPUT_CANDIDATES = [
    Path(
        "data_analysis/results/all_pa_with_li.parquet"
    ),
    Path(
        "all_pa_with_li.parquet"
    ),
]

OUTPUT_DIR = Path(
    "data_analysis/results"
)

OUTPUT_PA = (
    OUTPUT_DIR
    / "all_pa_with_li_train_frozen.parquet"
)

OUTPUT_DIST = (
    OUTPUT_DIR
    / "league_pa_distribution_train_frozen.json"
)

OUTPUT_BASELINE = (
    OUTPUT_DIR
    / "li_baseline_train_frozen.json"
)


PA_RESULTS = [
    "1B",
    "2B",
    "3B",
    "HR",
    "BB",
    "SO",
    "OUT",
    "GDP",
    "SF",
]


BASE_STATES = [
    "0",
    "1",
    "2",
    "3",
    "12",
    "13",
    "23",
    "123",
]


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def resolve_input_path():

    for path in INPUT_CANDIDATES:

        if path.exists():

            return path

    raise FileNotFoundError(
        "all_pa_with_li.parquet을 찾을 수 없습니다."
    )


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


def normalize_base_state(x):

    if pd.isna(x):

        raise ValueError(
            "base_state contains NaN"
        )

    if isinstance(
        x,
        (
            int,
            np.integer,
        ),
    ):

        s = str(
            int(x)
        )

    elif isinstance(
        x,
        (
            float,
            np.floating,
        ),
    ):

        if not float(
            x
        ).is_integer():

            raise ValueError(
                f"Invalid base_state: {x}"
            )

        s = str(
            int(x)
        )

    else:

        s = str(
            x
        ).strip()

        if s.endswith(
            ".0"
        ):

            s = s[:-2]

    if s not in BASE_STATES:

        raise ValueError(
            f"Unknown base_state: {x} -> {s}"
        )

    return s


def standard_clean_mask(df):

    quality = (
        df[
            "data_quality_flag"
        ]
        .fillna(
            "__NA__"
        )
        .astype(str)
    )

    result = (
        df[
            "pa_result"
        ]
        .astype(str)
    )

    mask = (
        (
            quality
            == ""
        )
        &
        (
            result
            != "UNK"
        )
        &
        (
            result.isin(
                PA_RESULTS
            )
        )
    )

    return mask


def describe_series(series):

    s = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    return {
        "count":
            int(
                len(s)
            ),

        "mean":
            float(
                s.mean()
            ),

        "std":
            float(
                s.std()
            ),

        "min":
            float(
                s.min()
            ),

        "p25":
            float(
                s.quantile(
                    0.25
                )
            ),

        "p50":
            float(
                s.quantile(
                    0.50
                )
            ),

        "p75":
            float(
                s.quantile(
                    0.75
                )
            ),

        "p95":
            float(
                s.quantile(
                    0.95
                )
            ),

        "p99":
            float(
                s.quantile(
                    0.99
                )
            ),

        "max":
            float(
                s.max()
            ),
    }


# ============================================================
# 1. Load LI dataset
# ============================================================

def load_li_data():

    print_section(
        "1. LOAD EXISTING LI DATA"
    )

    path = (
        resolve_input_path()
    )

    df = pd.read_parquet(
        path
    )

    print(
        "Input :",
        path,
    )

    print(
        "Shape :",
        df.shape,
    )

    required = [
        "game_id",
        "inning",
        "half",
        "out_count",
        "base_state",
        "score_diff_attacker",
        "pa_result",
        "data_quality_flag",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # 기존 full-data LI 보존
    if (
        "leverage_index"
        in df.columns
    ):

        df = (
            df.rename(
                columns={
                    "leverage_index":
                        "leverage_index_full_reference"
                }
            )
        )

        print(
            "Existing leverage_index preserved as:"
        )

        print(
            "  leverage_index_full_reference"
        )

    return df


# ============================================================
# 2. Resolve exact Bandit split
# ============================================================

def resolve_bandit_split():

    print_section(
        "2. RESOLVE BANDIT CHRONOLOGICAL SPLIT"
    )

    bandit_base.REWARD_MODE = (
        "wpa_per_pa"
    )

    bandit_base.ALPHA = (
        0.0
    )

    (
        base_pa,
        history,
        bullpen,
    ) = (
        bandit_base.load_data()
    )

    (
        decisions,
        first_app,
    ) = (
        bandit_base.build_decision_dataset(
            base_pa,
            history,
        )
    )

    (
        train,
        val,
        test,
    ) = (
        bandit_base.chronological_split(
            decisions
        )
    )

    train_games = set(
        train[
            "game_id"
        ]
        .apply(
            normalize_id
        )
    )

    val_games = set(
        val[
            "game_id"
        ]
        .apply(
            normalize_id
        )
    )

    test_games = set(
        test[
            "game_id"
        ]
        .apply(
            normalize_id
        )
    )

    if (
        train_games
        &
        val_games
    ):

        raise ValueError(
            "Train/Val game overlap"
        )

    if (
        train_games
        &
        test_games
    ):

        raise ValueError(
            "Train/Test game overlap"
        )

    if (
        val_games
        &
        test_games
    ):

        raise ValueError(
            "Val/Test game overlap"
        )

    print(
        f"Train decisions : "
        f"{len(train):,}"
    )

    print(
        f"Val decisions   : "
        f"{len(val):,}"
    )

    print(
        f"Test decisions  : "
        f"{len(test):,}"
    )

    print()

    print(
        f"Train games     : "
        f"{len(train_games):,}"
    )

    print(
        f"Val games       : "
        f"{len(val_games):,}"
    )

    print(
        f"Test games      : "
        f"{len(test_games):,}"
    )

    print()

    print(
        "Train dates:"
    )

    print(
        " ",
        train[
            "date"
        ].min(),
        "~",
        train[
            "date"
        ].max(),
    )

    print(
        "Val dates:"
    )

    print(
        " ",
        val[
            "date"
        ].min(),
        "~",
        val[
            "date"
        ].max(),
    )

    print(
        "Test dates:"
    )

    print(
        " ",
        test[
            "date"
        ].min(),
        "~",
        test[
            "date"
        ].max(),
    )

    return (
        train_games,
        val_games,
        test_games,
    )


# ============================================================
# 3. Build TRAIN-only P_league
# ============================================================

def build_train_distribution(
    df,
    train_games,
):

    print_section(
        "3. BUILD TRAIN-ONLY P_LEAGUE(r)"
    )

    game_key = (
        df[
            "game_id"
        ]
        .apply(
            normalize_id
        )
    )

    clean = (
        standard_clean_mask(
            df
        )
    )

    is_train = (
        game_key.isin(
            train_games
        )
    )

    train_df = (
        df[
            clean
            &
            is_train
        ]
        .copy()
    )

    train_df[
        "_out"
    ] = (
        pd.to_numeric(
            train_df[
                "out_count"
            ],
            errors="raise",
        )
        .astype(int)
    )

    train_df[
        "_base"
    ] = (
        train_df[
            "base_state"
        ]
        .apply(
            normalize_base_state
        )
    )

    print(
        f"Train clean PA : "
        f"{len(train_df):,}"
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_counts = (
        train_df[
            "pa_result"
        ]
        .value_counts()
        .reindex(
            PA_RESULTS,
            fill_value=0,
        )
    )

    overall_total = int(
        overall_counts.sum()
    )

    overall = {
        result:
            float(
                overall_counts[
                    result
                ]
                /
                overall_total
            )
        for result
        in PA_RESULTS
    }

    # --------------------------------------------------------
    # out_count × base_state conditional distribution
    # --------------------------------------------------------

    table = {}

    bucket_sizes = {}

    for out_count in [
        0,
        1,
        2,
    ]:

        table[
            str(
                out_count
            )
        ] = {}

        for base_state in (
            BASE_STATES
        ):

            bucket = (
                train_df[
                    (
                        train_df[
                            "_out"
                        ]
                        == out_count
                    )
                    &
                    (
                        train_df[
                            "_base"
                        ]
                        == base_state
                    )
                ]
            )

            n = int(
                len(
                    bucket
                )
            )

            bucket_key = (
                f"{out_count}_"
                f"{base_state}"
            )

            bucket_sizes[
                bucket_key
            ] = (
                n
            )

            if n == 0:

                raise ValueError(
                    "Empty TRAIN LI bucket: "
                    f"{bucket_key}"
                )

            counts = (
                bucket[
                    "pa_result"
                ]
                .value_counts()
                .reindex(
                    PA_RESULTS,
                    fill_value=0,
                )
            )

            probs = {
                result:
                    float(
                        counts[
                            result
                        ]
                        /
                        n
                    )
                for result
                in PA_RESULTS
            }

            prob_sum = sum(
                probs.values()
            )

            if not np.isclose(
                prob_sum,
                1.0,
                atol=1e-10,
            ):

                raise ValueError(
                    "Probability sum != 1 "
                    f"for {bucket_key}"
                )

            table[
                str(
                    out_count
                )
            ][
                base_state
            ] = (
                probs
            )

    print(
        f"Conditional buckets : "
        f"{len(bucket_sizes)}"
    )

    print(
        f"Minimum bucket size : "
        f"{min(bucket_sizes.values()):,}"
    )

    print()

    print(
        "TRAIN overall PA distribution:"
    )

    for result in (
        PA_RESULTS
    ):

        print(
            f"  {result:>3s}: "
            f"{overall[result] * 100:.3f}%"
        )

    distribution = {
        "source":
            "Bandit chronological TRAIN only",

        "conditioning_vars": [
            "out_count",
            "base_state",
        ],

        "pa_results":
            PA_RESULTS,

        "n_train_clean_pa":
            int(
                len(
                    train_df
                )
            ),

        "overall":
            overall,

        "bucket_sizes":
            bucket_sizes,

        "min_bucket_size":
            int(
                min(
                    bucket_sizes.values()
                )
            ),

        "table":
            table,
    }

    return distribution


# ============================================================
# 4. Build LI lookup
# ============================================================

def build_li_lookup(
    distribution,
):

    print_section(
        "4. BUILD TRAIN-FROZEN LI_RAW LOOKUP"
    )

    print(
        "NOTE:"
    )

    print(
        "  이번 단계에서는 기존 LI 구현과의 "
        "비교를 위해 -4~+4 score lookup 방식을 유지합니다."
    )

    rows = []

    total = (
        9
        *
        2
        *
        3
        *
        8
        *
        9
    )

    count = 0

    for inning in range(
        1,
        10,
    ):

        for half in (
            "top",
            "bot",
        ):

            for out_count in (
                0,
                1,
                2,
            ):

                for base_state in (
                    BASE_STATES
                ):

                    league_dist = (
                        distribution[
                            "table"
                        ][
                            str(
                                out_count
                            )
                        ][
                            base_state
                        ]
                    )

                    for score_diff in range(
                        -4,
                        5,
                    ):

                        li_raw = (
                            compute_li_raw(
                                inning=inning,
                                half=half,
                                out_count=out_count,
                                base_state=base_state,
                                score_diff_attacker=score_diff,
                                league_dist=league_dist,
                            )
                        )

                        if (
                            not np.isfinite(
                                li_raw
                            )
                        ):

                            raise ValueError(
                                "Non-finite LI_RAW"
                            )

                        if (
                            li_raw
                            <
                            0
                        ):

                            raise ValueError(
                                "Negative LI_RAW"
                            )

                        rows.append(
                            {
                                "_inning":
                                    int(
                                        inning
                                    ),

                                "_half":
                                    half,

                                "_out":
                                    int(
                                        out_count
                                    ),

                                "_base":
                                    base_state,

                                "_score_clip":
                                    int(
                                        score_diff
                                    ),

                                "_li_raw":
                                    float(
                                        li_raw
                                    ),
                            }
                        )

                        count += 1

    lookup = pd.DataFrame(
        rows
    )

    print(
        f"Lookup combinations : "
        f"{len(lookup):,}"
    )

    print(
        f"Expected            : "
        f"{total:,}"
    )

    if (
        len(lookup)
        != total
    ):

        raise ValueError(
            "Unexpected lookup size"
        )

    return lookup


# ============================================================
# 5. Apply lookup to full PA data
# ============================================================

def apply_lookup(
    df,
    lookup,
):

    print_section(
        "5. APPLY TRAIN-FROZEN LI_RAW TO ALL PA"
    )

    work = (
        df.copy()
    )

    work[
        "_row_order"
    ] = np.arange(
        len(
            work
        )
    )

    work[
        "_inning"
    ] = (
        pd.to_numeric(
            work[
                "inning"
            ],
            errors="raise",
        )
        .astype(int)
    )

    work[
        "_half"
    ] = (
        work[
            "half"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    work[
        "_out"
    ] = (
        pd.to_numeric(
            work[
                "out_count"
            ],
            errors="raise",
        )
        .astype(int)
    )

    work[
        "_base"
    ] = (
        work[
            "base_state"
        ]
        .apply(
            normalize_base_state
        )
    )

    score = (
        pd.to_numeric(
            work[
                "score_diff_attacker"
            ],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # 기존 희민님 LI 방식 유지
    #
    # clipping 문제는 팀원 확인 대상으로 두고
    # 이번 Train-frozen 실험에서는 계산법을 변경하지 않는다.
    # --------------------------------------------------------

    work[
        "_score_clip"
    ] = np.clip(
        score,
        -4,
        4,
    )

    work = (
        work.merge(
            lookup,
            on=[
                "_inning",
                "_half",
                "_out",
                "_base",
                "_score_clip",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    missing = int(
        work[
            "_li_raw"
        ]
        .isna()
        .sum()
    )

    print(
        f"Rows           : "
        f"{len(work):,}"
    )

    print(
        f"Missing LI_RAW : "
        f"{missing:,}"
    )

    if missing > 0:

        bad = (
            work[
                work[
                    "_li_raw"
                ]
                .isna()
            ][
                [
                    "inning",
                    "half",
                    "out_count",
                    "base_state",
                    "score_diff_attacker",
                ]
            ]
            .head(20)
        )

        print(
            bad.to_string(
                index=False
            )
        )

        raise ValueError(
            "LI lookup merge failed."
        )

    work = (
        work.sort_values(
            "_row_order"
        )
        .reset_index(
            drop=True
        )
    )

    return work


# ============================================================
# 6. Train-only normalization
# ============================================================

def normalize_li(
    work,
    train_games,
):

    print_section(
        "6. TRAIN-ONLY LI NORMALIZATION"
    )

    game_key = (
        work[
            "game_id"
        ]
        .apply(
            normalize_id
        )
    )

    clean = (
        standard_clean_mask(
            work
        )
    )

    is_train = (
        game_key.isin(
            train_games
        )
    )

    normalization_mask = (
        clean
        &
        is_train
    )

    train_raw = (
        work.loc[
            normalization_mask,
            "_li_raw",
        ]
    )

    li_league_avg_train = float(
        train_raw.mean()
    )

    if (
        not np.isfinite(
            li_league_avg_train
        )
        or
        li_league_avg_train
        <= 0
    ):

        raise ValueError(
            "Invalid TRAIN LI average."
        )

    work[
        "leverage_index_raw_train_frozen"
    ] = (
        work[
            "_li_raw"
        ]
    )

    work[
        "leverage_index"
    ] = (
        work[
            "_li_raw"
        ]
        /
        li_league_avg_train
    )

    train_normalized_mean = float(
        work.loc[
            normalization_mask,
            "leverage_index",
        ]
        .mean()
    )

    print(
        f"Train normalization PA : "
        f"{len(train_raw):,}"
    )

    print(
        f"li_league_avg_train    : "
        f"{li_league_avg_train:.10f}"
    )

    print(
        f"Train normalized mean  : "
        f"{train_normalized_mean:.10f}"
    )

    if not np.isclose(
        train_normalized_mean,
        1.0,
        atol=1e-10,
    ):

        raise ValueError(
            "TRAIN normalized LI mean != 1"
        )

    return (
        work,
        li_league_avg_train,
        normalization_mask,
    )


# ============================================================
# 7. Diagnostics
# ============================================================

def run_diagnostics(
    work,
    normalization_mask,
    li_league_avg_train,
    distribution,
):

    print_section(
        "7. DIAGNOSTICS"
    )

    train_distribution = (
        describe_series(
            work.loc[
                normalization_mask,
                "leverage_index",
            ]
        )
    )

    all_distribution = (
        describe_series(
            work[
                "leverage_index"
            ]
        )
    )

    print(
        "TRAIN normalized LI:"
    )

    for key, value in (
        train_distribution.items()
    ):

        if key == "count":

            print(
                f"  {key:>5s}: "
                f"{int(value):,}"
            )

        else:

            print(
                f"  {key:>5s}: "
                f"{value:.4f}"
            )

    print()

    print(
        "ALL PA LI using frozen Train baseline:"
    )

    for key, value in (
        all_distribution.items()
    ):

        if key == "count":

            print(
                f"  {key:>5s}: "
                f"{int(value):,}"
            )

        else:

            print(
                f"  {key:>5s}: "
                f"{value:.4f}"
            )

    old_vs_new = None

    if (
        "leverage_index_full_reference"
        in work.columns
    ):

        old = (
            pd.to_numeric(
                work[
                    "leverage_index_full_reference"
                ],
                errors="coerce",
            )
        )

        new = (
            work[
                "leverage_index"
            ]
        )

        valid = (
            old.notna()
            &
            new.notna()
            &
            np.isfinite(
                old
            )
            &
            np.isfinite(
                new
            )
        )

        diff = (
            new[
                valid
            ]
            -
            old[
                valid
            ]
        )

        mae = float(
            diff.abs().mean()
        )

        max_abs_diff = float(
            diff.abs().max()
        )

        pearson = float(
            np.corrcoef(
                old[
                    valid
                ],
                new[
                    valid
                ],
            )[0, 1]
        )

        old_vs_new = {
            "n":
                int(
                    valid.sum()
                ),

            "mae":
                mae,

            "max_abs_diff":
                max_abs_diff,

            "pearson":
                pearson,

            "fraction_abs_diff_gt_0_01":
                float(
                    (
                        diff.abs()
                        >
                        0.01
                    )
                    .mean()
                ),

            "fraction_abs_diff_gt_0_10":
                float(
                    (
                        diff.abs()
                        >
                        0.10
                    )
                    .mean()
                ),
        }

        print()

        print(
            "Original full-data LI vs "
            "Train-frozen LI:"
        )

        print(
            f"  MAE            : "
            f"{mae:.6f}"
        )

        print(
            f"  Max abs diff   : "
            f"{max_abs_diff:.6f}"
        )

        print(
            f"  Pearson        : "
            f"{pearson:.6f}"
        )

        print(
            f"  |diff| > .01  : "
            f"{old_vs_new['fraction_abs_diff_gt_0_01'] * 100:.2f}%"
        )

        print(
            f"  |diff| > .10  : "
            f"{old_vs_new['fraction_abs_diff_gt_0_10'] * 100:.2f}%"
        )

    metadata = {
        "purpose":
            "Bandit chronological evaluation",

        "method":
            (
                "Original LI calculation method "
                "with TRAIN-only P_league and "
                "TRAIN-only normalization."
            ),

        "clipping_policy":
            (
                "Original -4~+4 lookup method preserved. "
                "Clipping correctness is tracked separately."
            ),

        "n_train_clean_pa":
            distribution[
                "n_train_clean_pa"
            ],

        "min_train_bucket_size":
            distribution[
                "min_bucket_size"
            ],

        "li_league_avg_train":
            float(
                li_league_avg_train
            ),

        "train_distribution":
            train_distribution,

        "all_distribution":
            all_distribution,

        "old_full_vs_train_frozen":
            old_vs_new,
    }

    return metadata


# ============================================================
# 8. Save
# ============================================================

def save_outputs(
    work,
    distribution,
    metadata,
):

    print_section(
        "8. SAVE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    internal_cols = [
        "_row_order",
        "_inning",
        "_half",
        "_out",
        "_base",
        "_score_clip",
        "_li_raw",
    ]

    output = (
        work.drop(
            columns=[
                col
                for col in internal_cols
                if col in work.columns
            ]
        )
    )

    output.to_parquet(
        OUTPUT_PA,
        index=False,
    )

    with open(
        OUTPUT_DIST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            distribution,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        OUTPUT_BASELINE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Saved PA:"
    )

    print(
        " ",
        OUTPUT_PA,
    )

    print()

    print(
        "Saved TRAIN league distribution:"
    )

    print(
        " ",
        OUTPUT_DIST,
    )

    print()

    print(
        "Saved TRAIN LI baseline:"
    )

    print(
        " ",
        OUTPUT_BASELINE,
    )

    print()

    print(
        "Output shape:",
        output.shape,
    )

    print()

    print(
        "Bandit에서 사용할 컬럼:"
    )

    print(
        "  leverage_index"
    )

    print()

    print(
        "기존 LI reference:"
    )

    print(
        "  leverage_index_full_reference"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "STEP 13-1 - TRAIN-FROZEN LEVERAGE INDEX"
    )

    print()

    print(
        "This script changes ONLY:"
    )

    print(
        "  1. P_league source -> Bandit TRAIN only"
    )

    print(
        "  2. LI normalization -> Bandit TRAIN only"
    )

    print()

    print(
        "The existing -4~+4 LI lookup method "
        "is intentionally preserved."
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = (
        load_li_data()
    )

    # --------------------------------------------------------
    # Bandit split
    # --------------------------------------------------------

    (
        train_games,
        val_games,
        test_games,
    ) = (
        resolve_bandit_split()
    )

    # --------------------------------------------------------
    # TRAIN P_league
    # --------------------------------------------------------

    distribution = (
        build_train_distribution(
            df,
            train_games,
        )
    )

    # --------------------------------------------------------
    # LI lookup
    # --------------------------------------------------------

    lookup = (
        build_li_lookup(
            distribution
        )
    )

    # --------------------------------------------------------
    # Apply
    # --------------------------------------------------------

    work = (
        apply_lookup(
            df,
            lookup,
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    (
        work,
        li_league_avg_train,
        normalization_mask,
    ) = (
        normalize_li(
            work,
            train_games,
        )
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    metadata = (
        run_diagnostics(
            work,
            normalization_mask,
            li_league_avg_train,
            distribution,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        work,
        distribution,
        metadata,
    )

    print_section(
        "DONE"
    )


if __name__ == "__main__":
    main()
