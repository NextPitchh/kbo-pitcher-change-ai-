from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

INPUT_DIR = Path(
    "outputs/bandit/leverage/v3_li_ablation"
)

OUTPUT_DIR = Path(
    "outputs/bandit/leverage/v3_li_segments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LI_BINS = [
    -np.inf,
    0.5,
    1.5,
    2.5,
    np.inf,
]

LI_LABELS = [
    "low_<0.5",
    "normal_0.5-1.5",
    "high_1.5-2.5",
    "critical_>=2.5",
]


# ============================================================
# Helper
# ============================================================

def section(title):

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def load_decisions(split):

    path = (
        INPUT_DIR
        / f"{split}_decisions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find: {path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "decision_id",
        "game_id",
        "n_candidates",
        "leverage_index",

        "b0_rec",
        "b1_rec",

        "b0_agreement",
        "b1_agreement",

        "b0_rank",
        "b1_rank",
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

    numeric_cols = [
        "n_candidates",
        "leverage_index",
        "b0_agreement",
        "b1_agreement",
        "b0_rank",
        "b1_rank",
    ]

    for col in numeric_cols:

        df[col] = (
            pd.to_numeric(
                df[col],
                errors="raise",
            )
        )

    return df


# ============================================================
# LI buckets
# ============================================================

def add_li_bucket(df):

    out = df.copy()

    out[
        "li_bucket"
    ] = pd.cut(
        out[
            "leverage_index"
        ],
        bins=LI_BINS,
        labels=LI_LABELS,
        right=False,
    )

    if (
        out[
            "li_bucket"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Some decisions could not be assigned "
            "to an LI bucket."
        )

    return out


# ============================================================
# Metrics
# ============================================================

def model_metrics(
    df,
    prefix,
):

    agreement = float(
        df[
            f"{prefix}_agreement"
        ]
        .mean()
    )

    rank = (
        df[
            f"{prefix}_rank"
        ]
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


# ============================================================
# Recommendation change diagnostics
# ============================================================

def recommendation_change_metrics(
    df,
):

    changed = (
        df[
            "b0_rec"
        ]
        .astype(str)
        !=
        df[
            "b1_rec"
        ]
        .astype(str)
    )

    change_rate = float(
        changed.mean()
    )

    # --------------------------------------------------------
    # B1 win:
    # B0 틀림 + B1 맞음
    # --------------------------------------------------------

    b1_win = (
        (
            df[
                "b0_agreement"
            ]
            == 0
        )
        &
        (
            df[
                "b1_agreement"
            ]
            == 1
        )
    )

    # --------------------------------------------------------
    # B1 loss:
    # B0 맞음 + B1 틀림
    # --------------------------------------------------------

    b1_loss = (
        (
            df[
                "b0_agreement"
            ]
            == 1
        )
        &
        (
            df[
                "b1_agreement"
            ]
            == 0
        )
    )

    # --------------------------------------------------------
    # Rank improvement
    #
    # B1 rank가 작을수록 개선
    # --------------------------------------------------------

    rank_improved = (
        df[
            "b1_rank"
        ]
        <
        df[
            "b0_rank"
        ]
    )

    rank_worsened = (
        df[
            "b1_rank"
        ]
        >
        df[
            "b0_rank"
        ]
    )

    rank_same = (
        df[
            "b1_rank"
        ]
        ==
        df[
            "b0_rank"
        ]
    )

    return {
        "recommendation_change_rate":
            change_rate,

        "b1_win_rate":
            float(
                b1_win.mean()
            ),

        "b1_loss_rate":
            float(
                b1_loss.mean()
            ),

        "net_agreement_gain":
            float(
                b1_win.mean()
                -
                b1_loss.mean()
            ),

        "rank_improved_rate":
            float(
                rank_improved.mean()
            ),

        "rank_worsened_rate":
            float(
                rank_worsened.mean()
            ),

        "rank_same_rate":
            float(
                rank_same.mean()
            ),
    }


# ============================================================
# Analyze one group
# ============================================================

def analyze_group(
    split,
    group_name,
    df,
):

    b0 = (
        model_metrics(
            df,
            "b0",
        )
    )

    b1 = (
        model_metrics(
            df,
            "b1",
        )
    )

    change = (
        recommendation_change_metrics(
            df
        )
    )

    return {
        "split":
            split,

        "li_bucket":
            group_name,

        "n":
            int(
                len(df)
            ),

        "avg_li":
            float(
                df[
                    "leverage_index"
                ]
                .mean()
            ),

        "avg_candidates":
            float(
                df[
                    "n_candidates"
                ]
                .mean()
            ),

        "b0_agreement":
            b0[
                "agreement"
            ],

        "b1_agreement":
            b1[
                "agreement"
            ],

        "agreement_delta":
            (
                b1[
                    "agreement"
                ]
                -
                b0[
                    "agreement"
                ]
            ),

        "b0_top3":
            b0[
                "top3"
            ],

        "b1_top3":
            b1[
                "top3"
            ],

        "top3_delta":
            (
                b1[
                    "top3"
                ]
                -
                b0[
                    "top3"
                ]
            ),

        "b0_top5":
            b0[
                "top5"
            ],

        "b1_top5":
            b1[
                "top5"
            ],

        "top5_delta":
            (
                b1[
                    "top5"
                ]
                -
                b0[
                    "top5"
                ]
            ),

        "b0_mean_rank":
            b0[
                "mean_rank"
            ],

        "b1_mean_rank":
            b1[
                "mean_rank"
            ],

        "mean_rank_delta":
            (
                b1[
                    "mean_rank"
                ]
                -
                b0[
                    "mean_rank"
                ]
            ),

        "b0_median_rank":
            b0[
                "median_rank"
            ],

        "b1_median_rank":
            b1[
                "median_rank"
            ],

        **change,
    }


# ============================================================
# Split analysis
# ============================================================

def analyze_split(
    split,
    df,
):

    section(
        f"LI SEGMENT ANALYSIS - {split.upper()}"
    )

    work = (
        add_li_bucket(
            df
        )
    )

    rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    rows.append(
        analyze_group(
            split,
            "overall",
            work,
        )
    )

    # --------------------------------------------------------
    # LI buckets
    # --------------------------------------------------------

    for label in (
        LI_LABELS
    ):

        segment = (
            work[
                work[
                    "li_bucket"
                ]
                == label
            ]
        )

        if segment.empty:
            continue

        rows.append(
            analyze_group(
                split,
                label,
                segment,
            )
        )

    result = pd.DataFrame(
        rows
    )

    display = (
        result.copy()
    )

    percentage_cols = [
        "b0_agreement",
        "b1_agreement",
        "agreement_delta",

        "b0_top3",
        "b1_top3",
        "top3_delta",

        "b0_top5",
        "b1_top5",
        "top5_delta",

        "recommendation_change_rate",
        "b1_win_rate",
        "b1_loss_rate",
        "net_agreement_gain",

        "rank_improved_rate",
        "rank_worsened_rate",
        "rank_same_rate",
    ]

    for col in percentage_cols:

        display[
            col
        ] *= 100

    print()
    print(
        "Agreement / Ranking"
    )

    print(
        display[
            [
                "li_bucket",
                "n",
                "avg_li",
                "avg_candidates",

                "b0_agreement",
                "b1_agreement",
                "agreement_delta",

                "b0_top3",
                "b1_top3",
                "top3_delta",

                "b0_top5",
                "b1_top5",
                "top5_delta",

                "b0_mean_rank",
                "b1_mean_rank",
                "mean_rank_delta",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "avg_li":
                    lambda x:
                    f"{x:.3f}",

                "avg_candidates":
                    lambda x:
                    f"{x:.2f}",

                "b0_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "b1_agreement":
                    lambda x:
                    f"{x:.2f}%",

                "agreement_delta":
                    lambda x:
                    f"{x:+.2f}p",

                "b0_top3":
                    lambda x:
                    f"{x:.2f}%",

                "b1_top3":
                    lambda x:
                    f"{x:.2f}%",

                "top3_delta":
                    lambda x:
                    f"{x:+.2f}p",

                "b0_top5":
                    lambda x:
                    f"{x:.2f}%",

                "b1_top5":
                    lambda x:
                    f"{x:.2f}%",

                "top5_delta":
                    lambda x:
                    f"{x:+.2f}p",

                "b0_mean_rank":
                    lambda x:
                    f"{x:.2f}",

                "b1_mean_rank":
                    lambda x:
                    f"{x:.2f}",

                "mean_rank_delta":
                    lambda x:
                    f"{x:+.2f}",
            },
        )
    )

    print()
    print(
        "Recommendation / Rank changes"
    )

    print(
        display[
            [
                "li_bucket",
                "n",

                "recommendation_change_rate",

                "b1_win_rate",
                "b1_loss_rate",
                "net_agreement_gain",

                "rank_improved_rate",
                "rank_worsened_rate",
                "rank_same_rate",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "recommendation_change_rate":
                    lambda x:
                    f"{x:.2f}%",

                "b1_win_rate":
                    lambda x:
                    f"{x:.2f}%",

                "b1_loss_rate":
                    lambda x:
                    f"{x:.2f}%",

                "net_agreement_gain":
                    lambda x:
                    f"{x:+.2f}p",

                "rank_improved_rate":
                    lambda x:
                    f"{x:.2f}%",

                "rank_worsened_rate":
                    lambda x:
                    f"{x:.2f}%",

                "rank_same_rate":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    return (
        work,
        result,
    )


# ============================================================
# Simple robustness interpretation
# ============================================================

def print_focus(
    validation_result,
    test_result,
):

    section(
        "HIGH / CRITICAL LEVERAGE FOCUS"
    )

    focus_labels = [
        "high_1.5-2.5",
        "critical_>=2.5",
    ]

    combined = pd.concat(
        [
            validation_result,
            test_result,
        ],
        ignore_index=True,
    )

    focus = (
        combined[
            combined[
                "li_bucket"
            ]
            .isin(
                focus_labels
            )
        ]
        .copy()
    )

    focus[
        "agreement_delta_pp"
    ] = (
        focus[
            "agreement_delta"
        ]
        *
        100
    )

    focus[
        "top3_delta_pp"
    ] = (
        focus[
            "top3_delta"
        ]
        *
        100
    )

    focus[
        "top5_delta_pp"
    ] = (
        focus[
            "top5_delta"
        ]
        *
        100
    )

    print(
        focus[
            [
                "split",
                "li_bucket",
                "n",

                "agreement_delta_pp",
                "top3_delta_pp",
                "top5_delta_pp",

                "mean_rank_delta",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "agreement_delta_pp":
                    lambda x:
                    f"{x:+.2f}p",

                "top3_delta_pp":
                    lambda x:
                    f"{x:+.2f}p",

                "top5_delta_pp":
                    lambda x:
                    f"{x:+.2f}p",

                "mean_rank_delta":
                    lambda x:
                    f"{x:+.2f}",
            },
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 92
    )

    print(
        "STEP 13-4 - LEVERAGE-SEGMENT B0 VS B1 ANALYSIS"
    )

    print()

    print(
        "Goal:"
    )

    print(
        "Identify where leverage_index changes "
        "Bandit ranking performance."
    )

    print()

    print(
        "Segments:"
    )

    print(
        "  Low      : LI < 0.5"
    )

    print(
        "  Normal   : 0.5 <= LI < 1.5"
    )

    print(
        "  High     : 1.5 <= LI < 2.5"
    )

    print(
        "  Critical : LI >= 2.5"
    )

    print(
        "=" * 92
    )

    # ========================================================
    # Load
    # ========================================================

    validation = (
        load_decisions(
            "validation"
        )
    )

    test = (
        load_decisions(
            "test"
        )
    )

    print(
        f"Validation decisions : "
        f"{len(validation):,}"
    )

    print(
        f"Test decisions       : "
        f"{len(test):,}"
    )

    # ========================================================
    # Analyze
    # ========================================================

    (
        validation_segmented,
        validation_result,
    ) = (
        analyze_split(
            "validation",
            validation,
        )
    )

    (
        test_segmented,
        test_result,
    ) = (
        analyze_split(
            "test",
            test,
        )
    )

    # ========================================================
    # Focus
    # ========================================================

    print_focus(
        validation_result,
        test_result,
    )

    # ========================================================
    # Save
    # ========================================================

    section(
        "SAVE"
    )

    validation_result.to_csv(
        OUTPUT_DIR
        / "validation_li_segments.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_li_segments.csv",
        index=False,
    )

    validation_segmented.to_csv(
        OUTPUT_DIR
        / "validation_decisions_with_li_segment.csv",
        index=False,
    )

    test_segmented.to_csv(
        OUTPUT_DIR
        / "test_decisions_with_li_segment.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_DIR
        / "validation_li_segments.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_li_segments.csv"
    )

    # ========================================================
    # Done
    # ========================================================

    section(
        "DONE"
    )

    print(
        "NEXT:"
    )

    print(
        "Step 13-5 - B0 vs B1 Offline Policy Evaluation"
    )


if __name__ == "__main__":
    main()
