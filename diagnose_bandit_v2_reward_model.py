from pathlib import Path

import numpy as np
import pandas as pd

import bandit_v2_actual_bullpen as base
import bandit_v2_interactions as interactions


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = Path(
    "bandit_output_v2_reward_model_diagnostics"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

REWARD_MODE = "wpa_per_pa"
ALPHA = 0.0


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Metrics
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
    baseline_value,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    baseline_pred = np.full(
        len(y_true),
        baseline_value,
        dtype=float,
    )

    error = (
        y_true
        - y_pred
    )

    baseline_error = (
        y_true
        - baseline_pred
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

    baseline_mae = float(
        np.mean(
            np.abs(
                baseline_error
            )
        )
    )

    baseline_rmse = float(
        np.sqrt(
            np.mean(
                baseline_error ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            error ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_true
                - y_true.mean()
            ) ** 2
        )
    )

    if ss_tot > 0:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    else:

        r2 = np.nan

    true_series = pd.Series(
        y_true
    )

    pred_series = pd.Series(
        y_pred
    )

    pearson = float(
        true_series.corr(
            pred_series,
            method="pearson",
        )
    )

    spearman = float(
        true_series.corr(
            pred_series,
            method="spearman",
        )
    )

    return {
        "n":
            int(
                len(y_true)
            ),

        "actual_mean":
            float(
                y_true.mean()
            ),

        "actual_std":
            float(
                y_true.std()
            ),

        "pred_mean":
            float(
                y_pred.mean()
            ),

        "pred_std":
            float(
                y_pred.std()
            ),

        "mae":
            mae,

        "baseline_mae":
            baseline_mae,

        "rmse":
            rmse,

        "baseline_rmse":
            baseline_rmse,

        "r2":
            float(r2),

        "pearson":
            pearson,

        "spearman":
            spearman,
    }


# ============================================================
# Prediction
# ============================================================

def predict_reward(
    model,
    processor,
    df,
):

    X = (
        processor
        .transform_frame(
            df
        )
    )

    prediction = (
        X
        @ model.theta
    )

    return prediction


# ============================================================
# Segment Labels
# ============================================================

def add_segments(df):

    df = (
        df.copy()
    )

    df[
        "abs_score_diff"
    ] = (
        df[
            "score_diff_defense"
        ]
        .abs()
    )

    df[
        "inning_group"
    ] = (
        pd.cut(
            df["inning"],
            bins=[
                -np.inf,
                6,
                8,
                np.inf,
            ],
            labels=[
                "1-6",
                "7-8",
                "9",
            ],
        )
    )

    df[
        "score_group"
    ] = (
        pd.cut(
            df[
                "abs_score_diff"
            ],
            bins=[
                -0.1,
                1,
                3,
                np.inf,
            ],
            labels=[
                "close_0_1",
                "medium_2_3",
                "large_4plus",
            ],
        )
    )

    conditions = [
        (
            (df["inning"] >= 7)
            &
            (
                df[
                    "abs_score_diff"
                ]
                <= 1
            )
        ),

        (
            (df["inning"] >= 7)
            &
            (
                df[
                    "abs_score_diff"
                ]
                > 1
            )
        ),

        (
            (df["inning"] <= 6)
            &
            (
                df[
                    "abs_score_diff"
                ]
                <= 1
            )
        ),
    ]

    choices = [
        "late_close",
        "late_nonclose",
        "early_close",
    ]

    df[
        "state_group"
    ] = (
        np.select(
            conditions,
            choices,
            default="early_nonclose",
        )
    )

    return df


# ============================================================
# Segment Diagnostics
# ============================================================

def segment_metrics(
    df,
    segment_col,
    baseline_value,
):

    rows = []

    for segment, g in (
        df.groupby(
            segment_col,
            observed=True,
        )
    ):

        metrics = (
            regression_metrics(
                g["reward"],
                g["pred_reward"],
                baseline_value,
            )
        )

        metrics[
            "segment_type"
        ] = segment_col

        metrics[
            "segment"
        ] = str(
            segment
        )

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Predicted Reward Quintiles
# ============================================================

def reward_quintiles(
    df,
    split_name,
):

    work = (
        df[
            [
                "reward",
                "pred_reward",
            ]
        ]
        .copy()
    )

    # 가장 낮은 예상 reward -> Q1
    # 가장 높은 예상 reward -> Q5

    try:

        work[
            "predicted_quintile"
        ] = (
            pd.qcut(
                work[
                    "pred_reward"
                ],
                q=5,
                labels=[
                    "Q1_low",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5_high",
                ],
                duplicates="drop",
            )
        )

    except ValueError:

        work[
            "predicted_quintile"
        ] = (
            pd.qcut(
                work[
                    "pred_reward"
                ],
                q=5,
                duplicates="drop",
            )
        )

    result = (
        work.groupby(
            "predicted_quintile",
            observed=True,
        )
        .agg(
            n=(
                "reward",
                "size",
            ),

            predicted_reward_mean=(
                "pred_reward",
                "mean",
            ),

            actual_reward_mean=(
                "reward",
                "mean",
            ),

            actual_reward_median=(
                "reward",
                "median",
            ),
        )
        .reset_index()
    )

    result[
        "split"
    ] = split_name

    return result


# ============================================================
# Pretty Print
# ============================================================

def print_metrics_table(
    title,
    df,
):

    print_section(
        title
    )

    display = (
        df.copy()
    )

    cols = [
        "segment",
        "n",
        "mae",
        "baseline_mae",
        "rmse",
        "baseline_rmse",
        "r2",
        "pearson",
        "spearman",
    ]

    print(
        display[
            cols
        ]
        .to_string(
            index=False,
            formatters={
                "mae":
                    lambda x:
                    f"{x:.6f}",

                "baseline_mae":
                    lambda x:
                    f"{x:.6f}",

                "rmse":
                    lambda x:
                    f"{x:.6f}",

                "baseline_rmse":
                    lambda x:
                    f"{x:.6f}",

                "r2":
                    lambda x:
                    f"{x:.4f}",

                "pearson":
                    lambda x:
                    f"{x:.4f}",

                "spearman":
                    lambda x:
                    f"{x:.4f}",
            },
        )
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
        "- REWARD MODEL DIAGNOSTICS"
    )

    print(
        f"Reward : {REWARD_MODE}"
    )

    print(
        f"Alpha  : {ALPHA}"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 1. Data
    # ========================================================

    base.REWARD_MODE = (
        REWARD_MODE
    )

    base.ALPHA = (
        ALPHA
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

    # 정책 평가와 동일한 실제 bullpen 구간
    val = (
        val[
            val[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    test = (
        test[
            test[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    print_section(
        "DATA COUNTS"
    )

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(val):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    # ========================================================
    # 2. Train interaction reward model
    # ========================================================

    processor = (
        interactions
        .InteractionFeatureProcessor()
    )

    model = (
        interactions
        .train_interaction_bandit(
            train,
            processor,
        )
    )

    # ========================================================
    # 3. Predict logged actual actions
    # ========================================================

    train = (
        train.copy()
    )

    val = (
        val.copy()
    )

    test = (
        test.copy()
    )

    train[
        "pred_reward"
    ] = (
        predict_reward(
            model,
            processor,
            train,
        )
    )

    val[
        "pred_reward"
    ] = (
        predict_reward(
            model,
            processor,
            val,
        )
    )

    test[
        "pred_reward"
    ] = (
        predict_reward(
            model,
            processor,
            test,
        )
    )

    # train에서만 baseline 평균 계산
    baseline_value = float(
        train[
            "reward"
        ]
        .mean()
    )

    print_section(
        "TRAIN BASELINE"
    )

    print(
        "Train mean reward:",
        f"{baseline_value:.6f}",
    )

    # ========================================================
    # 4. Overall metrics
    # ========================================================

    overall_rows = []

    for (
        split_name,
        df,
    ) in [
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

        metrics = (
            regression_metrics(
                df[
                    "reward"
                ],
                df[
                    "pred_reward"
                ],
                baseline_value,
            )
        )

        metrics[
            "split"
        ] = (
            split_name
        )

        overall_rows.append(
            metrics
        )

    overall = (
        pd.DataFrame(
            overall_rows
        )
    )

    print_section(
        "OVERALL REWARD PREDICTION"
    )

    print(
        overall[
            [
                "split",
                "n",
                "actual_mean",
                "pred_mean",
                "mae",
                "baseline_mae",
                "rmse",
                "baseline_rmse",
                "r2",
                "pearson",
                "spearman",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "actual_mean":
                    lambda x:
                    f"{x:.6f}",

                "pred_mean":
                    lambda x:
                    f"{x:.6f}",

                "mae":
                    lambda x:
                    f"{x:.6f}",

                "baseline_mae":
                    lambda x:
                    f"{x:.6f}",

                "rmse":
                    lambda x:
                    f"{x:.6f}",

                "baseline_rmse":
                    lambda x:
                    f"{x:.6f}",

                "r2":
                    lambda x:
                    f"{x:.4f}",

                "pearson":
                    lambda x:
                    f"{x:.4f}",

                "spearman":
                    lambda x:
                    f"{x:.4f}",
            },
        )
    )

    # ========================================================
    # 5. Segment labels
    # ========================================================

    val = add_segments(
        val
    )

    test = add_segments(
        test
    )

    # ========================================================
    # 6. Test segment metrics
    # ========================================================

    segment_frames = []

    for col in [
        "inning_group",
        "score_group",
        "state_group",
    ]:

        segment_df = (
            segment_metrics(
                test,
                col,
                baseline_value,
            )
        )

        segment_frames.append(
            segment_df
        )

        print_metrics_table(
            (
                "TEST REWARD PREDICTION "
                f"- {col}"
            ),
            segment_df,
        )

    all_segments = (
        pd.concat(
            segment_frames,
            ignore_index=True,
        )
    )

    # ========================================================
    # 7. Prediction quintiles
    # ========================================================

    val_quintiles = (
        reward_quintiles(
            val,
            "validation",
        )
    )

    test_quintiles = (
        reward_quintiles(
            test,
            "test",
        )
    )

    quintiles = (
        pd.concat(
            [
                val_quintiles,
                test_quintiles,
            ],
            ignore_index=True,
        )
    )

    print_section(
        "PREDICTED REWARD QUINTILES"
    )

    print(
        quintiles.to_string(
            index=False,
            formatters={
                "predicted_reward_mean":
                    lambda x:
                    f"{x:.6f}",

                "actual_reward_mean":
                    lambda x:
                    f"{x:.6f}",

                "actual_reward_median":
                    lambda x:
                    f"{x:.6f}",
            },
        )
    )

    # ========================================================
    # 8. Save
    # ========================================================

    overall.to_csv(
        OUTPUT_DIR
        / "overall_metrics.csv",
        index=False,
    )

    all_segments.to_csv(
        OUTPUT_DIR
        / "test_segment_metrics.csv",
        index=False,
    )

    quintiles.to_csv(
        OUTPUT_DIR
        / "predicted_reward_quintiles.csv",
        index=False,
    )

    test[
        [
            "game_id",
            "date",
            "inning",
            "score_diff_defense",
            "pitcher_id",
            "reward",
            "pred_reward",
            "inning_group",
            "score_group",
            "state_group",
        ]
    ].to_csv(
        OUTPUT_DIR
        / "test_logged_action_predictions.csv",
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
        / "overall_metrics.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_segment_metrics.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "predicted_reward_quintiles.csv"
    )


if __name__ == "__main__":
    main()