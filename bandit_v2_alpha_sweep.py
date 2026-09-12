import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

INPUT_DIR = Path(
    "bandit_output_v2_interactions"
)

OUTPUT_DIR = Path(
    "bandit_output_v2_alpha_sweep"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


ALPHAS = [
    0.00,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]


DECISION_KEYS = [
    "game_id",
    "def_team",
    "appearance_order",
]


# ============================================================
# Helper
# ============================================================

def print_section(title):

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def normalize_bool(series):
    """
    CSV 로딩 시 bool이 문자열이 되는 경우까지 처리.
    """

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
# Load
# ============================================================

def load_candidates(split):

    path = (
        INPUT_DIR
        / f"{split}_candidates.csv"
    )

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "pitcher_id": str,
        },
    )

    df[
        "appearance_order"
    ] = (
        pd.to_numeric(
            df[
                "appearance_order"
            ],
            errors="raise",
        )
        .astype(int)
    )

    df[
        "is_actual"
    ] = normalize_bool(
        df[
            "is_actual"
        ]
    )

    print(
        f"{split:10s}: "
        f"{len(df):,} candidate rows"
    )

    return df


# ============================================================
# Baselines
# ============================================================

def calculate_baselines(df):

    group_sizes = (
        df.groupby(
            DECISION_KEYS
        )
        .size()
    )

    random_expected = (
        (1.0 / group_sizes)
        .mean()
    )

    # --------------------------------------------------------
    # Heuristic:
    # prior_n_apps 가장 큰 후보
    # --------------------------------------------------------

    h = df.copy()

    h[
        "prior_n_apps"
    ] = (
        pd.to_numeric(
            h[
                "prior_n_apps"
            ],
            errors="coerce",
        )
        .fillna(-1)
    )

    # tie가 있으면 pitcher_id 기준 deterministic
    h = h.sort_values(
        DECISION_KEYS
        + [
            "prior_n_apps",
            "pitcher_id",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
    )

    heuristic_top = (
        h.groupby(
            DECISION_KEYS,
            as_index=False,
        )
        .first()
    )

    heuristic_agreement = (
        heuristic_top[
            "is_actual"
        ]
        .mean()
    )

    return (
        float(
            random_expected
        ),
        float(
            heuristic_agreement
        ),
    )


# ============================================================
# Evaluate Alpha
# ============================================================

def evaluate_alpha(
    df,
    alpha,
):

    work = df.copy()

    # --------------------------------------------------------
    # UCB score
    # --------------------------------------------------------

    work[
        "score"
    ] = (
        work[
            "pred_mean"
        ]
        +
        alpha
        *
        work[
            "uncertainty"
        ]
    )

    # --------------------------------------------------------
    # deterministic tie break
    # --------------------------------------------------------

    work = work.sort_values(
        DECISION_KEYS
        + [
            "score",
            "pitcher_id",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            True,
        ],
    )

    # --------------------------------------------------------
    # Top-1
    # --------------------------------------------------------

    top = (
        work.groupby(
            DECISION_KEYS,
            as_index=False,
        )
        .first()
    )

    agreement = (
        top[
            "is_actual"
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Actual pitcher rank
    # --------------------------------------------------------

    rank_work = (
        df.copy()
    )

    rank_work[
        "score"
    ] = (
        rank_work[
            "pred_mean"
        ]
        +
        alpha
        *
        rank_work[
            "uncertainty"
        ]
    )

    rank_work[
        "rank"
    ] = (
        rank_work
        .groupby(
            DECISION_KEYS
        )["score"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    actual_rows = (
        rank_work[
            rank_work[
                "is_actual"
            ]
        ]
        .copy()
    )

    mean_rank = (
        actual_rows[
            "rank"
        ]
        .mean()
    )

    median_rank = (
        actual_rows[
            "rank"
        ]
        .median()
    )

    top3 = (
        actual_rows[
            "rank"
        ]
        .le(3)
        .mean()
    )

    top5 = (
        actual_rows[
            "rank"
        ]
        .le(5)
        .mean()
    )

    return {
        "alpha":
            float(alpha),

        "agreement":
            float(
                agreement
            ),

        "actual_rank_mean":
            float(
                mean_rank
            ),

        "actual_rank_median":
            float(
                median_rank
            ),

        "actual_top3":
            float(
                top3
            ),

        "actual_top5":
            float(
                top5
            ),
    }


# ============================================================
# Sweep
# ============================================================

def run_sweep(
    name,
    df,
):

    print_section(
        f"ALPHA SWEEP - {name.upper()}"
    )

    (
        random_expected,
        heuristic_agreement,
    ) = calculate_baselines(
        df
    )

    rows = []

    for alpha in ALPHAS:

        result = (
            evaluate_alpha(
                df,
                alpha,
            )
        )

        rows.append(
            result
        )

    result_df = (
        pd.DataFrame(
            rows
        )
    )

    print(
        f"Random expected : "
        f"{random_expected * 100:.2f}%"
    )

    print(
        f"Heuristic       : "
        f"{heuristic_agreement * 100:.2f}%"
    )

    print()

    display = (
        result_df.copy()
    )

    display[
        "agreement"
    ] *= 100

    display[
        "actual_top3"
    ] *= 100

    display[
        "actual_top5"
    ] *= 100

    print(
        display.to_string(
            index=False,
            formatters={
                "alpha":
                    lambda x:
                    f"{x:.2f}",

                "agreement":
                    lambda x:
                    f"{x:.2f}%",

                "actual_rank_mean":
                    lambda x:
                    f"{x:.2f}",

                "actual_rank_median":
                    lambda x:
                    f"{x:.2f}",

                "actual_top3":
                    lambda x:
                    f"{x:.2f}%",

                "actual_top5":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    result_df[
        "random_expected_agreement"
    ] = random_expected

    result_df[
        "heuristic_agreement"
    ] = heuristic_agreement

    result_df.to_csv(
        OUTPUT_DIR
        / f"{name}_alpha_sweep.csv",
        index=False,
    )

    return (
        result_df,
        random_expected,
        heuristic_agreement,
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
        "- ALPHA SWEEP"
    )

    print(
        "Model: Context x Pitcher "
        "Interactions"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # 이미 Step 7에서 계산된 candidate score 구성요소 사용
    # --------------------------------------------------------

    validation = (
        load_candidates(
            "validation"
        )
    )

    test = (
        load_candidates(
            "test"
        )
    )

    # ========================================================
    # Validation
    # ========================================================

    (
        val_results,
        val_random,
        val_heuristic,
    ) = run_sweep(
        "validation",
        validation,
    )

    # ========================================================
    # Validation에서만 alpha 선택
    # ========================================================

    best_idx = (
        val_results[
            "agreement"
        ]
        .idxmax()
    )

    best_alpha = float(
        val_results.loc[
            best_idx,
            "alpha",
        ]
    )

    best_val = float(
        val_results.loc[
            best_idx,
            "agreement",
        ]
    )

    print_section(
        "SELECT ALPHA USING VALIDATION ONLY"
    )

    print(
        f"Selected alpha        : "
        f"{best_alpha:.2f}"
    )

    print(
        f"Validation agreement  : "
        f"{best_val * 100:.2f}%"
    )

    print(
        f"Validation random     : "
        f"{val_random * 100:.2f}%"
    )

    print(
        f"Validation heuristic  : "
        f"{val_heuristic * 100:.2f}%"
    )

    # ========================================================
    # Test sweep는 진단용으로 출력
    # 최종 alpha 선택에는 사용하지 않음
    # ========================================================

    (
        test_results,
        test_random,
        test_heuristic,
    ) = run_sweep(
        "test",
        test,
    )

    selected_test = (
        test_results[
            np.isclose(
                test_results[
                    "alpha"
                ],
                best_alpha,
            )
        ]
        .iloc[0]
    )

    print_section(
        "FINAL TEST WITH VALIDATION-SELECTED ALPHA"
    )

    print(
        f"Alpha               : "
        f"{best_alpha:.2f}"
    )

    print(
        f"Test agreement      : "
        f"{selected_test['agreement'] * 100:.2f}%"
    )

    print(
        f"Random expected     : "
        f"{test_random * 100:.2f}%"
    )

    print(
        f"Heuristic           : "
        f"{test_heuristic * 100:.2f}%"
    )

    print(
        f"Actual rank mean    : "
        f"{selected_test['actual_rank_mean']:.2f}"
    )

    print(
        f"Actual Top-3        : "
        f"{selected_test['actual_top3'] * 100:.2f}%"
    )

    print(
        f"Actual Top-5        : "
        f"{selected_test['actual_top5'] * 100:.2f}%"
    )

    # ========================================================
    # Save selected result
    # ========================================================

    with open(
        OUTPUT_DIR
        / "selected_alpha.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "selection_rule":
                    (
                        "maximize validation "
                        "agreement"
                    ),

                "selected_alpha":
                    best_alpha,

                "validation_agreement":
                    best_val,

                "test_agreement":
                    float(
                        selected_test[
                            "agreement"
                        ]
                    ),

                "test_random":
                    test_random,

                "test_heuristic":
                    test_heuristic,

                "test_actual_rank_mean":
                    float(
                        selected_test[
                            "actual_rank_mean"
                        ]
                    ),

                "test_actual_top3":
                    float(
                        selected_test[
                            "actual_top3"
                        ]
                    ),

                "test_actual_top5":
                    float(
                        selected_test[
                            "actual_top5"
                        ]
                    ),
            },
            f,
            ensure_ascii=False,
            indent=2,
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
        / "validation_alpha_sweep.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_alpha_sweep.csv"
    )

    print(
        " ",
        OUTPUT_DIR
        / "selected_alpha.json"
    )


if __name__ == "__main__":
    main()