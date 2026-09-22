from pathlib import Path

import numpy as np
import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

RESULT_PATH = (
    "outputs/bandit/diagnostics/v2_reward_recheck/"
    "wpa_per_pa/"
    "test_decisions.csv"
)

OUTPUT_DIR = Path(
    "outputs/bandit/diagnostics/v2_segment_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helper
# ============================================================

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


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
# 1. LOAD FINAL MODEL RESULT
# ============================================================

print_section(
    "1. LOAD FINAL WPA_PER_PA RESULT"
)

result = pd.read_csv(
    RESULT_PATH,
    dtype={
        "game_id": str,
        "actual_pitcher": str,
    },
)

result["game_id"] = (
    result["game_id"]
    .astype(str)
)

result["actual_pitcher"] = (
    result["actual_pitcher"]
    .apply(normalize_id)
)

result["appearance_order"] = (
    pd.to_numeric(
        result[
            "appearance_order"
        ],
        errors="raise",
    )
    .astype(int)
)

print(
    "Final test decisions:",
    f"{len(result):,}",
)

print(
    "Overall model agreement:",
    f"{result['mean_match'].mean() * 100:.2f}%"
)

print(
    "Overall random:",
    f"{result['random_expected_match'].mean() * 100:.2f}%"
)

print(
    "Overall heuristic:",
    f"{result['heuristic_match'].mean() * 100:.2f}%"
)


# ============================================================
# 2. LOAD CONTEXT
# ============================================================

print_section(
    "2. LOAD TEST CONTEXT"
)

pa, history, bullpen = (
    base.load_data()
)

# reward는 이번 분석에 직접 필요 없지만
# 현재 최종 설정과 맞춰준다.
base.REWARD_MODE = (
    "wpa_per_pa"
)

decisions, first_app = (
    base.build_decision_dataset(
        pa,
        history,
    )
)

train, val, test = (
    base.chronological_split(
        decisions
    )
)

# 실제 명단이 존재하는 구간만
test = (
    test[
        test[
            "bullpen_source"
        ]
        == "preview"
    ]
    .copy()
)

test["pitcher_id"] = (
    test["pitcher_id"]
    .apply(normalize_id)
)

test["game_id"] = (
    test["game_id"]
    .astype(str)
)

test["appearance_order"] = (
    test[
        "appearance_order"
    ]
    .astype(int)
)


# ============================================================
# 3. RESULT + CONTEXT JOIN
# ============================================================

print_section(
    "3. MERGE RESULT WITH GAME CONTEXT"
)

context_cols = [
    "game_id",
    "def_team",
    "appearance_order",
    "pitcher_id",

    "out_count",
    "score_diff_defense",

    "is_base1",
    "is_base2",
    "is_base3",

    "bullpen_available_ratio",
    "n_pitchers_used",
]


context = (
    test[
        context_cols
    ]
    .copy()
)

context = (
    context.rename(
        columns={
            "pitcher_id":
                "actual_pitcher"
        }
    )
)


merged = (
    result.merge(
        context,
        on=[
            "game_id",
            "def_team",
            "appearance_order",
            "actual_pitcher",
        ],
        how="left",
        validate="one_to_one",
    )
)


missing_context = (
    merged["score_diff_defense"]
    .isna()
    .sum()
)

print(
    "Merged rows:",
    f"{len(merged):,}",
)

print(
    "Missing context:",
    f"{missing_context:,}",
)

if missing_context > 0:
    raise ValueError(
        "일부 결과에 context가 조인되지 않았습니다."
    )


# ============================================================
# 4. DERIVED STATE FEATURES
# ============================================================

print_section(
    "4. CREATE SEGMENT LABELS"
)


# ------------------------------------------------------------
# 4-1. 이닝
# ------------------------------------------------------------

merged["inning_group"] = (
    pd.cut(
        merged["inning"],
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


# ------------------------------------------------------------
# 4-2. 점수차
#
# 수비팀 기준 부호보다
# '얼마나 접전인가'를 보기 위해 절댓값 사용
# ------------------------------------------------------------

merged[
    "abs_score_diff"
] = (
    merged[
        "score_diff_defense"
    ]
    .abs()
)


merged["score_group"] = (
    pd.cut(
        merged[
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


# ------------------------------------------------------------
# 4-3. 주자
# ------------------------------------------------------------

merged[
    "has_runner"
] = (
    (
        merged[
            [
                "is_base1",
                "is_base2",
                "is_base3",
            ]
        ]
        .sum(axis=1)
        > 0
    )
)


merged["runner_group"] = np.where(
    merged["has_runner"],
    "runner_on",
    "bases_empty",
)


# ------------------------------------------------------------
# 4-4. 아웃
# ------------------------------------------------------------

merged["out_group"] = (
    merged["out_count"]
    .astype(int)
    .astype(str)
    + "_out"
)


# ------------------------------------------------------------
# 4-5. 후보 수
# ------------------------------------------------------------

merged["candidate_group"] = (
    pd.cut(
        merged["n_candidates"],
        bins=[
            -np.inf,
            9,
            11,
            np.inf,
        ],
        labels=[
            "small_le9",
            "medium_10_11",
            "large_ge12",
        ],
    )
)


# ------------------------------------------------------------
# 4-6. 불펜 잔여 비율
# ------------------------------------------------------------

merged["bullpen_ratio_group"] = (
    pd.cut(
        merged[
            "bullpen_available_ratio"
        ],
        bins=[
            -np.inf,
            0.50,
            0.75,
            np.inf,
        ],
        labels=[
            "low_le0.50",
            "mid_0.50_0.75",
            "high_gt0.75",
        ],
    )
)


# ------------------------------------------------------------
# 4-7. LI가 오기 전 임시 상황 그룹
#
# late_close:
# 7회 이후 + 1점차 이내
#
# LI를 대체하는 값은 절대 아니고
# 단순 구간 분석용 proxy
# ------------------------------------------------------------

conditions = [
    (
        (merged["inning"] >= 7)
        &
        (
            merged[
                "abs_score_diff"
            ]
            <= 1
        )
    ),

    (
        (merged["inning"] >= 7)
        &
        (
            merged[
                "abs_score_diff"
            ]
            > 1
        )
    ),

    (
        (merged["inning"] <= 6)
        &
        (
            merged[
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

merged["state_group"] = (
    np.select(
        conditions,
        choices,
        default="early_nonclose",
    )
)


print(
    "Segment labels created."
)


# ============================================================
# 5. SEGMENT METRICS
# ============================================================

def calculate_segment_metrics(
    df,
    column,
):
    """
    각 구간별로:
    - 표본 수
    - 평균 후보 수
    - Random
    - Heuristic
    - Model Top-1
    - Model - Random
    - Model - Heuristic
    - Top-3
    - Top-5
    - 실제 투수 평균 rank
    """

    rows = []

    for group_value, g in (
        df.groupby(
            column,
            observed=True,
        )
    ):

        if len(g) == 0:
            continue

        model = (
            g[
                "mean_match"
            ]
            .mean()
        )

        random = (
            g[
                "random_expected_match"
            ]
            .mean()
        )

        heuristic = (
            g[
                "heuristic_match"
            ]
            .mean()
        )

        top3 = (
            (
                g[
                    "actual_mean_rank"
                ]
                <= 3
            )
            .mean()
        )

        top5 = (
            (
                g[
                    "actual_mean_rank"
                ]
                <= 5
            )
            .mean()
        )

        rank_mean = (
            g[
                "actual_mean_rank"
            ]
            .mean()
        )

        rows.append(
            {
                "segment_type":
                    column,

                "segment":
                    str(
                        group_value
                    ),

                "n":
                    int(
                        len(g)
                    ),

                "avg_candidates":
                    float(
                        g[
                            "n_candidates"
                        ]
                        .mean()
                    ),

                "random":
                    float(
                        random
                    ),

                "heuristic":
                    float(
                        heuristic
                    ),

                "model":
                    float(
                        model
                    ),

                "model_minus_random":
                    float(
                        model
                        - random
                    ),

                "model_minus_heuristic":
                    float(
                        model
                        - heuristic
                    ),

                "top3":
                    float(
                        top3
                    ),

                "top5":
                    float(
                        top5
                    ),

                "mean_rank":
                    float(
                        rank_mean
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 6. RUN ALL SEGMENTS
# ============================================================

print_section(
    "5. SEGMENT PERFORMANCE"
)


segment_columns = [
    "inning_group",
    "score_group",
    "runner_group",
    "out_group",
    "candidate_group",
    "bullpen_ratio_group",
    "state_group",
]


all_results = []


for column in segment_columns:

    segment_result = (
        calculate_segment_metrics(
            merged,
            column,
        )
    )

    all_results.append(
        segment_result
    )

    print()
    print(
        "-" * 72
    )

    print(
        f"SEGMENT: {column}"
    )

    print(
        "-" * 72
    )

    display = (
        segment_result.copy()
    )

    percent_cols = [
        "random",
        "heuristic",
        "model",
        "model_minus_random",
        "model_minus_heuristic",
        "top3",
        "top5",
    ]

    for c in percent_cols:

        display[c] = (
            display[c]
            * 100
        )

    print(
        display[
            [
                "segment",
                "n",
                "avg_candidates",
                "random",
                "heuristic",
                "model",
                "model_minus_random",
                "top3",
                "top5",
                "mean_rank",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "avg_candidates":
                    lambda x:
                    f"{x:.2f}",

                "random":
                    lambda x:
                    f"{x:.2f}%",

                "heuristic":
                    lambda x:
                    f"{x:.2f}%",

                "model":
                    lambda x:
                    f"{x:.2f}%",

                "model_minus_random":
                    lambda x:
                    f"{x:+.2f}%p",

                "top3":
                    lambda x:
                    f"{x:.2f}%",

                "top5":
                    lambda x:
                    f"{x:.2f}%",

                "mean_rank":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )


# ============================================================
# 7. COMBINE + SAVE
# ============================================================

combined = (
    pd.concat(
        all_results,
        ignore_index=True,
    )
)


combined.to_csv(
    OUTPUT_DIR
    / "segment_performance.csv",
    index=False,
)


merged.to_csv(
    OUTPUT_DIR
    / "test_decisions_with_context.csv",
    index=False,
)


# ============================================================
# 8. BEST / WORST SEGMENTS
# ============================================================

print_section(
    "6. BEST SEGMENTS VS RANDOM"
)


eligible = (
    combined[
        combined[
            "n"
        ]
        >= 100
    ]
    .copy()
)


best = (
    eligible.sort_values(
        "model_minus_random",
        ascending=False,
    )
    .head(10)
)


display_best = (
    best.copy()
)

for c in [
    "random",
    "heuristic",
    "model",
    "model_minus_random",
]:

    display_best[c] = (
        display_best[c]
        * 100
    )


print(
    display_best[
        [
            "segment_type",
            "segment",
            "n",
            "random",
            "model",
            "model_minus_random",
            "heuristic",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "random":
                lambda x:
                f"{x:.2f}%",

            "model":
                lambda x:
                f"{x:.2f}%",

            "model_minus_random":
                lambda x:
                f"{x:+.2f}%p",

            "heuristic":
                lambda x:
                f"{x:.2f}%",
        },
    )
)


print_section(
    "7. WORST SEGMENTS VS RANDOM"
)


worst = (
    eligible.sort_values(
        "model_minus_random",
        ascending=True,
    )
    .head(10)
)


display_worst = (
    worst.copy()
)

for c in [
    "random",
    "heuristic",
    "model",
    "model_minus_random",
]:

    display_worst[c] = (
        display_worst[c]
        * 100
    )


print(
    display_worst[
        [
            "segment_type",
            "segment",
            "n",
            "random",
            "model",
            "model_minus_random",
            "heuristic",
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "random":
                lambda x:
                f"{x:.2f}%",

            "model":
                lambda x:
                f"{x:.2f}%",

            "model_minus_random":
                lambda x:
                f"{x:+.2f}%p",

            "heuristic":
                lambda x:
                f"{x:.2f}%",
        },
    )
)


# ============================================================
# DONE
# ============================================================

print_section(
    "DONE"
)

print(
    "Saved:"
)

print(
    " ",
    OUTPUT_DIR
    / "segment_performance.csv"
)

print(
    " ",
    OUTPUT_DIR
    / "test_decisions_with_context.csv"
)
