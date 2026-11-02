import json
from pathlib import Path

import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base
from kbo_bandit.core import bandit_v2_profile_fixed as profile_fix
from kbo_bandit.core import bandit_v2_interactions as interactions


# ============================================================
# Config
# ============================================================

OUTPUT_ROOT = Path(
    "outputs/bandit/diagnostics/v2_reward_recheck"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

REWARD_MODES = [
    "app_wpa",
    "wpa_per_pa",
]

# Step 8에서 validation으로 선택된 값
FIXED_ALPHA = 0.0


# ============================================================
# Helper
# ============================================================

def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Reward Diagnostics
# ============================================================

def reward_diagnostics(
    train,
    val,
    test,
    reward_mode,
):
    """
    Reward 분포와 appearance 길이 상관을 split별로 확인.

    app_wpa:
        reward = -Option B app_wpa

    wpa_per_pa:
        reward = -Option B app_wpa / n_pa
    """

    rows = []

    for split_name, df in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:

        reward = (
            df["reward"]
            .astype(float)
        )

        n_pa = (
            df["n_pa_optionB"]
            .astype(float)
        )

        row = {
            "reward_mode":
                reward_mode,

            "split":
                split_name,

            "n":
                int(len(df)),

            "mean":
                float(
                    reward.mean()
                ),

            "std":
                float(
                    reward.std()
                ),

            "skew":
                float(
                    reward.skew()
                ),

            "corr_with_n_pa":
                float(
                    reward.corr(
                        n_pa
                    )
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Single Reward Experiment
# ============================================================

def run_reward_experiment(
    reward_mode,
    pa,
    history,
    bullpen,
):

    print_section(
        f"REWARD EXPERIMENT: {reward_mode}"
    )

    # --------------------------------------------------------
    # 전역 설정
    # --------------------------------------------------------

    base.REWARD_MODE = (
        reward_mode
    )

    base.ALPHA = (
        FIXED_ALPHA
    )

    print(
        f"Reward mode : {base.REWARD_MODE}"
    )

    print(
        f"Alpha       : {base.ALPHA}"
    )

    # --------------------------------------------------------
    # Decision dataset
    #
    # copy를 사용해서 두 reward 실험이
    # 서로 영향을 주지 않게 함
    # --------------------------------------------------------

    (
        decisions,
        first_app,
    ) = (
        base.build_decision_dataset(
            pa.copy(),
            history.copy(),
        )
    )

    # --------------------------------------------------------
    # 실제 bullpen 후보군
    # --------------------------------------------------------

    bullpen_map = (
        base.build_bullpen_map(
            bullpen.copy()
        )
    )

    used_before_map = (
        base.build_used_before_map(
            first_app
        )
    )

    # --------------------------------------------------------
    # 동일 chronological split
    # --------------------------------------------------------

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
    # Reward distribution
    # ========================================================

    diagnostics = (
        reward_diagnostics(
            train,
            val,
            test,
            reward_mode,
        )
    )

    print()
    print(
        "Reward diagnostics:"
    )

    print(
        diagnostics.to_string(
            index=False,
            formatters={
                "mean":
                    lambda x:
                    f"{x:.6f}",

                "std":
                    lambda x:
                    f"{x:.6f}",

                "skew":
                    lambda x:
                    f"{x:.4f}",

                "corr_with_n_pa":
                    lambda x:
                    f"{x:.4f}",
            },
        )
    )

    # ========================================================
    # Candidate profile fix
    # ========================================================

    full_history = (
        profile_fix.prepare_full_history(
            history.copy()
        )
    )

    profile_index = (
        profile_fix.build_profile_index(
            full_history
        )
    )

    # ========================================================
    # Interaction model
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

    # alpha가 정말 0인지 확인
    print(
        f"Model alpha : {model.alpha}"
    )

    # ========================================================
    # Validation
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
    # Test
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

    output_dir = (
        OUTPUT_ROOT
        / reward_mode
    )

    output_dir.mkdir(
        exist_ok=True
    )

    diagnostics.to_csv(
        output_dir
        / "reward_diagnostics.csv",
        index=False,
    )

    val_result.to_csv(
        output_dir
        / "validation_decisions.csv",
        index=False,
    )

    test_result.to_csv(
        output_dir
        / "test_decisions.csv",
        index=False,
    )

    val_candidates.to_csv(
        output_dir
        / "validation_candidates.csv",
        index=False,
    )

    test_candidates.to_csv(
        output_dir
        / "test_candidates.csv",
        index=False,
    )

    with open(
        output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "reward_mode":
                    reward_mode,

                "alpha":
                    FIXED_ALPHA,

                "candidate_source":
                    "actual_bullpen_preview",

                "profile_method":
                    (
                        "global_full_history_"
                        "asof_snapshot"
                    ),

                "interactions":
                    True,

                "validation":
                    val_summary,

                "test":
                    test_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "reward_mode":
            reward_mode,

        "validation_random":
            val_summary[
                "random_agreement"
            ],

        "validation_heuristic":
            val_summary[
                "heuristic_agreement"
            ],

        "validation_model":
            val_summary[
                "mean_only_agreement"
            ],

        "validation_rank_mean":
            float(
                val_result[
                    "actual_mean_rank"
                ].mean()
            ),

        "validation_top3":
            float(
                (
                    val_result[
                        "actual_mean_rank"
                    ]
                    <= 3
                ).mean()
            ),

        "validation_top5":
            float(
                (
                    val_result[
                        "actual_mean_rank"
                    ]
                    <= 5
                ).mean()
            ),

        "test_random":
            test_summary[
                "random_agreement"
            ],

        "test_heuristic":
            test_summary[
                "heuristic_agreement"
            ],

        "test_model":
            test_summary[
                "mean_only_agreement"
            ],

        "test_rank_mean":
            float(
                test_result[
                    "actual_mean_rank"
                ].mean()
            ),

        "test_top3":
            float(
                (
                    test_result[
                        "actual_mean_rank"
                    ]
                    <= 3
                ).mean()
            ),

        "test_top5":
            float(
                (
                    test_result[
                        "actual_mean_rank"
                    ]
                    <= 5
                ).mean()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "CONTEXTUAL BANDIT V2 "
        "- FINAL REWARD A/B RECHECK"
    )

    print(
        "Candidate   : actual bullpen / preview"
    )

    print(
        "Profile     : fixed"
    )

    print(
        "Interaction : ON"
    )

    print(
        f"Alpha       : {FIXED_ALPHA}"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # 원본 데이터는 한 번만 load
    # --------------------------------------------------------

    (
        pa,
        history,
        bullpen,
    ) = (
        base.load_data()
    )

    comparison_rows = []

    # ========================================================
    # 두 Reward 비교
    # ========================================================

    for reward_mode in (
        REWARD_MODES
    ):

        result = (
            run_reward_experiment(
                reward_mode,
                pa,
                history,
                bullpen,
            )
        )

        comparison_rows.append(
            result
        )

    # ========================================================
    # Comparison
    # ========================================================

    comparison = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "reward_comparison.csv",
        index=False,
    )

    print_section(
        "FINAL REWARD COMPARISON"
    )

    display = (
        comparison.copy()
    )

    percent_cols = [
        "validation_random",
        "validation_heuristic",
        "validation_model",
        "validation_top3",
        "validation_top5",
        "test_random",
        "test_heuristic",
        "test_model",
        "test_top3",
        "test_top5",
    ]

    for col in percent_cols:

        display[col] = (
            display[col]
            * 100
        )

    print(
        display.to_string(
            index=False,
            formatters={
                "validation_random":
                    lambda x:
                    f"{x:.2f}%",

                "validation_heuristic":
                    lambda x:
                    f"{x:.2f}%",

                "validation_model":
                    lambda x:
                    f"{x:.2f}%",

                "validation_rank_mean":
                    lambda x:
                    f"{x:.2f}",

                "validation_top3":
                    lambda x:
                    f"{x:.2f}%",

                "validation_top5":
                    lambda x:
                    f"{x:.2f}%",

                "test_random":
                    lambda x:
                    f"{x:.2f}%",

                "test_heuristic":
                    lambda x:
                    f"{x:.2f}%",

                "test_model":
                    lambda x:
                    f"{x:.2f}%",

                "test_rank_mean":
                    lambda x:
                    f"{x:.2f}",

                "test_top3":
                    lambda x:
                    f"{x:.2f}%",

                "test_top5":
                    lambda x:
                    f"{x:.2f}%",
            },
        )
    )

    print_section(
        "IMPORTANT"
    )

    print(
        "app_wpa is included only as an "
        "ablation baseline."
    )

    print(
        "Do NOT adopt unnormalized app_wpa "
        "as the final reward."
    )

    print(
        "Reward choice should be interpreted "
        "with distribution, length correlation, "
        "ranking metrics, and validation results."
    )

    print_section(
        "DONE"
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_ROOT
        / "reward_comparison.csv"
    )


if __name__ == "__main__":
    main()
