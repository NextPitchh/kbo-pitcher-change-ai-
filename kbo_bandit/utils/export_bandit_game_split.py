from pathlib import Path

import pandas as pd

from kbo_bandit.core import bandit_v2_actual_bullpen as base


OUTPUT_PATH = Path(
    "data_analysis/results/bandit_game_split.csv"
)


EXPECTED = {
    "train": {
        "decisions": 34138,
        "games": 5030,
    },
    "validation": {
        "decisions": 7904,
        "games": 1078,
    },
    "test": {
        "decisions": 8135,
        "games": 1078,
    },
}


def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def build_game_split(
    df,
    split_name,
):

    work = df.copy()

    work["game_id"] = (
        work["game_id"]
        .astype(str)
    )

    work["date"] = pd.to_datetime(
        work["date"]
    )

    games = (
        work.groupby(
            "game_id",
            as_index=False,
        )
        .agg(
            date=("date", "min"),
            n_decisions=("game_id", "size"),
        )
    )

    games["split"] = split_name

    return games[
        [
            "game_id",
            "date",
            "split",
            "n_decisions",
        ]
    ]


def main():

    section(
        "LOAD BANDIT DATA"
    )

    base.REWARD_MODE = "wpa_per_pa"
    base.ALPHA = 0.0

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

    (
        train,
        val,
        test,
    ) = base.chronological_split(
        decisions
    )

    section(
        "BUILD GAME SPLIT"
    )

    train_games = build_game_split(
        train,
        "train",
    )

    val_games = build_game_split(
        val,
        "validation",
    )

    test_games = build_game_split(
        test,
        "test",
    )

    frames = {
        "train": (
            train,
            train_games,
        ),
        "validation": (
            val,
            val_games,
        ),
        "test": (
            test,
            test_games,
        ),
    }

    for split_name, (
        decision_df,
        game_df,
    ) in frames.items():

        actual_decisions = len(
            decision_df
        )

        actual_games = len(
            game_df
        )

        expected_decisions = (
            EXPECTED[
                split_name
            ][
                "decisions"
            ]
        )

        expected_games = (
            EXPECTED[
                split_name
            ][
                "games"
            ]
        )

        print(
            f"{split_name:10s} | "
            f"decisions "
            f"{actual_decisions:,} "
            f"(expected {expected_decisions:,}) | "
            f"games "
            f"{actual_games:,} "
            f"(expected {expected_games:,})"
        )

        if (
            actual_decisions
            != expected_decisions
        ):

            raise ValueError(
                f"{split_name} decision count mismatch"
            )

        if (
            actual_games
            != expected_games
        ):

            raise ValueError(
                f"{split_name} game count mismatch"
            )

    section(
        "CHECK GAME OVERLAP"
    )

    train_ids = set(
        train_games[
            "game_id"
        ]
    )

    val_ids = set(
        val_games[
            "game_id"
        ]
    )

    test_ids = set(
        test_games[
            "game_id"
        ]
    )

    print(
        "Train ∩ Validation:",
        len(
            train_ids
            &
            val_ids
        ),
    )

    print(
        "Train ∩ Test      :",
        len(
            train_ids
            &
            test_ids
        ),
    )

    print(
        "Validation ∩ Test :",
        len(
            val_ids
            &
            test_ids
        ),
    )

    if (
        train_ids
        &
        val_ids
    ):

        raise ValueError(
            "Train/Validation game overlap"
        )

    if (
        train_ids
        &
        test_ids
    ):

        raise ValueError(
            "Train/Test game overlap"
        )

    if (
        val_ids
        &
        test_ids
    ):

        raise ValueError(
            "Validation/Test game overlap"
        )

    section(
        "SAVE"
    )

    result = pd.concat(
        [
            train_games,
            val_games,
            test_games,
        ],
        ignore_index=True,
    )

    result = (
        result.sort_values(
            [
                "date",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Saved:"
    )

    print(
        " ",
        OUTPUT_PATH
    )

    print()
    print(
        f"Total games: "
        f"{len(result):,}"
    )

    print()
    print(
        result[
            "split"
        ]
        .value_counts(
            sort=False
        )
    )

    section(
        "DATE RANGE"
    )

    summary = (
        result.groupby(
            "split"
        )
        .agg(
            games=(
                "game_id",
                "nunique",
            ),

            start_date=(
                "date",
                "min",
            ),

            end_date=(
                "date",
                "max",
            ),
        )
    )

    print(
        summary.to_string()
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()
