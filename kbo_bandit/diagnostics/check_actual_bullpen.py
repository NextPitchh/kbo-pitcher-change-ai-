from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PA_PATH = (
    "data_analysis/results/"
    "all_pa_enriched_v3.parquet"
)

BULLPEN_PATH = (
    "data_analysis/results/"
    "all_game_bullpen.parquet"
)

OUTPUT_DIR = Path("outputs/bandit/core/v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper
# ============================================================

def normalize_id(x):
    """
    pitcher_id 타입 통일.

    parquet에 따라 12345 / 12345.0 / "12345"
    형태가 섞일 가능성에 대비한다.
    """

    if pd.isna(x):
        return "__MISSING__"

    s = str(x).strip()

    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    return s


def defense_team_from_game(game_id, half):
    """
    game_id:
    YYYYMMDD{away}{home}{seq}{year}

    확정 컨벤션:
    half="top" -> 원정팀 공격 -> 홈팀 수비
    half="bot" -> 홈팀 공격   -> 원정팀 수비
    """

    gid = str(game_id)

    away = gid[8:10]
    home = gid[10:12]

    if half == "top":
        return home

    return away


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 1. LOAD
# ============================================================

print_section("1. LOAD DATA")

pa = pd.read_parquet(PA_PATH)
bp = pd.read_parquet(BULLPEN_PATH)

print(f"PA      : {pa.shape}")
print(f"BULLPEN : {bp.shape}")


# ------------------------------------------------------------
# 타입 통일
# ------------------------------------------------------------

pa["game_id"] = (
    pa["game_id"]
    .astype(str)
)

bp["game_id"] = (
    bp["game_id"]
    .astype(str)
)

pa["pitcher_id"] = (
    pa["pitcher_id"]
    .apply(normalize_id)
)

bp["pitcher_id"] = (
    bp["pitcher_id"]
    .apply(normalize_id)
)

bp["team_code"] = (
    bp["team_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 2. BULLPEN FILE BASIC CHECK
# ============================================================

print_section("2. BULLPEN BASIC CHECK")


print(
    "Date range:",
    bp["game_id"].str[:8].min(),
    "~",
    bp["game_id"].str[:8].max(),
)


print(
    "Unique games:",
    f"{bp['game_id'].nunique():,}",
)


print(
    "Unique teams:",
    sorted(
        bp["team_code"].unique()
    ),
)


print()
print("is_listed_bullpen:")

print(
    bp["is_listed_bullpen"]
    .value_counts(
        dropna=False
    )
)


# ------------------------------------------------------------
# 중복 확인
# ------------------------------------------------------------

dup_count = (
    bp.duplicated(
        subset=[
            "game_id",
            "team_code",
            "pitcher_id",
        ]
    )
    .sum()
)

print()

print(
    "Duplicate "
    "(game_id, team_code, pitcher_id):",
    dup_count,
)


# ------------------------------------------------------------
# 경기당 팀 수
# ------------------------------------------------------------

teams_per_game = (
    bp.groupby("game_id")[
        "team_code"
    ]
    .nunique()
)

print()
print("Teams per game:")

print(
    teams_per_game
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 경기-팀별 명단 크기
# ------------------------------------------------------------

roster_size = (
    bp.groupby(
        [
            "game_id",
            "team_code",
        ]
    )["pitcher_id"]
    .nunique()
)

print()
print("Roster size per game-team:")

print(
    roster_size.describe()
)


# ============================================================
# 3. is_home CONSISTENCY
# ============================================================

print_section(
    "3. TEAM / is_home CONSISTENCY"
)


def expected_is_home(row):
    gid = row["game_id"]

    home_team = gid[10:12]

    return (
        row["team_code"]
        == home_team
    )


bp["expected_is_home"] = (
    bp.apply(
        expected_is_home,
        axis=1,
    )
)


home_match = (
    bp["is_home"].astype(bool)
    ==
    bp["expected_is_home"]
)


print(
    "is_home match:",
    f"{home_match.mean() * 100:.4f}%"
)


if not home_match.all():

    print()
    print("Mismatch examples:")

    print(
        bp.loc[
            ~home_match,
            [
                "game_id",
                "team_code",
                "is_home",
                "expected_is_home",
            ],
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# 4. BUILD ALL APPEARANCES
# ============================================================

print_section(
    "4. BUILD ALL APPEARANCES"
)


required = [
    "game_id",
    "_orig_idx",
    "half",
    "pitcher_id",
    "appearance_order",
    "is_starter",
    "bullpen_source",
]


missing = [
    c
    for c in required
    if c not in pa.columns
]


if missing:
    raise ValueError(
        f"PA missing columns: {missing}"
    )


# ------------------------------------------------------------
# appearance_order 타입 정리
# ------------------------------------------------------------

pa["appearance_order"] = (
    pd.to_numeric(
        pa["appearance_order"],
        errors="coerce",
    )
)


pa = (
    pa.dropna(
        subset=[
            "appearance_order"
        ]
    )
    .copy()
)


pa["appearance_order"] = (
    pa["appearance_order"]
    .astype(int)
)


appearance_keys = [
    "game_id",
    "half",
    "pitcher_id",
    "appearance_order",
]


# ------------------------------------------------------------
# 각 투수 appearance의 첫 PA
#
# 중요:
# 여기서는 선발을 제거하지 않는다.
# 선발도 "이미 사용된 투수" 계산에 필요하기 때문.
# ------------------------------------------------------------

first_app = (
    pa.sort_values(
        [
            "game_id",
            "_orig_idx",
        ]
    )
    .groupby(
        appearance_keys,
        as_index=False,
    )
    .first()
)


# ------------------------------------------------------------
# 수비팀 계산
# ------------------------------------------------------------

first_app["def_team"] = [
    defense_team_from_game(
        g,
        h,
    )
    for g, h in zip(
        first_app["game_id"],
        first_app["half"],
    )
]


first_app = (
    first_app.sort_values(
        [
            "game_id",
            "_orig_idx",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "All appearances:",
    f"{len(first_app):,}",
)


print(
    "All games:",
    f"{first_app['game_id'].nunique():,}",
)


# ============================================================
# 5. BUILD RELIEF DECISIONS
# ============================================================

print_section(
    "5. BUILD RELIEF DECISIONS"
)


# 실제 Bandit decision은 불펜 투입 시점만 사용
relief = (
    first_app[
        ~first_app[
            "is_starter"
        ].astype(bool)
    ]
    .copy()
)


relief = (
    relief.sort_values(
        [
            "game_id",
            "_orig_idx",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Relief decisions:",
    f"{len(relief):,}",
)


print(
    "Games:",
    f"{relief['game_id'].nunique():,}",
)


# ============================================================
# 6. BUILD ACTUAL BULLPEN MAP
# ============================================================

print_section(
    "6. BUILD ACTUAL BULLPEN MAP"
)


# 실제 명단에 등재된 투수만 사용
listed_bp = (
    bp[
        bp[
            "is_listed_bullpen"
        ].astype(bool)
    ]
    .copy()
)


bullpen_map = (
    listed_bp
    .groupby(
        [
            "game_id",
            "team_code",
        ]
    )["pitcher_id"]
    .apply(set)
    .to_dict()
)


print(
    "Game-team bullpen sets:",
    f"{len(bullpen_map):,}",
)


# ============================================================
# 7. BUILD USED-BEFORE LOOKUP
# ============================================================

print_section(
    "7. BUILD USED-BEFORE LOOKUP"
)


# ------------------------------------------------------------
# 핵심 수정점
#
# 이전 코드:
#   relief만 돌면서 used를 쌓음
#   -> 선발이 used에 포함되지 않음
#
# 수정 코드:
#   모든 appearance(first_app)를 이용해서
#   현재 appearance 이전에 이미 나온 투수를 계산
#
# 따라서 첫 번째 불펜 교체 때
# 선발투수도 자동으로 후보군에서 제거된다.
# ------------------------------------------------------------

used_before = {}


for (
    game_id,
    team,
), g in first_app.groupby(
    [
        "game_id",
        "def_team",
    ],
    sort=False,
):

    g = (
        g.sort_values(
            "_orig_idx"
        )
    )

    used = set()

    for _, row in g.iterrows():

        key = (
            str(game_id),
            str(team),
            int(
                row[
                    "appearance_order"
                ]
            ),
        )

        # 현재 투수가 나오기 "직전"까지
        # 이미 등판했던 투수들
        used_before[key] = (
            set(used)
        )

        # 현재 투수는
        # 다음 decision부터 used
        used.add(
            str(
                row[
                    "pitcher_id"
                ]
            )
        )


print(
    "Used-before states:",
    f"{len(used_before):,}",
)


# ============================================================
# 8. ACTION SET COVERAGE
# ============================================================

print_section(
    "8. CHECK ACTUAL ACTION SET"
)


records = []


for _, row in relief.iterrows():

    game_id = str(
        row["game_id"]
    )

    team = str(
        row["def_team"]
    )

    actual = str(
        row["pitcher_id"]
    )

    appearance_order = int(
        row["appearance_order"]
    )


    # --------------------------------------------------------
    # 1. 해당 경기 / 수비팀 실제 명단
    # --------------------------------------------------------

    game_team_key = (
        game_id,
        team,
    )


    listed = set(
        bullpen_map.get(
            game_team_key,
            set(),
        )
    )


    # --------------------------------------------------------
    # 2. 그 시점 이전까지 이미 사용한 모든 투수
    #    선발 포함
    # --------------------------------------------------------

    decision_key = (
        game_id,
        team,
        appearance_order,
    )


    used = set(
        used_before.get(
            decision_key,
            set(),
        )
    )


    # --------------------------------------------------------
    # 3. 실제 Action Set
    #
    # 당일 등록 투수
    # -
    # 이미 등판한 투수
    # --------------------------------------------------------

    available = (
        listed
        - used
    )


    actual_in_listed = (
        actual in listed
    )


    actual_in_available = (
        actual in available
    )


    records.append(
        {
            "game_id":
                game_id,

            "team_code":
                team,

            "half":
                row["half"],

            "appearance_order":
                appearance_order,

            "pitcher_id":
                actual,

            "bullpen_source":
                row[
                    "bullpen_source"
                ],

            "listed_count":
                len(listed),

            "used_count":
                len(used),

            "available_count":
                len(available),

            "actual_in_listed":
                actual_in_listed,

            "actual_in_available":
                actual_in_available,

            "has_bullpen_data":
                len(listed) > 0,
        }
    )


result = pd.DataFrame(
    records
)


# ============================================================
# 9. REPORT FUNCTION
# ============================================================

def report_result(
    df,
    title,
):

    print_section(title)

    if len(df) == 0:

        print(
            "No decisions."
        )

        return


    print(
        "Total decisions:",
        f"{len(df):,}",
    )


    print(
        "Has bullpen data:",
        f"{df['has_bullpen_data'].mean() * 100:.2f}%"
    )


    print(
        "Actual in listed roster:",
        f"{df['actual_in_listed'].mean() * 100:.2f}%"
    )


    print(
        "Actual in available set:",
        f"{df['actual_in_available'].mean() * 100:.2f}%"
    )


    print(
        "Avg listed pitchers:",
        f"{df['listed_count'].mean():.2f}"
    )


    print(
        "Avg used pitchers:",
        f"{df['used_count'].mean():.2f}"
    )


    print(
        "Avg available pitchers:",
        f"{df['available_count'].mean():.2f}"
    )


    print()
    print(
        "Available count distribution:"
    )

    print(
        df["available_count"]
        .describe()
    )


# ============================================================
# 10. REPORT - ALL
# ============================================================

report_result(
    result,
    "9. RESULT - ALL RELIEF DECISIONS",
)


# ============================================================
# 11. REPORT - PREVIEW ONLY
# ============================================================

preview = (
    result[
        result[
            "bullpen_source"
        ] == "preview"
    ]
    .copy()
)


report_result(
    preview,
    "10. RESULT - PREVIEW ONLY",
)


# ============================================================
# 12. REPORT - ESTIMATED ONLY
# ============================================================

estimated = (
    result[
        result[
            "bullpen_source"
        ] == "estimated"
    ]
    .copy()
)


report_result(
    estimated,
    "11. RESULT - ESTIMATED ONLY",
)


# ============================================================
# 13. PREVIEW MISSING ACTUAL
# ============================================================

missing_actual = (
    preview[
        ~preview[
            "actual_in_available"
        ]
    ]
    .copy()
)


print_section(
    "12. PREVIEW MISSING ACTUAL EXAMPLES"
)


print(
    "Missing count:",
    f"{len(missing_actual):,}",
)


if len(missing_actual) > 0:

    print(
        missing_actual
        .head(30)
        .to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# listed에는 있었는데 available에서만 빠진 케이스
#
# 이런 행이 존재하면
# used 계산 문제 / 재등판 문제를 확인해야 한다.
# ------------------------------------------------------------

removed_by_used = (
    preview[
        (
            preview[
                "actual_in_listed"
            ]
        )
        &
        (
            ~preview[
                "actual_in_available"
            ]
        )
    ]
    .copy()
)


print_section(
    "13. ACTUAL REMOVED BY USED FILTER"
)


print(
    "Count:",
    f"{len(removed_by_used):,}",
)


if len(removed_by_used) > 0:

    print(
        removed_by_used
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 14. SAVE
# ============================================================

result.to_csv(
    OUTPUT_DIR
    / "actual_bullpen_diagnostics.csv",
    index=False,
)


preview.to_csv(
    OUTPUT_DIR
    / "actual_bullpen_preview_diagnostics.csv",
    index=False,
)


missing_actual.to_csv(
    OUTPUT_DIR
    / "actual_bullpen_preview_missing.csv",
    index=False,
)


removed_by_used.to_csv(
    OUTPUT_DIR
    / "actual_removed_by_used_filter.csv",
    index=False,
)


print_section("DONE")


print("Saved:")

print(
    " ",
    OUTPUT_DIR
    / "actual_bullpen_diagnostics.csv",
)

print(
    " ",
    OUTPUT_DIR
    / "actual_bullpen_preview_diagnostics.csv",
)

print(
    " ",
    OUTPUT_DIR
    / "actual_bullpen_preview_missing.csv",
)

print(
    " ",
    OUTPUT_DIR
    / "actual_removed_by_used_filter.csv",
)
