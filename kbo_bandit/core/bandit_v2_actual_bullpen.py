import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

PA_PATH = (
    "data_analysis/results/"
    "all_pa_enriched_v3.parquet"
)

HISTORY_PATH = (
    "data_analysis/results/"
    "all_pitcher_history.parquet"
)

BULLPEN_PATH = (
    "data_analysis/results/"
    "all_game_bullpen.parquet"
)

OUTPUT_DIR = Path(
    "outputs/bandit/core/v2_actual_bullpen"
)

ALPHA = 0.5
RIDGE = 1.0

# Step 2에서 비교 결과
# app_wpa가 wpa_per_pa보다 Bandit agreement가 높았으므로
# 이번 후보군 실험에서는 app_wpa로 고정
REWARD_MODE = "app_wpa"


# ============================================================
# Feature 정의
# ============================================================

SITUATION_FEATURES = [
    "inning",
    "out_count",
    "score_diff_defense",
    "is_base1",
    "is_base2",
    "is_base3",
    "is_home_defense",
    "bullpen_available_ratio",
    "n_pitchers_used",
]

PITCHER_FEATURES = [
    "prior_n_apps_log",
    "prior_pitcher_reward_mean",
    "prior_wpa_std",
    "prior_bb_rate",
    "prior_so_rate",
    "prior_hr_rate",
    "prior_avg_pa_per_app",
]

FEATURES = (
    SITUATION_FEATURES
    + PITCHER_FEATURES
)


# ============================================================
# Utility
# ============================================================

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


def defense_team_from_game(
    game_id,
    half,
):
    """
    game_id:
    YYYYMMDD{away}{home}{seq}{year}

    top = 원정 공격 -> 홈 수비
    bot = 홈 공격   -> 원정 수비
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
# 1. 데이터 로드
# ============================================================

def load_data():

    print_section("1. LOAD DATA")

    pa = pd.read_parquet(
        PA_PATH
    )

    history = pd.read_parquet(
        HISTORY_PATH
    )

    bullpen = pd.read_parquet(
        BULLPEN_PATH
    )

    print(
        f"all_pa_enriched_v3     : "
        f"{pa.shape}"
    )

    print(
        f"all_pitcher_history    : "
        f"{history.shape}"
    )

    print(
        f"all_game_bullpen       : "
        f"{bullpen.shape}"
    )

    # --------------------------------------------------------
    # 타입 통일
    # --------------------------------------------------------

    for df in [
        pa,
        history,
        bullpen,
    ]:

        df["game_id"] = (
            df["game_id"]
            .astype(str)
        )

        df["pitcher_id"] = (
            df["pitcher_id"]
            .apply(normalize_id)
        )

    bullpen["team_code"] = (
        bullpen["team_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # v3에는 bullpen state가 이미 있어야 함
    # --------------------------------------------------------

    required_bp_state = [
        "bullpen_available_ratio",
        "n_pitchers_used",
    ]

    missing = [
        c
        for c in required_bp_state
        if c not in pa.columns
    ]

    if missing:

        raise ValueError(
            "v3 PA 데이터에 bullpen state "
            f"컬럼이 없습니다: {missing}"
        )

    print(
        "bullpen state          : "
        "v3에 이미 포함"
    )

    return (
        pa,
        history,
        bullpen,
    )


# ============================================================
# 2. 등판 데이터셋 생성
# ============================================================

def build_decision_dataset(
    pa,
    history,
):

    print_section(
        "2. BUILD RELIEF DECISION DATASET"
    )

    required_pa = [
        "game_id",
        "_orig_idx",
        "half",
        "pitcher_id",
        "appearance_order",
        "inning",
        "out_count",
        "score_diff_attacker",
        "is_base1",
        "is_base2",
        "is_base3",
        "bullpen_available_ratio",
        "n_pitchers_used",
        "reward_wpa_computed",
        "bullpen_source",
        "is_starter",
    ]

    missing = [
        c
        for c in required_pa
        if c not in pa.columns
    ]

    if missing:
        raise ValueError(
            f"PA missing columns: {missing}"
        )

    required_hist = [
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
        "is_starter",
        "prior_n_apps",
        "prior_wpa_mean",
        "prior_wpa_std",
        "prior_bb_rate",
        "prior_so_rate",
        "prior_hr_rate",
        "prior_avg_pa_per_app",
    ]

    missing = [
        c
        for c in required_hist
        if c not in history.columns
    ]

    if missing:
        raise ValueError(
            "pitcher history missing "
            f"columns: {missing}"
        )

    # --------------------------------------------------------
    # appearance_order 정리
    # --------------------------------------------------------

    for df in [
        pa,
        history,
    ]:

        df["appearance_order"] = (
            pd.to_numeric(
                df[
                    "appearance_order"
                ],
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

    history = (
        history.dropna(
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

    history["appearance_order"] = (
        history[
            "appearance_order"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # appearance 식별 key
    # --------------------------------------------------------

    keys = [
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
    ]

    # ========================================================
    # Option B WPA 재집계
    #
    # 기존 history.app_wpa는 Option A임을
    # Step 1에서 100% 확인했기 때문에 사용하지 않는다.
    # ========================================================

    option_b_app = (
        pa.groupby(
            keys,
            as_index=False,
        )
        .agg(
            app_wpa_optionB=(
                "reward_wpa_computed",
                "sum",
            ),
            n_pa_optionB=(
                "reward_wpa_computed",
                "size",
            ),
        )
    )

    # --------------------------------------------------------
    # 모든 appearance의 첫 PA
    #
    # 선발도 남겨둔다.
    # later: used-before 계산에 필요
    # --------------------------------------------------------

    pa = pa.sort_values(
        [
            "game_id",
            "_orig_idx",
        ]
    )

    first_cols = (
        keys
        + [
            "_orig_idx",
            "inning",
            "out_count",
            "score_diff_attacker",
            "is_base1",
            "is_base2",
            "is_base3",
            "bullpen_available_ratio",
            "n_pitchers_used",
            "bullpen_source",
            "is_starter",
        ]
    )

    first_app = (
        pa[first_cols]
        .groupby(
            keys,
            as_index=False,
        )
        .first()
    )

    # --------------------------------------------------------
    # 수비팀
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # history + 경기 시작상황
    # --------------------------------------------------------

    # first_app의 is_starter는 history와 겹치므로
    # merge 전에 제거
    first_for_merge = (
        first_app.drop(
            columns=[
                "is_starter"
            ]
        )
    )

    decisions = (
        history.merge(
            first_for_merge,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    )

    decisions = (
        decisions.merge(
            option_b_app,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Relief만 Bandit decision으로 사용
    # --------------------------------------------------------

    decisions = (
        decisions[
            ~decisions[
                "is_starter"
            ].astype(bool)
        ]
        .copy()
    )

    # --------------------------------------------------------
    # 날짜
    # --------------------------------------------------------

    decisions["date"] = (
        pd.to_datetime(
            decisions[
                "game_id"
            ].str[:8],
            format="%Y%m%d",
        )
    )

    # --------------------------------------------------------
    # 투수 관점 상황
    # --------------------------------------------------------

    decisions[
        "score_diff_defense"
    ] = (
        -decisions[
            "score_diff_attacker"
        ]
    )

    decisions[
        "is_home_defense"
    ] = (
        decisions["half"]
        == "top"
    ).astype(int)

    for c in [
        "is_base1",
        "is_base2",
        "is_base3",
    ]:

        decisions[c] = (
            decisions[c]
            .astype(int)
        )

    # ========================================================
    # Reward
    # ========================================================

    decisions[
        "reward_app_wpa"
    ] = (
        -decisions[
            "app_wpa_optionB"
        ]
    )

    decisions[
        "reward_wpa_per_pa"
    ] = (
        -decisions[
            "app_wpa_optionB"
        ]
        /
        decisions[
            "n_pa_optionB"
        ]
    )

    if REWARD_MODE == "app_wpa":

        decisions["reward"] = (
            decisions[
                "reward_app_wpa"
            ]
        )

    elif (
        REWARD_MODE
        == "wpa_per_pa"
    ):

        decisions["reward"] = (
            decisions[
                "reward_wpa_per_pa"
            ]
        )

    else:

        raise ValueError(
            f"Unknown REWARD_MODE: "
            f"{REWARD_MODE}"
        )

    # --------------------------------------------------------
    # prior WPA는 공격팀 관점이므로
    # 투수 관점으로 반전
    # --------------------------------------------------------

    decisions[
        "prior_pitcher_reward_mean"
    ] = (
        -decisions[
            "prior_wpa_mean"
        ]
    )

    decisions[
        "prior_n_apps_log"
    ] = (
        np.log1p(
            decisions[
                "prior_n_apps"
            ].fillna(0)
        )
    )

    decisions = (
        decisions.sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
        .reset_index(
            drop=True
        )
    )

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
        f"Relief decisions : "
        f"{len(decisions):,}"
    )

    print(
        f"Games            : "
        f"{decisions['game_id'].nunique():,}"
    )

    print(
        f"Pitchers         : "
        f"{decisions['pitcher_id'].nunique():,}"
    )

    return (
        decisions,
        first_app,
    )


# ============================================================
# 3. 실제 bullpen map
# ============================================================

def build_bullpen_map(
    bullpen,
):

    print_section(
        "3. BUILD ACTUAL BULLPEN MAP"
    )

    listed = (
        bullpen[
            bullpen[
                "is_listed_bullpen"
            ].astype(bool)
        ]
        .copy()
    )

    bullpen_map = (
        listed
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
        "Game-team bullpen sets : "
        f"{len(bullpen_map):,}"
    )

    return bullpen_map


# ============================================================
# 4. 현재 decision 이전 사용 투수 map
# ============================================================

def build_used_before_map(
    first_app,
):

    print_section(
        "4. BUILD USED-BEFORE MAP"
    )

    used_before = {}

    for (
        game_id,
        team,
    ), group in first_app.groupby(
        [
            "game_id",
            "def_team",
        ],
        sort=False,
    ):

        group = (
            group.sort_values(
                "_orig_idx"
            )
        )

        used = set()

        for _, row in (
            group.iterrows()
        ):

            key = (
                str(game_id),
                str(team),
                int(
                    row[
                        "appearance_order"
                    ]
                ),
            )

            # 현재 투수 투입 직전까지
            # 이미 등판한 모든 투수
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
        "Used-before states      : "
        f"{len(used_before):,}"
    )

    return used_before


# ============================================================
# 5. chronological split
# ============================================================

def chronological_split(df):

    print_section(
        "5. CHRONOLOGICAL SPLIT"
    )

    games = (
        df[
            [
                "game_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "date",
                "game_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    n = len(games)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train_games = set(
        games.iloc[
            :train_end
        ]["game_id"]
    )

    val_games = set(
        games.iloc[
            train_end:val_end
        ]["game_id"]
    )

    test_games = set(
        games.iloc[
            val_end:
        ]["game_id"]
    )

    train = (
        df[
            df[
                "game_id"
            ].isin(
                train_games
            )
        ]
        .copy()
    )

    val = (
        df[
            df[
                "game_id"
            ].isin(
                val_games
            )
        ]
        .copy()
    )

    test = (
        df[
            df[
                "game_id"
            ].isin(
                test_games
            )
        ]
        .copy()
    )

    print(
        f"Train : {len(train):,}"
    )

    print(
        f"Val   : {len(val):,}"
    )

    print(
        f"Test  : {len(test):,}"
    )

    print()

    print(
        "Train dates : "
        f"{train['date'].min().date()} "
        "~ "
        f"{train['date'].max().date()}"
    )

    print(
        "Val dates   : "
        f"{val['date'].min().date()} "
        "~ "
        f"{val['date'].max().date()}"
    )

    print(
        "Test dates  : "
        f"{test['date'].min().date()} "
        "~ "
        f"{test['date'].max().date()}"
    )

    return (
        train,
        val,
        test,
    )


# ============================================================
# 6. Feature preprocessing
# ============================================================

class FeatureProcessor:

    def __init__(self):

        self.medians = {}
        self.scaler = (
            StandardScaler()
        )

    def fit(self, train):

        x = (
            train[
                FEATURES
            ]
            .copy()
        )

        for col in FEATURES:

            med = (
                x[col]
                .median()
            )

            if pd.isna(med):
                med = 0.0

            self.medians[
                col
            ] = float(med)

            x[col] = (
                x[col]
                .fillna(med)
            )

        self.scaler.fit(x)

    def transform_frame(
        self,
        df,
    ):

        x = (
            df[
                FEATURES
            ]
            .copy()
        )

        for col in FEATURES:

            x[col] = (
                x[col]
                .fillna(
                    self.medians[
                        col
                    ]
                )
            )

        z = (
            self.scaler
            .transform(x)
        )

        return np.column_stack(
            [
                np.ones(
                    len(z)
                ),
                z,
            ]
        )

    def transform_values(
        self,
        values,
    ):

        arr = np.asarray(
            [
                (
                    self.medians[col]
                    if pd.isna(
                        values[col]
                    )
                    else values[col]
                )
                for col in FEATURES
            ],
            dtype=float,
        )

        z = (
            (
                arr
                - self.scaler.mean_
            )
            /
            self.scaler.scale_
        )

        return np.concatenate(
            [
                np.array(
                    [1.0]
                ),
                z,
            ]
        )


# ============================================================
# 7. Shared Linear UCB
# ============================================================

class LinearUCB:

    def __init__(
        self,
        dim,
        alpha=0.5,
        ridge=1.0,
    ):

        self.dim = dim
        self.alpha = alpha
        self.ridge = ridge

        self.A_inv = None
        self.theta = None

    def fit(
        self,
        X,
        reward,
    ):

        A = (
            self.ridge
            * np.eye(
                self.dim
            )
            + X.T @ X
        )

        b = (
            X.T
            @ reward
        )

        self.A_inv = (
            np.linalg.inv(A)
        )

        self.theta = (
            self.A_inv
            @ b
        )

    def score(
        self,
        x,
    ):

        mean = float(
            self.theta
            @ x
        )

        uncertainty = float(
            np.sqrt(
                max(
                    x
                    @ self.A_inv
                    @ x,
                    0.0,
                )
            )
        )

        ucb = (
            mean
            + self.alpha
            * uncertainty
        )

        return (
            ucb,
            mean,
            uncertainty,
        )


# ============================================================
# 8. Pitcher profile
# ============================================================

def profile_from_row(row):

    return {
        "pitcher_id":
            str(
                row[
                    "pitcher_id"
                ]
            ),

        "prior_n_apps":
            row[
                "prior_n_apps"
            ],

        "prior_n_apps_log":
            row[
                "prior_n_apps_log"
            ],

        "prior_pitcher_reward_mean":
            row[
                "prior_pitcher_reward_mean"
            ],

        "prior_wpa_std":
            row[
                "prior_wpa_std"
            ],

        "prior_bb_rate":
            row[
                "prior_bb_rate"
            ],

        "prior_so_rate":
            row[
                "prior_so_rate"
            ],

        "prior_hr_rate":
            row[
                "prior_hr_rate"
            ],

        "prior_avg_pa_per_app":
            row[
                "prior_avg_pa_per_app"
            ],
    }


def blank_profile(
    pitcher_id,
):
    """
    과거 relief profile이 없는 투수.

    후보군에서 제거하지 않고,
    결측치는 processor의 train median으로 처리한다.
    """

    return {
        "pitcher_id":
            str(pitcher_id),

        "prior_n_apps":
            0.0,

        "prior_n_apps_log":
            0.0,

        "prior_pitcher_reward_mean":
            np.nan,

        "prior_wpa_std":
            np.nan,

        "prior_bb_rate":
            np.nan,

        "prior_so_rate":
            np.nan,

        "prior_hr_rate":
            np.nan,

        "prior_avg_pa_per_app":
            np.nan,
    }


def candidate_feature_values(
    decision,
    profile,
):

    return {
        # 현재 경기 상황
        "inning":
            decision[
                "inning"
            ],

        "out_count":
            decision[
                "out_count"
            ],

        "score_diff_defense":
            decision[
                "score_diff_defense"
            ],

        "is_base1":
            decision[
                "is_base1"
            ],

        "is_base2":
            decision[
                "is_base2"
            ],

        "is_base3":
            decision[
                "is_base3"
            ],

        "is_home_defense":
            decision[
                "is_home_defense"
            ],

        "bullpen_available_ratio":
            decision[
                "bullpen_available_ratio"
            ],

        "n_pitchers_used":
            decision[
                "n_pitchers_used"
            ],

        # 후보 투수 과거 정보
        "prior_n_apps_log":
            profile[
                "prior_n_apps_log"
            ],

        "prior_pitcher_reward_mean":
            profile[
                "prior_pitcher_reward_mean"
            ],

        "prior_wpa_std":
            profile[
                "prior_wpa_std"
            ],

        "prior_bb_rate":
            profile[
                "prior_bb_rate"
            ],

        "prior_so_rate":
            profile[
                "prior_so_rate"
            ],

        "prior_hr_rate":
            profile[
                "prior_hr_rate"
            ],

        "prior_avg_pa_per_app":
            profile[
                "prior_avg_pa_per_app"
            ],
    }


# ============================================================
# 9. Model training
# ============================================================

def train_bandit(
    train,
    processor,
):

    print_section(
        "6. TRAIN LINEAR UCB"
    )

    processor.fit(train)

    X = (
        processor
        .transform_frame(
            train
        )
    )

    y = (
        train[
            "reward"
        ]
        .to_numpy(
            dtype=float
        )
    )

    model = LinearUCB(
        dim=X.shape[1],
        alpha=ALPHA,
        ridge=RIDGE,
    )

    model.fit(
        X,
        y,
    )

    print(
        "Training samples : "
        f"{len(train):,}"
    )

    print(
        "Feature dim      : "
        f"{X.shape[1]}"
    )

    print(
        "Alpha            : "
        f"{ALPHA}"
    )

    print(
        "Reward           : "
        f"{REWARD_MODE}"
    )

    return model


# ============================================================
# 10. Profile 초기화
# ============================================================

def seed_profiles(
    history_df,
):

    profiles = (
        defaultdict(dict)
    )

    hist = (
        history_df
        .sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
    )

    for _, row in (
        hist.iterrows()
    ):

        team = str(
            row[
                "def_team"
            ]
        )

        pid = str(
            row[
                "pitcher_id"
            ]
        )

        profiles[
            team
        ][pid] = (
            profile_from_row(
                row
            )
        )

    return profiles


# ============================================================
# 11. 실제 후보군
# ============================================================

def get_actual_candidates(
    decision,
    bullpen_map,
    used_before_map,
    profiles,
):

    game_id = str(
        decision[
            "game_id"
        ]
    )

    team = str(
        decision[
            "def_team"
        ]
    )

    actual = str(
        decision[
            "pitcher_id"
        ]
    )

    appearance_order = int(
        decision[
            "appearance_order"
        ]
    )

    # --------------------------------------------------------
    # 해당 경기 실제 bullpen 명단
    # --------------------------------------------------------

    listed = set(
        bullpen_map.get(
            (
                game_id,
                team,
            ),
            set(),
        )
    )

    # --------------------------------------------------------
    # 이미 등판한 모든 투수
    # 선발 포함
    # --------------------------------------------------------

    used = set(
        used_before_map.get(
            (
                game_id,
                team,
                appearance_order,
            ),
            set(),
        )
    )

    available_ids = (
        listed
        - used
    )

    actual_in_candidates = (
        actual
        in available_ids
    )

    candidates = []

    for pid in sorted(
        available_ids
    ):

        # 과거 profile이 있으면 사용
        if (
            pid
            in profiles[
                team
            ]
        ):

            profile = (
                profiles[
                    team
                ][pid]
            )

        # 실제 투수이며
        # 과거 relief profile이 없다면
        # 현재 decision의 leakage-free prior 사용 가능
        elif pid == actual:

            profile = (
                profile_from_row(
                    decision
                )
            )

        # 과거 profile 없는 후보도
        # Action Set에서는 제거하지 않는다.
        else:

            profile = (
                blank_profile(
                    pid
                )
            )

        candidates.append(
            profile
        )

    return (
        candidates,
        actual_in_candidates,
        len(listed),
        len(used),
    )


# ============================================================
# 12. Evaluation
# ============================================================

def evaluate(
    name,
    model,
    processor,
    seed_df,
    eval_df,
    bullpen_map,
    used_before_map,
):

    print_section(
        f"7. EVALUATE {name}"
    )

    profiles = (
        seed_profiles(
            seed_df
        )
    )

    # ========================================================
    # Task 2 main experiment:
    # 실제 명단이 존재하는 preview 구간만 평가
    # ========================================================

    eval_df = (
        eval_df[
            eval_df[
                "bullpen_source"
            ]
            == "preview"
        ]
        .copy()
    )

    eval_df = (
        eval_df.sort_values(
            [
                "date",
                "game_id",
                "_orig_idx",
            ]
        )
    )

    total_decisions = (
        len(eval_df)
    )

    candidate_hits = 0
    bullpen_data_hits = 0

    rows = []

    for _, decision in (
        eval_df.iterrows()
    ):

        actual = str(
            decision[
                "pitcher_id"
            ]
        )

        team = str(
            decision[
                "def_team"
            ]
        )

        (
            candidates,
            actual_in_candidates,
            listed_count,
            used_count,
        ) = get_actual_candidates(
            decision,
            bullpen_map,
            used_before_map,
            profiles,
        )

        if listed_count > 0:
            bullpen_data_hits += 1

        if actual_in_candidates:
            candidate_hits += 1

        # ----------------------------------------------------
        # 실제 행동이 후보군에 없으면
        # agreement 평가 불가능
        # ----------------------------------------------------

        if (
            actual_in_candidates
            and len(candidates) >= 2
        ):

            scored = []

            for candidate in (
                candidates
            ):

                values = (
                    candidate_feature_values(
                        decision,
                        candidate,
                    )
                )

                x = (
                    processor
                    .transform_values(
                        values
                    )
                )

                (
                    score,
                    mean,
                    uncertainty,
                ) = model.score(x)

                scored.append(
                    {
                        "pitcher_id":
                            candidate[
                                "pitcher_id"
                            ],

                        "score":
                            score,

                        "pred_mean":
                            mean,

                        "uncertainty":
                            uncertainty,

                        "prior_n_apps":
                            candidate[
                                "prior_n_apps"
                            ],
                    }
                )

            # ------------------------------------------------
            # Bandit 추천
            # ------------------------------------------------

            scored.sort(
                key=lambda x: (
                    x["score"]
                ),
                reverse=True,
            )

            recommended = (
                scored[0][
                    "pitcher_id"
                ]
            )

            # ------------------------------------------------
            # Heuristic:
            # 이전 등판 수 가장 많은 투수
            # ------------------------------------------------

            heuristic = max(
                scored,
                key=lambda x: (
                    -1
                    if pd.isna(
                        x[
                            "prior_n_apps"
                        ]
                    )
                    else x[
                        "prior_n_apps"
                    ]
                ),
            )[
                "pitcher_id"
            ]

            rows.append(
                {
                    "game_id":
                        decision[
                            "game_id"
                        ],

                    "date":
                        decision[
                            "date"
                        ],

                    "def_team":
                        team,

                    "inning":
                        decision[
                            "inning"
                        ],

                    "appearance_order":
                        decision[
                            "appearance_order"
                        ],

                    "actual_pitcher":
                        actual,

                    "bandit_pitcher":
                        recommended,

                    "heuristic_pitcher":
                        heuristic,

                    "n_candidates":
                        len(
                            candidates
                        ),

                    "listed_count":
                        listed_count,

                    "used_count":
                        used_count,

                    "bandit_match":
                        (
                            recommended
                            == actual
                        ),

                    "heuristic_match":
                        (
                            heuristic
                            == actual
                        ),

                    "random_expected_match":
                        (
                            1.0
                            /
                            len(
                                candidates
                            )
                        ),

                    "logged_reward":
                        decision[
                            "reward"
                        ],
                }
            )

        # ----------------------------------------------------
        # 시간이 한 단계 진행됐으므로
        # 실제 선택 투수 profile 갱신
        #
        # v1과 동일한 profile update 구조를 유지한다.
        # 이번 실험에서는 후보군만 변경하기 위함.
        # ----------------------------------------------------

        profiles[
            team
        ][actual] = (
            profile_from_row(
                decision
            )
        )

    result = (
        pd.DataFrame(
            rows
        )
    )

    candidate_coverage = (
        candidate_hits
        /
        total_decisions
        if total_decisions > 0
        else 0.0
    )

    bullpen_data_coverage = (
        bullpen_data_hits
        /
        total_decisions
        if total_decisions > 0
        else 0.0
    )

    if result.empty:

        print(
            "평가 가능한 decision이 "
            "없습니다."
        )

        return (
            result,
            {},
        )

    eval_coverage = (
        len(result)
        /
        total_decisions
    )

    avg_candidates = (
        result[
            "n_candidates"
        ]
        .mean()
    )

    random_agreement = (
        result[
            "random_expected_match"
        ]
        .mean()
    )

    heuristic_agreement = (
        result[
            "heuristic_match"
        ]
        .mean()
    )

    bandit_agreement = (
        result[
            "bandit_match"
        ]
        .mean()
    )

    summary = {
        "split":
            name,

        "candidate_source":
            "actual_bullpen_preview",

        "reward_mode":
            REWARD_MODE,

        "total_preview_relief_decisions":
            int(
                total_decisions
            ),

        "evaluated_decisions":
            int(
                len(result)
            ),

        "bullpen_data_coverage":
            float(
                bullpen_data_coverage
            ),

        "actual_candidate_coverage":
            float(
                candidate_coverage
            ),

        "evaluation_coverage":
            float(
                eval_coverage
            ),

        "avg_candidates":
            float(
                avg_candidates
            ),

        "random_expected_agreement":
            float(
                random_agreement
            ),

        "heuristic_agreement":
            float(
                heuristic_agreement
            ),

        "bandit_agreement":
            float(
                bandit_agreement
            ),
    }

    print(
        "Total preview relief decisions : "
        f"{total_decisions:,}"
    )

    print(
        "Bullpen data coverage           : "
        f"{bullpen_data_coverage * 100:.2f}%"
    )

    print(
        "Actual candidate coverage       : "
        f"{candidate_coverage * 100:.2f}%"
    )

    print(
        "Evaluated decisions             : "
        f"{len(result):,}"
    )

    print(
        "Evaluation coverage             : "
        f"{eval_coverage * 100:.2f}%"
    )

    print(
        "Avg candidates                  : "
        f"{avg_candidates:.2f}"
    )

    print()

    print(
        "Random expected agreement       : "
        f"{random_agreement * 100:.2f}%"
    )

    print(
        "Heuristic agreement             : "
        f"{heuristic_agreement * 100:.2f}%"
    )

    print(
        "Bandit agreement                : "
        f"{bandit_agreement * 100:.2f}%"
    )

    return (
        result,
        summary,
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
        "- ACTUAL BULLPEN"
    )

    print(
        f"REWARD MODE     : "
        f"{REWARD_MODE}"
    )

    print(
        "CANDIDATE SOURCE: "
        "actual bullpen / preview"
    )

    print(
        "=" * 72
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        pa,
        history,
        bullpen,
    ) = load_data()

    (
        decisions,
        first_app,
    ) = build_decision_dataset(
        pa,
        history,
    )

    bullpen_map = (
        build_bullpen_map(
            bullpen
        )
    )

    used_before_map = (
        build_used_before_map(
            first_app
        )
    )

    (
        train,
        val,
        test,
    ) = chronological_split(
        decisions
    )

    processor = (
        FeatureProcessor()
    )

    model = train_bandit(
        train,
        processor,
    )

    # ========================================================
    # Validation
    # ========================================================

    (
        val_result,
        val_summary,
    ) = evaluate(
        "VALIDATION",
        model,
        processor,
        seed_df=train,
        eval_df=val,
        bullpen_map=bullpen_map,
        used_before_map=(
            used_before_map
        ),
    )

    # ========================================================
    # Test
    #
    # test 시작 전에는
    # train + validation 정보 사용 가능
    # ========================================================

    test_seed = pd.concat(
        [
            train,
            val,
        ],
        ignore_index=True,
    )

    (
        test_result,
        test_summary,
    ) = evaluate(
        "TEST",
        model,
        processor,
        seed_df=test_seed,
        eval_df=test,
        bullpen_map=bullpen_map,
        used_before_map=(
            used_before_map
        ),
    )

    # ========================================================
    # Save
    # ========================================================

    val_result.to_csv(
        OUTPUT_DIR
        / "validation_decisions.csv",
        index=False,
    )

    test_result.to_csv(
        OUTPUT_DIR
        / "test_decisions.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "alpha":
                    ALPHA,

                "ridge":
                    RIDGE,

                "reward_mode":
                    REWARD_MODE,

                "candidate_source":
                    (
                        "actual_bullpen_"
                        "preview"
                    ),

                "features":
                    FEATURES,

                "validation":
                    val_summary,

                "test":
                    test_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_section(
        "DONE"
    )

    print("Saved:")

    print(
        " ",
        OUTPUT_DIR
        / "validation_decisions.csv",
    )

    print(
        " ",
        OUTPUT_DIR
        / "test_decisions.csv",
    )

    print(
        " ",
        OUTPUT_DIR
        / "summary.json",
    )


if __name__ == "__main__":
    main()
