import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

PA_PATH = "data_analysis/results/all_pa_enriched.parquet"
BULLPEN_PATH = "data_analysis/results/all_pa_bullpen_state.parquet"
HISTORY_PATH = "data_analysis/results/all_pitcher_history.parquet"

OUTPUT_DIR = Path("bandit_output")

ALPHA = 0.5
RIDGE = 1.0

# 실제 bullpen roster가 없으므로
# 최근 같은 팀에서 relief appearance가 있었던 투수를 후보로 근사
LOOKBACK_DAYS = 30
MAX_CANDIDATES = 10


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

FEATURES = SITUATION_FEATURES + PITCHER_FEATURES


# ============================================================
# Utility
# ============================================================

def defense_team_from_game(game_id: str, half: str) -> str:
    """
    game_id:
    YYYYMMDD + away(2) + home(2) + ...
    """

    game_id = str(game_id)

    away = game_id[8:10]
    home = game_id[10:12]

    # top = 원정 공격 → 홈 수비
    if half == "top":
        return home

    # bot = 홈 공격 → 원정 수비
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

    pa = pd.read_parquet(PA_PATH)
    history = pd.read_parquet(HISTORY_PATH)

    print(f"all_pa_enriched        : {pa.shape}")
    print(f"all_pitcher_history    : {history.shape}")

    # --------------------------------------------------------
    # bullpen state 컬럼이 enriched 안에 이미 있으면 그대로 사용
    # 없으면 별도 파일에서 가져옴
    # --------------------------------------------------------

    bullpen_cols = [
        "bullpen_available_ratio",
        "n_pitchers_used",
    ]

    missing_bp = [
        c for c in bullpen_cols
        if c not in pa.columns
    ]

    if missing_bp:
        bullpen = pd.read_parquet(BULLPEN_PATH)

        print(f"all_pa_bullpen_state   : {bullpen.shape}")

        merge_keys = ["game_id", "_orig_idx"]

        for key in merge_keys:
            if key not in bullpen.columns:
                raise ValueError(
                    f"{BULLPEN_PATH}에 merge key {key}가 없습니다."
                )

        cols = merge_keys + missing_bp

        pa = pa.merge(
            bullpen[cols],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
    else:
        print(
            "bullpen state          : "
            "all_pa_enriched에 이미 포함되어 있음"
        )

    return pa, history


# ============================================================
# 2. 등판 시작 상황 + 투수 history 결합
# ============================================================

def build_decision_dataset(pa, history):
    print_section("2. BUILD RELIEF DECISION DATASET")

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
    ]

    missing = [
        c for c in required_pa
        if c not in pa.columns
    ]

    if missing:
        raise ValueError(
            f"PA 데이터에 필요한 컬럼이 없습니다: {missing}"
        )

    required_hist = [
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
        "app_wpa",
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
        c for c in required_hist
        if c not in history.columns
    ]

    if missing:
        raise ValueError(
            f"pitcher history에 필요한 컬럼이 없습니다: {missing}"
        )

    # 타입 통일
    for df in [pa, history]:
        df["game_id"] = df["game_id"].astype(str)
        df["pitcher_id"] = (
            df["pitcher_id"]
            .fillna("__MISSING__")
            .astype(str)
        )

        df["appearance_order"] = (
            pd.to_numeric(
                df["appearance_order"],
                errors="coerce",
            )
        )

    pa = pa.dropna(
        subset=["appearance_order"]
    ).copy()

    history = history.dropna(
        subset=["appearance_order"]
    ).copy()

    pa["appearance_order"] = pa["appearance_order"].astype(int)
    history["appearance_order"] = history["appearance_order"].astype(int)

    # --------------------------------------------------------
    # 각 appearance의 첫 PA
    # = 투수가 투입되었을 때의 경기 상황
    # --------------------------------------------------------

    keys = [
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
    ]

    pa = pa.sort_values(
        ["game_id", "_orig_idx"]
    )

    state_cols = keys + [
        "_orig_idx",
        "inning",
        "out_count",
        "score_diff_attacker",
        "is_base1",
        "is_base2",
        "is_base3",
        "bullpen_available_ratio",
        "n_pitchers_used",
    ]

    first_pa = (
        pa[state_cols]
        .groupby(keys, as_index=False)
        .first()
    )

    # history가 appearance-level source of truth
    decisions = history.merge(
        first_pa,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Relief appearance만 사용
    # --------------------------------------------------------

    decisions = decisions[
        ~decisions["is_starter"].astype(bool)
    ].copy()

    # 날짜
    decisions["date"] = pd.to_datetime(
        decisions["game_id"].str[:8],
        format="%Y%m%d",
    )

    # 수비팀
    decisions["def_team"] = [
        defense_team_from_game(g, h)
        for g, h in zip(
            decisions["game_id"],
            decisions["half"],
        )
    ]

    # --------------------------------------------------------
    # 투수 관점으로 feature / reward 정렬
    # --------------------------------------------------------

    # 공격팀 점수차 → 수비팀 점수차
    decisions["score_diff_defense"] = (
        -decisions["score_diff_attacker"]
    )

    # top이면 홈팀 수비
    decisions["is_home_defense"] = (
        decisions["half"] == "top"
    ).astype(int)

    for c in [
        "is_base1",
        "is_base2",
        "is_base3",
    ]:
        decisions[c] = decisions[c].astype(int)

    # app_wpa = 공격팀 관점
    # 투수 reward = 부호 반전
    decisions["reward"] = -decisions["app_wpa"]

    # prior_wpa_mean도 공격팀 관점
    decisions["prior_pitcher_reward_mean"] = (
        -decisions["prior_wpa_mean"]
    )

    decisions["prior_n_apps_log"] = np.log1p(
        decisions["prior_n_apps"].fillna(0)
    )

    decisions = decisions.sort_values(
        [
            "date",
            "game_id",
            "appearance_order",
        ]
    ).reset_index(drop=True)

    print(
        f"Relief decisions : {len(decisions):,}"
    )
    print(
        f"Games            : "
        f"{decisions['game_id'].nunique():,}"
    )
    print(
        f"Pitchers         : "
        f"{decisions['pitcher_id'].nunique():,}"
    )

    return decisions


# ============================================================
# 3. chronological split
# ============================================================

def chronological_split(df):
    print_section("3. CHRONOLOGICAL SPLIT")

    games = (
        df[["game_id", "date"]]
        .drop_duplicates()
        .sort_values(
            ["date", "game_id"]
        )
        .reset_index(drop=True)
    )

    n = len(games)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_games = set(
        games.iloc[:train_end]["game_id"]
    )

    val_games = set(
        games.iloc[
            train_end:val_end
        ]["game_id"]
    )

    test_games = set(
        games.iloc[val_end:]["game_id"]
    )

    train = df[
        df["game_id"].isin(train_games)
    ].copy()

    val = df[
        df["game_id"].isin(val_games)
    ].copy()

    test = df[
        df["game_id"].isin(test_games)
    ].copy()

    print(f"Train : {len(train):,}")
    print(f"Val   : {len(val):,}")
    print(f"Test  : {len(test):,}")

    print()
    print(
        f"Train dates : "
        f"{train['date'].min().date()} "
        f"~ {train['date'].max().date()}"
    )

    print(
        f"Val dates   : "
        f"{val['date'].min().date()} "
        f"~ {val['date'].max().date()}"
    )

    print(
        f"Test dates  : "
        f"{test['date'].min().date()} "
        f"~ {test['date'].max().date()}"
    )

    return train, val, test


# ============================================================
# 4. Feature preprocessing
# ============================================================

class FeatureProcessor:
    def __init__(self):
        self.medians = {}
        self.scaler = StandardScaler()

    def fit(self, train):
        x = train[FEATURES].copy()

        for col in FEATURES:
            med = x[col].median()

            if pd.isna(med):
                med = 0.0

            self.medians[col] = float(med)

            x[col] = x[col].fillna(med)

        self.scaler.fit(x)

    def transform_frame(self, df):
        x = df[FEATURES].copy()

        for col in FEATURES:
            x[col] = x[col].fillna(
                self.medians[col]
            )

        z = self.scaler.transform(x)

        # intercept 추가
        return np.column_stack(
            [
                np.ones(len(z)),
                z,
            ]
        )

    def transform_values(self, values):
        arr = np.asarray(
            [
                self.medians[col]
                if pd.isna(values[col])
                else values[col]
                for col in FEATURES
            ],
            dtype=float,
        )

        z = (
            arr - self.scaler.mean_
        ) / self.scaler.scale_

        return np.concatenate(
            [
                np.array([1.0]),
                z,
            ]
        )


# ============================================================
# 5. Shared Linear UCB
# ============================================================

class LinearUCB:
    """
    후보마다 별도 모델을 만드는 Disjoint arm 방식 대신,
    context + pitcher feature에 대해 하나의 shared linear model 사용.

    따라서 신인 / test에서 처음 보는 pitcher_id도
    prior_* 특성을 이용해서 점수 계산 가능.
    """

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

    def fit(self, X, reward):
        A = (
            self.ridge * np.eye(self.dim)
            + X.T @ X
        )

        b = X.T @ reward

        self.A_inv = np.linalg.inv(A)
        self.theta = self.A_inv @ b

    def score(self, x):
        mean = float(
            self.theta @ x
        )

        uncertainty = float(
            np.sqrt(
                max(
                    x @ self.A_inv @ x,
                    0.0,
                )
            )
        )

        ucb = (
            mean
            + self.alpha * uncertainty
        )

        return ucb, mean, uncertainty


# ============================================================
# 6. Candidate feature
# ============================================================

def profile_from_row(row):
    return {
        "pitcher_id": str(row["pitcher_id"]),
        "last_seen_date": row["date"],

        "prior_n_apps": row["prior_n_apps"],
        "prior_n_apps_log": row["prior_n_apps_log"],

        "prior_pitcher_reward_mean":
            row["prior_pitcher_reward_mean"],

        "prior_wpa_std":
            row["prior_wpa_std"],

        "prior_bb_rate":
            row["prior_bb_rate"],

        "prior_so_rate":
            row["prior_so_rate"],

        "prior_hr_rate":
            row["prior_hr_rate"],

        "prior_avg_pa_per_app":
            row["prior_avg_pa_per_app"],
    }


def candidate_feature_values(
    decision,
    profile,
):
    return {
        # 현재 경기 상황
        "inning":
            decision["inning"],

        "out_count":
            decision["out_count"],

        "score_diff_defense":
            decision["score_diff_defense"],

        "is_base1":
            decision["is_base1"],

        "is_base2":
            decision["is_base2"],

        "is_base3":
            decision["is_base3"],

        "is_home_defense":
            decision["is_home_defense"],

        "bullpen_available_ratio":
            decision["bullpen_available_ratio"],

        "n_pitchers_used":
            decision["n_pitchers_used"],

        # 후보 투수 특성
        "prior_n_apps_log":
            profile["prior_n_apps_log"],

        "prior_pitcher_reward_mean":
            profile["prior_pitcher_reward_mean"],

        "prior_wpa_std":
            profile["prior_wpa_std"],

        "prior_bb_rate":
            profile["prior_bb_rate"],

        "prior_so_rate":
            profile["prior_so_rate"],

        "prior_hr_rate":
            profile["prior_hr_rate"],

        "prior_avg_pa_per_app":
            profile["prior_avg_pa_per_app"],
    }


# ============================================================
# 7. Model training
# ============================================================

def train_bandit(train, processor):
    print_section("4. TRAIN LINEAR UCB")

    processor.fit(train)

    X = processor.transform_frame(train)

    y = train["reward"].to_numpy(
        dtype=float
    )

    model = LinearUCB(
        dim=X.shape[1],
        alpha=ALPHA,
        ridge=RIDGE,
    )

    model.fit(X, y)

    print(
        f"Training samples : {len(train):,}"
    )

    print(
        f"Feature dim      : {X.shape[1]}"
    )

    print(
        f"Alpha            : {ALPHA}"
    )

    return model


# ============================================================
# 8. Candidate pool
# ============================================================

def seed_profiles(history_df):
    """
    평가 구간 시작 전에 알고 있는 과거 투수 정보.
    """

    profiles = defaultdict(dict)

    hist = history_df.sort_values(
        [
            "date",
            "game_id",
            "appearance_order",
        ]
    )

    for _, row in hist.iterrows():
        team = row["def_team"]
        pid = str(row["pitcher_id"])

        profiles[team][pid] = (
            profile_from_row(row)
        )

    return profiles


def get_candidates(
    decision,
    profiles,
    already_used,
):
    team = decision["def_team"]
    current_date = decision["date"]
    actual_pid = str(
        decision["pitcher_id"]
    )

    pool = []

    for pid, p in profiles[team].items():

        if pid in already_used:
            continue

        days = (
            current_date
            - p["last_seen_date"]
        ).days

        # 미래 정보 차단
        if days < 0:
            continue

        if days <= LOOKBACK_DAYS:
            pool.append(p)

    # 실제 투수가 원래 recent pool 안에 있었는가?
    actual_in_recent_pool = any(
        p["pitcher_id"] == actual_pid
        for p in pool
    )

    # Logged action은 평가 대상에 반드시 포함
    if not actual_in_recent_pool:

        actual_profile = (
            profile_from_row(decision)
        )

        if actual_pid not in already_used:
            pool.append(actual_profile)

    # 너무 많은 후보가 생기면 최근 등장 순으로 제한
    # 실제 행동은 반드시 유지
    if len(pool) > MAX_CANDIDATES:

        actual_profiles = [
            p for p in pool
            if p["pitcher_id"] == actual_pid
        ]

        others = [
            p for p in pool
            if p["pitcher_id"] != actual_pid
        ]

        others.sort(
            key=lambda p: (
                p["last_seen_date"],
                (
                    -1
                    if pd.isna(p["prior_n_apps"])
                    else p["prior_n_apps"]
                ),
            ),
            reverse=True,
        )

        keep_n = (
            MAX_CANDIDATES
            - len(actual_profiles)
        )

        pool = (
            actual_profiles
            + others[:keep_n]
        )

    return pool, actual_in_recent_pool


# ============================================================
# 9. Evaluation
# ============================================================

def evaluate(
    name,
    model,
    processor,
    seed_df,
    eval_df,
):
    print_section(
        f"5. EVALUATE {name}"
    )

    profiles = seed_profiles(seed_df)

    already_used = defaultdict(set)

    rows = []

    eval_df = eval_df.sort_values(
        [
            "date",
            "game_id",
            "appearance_order",
        ]
    )

    total_decisions = 0

    for _, decision in eval_df.iterrows():

        total_decisions += 1

        game_key = (
            decision["game_id"],
            decision["def_team"],
        )

        used = already_used[game_key]

        candidates, actual_in_pool = (
            get_candidates(
                decision,
                profiles,
                used,
            )
        )

        actual = str(
            decision["pitcher_id"]
        )

        if len(candidates) >= 2:

            scored = []

            for candidate in candidates:

                values = (
                    candidate_feature_values(
                        decision,
                        candidate,
                    )
                )

                x = (
                    processor.transform_values(
                        values
                    )
                )

                score, mean, uncertainty = (
                    model.score(x)
                )

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

            # Bandit 추천
            scored.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            recommended = (
                scored[0]["pitcher_id"]
            )

            # 휴리스틱:
            # 이전 등판 수 가장 많은 투수
            heuristic = max(
                scored,
                key=lambda x: (
                    -1
                    if pd.isna(
                        x["prior_n_apps"]
                    )
                    else x["prior_n_apps"]
                ),
            )["pitcher_id"]

            rows.append(
                {
                    "game_id":
                        decision["game_id"],

                    "date":
                        decision["date"],

                    "def_team":
                        decision["def_team"],

                    "inning":
                        decision["inning"],

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
                        len(candidates),

                    "bandit_match":
                        recommended == actual,

                    "heuristic_match":
                        heuristic == actual,

                    "actual_in_recent_pool":
                        actual_in_pool,

                    # 랜덤으로 하나 뽑으면
                    # 맞힐 기대 확률
                    "random_expected_match":
                        1.0 / len(candidates),

                    # 실제 감독이 선택한 투수의
                    # 실제 등판 reward
                    "logged_reward":
                        decision["reward"],
                }
            )

        # ----------------------------------------------------
        # 현재 선택은 이제 '과거'가 되었으므로
        # used / profile에 반영
        # ----------------------------------------------------

        used.add(actual)

        profiles[
            decision["def_team"]
        ][actual] = profile_from_row(
            decision
        )

    result = pd.DataFrame(rows)

    if result.empty:
        print(
            "평가 가능한 decision이 없습니다."
        )
        return result, {}

    bandit_agreement = (
        result["bandit_match"].mean()
    )

    heuristic_agreement = (
        result["heuristic_match"].mean()
    )

    random_agreement = (
        result[
            "random_expected_match"
        ].mean()
    )

    avg_candidates = (
        result["n_candidates"].mean()
    )

    candidate_coverage = (
        result[
            "actual_in_recent_pool"
        ].mean()
    )

    eval_coverage = (
        len(result) / total_decisions
    )

    summary = {
        "split": name,

        "total_relief_decisions":
            int(total_decisions),

        "evaluated_decisions":
            int(len(result)),

        "evaluation_coverage":
            float(eval_coverage),

        "avg_candidates":
            float(avg_candidates),

        "recent_pool_actual_coverage":
            float(candidate_coverage),

        "random_expected_agreement":
            float(random_agreement),

        "heuristic_agreement":
            float(heuristic_agreement),

        "bandit_agreement":
            float(bandit_agreement),
    }

    print(
        f"Total relief decisions       : "
        f"{total_decisions:,}"
    )

    print(
        f"Evaluated decisions          : "
        f"{len(result):,}"
    )

    print(
        f"Evaluation coverage          : "
        f"{eval_coverage * 100:.2f}%"
    )

    print(
        f"Avg candidates               : "
        f"{avg_candidates:.2f}"
    )

    print(
        f"Recent-pool actual coverage  : "
        f"{candidate_coverage * 100:.2f}%"
    )

    print()
    print(
        f"Random expected agreement    : "
        f"{random_agreement * 100:.2f}%"
    )

    print(
        f"Heuristic agreement          : "
        f"{heuristic_agreement * 100:.2f}%"
    )

    print(
        f"Bandit agreement             : "
        f"{bandit_agreement * 100:.2f}%"
    )

    return result, summary


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    pa, history = load_data()

    decisions = build_decision_dataset(
        pa,
        history,
    )

    train, val, test = (
        chronological_split(decisions)
    )

    processor = FeatureProcessor()

    model = train_bandit(
        train,
        processor,
    )

    # Validation
    val_result, val_summary = (
        evaluate(
            "VALIDATION",
            model,
            processor,
            seed_df=train,
            eval_df=val,
        )
    )

    # Test 시점에서는
    # train + validation의 과거 기록은 사용 가능
    test_seed = pd.concat(
        [train, val],
        ignore_index=True,
    )

    test_result, test_summary = (
        evaluate(
            "TEST",
            model,
            processor,
            seed_df=test_seed,
            eval_df=test,
        )
    )

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
        OUTPUT_DIR / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "alpha": ALPHA,
                "ridge": RIDGE,
                "lookback_days":
                    LOOKBACK_DAYS,
                "max_candidates":
                    MAX_CANDIDATES,
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

    print_section("DONE")

    print("Saved:")
    print(
        "  bandit_output/"
        "validation_decisions.csv"
    )
    print(
        "  bandit_output/"
        "test_decisions.csv"
    )
    print(
        "  bandit_output/"
        "summary.json"
    )


if __name__ == "__main__":
    main()