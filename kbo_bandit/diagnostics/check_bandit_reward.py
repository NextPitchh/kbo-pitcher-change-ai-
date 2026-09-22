import json
from pathlib import Path

import numpy as np
import pandas as pd


PA_PATH = "data_analysis/results/all_pa_enriched_v3.parquet"
APP_PATH = "data_analysis/results/all_pitcher_appearances.parquet"

OUT_DIR = Path("outputs/bandit/core/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = [
    "game_id",
    "half",
    "pitcher_id",
    "appearance_order",
]


def normalize_id(x):
    if pd.isna(x):
        return "__MISSING__"

    s = str(x).strip()

    # parquet 타입에 따라 12345.0으로 읽히는 경우 대응
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    return s


print("=" * 72)
print("1. LOAD")
print("=" * 72)

pa = pd.read_parquet(PA_PATH)
apps = pd.read_parquet(APP_PATH)

print("PA   :", pa.shape)
print("APPS :", apps.shape)


# ------------------------------------------------------------
# 타입 통일
# ------------------------------------------------------------

for df in (pa, apps):
    df["game_id"] = df["game_id"].astype(str)

    df["pitcher_id"] = (
        df["pitcher_id"]
        .apply(normalize_id)
    )

    df["appearance_order"] = pd.to_numeric(
        df["appearance_order"],
        errors="coerce",
    )


# ------------------------------------------------------------
# 3가지 집계 방식 비교
# ------------------------------------------------------------
#
# 어떤 필터를 적용해서 appearance 파일을 만들었는지 확인하기 위해
# 모두 비교한다.
# ------------------------------------------------------------

variants = {
    "all_rows": pa.copy(),

    "quality_clean": (
        pa[
            pa["data_quality_flag"].fillna("") == ""
        ].copy()
    ),

    "quality_clean_no_unk": (
        pa[
            (pa["data_quality_flag"].fillna("") == "")
            &
            (pa["pa_result"] != "UNK")
        ].copy()
    ),
}


results = {}

print()
print("=" * 72)
print("2. CHECK app_wpa == SUM(reward_wpa_computed)")
print("=" * 72)


for name, d in variants.items():

    agg = (
        d.groupby(KEYS, as_index=False)
        .agg(
            wpa_optionB_sum=(
                "reward_wpa_computed",
                "sum",
            ),
            n_pa_calc=(
                "reward_wpa_computed",
                "size",
            ),
        )
    )

    merged = apps.merge(
        agg,
        on=KEYS,
        how="inner",
    )

    merged["wpa_diff"] = (
        merged["app_wpa"]
        - merged["wpa_optionB_sum"]
    )

    abs_diff = merged["wpa_diff"].abs()

    mae = float(abs_diff.mean())
    max_error = float(abs_diff.max())

    close_ratio = float(
        np.isclose(
            merged["app_wpa"],
            merged["wpa_optionB_sum"],
            atol=1e-8,
            rtol=1e-7,
        ).mean()
    )

    npa_match = None

    if "n_pa" in merged.columns:
        npa_match = float(
            (
                merged["n_pa"]
                == merged["n_pa_calc"]
            ).mean()
        )

    results[name] = {
        "merged_rows": len(merged),
        "mae": mae,
        "max_error": max_error,
        "close_ratio": close_ratio,
        "n_pa_match_ratio": npa_match,
    }

    print()
    print(f"[{name}]")
    print(f"rows       : {len(merged):,}")
    print(f"MAE        : {mae:.12f}")
    print(f"MAX ERROR  : {max_error:.12f}")
    print(
        f"WPA match  : "
        f"{close_ratio * 100:.4f}%"
    )

    if npa_match is not None:
        print(
            f"n_pa match : "
            f"{npa_match * 100:.4f}%"
        )


# ------------------------------------------------------------
# 가장 잘 일치하는 variant
# ------------------------------------------------------------

best_name = min(
    results,
    key=lambda x: results[x]["mae"],
)

print()
print("=" * 72)
print("3. BEST MATCH")
print("=" * 72)

print("Best variant:", best_name)
print(results[best_name])


with open(
    OUT_DIR / "app_wpa_optionB_check.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "best_variant": best_name,
            "variants": results,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Reward distribution
# ============================================================

print()
print("=" * 72)
print("4. REWARD DISTRIBUTION")
print("=" * 72)


# Bandit v2에서는 Option B를 직접 집계해서 쓴다.
#
# HANDOFF 기준 학습용 정상 데이터:
# quality flag 정상 + UNK 제외
train_pa = pa[
    (pa["data_quality_flag"].fillna("") == "")
    &
    (pa["pa_result"] != "UNK")
].copy()


agg = (
    train_pa.groupby(KEYS, as_index=False)
    .agg(
        app_wpa_optionB=(
            "reward_wpa_computed",
            "sum",
        ),
        n_pa_optionB=(
            "reward_wpa_computed",
            "size",
        ),
        is_starter=(
            "is_starter",
            "first",
        ),
    )
)


# 불펜만
relief = agg[
    ~agg["is_starter"].astype(bool)
].copy()


relief = relief[
    relief["n_pa_optionB"] > 0
].copy()


# Reward A
relief["reward_app_wpa"] = (
    -relief["app_wpa_optionB"]
)


# Reward B
relief["reward_wpa_per_pa"] = (
    -relief["app_wpa_optionB"]
    / relief["n_pa_optionB"]
)


def reward_stats(df, col):

    s = df[col]

    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "corr_with_n_pa": float(
            s.corr(df["n_pa_optionB"])
        ),
        "min": float(s.min()),
        "median": float(s.median()),
        "max": float(s.max()),
    }


stats_a = reward_stats(
    relief,
    "reward_app_wpa",
)

stats_b = reward_stats(
    relief,
    "reward_wpa_per_pa",
)


print()
print("[A] -app_wpa")
for k, v in stats_a.items():
    print(f"{k:20s}: {v}")


print()
print("[B] -app_wpa / n_pa")
for k, v in stats_b.items():
    print(f"{k:20s}: {v}")


with open(
    OUT_DIR / "reward_diagnostics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "reward_app_wpa": stats_a,
            "reward_wpa_per_pa": stats_b,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


relief[
    [
        "game_id",
        "pitcher_id",
        "appearance_order",
        "n_pa_optionB",
        "app_wpa_optionB",
        "reward_app_wpa",
        "reward_wpa_per_pa",
    ]
].to_csv(
    OUT_DIR / "reward_diagnostics.csv",
    index=False,
)


print()
print("Saved:")
print(
    " outputs/bandit/core/v2/"
    "app_wpa_optionB_check.json"
)
print(
    " outputs/bandit/core/v2/"
    "reward_diagnostics.json"
)
