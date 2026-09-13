from pathlib import Path

import numpy as np
import pandas as pd

import bandit_v2_actual_bullpen as base


# ============================================================
# Config
# ============================================================

LI_PA_PATH = Path(
    "data_analysis/results/"
    "all_pa_with_li_train_frozen.parquet"
)


EXPECTED_DECISIONS = 50177

EXPECTED_TRAIN = 34138
EXPECTED_VAL = 7904
EXPECTED_TEST = 8135


# ============================================================
# Helper
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


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


def describe_li(name, df):

    s = pd.to_numeric(
        df["leverage_index"],
        errors="coerce",
    )

    print()
    print(f"{name} LI distribution")

    print(
        f"  count : {len(s):,}"
    )

    print(
        f"  mean  : {s.mean():.4f}"
    )

    print(
        f"  p25   : {s.quantile(0.25):.4f}"
    )

    print(
        f"  p50   : {s.quantile(0.50):.4f}"
    )

    print(
        f"  p75   : {s.quantile(0.75):.4f}"
    )

    print(
        f"  p95   : {s.quantile(0.95):.4f}"
    )

    print(
        f"  p99   : {s.quantile(0.99):.4f}"
    )

    print(
        f"  max   : {s.max():.4f}"
    )


# ============================================================
# 1. Load original Bandit data
# ============================================================

def load_original_bandit():

    section(
        "1. LOAD ORIGINAL BANDIT DATA"
    )

    base.REWARD_MODE = "wpa_per_pa"
    base.ALPHA = 0.0

    (
        pa,
        history,
        bullpen,
    ) = base.load_data()

    print(
        "Original PA shape :",
        pa.shape,
    )

    print(
        "PA pitcher_id dtype      :",
        pa["pitcher_id"].dtype,
    )

    print(
        "History pitcher_id dtype :",
        history["pitcher_id"].dtype,
    )

    return (
        pa,
        history,
        bullpen,
    )


# ============================================================
# 2. Load LI PA
# ============================================================

def load_li_pa():

    section(
        "2. LOAD TRAIN-FROZEN LI PA"
    )

    if not LI_PA_PATH.exists():

        raise FileNotFoundError(
            f"Cannot find: {LI_PA_PATH}"
        )

    li_pa = pd.read_parquet(
        LI_PA_PATH
    )

    print(
        "LI PA shape :",
        li_pa.shape,
    )

    print(
        "LI pitcher_id dtype :",
        li_pa["pitcher_id"].dtype,
    )

    required = [
        "_orig_idx",
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
        "leverage_index",
        "leverage_index_raw_train_frozen",
    ]

    missing = [
        col
        for col in required
        if col not in li_pa.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    li = pd.to_numeric(
        li_pa["leverage_index"],
        errors="coerce",
    )

    print(
        f"LI null   : "
        f"{li.isna().sum():,}"
    )

    print(
        f"LI finite : "
        f"{np.isfinite(li).sum():,}"
        f" / {len(li):,}"
    )

    print(
        f"LI < 0    : "
        f"{(li < 0).sum():,}"
    )

    if li.isna().any():

        raise ValueError(
            "leverage_index contains NaN."
        )

    if not np.isfinite(
        li
    ).all():

        raise ValueError(
            "leverage_index contains non-finite values."
        )

    if (
        li < 0
    ).any():

        raise ValueError(
            "leverage_index contains negative values."
        )

    return li_pa


# ============================================================
# 3. Verify PA identity
# ============================================================

def verify_pa_identity(
    original_pa,
    li_pa,
):

    section(
        "3. VERIFY ORIGINAL PA / LI PA IDENTITY"
    )

    print(
        f"Original rows : "
        f"{len(original_pa):,}"
    )

    print(
        f"LI rows       : "
        f"{len(li_pa):,}"
    )

    if (
        len(original_pa)
        != len(li_pa)
    ):

        raise ValueError(
            "PA row counts differ."
        )

    # --------------------------------------------------------
    # _orig_idx
    # --------------------------------------------------------

    old_idx = pd.to_numeric(
        original_pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    new_idx = pd.to_numeric(
        li_pa["_orig_idx"],
        errors="raise",
    ).to_numpy()

    same_idx = np.array_equal(
        old_idx,
        new_idx,
    )

    print(
        "Same _orig_idx row order :",
        same_idx,
    )

    if not same_idx:

        raise ValueError(
            "_orig_idx alignment changed."
        )

    # --------------------------------------------------------
    # identity columns
    #
    # dtype는 다를 수 있으므로 비교할 때만 문자열 표준화
    # --------------------------------------------------------

    for col in [
        "game_id",
        "half",
        "pitcher_id",
        "appearance_order",
    ]:

        if col == "pitcher_id":

            old = (
                original_pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

            new = (
                li_pa[col]
                .apply(normalize_id)
                .to_numpy()
            )

        else:

            old = (
                original_pa[col]
                .fillna("__NA__")
                .astype(str)
                .to_numpy()
            )

            new = (
                li_pa[col]
                .fillna("__NA__")
                .astype(str)
                .to_numpy()
            )

        same = np.array_equal(
            old,
            new,
        )

        print(
            f"Same {col:16s}: "
            f"{same}"
        )

        if not same:

            raise ValueError(
                f"PA identity mismatch: {col}"
            )


# ============================================================
# 4. Build ORIGINAL Bandit decisions
# ============================================================

def build_original_decisions(
    original_pa,
    history,
):

    section(
        "4. BUILD ORIGINAL BANDIT DECISIONS"
    )

    (
        decisions,
        first_app,
    ) = base.build_decision_dataset(
        original_pa,
        history,
    )

    print(
        f"Relief decisions : "
        f"{len(decisions):,}"
    )

    print(
        f"Expected         : "
        f"{EXPECTED_DECISIONS:,}"
    )

    if (
        len(decisions)
        != EXPECTED_DECISIONS
    ):

        raise ValueError(
            "Original Bandit decision count changed."
        )

    return (
        decisions,
        first_app,
    )


# ============================================================
# 5. Build _orig_idx -> LI lookup
# ============================================================

def build_li_lookup(
    li_pa,
):

    section(
        "5. BUILD _orig_idx -> LI LOOKUP"
    )

    work = li_pa[
        [
            "_orig_idx",
            "leverage_index",
            "leverage_index_raw_train_frozen",
        ]
        +
        (
            [
                "leverage_index_full_reference"
            ]
            if
            "leverage_index_full_reference"
            in li_pa.columns
            else
            []
        )
    ].copy()

    work["_orig_idx"] = pd.to_numeric(
        work["_orig_idx"],
        errors="raise",
    ).astype(int)

    duplicate_count = int(
        work["_orig_idx"]
        .duplicated()
        .sum()
    )

    print(
        f"Lookup rows          : "
        f"{len(work):,}"
    )

    print(
        f"Duplicate _orig_idx  : "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:

        raise ValueError(
            "_orig_idx must be unique."
        )

    return work


# ============================================================
# 6. Attach LI directly to Bandit decisions
# ============================================================

def attach_li_to_decisions(
    decisions,
    li_lookup,
):

    section(
        "6. ATTACH LI TO ORIGINAL BANDIT DECISIONS"
    )

    work = decisions.copy()

    work["_orig_idx"] = pd.to_numeric(
        work["_orig_idx"],
        errors="raise",
    ).astype(int)

    original_n = len(
        work
    )

    merged = work.merge(
        li_lookup,
        on="_orig_idx",
        how="left",
        validate="many_to_one",
    )

    if (
        len(merged)
        != original_n
    ):

        raise ValueError(
            "Decision count changed after LI merge."
        )

    missing = int(
        merged[
            "leverage_index"
        ]
        .isna()
        .sum()
    )

    print(
        f"Decisions before merge : "
        f"{original_n:,}"
    )

    print(
        f"Decisions after merge  : "
        f"{len(merged):,}"
    )

    print(
        f"Missing decision LI    : "
        f"{missing:,}"
    )

    if missing > 0:

        raise ValueError(
            "Some decisions do not have LI."
        )

    return merged


# ============================================================
# 7. Independently verify decision == first PA
# ============================================================

def verify_first_pa_alignment(
    decisions,
    li_pa,
):

    section(
        "7. VERIFY LI COMES FROM FIRST PA OF APPEARANCE"
    )

    pa = li_pa.copy()

    pa["game_id_key"] = (
        pa["game_id"]
        .astype(str)
    )

    pa["pitcher_id_key"] = (
        pa["pitcher_id"]
        .apply(
            normalize_id
        )
    )

    pa["half_key"] = (
        pa["half"]
        .astype(str)
    )

    pa["appearance_order_key"] = (
        pd.to_numeric(
            pa["appearance_order"],
            errors="coerce",
        )
    )

    pa["_orig_idx"] = pd.to_numeric(
        pa["_orig_idx"],
        errors="raise",
    ).astype(int)

    pa = pa.dropna(
        subset=[
            "appearance_order_key"
        ]
    ).copy()

    pa["appearance_order_key"] = (
        pa[
            "appearance_order_key"
        ]
        .astype(int)
    )

    # --------------------------------------------------------
    # 각 appearance의 가장 첫 번째 PA
    # --------------------------------------------------------

    first_pa = (
        pa.sort_values(
            "_orig_idx"
        )
        .groupby(
            [
                "game_id_key",
                "half_key",
                "pitcher_id_key",
                "appearance_order_key",
            ],
            as_index=False,
        )
        .first()
    )

    first_pa = first_pa[
        [
            "game_id_key",
            "half_key",
            "pitcher_id_key",
            "appearance_order_key",
            "_orig_idx",
            "leverage_index",
        ]
    ].rename(
        columns={
            "_orig_idx":
                "_first_pa_orig_idx",

            "leverage_index":
                "_first_pa_li",
        }
    )

    d = decisions.copy()

    d["game_id_key"] = (
        d["game_id"]
        .astype(str)
    )

    d["pitcher_id_key"] = (
        d["pitcher_id"]
        .apply(
            normalize_id
        )
    )

    d["half_key"] = (
        d["half"]
        .astype(str)
    )

    d["appearance_order_key"] = (
        pd.to_numeric(
            d["appearance_order"],
            errors="raise",
        )
        .astype(int)
    )

    check = d.merge(
        first_pa,
        on=[
            "game_id_key",
            "half_key",
            "pitcher_id_key",
            "appearance_order_key",
        ],
        how="left",
        validate="one_to_one",
    )

    missing = int(
        check[
            "_first_pa_orig_idx"
        ]
        .isna()
        .sum()
    )

    print(
        f"First-PA lookup missing : "
        f"{missing:,}"
    )

    if missing > 0:

        raise ValueError(
            "Some decisions could not be matched "
            "to appearance first PA."
        )

    same_idx = (
        check["_orig_idx"]
        ==
        check["_first_pa_orig_idx"]
    )

    li_diff = (
        check["leverage_index"]
        -
        check["_first_pa_li"]
    ).abs()

    print(
        f"Exact first-PA index match : "
        f"{same_idx.mean() * 100:.4f}%"
    )

    print(
        f"Index mismatch count       : "
        f"{(~same_idx).sum():,}"
    )

    print(
        f"LI match MAE               : "
        f"{li_diff.mean():.12f}"
    )

    print(
        f"LI match max diff          : "
        f"{li_diff.max():.12f}"
    )

    if not same_idx.all():

        bad = check.loc[
            ~same_idx,
            [
                "game_id",
                "half",
                "pitcher_id",
                "appearance_order",
                "_orig_idx",
                "_first_pa_orig_idx",
            ],
        ].head(
            20
        )

        print()
        print(
            bad.to_string(
                index=False
            )
        )

        raise ValueError(
            "Decision _orig_idx is not the "
            "appearance first PA."
        )

    if not np.allclose(
        check[
            "leverage_index"
        ],
        check[
            "_first_pa_li"
        ],
        atol=1e-12,
        rtol=0,
    ):

        raise ValueError(
            "Decision LI differs from first-PA LI."
        )

    return check


# ============================================================
# 8. Split verification
# ============================================================

def verify_split(
    decisions,
):

    section(
        "8. VERIFY CHRONOLOGICAL SPLIT"
    )

    (
        train,
        val,
        test,
    ) = base.chronological_split(
        decisions
    )

    rows = [
        (
            "Train",
            len(train),
            EXPECTED_TRAIN,
        ),
        (
            "Validation",
            len(val),
            EXPECTED_VAL,
        ),
        (
            "Test",
            len(test),
            EXPECTED_TEST,
        ),
    ]

    for (
        name,
        actual,
        expected,
    ) in rows:

        print(
            f"{name:10s}: "
            f"{actual:,} "
            f"(expected {expected:,})"
        )

        if (
            actual
            != expected
        ):

            raise ValueError(
                f"{name} count changed."
            )

    return (
        train,
        val,
        test,
    )


# ============================================================
# 9. LI diagnostics
# ============================================================

def run_li_diagnostics(
    train,
    val,
    test,
):

    section(
        "9. DECISION-LEVEL LI DISTRIBUTION"
    )

    describe_li(
        "TRAIN",
        train,
    )

    describe_li(
        "VALIDATION",
        val,
    )

    describe_li(
        "TEST",
        test,
    )

    # ========================================================
    # Test leverage buckets
    # ========================================================

    print()
    print(
        "TEST LI bucket counts"
    )

    buckets = pd.cut(
        test[
            "leverage_index"
        ],
        bins=[
            -np.inf,
            0.5,
            1.5,
            2.5,
            np.inf,
        ],
        labels=[
            "low_<0.5",
            "normal_0.5-1.5",
            "high_1.5-2.5",
            "critical_>=2.5",
        ],
        right=False,
    )

    counts = (
        buckets.value_counts(
            sort=False
        )
    )

    for (
        bucket,
        count,
    ) in counts.items():

        print(
            f"  {str(bucket):20s}: "
            f"{int(count):,} "
            f"({count / len(test) * 100:.2f}%)"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "STEP 13-2 - BANDIT DECISION / LI ALIGNMENT CHECK"
    )

    print()

    print(
        "Strategy:"
    )

    print(
        "Build Bandit decisions from the original PA data,"
    )

    print(
        "then attach leverage_index by immutable _orig_idx."
    )

    print()

    print(
        "This avoids dtype differences in the enriched LI parquet"
    )

    print(
        "and guarantees that adding LI does not alter Bandit decisions."
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Load
    # ========================================================

    (
        original_pa,
        history,
        bullpen,
    ) = load_original_bandit()

    li_pa = load_li_pa()

    # ========================================================
    # PA identity
    # ========================================================

    verify_pa_identity(
        original_pa,
        li_pa,
    )

    # ========================================================
    # Original decisions
    # ========================================================

    (
        decisions,
        first_app,
    ) = build_original_decisions(
        original_pa,
        history,
    )

    # ========================================================
    # LI lookup
    # ========================================================

    li_lookup = build_li_lookup(
        li_pa
    )

    # ========================================================
    # Attach
    # ========================================================

    decisions = attach_li_to_decisions(
        decisions,
        li_lookup,
    )

    # ========================================================
    # Independent first-PA verification
    # ========================================================

    verify_first_pa_alignment(
        decisions,
        li_pa,
    )

    # ========================================================
    # Split
    # ========================================================

    (
        train,
        val,
        test,
    ) = verify_split(
        decisions
    )

    # ========================================================
    # LI diagnostics
    # ========================================================

    run_li_diagnostics(
        train,
        val,
        test,
    )

    # ========================================================
    # Final
    # ========================================================

    section(
        "FINAL CHECK"
    )

    print(
        "PA identity unchanged         : PASS"
    )

    print(
        "Relief decisions unchanged    : PASS"
    )

    print(
        "Decision count                : PASS"
    )

    print(
        "Decision LI missing           : 0"
    )

    print(
        "First-PA LI alignment         : PASS"
    )

    print(
        "Chronological split           : PASS"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Step 13-3 - B0 vs B1(+LI) ablation"
    )

    section(
        "DONE"
    )


if __name__ == "__main__":
    main()