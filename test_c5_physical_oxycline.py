#!/usr/bin/env python3

"""
E-AWARE V2 — Step 5 C5B
Physically valid oxycline encoding test.

Only anchors where all 4 locations have valid GEBCO water depth
are included.

Models compared on identical anchor folds:

1. Core
   SST + salinity + surface O2

2. Core + oxycline reached before seabed

3. Core + reached flag + valid oxycline depth when reached

Additional diagnostic:
4. Raw oxycline on anchors where raw oxycline is complete,
   to see whether its previous performance survives removal
   of invalid-depth anchors.

Backgrounds are locally available habitat, NOT biological absence.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
)


INPUT = Path(
    "data/processed/"
    "eaware_v2_step4_coast_matched/"
    "eaware_v2_coast_matched_final_candidate.csv"
)

OUTDIR = Path(
    "data/models/"
    "eaware_v2_c5_physical_oxycline"
)

CORE = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
]

RANDOM_STATE = 42


def fit_model(X, y, groups):

    model = ConditionalLogit(
        y,
        X,
        groups=groups,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = model.fit(
            method="bfgs",
            maxiter=3000,
            disp=False,
        )

    return result


def matched_metrics(frame, scores):

    temp = frame[
        [
            "qc_anchor_id",
            "pb_target",
        ]
    ].copy()

    temp["score"] = scores

    concordances = []
    top1s = []
    ranks = []

    for _, g in temp.groupby(
        "qc_anchor_id"
    ):

        p = g.loc[
            g["pb_target"] == 1,
            "score",
        ].to_numpy()

        b = g.loc[
            g["pb_target"] == 0,
            "score",
        ].to_numpy()

        if len(p) != 1 or len(b) != 3:
            continue

        p = float(p[0])

        b = np.asarray(
            b,
            dtype=float,
        )

        concordance = np.mean(
            (p > b).astype(float)
            +
            0.5
            *
            (p == b).astype(float)
        )

        concordances.append(
            float(concordance)
        )

        scores4 = np.concatenate(
            [[p], b]
        )

        maximum = np.max(
            scores4
        )

        tied = np.sum(
            np.isclose(
                scores4,
                maximum,
            )
        )

        if np.isclose(
            p,
            maximum,
        ):
            top1s.append(
                1.0 / tied
            )
        else:
            top1s.append(
                0.0
            )

        greater = np.sum(
            scores4 > p
        )

        equal = np.sum(
            np.isclose(
                scores4,
                p,
            )
        )

        rank = (
            greater
            + 1
            + (equal - 1) / 2
        )

        ranks.append(
            float(rank)
        )

    return {
        "concordance":
            float(np.mean(concordances)),

        "top1":
            float(np.mean(top1s)),

        "rank":
            float(np.mean(ranks)),
    }


def make_folds(anchor_ids):

    anchors = np.asarray(
        sorted(
            np.unique(
                anchor_ids
            )
        )
    )

    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for train_i, test_i in splitter.split(
        anchors
    ):

        yield (
            set(anchors[train_i]),
            set(anchors[test_i]),
        )


def evaluate(
    df,
    features,
    model_name,
    folds,
):

    rows = []

    for fold_number, (
        train_anchors,
        test_anchors,
    ) in enumerate(
        folds,
        start=1,
    ):

        train = df[
            df["qc_anchor_id"].isin(
                train_anchors
            )
        ].copy()

        test = df[
            df["qc_anchor_id"].isin(
                test_anchors
            )
        ].copy()

        X_train_raw = (
            train[features]
            .astype(float)
            .to_numpy()
        )

        X_test_raw = (
            test[features]
            .astype(float)
            .to_numpy()
        )

        y_train = (
            train["pb_target"]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test["pb_target"]
            .astype(int)
            .to_numpy()
        )

        groups_train = (
            train["qc_anchor_id"]
            .astype(int)
            .to_numpy()
        )

        scaler = (
            StandardScaler()
            .fit(X_train_raw)
        )

        X_train = scaler.transform(
            X_train_raw
        )

        X_test = scaler.transform(
            X_test_raw
        )

        result = fit_model(
            X_train,
            y_train,
            groups_train,
        )

        params = np.asarray(
            result.params,
            dtype=float,
        )

        score = (
            X_test
            @
            params
        )

        auc = roc_auc_score(
            y_test,
            score,
        )

        ap = average_precision_score(
            y_test,
            score,
        )

        mm = matched_metrics(
            test,
            score,
        )

        rows.append(
            {
                "model":
                    model_name,

                "fold":
                    fold_number,

                "test_anchors":
                    test[
                        "qc_anchor_id"
                    ].nunique(),

                "auc":
                    float(auc),

                "ap":
                    float(ap),

                "concordance":
                    mm["concordance"],

                "top1":
                    mm["top1"],

                "presence_rank":
                    mm["rank"],
            }
        )

    return pd.DataFrame(rows)


def summarize(folds):

    return {
        "anchors":
            int(
                folds[
                    "test_anchors"
                ].sum()
            ),

        "mean_auc":
            folds["auc"].mean(),

        "sd_auc":
            folds["auc"].std(),

        "mean_ap":
            folds["ap"].mean(),

        "mean_concordance":
            folds[
                "concordance"
            ].mean(),

        "sd_concordance":
            folds[
                "concordance"
            ].std(),

        "mean_top1":
            folds["top1"].mean(),

        "mean_rank":
            folds[
                "presence_rank"
            ].mean(),
    }


def print_summary(name, result):

    print()
    print(name)

    print(
        "  Anchors:",
        result["anchors"]
    )

    print(
        "  Mean AUC:",
        f"{result['mean_auc']:.3f}"
    )

    print(
        "  AUC SD:",
        f"{result['sd_auc']:.3f}"
    )

    print(
        "  Mean AP:",
        f"{result['mean_ap']:.3f}"
    )

    print(
        "  Matched concordance:",
        f"{result['mean_concordance']:.3f}"
    )

    print(
        "  Concordance SD:",
        f"{result['sd_concordance']:.3f}"
    )

    print(
        "  Top-1:",
        f"{result['mean_top1']:.3f}"
    )

    print(
        "  Mean presence rank:",
        f"{result['mean_rank']:.3f}"
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    df["qc_anchor_id"] = (
        df["qc_anchor_id"]
        .astype(int)
    )

    df["pb_target"] = (
        df["pb_target"]
        .astype(int)
    )

    print(
        "E-AWARE Step 5 C5B — "
        "Physical Oxycline Test"
    )

    print()

    # ==================================================
    # REQUIRE VALID LOCAL WATER DEPTH
    # ==================================================

    depth_complete = (
        df.groupby(
            "qc_anchor_id"
        )[
            "historical_water_depth_m"
        ]
        .apply(
            lambda x:
                x.notna().all()
        )
    )

    valid_depth_anchors = set(
        depth_complete[
            depth_complete
        ]
        .index
        .astype(int)
    )

    work = df[
        df["qc_anchor_id"].isin(
            valid_depth_anchors
        )
    ].copy()

    print(
        "Anchors with valid water depth "
        "for all 4 rows:",
        work[
            "qc_anchor_id"
        ].nunique(),
        "/",
        df[
            "qc_anchor_id"
        ].nunique(),
    )

    print(
        "Rows:",
        len(work)
    )

    # ==================================================
    # PHYSICAL OXYCLINE ENCODING
    # ==================================================

    ACCEPTED = (
        "Accepted within local water column"
    )

    BELOW = (
        "Raw crossing below seabed; "
        "local threshold not reached"
    )

    NOT_REACHED = (
        "1 mL/L threshold not reached "
        "in available profile"
    )

    allowed = {
        ACCEPTED,
        BELOW,
        NOT_REACHED,
    }

    unexpected = set(
        work[
            "historical_oxycline_status"
        ]
        .dropna()
        .unique()
    ) - allowed

    if unexpected:

        raise RuntimeError(
            f"Unexpected statuses in valid-depth "
            f"subset: {unexpected}"
        )

    work[
        "oxycline_reached_before_seabed"
    ] = (
        work[
            "historical_oxycline_status"
        ]
        .eq(
            ACCEPTED
        )
        .astype(int)
    )

    # Valid physical depth contributes only when
    # threshold is actually reached before seabed.
    work[
        "oxycline_depth_when_reached_m"
    ] = np.where(
        work[
            "oxycline_reached_before_seabed"
        ] == 1,
        work[
            "historical_oxycline_m"
        ],
        0.0,
    )

    if work[
        "oxycline_depth_when_reached_m"
    ].isna().any():

        raise RuntimeError(
            "Accepted oxycline row has "
            "missing constrained depth."
        )

    print()

    print(
        "===== PHYSICAL OXYCLINE STATUS ====="
    )

    print(
        work[
            "historical_oxycline_status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Reached before seabed:",
        int(
            work[
                "oxycline_reached_before_seabed"
            ].sum()
        ),
        "/",
        len(work)
    )

    # ==================================================
    # IDENTICAL FOLDS FOR ALL PHYSICAL MODELS
    # ==================================================

    anchor_ids = (
        work[
            "qc_anchor_id"
        ]
        .unique()
    )

    folds = list(
        make_folds(
            anchor_ids
        )
    )

    MODELS = {
        "core": CORE,

        "core_plus_reached": (
            CORE
            +
            [
                "oxycline_reached_before_seabed"
            ]
        ),

        "core_plus_physical_oxycline": (
            CORE
            +
            [
                "oxycline_reached_before_seabed",
                "oxycline_depth_when_reached_m",
            ]
        ),
    }

    all_results = []

    print()
    print(
        "======================================"
    )
    print(
        "PHYSICAL OXYCLINE MODEL COMPARISON"
    )
    print(
        "======================================"
    )

    for name, features in MODELS.items():

        result = evaluate(
            work,
            features,
            name,
            folds,
        )

        all_results.append(
            result
        )

        summary = summarize(
            result
        )

        print_summary(
            name,
            summary
        )

    # ==================================================
    # RAW OXYCLINE DIAGNOSTIC
    # ==================================================

    raw_complete = (
        work.groupby(
            "qc_anchor_id"
        )[
            "historical_raw_oxycline_m"
        ]
        .apply(
            lambda x:
                x.notna().all()
        )
    )

    raw_anchors = set(
        raw_complete[
            raw_complete
        ]
        .index
        .astype(int)
    )

    raw_work = work[
        work[
            "qc_anchor_id"
        ].isin(
            raw_anchors
        )
    ].copy()

    print()
    print(
        "======================================"
    )

    print(
        "RAW OXYCLINE DIAGNOSTIC"
    )

    print(
        "======================================"
    )

    print(
        "Valid-depth + raw-complete anchors:",
        raw_work[
            "qc_anchor_id"
        ].nunique()
    )

    if (
        raw_work[
            "qc_anchor_id"
        ].nunique()
        >= 10
    ):

        raw_folds = list(
            make_folds(
                raw_work[
                    "qc_anchor_id"
                ].unique()
            )
        )

        raw_core = evaluate(
            raw_work,
            CORE,
            "raw_subset_core",
            raw_folds,
        )

        raw_extended = evaluate(
            raw_work,
            CORE
            +
            [
                "historical_raw_oxycline_m"
            ],
            "raw_subset_plus_raw_oxycline",
            raw_folds,
        )

        all_results.extend(
            [
                raw_core,
                raw_extended,
            ]
        )

        core_summary = summarize(
            raw_core
        )

        raw_summary = summarize(
            raw_extended
        )

        print_summary(
            "Raw-subset core",
            core_summary,
        )

        print_summary(
            "Raw-subset + raw oxycline",
            raw_summary,
        )

        print()

        print(
            "RAW OXYCLINE CHANGE "
            "AFTER REMOVING INVALID-DEPTH ANCHORS"
        )

        print(
            "  Δ AUC:",
            f"{raw_summary['mean_auc'] - core_summary['mean_auc']:+.3f}"
        )

        print(
            "  Δ concordance:",
            f"{raw_summary['mean_concordance'] - core_summary['mean_concordance']:+.3f}"
        )

        print(
            "  Δ Top-1:",
            f"{raw_summary['mean_top1'] - core_summary['mean_top1']:+.3f}"
        )

    # ==================================================
    # SAVE
    # ==================================================

    output = pd.concat(
        all_results,
        ignore_index=True,
    )

    output.to_csv(
        OUTDIR
        /
        "physical_oxycline_fold_metrics.csv",
        index=False,
    )

    work.to_csv(
        OUTDIR
        /
        "physical_oxycline_dataset.csv",
        index=False,
    )

    print()

    print(
        "Random expectations:"
    )

    print(
        "  Concordance ≈ 0.500"
    )

    print(
        "  Top-1 ≈ 0.250"
    )

    print(
        "  Mean rank ≈ 2.500"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "No 'No valid GEBCO water depth' "
        "row is used in this test."
    )

    print(
        "No below-seabed oxycline depth is "
        "used as a physical depth."
    )

    print()

    print(
        "===== STEP 5 C5B COMPLETE ====="
    )


if __name__ == "__main__":
    main()
