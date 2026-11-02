#!/usr/bin/env python3

"""
E-AWARE V2 — Step 5 C5
Ecological predictor comparison on final coast-matched dataset.

Backgrounds:
- same date
- local availability
- coast-distance matched
- NOT biological absences

Primary baseline:
    SST + salinity + surface O2

Additional ecological tests:
    + water depth
    + constrained oxycline depth
    + raw oxycline depth (exploratory only)

IMPORTANT:
Models are compared on the SAME anchor subset for each test.
Scores are relative habitat suitability rankings,
NOT calibrated probability of anchoveta presence.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
    "eaware_v2_c5_predictor_tests"
)

CORE = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
]

TESTS = {
    "depth": "historical_water_depth_m",

    "constrained_oxycline":
        "historical_oxycline_m",

    "raw_oxycline_exploratory":
        "historical_raw_oxycline_m",
}

RANDOM_STATE = 42


def fit_conditional(X, y, groups):

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
            "score"
        ].to_numpy()

        b = g.loc[
            g["pb_target"] == 0,
            "score"
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

        all_scores = np.concatenate(
            [[p], b]
        )

        maximum = np.max(
            all_scores
        )

        tied = np.sum(
            np.isclose(
                all_scores,
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
            all_scores > p
        )

        equal = np.sum(
            np.isclose(
                all_scores,
                p,
            )
        )

        rank = (
            greater
            +
            1
            +
            (equal - 1) / 2
        )

        ranks.append(
            float(rank)
        )

    return {
        "concordance":
            float(
                np.mean(
                    concordances
                )
            ),

        "top1":
            float(
                np.mean(
                    top1s
                )
            ),

        "rank":
            float(
                np.mean(
                    ranks
                )
            ),
    }


def make_anchor_folds(
    anchors,
    n_splits=5,
):

    anchors = np.asarray(
        sorted(
            np.unique(
                anchors
            )
        )
    )

    n_splits = min(
        n_splits,
        len(anchors),
    )

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for train_a, test_a in (
        splitter.split(
            anchors
        )
    ):

        yield (
            set(
                anchors[
                    train_a
                ]
            ),
            set(
                anchors[
                    test_a
                ]
            ),
        )


def evaluate(
    df,
    features,
    label,
):

    fold_rows = []

    anchors = (
        df["qc_anchor_id"]
        .astype(int)
        .to_numpy()
    )

    for fold, (
        train_anchors,
        test_anchors,
    ) in enumerate(
        make_anchor_folds(
            anchors
        ),
        start=1,
    ):

        train_mask = (
            df["qc_anchor_id"]
            .isin(
                train_anchors
            )
        )

        test_mask = (
            df["qc_anchor_id"]
            .isin(
                test_anchors
            )
        )

        train = (
            df[
                train_mask
            ]
            .copy()
        )

        test = (
            df[
                test_mask
            ]
            .copy()
        )

        X_train_raw = (
            train[
                features
            ]
            .astype(float)
            .to_numpy()
        )

        X_test_raw = (
            test[
                features
            ]
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
            train[
                "qc_anchor_id"
            ]
            .astype(int)
            .to_numpy()
        )

        scaler = (
            StandardScaler()
            .fit(
                X_train_raw
            )
        )

        X_train = (
            scaler.transform(
                X_train_raw
            )
        )

        X_test = (
            scaler.transform(
                X_test_raw
            )
        )

        result = fit_conditional(
            X_train,
            y_train,
            groups_train,
        )

        params = np.asarray(
            result.params,
            dtype=float,
        )

        scores = (
            X_test
            @
            params
        )

        auc = roc_auc_score(
            y_test,
            scores,
        )

        ap = average_precision_score(
            y_test,
            scores,
        )

        mm = matched_metrics(
            test,
            scores,
        )

        fold_rows.append(
            {
                "model":
                    label,

                "fold":
                    fold,

                "test_anchors":
                    test[
                        "qc_anchor_id"
                    ].nunique(),

                "auc":
                    float(
                        auc
                    ),

                "ap":
                    float(
                        ap
                    ),

                "concordance":
                    mm[
                        "concordance"
                    ],

                "top1":
                    mm[
                        "top1"
                    ],

                "presence_rank":
                    mm[
                        "rank"
                    ],
            }
        )

    folds = pd.DataFrame(
        fold_rows
    )

    summary = {
        "model":
            label,

        "anchors":
            int(
                df[
                    "qc_anchor_id"
                ].nunique()
            ),

        "rows":
            int(
                len(df)
            ),

        "mean_auc":
            float(
                folds["auc"]
                .mean()
            ),

        "sd_auc":
            float(
                folds["auc"]
                .std()
            ),

        "mean_ap":
            float(
                folds["ap"]
                .mean()
            ),

        "mean_concordance":
            float(
                folds[
                    "concordance"
                ]
                .mean()
            ),

        "sd_concordance":
            float(
                folds[
                    "concordance"
                ]
                .std()
            ),

        "mean_top1":
            float(
                folds["top1"]
                .mean()
            ),

        "mean_presence_rank":
            float(
                folds[
                    "presence_rank"
                ]
                .mean()
            ),
    }

    return folds, summary


def complete_anchor_subset(
    df,
    features,
):

    complete_rows = (
        df[
            features
        ]
        .notna()
        .all(axis=1)
    )

    temp = (
        df.assign(
            complete_row=
                complete_rows
        )
    )

    anchor_complete = (
        temp.groupby(
            "qc_anchor_id"
        )[
            "complete_row"
        ]
        .all()
    )

    good_anchors = set(
        anchor_complete[
            anchor_complete
        ]
        .index
        .astype(int)
    )

    return (
        df[
            df[
                "qc_anchor_id"
            ]
            .isin(
                good_anchors
            )
        ]
        .copy()
    )


def print_summary(summary):

    print(
        f"Anchors: "
        f"{summary['anchors']}"
    )

    print(
        f"Rows: "
        f"{summary['rows']}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{summary['mean_auc']:.3f}"
    )

    print(
        f"AUC SD: "
        f"{summary['sd_auc']:.3f}"
    )

    print(
        f"Mean AP: "
        f"{summary['mean_ap']:.3f}"
    )

    print(
        f"Matched concordance: "
        f"{summary['mean_concordance']:.3f}"
    )

    print(
        f"Concordance SD: "
        f"{summary['sd_concordance']:.3f}"
    )

    print(
        f"Top-1: "
        f"{summary['mean_top1']:.3f}"
    )

    print(
        f"Mean presence rank: "
        f"{summary['mean_presence_rank']:.3f}"
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    df["pb_target"] = (
        df["pb_target"]
        .astype(int)
    )

    df["qc_anchor_id"] = (
        df["qc_anchor_id"]
        .astype(int)
    )

    print(
        "E-AWARE V2 Step 5 C5 — "
        "Ecological Predictor Comparison"
    )

    print()

    print(
        "Final C5 rows:",
        len(df)
    )

    print(
        "Final C5 anchors:",
        df[
            "qc_anchor_id"
        ].nunique()
    )

    print(
        "Presences:",
        int(
            (
                df["pb_target"] == 1
            ).sum()
        )
    )

    print(
        "Backgrounds:",
        int(
            (
                df["pb_target"] == 0
            ).sum()
        )
    )

    print()

    # ==================================================
    # CORE MODEL ON ALL FINAL C5 ANCHORS
    # ==================================================

    core_df = complete_anchor_subset(
        df,
        CORE,
    )

    core_folds, core_summary = (
        evaluate(
            core_df,
            CORE,
            "core_all_C5",
        )
    )

    print(
        "======================================"
    )

    print(
        "CORE MODEL — ALL C5 ANCHORS"
    )

    print(
        "SST + salinity + surface O2"
    )

    print(
        "======================================"
    )

    print_summary(
        core_summary
    )

    all_folds = [
        core_folds
    ]

    all_summaries = [
        core_summary
    ]

    # ==================================================
    # FAIR PAIRED TESTS
    # ==================================================

    for test_name, extra in (
        TESTS.items()
    ):

        print()
        print(
            "======================================"
        )

        print(
            "TEST:",
            test_name
        )

        print(
            "Additional predictor:",
            extra
        )

        print(
            "======================================"
        )

        pair_features = (
            CORE
            +
            [extra]
        )

        subset = (
            complete_anchor_subset(
                df,
                pair_features,
            )
        )

        n_anchors = (
            subset[
                "qc_anchor_id"
            ]
            .nunique()
        )

        print(
            "Complete anchors:",
            n_anchors,
            "/",
            df[
                "qc_anchor_id"
            ].nunique()
        )

        if n_anchors < 10:

            print(
                "⚠ Too few complete anchors. "
                "Skipping model test."
            )

            continue

        base_folds, base_summary = (
            evaluate(
                subset,
                CORE,
                f"core_on_{test_name}_subset",
            )
        )

        extended_folds, extended_summary = (
            evaluate(
                subset,
                pair_features,
                f"core_plus_{test_name}",
            )
        )

        print()

        print(
            "--- FAIR BASELINE "
            "(same anchors) ---"
        )

        print_summary(
            base_summary
        )

        print()

        print(
            "--- EXTENDED MODEL ---"
        )

        print_summary(
            extended_summary
        )

        delta_concordance = (
            extended_summary[
                "mean_concordance"
            ]
            -
            base_summary[
                "mean_concordance"
            ]
        )

        delta_auc = (
            extended_summary[
                "mean_auc"
            ]
            -
            base_summary[
                "mean_auc"
            ]
        )

        delta_top1 = (
            extended_summary[
                "mean_top1"
            ]
            -
            base_summary[
                "mean_top1"
            ]
        )

        print()

        print(
            "CHANGE FROM FAIR BASELINE"
        )

        print(
            f"Δ AUC: "
            f"{delta_auc:+.3f}"
        )

        print(
            f"Δ concordance: "
            f"{delta_concordance:+.3f}"
        )

        print(
            f"Δ Top-1: "
            f"{delta_top1:+.3f}"
        )

        all_folds.extend(
            [
                base_folds,
                extended_folds,
            ]
        )

        all_summaries.extend(
            [
                base_summary,
                extended_summary,
            ]
        )

    # ==================================================
    # SAVE
    # ==================================================

    folds_out = pd.concat(
        all_folds,
        ignore_index=True,
    )

    summary_out = pd.DataFrame(
        all_summaries
    )

    folds_out.to_csv(
        OUTDIR
        /
        "c5_predictor_fold_metrics.csv",
        index=False,
    )

    summary_out.to_csv(
        OUTDIR
        /
        "c5_predictor_summary.csv",
        index=False,
    )

    print()
    print(
        "======================================"
    )

    print(
        "FINAL C5 PREDICTOR SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        summary_out.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()

    print(
        "Random expectations:"
    )

    print(
        "Matched concordance ≈ 0.500"
    )

    print(
        "Top-1 ≈ 0.250"
    )

    print(
        "Mean presence rank ≈ 2.500"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Distance from coast was deliberately "
        "NOT used as a predictor."
    )

    print(
        "Raw oxycline is exploratory only; "
        "the constrained oxycline is the "
        "scientifically preferred variable."
    )

    print()

    print(
        "===== STEP 5 C5 COMPLETE ====="
    )


if __name__ == "__main__":
    main()
