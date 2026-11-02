#!/usr/bin/env python3

"""
E-AWARE V2 Step 5
Local-availability Logistic Regression baseline.

Design:
- 1 observed anchoveta presence per anchor
- 3 locally available marine background points
- exact same date
- backgrounds within 10–50 km
- anchors kept together during validation

IMPORTANT:
The model predicts RELATIVE HABITAT SUITABILITY /
used-vs-locally-available habitat.

It does NOT estimate calibrated probability of anchoveta presence.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


INPUT = Path(
    "data/processed/eaware_v2_step4_local/"
    "eaware_v2_step4_local_balanced_candidate.csv"
)

OUTDIR = Path(
    "data/models/"
    "eaware_v2_local_baseline"
)

FEATURES = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
    "month_sin",
    "month_cos",
]

RANDOM_STATE = 42


def make_model():
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    required = (
        FEATURES
        +
        [
            "pb_target",
            "qc_anchor_id",
            "sample_type",
        ]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=required
    ).copy()

    df["pb_target"] = (
        df["pb_target"]
        .astype(int)
    )

    df["qc_anchor_id"] = (
        df["qc_anchor_id"]
        .astype(int)
    )

    print(
        "E-AWARE V2 Step 5 — "
        "Local Availability Logistic Baseline"
    )

    print()

    print(
        "Rows:",
        len(df)
    )

    print(
        "Presences:",
        int(
            (
                df["pb_target"]
                ==
                1
            ).sum()
        )
    )

    print(
        "Backgrounds:",
        int(
            (
                df["pb_target"]
                ==
                0
            ).sum()
        )
    )

    print(
        "Anchors:",
        df["qc_anchor_id"].nunique()
    )

    print()

    # --------------------------------------------------
    # Confirm every anchor still contains exactly 1 + 3
    # --------------------------------------------------

    anchor_check = (
        df.groupby(
            "qc_anchor_id"
        )
        .agg(
            rows=(
                "pb_target",
                "size",
            ),
            presences=(
                "pb_target",
                "sum",
            ),
        )
    )

    bad = anchor_check[
        ~(
            (anchor_check["rows"] == 4)
            &
            (anchor_check["presences"] == 1)
        )
    ]

    if len(bad):
        print(
            "Invalid anchors:"
        )
        print(
            bad.to_string()
        )

        raise RuntimeError(
            "Dataset is not perfectly "
            "balanced 1:3 by anchor."
        )

    print(
        "✅ Every anchor contains "
        "1 presence + 3 backgrounds."
    )

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    X = df[
        FEATURES
    ].copy()

    y = (
        df["pb_target"]
        .to_numpy()
    )

    groups = (
        df["qc_anchor_id"]
        .to_numpy()
    )

    # --------------------------------------------------
    # GROUPED CROSS-VALIDATION
    # --------------------------------------------------

    splitter = GroupKFold(
        n_splits=5
    )

    fold_rows = []

    oof = np.full(
        len(df),
        np.nan,
    )

    print()
    print(
        "===== ANCHOR-GROUPED CROSS-VALIDATION ====="
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        model = make_model()

        model.fit(
            X.iloc[train_idx],
            y[train_idx],
        )

        score = (
            model
            .predict_proba(
                X.iloc[test_idx]
            )[:, 1]
        )

        oof[
            test_idx
        ] = score

        y_test = y[
            test_idx
        ]

        pred = (
            score >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y_test,
            score,
        )

        ap = average_precision_score(
            y_test,
            score,
        )

        fold_row = {
            "fold":
                fold,

            "train_anchors":
                len(
                    np.unique(
                        groups[
                            train_idx
                        ]
                    )
                ),

            "test_anchors":
                len(
                    np.unique(
                        groups[
                            test_idx
                        ]
                    )
                ),

            "test_rows":
                len(test_idx),

            "test_presence":
                int(
                    (
                        y_test == 1
                    ).sum()
                ),

            "test_background":
                int(
                    (
                        y_test == 0
                    ).sum()
                ),

            "roc_auc":
                auc,

            "average_precision":
                ap,

            "balanced_accuracy_0_5":
                balanced_accuracy_score(
                    y_test,
                    pred,
                ),

            "precision_0_5":
                precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),

            "recall_0_5":
                recall_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),

            "f1_0_5":
                f1_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),
        }

        fold_rows.append(
            fold_row
        )

        print(
            f"Fold {fold}: "
            f"ROC-AUC={auc:.3f} | "
            f"AP={ap:.3f} | "
            f"anchors="
            f"{fold_row['test_anchors']}"
        )

    folds = pd.DataFrame(
        fold_rows
    )

    mean_auc = (
        folds[
            "roc_auc"
        ].mean()
    )

    sd_auc = (
        folds[
            "roc_auc"
        ].std()
    )

    mean_ap = (
        folds[
            "average_precision"
        ].mean()
    )

    sd_ap = (
        folds[
            "average_precision"
        ].std()
    )

    print()
    print(
        "===== BASELINE PERFORMANCE ====="
    )

    print(
        f"Mean ROC-AUC: "
        f"{mean_auc:.3f}"
    )

    print(
        f"ROC-AUC SD: "
        f"{sd_auc:.3f}"
    )

    print(
        f"Mean Average Precision: "
        f"{mean_ap:.3f}"
    )

    print(
        f"Average Precision SD: "
        f"{sd_ap:.3f}"
    )

    print(
        "Presence fraction:",
        round(
            float(
                y.mean()
            ),
            3,
        ),
    )

    # --------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------

    final_model = make_model()

    final_model.fit(
        X,
        y,
    )

    score = (
        final_model
        .predict_proba(
            X
        )[:, 1]
    )

    df[
        "full_fit_relative_suitability"
    ] = score

    # --------------------------------------------------
    # COEFFICIENTS
    # --------------------------------------------------

    logistic = (
        final_model
        .named_steps[
            "model"
        ]
    )

    coefficients = pd.DataFrame(
        {
            "predictor":
                FEATURES,

            "standardized_coefficient":
                logistic.coef_[0],
        }
    )

    coefficients[
        "absolute_coefficient"
    ] = (
        coefficients[
            "standardized_coefficient"
        ]
        .abs()
    )

    coefficients = (
        coefficients
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "===== STANDARDIZED COEFFICIENTS ====="
    )

    print(
        coefficients[
            [
                "predictor",
                "standardized_coefficient",
            ]
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    df.to_csv(
        OUTDIR
        /
        "local_baseline_training_dataset.csv",
        index=False,
    )

    folds.to_csv(
        OUTDIR
        /
        "local_baseline_fold_metrics.csv",
        index=False,
    )

    coefficients.to_csv(
        OUTDIR
        /
        "local_baseline_coefficients.csv",
        index=False,
    )

    bundle = {
        "pipeline":
            final_model,

        "features":
            FEATURES,

        "model_type":
            "logistic_regression",

        "background_design":
            "local_availability",

        "background_is_absence":
            False,

        "output_interpretation":
            "relative habitat suitability",

        "anchor_distance_km":
            "10-50",

        "background_date":
            "same as observed presence",

        "backgrounds_per_presence":
            3,
    }

    joblib.dump(
        bundle,
        OUTDIR
        /
        "eaware_local_baseline.joblib",
    )

    summary = {
        "rows":
            int(len(df)),

        "anchors":
            int(
                df[
                    "qc_anchor_id"
                ].nunique()
            ),

        "presences":
            int(
                (
                    y == 1
                ).sum()
            ),

        "backgrounds":
            int(
                (
                    y == 0
                ).sum()
            ),

        "mean_roc_auc":
            float(
                mean_auc
            ),

        "sd_roc_auc":
            float(
                sd_auc
            ),

        "mean_average_precision":
            float(
                mean_ap
            ),

        "sd_average_precision":
            float(
                sd_ap
            ),

        "presence_fraction":
            float(
                y.mean()
            ),

        "probability_of_presence":
            False,

        "interpretation":
            "relative habitat suitability",
    }

    with open(
        OUTDIR
        /
        "local_baseline_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "===== STEP 5 V2 COMPLETE ====="
    )

    print(
        "Model:",
        OUTDIR
        /
        "eaware_local_baseline.joblib"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The score represents relative "
        "habitat suitability against "
        "locally available habitat."
    )

    print(
        "It is NOT calibrated probability "
        "of anchoveta presence."
    )


if __name__ == "__main__":
    main()
