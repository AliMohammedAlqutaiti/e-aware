#!/usr/bin/env python3

"""
E-AWARE V2 Step 5
Baseline presence-background habitat suitability model.

IMPORTANT:
- Presence = observed Engraulis ringens occurrence.
- Background = environmentally available marine location.
- Background is NOT confirmed biological absence.
- Model output is RELATIVE HABITAT SUITABILITY.
- It must NOT be described as probability that anchoveta is present.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


INPUT = Path(
    "data/processed/eaware_v2_step4_pb/"
    "eaware_v2_step4_matched_historical_environment.csv"
)

OUTDIR = Path(
    "data/models/eaware_v2_step5_baseline"
)

RANDOM_STATE = 42

MAX_PHYSICS_DISTANCE_KM = 15.0
MAX_BGC_DISTANCE_KM = 25.0

FEATURES = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
    "month_sin",
    "month_cos",
]


def main():

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            precision_score,
            recall_score,
            f1_score,
            balanced_accuracy_score,
        )

        try:
            from sklearn.model_selection import (
                StratifiedGroupKFold,
            )
            use_stratified_group = True

        except ImportError:
            from sklearn.model_selection import GroupKFold
            use_stratified_group = False

        import joblib

    except ImportError as exc:
        raise SystemExit(
            "Missing modelling dependency. Run:\n"
            "pip install scikit-learn joblib\n\n"
            f"Original error: {exc}"
        )

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

    print(
        "E-AWARE V2 Step 5 — "
        "Presence-Background Baseline Model"
    )

    print()
    print("Input rows:", len(df))

    # --------------------------------
    # QC / exclusion tracking
    # --------------------------------

    df["exclusion_reason"] = ""

    bad_status = (
        df["match_status"]
        !=
        "Matched"
    )

    df.loc[
        bad_status,
        "exclusion_reason"
    ] = "No valid historical environmental match"

    far_physics = (
        df["match_status"].eq("Matched")
        &
        (
            df["physics_match_distance_km"]
            >
            MAX_PHYSICS_DISTANCE_KM
        )
    )

    df.loc[
        far_physics,
        "exclusion_reason"
    ] = (
        "Physics environmental grid "
        "match exceeds 15 km"
    )

    far_bgc = (
        df["match_status"].eq("Matched")
        &
        (
            df["bgc_match_distance_km"]
            >
            MAX_BGC_DISTANCE_KM
        )
    )

    df.loc[
        far_bgc,
        "exclusion_reason"
    ] = (
        "BGC environmental grid "
        "match exceeds 25 km"
    )

    missing_feature = (
        df[FEATURES]
        .isna()
        .any(axis=1)
    )

    missing_feature_mask = (
        df["exclusion_reason"].eq("")
        &
        missing_feature
    )

    df.loc[
        missing_feature_mask,
        "exclusion_reason"
    ] = "Missing baseline environmental predictor"

    bad_target = ~(
        df["pb_target"]
        .isin([0, 1])
    )

    df.loc[
        df["exclusion_reason"].eq("")
        &
        bad_target,
        "exclusion_reason"
    ] = "Invalid presence-background target"

    candidate = df[
        df["exclusion_reason"].eq("")
    ].copy()

    # --------------------------------
    # Keep only dates represented by
    # BOTH presence and background.
    #
    # This prevents dates whose observed
    # presence failed environmental
    # matching from contributing only
    # background samples.
    # --------------------------------

    date_class_count = (
        candidate
        .groupby("date")["pb_target"]
        .nunique()
    )

    valid_dates = (
        date_class_count[
            date_class_count >= 2
        ]
        .index
    )

    date_unbalanced = (
        candidate[
            ~candidate["date"].isin(
                valid_dates
            )
        ]
        .index
    )

    df.loc[
        date_unbalanced,
        "exclusion_reason"
    ] = (
        "Date does not retain both "
        "presence and background after QC"
    )

    modelling = df[
        df["exclusion_reason"].eq("")
    ].copy()

    modelling = (
        modelling
        .sort_values(
            [
                "date",
                "sample_type",
            ]
        )
        .reset_index(drop=True)
    )

    excluded = df[
        ~df["exclusion_reason"].eq("")
    ].copy()

    print()
    print("===== MODELLING DATASET =====")

    print(
        "Rows retained:",
        len(modelling)
    )

    print()
    print("By sample type:")
    print(
        modelling[
            "sample_type"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("By target:")
    print(
        modelling[
            "pb_target"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Unique dates:",
        modelling[
            "date"
        ].nunique()
    )

    print(
        "Date range:",
        modelling["date"].min(),
        "→",
        modelling["date"].max(),
    )

    # Safety checks

    if len(modelling) < 50:
        raise RuntimeError(
            "Too few rows remain for baseline modelling."
        )

    if (
        modelling["pb_target"]
        .nunique()
        !=
        2
    ):
        raise RuntimeError(
            "Presence-background model requires "
            "both target classes."
        )

    if (
        modelling["date"]
        .nunique()
        <
        5
    ):
        raise RuntimeError(
            "Insufficient independent dates "
            "for grouped validation."
        )

    # --------------------------------
    # Model
    # --------------------------------

    X = modelling[
        FEATURES
    ].copy()

    y = (
        modelling[
            "pb_target"
        ]
        .astype(int)
        .to_numpy()
    )

    groups = (
        modelling["date"]
        .dt
        .strftime("%Y-%m-%d")
        .to_numpy()
    )

    # No class_weight is used.
    # The background sampling ratio is
    # part of the presence-background design.
    pipeline = Pipeline(
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

    n_splits = min(
        5,
        modelling["date"].nunique(),
    )

    if use_stratified_group:

        splitter = (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
        )

        splits = splitter.split(
            X,
            y,
            groups,
        )

        validation_method = (
            "StratifiedGroupKFold grouped by exact date"
        )

    else:

        splitter = GroupKFold(
            n_splits=n_splits
        )

        splits = splitter.split(
            X,
            y,
            groups,
        )

        validation_method = (
            "GroupKFold grouped by exact date"
        )

    oof = np.full(
        len(modelling),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    print()
    print(
        "===== GROUPED CROSS-VALIDATION ====="
    )

    print(
        "Validation:",
        validation_method,
    )

    print(
        "Folds:",
        n_splits,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits,
        start=1,
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        if (
            len(
                np.unique(
                    y_train
                )
            )
            <
            2
            or
            len(
                np.unique(
                    y_test
                )
            )
            <
            2
        ):
            print(
                f"Fold {fold}: skipped "
                "because one class was missing."
            )
            continue

        pipeline.fit(
            X_train,
            y_train,
        )

        score = (
            pipeline
            .predict_proba(
                X_test
            )[
                :,
                1
            ]
        )

        oof[
            test_idx
        ] = score

        pred = (
            score
            >=
            0.5
        ).astype(int)

        metrics = {
            "fold":
                fold,

            "train_rows":
                len(train_idx),

            "test_rows":
                len(test_idx),

            "train_dates":
                len(
                    np.unique(
                        groups[
                            train_idx
                        ]
                    )
                ),

            "test_dates":
                len(
                    np.unique(
                        groups[
                            test_idx
                        ]
                    )
                ),

            "test_presence":
                int(
                    (
                        y_test
                        ==
                        1
                    ).sum()
                ),

            "test_background":
                int(
                    (
                        y_test
                        ==
                        0
                    ).sum()
                ),

            "roc_auc":
                roc_auc_score(
                    y_test,
                    score,
                ),

            "average_precision":
                average_precision_score(
                    y_test,
                    score,
                ),

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
            metrics
        )

        print(
            f"Fold {fold}: "
            f"ROC-AUC="
            f"{metrics['roc_auc']:.3f} | "
            f"AP="
            f"{metrics['average_precision']:.3f}"
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    valid_oof = np.isfinite(
        oof
    )

    if (
        valid_oof.sum()
        <
        len(modelling)
        *
        0.8
    ):
        raise RuntimeError(
            "Too many rows lack "
            "out-of-fold predictions."
        )

    overall_auc = roc_auc_score(
        y[
            valid_oof
        ],
        oof[
            valid_oof
        ],
    )

    overall_ap = average_precision_score(
        y[
            valid_oof
        ],
        oof[
            valid_oof
        ],
    )

    background_fraction = (
        (
            y[
                valid_oof
            ]
            ==
            0
        ).mean()
    )

    presence_fraction = (
        (
            y[
                valid_oof
            ]
            ==
            1
        ).mean()
    )

    print()
    print(
        "===== CROSS-VALIDATED PERFORMANCE ====="
    )

    print(
        f"Overall ROC-AUC: "
        f"{overall_auc:.3f}"
    )

    print(
        f"Overall Average Precision: "
        f"{overall_ap:.3f}"
    )

    print(
        f"Presence fraction: "
        f"{presence_fraction:.3f}"
    )

    print(
        "NOTE: predicted scores are "
        "relative habitat suitability, "
        "not calibrated occurrence probabilities."
    )

    # --------------------------------
    # Final fit on all retained data
    # --------------------------------

    pipeline.fit(
        X,
        y,
    )

    full_fit_score = (
        pipeline
        .predict_proba(
            X
        )[
            :,
            1
        ]
    )

    modelling[
        "oof_relative_suitability"
    ] = oof

    modelling[
        "full_fit_relative_suitability"
    ] = full_fit_score

    # Standardized-space coefficients

    model = (
        pipeline
        .named_steps[
            "model"
        ]
    )

    coefficients = pd.DataFrame(
        {
            "predictor":
                FEATURES,

            "standardized_coefficient":
                model.coef_[
                    0
                ],
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
        .reset_index(
            drop=True
        )
    )

    # --------------------------------
    # Save
    # --------------------------------

    modelling.to_csv(
        OUTDIR
        /
        "step5_training_dataset.csv",
        index=False,
    )

    excluded.to_csv(
        OUTDIR
        /
        "step5_excluded_rows.csv",
        index=False,
    )

    fold_metrics.to_csv(
        OUTDIR
        /
        "step5_cross_validation_metrics.csv",
        index=False,
    )

    coefficients.to_csv(
        OUTDIR
        /
        "step5_standardized_coefficients.csv",
        index=False,
    )

    bundle = {
        "pipeline":
            pipeline,

        "features":
            FEATURES,

        "model_type":
            "logistic_regression_presence_background",

        "output_interpretation":
            "relative habitat suitability",

        "background_is_absence":
            False,

        "validation_method":
            validation_method,

        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        bundle,
        OUTDIR
        /
        "eaware_presence_background_baseline.joblib",
    )

    summary = {
        "model":
            "Logistic Regression",

        "framework":
            "presence-background",

        "observed_presence_definition":
            "public Engraulis ringens occurrence",

        "background_definition":
            "generated marine background; not biological absence",

        "features":
            FEATURES,

        "rows_retained":
            len(modelling),

        "presence_rows":
            int(
                (
                    modelling[
                        "pb_target"
                    ]
                    ==
                    1
                ).sum()
            ),

        "background_rows":
            int(
                (
                    modelling[
                        "pb_target"
                    ]
                    ==
                    0
                ).sum()
            ),

        "rows_excluded":
            len(excluded),

        "unique_dates":
            int(
                modelling[
                    "date"
                ].nunique()
            ),

        "validation":
            validation_method,

        "overall_roc_auc":
            float(
                overall_auc
            ),

        "overall_average_precision":
            float(
                overall_ap
            ),

        "presence_fraction":
            float(
                presence_fraction
            ),

        "output_is_probability_of_presence":
            False,

        "output_interpretation":
            "relative habitat suitability",
    }

    with open(
        OUTDIR
        /
        "step5_model_summary.json",
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
        "===== STANDARDIZED COEFFICIENTS ====="
    )

    print(
        coefficients[
            [
                "predictor",
                "standardized_coefficient",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "===== STEP 5 COMPLETE ====="
    )

    print(
        "Training rows:",
        len(modelling)
    )

    print(
        "Excluded rows:",
        len(excluded)
    )

    print(
        "Model saved:",
        OUTDIR
        /
        "eaware_presence_background_baseline.joblib"
    )

    print()
    print(
        "DO NOT interpret model scores "
        "as probability of anchoveta presence."
    )


if __name__ == "__main__":
    main()
