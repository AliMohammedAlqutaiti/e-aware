#!/usr/bin/env python3

"""
E-AWARE V2 Step 5B
Spatial and temporal robustness validation.

This does NOT retrain the deployed E-AWARE model.

It tests whether the Step 5 logistic baseline remains useful when
entire spatial regions or years are held out.

Output remains RELATIVE HABITAT SUITABILITY.
Background points are NOT confirmed biological absences.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


INPUT = Path(
    "data/models/eaware_v2_step5_baseline/"
    "step5_training_dataset.csv"
)

OUTDIR = Path(
    "data/models/eaware_v2_step5_validation"
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


def evaluate_grouped(
    df,
    group_values,
    label,
):

    X = df[FEATURES].copy()

    y = (
        df["pb_target"]
        .astype(int)
        .to_numpy()
    )

    groups = np.asarray(
        group_values
    )

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        5,
        len(unique_groups),
    )

    if n_splits < 3:

        raise RuntimeError(
            f"{label}: insufficient groups "
            f"for robust validation."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    folds = []

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
            < 2
            or
            len(
                np.unique(
                    y_test
                )
            )
            < 2
        ):

            print(
                f"{label} fold {fold}: "
                "skipped — one class missing."
            )

            continue

        model = make_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            y_train,
        )

        score = (
            model
            .predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        oof[
            test_idx
        ] = score

        auc = roc_auc_score(
            y_test,
            score,
        )

        ap = average_precision_score(
            y_test,
            score,
        )

        folds.append(
            {
                "validation":
                    label,

                "fold":
                    fold,

                "train_rows":
                    len(train_idx),

                "test_rows":
                    len(test_idx),

                "train_groups":
                    len(
                        np.unique(
                            groups[
                                train_idx
                            ]
                        )
                    ),

                "test_groups":
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
                    auc,

                "average_precision":
                    ap,
            }
        )

        print(
            f"{label} fold {fold}: "
            f"ROC-AUC={auc:.3f} | "
            f"AP={ap:.3f}"
        )

    valid = np.isfinite(
        oof
    )

    if valid.sum() == 0:

        raise RuntimeError(
            f"{label}: no valid predictions."
        )

    overall_auc = roc_auc_score(
        y[
            valid
        ],
        oof[
            valid
        ],
    )

    overall_ap = average_precision_score(
        y[
            valid
        ],
        oof[
            valid
        ],
    )

    coverage = (
        valid.sum()
        /
        len(df)
    )

    summary = {
        "validation":
            label,

        "rows":
            len(df),

        "groups":
            len(
                unique_groups
            ),

        "folds":
            len(
                folds
            ),

        "prediction_coverage":
            float(
                coverage
            ),

        "roc_auc":
            float(
                overall_auc
            ),

        "average_precision":
            float(
                overall_ap
            ),
    }

    return (
        summary,
        pd.DataFrame(
            folds
        ),
    )


def spatial_groups(
    df,
    block_size,
):

    lat_block = np.floor(
        df["lat"].astype(float)
        /
        block_size
    ).astype(int)

    lon_block = np.floor(
        df["lon"].astype(float)
        /
        block_size
    ).astype(int)

    return (
        lat_block.astype(str)
        +
        "_"
        +
        lon_block.astype(str)
    )


def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing Step 5 dataset: {INPUT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        parse_dates=[
            "date"
        ],
    )

    required = (
        FEATURES
        +
        [
            "pb_target",
            "lat",
            "lon",
            "date",
        ]
    )

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=required
    ).copy()

    print(
        "E-AWARE V2 Step 5B — "
        "Spatial / Temporal Validation"
    )

    print()

    print(
        "Rows:",
        len(df)
    )

    print(
        "Presence:",
        int(
            (
                df["pb_target"]
                ==
                1
            ).sum()
        )
    )

    print(
        "Background:",
        int(
            (
                df["pb_target"]
                ==
                0
            ).sum()
        )
    )

    print()

    summaries = []

    all_folds = []

    # ----------------------------------
    # 1-degree spatial blocks
    # roughly ~111 km north/south
    # ----------------------------------

    print(
        "===== 1° SPATIAL BLOCK CV ====="
    )

    s1, f1 = evaluate_grouped(
        df,
        spatial_groups(
            df,
            1.0,
        ),
        "spatial_1_degree",
    )

    summaries.append(
        s1
    )

    all_folds.append(
        f1
    )

    print()

    # ----------------------------------
    # 2-degree spatial blocks
    # tougher geographic separation
    # ----------------------------------

    print(
        "===== 2° SPATIAL BLOCK CV ====="
    )

    s2, f2 = evaluate_grouped(
        df,
        spatial_groups(
            df,
            2.0,
        ),
        "spatial_2_degree",
    )

    summaries.append(
        s2
    )

    all_folds.append(
        f2
    )

    print()

    # ----------------------------------
    # Entire years held together
    # ----------------------------------

    print(
        "===== YEAR-GROUPED CV ====="
    )

    year_groups = (
        df[
            "date"
        ]
        .dt
        .year
        .astype(str)
    )

    sy, fy = evaluate_grouped(
        df,
        year_groups,
        "year_grouped",
    )

    summaries.append(
        sy
    )

    all_folds.append(
        fy
    )

    summary_df = pd.DataFrame(
        summaries
    )

    folds_df = pd.concat(
        all_folds,
        ignore_index=True,
    )

    summary_df.to_csv(
        OUTDIR
        /
        "step5b_validation_summary.csv",
        index=False,
    )

    folds_df.to_csv(
        OUTDIR
        /
        "step5b_fold_metrics.csv",
        index=False,
    )

    with open(
        OUTDIR
        /
        "step5b_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
        )

    print()

    print(
        "===== STEP 5B SUMMARY ====="
    )

    print(
        summary_df[
            [
                "validation",
                "groups",
                "roc_auc",
                "average_precision",
                "prediction_coverage",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "- Scores are relative habitat "
        "suitability discrimination."
    )

    print(
        "- Background is NOT biological absence."
    )

    print(
        "- Spatial performance is more important "
        "than the original random/date-only score "
        "for geographic deployment."
    )


if __name__ == "__main__":
    main()
