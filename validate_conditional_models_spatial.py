#!/usr/bin/env python3

"""
E-AWARE V2 — Step 5 V4
Spatial and temporal validation of matched conditional models.

Models:
1. coast_only
2. oceanography
3. oceanography_plus_coast

Validation:
- 1-degree anchor spatial blocks
- 2-degree anchor spatial blocks
- anchor year

All four rows belonging to an anchor stay together.

IMPORTANT:
Backgrounds are locally available habitat, NOT confirmed absence.
Scores represent relative habitat suitability / ranking.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
)


INPUT = Path(
    "data/processed/eaware_v2_step4_local/"
    "eaware_v2_step4_local_balanced_candidate.csv"
)

OUTDIR = Path(
    "data/models/eaware_v2_conditional_models"
)

MODELS = {
    "coast_only": [
        "historical_distance_coast_km",
    ],

    "oceanography": [
        "historical_sst_c",
        "historical_salinity_psu",
        "historical_surface_o2_ml_l",
    ],

    "oceanography_plus_coast": [
        "historical_sst_c",
        "historical_salinity_psu",
        "historical_surface_o2_ml_l",
        "historical_distance_coast_km",
    ],
}


def fit_model(X, y, anchors):

    model = ConditionalLogit(
        y,
        X,
        groups=anchors,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = model.fit(
            method="bfgs",
            maxiter=3000,
            disp=False,
        )

    return result


def matched_metrics(frame, score):

    temp = frame[
        [
            "qc_anchor_id",
            "pb_target",
        ]
    ].copy()

    temp["score"] = score

    concordance = []
    top1 = []
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

        concordance.append(
            float(
                np.mean(
                    (p > b).astype(float)
                    +
                    0.5
                    *
                    (p == b).astype(float)
                )
            )
        )

        all_scores = np.concatenate(
            [[p], b]
        )

        maximum = np.max(
            all_scores
        )

        tied_max = np.sum(
            np.isclose(
                all_scores,
                maximum,
            )
        )

        if np.isclose(
            p,
            maximum,
        ):
            top1.append(
                1.0 / tied_max
            )
        else:
            top1.append(
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
            + 1
            + (equal - 1) / 2
        )

        ranks.append(
            float(rank)
        )

    return {
        "within_anchor_concordance":
            float(
                np.mean(
                    concordance
                )
            ),

        "top1_accuracy":
            float(
                np.mean(
                    top1
                )
            ),

        "mean_presence_rank":
            float(
                np.mean(
                    ranks
                )
            ),

        "anchors":
            len(
                concordance
            ),
    }


def spatial_block(lat, lon, size):

    lat_block = np.floor(
        lat / size
    ).astype(int)

    lon_block = np.floor(
        lon / size
    ).astype(int)

    return (
        lat_block.astype(str)
        +
        "_"
        +
        lon_block.astype(str)
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
        "E-AWARE V2 Step 5 V4 — "
        "Spatial / Temporal Validation"
    )

    print()

    print(
        "Rows:",
        len(df)
    )

    print(
        "Anchors:",
        df[
            "qc_anchor_id"
        ].nunique()
    )

    # --------------------------------------------------
    # Anchor metadata comes from the presence row.
    # --------------------------------------------------

    presence_rows = (
        df[
            df["pb_target"] == 1
        ][
            [
                "qc_anchor_id",
                "lat",
                "lon",
                "date",
            ]
        ]
        .drop_duplicates(
            "qc_anchor_id"
        )
        .copy()
    )

    presence_rows = (
        presence_rows
        .rename(
            columns={
                "lat":
                    "anchor_lat",
                "lon":
                    "anchor_lon",
                "date":
                    "anchor_date",
            }
        )
    )

    df = df.merge(
        presence_rows,
        on="qc_anchor_id",
        how="left",
    )

    df["anchor_year"] = (
        pd.to_datetime(
            df["anchor_date"]
        )
        .dt.year
        .astype(int)
    )

    df["block_1deg"] = spatial_block(
        df["anchor_lat"],
        df["anchor_lon"],
        1.0,
    )

    df["block_2deg"] = spatial_block(
        df["anchor_lat"],
        df["anchor_lon"],
        2.0,
    )

    VALIDATIONS = {
        "spatial_1deg":
            "block_1deg",

        "spatial_2deg":
            "block_2deg",

        "year":
            "anchor_year",
    }

    print(
        "Unique 1° blocks:",
        df[
            "block_1deg"
        ].nunique()
    )

    print(
        "Unique 2° blocks:",
        df[
            "block_2deg"
        ].nunique()
    )

    print(
        "Unique years:",
        df[
            "anchor_year"
        ].nunique()
    )

    results = []

    # ==================================================
    # MODEL LOOP
    # ==================================================

    for model_name, features in (
        MODELS.items()
    ):

        print()
        print(
            "======================================"
        )
        print(
            "MODEL:",
            model_name
        )
        print(
            "======================================"
        )

        X_all = (
            df[
                features
            ]
            .astype(float)
        )

        y_all = (
            df["pb_target"]
            .to_numpy()
        )

        anchors_all = (
            df["qc_anchor_id"]
            .to_numpy()
        )

        # ==============================================
        # VALIDATION LOOP
        # ==============================================

        for validation_name, group_col in (
            VALIDATIONS.items()
        ):

            validation_groups = (
                df[group_col]
                .to_numpy()
            )

            unique_groups = (
                pd.Series(
                    validation_groups
                )
                .nunique()
            )

            n_splits = min(
                5,
                unique_groups,
            )

            if n_splits < 2:
                continue

            splitter = GroupKFold(
                n_splits=n_splits
            )

            fold_rows = []

            print()
            print(
                validation_name
            )

            for fold, (
                train_idx,
                test_idx,
            ) in enumerate(
                splitter.split(
                    X_all,
                    y_all,
                    validation_groups,
                ),
                start=1,
            ):

                X_train_raw = (
                    X_all
                    .iloc[
                        train_idx
                    ]
                    .to_numpy()
                )

                X_test_raw = (
                    X_all
                    .iloc[
                        test_idx
                    ]
                    .to_numpy()
                )

                y_train = (
                    y_all[
                        train_idx
                    ]
                )

                y_test = (
                    y_all[
                        test_idx
                    ]
                )

                anchor_train = (
                    anchors_all[
                        train_idx
                    ]
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

                result = fit_model(
                    X_train,
                    y_train,
                    anchor_train,
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

                ap = (
                    average_precision_score(
                        y_test,
                        score,
                    )
                )

                test_frame = (
                    df
                    .iloc[
                        test_idx
                    ]
                    .copy()
                )

                mm = matched_metrics(
                    test_frame,
                    score,
                )

                row = {
                    "model":
                        model_name,

                    "validation":
                        validation_name,

                    "fold":
                        fold,

                    "test_anchors":
                        mm[
                            "anchors"
                        ],

                    "roc_auc":
                        float(
                            auc
                        ),

                    "average_precision":
                        float(
                            ap
                        ),

                    "within_anchor_concordance":
                        mm[
                            "within_anchor_concordance"
                        ],

                    "top1_accuracy":
                        mm[
                            "top1_accuracy"
                        ],

                    "mean_presence_rank":
                        mm[
                            "mean_presence_rank"
                        ],
                }

                fold_rows.append(
                    row
                )

                results.append(
                    row
                )

                print(
                    f" Fold {fold}: "
                    f"AUC={auc:.3f} | "
                    f"Concordance="
                    f"{mm['within_anchor_concordance']:.3f} | "
                    f"Top1="
                    f"{mm['top1_accuracy']:.3f} | "
                    f"Rank="
                    f"{mm['mean_presence_rank']:.2f} | "
                    f"anchors="
                    f"{mm['anchors']}"
                )

            fold_df = pd.DataFrame(
                fold_rows
            )

            print(
                " Mean AUC:",
                round(
                    fold_df[
                        "roc_auc"
                    ].mean(),
                    3,
                )
            )

            print(
                " Mean concordance:",
                round(
                    fold_df[
                        "within_anchor_concordance"
                    ].mean(),
                    3,
                )
            )

            print(
                " Mean Top1:",
                round(
                    fold_df[
                        "top1_accuracy"
                    ].mean(),
                    3,
                )
            )

            print(
                " Mean presence rank:",
                round(
                    fold_df[
                        "mean_presence_rank"
                    ].mean(),
                    3,
                )
            )

    # --------------------------------------------------
    # SAVE + SUMMARY
    # --------------------------------------------------

    out = pd.DataFrame(
        results
    )

    out.to_csv(
        OUTDIR
        /
        "spatial_temporal_validation.csv",
        index=False,
    )

    summary = (
        out.groupby(
            [
                "model",
                "validation",
            ]
        )
        .agg(
            mean_auc=(
                "roc_auc",
                "mean",
            ),
            sd_auc=(
                "roc_auc",
                "std",
            ),
            mean_ap=(
                "average_precision",
                "mean",
            ),
            mean_concordance=(
                "within_anchor_concordance",
                "mean",
            ),
            sd_concordance=(
                "within_anchor_concordance",
                "std",
            ),
            mean_top1=(
                "top1_accuracy",
                "mean",
            ),
            mean_presence_rank=(
                "mean_presence_rank",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTDIR
        /
        "spatial_temporal_validation_summary.csv",
        index=False,
    )

    print()
    print(
        "======================================"
    )

    print(
        "STEP 5 V4 SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print(
        "Interpretation guide:"
    )

    print(
        "Random within-anchor concordance ≈ 0.500"
    )

    print(
        "Random Top-1 ≈ 0.250"
    )

    print(
        "Random mean presence rank ≈ 2.500"
    )

    print()

    print(
        "If coast_only performs nearly the same "
        "as oceanography_plus_coast, the apparent "
        "gain is mainly driven by coastline distance."
    )

    print(
        "If performance collapses under spatial "
        "blocking, geographic / observer bias is "
        "likely contributing."
    )

    print()

    print(
        "===== STEP 5 V4 COMPLETE ====="
    )


if __name__ == "__main__":
    main()
