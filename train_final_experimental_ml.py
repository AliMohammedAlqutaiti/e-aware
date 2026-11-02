#!/usr/bin/env python3

"""
E-AWARE V2
Final experimental ML model.

This model is NOT the primary operational E-AWARE model.

Primary operational model:
    Literature-based habitat suitability index (HSI).

Experimental ML:
    Matched conditional logistic regression using:
    - SST
    - salinity
    - surface dissolved oxygen
    - whether the 1 mL/L oxycline occurs before seabed
    - physically valid oxycline depth when reached

Training data:
    Coast-matched local-availability C5 dataset.

Interpretation:
    Relative used-vs-locally-available habitat score.

NOT:
    Probability of anchoveta presence.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

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
    "eaware_v2_final_experimental_ml"
)

MODEL_OUT = (
    OUTDIR
    /
    "eaware_experimental_conditional_ml.joblib"
)

COEFFICIENT_OUT = (
    OUTDIR
    /
    "experimental_ml_coefficients.csv"
)

METADATA_OUT = (
    OUTDIR
    /
    "experimental_ml_metadata.json"
)


CORE = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
]

FEATURES = CORE + [
    "oxycline_reached_before_seabed",
    "oxycline_depth_when_reached_m",
]

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

    # --------------------------------------------------
    # Require valid local water depth for every row
    # belonging to an anchor.
    # --------------------------------------------------

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

    valid_anchors = set(
        depth_complete[
            depth_complete
        ]
        .index
        .astype(int)
    )

    work = df[
        df["qc_anchor_id"]
        .isin(
            valid_anchors
        )
    ].copy()

    # --------------------------------------------------
    # Physically valid oxycline encoding.
    # --------------------------------------------------

    allowed = {
        ACCEPTED,
        BELOW,
        NOT_REACHED,
    }

    statuses = set(
        work[
            "historical_oxycline_status"
        ]
        .dropna()
        .unique()
    )

    unexpected = (
        statuses - allowed
    )

    if unexpected:

        raise RuntimeError(
            f"Unexpected oxycline statuses: "
            f"{unexpected}"
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

    if (
        work[FEATURES]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Missing values remain in "
            "experimental model predictors."
        )

    # --------------------------------------------------
    # Verify matched-set structure.
    # --------------------------------------------------

    check = (
        work.groupby(
            "qc_anchor_id"
        )
        .agg(
            rows=(
                "pb_target",
                "size",
            ),
            presence_count=(
                "pb_target",
                "sum",
            ),
        )
    )

    bad = check[
        ~(
            (check["rows"] == 4)
            &
            (check["presence_count"] == 1)
        )
    ]

    if len(bad):

        raise RuntimeError(
            "Invalid matched anchor sets."
        )

    X_raw = (
        work[
            FEATURES
        ]
        .astype(float)
        .to_numpy()
    )

    y = (
        work[
            "pb_target"
        ]
        .astype(int)
        .to_numpy()
    )

    groups = (
        work[
            "qc_anchor_id"
        ]
        .astype(int)
        .to_numpy()
    )

    scaler = (
        StandardScaler()
        .fit(
            X_raw
        )
    )

    X = scaler.transform(
        X_raw
    )

    model = ConditionalLogit(
        y,
        X,
        groups=groups,
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        result = model.fit(
            method="bfgs",
            maxiter=4000,
            disp=False,
        )

    coefficients = np.asarray(
        result.params,
        dtype=float,
    )

    coef_df = pd.DataFrame(
        {
            "predictor":
                FEATURES,

            "standardized_coefficient":
                coefficients,
        }
    )

    coef_df[
        "absolute_coefficient"
    ] = (
        coef_df[
            "standardized_coefficient"
        ].abs()
    )

    coef_df = (
        coef_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------
    # Training envelope for later live-data warnings.
    # --------------------------------------------------

    training_ranges = {}

    for feature in FEATURES:

        values = (
            work[
                feature
            ]
            .astype(float)
        )

        training_ranges[
            feature
        ] = {
            "min":
                float(
                    values.min()
                ),

            "max":
                float(
                    values.max()
                ),

            "mean":
                float(
                    values.mean()
                ),

            "median":
                float(
                    values.median()
                ),
        }

    bundle = {
        "model_name":
            "E-AWARE experimental conditional ML",

        "model_type":
            "conditional_logistic_regression",

        "role":
            "experimental_secondary_model",

        "primary_operational_model":
            "literature_based_HSI",

        "features":
            FEATURES,

        "scaler":
            scaler,

        "coefficients":
            coefficients,

        "intercept":
            None,

        "training_anchors":
            int(
                work[
                    "qc_anchor_id"
                ].nunique()
            ),

        "training_rows":
            int(
                len(work)
            ),

        "background_is_absence":
            False,

        "score_interpretation":
            (
                "relative used-vs-locally-available "
                "habitat suitability"
            ),

        "probability_of_presence":
            False,

        "training_ranges":
            training_ranges,

        "validation_summary":
            {
                "year_holdout_concordance":
                    0.598,

                "year_holdout_top1":
                    0.324,

                "year_holdout_weighted_auc":
                    0.550,

                "spatial_validation":
                    (
                        "insufficient because valid-depth "
                        "occurrences are strongly spatially "
                        "concentrated"
                    ),
            },

        "limitations": [
            (
                "Model trained on sparse public GBIF "
                "occurrence data."
            ),
            (
                "Spatial transfer could not be robustly "
                "validated."
            ),
            (
                "Scores must not be interpreted as "
                "probability of anchoveta presence."
            ),
            (
                "Model should be retrained if IMARPE "
                "survey observations become available."
            ),
        ],
    }

    joblib.dump(
        bundle,
        MODEL_OUT,
    )

    coef_df.to_csv(
        COEFFICIENT_OUT,
        index=False,
    )

    metadata = {
        k: v
        for k, v in bundle.items()
        if k not in {
            "scaler",
            "coefficients",
        }
    }

    with open(
        METADATA_OUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "===== FINAL EXPERIMENTAL ML MODEL ====="
    )

    print(
        "Training anchors:",
        work[
            "qc_anchor_id"
        ].nunique()
    )

    print(
        "Training rows:",
        len(work)
    )

    print()

    print(
        "STANDARDIZED COEFFICIENTS"
    )

    print(
        coef_df[
            [
                "predictor",
                "standardized_coefficient",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Saved model:"
    )

    print(
        MODEL_OUT
    )

    print()

    print(
        "Operational status:"
    )

    print(
        "HSI = PRIMARY"
    )

    print(
        "Conditional ML = EXPERIMENTAL SECONDARY"
    )

    print()

    print(
        "===== MODEL FREEZE COMPLETE ====="
    )


if __name__ == "__main__":
    main()
