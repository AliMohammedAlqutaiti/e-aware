from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path("data/imarpe")

FEATURES_INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

ML_INPUT = (
    ROOT
    / "phase2_hsi_fold_relative_predictions.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "phase2_literature_vs_ml_summary.csv"
)

BANDS_OUTPUT = (
    ROOT
    / "phase2_literature_vs_ml_bands.csv"
)

DISAGREEMENT_OUTPUT = (
    ROOT
    / "phase2_literature_vs_ml_disagreements.csv"
)


# ==========================================================
# EXACT ORIGINAL HSI CONSTANTS
# ==========================================================

OXYGEN_MMOL_M3_TO_ML_L = 0.0223916
OXYGEN_MMOL_M3_TO_MG_L = 0.032

MG_L_TO_ML_L = (
    OXYGEN_MMOL_M3_TO_ML_L
    / OXYGEN_MMOL_M3_TO_MG_L
)


# ==========================================================
# ORIGINAL APP FUNCTIONS
# ==========================================================

def habitat_reference(month):

    if month in (12, 1, 2):

        return {
            "season": "Summer",
            "temp_min": 17.6,
            "temp_max": 23.7,
            "sal_min": 32.30,
            "sal_max": 35.14,
            "oxygen_min": 5.9,
            "oxygen_max": 8.7,
        }


    if month in (6, 7, 8):

        return {
            "season": "Winter",
            "temp_min": 14.5,
            "temp_max": 18.8,
            "sal_min": 34.81,
            "sal_max": 35.12,
            "oxygen_min": 5.2,
            "oxygen_max": 6.3,
        }


    transition = (
        "Autumn (Transition)"
        if month in (3, 4, 5)
        else "Spring (Transition)"
    )


    return {
        "season": transition,
        "temp_min": 14.5,
        "temp_max": 23.7,
        "sal_min": 32.30,
        "sal_max": 35.14,
        "oxygen_min": 5.2,
        "oxygen_max": 8.7,
    }


def range_score(
    values,
    minimum,
    maximum,
    shoulder,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    score = np.ones_like(
        values
    )


    below = (
        values < minimum
    )

    above = (
        values > maximum
    )


    score[below] = (
        1
        -
        (
            minimum
            - values[below]
        )
        / shoulder
    )


    score[above] = (
        1
        -
        (
            values[above]
            - maximum
        )
        / shoulder
    )


    return np.clip(
        score,
        0,
        1,
    )


def hsi_class(score):

    if score < 0.30:
        return "Low"

    if score < 0.60:
        return "Moderate"

    if score < 0.80:
        return "High"

    return "Very High"


# ==========================================================
# LOAD DATA
# ==========================================================

features = pd.read_csv(
    FEATURES_INPUT,
    parse_dates=["date"],
)


ml = pd.read_csv(
    ML_INPUT
)


features["month"] = (
    features["date"]
    .dt.month
)


# Only records used in the validated Phase 2 model.
eligible = (
    features["date"].notna()
    & (
        features["date"]
        >= pd.Timestamp("1993-01-01")
    )
    & features[
        [
            "deploy_sst_c",
            "deploy_sss_psu",
            "deploy_oxygen_mg_l",
        ]
    ]
    .notna()
    .all(axis=1)
)


features = (
    features[
        eligible
    ]
    .copy()
)


# ==========================================================
# CONVERT OXYGEN EXACTLY AS APP REQUIRES
# ==========================================================

features[
    "deploy_oxygen_ml_l"
] = (
    features[
        "deploy_oxygen_mg_l"
    ]
    * MG_L_TO_ML_L
)


# ==========================================================
# REPRODUCE ORIGINAL LITERATURE HSI
# ==========================================================

literature_rows = []


for month, group in features.groupby(
    "month"
):

    ref = habitat_reference(
        int(month)
    )


    temp_score = range_score(
        group["deploy_sst_c"],
        ref["temp_min"],
        ref["temp_max"],
        2.0,
    )


    sal_score = range_score(
        group["deploy_sss_psu"],
        ref["sal_min"],
        ref["sal_max"],
        0.5,
    )


    oxygen_score = range_score(
        group["deploy_oxygen_ml_l"],
        ref["oxygen_min"],
        ref["oxygen_max"],
        1.0,
    )


    hsi = (
        temp_score
        + sal_score
        + oxygen_score
    ) / 3.0


    part = group[
        [
            "record_id",
            "date",
            "source_group",
            "presence",
            "deploy_sst_c",
            "deploy_sss_psu",
            "deploy_oxygen_mg_l",
            "deploy_oxygen_ml_l",
        ]
    ].copy()


    part["literature_season"] = (
        ref["season"]
    )

    part["temperature_score"] = (
        temp_score
    )

    part["salinity_score"] = (
        sal_score
    )

    part["oxygen_score"] = (
        oxygen_score
    )

    part["literature_hsi"] = (
        hsi
    )

    part["literature_class"] = [
        hsi_class(x)
        for x in hsi
    ]


    literature_rows.append(
        part
    )


literature = pd.concat(
    literature_rows,
    ignore_index=True,
)


# ==========================================================
# LITERATURE HSI RANK
#
# Only used for top-10% / top-20% comparison.
# Original literature classes still use the actual
# 0.30 / 0.60 / 0.80 HSI boundaries.
# ==========================================================

literature[
    "literature_percentile"
] = (
    literature[
        "literature_hsi"
    ]
    .rank(
        method="average",
        pct=True,
    )
)


# ==========================================================
# MERGE WITH STRICT ML HSI PREDICTIONS
# ==========================================================

comparison = ml.merge(
    literature[
        [
            "record_id",
            "literature_season",
            "temperature_score",
            "salinity_score",
            "oxygen_score",
            "literature_hsi",
            "literature_class",
            "literature_percentile",
        ]
    ],
    on="record_id",
    how="left",
    validate="many_to_one",
)


comparison[
    "ml_class"
] = comparison[
    "hsi"
].apply(
    hsi_class
)


# ==========================================================
# CLASS ORDERS FOR DISAGREEMENT
# ==========================================================

CLASS_ORDER = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 3,
}


comparison[
    "literature_class_number"
] = (
    comparison[
        "literature_class"
    ]
    .map(
        CLASS_ORDER
    )
)


comparison[
    "ml_class_number"
] = (
    comparison[
        "ml_class"
    ]
    .map(
        CLASS_ORDER
    )
)


comparison[
    "class_difference"
] = (
    comparison[
        "ml_class_number"
    ]
    -
    comparison[
        "literature_class_number"
    ]
)


# ==========================================================
# METRICS
# ==========================================================

summary_rows = []
band_rows = []


for validation in comparison[
    "validation"
].unique():

    subset = comparison[
        comparison[
            "validation"
        ]
        == validation
    ].copy()


    y = (
        subset[
            "presence"
        ]
        .astype(int)
        .to_numpy()
    )


    baseline = float(
        y.mean()
    )


    # ------------------------------------------------------
    # Continuous discrimination
    # ------------------------------------------------------

    literature_roc = roc_auc_score(
        y,
        subset[
            "literature_hsi"
        ],
    )

    literature_pr = (
        average_precision_score(
            y,
            subset[
                "literature_hsi"
            ],
        )
    )


    ml_roc = roc_auc_score(
        y,
        subset[
            "hsi"
        ],
    )

    ml_pr = (
        average_precision_score(
            y,
            subset[
                "hsi"
            ],
        )
    )


    rho, rho_p = spearmanr(
        subset[
            "literature_hsi"
        ],
        subset[
            "hsi"
        ],
    )


    # ------------------------------------------------------
    # Top 20% / 10%
    # ------------------------------------------------------

    literature_top20 = subset[
        subset[
            "literature_percentile"
        ]
        >= 0.80
    ]

    literature_top10 = subset[
        subset[
            "literature_percentile"
        ]
        >= 0.90
    ]


    ml_top20 = subset[
        subset[
            "hsi"
        ]
        >= 0.80
    ]

    ml_top10 = subset[
        subset[
            "hsi"
        ]
        >= 0.90
    ]


    def rate(frame):

        if len(frame) == 0:
            return np.nan

        return float(
            frame[
                "presence"
            ].mean()
        )


    summary_rows.extend(
        [
            {
                "validation":
                    validation,

                "model":
                    "LITERATURE_HSI",

                "roc_auc":
                    literature_roc,

                "pr_auc":
                    literature_pr,

                "baseline":
                    baseline,

                "top20_presence_rate":
                    rate(
                        literature_top20
                    ),

                "top20_lift":
                    (
                        rate(
                            literature_top20
                        )
                        / baseline
                    ),

                "top20_n":
                    len(
                        literature_top20
                    ),

                "top10_presence_rate":
                    rate(
                        literature_top10
                    ),

                "top10_lift":
                    (
                        rate(
                            literature_top10
                        )
                        / baseline
                    ),

                "top10_n":
                    len(
                        literature_top10
                    ),

                "spearman_with_other":
                    rho,
            },

            {
                "validation":
                    validation,

                "model":
                    "VALIDATED_ML_HSI",

                "roc_auc":
                    ml_roc,

                "pr_auc":
                    ml_pr,

                "baseline":
                    baseline,

                "top20_presence_rate":
                    rate(
                        ml_top20
                    ),

                "top20_lift":
                    (
                        rate(
                            ml_top20
                        )
                        / baseline
                    ),

                "top20_n":
                    len(
                        ml_top20
                    ),

                "top10_presence_rate":
                    rate(
                        ml_top10
                    ),

                "top10_lift":
                    (
                        rate(
                            ml_top10
                        )
                        / baseline
                    ),

                "top10_n":
                    len(
                        ml_top10
                    ),

                "spearman_with_other":
                    rho,
            },
        ]
    )


    # ------------------------------------------------------
    # BAND PERFORMANCE
    # ------------------------------------------------------

    for model_name, class_col in [
        (
            "LITERATURE_HSI",
            "literature_class",
        ),
        (
            "VALIDATED_ML_HSI",
            "ml_class",
        ),
    ]:

        bands = (
            subset.groupby(
                class_col,
                observed=False,
            )
            .agg(
                records=(
                    "presence",
                    "size",
                ),

                detections=(
                    "presence",
                    "sum",
                ),

                presence_rate=(
                    "presence",
                    "mean",
                ),
            )
            .reset_index()
        )


        bands[
            "lift_vs_baseline"
        ] = (
            bands[
                "presence_rate"
            ]
            / baseline
        )


        bands[
            "validation"
        ] = validation

        bands[
            "model"
        ] = model_name

        bands = bands.rename(
            columns={
                class_col:
                    "hsi_class"
            }
        )


        band_rows.append(
            bands
        )


# ==========================================================
# OUTPUT TABLES
# ==========================================================

summary = pd.DataFrame(
    summary_rows
)


bands = pd.concat(
    band_rows,
    ignore_index=True,
)


print("=" * 100)
print(
    "E-AWARE — LITERATURE HSI vs VALIDATED ML HSI"
)
print("=" * 100)


print(
    "\nOxygen conversion "
    "(mg/L → mL/L):",
    round(
        MG_L_TO_ML_L,
        7,
    )
)


print("\n" + "=" * 100)
print("SEASON COVERAGE")
print("=" * 100)


season_coverage = (
    literature.groupby(
        "literature_season"
    )
    .agg(
        records=(
            "presence",
            "size",
        ),

        detections=(
            "presence",
            "sum",
        ),
    )
)


season_coverage[
    "detection_rate"
] = (
    season_coverage[
        "detections"
    ]
    / season_coverage[
        "records"
    ]
)


print(
    season_coverage
    .round(4)
    .to_string()
)


print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)


print(
    summary[
        [
            "validation",
            "model",
            "roc_auc",
            "pr_auc",
            "baseline",
            "top20_presence_rate",
            "top20_lift",
            "top20_n",
            "top10_presence_rate",
            "top10_lift",
            "top10_n",
            "spearman_with_other",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("HSI BAND COMPARISON")
print("=" * 100)


for validation in bands[
    "validation"
].unique():

    print("\n" + validation)


    for model_name in [
        "LITERATURE_HSI",
        "VALIDATED_ML_HSI",
    ]:

        print(
            "\n",
            model_name,
            sep="",
        )


        table = bands[
            (
                bands[
                    "validation"
                ]
                == validation
            )
            &
            (
                bands[
                    "model"
                ]
                == model_name
            )
        ].copy()


        table[
            "order"
        ] = table[
            "hsi_class"
        ].map(
            CLASS_ORDER
        )


        table = table.sort_values(
            "order"
        )


        print(
            table[
                [
                    "hsi_class",
                    "records",
                    "detections",
                    "presence_rate",
                    "lift_vs_baseline",
                ]
            ]
            .round(4)
            .to_string(
                index=False
            )
        )


# ==========================================================
# AGREEMENT / DISAGREEMENT
# ==========================================================

print("\n" + "=" * 100)
print("MODEL AGREEMENT")
print("=" * 100)


for validation in comparison[
    "validation"
].unique():

    subset = comparison[
        comparison[
            "validation"
        ]
        == validation
    ]


    exact = (
        subset[
            "class_difference"
        ]
        == 0
    ).mean()


    within_one = (
        subset[
            "class_difference"
        ]
        .abs()
        <= 1
    ).mean()


    strong_disagreement = (
        subset[
            "class_difference"
        ]
        .abs()
        >= 2
    ).mean()


    print(
        f"\n{validation}"
    )

    print(
        "Exact class agreement:",
        round(
            exact,
            4
        )
    )

    print(
        "Within one class:",
        round(
            within_one,
            4
        )
    )

    print(
        "Strong disagreement (>=2 classes):",
        round(
            strong_disagreement,
            4
        )
    )


# ==========================================================
# SAVE STRONG DISAGREEMENTS
# ==========================================================

disagreement = comparison[
    comparison[
        "class_difference"
    ]
    .abs()
    >= 2
].copy()


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


bands.to_csv(
    BANDS_OUTPUT,
    index=False,
)


disagreement.to_csv(
    DISAGREEMENT_OUTPUT,
    index=False,
)


comparison.to_csv(
    ROOT
    / "phase2_literature_vs_ml_all_records.csv",
    index=False,
)


print("\nSaved:")
print(SUMMARY_OUTPUT)
print(BANDS_OUTPUT)
print(DISAGREEMENT_OUTPUT)
