from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/imarpe")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

MODEL_OUTPUT = (
    MODEL_DIR
    / "eaware_phase2_logistic.joblib"
)

REFERENCE_OUTPUT = (
    MODEL_DIR
    / "eaware_phase2_reference_scores.npy"
)

METADATA_OUTPUT = (
    MODEL_DIR
    / "eaware_phase2_model_metadata.json"
)

COEF_OUTPUT = (
    ROOT
    / "phase2_production_model_coefficients.csv"
)


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
)


eligible = (
    df["date"].notna()
    & (
        df["date"]
        >= pd.Timestamp("1993-01-01")
    )
)


required = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_bottom_depth_m",
    "sst_anomaly_c",
    "month_sin",
    "month_cos",
]


complete = (
    df[required]
    .notna()
    .all(axis=1)
)


data = df[
    eligible & complete
].copy()


data[
    "deploy_log_bottom_depth"
] = np.log1p(
    data[
        "deploy_bottom_depth_m"
    ]
)


FEATURES = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
    "sst_anomaly_c",
    "month_sin",
    "month_cos",
]


X = (
    data[
        FEATURES
    ]
    .to_numpy()
)


y = (
    data[
        "presence"
    ]
    .astype(int)
    .to_numpy()
)


# ==========================================================
# FINAL PRODUCTION MODEL
# ==========================================================

FINAL_C = 0.1


model = make_pipeline(
    StandardScaler(),

    LogisticRegression(
        C=FINAL_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    ),
)


model.fit(
    X,
    y,
)


# ==========================================================
# HISTORICAL REFERENCE SCORE DISTRIBUTION
#
# New operational cells will be converted to HSI using:
#
# fraction of historical reference scores <= new raw score
#
# This makes HSI a habitat-ranking index, NOT a probability.
# ==========================================================

training_scores = (
    model.predict_proba(
        X
    )[:, 1]
)


reference_scores = np.sort(
    training_scores
)


np.save(
    REFERENCE_OUTPUT,
    reference_scores,
)


joblib.dump(
    model,
    MODEL_OUTPUT,
)


# ==========================================================
# STANDARDISED COEFFICIENTS
# ==========================================================

logistic = (
    model.steps[-1][1]
)


coefficients = pd.DataFrame(
    {
        "feature": FEATURES,
        "standardized_coefficient":
            logistic.coef_[0],
    }
)


coefficients[
    "odds_multiplier_per_1sd"
] = np.exp(
    coefficients[
        "standardized_coefficient"
    ]
)


coefficients.to_csv(
    COEF_OUTPUT,
    index=False,
)


# ==========================================================
# MODEL METADATA
# ==========================================================

metadata = {
    "model_name":
        "E-AWARE Phase 2 Anchoveta Habitat Ranking Model",

    "model_type":
        "balanced logistic regression",

    "target":
        "anchoveta presence / absence",

    "records":
        int(len(data)),

    "detections":
        int(y.sum()),

    "non_detections":
        int((y == 0).sum()),

    "historical_sample_prevalence":
        float(y.mean()),

    "C":
        FINAL_C,

    "class_weight":
        "balanced",

    "features":
        FEATURES,

    "raw_score_interpretation":
        (
            "Internal balanced-logistic ranking score. "
            "Do not display as literal occurrence probability."
        ),

    "hsi_interpretation":
        (
            "Percentile of the model score relative to the "
            "historical E-AWARE training reference distribution."
        ),

    "hsi_classes": {
        "Low": [
            0.0,
            0.3,
        ],

        "Moderate": [
            0.3,
            0.6,
        ],

        "High": [
            0.6,
            0.8,
        ],

        "Very High": [
            0.8,
            1.0,
        ],
    },

    "validation": {
        "nested_leave_one_survey_out_roc_auc":
            0.7550,

        "nested_leave_one_survey_out_pr_auc":
            0.4242,

        "nested_leave_one_year_out_roc_auc":
            0.7269,

        "nested_leave_one_year_out_pr_auc":
            0.4010,

        "historical_baseline_presence":
            0.2435,
    },

    "operational_warning":
        (
            "HSI represents relative habitat suitability, "
            "not exact fish location and not a calibrated "
            "probability of anchoveta presence."
        ),
}


with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ==========================================================
# DEMONSTRATE HSI CONVERSION
# ==========================================================

def score_to_hsi(
    score,
):

    return (
        np.searchsorted(
            reference_scores,
            score,
            side="right",
        )
        / len(
            reference_scores
        )
    )


training_hsi = np.array(
    [
        score_to_hsi(
            value
        )
        for value in training_scores
    ]
)


print("=" * 90)
print(
    "E-AWARE PHASE 2 — "
    "PRODUCTION MODEL CREATED"
)
print("=" * 90)


print("\nTraining records:", len(data))
print(
    "Detections:",
    int(y.sum())
)
print(
    "Prevalence:",
    round(
        y.mean(),
        4
    )
)


print("\nModel:")
print(
    "Logistic Regression"
)
print(
    "C:",
    FINAL_C
)
print(
    "Class weight: balanced"
)


print("\nFeatures:")

for feature in FEATURES:

    print(
        " -",
        feature
    )


print("\n" + "=" * 90)
print(
    "STANDARDISED FINAL COEFFICIENTS"
)
print("=" * 90)


print(
    coefficients
    .sort_values(
        "standardized_coefficient",
        ascending=False,
    )
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("REFERENCE SCORE DISTRIBUTION")
print("=" * 90)


for q in [
    0.10,
    0.30,
    0.50,
    0.60,
    0.80,
    0.90,
    0.95,
]:

    print(
        f"{int(q * 100):>2}%:",
        round(
            float(
                np.quantile(
                    reference_scores,
                    q,
                )
            ),
            6,
        )
    )


print("\nHSI range:")
print(
    round(
        training_hsi.min(),
        4
    ),
    "to",
    round(
        training_hsi.max(),
        4
    )
)


print("\nSaved:")
print(MODEL_OUTPUT)
print(REFERENCE_OUTPUT)
print(METADATA_OUTPUT)
print(COEF_OUTPUT)
