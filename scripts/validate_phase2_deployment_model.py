from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_deployment_consistent.csv"
)

OUTPUT = (
    ROOT
    / "phase2_deployment_validation.csv"
)

FOLD_OUTPUT = (
    ROOT
    / "phase2_deployment_validation_folds.csv"
)


df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
)

eligible = (
    df["date"].notna()
    & (df["date"] >= pd.Timestamp("1993-01-01"))
)


required = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_chlorophyll_mg_m3",
    "deploy_bottom_depth_m",
    "sst_anomaly_c",
]


complete = (
    df[required]
    .notna()
    .all(axis=1)
)


data = df[
    eligible & complete
].copy()


print("=" * 90)
print("E-AWARE PHASE 2 — DEPLOYMENT MODEL VALIDATION")
print("=" * 90)

print("\nRecords:", len(data))
print(
    "Independent surveys:",
    data["source_group"].nunique()
)
print(
    "Detections:",
    int(data["presence"].sum())
)
print(
    "Non-detections:",
    int((data["presence"] == 0).sum())
)
print(
    "Prevalence:",
    round(data["presence"].mean(), 4)
)


# ----------------------------------------------------------
# Transform skewed predictors.
# ----------------------------------------------------------

data["deploy_log_bottom_depth"] = np.log1p(
    data["deploy_bottom_depth_m"]
)

data["deploy_log_chlorophyll"] = np.log1p(
    data["deploy_chlorophyll_mg_m3"]
)


core = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
]

core_chl = core + [
    "deploy_log_chlorophyll",
]

core_anomaly = core + [
    "sst_anomaly_c",
]

core_chl_anomaly = core + [
    "deploy_log_chlorophyll",
    "sst_anomaly_c",
]


# ----------------------------------------------------------
# Diagnostic within-survey versions.
# These are NOT candidates for deployment.
# ----------------------------------------------------------

def make_centered(features):

    centered = []

    for col in features:

        new = f"{col}_within_survey"

        data[new] = (
            data[col]
            - data.groupby(
                "source_group"
            )[col].transform("mean")
        )

        centered.append(new)

    return centered


core_within = make_centered(core)
core_chl_within = make_centered(core_chl)
core_anomaly_within = make_centered(core_anomaly)
core_chl_anomaly_within = make_centered(
    core_chl_anomaly
)


def evaluate(features, label):

    X = data[features].to_numpy()

    y = (
        data["presence"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        data["source_group"]
        .to_numpy()
    )

    logo = LeaveOneGroupOut()

    pred = np.full(
        len(data),
        np.nan
    )

    fold_results = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups
        ),
        start=1,
    ):

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        )

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        probability = (
            model.predict_proba(
                X[test_idx]
            )[:, 1]
        )

        pred[test_idx] = probability

        y_test = y[test_idx]

        group_name = groups[
            test_idx
        ][0]

        # Per-survey ROC is undefined if the survey
        # contains only one class.
        if len(np.unique(y_test)) == 2:

            fold_roc = roc_auc_score(
                y_test,
                probability
            )

            fold_pr = average_precision_score(
                y_test,
                probability
            )

        else:

            fold_roc = np.nan

            fold_pr = np.nan

        fold_results.append(
            {
                "model": label,
                "fold": fold,
                "source_group": group_name,
                "records": len(test_idx),
                "detections": int(
                    y_test.sum()
                ),
                "prevalence": float(
                    y_test.mean()
                ),
                "roc_auc": fold_roc,
                "pr_auc": fold_pr,
                "mean_prediction": float(
                    probability.mean()
                ),
            }
        )


    pooled_roc = roc_auc_score(
        y,
        pred
    )

    pooled_pr = average_precision_score(
        y,
        pred
    )

    folds = pd.DataFrame(
        fold_results
    )


    result = {
        "model": label,
        "records": len(y),
        "surveys": len(
            np.unique(groups)
        ),
        "pooled_roc_auc": pooled_roc,
        "pooled_pr_auc": pooled_pr,
        "baseline_prevalence": y.mean(),
        "median_mixed_survey_roc_auc": (
            folds["roc_auc"]
            .dropna()
            .median()
        ),
        "median_mixed_survey_pr_auc": (
            folds["pr_auc"]
            .dropna()
            .median()
        ),
    }

    return result, folds


experiments = [
    (
        "DEPLOY_CORE_RAW",
        core,
    ),
    (
        "DEPLOY_CORE_CHL_RAW",
        core_chl,
    ),
    (
        "DEPLOY_CORE_ANOMALY_RAW",
        core_anomaly,
    ),
    (
        "DEPLOY_CORE_CHL_ANOMALY_RAW",
        core_chl_anomaly,
    ),

    (
        "DEPLOY_CORE_WITHIN_DIAGNOSTIC",
        core_within,
    ),
    (
        "DEPLOY_CORE_CHL_WITHIN_DIAGNOSTIC",
        core_chl_within,
    ),
    (
        "DEPLOY_CORE_ANOMALY_WITHIN_DIAGNOSTIC",
        core_anomaly_within,
    ),
    (
        "DEPLOY_CORE_CHL_ANOMALY_WITHIN_DIAGNOSTIC",
        core_chl_anomaly_within,
    ),
]


results = []
fold_tables = []


for label, features in experiments:

    result, folds = evaluate(
        features,
        label
    )

    results.append(result)
    fold_tables.append(folds)

    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    print(
        "Pooled ROC-AUC:",
        round(
            result["pooled_roc_auc"],
            4
        )
    )

    print(
        "Pooled PR-AUC:",
        round(
            result["pooled_pr_auc"],
            4
        )
    )

    print(
        "Baseline prevalence:",
        round(
            result["baseline_prevalence"],
            4
        )
    )

    print(
        "Median mixed-survey ROC-AUC:",
        round(
            result[
                "median_mixed_survey_roc_auc"
            ],
            4
        )
    )

    print(
        "Median mixed-survey PR-AUC:",
        round(
            result[
                "median_mixed_survey_pr_auc"
            ],
            4
        )
    )


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 90)
print("FINAL DEPLOYMENT MODEL COMPARISON")
print("=" * 90)

print(
    results_df[
        [
            "model",
            "pooled_roc_auc",
            "pooled_pr_auc",
            "baseline_prevalence",
            "median_mixed_survey_roc_auc",
            "median_mixed_survey_pr_auc",
        ]
    ]
    .sort_values(
        "pooled_roc_auc",
        ascending=False
    )
    .round(4)
    .to_string(index=False)
)


results_df.to_csv(
    OUTPUT,
    index=False
)

pd.concat(
    fold_tables,
    ignore_index=True
).to_csv(
    FOLD_OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)
print(FOLD_OUTPUT)
