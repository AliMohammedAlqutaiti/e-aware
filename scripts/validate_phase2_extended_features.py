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
    / "E_AWARE_IMARPE_phase2_enriched.csv"
)

OUTPUT = (
    ROOT
    / "phase2_extended_feature_validation.csv"
)


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
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "bottom_depth_m",
    "sst_anomaly_c",
    "chlorophyll_mg_m3",
]


complete = (
    df[required]
    .notna()
    .all(axis=1)
)


data = df[
    eligible & complete
].copy()


# ----------------------------------------------------------
# Keep only mixed surveys for this diagnostic.
# ----------------------------------------------------------

group_summary = (
    data.groupby("source_group")
    .agg(
        records=("presence", "size"),
        detections=("presence", "sum"),
    )
)


group_summary["non_detections"] = (
    group_summary["records"]
    - group_summary["detections"]
)


mixed_groups = (
    group_summary[
        (group_summary["detections"] > 0)
        & (
            group_summary["non_detections"]
            > 0
        )
    ]
    .index
)


data = data[
    data["source_group"]
    .isin(mixed_groups)
].copy()


print("=" * 90)
print("PHASE 2 — EXTENDED FEATURE VALIDATION")
print("=" * 90)

print("\nRecords:", len(data))
print(
    "Groups:",
    data["source_group"].nunique()
)
print(
    "Detections:",
    int(data["presence"].sum())
)
print(
    "Non-detections:",
    int(
        (data["presence"] == 0).sum()
    )
)


# ----------------------------------------------------------
# Transform strongly skewed variables.
# ----------------------------------------------------------

data["log_bottom_depth"] = np.log1p(
    data["bottom_depth_m"]
)

data["log_chlorophyll"] = np.log1p(
    data["chlorophyll_mg_m3"]
)


core_features = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "log_bottom_depth",
]


extended_features = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "log_bottom_depth",
    "sst_anomaly_c",
    "log_chlorophyll",
]


# Separate experiment:
# chlorophyll added, but NOT anomaly.
chlorophyll_features = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "log_bottom_depth",
    "log_chlorophyll",
]


# Separate experiment:
# anomaly added, but NOT chlorophyll.
anomaly_features = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "log_bottom_depth",
    "sst_anomaly_c",
]


def make_within_features(
    dataframe,
    features,
):

    out = []

    for col in features:

        new = (
            f"{col}_within"
        )

        dataframe[new] = (
            dataframe[col]
            - dataframe.groupby(
                "source_group"
            )[col].transform("mean")
        )

        out.append(new)

    return out


core_within = make_within_features(
    data,
    core_features,
)

chlorophyll_within = make_within_features(
    data,
    chlorophyll_features,
)

anomaly_within = make_within_features(
    data,
    anomaly_features,
)

extended_within = make_within_features(
    data,
    extended_features,
)


def evaluate(
    dataframe,
    features,
    label,
):

    X = dataframe[
        features
    ].to_numpy()

    y = (
        dataframe["presence"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        dataframe["source_group"]
        .to_numpy()
    )

    logo = LeaveOneGroupOut()

    predictions = np.full(
        len(dataframe),
        np.nan,
    )

    folds = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups,
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
            y[train_idx],
        )

        probability = (
            model.predict_proba(
                X[test_idx]
            )[:, 1]
        )

        predictions[
            test_idx
        ] = probability

        y_test = y[
            test_idx
        ]

        group_name = (
            groups[
                test_idx
            ][0]
        )

        folds.append(
            {
                "model": label,
                "fold": fold,
                "source_group": group_name,
                "records": len(test_idx),
                "detections": int(
                    y_test.sum()
                ),
                "roc_auc": roc_auc_score(
                    y_test,
                    probability,
                ),
                "pr_auc": average_precision_score(
                    y_test,
                    probability,
                ),
            }
        )

    pooled_roc = roc_auc_score(
        y,
        predictions,
    )

    pooled_pr = average_precision_score(
        y,
        predictions,
    )

    fold_df = pd.DataFrame(
        folds
    )

    result = {
        "model": label,
        "features": ", ".join(features),
        "pooled_roc_auc": pooled_roc,
        "pooled_pr_auc": pooled_pr,
        "baseline_prevalence": y.mean(),
        "median_fold_roc_auc": (
            fold_df["roc_auc"].median()
        ),
        "median_fold_pr_auc": (
            fold_df["pr_auc"].median()
        ),
    }

    return (
        result,
        fold_df,
        predictions,
    )


experiments = [
    (
        "CORE_RAW",
        core_features,
    ),
    (
        "CORE_WITHIN",
        core_within,
    ),

    (
        "CORE_PLUS_CHL_RAW",
        chlorophyll_features,
    ),
    (
        "CORE_PLUS_CHL_WITHIN",
        chlorophyll_within,
    ),

    (
        "CORE_PLUS_ANOMALY_RAW",
        anomaly_features,
    ),
    (
        "CORE_PLUS_ANOMALY_WITHIN",
        anomaly_within,
    ),

    (
        "CORE_PLUS_CHL_ANOMALY_RAW",
        extended_features,
    ),
    (
        "CORE_PLUS_CHL_ANOMALY_WITHIN",
        extended_within,
    ),
]


results = []
all_folds = []


for label, features in experiments:

    result, folds, predictions = (
        evaluate(
            data,
            features,
            label,
        )
    )

    results.append(result)
    all_folds.append(folds)

    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    print(
        "Pooled ROC-AUC:",
        round(
            result[
                "pooled_roc_auc"
            ],
            3,
        )
    )

    print(
        "Pooled PR-AUC:",
        round(
            result[
                "pooled_pr_auc"
            ],
            3,
        )
    )

    print(
        "Median survey ROC-AUC:",
        round(
            result[
                "median_fold_roc_auc"
            ],
            3,
        )
    )

    print(
        "Median survey PR-AUC:",
        round(
            result[
                "median_fold_pr_auc"
            ],
            3,
        )
    )


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(
    results_df[
        [
            "model",
            "pooled_roc_auc",
            "pooled_pr_auc",
            "baseline_prevalence",
            "median_fold_roc_auc",
            "median_fold_pr_auc",
        ]
    ]
    .sort_values(
        "pooled_roc_auc",
        ascending=False,
    )
    .round(4)
    .to_string(
        index=False
    )
)


results_df.to_csv(
    OUTPUT,
    index=False,
)


fold_output = (
    ROOT
    / "phase2_extended_feature_fold_results.csv"
)

pd.concat(
    all_folds,
    ignore_index=True,
).to_csv(
    fold_output,
    index=False,
)


print("\nSaved:")
print(OUTPUT)
print(fold_output)
