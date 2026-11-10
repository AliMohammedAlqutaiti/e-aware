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
    / "E_AWARE_IMARPE_phase2_copernicus_filled.csv"
)

OUTPUT = (
    ROOT
    / "phase2_core_within_survey_oof_predictions.csv"
)


df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
)

eligible = (
    df["date"].notna()
    & (df["date"] >= pd.Timestamp("1993-01-01"))
)

core = (
    df["sst_c"].notna()
    & df["sss_psu"].notna()
    & df["oxygen_mg_l"].notna()
    & df["bottom_depth_m"].notna()
)

data = df[
    eligible & core
].copy()


# ----------------------------------------------------------
# Group audit
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

group_summary["detection_rate"] = (
    group_summary["detections"]
    / group_summary["records"]
)

group_summary["group_type"] = np.where(
    group_summary["detections"] == 0,
    "all_absence",
    np.where(
        group_summary["non_detections"] == 0,
        "all_presence",
        "mixed",
    ),
)


print("=" * 90)
print("PHASE 2 — SURVEY CLASS STRUCTURE")
print("=" * 90)

print(
    group_summary
    .sort_values(
        "detection_rate",
        ascending=False,
    )
    .to_string()
)

print("\nGroup-type counts:")
print(
    group_summary[
        "group_type"
    ]
    .value_counts()
    .to_string()
)


# ----------------------------------------------------------
# Use only mixed surveys for the within-survey diagnostic.
# We DO NOT delete the single-class surveys from the master
# dataset. They are excluded only from this diagnostic.
# ----------------------------------------------------------

mixed_groups = (
    group_summary[
        group_summary["group_type"]
        == "mixed"
    ]
    .index
)

mixed = data[
    data["source_group"]
    .isin(mixed_groups)
].copy()


print("\n" + "=" * 90)
print("MIXED-SURVEY DIAGNOSTIC DATA")
print("=" * 90)

print("Records:", len(mixed))
print(
    "Groups:",
    mixed["source_group"].nunique()
)
print(
    "Detections:",
    int(mixed["presence"].sum())
)
print(
    "Non-detections:",
    int(
        (mixed["presence"] == 0).sum()
    )
)


# Depth is strongly right-skewed.
mixed["log_bottom_depth"] = np.log1p(
    mixed["bottom_depth_m"]
)

raw_features = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "log_bottom_depth",
]


# ----------------------------------------------------------
# Within-survey centering.
#
# This is a DIAGNOSTIC transformation, not the deployment
# model. It removes the average environmental differences
# between surveys and asks whether relative conditions
# within the same survey contain predictive signal.
# ----------------------------------------------------------

within_features = []

for col in raw_features:

    new_col = (
        f"{col}_within_survey"
    )

    mixed[new_col] = (
        mixed[col]
        - mixed.groupby(
            "source_group"
        )[col].transform("mean")
    )

    within_features.append(
        new_col
    )


def run_logo(
    dataframe,
    features,
    label,
):

    X = dataframe[
        features
    ].to_numpy()

    y = dataframe[
        "presence"
    ].astype(int).to_numpy()

    groups = dataframe[
        "source_group"
    ].to_numpy()

    logo = LeaveOneGroupOut()

    predictions = np.full(
        len(dataframe),
        np.nan,
    )

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
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

        prob = model.predict_proba(
            X[test_idx]
        )[:, 1]

        predictions[
            test_idx
        ] = prob

        y_test = y[
            test_idx
        ]

        group_name = groups[
            test_idx
        ][0]

        fold_rows.append(
            {
                "fold": fold,
                "source_group": group_name,
                "records": len(test_idx),
                "detections": int(
                    y_test.sum()
                ),
                "roc_auc": (
                    roc_auc_score(
                        y_test,
                        prob,
                    )
                ),
                "pr_auc": (
                    average_precision_score(
                        y_test,
                        prob,
                    )
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

    folds = pd.DataFrame(
        fold_rows
    )

    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    print(
        "Pooled ROC-AUC:",
        round(
            pooled_roc,
            3,
        ),
    )

    print(
        "Pooled PR-AUC:",
        round(
            pooled_pr,
            3,
        ),
    )

    print(
        "Baseline prevalence:",
        round(
            y.mean(),
            3,
        ),
    )

    print(
        "\nMedian held-out survey ROC-AUC:",
        round(
            folds["roc_auc"].median(),
            3,
        ),
    )

    print(
        "Median held-out survey PR-AUC:",
        round(
            folds["pr_auc"].median(),
            3,
        ),
    )

    print("\nPer-survey results:")

    print(
        folds
        .sort_values(
            "roc_auc",
            ascending=False,
        )
        .round(3)
        .to_string(
            index=False
        )
    )

    return predictions, folds


raw_pred, raw_folds = run_logo(
    mixed,
    raw_features,
    "RAW ENVIRONMENTAL FEATURES — LEAVE ONE SURVEY OUT",
)

within_pred, within_folds = run_logo(
    mixed,
    within_features,
    "WITHIN-SURVEY CENTERED FEATURES — LEAVE ONE SURVEY OUT",
)


mixed[
    "raw_core_oof_probability"
] = raw_pred

mixed[
    "within_survey_core_oof_probability"
] = within_pred

mixed[
    [
        "record_id",
        "source_group",
        "date",
        "lat",
        "lon",
        "presence",
        "sst_c",
        "sss_psu",
        "oxygen_mg_l",
        "bottom_depth_m",
        "raw_core_oof_probability",
        "within_survey_core_oof_probability",
    ]
].to_csv(
    OUTPUT,
    index=False,
)


print("\n" + "=" * 90)
print("INTERPRETATION GUIDE")
print("=" * 90)

print(
    """
If RAW performance is good but WITHIN-SURVEY performance
falls strongly, much of the apparent predictive ability
comes from differences between surveys rather than from
local anchoveta habitat relationships.

If WITHIN-SURVEY ROC-AUC remains clearly above 0.5 and
PR-AUC remains clearly above prevalence, the environmental
variables contain useful local signal even after survey
effects are removed.
"""
)

print("Saved:")
print(OUTPUT)
