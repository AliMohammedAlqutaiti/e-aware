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
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

OUTPUT = (
    ROOT
    / "phase2_coast_season_validation.csv"
)

FOLD_OUTPUT = (
    ROOT
    / "phase2_coast_season_validation_folds.csv"
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
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_bottom_depth_m",
    "sst_anomaly_c",
    "distance_to_coast_km",
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


# ----------------------------------------------------------
# Transform skewed distance variables.
# ----------------------------------------------------------

data["deploy_log_bottom_depth"] = np.log1p(
    data["deploy_bottom_depth_m"]
)

data["log_distance_to_coast"] = np.log1p(
    data["distance_to_coast_km"]
)


print("=" * 90)
print(
    "E-AWARE PHASE 2 — "
    "COAST + SEASON VALIDATION"
)
print("=" * 90)

print("\nRecords:", len(data))

print(
    "Surveys:",
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

print(
    "Prevalence:",
    round(
        data["presence"].mean(),
        4
    )
)


# ----------------------------------------------------------
# Existing best deployment benchmark.
# ----------------------------------------------------------

benchmark = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
    "sst_anomaly_c",
]


benchmark_coast = benchmark + [
    "log_distance_to_coast",
]


benchmark_season = benchmark + [
    "month_sin",
    "month_cos",
]


benchmark_coast_season = benchmark + [
    "log_distance_to_coast",
    "month_sin",
    "month_cos",
]


# Test whether distance-to-coast could replace bathymetry.
coast_without_depth = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "sst_anomaly_c",
    "log_distance_to_coast",
]


# Test both geography variables without season.
depth_and_coast = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
    "sst_anomaly_c",
    "log_distance_to_coast",
]


def evaluate(
    features,
    label,
):

    X = data[
        features
    ].to_numpy()

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

    prediction = np.full(
        len(data),
        np.nan
    )

    fold_rows = []


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

        prediction[
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


        if len(
            np.unique(
                y_test
            )
        ) == 2:

            fold_roc = (
                roc_auc_score(
                    y_test,
                    probability,
                )
            )

            fold_pr = (
                average_precision_score(
                    y_test,
                    probability,
                )
            )

        else:

            fold_roc = np.nan
            fold_pr = np.nan


        fold_rows.append(
            {
                "model": label,
                "fold": fold,
                "source_group": group_name,
                "records": len(
                    test_idx
                ),
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


    folds = pd.DataFrame(
        fold_rows
    )


    result = {
        "model": label,

        "features": ", ".join(
            features
        ),

        "pooled_roc_auc": (
            roc_auc_score(
                y,
                prediction,
            )
        ),

        "pooled_pr_auc": (
            average_precision_score(
                y,
                prediction,
            )
        ),

        "baseline_prevalence": (
            y.mean()
        ),

        "median_mixed_survey_roc_auc": (
            folds[
                "roc_auc"
            ]
            .dropna()
            .median()
        ),

        "median_mixed_survey_pr_auc": (
            folds[
                "pr_auc"
            ]
            .dropna()
            .median()
        ),
    }


    return (
        result,
        folds,
    )


experiments = [
    (
        "BENCHMARK",
        benchmark,
    ),

    (
        "BENCHMARK_PLUS_COAST",
        benchmark_coast,
    ),

    (
        "BENCHMARK_PLUS_SEASON",
        benchmark_season,
    ),

    (
        "BENCHMARK_PLUS_COAST_SEASON",
        benchmark_coast_season,
    ),

    (
        "COAST_REPLACES_DEPTH",
        coast_without_depth,
    ),

    (
        "DEPTH_AND_COAST",
        depth_and_coast,
    ),
]


results = []
fold_tables = []


for label, features in experiments:

    result, folds = evaluate(
        features,
        label,
    )

    results.append(
        result
    )

    fold_tables.append(
        folds
    )


    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)

    print(
        "Pooled ROC-AUC:",
        round(
            result[
                "pooled_roc_auc"
            ],
            4
        )
    )

    print(
        "Pooled PR-AUC:",
        round(
            result[
                "pooled_pr_auc"
            ],
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
print(
    "FINAL COAST + SEASON MODEL COMPARISON"
)
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
        ascending=False,
    )
    .round(4)
    .to_string(
        index=False
    )
)


results_df.to_csv(
    OUTPUT,
    index=False
)


pd.concat(
    fold_tables,
    ignore_index=True,
).to_csv(
    FOLD_OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)
print(FOLD_OUTPUT)
