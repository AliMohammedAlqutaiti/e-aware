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
    / "E_AWARE_IMARPE_phase2_deployment_relative.csv"
)

OUTPUT = (
    ROOT
    / "phase2_spatial_model_validation.csv"
)

FOLD_OUTPUT = (
    ROOT
    / "phase2_spatial_model_validation_folds.csv"
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
    "sst_spatial_percentile",
    "chlorophyll_spatial_percentile",
]


complete = (
    df[required]
    .notna()
    .all(axis=1)
)


data = df[
    eligible & complete
].copy()


data["deploy_log_bottom_depth"] = np.log1p(
    data["deploy_bottom_depth_m"]
)


print("=" * 90)
print(
    "E-AWARE PHASE 2 — "
    "DEPLOYABLE SPATIAL FEATURE VALIDATION"
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


core = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
]


benchmark = core + [
    "sst_anomaly_c",
]


sst_spatial = benchmark + [
    "sst_spatial_percentile",
]


chl_spatial = benchmark + [
    "chlorophyll_spatial_percentile",
]


sst_chl_spatial = benchmark + [
    "sst_spatial_percentile",
    "chlorophyll_spatial_percentile",
]


spatial_without_climatology = core + [
    "sst_spatial_percentile",
    "chlorophyll_spatial_percentile",
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

    predictions = np.full(
        len(data),
        np.nan
    )

    folds = []


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

        predictions[
            test_idx
        ] = probability

        y_test = y[
            test_idx
        ]

        group_name = groups[
            test_idx
        ][0]


        if len(
            np.unique(y_test)
        ) == 2:

            fold_roc = roc_auc_score(
                y_test,
                probability
            )

            fold_pr = (
                average_precision_score(
                    y_test,
                    probability
                )
            )

        else:

            fold_roc = np.nan
            fold_pr = np.nan


        folds.append(
            {
                "model": label,
                "fold": fold,
                "source_group": group_name,
                "records": len(test_idx),
                "detections": int(
                    y_test.sum()
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
        predictions
    )

    pooled_pr = average_precision_score(
        y,
        predictions
    )

    fold_df = pd.DataFrame(
        folds
    )


    result = {
        "model": label,
        "features": ", ".join(features),

        "pooled_roc_auc": pooled_roc,

        "pooled_pr_auc": pooled_pr,

        "baseline_prevalence": (
            y.mean()
        ),

        "median_mixed_survey_roc_auc": (
            fold_df[
                "roc_auc"
            ]
            .dropna()
            .median()
        ),

        "median_mixed_survey_pr_auc": (
            fold_df[
                "pr_auc"
            ]
            .dropna()
            .median()
        ),
    }


    return (
        result,
        fold_df,
        predictions,
    )


experiments = [
    (
        "BENCHMARK_CORE_PLUS_CLIM_ANOMALY",
        benchmark,
    ),

    (
        "PLUS_SPATIAL_SST",
        sst_spatial,
    ),

    (
        "PLUS_SPATIAL_CHL",
        chl_spatial,
    ),

    (
        "PLUS_SPATIAL_SST_AND_CHL",
        sst_chl_spatial,
    ),

    (
        "SPATIAL_SST_CHL_NO_CLIM_ANOMALY",
        spatial_without_climatology,
    ),
]


results = []
fold_tables = []


for label, features in experiments:

    (
        result,
        folds,
        predictions,
    ) = evaluate(
        features,
        label
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
print("FINAL SPATIAL MODEL COMPARISON")
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
    ignore_index=True
).to_csv(
    FOLD_OUTPUT,
    index=False
)


print("\nSaved:")
print(OUTPUT)
print(FOLD_OUTPUT)


if __name__ == "__main__":
    pass
