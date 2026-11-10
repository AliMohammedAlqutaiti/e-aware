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
    / "phase2_year_holdout_validation.csv"
)

FOLD_OUTPUT = (
    ROOT
    / "phase2_year_holdout_folds.csv"
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


data["year"] = (
    data["date"]
    .dt.year
)


data["deploy_log_bottom_depth"] = np.log1p(
    data["deploy_bottom_depth_m"]
)


print("=" * 90)
print("E-AWARE PHASE 2 — LEAVE-ONE-YEAR-OUT VALIDATION")
print("=" * 90)

print("\nRecords:", len(data))
print(
    "Years:",
    data["year"].nunique()
)
print(
    "Year range:",
    data["year"].min(),
    "-",
    data["year"].max()
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


benchmark = [
    "deploy_sst_c",
    "deploy_sss_psu",
    "deploy_oxygen_mg_l",
    "deploy_log_bottom_depth",
    "sst_anomaly_c",
]


season_only = [
    "month_sin",
    "month_cos",
]


benchmark_season = benchmark + [
    "month_sin",
    "month_cos",
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
        data["year"]
        .to_numpy()
    )

    logo = LeaveOneGroupOut()

    prediction = np.full(
        len(data),
        np.nan,
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

        held_out_year = int(
            groups[
                test_idx
            ][0]
        )


        if len(
            np.unique(
                y_test
            )
        ) == 2:

            fold_roc = roc_auc_score(
                y_test,
                probability,
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
                "held_out_year": held_out_year,
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

        "median_year_roc_auc": (
            folds[
                "roc_auc"
            ]
            .dropna()
            .median()
        ),

        "median_year_pr_auc": (
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
        "YEAR_HOLDOUT_BENCHMARK",
        benchmark,
    ),

    (
        "YEAR_HOLDOUT_SEASON_ONLY",
        season_only,
    ),

    (
        "YEAR_HOLDOUT_BENCHMARK_PLUS_SEASON",
        benchmark_season,
    ),
]


results = []
all_folds = []


for label, features in experiments:

    result, folds = evaluate(
        features,
        label,
    )

    results.append(
        result
    )

    all_folds.append(
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
        "Median valid-year ROC-AUC:",
        round(
            result[
                "median_year_roc_auc"
            ],
            4
        )
    )

    print(
        "Median valid-year PR-AUC:",
        round(
            result[
                "median_year_pr_auc"
            ],
            4
        )
    )


results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 90)
print("YEAR-HOLDOUT MODEL COMPARISON")
print("=" * 90)


print(
    results_df[
        [
            "model",
            "pooled_roc_auc",
            "pooled_pr_auc",
            "baseline_prevalence",
            "median_year_roc_auc",
            "median_year_pr_auc",
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


print("\n" + "=" * 90)
print("YEAR STRUCTURE")
print("=" * 90)


year_summary = (
    data.groupby(
        "year"
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


year_summary[
    "detection_rate"
] = (
    year_summary[
        "detections"
    ]
    / year_summary[
        "records"
    ]
)


print(
    year_summary
    .round(4)
    .to_string()
)


results_df.to_csv(
    OUTPUT,
    index=False,
)


pd.concat(
    all_folds,
    ignore_index=True,
).to_csv(
    FOLD_OUTPUT,
    index=False,
)


print("\nSaved:")
print(OUTPUT)
print(FOLD_OUTPUT)
