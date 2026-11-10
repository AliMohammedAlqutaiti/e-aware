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
    / "phase2_logistic_tuning_screen.csv"
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


data["deploy_log_bottom_depth"] = (
    np.log1p(
        data["deploy_bottom_depth_m"]
    )
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
    data[FEATURES]
    .to_numpy()
)

y = (
    data["presence"]
    .astype(int)
    .to_numpy()
)


def evaluate(
    groups,
    C,
    class_weight,
):

    logo = LeaveOneGroupOut()

    prediction = np.full(
        len(data),
        np.nan,
    )


    for (
        train_idx,
        test_idx,
    ) in logo.split(
        X,
        y,
        groups,
    ):

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=5000,
                random_state=42,
            ),
        )

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        prediction[
            test_idx
        ] = (
            model.predict_proba(
                X[test_idx]
            )[:, 1]
        )


    return {
        "roc_auc": roc_auc_score(
            y,
            prediction,
        ),

        "pr_auc": (
            average_precision_score(
                y,
                prediction,
            )
        ),
    }


C_VALUES = [
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]


WEIGHTS = [
    None,
    "balanced",
]


rows = []


print("=" * 90)
print("E-AWARE — LOGISTIC REGRESSION TUNING SCREEN")
print("=" * 90)


for C in C_VALUES:

    for weight in WEIGHTS:

        survey = evaluate(
            data[
                "source_group"
            ].to_numpy(),
            C,
            weight,
        )

        year = evaluate(
            data[
                "year"
            ].to_numpy(),
            C,
            weight,
        )


        row = {
            "C": C,

            "class_weight": (
                "none"
                if weight is None
                else weight
            ),

            "survey_roc_auc": (
                survey[
                    "roc_auc"
                ]
            ),

            "survey_pr_auc": (
                survey[
                    "pr_auc"
                ]
            ),

            "year_roc_auc": (
                year[
                    "roc_auc"
                ]
            ),

            "year_pr_auc": (
                year[
                    "pr_auc"
                ]
            ),
        }


        row["mean_roc_auc"] = np.mean(
            [
                row[
                    "survey_roc_auc"
                ],
                row[
                    "year_roc_auc"
                ],
            ]
        )


        row["mean_pr_auc"] = np.mean(
            [
                row[
                    "survey_pr_auc"
                ],
                row[
                    "year_pr_auc"
                ],
            ]
        )


        row["worst_roc_auc"] = min(
            row[
                "survey_roc_auc"
            ],
            row[
                "year_roc_auc"
            ],
        )


        row["worst_pr_auc"] = min(
            row[
                "survey_pr_auc"
            ],
            row[
                "year_pr_auc"
            ],
        )


        rows.append(
            row
        )


        print(
            f"C={C:<5} "
            f"weight={str(weight):<8} "
            f"survey ROC={survey['roc_auc']:.4f} "
            f"PR={survey['pr_auc']:.4f} | "
            f"year ROC={year['roc_auc']:.4f} "
            f"PR={year['pr_auc']:.4f}"
        )


results = pd.DataFrame(
    rows
)


results = results.sort_values(
    [
        "worst_roc_auc",
        "worst_pr_auc",
        "mean_roc_auc",
    ],
    ascending=False,
)


print("\n" + "=" * 90)
print("LOGISTIC TUNING COMPARISON")
print("=" * 90)


print(
    results
    .round(4)
    .to_string(
        index=False
    )
)


results.to_csv(
    OUTPUT,
    index=False,
)


print("\nBest robustness-screen configuration:")

print(
    results
    .head(1)
    .to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT)
