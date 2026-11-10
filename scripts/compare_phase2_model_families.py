from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

OUTPUT = (
    ROOT
    / "phase2_model_family_comparison.csv"
)

FOLD_OUTPUT = (
    ROOT
    / "phase2_model_family_folds.csv"
)


# ==========================================================
# LOAD DATA
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


data["year"] = (
    data["date"]
    .dt.year
)


data["deploy_log_bottom_depth"] = np.log1p(
    data["deploy_bottom_depth_m"]
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


print("=" * 90)
print("E-AWARE PHASE 2 — MODEL FAMILY COMPETITION")
print("=" * 90)

print("\nRecords:", len(data))

print(
    "Surveys:",
    data["source_group"].nunique()
)

print(
    "Years:",
    data["year"].nunique()
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

print("\nSelected predictors:")

for feature in FEATURES:
    print(" -", feature)


# ==========================================================
# MODEL FACTORIES
#
# Fixed, conservative hyperparameters.
# We are selecting the MODEL FAMILY here,
# not doing final hyperparameter tuning yet.
# ==========================================================

def build_model(name):

    if name == "LOGISTIC":

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                C=1.0,
                random_state=42,
            ),
        )


    if name == "RANDOM_FOREST":

        return RandomForestClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )


    if name == "HIST_GRADIENT_BOOSTING":

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )


    raise ValueError(
        f"Unknown model: {name}"
    )


MODEL_NAMES = [
    "LOGISTIC",
    "RANDOM_FOREST",
    "HIST_GRADIENT_BOOSTING",
]


# ==========================================================
# GROUPED EVALUATION
# ==========================================================

def evaluate(
    model_name,
    group_column,
    validation_name,
):

    X = (
        data[FEATURES]
        .to_numpy()
    )

    y = (
        data["presence"]
        .astype(int)
        .to_numpy()
    )

    groups = (
        data[group_column]
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


        model = build_model(
            model_name
        )


        # ----------------------------------------------
        # Gradient boosting does not rely on the same
        # class_weight mechanism, so give it balanced
        # sample weights explicitly.
        # ----------------------------------------------

        if (
            model_name
            == "HIST_GRADIENT_BOOSTING"
        ):

            sample_weight = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y[train_idx],
                )
            )

            model.fit(
                X[train_idx],
                y[train_idx],
                sample_weight=sample_weight,
            )

        else:

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


        held_out_group = (
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
                "model": model_name,
                "validation": validation_name,
                "fold": fold,
                "held_out_group": held_out_group,
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


    fold_df = pd.DataFrame(
        fold_rows
    )


    result = {
        "model": model_name,
        "validation": validation_name,

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

        "median_valid_fold_roc_auc": (
            fold_df[
                "roc_auc"
            ]
            .dropna()
            .median()
        ),

        "median_valid_fold_pr_auc": (
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
    )


# ==========================================================
# RUN BOTH VALIDATION SCHEMES
# ==========================================================

results = []
fold_tables = []


validation_schemes = [
    (
        "source_group",
        "LEAVE_ONE_SURVEY_OUT",
    ),
    (
        "year",
        "LEAVE_ONE_YEAR_OUT",
    ),
]


for model_name in MODEL_NAMES:

    for (
        group_column,
        validation_name,
    ) in validation_schemes:


        result, folds = evaluate(
            model_name,
            group_column,
            validation_name,
        )


        results.append(
            result
        )

        fold_tables.append(
            folds
        )


        print("\n" + "=" * 90)

        print(
            model_name,
            "—",
            validation_name,
        )

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
            "Median fold ROC-AUC:",
            round(
                result[
                    "median_valid_fold_roc_auc"
                ],
                4
            )
        )


        print(
            "Median fold PR-AUC:",
            round(
                result[
                    "median_valid_fold_pr_auc"
                ],
                4
            )
        )


# ==========================================================
# SUMMARY TABLE
# ==========================================================

results_df = pd.DataFrame(
    results
)


pivot_roc = (
    results_df
    .pivot(
        index="model",
        columns="validation",
        values="pooled_roc_auc",
    )
)


pivot_pr = (
    results_df
    .pivot(
        index="model",
        columns="validation",
        values="pooled_pr_auc",
    )
)


summary = pd.DataFrame(
    index=MODEL_NAMES
)


summary[
    "survey_roc_auc"
] = (
    pivot_roc[
        "LEAVE_ONE_SURVEY_OUT"
    ]
)


summary[
    "survey_pr_auc"
] = (
    pivot_pr[
        "LEAVE_ONE_SURVEY_OUT"
    ]
)


summary[
    "year_roc_auc"
] = (
    pivot_roc[
        "LEAVE_ONE_YEAR_OUT"
    ]
)


summary[
    "year_pr_auc"
] = (
    pivot_pr[
        "LEAVE_ONE_YEAR_OUT"
    ]
)


# Simple robustness summary.
# This is NOT a new performance metric;
# it just helps compare consistency between
# the two validation schemes.

summary[
    "mean_roc_auc"
] = (
    summary[
        [
            "survey_roc_auc",
            "year_roc_auc",
        ]
    ]
    .mean(axis=1)
)


summary[
    "mean_pr_auc"
] = (
    summary[
        [
            "survey_pr_auc",
            "year_pr_auc",
        ]
    ]
    .mean(axis=1)
)


print("\n" + "=" * 90)
print("FINAL MODEL FAMILY COMPARISON")
print("=" * 90)


print(
    summary
    .sort_values(
        [
            "mean_roc_auc",
            "mean_pr_auc",
        ],
        ascending=False,
    )
    .round(4)
    .to_string()
)


# ==========================================================
# SAVE
# ==========================================================

results_df.to_csv(
    OUTPUT,
    index=False,
)


pd.concat(
    fold_tables,
    ignore_index=True,
).to_csv(
    FOLD_OUTPUT,
    index=False,
)


summary.to_csv(
    ROOT
    / "phase2_model_family_summary.csv"
)


print("\nSaved:")
print(OUTPUT)
print(FOLD_OUTPUT)
print(
    ROOT
    / "phase2_model_family_summary.csv"
)
