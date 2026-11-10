from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "phase2_final_nested_validation_summary.csv"
)

OOF_OUTPUT = (
    ROOT
    / "phase2_final_nested_oof_predictions.csv"
)

COEF_OUTPUT = (
    ROOT
    / "phase2_final_nested_coefficients.csv"
)


# ==========================================================
# DATA
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


C_VALUES = [
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]


X = data[
    FEATURES
].to_numpy()

y = (
    data["presence"]
    .astype(int)
    .to_numpy()
)


print("=" * 90)
print("E-AWARE PHASE 2 — STRICT NESTED GROUPED VALIDATION")
print("=" * 90)

print("\nRecords:", len(data))
print(
    "Detections:",
    int(y.sum())
)
print(
    "Non-detections:",
    int((y == 0).sum())
)
print(
    "Prevalence:",
    round(y.mean(), 4)
)


# ==========================================================
# MODEL
# ==========================================================

def build_model(C):

    return make_pipeline(
        StandardScaler(),

        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )


# ==========================================================
# INNER CV — CHOOSE C USING TRAINING DATA ONLY
# ==========================================================

def choose_c(
    X_train,
    y_train,
    groups_train,
):

    logo = LeaveOneGroupOut()

    rows = []


    for C in C_VALUES:

        prediction = np.full(
            len(y_train),
            np.nan,
        )


        for (
            inner_train,
            inner_test,
        ) in logo.split(
            X_train,
            y_train,
            groups_train,
        ):

            model = build_model(C)

            model.fit(
                X_train[
                    inner_train
                ],
                y_train[
                    inner_train
                ],
            )

            prediction[
                inner_test
            ] = (
                model.predict_proba(
                    X_train[
                        inner_test
                    ]
                )[:, 1]
            )


        roc = roc_auc_score(
            y_train,
            prediction,
        )

        pr = average_precision_score(
            y_train,
            prediction,
        )


        # Equal-weight screening criterion.
        # Used only for hyperparameter selection.
        selection_score = (
            roc + pr
        ) / 2.0


        rows.append(
            {
                "C": C,
                "roc_auc": roc,
                "pr_auc": pr,
                "selection_score": selection_score,
            }
        )


    results = pd.DataFrame(
        rows
    )


    # If scores are extremely close, prefer
    # stronger regularisation (smaller C).
    results = results.sort_values(
        [
            "selection_score",
            "C",
        ],
        ascending=[
            False,
            True,
        ],
    )


    return (
        float(
            results.iloc[0][
                "C"
            ]
        ),
        results,
    )


# ==========================================================
# OUTER VALIDATION
# ==========================================================

def nested_validation(
    group_column,
    validation_name,
):

    groups = (
        data[
            group_column
        ]
        .to_numpy()
    )

    outer = LeaveOneGroupOut()


    oof_probability = np.full(
        len(data),
        np.nan,
    )


    fold_rows = []

    coefficient_rows = []


    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):


        X_train = X[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        groups_train = groups[
            train_idx
        ]


        # ----------------------------------------------
        # Tune C INSIDE the outer training set.
        # ----------------------------------------------

        best_c, inner_results = choose_c(
            X_train,
            y_train,
            groups_train,
        )


        # ----------------------------------------------
        # Fit selected model on complete outer training.
        # ----------------------------------------------

        model = build_model(
            best_c
        )


        model.fit(
            X_train,
            y_train,
        )


        probability = (
            model.predict_proba(
                X[
                    test_idx
                ]
            )[:, 1]
        )


        oof_probability[
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

                "selected_C": best_c,

                "roc_auc": fold_roc,

                "pr_auc": fold_pr,

                "mean_prediction": float(
                    probability.mean()
                ),
            }
        )


        # ----------------------------------------------
        # Standardised coefficients.
        # Because StandardScaler is inside the pipeline,
        # these are comparable across outer folds.
        # ----------------------------------------------

        logistic = (
            model.steps[-1][1]
        )

        coefficients = (
            logistic.coef_[0]
        )


        for feature, coefficient in zip(
            FEATURES,
            coefficients,
        ):

            coefficient_rows.append(
                {
                    "validation": validation_name,
                    "fold": fold,
                    "held_out_group": held_out_group,
                    "selected_C": best_c,
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                }
            )


        print(
            f"{validation_name} "
            f"fold {fold:02d}: "
            f"held out={held_out_group} "
            f"C={best_c}"
        )


    # ======================================================
    # POOLED OUT-OF-FOLD PERFORMANCE
    # ======================================================

    pooled_roc = roc_auc_score(
        y,
        oof_probability,
    )

    pooled_pr = average_precision_score(
        y,
        oof_probability,
    )

    brier = brier_score_loss(
        y,
        oof_probability,
    )


    fold_df = pd.DataFrame(
        fold_rows
    )


    summary = {
        "validation": validation_name,

        "records": len(y),

        "groups": len(
            np.unique(
                groups
            )
        ),

        "pooled_roc_auc": pooled_roc,

        "pooled_pr_auc": pooled_pr,

        "baseline_prevalence": (
            y.mean()
        ),

        "brier_score_raw": brier,

        "mean_oof_probability": (
            oof_probability.mean()
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

        "most_selected_C": (
            fold_df[
                "selected_C"
            ]
            .mode()
            .iloc[0]
        ),
    }


    prediction_df = pd.DataFrame(
        {
            "record_id": (
                data["record_id"]
                .to_numpy()
            ),

            "source_group": (
                data["source_group"]
                .to_numpy()
            ),

            "date": (
                data["date"]
                .to_numpy()
            ),

            "year": (
                data["year"]
                .to_numpy()
            ),

            "lat": (
                data["lat"]
                .to_numpy()
            ),

            "lon": (
                data["lon"]
                .to_numpy()
            ),

            "presence": y,

            "validation": validation_name,

            "oof_probability_raw": (
                oof_probability
            ),
        }
    )


    return (
        summary,
        fold_df,
        prediction_df,
        pd.DataFrame(
            coefficient_rows
        ),
    )


# ==========================================================
# RUN STRICT SURVEY AND YEAR VALIDATION
# ==========================================================

all_summaries = []
all_folds = []
all_predictions = []
all_coefficients = []


for (
    group_column,
    validation_name,
) in [
    (
        "source_group",
        "NESTED_LEAVE_ONE_SURVEY_OUT",
    ),

    (
        "year",
        "NESTED_LEAVE_ONE_YEAR_OUT",
    ),
]:

    print("\n" + "=" * 90)
    print(validation_name)
    print("=" * 90)


    (
        summary,
        folds,
        predictions,
        coefficients,
    ) = nested_validation(
        group_column,
        validation_name,
    )


    all_summaries.append(
        summary
    )

    all_folds.append(
        folds
    )

    all_predictions.append(
        predictions
    )

    all_coefficients.append(
        coefficients
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

summary_df = pd.DataFrame(
    all_summaries
)


print("\n" + "=" * 90)
print("STRICT NESTED VALIDATION RESULTS")
print("=" * 90)


print(
    summary_df
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# COEFFICIENT STABILITY
# ==========================================================

coef_df = pd.concat(
    all_coefficients,
    ignore_index=True,
)


coef_summary = (
    coef_df
    .groupby(
        [
            "validation",
            "feature",
        ]
    )[
        "standardized_coefficient"
    ]
    .agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    )
    .reset_index()
)


print("\n" + "=" * 90)
print("STANDARDISED COEFFICIENT STABILITY")
print("=" * 90)


print(
    coef_summary
    .round(4)
    .to_string(
        index=False
    )
)


# ==========================================================
# SELECTED C COUNTS
# ==========================================================

fold_df = pd.concat(
    all_folds,
    ignore_index=True,
)


print("\n" + "=" * 90)
print("SELECTED C COUNTS")
print("=" * 90)


print(
    fold_df.groupby(
        [
            "validation",
            "selected_C",
        ]
    )
    .size()
    .rename(
        "folds"
    )
    .to_string()
)


# ==========================================================
# SAVE
# ==========================================================

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


pd.concat(
    all_predictions,
    ignore_index=True,
).to_csv(
    OOF_OUTPUT,
    index=False,
)


coef_df.to_csv(
    COEF_OUTPUT,
    index=False,
)


fold_df.to_csv(
    ROOT
    / "phase2_final_nested_fold_results.csv",
    index=False,
)


coef_summary.to_csv(
    ROOT
    / "phase2_final_coefficient_stability.csv",
    index=False,
)


print("\nSaved:")
print(SUMMARY_OUTPUT)
print(OOF_OUTPUT)
print(COEF_OUTPUT)
