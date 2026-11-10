from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
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
    / "phase2_nested_calibration_summary.csv"
)

OOF_OUTPUT = (
    ROOT
    / "phase2_nested_calibrated_oof.csv"
)

BIN_OUTPUT = (
    ROOT
    / "phase2_nested_calibration_bins.csv"
)


C_VALUES = [
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
]


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


print("=" * 90)
print(
    "E-AWARE PHASE 2 — "
    "STRICT NESTED PROBABILITY CALIBRATION"
)
print("=" * 90)

print("\nRecords:", len(data))
print("Detections:", int(y.sum()))
print(
    "Prevalence:",
    round(y.mean(), 4)
)


# ==========================================================
# BASE MODEL
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
# INNER OOF PREDICTIONS
# ==========================================================

def grouped_oof_predictions(
    X_train,
    y_train,
    groups_train,
    C,
):

    logo = LeaveOneGroupOut()

    prediction = np.full(
        len(y_train),
        np.nan,
    )


    for (
        train_idx,
        test_idx,
    ) in logo.split(
        X_train,
        y_train,
        groups_train,
    ):

        model = build_model(C)

        model.fit(
            X_train[train_idx],
            y_train[train_idx],
        )

        prediction[test_idx] = (
            model.predict_proba(
                X_train[test_idx]
            )[:, 1]
        )


    return prediction


# ==========================================================
# CHOOSE C INSIDE OUTER TRAINING SET
# ==========================================================

def choose_c(
    X_train,
    y_train,
    groups_train,
):

    rows = []


    for C in C_VALUES:

        pred = grouped_oof_predictions(
            X_train,
            y_train,
            groups_train,
            C,
        )


        roc = roc_auc_score(
            y_train,
            pred,
        )

        pr = average_precision_score(
            y_train,
            pred,
        )


        rows.append(
            {
                "C": C,
                "roc": roc,
                "pr": pr,
                "score": (
                    roc + pr
                ) / 2.0,
            }
        )


    table = pd.DataFrame(
        rows
    )


    table = table.sort_values(
        [
            "score",
            "C",
        ],
        ascending=[
            False,
            True,
        ],
    )


    return float(
        table.iloc[0]["C"]
    )


# ==========================================================
# LOGIT TRANSFORM FOR PLATT SCALING
# ==========================================================

def probability_to_logit(
    probability,
):

    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )


    return np.log(
        p / (1 - p)
    )


# ==========================================================
# CALIBRATION METRICS
# ==========================================================

def expected_calibration_error(
    y_true,
    probability,
    bins=10,
):

    edges = np.linspace(
        0,
        1,
        bins + 1,
    )


    total = len(y_true)

    ece = 0.0


    for i in range(bins):

        if i == bins - 1:

            mask = (
                (probability >= edges[i])
                & (probability <= edges[i + 1])
            )

        else:

            mask = (
                (probability >= edges[i])
                & (probability < edges[i + 1])
            )


        n = int(
            mask.sum()
        )


        if n == 0:
            continue


        mean_pred = float(
            probability[mask].mean()
        )

        observed = float(
            y_true[mask].mean()
        )


        ece += (
            n / total
        ) * abs(
            mean_pred
            - observed
        )


    return ece


def calibration_bins(
    y_true,
    probability,
    validation,
    method,
):

    frame = pd.DataFrame(
        {
            "y": y_true,
            "p": probability,
        }
    )


    # Quantile bins produce more useful diagnostics
    # than many empty fixed-width bins.
    frame["bin"] = pd.qcut(
        frame["p"],
        q=10,
        duplicates="drop",
    )


    result = (
        frame.groupby(
            "bin",
            observed=False,
        )
        .agg(
            records=("y", "size"),
            mean_predicted_probability=("p", "mean"),
            observed_presence_rate=("y", "mean"),
        )
        .reset_index()
    )


    result[
        "validation"
    ] = validation

    result[
        "method"
    ] = method


    return result


# ==========================================================
# STRICT OUTER VALIDATION
# ==========================================================

def run_nested_calibration(
    group_column,
    validation_name,
):

    groups = (
        data[group_column]
        .to_numpy()
    )


    outer = LeaveOneGroupOut()


    raw_oof = np.full(
        len(data),
        np.nan,
    )

    platt_oof = np.full(
        len(data),
        np.nan,
    )

    isotonic_oof = np.full(
        len(data),
        np.nan,
    )


    selected_c = []


    for fold, (
        outer_train,
        outer_test,
    ) in enumerate(
        outer.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):


        X_train = X[
            outer_train
        ]

        y_train = y[
            outer_train
        ]

        groups_train = groups[
            outer_train
        ]


        # ----------------------------------------------
        # 1. Choose C using ONLY outer training data.
        # ----------------------------------------------

        best_c = choose_c(
            X_train,
            y_train,
            groups_train,
        )


        selected_c.append(
            best_c
        )


        # ----------------------------------------------
        # 2. Generate inner OOF probabilities using
        #    the selected C.
        # ----------------------------------------------

        inner_raw = grouped_oof_predictions(
            X_train,
            y_train,
            groups_train,
            best_c,
        )


        # ----------------------------------------------
        # 3. Fit Platt calibration.
        #
        # Fit a normal, unweighted logistic model to
        # the logit of inner OOF predictions.
        # ----------------------------------------------

        platt = LogisticRegression(
            C=1e6,
            class_weight=None,
            max_iter=5000,
            random_state=42,
        )


        platt.fit(
            probability_to_logit(
                inner_raw
            ).reshape(-1, 1),
            y_train,
        )


        # ----------------------------------------------
        # 4. Fit isotonic calibration.
        # ----------------------------------------------

        isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )


        isotonic.fit(
            inner_raw,
            y_train,
        )


        # ----------------------------------------------
        # 5. Fit base model on entire outer training set.
        # ----------------------------------------------

        model = build_model(
            best_c
        )


        model.fit(
            X_train,
            y_train,
        )


        raw_test = (
            model.predict_proba(
                X[outer_test]
            )[:, 1]
        )


        # ----------------------------------------------
        # 6. Calibrate completely unseen outer fold.
        # ----------------------------------------------

        platt_test = (
            platt.predict_proba(
                probability_to_logit(
                    raw_test
                ).reshape(-1, 1)
            )[:, 1]
        )


        isotonic_test = isotonic.predict(
            raw_test
        )


        raw_oof[
            outer_test
        ] = raw_test

        platt_oof[
            outer_test
        ] = platt_test

        isotonic_oof[
            outer_test
        ] = isotonic_test


        held_out = (
            groups[
                outer_test
            ][0]
        )


        print(
            f"{validation_name} "
            f"fold {fold:02d}: "
            f"held out={held_out} "
            f"C={best_c}"
        )


    # ======================================================
    # METRICS
    # ======================================================

    methods = {
        "RAW_BALANCED_LOGISTIC":
            raw_oof,

        "PLATT_CALIBRATED":
            platt_oof,

        "ISOTONIC_CALIBRATED":
            isotonic_oof,
    }


    summary_rows = []

    bin_tables = []


    for method, probability in methods.items():

        summary_rows.append(
            {
                "validation": validation_name,

                "method": method,

                "roc_auc": roc_auc_score(
                    y,
                    probability,
                ),

                "pr_auc": (
                    average_precision_score(
                        y,
                        probability,
                    )
                ),

                "brier_score": (
                    brier_score_loss(
                        y,
                        probability,
                    )
                ),

                "log_loss": log_loss(
                    y,
                    probability,
                    labels=[0, 1],
                ),

                "ece_10": (
                    expected_calibration_error(
                        y,
                        probability,
                        bins=10,
                    )
                ),

                "mean_probability": float(
                    probability.mean()
                ),

                "actual_prevalence": float(
                    y.mean()
                ),

                "most_selected_C": (
                    pd.Series(
                        selected_c
                    )
                    .mode()
                    .iloc[0]
                ),
            }
        )


        bin_tables.append(
            calibration_bins(
                y,
                probability,
                validation_name,
                method,
            )
        )


    prediction_table = pd.DataFrame(
        {
            "record_id":
                data["record_id"].to_numpy(),

            "source_group":
                data["source_group"].to_numpy(),

            "date":
                data["date"].to_numpy(),

            "year":
                data["year"].to_numpy(),

            "presence":
                y,

            "validation":
                validation_name,

            "raw_probability":
                raw_oof,

            "platt_probability":
                platt_oof,

            "isotonic_probability":
                isotonic_oof,
        }
    )


    return (
        pd.DataFrame(
            summary_rows
        ),

        prediction_table,

        pd.concat(
            bin_tables,
            ignore_index=True,
        ),
    )


# ==========================================================
# RUN SURVEY + YEAR VALIDATION
# ==========================================================

summaries = []
predictions = []
bins = []


for group_column, validation_name in [
    (
        "source_group",
        "NESTED_SURVEY",
    ),

    (
        "year",
        "NESTED_YEAR",
    ),
]:

    print("\n" + "=" * 90)
    print(validation_name)
    print("=" * 90)


    (
        summary,
        prediction,
        bin_table,
    ) = run_nested_calibration(
        group_column,
        validation_name,
    )


    summaries.append(
        summary
    )

    predictions.append(
        prediction
    )

    bins.append(
        bin_table
    )


# ==========================================================
# FINAL RESULTS
# ==========================================================

summary_df = pd.concat(
    summaries,
    ignore_index=True,
)


print("\n" + "=" * 90)
print("STRICT CALIBRATION COMPARISON")
print("=" * 90)


print(
    summary_df[
        [
            "validation",
            "method",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "ece_10",
            "mean_probability",
            "actual_prevalence",
            "most_selected_C",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


pd.concat(
    predictions,
    ignore_index=True,
).to_csv(
    OOF_OUTPUT,
    index=False,
)


pd.concat(
    bins,
    ignore_index=True,
).to_csv(
    BIN_OUTPUT,
    index=False,
)


print("\nSaved:")
print(SUMMARY_OUTPUT)
print(OOF_OUTPUT)
print(BIN_OUTPUT)
