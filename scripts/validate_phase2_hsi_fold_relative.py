from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

FOLD_SELECTIONS = (
    ROOT
    / "phase2_final_nested_fold_results.csv"
)

OUTPUT = (
    ROOT
    / "phase2_hsi_fold_relative_validation.csv"
)

PRED_OUTPUT = (
    ROOT
    / "phase2_hsi_fold_relative_predictions.csv"
)


# ==========================================================
# DATA
# ==========================================================

df = pd.read_csv(
    INPUT,
    parse_dates=["date"],
)

selected = pd.read_csv(
    FOLD_SELECTIONS
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


X = data[
    FEATURES
].to_numpy()

y = (
    data["presence"]
    .astype(int)
    .to_numpy()
)


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


def percentile_against_training(
    training_scores,
    test_scores,
):

    reference = np.sort(
        np.asarray(
            training_scores,
            dtype=float,
        )
    )

    ranks = np.searchsorted(
        reference,
        np.asarray(
            test_scores,
            dtype=float,
        ),
        side="right",
    )

    return (
        ranks
        / len(reference)
    )


def run_validation(
    group_column,
    validation_name,
):

    groups = (
        data[group_column]
        .to_numpy()
    )

    logo = LeaveOneGroupOut()

    raw_oof = np.full(
        len(data),
        np.nan,
    )

    hsi_oof = np.full(
        len(data),
        np.nan,
    )

    fold_rows = []


    selection_table = selected[
        selected["validation"]
        == validation_name
    ]


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


        match = selection_table[
            selection_table["fold"]
            == fold
        ]


        if len(match) != 1:

            raise ValueError(
                f"Could not find selected C "
                f"for {validation_name} fold {fold}"
            )


        C = float(
            match.iloc[0][
                "selected_C"
            ]
        )


        model = build_model(C)

        model.fit(
            X[train_idx],
            y[train_idx],
        )


        # Same exact model generates both
        # training-reference and held-out scores.
        train_score = (
            model.predict_proba(
                X[train_idx]
            )[:, 1]
        )

        test_score = (
            model.predict_proba(
                X[test_idx]
            )[:, 1]
        )


        test_hsi = (
            percentile_against_training(
                train_score,
                test_score,
            )
        )


        raw_oof[
            test_idx
        ] = test_score

        hsi_oof[
            test_idx
        ] = test_hsi


        fold_rows.append(
            {
                "validation":
                    validation_name,

                "fold":
                    fold,

                "held_out_group":
                    groups[test_idx][0],

                "selected_C":
                    C,

                "records":
                    len(test_idx),

                "mean_raw_score":
                    float(
                        test_score.mean()
                    ),

                "mean_hsi":
                    float(
                        test_hsi.mean()
                    ),
            }
        )


    result = pd.DataFrame(
        {
            "record_id":
                data["record_id"]
                .to_numpy(),

            "source_group":
                data["source_group"]
                .to_numpy(),

            "year":
                data["year"]
                .to_numpy(),

            "presence":
                y,

            "validation":
                validation_name,

            "raw_score":
                raw_oof,

            "hsi":
                hsi_oof,
        }
    )


    result["hsi_class"] = pd.cut(
        result["hsi"],
        bins=[
            0.0,
            0.3,
            0.6,
            0.8,
            1.0,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        include_lowest=True,
    )


    # Deciles for deeper diagnostic.
    result["hsi_decile"] = pd.cut(
        result["hsi"],
        bins=np.linspace(
            0,
            1,
            11,
        ),
        labels=[
            "0-10",
            "10-20",
            "20-30",
            "30-40",
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100",
        ],
        include_lowest=True,
    )


    return (
        result,
        pd.DataFrame(
            fold_rows
        ),
    )


all_predictions = []
all_folds = []


for group_column, validation_name in [
    (
        "source_group",
        "NESTED_LEAVE_ONE_SURVEY_OUT",
    ),
    (
        "year",
        "NESTED_LEAVE_ONE_YEAR_OUT",
    ),
]:

    prediction, folds = run_validation(
        group_column,
        validation_name,
    )

    all_predictions.append(
        prediction
    )

    all_folds.append(
        folds
    )


predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)


summaries = []


print("=" * 90)
print(
    "E-AWARE — CORRECTED FOLD-RELATIVE HSI VALIDATION"
)
print("=" * 90)


for validation in predictions[
    "validation"
].unique():

    subset = predictions[
        predictions["validation"]
        == validation
    ].copy()

    baseline = (
        subset["presence"]
        .mean()
    )


    bands = (
        subset.groupby(
            "hsi_class",
            observed=False,
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
            observed_presence_rate=(
                "presence",
                "mean",
            ),
            mean_hsi=(
                "hsi",
                "mean",
            ),
            mean_raw_score=(
                "raw_score",
                "mean",
            ),
        )
        .reset_index()
    )


    bands[
        "lift_vs_baseline"
    ] = (
        bands[
            "observed_presence_rate"
        ]
        / baseline
    )


    bands[
        "validation"
    ] = validation


    summaries.append(
        bands
    )


    print("\n" + "=" * 90)
    print(validation)
    print("=" * 90)

    print(
        "Baseline:",
        round(
            baseline,
            4
        )
    )


    print("\nHSI BANDS")

    print(
        bands[
            [
                "hsi_class",
                "records",
                "detections",
                "observed_presence_rate",
                "lift_vs_baseline",
                "mean_hsi",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    deciles = (
        subset.groupby(
            "hsi_decile",
            observed=False,
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
            presence_rate=(
                "presence",
                "mean",
            ),
            mean_hsi=(
                "hsi",
                "mean",
            ),
        )
        .reset_index()
    )


    print("\nHSI DECILES")

    print(
        deciles
        .round(4)
        .to_string(
            index=False
        )
    )


    for threshold in [
        0.8,
        0.9,
    ]:

        top = subset[
            subset["hsi"]
            >= threshold
        ]

        rate = (
            top["presence"]
            .mean()
        )

        print(
            f"\nHSI >= {threshold:.1f}: "
            f"presence={rate:.4f}, "
            f"lift={rate/baseline:.3f}x, "
            f"n={len(top)}"
        )


summary_df = pd.concat(
    summaries,
    ignore_index=True,
)


summary_df.to_csv(
    OUTPUT,
    index=False,
)


predictions.to_csv(
    PRED_OUTPUT,
    index=False,
)


pd.concat(
    all_folds,
    ignore_index=True,
).to_csv(
    ROOT
    / "phase2_hsi_fold_relative_folds.csv",
    index=False,
)


print("\nSaved:")
print(OUTPUT)
print(PRED_OUTPUT)
