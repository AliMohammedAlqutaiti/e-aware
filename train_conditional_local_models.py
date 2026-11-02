#!/usr/bin/env python3

"""
E-AWARE V2 — Step 5 V3
Matched Conditional Logistic Regression.

Two nested models:

A) Oceanography
   SST + salinity + surface dissolved oxygen

B) Oceanography + Coast
   SST + salinity + surface dissolved oxygen
   + distance from coast

Each anchoveta presence and its 3 local backgrounds form
one matched stratum.

IMPORTANT:
- Backgrounds are NOT biological absences.
- Scores represent relative habitat suitability /
  used-vs-locally-available habitat.
- Model B may contain observer/accessibility bias through
  distance from coast, so it is treated as a comparison,
  not automatically the final model.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
)


INPUT = Path(
    "data/processed/eaware_v2_step4_local/"
    "eaware_v2_step4_local_balanced_candidate.csv"
)

OUTDIR = Path(
    "data/models/"
    "eaware_v2_conditional_models"
)

MODELS = {
    "oceanography": [
        "historical_sst_c",
        "historical_salinity_psu",
        "historical_surface_o2_ml_l",
    ],

    "oceanography_plus_coast": [
        "historical_sst_c",
        "historical_salinity_psu",
        "historical_surface_o2_ml_l",
        "historical_distance_coast_km",
    ],
}

N_SPLITS = 5


def fit_conditional(
    X,
    y,
    groups,
):
    """
    Fit Conditional Logistic Regression.

    No intercept is included because the conditional model
    handles a group-specific intercept implicitly.
    """

    model = ConditionalLogit(
        y,
        X,
        groups=groups,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = model.fit(
            method="bfgs",
            maxiter=2000,
            disp=False,
        )

    return result


def matched_metrics(
    frame,
    scores,
):
    """
    Evaluate ranking inside each matched anchor.

    Each anchor contains:
      1 observed presence
      3 available backgrounds

    Metrics:
      within_anchor_concordance:
          probability the observed presence scores above
          a background from the same anchor.
          Random expectation = 0.50

      top1_accuracy:
          fraction of anchors where the observed presence
          is ranked highest among all four locations.
          Random expectation = 0.25

      mean_presence_rank:
          rank of presence within its four choices.
          1 = best, 4 = worst.
          Random expectation ~2.5
    """

    temp = frame[
        [
            "qc_anchor_id",
            "pb_target",
        ]
    ].copy()

    temp["score"] = scores

    concordances = []
    top1_values = []
    ranks = []

    for anchor, group in temp.groupby(
        "qc_anchor_id"
    ):

        presence_scores = group.loc[
            group["pb_target"] == 1,
            "score",
        ].to_numpy()

        background_scores = group.loc[
            group["pb_target"] == 0,
            "score",
        ].to_numpy()

        if (
            len(presence_scores) != 1
            or
            len(background_scores) != 3
        ):
            continue

        p = float(
            presence_scores[0]
        )

        b = np.asarray(
            background_scores,
            dtype=float,
        )

        # Pairwise concordance with 0.5 credit for ties.
        concordance = np.mean(
            (p > b).astype(float)
            +
            0.5
            *
            (p == b).astype(float)
        )

        concordances.append(
            float(concordance)
        )

        all_scores = np.concatenate(
            [
                [p],
                b,
            ]
        )

        maximum = np.max(
            all_scores
        )

        tied_max = np.sum(
            np.isclose(
                all_scores,
                maximum,
            )
        )

        if np.isclose(
            p,
            maximum,
        ):
            # Fractional credit in the unlikely case of ties.
            top1_values.append(
                1.0 / tied_max
            )
        else:
            top1_values.append(
                0.0
            )

        # Average rank for ties, descending score.
        greater = np.sum(
            all_scores > p
        )

        equal = np.sum(
            np.isclose(
                all_scores,
                p,
            )
        )

        rank = (
            greater
            +
            1
            +
            (equal - 1) / 2
        )

        ranks.append(
            float(rank)
        )

    return {
        "anchors":
            len(concordances),

        "within_anchor_concordance":
            float(
                np.mean(
                    concordances
                )
            ),

        "top1_accuracy":
            float(
                np.mean(
                    top1_values
                )
            ),

        "mean_presence_rank":
            float(
                np.mean(
                    ranks
                )
            ),
    }


def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    df["pb_target"] = (
        df["pb_target"]
        .astype(int)
    )

    df["qc_anchor_id"] = (
        df["qc_anchor_id"]
        .astype(int)
    )

    print(
        "E-AWARE V2 Step 5 V3 — "
        "Matched Conditional Logistic Models"
    )

    print()

    print(
        "Rows:",
        len(df)
    )

    print(
        "Anchors:",
        df[
            "qc_anchor_id"
        ].nunique()
    )

    print(
        "Presences:",
        int(
            (
                df["pb_target"]
                ==
                1
            ).sum()
        )
    )

    print(
        "Backgrounds:",
        int(
            (
                df["pb_target"]
                ==
                0
            ).sum()
        )
    )

    print()

    # --------------------------------------------------
    # Confirm matched design
    # --------------------------------------------------

    check = (
        df.groupby(
            "qc_anchor_id"
        )
        .agg(
            rows=(
                "pb_target",
                "size",
            ),
            presences=(
                "pb_target",
                "sum",
            ),
        )
    )

    bad = check[
        ~(
            (check["rows"] == 4)
            &
            (check["presences"] == 1)
        )
    ]

    if len(bad):

        raise RuntimeError(
            "Dataset no longer contains "
            "exactly 1 presence + 3 backgrounds "
            "per anchor."
        )

    print(
        "✅ Matched design confirmed:"
    )

    print(
        "   1 presence + 3 backgrounds "
        "for every anchor."
    )

    print()

    groups_all = (
        df["qc_anchor_id"]
        .to_numpy()
    )

    y_all = (
        df["pb_target"]
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    all_fold_results = []
    model_summaries = {}

    # ==================================================
    # MODEL LOOP
    # ==================================================

    for model_name, features in (
        MODELS.items()
    ):

        print(
            "\n======================================"
        )

        print(
            "MODEL:",
            model_name
        )

        print(
            "Features:"
        )

        for feature in features:
            print(
                " -",
                feature
            )

        print(
            "======================================"
        )

        missing = [
            c
            for c in features
            if c not in df.columns
        ]

        if missing:

            raise RuntimeError(
                f"Missing columns: {missing}"
            )

        complete = (
            df[
                features
            ]
            .notna()
            .all(axis=1)
        )

        if not complete.all():

            raise RuntimeError(
                f"{model_name} has missing "
                "predictor values."
            )

        X_df = (
            df[
                features
            ]
            .astype(float)
            .copy()
        )

        fold_metrics = []

        for fold, (
            train_idx,
            test_idx,
        ) in enumerate(
            splitter.split(
                X_df,
                y_all,
                groups_all,
            ),
            start=1,
        ):

            X_train_raw = (
                X_df
                .iloc[
                    train_idx
                ]
                .to_numpy()
            )

            X_test_raw = (
                X_df
                .iloc[
                    test_idx
                ]
                .to_numpy()
            )

            y_train = (
                y_all[
                    train_idx
                ]
            )

            y_test = (
                y_all[
                    test_idx
                ]
            )

            group_train = (
                groups_all[
                    train_idx
                ]
            )

            scaler = (
                StandardScaler()
                .fit(
                    X_train_raw
                )
            )

            X_train = (
                scaler.transform(
                    X_train_raw
                )
            )

            X_test = (
                scaler.transform(
                    X_test_raw
                )
            )

            result = fit_conditional(
                X_train,
                y_train,
                group_train,
            )

            params = np.asarray(
                result.params,
                dtype=float,
            )

            # Linear predictor only.
            # Ranking is what matters here;
            # this is NOT calibrated presence probability.
            scores = (
                X_test
                @
                params
            )

            auc = roc_auc_score(
                y_test,
                scores,
            )

            ap = average_precision_score(
                y_test,
                scores,
            )

            test_frame = (
                df.iloc[
                    test_idx
                ]
                .copy()
            )

            matched = matched_metrics(
                test_frame,
                scores,
            )

            row = {
                "model":
                    model_name,

                "fold":
                    fold,

                "test_anchors":
                    matched[
                        "anchors"
                    ],

                "roc_auc":
                    float(
                        auc
                    ),

                "average_precision":
                    float(
                        ap
                    ),

                "within_anchor_concordance":
                    matched[
                        "within_anchor_concordance"
                    ],

                "top1_accuracy":
                    matched[
                        "top1_accuracy"
                    ],

                "mean_presence_rank":
                    matched[
                        "mean_presence_rank"
                    ],
            }

            fold_metrics.append(
                row
            )

            all_fold_results.append(
                row
            )

            print(
                f"Fold {fold}: "
                f"AUC={auc:.3f} | "
                f"AP={ap:.3f} | "
                f"matched concordance="
                f"{matched['within_anchor_concordance']:.3f} | "
                f"Top-1="
                f"{matched['top1_accuracy']:.3f} | "
                f"presence rank="
                f"{matched['mean_presence_rank']:.2f}"
            )

        fold_df = pd.DataFrame(
            fold_metrics
        )

        # ----------------------------------------------
        # Summary
        # ----------------------------------------------

        summary = {
            "model":
                model_name,

            "features":
                features,

            "mean_roc_auc":
                float(
                    fold_df[
                        "roc_auc"
                    ].mean()
                ),

            "sd_roc_auc":
                float(
                    fold_df[
                        "roc_auc"
                    ].std()
                ),

            "mean_average_precision":
                float(
                    fold_df[
                        "average_precision"
                    ].mean()
                ),

            "mean_within_anchor_concordance":
                float(
                    fold_df[
                        "within_anchor_concordance"
                    ].mean()
                ),

            "sd_within_anchor_concordance":
                float(
                    fold_df[
                        "within_anchor_concordance"
                    ].std()
                ),

            "mean_top1_accuracy":
                float(
                    fold_df[
                        "top1_accuracy"
                    ].mean()
                ),

            "mean_presence_rank":
                float(
                    fold_df[
                        "mean_presence_rank"
                    ].mean()
                ),
        }

        print()

        print(
            "----- MODEL SUMMARY -----"
        )

        print(
            "Mean ROC-AUC:",
            round(
                summary[
                    "mean_roc_auc"
                ],
                3,
            )
        )

        print(
            "ROC-AUC SD:",
            round(
                summary[
                    "sd_roc_auc"
                ],
                3,
            )
        )

        print(
            "Mean AP:",
            round(
                summary[
                    "mean_average_precision"
                ],
                3,
            )
        )

        print(
            "Mean within-anchor concordance:",
            round(
                summary[
                    "mean_within_anchor_concordance"
                ],
                3,
            ),
            "(random = 0.500)"
        )

        print(
            "Mean Top-1 accuracy:",
            round(
                summary[
                    "mean_top1_accuracy"
                ],
                3,
            ),
            "(random = 0.250)"
        )

        print(
            "Mean presence rank:",
            round(
                summary[
                    "mean_presence_rank"
                ],
                3,
            ),
            "(random ≈ 2.500)"
        )

        # ----------------------------------------------
        # Final full-data fit
        # ----------------------------------------------

        scaler_final = (
            StandardScaler()
            .fit(
                X_df.to_numpy()
            )
        )

        X_final = (
            scaler_final.transform(
                X_df.to_numpy()
            )
        )

        final_result = fit_conditional(
            X_final,
            y_all,
            groups_all,
        )

        coefficients = pd.DataFrame(
            {
                "predictor":
                    features,

                "standardized_coefficient":
                    np.asarray(
                        final_result.params,
                        dtype=float,
                    ),
            }
        )

        coefficients[
            "absolute_coefficient"
        ] = (
            coefficients[
                "standardized_coefficient"
            ]
            .abs()
        )

        coefficients = (
            coefficients
            .sort_values(
                "absolute_coefficient",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        print()

        print(
            "Standardized conditional "
            "coefficients:"
        )

        print(
            coefficients[
                [
                    "predictor",
                    "standardized_coefficient",
                ]
            ]
            .to_string(
                index=False
            )
        )

        bundle = {
            "model_type":
                "conditional_logistic_regression",

            "model_name":
                model_name,

            "features":
                features,

            "scaler":
                scaler_final,

            "coefficients":
                np.asarray(
                    final_result.params,
                    dtype=float,
                ),

            "intercept":
                None,

            "background_is_absence":
                False,

            "interpretation":
                "relative habitat suitability",

            "matched_design":
                "1 observed presence + 3 "
                "same-date local availability "
                "backgrounds",

            "warning":
                (
                    "Scores are ranking / relative "
                    "suitability scores and are not "
                    "calibrated probability of presence."
                ),
        }

        joblib.dump(
            bundle,
            OUTDIR
            /
            f"{model_name}.joblib",
        )

        coefficients.to_csv(
            OUTDIR
            /
            f"{model_name}_coefficients.csv",
            index=False,
        )

        model_summaries[
            model_name
        ] = summary

    # ==================================================
    # COMPARISON
    # ==================================================

    folds_out = pd.DataFrame(
        all_fold_results
    )

    folds_out.to_csv(
        OUTDIR
        /
        "conditional_model_fold_metrics.csv",
        index=False,
    )

    with open(
        OUTDIR
        /
        "conditional_model_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            model_summaries,
            f,
            indent=2,
        )

    print(
        "\n======================================"
    )

    print(
        "STEP 5 V3 MODEL COMPARISON"
    )

    print(
        "======================================"
    )

    for name, s in (
        model_summaries.items()
    ):

        print()

        print(
            name
        )

        print(
            "  ROC-AUC:",
            round(
                s[
                    "mean_roc_auc"
                ],
                3,
            )
        )

        print(
            "  Within-anchor concordance:",
            round(
                s[
                    "mean_within_anchor_concordance"
                ],
                3,
            )
        )

        print(
            "  Top-1 accuracy:",
            round(
                s[
                    "mean_top1_accuracy"
                ],
                3,
            )
        )

        print(
            "  Mean presence rank:",
            round(
                s[
                    "mean_presence_rank"
                ],
                3,
            )
        )

    print()

    print(
        "Random matched expectations:"
    )

    print(
        "  Within-anchor concordance = 0.500"
    )

    print(
        "  Top-1 accuracy = 0.250"
    )

    print(
        "  Mean presence rank ≈ 2.500"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT automatically select the "
        "coast-distance model if it performs "
        "better."
    )

    print(
        "Distance from coast may include "
        "GBIF observer-accessibility bias."
    )

    print()

    print(
        "===== STEP 5 V3 COMPLETE ====="
    )


if __name__ == "__main__":
    main()
