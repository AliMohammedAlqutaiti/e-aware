#!/usr/bin/env python3

"""
E-AWARE V2 — Final C5 validation gate.

Candidate models:

1) core_all
   SST + salinity + surface O2
   Uses all final C5 anchors.

2) core_valid_depth
   Same core predictors,
   restricted to anchors with valid local water depth
   for all 4 locations.

3) core_plus_physical_oxycline
   Core +
   - oxycline reached before seabed
   - oxycline depth when physically reached

Validation:
- grouped 5-fold by year
- leave-one-1-degree-spatial-block-out
- leave-one-2-degree-spatial-block-out

Primary validation metric:
held-out within-anchor ranking.

Background = local available habitat,
NOT confirmed anchoveta absence.

Scores are relative suitability rankings,
NOT calibrated probability of presence.
"""

from pathlib import Path
import warnings

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
    "data/processed/"
    "eaware_v2_step4_coast_matched/"
    "eaware_v2_coast_matched_final_candidate.csv"
)

OUTDIR = Path(
    "data/models/"
    "eaware_v2_c5_final_validation"
)

CORE = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
]

ACCEPTED = (
    "Accepted within local water column"
)

BELOW = (
    "Raw crossing below seabed; "
    "local threshold not reached"
)

NOT_REACHED = (
    "1 mL/L threshold not reached "
    "in available profile"
)


def fit_model(
    X,
    y,
    groups,
):

    model = ConditionalLogit(
        y,
        X,
        groups=groups,
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        result = model.fit(
            method="bfgs",
            maxiter=4000,
            disp=False,
        )

    return result


def matched_metrics(
    frame,
    score_col="score",
):

    concordance = []
    top1 = []
    ranks = []

    for _, g in frame.groupby(
        "qc_anchor_id"
    ):

        p = g.loc[
            g["pb_target"] == 1,
            score_col,
        ].to_numpy(
            dtype=float
        )

        b = g.loc[
            g["pb_target"] == 0,
            score_col,
        ].to_numpy(
            dtype=float
        )

        if (
            len(p) != 1
            or
            len(b) != 3
        ):
            continue

        p = float(
            p[0]
        )

        concordance.append(
            float(
                np.mean(
                    (p > b).astype(float)
                    +
                    0.5
                    *
                    (p == b).astype(float)
                )
            )
        )

        values = np.concatenate(
            [[p], b]
        )

        maximum = np.max(
            values
        )

        tied_max = np.sum(
            np.isclose(
                values,
                maximum,
            )
        )

        if np.isclose(
            p,
            maximum,
        ):
            top1.append(
                1.0
                /
                tied_max
            )
        else:
            top1.append(
                0.0
            )

        greater = np.sum(
            values > p
        )

        equal = np.sum(
            np.isclose(
                values,
                p,
            )
        )

        rank = (
            greater
            +
            1
            +
            (equal - 1)
            /
            2
        )

        ranks.append(
            float(rank)
        )

    return {
        "anchors":
            len(concordance),

        "concordance":
            float(
                np.mean(
                    concordance
                )
            ),

        "top1":
            float(
                np.mean(
                    top1
                )
            ),

        "rank":
            float(
                np.mean(
                    ranks
                )
            ),
    }


def prepare_data():

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    df["qc_anchor_id"] = (
        df["qc_anchor_id"]
        .astype(int)
    )

    df["pb_target"] = (
        df["pb_target"]
        .astype(int)
    )

    # ------------------------------------------
    # Recover anchor location/date from presence.
    # ------------------------------------------

    anchor_meta = (
        df[
            df["pb_target"] == 1
        ][
            [
                "qc_anchor_id",
                "lat",
                "lon",
                "date",
            ]
        ]
        .drop_duplicates(
            "qc_anchor_id"
        )
        .rename(
            columns={
                "lat":
                    "anchor_lat",

                "lon":
                    "anchor_lon",

                "date":
                    "anchor_date",
            }
        )
    )

    df = df.merge(
        anchor_meta,
        on="qc_anchor_id",
        how="left",
    )

    df["anchor_year"] = (
        pd.to_datetime(
            df["anchor_date"]
        )
        .dt.year
        .astype(int)
    )

    df["block_1deg"] = (
        np.floor(
            df["anchor_lat"]
        )
        .astype(int)
        .astype(str)
        +
        "_"
        +
        np.floor(
            df["anchor_lon"]
        )
        .astype(int)
        .astype(str)
    )

    df["block_2deg"] = (
        np.floor(
            df["anchor_lat"] / 2
        )
        .astype(int)
        .astype(str)
        +
        "_"
        +
        np.floor(
            df["anchor_lon"] / 2
        )
        .astype(int)
        .astype(str)
    )

    # ------------------------------------------
    # Valid-depth subset.
    # ------------------------------------------

    depth_complete = (
        df.groupby(
            "qc_anchor_id"
        )[
            "historical_water_depth_m"
        ]
        .apply(
            lambda x:
                x.notna().all()
        )
    )

    valid_depth_anchors = set(
        depth_complete[
            depth_complete
        ]
        .index
        .astype(int)
    )

    df[
        "valid_depth_anchor"
    ] = (
        df["qc_anchor_id"]
        .isin(
            valid_depth_anchors
        )
    )

    # ------------------------------------------
    # Physical oxycline encoding.
    # ------------------------------------------

    df[
        "oxycline_reached_before_seabed"
    ] = np.nan

    df[
        "oxycline_depth_when_reached_m"
    ] = np.nan

    mask = (
        df[
            "valid_depth_anchor"
        ]
    )

    valid_statuses = {
        ACCEPTED,
        BELOW,
        NOT_REACHED,
    }

    found = set(
        df.loc[
            mask,
            "historical_oxycline_status",
        ]
        .dropna()
        .unique()
    )

    unexpected = (
        found
        -
        valid_statuses
    )

    if unexpected:

        raise RuntimeError(
            "Unexpected oxycline status: "
            f"{unexpected}"
        )

    df.loc[
        mask,
        "oxycline_reached_before_seabed",
    ] = (
        df.loc[
            mask,
            "historical_oxycline_status",
        ]
        .eq(
            ACCEPTED
        )
        .astype(int)
    )

    df.loc[
        mask,
        "oxycline_depth_when_reached_m",
    ] = np.where(
        df.loc[
            mask,
            "historical_oxycline_status",
        ]
        .eq(
            ACCEPTED
        ),
        df.loc[
            mask,
            "historical_oxycline_m",
        ],
        0.0,
    )

    return df


def get_splits(
    df,
    validation,
):

    if validation == "year":

        anchor_frame = (
            df[
                [
                    "qc_anchor_id",
                    "anchor_year",
                ]
            ]
            .drop_duplicates(
                "qc_anchor_id"
            )
            .sort_values(
                "qc_anchor_id"
            )
            .reset_index(
                drop=True
            )
        )

        groups = (
            anchor_frame[
                "anchor_year"
            ]
            .to_numpy()
        )

        anchors = (
            anchor_frame[
                "qc_anchor_id"
            ]
            .to_numpy()
        )

        n_splits = min(
            5,
            pd.Series(
                groups
            ).nunique(),
        )

        splitter = GroupKFold(
            n_splits=n_splits
        )

        for fold, (
            train_i,
            test_i,
        ) in enumerate(
            splitter.split(
                anchors,
                groups=groups,
            ),
            start=1,
        ):

            yield (
                f"year_fold_{fold}",
                set(
                    anchors[
                        train_i
                    ]
                ),
                set(
                    anchors[
                        test_i
                    ]
                ),
            )

        return

    if validation == "spatial_1deg":

        group_col = (
            "block_1deg"
        )

    elif validation == "spatial_2deg":

        group_col = (
            "block_2deg"
        )

    else:

        raise ValueError(
            validation
        )

    anchor_frame = (
        df[
            [
                "qc_anchor_id",
                group_col,
            ]
        ]
        .drop_duplicates(
            "qc_anchor_id"
        )
    )

    all_anchors = set(
        anchor_frame[
            "qc_anchor_id"
        ].astype(int)
    )

    for block, g in (
        anchor_frame.groupby(
            group_col
        )
    ):

        test_anchors = set(
            g[
                "qc_anchor_id"
            ].astype(int)
        )

        train_anchors = (
            all_anchors
            -
            test_anchors
        )

        # Avoid attempting a nearly empty
        # conditional model.
        if len(
            train_anchors
        ) < 8:

            print(
                f"Skipping block {block}: "
                f"only {len(train_anchors)} "
                f"training anchors."
            )

            continue

        yield (
            str(block),
            train_anchors,
            test_anchors,
        )


def validate(
    df,
    features,
    model_name,
    validation,
):

    rows = []
    oof_parts = []

    for fold_name, (
        train_anchors
    ), (
        test_anchors
    ) in [
        (
            name,
            tr,
            te,
        )
        for name, tr, te
        in get_splits(
            df,
            validation,
        )
    ]:

        train = df[
            df[
                "qc_anchor_id"
            ].isin(
                train_anchors
            )
        ].copy()

        test = df[
            df[
                "qc_anchor_id"
            ].isin(
                test_anchors
            )
        ].copy()

        if (
            train[
                "qc_anchor_id"
            ].nunique()
            < 8
        ):
            continue

        X_train_raw = (
            train[
                features
            ]
            .astype(float)
            .to_numpy()
        )

        X_test_raw = (
            test[
                features
            ]
            .astype(float)
            .to_numpy()
        )

        y_train = (
            train[
                "pb_target"
            ]
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test[
                "pb_target"
            ]
            .astype(int)
            .to_numpy()
        )

        train_groups = (
            train[
                "qc_anchor_id"
            ]
            .astype(int)
            .to_numpy()
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

        try:

            result = fit_model(
                X_train,
                y_train,
                train_groups,
            )

        except Exception as exc:

            print(
                f"Fit failed "
                f"{model_name} "
                f"{validation} "
                f"{fold_name}: "
                f"{exc}"
            )

            continue

        params = np.asarray(
            result.params,
            dtype=float,
        )

        score = (
            X_test
            @
            params
        )

        test["score"] = (
            score
        )

        mm = matched_metrics(
            test
        )

        auc = roc_auc_score(
            y_test,
            score,
        )

        ap = average_precision_score(
            y_test,
            score,
        )

        row = {
            "model":
                model_name,

            "validation":
                validation,

            "fold":
                fold_name,

            "train_anchors":
                train[
                    "qc_anchor_id"
                ].nunique(),

            "test_anchors":
                test[
                    "qc_anchor_id"
                ].nunique(),

            "auc":
                float(
                    auc
                ),

            "ap":
                float(
                    ap
                ),

            "concordance":
                mm[
                    "concordance"
                ],

            "top1":
                mm[
                    "top1"
                ],

            "rank":
                mm[
                    "rank"
                ],
        }

        rows.append(
            row
        )

        oof_parts.append(
            test[
                [
                    "qc_anchor_id",
                    "pb_target",
                    "score",
                ]
            ]
        )

        print(
            f"  {fold_name}: "
            f"train={row['train_anchors']} "
            f"| test={row['test_anchors']} "
            f"| AUC={auc:.3f} "
            f"| concordance="
            f"{mm['concordance']:.3f} "
            f"| Top1="
            f"{mm['top1']:.3f}"
        )

    fold_df = pd.DataFrame(
        rows
    )

    if (
        fold_df.empty
        or
        not oof_parts
    ):

        return (
            fold_df,
            None,
        )

    oof = pd.concat(
        oof_parts,
        ignore_index=True,
    )

    pooled = matched_metrics(
        oof
    )

    weights = (
        fold_df[
            "test_anchors"
        ]
        .to_numpy(
            dtype=float
        )
    )

    weighted_auc = (
        np.average(
            fold_df[
                "auc"
            ],
            weights=weights,
        )
    )

    weighted_ap = (
        np.average(
            fold_df[
                "ap"
            ],
            weights=weights,
        )
    )

    summary = {
        "model":
            model_name,

        "validation":
            validation,

        "evaluated_anchors":
            pooled[
                "anchors"
            ],

        "weighted_fold_auc":
            float(
                weighted_auc
            ),

        "weighted_fold_ap":
            float(
                weighted_ap
            ),

        "pooled_matched_concordance":
            pooled[
                "concordance"
            ],

        "pooled_top1":
            pooled[
                "top1"
            ],

        "pooled_presence_rank":
            pooled[
                "rank"
            ],
    }

    return (
        fold_df,
        summary,
    )


def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = prepare_data()

    all52 = (
        df.copy()
    )

    valid34 = (
        df[
            df[
                "valid_depth_anchor"
            ]
        ]
        .copy()
    )

    print(
        "E-AWARE V2 — "
        "FINAL C5 VALIDATION GATE"
    )

    print()

    print(
        "All C5 anchors:",
        all52[
            "qc_anchor_id"
        ].nunique()
    )

    print(
        "Valid-depth anchors:",
        valid34[
            "qc_anchor_id"
        ].nunique()
    )

    print(
        "1° blocks all C5:",
        all52[
            all52[
                "pb_target"
            ] == 1
        ][
            "block_1deg"
        ].nunique()
    )

    print(
        "2° blocks all C5:",
        all52[
            all52[
                "pb_target"
            ] == 1
        ][
            "block_2deg"
        ].nunique()
    )

    print(
        "Years all C5:",
        all52[
            all52[
                "pb_target"
            ] == 1
        ][
            "anchor_year"
        ].nunique()
    )

    MODELS = [
        (
            "core_all52",
            all52,
            CORE,
        ),

        (
            "core_valid_depth34",
            valid34,
            CORE,
        ),

        (
            "core_plus_physical_oxycline34",
            valid34,
            CORE
            +
            [
                "oxycline_reached_before_seabed",
                "oxycline_depth_when_reached_m",
            ],
        ),
    ]

    validations = [
        "year",
        "spatial_1deg",
        "spatial_2deg",
    ]

    all_folds = []
    summaries = []

    for (
        model_name,
        model_df,
        features,
    ) in MODELS:

        print()
        print(
            "======================================"
        )

        print(
            "MODEL:",
            model_name
        )

        print(
            "Anchors:",
            model_df[
                "qc_anchor_id"
            ].nunique()
        )

        print(
            "Features:"
        )

        for f in features:
            print(
                " -",
                f
            )

        print(
            "======================================"
        )

        for validation in validations:

            print()
            print(
                validation
            )

            result = validate(
                model_df,
                features,
                model_name,
                validation,
            )

            if len(result) == 2:

                folds, summary = result

            else:

                folds, summary = (
                    result[0],
                    result[1],
                )

            if not folds.empty:

                all_folds.append(
                    folds
                )

            if summary is None:
                continue

            summaries.append(
                summary
            )

            print(
                "  ---- POOLED HELD-OUT RESULT ----"
            )

            print(
                "  Anchors:",
                summary[
                    "evaluated_anchors"
                ]
            )

            print(
                "  Weighted fold AUC:",
                f"{summary['weighted_fold_auc']:.3f}"
            )

            print(
                "  Matched concordance:",
                f"{summary['pooled_matched_concordance']:.3f}"
            )

            print(
                "  Top-1:",
                f"{summary['pooled_top1']:.3f}"
            )

            print(
                "  Presence rank:",
                f"{summary['pooled_presence_rank']:.3f}"
            )

    if all_folds:

        pd.concat(
            all_folds,
            ignore_index=True,
        ).to_csv(
            OUTDIR
            /
            "final_validation_fold_metrics.csv",
            index=False,
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUTDIR
        /
        "final_validation_summary.csv",
        index=False,
    )

    print()
    print(
        "======================================"
    )

    print(
        "FINAL VALIDATION SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()

    print(
        "Random expectations:"
    )

    print(
        "  Matched concordance = 0.500"
    )

    print(
        "  Top-1 = 0.250"
    )

    print(
        "  Presence rank ≈ 2.500"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Matched concordance / Top-1 are "
        "the primary validation metrics."
    )

    print(
        "AUC is reported only as a "
        "weighted descriptive fold metric."
    )

    print(
        "Scores from different fitted folds "
        "are NOT pooled into one global AUC."
    )

    print()

    print(
        "===== FINAL C5 VALIDATION COMPLETE ====="
    )


if __name__ == "__main__":
    main()
