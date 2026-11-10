from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "phase2_final_nested_oof_predictions.csv"
)

OUTPUT = (
    ROOT
    / "phase2_hsi_band_validation.csv"
)


df = pd.read_csv(
    INPUT
)


print("=" * 90)
print("E-AWARE PHASE 2 — HSI BAND VALIDATION")
print("=" * 90)


all_results = []


for validation in sorted(
    df["validation"].unique()
):

    subset = (
        df[
            df["validation"]
            == validation
        ]
        .copy()
    )


    # ------------------------------------------------------
    # Convert raw model ranking to empirical percentile.
    #
    # 0.90 = model score is higher than approximately
    # 90% of historical held-out observations.
    # ------------------------------------------------------

    subset["hsi_percentile"] = (
        subset[
            "oof_probability_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )


    # Keep the original E-AWARE display bands:
    #
    # 0.0–0.3  Low
    # 0.3–0.6  Moderate
    # 0.6–0.8  High
    # 0.8–1.0  Very High
    # ------------------------------------------------------

    subset["hsi_class"] = pd.cut(
        subset["hsi_percentile"],
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


    baseline = (
        subset[
            "presence"
        ]
        .mean()
    )


    summary = (
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
                "hsi_percentile",
                "mean",
            ),

            min_hsi=(
                "hsi_percentile",
                "min",
            ),

            max_hsi=(
                "hsi_percentile",
                "max",
            ),

            mean_raw_score=(
                "oof_probability_raw",
                "mean",
            ),
        )
        .reset_index()
    )


    summary[
        "lift_vs_baseline"
    ] = (
        summary[
            "observed_presence_rate"
        ]
        / baseline
    )


    summary[
        "validation"
    ] = validation


    all_results.append(
        summary
    )


    print("\n" + "=" * 90)
    print(validation)
    print("=" * 90)

    print(
        "Baseline presence rate:",
        round(
            baseline,
            4
        )
    )

    print(
        summary[
            [
                "hsi_class",
                "records",
                "detections",
                "observed_presence_rate",
                "lift_vs_baseline",
                "mean_hsi",
                "mean_raw_score",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    # ------------------------------------------------------
    # Top-zone concentration.
    # ------------------------------------------------------

    top20 = (
        subset[
            subset[
                "hsi_percentile"
            ]
            >= 0.80
        ]
    )


    top10 = (
        subset[
            subset[
                "hsi_percentile"
            ]
            >= 0.90
        ]
    )


    print(
        "\nTop 20% presence rate:",
        round(
            top20[
                "presence"
            ].mean(),
            4
        )
    )

    print(
        "Top 20% lift:",
        round(
            top20[
                "presence"
            ].mean()
            / baseline,
            3
        ),
        "x"
    )


    print(
        "Top 10% presence rate:",
        round(
            top10[
                "presence"
            ].mean(),
            4
        )
    )

    print(
        "Top 10% lift:",
        round(
            top10[
                "presence"
            ].mean()
            / baseline,
            3
        ),
        "x"
    )


result = pd.concat(
    all_results,
    ignore_index=True,
)


result.to_csv(
    OUTPUT,
    index=False,
)


print("\nSaved:")
print(OUTPUT)
