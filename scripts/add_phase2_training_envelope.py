from pathlib import Path
import json

import numpy as np
import pandas as pd


DATA = Path(
    "data/imarpe/E_AWARE_IMARPE_phase2_final_features.csv"
)

METADATA = Path(
    "models/eaware_phase2_model_metadata.json"
)


df = pd.read_csv(
    DATA,
    parse_dates=["date"],
)


eligible = (
    df["date"].notna()
    &
    (
        df["date"]
        >= pd.Timestamp("1993-01-01")
    )
)


df = df[
    eligible
].copy()


df[
    "deploy_log_bottom_depth"
] = np.log1p(
    df[
        "deploy_bottom_depth_m"
    ]
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


complete = (
    df[FEATURES]
    .notna()
    .all(axis=1)
)


df = df[
    complete
].copy()


training_envelope = {}


rows = []


for feature in FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce",
    ).dropna()


    stats = {
        "min":
            float(values.min()),

        "p01":
            float(
                values.quantile(0.01)
            ),

        "p05":
            float(
                values.quantile(0.05)
            ),

        "median":
            float(
                values.median()
            ),

        "p95":
            float(
                values.quantile(0.95)
            ),

        "p99":
            float(
                values.quantile(0.99)
            ),

        "max":
            float(values.max()),

        "mean":
            float(values.mean()),

        "std":
            float(
                values.std(ddof=0)
            ),
    }


    training_envelope[
        feature
    ] = stats


    rows.append(
        {
            "feature":
                feature,

            **stats,
        }
    )


metadata = json.loads(
    METADATA.read_text(
        encoding="utf-8"
    )
)


metadata[
    "training_envelope"
] = training_envelope


metadata[
    "domain_rule"
] = {
    "operational_definition":
        (
            "A live prediction is considered outside "
            "the validated ML domain when one or more "
            "production predictors fall outside the "
            "observed Phase 2 training min-max range."
        ),

    "handling":
        (
            "Do not interpret historical-percentile ML HSI "
            "as validated habitat suitability outside this "
            "envelope. Use literature HSI as operational "
            "fallback and retain ML ranking as diagnostic only."
        ),
}


METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


table = pd.DataFrame(
    rows
)


print("=" * 100)
print(
    "E-AWARE PRODUCTION ML — TRAINING ENVELOPE"
)
print("=" * 100)


print(
    table[
        [
            "feature",
            "min",
            "p05",
            "median",
            "p95",
            "max",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nTraining records:",
    len(df)
)


print(
    "\nUpdated:",
    METADATA
)
