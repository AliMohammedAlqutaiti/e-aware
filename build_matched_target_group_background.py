#!/usr/bin/env python3

"""
E-AWARE V2 Step 4C3
Strict matched target-group background builder.

Hierarchy:
1. same dataset + same basis + same month
2. same dataset + same basis + same year, <=90 days
3. same basis + same month
4. same basis + same year, <=90 days

All candidates must:
- be other ray-finned fish observations
- be <=50 km from the anchoveta anchor
- be >2 km from known anchoveta presence locations
- pass marine / GEBCO / coastline QC
- NOT be interpreted as biological absence

Only anchoveta presences with >=1 matched background are retained
in the final matched modelling dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import build_target_group_background as base


INPUT = Path(
    "data/public_occurrences/"
    "anchoveta_public_presences.csv"
)

OUTDIR = Path(
    "data/public_occurrences"
)

BACKGROUND_OUT = (
    OUTDIR
    / "anchoveta_target_group_matched_background.csv"
)

PRESENCE_OUT = (
    OUTDIR
    / "anchoveta_target_group_matched_presences.csv"
)

COMBINED_OUT = (
    OUTDIR
    / "anchoveta_presence_target_group_matched.csv"
)

QC_OUT = (
    OUTDIR
    / "target_group_matched_qc.csv"
)


BACKGROUND_PER_PRESENCE = 3

MAX_ANCHOR_DISTANCE_KM = 50.0
MAX_YEAR_FALLBACK_DAYS = 90

LOCAL_BOX_DEGREES = 0.75

MONTH_QUERY_LIMIT = 1500
YEAR_QUERY_LIMIT = 3000


def local_geometry(lat, lon):

    d = LOCAL_BOX_DEGREES

    west = lon - d
    east = lon + d
    south = lat - d
    north = lat + d

    return (
        "POLYGON(("
        f"{west} {south},"
        f"{east} {south},"
        f"{east} {north},"
        f"{west} {north},"
        f"{west} {south}"
        "))"
    )


def query_params(
    anchor,
    use_dataset,
    use_month,
):

    date = pd.Timestamp(
        anchor["date"]
    )

    basis = base.anchor_basis(
        anchor
    )

    if basis is None:
        return None

    params = {
        "taxonKey":
            base.FISH_TAXON_KEY,

        "checklistKey":
            base.COL_XR_CHECKLIST_KEY,

        "country":
            "PE",

        "hasCoordinate":
            "true",

        "hasGeospatialIssue":
            "false",

        "occurrenceStatus":
            "PRESENT",

        "basisOfRecord":
            basis,

        "year":
            int(date.year),

        "geometry":
            local_geometry(
                float(anchor["lat"]),
                float(anchor["lon"]),
            ),
    }

    if use_month:

        params["month"] = int(
            date.month
        )

    if use_dataset:

        dataset_key = anchor.get(
            "dataset_key"
        )

        if (
            pd.isna(dataset_key)
            or
            not str(dataset_key).strip()
        ):
            return None

        params["datasetKey"] = str(
            dataset_key
        )

    return params


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    presence = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    presence = (
        presence
        .dropna(
            subset=[
                "date",
                "lat",
                "lon",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "E-AWARE V2 Step 4C3 — "
        "Strict Matched Target-Group Background"
    )

    print()

    print(
        "Anchoveta presences:",
        len(presence)
    )

    print(
        "Maximum anchor distance:",
        MAX_ANCHOR_DISTANCE_KM,
        "km"
    )

    print(
        "Year-fallback temporal limit:",
        MAX_YEAR_FALLBACK_DAYS,
        "days"
    )

    print()

    presence_coordinates = (
        presence[
            ["lat", "lon"]
        ]
        .astype(float)
        .to_numpy()
    )

    query_cache = {}

    coastline_cache = {}
    gebco_cache = {}

    globally_used = set()

    accepted = []

    tiers = [
        {
            "name":
                "same_dataset_month",

            "dataset":
                True,

            "month":
                True,

            "max_days":
                35,

            "limit":
                MONTH_QUERY_LIMIT,
        },

        {
            "name":
                "same_dataset_90d",

            "dataset":
                True,

            "month":
                False,

            "max_days":
                MAX_YEAR_FALLBACK_DAYS,

            "limit":
                YEAR_QUERY_LIMIT,
        },

        {
            "name":
                "same_basis_month",

            "dataset":
                False,

            "month":
                True,

            "max_days":
                35,

            "limit":
                MONTH_QUERY_LIMIT,
        },

        {
            "name":
                "same_basis_90d",

            "dataset":
                False,

            "month":
                False,

            "max_days":
                MAX_YEAR_FALLBACK_DAYS,

            "limit":
                YEAR_QUERY_LIMIT,
        },
    ]

    for i, anchor in presence.iterrows():

        anchor_number = i + 1

        anchor_date = pd.Timestamp(
            anchor["date"]
        )

        anchor_lat = float(
            anchor["lat"]
        )

        anchor_lon = float(
            anchor["lon"]
        )

        selected = []

        for tier in tiers:

            if (
                len(selected)
                >=
                BACKGROUND_PER_PRESENCE
            ):
                break

            params = query_params(
                anchor,
                use_dataset=tier["dataset"],
                use_month=tier["month"],
            )

            if params is None:
                continue

            cache_key = tuple(
                sorted(
                    (str(k), str(v))
                    for k, v
                    in params.items()
                )
            )

            if cache_key not in query_cache:

                query_cache[
                    cache_key
                ] = base.fetch_records(
                    params,
                    tier["limit"],
                )

            records = query_cache[
                cache_key
            ]

            candidates = []

            for record in records:

                row = base.record_to_row(
                    record,
                    tier["name"],
                    anchor_number,
                )

                if not base.basic_qc(
                    row
                ):
                    continue

                row_date = pd.Timestamp(
                    row["date"]
                )

                date_diff = abs(
                    (
                        row_date
                        -
                        anchor_date
                    ).days
                )

                if (
                    date_diff
                    >
                    tier["max_days"]
                ):
                    continue

                anchor_distance = (
                    base.haversine_km(
                        anchor_lat,
                        anchor_lon,
                        float(row["lat"]),
                        float(row["lon"]),
                    )
                )

                if (
                    anchor_distance
                    >
                    MAX_ANCHOR_DISTANCE_KM
                ):
                    continue

                # Avoid essentially contradictory
                # presence/background labels at
                # almost identical locations.
                if anchor_distance < 2.0:
                    continue

                candidate_id = (
                    base.candidate_key(
                        row
                    )
                )

                if (
                    candidate_id
                    in
                    globally_used
                ):
                    continue

                candidates.append(
                    (
                        date_diff,
                        anchor_distance,
                        candidate_id,
                        row,
                    )
                )

            # Prefer closest in time,
            # then closest in space.
            candidates.sort(
                key=lambda x: (
                    x[0],
                    x[1],
                )
            )

            for (
                date_diff,
                anchor_distance,
                candidate_id,
                row,
            ) in candidates:

                if (
                    len(selected)
                    >=
                    BACKGROUND_PER_PRESENCE
                ):
                    break

                marine_row = (
                    base.marine_qc(
                        row,
                        presence_coordinates,
                        coastline_cache,
                        gebco_cache,
                    )
                )

                if marine_row is None:
                    continue

                marine_row[
                    "anchor_date"
                ] = anchor_date

                marine_row[
                    "anchor_lat"
                ] = anchor_lat

                marine_row[
                    "anchor_lon"
                ] = anchor_lon

                marine_row[
                    "anchor_distance_km"
                ] = float(
                    anchor_distance
                )

                marine_row[
                    "date_difference_days"
                ] = int(
                    date_diff
                )

                marine_row[
                    "pair_tier"
                ] = tier[
                    "name"
                ]

                globally_used.add(
                    candidate_id
                )

                selected.append(
                    marine_row
                )

                accepted.append(
                    marine_row
                )

        print(
            f"[{anchor_number:02d}/"
            f"{len(presence)}] "
            f"{anchor_date.date()} "
            f"→ {len(selected)} "
            f"matched backgrounds "
            f"| total={len(accepted)}"
        )

    if not accepted:

        raise RuntimeError(
            "No matched target-group "
            "backgrounds survived QC."
        )

    background = pd.DataFrame(
        accepted
    )

    background[
        "presence"
    ] = np.nan

    background[
        "pb_target"
    ] = 0

    background[
        "sample_type"
    ] = "background"

    background[
        "background_type"
    ] = "matched_target_group"

    background[
        "source"
    ] = "GBIF_target_group_matched"

    background[
        "absence_semantics_verified"
    ] = False

    background.to_csv(
        BACKGROUND_OUT,
        index=False,
    )

    represented = sorted(
        background[
            "anchor_presence_row"
        ]
        .astype(int)
        .unique()
    )

    matched_presence = (
        presence.iloc[
            [
                x - 1
                for x
                in represented
            ]
        ]
        .copy()
    )

    matched_presence[
        "presence"
    ] = 1

    matched_presence[
        "pb_target"
    ] = 1

    matched_presence[
        "sample_type"
    ] = "presence"

    matched_presence[
        "background_type"
    ] = np.nan

    if (
        "source"
        not in
        matched_presence.columns
    ):
        matched_presence[
            "source"
        ] = "GBIF"

    matched_presence.to_csv(
        PRESENCE_OUT,
        index=False,
    )

    combined = pd.concat(
        [
            matched_presence,
            background,
        ],
        ignore_index=True,
        sort=False,
    )

    combined[
        "date"
    ] = pd.to_datetime(
        combined["date"],
        errors="coerce",
    )

    combined = (
        combined
        .sort_values(
            [
                "date",
                "sample_type",
            ]
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
    )

    per_anchor = (
        background[
            "anchor_presence_row"
        ]
        .value_counts()
    )

    tiers_used = (
        background[
            "pair_tier"
        ]
        .value_counts()
    )

    print()
    print(
        "===== STEP 4C3 RESULTS ====="
    )

    print(
        "Original anchoveta presences:",
        len(presence)
    )

    print(
        "Matched anchoveta presences:",
        len(matched_presence)
    )

    print(
        "Matched target-group backgrounds:",
        len(background)
    )

    print(
        "Anchors represented:",
        len(represented),
        "/",
        len(presence)
    )

    print()

    print(
        "Backgrounds per represented anchor:"
    )

    print(
        per_anchor
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Selection tiers:"
    )

    print(
        tiers_used.to_string()
    )

    print()

    print(
        "ANCHOR DISTANCE"
    )

    print(
        "Median:",
        round(
            background[
                "anchor_distance_km"
            ].median(),
            1,
        ),
        "km"
    )

    print(
        "Mean:",
        round(
            background[
                "anchor_distance_km"
            ].mean(),
            1,
        ),
        "km"
    )

    print(
        "Maximum:",
        round(
            background[
                "anchor_distance_km"
            ].max(),
            1,
        ),
        "km"
    )

    print()

    print(
        "TEMPORAL DIFFERENCE"
    )

    print(
        "Median:",
        round(
            background[
                "date_difference_days"
            ].median(),
            1,
        ),
        "days"
    )

    print(
        "Mean:",
        round(
            background[
                "date_difference_days"
            ].mean(),
            1,
        ),
        "days"
    )

    print(
        "Maximum:",
        int(
            background[
                "date_difference_days"
            ].max()
        ),
        "days"
    )

    print()

    print(
        "BACKGROUND DEPTH"
    )

    depth = background[
        "gebco_depth_m"
    ]

    print(
        "<=200 m:",
        int(
            (
                depth <= 200
            ).sum()
        )
    )

    print(
        "200–1000 m:",
        int(
            (
                (depth > 200)
                &
                (depth <= 1000)
            ).sum()
        )
    )

    print(
        "1000–3000 m:",
        int(
            (
                (depth > 1000)
                &
                (depth <= 3000)
            ).sum()
        )
    )

    print(
        ">3000 m:",
        int(
            (
                depth > 3000
            ).sum()
        )
    )

    qc = pd.DataFrame(
        [
            [
                "Original presences",
                len(presence),
            ],
            [
                "Matched presences",
                len(matched_presence),
            ],
            [
                "Matched backgrounds",
                len(background),
            ],
            [
                "Anchors represented",
                len(represented),
            ],
            [
                "Median anchor distance km",
                background[
                    "anchor_distance_km"
                ].median(),
            ],
            [
                "Median date difference days",
                background[
                    "date_difference_days"
                ].median(),
            ],
            [
                "Confirmed biological absences",
                0,
            ],
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    qc.to_csv(
        QC_OUT,
        index=False,
    )

    print()
    print(
        "Combined matched dataset:"
    )

    print(
        COMBINED_OUT
    )

    print()
    print(
        "IMPORTANT: pb_target=0 remains "
        "target-group background, "
        "not confirmed anchoveta absence."
    )


if __name__ == "__main__":
    main()
