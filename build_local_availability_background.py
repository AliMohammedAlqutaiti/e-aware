#!/usr/bin/env python3

"""
E-AWARE V2 Step 4C4
Local availability background builder.

For each observed Engraulis ringens presence:
- generate 3 marine background points
- exact same date as the presence
- 10–50 km from the anchor
- <=50 nautical miles from coastline
- ocean according to GEBCO
- >2 km from every known anchoveta presence

BACKGROUND IS NOT BIOLOGICAL ABSENCE.

The resulting model should be interpreted as relative habitat
suitability / used-vs-locally-available habitat.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from preprocess_historical import (
    gebco,
    coastline_lines,
    coast_distance,
    KM_PER_NAUTICAL_MILE,
)


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    "data/public_occurrences/"
    "anchoveta_public_presences.csv"
)

OUTDIR = Path(
    "data/public_occurrences"
)

BACKGROUND_OUT = (
    OUTDIR
    / "anchoveta_local_availability_background.csv"
)

COMBINED_OUT = (
    OUTDIR
    / "anchoveta_presence_local_availability.csv"
)

QC_OUT = (
    OUTDIR
    / "local_availability_background_qc.csv"
)

BACKGROUNDS_PER_PRESENCE = 3

MIN_ANCHOR_DISTANCE_KM = 10.0
MAX_ANCHOR_DISTANCE_KM = 50.0

MIN_DISTANCE_FROM_ANY_PRESENCE_KM = 2.0

MAX_COAST_DISTANCE_NM = 50.0

MAX_COAST_DISTANCE_KM = (
    MAX_COAST_DISTANCE_NM
    *
    KM_PER_NAUTICAL_MILE
)

MAX_ATTEMPTS_PER_PRESENCE = 1000

RANDOM_SEED = 42

MIN_LAT = -18.5
MAX_LAT = -3.0

MIN_LON = -86.0
MAX_LON = -70.0


# ============================================================
# GEOGRAPHY
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    r = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(
        lat2 - lat1
    )

    dl = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dl / 2) ** 2
    )

    return (
        2
        *
        r
        *
        math.asin(
            math.sqrt(a)
        )
    )


def destination_point(
    lat,
    lon,
    distance_km,
    bearing_deg,
):
    r = 6371.0088

    angular = (
        distance_km
        /
        r
    )

    bearing = math.radians(
        bearing_deg
    )

    lat1 = math.radians(
        lat
    )

    lon1 = math.radians(
        lon
    )

    lat2 = math.asin(
        math.sin(lat1)
        *
        math.cos(angular)
        +
        math.cos(lat1)
        *
        math.sin(angular)
        *
        math.cos(bearing)
    )

    lon2 = (
        lon1
        +
        math.atan2(
            math.sin(bearing)
            *
            math.sin(angular)
            *
            math.cos(lat1),

            math.cos(angular)
            -
            math.sin(lat1)
            *
            math.sin(lat2),
        )
    )

    return (
        math.degrees(lat2),
        math.degrees(lon2),
    )


def coastline_for_latitude(
    lat,
    cache,
):
    band_size = 2.0

    band_index = math.floor(
        (
            lat
            -
            MIN_LAT
        )
        /
        band_size
    )

    south = (
        MIN_LAT
        +
        band_index
        *
        band_size
    )

    north = min(
        south
        +
        band_size,
        MAX_LAT,
    )

    key = (
        round(south, 2),
        round(north, 2),
    )

    if key not in cache:

        print(
            f"  Loading coastline "
            f"{south:.1f} to "
            f"{north:.1f}"
        )

        cache[key] = coastline_lines(
            MIN_LON,
            MAX_LON,
            south,
            north,
        )

    return cache[key]


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
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

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    presence_coordinates = (
        presence[
            ["lat", "lon"]
        ]
        .astype(float)
        .to_numpy()
    )

    coastline_cache = {}
    gebco_cache = {}

    used_locations = set()

    background_rows = []

    failed_anchors = []

    print(
        "E-AWARE V2 Step 4C4 — "
        "Local Availability Background"
    )

    print()

    print(
        "Anchoveta presences:",
        len(presence)
    )

    print(
        "Backgrounds requested:",
        len(presence)
        *
        BACKGROUNDS_PER_PRESENCE
    )

    print(
        "Anchor-distance range:",
        f"{MIN_ANCHOR_DISTANCE_KM}"
        "–"
        f"{MAX_ANCHOR_DISTANCE_KM} km"
    )

    print(
        "Background date: exact anchor date"
    )

    print()

    for index, anchor in (
        presence.iterrows()
    ):

        anchor_number = (
            index + 1
        )

        anchor_date = pd.Timestamp(
            anchor["date"]
        )

        anchor_lat = float(
            anchor["lat"]
        )

        anchor_lon = float(
            anchor["lon"]
        )

        accepted = []

        attempts = 0

        while (
            len(accepted)
            <
            BACKGROUNDS_PER_PRESENCE
            and
            attempts
            <
            MAX_ATTEMPTS_PER_PRESENCE
        ):

            attempts += 1

            # Uniform sampling by area
            # within the 10–50 km annulus.
            radius = math.sqrt(
                rng.uniform(
                    MIN_ANCHOR_DISTANCE_KM ** 2,
                    MAX_ANCHOR_DISTANCE_KM ** 2,
                )
            )

            bearing = rng.uniform(
                0.0,
                360.0,
            )

            lat, lon = destination_point(
                anchor_lat,
                anchor_lon,
                radius,
                bearing,
            )

            if not (
                MIN_LAT
                <=
                lat
                <=
                MAX_LAT
            ):
                continue

            if not (
                MIN_LON
                <=
                lon
                <=
                MAX_LON
            ):
                continue

            location_key = (
                round(lat, 4),
                round(lon, 4),
            )

            if (
                location_key
                in
                used_locations
            ):
                continue

            # Avoid generating a background
            # almost directly on top of any
            # known anchoveta occurrence.
            nearest_presence_km = min(
                haversine_km(
                    lat,
                    lon,
                    float(p_lat),
                    float(p_lon),
                )
                for p_lat, p_lon
                in presence_coordinates
            )

            if (
                nearest_presence_km
                <
                MIN_DISTANCE_FROM_ANY_PRESENCE_KM
            ):
                continue

            try:

                coastlines = (
                    coastline_for_latitude(
                        lat,
                        coastline_cache,
                    )
                )

                if not coastlines:
                    continue

                distance_coast_km = (
                    coast_distance(
                        lat,
                        lon,
                        coastlines,
                    )
                )

                if not np.isfinite(
                    distance_coast_km
                ):
                    continue

                if (
                    distance_coast_km
                    >
                    MAX_COAST_DISTANCE_KM
                ):
                    continue

                if (
                    location_key
                    not in
                    gebco_cache
                ):

                    gebco_cache[
                        location_key
                    ] = gebco(
                        lat,
                        lon,
                    )

                (
                    elevation,
                    depth,
                    is_ocean,
                ) = gebco_cache[
                    location_key
                ]

                if not is_ocean:
                    continue

            except Exception:
                continue

            actual_anchor_distance = (
                haversine_km(
                    anchor_lat,
                    anchor_lon,
                    lat,
                    lon,
                )
            )

            row = {
                "date":
                    anchor_date,

                "lat":
                    lat,

                "lon":
                    lon,

                "presence":
                    np.nan,

                "pb_target":
                    0,

                "sample_type":
                    "background",

                "background_type":
                    "local_availability",

                "source":
                    "generated_local_availability",

                "absence_semantics_verified":
                    False,

                "anchor_presence_row":
                    anchor_number,

                "anchor_date":
                    anchor_date,

                "anchor_lat":
                    anchor_lat,

                "anchor_lon":
                    anchor_lon,

                "anchor_distance_km":
                    actual_anchor_distance,

                "date_difference_days":
                    0,

                "nearest_known_presence_km":
                    nearest_presence_km,

                "background_distance_coast_km":
                    distance_coast_km,

                "background_distance_coast_nm":
                    (
                        distance_coast_km
                        /
                        KM_PER_NAUTICAL_MILE
                    ),

                "gebco_elevation_m":
                    elevation,

                "gebco_depth_m":
                    depth,
            }

            accepted.append(
                row
            )

            background_rows.append(
                row
            )

            used_locations.add(
                location_key
            )

        print(
            f"[{anchor_number:02d}/"
            f"{len(presence)}] "
            f"{anchor_date.date()} "
            f"→ {len(accepted)} "
            f"backgrounds "
            f"| attempts={attempts} "
            f"| total="
            f"{len(background_rows)}"
        )

        if (
            len(accepted)
            <
            BACKGROUNDS_PER_PRESENCE
        ):

            failed_anchors.append(
                anchor_number
            )

    if not background_rows:

        raise RuntimeError(
            "No local availability "
            "backgrounds generated."
        )

    background = pd.DataFrame(
        background_rows
    )

    background.to_csv(
        BACKGROUND_OUT,
        index=False,
    )

    # ---------------------------------------------
    # Only include anchors that obtained at least
    # one background.
    # ---------------------------------------------

    represented = sorted(
        background[
            "anchor_presence_row"
        ]
        .astype(int)
        .unique()
    )

    presence_out = (
        presence.iloc[
            [
                i - 1
                for i
                in represented
            ]
        ]
        .copy()
    )

    presence_out[
        "presence"
    ] = 1

    presence_out[
        "pb_target"
    ] = 1

    presence_out[
        "sample_type"
    ] = "presence"

    presence_out[
        "background_type"
    ] = np.nan

    if (
        "source"
        not in
        presence_out.columns
    ):
        presence_out[
            "source"
        ] = "GBIF"

    combined = pd.concat(
        [
            presence_out,
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

    counts = (
        background[
            "anchor_presence_row"
        ]
        .value_counts()
    )

    print()
    print(
        "===== STEP 4C4 RESULTS ====="
    )

    print(
        "Original anchoveta presences:",
        len(presence)
    )

    print(
        "Represented anchoveta presences:",
        len(represented)
    )

    print(
        "Local availability backgrounds:",
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
        "Backgrounds per anchor:"
    )

    print(
        counts
        .value_counts()
        .sort_index()
        .to_string()
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
        "Minimum:",
        round(
            background[
                "anchor_distance_km"
            ].min(),
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
        "All backgrounds:",
        "0 days"
    )

    print()

    print(
        "COAST DISTANCE"
    )

    print(
        "Median:",
        round(
            background[
                "background_distance_coast_nm"
            ].median(),
            1,
        ),
        "nm"
    )

    print(
        "Maximum:",
        round(
            background[
                "background_distance_coast_nm"
            ].max(),
            1,
        ),
        "nm"
    )

    print()

    print(
        "BACKGROUND DEPTH"
    )

    depth = (
        background[
            "gebco_depth_m"
        ]
    )

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

    print()

    print(
        "Failed anchors:",
        failed_anchors
    )

    print()

    print(
        "Combined dataset:",
        COMBINED_OUT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "pb_target=0 means locally "
        "available marine habitat, "
        "NOT biological absence."
    )

    qc = pd.DataFrame(
        [
            [
                "Original presences",
                len(presence),
            ],
            [
                "Represented presences",
                len(represented),
            ],
            [
                "Local backgrounds",
                len(background),
            ],
            [
                "Median anchor distance km",
                background[
                    "anchor_distance_km"
                ].median(),
            ],
            [
                "Maximum anchor distance km",
                background[
                    "anchor_distance_km"
                ].max(),
            ],
            [
                "Maximum temporal difference days",
                0,
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


if __name__ == "__main__":
    main()
