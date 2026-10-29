#!/usr/bin/env python3

"""
E-AWARE V2 Step 4C
Presence-background dataset builder.

Observed Engraulis ringens records remain PRESENCES.
Generated points are BACKGROUND points, NOT biological absences.

Background design:
- Peruvian Pacific coastal ocean
- maximum 100 nautical miles from coastline
- GEBCO-confirmed ocean cells
- same date distribution as observed presences
- deterministic random seed for reproducibility
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


INPUT = Path(
    "data/public_occurrences/anchoveta_public_presences.csv"
)

OUTDIR = Path(
    "data/public_occurrences"
)

BACKGROUND_OUT = (
    OUTDIR
    / "anchoveta_background_points.csv"
)

COMBINED_OUT = (
    OUTDIR
    / "anchoveta_presence_background.csv"
)

QC_OUT = (
    OUTDIR
    / "presence_background_qc.csv"
)

BACKGROUND_PER_PRESENCE = 3

RANDOM_SEED = 42

# Broad Peruvian Pacific search envelope.
# Final accepted points must also pass ocean + coast-distance QC.
MIN_LAT = -18.5
MAX_LAT = -3.0

MIN_LON = -84.5
MAX_LON = -69.0

MAX_COAST_DISTANCE_NM = 100.0

MAX_COAST_DISTANCE_KM = (
    MAX_COAST_DISTANCE_NM
    *
    KM_PER_NAUTICAL_MILE
)

# Avoid generating essentially identical locations.
ROUND_DECIMALS = 4

MAX_ATTEMPTS_PER_BACKGROUND = 250


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    r = 6371.0088

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

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


def coastline_band(
    lat,
    cache,
):
    """
    Load NOAA coastline in ~2 degree latitude bands.
    This avoids repeatedly downloading the whole Peru coastline.
    """

    band_size = 2.0

    index = math.floor(
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
        index
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
        round(
            south,
            2
        ),
        round(
            north,
            2
        ),
    )

    if key not in cache:

        print(
            "Loading coastline band:",
            south,
            "to",
            north,
        )

        cache[key] = coastline_lines(
            MIN_LON,
            MAX_LON,
            south,
            north,
        )

    return cache[key]


def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    presence = pd.read_csv(
        INPUT,
        parse_dates=[
            "date"
        ],
    )

    presence = presence[
        presence[
            "date"
        ].notna()
        &
        presence[
            "lat"
        ].notna()
        &
        presence[
            "lon"
        ].notna()
    ].copy()

    presence = presence.reset_index(
        drop=True
    )

    print(
        "E-AWARE Step 4C — "
        "Presence-background builder"
    )

    print()

    print(
        "Observed presences:",
        len(
            presence
        ),
    )

    print(
        "Background per presence:",
        BACKGROUND_PER_PRESENCE,
    )

    target_background = (
        len(
            presence
        )
        *
        BACKGROUND_PER_PRESENCE
    )

    print(
        "Target background points:",
        target_background,
    )

    print(
        "Background marine domain:",
        f"0–{MAX_COAST_DISTANCE_NM:.0f} nm "
        "from the Peruvian Pacific coast",
    )

    print()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    coastline_cache = {}

    backgrounds = []

    used = set()

    presence_coords = (
        presence[
            [
                "lat",
                "lon",
            ]
        ]
        .astype(float)
        .to_numpy()
    )

    total_attempts = 0

    for presence_index, prow in (
        presence.iterrows()
    ):

        created = 0

        attempts = 0

        while (
            created
            <
            BACKGROUND_PER_PRESENCE
        ):

            attempts += 1

            total_attempts += 1

            if (
                attempts
                >
                MAX_ATTEMPTS_PER_BACKGROUND
                *
                BACKGROUND_PER_PRESENCE
            ):

                print(
                    "WARNING: could only create",
                    created,
                    "background points for presence row",
                    presence_index + 1,
                )

                break

            lat = float(
                rng.uniform(
                    MIN_LAT,
                    MAX_LAT,
                )
            )

            lon = float(
                rng.uniform(
                    MIN_LON,
                    MAX_LON,
                )
            )

            key = (
                prow[
                    "date"
                ].strftime(
                    "%Y-%m-%d"
                ),
                round(
                    lat,
                    ROUND_DECIMALS,
                ),
                round(
                    lon,
                    ROUND_DECIMALS,
                ),
            )

            if key in used:
                continue

            try:

                lines = coastline_band(
                    lat,
                    coastline_cache,
                )

                if not lines:
                    continue

                distance_km = coast_distance(
                    lat,
                    lon,
                    lines,
                )

                if (
                    not np.isfinite(
                        distance_km
                    )
                ):
                    continue

                if (
                    distance_km
                    >
                    MAX_COAST_DISTANCE_KM
                ):
                    continue

                elevation, depth, is_ocean = gebco(
                    lat,
                    lon,
                )

                if not is_ocean:
                    continue

                # Avoid an exact/near duplicate of a known occurrence.
                nearest_presence_km = min(
                    haversine_km(
                        lat,
                        lon,
                        float(
                            p_lat
                        ),
                        float(
                            p_lon
                        ),
                    )
                    for p_lat, p_lon
                    in presence_coords
                )

                if nearest_presence_km < 2.0:
                    continue

            except Exception:
                continue

            used.add(
                key
            )

            backgrounds.append(
                {
                    "date":
                        prow[
                            "date"
                        ],

                    "lat":
                        lat,

                    "lon":
                        lon,

                    # IMPORTANT:
                    # background != confirmed absence
                    "presence":
                        np.nan,

                    "pb_target":
                        0,

                    "sample_type":
                        "background",

                    "source":
                        "generated_background",

                    "background_for_presence_row":
                        int(
                            presence_index
                            +
                            1
                        ),

                    "background_distance_coast_km":
                        float(
                            distance_km
                        ),

                    "background_distance_coast_nm":
                        float(
                            distance_km
                            /
                            KM_PER_NAUTICAL_MILE
                        ),

                    "gebco_elevation_m":
                        elevation,

                    "gebco_depth_m":
                        depth,

                    "nearest_known_presence_km":
                        nearest_presence_km,
                }
            )

            created += 1

        if (
            (
                presence_index
                +
                1
            )
            %
            10
            ==
            0
            or
            (
                presence_index
                +
                1
            )
            ==
            len(
                presence
            )
        ):

            print(
                f"Processed "
                f"{presence_index + 1}/"
                f"{len(presence)} presences | "
                f"backgrounds created: "
                f"{len(backgrounds)}"
            )

    background = pd.DataFrame(
        backgrounds
    )

    background.to_csv(
        BACKGROUND_OUT,
        index=False,
    )

    # -----------------------------
    # Presence side
    # -----------------------------

    presence[
        "sample_type"
    ] = "presence"

    presence[
        "pb_target"
    ] = 1

    presence[
        "presence"
    ] = 1

    # Keep the original GBIF source label
    if "source" not in presence.columns:
        presence[
            "source"
        ] = "GBIF"

    # -----------------------------
    # Combine
    # -----------------------------

    combined = pd.concat(
        [
            presence,
            background,
        ],
        ignore_index=True,
        sort=False,
    )

    combined[
        "date"
    ] = pd.to_datetime(
        combined[
            "date"
        ],
        errors="coerce",
    )

    combined = combined.sort_values(
        [
            "date",
            "sample_type",
        ]
    ).reset_index(
        drop=True
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
    )

    # -----------------------------
    # QC
    # -----------------------------

    qc = pd.DataFrame(
        [
            [
                "Observed presences",
                int(
                    (
                        combined[
                            "sample_type"
                        ]
                        ==
                        "presence"
                    ).sum()
                ),
            ],

            [
                "Generated background",
                int(
                    (
                        combined[
                            "sample_type"
                        ]
                        ==
                        "background"
                    ).sum()
                ),
            ],

            [
                "Confirmed biological absences",
                0,
            ],

            [
                "Presence-background total",
                len(
                    combined
                ),
            ],

            [
                "Random seed",
                RANDOM_SEED,
            ],

            [
                "Maximum background coast distance nm",
                MAX_COAST_DISTANCE_NM,
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
        "===== STEP 4C RESULTS ====="
    )

    print(
        "Observed presences:",
        len(
            presence
        ),
    )

    print(
        "Generated backgrounds:",
        len(
            background
        ),
    )

    print(
        "Total modelling rows:",
        len(
            combined
        ),
    )

    print(
        "Confirmed biological absences: 0"
    )

    if len(
        background
    ):

        print()

        print(
            "Background coast-distance range:"
        )

        print(
            f"{background['background_distance_coast_nm'].min():.2f}"
            " – "
            f"{background['background_distance_coast_nm'].max():.2f}"
            " nm"
        )

        print()

        print(
            "Background depth range:"
        )

        print(
            f"{background['gebco_depth_m'].min():.1f}"
            " – "
            f"{background['gebco_depth_m'].max():.1f}"
            " m"
        )

    print()

    print(
        "Background file:",
        BACKGROUND_OUT,
    )

    print(
        "Combined file:",
        COMBINED_OUT,
    )

    print(
        "QC file:",
        QC_OUT,
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "pb_target=0 means BACKGROUND, "
        "not confirmed absence."
    )


if __name__ == "__main__":
    main()
