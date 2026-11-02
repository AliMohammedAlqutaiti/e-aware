#!/usr/bin/env python3

"""
E-AWARE V2 Step 4C5
Coast-matched local availability backgrounds.

Purpose:
Reduce the artificial coast-distance separation created by
Step 4C4.

For each observed Engraulis ringens presence:

- same exact date
- 3 generated local availability backgrounds
- 10–50 km from the observed presence
- ocean according to GEBCO
- <=50 nautical miles from coastline
- >2 km from every known anchoveta presence
- background coastline distance must be within ±5 km
  of the observed presence coastline distance

IMPORTANT:
pb_target=0 means locally available marine habitat.
It does NOT mean biological absence.
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
    / "anchoveta_coast_matched_background.csv"
)

PRESENCE_OUT = (
    OUTDIR
    / "anchoveta_coast_matched_presences.csv"
)

COMBINED_OUT = (
    OUTDIR
    / "anchoveta_presence_coast_matched.csv"
)

QC_OUT = (
    OUTDIR
    / "coast_matched_background_qc.csv"
)


BACKGROUNDS_PER_PRESENCE = 3

MIN_ANCHOR_DISTANCE_KM = 10.0
MAX_ANCHOR_DISTANCE_KM = 50.0

COAST_DISTANCE_TOLERANCE_KM = 5.0

MIN_DISTANCE_FROM_ANY_PRESENCE_KM = 2.0

MAX_COAST_DISTANCE_NM = 50.0

MAX_COAST_DISTANCE_KM = (
    MAX_COAST_DISTANCE_NM
    *
    KM_PER_NAUTICAL_MILE
)

MAX_ATTEMPTS_PER_PRESENCE = 5000

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
    retained_presence_rows = []
    qc_rows = []

    print(
        "E-AWARE V2 Step 4C5 — "
        "Coast-Matched Local Availability"
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
        "Anchor distance:",
        f"{MIN_ANCHOR_DISTANCE_KM}"
        "–"
        f"{MAX_ANCHOR_DISTANCE_KM} km"
    )

    print(
        "Coast matching tolerance:",
        f"±{COAST_DISTANCE_TOLERANCE_KM} km"
    )

    print(
        "Background date:",
        "exact anchor date"
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

        # ------------------------------------------------
        # Calculate observed presence coast distance.
        # Presence itself is not rejected if a coarse
        # bathymetry product labels the point awkwardly.
        # ------------------------------------------------

        try:

            anchor_coastlines = (
                coastline_for_latitude(
                    anchor_lat,
                    coastline_cache,
                )
            )

            anchor_coast_km = (
                coast_distance(
                    anchor_lat,
                    anchor_lon,
                    anchor_coastlines,
                )
            )

        except Exception as exc:

            print(
                f"[{anchor_number:02d}/"
                f"{len(presence)}] "
                f"{anchor_date.date()} "
                f"→ anchor coast-distance failure: "
                f"{exc}"
            )

            qc_rows.append(
                {
                    "anchor_presence_row":
                        anchor_number,

                    "date":
                        anchor_date,

                    "status":
                        "anchor_coast_failure",

                    "backgrounds":
                        0,
                }
            )

            continue

        if not np.isfinite(
            anchor_coast_km
        ):

            print(
                f"[{anchor_number:02d}/"
                f"{len(presence)}] "
                f"{anchor_date.date()} "
                f"→ invalid anchor coast distance"
            )

            continue

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
            # in 10–50 km annulus.
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

            # --------------------------------------------
            # Avoid near-duplicate known anchoveta points.
            # --------------------------------------------

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

            # --------------------------------------------
            # Coastline matching FIRST.
            # This avoids unnecessary GEBCO calls.
            # --------------------------------------------

            try:

                candidate_coastlines = (
                    coastline_for_latitude(
                        lat,
                        coastline_cache,
                    )
                )

                if not candidate_coastlines:
                    continue

                candidate_coast_km = (
                    coast_distance(
                        lat,
                        lon,
                        candidate_coastlines,
                    )
                )

            except Exception:
                continue

            if not np.isfinite(
                candidate_coast_km
            ):
                continue

            if (
                candidate_coast_km
                >
                MAX_COAST_DISTANCE_KM
            ):
                continue

            coast_delta_km = (
                candidate_coast_km
                -
                anchor_coast_km
            )

            if (
                abs(
                    coast_delta_km
                )
                >
                COAST_DISTANCE_TOLERANCE_KM
            ):
                continue

            # --------------------------------------------
            # GEBCO ocean QC only after coast match.
            # --------------------------------------------

            try:

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

            except Exception:
                continue

            if not is_ocean:
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
                    "coast_matched_local_availability",

                "source":
                    "generated_coast_matched_local_availability",

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

                "anchor_distance_coast_km":
                    anchor_coast_km,

                "background_distance_coast_km":
                    candidate_coast_km,

                "coast_distance_delta_km":
                    coast_delta_km,

                "abs_coast_distance_delta_km":
                    abs(
                        coast_delta_km
                    ),

                "background_distance_coast_nm":
                    (
                        candidate_coast_km
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

        status = (
            "complete"
            if len(accepted)
            ==
            BACKGROUNDS_PER_PRESENCE
            else
            "incomplete"
        )

        print(
            f"[{anchor_number:02d}/"
            f"{len(presence)}] "
            f"{anchor_date.date()} "
            f"coast={anchor_coast_km:.2f} km "
            f"→ {len(accepted)} backgrounds "
            f"| attempts={attempts} "
            f"| total="
            f"{len(background_rows)}"
        )

        qc_rows.append(
            {
                "anchor_presence_row":
                    anchor_number,

                "date":
                    anchor_date,

                "anchor_lat":
                    anchor_lat,

                "anchor_lon":
                    anchor_lon,

                "anchor_distance_coast_km":
                    anchor_coast_km,

                "backgrounds":
                    len(accepted),

                "attempts":
                    attempts,

                "status":
                    status,
            }
        )

        # Only retain complete 1 + 3 sets.
        if (
            len(accepted)
            ==
            BACKGROUNDS_PER_PRESENCE
        ):

            p = anchor.copy()

            p["presence"] = 1
            p["pb_target"] = 1
            p["sample_type"] = "presence"

            p["background_type"] = np.nan

            p[
                "anchor_presence_row"
            ] = anchor_number

            p[
                "anchor_date"
            ] = anchor_date

            p[
                "anchor_lat"
            ] = anchor_lat

            p[
                "anchor_lon"
            ] = anchor_lon

            p[
                "anchor_distance_coast_km"
            ] = anchor_coast_km

            retained_presence_rows.append(
                p
            )


    if not background_rows:

        raise RuntimeError(
            "No coast-matched backgrounds "
            "were generated."
        )


    background = pd.DataFrame(
        background_rows
    )

    qc = pd.DataFrame(
        qc_rows
    )

    complete_anchor_ids = set(
        qc.loc[
            qc["status"] == "complete",
            "anchor_presence_row",
        ].astype(int)
    )

    # Remove partial background sets.
    background = background[
        background[
            "anchor_presence_row"
        ].astype(int)
        .isin(
            complete_anchor_ids
        )
    ].copy()

    presence_out = pd.DataFrame(
        retained_presence_rows
    )

    background.to_csv(
        BACKGROUND_OUT,
        index=False,
    )

    presence_out.to_csv(
        PRESENCE_OUT,
        index=False,
    )

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
                "anchor_presence_row",
                "sample_type",
            ]
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
    )

    qc.to_csv(
        QC_OUT,
        index=False,
    )


    # ====================================================
    # FINAL QC
    # ====================================================

    print()
    print(
        "===== STEP 4C5 RESULTS ====="
    )

    print(
        "Original anchoveta presences:",
        len(presence)
    )

    print(
        "Complete coast-matched presences:",
        len(presence_out)
    )

    print(
        "Coast-matched backgrounds:",
        len(background)
    )

    print(
        "Complete anchors:",
        len(complete_anchor_ids),
        "/",
        len(presence)
    )

    print()

    if len(background):

        print(
            "ANCHOR DISTANCE"
        )

        print(
            "Median:",
            round(
                background[
                    "anchor_distance_km"
                ].median(),
                2,
            ),
            "km"
        )

        print(
            "Minimum:",
            round(
                background[
                    "anchor_distance_km"
                ].min(),
                2,
            ),
            "km"
        )

        print(
            "Maximum:",
            round(
                background[
                    "anchor_distance_km"
                ].max(),
                2,
            ),
            "km"
        )

        print()

        print(
            "COAST-DISTANCE MATCH"
        )

        print(
            "Median absolute difference:",
            round(
                background[
                    "abs_coast_distance_delta_km"
                ].median(),
                2,
            ),
            "km"
        )

        print(
            "Mean absolute difference:",
            round(
                background[
                    "abs_coast_distance_delta_km"
                ].mean(),
                2,
            ),
            "km"
        )

        print(
            "Maximum absolute difference:",
            round(
                background[
                    "abs_coast_distance_delta_km"
                ].max(),
                2,
            ),
            "km"
        )

        print(
            "Background farther offshore:",
            round(
                100
                *
                (
                    background[
                        "coast_distance_delta_km"
                    ]
                    >
                    0
                ).mean(),
                1,
            ),
            "%"
        )

        print(
            "Background closer to coast:",
            round(
                100
                *
                (
                    background[
                        "coast_distance_delta_km"
                    ]
                    <
                    0
                ).mean(),
                1,
            ),
            "%"
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

    failed = qc[
        qc["status"] != "complete"
    ]

    print()
    print(
        "Incomplete / failed anchors:",
        len(failed)
    )

    if len(failed):

        print(
            failed[
                [
                    "anchor_presence_row",
                    "date",
                    "backgrounds",
                    "attempts",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "Combined dataset:"
    )

    print(
        COMBINED_OUT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "pb_target=0 remains locally "
        "available habitat, NOT anchoveta absence."
    )

    print()

    print(
        "Do NOT start Copernicus preprocessing "
        "until Step 4C5 QC has been reviewed."
    )


if __name__ == "__main__":
    main()
