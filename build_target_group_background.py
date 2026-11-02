#!/usr/bin/env python3

"""
E-AWARE V2 Step 4C2
Target-group background builder using GBIF Catalogue of Life XR.

Presence:
    Observed Engraulis ringens occurrence.

Background:
    Real observations of OTHER ray-finned fishes in Peru,
    matched to the anchoveta sampling process.

Priority:
    1. Same basis of record + same year + same month.
    2. Same basis of record + same year.

IMPORTANT:
    Background != confirmed biological absence.
    pb_target=0 means target-group background only.
"""

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import requests

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
    / "anchoveta_target_group_background.csv"
)

COMBINED_OUT = (
    OUTDIR
    / "anchoveta_presence_target_group.csv"
)

QC_OUT = (
    OUTDIR
    / "target_group_background_qc.csv"
)


GBIF_API = (
    "https://api.gbif.org/v1/occurrence/search"
)

# Catalogue of Life Extended Release
COL_XR_CHECKLIST_KEY = (
    "7ddf754f-d193-4cc9-b351-99906754a03b"
)

# Actinopterygii
FISH_TAXON_KEY = "8VR36"

# Broad Pacific coastal corridor following Peru.
# This is only a GBIF retrieval filter.
# Final acceptance is still checked using
# GEBCO ocean status + calculated coast distance.
COASTAL_GEOMETRY = (
    "POLYGON(("
    "-84.5 -3.0,"
    "-74.5 -18.5,"
    "-68.7 -18.5,"
    "-78.6 -3.0,"
    "-84.5 -3.0"
    "))"
)

BACKGROUND_PER_PRESENCE = 3

RANDOM_SEED = 42

MONTH_QUERY_LIMIT = 1500
YEAR_QUERY_LIMIT = 3000

MIN_LAT = -18.5
MAX_LAT = -3.0

MIN_LON = -86.0
MAX_LON = -70.0

MAX_COORDINATE_UNCERTAINTY_M = 10000

MAX_COAST_DISTANCE_NM = 50.0

MAX_COAST_DISTANCE_KM = (
    MAX_COAST_DISTANCE_NM
    *
    KM_PER_NAUTICAL_MILE
)

MIN_DISTANCE_FROM_PRESENCE_KM = 2.0


# ============================================================
# HTTP
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "E-AWARE University of Portsmouth "
            "academic prototype"
    }
)


def get_json(
    url,
    params,
    attempts=3,
    timeout=60,
):

    last_error = None

    for attempt in range(
        attempts
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=timeout,
            )

            response.raise_for_status()

            return response.json()

        except Exception as exc:

            last_error = exc

            if (
                attempt
                <
                attempts - 1
            ):
                time.sleep(
                    1.5
                    *
                    (
                        attempt
                        +
                        1
                    )
                )

    raise RuntimeError(
        f"GBIF request failed: "
        f"{last_error}"
    )


def fetch_records(
    params,
    max_records,
):

    records = []

    offset = 0
    page_size = 300

    while (
        len(records)
        <
        max_records
    ):

        query = dict(
            params
        )

        query[
            "limit"
        ] = min(
            page_size,
            max_records
            -
            len(records),
        )

        query[
            "offset"
        ] = offset

        payload = get_json(
            GBIF_API,
            query,
        )

        batch = payload.get(
            "results",
            [],
        )

        if not batch:
            break

        records.extend(
            batch
        )

        offset += len(
            batch
        )

        if payload.get(
            "endOfRecords",
            False,
        ):
            break

        if (
            len(batch)
            <
            query["limit"]
        ):
            break

        time.sleep(
            0.08
        )

    return records


# ============================================================
# GENERAL HELPERS
# ============================================================

def exact_date(
    record
):

    try:

        year = int(
            record.get(
                "year"
            )
        )

        month = int(
            record.get(
                "month"
            )
        )

        day = int(
            record.get(
                "day"
            )
        )

        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
        )

    except Exception:

        return pd.NaT


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    earth_radius = 6371.0088

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            delta_lat / 2
        ) ** 2
        +
        math.cos(
            p1
        )
        *
        math.cos(
            p2
        )
        *
        math.sin(
            delta_lon / 2
        ) ** 2
    )

    return (
        2
        *
        earth_radius
        *
        math.asin(
            math.sqrt(
                a
            )
        )
    )


def anchor_basis(
    row
):

    for column in [
        "basis_of_record",
        "basisOfRecord",
    ]:

        if (
            column in row.index
            and
            pd.notna(
                row[column]
            )
        ):

            value = str(
                row[column]
            ).strip()

            if value:
                return value

    return None


def candidate_key(
    row
):

    gbif_id = row.get(
        "gbif_id"
    )

    if pd.notna(
        gbif_id
    ):

        return (
            "gbif",
            str(
                gbif_id
            ),
        )

    return (
        "location",
        str(
            row["date"]
        ),
        round(
            float(
                row["lat"]
            ),
            4,
        ),
        round(
            float(
                row["lon"]
            ),
            4,
        ),
        str(
            row.get(
                "scientific_name",
                "",
            )
        ),
    )


# ============================================================
# GBIF RECORD → ROW
# ============================================================

def record_to_row(
    record,
    selection_tier,
    anchor_index,
):

    try:

        lat = float(
            record[
                "decimalLatitude"
            ]
        )

        lon = float(
            record[
                "decimalLongitude"
            ]
        )

    except Exception:

        return None

    date = exact_date(
        record
    )

    if pd.isna(
        date
    ):
        return None

    uncertainty = record.get(
        "coordinateUncertaintyInMeters"
    )

    try:

        uncertainty = (
            float(
                uncertainty
            )
            if uncertainty
            is not None
            else np.nan
        )

    except Exception:

        uncertainty = np.nan

    scientific_name = str(
        record.get(
            "scientificName"
        )
        or
        ""
    ).strip()

    accepted_name = str(
        record.get(
            "acceptedScientificName"
        )
        or
        scientific_name
    ).strip()

    return {
        "date":
            date,

        "lat":
            lat,

        "lon":
            lon,

        "gbif_id":
            record.get(
                "key"
            ),

        "occurrence_id":
            record.get(
                "occurrenceID"
            ),

        "scientific_name":
            scientific_name,

        "accepted_scientific_name":
            accepted_name,

        "species_key":
            record.get(
                "speciesKey"
            ),

        "taxon_key":
            record.get(
                "taxonKey"
            ),

        "family":
            record.get(
                "family"
            ),

        "basis_of_record":
            record.get(
                "basisOfRecord"
            ),

        "dataset_key":
            record.get(
                "datasetKey"
            ),

        "dataset_name":
            record.get(
                "datasetTitle"
            ),

        "coordinate_uncertainty_m":
            uncertainty,

        "selection_tier":
            selection_tier,

        "anchor_presence_row":
            int(
                anchor_index
            ),
    }


# ============================================================
# BASIC RECORD QC
# ============================================================

def basic_qc(
    row
):

    if row is None:
        return False

    lat = row[
        "lat"
    ]

    lon = row[
        "lon"
    ]

    if not (
        MIN_LAT
        <=
        lat
        <=
        MAX_LAT
    ):
        return False

    if not (
        MIN_LON
        <=
        lon
        <=
        MAX_LON
    ):
        return False

    if (
        row["date"]
        <
        pd.Timestamp(
            "1993-01-01"
        )
    ):
        return False

    uncertainty = row[
        "coordinate_uncertainty_m"
    ]

    if (
        pd.notna(
            uncertainty
        )
        and
        uncertainty
        >
        MAX_COORDINATE_UNCERTAINTY_M
    ):
        return False

    names = [
        str(
            row.get(
                "scientific_name",
                ""
            )
        ).lower(),

        str(
            row.get(
                "accepted_scientific_name",
                ""
            )
        ).lower(),
    ]

    # Exclude the target species itself.
    for name in names:

        if (
            name.startswith(
                "engraulis ringens"
            )
        ):
            return False

    return True


# ============================================================
# COASTLINE CACHE
# ============================================================

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
        round(
            south,
            2,
        ),
        round(
            north,
            2,
        ),
    )

    if key not in cache:

        print(
            f"  Loading coastline "
            f"{south:.1f} to "
            f"{north:.1f}"
        )

        cache[
            key
        ] = coastline_lines(
            MIN_LON,
            MAX_LON,
            south,
            north,
        )

    return cache[
        key
    ]


# ============================================================
# MARINE QC
# ============================================================

def marine_qc(
    row,
    presence_coordinates,
    coastline_cache,
    gebco_cache,
):

    lat = float(
        row[
            "lat"
        ]
    )

    lon = float(
        row[
            "lon"
        ]
    )

    try:

        coastlines = (
            coastline_for_latitude(
                lat,
                coastline_cache,
            )
        )

        if not coastlines:
            return None

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
            return None

        if (
            distance_coast_km
            >
            MAX_COAST_DISTANCE_KM
        ):
            return None

        location_key = (
            round(
                lat,
                4,
            ),
            round(
                lon,
                4,
            ),
        )

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
            return None

        nearest_presence = min(
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
            in presence_coordinates
        )

        if (
            nearest_presence
            <
            MIN_DISTANCE_FROM_PRESENCE_KM
        ):
            return None

    except Exception:

        return None

    output = dict(
        row
    )

    output[
        "background_distance_coast_km"
    ] = float(
        distance_coast_km
    )

    output[
        "background_distance_coast_nm"
    ] = float(
        distance_coast_km
        /
        KM_PER_NAUTICAL_MILE
    )

    output[
        "gebco_elevation_m"
    ] = elevation

    output[
        "gebco_depth_m"
    ] = depth

    output[
        "nearest_known_presence_km"
    ] = float(
        nearest_presence
    )

    return output


# ============================================================
# QUERY
# ============================================================

def query_parameters(
    basis,
    year,
    month=None,
):

    params = {
        "taxonKey":
            FISH_TAXON_KEY,

        "checklistKey":
            COL_XR_CHECKLIST_KEY,

        "geometry":
            COASTAL_GEOMETRY,

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
            int(
                year
            ),
    }

    if month is not None:

        params[
            "month"
        ] = int(
            month
        )

    return params


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing input: "
            f"{INPUT}"
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

    presence = (
        presence
        .dropna(
            subset=[
                "date",
                "lat",
                "lon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    target_total = (
        len(
            presence
        )
        *
        BACKGROUND_PER_PRESENCE
    )

    print(
        "E-AWARE V2 Step 4C2 — "
        "Target-Group Background"
    )

    print()

    print(
        "Anchoveta presences:",
        len(
            presence
        ),
    )

    print(
        "Requested backgrounds:",
        target_total,
    )

    print(
        "GBIF fish taxon key:",
        FISH_TAXON_KEY,
    )

    print(
        "Taxonomy: COL XR"
    )

    print()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    query_cache = {}

    coastline_cache = {}

    gebco_cache = {}

    used_candidates = set()

    accepted_rows = []

    total_records_seen = 0

    basic_qc_pass = 0

    missing_basis = 0

    presence_coordinates = (
        presence[
            [
                "lat",
                "lon",
            ]
        ]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Process each anchoveta presence separately.
    # --------------------------------------------------------

    for index, anchor in (
        presence.iterrows()
    ):

        anchor_number = (
            index
            +
            1
        )

        date = pd.Timestamp(
            anchor[
                "date"
            ]
        )

        basis = anchor_basis(
            anchor
        )

        if basis is None:

            missing_basis += 1

            print(
                f"[{anchor_number}/"
                f"{len(presence)}] "
                "no basisOfRecord — skipped"
            )

            continue

        selected_for_anchor = []

        tiers = [
            (
                "same_basis_month",
                query_parameters(
                    basis=basis,
                    year=date.year,
                    month=date.month,
                ),
                MONTH_QUERY_LIMIT,
            ),

            (
                "same_basis_year",
                query_parameters(
                    basis=basis,
                    year=date.year,
                    month=None,
                ),
                YEAR_QUERY_LIMIT,
            ),
        ]

        for (
            tier_name,
            params,
            max_records,
        ) in tiers:

            if (
                len(
                    selected_for_anchor
                )
                >=
                BACKGROUND_PER_PRESENCE
            ):
                break

            cache_key = tuple(
                sorted(
                    (
                        str(k),
                        str(v),
                    )
                    for k, v
                    in params.items()
                )
            )

            if (
                cache_key
                not in
                query_cache
            ):

                query_cache[
                    cache_key
                ] = fetch_records(
                    params,
                    max_records,
                )

            records = query_cache[
                cache_key
            ]

            if not records:
                continue

            order = rng.permutation(
                len(
                    records
                )
            )

            for record_index in order:

                if (
                    len(
                        selected_for_anchor
                    )
                    >=
                    BACKGROUND_PER_PRESENCE
                ):
                    break

                record = records[
                    int(
                        record_index
                    )
                ]

                total_records_seen += 1

                row = record_to_row(
                    record,
                    tier_name,
                    anchor_number,
                )

                if not basic_qc(
                    row
                ):
                    continue

                basic_qc_pass += 1

                key = candidate_key(
                    row
                )

                if (
                    key
                    in
                    used_candidates
                ):
                    continue

                marine_row = marine_qc(
                    row,
                    presence_coordinates,
                    coastline_cache,
                    gebco_cache,
                )

                if (
                    marine_row
                    is None
                ):
                    continue

                used_candidates.add(
                    key
                )

                selected_for_anchor.append(
                    marine_row
                )

                accepted_rows.append(
                    marine_row
                )

        print(
            f"[{anchor_number}/"
            f"{len(presence)}] "
            f"{basis} "
            f"{date.year}-"
            f"{date.month:02d} "
            f"→ "
            f"{len(selected_for_anchor)} "
            f"backgrounds "
            f"| total "
            f"{len(accepted_rows)}"
        )

    if not accepted_rows:

        raise RuntimeError(
            "No target-group backgrounds "
            "survived QC."
        )

    # --------------------------------------------------------
    # BACKGROUND DATAFRAME
    # --------------------------------------------------------

    background = pd.DataFrame(
        accepted_rows
    )

    background[
        "presence"
    ] = np.nan

    background[
        "pb_target"
    ] = 0

    # Keep sample_type compatible with
    # the current Step 4/5 pipeline.
    background[
        "sample_type"
    ] = "background"

    background[
        "background_type"
    ] = "target_group"

    background[
        "source"
    ] = "GBIF_target_group"

    background[
        "absence_semantics_verified"
    ] = False

    background = (
        background
        .sort_values(
            [
                "date",
                "lat",
                "lon",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    background.to_csv(
        BACKGROUND_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # PRESENCE DATAFRAME
    # --------------------------------------------------------

    presence_out = (
        presence.copy()
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

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

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
        combined[
            "date"
        ],
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
        .reset_index(
            drop=True
        )
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    tier_counts = (
        background[
            "selection_tier"
        ]
        .value_counts()
    )

    basis_counts = (
        background[
            "basis_of_record"
        ]
        .value_counts(
            dropna=False
        )
    )

    unique_datasets = (
        background[
            "dataset_key"
        ]
        .nunique(
            dropna=True
        )
    )

    anchors_with_background = (
        background[
            "anchor_presence_row"
        ]
        .nunique()
    )

    qc = pd.DataFrame(
        [
            [
                "Observed anchoveta presences",
                len(
                    presence
                ),
            ],
            [
                "Target backgrounds requested",
                target_total,
            ],
            [
                "Final target-group backgrounds",
                len(
                    background
                ),
            ],
            [
                "Anchoveta anchors with >=1 background",
                anchors_with_background,
            ],
            [
                "Anchoveta anchors missing basisOfRecord",
                missing_basis,
            ],
            [
                "GBIF candidate records inspected",
                total_records_seen,
            ],
            [
                "Candidates passing basic QC",
                basic_qc_pass,
            ],
            [
                "Same basis/month backgrounds",
                int(
                    tier_counts.get(
                        "same_basis_month",
                        0,
                    )
                ),
            ],
            [
                "Same basis/year backgrounds",
                int(
                    tier_counts.get(
                        "same_basis_year",
                        0,
                    )
                ),
            ],
            [
                "Unique background datasets",
                int(
                    unique_datasets
                ),
            ],
            [
                "Combined modelling rows",
                len(
                    combined
                ),
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

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print(
        "===== STEP 4C2 RESULTS ====="
    )

    print(
        "Observed anchoveta presences:",
        len(
            presence
        ),
    )

    print(
        "Target backgrounds requested:",
        target_total,
    )

    print(
        "Final target-group backgrounds:",
        len(
            background
        ),
    )

    print(
        "Anchoveta anchors represented:",
        anchors_with_background,
        "/",
        len(
            presence
        ),
    )

    print(
        "GBIF records inspected:",
        total_records_seen,
    )

    print(
        "Candidates passing basic QC:",
        basic_qc_pass,
    )

    print(
        "Unique background datasets:",
        unique_datasets,
    )

    print()

    print(
        "Background selection tiers:"
    )

    print(
        tier_counts.to_string()
    )

    print()

    print(
        "Basis of record:"
    )

    print(
        basis_counts.to_string()
    )

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
        "pb_target=0 is target-group "
        "background, NOT confirmed "
        "anchoveta absence."
    )


if __name__ == "__main__":
    main()
