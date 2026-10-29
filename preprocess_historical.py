#!/usr/bin/env python3
"""
E-AWARE V2 Step 4 — Offline Historical Preprocessing

Builds a reproducible historical anchoveta + environmental predictor table
outside Streamlit. This script does NOT train a machine-learning model.

Dry-run QC only:
    python preprocess_historical.py --dry-run

Small network smoke test:
    python preprocess_historical.py --max-groups 3

Full resumable run:
    python preprocess_historical.py

IMPORTANT:
Use --confirm-explicit-zero-is-absence only when the source documentation
confirms that 0 in an explicit presence/absence field is a genuine surveyed
absence/non-detection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr


MIN_LON, MAX_LON = -81.50, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36

HIST_PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
HIST_BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
HISTORICAL_COPERNICUS_START = pd.Timestamp("1993-01-01")

GEBCO_API_URL = "https://api.odb.ntu.edu.tw/gebco"
NOAA_COASTLINE_QUERY_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/EMAG2v3/MapServer/1/query"
)

OXYGEN_MMOL_M3_TO_ML_L = 0.0223916
OXYCLINE_THRESHOLD_ML_L = 1.0
KM_PER_NAUTICAL_MILE = 1.852

DEFAULT_INPUTS = [
    Path("data/anchoveta_observations.csv"),
    Path("data/imarpe_anchoveta_observations.csv"),
]

DEFAULT_OUTPUT = Path("data/processed/eaware_v2_step4")
DEFAULT_TILE_SIZE = 0.5
DEFAULT_MARGIN = 0.20

CORE_PREDICTORS = [
    "historical_sst_c",
    "historical_salinity_psu",
    "historical_surface_o2_ml_l",
    "historical_water_depth_m",
    "historical_distance_coast_km",
    "month_sin",
    "month_cos",
]

ALIASES = {
    "date": [
        "date",
        "datetime",
        "time",
        "fecha",
        "sample_date",
        "survey_date",
    ],
    "lat": [
        "lat",
        "latitude",
        "latitud",
        "y",
    ],
    "lon": [
        "lon",
        "longitude",
        "longitud",
        "lng",
        "x",
    ],
    "presence": [
        "presence",
        "present",
        "occurrence",
        "presence_absence",
        "pres_abs",
        "anchovy_presence",
        "anchoveta_presence",
        "fish_presence",
    ],
    "abundance": [
        "abundance",
        "anchovy_abundance",
        "anchoveta_abundance",
        "density",
        "acoustic_abundance",
        "sa",
        "nasc",
        "biomass_index",
    ],
    "survey_id": [
        "survey_id",
        "survey",
        "cruise_id",
        "cruise",
        "crucero",
        "campaign",
        "expedition",
    ],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_alias(columns, canonical: str):
    lookup = {
        norm(c): c
        for c in columns
    }

    for alias in ALIASES[canonical]:
        if alias in lookup:
            return lookup[alias]

    return None


def austral_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"

    if month in (3, 4, 5):
        return "Autumn"

    if month in (6, 7, 8):
        return "Winter"

    return "Spring"


def parse_presence(series: pd.Series) -> pd.Series:
    yes = {
        "1",
        "true",
        "yes",
        "y",
        "present",
        "presence",
        "si",
        "sí",
    }

    no = {
        "0",
        "false",
        "no",
        "n",
        "absent",
        "absence",
    }

    def one(v):
        if pd.isna(v):
            return np.nan

        if isinstance(
            v,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            x = float(v)

            if x == 1:
                return 1.0

            if x == 0:
                return 0.0

            return np.nan

        t = str(v).strip().lower()

        if t in yes:
            return 1.0

        if t in no:
            return 0.0

        return np.nan

    return series.apply(one).astype(float)


def standardise(
    raw: pd.DataFrame,
    confirm_explicit_zero: bool,
    confirm_abundance_zero: bool,
):
    mapping = {
        k: find_alias(
            raw.columns,
            k,
        )
        for k in ALIASES
    }

    missing = [
        k
        for k in (
            "date",
            "lat",
            "lon",
        )
        if mapping[k] is None
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(
        index=raw.index
    )

    out["date"] = pd.to_datetime(
        raw[
            mapping["date"]
        ],
        errors="coerce",
    )

    out["lat"] = pd.to_numeric(
        raw[
            mapping["lat"]
        ],
        errors="coerce",
    )

    out["lon"] = pd.to_numeric(
        raw[
            mapping["lon"]
        ],
        errors="coerce",
    )

    target_absence_verified = False
    target_method = (
        "No usable binary target"
    )

    if mapping["presence"] is not None:

        out["presence"] = parse_presence(
            raw[
                mapping["presence"]
            ]
        )

        has_zero = bool(
            (
                out["presence"]
                ==
                0
            ).any()
        )

        target_absence_verified = bool(
            has_zero
            and
            confirm_explicit_zero
        )

        target_method = (
            "Explicit binary presence/absence; "
            "absence semantics VERIFIED"
            if target_absence_verified
            else
            "Explicit binary presence/absence; "
            "absence semantics NOT verified"
        )

    elif mapping["abundance"] is not None:

        abundance = pd.to_numeric(
            raw[
                mapping["abundance"]
            ],
            errors="coerce",
        )

        out["abundance"] = abundance

        if confirm_abundance_zero:

            out["presence"] = np.where(
                abundance.notna(),
                (
                    abundance
                    >
                    0
                ).astype(float),
                np.nan,
            )

            target_absence_verified = bool(
                (
                    abundance
                    ==
                    0
                ).any()
            )

            target_method = (
                "Abundance-derived binary target; "
                "zero-as-absence VERIFIED"
            )

        else:

            out["presence"] = np.where(
                abundance > 0,
                1.0,
                np.nan,
            )

            target_method = (
                "Presence-only from positive abundance; "
                "zero left unknown"
            )

    else:
        out["presence"] = np.nan

    if "abundance" not in out.columns:

        if mapping["abundance"] is not None:

            out["abundance"] = pd.to_numeric(
                raw[
                    mapping["abundance"]
                ],
                errors="coerce",
            )

        else:
            out["abundance"] = np.nan

    if mapping["survey_id"] is not None:

        out["survey_id"] = (
            raw[
                mapping["survey_id"]
            ]
            .astype("string")
        )

    else:

        out["survey_id"] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )

    normalized = {
        norm(c): c
        for c in raw.columns
    }

    provenance = {
        "dataset_id": [
            "dataset_id",
            "datasetid",
        ],
        "dataset_name": [
            "dataset_name",
            "datasetname",
        ],
        "event_id": [
            "event_id",
            "eventid",
        ],
        "occurrence_id": [
            "occurrence_id",
            "occurrenceid",
        ],
        "basis_of_record": [
            "basis_of_record",
            "basisofrecord",
        ],
        "coordinate_uncertainty_m": [
            "coordinate_uncertainty_m",
            "coordinateuncertaintyinmeters",
        ],
        "obis_stage": [
            "obis_stage",
        ],
    }

    for target, options in provenance.items():

        source = next(
            (
                normalized[x]
                for x in options
                if x in normalized
            ),
            None,
        )

        if source is not None:
            out[target] = raw[source].values

    # Preserve presence-background modelling fields.
    # pb_target=0 is BACKGROUND, not confirmed biological absence.
    for passthrough in (
        "sample_type",
        "pb_target",
        "source",
    ):

        if passthrough in raw.columns:
            out[passthrough] = raw[passthrough].values

    out["source_row"] = np.arange(
        1,
        len(out) + 1,
    )

    out["valid_coordinates"] = (
        out["lat"].between(
            -90,
            90,
        )
        &
        out["lon"].between(
            -180,
            180,
        )
    )

    out["within_eaware_study_area"] = (
        out["lat"].between(
            MIN_LAT,
            MAX_LAT,
        )
        &
        out["lon"].between(
            MIN_LON,
            MAX_LON,
        )
    )

    out["within_peru_lat_band"] = (
        out["lat"].between(
            -20.5,
            0.5,
        )
    )

    dates = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    month = (
        dates
        .dt
        .month
        .astype("Int64")
    )

    angle = (
        2
        *
        np.pi
        *
        (
            month.astype(float)
            -
            1
        )
        /
        12
    )

    out["month_number"] = month

    out["year"] = (
        dates
        .dt
        .year
        .astype("Int64")
    )

    out["austral_season"] = (
        month.apply(
            lambda m:
                austral_season(
                    int(m)
                )
                if pd.notna(m)
                else None
        )
    )

    out["month_sin"] = np.sin(
        angle
    )

    out["month_cos"] = np.cos(
        angle
    )

    usable = out[
        out["date"].notna()
        &
        out["valid_coordinates"]
    ].copy().reset_index(
        drop=True
    )

    presence = int(
        (
            usable["presence"]
            ==
            1
        ).sum()
    )

    absence = int(
        (
            usable["presence"]
            ==
            0
        ).sum()
    )

    meta = {
        "raw_rows":
            int(
                len(out)
            ),

        "usable_rows":
            int(
                len(usable)
            ),

        "presence":
            presence,

        "absence":
            absence,

        "unknown_target":
            int(
                usable[
                    "presence"
                ]
                .isna()
                .sum()
            ),

        "inside_eaware":
            int(
                usable[
                    "within_eaware_study_area"
                ].sum()
            ),

        "pre1993":
            int(
                (
                    usable["date"]
                    <
                    HISTORICAL_COPERNICUS_START
                ).sum()
            ),

        "target_absence_verified":
            bool(
                target_absence_verified
            ),

        "target_method":
            target_method,

        "training_ready":
            bool(
                target_absence_verified
                and
                presence > 0
                and
                absence > 0
            ),

        "mapping": {
            k:
                (
                    str(v)
                    if v is not None
                    else None
                )
            for k, v
            in mapping.items()
        },
    }

    return usable, meta


def get_credentials():

    username = os.getenv(
        "COPERNICUS_USERNAME",
        "",
    ).strip()

    password = os.getenv(
        "COPERNICUS_PASSWORD",
        "",
    ).strip()

    if username and password:
        return username, password

    secrets = Path(
        ".streamlit/secrets.toml"
    )

    if secrets.exists():

        import tomllib

        data = tomllib.loads(
            secrets.read_text(
                encoding="utf-8"
            )
        )

        username = str(
            data.get(
                "COPERNICUS_USERNAME",
                "",
            )
        ).strip()

        password = str(
            data.get(
                "COPERNICUS_PASSWORD",
                "",
            )
        ).strip()

    if not username or not password:

        raise RuntimeError(
            "Copernicus credentials not found. "
            "Add COPERNICUS_USERNAME and "
            "COPERNICUS_PASSWORD to "
            ".streamlit/secrets.toml or "
            "environment variables."
        )

    return username, password


def validate_credentials():

    import copernicusmarine

    username, password = (
        get_credentials()
    )

    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"

    ok = copernicusmarine.login(
        username=username,
        password=password,
        check_credentials_valid=True,
    )

    if not ok:

        raise RuntimeError(
            "Copernicus rejected "
            "the supplied credentials."
        )

    return username, password


def subset_dataset(
    dataset_id,
    variables,
    date_string,
    bbox,
    username,
    password,
    depth=None,
):

    import copernicusmarine

    with tempfile.TemporaryDirectory() as temp_dir:

        args = dict(
            dataset_id=
                dataset_id,

            variables=
                list(
                    variables
                ),

            minimum_longitude=
                float(
                    bbox[0]
                ),

            maximum_longitude=
                float(
                    bbox[1]
                ),

            minimum_latitude=
                float(
                    bbox[2]
                ),

            maximum_latitude=
                float(
                    bbox[3]
                ),

            start_datetime=
                f"{date_string}T00:00:00",

            end_datetime=
                f"{date_string}T23:59:59",

            username=
                username,

            password=
                password,

            output_directory=
                temp_dir,

            output_filename=
                "subset.nc",

            disable_progress_bar=
                True,
        )

        if depth is not None:

            args[
                "minimum_depth"
            ] = depth[0]

            args[
                "maximum_depth"
            ] = depth[1]

        response = (
            copernicusmarine.subset(
                **args
            )
        )

        path = Path(
            str(
                response.file_path
            )
        )

        if not path.exists():

            candidates = list(
                Path(
                    temp_dir
                ).glob(
                    "*.nc"
                )
            )

            if not candidates:

                raise RuntimeError(
                    "Copernicus returned "
                    "no NetCDF file."
                )

            path = candidates[0]

        with xr.open_dataset(
            path
        ) as ds:

            return ds.load()


def normalize_xy(df):

    rename = {}

    for c in df.columns:

        lc = str(c).lower()

        if lc in (
            "latitude",
            "lat",
        ):
            rename[c] = "lat"

        elif lc in (
            "longitude",
            "lon",
        ):
            rename[c] = "lon"

    return df.rename(
        columns=rename
    )


def load_physics(
    date_string,
    bbox,
    username,
    password,
):

    ds = subset_dataset(
        HIST_PHY_DATASET,
        [
            "thetao",
            "so",
        ],
        date_string,
        bbox,
        username,
        password,
        depth=(
            0,
            1,
        ),
    )

    frames = []

    for var, name in (
        (
            "thetao",
            "historical_sst_c",
        ),
        (
            "so",
            "historical_salinity_psu",
        ),
    ):

        if var not in ds.data_vars:

            raise RuntimeError(
                f"Missing historical "
                f"physics variable: {var}"
            )

        da = ds[var]

        if "depth" in da.dims:

            da = da.isel(
                depth=0
            )

        f = normalize_xy(
            da.to_dataframe(
                name=name
            ).reset_index()
        )

        f["lat"] = pd.to_numeric(
            f["lat"],
            errors="coerce",
        )

        f["lon"] = pd.to_numeric(
            f["lon"],
            errors="coerce",
        )

        f = f.dropna(
            subset=[
                "lat",
                "lon",
                name,
            ]
        )

        frames.append(
            f[
                [
                    "lat",
                    "lon",
                    name,
                ]
            ].copy()
        )

    temp, sal = frames

    temp["lat_key"] = (
        temp["lat"]
        .round(4)
    )

    temp["lon_key"] = (
        temp["lon"]
        .round(4)
    )

    sal["lat_key"] = (
        sal["lat"]
        .round(4)
    )

    sal["lon_key"] = (
        sal["lon"]
        .round(4)
    )

    return (
        temp.merge(
            sal[
                [
                    "lat_key",
                    "lon_key",
                    "historical_salinity_psu",
                ]
            ],
            on=[
                "lat_key",
                "lon_key",
            ],
            how="inner",
        )
        [
            [
                "lat",
                "lon",
                "historical_sst_c",
                "historical_salinity_psu",
            ]
        ]
    )


def load_bgc(
    date_string,
    bbox,
    username,
    password,
):

    ds = subset_dataset(
        HIST_BGC_DATASET,
        [
            "o2",
        ],
        date_string,
        bbox,
        username,
        password,
    )

    if "o2" not in ds.data_vars:

        raise RuntimeError(
            "Missing historical "
            "BGC variable: o2"
        )

    df = normalize_xy(
        ds["o2"]
        .to_dataframe(
            name="oxygen_mmol_m3"
        )
        .reset_index()
    )

    for c in (
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3",
    ):

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "depth",
            "lat",
            "lon",
            "oxygen_mmol_m3",
        ]
    ).copy()

    df["oxygen_ml_l"] = (
        df[
            "oxygen_mmol_m3"
        ]
        *
        OXYGEN_MMOL_M3_TO_ML_L
    )

    return df[
        [
            "depth",
            "lat",
            "lon",
            "oxygen_mmol_m3",
            "oxygen_ml_l",
        ]
    ]


def nearest(
    target,
    source,
):

    tlat = target[
        "lat"
    ].to_numpy(
        float
    )

    tlon = target[
        "lon"
    ].to_numpy(
        float
    )

    slat = source[
        "lat"
    ].to_numpy(
        float
    )

    slon = source[
        "lon"
    ].to_numpy(
        float
    )

    dlat = (
        tlat[:, None]
        -
        slat[None, :]
    ) * 111.32

    mean_lat = np.radians(
        (
            tlat[:, None]
            +
            slat[None, :]
        )
        /
        2
    )

    dlon = (
        tlon[:, None]
        -
        slon[None, :]
    ) * (
        111.32
        *
        np.cos(
            mean_lat
        )
    )

    dist = np.sqrt(
        dlat ** 2
        +
        dlon ** 2
    )

    idx = np.argmin(
        dist,
        axis=1,
    )

    return (
        idx,
        dist[
            np.arange(
                len(target)
            ),
            idx,
        ],
    )


def oxycline(
    profile
):

    clean = (
        profile[
            [
                "depth",
                "oxygen_ml_l",
            ]
        ]
        .dropna()
        .sort_values(
            "depth"
        )
        .drop_duplicates(
            "depth"
        )
    )

    if clean.empty:

        return (
            np.nan,
            "No valid oxygen profile",
        )

    z = clean[
        "depth"
    ].to_numpy(
        float
    )

    o = clean[
        "oxygen_ml_l"
    ].to_numpy(
        float
    )

    if (
        o[0]
        <=
        OXYCLINE_THRESHOLD_ML_L
    ):

        return (
            float(
                z[0]
            ),
            "Threshold reached at "
            "shallowest model level",
        )

    for i in range(
        1,
        len(z),
    ):

        if (
            o[i - 1]
            >
            OXYCLINE_THRESHOLD_ML_L
            and
            o[i]
            <=
            OXYCLINE_THRESHOLD_ML_L
        ):

            if (
                o[i]
                ==
                o[i - 1]
            ):

                return (
                    float(
                        z[i]
                    ),
                    "Threshold crossing",
                )

            depth = (
                z[i - 1]
                +
                (
                    (
                        OXYCLINE_THRESHOLD_ML_L
                        -
                        o[i - 1]
                    )
                    /
                    (
                        o[i]
                        -
                        o[i - 1]
                    )
                )
                *
                (
                    z[i]
                    -
                    z[i - 1]
                )
            )

            return (
                float(
                    depth
                ),
                "Interpolated "
                "1 mL/L crossing",
            )

    return (
        np.nan,
        "1 mL/L threshold "
        "not reached in available profile",
    )


def request_json(
    url,
    params=None,
    timeout=45,
    attempts=3,
):

    last = None

    for attempt in range(
        attempts
    ):

        try:

            r = requests.get(
                url,
                params=params,
                headers={
                    "Accept":
                        "application/json",

                    "User-Agent":
                        "E-AWARE Step4 "
                        "academic prototype",
                },
                timeout=timeout,
            )

            r.raise_for_status()

            return r.json()

        except Exception as exc:

            last = exc

            if attempt + 1 < attempts:

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
        f"Request failed: {last}"
    )


def first_number(
    value
):

    if value is None:
        return None

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):

        v = float(
            value
        )

        return (
            v
            if math.isfinite(v)
            else None
        )

    if isinstance(
        value,
        str,
    ):

        try:
            return float(
                value
            )

        except ValueError:
            return None

    if isinstance(
        value,
        dict,
    ):

        if "z" in value:

            return first_number(
                value["z"]
            )

        for k in (
            "data",
            "result",
            "results",
            "records",
        ):

            if k in value:

                x = first_number(
                    value[k]
                )

                if x is not None:
                    return x

    if isinstance(
        value,
        list,
    ):

        for item in value:

            x = first_number(
                item
            )

            if x is not None:
                return x

    return None


def gebco(
    lat,
    lon,
):

    payload = request_json(
        GEBCO_API_URL,
        params={
            "lon":
                f"{lon:.6f}",

            "lat":
                f"{lat:.6f}",

            "mode":
                "point",
        },
        timeout=30,
    )

    value = first_number(
        payload.get(
            "z"
        )
        if isinstance(
            payload,
            dict,
        )
        else payload
    )

    if value is None:

        return (
            np.nan,
            np.nan,
            False,
        )

    is_ocean = (
        value
        <
        0
    )

    depth = (
        -value
        if is_ocean
        else np.nan
    )

    return (
        float(
            value
        ),
        (
            float(
                depth
            )
            if is_ocean
            else np.nan
        ),
        bool(
            is_ocean
        ),
    )


def coastline_lines(
    west,
    east,
    south,
    north,
):

    common = {
        "where":
            "1=1",

        "geometry":
            f"{west},{south},{east},{north}",

        "geometryType":
            "esriGeometryEnvelope",

        "inSR":
            "4326",

        "spatialRel":
            "esriSpatialRelIntersects",

        "outFields":
            "OBJECTID",

        "returnGeometry":
            "true",

        "outSR":
            "4326",

        "f":
            "geojson",
    }

    payload = request_json(
        NOAA_COASTLINE_QUERY_URL,
        params=common,
    )

    lines = []

    for feature in payload.get(
        "features",
        [],
    ):

        geom = (
            feature.get(
                "geometry"
            )
            or
            {}
        )

        if (
            geom.get(
                "type"
            )
            ==
            "LineString"
        ):

            lines.append(
                geom.get(
                    "coordinates",
                    [],
                )
            )

        elif (
            geom.get(
                "type"
            )
            ==
            "MultiLineString"
        ):

            lines.extend(
                geom.get(
                    "coordinates",
                    [],
                )
            )

    return lines


def segment_distance_km(
    lat,
    lon,
    lon1,
    lat1,
    lon2,
    lat2,
):

    ky = 111.32

    kx = (
        111.32
        *
        math.cos(
            math.radians(
                lat
            )
        )
    )

    x1 = (
        lon1
        -
        lon
    ) * kx

    y1 = (
        lat1
        -
        lat
    ) * ky

    x2 = (
        lon2
        -
        lon
    ) * kx

    y2 = (
        lat2
        -
        lat
    ) * ky

    dx = (
        x2
        -
        x1
    )

    dy = (
        y2
        -
        y1
    )

    denom = (
        dx
        *
        dx
        +
        dy
        *
        dy
    )

    if denom == 0:

        return math.hypot(
            x1,
            y1,
        )

    t = max(
        0.0,
        min(
            1.0,
            -
            (
                x1 * dx
                +
                y1 * dy
            )
            /
            denom,
        ),
    )

    return math.hypot(
        x1
        +
        t * dx,
        y1
        +
        t * dy,
    )


def coast_distance(
    lat,
    lon,
    lines,
):

    best = math.inf

    for line in lines:

        for i in range(
            len(line) - 1
        ):

            p1 = line[i]
            p2 = line[i + 1]

            if (
                len(p1) >= 2
                and
                len(p2) >= 2
            ):

                best = min(
                    best,
                    segment_distance_km(
                        lat,
                        lon,
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                    ),
                )

    return (
        float(
            best
        )
        if math.isfinite(
            best
        )
        else np.nan
    )


def tile_key(
    lat,
    lon,
    size,
):

    return (
        math.floor(
            lat / size
        )
        *
        size,

        math.floor(
            lon / size
        )
        *
        size,
    )


def safe_bbox(
    group,
    margin,
):

    return (
        max(
            -179.99,
            float(
                group["lon"].min()
            )
            -
            margin,
        ),

        min(
            179.99,
            float(
                group["lon"].max()
            )
            +
            margin,
        ),

        max(
            -79.9,
            float(
                group["lat"].min()
            )
            -
            margin,
        ),

        min(
            89.9,
            float(
                group["lat"].max()
            )
            +
            margin,
        ),
    )


def empty_environment(
    df
):

    out = df.copy()

    defaults = {
        "match_status":
            "Pending",

        "historical_sst_c":
            np.nan,

        "historical_salinity_psu":
            np.nan,

        "historical_surface_o2_ml_l":
            np.nan,

        "historical_raw_oxycline_m":
            np.nan,

        "historical_oxycline_m":
            np.nan,

        "historical_oxycline_status":
            None,

        "gebco_elevation_m":
            np.nan,

        "historical_water_depth_m":
            np.nan,

        "is_ocean_gebco":
            False,

        "historical_distance_coast_km":
            np.nan,

        "historical_distance_coast_nm":
            np.nan,

        "physics_match_distance_km":
            np.nan,

        "bgc_match_distance_km":
            np.nan,
    }

    for c, v in defaults.items():

        if c not in out.columns:
            out[c] = v

    return out


def match_group(
    group,
    date_string,
    username,
    password,
    margin,
):

    out = empty_environment(
        group.reset_index(
            drop=True
        )
    )

    try:

        lines = coastline_lines(
            float(
                out["lon"].min()
            )
            -
            0.8,

            float(
                out["lon"].max()
            )
            +
            0.8,

            float(
                out["lat"].min()
            )
            -
            0.8,

            float(
                out["lat"].max()
            )
            +
            0.8,
        )

    except Exception:
        lines = []

    for idx, row in out.iterrows():

        try:

            elev, depth, is_ocean = gebco(
                float(
                    row["lat"]
                ),
                float(
                    row["lon"]
                ),
            )

            out.at[
                idx,
                "gebco_elevation_m",
            ] = elev

            out.at[
                idx,
                "historical_water_depth_m",
            ] = depth

            out.at[
                idx,
                "is_ocean_gebco",
            ] = is_ocean

        except Exception:
            pass

        if lines:

            d = coast_distance(
                float(
                    row["lat"]
                ),
                float(
                    row["lon"]
                ),
                lines,
            )

            out.at[
                idx,
                "historical_distance_coast_km",
            ] = d

            if pd.notna(d):

                out.at[
                    idx,
                    "historical_distance_coast_nm",
                ] = (
                    d
                    /
                    KM_PER_NAUTICAL_MILE
                )

    pre1993 = (
        pd.to_datetime(
            out["date"]
        )
        <
        HISTORICAL_COPERNICUS_START
    )

    out.loc[
        pre1993,
        "match_status",
    ] = (
        "No Copernicus match — "
        "date before 1993-01-01"
    )

    # GEBCO is used as a bathymetry/depth estimate only.
    #
    # Do NOT reject a fish observation from Copernicus matching simply
    # because the exact GEBCO raster cell returns a small positive
    # elevation. Nearshore occurrence coordinates can fall into a
    # coastal/land raster cell even though the observation represents
    # a real marine record.
    #
    # This mirrors the working Streamlit Step 2 behaviour:
    # - retain the observation for Copernicus physics/BGC matching;
    # - leave historical_water_depth_m missing when GEBCO is non-ocean;
    # - report the missing depth later through QC;
    # - oxycline remains unconstrained by local depth when depth is absent.

    target_mask = ~pre1993

    targets = out.loc[
        target_mask
    ].copy()

    if targets.empty:
        return out

    bbox = safe_bbox(
        targets,
        margin,
    )

    try:

        physics = load_physics(
            date_string,
            bbox,
            username,
            password,
        )

        bgc = load_bgc(
            date_string,
            bbox,
            username,
            password,
        )

        # A coastal or inland observation can produce a valid Copernicus
        # download but no valid ocean cells after land/missing-value masking.
        # Do not call nearest-neighbour matching on an empty grid and do not
        # force the observation onto a distant ocean cell.
        if physics.empty or bgc.empty:

            out.loc[
                target_mask,
                "match_status",
            ] = (
                "No historical environmental grid match"
            )

            return out

    except Exception as exc:

        out.loc[
            target_mask,
            "match_status",
        ] = (
            "Copernicus request failed: "
            f"{str(exc)[:160]}"
        )

        return out

    pidx, pdist = nearest(
        targets,
        physics[
            [
                "lat",
                "lon",
            ]
        ],
    )

    bgc_grid = (
        bgc[
            [
                "lat",
                "lon",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    bidx, bdist = nearest(
        targets,
        bgc_grid,
    )

    nearest_p = (
        physics
        .iloc[pidx]
        .reset_index(
            drop=True
        )
    )

    nearest_b = (
        bgc_grid
        .iloc[bidx]
        .reset_index(
            drop=True
        )
    )

    for pos, idx in enumerate(
        targets.index.to_list()
    ):

        p = nearest_p.iloc[
            pos
        ]

        out.at[
            idx,
            "historical_sst_c",
        ] = float(
            p[
                "historical_sst_c"
            ]
        )

        out.at[
            idx,
            "historical_salinity_psu",
        ] = float(
            p[
                "historical_salinity_psu"
            ]
        )

        out.at[
            idx,
            "physics_match_distance_km",
        ] = float(
            pdist[pos]
        )

        out.at[
            idx,
            "bgc_match_distance_km",
        ] = float(
            bdist[pos]
        )

        slat = float(
            nearest_b.iloc[
                pos
            ][
                "lat"
            ]
        )

        slon = float(
            nearest_b.iloc[
                pos
            ][
                "lon"
            ]
        )

        profile = (
            bgc[
                (
                    bgc["lat"]
                    ==
                    slat
                )
                &
                (
                    bgc["lon"]
                    ==
                    slon
                )
            ]
            .sort_values(
                "depth"
            )
        )

        if profile.empty:

            out.at[
                idx,
                "match_status",
            ] = (
                "No BGC oxygen profile"
            )

            continue

        out.at[
            idx,
            "historical_surface_o2_ml_l",
        ] = float(
            profile.iloc[
                0
            ][
                "oxygen_ml_l"
            ]
        )

        raw_depth, _ = oxycline(
            profile
        )

        out.at[
            idx,
            "historical_raw_oxycline_m",
        ] = raw_depth

        local_depth = out.at[
            idx,
            "historical_water_depth_m",
        ]

        if pd.notna(
            local_depth
        ):

            valid = profile[
                profile[
                    "depth"
                ]
                <=
                float(
                    local_depth
                )
            ]

            oxy_depth, oxy_status = oxycline(
                valid
            )

            out.at[
                idx,
                "historical_oxycline_m",
            ] = oxy_depth

            if pd.notna(
                oxy_depth
            ):

                out.at[
                    idx,
                    "historical_oxycline_status",
                ] = (
                    "Accepted within "
                    "local water column"
                )

            elif (
                pd.notna(
                    raw_depth
                )
                and
                raw_depth
                >
                float(
                    local_depth
                )
            ):

                out.at[
                    idx,
                    "historical_oxycline_status",
                ] = (
                    "Raw crossing below seabed; "
                    "local threshold not reached"
                )

            else:

                out.at[
                    idx,
                    "historical_oxycline_status",
                ] = oxy_status

        else:

            out.at[
                idx,
                "historical_oxycline_status",
            ] = (
                "No valid GEBCO water depth"
            )

        out.at[
            idx,
            "match_status",
        ] = "Matched"

    return out


def input_path(
    requested
):

    if requested:
        return Path(
            requested
        )

    for path in DEFAULT_INPUTS:

        if path.exists():
            return path

    raise FileNotFoundError(
        "No observation CSV found. Tried: "
        +
        ", ".join(
            str(x)
            for x
            in DEFAULT_INPUTS
        )
    )


def sha256(
    path
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024
                    *
                    1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def write_json(
    path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def qc_outputs(
    df,
    meta,
    output,
):

    qc = (
        output
        /
        "qc"
    )

    qc.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_keys = [
        "date",
        "lat",
        "lon",
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_keys,
        keep=False,
    )

    summary = pd.DataFrame(
        [
            [
                "Raw rows",
                meta["raw_rows"],
            ],
            [
                "Usable rows",
                meta["usable_rows"],
            ],
            [
                "Presence",
                meta["presence"],
            ],
            [
                "Absence",
                meta["absence"],
            ],
            [
                "Unknown target",
                meta["unknown_target"],
            ],
            [
                "Inside E-AWARE area",
                meta["inside_eaware"],
            ],
            [
                "Before 1993",
                meta["pre1993"],
            ],
            [
                "Duplicates flagged",
                int(
                    duplicate_mask.sum()
                ),
            ],
            [
                "Absence semantics verified",
                meta[
                    "target_absence_verified"
                ],
            ],
            [
                "Training-ready target",
                meta[
                    "training_ready"
                ],
            ],
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    summary.to_csv(
        qc
        /
        "observation_qc_summary.csv",
        index=False,
    )

    df.loc[
        duplicate_mask
    ].to_csv(
        qc
        /
        "duplicate_observations.csv",
        index=False,
    )

    write_json(
        qc
        /
        "ingestion_metadata.json",
        meta,
    )


def final_qc(
    matched,
    meta,
    output,
):

    matched_mask = (
        matched[
            "match_status"
        ].eq(
            "Matched"
        )
    )

    binary = (
        matched[
            "presence"
        ]
        .isin(
            [
                0.0,
                1.0,
            ]
        )
    )

    complete = (
        matched[
            CORE_PREDICTORS
        ]
        .notna()
        .all(
            axis=1
        )
    )

    review = matched.loc[
        matched_mask
        &
        binary
        &
        complete
    ].copy()

    rp = int(
        (
            review[
                "presence"
            ]
            ==
            1
        ).sum()
    )

    ra = int(
        (
            review[
                "presence"
            ]
            ==
            0
        ).sum()
    )

    ready = bool(
        meta[
            "target_absence_verified"
        ]
        and
        rp > 0
        and
        ra > 0
    )

    candidate = (
        review.copy()
        if ready
        else
        review.iloc[
            0:0
        ].copy()
    )

    review.to_csv(
        output
        /
        "eaware_v2_step4_environment_complete_binary_review.csv",
        index=False,
    )

    candidate.to_csv(
        output
        /
        "eaware_v2_step4_core_training_candidate.csv",
        index=False,
    )

    rows = []

    for col in (
        CORE_PREDICTORS
        +
        [
            "historical_oxycline_m"
        ]
    ):

        good = int(
            matched[
                col
            ]
            .notna()
            .sum()
        )

        rows.append(
            {
                "Predictor":
                    col,

                "Non_missing":
                    good,

                "Missing":
                    int(
                        len(
                            matched
                        )
                        -
                        good
                    ),

                "Completeness_percent":
                    (
                        100
                        *
                        good
                        /
                        len(
                            matched
                        )
                        if len(
                            matched
                        )
                        else np.nan
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        output
        /
        "qc"
        /
        "predictor_completeness.csv",
        index=False,
    )

    return {
        "copernicus_matched":
            int(
                matched_mask.sum()
            ),

        "environment_complete_binary_review_rows":
            int(
                len(
                    review
                )
            ),

        "training_target_ready":
            ready,

        "core_candidate_rows":
            int(
                len(
                    candidate
                )
            ),
    }


def parse_args():

    p = argparse.ArgumentParser(
        description=
            "E-AWARE V2 Step 4 "
            "offline preprocessing"
    )

    p.add_argument(
        "--input",
        default=None,
    )

    p.add_argument(
        "--output-dir",
        default=
            str(
                DEFAULT_OUTPUT
            ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--max-groups",
        type=int,
        default=None,
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
    )

    p.add_argument(
        "--tile-size",
        type=float,
        default=
            DEFAULT_TILE_SIZE,
    )

    p.add_argument(
        "--bbox-margin",
        type=float,
        default=
            DEFAULT_MARGIN,
    )

    p.add_argument(
        "--reset-run",
        action="store_true",
    )

    p.add_argument(
        "--confirm-explicit-zero-is-absence",
        action="store_true",
    )

    p.add_argument(
        "--confirm-zero-abundance-is-absence",
        action="store_true",
    )

    return p.parse_args()


def main():

    args = parse_args()

    source = input_path(
        args.input
    ).resolve()

    output = Path(
        args.output_dir
    ).resolve()

    parts = (
        output
        /
        "parts"
    )

    if (
        args.reset_run
        and
        output.exists()
    ):

        import shutil

        shutil.rmtree(
            output
        )

    parts.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output
        /
        "qc"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "E-AWARE V2 Step 4 — "
        "Offline Historical Preprocessing"
    )

    log(
        f"Input: {source}"
    )

    log(
        f"Output: {output}"
    )

    raw = pd.read_csv(
        source,
        low_memory=False,
    )

    historical, meta = standardise(
        raw,
        args.confirm_explicit_zero_is_absence,
        args.confirm_zero_abundance_is_absence,
    )

    if (
        args.max_rows
        is not None
    ):

        historical = (
            historical
            .head(
                max(
                    0,
                    args.max_rows,
                )
            )
            .copy()
        )

    historical.to_csv(
        output
        /
        "standardised_observations.csv",
        index=False,
    )

    qc_outputs(
        historical,
        meta,
        output,
    )

    log(
        f"Usable observations: "
        f"{len(historical)} | "
        f"Presence: "
        f"{(historical['presence'] == 1).sum()} | "
        f"Absence: "
        f"{(historical['presence'] == 0).sum()} | "
        f"Unknown: "
        f"{historical['presence'].isna().sum()}"
    )

    log(
        f"Target method: "
        f"{meta['target_method']}"
    )

    manifest = {
        "created_utc":
            now_iso(),

        "input":
            str(
                source
            ),

        "input_sha256":
            sha256(
                source
            ),

        "usable_rows":
            int(
                len(
                    historical
                )
            ),

        "dry_run":
            bool(
                args.dry_run
            ),

        "target_method":
            meta[
                "target_method"
            ],

        "target_absence_verified":
            meta[
                "target_absence_verified"
            ],
    }

    if args.dry_run:

        write_json(
            output
            /
            "run_manifest.json",
            manifest,
        )

        log(
            "Dry-run complete. "
            "No environmental APIs were called."
        )

        return

    username, password = (
        validate_credentials()
    )

    log(
        "Copernicus credentials validated."
    )

    if historical.empty:

        raise RuntimeError(
            "No usable observations "
            "remain after validation."
        )

    work = historical.copy()

    work["match_date"] = (
        pd.to_datetime(
            work["date"]
        )
        .dt
        .strftime(
            "%Y-%m-%d"
        )
    )

    work[
        [
            "tile_lat",
            "tile_lon",
        ]
    ] = work.apply(
        lambda r:
            pd.Series(
                tile_key(
                    float(
                        r["lat"]
                    ),
                    float(
                        r["lon"]
                    ),
                    args.tile_size,
                )
            ),
        axis=1,
    )

    groups = list(
        work.groupby(
            [
                "match_date",
                "tile_lat",
                "tile_lon",
            ],
            sort=True,
        )
    )

    selected = (
        groups[
            :args.max_groups
        ]
        if args.max_groups
        is not None
        else groups
    )

    log(
        f"Date/tile groups selected: "
        f"{len(selected)} "
        f"of {len(groups)}"
    )

    for n, (
        (
            date_string,
            tile_lat,
            tile_lon,
        ),
        group,
    ) in enumerate(
        selected,
        start=1,
    ):

        key = (
            f"{date_string}|"
            f"{tile_lat:.4f}|"
            f"{tile_lon:.4f}"
        )

        name = hashlib.sha1(
            key.encode()
        ).hexdigest()[
            :12
        ]

        part = (
            parts
            /
            f"{date_string}_{name}.csv"
        )

        if part.exists():

            log(
                f"[{n}/{len(selected)}] "
                f"Resume skip: "
                f"{date_string} "
                f"tile "
                f"{tile_lat},{tile_lon}"
            )

            continue

        log(
            f"[{n}/{len(selected)}] "
            f"Matching "
            f"{len(group)} row(s): "
            f"{date_string} "
            f"tile "
            f"{tile_lat},{tile_lon}"
        )

        base = group.drop(
            columns=[
                "match_date",
                "tile_lat",
                "tile_lon",
            ]
        )

        try:

            matched = match_group(
                base,
                str(
                    date_string
                ),
                username,
                password,
                args.bbox_margin,
            )

        except Exception as exc:

            matched = empty_environment(
                base.reset_index(
                    drop=True
                )
            )

            matched[
                "match_status"
            ] = (
                "Step 4 group failed: "
                f"{str(exc)[:160]}"
            )

        matched.to_csv(
            part,
            index=False,
        )

    part_files = sorted(
        parts.glob(
            "*.csv"
        )
    )

    if not part_files:

        raise RuntimeError(
            "No checkpoint part files "
            "were produced."
        )

    assembled = pd.concat(
        [
            pd.read_csv(
                x,
                parse_dates=[
                    "date"
                ],
            )
            for x
            in part_files
        ],
        ignore_index=True,
    )

    assembled = (
        assembled
        .sort_values(
            "source_row"
        )
        .drop_duplicates(
            "source_row",
            keep="last",
        )
    )

    done = set(
        pd.to_numeric(
            assembled[
                "source_row"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    remaining = historical[
        ~historical[
            "source_row"
        ].isin(
            done
        )
    ].copy()

    if not remaining.empty:

        remaining = empty_environment(
            remaining
        )

        remaining[
            "match_status"
        ] = (
            "Not processed in this run"
        )

        final = pd.concat(
            [
                assembled,
                remaining,
            ],
            ignore_index=True,
        )

    else:
        final = assembled

    final = (
        final
        .sort_values(
            "source_row"
        )
        .reset_index(
            drop=True
        )
    )

    complete = (
        len(done)
        ==
        len(historical)
    )

    filename = (
        "eaware_v2_step4_"
        "matched_historical_environment.csv"
        if complete
        else
        "eaware_v2_step4_"
        "matched_partial.csv"
    )

    final.to_csv(
        output
        /
        filename,
        index=False,
    )

    stats = final_qc(
        final,
        meta,
        output,
    )

    manifest.update(
        stats
    )

    manifest.update(
        {
            "updated_utc":
                now_iso(),

            "processed_rows":
                len(done),

            "complete":
                complete,

            "final_table":
                filename,
        }
    )

    write_json(
        output
        /
        "run_manifest.json",
        manifest,
    )

    log(
        "Step 4 assembly complete."
    )

    log(
        f"Copernicus matched: "
        f"{stats['copernicus_matched']}"
    )

    log(
        "Environment-complete binary "
        "rows for review: "
        f"{stats['environment_complete_binary_review_rows']}"
    )

    log(
        f"Training target ready: "
        f"{stats['training_target_ready']}"
    )

    log(
        f"Core training-candidate rows: "
        f"{stats['core_candidate_rows']}"
    )

    log(
        "No machine-learning model was trained."
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        log(
            "Interrupted. Completed checkpoint "
            "files remain available for resume."
        )

        raise SystemExit(
            130
        )

    except Exception as exc:

        log(
            f"ERROR: {exc}"
        )

        raise
