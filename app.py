import os
import math
import json
import tempfile
from pathlib import Path
from datetime import date, timedelta, datetime

import requests
import copernicusmarine
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import xarray as xr

# ============================================================
# PAGE / STUDY AREA
# ============================================================
st.set_page_config(page_title="E-AWARE", page_icon="🐟", layout="wide")
st.title("🐟 E-AWARE")
st.caption(
    "El Niño Adaptive Fisheries Intelligence — "
    "Primary Literature-Based HSI + Experimental ML"
)

st.info(
    "E-AWARE uses the literature-based Habitat Suitability Index (HSI) "
    "as its PRIMARY habitat model. "
    "A coast-matched conditional logistic model is retained as an "
    "EXPERIMENTAL secondary research layer. "
    "The experimental ML score is relative habitat suitability, "
    "not probability of fish presence."
)

MIN_LON, MAX_LON = -82.15, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36
MAX_OPERATIONAL_DISTANCE_NM = 50.0

COMMUNITIES = pd.DataFrame([
    {"name": "Lobitos", "lat": -4.4567, "lon": -81.2849},
    {"name": "Siches", "lat": -4.4890, "lon": -81.2680},
    {"name": "Piedritas", "lat": -4.5190, "lon": -81.2630},
])

# ============================================================
# DATA SOURCES / CONSTANTS
# ============================================================
LIVE_TEMP_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
LIVE_SAL_DATASET = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
LIVE_SSTA_DATASET = "cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m"
LIVE_BGC_DATASET = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"

HIST_PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
HIST_BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
HISTORICAL_COPERNICUS_START = pd.Timestamp("1993-01-01")

GEBCO_API_URL = "https://api.odb.ntu.edu.tw/gebco"
NOAA_COASTLINE_QUERY_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/EMAG2v3/MapServer/1/query"
)
BASEMAP_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

COASTLINE_QUERY_MARGIN_DEG = 0.75
KM_PER_NAUTICAL_MILE = 1.852
OXYGEN_MMOL_M3_TO_ML_L = 0.0223916
OXYCLINE_THRESHOLD_ML_L = 1.0
OBSERVATION_REPO_PATH = Path("data/anchoveta_observations.csv")
LEGACY_OBSERVATION_REPO_PATH = Path("data/imarpe_anchoveta_observations.csv")

EXPERIMENTAL_MODEL_MANIFEST = Path(
    "model_artifacts/eaware_experimental_ml.json"
)

# Engineering QC warning thresholds for nearest-grid matching.
# These are diagnostic thresholds, not biological habitat thresholds.
QC_MAX_PHYSICS_MATCH_KM = 15.0
QC_MAX_BGC_MATCH_KM = 25.0

# ============================================================
# GENERAL HELPERS
# ============================================================

@st.cache_data
def load_experimental_model_manifest():

    if not EXPERIMENTAL_MODEL_MANIFEST.exists():
        return None

    try:
        return json.loads(
            EXPERIMENTAL_MODEL_MANIFEST.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return None


def get_credentials():
    username = st.secrets.get("COPERNICUS_USERNAME", "")
    password = st.secrets.get("COPERNICUS_PASSWORD", "")

    if not username or not password:
        raise RuntimeError(
            "Copernicus credentials are missing from Streamlit Secrets."
        )

    return username, password


def resolve_downloaded_file(response, temp_dir, filename):
    path = Path(str(response.file_path))

    if path.exists():
        return path

    expected = Path(temp_dir) / filename

    if expected.exists():
        return expected

    candidates = list(Path(temp_dir).glob("*.nc"))

    if not candidates:
        raise RuntimeError(
            "Copernicus download completed but no NetCDF file was found."
        )

    return candidates[0]


def normalize_xy_columns(df):
    rename = {}

    for c in df.columns:
        lc = str(c).lower()

        if lc in ("latitude", "lat"):
            rename[c] = "lat"

        elif lc in ("longitude", "lon"):
            rename[c] = "lon"

    return df.rename(columns=rename)


def prepare_surface_dataframe(df, variable_name):
    df = normalize_xy_columns(df.copy())

    required = [
        "time",
        "lat",
        "lon",
        variable_name
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns for {variable_name}: {missing}"
        )

    df = df.dropna(
        subset=[variable_name]
    ).copy()

    df["date"] = pd.to_datetime(
        df["time"]
    ).dt.date

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    df["lat_key"] = df["lat"].round(3)
    df["lon_key"] = df["lon"].round(3)

    return df[
        [
            "date",
            "lat",
            "lon",
            "lat_key",
            "lon_key",
            variable_name
        ]
    ].dropna()


def copernicus_subset_to_file(
    dataset_id,
    variables,
    start_dt,
    end_dt,
    bbox,
    filename,
    minimum_depth=None,
    maximum_depth=None
):
    username, password = get_credentials()

    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"

    min_lon, max_lon, min_lat, max_lat = bbox

    with tempfile.TemporaryDirectory() as temp_dir:

        args = dict(
            dataset_id=dataset_id,
            variables=list(variables),
            minimum_longitude=float(min_lon),
            maximum_longitude=float(max_lon),
            minimum_latitude=float(min_lat),
            maximum_latitude=float(max_lat),
            start_datetime=start_dt,
            end_datetime=end_dt,
            username=username,
            password=password,
            output_filename=filename,
            output_directory=temp_dir,
            disable_progress_bar=True,
        )

        if minimum_depth is not None:
            args["minimum_depth"] = float(
                minimum_depth
            )

        if maximum_depth is not None:
            args["maximum_depth"] = float(
                maximum_depth
            )

        response = copernicusmarine.subset(
            **args
        )

        path = resolve_downloaded_file(
            response,
            temp_dir,
            filename
        )

        with xr.open_dataset(path) as ds:
            return ds.load()


# ============================================================
# LIVE COPERNICUS DATA
# ============================================================
def download_live_surface(
    dataset_id,
    variable,
    output_name,
    start_date,
    end_date,
    use_depth=True
):
    bbox = (
        MIN_LON,
        MAX_LON,
        MIN_LAT,
        MAX_LAT
    )

    depth_min = (
        0 if use_depth
        else None
    )

    depth_max = (
        1 if use_depth
        else None
    )

    ds = copernicus_subset_to_file(
        dataset_id,
        [variable],
        f"{start_date}T00:00:00",
        f"{end_date}T23:59:59",
        bbox,
        f"{output_name}.nc",
        depth_min,
        depth_max
    )

    if variable not in ds.data_vars:
        raise RuntimeError(
            f"Variable '{variable}' was not found in {output_name}."
        )

    da = ds[variable]

    if "depth" in da.dims:
        da = da.isel(
            depth=0
        )

    return prepare_surface_dataframe(
        da.to_dataframe(
            name=output_name
        ).reset_index(),
        output_name
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_temperature(
    start_date,
    end_date
):
    return download_live_surface(
        LIVE_TEMP_DATASET,
        "thetao",
        "temperature",
        start_date,
        end_date,
        True
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_salinity(
    start_date,
    end_date
):
    return download_live_surface(
        LIVE_SAL_DATASET,
        "so",
        "salinity",
        start_date,
        end_date,
        True
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_sst_anomaly(
    start_date,
    end_date
):
    return download_live_surface(
        LIVE_SSTA_DATASET,
        "sea_surface_temperature_anomaly",
        "sst_anomaly",
        start_date,
        end_date,
        False
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_oxygen_profile(
    start_date,
    end_date
):
    bbox = (
        MIN_LON,
        MAX_LON,
        MIN_LAT,
        MAX_LAT
    )

    ds = copernicus_subset_to_file(
        LIVE_BGC_DATASET,
        ["o2"],
        f"{start_date}T00:00:00",
        f"{end_date}T23:59:59",
        bbox,
        "oxygen_profile.nc"
    )

    if "o2" not in ds.data_vars:
        raise RuntimeError(
            "Variable 'o2' was not found "
            "in the oxygen-profile file."
        )

    df = normalize_xy_columns(
        ds["o2"]
        .to_dataframe(
            name="oxygen_mmol_m3"
        )
        .reset_index()
    )

    required = [
        "time",
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing oxygen-profile columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "oxygen_mmol_m3",
            "depth",
            "lat",
            "lon"
        ]
    ).copy()

    df["date"] = pd.to_datetime(
        df["time"]
    ).dt.date

    for c in [
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df["lat_key"] = df["lat"].round(3)
    df["lon_key"] = df["lon"].round(3)

    df["oxygen_ml_l"] = (
        df["oxygen_mmol_m3"]
        *
        OXYGEN_MMOL_M3_TO_ML_L
    )

    return df[
        [
            "date",
            "depth",
            "lat",
            "lon",
            "lat_key",
            "lon_key",
            "oxygen_mmol_m3",
            "oxygen_ml_l"
        ]
    ].dropna()


def surface_oxygen_from_profile(
    profile
):
    return (
        profile
        .sort_values("depth")
        .groupby(
            [
                "date",
                "lat_key",
                "lon_key"
            ],
            as_index=False
        )
        .head(1)
        [
            [
                "date",
                "lat",
                "lon",
                "lat_key",
                "lon_key",
                "oxygen_mmol_m3",
                "oxygen_ml_l"
            ]
        ]
        .copy()
    )


def estimate_oxycline_depth(
    profile,
    threshold_ml_l=
        OXYCLINE_THRESHOLD_ML_L
):
    clean = (
        profile[
            [
                "depth",
                "oxygen_ml_l"
            ]
        ]
        .dropna()
        .sort_values("depth")
        .drop_duplicates(
            subset=["depth"]
        )
    )

    if clean.empty:
        return (
            np.nan,
            "No valid oxygen profile"
        )

    depths = clean[
        "depth"
    ].to_numpy(
        float
    )

    oxygen = clean[
        "oxygen_ml_l"
    ].to_numpy(
        float
    )

    if oxygen[0] <= threshold_ml_l:

        return (
            float(depths[0]),
            "Threshold reached at shallowest model level"
        )

    for i in range(
        1,
        len(depths)
    ):

        o1 = oxygen[i - 1]
        o2 = oxygen[i]

        if (
            o1 > threshold_ml_l
            and
            o2 <= threshold_ml_l
        ):

            z1 = depths[i - 1]
            z2 = depths[i]

            if o2 == o1:
                z = z2

            else:
                z = (
                    z1
                    +
                    (
                        (
                            threshold_ml_l
                            -
                            o1
                        )
                        /
                        (
                            o2
                            -
                            o1
                        )
                    )
                    *
                    (
                        z2
                        -
                        z1
                    )
                )

            return (
                float(z),
                "Interpolated 1 mL/L crossing"
            )

    return (
        np.nan,
        "1 mL/L threshold not reached in available profile"
    )


def merge_physics_data(
    temperature,
    salinity,
    anomaly
):
    keys = [
        "date",
        "lat_key",
        "lon_key"
    ]

    out = (
        temperature
        .merge(
            salinity[
                keys
                +
                ["salinity"]
            ],
            on=keys,
            how="inner"
        )
        .merge(
            anomaly[
                keys
                +
                ["sst_anomaly"]
            ],
            on=keys,
            how="left"
        )
    )

    if out.empty:
        raise RuntimeError(
            "No matching physics grid cells were found."
        )

    return out


def nearest_source_indices(
    target_rows,
    source_rows
):
    tlat = target_rows[
        "lat"
    ].to_numpy(
        float
    )

    tlon = target_rows[
        "lon"
    ].to_numpy(
        float
    )

    slat = source_rows[
        "lat"
    ].to_numpy(
        float
    )

    slon = source_rows[
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
        2.0
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
        axis=1
    )

    return (
        idx,
        dist[
            np.arange(
                len(target_rows)
            ),
            idx
        ]
    )


def attach_nearest_oxygen(
    physics,
    oxygen
):
    result = physics.copy()

    for c in [
        "oxygen_mmol_m3",
        "oxygen_ml_l",
        "oxygen_source_distance_km"
    ]:
        result[c] = np.nan

    for d in sorted(
        result["date"].unique()
    ):

        mask = (
            result["date"]
            ==
            d
        )

        target = result.loc[
            mask
        ]

        source = oxygen[
            oxygen["date"]
            ==
            d
        ]

        if source.empty:
            continue

        idx, dist = nearest_source_indices(
            target,
            source
        )

        nearest = source.iloc[
            idx
        ]

        result.loc[
            mask,
            "oxygen_mmol_m3"
        ] = nearest[
            "oxygen_mmol_m3"
        ].to_numpy(
            float
        )

        result.loc[
            mask,
            "oxygen_ml_l"
        ] = nearest[
            "oxygen_ml_l"
        ].to_numpy(
            float
        )

        result.loc[
            mask,
            "oxygen_source_distance_km"
        ] = dist

    if result[
        "oxygen_ml_l"
    ].isna().all():

        raise RuntimeError(
            "Oxygen data could not be "
            "matched to the physics grid."
        )

    return result


# ============================================================
# GEBCO
# ============================================================
def first_numeric(
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
            np.floating
        )
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
        str
    ):
        try:
            v = float(
                value
            )

            return (
                v
                if math.isfinite(v)
                else None
            )

        except ValueError:
            return None

    if isinstance(
        value,
        dict
    ):

        if "z" in value:
            return first_numeric(
                value["z"]
            )

        for k in (
            "data",
            "result",
            "results",
            "records"
        ):
            if k in value:

                found = first_numeric(
                    value[k]
                )

                if found is not None:
                    return found

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        for item in value:

            found = first_numeric(
                item
            )

            if found is not None:
                return found

    return None


def numeric_list(
    value
):
    if value is None:
        return []

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
            str
        )
    ):

        n = first_numeric(
            value
        )

        return (
            []
            if n is None
            else [n]
        )

    if isinstance(
        value,
        dict
    ):

        if "z" in value:
            return numeric_list(
                value["z"]
            )

        for k in (
            "data",
            "result",
            "results",
            "records"
        ):

            if k in value:

                vals = numeric_list(
                    value[k]
                )

                if vals:
                    return vals

        return []

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        out = []

        for item in value:

            n = (
                first_numeric(
                    item["z"]
                )
                if (
                    isinstance(
                        item,
                        dict
                    )
                    and
                    "z" in item
                )
                else
                first_numeric(
                    item
                )
            )

            if n is not None:
                out.append(
                    float(n)
                )

        return out

    return []


def validate_elevation(
    value
):
    if value is None:
        raise RuntimeError(
            "GEBCO API returned no elevation value."
        )

    value = float(
        value
    )

    if (
        not math.isfinite(value)
        or
        value < -12000
        or
        value > 9000
    ):
        raise RuntimeError(
            f"GEBCO returned an implausible elevation: {value} m."
        )

    return value


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_gebco_elevation(
    lat,
    lon
):
    r = requests.get(
        GEBCO_API_URL,
        params={
            "lon":
                f"{float(lon):.6f}",
            "lat":
                f"{float(lat):.6f}",
            "mode":
                "point"
        },
        headers={
            "Accept":
                "application/json",
            "User-Agent":
                "E-AWARE/2.2 academic prototype"
        },
        timeout=30,
    )

    r.raise_for_status()

    payload = r.json()

    value = first_numeric(
        payload["z"]
        if (
            isinstance(
                payload,
                dict
            )
            and
            "z" in payload
        )
        else
        payload
    )

    return validate_elevation(
        value
    )


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_gebco_batch(
    points
):
    points = tuple(
        (
            float(lat),
            float(lon)
        )
        for lat, lon in points
    )

    if not points:
        return []

    params = {
        "lon":
            ",".join(
                f"{lon:.6f}"
                for lat, lon in points
            ),
        "lat":
            ",".join(
                f"{lat:.6f}"
                for lat, lon in points
            ),
        "mode":
            "point",
    }

    try:
        r = requests.get(
            GEBCO_API_URL,
            params=params,
            headers={
                "Accept":
                    "application/json",
                "User-Agent":
                    "E-AWARE/2.2 academic prototype"
            },
            timeout=40,
        )

        r.raise_for_status()

        payload = r.json()

        vals = numeric_list(
            payload["z"]
            if (
                isinstance(
                    payload,
                    dict
                )
                and
                "z" in payload
            )
            else
            payload
        )

        if len(vals) == len(points):

            return [
                validate_elevation(v)
                for v in vals
            ]

    except Exception:
        pass

    return [
        get_gebco_elevation(
            lat,
            lon
        )
        for lat, lon in points
    ]


def attach_gebco_bathymetry(
    dataframe
):
    result = dataframe.copy()

    grid = (
        result[
            [
                "lat",
                "lon",
                "lat_key",
                "lon_key"
            ]
        ]
        .drop_duplicates(
            [
                "lat_key",
                "lon_key"
            ]
        )
        .sort_values(
            [
                "lat",
                "lon"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    points = tuple(
        (
            float(r.lat),
            float(r.lon)
        )
        for r in grid.itertuples(
            index=False
        )
    )

    elevations = get_gebco_batch(
        points
    )

    bath = grid[
        [
            "lat_key",
            "lon_key"
        ]
    ].copy()

    bath[
        "gebco_elevation_m"
    ] = np.asarray(
        elevations,
        float
    )

    bath[
        "is_ocean_gebco"
    ] = (
        bath[
            "gebco_elevation_m"
        ]
        <
        0
    )

    bath[
        "water_depth_m"
    ] = np.where(
        bath[
            "is_ocean_gebco"
        ],
        -
        bath[
            "gebco_elevation_m"
        ],
        np.nan
    )

    result = result.merge(
        bath,
        on=[
            "lat_key",
            "lon_key"
        ],
        how="left"
    )

    result = result[
        result[
            "is_ocean_gebco"
        ]
        ==
        True
    ].copy()

    if result.empty:
        raise RuntimeError(
            "GEBCO land filtering removed all E-AWARE cells."
        )

    return result


# ============================================================
# OXYCLINE MAPPING
# ============================================================
def attach_seabed_constrained_oxycline(
    physics,
    oxygen_profile
):
    result = physics.copy()

    defaults = {
        "raw_oxycline_depth_m":
            np.nan,
        "raw_oxycline_status":
            None,
        "oxycline_depth_m":
            np.nan,
        "oxycline_validation_status":
            None,
        "oxycline_source_distance_km":
            np.nan,
        "oxycline_source_lat":
            np.nan,
        "oxycline_source_lon":
            np.nan,
        "bgc_profile_max_depth_m":
            np.nan,
        "bgc_profile_levels":
            np.nan,
        "local_profile_max_depth_m":
            np.nan,
        "local_profile_levels":
            np.nan,
        "local_depth_gap_m":
            np.nan,
    }

    for c, v in defaults.items():
        result[c] = v

    for d in sorted(
        result["date"].unique()
    ):

        mask = (
            result["date"]
            ==
            d
        )

        target = result.loc[
            mask
        ]

        source_grid = (
            oxygen_profile[
                oxygen_profile[
                    "date"
                ]
                ==
                d
            ]
            [
                [
                    "lat",
                    "lon",
                    "lat_key",
                    "lon_key"
                ]
            ]
            .drop_duplicates(
                [
                    "lat_key",
                    "lon_key"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if source_grid.empty:
            continue

        idx, dist = nearest_source_indices(
            target,
            source_grid
        )

        nearest = source_grid.iloc[
            idx
        ].reset_index(
            drop=True
        )

        for pos, target_index in enumerate(
            target.index.to_list()
        ):

            row = result.loc[
                target_index
            ]

            src = nearest.iloc[
                pos
            ]

            water_depth = float(
                row[
                    "water_depth_m"
                ]
            )

            profile = oxygen_profile[
                (
                    oxygen_profile[
                        "date"
                    ]
                    ==
                    d
                )
                &
                (
                    oxygen_profile[
                        "lat_key"
                    ]
                    ==
                    float(
                        src[
                            "lat_key"
                        ]
                    )
                )
                &
                (
                    oxygen_profile[
                        "lon_key"
                    ]
                    ==
                    float(
                        src[
                            "lon_key"
                        ]
                    )
                )
            ].sort_values(
                "depth"
            )

            if profile.empty:

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    "No mapped BGC oxygen profile"
                )

                continue

            raw_depth, raw_status = (
                estimate_oxycline_depth(
                    profile
                )
            )

            valid_profile = profile[
                profile[
                    "depth"
                ]
                <=
                water_depth
            ].copy()

            result.at[
                target_index,
                "raw_oxycline_depth_m"
            ] = raw_depth

            result.at[
                target_index,
                "raw_oxycline_status"
            ] = raw_status

            result.at[
                target_index,
                "oxycline_source_distance_km"
            ] = float(
                dist[pos]
            )

            result.at[
                target_index,
                "oxycline_source_lat"
            ] = float(
                src[
                    "lat"
                ]
            )

            result.at[
                target_index,
                "oxycline_source_lon"
            ] = float(
                src[
                    "lon"
                ]
            )

            result.at[
                target_index,
                "bgc_profile_max_depth_m"
            ] = float(
                profile[
                    "depth"
                ].max()
            )

            result.at[
                target_index,
                "bgc_profile_levels"
            ] = int(
                profile[
                    "depth"
                ].nunique()
            )

            if valid_profile.empty:

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    "No BGC levels within local GEBCO water column"
                )

                continue

            max_valid = float(
                valid_profile[
                    "depth"
                ].max()
            )

            result.at[
                target_index,
                "local_profile_max_depth_m"
            ] = max_valid

            result.at[
                target_index,
                "local_profile_levels"
            ] = int(
                valid_profile[
                    "depth"
                ].nunique()
            )

            result.at[
                target_index,
                "local_depth_gap_m"
            ] = max(
                0.0,
                water_depth
                -
                max_valid
            )

            local_depth, _ = (
                estimate_oxycline_depth(
                    valid_profile
                )
            )

            if pd.notna(
                local_depth
            ):

                result.at[
                    target_index,
                    "oxycline_depth_m"
                ] = float(
                    local_depth
                )

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    "Accepted — 1 mL/L crossing within local water column"
                )

            elif (
                pd.notna(
                    raw_depth
                )
                and
                float(
                    raw_depth
                )
                >
                water_depth
            ):

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    f"Threshold not reached in valid levels; "
                    f"raw crossing {float(raw_depth):.1f} m "
                    f"is below local {water_depth:.1f} m seabed"
                )

            else:

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    "1 mL/L threshold not reached within valid levels"
                )

    return result


# ============================================================
# NOAA COASTLINE / DISTANCE
# ============================================================
@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_noaa_coastline_lines(
    west=None,
    east=None,
    south=None,
    north=None
):
    if west is None:

        west = (
            MIN_LON
            -
            COASTLINE_QUERY_MARGIN_DEG
        )

        east = (
            MAX_LON
            +
            COASTLINE_QUERY_MARGIN_DEG
        )

        south = (
            MIN_LAT
            -
            COASTLINE_QUERY_MARGIN_DEG
        )

        north = (
            MAX_LAT
            +
            COASTLINE_QUERY_MARGIN_DEG
        )

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
    }

    headers = {
        "Accept":
            "application/json",
        "User-Agent":
            "E-AWARE/2.2 academic prototype"
    }

    try:
        params = dict(
            common,
            f="geojson"
        )

        r = requests.get(
            NOAA_COASTLINE_QUERY_URL,
            params=params,
            headers=headers,
            timeout=45
        )

        r.raise_for_status()

        lines = []

        for feature in r.json().get(
            "features",
            []
        ):

            geom = feature.get(
                "geometry"
            ) or {}

            if geom.get(
                "type"
            ) == "LineString":

                lines.append(
                    geom.get(
                        "coordinates",
                        []
                    )
                )

            elif geom.get(
                "type"
            ) == "MultiLineString":

                lines.extend(
                    geom.get(
                        "coordinates",
                        []
                    )
                )

        if lines:
            return lines

    except Exception:
        pass

    params = dict(
        common,
        f="json"
    )

    r = requests.get(
        NOAA_COASTLINE_QUERY_URL,
        params=params,
        headers=headers,
        timeout=45
    )

    r.raise_for_status()

    payload = r.json()

    if "error" in payload:
        raise RuntimeError(
            f"NOAA coastline service error: {payload['error']}"
        )

    lines = []

    for feature in payload.get(
        "features",
        []
    ):

        lines.extend(
            (
                feature.get(
                    "geometry"
                )
                or
                {}
            ).get(
                "paths",
                []
            )
        )

    if not lines:
        raise RuntimeError(
            "NOAA coastline query returned no coastline geometry."
        )

    return lines


def point_to_segment_distance_km(
    point_lat,
    point_lon,
    lon1,
    lat1,
    lon2,
    lat2
):
    ky = 111.32

    kx = (
        111.32
        *
        math.cos(
            math.radians(
                point_lat
            )
        )
    )

    x1 = (
        float(lon1)
        -
        point_lon
    ) * kx

    y1 = (
        float(lat1)
        -
        point_lat
    ) * ky

    x2 = (
        float(lon2)
        -
        point_lon
    ) * kx

    y2 = (
        float(lat2)
        -
        point_lat
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
            y1
        )

    t = max(
        0.0,
        min(
            1.0,
            -
            (
                x1
                *
                dx
                +
                y1
                *
                dy
            )
            /
            denom
        )
    )

    return math.hypot(
        x1
        +
        t
        *
        dx,
        y1
        +
        t
        *
        dy
    )


def distance_to_nearest_coast_km(
    lat,
    lon,
    coastline_lines
):
    minimum = math.inf

    for line in coastline_lines:

        if (
            not line
            or
            len(line) < 2
        ):
            continue

        for i in range(
            len(line)
            -
            1
        ):

            p1 = line[i]
            p2 = line[i + 1]

            if (
                len(p1) < 2
                or
                len(p2) < 2
            ):
                continue

            minimum = min(
                minimum,
                point_to_segment_distance_km(
                    lat,
                    lon,
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1]
                )
            )

    if not math.isfinite(
        minimum
    ):
        raise RuntimeError(
            "No valid coastline segments were available."
        )

    return float(
        minimum
    )


def attach_distance_to_coast(
    dataframe
):
    result = dataframe.copy()

    lines = load_noaa_coastline_lines()

    grid = (
        result[
            [
                "lat",
                "lon",
                "lat_key",
                "lon_key"
            ]
        ]
        .drop_duplicates(
            [
                "lat_key",
                "lon_key"
            ]
        )
    )

    coast = grid[
        [
            "lat_key",
            "lon_key"
        ]
    ].copy()

    coast[
        "distance_coast_km"
    ] = [
        distance_to_nearest_coast_km(
            float(r.lat),
            float(r.lon),
            lines
        )
        for r in grid.itertuples(
            index=False
        )
    ]

    coast[
        "distance_coast_nm"
    ] = (
        coast[
            "distance_coast_km"
        ]
        /
        KM_PER_NAUTICAL_MILE
    )

    return result.merge(
        coast,
        on=[
            "lat_key",
            "lon_key"
        ],
        how="left"
    )


# ============================================================
# TEMPORAL / HSI
# ============================================================
def austral_season(
    month
):
    if month in (
        12,
        1,
        2
    ):
        return "Summer"

    if month in (
        3,
        4,
        5
    ):
        return "Autumn"

    if month in (
        6,
        7,
        8
    ):
        return "Winter"

    return "Spring"


def add_temporal_features(
    dataframe
):
    df = dataframe.copy()

    dates = pd.to_datetime(
        df["date"]
    )

    month = dates.dt.month.astype(
        int
    )

    angle = (
        2.0
        *
        np.pi
        *
        (
            month
            -
            1
        )
        /
        12.0
    )

    df["month_number"] = month
    df["month_name"] = dates.dt.month_name()

    df[
        "austral_season"
    ] = month.apply(
        austral_season
    )

    df[
        "month_sin"
    ] = np.sin(
        angle
    )

    df[
        "month_cos"
    ] = np.cos(
        angle
    )

    return df


def habitat_reference(
    month
):
    if month in (
        12,
        1,
        2
    ):

        return dict(
            season=
                "Summer",
            reference_basis=
                "Published summer range",
            temp_min=
                17.6,
            temp_max=
                23.7,
            sal_min=
                32.30,
            sal_max=
                35.14,
            oxygen_min=
                5.9,
            oxygen_max=
                8.7
        )

    if month in (
        6,
        7,
        8
    ):

        return dict(
            season=
                "Winter",
            reference_basis=
                "Published winter range",
            temp_min=
                14.5,
            temp_max=
                18.8,
            sal_min=
                34.81,
            sal_max=
                35.12,
            oxygen_min=
                5.2,
            oxygen_max=
                6.3
        )

    transition = (
        "Autumn (Transition)"
        if month in (
            3,
            4,
            5
        )
        else
        "Spring (Transition)"
    )

    return dict(
        season=
            transition,
        reference_basis=
            "Transition envelope — no invented seasonal threshold",
        temp_min=
            14.5,
        temp_max=
            23.7,
        sal_min=
            32.30,
        sal_max=
            35.14,
        oxygen_min=
            5.2,
        oxygen_max=
            8.7
    )


def range_score(
    values,
    minimum,
    maximum,
    shoulder
):
    values = np.asarray(
        values,
        float
    )

    score = np.ones_like(
        values
    )

    below = (
        values
        <
        minimum
    )

    above = (
        values
        >
        maximum
    )

    score[
        below
    ] = (
        1
        -
        (
            minimum
            -
            values[
                below
            ]
        )
        /
        shoulder
    )

    score[
        above
    ] = (
        1
        -
        (
            values[
                above
            ]
            -
            maximum
        )
        /
        shoulder
    )

    return np.clip(
        score,
        0,
        1
    )


def calculate_hsi(
    dataframe,
    month
):
    df = dataframe.copy()

    ref = habitat_reference(
        month
    )

    df[
        "temperature_score"
    ] = range_score(
        df[
            "temperature"
        ],
        ref[
            "temp_min"
        ],
        ref[
            "temp_max"
        ],
        2.0
    )

    df[
        "salinity_score"
    ] = range_score(
        df[
            "salinity"
        ],
        ref[
            "sal_min"
        ],
        ref[
            "sal_max"
        ],
        0.5
    )

    df[
        "oxygen_score"
    ] = range_score(
        df[
            "oxygen_ml_l"
        ],
        ref[
            "oxygen_min"
        ],
        ref[
            "oxygen_max"
        ],
        1.0
    )

    df[
        "hsi"
    ] = (
        (
            df[
                "temperature_score"
            ]
            +
            df[
                "salinity_score"
            ]
            +
            df[
                "oxygen_score"
            ]
        )
        /
        3.0
    ).clip(
        0,
        1
    )

    return df


def score_experimental_ml_live(
    dataframe,
    manifest
):
    """
    Apply the frozen experimental conditional-logistic model
    to the live environmental grid.

    IMPORTANT:
    - Output is a relative ranking score.
    - It is NOT probability of anchoveta presence.
    - It does NOT replace or modify the operational HSI.
    """

    df = dataframe.copy()

    diagnostics = {
        "available": False,
        "reason": None,
        "cells_scored": 0,
        "cells_inside_training_envelope": 0,
        "inside_training_envelope_pct": np.nan,
        "feature_envelope": [],
    }

    df[
        "experimental_ml_score"
    ] = np.nan

    df[
        "experimental_ml_relative_rank"
    ] = np.nan

    df[
        "experimental_ml_outside_training_count"
    ] = np.nan

    df[
        "experimental_ml_in_training_envelope"
    ] = False

    if not manifest:

        diagnostics[
            "reason"
        ] = (
            "Experimental model manifest unavailable."
        )

        return df, diagnostics

    required_live = [
        "temperature",
        "salinity",
        "oxygen_ml_l",
        "water_depth_m",
        "oxycline_depth_m",
    ]

    # oxycline_depth_m is allowed to contain NaN because
    # NaN means that the 1 mL/L threshold was not reached
    # before the local seabed.
    required_nonnullable = [
        "temperature",
        "salinity",
        "oxygen_ml_l",
        "water_depth_m",
    ]

    missing_columns = [
        c
        for c in required_live
        if c not in df.columns
    ]

    if missing_columns:

        diagnostics[
            "reason"
        ] = (
            "Missing live predictor columns: "
            +
            ", ".join(
                missing_columns
            )
        )

        return df, diagnostics

    # --------------------------------------------------------
    # Reproduce the physical oxycline encoding used in
    # final experimental model training.
    # --------------------------------------------------------

    df[
        "oxycline_reached_before_seabed"
    ] = (
        df[
            "oxycline_depth_m"
        ]
        .notna()
        .astype(int)
    )

    df[
        "oxycline_depth_when_reached_m"
    ] = np.where(
        df[
            "oxycline_reached_before_seabed"
        ] == 1,
        df[
            "oxycline_depth_m"
        ],
        0.0,
    )

    live_feature_map = {
        "historical_sst_c":
            "temperature",

        "historical_salinity_psu":
            "salinity",

        "historical_surface_o2_ml_l":
            "oxygen_ml_l",

        "oxycline_reached_before_seabed":
            "oxycline_reached_before_seabed",

        "oxycline_depth_when_reached_m":
            "oxycline_depth_when_reached_m",
    }

    features = manifest.get(
        "features",
        []
    )

    unresolved = [
        feature
        for feature in features
        if feature not in live_feature_map
    ]

    if unresolved:

        diagnostics[
            "reason"
        ] = (
            "Unsupported experimental feature(s): "
            +
            ", ".join(
                unresolved
            )
        )

        return df, diagnostics

    valid_mask = (
        df[
            required_nonnullable
        ]
        .notna()
        .all(
            axis=1
        )
    )

    if not valid_mask.any():

        diagnostics[
            "reason"
        ] = (
            "No live grid cells contain all "
            "experimental predictors."
        )

        return df, diagnostics

    live_columns = [
        live_feature_map[
            feature
        ]
        for feature in features
    ]

    X = (
        df.loc[
            valid_mask,
            live_columns
        ]
        .astype(float)
        .to_numpy()
    )

    mean = np.asarray(
        manifest.get(
            "scaler_mean",
            []
        ),
        dtype=float,
    )

    scale = np.asarray(
        manifest.get(
            "scaler_scale",
            []
        ),
        dtype=float,
    )

    coefficients = np.asarray(
        manifest.get(
            "coefficients",
            []
        ),
        dtype=float,
    )

    if not (
        len(features)
        ==
        len(mean)
        ==
        len(scale)
        ==
        len(coefficients)
    ):

        diagnostics[
            "reason"
        ] = (
            "Experimental model manifest dimensions "
            "are inconsistent."
        )

        return df, diagnostics

    safe_scale = np.where(
        scale == 0,
        1.0,
        scale,
    )

    standardized = (
        X
        -
        mean
    ) / safe_scale

    score = (
        standardized
        @
        coefficients
    )

    valid_index = (
        df.index[
            valid_mask
        ]
    )

    df.loc[
        valid_index,
        "experimental_ml_score",
    ] = score

    # Percentile ranking ONLY within the current live grid.
    # 1.0 = highest experimental relative score today.
    df.loc[
        valid_index,
        "experimental_ml_relative_rank",
    ] = (
        pd.Series(
            score,
            index=valid_index,
        )
        .rank(
            method="average",
            pct=True,
        )
    )

    # --------------------------------------------------------
    # Training-envelope check
    # --------------------------------------------------------

    training_ranges = (
        manifest.get(
            "training_ranges",
            {}
        )
    )

    outside_count = np.zeros(
        len(valid_index),
        dtype=int,
    )

    feature_envelope = []

    for position, feature in enumerate(
        features
    ):

        limits = training_ranges.get(
            feature
        )

        if not limits:
            continue

        minimum = limits.get(
            "min"
        )

        maximum = limits.get(
            "max"
        )

        if (
            minimum is None
            or
            maximum is None
        ):
            continue

        values = X[
            :,
            position
        ]

        outside_mask = (
            (
                values
                <
                float(
                    minimum
                )
            )
            |
            (
                values
                >
                float(
                    maximum
                )
            )
        )

        outside_count += (
            outside_mask.astype(int)
        )

        feature_envelope.append(
            {
                "Predictor":
                    feature,

                "Training Min":
                    float(minimum),

                "Training Max":
                    float(maximum),

                "Live Min":
                    float(
                        np.nanmin(
                            values
                        )
                    ),

                "Live Max":
                    float(
                        np.nanmax(
                            values
                        )
                    ),

                "Cells Outside":
                    int(
                        outside_mask.sum()
                    ),

                "Cells Scored":
                    int(
                        len(values)
                    ),
            }
        )

    df.loc[
        valid_index,
        "experimental_ml_outside_training_count",
    ] = outside_count

    inside = (
        outside_count
        ==
        0
    )

    df.loc[
        valid_index,
        "experimental_ml_in_training_envelope",
    ] = inside

    diagnostics[
        "available"
    ] = True

    diagnostics[
        "cells_scored"
    ] = int(
        len(
            valid_index
        )
    )

    diagnostics[
        "cells_inside_training_envelope"
    ] = int(
        inside.sum()
    )

    diagnostics[
        "inside_training_envelope_pct"
    ] = float(
        100.0
        *
        inside.mean()
    )

    diagnostics[
        "feature_envelope"
    ] = feature_envelope

    return df, diagnostics


def experimental_rank_agreement_label(
    correlation
):
    """
    Dashboard diagnostic only.
    These are NOT biological thresholds and are NOT
    model-validation categories.
    """

    if pd.isna(
        correlation
    ):
        return "Unavailable"

    if correlation >= 0.60:
        return "High"

    if correlation >= 0.30:
        return "Moderate"

    return "Low"


def assess_fisheries_protection(
    best,
    hsi_confidence,
    juvenile_pct=None
):
    """
    Fisheries protection decision-support layer.

    IMPORTANT:
    - E-AWARE does NOT declare legal fishing closures.
    - Official closures/restrictions must come from PRODUCE.
    - Juvenile status is assessed only when actual biometric
      juvenile data are available.
    - HSI remains habitat guidance, not regulatory guidance.
    """

    # --------------------------------------------------------
    # Official regulatory status
    # --------------------------------------------------------

    official_status = "NOT LIVE-VERIFIED"

    official_detail = (
        "E-AWARE is not currently connected to a live PRODUCE "
        "regulatory feed. Current closures and restrictions "
        "must be verified from official Peruvian authorities."
    )


    # --------------------------------------------------------
    # Juvenile protection
    #
    # Prototype reference:
    # Anchoveta minimum total length = 12 cm
    # Maximum juvenile tolerance = 10%
    #
    # E-AWARE only applies this logic if a real juvenile
    # percentage is supplied.
    # --------------------------------------------------------

    if juvenile_pct is None:

        juvenile_status = "NOT ASSESSED"

        juvenile_detail = (
            "No current biometric juvenile-incidence data "
            "are available to E-AWARE."
        )

    else:

        juvenile_pct = float(
            juvenile_pct
        )

        if juvenile_pct <= 10.0:

            juvenile_status = "WITHIN REFERENCE TOLERANCE"

            juvenile_detail = (
                f"Juvenile incidence = {juvenile_pct:.1f}%. "
                "This is within the 10% reference tolerance. "
                "Official fishing status must still be verified."
            )

        else:

            juvenile_status = "CONSERVATION CAUTION"

            juvenile_detail = (
                f"Juvenile incidence = {juvenile_pct:.1f}%. "
                "This exceeds the 10% reference tolerance. "
                "E-AWARE flags conservation caution only; "
                "it does not declare a legal closure."
            )


    # --------------------------------------------------------
    # E-AWARE ecological advisory
    #
    # This uses the PRIMARY HSI only.
    # It is habitat interpretation, NOT regulation.
    # --------------------------------------------------------

    try:

        current_hsi = float(
            best[
                "hsi"
            ]
        )

    except Exception:

        current_hsi = np.nan


    confidence = hsi_confidence[
        "overall_confidence"
    ]


    if confidence == "Low":

        ecological_status = "USE WITH CAUTION"

        ecological_detail = (
            "Environmental support for the current HSI is limited. "
            "Do not rely on the habitat advisory without additional "
            "local and official information."
        )

    elif pd.isna(
        current_hsi
    ):

        ecological_status = "UNAVAILABLE"

        ecological_detail = (
            "A valid HSI value is not available for the current zone."
        )

    elif current_hsi < 0.30:

        ecological_status = "LOW HABITAT SUITABILITY"

        ecological_detail = (
            "Current environmental conditions indicate low "
            "anchoveta habitat suitability in the highest-scoring "
            "available E-AWARE cell."
        )

    elif current_hsi < 0.60:

        ecological_status = "MODERATE HABITAT SUITABILITY"

        ecological_detail = (
            "Current environmental conditions indicate moderate "
            "anchoveta habitat suitability. This is habitat guidance "
            "only and must not be interpreted as fishing permission."
        )

    elif current_hsi < 0.80:

        ecological_status = "HIGH HABITAT SUITABILITY"

        ecological_detail = (
            "Current environmental conditions indicate relatively "
            "high habitat suitability. This does not override any "
            "official fishing restriction."
        )

    else:

        ecological_status = "VERY HIGH HABITAT SUITABILITY"

        ecological_detail = (
            "Current environmental conditions indicate very high "
            "habitat suitability. This remains a habitat prediction "
            "and does not override regulation."
        )


    return {
        "official_status":
            official_status,

        "official_detail":
            official_detail,

        "juvenile_status":
            juvenile_status,

        "juvenile_detail":
            juvenile_detail,

        "juvenile_pct":
            juvenile_pct,

        "ecological_status":
            ecological_status,

        "ecological_detail":
            ecological_detail,

        "minimum_length_cm":
            12.0,

        "juvenile_tolerance_pct":
            10.0,

        "regulatory_authority":
            "PRODUCE",

        "habitat_only":
            True,
    }


def assess_primary_hsi_confidence(
    ocean,
    current_date,
    current_data,
    available_dates
):
    """
    Engineering confidence diagnostic for the PRIMARY
    literature-based HSI.

    This is NOT probability of fish presence.

    Confidence considers:
    1. completeness of the three HSI predictors;
    2. freshness of the live environmental date;
    3. availability of forecast dates.

    Because the literature HSI has not yet been independently
    validated against fish observations, overall confidence is
    capped at MODERATE.
    """

    required_hsi = [
        "temperature",
        "salinity",
        "oxygen_ml_l",
    ]

    same_date = ocean[
        ocean[
            "date"
        ]
        ==
        current_date
    ].copy()

    # --------------------------------------------------------
    # HSI predictor completeness
    # --------------------------------------------------------

    if same_date.empty:

        coverage_pct = 0.0

    else:

        complete_mask = (
            same_date[
                required_hsi
            ]
            .notna()
            .all(
                axis=1
            )
        )

        coverage_pct = float(
            100.0
            *
            complete_mask.mean()
        )

    if coverage_pct >= 90.0:

        coverage_score = 2
        coverage_label = "Good"

    elif coverage_pct >= 70.0:

        coverage_score = 1
        coverage_label = "Partial"

    else:

        coverage_score = 0
        coverage_label = "Limited"


    # --------------------------------------------------------
    # Data freshness
    #
    # These are engineering diagnostic thresholds,
    # not biological habitat thresholds.
    # --------------------------------------------------------

    today = pd.Timestamp(
        date.today()
    ).normalize()

    data_day = pd.Timestamp(
        current_date
    ).normalize()

    age_days = max(
        0,
        int(
            (
                today
                -
                data_day
            ).days
        )
    )

    if age_days <= 2:

        freshness_score = 2
        freshness_label = "Current"

    elif age_days <= 5:

        freshness_score = 1
        freshness_label = "Recent"

    else:

        freshness_score = 0
        freshness_label = "Stale"


    # --------------------------------------------------------
    # Forecast-date availability
    # --------------------------------------------------------

    future_dates = [
        d
        for d in available_dates
        if pd.Timestamp(
            d
        )
        >=
        pd.Timestamp(
            current_date
        )
    ]

    forecast_date_count = len(
        future_dates
    )

    if forecast_date_count >= 3:

        forecast_score = 2
        forecast_label = "Good"

    elif forecast_date_count >= 2:

        forecast_score = 1
        forecast_label = "Limited"

    else:

        forecast_score = 0
        forecast_label = "Insufficient"


    # --------------------------------------------------------
    # Raw operational support
    # --------------------------------------------------------

    support_score = (
        coverage_score
        +
        freshness_score
        +
        forecast_score
    )

    if support_score >= 5:

        raw_support = "High"

    elif support_score >= 3:

        raw_support = "Moderate"

    else:

        raw_support = "Low"


    # --------------------------------------------------------
    # Evidence ceiling
    #
    # HSI is literature-based but not yet independently
    # validated against observed anchoveta distributions.
    # --------------------------------------------------------

    if raw_support == "High":

        overall_confidence = "Moderate"

    else:

        overall_confidence = raw_support


    # --------------------------------------------------------
    # Spatial diagnostics — displayed but NOT used to create
    # the confidence category.
    # --------------------------------------------------------

    hsi_gap = np.nan
    near_best_cells = 0

    if (
        "hsi"
        in current_data.columns
        and
        current_data[
            "hsi"
        ].notna().any()
    ):

        max_hsi = float(
            current_data[
                "hsi"
            ].max()
        )

        median_hsi = float(
            current_data[
                "hsi"
            ].median()
        )

        hsi_gap = (
            max_hsi
            -
            median_hsi
        )

        near_best_cells = int(
            (
                current_data[
                    "hsi"
                ]
                >=
                max_hsi - 0.01
            ).sum()
        )


    return {
        "overall_confidence":
            overall_confidence,

        "raw_operational_support":
            raw_support,

        "coverage_pct":
            coverage_pct,

        "coverage_label":
            coverage_label,

        "age_days":
            age_days,

        "freshness_label":
            freshness_label,

        "forecast_date_count":
            forecast_date_count,

        "forecast_label":
            forecast_label,

        "support_score":
            support_score,

        "hsi_gap":
            hsi_gap,

        "near_best_cells":
            near_best_cells,

        "evidence_ceiling":
            "Moderate",

        "independently_validated":
            False,
    }


def hsi_class(
    value
):
    if value >= 0.8:
        return "Very High"

    if value >= 0.6:
        return "High"

    if value >= 0.3:
        return "Moderate"

    return "Low"


def habitat_centroid(
    dataframe
):
    if dataframe.empty:
        return None

    valid_hsi = dataframe[
        pd.to_numeric(
            dataframe["hsi"],
            errors="coerce"
        ).notna()
    ].copy()

    if (
        valid_hsi.empty
        or
        float(valid_hsi["hsi"].max()) <= 0
    ):
        return None

    threshold = float(
        valid_hsi[
            "hsi"
        ].quantile(
            0.75
        )
    )

    best_area = valid_hsi[
        (
            valid_hsi[
                "hsi"
            ]
            >=
            threshold
        )
        &
        (
            valid_hsi[
                "hsi"
            ]
            >
            0
        )
    ].copy()

    if best_area.empty:
        best_area = valid_hsi.nlargest(
            1,
            "hsi"
        ).copy()

    w = np.clip(
        best_area[
            "hsi"
        ].to_numpy(),
        0.01,
        None
    )

    return (
        float(
            np.average(
                best_area[
                    "lat"
                ],
                weights=w
            )
        ),
        float(
            np.average(
                best_area[
                    "lon"
                ],
                weights=w
            )
        )
    )


# ============================================================
# MOVEMENT
# ============================================================
def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):
    r = 6371.0

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dp = math.radians(
        lat2
        -
        lat1
    )

    dl = math.radians(
        lon2
        -
        lon1
    )

    a = (
        math.sin(
            dp
            /
            2
        )
        ** 2
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
            dl
            /
            2
        )
        ** 2
    )

    return (
        2
        *
        r
        *
        math.asin(
            math.sqrt(
                a
            )
        )
    )


def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):
    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dl = math.radians(
        lon2
        -
        lon1
    )

    x = (
        math.sin(
            dl
        )
        *
        math.cos(
            p2
        )
    )

    y = (
        math.cos(
            p1
        )
        *
        math.sin(
            p2
        )
        -
        math.sin(
            p1
        )
        *
        math.cos(
            p2
        )
        *
        math.cos(
            dl
        )
    )

    return (
        math.degrees(
            math.atan2(
                x,
                y
            )
        )
        +
        360
    ) % 360


def direction_name(
    bearing
):
    dirs = [
        "North ↑",
        "North-East ↗",
        "East →",
        "South-East ↘",
        "South ↓",
        "South-West ↙",
        "West ←",
        "North-West ↖"
    ]

    return dirs[
        int(
            (
                bearing
                +
                22.5
            )
            //
            45
        )
        %
        8
    ]


# ============================================================
# V2 STEP 1 — HISTORICAL INGESTION
# ============================================================
COLUMN_ALIASES = {
    "date": [
        "date",
        "datetime",
        "time",
        "fecha",
        "sample_date",
        "survey_date"
    ],

    "lat": [
        "lat",
        "latitude",
        "latitud",
        "y"
    ],

    "lon": [
        "lon",
        "longitude",
        "longitud",
        "lng",
        "x"
    ],

    "presence": [
        "presence",
        "present",
        "occurrence",
        "presence_absence",
        "pres_abs",
        "anchovy_presence",
        "anchoveta_presence",
        "fish_presence"
    ],

    "abundance": [
        "abundance",
        "anchovy_abundance",
        "anchoveta_abundance",
        "density",
        "acoustic_abundance",
        "sa",
        "nasc",
        "biomass_index"
    ],

    "length_cm": [
        "length_cm",
        "length",
        "size_cm",
        "talla_cm",
        "fish_length_cm"
    ],

    "catch": [
        "catch",
        "catch_t",
        "catch_tonnes",
        "volume",
        "volumen",
        "landings"
    ],

    "survey_id": [
        "survey_id",
        "survey",
        "cruise_id",
        "cruise",
        "crucero",
        "campaign",
        "expedition"
    ],
}


def normalise_column_name(
    name
):
    return (
        str(
            name
        )
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "-",
            "_"
        )
    )


def find_alias(
    columns,
    canonical
):
    normalized = {
        normalise_column_name(c):
            c
        for c in columns
    }

    for alias in COLUMN_ALIASES[
        canonical
    ]:

        if alias in normalized:
            return normalized[
                alias
            ]

    return None


def parse_presence(
    series
):
    def parse_value(value):
        if pd.isna(value):
            return np.nan

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating
            )
        ):
            numeric = float(
                value
            )

            if numeric == 1:
                return 1.0

            if numeric == 0:
                return 0.0

            return np.nan

        text = str(
            value
        ).strip().lower()

        yes = {
            "1",
            "true",
            "yes",
            "y",
            "present",
            "presence",
            "si",
            "sí"
        }

        no = {
            "0",
            "false",
            "no",
            "n",
            "absent",
            "absence"
        }

        if text in yes:
            return 1.0

        if text in no:
            return 0.0

        return np.nan

    return series.apply(
        parse_value
    ).astype(
        float
    )


def standardise_historical_observations(
    raw,
    abundance_zero_is_absence=False
):
    df = raw.copy()

    mapping = {
        k:
            find_alias(
                df.columns,
                k
            )
        for k in COLUMN_ALIASES
    }

    missing = [
        c
        for c in (
            "date",
            "lat",
            "lon"
        )
        if mapping[
            c
        ]
        is None
    ]

    if missing:
        raise ValueError(
            "Historical file is missing required field(s): "
            +
            ", ".join(
                missing
            )
        )

    out = pd.DataFrame()

    out[
        "date"
    ] = pd.to_datetime(
        df[
            mapping[
                "date"
            ]
        ],
        errors="coerce"
    )

    out[
        "lat"
    ] = pd.to_numeric(
        df[
            mapping[
                "lat"
            ]
        ],
        errors="coerce"
    )

    out[
        "lon"
    ] = pd.to_numeric(
        df[
            mapping[
                "lon"
            ]
        ],
        errors="coerce"
    )

    presence_source = "Unavailable"
    target_absence_verified = False
    target_method = "No usable binary target"

    if mapping[
        "presence"
    ] is not None:

        out[
            "presence"
        ] = parse_presence(
            df[
                mapping[
                    "presence"
                ]
            ]
        )

        presence_source = (
            f"Explicit presence/absence field: "
            f"{mapping['presence']}"
        )

        target_absence_verified = bool(
            (
                out[
                    "presence"
                ]
                ==
                0
            ).any()
        )

        target_method = (
            "Explicit binary presence/absence field"
        )

    elif mapping[
        "abundance"
    ] is not None:

        abundance = pd.to_numeric(
            df[
                mapping[
                    "abundance"
                ]
            ],
            errors="coerce"
        )

        if abundance_zero_is_absence:

            out[
                "presence"
            ] = np.where(
                abundance.notna(),
                (
                    abundance
                    >
                    0
                ).astype(
                    float
                ),
                np.nan
            )

            presence_source = (
                "Derived from abundance after documented confirmation "
                "that zero represents surveyed absence: "
                f"{mapping['abundance']}"
            )

            target_absence_verified = bool(
                (
                    abundance
                    ==
                    0
                ).any()
            )

            target_method = (
                "Abundance-derived binary target with confirmed zero-as-absence"
            )

        else:

            out[
                "presence"
            ] = np.where(
                abundance
                >
                0,
                1.0,
                np.nan
            )

            presence_source = (
                "Positive abundance used as presence only; zero abundance "
                "is left unknown until survey methodology confirms absence: "
                f"{mapping['abundance']}"
            )

            target_method = (
                "Presence-only target derived from positive abundance"
            )

    else:
        out[
            "presence"
        ] = np.nan

    for canonical in (
        "abundance",
        "length_cm",
        "catch"
    ):

        source = mapping[
            canonical
        ]

        out[
            canonical
        ] = (
            pd.to_numeric(
                df[source],
                errors="coerce"
            )
            if source is not None
            else
            np.nan
        )

    survey_source = mapping[
        "survey_id"
    ]

    out[
        "survey_id"
    ] = (
        df[
            survey_source
        ]
        .astype(str)
        .replace(
            {
                "nan": None,
                "None": None,
                "": None
            }
        )
        if survey_source is not None
        else
        None
    )

    out[
        "source_row"
    ] = np.arange(
        1,
        len(out)
        +
        1
    )

    out[
        "valid_coordinates"
    ] = (
        out[
            "lat"
        ].between(
            -90,
            90
        )
        &
        out[
            "lon"
        ].between(
            -180,
            180
        )
    )

    out[
        "within_peru_lat_band"
    ] = out[
        "lat"
    ].between(
        -20.5,
        0.5
    )

    out[
        "within_eaware_study_area"
    ] = (
        out[
            "lat"
        ].between(
            MIN_LAT,
            MAX_LAT
        )
        &
        out[
            "lon"
        ].between(
            MIN_LON,
            MAX_LON
        )
    )

    out[
        "month_number"
    ] = out[
        "date"
    ].dt.month

    out[
        "year"
    ] = out[
        "date"
    ].dt.year

    out[
        "austral_season"
    ] = out[
        "month_number"
    ].apply(
        lambda m:
            austral_season(
                int(m)
            )
            if pd.notna(m)
            else None
    )

    usable = out[
        out[
            "date"
        ].notna()
        &
        out[
            "valid_coordinates"
        ]
    ].copy()

    known_presence = usable[
        "presence"
    ].dropna()

    presence_count = int(
        (
            known_presence
            ==
            1
        ).sum()
    )

    absence_count = int(
        (
            known_presence
            ==
            0
        ).sum()
    )

    metadata = {
        "mapping":
            mapping,

        "presence_source":
            presence_source,

        "target_method":
            target_method,

        "target_absence_verified":
            bool(
                target_absence_verified
            ),

        "raw_rows":
            int(
                len(out)
            ),

        "usable_rows":
            int(
                len(usable)
            ),

        "invalid_date_rows":
            int(
                out[
                    "date"
                ].isna().sum()
            ),

        "invalid_coordinate_rows":
            int(
                (
                    ~
                    out[
                        "valid_coordinates"
                    ]
                ).sum()
            ),

        "study_area_rows":
            int(
                usable[
                    "within_eaware_study_area"
                ].sum()
            ),

        "presence_known_rows":
            int(
                known_presence.shape[0]
            ),

        "presence_rows":
            presence_count,

        "absence_rows":
            absence_count,

        "survey_id_available":
            bool(
                survey_source is not None
            ),
    }

    metadata[
        "training_ready"
    ] = (
        metadata[
            "usable_rows"
        ]
        >
        0
        and
        metadata[
            "presence_rows"
        ]
        >
        0
        and
        metadata[
            "absence_rows"
        ]
        >
        0
        and
        metadata[
            "target_absence_verified"
        ]
    )

    return (
        usable,
        metadata
    )

def read_historical_source():
    """Load the repository-managed anchoveta observation dataset automatically."""

    for path in (
        OBSERVATION_REPO_PATH,
        LEGACY_OBSERVATION_REPO_PATH,
    ):
        if path.exists():
            return (
                pd.read_csv(path),
                str(path)
            )

    return (
        None,
        None
    )


def describe_observation_dataset(raw, source_path):
    """Return transparent source/role labels without overstating provenance."""

    normalized = {
        normalise_column_name(c)
        for c in raw.columns
    }

    is_obis = bool(
        {
            "obis_stage",
            "obis_id",
            "dataset_id",
            "basis_of_record",
        }
        &
        normalized
    ) and (
        "obis_stage" in normalized
        or
        "obis_id" in normalized
    )

    if is_obis:
        return {
            "source_label": "OBIS interim anchoveta occurrence dataset",
            "dataset_role": (
                "Pipeline validation only — not the final supervised "
                "presence/absence training dataset"
            ),
            "status_kind": "info",
        }

    return {
        "source_label": "Repository-managed anchoveta observation dataset",
        "dataset_role": (
            "Candidate observation dataset — training suitability is determined "
            "by the data quality-control audit"
        ),
        "status_kind": "info",
    }


# ============================================================
# V2 STEP 3 — DATA QUALITY CONTROL HELPERS
# ============================================================
def observation_qc_audit(
    historical,
    metadata
):
    if historical.empty:
        return {
            "summary": pd.DataFrame(),
            "duplicates": historical.copy(),
            "class_balance": pd.DataFrame(),
            "year_counts": pd.DataFrame(),
            "season_counts": pd.DataFrame(),
            "status": "Not ready — no usable observations"
        }

    duplicate_keys = [
        "date",
        "lat",
        "lon"
    ]

    if (
        "survey_id" in historical.columns
        and
        historical[
            "survey_id"
        ].notna().any()
    ):
        duplicate_keys.append(
            "survey_id"
        )

    duplicate_mask = historical.duplicated(
        subset=duplicate_keys,
        keep=False
    )

    duplicates = historical[
        duplicate_mask
    ].copy()

    dates = pd.to_datetime(
        historical[
            "date"
        ],
        errors="coerce"
    )

    pre1993 = int(
        (
            dates
            <
            HISTORICAL_COPERNICUS_START
        ).sum()
    )

    outside_peru_lat = int(
        (
            ~historical[
                "within_peru_lat_band"
            ]
        ).sum()
    )

    presence_count = int(
        (
            historical[
                "presence"
            ]
            ==
            1
        ).sum()
    )

    absence_count = int(
        (
            historical[
                "presence"
            ]
            ==
            0
        ).sum()
    )

    unknown_target = int(
        historical[
            "presence"
        ].isna().sum()
    )

    known_total = (
        presence_count
        +
        absence_count
    )

    presence_pct = (
        100.0
        *
        presence_count
        /
        known_total
        if known_total
        else
        np.nan
    )

    absence_pct = (
        100.0
        *
        absence_count
        /
        known_total
        if known_total
        else
        np.nan
    )

    class_balance = pd.DataFrame(
        [
            {
                "Class": "Presence",
                "Rows": presence_count,
                "Percent of known target": presence_pct
            },
            {
                "Class": "Absence",
                "Rows": absence_count,
                "Percent of known target": absence_pct
            },
            {
                "Class": "Unknown",
                "Rows": unknown_target,
                "Percent of known target": np.nan
            }
        ]
    )

    year_counts = (
        historical
        .dropna(
            subset=[
                "year"
            ]
        )
        .groupby(
            "year"
        )
        .size()
        .reset_index(
            name="observations"
        )
        .sort_values(
            "year"
        )
    )

    season_counts = (
        historical
        .dropna(
            subset=[
                "austral_season"
            ]
        )
        .groupby(
            "austral_season"
        )
        .size()
        .reset_index(
            name="observations"
        )
        .sort_values(
            "observations",
            ascending=False
        )
    )

    survey_count = (
        int(
            historical[
                "survey_id"
            ]
            .dropna()
            .nunique()
        )
        if (
            "survey_id" in historical.columns
            and
            historical[
                "survey_id"
            ].notna().any()
        )
        else
        0
    )

    summary = pd.DataFrame(
        [
            {
                "QC check": "Usable observations",
                "Value": int(len(historical)),
                "Status": "Pass" if len(historical) else "Fail",
                "Interpretation": "Rows with valid date and global coordinates"
            },
            {
                "QC check": "Invalid dates in raw file",
                "Value": metadata.get("invalid_date_rows", 0),
                "Status": "Pass" if metadata.get("invalid_date_rows", 0) == 0 else "Review",
                "Interpretation": "Rows excluded during ingestion"
            },
            {
                "QC check": "Invalid coordinates in raw file",
                "Value": metadata.get("invalid_coordinate_rows", 0),
                "Status": "Pass" if metadata.get("invalid_coordinate_rows", 0) == 0 else "Review",
                "Interpretation": "Coordinates outside ±90° latitude or ±180° longitude"
            },
            {
                "QC check": "Outside broad Peru latitude band",
                "Value": outside_peru_lat,
                "Status": "Pass" if outside_peru_lat == 0 else "Review",
                "Interpretation": "Geographic screening flag only; not a land/ocean test"
            },
            {
                "QC check": "Duplicate date/location records",
                "Value": int(duplicate_mask.sum()),
                "Status": "Pass" if int(duplicate_mask.sum()) == 0 else "Review",
                "Interpretation": "Duplicates are flagged, not automatically deleted"
            },
            {
                "QC check": "Observations before Copernicus historical coverage",
                "Value": pre1993,
                "Status": "Pass" if pre1993 == 0 else "Review",
                "Interpretation": "Pre-1993 observations cannot receive current historical Copernicus predictors"
            },
            {
                "QC check": "Inside current E-AWARE study box",
                "Value": int(historical["within_eaware_study_area"].sum()),
                "Status": "Info",
                "Interpretation": "Lobitos–Piedritas–Siches application area"
            },
            {
                "QC check": "Presence records",
                "Value": presence_count,
                "Status": "Pass" if presence_count > 0 else "Fail",
                "Interpretation": "Binary target class = 1"
            },
            {
                "QC check": "Absence records",
                "Value": absence_count,
                "Status": "Pass" if absence_count > 0 else "Fail",
                "Interpretation": "Must represent genuine surveyed non-detection/absence"
            },
            {
                "QC check": "Unknown target rows",
                "Value": unknown_target,
                "Status": "Info" if unknown_target == 0 else "Review",
                "Interpretation": "Rows are retained but cannot train a binary presence/absence model"
            },
            {
                "QC check": "Survey/cruise identifier availability",
                "Value": survey_count,
                "Status": "Pass" if survey_count > 0 else "Review",
                "Interpretation": "Useful for later survey-level train/test splitting"
            },
        ]
    )

    if not metadata.get(
        "target_absence_verified",
        False
    ):
        status = (
            "Not training-ready — absence/non-detection semantics are not verified"
        )
    elif presence_count == 0 or absence_count == 0:
        status = (
            "Not training-ready — both binary target classes are required"
        )
    else:
        status = (
            "Structurally suitable for environmental matching; survey methodology and sampling design still require review"
        )

    return {
        "summary": summary,
        "duplicates": duplicates,
        "class_balance": class_balance,
        "year_counts": year_counts,
        "season_counts": season_counts,
        "status": status,
        "date_min": dates.min(),
        "date_max": dates.max(),
        "survey_count": survey_count,
    }


def matched_qc_audit(
    matched,
    metadata
):
    if matched is None or matched.empty:
        return {
            "summary": pd.DataFrame(),
            "predictor_completeness": pd.DataFrame(),
            "candidate_rows": pd.DataFrame(),
            "status": "Run historical environmental matching before environmental QC"
        }

    work = matched.copy()

    core_predictors = [
        "historical_sst_c",
        "historical_salinity_psu",
        "historical_surface_o2_ml_l",
        "historical_water_depth_m",
        "historical_distance_coast_km",
        "month_sin",
        "month_cos"
    ]

    optional_predictors = [
        "historical_oxycline_m"
    ]

    matched_mask = (
        work[
            "match_status"
        ]
        ==
        "Matched"
    )

    binary_target_mask = work[
        "presence"
    ].isin(
        [
            0,
            1
        ]
    )

    marine_depth_mask = (
        work[
            "historical_water_depth_m"
        ].notna()
        &
        (
            work[
                "historical_water_depth_m"
            ]
            >
            0
        )
    )

    core_complete_mask = work[
        core_predictors
    ].notna().all(
        axis=1
    )

    work[
        "qc_possible_land_or_missing_bathymetry"
    ] = (
        matched_mask
        &
        ~marine_depth_mask
    )

    work[
        "qc_physics_match_distance_warning"
    ] = (
        work[
            "physics_match_distance_km"
        ]
        >
        QC_MAX_PHYSICS_MATCH_KM
    )

    work[
        "qc_bgc_match_distance_warning"
    ] = (
        work[
            "bgc_match_distance_km"
        ]
        >
        QC_MAX_BGC_MATCH_KM
    )

    work[
        "qc_core_predictors_complete"
    ] = core_complete_mask

    work[
        "qc_binary_target_known"
    ] = binary_target_mask

    work[
        "qc_training_candidate_core"
    ] = (
        matched_mask
        &
        binary_target_mask
        &
        marine_depth_mask
        &
        core_complete_mask
    )

    predictor_rows = []

    for col in (
        core_predictors
        +
        optional_predictors
    ):
        available = int(
            work[
                col
            ].notna().sum()
        )

        predictor_rows.append(
            {
                "Predictor": col,
                "Available rows": available,
                "Missing rows": int(len(work) - available),
                "Completeness %": (
                    100.0
                    *
                    available
                    /
                    len(work)
                    if len(work)
                    else
                    np.nan
                ),
                "Role": (
                    "Core candidate predictor"
                    if col in core_predictors
                    else
                    "Additional predictor / availability audit"
                )
            }
        )

    predictor_completeness = pd.DataFrame(
        predictor_rows
    )

    candidate_rows = work[
        work[
            "qc_training_candidate_core"
        ]
    ].copy()

    candidate_presence = int(
        (
            candidate_rows[
                "presence"
            ]
            ==
            1
        ).sum()
    )

    candidate_absence = int(
        (
            candidate_rows[
                "presence"
            ]
            ==
            0
        ).sum()
    )

    possible_land = int(
        work[
            "qc_possible_land_or_missing_bathymetry"
        ].sum()
    )

    physics_warnings = int(
        work[
            "qc_physics_match_distance_warning"
        ].sum()
    )

    bgc_warnings = int(
        work[
            "qc_bgc_match_distance_warning"
        ].sum()
    )

    missing_oxycline = int(
        work[
            "historical_oxycline_m"
        ].isna().sum()
    )

    duplicate_mask = work.duplicated(
        subset=[
            "date",
            "lat",
            "lon"
        ],
        keep=False
    )

    summary = pd.DataFrame(
        [
            {
                "QC check": "Matched observations",
                "Value": int(matched_mask.sum()),
                "Status": "Pass" if int(matched_mask.sum()) > 0 else "Fail",
                "Interpretation": "Rows with successful historical Copernicus matching"
            },
            {
                "QC check": "Core environmental predictors complete",
                "Value": int(core_complete_mask.sum()),
                "Status": "Pass" if int(core_complete_mask.sum()) > 0 else "Fail",
                "Interpretation": "SST, salinity, surface oxygen, bathymetry, coast distance and cyclic month"
            },
            {
                "QC check": "Possible land / missing bathymetry",
                "Value": possible_land,
                "Status": "Pass" if possible_land == 0 else "Review",
                "Interpretation": "Rows without a positive GEBCO ocean depth"
            },
            {
                "QC check": f"Physics match > {QC_MAX_PHYSICS_MATCH_KM:.0f} km",
                "Value": physics_warnings,
                "Status": "Pass" if physics_warnings == 0 else "Review",
                "Interpretation": "Engineering QC warning relative to the ~0.083° physics grid"
            },
            {
                "QC check": f"BGC match > {QC_MAX_BGC_MATCH_KM:.0f} km",
                "Value": bgc_warnings,
                "Status": "Pass" if bgc_warnings == 0 else "Review",
                "Interpretation": "Engineering QC warning relative to the ~0.25° BGC grid"
            },
            {
                "QC check": "Missing oxycline values",
                "Value": missing_oxycline,
                "Status": "Info" if missing_oxycline == 0 else "Review",
                "Interpretation": "A missing oxycline can mean the 1 mL/L threshold was not reached within valid sampled levels"
            },
            {
                "QC check": "Duplicate matched date/location rows",
                "Value": int(duplicate_mask.sum()),
                "Status": "Pass" if int(duplicate_mask.sum()) == 0 else "Review",
                "Interpretation": "Review repeated observations before model fitting"
            },
            {
                "QC check": "Core training-candidate rows",
                "Value": int(len(candidate_rows)),
                "Status": "Pass" if len(candidate_rows) > 0 else "Fail",
                "Interpretation": "Matched marine rows with binary target and complete core predictors"
            },
            {
                "QC check": "Candidate presence rows",
                "Value": candidate_presence,
                "Status": "Pass" if candidate_presence > 0 else "Fail",
                "Interpretation": "Presence class after environmental QC"
            },
            {
                "QC check": "Candidate absence rows",
                "Value": candidate_absence,
                "Status": "Pass" if candidate_absence > 0 else "Fail",
                "Interpretation": "Absence class after environmental QC"
            },
        ]
    )

    if not metadata.get(
        "target_absence_verified",
        False
    ):
        status = (
            "Not training-ready — target absence semantics are not verified"
        )
    elif candidate_presence == 0 or candidate_absence == 0:
        status = (
            "Not training-ready — both classes must survive environmental QC"
        )
    elif len(candidate_rows) == 0:
        status = (
            "Not training-ready — no rows have the complete core predictor set"
        )
    else:
        status = (
            "Candidate training table passes structural QC; sampling design, survey dependence and leakage controls must still be reviewed before modelling"
        )

    return {
        "summary": summary,
        "predictor_completeness": predictor_completeness,
        "candidate_rows": candidate_rows,
        "annotated": work,
        "status": status,
        "candidate_presence": candidate_presence,
        "candidate_absence": candidate_absence,
    }


# ============================================================
# V2 STEP 2 — HISTORICAL ENVIRONMENTAL MATCHING
# ============================================================
def tile_key(
    lat,
    lon,
    tile_size=0.5
):
    return (
        math.floor(
            float(lat)
            /
            tile_size
        )
        *
        tile_size,

        math.floor(
            float(lon)
            /
            tile_size
        )
        *
        tile_size
    )


def safe_bbox(
    rows,
    margin=0.20
):
    min_lon = max(
        -179.99,
        float(
            rows[
                "lon"
            ].min()
        )
        -
        margin
    )

    max_lon = min(
        179.99,
        float(
            rows[
                "lon"
            ].max()
        )
        +
        margin
    )

    min_lat = max(
        -79.9,
        float(
            rows[
                "lat"
            ].min()
        )
        -
        margin
    )

    max_lat = min(
        89.9,
        float(
            rows[
                "lat"
            ].max()
        )
        +
        margin
    )

    return (
        min_lon,
        max_lon,
        min_lat,
        max_lat
    )


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_historical_physics_group(
    date_string,
    min_lon,
    max_lon,
    min_lat,
    max_lat
):
    ds = copernicus_subset_to_file(
        HIST_PHY_DATASET,
        [
            "thetao",
            "so"
        ],
        f"{date_string}T00:00:00",
        f"{date_string}T23:59:59",
        (
            min_lon,
            max_lon,
            min_lat,
            max_lat
        ),
        f"hist_phy_{date_string}.nc",
        0,
        1
    )

    frames = []

    for var, name in (
        (
            "thetao",
            "historical_sst_c"
        ),
        (
            "so",
            "historical_salinity_psu"
        )
    ):

        if var not in ds.data_vars:
            raise RuntimeError(
                f"Historical physics variable '{var}' not found."
            )

        da = ds[
            var
        ]

        if "depth" in da.dims:
            da = da.isel(
                depth=0
            )

        frame = normalize_xy_columns(
            da.to_dataframe(
                name=name
            ).reset_index()
        )

        frame[
            "lat"
        ] = pd.to_numeric(
            frame[
                "lat"
            ],
            errors="coerce"
        )

        frame[
            "lon"
        ] = pd.to_numeric(
            frame[
                "lon"
            ],
            errors="coerce"
        )

        frame = frame.dropna(
            subset=[
                "lat",
                "lon",
                name
            ]
        )

        frames.append(
            frame[
                [
                    "lat",
                    "lon",
                    name
                ]
            ]
        )

    temp, sal = frames

    temp[
        "lat_key"
    ] = temp[
        "lat"
    ].round(
        4
    )

    temp[
        "lon_key"
    ] = temp[
        "lon"
    ].round(
        4
    )

    sal[
        "lat_key"
    ] = sal[
        "lat"
    ].round(
        4
    )

    sal[
        "lon_key"
    ] = sal[
        "lon"
    ].round(
        4
    )

    merged = temp.merge(
        sal[
            [
                "lat_key",
                "lon_key",
                "historical_salinity_psu"
            ]
        ],
        on=[
            "lat_key",
            "lon_key"
        ],
        how="inner"
    )

    return merged[
        [
            "lat",
            "lon",
            "historical_sst_c",
            "historical_salinity_psu"
        ]
    ]


@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_historical_bgc_group(
    date_string,
    min_lon,
    max_lon,
    min_lat,
    max_lat
):
    ds = copernicus_subset_to_file(
        HIST_BGC_DATASET,
        ["o2"],
        f"{date_string}T00:00:00",
        f"{date_string}T23:59:59",
        (
            min_lon,
            max_lon,
            min_lat,
            max_lat
        ),
        f"hist_bgc_{date_string}.nc"
    )

    if "o2" not in ds.data_vars:
        raise RuntimeError(
            "Historical BGC variable 'o2' not found."
        )

    df = normalize_xy_columns(
        ds[
            "o2"
        ]
        .to_dataframe(
            name="oxygen_mmol_m3"
        )
        .reset_index()
    )

    required = [
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Historical BGC missing columns: {missing}"
        )

    for c in [
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]:

        df[
            c
        ] = pd.to_numeric(
            df[
                c
            ],
            errors="coerce"
        )

    df = df.dropna(
        subset=
            required
    ).copy()

    df[
        "oxygen_ml_l"
    ] = (
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
            "oxygen_ml_l"
        ]
    ]


def nearest_row(
    target_lat,
    target_lon,
    source
):
    if source.empty:
        return (
            None,
            np.nan
        )

    target = pd.DataFrame(
        {
            "lat":
                [
                    float(
                        target_lat
                    )
                ],

            "lon":
                [
                    float(
                        target_lon
                    )
                ]
        }
    )

    idx, dist = nearest_source_indices(
        target,
        source[
            [
                "lat",
                "lon"
            ]
        ]
    )

    return (
        source.iloc[
            int(
                idx[0]
            )
        ],
        float(
            dist[0]
        )
    )


def nearest_profile(
    target_lat,
    target_lon,
    source
):
    grid = (
        source[
            [
                "lat",
                "lon"
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    row, dist = nearest_row(
        target_lat,
        target_lon,
        grid
    )

    if row is None:
        return (
            pd.DataFrame(),
            np.nan,
            np.nan,
            np.nan
        )

    slat = float(
        row[
            "lat"
        ]
    )

    slon = float(
        row[
            "lon"
        ]
    )

    profile = source[
        (
            source[
                "lat"
            ]
            ==
            slat
        )
        &
        (
            source[
                "lon"
            ]
            ==
            slon
        )
    ].sort_values(
        "depth"
    ).copy()

    return (
        profile,
        dist,
        slat,
        slon
    )


def historical_static_features(
    rows
):
    out = (
        rows
        .copy()
        .reset_index(
            drop=True
        )
    )

    points = tuple(
        (
            float(r.lat),
            float(r.lon)
        )
        for r in out.itertuples(
            index=False
        )
    )

    elevations = get_gebco_batch(
        points
    )

    out[
        "gebco_elevation_m"
    ] = elevations

    out[
        "historical_water_depth_m"
    ] = np.where(
        np.asarray(
            elevations
        )
        <
        0,

        -
        np.asarray(
            elevations
        ),

        np.nan
    )

    out[
        "historical_distance_coast_km"
    ] = np.nan

    out[
        "historical_distance_coast_nm"
    ] = np.nan

    out[
        [
            "tile_lat",
            "tile_lon"
        ]
    ] = out.apply(
        lambda r:
            pd.Series(
                tile_key(
                    r[
                        "lat"
                    ],
                    r[
                        "lon"
                    ]
                )
            ),
        axis=1
    )

    for (_, _), group in out.groupby(
        [
            "tile_lat",
            "tile_lon"
        ],
        sort=False
    ):

        west = (
            float(
                group[
                    "lon"
                ].min()
            )
            -
            0.8
        )

        east = (
            float(
                group[
                    "lon"
                ].max()
            )
            +
            0.8
        )

        south = (
            float(
                group[
                    "lat"
                ].min()
            )
            -
            0.8
        )

        north = (
            float(
                group[
                    "lat"
                ].max()
            )
            +
            0.8
        )

        try:
            lines = load_noaa_coastline_lines(
                west,
                east,
                south,
                north
            )

            distances = [
                distance_to_nearest_coast_km(
                    float(r.lat),
                    float(r.lon),
                    lines
                )
                for r in group.itertuples(
                    index=False
                )
            ]

            out.loc[
                group.index,
                "historical_distance_coast_km"
            ] = distances

            out.loc[
                group.index,
                "historical_distance_coast_nm"
            ] = (
                np.asarray(
                    distances
                )
                /
                KM_PER_NAUTICAL_MILE
            )

        except Exception:
            pass

    return out.drop(
        columns=[
            "tile_lat",
            "tile_lon"
        ]
    )


def match_historical_environment(
    observations
):
    if observations.empty:
        return observations.copy()

    work = (
        observations
        .copy()
        .reset_index(
            drop=True
        )
    )

    work[
        "match_status"
    ] = "Pending"

    work[
        "historical_sst_c"
    ] = np.nan

    work[
        "historical_salinity_psu"
    ] = np.nan

    work[
        "historical_surface_o2_ml_l"
    ] = np.nan

    work[
        "historical_oxycline_m"
    ] = np.nan

    work[
        "historical_raw_oxycline_m"
    ] = np.nan

    work[
        "historical_water_depth_m"
    ] = np.nan

    work[
        "historical_distance_coast_km"
    ] = np.nan

    work[
        "historical_distance_coast_nm"
    ] = np.nan

    work[
        "physics_match_distance_km"
    ] = np.nan

    work[
        "bgc_match_distance_km"
    ] = np.nan

    work[
        "historical_oxycline_status"
    ] = None

    eligible_mask = (
        pd.to_datetime(
            work[
                "date"
            ]
        )
        >=
        HISTORICAL_COPERNICUS_START
    )

    work.loc[
        ~eligible_mask,
        "match_status"
    ] = (
        "No Copernicus match — date before 1993-01-01"
    )

    eligible = work[
        eligible_mask
    ].copy()

    if eligible.empty:
        return work

    static = historical_static_features(
        eligible[
            [
                "lat",
                "lon"
            ]
        ].copy()
    )

    for col in [
        "historical_water_depth_m",
        "historical_distance_coast_km",
        "historical_distance_coast_nm"
    ]:

        work.loc[
            eligible.index,
            col
        ] = static[
            col
        ].to_numpy()

    eligible[
        "match_date"
    ] = pd.to_datetime(
        eligible[
            "date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    eligible[
        [
            "tile_lat",
            "tile_lon"
        ]
    ] = eligible.apply(
        lambda r:
            pd.Series(
                tile_key(
                    r[
                        "lat"
                    ],
                    r[
                        "lon"
                    ]
                )
            ),
        axis=1
    )

    groups = eligible.groupby(
        [
            "match_date",
            "tile_lat",
            "tile_lon"
        ],
        sort=True
    )

    for (
        date_string,
        _,
        _
    ), group in groups:

        bbox = safe_bbox(
            group
        )

        (
            min_lon,
            max_lon,
            min_lat,
            max_lat
        ) = bbox

        try:
            physics = (
                load_historical_physics_group(
                    date_string,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat
                )
            )

            bgc = (
                load_historical_bgc_group(
                    date_string,
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat
                )
            )

        except Exception as exc:

            work.loc[
                group.index,
                "match_status"
            ] = (
                "Copernicus request failed: "
                f"{str(exc)[:120]}"
            )

            continue

        for idx in group.index:

            lat = float(
                work.at[
                    idx,
                    "lat"
                ]
            )

            lon = float(
                work.at[
                    idx,
                    "lon"
                ]
            )

            physics_row, physics_dist = (
                nearest_row(
                    lat,
                    lon,
                    physics
                )
            )

            (
                profile,
                bgc_dist,
                _,
                _
            ) = nearest_profile(
                lat,
                lon,
                bgc
            )

            if (
                physics_row is None
                or
                profile.empty
            ):

                work.at[
                    idx,
                    "match_status"
                ] = (
                    "No historical environmental grid match"
                )

                continue

            work.at[
                idx,
                "historical_sst_c"
            ] = float(
                physics_row[
                    "historical_sst_c"
                ]
            )

            work.at[
                idx,
                "historical_salinity_psu"
            ] = float(
                physics_row[
                    "historical_salinity_psu"
                ]
            )

            work.at[
                idx,
                "physics_match_distance_km"
            ] = physics_dist

            work.at[
                idx,
                "bgc_match_distance_km"
            ] = bgc_dist

            surface_row = (
                profile
                .sort_values(
                    "depth"
                )
                .iloc[0]
            )

            work.at[
                idx,
                "historical_surface_o2_ml_l"
            ] = float(
                surface_row[
                    "oxygen_ml_l"
                ]
            )

            raw_depth, _ = (
                estimate_oxycline_depth(
                    profile
                )
            )

            work.at[
                idx,
                "historical_raw_oxycline_m"
            ] = raw_depth

            local_depth = work.at[
                idx,
                "historical_water_depth_m"
            ]

            if pd.isna(
                local_depth
            ):

                work.at[
                    idx,
                    "historical_oxycline_status"
                ] = (
                    "No valid GEBCO water depth"
                )

            else:

                valid_profile = profile[
                    profile[
                        "depth"
                    ]
                    <=
                    float(
                        local_depth
                    )
                ].copy()

                oxy_depth, oxy_status = (
                    estimate_oxycline_depth(
                        valid_profile
                    )
                )

                work.at[
                    idx,
                    "historical_oxycline_m"
                ] = oxy_depth

                if pd.notna(
                    oxy_depth
                ):

                    work.at[
                        idx,
                        "historical_oxycline_status"
                    ] = (
                        "Accepted within local water column"
                    )

                elif (
                    pd.notna(
                        raw_depth
                    )
                    and
                    float(
                        raw_depth
                    )
                    >
                    float(
                        local_depth
                    )
                ):

                    work.at[
                        idx,
                        "historical_oxycline_status"
                    ] = (
                        "Raw crossing below seabed; "
                        "local threshold not reached"
                    )

                else:

                    work.at[
                        idx,
                        "historical_oxycline_status"
                    ] = oxy_status

            work.at[
                idx,
                "match_status"
            ] = "Matched"

    dates = pd.to_datetime(
        work[
            "date"
        ]
    )

    month = dates.dt.month.astype(
        "Int64"
    )

    angle = (
        2
        *
        np.pi
        *
        (
            month.astype(
                float
            )
            -
            1
        )
        /
        12.0
    )

    work[
        "month_number"
    ] = month

    work[
        "month_sin"
    ] = np.sin(
        angle
    )

    work[
        "month_cos"
    ] = np.cos(
        angle
    )

    work[
        "austral_season"
    ] = month.apply(
        lambda m:
            austral_season(
                int(m)
            )
            if pd.notna(m)
            else None
    )

    return work


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.header(
        "E-AWARE Controls"
    )

    st.markdown(
        "### Study Area"
    )

    st.write(
        "Costanera Corridor"
    )

    st.write(
        "• Lobitos"
    )

    st.write(
        "• Siches"
    )

    st.write(
        "• Piedritas"
    )

    st.divider()

    selected_date = st.date_input(
        "Ocean Date",
        value=
            date.today()
    )

    st.caption(
        "E-AWARE requests the selected day "
        "plus the following 72 hours."
    )

    if st.button(
        "🔄 Refresh Live Data",
        width="stretch"
    ):

        st.cache_data.clear()

        st.session_state.pop(
            "historical_matched",
            None
        )

    st.divider()

    st.markdown(
        "### V2 Observation Data"
    )

    if (
        OBSERVATION_REPO_PATH.exists()
        or
        LEGACY_OBSERVATION_REPO_PATH.exists()
    ):
        st.success(
            "Repository observation dataset detected"
        )
    else:
        st.warning(
            "No repository observation dataset detected"
        )

    st.caption(
        "E-AWARE loads the repository dataset automatically. "
        "Dataset replacement and large-scale preprocessing are handled "
        "outside the live dashboard."
    )


# ============================================================
# LIVE ENVIRONMENTAL LOAD
# ============================================================
end_date = (
    selected_date
    +
    timedelta(
        days=3
    )
)

live_load_stage = "Initialisation"


def _eaware_test_live_failure(stage):
    """
    Controlled failure hook used only for engineering tests.

    It has no effect unless EAWARE_TEST_FAILURE_STAGE is
    deliberately supplied as an environment variable.
    """

    import os

    requested = os.environ.get(
        "EAWARE_TEST_FAILURE_STAGE",
        ""
    ).strip()

    if (
        requested
        and
        requested.casefold()
        ==
        stage.casefold()
    ):
        raise RuntimeError(
            f"Simulated E-AWARE failure at stage: {stage}"
        )


try:

    with st.status(
        "Loading E-AWARE environmental data...",
        expanded=True
    ) as status:

        live_load_stage = "Copernicus credentials"

        _eaware_test_live_failure(
            live_load_stage
        )

        username, password = (
            get_credentials()
        )

        status.write(
            "🔐 Checking Copernicus credentials..."
        )

        live_load_stage = "Copernicus authentication"

        _eaware_test_live_failure(
            live_load_stage
        )

        valid = copernicusmarine.login(
            username=
                username,
            password=
                password,
            check_credentials_valid=
                True
        )

        if not valid:
            raise RuntimeError(
                "Copernicus rejected the supplied credentials."
            )

        status.write(
            "🌡️ 1/7 Temperature"
        )

        live_load_stage = "Temperature"

        _eaware_test_live_failure(
            live_load_stage
        )

        temperature_data = (
            load_temperature(
                selected_date,
                end_date
            )
        )

        status.write(
            "🧂 2/7 Salinity"
        )

        live_load_stage = "Salinity"

        _eaware_test_live_failure(
            live_load_stage
        )

        salinity_data = (
            load_salinity(
                selected_date,
                end_date
            )
        )

        status.write(
            "🌊 3/7 SST anomaly"
        )

        live_load_stage = "SST anomaly"

        _eaware_test_live_failure(
            live_load_stage
        )

        anomaly_data = (
            load_sst_anomaly(
                selected_date,
                end_date
            )
        )

        status.write(
            "🫧 4/7 3D oxygen profile"
        )

        live_load_stage = "Oxygen profile"

        _eaware_test_live_failure(
            live_load_stage
        )

        oxygen_profile_data = (
            load_oxygen_profile(
                selected_date,
                end_date
            )
        )

        oxygen_surface = (
            surface_oxygen_from_profile(
                oxygen_profile_data
            )
        )

        ocean = merge_physics_data(
            temperature_data,
            salinity_data,
            anomaly_data
        )

        ocean = attach_nearest_oxygen(
            ocean,
            oxygen_surface
        )

        status.write(
            "🗻 5/7 GEBCO bathymetry"
        )

        live_load_stage = "GEBCO bathymetry"

        _eaware_test_live_failure(
            live_load_stage
        )

        ocean = attach_gebco_bathymetry(
            ocean
        )

        status.write(
            "📏 6/7 Distance from coast"
        )

        live_load_stage = "Distance from coast"

        _eaware_test_live_failure(
            live_load_stage
        )

        ocean = attach_distance_to_coast(
            ocean
        )

        # Operational E-AWARE coastal corridor.
        # The larger bounding box is used for data retrieval,
        # but habitat analysis is limited to ocean cells within
        # 50 nautical miles of the coastline.
        ocean = ocean[
            ocean["distance_coast_nm"]
            <= MAX_OPERATIONAL_DISTANCE_NM
        ].copy()

        if ocean.empty:
            raise RuntimeError(
                "No valid ocean cells remain within the "
                "50 nautical mile E-AWARE coastal corridor."
            )

        status.write(
            "🫧 7/7 Seabed-constrained oxycline"
        )

        live_load_stage = "Seabed-constrained oxycline"

        _eaware_test_live_failure(
            live_load_stage
        )

        ocean = attach_seabed_constrained_oxycline(
            ocean,
            oxygen_profile_data
        )

        ocean = add_temporal_features(
            ocean
        )

        status.update(
            label=
                "✅ E-AWARE live environmental data loaded",
            state=
                "complete",
            expanded=
                False
        )

except Exception as error:

    import json as _failure_json
    import os as _failure_os
    from pathlib import Path as _FailurePath
    from datetime import datetime as _FailureDateTime
    from datetime import timezone as _FailureTimezone

    failure_message = (
        f"E-AWARE live-data failure during "
        f"{live_load_stage}: "
        f"{type(error).__name__}: {error}"
    )

    # Scheduled automation receives an explicit failure marker.
    # This prevents waiting for an advisory that can never be
    # generated and ensures no normal email is sent.
    if (
        _failure_os.environ.get(
            "EAWARE_AUTOMATION_MODE"
        )
        ==
        "1"
    ):

        failure_payload = {
            "generated_at_utc":
                _FailureDateTime.now(
                    _FailureTimezone.utc
                ).isoformat(),

            "stage":
                live_load_stage,

            "error_type":
                type(error).__name__,

            "message":
                str(error),
        }

        _FailurePath(
            "/tmp/eaware_failure.json"
        ).write_text(
            _failure_json.dumps(
                failure_payload,
                indent=2
            ),
            encoding="utf-8"
        )

    st.error(
        "❌ E-AWARE could not load all live environmental data."
    )

    st.warning(
        f"Failure stage: {live_load_stage}"
    )

    st.code(
        str(
            error
        )
    )

    st.caption(
        "No habitat advisory is generated from incomplete "
        "environmental data, and scheduled delivery is stopped."
    )

    st.stop()


# ============================================================
# CURRENT CONDITIONS / MAP
# ============================================================
available_dates = sorted(
    ocean[
        "date"
    ].unique()
)

if not available_dates:

    st.error(
        "No valid ocean dates were returned."
    )

    st.stop()

current_date = (
    selected_date
    if selected_date in available_dates
    else
    available_dates[0]
)

current_data = (
    ocean[
        ocean[
            "date"
        ]
        ==
        current_date
    ]
    .dropna(
        subset=[
            "oxygen_ml_l",
            "water_depth_m",
            "distance_coast_km",
            "distance_coast_nm"
        ]
    )
    .copy()
)

if current_data.empty:

    st.error(
        "No complete environmental grid cells "
        "are available for the selected date."
    )

    st.stop()

current_data = calculate_hsi(
    current_data,
    current_date.month
)

experimental_manifest = (
    load_experimental_model_manifest()
)

current_data, experimental_ml_diagnostics = (
    score_experimental_ml_live(
        current_data,
        experimental_manifest
    )
)

current_data[
    "class"
] = current_data[
    "hsi"
].apply(
    hsi_class
)

best = current_data.loc[
    current_data[
        "hsi"
    ].idxmax()
]

hsi_confidence = (
    assess_primary_hsi_confidence(
        ocean,
        current_date,
        current_data,
        available_dates
    )
)

# No live juvenile biometric feed is connected yet.
# Keep this as None until verified juvenile data are available.
current_juvenile_pct = None

fisheries_protection = (
    assess_fisheries_protection(
        best,
        hsi_confidence,
        juvenile_pct=current_juvenile_pct
    )
)

reference = habitat_reference(
    current_date.month
)

experimental_ml_best = None
experimental_rank_correlation = np.nan
experimental_rank_correlation_reason = None
experimental_best_zone_distance_km = np.nan
experimental_agreement = "Unavailable"

if (
    experimental_ml_diagnostics[
        "available"
    ]
    and
    current_data[
        "experimental_ml_score"
    ].notna().any()
):

    experimental_ml_best = (
        current_data.loc[
            current_data[
                "experimental_ml_score"
            ].idxmax()
        ]
    )

    comparison = current_data[
        [
            "hsi",
            "experimental_ml_score",
        ]
    ].dropna()

    if len(
        comparison
    ) >= 2:

        hsi_rank = (
            comparison[
                "hsi"
            ]
            .rank(
                method="average"
            )
        )

        ml_rank = (
            comparison[
                "experimental_ml_score"
            ]
            .rank(
                method="average"
            )
        )

        if hsi_rank.nunique() < 2:

            experimental_rank_correlation_reason = (
                "HSI values are tied across the scored "
                "live grid cells."
            )

        elif ml_rank.nunique() < 2:

            experimental_rank_correlation_reason = (
                "Experimental ML scores are tied across "
                "the scored live grid cells."
            )

        else:

            experimental_rank_correlation = (
                hsi_rank.corr(
                    ml_rank
                )
            )

    envelope_pct = (
        experimental_ml_diagnostics.get(
            "inside_training_envelope_pct",
            np.nan
        )
    )

    if (
        pd.notna(
            envelope_pct
        )
        and
        envelope_pct <= 0
    ):

        experimental_agreement = (
            "Not interpreted"
        )

    else:

        experimental_agreement = (
            experimental_rank_agreement_label(
                experimental_rank_correlation
            )
        )

    experimental_best_zone_distance_km = (
        haversine(
            float(
                best[
                    "lat"
                ]
            ),
            float(
                best[
                    "lon"
                ]
            ),
            float(
                experimental_ml_best[
                    "lat"
                ]
            ),
            float(
                experimental_ml_best[
                    "lon"
                ]
            ),
        )
    )

coastline_lines = (
    load_noaa_coastline_lines()
)

st.subheader(
    "🌊 Current Ocean Conditions"
)

cols = st.columns(
    4
)

cols[0].metric(
    "Data Date",
    str(
        current_date
    )
)

cols[1].metric(
    "Highest HSI",
    f"{best['hsi']:.2f}"
)

cols[2].metric(
    "Highest-HSI SST",
    f"{best['temperature']:.2f} °C"
)

cols[3].metric(
    "Dissolved Oxygen",
    f"{best['oxygen_ml_l']:.2f} mL/L"
)

cols = st.columns(
    4
)

cols[0].metric(
    "Salinity",
    f"{best['salinity']:.2f} PSU"
)

cols[1].metric(
    "SST Anomaly",
    (
        f"{best['sst_anomaly']:+.2f} °C"
        if pd.notna(
            best[
                "sst_anomaly"
            ]
        )
        else
        "Unavailable"
    )
)

cols[2].metric(
    "GEBCO Seafloor Depth",
    f"{best['water_depth_m']:.0f} m"
)

cols[3].metric(
    "Distance from Coast",
    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)

st.subheader(
    "🧪 Experimental ML Comparison"
)

st.caption(
    "Secondary research layer only. "
    "The experimental conditional model does not change "
    "the HSI, best-zone selection, forecast habitat shift "
    "or fisher SMS."
)

if (
    experimental_ml_diagnostics[
        "available"
    ]
    and
    experimental_ml_best is not None
):

    cols = st.columns(
        4
    )

    envelope_pct = (
        experimental_ml_diagnostics[
            "inside_training_envelope_pct"
        ]
    )

    ml_live_status = (
        "Out-of-envelope"
        if envelope_pct <= 0
        else
        "Exploratory"
    )

    cols[0].metric(
        "Experimental ML Status",
        ml_live_status
    )

    cols[1].metric(
        "HSI ↔ ML Rank Agreement",
        experimental_agreement
    )

    cols[2].metric(
        "Rank Correlation",
        (
            f"{experimental_rank_correlation:.2f}"
            if pd.notna(
                experimental_rank_correlation
            )
            else
            "Unavailable"
        )
    )

    cols[3].metric(
        "Best-Zone Separation",
        (
            f"{experimental_best_zone_distance_km:.2f} km"
            if pd.notna(
                experimental_best_zone_distance_km
            )
            else
            "Unavailable"
        )
    )

    if (
        pd.isna(
            experimental_rank_correlation
        )
        and
        experimental_rank_correlation_reason
    ):

        st.caption(
            "Rank correlation unavailable: "
            +
            experimental_rank_correlation_reason
        )

    cols = st.columns(
        4
    )

    cols[0].metric(
        "ML Best Latitude",
        f"{experimental_ml_best['lat']:.4f}"
    )

    cols[1].metric(
        "ML Best Longitude",
        f"{experimental_ml_best['lon']:.4f}"
    )

    cols[2].metric(
        "Live Cells Scored",
        (
            f"{experimental_ml_diagnostics['cells_scored']:,}"
        )
    )

    cols[3].metric(
        "Inside Training Envelope",
        (
            f"{experimental_ml_diagnostics['inside_training_envelope_pct']:.1f}%"
        )
    )

    ml_best_in_envelope = bool(
        experimental_ml_best[
            "experimental_ml_in_training_envelope"
        ]
    )

    if envelope_pct <= 0:

        st.error(
            "All currently scored live cells are outside "
            "the experimental ML training envelope. "
            "The ML ranking and best-zone separation are "
            "shown for research transparency only and should "
            "not be interpreted as reliable habitat guidance."
        )

    elif ml_best_in_envelope:

        st.success(
            "The experimental ML best-ranked live cell "
            "is inside the environmental range represented "
            "in the model training data."
        )

    else:

        outside_count = int(
            experimental_ml_best[
                "experimental_ml_outside_training_count"
            ]
        )

        st.warning(
            "The experimental ML best-ranked live cell is "
            f"outside the training range for {outside_count} "
            "predictor(s). Treat this experimental ranking "
            "with additional caution."
        )

    with st.expander(
        "🔎 Live vs Experimental ML Training Ranges"
    ):

        envelope_table = pd.DataFrame(
            experimental_ml_diagnostics.get(
                "feature_envelope",
                []
            )
        )

        if not envelope_table.empty:

            st.dataframe(
                envelope_table,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "A cell is outside the training envelope when "
                "at least one live predictor is below the minimum "
                "or above the maximum observed in the final "
                "34-anchor experimental training dataset."
            )

        else:

            st.info(
                "No predictor-range diagnostics are available."
            )


    with st.expander(
        "What does the experimental ML score mean?"
    ):

        st.markdown(
            """
- The model was trained using **34 matched anchoveta presence sets** after strict environmental and bathymetric QC.
- Background locations represent **locally available habitat**, not confirmed fish absence.
- The displayed ML value is ranked **relative to the current live grid**.
- It is **not** a probability that anchoveta are present at a coordinate.
- Year-held-out matched concordance was **0.598** and Top-1 performance was **0.324**.
- Robust spatial transfer could **not** be established because the valid historical observations were strongly geographically concentrated.
- The literature-based **HSI remains the primary habitat model**.
"""
        )

        st.caption(
            "High / Moderate / Low rank-agreement labels are "
            "dashboard diagnostics only. They are not biological "
            "or validated fisheries-management thresholds."
        )

else:

    st.warning(
        "Experimental ML live scoring is unavailable: "
        +
        str(
            experimental_ml_diagnostics.get(
                "reason",
                "unknown reason"
            )
        )
    )


st.subheader(
    "🎯 Primary HSI Confidence"
)

st.caption(
    "Confidence describes how well the current HSI output is "
    "supported by available environmental data. "
    "It is not a probability that fish are present."
)

confidence_cols = st.columns(
    4
)

confidence_cols[0].metric(
    "HSI Confidence",
    hsi_confidence[
        "overall_confidence"
    ]
)

confidence_cols[1].metric(
    "HSI Data Coverage",
    (
        f"{hsi_confidence['coverage_pct']:.0f}%"
    )
)

confidence_cols[2].metric(
    "Data Age",
    (
        f"{hsi_confidence['age_days']} day"
        if hsi_confidence[
            "age_days"
        ] == 1
        else
        f"{hsi_confidence['age_days']} days"
    )
)

confidence_cols[3].metric(
    "Forecast Dates Available",
    hsi_confidence[
        "forecast_date_count"
    ]
)

if (
    hsi_confidence[
        "overall_confidence"
    ]
    ==
    "Low"
):

    st.warning(
        "Primary HSI confidence is LOW. "
        "Treat the current habitat pattern cautiously because "
        "environmental data coverage, freshness or forecast "
        "availability is limited."
    )

else:

    st.info(
        "Primary HSI confidence is capped at MODERATE because "
        "the literature-based HSI has not yet been independently "
        "validated against observed anchoveta distributions."
    )


with st.expander(
    "🔎 How is HSI confidence determined?"
):

    detail_cols = st.columns(
        3
    )

    detail_cols[0].metric(
        "Predictor Completeness",
        hsi_confidence[
            "coverage_label"
        ]
    )

    detail_cols[1].metric(
        "Environmental Freshness",
        hsi_confidence[
            "freshness_label"
        ]
    )

    detail_cols[2].metric(
        "Forecast Coverage",
        hsi_confidence[
            "forecast_label"
        ]
    )

    if pd.notna(
        hsi_confidence[
            "hsi_gap"
        ]
    ):

        st.write(
            "**Spatial diagnostic:** "
            f"best HSI exceeds the median grid-cell HSI by "
            f"{hsi_confidence['hsi_gap']:.2f}. "
            f"{hsi_confidence['near_best_cells']} live grid cell(s) "
            "are within 0.01 HSI of the current best cell."
        )

    st.markdown(
        """
**Confidence rules**

- HSI predictor coverage ≥90% → good data support.
- Coverage 70–89% → partial support.
- Coverage <70% → limited support.
- Environmental data ≤2 days old → current.
- 3 or more current/future forecast dates → good forecast coverage.
- These thresholds are **engineering diagnostics**, not biological thresholds.
- Even with excellent environmental data, overall confidence is currently capped at **Moderate** because the HSI has not yet been independently validated against observed fish distributions.
- Confidence must never be interpreted as probability of anchoveta presence.
"""
    )


st.subheader(
    "🐟 Fisheries Protection & Regulation"
)

st.caption(
    "This panel separates official regulation, juvenile protection "
    "and E-AWARE habitat guidance. E-AWARE does not declare legal "
    "fishing closures."
)

protection_cols = st.columns(
    3
)

protection_cols[0].metric(
    "Official Fishing Status",
    fisheries_protection[
        "official_status"
    ]
)

protection_cols[1].metric(
    "Juvenile Protection",
    fisheries_protection[
        "juvenile_status"
    ]
)

ecological_display = {
    "LOW HABITAT SUITABILITY": "LOW",
    "MODERATE HABITAT SUITABILITY": "MODERATE",
    "HIGH HABITAT SUITABILITY": "HIGH",
    "VERY HIGH HABITAT SUITABILITY": "VERY HIGH",
    "USE WITH CAUTION": "CAUTION",
    "UNAVAILABLE": "UNAVAILABLE",
}.get(
    fisheries_protection[
        "ecological_status"
    ],
    fisheries_protection[
        "ecological_status"
    ]
)

protection_cols[2].metric(
    "E-AWARE Ecological Advisory",
    ecological_display
)

st.warning(
    "Official fishing status is NOT live-verified. "
    "Before fishing, users must check current PRODUCE regulations, "
    "temporary suspensions and local restrictions."
)

with st.expander(
    "🛡️ Fisheries protection details"
):

    st.markdown(
        f"""
### Official regulatory status

**{fisheries_protection['official_status']}**

{fisheries_protection['official_detail']}

### Juvenile protection

**{fisheries_protection['juvenile_status']}**

{fisheries_protection['juvenile_detail']}

Prototype regulatory reference used by E-AWARE:

- Anchoveta minimum total length: **{fisheries_protection['minimum_length_cm']:.0f} cm**
- Maximum juvenile tolerance reference: **{fisheries_protection['juvenile_tolerance_pct']:.0f}%**
- Regulatory authority: **{fisheries_protection['regulatory_authority']}**

### E-AWARE ecological interpretation

**{fisheries_protection['ecological_status']}**

{fisheries_protection['ecological_detail']}

E-AWARE provides **habitat decision support only**. Habitat suitability, predicted movement and environmental conditions must never be interpreted as legal permission to fish.
"""
    )


st.subheader(
    "🗓️ Temporal / Seasonal Predictor"
)

cols = st.columns(
    5
)

cols[0].metric(
    "Month",
    (
        f"{best['month_name']} "
        f"({int(best['month_number'])})"
    )
)

cols[1].metric(
    "Austral Season",
    str(
        best[
            "austral_season"
        ]
    )
)

cols[2].metric(
    "Month sin",
    f"{float(best['month_sin']):.3f}"
)

cols[3].metric(
    "Month cos",
    f"{float(best['month_cos']):.3f}"
)

cols[4].metric(
    "HSI Weight",
    "Not included"
)

st.subheader(
    "🗺️ Anchoveta Habitat Map"
)

map_mode = st.radio(
    "Map Layer",
    [
        "Habitat Suitability",
        "Temperature",
        "Salinity",
        "Dissolved Oxygen",
        "Bathymetry",
        "Distance from Coast",
        "SST Anomaly"
    ],
    horizontal=True,
)

map_data = (
    current_data.copy()
)


def scaled(
    values
):
    values = values.astype(
        float
    )

    spread = max(
        float(
            values.max()
            -
            values.min()
        ),
        1e-9
    )

    return (
        values
        -
        values.min()
    ) / spread


if map_mode == "Habitat Suitability":

    map_data[
        "r"
    ] = (
        1
        -
        map_data[
            "hsi"
        ]
    ) * 220

    map_data[
        "g"
    ] = (
        map_data[
            "hsi"
        ]
        *
        210
    )

    map_data[
        "b"
    ] = 70

elif map_mode == "Temperature":

    n = scaled(
        map_data[
            "temperature"
        ]
    )

    map_data[
        "r"
    ] = (
        80
        +
        170
        *
        n
    )

    map_data[
        "g"
    ] = 80

    map_data[
        "b"
    ] = (
        220
        -
        150
        *
        n
    )

elif map_mode == "Salinity":

    n = scaled(
        map_data[
            "salinity"
        ]
    )

    map_data[
        "r"
    ] = 50

    map_data[
        "g"
    ] = (
        100
        +
        120
        *
        n
    )

    map_data[
        "b"
    ] = 220

elif map_mode == "Dissolved Oxygen":

    n = scaled(
        map_data[
            "oxygen_ml_l"
        ]
    )

    map_data[
        "r"
    ] = 80

    map_data[
        "g"
    ] = (
        130
        +
        100
        *
        n
    )

    map_data[
        "b"
    ] = (
        230
        -
        80
        *
        n
    )

elif map_mode == "Bathymetry":

    n = scaled(
        map_data[
            "water_depth_m"
        ]
    )

    map_data[
        "r"
    ] = (
        70
        -
        50
        *
        n
    )

    map_data[
        "g"
    ] = (
        180
        -
        110
        *
        n
    )

    map_data[
        "b"
    ] = (
        210
        +
        40
        *
        n
    )

elif map_mode == "Distance from Coast":

    n = scaled(
        map_data[
            "distance_coast_km"
        ]
    )

    map_data[
        "r"
    ] = (
        60
        +
        150
        *
        n
    )

    map_data[
        "g"
    ] = (
        190
        -
        100
        *
        n
    )

    map_data[
        "b"
    ] = (
        220
        -
        80
        *
        n
    )

else:

    vals = map_data[
        "sst_anomaly"
    ].fillna(
        0
    )

    spread = max(
        abs(
            float(
                vals.min()
            )
        ),
        abs(
            float(
                vals.max()
            )
        ),
        0.1
    )

    n = (
        vals
        /
        spread
        +
        1
    ) / 2

    map_data[
        "r"
    ] = (
        40
        +
        200
        *
        n
    )

    map_data[
        "g"
    ] = 100

    map_data[
        "b"
    ] = (
        240
        -
        190
        *
        n
    )

for (
    source,
    display,
    decimals
) in [
    (
        "hsi",
        "hsi_display",
        2
    ),
    (
        "temperature",
        "temperature_display",
        2
    ),
    (
        "salinity",
        "salinity_display",
        2
    ),
    (
        "oxygen_ml_l",
        "oxygen_display",
        2
    ),
    (
        "water_depth_m",
        "depth_display",
        0
    ),
    (
        "distance_coast_km",
        "coast_km_display",
        2
    ),
    (
        "distance_coast_nm",
        "coast_nm_display",
        2
    ),
    (
        "sst_anomaly",
        "anomaly_display",
        2
    ),
]:

    map_data[
        display
    ] = map_data[
        source
    ].round(
        decimals
    )

ocean_layer = pdk.Layer(
    "ScatterplotLayer",
    data=
        map_data,
    get_position=
        [
            "lon",
            "lat"
        ],
    get_fill_color=
        [
            "r",
            "g",
            "b",
            190
        ],
    get_radius=
        700,
    radius_min_pixels=
        6,
    radius_max_pixels=
        18,
    pickable=
        True
)

community_layer = pdk.Layer(
    "ScatterplotLayer",
    data=
        COMMUNITIES,
    get_position=
        [
            "lon",
            "lat"
        ],
    get_fill_color=
        [
            255,
            255,
            255,
            240
        ],
    get_line_color=
        [
            20,
            20,
            20,
            255
        ],
    stroked=
        True,
    line_width_min_pixels=
        2,
    get_radius=
        350,
    pickable=
        True
)

# Automatically frame the valid live E-AWARE ocean cells.
map_center_lat = float(map_data["lat"].mean())
map_center_lon = float(map_data["lon"].mean())

map_lat_span = float(
    map_data["lat"].max()
    -
    map_data["lat"].min()
)

map_lon_span = float(
    map_data["lon"].max()
    -
    map_data["lon"].min()
)

map_span = max(
    map_lat_span,
    map_lon_span
)

if map_span >= 0.90:
    map_zoom = 7.8
elif map_span >= 0.60:
    map_zoom = 8.2
elif map_span >= 0.35:
    map_zoom = 8.7
else:
    map_zoom = 9.3

st.pydeck_chart(
    pdk.Deck(
        map_style=
            BASEMAP_URL,

        initial_view_state=
            pdk.ViewState(
                latitude=
                    map_center_lat,
                longitude=
                    map_center_lon,
                zoom=
                    map_zoom
            ),

        layers=[
            ocean_layer,
            community_layer
        ],

        tooltip={
            "html":
                "<b>{class} Habitat</b><br/>"
                "HSI: {hsi_display}<br/>"
                "SST: {temperature_display} °C<br/>"
                "Salinity: {salinity_display} PSU<br/>"
                "Oxygen: {oxygen_display} mL/L<br/>"
                "Water depth: {depth_display} m<br/>"
                "Distance from coast: "
                "{coast_km_display} km / "
                "{coast_nm_display} nm<br/>"
                "SST anomaly: {anomaly_display} °C"
        },
    ),
    width="stretch"
)


# ============================================================
# OXYCLINE STATUS
# ============================================================
st.subheader(
    "🫧 Seabed-Constrained Oxycline"
)

raw_oxycline = best.get(
    "raw_oxycline_depth_m",
    np.nan
)

local_oxycline = best.get(
    "oxycline_depth_m",
    np.nan
)

local_maximum = best.get(
    "local_profile_max_depth_m",
    np.nan
)

local_levels = best.get(
    "local_profile_levels",
    np.nan
)

local_gap = best.get(
    "local_depth_gap_m",
    np.nan
)

cols = st.columns(
    4
)

cols[0].metric(
    "Raw Full-Profile Oxycline",
    (
        f"{raw_oxycline:.1f} m"
        if pd.notna(
            raw_oxycline
        )
        else
        "Not detected"
    )
)

cols[1].metric(
    "Local GEBCO Water Depth",
    f"{best['water_depth_m']:.1f} m"
)

cols[2].metric(
    "Local Oxycline",
    (
        f"{local_oxycline:.1f} m"
        if pd.notna(
            local_oxycline
        )
        else
        "Not reached"
    )
)

cols[3].metric(
    "HSI Weight",
    "Not included"
)

cols = st.columns(
    3
)

cols[0].metric(
    "Deepest BGC Level Used",
    (
        f"{local_maximum:.1f} m"
        if pd.notna(
            local_maximum
        )
        else
        "Unavailable"
    )
)

cols[1].metric(
    "BGC Levels Inside Water Column",
    (
        f"{int(local_levels)}"
        if pd.notna(
            local_levels
        )
        else
        "Unavailable"
    )
)

cols[2].metric(
    "Unsampled Gap to Seabed",
    (
        f"{local_gap:.1f} m"
        if pd.notna(
            local_gap
        )
        else
        "Unavailable"
    )
)

st.info(
    str(
        best.get(
            "oxycline_validation_status",
            "Unavailable"
        )
    )
)


# ============================================================
# HIGHEST HABITAT / FORECAST SHIFT
# ============================================================
st.subheader(
    "📍 Highest HSI Cell Within Configured Corridor"
)

cols = st.columns(
    5
)

cols[0].metric(
    "Latitude",
    f"{best['lat']:.4f}"
)

cols[1].metric(
    "Longitude",
    f"{best['lon']:.4f}"
)

cols[2].metric(
    "Suitability",
    hsi_class(
        best[
            "hsi"
        ]
    )
)

cols[3].metric(
    "GEBCO Seafloor Depth",
    f"{best['water_depth_m']:.0f} m"
)

cols[4].metric(
    "Coast Distance",
    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)

st.caption(
    "This is the grid cell with the highest relative HSI inside the "
    "configured E-AWARE corridor. GEBCO depth is seabed bathymetry beneath "
    "the cell and is not a predicted anchoveta swimming or fishing depth."
)

st.subheader(
    "🐟 Literature-Based HSI Reference Envelope"
)

cols = st.columns(
    4
)

cols[0].metric(
    "Reference Temperature",
    (
        f"{reference['temp_min']} – "
        f"{reference['temp_max']} °C"
    )
)

cols[1].metric(
    "Reference Salinity",
    (
        f"{reference['sal_min']} – "
        f"{reference['sal_max']} PSU"
    )
)

cols[2].metric(
    "Reference Oxygen",
    (
        f"{reference['oxygen_min']} – "
        f"{reference['oxygen_max']} mL/L"
    )
)

cols[3].metric(
    "Season",
    reference[
        "season"
    ]
)

st.info(
    "The primary HSI uses temperature, salinity and "
    "surface dissolved oxygen from the literature-based "
    "reference ranges. The experimental ML layer is displayed "
    "separately and does not alter the HSI forecast."
)

forecast_centres = []
forecast_results = []

for forecast_date in available_dates:

    frame = ocean[
        ocean[
            "date"
        ]
        ==
        forecast_date
    ].dropna(
        subset=[
            "oxygen_ml_l",
            "water_depth_m",
            "distance_coast_km",
            "distance_coast_nm"
        ]
    ).copy()

    if frame.empty:
        continue

    frame = calculate_hsi(
        frame,
        forecast_date.month
    )

    centre = habitat_centroid(
        frame
    )

    coast_km = None
    coast_nm = None

    if centre:

        coast_km = (
            distance_to_nearest_coast_km(
                centre[0],
                centre[1],
                coastline_lines
            )
        )

        coast_nm = (
            coast_km
            /
            KM_PER_NAUTICAL_MILE
        )

        forecast_centres.append(
            {
                "date":
                    forecast_date,

                "lat":
                    centre[0],

                "lon":
                    centre[1],

                "distance_coast_km":
                    coast_km,

                "distance_coast_nm":
                    coast_nm
            }
        )

    bf = frame.loc[
        frame[
            "hsi"
        ].idxmax()
    ]

    forecast_results.append(
        {
            "Date":
                forecast_date,

            "Month":
                str(
                    bf[
                        "month_name"
                    ]
                ),

            "Austral Season":
                str(
                    bf[
                        "austral_season"
                    ]
                ),

            "Highest HSI":
                round(
                    float(
                        bf[
                            "hsi"
                        ]
                    ),
                    2
                ),

            "Temperature °C":
                round(
                    float(
                        bf[
                            "temperature"
                        ]
                    ),
                    2
                ),

            "Salinity PSU":
                round(
                    float(
                        bf[
                            "salinity"
                        ]
                    ),
                    2
                ),

            "Oxygen mL/L":
                round(
                    float(
                        bf[
                            "oxygen_ml_l"
                        ]
                    ),
                    2
                ),

            "Water Depth m":
                round(
                    float(
                        bf[
                            "water_depth_m"
                        ]
                    ),
                    0
                ),

            "Distance Coast km":
                round(
                    float(
                        bf[
                            "distance_coast_km"
                        ]
                    ),
                    2
                ),

            "Distance Coast nm":
                round(
                    float(
                        bf[
                            "distance_coast_nm"
                        ]
                    ),
                    2
                ),

            "Local Oxycline m":
                (
                    round(
                        float(
                            bf[
                                "oxycline_depth_m"
                            ]
                        ),
                        1
                    )
                    if pd.notna(
                        bf[
                            "oxycline_depth_m"
                        ]
                    )
                    else
                    None
                ),

            "SST Anomaly °C":
                (
                    round(
                        float(
                            bf[
                                "sst_anomaly"
                            ]
                        ),
                        2
                    )
                    if pd.notna(
                        bf[
                            "sst_anomaly"
                        ]
                    )
                    else
                    None
                ),

            "Habitat Centre Lat":
                (
                    round(
                        centre[0],
                        4
                    )
                    if centre
                    else
                    None
                ),

            "Habitat Centre Lon":
                (
                    round(
                        centre[1],
                        4
                    )
                    if centre
                    else
                    None
                ),

            "Centre Coast km":
                (
                    round(
                        coast_km,
                        2
                    )
                    if coast_km is not None
                    else
                    None
                ),

            "Centre Coast nm":
                (
                    round(
                        coast_nm,
                        2
                    )
                    if coast_nm is not None
                    else
                    None
                ),
        }
    )


# ============================================================
# HSI COMPONENT TRANSPARENCY
# ============================================================
st.markdown(
    "#### 🧮 HSI Component Score Breakdown"
)

hsi_cols = st.columns(4)

hsi_cols[0].metric(
    "Temperature Score",
    f"{float(best['temperature_score']):.2f}"
)

hsi_cols[1].metric(
    "Salinity Score",
    f"{float(best['salinity_score']):.2f}"
)

hsi_cols[2].metric(
    "Oxygen Score",
    f"{float(best['oxygen_score']):.2f}"
)

hsi_cols[3].metric(
    "Final HSI",
    f"{float(best['hsi']):.2f}"
)

st.caption(
    "The primary HSI is the equal-weight mean of the "
    "temperature, salinity and surface dissolved-oxygen scores. "
    "A value outside the central literature reference envelope "
    "does not automatically force the whole HSI to zero because "
    "each environmental variable is scored independently before "
    "the three scores are averaged."
)

with st.expander(
    "How is the HSI calculated?",
    expanded=False
):
    st.markdown(
        """
**Primary HSI formula**

`HSI = (Temperature Score + Salinity Score + Oxygen Score) / 3`

Each component ranges from **0 to 1**.

- **1.00** means the value is inside its literature reference range.
- Values outside the reference range decrease through the existing
  shoulder/taper function.
- Scores are clipped between **0 and 1**.
- Temperature, salinity and oxygen currently have equal weight.
- Bathymetry, coast distance, oxycline and season are not given
  hand-built HSI weights.

This breakdown explains the primary HSI only. It does not make
the HSI a probability that anchoveta are present.
        """
    )


st.subheader(
    "🧭 Predicted Suitable-Habitat Shift"
)

direction_text = "Unavailable"
distance_text = "—"
bearing_text = "—"
period_text = "Insufficient forecast data"
coastal_interpretation = "Unavailable"

start = None
end = None
forecast_hours = None

if len(
    forecast_centres
) >= 2:

    start = forecast_centres[0]
    end = forecast_centres[-1]

    forecast_hours = int(
        round(
            (
                pd.Timestamp(
                    end["date"]
                )
                -
                pd.Timestamp(
                    start["date"]
                )
            ).total_seconds()
            /
            3600.0
        )
    )

    distance = haversine(
        start[
            "lat"
        ],
        start[
            "lon"
        ],
        end[
            "lat"
        ],
        end[
            "lon"
        ]
    )

    bearing = calculate_bearing(
        start[
            "lat"
        ],
        start[
            "lon"
        ],
        end[
            "lat"
        ],
        end[
            "lon"
        ]
    )

    distance_text = (
        f"{distance:.2f} km"
    )

    if distance >= 2.0:

        bearing_text = (
            f"{bearing:.1f}°"
        )

        direction_text = (
            direction_name(
                bearing
            )
        )

    else:

        bearing_text = "—"
        direction_text = "No clear shift"

    period_text = (
        f"{start['date'].strftime('%d %b %Y')} "
        f"→ "
        f"{end['date'].strftime('%d %b %Y')}"
    )

    delta = (
        end[
            "distance_coast_km"
        ]
        -
        start[
            "distance_coast_km"
        ]
    )

    coastal_interpretation = (
        "No clear coastal/offshore change"
        if abs(
            delta
        ) < 0.5
        else
        (
            f"Closer to coast by {abs(delta):.2f} km"
            if delta < 0
            else
            f"Farther offshore by {abs(delta):.2f} km"
        )
    )

cols = st.columns(
    3
)

cols[0].metric(
    "Suitable-Habitat Shift",
    direction_text
)

cols[1].metric(
    "Centroid Shift",
    distance_text
)

cols[2].metric(
    "Bearing",
    bearing_text
)

st.markdown(
    f"**Forecast Period:** {period_text}"
)

st.caption(
    "The reported shift is movement of the HSI-weighted centroid of the "
    "relatively highest-suitability cells, not movement of individual fish. "
    "A large value can occur when relative habitat suitability redistributes "
    "from one part of the study corridor to another over the forecast period."
)

if (
    start
    and
    end
):

    cols = st.columns(
        3
    )

    cols[0].metric(
        "Start Distance from Coast",
        (
            f"{start['distance_coast_km']:.2f} km / "
            f"{start['distance_coast_nm']:.2f} nm"
        )
    )

    cols[1].metric(
        (
            f"+{forecast_hours} h Distance from Coast"
            if forecast_hours is not None
            else
            "Future Distance from Coast"
        ),
        (
            f"{end['distance_coast_km']:.2f} km / "
            f"{end['distance_coast_nm']:.2f} nm"
        )
    )

    cols[2].metric(
        "Interpretation",
        coastal_interpretation
    )

st.subheader(
    (
        f"📅 {forecast_hours}-Hour Habitat Forecast"
        if forecast_hours is not None
        else
        "📅 Habitat Forecast"
    )
)

st.dataframe(
    pd.DataFrame(
        forecast_results
    ),
    width="stretch",
    hide_index=True
)


# ============================================================
# V2 HISTORICAL UI
# ============================================================
st.divider()

st.header(
    "🧬 Anchoveta Observation Dataset"
)

historical_raw, historical_source = (
    read_historical_source()
)

if historical_raw is None:

    st.warning(
        "No repository-managed anchoveta observation dataset is available. "
        "Add a traceable dataset as data/anchoveta_observations.csv. "
        "E-AWARE will not create fake fish records."
    )

else:

    dataset_descriptor = describe_observation_dataset(
        historical_raw,
        historical_source
    )

    st.caption(
        f"Repository file: **{historical_source}**"
    )

    st.info(
        f"**Dataset source:** {dataset_descriptor['source_label']}  \n"
        f"**Dataset role:** {dataset_descriptor['dataset_role']}"
    )

    try:

        detected_presence = find_alias(
            historical_raw.columns,
            "presence"
        )

        detected_abundance = find_alias(
            historical_raw.columns,
            "abundance"
        )

        abundance_zero_is_absence = False

        if (
            detected_presence is None
            and
            detected_abundance is not None
        ):

            st.warning(
                "This file has an abundance field but no explicit "
                "presence/absence field. Positive abundance can confirm "
                "presence, but zero abundance is not automatically treated "
                "as surveyed absence."
            )

            abundance_zero_is_absence = st.checkbox(
                "Dataset documentation confirms that abundance = 0 represents a genuine surveyed absence/non-detection",
                value=False,
                help=(
                    "Enable this only when the survey methodology explicitly "
                    "supports interpreting zero abundance as an absence. "
                    "Do not enable it for ordinary presence-only occurrence data."
                )
            )

        historical, meta = (
            standardise_historical_observations(
                historical_raw,
                abundance_zero_is_absence=
                    abundance_zero_is_absence
            )
        )

        st.subheader(
            "Observation Ingestion & Validation"
        )

        cols = st.columns(
            5
        )

        cols[0].metric(
            "Raw Rows",
            f"{meta['raw_rows']:,}"
        )

        cols[1].metric(
            "Usable Rows",
            f"{meta['usable_rows']:,}"
        )

        cols[2].metric(
            "Presence Known",
            f"{meta['presence_known_rows']:,}"
        )

        cols[3].metric(
            "Inside E-AWARE Area",
            f"{meta['study_area_rows']:,}"
        )

        cols[4].metric(
            "Target Ready",
            (
                "Yes"
                if meta[
                    "training_ready"
                ]
                else
                "No"
            )
        )

        st.caption(
            f"Presence handling: "
            f"{meta['presence_source']}"
        )

        st.caption(
            f"Target method: **{meta['target_method']}** | "
            f"Absence semantics verified: "
            f"**{'Yes' if meta['target_absence_verified'] else 'No'}**"
        )

        mapping_rows = [
            {
                "E-AWARE field":
                    k,

                "Detected source column":
                    (
                        v
                        if v is not None
                        else
                        "Not detected"
                    )
            }

            for k, v in meta[
                "mapping"
            ].items()
        ]

        with st.expander(
            "Detected column mapping"
        ):

            st.dataframe(
                pd.DataFrame(
                    mapping_rows
                ),
                width="stretch",
                hide_index=True
            )

        st.subheader(
            "Observation Data Quality Control"
        )

        observation_qc = observation_qc_audit(
            historical,
            meta
        )

        qc_cols = st.columns(
            4
        )

        qc_cols[0].metric(
            "Presence",
            f"{meta['presence_rows']:,}"
        )

        qc_cols[1].metric(
            "Absence",
            f"{meta['absence_rows']:,}"
        )

        qc_cols[2].metric(
            "Duplicates Flagged",
            f"{len(observation_qc['duplicates']):,}"
        )

        qc_cols[3].metric(
            "Surveys / Cruises",
            (
                f"{observation_qc['survey_count']:,}"
                if observation_qc['survey_count']
                else
                "Not available"
            )
        )

        if meta[
            "training_ready"
        ]:
            st.success(
                observation_qc[
                    "status"
                ]
            )
        else:
            st.warning(
                observation_qc[
                    "status"
                ]
            )

        with st.expander(
            "Open observation QC details",
            expanded=False
        ):

            st.dataframe(
                observation_qc[
                    "summary"
                ],
                width="stretch",
                hide_index=True
            )

            c1, c2 = st.columns(
                2
            )

            with c1:
                st.markdown(
                    "**Target class balance**"
                )
                st.dataframe(
                    observation_qc[
                        "class_balance"
                    ],
                    width="stretch",
                    hide_index=True
                )

            with c2:
                st.markdown(
                    "**Seasonal distribution**"
                )
                st.dataframe(
                    observation_qc[
                        "season_counts"
                    ],
                    width="stretch",
                    hide_index=True
                )

            st.markdown(
                "**Observations by year**"
            )
            st.dataframe(
                observation_qc[
                    "year_counts"
                ],
                width="stretch",
                hide_index=True
            )

            if not observation_qc[
                "duplicates"
            ].empty:
                st.markdown(
                    "**Duplicate records requiring review**"
                )
                st.dataframe(
                    observation_qc[
                        "duplicates"
                    ],
                    width="stretch",
                    hide_index=True
                )

            st.download_button(
                "⬇️ Download observation QC summary",
                data=
                    observation_qc[
                        "summary"
                    ]
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    ),
                file_name=
                    "eaware_v2_step3_observation_qc.csv",
                mime=
                    "text/csv"
            )

        st.subheader(
            "Historical Environmental Matching"
        )

        st.write(
            "Each selected observation is matched to daily historical "
            "Copernicus physics and biogeochemistry at the same date "
            "and nearest grid location. GEBCO depth, coastline distance "
            "and cyclic month features are added independently. "
            "No HSI score or machine-learning prediction is created here."
        )

        pre1993 = int(
            (
                pd.to_datetime(
                    historical[
                        "date"
                    ]
                )
                <
                HISTORICAL_COPERNICUS_START
            ).sum()
        )

        if pre1993:

            st.info(
                f"{pre1993:,} observation(s) are before "
                "1 January 1993 and will be labelled as "
                "having no Copernicus historical match."
            )

        scope = st.radio(
            "Matching scope",
            [
                "Current E-AWARE study area",
                "All validated observations"
            ],
            horizontal=True
        )

        pool = (
            historical[
                historical[
                    "within_eaware_study_area"
                ]
            ].copy()

            if scope
            ==
            "Current E-AWARE study area"

            else

            historical.copy()
        )

        if pool.empty:

            st.warning(
                "There are no validated observations "
                "in the selected matching scope."
            )

        else:

            limit_options = [
                10,
                25,
                50,
                100,
                250
            ]

            max_rows = st.selectbox(
                "Maximum observations to match in this Streamlit run",
                limit_options,
                index=2
            )

            if len(
                pool
            ) > max_rows:

                st.caption(
                    f"{len(pool):,} observations are available; "
                    f"this run will match the first {max_rows:,} "
                    "after sorting by date. Large archives should "
                    "be batch-processed outside the interactive "
                    "dashboard later."
                )

            selected_hist = (
                pool
                .sort_values(
                    [
                        "date",
                        "source_row"
                    ]
                )
                .head(
                    max_rows
                )
                .copy()
            )

            if st.button(
                "🔗 Match Historical Environmental Predictors",
                type="primary",
                width="stretch"
            ):

                with st.status(
                    "Matching historical observations "
                    "to environmental data...",
                    expanded=True
                ) as hist_status:

                    hist_status.write(
                        "Checking dates and Copernicus historical coverage..."
                    )

                    matched = (
                        match_historical_environment(
                            selected_hist
                        )
                    )

                    st.session_state[
                        "historical_matched"
                    ] = matched

                    hist_status.update(
                        label=
                            "✅ Historical environmental matching complete",
                        state=
                            "complete",
                        expanded=
                            False
                    )

            matched = st.session_state.get(
                "historical_matched"
            )

            if (
                isinstance(
                    matched,
                    pd.DataFrame
                )
                and
                not matched.empty
            ):

                matched_count = int(
                    (
                        matched[
                            "match_status"
                        ]
                        ==
                        "Matched"
                    ).sum()
                )

                unmatched_count = int(
                    len(
                        matched
                    )
                    -
                    matched_count
                )

                complete_predictors = int(
                    (
                        (
                            matched[
                                "match_status"
                            ]
                            ==
                            "Matched"
                        )
                        &
                        matched[
                            [
                                "historical_sst_c",
                                "historical_salinity_psu",
                                "historical_surface_o2_ml_l",
                                "historical_water_depth_m",
                                "historical_distance_coast_km"
                            ]
                        ]
                        .notna()
                        .all(
                            axis=1
                        )
                    ).sum()
                )

                cols = st.columns(
                    4
                )

                cols[0].metric(
                    "Selected Observations",
                    f"{len(matched):,}"
                )

                cols[1].metric(
                    "Copernicus Matched",
                    f"{matched_count:,}"
                )

                cols[2].metric(
                    "Unmatched",
                    f"{unmatched_count:,}"
                )

                cols[3].metric(
                    "Core Predictors Complete",
                    f"{complete_predictors:,}"
                )

                display_cols = [
                    "source_row",
                    "date",
                    "lat",
                    "lon",
                    "survey_id",
                    "presence",
                    "abundance",
                    "historical_sst_c",
                    "historical_salinity_psu",
                    "historical_surface_o2_ml_l",
                    "historical_water_depth_m",
                    "historical_distance_coast_km",
                    "historical_distance_coast_nm",
                    "historical_oxycline_m",
                    "physics_match_distance_km",
                    "bgc_match_distance_km",
                    "month_number",
                    "austral_season",
                    "month_sin",
                    "month_cos",
                    "match_status",
                ]

                st.dataframe(
                    matched[
                        display_cols
                    ],
                    width="stretch",
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download matched training-table candidate",
                    data=
                        matched
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        ),
                    file_name=
                        "eaware_v2_step2_matched_historical_environment.csv",
                    mime=
                        "text/csv",
                )

                if matched_count > 0:

                    st.success(
                        "Historical environmental matching has created a historical "
                        "observation + environmental predictor table. "
                        "The next stage is quality control and predictor "
                        "completeness checking before any model training."
                    )

                else:

                    st.warning(
                        "No selected observations were successfully matched. "
                        "Review dates, coordinates and the status column "
                        "before continuing."
                    )

                st.subheader(
                    "Matched Training-Table Quality Control"
                )

                matched_qc = matched_qc_audit(
                    matched,
                    meta
                )

                candidate_rows = matched_qc[
                    "candidate_rows"
                ]

                qc_cols = st.columns(
                    4
                )

                qc_cols[0].metric(
                    "Core Candidate Rows",
                    f"{len(candidate_rows):,}"
                )

                qc_cols[1].metric(
                    "Candidate Presence",
                    f"{matched_qc['candidate_presence']:,}"
                )

                qc_cols[2].metric(
                    "Candidate Absence",
                    f"{matched_qc['candidate_absence']:,}"
                )

                qc_cols[3].metric(
                    "Candidate Completeness",
                    (
                        f"{100.0 * len(candidate_rows) / len(matched):.1f}%"
                        if len(matched)
                        else
                        "0.0%"
                    )
                )

                if (
                    meta[
                        "target_absence_verified"
                    ]
                    and
                    matched_qc[
                        "candidate_presence"
                    ]
                    >
                    0
                    and
                    matched_qc[
                        "candidate_absence"
                    ]
                    >
                    0
                ):
                    st.success(
                        matched_qc[
                            "status"
                        ]
                    )
                else:
                    st.warning(
                        matched_qc[
                            "status"
                        ]
                    )

                st.dataframe(
                    matched_qc[
                        "summary"
                    ],
                    width="stretch",
                    hide_index=True
                )

                with st.expander(
                    "Predictor completeness and QC-annotated rows",
                    expanded=False
                ):

                    st.markdown(
                        "**Predictor completeness**"
                    )

                    st.dataframe(
                        matched_qc[
                            "predictor_completeness"
                        ],
                        width="stretch",
                        hide_index=True
                    )

                    st.markdown(
                        "**QC-annotated matched rows**"
                    )

                    st.dataframe(
                        matched_qc[
                            "annotated"
                        ],
                        width="stretch",
                        hide_index=True
                    )

                st.download_button(
                    "⬇️ Download data-quality QC summary",
                    data=
                        matched_qc[
                            "summary"
                        ]
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        ),
                    file_name=
                        "eaware_v2_step3_matched_qc_summary.csv",
                    mime=
                        "text/csv"
                )

                st.download_button(
                    "⬇️ Download QC-filtered core training candidates",
                    data=
                        candidate_rows
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        ),
                    file_name=
                        "eaware_v2_step3_training_candidates.csv",
                    mime=
                        "text/csv",
                    disabled=
                        candidate_rows.empty
                )

    except Exception as historical_error:

        st.error(
            "Historical observation processing failed."
        )

        st.code(
            str(
                historical_error
            )
        )


# ============================================================
# FISHER ADVISORY / SMS PREVIEW
# ============================================================
st.subheader(
    "📱 Fisher SMS Advisory Preview"
)

best_hsi = float(
    best[
        "hsi"
    ]
)

# ------------------------------------------------------------
# Bathymetry at the highest-HSI grid cell.
#
# This is seafloor depth, NOT predicted fish depth.
# E-AWARE currently has no validated vertical-distribution
# model for anchoveta.
# ------------------------------------------------------------

best_seafloor_depth_text = (
    f"{float(best['water_depth_m']):.0f} m"
    if pd.notna(
        best[
            "water_depth_m"
        ]
    )
    else
    "Unavailable"
)


best_zone_text = (
    f"{best['lat']:.4f}, "
    f"{best['lon']:.4f}"
)


sms_message = f"""E-AWARE Fisheries Advisory

Date:
{current_date}

Anchoveta habitat suitability:
{hsi_class(best_hsi).upper()} (HSI {best_hsi:.2f})

Highest HSI grid cell within configured corridor:
{best_zone_text}

GEBCO seafloor depth beneath HSI cell:
{best_seafloor_depth_text}
(Not predicted fish depth)

Predicted suitable-habitat shift:
{direction_text}

Estimated habitat-centroid shift:
{distance_text}

Relative to coastline:
{coastal_interpretation}

Conditions at highest-HSI cell:
SST {best['temperature']:.1f} °C
Salinity {best['salinity']:.2f} PSU
Oxygen {best['oxygen_ml_l']:.2f} mL/L

Distance from coast:
{best['distance_coast_km']:.2f} km / {best['distance_coast_nm']:.2f} nm

Official regulatory status:
{fisheries_protection['official_status']}

Juvenile protection status:
{fisheries_protection['juvenile_status']}

E-AWARE ecological advisory:
{ecological_display}

IMPORTANT:
Official fishing status is not live-verified.
Check current PRODUCE closures, temporary suspensions
and local fishing restrictions before fishing.

E-AWARE identifies higher-suitability habitat zones,
not exact fish locations, legal fishing permission,
or navigation routes.
"""

# ============================================================
# AUTOMATION SNAPSHOT
#
# The scheduled backend uses the SAME advisory generated by
# the Streamlit dashboard. No second HSI model is created.
# ============================================================

import os as _eaware_os

if _eaware_os.environ.get("EAWARE_AUTOMATION_MODE") == "1":

    import json as _eaware_json
    from pathlib import Path as _EawarePath
    from datetime import datetime as _EawareDateTime
    from datetime import timezone as _EawareTimezone

    _automation_payload = {
        "generated_at_utc":
            _EawareDateTime.now(
                _EawareTimezone.utc
            ).isoformat(),

        "data_date":
            str(current_date),

        "hsi":
            float(best_hsi),

        "hsi_class":
            hsi_class(best_hsi).upper(),

        "advisory":
            sms_message,
    }

    _EawarePath(
        "/tmp/eaware_advisory.json"
    ).write_text(
        _eaware_json.dumps(
            _automation_payload,
            indent=2
        ),
        encoding="utf-8"
    )


st.text_area(
    "Example SMS advisory",
    sms_message,
    height=680
)

# ============================================================
# REAL EMAIL ADVISORY DELIVERY — GMAIL SMTP
# ============================================================

st.subheader(
    "📧 Email Advisory Delivery"
)

st.caption(
    "Enter an email address to receive the current E-AWARE "
    "fisheries advisory. SMS/GSM remains a proposed future "
    "implementation."
)

gmail_configured = (
    "GMAIL_SENDER" in st.secrets
    and
    "GMAIL_APP_PASSWORD" in st.secrets
)

recipient_email = st.text_input(
    "Recipient email address",
    placeholder="name@example.com",
    key="eaware_recipient_email"
)

if not gmail_configured:

    st.warning(
        "Email delivery is unavailable in this local environment. The deployed backend is configured separately."
    )


else:

    if st.button(
        "📧 Send Current E-AWARE Advisory",
        type="primary",
        width="stretch"
    ):

        try:

            import html
            import re
            import smtplib

            from email.message import EmailMessage


            recipient_email = recipient_email.strip()

            email_pattern = (
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            )

            if not re.match(
                email_pattern,
                recipient_email
            ):

                st.error(
                    "Please enter a valid email address."
                )

            else:

                sender_email = str(
                    st.secrets[
                        "GMAIL_SENDER"
                    ]
                ).strip()

                app_password = str(
                    st.secrets[
                        "GMAIL_APP_PASSWORD"
                    ]
                ).replace(
                    " ",
                    ""
                ).strip()

                safe_message = html.escape(
                    sms_message
                )

                email_html = f"""
                <div style="
                    font-family: Arial, sans-serif;
                    max-width: 700px;
                    margin: auto;
                    color: #222;
                ">

                    <h1>
                        E-AWARE Fisheries Advisory
                    </h1>

                    <p style="color:#666;">
                        El Niño Adaptive Fisheries Intelligence
                    </p>

                    <div style="
                        background:#f4f4f4;
                        border-radius:12px;
                        padding:22px;
                        margin-top:20px;
                    ">

                        <pre style="
                            white-space:pre-wrap;
                            font-family:Arial, sans-serif;
                            line-height:1.6;
                            margin:0;
                        ">{safe_message}</pre>

                    </div>

                    <p style="
                        margin-top:24px;
                        font-size:13px;
                        color:#666;
                    ">
                        E-AWARE provides habitat decision support only.
                        It does not provide legal fishing permission,
                        exact fish locations or navigation instructions.
                        Current PRODUCE regulations must be verified
                        before fishing.
                    </p>

                </div>
                """

                msg = EmailMessage()

                msg[
                    "Subject"
                ] = (
                    f"E-AWARE Fisheries Advisory — "
                    f"{current_date}"
                )

                msg[
                    "From"
                ] = (
                    f"E-AWARE <{sender_email}>"
                )

                msg[
                    "To"
                ] = recipient_email

                msg.set_content(
                    sms_message
                )

                msg.add_alternative(
                    email_html,
                    subtype="html"
                )

                with smtplib.SMTP(
                    "smtp.gmail.com",
                    587,
                    timeout=30
                ) as smtp:

                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()

                    smtp.login(
                        sender_email,
                        app_password
                    )

                    smtp.send_message(
                        msg
                    )

                st.success(
                    "✅ E-AWARE advisory sent successfully to "
                    f"{recipient_email}"
                )

        except Exception as exc:

            st.error(
                "Email delivery failed. "
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# DAILY ADVISORY SUBSCRIPTION — OPERATIONAL MODE
# ============================================================


def send_subscription_confirmation_email(recipient_email):
    """Send an immediate confirmation after a successful subscription."""
    from email.message import EmailMessage
    import smtplib

    sender = str(st.secrets.get("GMAIL_SENDER", "")).strip()
    app_password = str(st.secrets.get("GMAIL_APP_PASSWORD", "")).strip()

    if not sender or not app_password:
        raise RuntimeError(
            "Gmail credentials are not configured."
        )

    message = EmailMessage()

    message["From"] = sender
    message["To"] = recipient_email
    message["Subject"] = "E-AWARE Daily Advisory Subscription Confirmed"

    message.set_content(
        """E-AWARE Daily Advisory Subscription Confirmed

Your email address has been successfully registered for the E-AWARE daily fisheries advisory.

You will receive the latest E-AWARE advisory automatically each day at approximately 4:00 AM Peru time.

The advisory provides decision-support information including:
- Anchoveta habitat suitability
- Environmental conditions
- Predicted suitable-habitat shift
- Coastal distance information
- Ecological guidance

Please check official fishing restrictions and safety information before departure.

E-AWARE
El Niño Adaptive Fisheries Intelligence

Decision-support research prototype.
"""
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, app_password)
        smtp.send_message(message)



def send_unsubscription_confirmation_email(recipient_email):
    """Send an immediate confirmation after a successful unsubscribe."""
    from email.message import EmailMessage
    import smtplib

    sender = str(st.secrets.get("GMAIL_SENDER", "")).strip()
    app_password = str(st.secrets.get("GMAIL_APP_PASSWORD", "")).strip()

    if not sender or not app_password:
        raise RuntimeError(
            "Gmail credentials are not configured."
        )

    message = EmailMessage()

    message["From"] = sender
    message["To"] = recipient_email
    message["Subject"] = "E-AWARE Daily Advisory Unsubscription Confirmed"

    message.set_content(
        """E-AWARE Daily Advisory Unsubscription Confirmed

Your email address has been successfully removed from the active E-AWARE daily advisory subscription list.

You will no longer receive the automatic 4:00 AM Peru-time daily advisory.

You may subscribe again at any time through the E-AWARE dashboard.

E-AWARE
El Niño Adaptive Fisheries Intelligence

Decision-support research prototype.
"""
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, app_password)
        smtp.send_message(message)


st.subheader(
    "🔔 Daily Advisory Subscription"
)

st.caption(
    "Scheduled delivery: subscribe once to receive the E-AWARE "
    "daily advisory automatically. The prototype backend is "
    "scheduled for 4:00 AM Peru time."
)

supabase_configured = (
    "SUPABASE_URL" in st.secrets
    and
    "SUPABASE_SECRET_KEY" in st.secrets
)

subscription_email = st.text_input(
    "Email for daily advisory",
    placeholder="name@example.com",
    key="eaware_subscription_email"
)

subscription_cols = st.columns(2)

if not supabase_configured:

    st.warning(
        "Daily subscription management is unavailable in this local environment. The deployed backend is configured separately."
    )


else:

    if subscription_cols[0].button(
        "🔔 Subscribe to Daily Advisory",
        width="stretch"
    ):

        try:

            import re
            from datetime import datetime, timezone
            from supabase import create_client

            clean_email = (
                subscription_email
                .strip()
                .lower()
            )

            email_pattern = (
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            )

            if not re.match(
                email_pattern,
                clean_email
            ):

                st.error(
                    "Please enter a valid email address."
                )

            else:

                supabase_client = create_client(
                    str(
                        st.secrets[
                            "SUPABASE_URL"
                        ]
                    ),
                    str(
                        st.secrets[
                            "SUPABASE_SECRET_KEY"
                        ]
                    )
                )

                existing = (
                    supabase_client
                    .table(
                        "eaware_subscribers"
                    )
                    .select(
                        "id,email,active"
                    )
                    .eq(
                        "email",
                        clean_email
                    )
                    .execute()
                )

                now_iso = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                if existing.data:

                    (
                        supabase_client
                        .table(
                            "eaware_subscribers"
                        )
                        .update(
                            {
                                "active": True,
                                "updated_at": now_iso,
                            }
                        )
                        .eq(
                            "email",
                            clean_email
                        )
                        .execute()
                    )

                else:

                    (
                        supabase_client
                        .table(
                            "eaware_subscribers"
                        )
                        .insert(
                            {
                                "email": clean_email,
                                "active": True,
                                "updated_at": now_iso,
                            }
                        )
                        .execute()
                    )

                st.success(
                    "✅ Subscribed successfully. "
                    "This address is now registered for the "
                    "daily E-AWARE advisory."
                )

                try:
                    send_subscription_confirmation_email(
                        clean_email
                    )

                    st.success(
                        "📧 A subscription confirmation email "
                        "has been sent."
                    )

                except Exception as email_exc:
                    st.warning(
                        "The subscription was saved successfully, "
                        "but the confirmation email could not be sent. "
                        f"{type(email_exc).__name__}: {email_exc}"
                    )

        except Exception as exc:

            st.error(
                "Subscription failed. "
                f"{type(exc).__name__}: {exc}"
            )


    if subscription_cols[1].button(
        "🔕 Unsubscribe",
        width="stretch"
    ):

        try:

            import re
            from datetime import datetime, timezone
            from supabase import create_client

            clean_email = (
                subscription_email
                .strip()
                .lower()
            )

            email_pattern = (
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            )

            if not re.match(
                email_pattern,
                clean_email
            ):

                st.error(
                    "Please enter a valid email address."
                )

            else:

                supabase_client = create_client(
                    str(
                        st.secrets[
                            "SUPABASE_URL"
                        ]
                    ),
                    str(
                        st.secrets[
                            "SUPABASE_SECRET_KEY"
                        ]
                    )
                )

                result = (
                    supabase_client
                    .table(
                        "eaware_subscribers"
                    )
                    .update(
                        {
                            "active": False,
                            "updated_at":
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                        }
                    )
                    .eq(
                        "email",
                        clean_email
                    )
                    .execute()
                )

                if result.data:

                    st.success(
                        "✅ Unsubscribed successfully."
                    )

                    try:
                        send_unsubscription_confirmation_email(
                            clean_email
                        )

                        st.success(
                            "📧 An unsubscription confirmation email "
                            "has been sent."
                        )

                    except Exception as email_exc:
                        st.warning(
                            "The subscription was removed successfully, "
                            "but the confirmation email could not be sent. "
                            f"{type(email_exc).__name__}: {email_exc}"
                        )

                else:

                    st.info(
                        "That email was not found in the "
                        "subscriber list."
                    )

        except Exception as exc:

            st.error(
                "Unsubscribe failed. "
                f"{type(exc).__name__}: {exc}"
            )


st.info(
    "📤 **Send Now** provides manual/demo delivery. "
    "🔔 **Daily Advisory Subscription** enables scheduled prototype "
    "delivery. Active subscribers are processed by the "
    "scheduled 4:00 AM Peru backend."
)


st.caption(
    "The displayed depth is GEBCO seafloor bathymetry at the "
    "highest-HSI grid cell. E-AWARE does not currently predict "
    "a validated anchoveta fishing-depth band."
)


# ============================================================
# TRANSPARENCY / STATUS
# ============================================================
st.subheader(
    "📊 Data Sources & Transparency"
)

source_table = pd.DataFrame(
    [
        {
            "Source":
                "Copernicus Marine Physics Analysis & Forecast",
            "Use":
                "Live SST, salinity and SST anomaly"
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry Analysis & Forecast",
            "Use":
                "Live surface oxygen and vertical oxygen profile"
        },

        {
            "Source":
                "Copernicus Global Ocean Physics Reanalysis",
            "Use":
                "Historical SST and salinity from 1993 onward"
        },

        {
            "Source":
                "Copernicus Global Ocean Biogeochemistry Hindcast",
            "Use":
                "Historical oxygen and oxycline from 1993 onward"
        },

        {
            "Source":
                "GEBCO_2026 via Ocean Data Bank API",
            "Use":
                "Bathymetry / local water-column limit"
        },

        {
            "Source":
                "NOAA NCEI coastline",
            "Use":
                "Nearest-coast distance"
        },

        {
            "Source":
                "Repository-managed anchoveta observations",
            "Use":
                "Traceable fish-observation data for V2 pipeline validation "
                "and, when scientifically suitable, future model training"
        },
    ]
)

st.dataframe(
    source_table,
    width="stretch",
    hide_index=True
)

st.subheader(
    "🧪 E-AWARE Model Status"
)

model_manifest = (
    load_experimental_model_manifest()
)

cols = st.columns(
    2
)

cols[0].success(
    "**PRIMARY — Literature HSI**\n\n"
    "The live habitat map, best-zone selection and "
    "72-hour habitat-shift forecast continue to use "
    "the literature-based Habitat Suitability Index."
)

if model_manifest:

    validation = model_manifest.get(
        "validation_summary",
        {}
    )

    year_concordance = validation.get(
        "year_holdout_concordance"
    )

    year_top1 = validation.get(
        "year_holdout_top1"
    )

    anchors = model_manifest.get(
        "training_anchors",
        "—"
    )

    concordance_text = (
        f"{year_concordance:.3f}"
        if isinstance(
            year_concordance,
            (int, float)
        )
        else
        "—"
    )

    top1_text = (
        f"{year_top1:.3f}"
        if isinstance(
            year_top1,
            (int, float)
        )
        else
        "—"
    )

    cols[1].info(
        "**SECONDARY — Experimental Conditional ML**\n\n"
        f"Training anchors: **{anchors}**  \n"
        f"Year-held-out matched concordance: "
        f"**{concordance_text}**  \n"
        f"Year-held-out Top-1: **{top1_text}**  \n\n"
        "Spatial transfer could not be robustly validated. "
        "The score must not be interpreted as probability "
        "of anchoveta presence."
    )

else:

    cols[1].warning(
        "**SECONDARY — Experimental Conditional ML**\n\n"
        "The experimental-model manifest is not available "
        "in this deployment. The primary HSI remains fully "
        "available."
    )

with st.expander(
    "⚙️ E-AWARE Model & Data Limitations"
):

    limitation_lines = [
        "- Historical Copernicus matching begins on **1 January 1993**. Earlier observations remain in the table but are not assigned invented environmental values.",
        "- Physics is matched to the nearest 0.083-degree reanalysis grid cell; biogeochemistry is matched independently to the nearest 0.25-degree hindcast grid cell.",
        "- Historical oxycline is calculated from the 1 mL/L dissolved-oxygen threshold and constrained by local GEBCO water depth.",
        "- A missing historical record is never interpreted as fish absence.",
        "- The Streamlit matcher intentionally limits the number of observations processed in one interactive run. A large national archive should later be batch-processed offline and versioned.",
        "- A true historical SST-anomaly predictor is not included in the historical environmental matching. It should be calculated from a documented historical climatology rather than copied from the live forecast-anomaly product.",
        "- Historical environmental matching creates candidate predictor rows only; it does not prove habitat-model accuracy.",
        "- Data-quality QC checks structural and environmental data quality, but it cannot prove that survey absences are valid unless the source methodology supports that interpretation.",
        "- QC match-distance thresholds are engineering warning limits for investigation; they are not biological habitat thresholds.",
        "- Passing data-quality QC does not remove sampling bias, spatial autocorrelation or survey-design dependence.",
    ]

    st.markdown(
        "\n".join(limitation_lines)
    )

st.divider()

st.caption(
    "E-AWARE | "
    "Literature HSI = primary habitat model | "
    "Conditional ML = experimental secondary layer | "
    "Decision-support research prototype"
)

st.caption(
    "Habitat suitability is a model estimate and must not be used "
    "as navigation, safety or legal fishing advice."
)
