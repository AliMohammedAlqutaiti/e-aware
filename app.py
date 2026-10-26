import os
import math
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
st.caption("El Niño Adaptive Fisheries Intelligence — V2 Step 3")
st.info(
    "V2 Step 3 adds a training-data quality-control audit to the anchoveta "
    "observation and historical environmental-matching pipeline. It still does not "
    "train a machine-learning model."
)

MIN_LON, MAX_LON = -81.50, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36

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

# Engineering QC warning thresholds for nearest-grid matching.
# These are diagnostic thresholds, not biological habitat thresholds.
QC_MAX_PHYSICS_MATCH_KM = 15.0
QC_MAX_BGC_MATCH_KM = 25.0

# ============================================================
# GENERAL HELPERS
# ============================================================
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

    threshold = float(
        dataframe[
            "hsi"
        ].quantile(
            0.75
        )
    )

    best_area = dataframe[
        dataframe[
            "hsi"
        ]
        >=
        threshold
    ].copy()

    if best_area.empty:
        best_area = dataframe.nlargest(
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
            "by the Step 3 quality-control audit"
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
            "status": "Run V2 Step 2 matching before environmental QC"
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
        use_container_width=True
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

try:

    with st.status(
        "Loading E-AWARE environmental data...",
        expanded=True
    ) as status:

        username, password = (
            get_credentials()
        )

        status.write(
            "🔐 Checking Copernicus credentials..."
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

        temperature_data = (
            load_temperature(
                selected_date,
                end_date
            )
        )

        status.write(
            "🧂 2/7 Salinity"
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

        anomaly_data = (
            load_sst_anomaly(
                selected_date,
                end_date
            )
        )

        status.write(
            "🫧 4/7 3D oxygen profile"
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

        ocean = attach_gebco_bathymetry(
            ocean
        )

        status.write(
            "📏 6/7 Distance from coast"
        )

        ocean = attach_distance_to_coast(
            ocean
        )

        status.write(
            "🫧 7/7 Seabed-constrained oxycline"
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

    st.error(
        "❌ E-AWARE could not load all live environmental data."
    )

    st.code(
        str(
            error
        )
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

reference = habitat_reference(
    current_date.month
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
    "GEBCO Depth Estimate",
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

st.pydeck_chart(
    pdk.Deck(
        map_style=
            BASEMAP_URL,

        initial_view_state=
            pdk.ViewState(
                latitude=
                    -4.49,
                longitude=
                    -81.36,
                zoom=
                    9.5
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
    use_container_width=True
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
    "📍 Highest Relative Habitat Suitability"
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
    "GEBCO Depth",
    f"{best['water_depth_m']:.0f} m"
)

cols[4].metric(
    "Coast Distance",
    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)

st.subheader(
    "🐟 Current MVP Habitat Reference"
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

st.warning(
    "The temporary HSI still uses only temperature, "
    "salinity and surface dissolved oxygen. "
    "Historical matching does not change the live HSI."
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

st.subheader(
    "🧭 Predicted Habitat Shift"
)

direction_text = "Unavailable"
distance_text = "—"
bearing_text = "—"
period_text = "Insufficient forecast data"
coastal_interpretation = "Unavailable"

start = None
end = None

if len(
    forecast_centres
) >= 2:

    start = forecast_centres[0]
    end = forecast_centres[-1]

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

    bearing_text = (
        f"{bearing:.1f}°"
    )

    direction_text = (
        direction_name(
            bearing
        )
        if distance >= 2.0
        else
        "No clear shift"
    )

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
    "Predicted Direction",
    direction_text
)

cols[1].metric(
    "Total Shift",
    distance_text
)

cols[2].metric(
    "Bearing",
    bearing_text
)

st.markdown(
    f"**Forecast Period:** {period_text}"
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
        "+72 h Distance from Coast",
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
    "📅 72-Hour Habitat Forecast"
)

st.dataframe(
    pd.DataFrame(
        forecast_results
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# V2 HISTORICAL UI
# ============================================================
st.divider()

st.header(
    "🧬 V2 — Anchoveta Observation Dataset"
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
            "V2 Step 1 — Ingestion & Validation"
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
                use_container_width=True,
                hide_index=True
            )

        st.subheader(
            "V2 Step 3A — Observation Data Quality Control"
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
            "Open V2 Step 3A observation QC details",
            expanded=False
        ):

            st.dataframe(
                observation_qc[
                    "summary"
                ],
                use_container_width=True,
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
                    use_container_width=True,
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
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(
                "**Observations by year**"
            )
            st.dataframe(
                observation_qc[
                    "year_counts"
                ],
                use_container_width=True,
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
                    use_container_width=True,
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
            "V2 Step 2 — Historical Environmental Matching"
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
                use_container_width=True
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
                            "✅ V2 Step 2 matching complete",
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
                    use_container_width=True,
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
                        "V2 Step 2 has created a historical "
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
                    "V2 Step 3B — Matched Training-Table Quality Control"
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
                    use_container_width=True,
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
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown(
                        "**QC-annotated matched rows**"
                    )

                    st.dataframe(
                        matched_qc[
                            "annotated"
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                st.download_button(
                    "⬇️ Download V2 Step 3 QC summary",
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
# SMS PREVIEW
# ============================================================
st.subheader(
    "📱 Fisher SMS Alert Preview"
)

best_hsi = float(
    best[
        "hsi"
    ]
)

sms_message = f"""E-AWARE Fisheries Alert

Anchoveta habitat suitability:
{hsi_class(best_hsi).upper()}

Habitat Suitability Index:
{best_hsi:.2f}

Predicted suitable-habitat shift:
{direction_text}

Estimated habitat shift:
{distance_text}

Relative to coastline:
{coastal_interpretation}

Sea temperature:
{best['temperature']:.1f} °C

Salinity:
{best['salinity']:.2f} PSU

Dissolved oxygen:
{best['oxygen_ml_l']:.2f} mL/L

GEBCO depth estimate:
{best['water_depth_m']:.0f} m

Distance from coast:
{best['distance_coast_km']:.2f} km / {best['distance_coast_nm']:.2f} nm

Check official fishing restrictions before departure.

E-AWARE provides habitat guidance only.
"""

st.text_area(
    "Example SMS",
    sms_message,
    height=500
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
    use_container_width=True,
    hide_index=True
)

st.subheader(
    "🧪 V2 Step 3 Model Status"
)

cols = st.columns(
    2
)

cols[0].success(
    "Historical observations can now be ingested, audited, matched to "
    "historical environmental predictors and QC-screened without "
    "altering the live HSI."
)

cols[1].warning(
    "No machine-learning model is trained yet. "
    "The next stage after a defensible historical dataset passes Step 3 QC "
    "is a versioned final training table, followed by a Logistic Regression baseline."
)

with st.expander(
    "⚙️ V2 Step 3 Limitations"
):

    limitation_lines = [
        "- Historical Copernicus matching begins on **1 January 1993**. Earlier observations remain in the table but are not assigned invented environmental values.",
        "- Physics is matched to the nearest 0.083-degree reanalysis grid cell; biogeochemistry is matched independently to the nearest 0.25-degree hindcast grid cell.",
        "- Historical oxycline is calculated from the 1 mL/L dissolved-oxygen threshold and constrained by local GEBCO water depth.",
        "- A missing historical record is never interpreted as fish absence.",
        "- The Streamlit matcher intentionally limits the number of observations processed in one interactive run. A large national archive should later be batch-processed offline and versioned.",
        "- A true historical SST-anomaly predictor is not added in Step 2. It should be calculated from a documented historical climatology rather than copied from the live forecast-anomaly product.",
        "- Step 2 creates candidate predictor rows only; it does not prove habitat-model accuracy.",
        "- Step 3 QC checks structural and environmental data quality, but it cannot prove that survey absences are valid unless the source methodology supports that interpretation.",
        "- QC match-distance thresholds are engineering warning limits for investigation; they are not biological habitat thresholds.",
        "- Passing Step 3 QC does not remove sampling bias, spatial autocorrelation or survey-design dependence.",
    ]

    st.markdown(
        "\n".join(limitation_lines)
    )

st.divider()

st.caption(
    "E-AWARE V2 Step 3 | "
    "Historical observation ↔ environmental matching ↔ training-data QC | "
    "Decision-support research prototype"
)

st.caption(
    "Habitat suitability is a model estimate and must not be used "
    "as navigation, safety or legal fishing advice."
)
