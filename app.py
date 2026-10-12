import os
import math
import re
import tempfile
from pathlib import Path
from datetime import date, timedelta

import copernicusmarine
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import xarray as xr


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="E-AWARE",
    page_icon="🐟",
    layout="wide",
)

st.title("🐟 E-AWARE")
st.caption("El Niño Adaptive Fisheries Intelligence — MVP V1.2")

st.info(
    "E-AWARE estimates suitable anchoveta habitat from environmental conditions. "
    "It does not show the exact location or swimming direction of individual fish schools."
)


# ============================================================
# STUDY AREA
# ============================================================

MIN_LON = -81.50
MAX_LON = -81.20
MIN_LAT = -4.60
MAX_LAT = -4.36

COMMUNITIES = pd.DataFrame(
    [
        {"name": "Lobitos", "lat": -4.4567, "lon": -81.2849},
        {"name": "Siches", "lat": -4.4890, "lon": -81.2680},
        {"name": "Piedritas", "lat": -4.5190, "lon": -81.2630},
    ]
)


# ============================================================
# COPERNICUS DATASETS
# ============================================================

TEMPERATURE_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
SALINITY_DATASET = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
SST_ANOMALY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m"
OXYGEN_DATASET = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"


# ============================================================
# GEBCO 2026 WMS
# ============================================================

GEBCO_WMS_URL = "https://wms.gebco.net/mapserv?"
GEBCO_WMS_LAYER = "gebco_latest_2"


# ============================================================
# COPERNICUS CREDENTIALS
# ============================================================

def get_credentials():

    username = st.secrets.get(
        "COPERNICUS_USERNAME",
        "",
    )

    password = st.secrets.get(
        "COPERNICUS_PASSWORD",
        "",
    )

    if not username or not password:

        raise RuntimeError(
            "Copernicus credentials are missing from "
            "Streamlit Secrets."
        )

    return username, password


# ============================================================
# CLEAN COPERNICUS DATA
# ============================================================

def prepare_dataframe(
    dataframe,
    variable_name,
):

    df = dataframe.copy()

    rename_map = {}

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if column_lower in [
            "latitude",
            "lat",
        ]:

            rename_map[
                column
            ] = "lat"

        elif column_lower in [
            "longitude",
            "lon",
        ]:

            rename_map[
                column
            ] = "lon"

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "time",
        "lat",
        "lon",
        variable_name,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns for "
            f"{variable_name}: {missing}"
        )

    df = df.dropna(
        subset=[
            variable_name
        ]
    ).copy()

    df[
        "date"
    ] = (
        pd.to_datetime(
            df[
                "time"
            ]
        )
        .dt.date
    )

    df[
        "lat"
    ] = (
        df[
            "lat"
        ]
        .astype(
            float
        )
    )

    df[
        "lon"
    ] = (
        df[
            "lon"
        ]
        .astype(
            float
        )
    )

    df[
        "lat_key"
    ] = (
        df[
            "lat"
        ]
        .round(
            3
        )
    )

    df[
        "lon_key"
    ] = (
        df[
            "lon"
        ]
        .round(
            3
        )
    )

    return df[
        [
            "date",
            "lat",
            "lon",
            "lat_key",
            "lon_key",
            variable_name,
        ]
    ]


# ============================================================
# DOWNLOAD COPERNICUS SUBSET
# ============================================================

def download_subset(
    dataset_id,
    variable,
    output_name,
    start_date,
    end_date,
    use_depth=True,
):

    username, password = (
        get_credentials()
    )

    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"

    with tempfile.TemporaryDirectory() as temp_dir:

        filename = (
            f"{output_name}.nc"
        )

        subset_arguments = {

            "dataset_id":
                dataset_id,

            "variables":
                [
                    variable
                ],

            "minimum_longitude":
                MIN_LON,

            "maximum_longitude":
                MAX_LON,

            "minimum_latitude":
                MIN_LAT,

            "maximum_latitude":
                MAX_LAT,

            "start_datetime":
                f"{start_date}T00:00:00",

            "end_datetime":
                f"{end_date}T23:59:59",

            "username":
                username,

            "password":
                password,

            "service":
                "arco-geo-series",

            "output_filename":
                filename,

            "output_directory":
                temp_dir,

            "disable_progress_bar":
                True,
        }

        if use_depth:

            subset_arguments[
                "minimum_depth"
            ] = 0

            subset_arguments[
                "maximum_depth"
            ] = 1

        response = (
            copernicusmarine.subset(
                **subset_arguments
            )
        )

        possible_path = Path(
            str(
                response.file_path
            )
        )

        if possible_path.exists():

            file_path = (
                possible_path
            )

        else:

            file_path = (
                Path(
                    temp_dir
                )
                /
                filename
            )

        if not file_path.exists():

            netcdf_files = list(
                Path(
                    temp_dir
                )
                .glob(
                    "*.nc"
                )
            )

            if not netcdf_files:

                raise RuntimeError(
                    f"{output_name} download "
                    "completed but no NetCDF "
                    "file was found."
                )

            file_path = (
                netcdf_files[
                    0
                ]
            )

        with xr.open_dataset(
            file_path
        ) as ds:

            if variable not in ds.data_vars:

                raise RuntimeError(
                    f"Variable '{variable}' "
                    f"was not found in the "
                    f"{output_name} file."
                )

            data_array = (
                ds[
                    variable
                ]
            )

            if "depth" in data_array.dims:

                data_array = (
                    data_array.isel(
                        depth=0
                    )
                )

            data_array = (
                data_array.load()
            )

            dataframe = (
                data_array
                .to_dataframe(
                    name=output_name
                )
                .reset_index()
            )

    return prepare_dataframe(
        dataframe,
        output_name,
    )


# ============================================================
# TEMPERATURE
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_temperature(
    start_date,
    end_date,
):

    return download_subset(

        dataset_id=
            TEMPERATURE_DATASET,

        variable=
            "thetao",

        output_name=
            "temperature",

        start_date=
            start_date,

        end_date=
            end_date,

        use_depth=True,
    )


# ============================================================
# SALINITY
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_salinity(
    start_date,
    end_date,
):

    return download_subset(

        dataset_id=
            SALINITY_DATASET,

        variable=
            "so",

        output_name=
            "salinity",

        start_date=
            start_date,

        end_date=
            end_date,

        use_depth=True,
    )


# ============================================================
# SST ANOMALY
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_sst_anomaly(
    start_date,
    end_date,
):

    return download_subset(

        dataset_id=
            SST_ANOMALY_DATASET,

        variable=
            "sea_surface_temperature_anomaly",

        output_name=
            "sst_anomaly",

        start_date=
            start_date,

        end_date=
            end_date,

        use_depth=False,
    )


# ============================================================
# DISSOLVED OXYGEN
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_oxygen(
    start_date,
    end_date,
):

    oxygen = download_subset(

        dataset_id=
            OXYGEN_DATASET,

        variable=
            "o2",

        output_name=
            "oxygen_mmol_m3",

        start_date=
            start_date,

        end_date=
            end_date,

        use_depth=True,
    )

    oxygen[
        "oxygen_ml_l"
    ] = (

        oxygen[
            "oxygen_mmol_m3"
        ]

        *
        0.0223916
    )

    return oxygen


# ============================================================
# MERGE PHYSICS DATASETS
# ============================================================

def merge_physics_data(
    temperature,
    salinity,
    anomaly,
):

    keys = [
        "date",
        "lat_key",
        "lon_key",
    ]

    merged = (

        temperature

        .merge(

            salinity[
                keys
                +
                [
                    "salinity"
                ]
            ],

            on=keys,

            how="inner",
        )

        .merge(

            anomaly[
                keys
                +
                [
                    "sst_anomaly"
                ]
            ],

            on=keys,

            how="left",
        )
    )

    if merged.empty:

        raise RuntimeError(
            "Copernicus physics data downloaded "
            "successfully, but no matching grid "
            "cells were found."
        )

    return merged


# ============================================================
# MAP OXYGEN GRID TO PHYSICS GRID
# ============================================================

def attach_nearest_oxygen(
    physics_df,
    oxygen_df,
):

    result = (
        physics_df.copy()
    )

    result[
        "oxygen_mmol_m3"
    ] = np.nan

    result[
        "oxygen_ml_l"
    ] = np.nan

    result[
        "oxygen_source_distance_km"
    ] = np.nan

    for target_date in sorted(
        result[
            "date"
        ]
        .unique()
    ):

        target_mask = (

            result[
                "date"
            ]

            ==
            target_date
        )

        target_rows = (
            result.loc[
                target_mask
            ]
        )

        source_rows = oxygen_df[

            oxygen_df[
                "date"
            ]

            ==
            target_date

        ]

        if source_rows.empty:

            continue

        target_lat = (

            target_rows[
                "lat"
            ]

            .to_numpy(
                dtype=float
            )
        )

        target_lon = (

            target_rows[
                "lon"
            ]

            .to_numpy(
                dtype=float
            )
        )

        source_lat = (

            source_rows[
                "lat"
            ]

            .to_numpy(
                dtype=float
            )
        )

        source_lon = (

            source_rows[
                "lon"
            ]

            .to_numpy(
                dtype=float
            )
        )

        dlat_km = (

            target_lat[
                :,
                None
            ]

            -

            source_lat[
                None,
                :
            ]

        ) * 111.32

        mean_lat = np.radians(

            (

                target_lat[
                    :,
                    None
                ]

                +

                source_lat[
                    None,
                    :
                ]

            )

            /
            2.0
        )

        dlon_km = (

            target_lon[
                :,
                None
            ]

            -

            source_lon[
                None,
                :
            ]

        ) * (

            111.32

            *

            np.cos(
                mean_lat
            )
        )

        distance_km = np.sqrt(

            dlat_km
            ** 2

            +

            dlon_km
            ** 2
        )

        nearest_idx = np.argmin(

            distance_km,

            axis=1,
        )

        mapped_mmol = (

            source_rows
            .iloc[
                nearest_idx
            ][
                "oxygen_mmol_m3"
            ]

            .to_numpy(
                dtype=float
            )
        )

        mapped_ml_l = (

            source_rows
            .iloc[
                nearest_idx
            ][
                "oxygen_ml_l"
            ]

            .to_numpy(
                dtype=float
            )
        )

        mapped_distance = distance_km[

            np.arange(
                len(
                    target_rows
                )
            ),

            nearest_idx,
        ]

        result.loc[
            target_mask,
            "oxygen_mmol_m3"
        ] = mapped_mmol

        result.loc[
            target_mask,
            "oxygen_ml_l"
        ] = mapped_ml_l

        result.loc[
            target_mask,
            "oxygen_source_distance_km"
        ] = mapped_distance

    if result[
        "oxygen_ml_l"
    ].isna().all():

        raise RuntimeError(
            "Oxygen data downloaded, but it "
            "could not be matched to the physics grid."
        )

    return result


# ============================================================
# GEBCO RESPONSE PARSER
# ============================================================

def parse_gebco_elevation(
    response_text,
):

    clean_text = re.sub(
        r"<[^>]+>",
        " ",
        response_text,
    )

    clean_text = re.sub(
        r"\s+",
        " ",
        clean_text,
    )

    patterns = [

        r"(?i)elevation[^-+\d]*(-?\d+(?:\.\d+)?)",

        r"(?i)height_above_mean_sea_level[^-+\d]*"
        r"(-?\d+(?:\.\d+)?)",

        r"(?i)value[^-+\d]*(-?\d+(?:\.\d+)?)",

        r"(?i)pixel[^-+\d]*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            clean_text,
        )

        if match:

            return float(
                match.group(
                    1
                )
            )

    return None


# ============================================================
# GEBCO POINT LOOKUP
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def get_gebco_elevation(
    lat,
    lon,
):

    lat = float(
        lat
    )

    lon = float(
        lon
    )

    delta = 0.002

    attempts = [

        {
            "SERVICE":
                "WMS",

            "VERSION":
                "1.3.0",

            "REQUEST":
                "GetFeatureInfo",

            "LAYERS":
                GEBCO_WMS_LAYER,

            "QUERY_LAYERS":
                GEBCO_WMS_LAYER,

            "CRS":
                "EPSG:4326",

            "BBOX":
                (
                    f"{lat - delta},"
                    f"{lon - delta},"
                    f"{lat + delta},"
                    f"{lon + delta}"
                ),

            "WIDTH":
                101,

            "HEIGHT":
                101,

            "I":
                50,

            "J":
                50,

            "INFO_FORMAT":
                "text/plain",
        },

        {
            "SERVICE":
                "WMS",

            "VERSION":
                "1.3.0",

            "REQUEST":
                "GetFeatureInfo",

            "LAYERS":
                GEBCO_WMS_LAYER,

            "QUERY_LAYERS":
                GEBCO_WMS_LAYER,

            "CRS":
                "EPSG:4326",

            "BBOX":
                (
                    f"{lat - delta},"
                    f"{lon - delta},"
                    f"{lat + delta},"
                    f"{lon + delta}"
                ),

            "WIDTH":
                101,

            "HEIGHT":
                101,

            "I":
                50,

            "J":
                50,

            "INFO_FORMAT":
                "text/html",
        },

        {
            "SERVICE":
                "WMS",

            "VERSION":
                "1.1.1",

            "REQUEST":
                "GetFeatureInfo",

            "LAYERS":
                GEBCO_WMS_LAYER,

            "QUERY_LAYERS":
                GEBCO_WMS_LAYER,

            "SRS":
                "EPSG:4326",

            "BBOX":
                (
                    f"{lon - delta},"
                    f"{lat - delta},"
                    f"{lon + delta},"
                    f"{lat + delta}"
                ),

            "WIDTH":
                101,

            "HEIGHT":
                101,

            "X":
                50,

            "Y":
                50,

            "INFO_FORMAT":
                "text/plain",
        },
    ]

    last_error = None

    for params in attempts:

        try:

            response = requests.get(

                GEBCO_WMS_URL,

                params=params,

                timeout=20,
            )

            response.raise_for_status()

            elevation = (
                parse_gebco_elevation(
                    response.text
                )
            )

            if elevation is not None:

                return elevation

            last_error = (
                "GEBCO returned a response, "
                "but no elevation value could "
                "be parsed from it."
            )

        except Exception as error:

            last_error = str(
                error
            )

    raise RuntimeError(
        "GEBCO elevation lookup failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# BUILD GEBCO GRID
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def build_gebco_grid(
    unique_points,
):

    records = []

    for point in unique_points:

        lat = float(
            point[
                0
            ]
        )

        lon = float(
            point[
                1
            ]
        )

        elevation_m = (
            get_gebco_elevation(
                lat,
                lon,
            )
        )

        is_ocean = (
            elevation_m
            <
            0
        )

        water_depth_m = (

            -elevation_m

            if is_ocean

            else np.nan
        )

        records.append(
            {
                "lat_key":
                    round(
                        lat,
                        3,
                    ),

                "lon_key":
                    round(
                        lon,
                        3,
                    ),

                "gebco_elevation_m":
                    float(
                        elevation_m
                    ),

                "water_depth_m":
                    (
                        float(
                            water_depth_m
                        )

                        if is_ocean

                        else np.nan
                    ),

                "is_ocean_gebco":
                    bool(
                        is_ocean
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# ATTACH GEBCO BATHYMETRY
# ============================================================

def attach_gebco_bathymetry(
    dataframe,
):

    result = (
        dataframe.copy()
    )

    unique_grid = (

        result[
            [
                "lat",
                "lon",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            [
                "lat",
                "lon",
            ]
        )
    )

    unique_points = tuple(

        (
            float(
                row.lat
            ),
            float(
                row.lon
            ),
        )

        for row in unique_grid.itertuples(
            index=False
        )
    )

    gebco_grid = (
        build_gebco_grid(
            unique_points
        )
    )

    result = result.merge(

        gebco_grid,

        on=[
            "lat_key",
            "lon_key",
        ],

        how="left",
    )

    if result[
        "gebco_elevation_m"
    ].isna().all():

        raise RuntimeError(
            "GEBCO bathymetry could not "
            "be attached to the E-AWARE grid."
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
            "GEBCO filtering removed all cells. "
            "Please check the bathymetry lookup."
        )

    return result


# ============================================================
# HABITAT REFERENCE
# ============================================================

def habitat_reference(
    month,
):

    if month in [
        12,
        1,
        2,
    ]:

        return {

            "season":
                "Summer",

            "temp_min":
                17.6,

            "temp_max":
                23.7,

            "sal_min":
                32.30,

            "sal_max":
                35.14,

            "oxygen_min":
                5.9,

            "oxygen_max":
                8.7,
        }

    if month in [
        6,
        7,
        8,
    ]:

        return {

            "season":
                "Winter",

            "temp_min":
                14.5,

            "temp_max":
                18.8,

            "sal_min":
                34.81,

            "sal_max":
                35.12,

            "oxygen_min":
                5.2,

            "oxygen_max":
                6.3,
        }

    return {

        "season":
            "Transition (MVP)",

        "temp_min":
            14.5,

        "temp_max":
            23.7,

        "sal_min":
            32.30,

        "sal_max":
            35.14,

        "oxygen_min":
            5.2,

        "oxygen_max":
            8.7,
    }


# ============================================================
# RANGE SCORE
# ============================================================

def range_score(
    values,
    minimum,
    maximum,
    shoulder,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    score = np.ones_like(
        values,
        dtype=float,
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
        1,
    )


# ============================================================
# HSI — V1.2
# ============================================================

def calculate_hsi(
    dataframe,
    month,
):

    df = (
        dataframe.copy()
    )

    reference = (
        habitat_reference(
            month
        )
    )

    df[
        "temperature_score"
    ] = range_score(

        df[
            "temperature"
        ],

        reference[
            "temp_min"
        ],

        reference[
            "temp_max"
        ],

        shoulder=2.0,
    )

    df[
        "salinity_score"
    ] = range_score(

        df[
            "salinity"
        ],

        reference[
            "sal_min"
        ],

        reference[
            "sal_max"
        ],

        shoulder=0.5,
    )

    df[
        "oxygen_score"
    ] = range_score(

        df[
            "oxygen_ml_l"
        ],

        reference[
            "oxygen_min"
        ],

        reference[
            "oxygen_max"
        ],

        shoulder=1.0,
    )

    # ========================================================
    # IMPORTANT V1.2 MODEL DECISION
    #
    # Bathymetry is now available as an environmental
    # predictor but is NOT yet assigned a fixed HSI score.
    #
    # Current temporary HSI:
    #
    # Temperature = 1/3
    # Salinity    = 1/3
    # Oxygen      = 1/3
    # ========================================================

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
        1,
    )

    return df


# ============================================================
# HSI CLASS
# ============================================================

def hsi_class(
    value,
):

    if value >= 0.80:

        return "Very High"

    if value >= 0.60:

        return "High"

    if value >= 0.30:

        return "Moderate"

    return "Low"


# ============================================================
# HABITAT CENTROID
# ============================================================

def habitat_centroid(
    dataframe,
):

    if dataframe.empty:

        return None

    threshold = float(

        dataframe[
            "hsi"
        ]

        .quantile(
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

        best_area = (

            dataframe

            .nlargest(
                1,
                "hsi",
            )

            .copy()
        )

    weights = np.clip(

        best_area[
            "hsi"
        ]

        .to_numpy(),

        0.01,

        None,
    )

    latitude = np.average(

        best_area[
            "lat"
        ],

        weights=weights,
    )

    longitude = np.average(

        best_area[
            "lon"
        ],

        weights=weights,
    )

    return (
        float(
            latitude
        ),
        float(
            longitude
        ),
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2,
):

    earth_radius = (
        6371.0
    )

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    delta_phi = math.radians(
        lat2
        -
        lat1
    )

    delta_lambda = math.radians(
        lon2
        -
        lon1
    )

    a = (

        math.sin(
            delta_phi
            /
            2
        )
        ** 2

        +

        math.cos(
            phi1
        )

        *

        math.cos(
            phi2
        )

        *

        math.sin(
            delta_lambda
            /
            2
        )
        ** 2
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


# ============================================================
# BEARING
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2,
):

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    delta_lambda = math.radians(
        lon2
        -
        lon1
    )

    x = (

        math.sin(
            delta_lambda
        )

        *

        math.cos(
            phi2
        )
    )

    y = (

        math.cos(
            phi1
        )

        *

        math.sin(
            phi2
        )

        -

        math.sin(
            phi1
        )

        *

        math.cos(
            phi2
        )

        *

        math.cos(
            delta_lambda
        )
    )

    bearing = math.degrees(

        math.atan2(
            x,
            y,
        )
    )

    return (

        bearing
        +
        360

    ) % 360


# ============================================================
# DIRECTION
# ============================================================

def direction_name(
    bearing,
):

    directions = [

        "North ↑",

        "North-East ↗",

        "East →",

        "South-East ↘",

        "South ↓",

        "South-West ↙",

        "West ←",

        "North-West ↖",
    ]

    index = int(

        (
            bearing
            +
            22.5
        )

        //
        45

    ) % 8

    return directions[
        index
    ]


# ============================================================
# MOVEMENT COMPONENTS
# ============================================================

def movement_components(
    lat1,
    lon1,
    lat2,
    lon2,
):

    north_km = (

        lat2
        -
        lat1

    ) * 111.32

    mean_latitude = math.radians(

        (
            lat1
            +
            lat2
        )

        /
        2
    )

    east_km = (

        lon2
        -
        lon1

    ) * (

        111.32

        *

        math.cos(
            mean_latitude
        )
    )

    return (
        north_km,
        east_km,
    )


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

        value=date.today(),
    )

    st.caption(
        "E-AWARE requests the selected "
        "day plus the following 72 hours."
    )

    if st.button(
        "🔄 Refresh Data",
        use_container_width=True,
    ):

        st.cache_data.clear()


# ============================================================
# DATE RANGE
# ============================================================

end_date = (

    selected_date

    +

    timedelta(
        days=3
    )
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.status(

        "Loading E-AWARE environmental data...",

        expanded=True,

    ) as status:

        status.write(
            "🔐 Checking Copernicus credentials..."
        )

        username, password = (
            get_credentials()
        )

        credential_check = (
            copernicusmarine.login(

                username=username,

                password=password,

                check_credentials_valid=True,
            )
        )

        if not credential_check:

            raise RuntimeError(
                "Copernicus rejected "
                "the supplied credentials."
            )

        status.write(
            "✅ Copernicus credentials valid"
        )

        status.write(
            "🌡️ 1/5 Downloading temperature..."
        )

        temperature_data = (
            load_temperature(

                selected_date,

                end_date,
            )
        )

        status.write(

            f"✅ Temperature loaded "
            f"({len(temperature_data)} records)"
        )

        status.write(
            "🧂 2/5 Downloading salinity..."
        )

        salinity_data = (
            load_salinity(

                selected_date,

                end_date,
            )
        )

        status.write(

            f"✅ Salinity loaded "
            f"({len(salinity_data)} records)"
        )

        status.write(
            "🌊 3/5 Downloading SST anomaly..."
        )

        anomaly_data = (
            load_sst_anomaly(

                selected_date,

                end_date,
            )
        )

        status.write(

            f"✅ SST anomaly loaded "
            f"({len(anomaly_data)} records)"
        )

        status.write(
            "🫧 4/5 Downloading dissolved oxygen..."
        )

        oxygen_data = (
            load_oxygen(

                selected_date,

                end_date,
            )
        )

        status.write(

            f"✅ Oxygen loaded "
            f"({len(oxygen_data)} records)"
        )

        status.write(
            "🗻 5/5 Loading GEBCO 2026 bathymetry..."
        )

        ocean = (
            merge_physics_data(

                temperature_data,

                salinity_data,

                anomaly_data,
            )
        )

        ocean = (
            attach_nearest_oxygen(

                ocean,

                oxygen_data,
            )
        )

        ocean = (
            attach_gebco_bathymetry(
                ocean
            )
        )

        status.write(
            "✅ GEBCO bathymetry attached "
            "and land cells filtered"
        )

        status.update(

            label=
                "✅ E-AWARE V1.2 environmental data loaded",

            state=
                "complete",

            expanded=
                False,
        )


except Exception as error:

    st.error(
        "❌ E-AWARE V1.2 could not load "
        "all environmental data."
    )

    st.write(
        "Technical error:"
    )

    st.code(
        str(
            error
        )
    )

    st.stop()


# ============================================================
# AVAILABLE DATES
# ============================================================

available_dates = sorted(

    ocean[
        "date"
    ]

    .unique()
)


if len(
    available_dates
) == 0:

    st.error(
        "No valid ocean dates were returned."
    )

    st.stop()


# ============================================================
# CURRENT DATE
# ============================================================

if selected_date in available_dates:

    current_date = (
        selected_date
    )

else:

    current_date = (
        available_dates[
            0
        ]
    )


# ============================================================
# CURRENT DATA
# ============================================================

current_data = ocean[

    ocean[
        "date"
    ]

    ==
    current_date

].copy()


current_data = (

    current_data

    .dropna(
        subset=[
            "oxygen_ml_l",
            "water_depth_m",
        ]
    )

    .copy()
)


if current_data.empty:

    st.error(
        "No complete oxygen + bathymetry "
        "matched ocean cells are available "
        "for the selected date."
    )

    st.stop()


# ============================================================
# CURRENT HSI
# ============================================================

current_data = (
    calculate_hsi(

        current_data,

        current_date.month,
    )
)


current_data[
    "class"
] = (

    current_data[
        "hsi"
    ]

    .apply(
        hsi_class
    )
)


best = current_data.loc[

    current_data[
        "hsi"
    ]

    .idxmax()
]


reference = (
    habitat_reference(
        current_date.month
    )
)


# ============================================================
# CURRENT CONDITIONS
# ============================================================

st.subheader(
    "🌊 Current Ocean Conditions"
)


metric1, metric2, metric3, metric4 = (
    st.columns(
        4
    )
)


metric1.metric(
    "Data Date",
    str(
        current_date
    )
)


metric2.metric(
    "Highest HSI",
    f"{best['hsi']:.2f}"
)


metric3.metric(
    "Highest-HSI SST",
    f"{best['temperature']:.2f} °C"
)


metric4.metric(
    "Dissolved Oxygen",
    f"{best['oxygen_ml_l']:.2f} mL/L"
)


context1, context2, context3 = (
    st.columns(
        3
    )
)


context1.metric(
    "Salinity",
    f"{best['salinity']:.2f} PSU"
)


context2.metric(

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
    ),
)


context3.metric(
    "Water Depth",
    f"{best['water_depth_m']:.0f} m"
)


st.caption(
    "V1.2 adds GEBCO 2026 bathymetry as a "
    "spatial environmental predictor. Depth is "
    "displayed and stored, but is not yet given "
    "an HSI weight."
)


# ============================================================
# HABITAT MAP
# ============================================================

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
        "SST Anomaly",
    ],

    horizontal=True,
)


map_data = (
    current_data.copy()
)


# ============================================================
# MAP COLOURS
# ============================================================

if map_mode == (
    "Habitat Suitability"
):

    map_data[
        "r"
    ] = (

        (
            1
            -
            map_data[
                "hsi"
            ]
        )

        *
        220
    )

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


elif map_mode == (
    "Temperature"
):

    values = (
        map_data[
            "temperature"
        ]
    )

    spread = max(

        float(
            values.max()
            -
            values.min()
        ),

        0.01,
    )

    normalised = (

        values
        -
        values.min()

    ) / spread

    map_data[
        "r"
    ] = (

        80

        +
        170
        *
        normalised
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
        normalised
    )


elif map_mode == (
    "Salinity"
):

    values = (
        map_data[
            "salinity"
        ]
    )

    spread = max(

        float(
            values.max()
            -
            values.min()
        ),

        0.001,
    )

    normalised = (

        values
        -
        values.min()

    ) / spread

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
        normalised
    )

    map_data[
        "b"
    ] = 220


elif map_mode == (
    "Dissolved Oxygen"
):

    values = (
        map_data[
            "oxygen_ml_l"
        ]
    )

    spread = max(

        float(
            values.max()
            -
            values.min()
        ),

        0.001,
    )

    normalised = (

        values
        -
        values.min()

    ) / spread

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
        normalised
    )

    map_data[
        "b"
    ] = (

        230

        -
        80
        *
        normalised
    )


elif map_mode == (
    "Bathymetry"
):

    values = (
        map_data[
            "water_depth_m"
        ]
    )

    spread = max(

        float(
            values.max()
            -
            values.min()
        ),

        1.0,
    )

    normalised = (

        values
        -
        values.min()

    ) / spread

    map_data[
        "r"
    ] = (

        60

        -
        35
        *
        normalised
    )

    map_data[
        "g"
    ] = (

        170

        -
        90
        *
        normalised
    )

    map_data[
        "b"
    ] = (

        210

        +
        40
        *
        normalised
    )


else:

    values = (

        map_data[
            "sst_anomaly"
        ]

        .fillna(
            0
        )
    )

    spread = max(

        abs(
            float(
                values.min()
            )
        ),

        abs(
            float(
                values.max()
            )
        ),

        0.1,
    )

    normalised = (

        (
            values
            /
            spread
        )

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
        normalised
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
        normalised
    )


# ============================================================
# MAP LAYERS
# ============================================================

ocean_layer = pdk.Layer(

    "ScatterplotLayer",

    data=
        map_data,

    get_position=
        [
            "lon",
            "lat",
        ],

    get_fill_color=
        [
            "r",
            "g",
            "b",
            190,
        ],

    get_radius=
        700,

    radius_min_pixels=
        6,

    radius_max_pixels=
        18,

    pickable=
        True,
)


community_layer = pdk.Layer(

    "ScatterplotLayer",

    data=
        COMMUNITIES,

    get_position=
        [
            "lon",
            "lat",
        ],

    get_fill_color=
        [
            255,
            255,
            255,
            240,
        ],

    get_line_color=
        [
            20,
            20,
            20,
            255,
        ],

    stroked=
        True,

    line_width_min_pixels=
        2,

    get_radius=
        350,

    pickable=
        True,
)


st.pydeck_chart(

    pdk.Deck(

        map_style=(

            "https://basemaps.cartocdn.com/"
            "gl/positron-gl-style/style.json"
        ),

        initial_view_state=
            pdk.ViewState(

                latitude=
                    -4.49,

                longitude=
                    -81.36,

                zoom=
                    9.5,

                pitch=
                    0,
            ),

        layers=[
            ocean_layer,
            community_layer,
        ],

        tooltip={

            "html":

                "<b>{class} Habitat</b><br/>"
                "HSI: {hsi}<br/>"
                "SST: {temperature} °C<br/>"
                "Salinity: {salinity} PSU<br/>"
                "Oxygen: {oxygen_ml_l} mL/L<br/>"
                "Water depth: {water_depth_m} m<br/>"
                "GEBCO elevation: {gebco_elevation_m} m<br/>"
                "SST anomaly: {sst_anomaly} °C"
        },
    ),

    use_container_width=True,
)


st.caption(
    "GEBCO 2026 bathymetry is queried from the "
    "official GEBCO WMS. Negative GEBCO elevation "
    "is converted to positive water depth, and cells "
    "identified as land are excluded from the "
    "E-AWARE ocean grid."
)


# ============================================================
# HIGHEST RELATIVE HABITAT
# ============================================================

st.subheader(
    "📍 Highest Relative Habitat Suitability"
)


zone1, zone2, zone3, zone4 = (
    st.columns(
        4
    )
)


zone1.metric(
    "Latitude",
    f"{best['lat']:.4f}"
)


zone2.metric(
    "Longitude",
    f"{best['lon']:.4f}"
)


zone3.metric(
    "Suitability",
    hsi_class(
        best[
            "hsi"
        ]
    )
)


zone4.metric(
    "Water Depth",
    f"{best['water_depth_m']:.0f} m"
)


st.caption(
    "This is the highest-scoring E-AWARE cell "
    "within the current study area. Bathymetry "
    "is shown as environmental context in V1.2 "
    "and is not yet weighted in the HSI."
)


# ============================================================
# HABITAT REFERENCE
# ============================================================

st.subheader(
    "🐟 Why is this habitat considered suitable?"
)


reference1, reference2, reference3, reference4 = (
    st.columns(
        4
    )
)


reference1.metric(

    "Reference Temperature",

    (
        f"{reference['temp_min']} – "
        f"{reference['temp_max']} °C"
    )
)


reference2.metric(

    "Reference Salinity",

    (
        f"{reference['sal_min']} – "
        f"{reference['sal_max']} PSU"
    )
)


reference3.metric(

    "Reference Oxygen",

    (
        f"{reference['oxygen_min']} – "
        f"{reference['oxygen_max']} mL/L"
    )
)


reference4.metric(
    "Season",
    reference[
        "season"
    ]
)


st.warning(
    "V1.2 continues to calculate HSI from "
    "temperature, salinity and surface dissolved "
    "oxygen. Bathymetry has now been integrated "
    "as a predictor, but no fixed bathymetry "
    "suitability score has been invented. "
    "Depth will later be incorporated through "
    "model training and validation."
)


# ============================================================
# FORECAST PROCESSING
# ============================================================

forecast_centres = []

forecast_results = []


for forecast_date in available_dates:

    forecast_frame = ocean[

        ocean[
            "date"
        ]

        ==
        forecast_date

    ].copy()

    forecast_frame = (

        forecast_frame

        .dropna(
            subset=[
                "oxygen_ml_l",
                "water_depth_m",
            ]
        )

        .copy()
    )

    if forecast_frame.empty:

        continue

    forecast_frame = (
        calculate_hsi(

            forecast_frame,

            forecast_date.month,
        )
    )

    centre = (
        habitat_centroid(
            forecast_frame
        )
    )

    if centre is not None:

        forecast_centres.append(

            {
                "date":
                    forecast_date,

                "lat":
                    centre[
                        0
                    ],

                "lon":
                    centre[
                        1
                    ],
            }
        )

    best_forecast = forecast_frame.loc[

        forecast_frame[
            "hsi"
        ]

        .idxmax()
    ]

    forecast_results.append(

        {
            "Date":
                forecast_date,

            "Highest HSI":
                round(
                    float(
                        best_forecast[
                            "hsi"
                        ]
                    ),
                    2,
                ),

            "Temperature °C":
                round(
                    float(
                        best_forecast[
                            "temperature"
                        ]
                    ),
                    2,
                ),

            "Salinity PSU":
                round(
                    float(
                        best_forecast[
                            "salinity"
                        ]
                    ),
                    2,
                ),

            "Oxygen mL/L":
                round(
                    float(
                        best_forecast[
                            "oxygen_ml_l"
                        ]
                    ),
                    2,
                ),

            "Water Depth m":
                round(
                    float(
                        best_forecast[
                            "water_depth_m"
                        ]
                    ),
                    0,
                ),

            "SST Anomaly °C":
                (

                    round(
                        float(
                            best_forecast[
                                "sst_anomaly"
                            ]
                        ),
                        2,
                    )

                    if pd.notna(
                        best_forecast[
                            "sst_anomaly"
                        ]
                    )

                    else None
                ),

            "Habitat Centre Lat":
                (

                    round(
                        centre[
                            0
                        ],
                        4,
                    )

                    if centre is not None

                    else None
                ),

            "Habitat Centre Lon":
                (

                    round(
                        centre[
                            1
                        ],
                        4,
                    )

                    if centre is not None

                    else None
                ),
        }
    )


# ============================================================
# HABITAT SHIFT
# ============================================================

st.subheader(
    "🧭 Predicted Habitat Shift"
)


direction_text = (
    "Unavailable"
)

distance_text = (
    "—"
)

bearing_text = (
    "—"
)

period_text = (
    "Insufficient forecast data"
)

start_point_text = (
    "—"
)

end_point_text = (
    "—"
)

north_south_text = (
    "—"
)

east_west_text = (
    "—"
)

start = None
end = None
distance = None
bearing = None


if len(
    forecast_centres
) >= 2:

    start = (
        forecast_centres[
            0
        ]
    )

    end = (
        forecast_centres[
            -1
        ]
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
        ],
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
        ],
    )

    north_km, east_km = (
        movement_components(

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
            ],
        )
    )

    distance_text = (
        f"{distance:.2f} km"
    )

    bearing_text = (
        f"{bearing:.1f}°"
    )

    period_text = (

        f"{start['date'].strftime('%d %b %Y')} "
        f"→ "
        f"{end['date'].strftime('%d %b %Y')}"
    )

    start_point_text = (

        f"{start['lat']:.4f}, "
        f"{start['lon']:.4f}"
    )

    end_point_text = (

        f"{end['lat']:.4f}, "
        f"{end['lon']:.4f}"
    )

    if north_km > 0:

        north_south_text = (
            f"{abs(north_km):.2f} km North"
        )

    elif north_km < 0:

        north_south_text = (
            f"{abs(north_km):.2f} km South"
        )

    else:

        north_south_text = (
            "0.00 km"
        )

    if east_km > 0:

        east_west_text = (
            f"{abs(east_km):.2f} km East"
        )

    elif east_km < 0:

        east_west_text = (
            f"{abs(east_km):.2f} km West"
        )

    else:

        east_west_text = (
            "0.00 km"
        )

    if distance >= 2.0:

        direction_text = (
            direction_name(
                bearing
            )
        )

    else:

        direction_text = (
            "No clear shift"
        )


# ============================================================
# SHIFT METRICS
# ============================================================

shift1, shift2, shift3 = (
    st.columns(
        3
    )
)


shift1.metric(
    "Predicted Direction",
    direction_text
)


shift2.metric(
    "Total Shift",
    distance_text
)


shift3.metric(
    "Bearing",
    bearing_text
)


st.markdown(
    f"**Forecast Period:** "
    f"{period_text}"
)


# ============================================================
# MOVEMENT BREAKDOWN
# ============================================================

st.markdown(
    "#### Movement Breakdown"
)


movement1, movement2 = (
    st.columns(
        2
    )
)


movement1.metric(
    "North / South",
    north_south_text
)


movement2.metric(
    "East / West",
    east_west_text
)


# ============================================================
# HABITAT CENTRE POSITIONS
# ============================================================

st.markdown(
    "#### Habitat-Centre Positions"
)


position1, position2 = (
    st.columns(
        2
    )
)


position1.metric(
    "Current / Start Position",
    start_point_text
)


position2.metric(
    "+72 h Forecast Position",
    end_point_text
)


st.info(
    "This is the forecast shift of the "
    "relatively highest-suitability habitat "
    "within the study area. It does not mean "
    "individual anchoveta schools are being "
    "directly tracked."
)


# ============================================================
# HABITAT SHIFT MAP
# ============================================================

st.markdown(
    "#### 🗺️ Habitat Shift Map"
)


if (
    start is not None
    and
    end is not None
):

    shift_path_data = [

        {
            "path":
                [
                    [
                        start[
                            "lon"
                        ],
                        start[
                            "lat"
                        ],
                    ],

                    [
                        end[
                            "lon"
                        ],
                        end[
                            "lat"
                        ],
                    ],
                ],

            "name":
                (
                    f"{direction_text} • "
                    f"{distance_text}"
                ),
        }
    ]

    start_marker = pd.DataFrame(

        [
            {
                "label":
                    "Current habitat centre",

                "lat":
                    start[
                        "lat"
                    ],

                "lon":
                    start[
                        "lon"
                    ],
            }
        ]
    )

    end_marker = pd.DataFrame(

        [
            {
                "label":
                    "+72 h habitat centre",

                "lat":
                    end[
                        "lat"
                    ],

                "lon":
                    end[
                        "lon"
                    ],
            }
        ]
    )

    label_points = pd.DataFrame(

        [
            {
                "label":
                    "START",

                "lat":
                    start[
                        "lat"
                    ],

                "lon":
                    start[
                        "lon"
                    ],
            },

            {
                "label":
                    "+72 h",

                "lat":
                    end[
                        "lat"
                    ],

                "lon":
                    end[
                        "lon"
                    ],
            },
        ]
    )

    path_layer = pdk.Layer(

        "PathLayer",

        data=
            shift_path_data,

        get_path=
            "path",

        get_color=
            [
                0,
                140,
                255,
                230,
            ],

        get_width=
            5,

        width_min_pixels=
            5,

        pickable=
            True,
    )

    start_layer = pdk.Layer(

        "ScatterplotLayer",

        data=
            start_marker,

        get_position=
            [
                "lon",
                "lat",
            ],

        get_fill_color=
            [
                30,
                190,
                90,
                240,
            ],

        get_line_color=
            [
                255,
                255,
                255,
                255,
            ],

        stroked=
            True,

        line_width_min_pixels=
            2,

        get_radius=
            500,

        radius_min_pixels=
            8,

        pickable=
            True,
    )

    end_layer = pdk.Layer(

        "ScatterplotLayer",

        data=
            end_marker,

        get_position=
            [
                "lon",
                "lat",
            ],

        get_fill_color=
            [
                255,
                145,
                40,
                240,
            ],

        get_line_color=
            [
                255,
                255,
                255,
                255,
            ],

        stroked=
            True,

        line_width_min_pixels=
            2,

        get_radius=
            500,

        radius_min_pixels=
            8,

        pickable=
            True,
    )

    label_layer = pdk.Layer(

        "TextLayer",

        data=
            label_points,

        get_position=
            [
                "lon",
                "lat",
            ],

        get_text=
            "label",

        get_size=
            16,

        get_color=
            [
                20,
                20,
                20,
                255,
            ],

        get_pixel_offset=
            [
                0,
                -18,
            ],

        pickable=
            False,
    )

    shift_mid_lat = (

        start[
            "lat"
        ]

        +

        end[
            "lat"
        ]

    ) / 2

    shift_mid_lon = (

        start[
            "lon"
        ]

        +

        end[
            "lon"
        ]

    ) / 2

    st.pydeck_chart(

        pdk.Deck(

            map_style=(

                "https://basemaps.cartocdn.com/"
                "gl/positron-gl-style/style.json"
            ),

            initial_view_state=
                pdk.ViewState(

                    latitude=
                        shift_mid_lat,

                    longitude=
                        shift_mid_lon,

                    zoom=
                        10.3,

                    pitch=
                        0,
                ),

            layers=
                [
                    path_layer,
                    start_layer,
                    end_layer,
                    label_layer,
                ],

            tooltip=
                {
                    "html":
                        "<b>{label}{name}</b>"
                },
        ),

        use_container_width=True,
    )

    st.caption(
        "🟢 Current habitat centre   •   "
        "🟠 +72 h forecast habitat centre   •   "
        "🔵 Predicted habitat-centre shift"
    )

    st.markdown(

        f"**E-AWARE interpretation:** "
        f"The centre of the relatively most "
        f"suitable anchoveta habitat is forecast "
        f"to shift **{distance_text} "
        f"{direction_text.lower()}** over the "
        f"forecast period, at a bearing of "
        f"**{bearing_text}**."
    )


else:

    st.warning(
        "Not enough forecast information "
        "is available to draw the "
        "habitat-shift map."
    )


# ============================================================
# 72-HOUR FORECAST
# ============================================================

st.subheader(
    "📅 72-Hour Habitat Forecast"
)


st.dataframe(

    pd.DataFrame(
        forecast_results
    ),

    use_container_width=True,

    hide_index=True,
)


# ============================================================
# SMS ALERT PREVIEW
# ============================================================

st.subheader(
    "📱 Fisher SMS Alert Preview"
)


best_hsi = float(
    best[
        "hsi"
    ]
)


habitat_status = (

    hsi_class(
        best_hsi
    )

    .upper()
)


sms_message = f"""E-AWARE Fisheries Alert

Anchoveta habitat suitability:
{habitat_status}

Habitat Suitability Index:
{best_hsi:.2f}

Predicted suitable-habitat shift:
{direction_text}

Estimated habitat shift:
{distance_text}

Bearing:
{bearing_text}

Sea temperature:
{best['temperature']:.1f} °C

Salinity:
{best['salinity']:.2f} PSU

Dissolved oxygen:
{best['oxygen_ml_l']:.2f} mL/L

Water depth at highest-HSI cell:
{best['water_depth_m']:.0f} m

Check official fishing restrictions before departure.

E-AWARE provides habitat guidance only.
"""


st.text_area(

    "Example SMS",

    sms_message,

    height=430,
)


st.caption(
    "MVP V1.2 only previews the SMS. "
    "Real SMS delivery will be added later. "
    "Depth is shown as context and is not "
    "yet part of the HSI weighting."
)


# ============================================================
# DATA SOURCES
# ============================================================

st.subheader(
    "📊 Data Sources & Transparency"
)


source_table = pd.DataFrame(

    [
        {
            "Source":
                "Copernicus Marine Physics",

            "Status":
                "Analysis / forecast",

            "Use":
                "Sea temperature and salinity",
        },

        {
            "Source":
                "Copernicus Marine Physics",

            "Status":
                "Analysis / forecast",

            "Use":
                "SST anomaly",
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry",

            "Status":
                "Analysis / forecast",

            "Use":
                "Surface dissolved oxygen",
        },

        {
            "Source":
                "GEBCO 2026",

            "Status":
                "Static terrain model",

            "Use":
                "Bathymetry / water depth and land filtering",
        },

        {
            "Source":
                "Castillo et al. (2019)",

            "Status":
                "Scientific reference",

            "Use":
                "Anchoveta temperature, salinity and oxygen ranges",
        },

        {
            "Source":
                "Hill et al. (2026)",

            "Status":
                "Scientific reference",

            "Use":
                "Supports bathymetry as a habitat-model predictor",
        },

        {
            "Source":
                "IMARPE",

            "Status":
                "Scientific observations",

            "Use":
                "Future training / validation",
        },

        {
            "Source":
                "PRODUCE",

            "Status":
                "Regulatory",

            "Use":
                "Future protection / closure layer",
        },
    ]
)


st.dataframe(

    source_table,

    use_container_width=True,

    hide_index=True,
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "🧪 V1.2 Model Status"
)


status1, status2 = (
    st.columns(
        2
    )
)


status1.success(
    "Bathymetry integrated: GEBCO water "
    "depth is now associated with every "
    "retained E-AWARE ocean grid cell."
)


status2.warning(
    "Bathymetry is not yet weighted in the "
    "HSI. A fixed depth score will not be "
    "added until there is a defensible model "
    "or trained relationship."
)


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "⚙️ MVP V1.2 Limitations"
):

    st.markdown(
        """
### Current HSI inputs

- Sea temperature
- Salinity
- Surface dissolved oxygen

### Additional environmental predictors/context

- SST anomaly
- GEBCO bathymetry / water depth
- Season

### Bathymetry treatment in V1.2

GEBCO provides elevation relative to sea level.

For ocean cells:

**Water depth = -1 × GEBCO elevation**

Bathymetry is now stored and displayed for every retained
E-AWARE grid cell.

However, depth is **not yet given a hand-built HSI score
or weighting**.

This is intentional. The literature supports bathymetry as
an anchoveta habitat predictor, but the current project has
not established a universal fixed depth range that would
justify an arbitrary scoring function.

### GEBCO land filtering

Cells where GEBCO elevation is zero or positive are treated
as land and removed from the habitat grid.

### Oxygen limitation

The Copernicus biogeochemistry oxygen product is coarser
than the physics grid.

V1.2 therefore maps each physics cell to the nearest oxygen
grid cell for the same day.

### Habitat-shift calculation

E-AWARE calculates the centre of the top 25% highest-scoring
habitat cells for each forecast day and compares the current
centre with the final forecast centre.

The movement shown is a shift in **predicted habitat
suitability**, not direct tracking of anchoveta.

### Not yet integrated

- Oxycline depth
- Distance from coast
- Chlorophyll
- Ocean currents in the HSI
- Historical IMARPE anchoveta observations
- Juvenile protection
- Fishing closures

### Model status

V1.2 remains a transparent engineering prototype.

The HSI still uses equal temporary weights for temperature,
salinity and oxygen.

Bathymetry is now available as a predictor for later model
training and validation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "E-AWARE MVP V1.2 | "
    "El Niño Adaptive Fisheries Intelligence | "
    "Decision-support research prototype"
)


st.caption(
    "Habitat suitability is a model estimate. "
    "It is not confirmed fish location, catch advice, "
    "or a legal fishing instruction. GEBCO bathymetry "
    "must not be used for navigation or safety at sea."
)
