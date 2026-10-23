import os
import math
import tempfile
from pathlib import Path
from datetime import date, timedelta

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

st.set_page_config(
    page_title="E-AWARE",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 E-AWARE")

st.caption(
    "El Niño Adaptive Fisheries Intelligence — V2 Step 1"
)

st.info(
    "E-AWARE estimates suitable anchoveta habitat from "
    "environmental conditions. V2 Step 1 adds historical "
    "anchoveta observation ingestion and validation, but "
    "does not train a machine-learning habitat model yet."
)


MIN_LON = -81.50
MAX_LON = -81.20

MIN_LAT = -4.60
MAX_LAT = -4.36


COMMUNITIES = pd.DataFrame(
    [
        {
            "name": "Lobitos",
            "lat": -4.4567,
            "lon": -81.2849
        },
        {
            "name": "Siches",
            "lat": -4.4890,
            "lon": -81.2680
        },
        {
            "name": "Piedritas",
            "lat": -4.5190,
            "lon": -81.2630
        }
    ]
)


# ============================================================
# DATA SOURCES / CONSTANTS
# ============================================================

TEMPERATURE_DATASET = (
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
)

SALINITY_DATASET = (
    "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
)

SST_ANOMALY_DATASET = (
    "cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m"
)

OXYGEN_DATASET = (
    "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
)


GEBCO_API_URL = (
    "https://api.odb.ntu.edu.tw/gebco"
)


NOAA_COASTLINE_QUERY_URL = (
    "https://gis.ngdc.noaa.gov/"
    "arcgis/rest/services/"
    "EMAG2v3/MapServer/1/query"
)


BASEMAP_URL = (
    "https://basemaps.cartocdn.com/"
    "gl/positron-gl-style/style.json"
)


COASTLINE_QUERY_MARGIN_DEG = 0.75

KM_PER_NAUTICAL_MILE = 1.852

OXYGEN_MMOL_M3_TO_ML_L = 0.0223916

OXYCLINE_THRESHOLD_ML_L = 1.0


HISTORICAL_REPO_PATH = Path(
    "data/imarpe_anchoveta_observations.csv"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_credentials():

    username = st.secrets.get(
        "COPERNICUS_USERNAME",
        ""
    )

    password = st.secrets.get(
        "COPERNICUS_PASSWORD",
        ""
    )

    if not username or not password:

        raise RuntimeError(
            "Copernicus credentials are missing "
            "from Streamlit Secrets."
        )

    return username, password


def resolve_downloaded_file(
    response,
    temp_dir,
    filename
):

    possible = Path(
        str(
            response.file_path
        )
    )


    if possible.exists():

        return possible


    expected = (
        Path(
            temp_dir
        )
        /
        filename
    )


    if expected.exists():

        return expected


    candidates = list(
        Path(
            temp_dir
        )
        .glob(
            "*.nc"
        )
    )


    if not candidates:

        raise RuntimeError(
            "Copernicus download completed "
            "but no NetCDF file was found."
        )


    return candidates[
        0
    ]


def prepare_dataframe(
    dataframe,
    variable_name
):

    df = dataframe.copy()

    rename_map = {}


    for column in df.columns:

        column_lower = str(
            column
        ).lower()


        if column_lower in [
            "latitude",
            "lat"
        ]:

            rename_map[
                column
            ] = "lat"


        elif column_lower in [
            "longitude",
            "lon"
        ]:

            rename_map[
                column
            ] = "lon"


    df = df.rename(
        columns=rename_map
    )


    required = [
        "time",
        "lat",
        "lon",
        variable_name
    ]


    missing = [
        column
        for column in required
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
            variable_name
        ]
    ]


# ============================================================
# COPERNICUS SURFACE DATA
# ============================================================

def download_subset(
    dataset_id,
    variable,
    output_name,
    start_date,
    end_date,
    use_depth=True
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


        arguments = {

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
                True
        }


        if use_depth:

            arguments[
                "minimum_depth"
            ] = 0

            arguments[
                "maximum_depth"
            ] = 1


        response = (
            copernicusmarine.subset(
                **arguments
            )
        )


        file_path = (
            resolve_downloaded_file(

                response,

                temp_dir,

                filename
            )
        )


        with xr.open_dataset(
            file_path
        ) as dataset:


            if variable not in dataset.data_vars:

                raise RuntimeError(
                    f"Variable '{variable}' "
                    f"was not found in "
                    f"{output_name}."
                )


            data_array = (
                dataset[
                    variable
                ]
            )


            if "depth" in data_array.dims:

                data_array = (
                    data_array.isel(
                        depth=0
                    )
                )


            dataframe = (

                data_array

                .load()

                .to_dataframe(
                    name=output_name
                )

                .reset_index()
            )


    return prepare_dataframe(

        dataframe,

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

    return download_subset(

        TEMPERATURE_DATASET,

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

    return download_subset(

        SALINITY_DATASET,

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

    return download_subset(

        SST_ANOMALY_DATASET,

        "sea_surface_temperature_anomaly",

        "sst_anomaly",

        start_date,

        end_date,

        False
    )


# ============================================================
# COPERNICUS 3D OXYGEN
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_oxygen_profile(
    start_date,
    end_date
):

    username, password = (
        get_credentials()
    )


    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"


    with tempfile.TemporaryDirectory() as temp_dir:

        filename = (
            "oxygen_profile.nc"
        )


        response = (
            copernicusmarine.subset(

                dataset_id=
                    OXYGEN_DATASET,

                variables=
                    [
                        "o2"
                    ],

                minimum_longitude=
                    MIN_LON,

                maximum_longitude=
                    MAX_LON,

                minimum_latitude=
                    MIN_LAT,

                maximum_latitude=
                    MAX_LAT,

                start_datetime=
                    f"{start_date}T00:00:00",

                end_datetime=
                    f"{end_date}T23:59:59",

                username=
                    username,

                password=
                    password,

                service=
                    "arco-geo-series",

                output_filename=
                    filename,

                output_directory=
                    temp_dir,

                disable_progress_bar=
                    True
            )
        )


        file_path = (
            resolve_downloaded_file(

                response,

                temp_dir,

                filename
            )
        )


        with xr.open_dataset(
            file_path
        ) as dataset:


            if "o2" not in dataset.data_vars:

                raise RuntimeError(
                    "Variable 'o2' was not "
                    "found in the oxygen-profile file."
                )


            dataframe = (

                dataset[
                    "o2"
                ]

                .load()

                .to_dataframe(
                    name="oxygen_mmol_m3"
                )

                .reset_index()
            )


    rename_map = {}


    for column in dataframe.columns:

        column_lower = str(
            column
        ).lower()


        if column_lower in [
            "latitude",
            "lat"
        ]:

            rename_map[
                column
            ] = "lat"


        elif column_lower in [
            "longitude",
            "lon"
        ]:

            rename_map[
                column
            ] = "lon"


    dataframe = dataframe.rename(
        columns=rename_map
    )


    required = [
        "time",
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]


    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]


    if missing:

        raise RuntimeError(
            f"Missing oxygen-profile "
            f"columns: {missing}"
        )


    dataframe = dataframe.dropna(
        subset=[
            "oxygen_mmol_m3",
            "depth"
        ]
    ).copy()


    dataframe[
        "date"
    ] = (
        pd.to_datetime(
            dataframe[
                "time"
            ]
        )
        .dt.date
    )


    dataframe[
        "depth"
    ] = (
        dataframe[
            "depth"
        ]
        .astype(
            float
        )
    )


    dataframe[
        "lat"
    ] = (
        dataframe[
            "lat"
        ]
        .astype(
            float
        )
    )


    dataframe[
        "lon"
    ] = (
        dataframe[
            "lon"
        ]
        .astype(
            float
        )
    )


    dataframe[
        "lat_key"
    ] = (
        dataframe[
            "lat"
        ]
        .round(
            3
        )
    )


    dataframe[
        "lon_key"
    ] = (
        dataframe[
            "lon"
        ]
        .round(
            3
        )
    )


    dataframe[
        "oxygen_ml_l"
    ] = (

        dataframe[
            "oxygen_mmol_m3"
        ]

        .astype(
            float
        )

        *
        OXYGEN_MMOL_M3_TO_ML_L
    )


    return dataframe[
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
    ]


def surface_oxygen_from_profile(
    profile
):

    return (

        profile

        .sort_values(
            "depth"
        )

        .groupby(
            [
                "date",
                "lat_key",
                "lon_key"
            ],
            as_index=False
        )

        .head(
            1
        )

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

        .sort_values(
            "depth"
        )

        .drop_duplicates(
            subset=[
                "depth"
            ]
        )
    )


    if clean.empty:

        return (
            np.nan,
            "No valid oxygen profile"
        )


    depths = (
        clean[
            "depth"
        ]
        .to_numpy(
            dtype=float
        )
    )


    oxygen = (
        clean[
            "oxygen_ml_l"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if oxygen[
        0
    ] <= threshold_ml_l:

        return (

            float(
                depths[
                    0
                ]
            ),

            (
                "Threshold reached at "
                "shallowest model level"
            )
        )


    for index in range(
        1,
        len(
            depths
        )
    ):


        upper_oxygen = (
            oxygen[
                index
                -
                1
            ]
        )


        lower_oxygen = (
            oxygen[
                index
            ]
        )


        if (
            upper_oxygen
            >
            threshold_ml_l

            and

            lower_oxygen
            <=
            threshold_ml_l
        ):


            upper_depth = (
                depths[
                    index
                    -
                    1
                ]
            )


            lower_depth = (
                depths[
                    index
                ]
            )


            if lower_oxygen == upper_oxygen:

                crossing_depth = (
                    lower_depth
                )


            else:

                fraction = (

                    threshold_ml_l
                    -
                    upper_oxygen

                ) / (

                    lower_oxygen
                    -
                    upper_oxygen
                )


                crossing_depth = (

                    upper_depth

                    +

                    fraction
                    *
                    (
                        lower_depth
                        -
                        upper_depth
                    )
                )


            return (

                float(
                    crossing_depth
                ),

                (
                    "Interpolated "
                    "1 mL/L crossing"
                )
            )


    return (

        np.nan,

        (
            "1 mL/L threshold not reached "
            "in available profile"
        )
    )


# ============================================================
# PHYSICS / NEAREST BGC MAPPING
# ============================================================

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

            how="inner"
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

            how="left"
        )
    )


    if merged.empty:

        raise RuntimeError(
            "No matching physics grid cells "
            "were found."
        )


    return merged


def nearest_source_indices(
    target_rows,
    source_rows
):

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


    mean_latitude = np.radians(

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
            mean_latitude
        )
    )


    distance_km = np.sqrt(

        dlat_km
        ** 2

        +

        dlon_km
        ** 2
    )


    nearest_index = np.argmin(

        distance_km,

        axis=1
    )


    nearest_distance = distance_km[

        np.arange(
            len(
                target_rows
            )
        ),

        nearest_index
    ]


    return (
        nearest_index,
        nearest_distance
    )


def attach_nearest_oxygen(
    physics,
    oxygen
):

    result = (
        physics.copy()
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


        source_rows = oxygen[

            oxygen[
                "date"
            ]

            ==
            target_date

        ]


        if source_rows.empty:

            continue


        (
            nearest_index,
            nearest_distance
        ) = nearest_source_indices(

            target_rows,

            source_rows
        )


        nearest_rows = (

            source_rows

            .iloc[
                nearest_index
            ]
        )


        result.loc[
            target_mask,
            "oxygen_mmol_m3"
        ] = (

            nearest_rows[
                "oxygen_mmol_m3"
            ]

            .to_numpy(
                dtype=float
            )
        )


        result.loc[
            target_mask,
            "oxygen_ml_l"
        ] = (

            nearest_rows[
                "oxygen_ml_l"
            ]

            .to_numpy(
                dtype=float
            )
        )


        result.loc[
            target_mask,
            "oxygen_source_distance_km"
        ] = (
            nearest_distance
        )


    if result[
        "oxygen_ml_l"
    ].isna().all():

        raise RuntimeError(
            "Oxygen data could not be "
            "matched to the physics grid."
        )


    return result


# ============================================================
# GEBCO HELPERS
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

        value = float(
            value
        )

        if math.isfinite(
            value
        ):

            return value

        return None


    if isinstance(
        value,
        str
    ):

        try:

            value = float(
                value
            )

            if math.isfinite(
                value
            ):

                return value

        except ValueError:

            return None


    if isinstance(
        value,
        dict
    ):

        if "z" in value:

            return first_numeric(
                value[
                    "z"
                ]
            )


        for key in [
            "data",
            "result",
            "results",
            "records"
        ]:

            if key in value:

                found = first_numeric(
                    value[
                        key
                    ]
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

        number = first_numeric(
            value
        )


        if number is None:

            return []


        return [
            number
        ]


    if isinstance(
        value,
        dict
    ):


        if "z" in value:

            return numeric_list(
                value[
                    "z"
                ]
            )


        for key in [
            "data",
            "result",
            "results",
            "records"
        ]:

            if key in value:

                values = numeric_list(
                    value[
                        key
                    ]
                )


                if values:

                    return values


        return []


    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        output = []


        for item in value:


            if (
                isinstance(
                    item,
                    dict
                )

                and

                "z" in item
            ):

                number = first_numeric(
                    item[
                        "z"
                    ]
                )


            else:

                number = first_numeric(
                    item
                )


            if number is not None:

                output.append(
                    float(
                        number
                    )
                )


        return output


    return []


def validate_elevation(
    value
):

    if value is None:

        raise RuntimeError(
            "GEBCO API returned "
            "no elevation value."
        )


    value = float(
        value
    )


    if not math.isfinite(
        value
    ):

        raise RuntimeError(
            "GEBCO returned a non-finite "
            "elevation value."
        )


    if (
        value
        <
        -12000

        or

        value
        >
        9000
    ):

        raise RuntimeError(
            f"GEBCO returned an "
            f"implausible elevation: "
            f"{value} m."
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

    response = requests.get(

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
                "E-AWARE/2.0 academic prototype"
        },

        timeout=30
    )


    response.raise_for_status()


    payload = (
        response.json()
    )


    if (
        isinstance(
            payload,
            dict
        )

        and

        "z" in payload
    ):

        z_value = first_numeric(
            payload[
                "z"
            ]
        )


    else:

        z_value = first_numeric(
            payload
        )


    return validate_elevation(
        z_value
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
            float(
                lat
            ),

            float(
                lon
            )
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
            "point"
    }


    try:

        response = requests.get(

            GEBCO_API_URL,

            params=params,

            headers={

                "Accept":
                    "application/json",

                "User-Agent":
                    "E-AWARE/2.0 academic prototype"
            },

            timeout=40
        )


        response.raise_for_status()


        payload = (
            response.json()
        )


        if (
            isinstance(
                payload,
                dict
            )

            and

            "z" in payload
        ):

            values = numeric_list(
                payload[
                    "z"
                ]
            )


        else:

            values = numeric_list(
                payload
            )


        if len(
            values
        ) == len(
            points
        ):

            return [

                validate_elevation(
                    value
                )

                for value in values
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

    result = (
        dataframe.copy()
    )


    unique_grid = (

        result[
            [
                "lat",
                "lon",
                "lat_key",
                "lon_key"
            ]
        ]

        .drop_duplicates(
            subset=[
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
            float(
                row.lat
            ),

            float(
                row.lon
            )
        )

        for row in unique_grid.itertuples(
            index=False
        )
    )


    elevations = (
        get_gebco_batch(
            points
        )
    )


    if len(
        elevations
    ) != len(
        unique_grid
    ):

        raise RuntimeError(
            "GEBCO returned a different "
            "number of values than requested."
        )


    bathymetry_grid = (

        unique_grid[
            [
                "lat_key",
                "lon_key"
            ]
        ]

        .copy()
    )


    bathymetry_grid[
        "gebco_elevation_m"
    ] = np.asarray(
        elevations,
        dtype=float
    )


    bathymetry_grid[
        "is_ocean_gebco"
    ] = (

        bathymetry_grid[
            "gebco_elevation_m"
        ]

        <
        0
    )


    bathymetry_grid[
        "water_depth_m"
    ] = np.where(

        bathymetry_grid[
            "is_ocean_gebco"
        ],

        -

        bathymetry_grid[
            "gebco_elevation_m"
        ],

        np.nan
    )


    result = result.merge(

        bathymetry_grid,

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
            "GEBCO land filtering "
            "removed all E-AWARE cells."
        )


    return result


# ============================================================
# SEABED-CONSTRAINED OXYCLINE
# ============================================================

def attach_seabed_constrained_oxycline(
    physics,
    oxygen_profile
):

    result = (
        physics.copy()
    )


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
            np.nan
    }


    for column, default in defaults.items():

        result[
            column
        ] = default


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


        source_grid = (

            oxygen_profile[

                oxygen_profile[
                    "date"
                ]

                ==
                target_date

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
                subset=[
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


        (
            nearest_index,
            nearest_distance
        ) = nearest_source_indices(

            target_rows,

            source_grid
        )


        nearest_sources = (

            source_grid

            .iloc[
                nearest_index
            ]

            .reset_index(
                drop=True
            )
        )


        target_indices = (
            target_rows
            .index
            .to_list()
        )


        for row_position, target_index in enumerate(
            target_indices
        ):


            target_row = (
                result.loc[
                    target_index
                ]
            )


            source_row = (
                nearest_sources
                .iloc[
                    row_position
                ]
            )


            local_water_depth = float(

                target_row[
                    "water_depth_m"
                ]
            )


            source_lat_key = float(

                source_row[
                    "lat_key"
                ]
            )


            source_lon_key = float(

                source_row[
                    "lon_key"
                ]
            )


            profile = oxygen_profile[

                (
                    oxygen_profile[
                        "date"
                    ]

                    ==
                    target_date
                )

                &

                (
                    oxygen_profile[
                        "lat_key"
                    ]

                    ==
                    source_lat_key
                )

                &

                (
                    oxygen_profile[
                        "lon_key"
                    ]

                    ==
                    source_lon_key
                )

            ].copy()


            profile = profile.sort_values(
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


            (
                raw_depth,
                raw_status
            ) = estimate_oxycline_depth(
                profile
            )


            local_profile = profile[

                profile[
                    "depth"
                ]

                <=
                local_water_depth

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

                nearest_distance[
                    row_position
                ]
            )


            result.at[
                target_index,
                "oxycline_source_lat"
            ] = float(

                source_row[
                    "lat"
                ]
            )


            result.at[
                target_index,
                "oxycline_source_lon"
            ] = float(

                source_row[
                    "lon"
                ]
            )


            result.at[
                target_index,
                "bgc_profile_max_depth_m"
            ] = float(

                profile[
                    "depth"
                ]
                .max()
            )


            result.at[
                target_index,
                "bgc_profile_levels"
            ] = int(

                profile[
                    "depth"
                ]
                .nunique()
            )


            if local_profile.empty:

                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (
                    "No BGC depth levels fall "
                    "within the local GEBCO water column"
                )

                continue


            local_max_depth = float(

                local_profile[
                    "depth"
                ]
                .max()
            )


            local_levels = int(

                local_profile[
                    "depth"
                ]
                .nunique()
            )


            local_gap = max(

                0.0,

                local_water_depth
                -
                local_max_depth
            )


            result.at[
                target_index,
                "local_profile_max_depth_m"
            ] = local_max_depth


            result.at[
                target_index,
                "local_profile_levels"
            ] = local_levels


            result.at[
                target_index,
                "local_depth_gap_m"
            ] = local_gap


            (
                local_depth,
                _
            ) = estimate_oxycline_depth(
                local_profile
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
                    "Accepted — 1 mL/L crossing "
                    "detected within the local "
                    "GEBCO water column"
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
                local_water_depth
            ):


                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (

                    "1 mL/L threshold not reached "
                    "in valid BGC levels within the local "
                    f"{local_water_depth:.1f} m water column; "
                    "raw full-profile crossing occurs "
                    f"at {float(raw_depth):.1f} m"
                )


            else:


                result.at[
                    target_index,
                    "oxycline_validation_status"
                ] = (

                    "1 mL/L threshold not reached "
                    "in valid BGC levels within the local "
                    f"{local_water_depth:.1f} m water column"
                )


    if result[
        "oxycline_validation_status"
    ].isna().all():

        raise RuntimeError(
            "Oxycline profiles could not "
            "be mapped to the E-AWARE grid."
        )


    return result


# ============================================================
# NOAA COASTLINE
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def load_noaa_coastline_lines():

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


    common_parameters = {

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
            "4326"
    }


    headers = {

        "Accept":
            "application/json",

        "User-Agent":
            "E-AWARE/2.0 academic prototype"
    }


    try:

        parameters = dict(
            common_parameters
        )


        parameters[
            "f"
        ] = "geojson"


        response = requests.get(

            NOAA_COASTLINE_QUERY_URL,

            params=parameters,

            headers=headers,

            timeout=45
        )


        response.raise_for_status()


        payload = (
            response.json()
        )


        line_strings = []


        for feature in payload.get(
            "features",
            []
        ):


            geometry = feature.get(
                "geometry"
            ) or {}


            geometry_type = geometry.get(
                "type"
            )


            coordinates = geometry.get(
                "coordinates"
            )


            if (
                geometry_type
                ==
                "LineString"

                and

                coordinates
            ):

                line_strings.append(
                    coordinates
                )


            elif (
                geometry_type
                ==
                "MultiLineString"

                and

                coordinates
            ):

                line_strings.extend(
                    coordinates
                )


        if line_strings:

            return line_strings


    except Exception:

        pass


    parameters = dict(
        common_parameters
    )


    parameters[
        "f"
    ] = "json"


    response = requests.get(

        NOAA_COASTLINE_QUERY_URL,

        params=parameters,

        headers=headers,

        timeout=45
    )


    response.raise_for_status()


    payload = (
        response.json()
    )


    if "error" in payload:

        raise RuntimeError(
            "NOAA coastline service error: "
            f"{payload['error']}"
        )


    line_strings = []


    for feature in payload.get(
        "features",
        []
    ):


        geometry = feature.get(
            "geometry"
        ) or {}


        line_strings.extend(
            geometry.get(
                "paths",
                []
            )
        )


    if not line_strings:

        raise RuntimeError(
            "NOAA coastline query returned "
            "no coastline geometry."
        )


    return line_strings


def point_to_segment_distance_km(
    point_lat,
    point_lon,
    lon1,
    lat1,
    lon2,
    lat2
):

    km_per_degree_lat = 111.32


    km_per_degree_lon = (

        111.32

        *

        math.cos(
            math.radians(
                point_lat
            )
        )
    )


    x1 = (

        float(
            lon1
        )

        -
        point_lon

    ) * km_per_degree_lon


    y1 = (

        float(
            lat1
        )

        -
        point_lat

    ) * km_per_degree_lat


    x2 = (

        float(
            lon2
        )

        -
        point_lon

    ) * km_per_degree_lon


    y2 = (

        float(
            lat2
        )

        -
        point_lat

    ) * km_per_degree_lat


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


    segment_length_squared = (

        dx
        *
        dx

        +

        dy
        *
        dy
    )


    if segment_length_squared == 0:

        return math.hypot(
            x1,
            y1
        )


    t = -(

        x1
        *
        dx

        +

        y1
        *
        dy

    ) / segment_length_squared


    t = max(
        0.0,
        min(
            1.0,
            t
        )
    )


    closest_x = (
        x1
        +
        t
        *
        dx
    )


    closest_y = (
        y1
        +
        t
        *
        dy
    )


    return math.hypot(
        closest_x,
        closest_y
    )


def distance_to_nearest_coast_km(
    lat,
    lon,
    coastline_lines
):

    minimum_distance = (
        math.inf
    )


    for line in coastline_lines:


        if (
            not line

            or

            len(
                line
            )
            <
            2
        ):

            continue


        for index in range(
            len(
                line
            )
            -
            1
        ):


            point1 = line[
                index
            ]


            point2 = line[
                index
                +
                1
            ]


            if (
                len(
                    point1
                )
                <
                2

                or

                len(
                    point2
                )
                <
                2
            ):

                continue


            distance = (
                point_to_segment_distance_km(

                    float(
                        lat
                    ),

                    float(
                        lon
                    ),

                    float(
                        point1[
                            0
                        ]
                    ),

                    float(
                        point1[
                            1
                        ]
                    ),

                    float(
                        point2[
                            0
                        ]
                    ),

                    float(
                        point2[
                            1
                        ]
                    )
                )
            )


            minimum_distance = min(

                minimum_distance,

                distance
            )


    if not math.isfinite(
        minimum_distance
    ):

        raise RuntimeError(
            "No valid coastline segments "
            "were available."
        )


    return float(
        minimum_distance
    )


def attach_distance_to_coast(
    dataframe
):

    result = (
        dataframe.copy()
    )


    coastline_lines = (
        load_noaa_coastline_lines()
    )


    unique_grid = (

        result[
            [
                "lat",
                "lon",
                "lat_key",
                "lon_key"
            ]
        ]

        .drop_duplicates(
            subset=[
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


    distances = []


    for row in unique_grid.itertuples(
        index=False
    ):


        distance = (
            distance_to_nearest_coast_km(

                float(
                    row.lat
                ),

                float(
                    row.lon
                ),

                coastline_lines
            )
        )


        distances.append(
            distance
        )


    coast_grid = (

        unique_grid[
            [
                "lat_key",
                "lon_key"
            ]
        ]

        .copy()
    )


    coast_grid[
        "distance_coast_km"
    ] = np.asarray(
        distances,
        dtype=float
    )


    coast_grid[
        "distance_coast_nm"
    ] = (

        coast_grid[
            "distance_coast_km"
        ]

        /
        KM_PER_NAUTICAL_MILE
    )


    result = result.merge(

        coast_grid,

        on=[
            "lat_key",
            "lon_key"
        ],

        how="left"
    )


    if result[
        "distance_coast_km"
    ].isna().all():

        raise RuntimeError(
            "Distance from coast could "
            "not be attached."
        )


    return result


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def austral_season(
    month
):

    if month in [
        12,
        1,
        2
    ]:

        return "Summer"


    if month in [
        3,
        4,
        5
    ]:

        return "Autumn"


    if month in [
        6,
        7,
        8
    ]:

        return "Winter"


    return "Spring"


def add_temporal_features(
    dataframe
):

    df = (
        dataframe.copy()
    )


    dates = pd.to_datetime(
        df[
            "date"
        ]
    )


    month = (
        dates
        .dt.month
        .astype(
            int
        )
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


    df[
        "month_number"
    ] = month


    df[
        "month_name"
    ] = (
        dates
        .dt.month_name()
    )


    df[
        "austral_season"
    ] = (
        month.apply(
            austral_season
        )
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


# ============================================================
# HABITAT REFERENCE / HSI
# ============================================================

def habitat_reference(
    month
):

    if month in [
        12,
        1,
        2
    ]:

        return {

            "season":
                "Summer",

            "reference_basis":
                "Published summer range",

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
                8.7
        }


    if month in [
        6,
        7,
        8
    ]:

        return {

            "season":
                "Winter",

            "reference_basis":
                "Published winter range",

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
                6.3
        }


    season_name = (

        "Autumn (Transition)"

        if month in [
            3,
            4,
            5
        ]

        else

        "Spring (Transition)"
    )


    return {

        "season":
            season_name,

        "reference_basis":
            (
                "Transition envelope — "
                "no invented seasonal threshold"
            ),

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
            8.7
    }


def range_score(
    values,
    minimum,
    maximum,
    shoulder
):

    values = np.asarray(
        values,
        dtype=float
    )


    score = np.ones_like(
        values,
        dtype=float
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

        2.0
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

        0.5
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
    )


    df[
        "hsi"
    ] = (
        df[
            "hsi"
        ]
        .clip(
            0,
            1
        )
    )


    return df


def hsi_class(
    value
):

    if value >= 0.80:

        return "Very High"


    if value >= 0.60:

        return "High"


    if value >= 0.30:

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
                "hsi"
            )

            .copy()
        )


    weights = np.clip(

        best_area[
            "hsi"
        ]

        .to_numpy(),

        0.01,

        None
    )


    latitude = np.average(

        best_area[
            "lat"
        ],

        weights=weights
    )


    longitude = np.average(

        best_area[
            "lon"
        ],

        weights=weights
    )


    return (
        float(
            latitude
        ),
        float(
            longitude
        )
    )


# ============================================================
# MOVEMENT FUNCTIONS
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0


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


def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
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

    directions = [

        "North ↑",

        "North-East ↗",

        "East →",

        "South-East ↘",

        "South ↓",

        "South-West ↙",

        "West ←",

        "North-West ↖"
    ]


    return directions[

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


def movement_components(
    lat1,
    lon1,
    lat2,
    lon2
):

    north_km = (

        lat2
        -
        lat1

    ) * 111.32


    east_km = (

        lon2
        -
        lon1

    ) * (

        111.32

        *

        math.cos(
            math.radians(
                (
                    lat1
                    +
                    lat2
                )
                /
                2
            )
        )
    )


    return (
        north_km,
        east_km
    )


# ============================================================
# V2 STEP 1 — HISTORICAL OBSERVATION INGESTION
# ============================================================

COLUMN_ALIASES = {

    "date":
        [
            "date",
            "datetime",
            "time",
            "fecha",
            "sample_date",
            "survey_date"
        ],

    "lat":
        [
            "lat",
            "latitude",
            "latitud",
            "y"
        ],

    "lon":
        [
            "lon",
            "longitude",
            "longitud",
            "lng",
            "x"
        ],

    "presence":
        [
            "presence",
            "present",
            "occurrence",
            "presence_absence",
            "pres_abs",
            "anchovy_presence",
            "anchoveta_presence",
            "fish_presence"
        ],

    "abundance":
        [
            "abundance",
            "anchovy_abundance",
            "anchoveta_abundance",
            "density",
            "acoustic_abundance",
            "sa",
            "nasc",
            "biomass_index"
        ],

    "length_cm":
        [
            "length_cm",
            "length",
            "size_cm",
            "talla_cm",
            "fish_length_cm"
        ],

    "catch":
        [
            "catch",
            "catch_t",
            "catch_tonnes",
            "volume",
            "volumen",
            "landings"
        ]
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

        normalise_column_name(
            column
        ):
            column

        for column in columns
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

    if pd.api.types.is_numeric_dtype(
        series
    ):


        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )


        unique_values = set(

            numeric

            .dropna()

            .unique()

            .tolist()
        )


        if unique_values.issubset(
            {
                0,
                1
            }
        ):

            return numeric


        return pd.Series(

            np.where(

                numeric.notna(),

                (
                    numeric
                    >
                    0
                )
                .astype(
                    float
                ),

                np.nan
            ),

            index=
                series.index,

            dtype=
                float
        )


    lowered = (

        series

        .astype(
            str
        )

        .str.strip()

        .str.lower()
    )


    true_values = {

        "1",
        "true",
        "yes",
        "y",
        "present",
        "presence",
        "si",
        "sí"
    }


    false_values = {

        "0",
        "false",
        "no",
        "n",
        "absent",
        "absence"
    }


    mapped = []


    for value in lowered:


        if value in true_values:

            mapped.append(
                1.0
            )


        elif value in false_values:

            mapped.append(
                0.0
            )


        else:

            mapped.append(
                np.nan
            )


    return pd.Series(

        mapped,

        index=
            series.index,

        dtype=
            float
    )


def standardise_historical_observations(
    raw
):

    dataframe = (
        raw.copy()
    )


    source_columns = list(
        dataframe.columns
    )


    mapping = {

        canonical:
            find_alias(
                source_columns,
                canonical
            )

        for canonical in COLUMN_ALIASES
    }


    missing_core = [

        column

        for column in [
            "date",
            "lat",
            "lon"
        ]

        if mapping[
            column
        ]
        is None
    ]


    if missing_core:

        raise ValueError(

            "Historical file is missing "
            "required field(s): "

            +

            ", ".join(
                missing_core
            )
        )


    output = pd.DataFrame()


    output[
        "date"
    ] = pd.to_datetime(

        dataframe[
            mapping[
                "date"
            ]
        ],

        errors=
            "coerce"
    )


    output[
        "lat"
    ] = pd.to_numeric(

        dataframe[
            mapping[
                "lat"
            ]
        ],

        errors=
            "coerce"
    )


    output[
        "lon"
    ] = pd.to_numeric(

        dataframe[
            mapping[
                "lon"
            ]
        ],

        errors=
            "coerce"
    )


    presence_source = (
        "Unavailable"
    )


    if mapping[
        "presence"
    ] is not None:


        output[
            "presence"
        ] = parse_presence(

            dataframe[
                mapping[
                    "presence"
                ]
            ]
        )


        presence_source = (

            "Existing field: "

            +

            str(
                mapping[
                    "presence"
                ]
            )
        )


    elif mapping[
        "abundance"
    ] is not None:


        abundance = pd.to_numeric(

            dataframe[
                mapping[
                    "abundance"
                ]
            ],

            errors=
                "coerce"
        )


        output[
            "abundance"
        ] = abundance


        output[
            "presence"
        ] = np.where(

            abundance.notna(),

            (
                abundance
                >
                0
            )
            .astype(
                float
            ),

            np.nan
        )


        presence_source = (

            "Derived from abundance > 0 using: "

            +

            str(
                mapping[
                    "abundance"
                ]
            )
        )


    else:

        output[
            "presence"
        ] = np.nan


    if (
        mapping[
            "abundance"
        ]
        is not None

        and

        "abundance"
        not in output.columns
    ):


        output[
            "abundance"
        ] = pd.to_numeric(

            dataframe[
                mapping[
                    "abundance"
                ]
            ],

            errors=
                "coerce"
        )


    if "abundance" not in output.columns:

        output[
            "abundance"
        ] = np.nan


    if mapping[
        "length_cm"
    ] is not None:


        output[
            "length_cm"
        ] = pd.to_numeric(

            dataframe[
                mapping[
                    "length_cm"
                ]
            ],

            errors=
                "coerce"
        )


    else:

        output[
            "length_cm"
        ] = np.nan


    if mapping[
        "catch"
    ] is not None:


        output[
            "catch"
        ] = pd.to_numeric(

            dataframe[
                mapping[
                    "catch"
                ]
            ],

            errors=
                "coerce"
        )


    else:

        output[
            "catch"
        ] = np.nan


    output[
        "source_row"
    ] = np.arange(

        1,

        len(
            output
        )
        +
        1
    )


    output[
        "valid_coordinates"
    ] = (

        output[
            "lat"
        ]
        .between(
            -90,
            90
        )

        &

        output[
            "lon"
        ]
        .between(
            -180,
            180
        )
    )


    output[
        "within_peru_lat_band"
    ] = (

        output[
            "lat"
        ]
        .between(
            -20.5,
            0.5
        )
    )


    output[
        "within_eaware_study_area"
    ] = (

        output[
            "lat"
        ]
        .between(
            MIN_LAT,
            MAX_LAT
        )

        &

        output[
            "lon"
        ]
        .between(
            MIN_LON,
            MAX_LON
        )
    )


    output[
        "month_number"
    ] = (
        output[
            "date"
        ]
        .dt.month
    )


    output[
        "year"
    ] = (
        output[
            "date"
        ]
        .dt.year
    )


    output[
        "austral_season"
    ] = (

        output[
            "month_number"
        ]

        .apply(

            lambda month:

                austral_season(
                    int(
                        month
                    )
                )

                if pd.notna(
                    month
                )

                else None
        )
    )


    usable = output[

        output[
            "date"
        ]
        .notna()

        &

        output[
            "valid_coordinates"
        ]

    ].copy()


    metadata = {

        "mapping":
            mapping,

        "presence_source":
            presence_source,

        "raw_rows":
            int(
                len(
                    output
                )
            ),

        "usable_rows":
            int(
                len(
                    usable
                )
            ),

        "invalid_date_rows":
            int(
                output[
                    "date"
                ]
                .isna()
                .sum()
            ),

        "invalid_coordinate_rows":
            int(
                (
                    ~
                    output[
                        "valid_coordinates"
                    ]
                )
                .sum()
            ),

        "study_area_rows":
            int(
                usable[
                    "within_eaware_study_area"
                ]
                .sum()
            ),

        "presence_known_rows":
            int(
                usable[
                    "presence"
                ]
                .notna()
                .sum()
            )
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
            "presence_known_rows"
        ]
        >
        0

        and

        usable[
            "presence"
        ]
        .dropna()
        .nunique()
        >=
        2
    )


    return (
        usable,
        metadata
    )


def read_historical_source(
    uploaded_file
):

    if uploaded_file is not None:

        return (

            pd.read_csv(
                uploaded_file
            ),

            "Uploaded CSV"
        )


    if HISTORICAL_REPO_PATH.exists():

        return (

            pd.read_csv(
                HISTORICAL_REPO_PATH
            ),

            str(
                HISTORICAL_REPO_PATH
            )
        )


    return (
        None,
        None
    )


def historical_template_csv():

    template = pd.DataFrame(

        [
            {
                "date":
                    "2008-04-15",

                "latitude":
                    -4.45,

                "longitude":
                    -81.30,

                "presence":
                    1,

                "abundance":
                    120.0,

                "length_cm":
                    13.2
            },

            {
                "date":
                    "2008-04-15",

                "latitude":
                    -4.48,

                "longitude":
                    -81.38,

                "presence":
                    0,

                "abundance":
                    0.0,

                "length_cm":
                    ""
            }
        ]
    )


    return (
        template

        .to_csv(
            index=False
        )

        .encode(
            "utf-8"
        )
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


    st.divider()


    st.markdown(
        "### V2 Historical Data"
    )


    uploaded_historical = st.file_uploader(

        "Historical anchoveta CSV",

        type=[
            "csv"
        ],

        help=(
            "If data/imarpe_anchoveta_observations.csv "
            "exists in the repository, E-AWARE uses it "
            "automatically. An uploaded CSV overrides "
            "the repository file."
        )
    )


# ============================================================
# LIVE ENVIRONMENTAL DATA
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


        credentials_valid = (
            copernicusmarine.login(

                username=
                    username,

                password=
                    password,

                check_credentials_valid=
                    True
            )
        )


        if not credentials_valid:

            raise RuntimeError(
                "Copernicus rejected "
                "the supplied credentials."
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


        ocean = (
            merge_physics_data(

                temperature_data,

                salinity_data,

                anomaly_data
            )
        )


        ocean = (
            attach_nearest_oxygen(

                ocean,

                oxygen_surface
            )
        )


        status.write(
            "🗻 5/7 GEBCO bathymetry"
        )


        ocean = (
            attach_gebco_bathymetry(
                ocean
            )
        )


        status.write(
            "📏 6/7 Distance from coast"
        )


        ocean = (
            attach_distance_to_coast(
                ocean
            )
        )


        status.write(
            "🫧 7/7 Seabed-constrained oxycline"
        )


        ocean = (
            attach_seabed_constrained_oxycline(

                ocean,

                oxygen_profile_data
            )
        )


        ocean = (
            add_temporal_features(
                ocean
            )
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
        "❌ E-AWARE could not load all "
        "live environmental data."
    )


    st.code(
        str(
            error
        )
    )


    st.stop()


# ============================================================
# CURRENT CONDITIONS
# ============================================================

available_dates = sorted(

    ocean[
        "date"
    ]

    .unique()
)


if not available_dates:

    st.error(
        "No valid ocean dates were returned."
    )

    st.stop()


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


current_data = (
    calculate_hsi(

        current_data,

        current_date.month
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


coastline_lines = (
    load_noaa_coastline_lines()
)


# ============================================================
# CURRENT CONDITIONS UI
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


context1, context2, context3, context4 = (
    st.columns(
        4
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
    )
)


context3.metric(
    "GEBCO Depth Estimate",
    f"{best['water_depth_m']:.0f} m"
)


context4.metric(

    "Distance from Coast",

    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)


# ============================================================
# TEMPORAL / SEASONAL PREDICTOR
# ============================================================

st.subheader(
    "🗓️ Temporal / Seasonal Predictor"
)


temporal1, temporal2, temporal3, temporal4, temporal5 = (
    st.columns(
        5
    )
)


temporal1.metric(

    "Month",

    (
        f"{best['month_name']} "
        f"({int(best['month_number'])})"
    )
)


temporal2.metric(
    "Austral Season",
    str(
        best[
            "austral_season"
        ]
    )
)


temporal3.metric(
    "Month sin",
    f"{float(best['month_sin']):.3f}"
)


temporal4.metric(
    "Month cos",
    f"{float(best['month_cos']):.3f}"
)


temporal5.metric(
    "HSI Weight",
    "Not included"
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
        "Distance from Coast",
        "SST Anomaly"
    ],

    horizontal=True
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

    normalised = scaled(
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


elif map_mode == "Salinity":

    normalised = scaled(
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
        normalised
    )


    map_data[
        "b"
    ] = 220


elif map_mode == "Dissolved Oxygen":

    normalised = scaled(
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


elif map_mode == "Bathymetry":

    normalised = scaled(
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
        normalised
    )


    map_data[
        "g"
    ] = (

        180

        -
        110
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


elif map_mode == "Distance from Coast":

    normalised = scaled(
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
        normalised
    )


    map_data[
        "g"
    ] = (

        190

        -
        100
        *
        normalised
    )


    map_data[
        "b"
    ] = (

        220

        -
        80
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

        0.1
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


display_columns = [

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
    )
]


for (
    source_column,
    display_column,
    decimals
) in display_columns:


    map_data[
        display_column
    ] = (

        map_data[
            source_column
        ]

        .round(
            decimals
        )
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
                    9.5,

                pitch=
                    0
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
        }
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


validation_status = str(
    best.get(
        "oxycline_validation_status",
        "Unavailable"
    )
)


oxycline1, oxycline2, oxycline3, oxycline4 = (
    st.columns(
        4
    )
)


oxycline1.metric(

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


oxycline2.metric(
    "Local GEBCO Water Depth",
    f"{best['water_depth_m']:.1f} m"
)


oxycline3.metric(

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


oxycline4.metric(
    "HSI Weight",
    "Not included"
)


detail1, detail2, detail3 = (
    st.columns(
        3
    )
)


detail1.metric(

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


detail2.metric(

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


detail3.metric(

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
    validation_status
)


# ============================================================
# HIGHEST HABITAT / REFERENCE
# ============================================================

st.subheader(
    "📍 Highest Relative Habitat Suitability"
)


zone1, zone2, zone3, zone4, zone5 = (
    st.columns(
        5
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
    "GEBCO Depth",
    f"{best['water_depth_m']:.0f} m"
)


zone5.metric(

    "Coast Distance",

    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)


st.subheader(
    "🐟 Current MVP Habitat Reference"
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
    "The temporary HSI still uses only temperature, "
    "salinity and surface dissolved oxygen. Bathymetry, "
    "distance from coast, oxycline and month/season remain "
    "independent predictors until historical observations "
    "are prepared for model training."
)


# ============================================================
# FORECAST / HABITAT SHIFT
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
                "distance_coast_km",
                "distance_coast_nm"
            ]
        )

        .copy()
    )


    if forecast_frame.empty:

        continue


    forecast_frame = (
        calculate_hsi(

            forecast_frame,

            forecast_date.month
        )
    )


    centre = (
        habitat_centroid(
            forecast_frame
        )
    )


    centre_coast_km = None

    centre_coast_nm = None


    if centre is not None:


        centre_coast_km = (
            distance_to_nearest_coast_km(

                centre[
                    0
                ],

                centre[
                    1
                ],

                coastline_lines
            )
        )


        centre_coast_nm = (

            centre_coast_km

            /
            KM_PER_NAUTICAL_MILE
        )


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

                "distance_coast_km":
                    centre_coast_km,

                "distance_coast_nm":
                    centre_coast_nm
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

            "Month":
                str(
                    best_forecast[
                        "month_name"
                    ]
                ),

            "Austral Season":
                str(
                    best_forecast[
                        "austral_season"
                    ]
                ),

            "Highest HSI":
                round(
                    float(
                        best_forecast[
                            "hsi"
                        ]
                    ),
                    2
                ),

            "Temperature °C":
                round(
                    float(
                        best_forecast[
                            "temperature"
                        ]
                    ),
                    2
                ),

            "Salinity PSU":
                round(
                    float(
                        best_forecast[
                            "salinity"
                        ]
                    ),
                    2
                ),

            "Oxygen mL/L":
                round(
                    float(
                        best_forecast[
                            "oxygen_ml_l"
                        ]
                    ),
                    2
                ),

            "Water Depth m":
                round(
                    float(
                        best_forecast[
                            "water_depth_m"
                        ]
                    ),
                    0
                ),

            "Distance Coast km":
                round(
                    float(
                        best_forecast[
                            "distance_coast_km"
                        ]
                    ),
                    2
                ),

            "Distance Coast nm":
                round(
                    float(
                        best_forecast[
                            "distance_coast_nm"
                        ]
                    ),
                    2
                ),

            "Local Oxycline m":
                (

                    round(
                        float(
                            best_forecast[
                                "oxycline_depth_m"
                            ]
                        ),
                        1
                    )

                    if pd.notna(
                        best_forecast[
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
                            best_forecast[
                                "sst_anomaly"
                            ]
                        ),
                        2
                    )

                    if pd.notna(
                        best_forecast[
                            "sst_anomaly"
                        ]
                    )

                    else

                    None
                ),

            "Habitat Centre Lat":
                (

                    round(
                        centre[
                            0
                        ],
                        4
                    )

                    if centre

                    else

                    None
                ),

            "Habitat Centre Lon":
                (

                    round(
                        centre[
                            1
                        ],
                        4
                    )

                    if centre

                    else

                    None
                ),

            "Centre Coast km":
                (

                    round(
                        centre_coast_km,
                        2
                    )

                    if centre_coast_km
                    is not None

                    else

                    None
                ),

            "Centre Coast nm":
                (

                    round(
                        centre_coast_nm,
                        2
                    )

                    if centre_coast_nm
                    is not None

                    else

                    None
                )
        }
    )


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

start_coast_text = (
    "—"
)

end_coast_text = (
    "—"
)

coastal_interpretation = (
    "Unavailable"
)


start = None

end = None


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
            ]
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


    if north_km >= 0:

        north_south_text = (
            f"{abs(north_km):.2f} km North"
        )


    else:

        north_south_text = (
            f"{abs(north_km):.2f} km South"
        )


    if east_km >= 0:

        east_west_text = (
            f"{abs(east_km):.2f} km East"
        )


    else:

        east_west_text = (
            f"{abs(east_km):.2f} km West"
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


    start_coast_text = (

        f"{start['distance_coast_km']:.2f} km / "
        f"{start['distance_coast_nm']:.2f} nm"
    )


    end_coast_text = (

        f"{end['distance_coast_km']:.2f} km / "
        f"{end['distance_coast_nm']:.2f} nm"
    )


    coast_delta = (

        end[
            "distance_coast_km"
        ]

        -

        start[
            "distance_coast_km"
        ]
    )


    if abs(
        coast_delta
    ) < 0.5:

        coastal_interpretation = (
            "No clear coastal/offshore change"
        )


    elif coast_delta < 0:

        coastal_interpretation = (

            f"Closer to coast by "
            f"{abs(coast_delta):.2f} km"
        )


    else:

        coastal_interpretation = (

            f"Farther offshore by "
            f"{abs(coast_delta):.2f} km"
        )


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


coast1, coast2, coast3 = (
    st.columns(
        3
    )
)


coast1.metric(
    "Start Distance from Coast",
    start_coast_text
)


coast2.metric(
    "+72 h Distance from Coast",
    end_coast_text
)


coast3.metric(
    "Interpretation",
    coastal_interpretation
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


    shift_path = [

        {
            "path":
                [
                    [
                        start[
                            "lon"
                        ],
                        start[
                            "lat"
                        ]
                    ],

                    [
                        end[
                            "lon"
                        ],
                        end[
                            "lat"
                        ]
                    ]
                ],

            "name":
                (
                    f"{direction_text} • "
                    f"{distance_text}"
                )
        }
    ]


    marker_data = pd.DataFrame(

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

                "r":
                    30,

                "g":
                    190,

                "b":
                    90
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

                "r":
                    255,

                "g":
                    145,

                "b":
                    40
            }
        ]
    )


    path_layer = pdk.Layer(

        "PathLayer",

        data=
            shift_path,

        get_path=
            "path",

        get_color=
            [
                0,
                140,
                255,
                230
            ],

        get_width=
            5,

        width_min_pixels=
            5,

        pickable=
            True
    )


    marker_layer = pdk.Layer(

        "ScatterplotLayer",

        data=
            marker_data,

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
                240
            ],

        get_radius=
            500,

        radius_min_pixels=
            8,

        pickable=
            True
    )


    text_layer = pdk.Layer(

        "TextLayer",

        data=
            marker_data,

        get_position=
            [
                "lon",
                "lat"
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
                255
            ],

        get_pixel_offset=
            [
                0,
                -18
            ]
    )


    st.pydeck_chart(

        pdk.Deck(

            map_style=
                BASEMAP_URL,

            initial_view_state=
                pdk.ViewState(

                    latitude=
                        (
                            start[
                                "lat"
                            ]
                            +
                            end[
                                "lat"
                            ]
                        )
                        /
                        2,

                    longitude=
                        (
                            start[
                                "lon"
                            ]
                            +
                            end[
                                "lon"
                            ]
                        )
                        /
                        2,

                    zoom=
                        10.3
                ),

            layers=[
                path_layer,
                marker_layer,
                text_layer
            ]
        ),

        use_container_width=True
    )


# ============================================================
# 72-HOUR FORECAST TABLE
# ============================================================

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
# V2 STEP 1 — HISTORICAL OBSERVATIONS UI
# ============================================================

st.divider()


st.header(
    "🧬 V2 Step 1 — Historical Anchoveta Observations"
)


st.write(
    "This stage prepares historical anchoveta observations "
    "for later model training. It validates the file structure, "
    "dates, coordinates and presence/absence information. "
    "It does not train XGBoost, a Generalized Additive Model, "
    "or any other habitat model yet."
)


st.download_button(

    "⬇️ Download CSV structure example",

    data=
        historical_template_csv(),

    file_name=
        "eaware_historical_anchoveta_template.csv",

    mime=
        "text/csv"
)


(
    historical_raw,
    historical_source
) = read_historical_source(
    uploaded_historical
)


if historical_raw is None:


    st.warning(
        "No historical observation CSV is loaded yet. "
        "Add the real data as "
        "`data/imarpe_anchoveta_observations.csv` "
        "in the GitHub repository, or upload a CSV "
        "using the sidebar. E-AWARE will not create "
        "fake historical fish observations."
    )


    st.markdown(
        "**Minimum required fields:** "
        "date, latitude, longitude"
    )


    st.markdown(
        "**For supervised presence/absence training:** "
        "the file must also contain either a "
        "presence/absence field or a defensible abundance "
        "field from which presence can be derived."
    )


else:


    st.caption(
        f"Historical source currently loaded: "
        f"**{historical_source}**"
    )


    try:


        (
            historical,
            historical_metadata
        ) = standardise_historical_observations(
            historical_raw
        )


        history1, history2, history3, history4, history5 = (
            st.columns(
                5
            )
        )


        history1.metric(
            "Raw Rows",
            f"{historical_metadata['raw_rows']:,}"
        )


        history2.metric(
            "Usable Rows",
            f"{historical_metadata['usable_rows']:,}"
        )


        history3.metric(
            "Presence Known",
            f"{historical_metadata['presence_known_rows']:,}"
        )


        history4.metric(
            "Inside E-AWARE Area",
            f"{historical_metadata['study_area_rows']:,}"
        )


        history5.metric(

            "Training Ready",

            (
                "Yes"

                if historical_metadata[
                    "training_ready"
                ]

                else

                "No"
            )
        )


        st.caption(
            "Presence handling: "
            f"{historical_metadata['presence_source']}"
        )


        mapping_rows = []


        for (
            canonical,
            source_column
        ) in historical_metadata[
            "mapping"
        ].items():


            mapping_rows.append(

                {
                    "E-AWARE field":
                        canonical,

                    "Detected source column":
                        (
                            source_column

                            if source_column
                            is not None

                            else

                            "Not detected"
                        )
                }
            )


        st.markdown(
            "#### Detected Column Mapping"
        )


        st.dataframe(

            pd.DataFrame(
                mapping_rows
            ),

            use_container_width=True,

            hide_index=True
        )


        if (
            historical_metadata[
                "invalid_date_rows"
            ]
            >
            0

            or

            historical_metadata[
                "invalid_coordinate_rows"
            ]
            >
            0
        ):


            st.warning(

                "Validation excluded "
                f"{historical_metadata['invalid_date_rows']} "
                "rows with invalid dates and "
                f"{historical_metadata['invalid_coordinate_rows']} "
                "rows with invalid coordinates."
            )


        if historical_metadata[
            "study_area_rows"
        ] == 0:


            st.warning(
                "The loaded historical file contains no "
                "validated records inside the current "
                "E-AWARE Lobitos study bounding box. "
                "This is not treated as fish absence. "
                "It only means no observations in the "
                "loaded file intersect this current box."
            )


        if historical_metadata[
            "presence_known_rows"
        ] == 0:


            st.warning(
                "No presence/absence target could be identified. "
                "E-AWARE will not train a supervised presence model "
                "from this file until a defensible target variable "
                "is available."
            )


        elif (
            historical[
                "presence"
            ]
            .dropna()
            .nunique()
            <
            2
        ):


            st.warning(
                "The loaded data contains only one presence class. "
                "A presence/absence classifier requires both positive "
                "and negative observations, or a separately justified "
                "presence-background design."
            )


        else:


            presence_counts = (

                historical[
                    "presence"
                ]

                .value_counts(
                    dropna=True
                )

                .sort_index()
            )


            absence_count = int(
                presence_counts.get(
                    0.0,
                    0
                )
            )


            presence_count = int(
                presence_counts.get(
                    1.0,
                    0
                )
            )


            known_total = max(

                absence_count
                +
                presence_count,

                1
            )


            presence1, presence2, presence3 = (
                st.columns(
                    3
                )
            )


            presence1.metric(
                "Observed Absence",
                f"{absence_count:,}"
            )


            presence2.metric(
                "Observed Presence",
                f"{presence_count:,}"
            )


            presence3.metric(

                "Presence Rate",

                (
                    f"{100 * presence_count / known_total:.1f}%"
                )
            )


        if not historical.empty:


            historical_map = (

                historical

                .dropna(
                    subset=[
                        "lat",
                        "lon"
                    ]
                )

                .copy()
            )


            historical_map[
                "presence_label"
            ] = (

                historical_map[
                    "presence"
                ]

                .map(
                    {
                        1.0:
                            "Presence",

                        0.0:
                            "Absence"
                    }
                )

                .fillna(
                    "Unknown"
                )
            )


            historical_map[
                "r"
            ] = np.where(

                historical_map[
                    "presence"
                ]
                ==
                1,

                230,

                np.where(

                    historical_map[
                        "presence"
                    ]
                    ==
                    0,

                    60,

                    150
                )
            )


            historical_map[
                "g"
            ] = np.where(

                historical_map[
                    "presence"
                ]
                ==
                1,

                90,

                np.where(

                    historical_map[
                        "presence"
                    ]
                    ==
                    0,

                    120,

                    150
                )
            )


            historical_map[
                "b"
            ] = np.where(

                historical_map[
                    "presence"
                ]
                ==
                1,

                60,

                np.where(

                    historical_map[
                        "presence"
                    ]
                    ==
                    0,

                    220,

                    150
                )
            )


            historical_map[
                "date_display"
            ] = (

                historical_map[
                    "date"
                ]

                .dt.strftime(
                    "%Y-%m-%d"
                )
            )


            st.markdown(
                "#### Historical Observation Map"
            )


            historical_layer = pdk.Layer(

                "ScatterplotLayer",

                data=
                    historical_map,

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
                        175
                    ],

                get_radius=
                    4500,

                radius_min_pixels=
                    3,

                radius_max_pixels=
                    10,

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
                                -9.5,

                            longitude=
                                -78.5,

                            zoom=
                                4.6
                        ),

                    layers=[
                        historical_layer
                    ],

                    tooltip={

                        "html":

                            "<b>Historical anchoveta "
                            "observation</b><br/>"
                            "Date: {date_display}<br/>"
                            "Status: {presence_label}<br/>"
                            "Latitude: {lat}<br/>"
                            "Longitude: {lon}"
                    }
                ),

                use_container_width=True
            )


            study_historical = historical[

                historical[
                    "within_eaware_study_area"
                ]

            ].copy()


            st.markdown(
                "#### Historical Records Inside "
                "Current E-AWARE Study Area"
            )


            if study_historical.empty:


                st.info(
                    "No validated historical records "
                    "from the loaded file fall inside "
                    "the current study box."
                )


            else:


                display_columns = [

                    "date",
                    "lat",
                    "lon",
                    "presence",
                    "abundance",
                    "length_cm",
                    "austral_season"
                ]


                st.dataframe(

                    study_historical[
                        display_columns
                    ]

                    .sort_values(
                        "date"
                    ),

                    use_container_width=True,

                    hide_index=True
                )


        if historical_metadata[
            "training_ready"
        ]:


            st.success(
                "Historical observations passed the "
                "basic V2 Step 1 structure checks. "
                "The next stage can match each historical "
                "observation to environmental predictors "
                "before any model is trained."
            )


        else:


            st.info(
                "V2 Step 1 ingestion is working, but "
                "this dataset is not yet ready for "
                "supervised presence/absence model training."
            )


    except Exception as historical_error:


        st.error(
            "Historical observation validation failed."
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
# DATA SOURCES / TRANSPARENCY
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
                "Temperature, salinity and SST anomaly"
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry",

            "Status":
                "3D analysis / forecast",

            "Use":
                "Surface oxygen and vertical oxygen profiles"
        },

        {
            "Source":
                "GEBCO_2026 via Ocean Data Bank API",

            "Status":
                "Static terrain model",

            "Use":
                "Bathymetry and local water-column limit"
        },

        {
            "Source":
                "NOAA NCEI coastline",

            "Status":
                "Static coastline geometry",

            "Use":
                "Nearest-coast distance"
        },

        {
            "Source":
                "Historical anchoveta CSV",

            "Status":
                "V2 input when supplied",

            "Use":
                "Observed presence/absence or abundance "
                "for later model training"
        },

        {
            "Source":
                "IMARPE",

            "Status":
                "Scientific observation source",

            "Use":
                "Target source for historical "
                "anchoveta observations"
        }
    ]
)


st.dataframe(

    source_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "🧪 V2 Step 1 Model Status"
)


status1, status2 = (
    st.columns(
        2
    )
)


status1.success(
    "E-AWARE can now ingest and validate a historical "
    "anchoveta CSV without changing the live V1.5 HSI."
)


status2.warning(
    "No machine-learning model is trained in V2 Step 1. "
    "Historical fish observations must first be validated "
    "and matched to environmental predictors at the same "
    "date and location."
)


with st.expander(
    "⚙️ V2 Step 1 Limitations"
):

    st.markdown(
        """
- E-AWARE does not fabricate historical fish observations.
- The public IMARPE pages used by the project describe anchoveta monitoring, but this app does not assume that those pages expose a stable machine-readable historical Application Programming Interface (API).
- A historical Comma-Separated Values (CSV) file must therefore be supplied from an authorised and traceable source before training.
- Presence is used directly when a valid presence field exists.
- If only abundance is supplied, V2 Step 1 can derive a provisional binary target as abundance greater than zero. This must still be documented and justified before final model training.
- Records outside the current Lobitos study box are retained for validation and future model development.
- Lack of historical records inside the current study box is never treated as biological absence.
- The temporary live Habitat Suitability Index (HSI) still uses temperature, salinity and surface dissolved oxygen only.
- Bathymetry, distance from coast, oxycline and temporal variables remain independent future predictors.
- V2 Step 2 should match each historical observation to environmental predictors at the same spatial and temporal coordinates before comparing candidate habitat models.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "E-AWARE V2 Step 1 | "
    "Historical observation ingestion and validation | "
    "Decision-support research prototype"
)


st.caption(
    "Habitat suitability is a model estimate "
    "and must not be used as navigation, safety "
    "or legal fishing advice."
)
