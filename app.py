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

st.set_page_config(page_title="E-AWARE", page_icon="🐟", layout="wide")
st.title("🐟 E-AWARE")
st.caption("El Niño Adaptive Fisheries Intelligence — MVP V1.4 Step 1")

st.info(
    "E-AWARE estimates suitable anchoveta habitat from environmental conditions. "
    "It does not show the exact location or swimming direction of individual fish schools."
)


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
# DATA SOURCES
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


COASTLINE_QUERY_MARGIN_DEG = 0.75

BASEMAP_URL = (
    "https://basemaps.cartocdn.com/"
    "gl/positron-gl-style/style.json"
)

KM_PER_NAUTICAL_MILE = 1.852

OXYGEN_MMOL_M3_TO_ML_L = 0.0223916

OXYCLINE_THRESHOLD_ML_L = 1.0


# ============================================================
# COPERNICUS CREDENTIALS
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


# ============================================================
# PREPARE COPERNICUS 2D DATA
# ============================================================

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


    required_columns = [
        "time",
        "lat",
        "lon",
        variable_name
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
            variable_name
        ]
    ]


# ============================================================
# DOWNLOAD COPERNICUS 2D / SURFACE SUBSET
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
                True
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
        output_name
    )


# ============================================================
# STANDARD COPERNICUS LOADERS
# ============================================================

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
# V1.4 — FULL 3D OXYGEN PROFILE
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
                    "Oxygen-profile download "
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


            if "o2" not in ds.data_vars:

                raise RuntimeError(
                    "Variable 'o2' was not found "
                    "in the oxygen-profile file."
                )


            data_array = (
                ds[
                    "o2"
                ]
                .load()
            )


            dataframe = (
                data_array
                .to_dataframe(
                    name=
                        "oxygen_mmol_m3"
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


    dataframe = (
        dataframe.rename(
            columns=rename_map
        )
    )


    required_columns = [
        "time",
        "depth",
        "lat",
        "lon",
        "oxygen_mmol_m3"
    ]


    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


    if missing:

        raise RuntimeError(
            f"Missing oxygen-profile columns: "
            f"{missing}"
        )


    dataframe = (

        dataframe

        .dropna(
            subset=[
                "oxygen_mmol_m3",
                "depth"
            ]
        )

        .copy()
    )


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


# ============================================================
# SURFACE OXYGEN FROM FULL PROFILE
# ============================================================

def surface_oxygen_from_profile(
    profile_dataframe
):

    surface = (

        profile_dataframe

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

        .copy()
    )


    return surface[
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


# ============================================================
# CALCULATE OXYCLINE DEPTH
# ============================================================

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


    # If the shallowest model level is already
    # at or below the 1 mL/L threshold.

    if oxygen[
        0
    ] <= threshold_ml_l:

        return (

            float(
                depths[
                    0
                ]
            ),

            "Threshold reached at shallowest model level"
        )


    # Search downward for the first crossing
    # from > 1 mL/L to <= 1 mL/L.

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


            # Linear interpolation between
            # the two model depth levels.

            if (
                lower_oxygen
                ==
                upper_oxygen
            ):

                crossing_depth = (
                    lower_depth
                )


            else:

                fraction = (

                    (
                        threshold_ml_l
                        -
                        upper_oxygen
                    )

                    /

                    (
                        lower_oxygen
                        -
                        upper_oxygen
                    )
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

                "Interpolated 1 mL/L crossing"
            )


    return (

        np.nan,

        (
            "1 mL/L threshold not reached "
            "in available water column"
        )
    )


# ============================================================
# BUILD OXYCLINE TABLE
# ============================================================

def calculate_oxycline_table(
    profile_dataframe
):

    records = []


    group_columns = [
        "date",
        "lat_key",
        "lon_key"
    ]


    for (
        target_date,
        lat_key,
        lon_key
    ), group in profile_dataframe.groupby(
        group_columns
    ):


        group = (
            group.sort_values(
                "depth"
            )
        )


        (
            oxycline_depth,
            status
        ) = estimate_oxycline_depth(
            group
        )


        records.append(
            {
                "date":
                    target_date,

                "lat":
                    float(
                        group[
                            "lat"
                        ]
                        .iloc[
                            0
                        ]
                    ),

                "lon":
                    float(
                        group[
                            "lon"
                        ]
                        .iloc[
                            0
                        ]
                    ),

                "lat_key":
                    float(
                        lat_key
                    ),

                "lon_key":
                    float(
                        lon_key
                    ),

                "oxycline_depth_m":
                    oxycline_depth,

                "oxycline_status":
                    status,

                "profile_min_depth_m":
                    float(
                        group[
                            "depth"
                        ]
                        .min()
                    ),

                "profile_max_depth_m":
                    float(
                        group[
                            "depth"
                        ]
                        .max()
                    ),

                "profile_levels":
                    int(
                        group[
                            "depth"
                        ]
                        .nunique()
                    )
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# MERGE PHYSICS DATA
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
            "Copernicus datasets loaded, "
            "but no matching physics grid "
            "cells were found."
        )


    return merged


# ============================================================
# MAP SURFACE OXYGEN TO PHYSICS GRID
# ============================================================

def attach_nearest_oxygen(
    physics_dataframe,
    oxygen_dataframe
):

    result = (
        physics_dataframe.copy()
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


        source_rows = oxygen_dataframe[

            oxygen_dataframe[
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


        result.loc[
            target_mask,
            "oxygen_mmol_m3"
        ] = (

            source_rows

            .iloc[
                nearest_index
            ][
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

            source_rows

            .iloc[
                nearest_index
            ][
                "oxygen_ml_l"
            ]

            .to_numpy(
                dtype=float
            )
        )


        result.loc[
            target_mask,
            "oxygen_source_distance_km"
        ] = distance_km[

            np.arange(
                len(
                    target_rows
                )
            ),

            nearest_index
        ]


    if result[
        "oxygen_ml_l"
    ].isna().all():

        raise RuntimeError(
            "Oxygen data could not be "
            "matched to the physics grid."
        )


    return result


# ============================================================
# MAP OXYCLINE TO PHYSICS GRID
# ============================================================

def attach_nearest_oxycline(
    physics_dataframe,
    oxycline_dataframe
):

    result = (
        physics_dataframe.copy()
    )


    result[
        "oxycline_depth_m"
    ] = np.nan


    result[
        "oxycline_source_distance_km"
    ] = np.nan


    result[
        "oxycline_status"
    ] = None


    result[
        "oxycline_profile_max_depth_m"
    ] = np.nan


    result[
        "oxycline_profile_levels"
    ] = np.nan


    mapped_any = False


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


        source_rows = oxycline_dataframe[

            oxycline_dataframe[
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


        nearest_rows = (

            source_rows
            .iloc[
                nearest_index
            ]
        )


        result.loc[
            target_mask,
            "oxycline_depth_m"
        ] = (

            nearest_rows[
                "oxycline_depth_m"
            ]

            .to_numpy(
                dtype=float
            )
        )


        result.loc[
            target_mask,
            "oxycline_source_distance_km"
        ] = distance_km[

            np.arange(
                len(
                    target_rows
                )
            ),

            nearest_index
        ]


        result.loc[
            target_mask,
            "oxycline_status"
        ] = (

            nearest_rows[
                "oxycline_status"
            ]

            .astype(
                str
            )

            .to_numpy()
        )


        result.loc[
            target_mask,
            "oxycline_profile_max_depth_m"
        ] = (

            nearest_rows[
                "profile_max_depth_m"
            ]

            .to_numpy(
                dtype=float
            )
        )


        result.loc[
            target_mask,
            "oxycline_profile_levels"
        ] = (

            nearest_rows[
                "profile_levels"
            ]

            .to_numpy(
                dtype=float
            )
        )


        mapped_any = True


    if not mapped_any:

        raise RuntimeError(
            "Oxycline results could not be "
            "matched to the E-AWARE physics grid."
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
            "GEBCO API returned a "
            "non-finite elevation value."
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
            "GEBCO API returned "
            f"an implausible elevation: "
            f"{value} m."
        )


    return value


# ============================================================
# SINGLE GEBCO POINT
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_gebco_elevation(
    lat,
    lon
):

    params = {

        "lon":
            f"{float(lon):.6f}",

        "lat":
            f"{float(lat):.6f}",

        "mode":
            "point"
    }


    headers = {

        "Accept":
            "application/json",

        "User-Agent":
            "E-AWARE/1.4 academic prototype"
    }


    response = requests.get(

        GEBCO_API_URL,

        params=params,

        headers=headers,

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


# ============================================================
# BATCH GEBCO REQUEST
# ============================================================

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


    headers = {

        "Accept":
            "application/json",

        "User-Agent":
            "E-AWARE/1.4 academic prototype"
    }


    try:

        response = requests.get(

            GEBCO_API_URL,

            params=params,

            headers=headers,

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


# ============================================================
# ATTACH GEBCO BATHYMETRY
# ============================================================

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


    gebco_grid = (

        unique_grid[
            [
                "lat_key",
                "lon_key"
            ]
        ]

        .copy()
    )


    gebco_grid[
        "gebco_elevation_m"
    ] = np.asarray(
        elevations,
        dtype=float
    )


    gebco_grid[
        "is_ocean_gebco"
    ] = (

        gebco_grid[
            "gebco_elevation_m"
        ]

        <
        0
    )


    gebco_grid[
        "water_depth_m"
    ] = np.where(

        gebco_grid[
            "is_ocean_gebco"
        ],

        -

        gebco_grid[
            "gebco_elevation_m"
        ],

        np.nan
    )


    result = result.merge(

        gebco_grid,

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
# LOAD NOAA COASTLINE
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


    common_params = {

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
            "E-AWARE/1.4 academic prototype"
    }


    # --------------------------------------------------------
    # TRY GEOJSON FIRST
    # --------------------------------------------------------

    try:

        params = dict(
            common_params
        )


        params[
            "f"
        ] = "geojson"


        response = requests.get(

            NOAA_COASTLINE_QUERY_URL,

            params=params,

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


    # --------------------------------------------------------
    # FALLBACK TO ARCGIS JSON
    # --------------------------------------------------------

    params = dict(
        common_params
    )


    params[
        "f"
    ] = "json"


    response = requests.get(

        NOAA_COASTLINE_QUERY_URL,

        params=params,

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


        paths = geometry.get(
            "paths",
            []
        )


        line_strings.extend(
            paths
        )


    if not line_strings:

        raise RuntimeError(
            "NOAA coastline query "
            "returned no coastline geometry."
        )


    return line_strings


# ============================================================
# POINT TO COASTLINE SEGMENT DISTANCE
# ============================================================

def point_to_segment_distance_km(
    point_lat,
    point_lon,
    lon1,
    lat1,
    lon2,
    lat2
):

    km_per_degree_lat = (
        111.32
    )


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


# ============================================================
# DISTANCE TO NEAREST COAST
# ============================================================

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


            if (
                distance
                <
                minimum_distance
            ):

                minimum_distance = (
                    distance
                )


    if not math.isfinite(
        minimum_distance
    ):

        raise RuntimeError(
            "No valid coastline segments "
            "were available for distance calculation."
        )


    return float(
        minimum_distance
    )


# ============================================================
# ATTACH DISTANCE FROM COAST
# ============================================================

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


    distances_km = []


    for row in unique_grid.itertuples(
        index=False
    ):


        distance_km = (
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


        distances_km.append(
            distance_km
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
        distances_km,
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
            "Distance from coast could not "
            "be attached to the E-AWARE grid."
        )


    return result


# ============================================================
# HABITAT REFERENCE
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
            8.7
    }


# ============================================================
# RANGE SCORE
# ============================================================

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


# ============================================================
# HSI — V1.4 STEP 1
# ============================================================

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


    # IMPORTANT:
    #
    # Bathymetry, coast distance and oxycline
    # depth remain OUTSIDE the HSI.
    #
    # Current temporary HSI:
    #
    # Temperature = 1/3
    # Salinity = 1/3
    # Surface oxygen = 1/3

    df[
        "hsi"
    ] = (

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

    ) / 3.0


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


# ============================================================
# HSI CLASS
# ============================================================

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


# ============================================================
# HABITAT CENTROID
# ============================================================

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

        value=date.today()
    )


    st.caption(
        "E-AWARE requests the selected day "
        "plus the following 72 hours."
    )


    if st.button(
        "🔄 Refresh Data",
        use_container_width=True
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

        "Loading E-AWARE V1.4 Step 1 environmental data...",

        expanded=True

    ) as status:


        status.write(
            "🔐 Checking Copernicus credentials..."
        )


        username, password = (
            get_credentials()
        )


        credentials_valid = (
            copernicusmarine.login(

                username=username,

                password=password,

                check_credentials_valid=True
            )
        )


        if not credentials_valid:

            raise RuntimeError(
                "Copernicus rejected "
                "the supplied credentials."
            )


        status.write(
            "✅ Copernicus credentials valid"
        )


        status.write(
            "🌡️ 1/7 Downloading temperature..."
        )


        temperature_data = (
            load_temperature(

                selected_date,

                end_date
            )
        )


        status.write(
            f"✅ Temperature loaded "
            f"({len(temperature_data)} records)"
        )


        status.write(
            "🧂 2/7 Downloading salinity..."
        )


        salinity_data = (
            load_salinity(

                selected_date,

                end_date
            )
        )


        status.write(
            f"✅ Salinity loaded "
            f"({len(salinity_data)} records)"
        )


        status.write(
            "🌊 3/7 Downloading SST anomaly..."
        )


        anomaly_data = (
            load_sst_anomaly(

                selected_date,

                end_date
            )
        )


        status.write(
            f"✅ SST anomaly loaded "
            f"({len(anomaly_data)} records)"
        )


        status.write(
            "🫧 4/7 Downloading full "
            "3D dissolved-oxygen profile..."
        )


        oxygen_profile_data = (
            load_oxygen_profile(

                selected_date,

                end_date
            )
        )


        oxygen_data = (
            surface_oxygen_from_profile(
                oxygen_profile_data
            )
        )


        oxycline_data = (
            calculate_oxycline_table(
                oxygen_profile_data
            )
        )


        status.write(

            f"✅ Oxygen profile loaded "
            f"({len(oxygen_profile_data)} depth records; "
            f"{len(oxycline_data)} grid-cell/day profiles)"
        )


        status.write(
            "🔗 Combining Copernicus datasets..."
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

                oxygen_data
            )
        )


        status.write(
            "🗻 5/7 Retrieving GEBCO_2026 bathymetry..."
        )


        ocean = (
            attach_gebco_bathymetry(
                ocean
            )
        )


        status.write(
            "✅ GEBCO_2026 bathymetry attached "
            "and land cells filtered"
        )


        status.write(
            "📏 6/7 Calculating distance "
            "to nearest coastline..."
        )


        ocean = (
            attach_distance_to_coast(
                ocean
            )
        )


        status.write(
            "✅ Coast distance stored in "
            "km and nautical miles"
        )


        status.write(
            "🫧 7/7 Mapping 1 mL/L "
            "oxycline depth to E-AWARE grid..."
        )


        ocean = (
            attach_nearest_oxycline(

                ocean,

                oxycline_data
            )
        )


        valid_oxyclines = int(

            ocean[
                "oxycline_depth_m"
            ]

            .notna()

            .sum()
        )


        status.write(

            f"✅ Oxycline mapping complete "
            f"({valid_oxyclines} mapped cells "
            f"with a detected crossing)"
        )


        status.update(

            label=
                "✅ E-AWARE V1.4 Step 1 "
                "environmental data loaded",

            state=
                "complete",

            expanded=
                False
        )


except Exception as error:

    st.error(
        "❌ E-AWARE V1.4 Step 1 could not "
        "load all environmental data."
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
# CURRENT DATA
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
        "No complete environmental grid "
        "cells are available for the selected date."
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


st.caption(
    "V1.4 Step 1 keeps the established V1.3 dashboard "
    "unchanged while adding a calculated oxycline-depth "
    "field for validation. Oxycline depth is not yet "
    "included in the HSI weighting."
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

        0.000000001
    )


    return (

        values
        -
        values.min()

    ) / spread


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


elif map_mode == (
    "Salinity"
):

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


elif map_mode == (
    "Dissolved Oxygen"
):

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


elif map_mode == (
    "Bathymetry"
):

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


elif map_mode == (
    "Distance from Coast"
):

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


# ============================================================
# MAP DISPLAY VALUES
# ============================================================

map_data[
    "hsi_display"
] = (
    map_data[
        "hsi"
    ]
    .round(
        2
    )
)


map_data[
    "temperature_display"
] = (
    map_data[
        "temperature"
    ]
    .round(
        2
    )
)


map_data[
    "salinity_display"
] = (
    map_data[
        "salinity"
    ]
    .round(
        2
    )
)


map_data[
    "oxygen_display"
] = (
    map_data[
        "oxygen_ml_l"
    ]
    .round(
        2
    )
)


map_data[
    "depth_display"
] = (
    map_data[
        "water_depth_m"
    ]
    .round(
        0
    )
)


map_data[
    "coast_km_display"
] = (
    map_data[
        "distance_coast_km"
    ]
    .round(
        2
    )
)


map_data[
    "coast_nm_display"
] = (
    map_data[
        "distance_coast_nm"
    ]
    .round(
        2
    )
)


map_data[
    "anomaly_display"
] = (
    map_data[
        "sst_anomaly"
    ]
    .round(
        2
    )
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


st.caption(
    "The Distance from Coast layer shows the "
    "shortest calculated distance from each "
    "E-AWARE ocean grid cell to the nearest "
    "NOAA coastline segment."
)


# ============================================================
# HIGHEST RELATIVE HABITAT
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


st.caption(
    "Bathymetry and coast distance are displayed "
    "as environmental context. Neither currently "
    "changes the HSI score."
)


# ============================================================
# DISTANCE TO COAST VALIDATION
# ============================================================

st.subheader(
    "📏 V1.3 Step 2 — Distance-to-Coast"
)


validation1, validation2, validation3, validation4 = (
    st.columns(
        4
    )
)


validation1.metric(

    "Highest-HSI Coast Distance",

    f"{best['distance_coast_km']:.2f} km"
)


validation2.metric(

    "Nautical Miles",

    f"{best['distance_coast_nm']:.2f} nm"
)


validation3.metric(
    "Coastline Dataset",
    "NOAA NCEI"
)


validation4.metric(
    "HSI Weight",
    "Not included"
)


validation_table = current_data[
    [
        "lat",
        "lon",
        "hsi",
        "water_depth_m",
        "distance_coast_km",
        "distance_coast_nm"
    ]
].copy()


validation_table = (

    validation_table

    .sort_values(
        "distance_coast_km"
    )

    .reset_index(
        drop=True
    )
)


validation_table[
    "lat"
] = (
    validation_table[
        "lat"
    ]
    .round(
        4
    )
)


validation_table[
    "lon"
] = (
    validation_table[
        "lon"
    ]
    .round(
        4
    )
)


validation_table[
    "hsi"
] = (
    validation_table[
        "hsi"
    ]
    .round(
        2
    )
)


validation_table[
    "water_depth_m"
] = (
    validation_table[
        "water_depth_m"
    ]
    .round(
        0
    )
)


validation_table[
    "distance_coast_km"
] = (
    validation_table[
        "distance_coast_km"
    ]
    .round(
        2
    )
)


validation_table[
    "distance_coast_nm"
] = (
    validation_table[
        "distance_coast_nm"
    ]
    .round(
        2
    )
)


validation_table = (
    validation_table.rename(

        columns={

            "lat":
                "Latitude",

            "lon":
                "Longitude",

            "hsi":
                "HSI",

            "water_depth_m":
                "GEBCO Depth Estimate (m)",

            "distance_coast_km":
                "Distance from Coast (km)",

            "distance_coast_nm":
                "Distance from Coast (nm)"
        }
    )
)


st.dataframe(

    validation_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# V1.4 STEP 1 — OXYCLINE VALIDATION
# ============================================================

st.subheader(
    "🫧 V1.4 Step 1 — Oxycline Depth Validation"
)


best_oxycline_available = pd.notna(

    best.get(
        "oxycline_depth_m",
        np.nan
    )
)


if best_oxycline_available:

    best_oxycline_text = (

        f"{best['oxycline_depth_m']:.1f} m"
    )

else:

    best_oxycline_text = (
        "Not detected"
    )


oxycline1, oxycline2, oxycline3, oxycline4 = (
    st.columns(
        4
    )
)


oxycline1.metric(

    "Highest-HSI Oxycline Depth",

    best_oxycline_text
)


oxycline2.metric(

    "Oxycline Threshold",

    f"{OXYCLINE_THRESHOLD_ML_L:.1f} mL/L O₂"
)


oxycline3.metric(
    "Source",
    "Copernicus BGC 3D"
)


oxycline4.metric(
    "HSI Weight",
    "Not included"
)


st.info(
    "V1.4 Step 1 defines oxycline depth as the first "
    "depth where the vertical dissolved-oxygen profile "
    "crosses 1 mL/L. When the threshold falls between "
    "two Copernicus model levels, E-AWARE estimates "
    "the crossing depth using linear interpolation. "
    "If the threshold is not reached within the available "
    "water column, the oxycline is reported as not detected "
    "rather than inventing a depth."
)


oxycline_validation = current_data[
    [
        "lat",
        "lon",
        "hsi",
        "oxygen_ml_l",
        "oxycline_depth_m",
        "oxycline_source_distance_km",
        "oxycline_profile_max_depth_m",
        "oxycline_profile_levels",
        "oxycline_status"
    ]
].copy()


oxycline_validation[
    "lat"
] = (
    oxycline_validation[
        "lat"
    ]
    .round(
        4
    )
)


oxycline_validation[
    "lon"
] = (
    oxycline_validation[
        "lon"
    ]
    .round(
        4
    )
)


oxycline_validation[
    "hsi"
] = (
    oxycline_validation[
        "hsi"
    ]
    .round(
        2
    )
)


oxycline_validation[
    "oxygen_ml_l"
] = (
    oxycline_validation[
        "oxygen_ml_l"
    ]
    .round(
        2
    )
)


oxycline_validation[
    "oxycline_depth_m"
] = (
    oxycline_validation[
        "oxycline_depth_m"
    ]
    .round(
        1
    )
)


oxycline_validation[
    "oxycline_source_distance_km"
] = (
    oxycline_validation[
        "oxycline_source_distance_km"
    ]
    .round(
        2
    )
)


oxycline_validation[
    "oxycline_profile_max_depth_m"
] = (
    oxycline_validation[
        "oxycline_profile_max_depth_m"
    ]
    .round(
        0
    )
)


oxycline_validation[
    "oxycline_profile_levels"
] = (
    oxycline_validation[
        "oxycline_profile_levels"
    ]
    .round(
        0
    )
)


oxycline_validation = (
    oxycline_validation.rename(

        columns={

            "lat":
                "Latitude",

            "lon":
                "Longitude",

            "hsi":
                "HSI",

            "oxygen_ml_l":
                "Surface O₂ (mL/L)",

            "oxycline_depth_m":
                "Oxycline Depth (m)",

            "oxycline_source_distance_km":
                "BGC Source Distance (km)",

            "oxycline_profile_max_depth_m":
                "Profile Max Depth (m)",

            "oxycline_profile_levels":
                "Profile Levels",

            "oxycline_status":
                "Oxycline Status"
        }
    )
)


st.dataframe(

    oxycline_validation,

    use_container_width=True,

    hide_index=True
)


if (
    current_data[
        "oxycline_depth_m"
    ]
    .notna()
    .sum()
    ==
    0
):

    st.warning(
        "The 1 mL/L threshold was not detected "
        "in the mapped Copernicus oxygen profiles "
        "for the current date. This is a valid "
        "model result and should not be replaced "
        "by an assumed oxycline depth."
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
    "V1.4 Step 1 still calculates HSI only from "
    "temperature, salinity and surface dissolved oxygen. "
    "Bathymetry, distance from coast and oxycline depth "
    "are stored as additional predictors but are not yet "
    "assigned fixed suitability weights."
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


# ============================================================
# COASTAL / OFFSHORE CHANGE
# ============================================================

st.markdown(
    "#### Coastal / Offshore Change"
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


st.info(
    "The coastal/offshore interpretation describes "
    "a change in the predicted centre of relatively "
    "suitable habitat. It is not direct tracking of "
    "individual anchoveta schools."
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
                    ]
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
                    ]
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
                    ]
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
                    ]
            }
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
                230
            ],

        get_width=
            5,

        width_min_pixels=
            5,

        pickable=
            True
    )


    start_layer = pdk.Layer(

        "ScatterplotLayer",

        data=
            start_marker,

        get_position=
            [
                "lon",
                "lat"
            ],

        get_fill_color=
            [
                30,
                190,
                90,
                240
            ],

        get_line_color=
            [
                255,
                255,
                255,
                255
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
            True
    )


    end_layer = pdk.Layer(

        "ScatterplotLayer",

        data=
            end_marker,

        get_position=
            [
                "lon",
                "lat"
            ],

        get_fill_color=
            [
                255,
                145,
                40,
                240
            ],

        get_line_color=
            [
                255,
                255,
                255,
                255
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
            True
    )


    label_layer = pdk.Layer(

        "TextLayer",

        data=
            label_points,

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
            ],

        pickable=
            False
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

            map_style=
                BASEMAP_URL,

            initial_view_state=
                pdk.ViewState(

                    latitude=
                        shift_mid_lat,

                    longitude=
                        shift_mid_lon,

                    zoom=
                        10.3,

                    pitch=
                        0
                ),

            layers=[
                path_layer,
                start_layer,
                end_layer,
                label_layer
            ],

            tooltip={
                "html":
                    "<b>{label}{name}</b>"
            }
        ),

        use_container_width=True
    )


    st.caption(
        "🟢 Current habitat centre   •   "
        "🟠 +72 h forecast habitat centre   •   "
        "🔵 Predicted habitat-centre shift"
    )


    st.markdown(

        f"**E-AWARE interpretation:** "
        f"The centre of the relatively most suitable "
        f"anchoveta habitat is forecast to shift "
        f"**{distance_text} {direction_text.lower()}**. "
        f"Relative to the coastline, it is predicted "
        f"to be **{coastal_interpretation.lower()}**."
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

    hide_index=True
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


st.caption(
    "V1.4 Step 1 keeps oxycline depth out of "
    "the fisher-facing SMS while the new vertical "
    "oxygen calculation is being validated."
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
                "Sea temperature and salinity"
        },

        {
            "Source":
                "Copernicus Marine Physics",

            "Status":
                "Analysis / forecast",

            "Use":
                "SST anomaly"
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry",

            "Status":
                "3D analysis / forecast",

            "Use":
                "Surface dissolved oxygen and "
                "vertical oxygen profiles"
        },

        {
            "Source":
                "Copernicus Marine Biogeochemistry",

            "Status":
                "Derived predictor",

            "Use":
                "1 mL/L oxycline depth by "
                "vertical interpolation"
        },

        {
            "Source":
                "GEBCO_2026 via Ocean Data Bank API",

            "Status":
                "Static terrain model",

            "Use":
                "Bathymetry / water depth "
                "and land filtering"
        },

        {
            "Source":
                "NOAA NCEI High-Res Coastline",

            "Status":
                "Static coastline geometry",

            "Use":
                "Nearest-coast distance in km "
                "and nautical miles"
        },

        {
            "Source":
                "Castillo et al. (2019)",

            "Status":
                "Scientific reference",

            "Use":
                "Anchoveta environmental reference ranges"
        },

        {
            "Source":
                "Hill et al. (2026)",

            "Status":
                "Scientific reference",

            "Use":
                "Supports bathymetry, coast distance "
                "and 1 mL/L oxycline depth "
                "as habitat predictors"
        },

        {
            "Source":
                "IMARPE",

            "Status":
                "Scientific observations",

            "Use":
                "Future training / validation"
        },

        {
            "Source":
                "PRODUCE",

            "Status":
                "Regulatory",

            "Use":
                "Future protection / closure layer"
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
    "🧪 V1.4 Step 1 Model Status"
)


status1, status2 = (
    st.columns(
        2
    )
)


status1.success(
    "V1.4 Step 1 now retrieves the full vertical "
    "Copernicus dissolved-oxygen profile and estimates "
    "the 1 mL/L oxycline depth using linear interpolation "
    "between model levels."
)


status2.warning(
    "Oxycline depth is currently a validation predictor "
    "only. It is not yet included in the HSI, map, "
    "forecast guidance or SMS until its behaviour is "
    "checked against the project study area."
)


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "⚙️ MVP V1.4 Step 1 Limitations"
):

    st.markdown(
        """
### Current HSI inputs

- Sea temperature
- Salinity
- Surface dissolved oxygen

### Additional predictors / context

- SST anomaly
- GEBCO bathymetry / water depth
- Distance from nearest coastline
- Oxycline depth at 1 mL/L O₂
- Season

### Oxycline calculation

V1.4 Step 1 downloads the full vertical dissolved-oxygen
profile from the Copernicus Marine biogeochemistry product.

The oxycline is defined as the first downward crossing of:

**1 mL/L O₂**

If the threshold lies between two Copernicus model depth
levels, E-AWARE estimates the crossing depth using linear
interpolation.

If the 1 mL/L threshold is not reached anywhere in the
available model water column, E-AWARE reports the oxycline
as unavailable rather than assigning an assumed value.

### Oxygen-grid limitation

The Copernicus biogeochemistry product has a coarser
horizontal grid than the physics product.

Surface oxygen and oxycline depth are therefore mapped
from the nearest BGC grid location to each E-AWARE
physics-grid cell.

This means multiple E-AWARE physics cells can share the
same underlying oxygen profile and oxycline estimate.

### HSI treatment

Oxycline depth is not included in the HSI during
V1.4 Step 1.

The current temporary HSI remains:

**Temperature + Salinity + Surface Oxygen**

with equal weights.

### Bathymetry

GEBCO bathymetry remains an additional spatial predictor
and is not assigned a fixed HSI score.

### Distance from coast

Distance from coast remains an additional spatial predictor
and is not assigned a fixed HSI score.

### Coastal/offshore shift

The app continues to compare the coastline distance of the
current predicted habitat centre with the +72 h habitat
centre.

This describes movement of predicted suitable habitat,
not the movement of individual fish schools.

### Still not integrated

- Oxycline-depth HSI weighting
- Oxycline map layer
- Oxycline fisher-facing guidance
- Distance-from-coast HSI weighting
- Bathymetry HSI weighting
- Chlorophyll
- Ocean currents in the HSI
- Historical IMARPE anchoveta observations
- Juvenile protection
- Fishing closures
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "E-AWARE MVP V1.4 Step 1 | "
    "El Niño Adaptive Fisheries Intelligence | "
    "Decision-support research prototype"
)


st.caption(
    "Habitat suitability is a model estimate. "
    "It is not confirmed fish location, catch advice, "
    "or a legal fishing instruction. Bathymetry, "
    "coast-distance and oxycline outputs must not "
    "be used for navigation or safety at sea."
)
