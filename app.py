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
st.caption("El Niño Adaptive Fisheries Intelligence — MVP V1.4 Step 2")
st.info(
    "E-AWARE estimates suitable anchoveta habitat from environmental conditions. "
    "It does not show the exact location or swimming direction of individual fish schools."
)

MIN_LON, MAX_LON = -81.50, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36

COMMUNITIES = pd.DataFrame([
    {"name": "Lobitos", "lat": -4.4567, "lon": -81.2849},
    {"name": "Siches", "lat": -4.4890, "lon": -81.2680},
    {"name": "Piedritas", "lat": -4.5190, "lon": -81.2630},
])

TEMPERATURE_DATASET = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
SALINITY_DATASET = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
SST_ANOMALY_DATASET = "cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m"
OXYGEN_DATASET = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"

GEBCO_API_URL = "https://api.odb.ntu.edu.tw/gebco"
NOAA_COASTLINE_QUERY_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/EMAG2v3/MapServer/1/query"
)
BASEMAP_URL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

COASTLINE_QUERY_MARGIN_DEG = 0.75
KM_PER_NAUTICAL_MILE = 1.852
OXYGEN_MMOL_M3_TO_ML_L = 0.0223916
OXYCLINE_THRESHOLD_ML_L = 1.0


# ============================================================
# COPERNICUS
# ============================================================

def get_credentials():
    username = st.secrets.get("COPERNICUS_USERNAME", "")
    password = st.secrets.get("COPERNICUS_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("Copernicus credentials are missing from Streamlit Secrets.")
    return username, password


def prepare_dataframe(dataframe, variable_name):
    df = dataframe.copy()
    rename_map = {}

    for column in df.columns:
        c = str(column).lower()

        if c in ["latitude", "lat"]:
            rename_map[column] = "lat"

        elif c in ["longitude", "lon"]:
            rename_map[column] = "lon"

    df = df.rename(columns=rename_map)

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
        subset=[
            variable_name
        ]
    ).copy()

    df["date"] = pd.to_datetime(
        df["time"]
    ).dt.date

    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

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
    ]


def download_subset(
    dataset_id,
    variable,
    output_name,
    start_date,
    end_date,
    use_depth=True
):
    username, password = get_credentials()

    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"

    with tempfile.TemporaryDirectory() as temp_dir:

        filename = f"{output_name}.nc"

        args = {
            "dataset_id": dataset_id,
            "variables": [variable],
            "minimum_longitude": MIN_LON,
            "maximum_longitude": MAX_LON,
            "minimum_latitude": MIN_LAT,
            "maximum_latitude": MAX_LAT,
            "start_datetime": f"{start_date}T00:00:00",
            "end_datetime": f"{end_date}T23:59:59",
            "username": username,
            "password": password,
            "service": "arco-geo-series",
            "output_filename": filename,
            "output_directory": temp_dir,
            "disable_progress_bar": True,
        }

        if use_depth:
            args["minimum_depth"] = 0
            args["maximum_depth"] = 1

        response = copernicusmarine.subset(
            **args
        )

        possible = Path(
            str(
                response.file_path
            )
        )

        file_path = (
            possible
            if possible.exists()
            else Path(temp_dir) / filename
        )

        if not file_path.exists():

            candidates = list(
                Path(temp_dir).glob("*.nc")
            )

            if not candidates:
                raise RuntimeError(
                    f"{output_name} download completed "
                    "but no NetCDF file was found."
                )

            file_path = candidates[0]

        with xr.open_dataset(
            file_path
        ) as ds:

            if variable not in ds.data_vars:

                raise RuntimeError(
                    f"Variable '{variable}' "
                    f"was not found in {output_name}."
                )

            da = ds[
                variable
            ]

            if "depth" in da.dims:

                da = da.isel(
                    depth=0
                )

            df = (
                da
                .load()
                .to_dataframe(
                    name=output_name
                )
                .reset_index()
            )

    return prepare_dataframe(
        df,
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
# FULL 3D OXYGEN PROFILE
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_oxygen_profile(
    start_date,
    end_date
):

    username, password = get_credentials()

    os.environ[
        "COPERNICUSMARINE_TRUST_ENV"
    ] = "False"

    with tempfile.TemporaryDirectory() as temp_dir:

        filename = "oxygen_profile.nc"

        response = copernicusmarine.subset(
            dataset_id=OXYGEN_DATASET,
            variables=["o2"],
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            username=username,
            password=password,
            service="arco-geo-series",
            output_filename=filename,
            output_directory=temp_dir,
            disable_progress_bar=True,
        )

        possible = Path(
            str(
                response.file_path
            )
        )

        file_path = (
            possible
            if possible.exists()
            else Path(temp_dir) / filename
        )

        if not file_path.exists():

            candidates = list(
                Path(temp_dir).glob("*.nc")
            )

            if not candidates:
                raise RuntimeError(
                    "No NetCDF file was found "
                    "for the oxygen profile."
                )

            file_path = candidates[0]

        with xr.open_dataset(
            file_path
        ) as ds:

            if "o2" not in ds.data_vars:

                raise RuntimeError(
                    "Variable 'o2' was not found "
                    "in the oxygen-profile file."
                )

            df = (
                ds["o2"]
                .load()
                .to_dataframe(
                    name="oxygen_mmol_m3"
                )
                .reset_index()
            )

    rename_map = {}

    for column in df.columns:

        c = str(
            column
        ).lower()

        if c in [
            "latitude",
            "lat"
        ]:

            rename_map[
                column
            ] = "lat"

        elif c in [
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
            "depth"
        ]
    ).copy()

    df["date"] = pd.to_datetime(
        df["time"]
    ).dt.date

    df["depth"] = df[
        "depth"
    ].astype(float)

    df["lat"] = df[
        "lat"
    ].astype(float)

    df["lon"] = df[
        "lon"
    ].astype(float)

    df["lat_key"] = df[
        "lat"
    ].round(3)

    df["lon_key"] = df[
        "lon"
    ].round(3)

    df["oxygen_ml_l"] = (
        df[
            "oxygen_mmol_m3"
        ].astype(float)
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

    depths = clean[
        "depth"
    ].to_numpy(
        dtype=float
    )

    oxygen = clean[
        "oxygen_ml_l"
    ].to_numpy(
        dtype=float
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
            "Threshold reached at shallowest model level"
        )

    for i in range(
        1,
        len(
            depths
        )
    ):

        o1 = oxygen[
            i - 1
        ]

        o2 = oxygen[
            i
        ]

        if (
            o1
            >
            threshold_ml_l
            and
            o2
            <=
            threshold_ml_l
        ):

            d1 = depths[
                i - 1
            ]

            d2 = depths[
                i
            ]

            if o2 == o1:

                crossing = d2

            else:

                fraction = (
                    threshold_ml_l
                    -
                    o1
                ) / (
                    o2
                    -
                    o1
                )

                crossing = (
                    d1
                    +
                    fraction
                    *
                    (
                        d2
                        -
                        d1
                    )
                )

            return (
                float(
                    crossing
                ),
                "Interpolated 1 mL/L crossing"
            )

    return (
        np.nan,
        "1 mL/L threshold not reached in available BGC profile"
    )


def calculate_oxycline_table(
    profile
):

    records = []

    for (
        target_date,
        lat_key,
        lon_key
    ), group in profile.groupby(
        [
            "date",
            "lat_key",
            "lon_key"
        ]
    ):

        group = group.sort_values(
            "depth"
        )

        raw_depth, raw_status = (
            estimate_oxycline_depth(
                group
            )
        )

        records.append(
            {
                "date":
                    target_date,

                "lat":
                    float(
                        group[
                            "lat"
                        ].iloc[
                            0
                        ]
                    ),

                "lon":
                    float(
                        group[
                            "lon"
                        ].iloc[
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

                "raw_oxycline_depth_m":
                    raw_depth,

                "raw_oxycline_status":
                    raw_status,

                "profile_min_depth_m":
                    float(
                        group[
                            "depth"
                        ].min()
                    ),

                "profile_max_depth_m":
                    float(
                        group[
                            "depth"
                        ].max()
                    ),

                "profile_levels":
                    int(
                        group[
                            "depth"
                        ].nunique()
                    )
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# MERGE / NEAREST BGC MAPPING
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
            "No matching physics grid cells were found."
        )

    return merged


def nearest_source_indices(
    target_rows,
    source_rows
):

    target_lat = target_rows[
        "lat"
    ].to_numpy(
        dtype=float
    )

    target_lon = target_rows[
        "lon"
    ].to_numpy(
        dtype=float
    )

    source_lat = source_rows[
        "lat"
    ].to_numpy(
        dtype=float
    )

    source_lon = source_rows[
        "lon"
    ].to_numpy(
        dtype=float
    )

    dlat = (
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

    dlon = (
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

    distance = np.sqrt(
        dlat
        ** 2
        +
        dlon
        ** 2
    )

    idx = np.argmin(
        distance,
        axis=1
    )

    nearest_distance = distance[
        np.arange(
            len(
                target_rows
            )
        ),
        idx
    ]

    return (
        idx,
        nearest_distance
    )


def attach_nearest_oxygen(
    physics,
    oxygen
):

    result = physics.copy()

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
        ].unique()
    ):

        mask = (
            result[
                "date"
            ]
            ==
            target_date
        )

        target = result.loc[
            mask
        ]

        source = oxygen[
            oxygen[
                "date"
            ]
            ==
            target_date
        ]

        if source.empty:
            continue

        idx, dist = (
            nearest_source_indices(
                target,
                source
            )
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
            "Oxygen data could not be matched "
            "to the physics grid."
        )

    return result


def attach_nearest_oxycline(
    physics,
    oxycline
):

    result = physics.copy()

    result[
        "raw_oxycline_depth_m"
    ] = np.nan

    result[
        "raw_oxycline_status"
    ] = None

    result[
        "oxycline_source_distance_km"
    ] = np.nan

    result[
        "oxycline_profile_max_depth_m"
    ] = np.nan

    result[
        "oxycline_profile_levels"
    ] = np.nan

    mapped = False

    for target_date in sorted(
        result[
            "date"
        ].unique()
    ):

        mask = (
            result[
                "date"
            ]
            ==
            target_date
        )

        target = result.loc[
            mask
        ]

        source = oxycline[
            oxycline[
                "date"
            ]
            ==
            target_date
        ]

        if source.empty:
            continue

        idx, dist = (
            nearest_source_indices(
                target,
                source
            )
        )

        nearest = source.iloc[
            idx
        ]

        result.loc[
            mask,
            "raw_oxycline_depth_m"
        ] = nearest[
            "raw_oxycline_depth_m"
        ].to_numpy(
            float
        )

        result.loc[
            mask,
            "raw_oxycline_status"
        ] = nearest[
            "raw_oxycline_status"
        ].astype(
            str
        ).to_numpy()

        result.loc[
            mask,
            "oxycline_source_distance_km"
        ] = dist

        result.loc[
            mask,
            "oxycline_profile_max_depth_m"
        ] = nearest[
            "profile_max_depth_m"
        ].to_numpy(
            float
        )

        result.loc[
            mask,
            "oxycline_profile_levels"
        ] = nearest[
            "profile_levels"
        ].to_numpy(
            float
        )

        mapped = True

    if not mapped:

        raise RuntimeError(
            "Oxycline results could not "
            "be matched to the E-AWARE grid."
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

        value = float(
            value
        )

        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        str
    ):

        try:

            value = float(
                value
            )

            return (
                value
                if math.isfinite(
                    value
                )
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

        n = first_numeric(
            value
        )

        return (
            []
            if n is None
            else [
                n
            ]
        )

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

            n = (
                first_numeric(
                    item[
                        "z"
                    ]
                )
                if isinstance(
                    item,
                    dict
                )
                and
                "z" in item
                else
                first_numeric(
                    item
                )
            )

            if n is not None:

                output.append(
                    float(
                        n
                    )
                )

        return output

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
            f"GEBCO returned an implausible "
            f"elevation: {value} m."
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
                "E-AWARE/1.4 academic prototype"
        },
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()

    z = (
        first_numeric(
            payload[
                "z"
            ]
        )
        if isinstance(
            payload,
            dict
        )
        and
        "z" in payload
        else
        first_numeric(
            payload
        )
    )

    return validate_elevation(
        z
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
                    "E-AWARE/1.4 academic prototype"
            },
            timeout=40
        )

        response.raise_for_status()

        payload = response.json()

        values = (
            numeric_list(
                payload[
                    "z"
                ]
            )
            if isinstance(
                payload,
                dict
            )
            and
            "z" in payload
            else
            numeric_list(
                payload
            )
        )

        if len(
            values
        ) == len(
            points
        ):

            return [
                validate_elevation(
                    v
                )
                for v in values
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
                r.lat
            ),
            float(
                r.lon
            )
        )
        for r in unique_grid.itertuples(
            index=False
        )
    )

    elevations = get_gebco_batch(
        points
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

    bathy = unique_grid[
        [
            "lat_key",
            "lon_key"
        ]
    ].copy()

    bathy[
        "gebco_elevation_m"
    ] = np.asarray(
        elevations,
        dtype=float
    )

    bathy[
        "is_ocean_gebco"
    ] = (
        bathy[
            "gebco_elevation_m"
        ]
        <
        0
    )

    bathy[
        "water_depth_m"
    ] = np.where(
        bathy[
            "is_ocean_gebco"
        ],
        -
        bathy[
            "gebco_elevation_m"
        ],
        np.nan
    )

    result = result.merge(
        bathy,
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
# PHYSICAL OXYCLINE VALIDATION
# ============================================================

def validate_oxycline_against_bathymetry(
    dataframe
):

    result = dataframe.copy()

    result[
        "oxycline_depth_m"
    ] = np.nan

    result[
        "oxycline_validation_status"
    ] = ""

    result[
        "oxycline_below_seabed_by_m"
    ] = np.nan

    for index, row in result.iterrows():

        raw_depth = row.get(
            "raw_oxycline_depth_m",
            np.nan
        )

        local_depth = row.get(
            "water_depth_m",
            np.nan
        )

        raw_status = str(
            row.get(
                "raw_oxycline_status",
                ""
            )
        )

        if pd.isna(
            raw_depth
        ):

            result.at[
                index,
                "oxycline_validation_status"
            ] = (
                raw_status
                if raw_status
                else
                "Oxycline not detected in BGC profile"
            )

            continue

        if pd.isna(
            local_depth
        ):

            result.at[
                index,
                "oxycline_validation_status"
            ] = (
                "Cannot validate oxycline because "
                "local GEBCO depth is unavailable"
            )

            continue

        raw_depth = float(
            raw_depth
        )

        local_depth = float(
            local_depth
        )

        if raw_depth <= local_depth:

            result.at[
                index,
                "oxycline_depth_m"
            ] = raw_depth

            result.at[
                index,
                "oxycline_validation_status"
            ] = (
                "Accepted — oxycline lies within "
                "local GEBCO water column"
            )

        else:

            result.at[
                index,
                "oxycline_below_seabed_by_m"
            ] = (
                raw_depth
                -
                local_depth
            )

            result.at[
                index,
                "oxycline_validation_status"
            ] = (
                "Rejected — raw BGC oxycline falls "
                "below local GEBCO seabed"
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
            "4326"
    }

    headers = {
        "Accept":
            "application/json",
        "User-Agent":
            "E-AWARE/1.4 academic prototype"
    }

    try:

        params = dict(
            common
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

        payload = response.json()

        lines = []

        for feature in payload.get(
            "features",
            []
        ):

            g = feature.get(
                "geometry"
            ) or {}

            if (
                g.get(
                    "type"
                )
                ==
                "LineString"
                and
                g.get(
                    "coordinates"
                )
            ):

                lines.append(
                    g[
                        "coordinates"
                    ]
                )

            elif (
                g.get(
                    "type"
                )
                ==
                "MultiLineString"
                and
                g.get(
                    "coordinates"
                )
            ):

                lines.extend(
                    g[
                        "coordinates"
                    ]
                )

        if lines:
            return lines

    except Exception:
        pass

    params = dict(
        common
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

    payload = response.json()

    if "error" in payload:

        raise RuntimeError(
            f"NOAA coastline service error: "
            f"{payload['error']}"
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
            "NOAA coastline query returned "
            "no coastline geometry."
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

    km_lat = 111.32

    km_lon = (
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
    ) * km_lon

    y1 = (
        float(
            lat1
        )
        -
        point_lat
    ) * km_lat

    x2 = (
        float(
            lon2
        )
        -
        point_lon
    ) * km_lon

    y2 = (
        float(
            lat2
        )
        -
        point_lat
    ) * km_lat

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

    length2 = (
        dx
        *
        dx
        +
        dy
        *
        dy
    )

    if length2 == 0:

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

    ) / length2

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

    minimum = math.inf

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

        for i in range(
            len(
                line
            )
            -
            1
        ):

            p1 = line[
                i
            ]

            p2 = line[
                i + 1
            ]

            if (
                len(
                    p1
                )
                <
                2
                or
                len(
                    p2
                )
                <
                2
            ):

                continue

            d = point_to_segment_distance_km(
                float(
                    lat
                ),
                float(
                    lon
                ),
                float(
                    p1[
                        0
                    ]
                ),
                float(
                    p1[
                        1
                    ]
                ),
                float(
                    p2[
                        0
                    ]
                ),
                float(
                    p2[
                        1
                    ]
                )
            )

            minimum = min(
                minimum,
                d
            )

    if not math.isfinite(
        minimum
    ):

        raise RuntimeError(
            "No valid coastline segments "
            "were available."
        )

    return float(
        minimum
    )


def attach_distance_to_coast(
    dataframe
):

    result = dataframe.copy()

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

    distances = [
        distance_to_nearest_coast_km(
            float(
                r.lat
            ),
            float(
                r.lon
            ),
            coastline_lines
        )
        for r in unique_grid.itertuples(
            index=False
        )
    ]

    coast = unique_grid[
        [
            "lat_key",
            "lon_key"
        ]
    ].copy()

    coast[
        "distance_coast_km"
    ] = np.asarray(
        distances,
        dtype=float
    )

    coast[
        "distance_coast_nm"
    ] = (
        coast[
            "distance_coast_km"
        ]
        /
        KM_PER_NAUTICAL_MILE
    )

    result = result.merge(
        coast,
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
# HABITAT / HSI
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
                weights=weights
            )
        ),
        float(
            np.average(
                best_area[
                    "lon"
                ],
                weights=weights
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

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2
        -
        lat1
    )

    dlambda = math.radians(
        lon2
        -
        lon1
    )

    a = (
        math.sin(
            dphi
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
            dlambda
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

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dlambda = math.radians(
        lon2
        -
        lon1
    )

    x = (
        math.sin(
            dlambda
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
            dlambda
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
        "Loading E-AWARE V1.4 Step 2 environmental data...",
        expanded=True
    ) as status:

        username, password = get_credentials()

        status.write(
            "🔐 Checking Copernicus credentials..."
        )

        if not copernicusmarine.login(
            username=username,
            password=password,
            check_credentials_valid=True
        ):

            raise RuntimeError(
                "Copernicus rejected the supplied credentials."
            )

        status.write(
            "🌡️ 1/7 Downloading temperature..."
        )

        temperature_data = load_temperature(
            selected_date,
            end_date
        )

        status.write(
            "🧂 2/7 Downloading salinity..."
        )

        salinity_data = load_salinity(
            selected_date,
            end_date
        )

        status.write(
            "🌊 3/7 Downloading SST anomaly..."
        )

        anomaly_data = load_sst_anomaly(
            selected_date,
            end_date
        )

        status.write(
            "🫧 4/7 Downloading full 3D oxygen profile..."
        )

        oxygen_profile_data = load_oxygen_profile(
            selected_date,
            end_date
        )

        oxygen_data = surface_oxygen_from_profile(
            oxygen_profile_data
        )

        oxycline_data = calculate_oxycline_table(
            oxygen_profile_data
        )

        ocean = merge_physics_data(
            temperature_data,
            salinity_data,
            anomaly_data
        )

        ocean = attach_nearest_oxygen(
            ocean,
            oxygen_data
        )

        ocean = attach_nearest_oxycline(
            ocean,
            oxycline_data
        )

        status.write(
            "🗻 5/7 Retrieving GEBCO_2026 bathymetry..."
        )

        ocean = attach_gebco_bathymetry(
            ocean
        )

        status.write(
            "📏 6/7 Calculating distance from coast..."
        )

        ocean = attach_distance_to_coast(
            ocean
        )

        status.write(
            "🫧 7/7 Applying physical oxycline validation..."
        )

        ocean = validate_oxycline_against_bathymetry(
            ocean
        )

        accepted = int(
            ocean[
                "oxycline_depth_m"
            ]
            .notna()
            .sum()
        )

        rejected = int(
            ocean[
                "oxycline_validation_status"
            ]
            .str.startswith(
                "Rejected",
                na=False
            )
            .sum()
        )

        status.write(
            f"✅ Oxycline validation: "
            f"{accepted} accepted, "
            f"{rejected} rejected below seabed"
        )

        status.update(
            label=
                "✅ E-AWARE V1.4 Step 2 "
                "environmental data loaded",
            state=
                "complete",
            expanded=
                False
        )

except Exception as error:

    st.error(
        "❌ E-AWARE V1.4 Step 2 could not "
        "load all environmental data."
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


current_data = current_data.dropna(
    subset=[
        "oxygen_ml_l",
        "water_depth_m",
        "distance_coast_km",
        "distance_coast_nm"
    ]
).copy()


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


coastline_lines = load_noaa_coastline_lines()


# ============================================================
# CURRENT CONDITIONS
# ============================================================

st.subheader(
    "🌊 Current Ocean Conditions"
)


c1, c2, c3, c4 = st.columns(
    4
)


c1.metric(
    "Data Date",
    str(
        current_date
    )
)


c2.metric(
    "Highest HSI",
    f"{best['hsi']:.2f}"
)


c3.metric(
    "Highest-HSI SST",
    f"{best['temperature']:.2f} °C"
)


c4.metric(
    "Dissolved Oxygen",
    f"{best['oxygen_ml_l']:.2f} mL/L"
)


c1, c2, c3, c4 = st.columns(
    4
)


c1.metric(
    "Salinity",
    f"{best['salinity']:.2f} PSU"
)


c2.metric(
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


c3.metric(
    "GEBCO Depth Estimate",
    f"{best['water_depth_m']:.0f} m"
)


c4.metric(
    "Distance from Coast",
    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)


st.caption(
    "A raw BGC oxycline is now accepted only if "
    "it lies within the local GEBCO water column."
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


map_data = current_data.copy()


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

    values = map_data[
        "sst_anomaly"
    ].fillna(
        0
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

    n = (
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


map_data[
    "hsi_display"
] = map_data[
    "hsi"
].round(
    2
)


map_data[
    "temperature_display"
] = map_data[
    "temperature"
].round(
    2
)


map_data[
    "salinity_display"
] = map_data[
    "salinity"
].round(
    2
)


map_data[
    "oxygen_display"
] = map_data[
    "oxygen_ml_l"
].round(
    2
)


map_data[
    "depth_display"
] = map_data[
    "water_depth_m"
].round(
    0
)


map_data[
    "coast_km_display"
] = map_data[
    "distance_coast_km"
].round(
    2
)


map_data[
    "coast_nm_display"
] = map_data[
    "distance_coast_nm"
].round(
    2
)


map_data[
    "anomaly_display"
] = map_data[
    "sst_anomaly"
].round(
    2
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
        layers=
            [
                ocean_layer,
                community_layer
            ],
        tooltip=
            {
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
    use_container_width=
        True
)


# ============================================================
# HIGHEST RELATIVE HABITAT
# ============================================================

st.subheader(
    "📍 Highest Relative Habitat Suitability"
)


z1, z2, z3, z4, z5 = st.columns(
    5
)


z1.metric(
    "Latitude",
    f"{best['lat']:.4f}"
)


z2.metric(
    "Longitude",
    f"{best['lon']:.4f}"
)


z3.metric(
    "Suitability",
    hsi_class(
        best[
            "hsi"
        ]
    )
)


z4.metric(
    "GEBCO Depth",
    f"{best['water_depth_m']:.0f} m"
)


z5.metric(
    "Coast Distance",
    (
        f"{best['distance_coast_km']:.2f} km / "
        f"{best['distance_coast_nm']:.2f} nm"
    )
)


# ============================================================
# OXYCLINE PHYSICAL VALIDATION
# ============================================================

st.subheader(
    "🫧 V1.4 Step 2 — Oxycline Physical Validation"
)


raw_oxycline = best.get(
    "raw_oxycline_depth_m",
    np.nan
)


validated_oxycline = best.get(
    "oxycline_depth_m",
    np.nan
)


validation_status = str(
    best.get(
        "oxycline_validation_status",
        "Unavailable"
    )
)


raw_text = (
    f"{raw_oxycline:.1f} m"
    if pd.notna(
        raw_oxycline
    )
    else
    "Not detected"
)


validated_text = (
    f"{validated_oxycline:.1f} m"
    if pd.notna(
        validated_oxycline
    )
    else
    "Rejected / unavailable"
)


o1, o2, o3, o4 = st.columns(
    4
)


o1.metric(
    "Raw Copernicus Oxycline",
    raw_text
)


o2.metric(
    "Local GEBCO Water Depth",
    f"{best['water_depth_m']:.1f} m"
)


o3.metric(
    "Validated Oxycline",
    validated_text
)


o4.metric(
    "HSI Weight",
    "Not included"
)


if validation_status.startswith(
    "Accepted"
):

    st.success(
        validation_status
    )


elif validation_status.startswith(
    "Rejected"
):

    below_by = best.get(
        "oxycline_below_seabed_by_m",
        np.nan
    )

    extra = (
        f" The raw estimate is "
        f"{below_by:.1f} m deeper than the local seabed."
        if pd.notna(
            below_by
        )
        else
        ""
    )

    st.warning(
        validation_status
        +
        "."
        +
        extra
        +
        " E-AWARE therefore keeps the raw value only "
        "for transparency and does not treat it as a "
        "valid local oxycline."
    )


else:

    st.info(
        validation_status
    )


st.caption(
    "Physical rule: accept the mapped oxycline only when "
    "raw oxycline depth ≤ local GEBCO water depth."
)


validation_table = current_data[
    [
        "lat",
        "lon",
        "hsi",
        "water_depth_m",
        "oxygen_ml_l",
        "raw_oxycline_depth_m",
        "oxycline_depth_m",
        "oxycline_below_seabed_by_m",
        "oxycline_source_distance_km",
        "oxycline_profile_max_depth_m",
        "oxycline_profile_levels",
        "oxycline_validation_status"
    ]
].copy()


for col, decimals in {
    "lat":
        4,
    "lon":
        4,
    "hsi":
        2,
    "water_depth_m":
        1,
    "oxygen_ml_l":
        2,
    "raw_oxycline_depth_m":
        1,
    "oxycline_depth_m":
        1,
    "oxycline_below_seabed_by_m":
        1,
    "oxycline_source_distance_km":
        2,
    "oxycline_profile_max_depth_m":
        0,
    "oxycline_profile_levels":
        0
}.items():

    validation_table[
        col
    ] = validation_table[
        col
    ].round(
        decimals
    )


validation_table = validation_table.rename(
    columns=
        {
            "lat":
                "Latitude",
            "lon":
                "Longitude",
            "hsi":
                "HSI",
            "water_depth_m":
                "Local GEBCO Depth (m)",
            "oxygen_ml_l":
                "Surface O₂ (mL/L)",
            "raw_oxycline_depth_m":
                "Raw BGC Oxycline (m)",
            "oxycline_depth_m":
                "Validated Oxycline (m)",
            "oxycline_below_seabed_by_m":
                "Below Seabed By (m)",
            "oxycline_source_distance_km":
                "BGC Source Distance (km)",
            "oxycline_profile_max_depth_m":
                "BGC Profile Max Depth (m)",
            "oxycline_profile_levels":
                "BGC Profile Levels",
            "oxycline_validation_status":
                "Validation Status"
        }
)


st.dataframe(
    validation_table,
    use_container_width=
        True,
    hide_index=
        True
)


# ============================================================
# HABITAT REFERENCE
# ============================================================

st.subheader(
    "🐟 Why is this habitat considered suitable?"
)


r1, r2, r3, r4 = st.columns(
    4
)


r1.metric(
    "Reference Temperature",
    (
        f"{reference['temp_min']} – "
        f"{reference['temp_max']} °C"
    )
)


r2.metric(
    "Reference Salinity",
    (
        f"{reference['sal_min']} – "
        f"{reference['sal_max']} PSU"
    )
)


r3.metric(
    "Reference Oxygen",
    (
        f"{reference['oxygen_min']} – "
        f"{reference['oxygen_max']} mL/L"
    )
)


r4.metric(
    "Season",
    reference[
        "season"
    ]
)


st.warning(
    "V1.4 Step 2 still calculates HSI only from temperature, "
    "salinity and surface dissolved oxygen. Bathymetry, distance "
    "from coast and validated oxycline remain additional predictors "
    "and are not assigned fixed HSI weights."
)


# ============================================================
# FORECAST
# ============================================================

forecast_centres = []

forecast_results = []


for forecast_date in available_dates:

    frame = ocean[
        ocean[
            "date"
        ]
        ==
        forecast_date
    ].copy()

    frame = frame.dropna(
        subset=
            [
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

    centre_coast_km = None

    centre_coast_nm = None

    if centre is not None:

        centre_coast_km = distance_to_nearest_coast_km(
            centre[
                0
            ],
            centre[
                1
            ],
            coastline_lines
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

    best_f = frame.loc[
        frame[
            "hsi"
        ].idxmax()
    ]

    forecast_results.append(
        {
            "Date":
                forecast_date,

            "Highest HSI":
                round(
                    float(
                        best_f[
                            "hsi"
                        ]
                    ),
                    2
                ),

            "Temperature °C":
                round(
                    float(
                        best_f[
                            "temperature"
                        ]
                    ),
                    2
                ),

            "Salinity PSU":
                round(
                    float(
                        best_f[
                            "salinity"
                        ]
                    ),
                    2
                ),

            "Oxygen mL/L":
                round(
                    float(
                        best_f[
                            "oxygen_ml_l"
                        ]
                    ),
                    2
                ),

            "Water Depth m":
                round(
                    float(
                        best_f[
                            "water_depth_m"
                        ]
                    ),
                    0
                ),

            "Distance Coast km":
                round(
                    float(
                        best_f[
                            "distance_coast_km"
                        ]
                    ),
                    2
                ),

            "Distance Coast nm":
                round(
                    float(
                        best_f[
                            "distance_coast_nm"
                        ]
                    ),
                    2
                ),

            "Raw Oxycline m":
                (
                    round(
                        float(
                            best_f[
                                "raw_oxycline_depth_m"
                            ]
                        ),
                        1
                    )
                    if pd.notna(
                        best_f[
                            "raw_oxycline_depth_m"
                        ]
                    )
                    else
                    None
                ),

            "Validated Oxycline m":
                (
                    round(
                        float(
                            best_f[
                                "oxycline_depth_m"
                            ]
                        ),
                        1
                    )
                    if pd.notna(
                        best_f[
                            "oxycline_depth_m"
                        ]
                    )
                    else
                    None
                ),

            "Oxycline Status":
                str(
                    best_f[
                        "oxycline_validation_status"
                    ]
                ),

            "SST Anomaly °C":
                (
                    round(
                        float(
                            best_f[
                                "sst_anomaly"
                            ]
                        ),
                        2
                    )
                    if pd.notna(
                        best_f[
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
                    if centre_coast_km is not None
                    else
                    None
                ),

            "Centre Coast nm":
                (
                    round(
                        centre_coast_nm,
                        2
                    )
                    if centre_coast_nm is not None
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


direction_text = "Unavailable"
distance_text = "—"
bearing_text = "—"
period_text = "Insufficient forecast data"

start_point_text = "—"
end_point_text = "—"

north_south_text = "—"
east_west_text = "—"

start_coast_text = "—"
end_coast_text = "—"

coastal_interpretation = "Unavailable"

start = None

end = None


if len(
    forecast_centres
) >= 2:

    start = forecast_centres[
        0
    ]

    end = forecast_centres[
        -1
    ]

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

    north_km, east_km = movement_components(
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

    north_south_text = (
        f"{abs(north_km):.2f} km North"
        if north_km > 0
        else
        f"{abs(north_km):.2f} km South"
        if north_km < 0
        else
        "0.00 km"
    )

    east_west_text = (
        f"{abs(east_km):.2f} km East"
        if east_km > 0
        else
        f"{abs(east_km):.2f} km West"
        if east_km < 0
        else
        "0.00 km"
    )

    direction_text = (
        direction_name(
            bearing
        )
        if distance >= 2.0
        else
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


s1, s2, s3 = st.columns(
    3
)


s1.metric(
    "Predicted Direction",
    direction_text
)


s2.metric(
    "Total Shift",
    distance_text
)


s3.metric(
    "Bearing",
    bearing_text
)


st.markdown(
    f"**Forecast Period:** "
    f"{period_text}"
)


m1, m2 = st.columns(
    2
)


m1.metric(
    "North / South",
    north_south_text
)


m2.metric(
    "East / West",
    east_west_text
)


p1, p2 = st.columns(
    2
)


p1.metric(
    "Current / Start Position",
    start_point_text
)


p2.metric(
    "+72 h Forecast Position",
    end_point_text
)


co1, co2, co3 = st.columns(
    3
)


co1.metric(
    "Start Distance from Coast",
    start_coast_text
)


co2.metric(
    "+72 h Distance from Coast",
    end_coast_text
)


co3.metric(
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
                        10.3,
                    pitch=
                        0
                ),
            layers=
                [
                    path_layer,
                    start_layer,
                    end_layer,
                    label_layer
                ],
            tooltip=
                {
                    "html":
                        "<b>{label}{name}</b>"
                }
        ),
        use_container_width=
            True
    )

    st.markdown(
        f"**E-AWARE interpretation:** "
        f"The centre of relatively suitable habitat "
        f"is forecast to shift "
        f"**{distance_text} {direction_text.lower()}**. "
        f"Relative to the coastline, it is predicted "
        f"to be **{coastal_interpretation.lower()}**."
    )

else:

    st.warning(
        "Not enough forecast information is available "
        "to draw the habitat-shift map."
    )


# ============================================================
# FORECAST TABLE
# ============================================================

st.subheader(
    "📅 72-Hour Habitat Forecast"
)


st.dataframe(
    pd.DataFrame(
        forecast_results
    ),
    use_container_width=
        True,
    hide_index=
        True
)


# ============================================================
# SMS
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
    height=
        500
)


st.caption(
    "Validated oxycline remains out of the fisher-facing "
    "SMS and HSI while V1.4 is being validated."
)


# ============================================================
# SOURCES
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
                "Derived from Copernicus BGC",
            "Status":
                "Validation predictor",
            "Use":
                "Raw 1 mL/L oxycline by vertical interpolation"
        },
        {
            "Source":
                "GEBCO_2026 via Ocean Data Bank API",
            "Status":
                "Static terrain model",
            "Use":
                "Bathymetry, land filtering and oxycline physical validation"
        },
        {
            "Source":
                "NOAA NCEI High-Res Coastline",
            "Status":
                "Static coastline geometry",
            "Use":
                "Distance from coast in km and nautical miles"
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
                "Supports bathymetry, coast distance and 1 mL/L oxycline as predictors"
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
    use_container_width=
        True,
    hide_index=
        True
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "🧪 V1.4 Step 2 Model Status"
)


s1, s2 = st.columns(
    2
)


s1.success(
    "Raw Copernicus oxycline estimates are checked "
    "against local GEBCO seabed depth before acceptance."
)


s2.warning(
    "Validated oxycline is still not included in the HSI. "
    "Rejected values are kept only for transparency."
)


with st.expander(
    "⚙️ MVP V1.4 Step 2 Limitations"
):

    st.markdown(
        """
### Current HSI inputs
- Sea temperature
- Salinity
- Surface dissolved oxygen

### Additional predictors / context
- SST anomaly
- GEBCO bathymetry
- Distance from coast
- Validated oxycline depth
- Season

### Oxycline physical constraint

The raw oxycline is derived from the nearest Copernicus BGC
profile using the first downward crossing of **1 mL/L O₂**.

Because the BGC grid is coarser than the E-AWARE physics grid,
the nearest BGC profile can represent water deeper than the
local E-AWARE cell.

V1.4 Step 2 therefore applies:

**Accept oxycline only if raw oxycline depth ≤ local GEBCO water depth.**

If the raw oxycline lies below the local seabed, the value is rejected.

### Important interpretation

A rejected below-seabed value does **not** prove that no local
oxycline exists above the seabed.

It means the nearest coarse BGC profile does not provide a
physically valid local oxycline estimate for that finer
E-AWARE cell.

### HSI

The temporary HSI remains:

**Temperature + Salinity + Surface Oxygen**

with equal weights.

### Still not integrated
- Oxycline HSI weighting
- Oxycline map layer
- Historical IMARPE observations
- Trained habitat model
- Chlorophyll
- Ocean currents
- Juvenile protection
- Fishing closures
        """
    )


st.divider()


st.caption(
    "E-AWARE MVP V1.4 Step 2 | "
    "El Niño Adaptive Fisheries Intelligence | "
    "Decision-support research prototype"
)


st.caption(
    "Habitat suitability is a model estimate and must not "
    "be used as navigation, safety or legal fishing advice."
)
