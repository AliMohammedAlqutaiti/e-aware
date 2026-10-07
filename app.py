import os
import math
import tempfile
from pathlib import Path
from datetime import date, timedelta

import copernicusmarine
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import xarray as xr


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="E-AWARE",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 E-AWARE")
st.caption("El Niño Adaptive Fisheries Intelligence — MVP V1")

st.info(
    "E-AWARE estimates suitable anchoveta habitat. "
    "It does not show the exact location or swimming direction "
    "of individual fish schools."
)


# ============================================================
# STUDY AREA
# Costanera corridor / northern Peru
# ============================================================

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
# COPERNICUS DATASET IDs
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


# ============================================================
# COPERNICUS LOGIN
# ============================================================

def configure_copernicus():

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
            "Copernicus credentials are missing from "
            "Streamlit Secrets."
        )

    os.environ[
        "COPERNICUSMARINE_SERVICE_USERNAME"
    ] = username

    os.environ[
        "COPERNICUSMARINE_SERVICE_PASSWORD"
    ] = password


# ============================================================
# DOWNLOAD A SMALL COPERNICUS NETCDF SUBSET
# ============================================================

def download_subset(
    dataset_id,
    variable,
    output_name,
    start_date,
    end_date,
    use_depth=True
):

    configure_copernicus()

    with tempfile.TemporaryDirectory() as temp_folder:

        filename = (
            f"{output_name}.nc"
        )

        arguments = {

            "dataset_id":
                dataset_id,

            "variables":
                [variable],

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

            "output_filename":
                filename,

            "output_directory":
                temp_folder
        }


        # Temperature and salinity are 3D datasets.
        # Only the surface layer is needed.

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


        file_path = Path(
            response.file_path
        )


        if not file_path.exists():

            raise RuntimeError(
                f"{output_name} file was not created."
            )


        with xr.open_dataset(
            file_path
        ) as dataset:


            if variable not in dataset:

                raise RuntimeError(
                    f"Variable '{variable}' was not found "
                    f"in the downloaded {output_name} dataset."
                )


            data_array = (
                dataset[
                    variable
                ]
            )


            # Select first surface depth if required.

            if "depth" in data_array.dims:

                data_array = (
                    data_array.isel(
                        depth=0
                    )
                )


            # Load into memory before
            # TemporaryDirectory deletes the file.

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
# CLEAN DOWNLOADED DATA
# ============================================================

def prepare_dataframe(
    dataframe,
    variable_name
):

    df = (
        dataframe.copy()
    )


    rename_map = {}


    for column in df.columns:

        lower_name = (
            str(
                column
            )
            .lower()
        )


        if lower_name in [
            "latitude",
            "lat"
        ]:

            rename_map[
                column
            ] = "lat"


        elif lower_name in [
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


    missing_columns = [

        column

        for column
        in required_columns

        if column
        not in df.columns
    ]


    if missing_columns:

        raise RuntimeError(
            f"Missing columns for {variable_name}: "
            f"{missing_columns}"
        )


    df = (

        df

        .dropna(
            subset=[
                variable_name
            ]
        )

        .copy()
    )


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
# INDIVIDUAL DOWNLOAD FUNCTIONS
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

        use_depth=True
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

        use_depth=True
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

        use_depth=False
    )


# ============================================================
# MERGE OCEAN DATA
# ============================================================

def merge_ocean_data(
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
            "Temperature, salinity and SST anomaly downloaded, "
            "but no matching ocean grid cells were found."
        )


    return merged


# ============================================================
# ANCHOVETA HABITAT REFERENCE
# ============================================================

def habitat_reference(
    month
):

    # Southern Hemisphere summer

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
                35.14
        }


    # Southern Hemisphere winter

    elif month in [
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
                35.12
        }


    # Transition-season approximation
    # for MVP V1 only.

    else:

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
                35.14
        }


# ============================================================
# RANGE SUITABILITY SCORE
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
# HABITAT SUITABILITY INDEX
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

        shoulder=2.0
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

        shoulder=0.5
    )


    # ========================================================
    # MVP V1 HSI
    #
    # Temperature = 50%
    # Salinity    = 50%
    #
    # These are temporary engineering weights.
    # ========================================================

    df[
        "hsi"
    ] = (

        0.50
        *
        df[
            "temperature_score"
        ]

        +

        0.50
        *
        df[
            "salinity_score"
        ]

    ).clip(
        0,
        1
    )


    return df


# ============================================================
# HSI CATEGORY
# ============================================================

def hsi_class(
    value
):

    if value >= 0.80:

        return (
            "Very High"
        )


    elif value >= 0.60:

        return (
            "High"
        )


    elif value >= 0.35:

        return (
            "Moderate"
        )


    return (
        "Low"
    )


# ============================================================
# HABITAT CENTROID
# ============================================================

def habitat_centroid(
    dataframe
):

    if dataframe.empty:

        return None


    threshold = max(

        0.60,

        float(
            dataframe[
                "hsi"
            ]

            .quantile(
                0.75
            )
        )
    )


    high_habitat = dataframe[

        dataframe[
            "hsi"
        ]

        >=
        threshold

    ].copy()


    if high_habitat.empty:

        return None


    weights = (

        high_habitat[
            "hsi"
        ]

        .to_numpy()
    )


    latitude = np.average(

        high_habitat[
            "lat"
        ],

        weights=weights
    )


    longitude = np.average(

        high_habitat[
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
# DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = (
        6371.0
    )


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
        radius

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


    bearing = math.degrees(

        math.atan2(
            x,
            y
        )
    )


    return (

        bearing
        +
        360

    ) % 360


# ============================================================
# DIRECTION NAME
# ============================================================

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
# REQUEST PERIOD
# ============================================================

end_date = (

    selected_date

    +

    timedelta(
        days=3
    )
)


# ============================================================
# DOWNLOAD COPERNICUS DATA
# ============================================================

try:

    with st.status(
        "Downloading Copernicus Marine data...",
        expanded=True
    ) as status:


        status.write(
            "🔐 Checking Copernicus credentials..."
        )


        configure_copernicus()


        status.write(
            "🌡️ 1/3 Downloading temperature subset..."
        )


        temperature_data = load_temperature(

            selected_date,
            end_date
        )


        status.write(
            f"✅ Temperature loaded "
            f"({len(temperature_data)} cells)"
        )


        status.write(
            "🧂 2/3 Downloading salinity subset..."
        )


        salinity_data = load_salinity(

            selected_date,
            end_date
        )


        status.write(
            f"✅ Salinity loaded "
            f"({len(salinity_data)} cells)"
        )


        status.write(
            "🌊 3/3 Downloading official SST anomaly..."
        )


        anomaly_data = load_sst_anomaly(

            selected_date,
            end_date
        )


        status.write(
            f"✅ SST anomaly loaded "
            f"({len(anomaly_data)} cells)"
        )


        status.write(
            "🔗 Combining environmental datasets..."
        )


        ocean = merge_ocean_data(

            temperature_data,
            salinity_data,
            anomaly_data
        )


        status.update(

            label=(
                "✅ Copernicus Marine data loaded"
            ),

            state="complete",

            expanded=False
        )


except Exception as error:

    st.error(
        "❌ E-AWARE could not load Copernicus Marine data."
    )


    st.write(
        "Error details:"
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


if not available_dates:

    st.error(
        "No valid ocean dates were returned."
    )

    st.stop()


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


current_data = calculate_hsi(

    current_data,
    current_date.month
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


reference = habitat_reference(

    current_date.month
)


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    "🌊 Ocean Conditions"
)


metric1, metric2, metric3, metric4 = st.columns(
    4
)


metric1.metric(

    "Ocean Data Date",

    str(
        current_date
    )
)


metric2.metric(

    "Highest HSI",

    f"{best['hsi']:.2f}"
)


metric3.metric(

    "Best Habitat SST",

    f"{best['temperature']:.2f} °C"
)


if pd.notna(
    best[
        "sst_anomaly"
    ]
):

    anomaly_text = (

        f"{best['sst_anomaly']:+.2f} °C"
    )

else:

    anomaly_text = (
        "Unavailable"
    )


metric4.metric(

    "SST Anomaly",

    anomaly_text
)


st.caption(
    "Copernicus supplies the environmental data. "
    "E-AWARE calculates the Habitat Suitability Index."
)


# ============================================================
# MAP
# ============================================================

st.subheader(
    "🗺️ Anchoveta Habitat Map"
)


map_mode = st.radio(

    "Choose Map Layer",

    [
        "Habitat Suitability",
        "Temperature",
        "SST Anomaly",
        "Salinity"
    ],

    horizontal=True
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

        0.01
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
    "SST Anomaly"
):

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


else:

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

        0.001
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


# ============================================================
# OCEAN MAP LAYER
# ============================================================

ocean_layer = pdk.Layer(

    "ScatterplotLayer",

    data=map_data,

    get_position=[
        "lon",
        "lat"
    ],

    get_fill_color=[
        "r",
        "g",
        "b",
        190
    ],

    get_radius=700,

    radius_min_pixels=6,

    radius_max_pixels=18,

    pickable=True
)


# ============================================================
# COMMUNITY MARKERS
# ============================================================

community_layer = pdk.Layer(

    "ScatterplotLayer",

    data=COMMUNITIES,

    get_position=[
        "lon",
        "lat"
    ],

    get_fill_color=[
        255,
        255,
        255,
        240
    ],

    get_line_color=[
        20,
        20,
        20,
        255
    ],

    stroked=True,

    line_width_min_pixels=2,

    get_radius=350,

    pickable=True
)


# ============================================================
# DISPLAY MAP
# ============================================================

st.pydeck_chart(

    pdk.Deck(

        map_style=(
            "https://basemaps.cartocdn.com/"
            "gl/positron-gl-style/style.json"
        ),

        initial_view_state=pdk.ViewState(

            latitude=-4.49,

            longitude=-81.36,

            zoom=9.5,

            pitch=0
        ),

        layers=[
            ocean_layer,
            community_layer
        ],

        tooltip={

            "html":

                "<b>{class} Habitat</b><br/>"

                "HSI: {hsi}<br/>"

                "SST: {temperature} °C<br/>"

                "SST anomaly: {sst_anomaly} °C<br/>"

                "Salinity: {salinity} PSU"
        }
    ),

    use_container_width=True
)


st.caption(
    "Only valid Copernicus ocean cells are included."
)


# ============================================================
# BEST HABITAT
# ============================================================

st.subheader(
    "📍 Best Predicted Habitat Zone"
)


zone1, zone2, zone3 = st.columns(
    3
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


# ============================================================
# HABITAT REFERENCE
# ============================================================

st.subheader(
    "🐟 Why is this habitat considered suitable?"
)


reference1, reference2, reference3 = st.columns(
    3
)


reference1.metric(

    "Temperature Range",

    (
        f"{reference['temp_min']} – "
        f"{reference['temp_max']} °C"
    )
)


reference2.metric(

    "Salinity Range",

    (
        f"{reference['sal_min']} – "
        f"{reference['sal_max']} PSU"
    )
)


reference3.metric(

    "Season",

    reference[
        "season"
    ]
)


st.warning(
    "MVP V1 currently uses temperature and salinity "
    "with equal weighting. It is not yet the final "
    "trained anchoveta prediction model."
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


    forecast_frame = calculate_hsi(

        forecast_frame,
        forecast_date.month
    )


    centre = habitat_centroid(

        forecast_frame
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
                    ]
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

                    else None
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
    "No clear shift"
)


distance_text = (
    "—"
)


period_text = (
    "Insufficient forecast data"
)


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


    distance_text = (
        f"{distance:.1f} km"
    )


    period_text = (

        f"{start['date']}"
        " → "
        f"{end['date']}"
    )


    # Do not present a compass direction
    # if movement is smaller than 2 km.

    if distance >= 2.0:


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


        direction_text = (
            direction_name(
                bearing
            )
        )


shift1, shift2, shift3 = st.columns(
    3
)


shift1.metric(

    "Predicted Habitat Direction",

    direction_text
)


shift2.metric(

    "Habitat Centre Shift",

    distance_text
)


shift3.metric(

    "Forecast Period",

    period_text
)


st.info(
    "This direction describes the forecast movement "
    "of suitable habitat. It does not mean that individual "
    "anchoveta schools are being directly tracked."
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

    use_container_width=True,

    hide_index=True
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

Sea temperature:
{best['temperature']:.1f} °C

Salinity:
{best['salinity']:.2f} PSU

Check official fishing restrictions before departure.

E-AWARE provides habitat guidance only.
"""


st.text_area(

    "Example SMS",

    sms_message,

    height=260
)


st.caption(
    "MVP V1 only previews this message. "
    "Real SMS delivery will be added later."
)


# ============================================================
# DATA TRANSPARENCY
# ============================================================

st.subheader(
    "📊 Data Sources & Transparency"
)


source_table = pd.DataFrame(
    [
        {
            "Source":
                "Copernicus Marine",

            "Status":
                "Analysis / forecast",

            "Used For":
                "Sea temperature"
        },

        {
            "Source":
                "Copernicus Marine",

            "Status":
                "Analysis / forecast",

            "Used For":
                "Salinity"
        },

        {
            "Source":
                "Copernicus Marine",

            "Status":
                "Analysis / forecast",

            "Used For":
                "SST anomaly"
        },

        {
            "Source":
                "Anchoveta habitat research",

            "Status":
                "Historical scientific reference",

            "Used For":
                "Temperature and salinity ranges"
        },

        {
            "Source":
                "IMARPE",

            "Status":
                "Scientific observations",

            "Used For":
                "Future model training and validation"
        },

        {
            "Source":
                "PRODUCE",

            "Status":
                "Regulatory updates",

            "Used For":
                "Future fisheries protection layer"
        }
    ]
)


st.dataframe(

    source_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander(
    "⚙️ MVP V1 Limitations"
):

    st.markdown(
        """
### Current HSI inputs

- Sea temperature
- Salinity

### Environmental context

- SST anomaly

### Not yet included

- Dissolved oxygen
- Oxycline depth
- Bathymetry
- Distance from coast
- Chlorophyll
- Ocean currents
- IMARPE fish observations
- Juvenile protection
- Fisheries closures

### Model status

The current HSI is a transparent engineering prototype.

It is not yet a trained fish-distribution model.

The next development stages will add historical anchoveta
observations and test the model against observations that were
not used during training.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "E-AWARE MVP V1 | "
    "El Niño Adaptive Fisheries Intelligence | "
    "Decision-support research prototype"
)


st.caption(
    "Habitat suitability is a model estimate. "
    "It is not confirmed fish location, catch advice, "
    "or a legal fishing instruction."
)
