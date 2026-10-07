import os
import math
from datetime import date, timedelta

import copernicusmarine
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# PAGE
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
    "of fish schools."
)


# ============================================================
# STUDY AREA
# Costanera corridor / Lobitos marine area
# ============================================================

MIN_LON = -81.50
MAX_LON = -81.20

MIN_LAT = -4.60
MAX_LAT = -4.36


COMMUNITIES = pd.DataFrame(
    [
        {
            "name": "Lobitos",
            "lat": -4.45279,
            "lon": -81.27771
        },
        {
            "name": "Siches",
            "lat": -4.48907,
            "lon": -81.26827
        },
        {
            "name": "Piedritas",
            "lat": -4.51905,
            "lon": -81.26336
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
            "Copernicus credentials are missing. "
            "Add COPERNICUS_USERNAME and "
            "COPERNICUS_PASSWORD in Streamlit Secrets."
        )


    os.environ[
        "COPERNICUSMARINE_SERVICE_USERNAME"
    ] = username


    os.environ[
        "COPERNICUSMARINE_SERVICE_PASSWORD"
    ] = password


# ============================================================
# CONVERT COPERNICUS FIELD TO DATAFRAME
# ============================================================

def field_to_dataframe(
    ds,
    variable,
    output_name
):

    da = ds[
        variable
    ]


    # Surface layer only
    if "depth" in da.dims:

        da = da.isel(
            depth=0
        )


    da = da.load()


    df = (
        da
        .to_dataframe(
            name=output_name
        )
        .reset_index()
    )


    rename_map = {}


    for column in df.columns:

        lower = str(
            column
        ).lower()


        if lower in (
            "latitude",
            "lat"
        ):

            rename_map[
                column
            ] = "lat"


        elif lower in (
            "longitude",
            "lon"
        ):

            rename_map[
                column
            ] = "lon"


    df = df.rename(
        columns=rename_map
    )


    if (
        "lat" not in df.columns
        or
        "lon" not in df.columns
    ):

        raise RuntimeError(
            f"Could not find latitude/longitude for {output_name}."
        )


    if "time" not in df.columns:

        raise RuntimeError(
            f"Could not find time coordinate for {output_name}."
        )


    df = df.dropna(
        subset=[
            output_name
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
            output_name
        ]
    ]


# ============================================================
# GET COPERNICUS DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def get_ocean_data(
    start_date,
    days_ahead=3
):

    configure_copernicus()


    end_date = (
        start_date
        +
        timedelta(
            days=days_ahead
        )
    )


    common = dict(

        minimum_longitude=MIN_LON,

        maximum_longitude=MAX_LON,

        minimum_latitude=MIN_LAT,

        maximum_latitude=MAX_LAT,

        start_datetime=(
            f"{start_date}T00:00:00"
        ),

        end_datetime=(
            f"{end_date}T23:59:59"
        )
    )


    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature_ds = (
        copernicusmarine.open_dataset(

            dataset_id=(
                TEMPERATURE_DATASET
            ),

            variables=[
                "thetao"
            ],

            minimum_depth=0,

            maximum_depth=1,

            **common
        )
    )


    temperature_df = (
        field_to_dataframe(

            temperature_ds,

            "thetao",

            "temperature"
        )
    )


    # ========================================================
    # SALINITY
    # ========================================================

    salinity_ds = (
        copernicusmarine.open_dataset(

            dataset_id=(
                SALINITY_DATASET
            ),

            variables=[
                "so"
            ],

            minimum_depth=0,

            maximum_depth=1,

            **common
        )
    )


    salinity_df = (
        field_to_dataframe(

            salinity_ds,

            "so",

            "salinity"
        )
    )


    # ========================================================
    # SST ANOMALY
    # ========================================================

    anomaly_ds = (
        copernicusmarine.open_dataset(

            dataset_id=(
                SST_ANOMALY_DATASET
            ),

            **common
        )
    )


    anomaly_candidates = [

        name

        for name
        in anomaly_ds.data_vars

        if "anomal"
        in name.lower()
    ]


    if not anomaly_candidates:

        anomaly_candidates = list(
            anomaly_ds.data_vars
        )


    if not anomaly_candidates:

        raise RuntimeError(
            "No SST anomaly variable "
            "was returned by Copernicus."
        )


    anomaly_variable = (
        anomaly_candidates[
            0
        ]
    )


    anomaly_df = (
        field_to_dataframe(

            anomaly_ds,

            anomaly_variable,

            "sst_anomaly"
        )
    )


    # ========================================================
    # MERGE DATA
    # ========================================================

    keys = [
        "date",
        "lat_key",
        "lon_key"
    ]


    merged = (

        temperature_df

        .merge(

            salinity_df[
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

            anomaly_df[
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
            "Copernicus returned "
            "no overlapping ocean cells."
        )


    return merged


# ============================================================
# ANCHOVETA HABITAT REFERENCE
# ============================================================

def habitat_reference(
    month
):

    # Southern Hemisphere summer

    if month in (
        12,
        1,
        2
    ):

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

    if month in (
        6,
        7,
        8
    ):

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


    # Spring / Autumn
    # This is deliberately labelled
    # as an MVP engineering approximation.

    return {

        "season":
            "Transition "
            "(MVP approximation)",

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
# HABITAT SUITABILITY INDEX
# ============================================================

def calculate_hsi(
    df,
    month
):

    result = (
        df.copy()
    )


    reference = (
        habitat_reference(
            month
        )
    )


    result[
        "temperature_score"
    ] = range_score(

        result[
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


    result[
        "salinity_score"
    ] = range_score(

        result[
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
    # MVP V1
    #
    # Temperature = 50%
    # Salinity = 50%
    #
    # These are temporary engineering weights.
    # Final weights will come from the trained model.
    # ========================================================

    result[
        "hsi"
    ] = (

        0.50
        *
        result[
            "temperature_score"
        ]

        +

        0.50
        *
        result[
            "salinity_score"
        ]

    ).clip(
        0,
        1
    )


    return result


# ============================================================
# HSI CLASS
# ============================================================

def hsi_class(
    value
):

    if value >= 0.80:

        return (
            "Very High"
        )


    if value >= 0.60:

        return (
            "High"
        )


    if value >= 0.35:

        return (
            "Moderate"
        )


    return (
        "Low"
    )


# ============================================================
# CENTRE OF HIGH-SUITABILITY HABITAT
# ============================================================

def habitat_centroid(
    df
):

    if df.empty:

        return None


    threshold = max(

        0.60,

        float(
            df[
                "hsi"
            ]
            .quantile(
                0.75
            )
        )
    )


    high_habitat = df[

        df[
            "hsi"
        ]
        >= threshold

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

    earth_radius_km = (
        6371.0
    )


    p1 = math.radians(
        lat1
    )


    p2 = math.radians(
        lat2
    )


    dlat = math.radians(
        lat2
        -
        lat1
    )


    dlon = math.radians(
        lon2
        -
        lon1
    )


    a = (

        math.sin(
            dlat
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
            dlon
            /
            2
        )
        ** 2
    )


    return (

        2

        *
        earth_radius_km

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

    p1 = math.radians(
        lat1
    )


    p2 = math.radians(
        lat2
    )


    dlon = math.radians(
        lon2
        -
        lon1
    )


    x = (

        math.sin(
            dlon
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
            dlon
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


    st.write(
        "Study area:"
    )


    st.write(
        "Lobitos • Siches • Piedritas"
    )


    selected_date = (
        st.date_input(

            "Ocean date",

            value=date.today()
        )
    )


    st.caption(
        "The app requests the selected "
        "day plus up to 72 hours."
    )


    if st.button(
        "🔄 Refresh data",
        use_container_width=True
    ):

        st.cache_data.clear()


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading Copernicus Marine "
        "analysis/forecast data..."
    ):

        ocean = get_ocean_data(

            selected_date,

            days_ahead=3
        )


except Exception as error:

    st.error(
        "Could not load Copernicus data."
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
        "No ocean data was returned "
        "for this request."
    )

    st.stop()


# ============================================================
# CURRENT DAY
# ============================================================

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


# ============================================================
# TOP METRICS
# ============================================================

st.subheader(
    "Current / forecast ocean conditions"
)


m1, m2, m3, m4 = st.columns(
    4
)


m1.metric(

    "Data date",

    str(
        current_date
    )
)


m2.metric(

    "Highest HSI",

    f"{best['hsi']:.2f}"
)


m3.metric(

    "Highest-cell SST",

    f"{best['temperature']:.1f} °C"
)


m4.metric(

    "SST anomaly",

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


st.caption(
    "Ocean inputs come from Copernicus Marine. "
    "HSI is an E-AWARE MVP estimate."
)


# ============================================================
# HABITAT MAP
# ============================================================

st.subheader(
    "🗺 Anchoveta habitat map"
)


map_mode = st.radio(

    "Choose map layer",

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
# COLOURS
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

    ).clip(
        20,
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

    ).clip(
        30,
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


    normalized = (

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
        normalized
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
        normalized
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


    normalized = (

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
        normalized
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
        normalized
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


    normalized = (

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
        normalized
    )


    map_data[
        "b"
    ] = 220


# ============================================================
# OCEAN LAYER
# ============================================================

ocean_layer = pdk.Layer(

    "ScatterplotLayer",

    data=map_data,

    get_position="[lon, lat]",

    get_fill_color="[r, g, b, 190]",

    get_radius=700,

    radius_min_pixels=5,

    radius_max_pixels=16,

    pickable=True
)


# ============================================================
# COMMUNITY LAYER
# ============================================================

community_layer = pdk.Layer(

    "ScatterplotLayer",

    data=COMMUNITIES,

    get_position="[lon, lat]",

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

        initial_view_state=(
            pdk.ViewState(

                latitude=-4.49,

                longitude=-81.36,

                zoom=9.5,

                pitch=0
            )
        ),

        layers=[

            ocean_layer,

            community_layer
        ],

        tooltip={

            "html":

                "<b>{class} habitat</b><br/>"

                "HSI: {hsi}<br/>"

                "SST: {temperature} °C<br/>"

                "SST anomaly: {sst_anomaly} °C<br/>"

                "Salinity: {salinity} PSU"
        }
    ),

    use_container_width=True
)


st.caption(
    "Missing/land cells are excluded because "
    "Copernicus returns no valid ocean values there."
)


# ============================================================
# HABITAT CONDITIONS
# ============================================================

st.subheader(
    "Why is this habitat considered suitable?"
)


r1, r2, r3 = st.columns(
    3
)


r1.metric(

    "Reference temperature",

    (
        f"{reference['temp_min']}"
        "–"
        f"{reference['temp_max']} °C"
    )
)


r2.metric(

    "Reference salinity",

    (
        f"{reference['sal_min']}"
        "–"
        f"{reference['sal_max']} PSU"
    )
)


r3.metric(

    "Season",

    reference[
        "season"
    ]
)


st.warning(
    "V1 is a transparent reference-rule MVP. "
    "Temperature and salinity currently have equal weight. "
    "It is not yet the trained IMARPE anchoveta model."
)


# ============================================================
# FORECAST CALCULATIONS
# ============================================================

forecast_centres = []

forecast_rows = []


for forecast_date in available_dates:


    frame = ocean[

        ocean[
            "date"
        ]

        ==
        forecast_date

    ].copy()


    frame = calculate_hsi(

        frame,

        forecast_date.month
    )


    centre = habitat_centroid(
        frame
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


    best_day = frame.loc[

        frame[
            "hsi"
        ]

        .idxmax()
    ]


    forecast_rows.append(

        {
            "Date":
                forecast_date,

            "Highest HSI":
                round(

                    float(
                        best_day[
                            "hsi"
                        ]
                    ),

                    2
                ),

            "SST °C":
                round(

                    float(
                        best_day[
                            "temperature"
                        ]
                    ),

                    2
                ),

            "Salinity PSU":
                round(

                    float(
                        best_day[
                            "salinity"
                        ]
                    ),

                    2
                ),

            "SST anomaly °C":

                (
                    round(

                        float(
                            best_day[
                                "sst_anomaly"
                            ]
                        ),

                        2
                    )

                    if pd.notna(

                        best_day[
                            "sst_anomaly"
                        ]
                    )

                    else
                    None
                )
        }
    )


# ============================================================
# HABITAT SHIFT
# ============================================================

st.subheader(
    "🧭 Predicted habitat shift"
)


direction_text = (
    "No clear shift"
)


distance_text = (
    "—"
)


period_text = (
    "Not enough forecast data"
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


    distance_km = haversine(

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


    # Ignore tiny shifts below 2 km
    # because the model grid is coarse.

    if distance_km >= 2.0:


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


        distance_text = (
            f"{distance_km:.1f} km"
        )


    else:

        direction_text = (
            "No clear shift"
        )


        distance_text = (
            f"{distance_km:.1f} km"
        )


    period_text = (

        f"{start['date']}"
        " → "
        f"{end['date']}"
    )


f1, f2, f3 = st.columns(
    3
)


f1.metric(

    "Predicted habitat direction",

    direction_text
)


f2.metric(

    "Habitat-centre shift",

    distance_text
)


f3.metric(

    "Forecast period",

    period_text
)


st.info(
    "The direction describes where the high-suitability "
    "habitat is forecast to shift. "
    "It does not mean individual anchoveta schools are "
    "confirmed to be swimming in that direction."
)


# ============================================================
# 72 HOUR FORECAST TABLE
# ============================================================

st.subheader(
    "72-hour habitat forecast"
)


forecast_table = pd.DataFrame(
    forecast_rows
)


st.dataframe(

    forecast_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# SMS PREVIEW
# ============================================================

st.subheader(
    "📱 Fisher SMS preview"
)


best_hsi = float(
    best[
        "hsi"
    ]
)


habitat_text = (
    hsi_class(
        best_hsi
    )
    .upper()
)


sms_message = f"""
E-AWARE Fisheries Alert

Anchoveta habitat suitability: {habitat_text}

Habitat Suitability Index: {best_hsi:.2f}

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

    "Example SMS sent to fishers",

    sms_message,

    height=230
)


st.caption(
    "V1 only previews the message. "
    "It does not yet send a real SMS."
)


# ============================================================
# SOURCE TRANSPARENCY
# ============================================================

st.subheader(
    "Scientific sources & transparency"
)


source_table = pd.DataFrame(
    [
        {
            "Source":
                "Copernicus Marine",

            "Status":
                "Daily analysis / forecast",

            "Use in V1":
                "Temperature, salinity and SST anomaly"
        },

        {
            "Source":
                "Anchoveta habitat study",

            "Status":
                "Historical scientific reference",

            "Use in V1":
                "Seasonal temperature and salinity ranges"
        },

        {
            "Source":
                "IMARPE SIOFEN",

            "Status":
                "Latest scientific observation",

            "Use in V1":
                "Reference now; training/validation later"
        },

        {
            "Source":
                "IMARPE Daily Oceanographic Bulletin",

            "Status":
                "Daily / latest",

            "Use in V1":
                "Supporting Peru ocean context"
        },

        {
            "Source":
                "PRODUCE",

            "Status":
                "Regulatory updates",

            "Use in V1":
                "Future closure/protection logic"
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
    "MVP V1 limitations"
):

    st.write(
        """
Current HSI variables:

- Sea temperature
- Salinity

Displayed but not weighted:

- SST anomaly

Planned next integrations:

- Dissolved oxygen / oxycline
- Bathymetry
- Distance from coast
- Historical anchoveta observations
- Model training and independent validation
- Regulatory / juvenile-protection logic

The transition-season habitat envelope is an MVP engineering
approximation and must be replaced by a validated seasonal model.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "E-AWARE MVP V1 | "
    "Decision-support research prototype | "
    "Not confirmed fish location and not a legal fishing instruction."
)
