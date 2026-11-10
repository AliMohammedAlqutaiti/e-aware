from pathlib import Path
from tempfile import TemporaryDirectory
import json

import copernicusmarine
import joblib
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIG
# ============================================================

MIN_LON, MAX_LON = -82.15, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36

TEST_DATES = [
    "2024-11-10",
    "2026-03-15",
]

TEMP_DATASET = (
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
)

SAL_DATASET = (
    "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
)

BGC_DATASET = (
    "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
)

CLIM_DATASET = (
    "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"
)

GEBCO = Path(
    "data/imarpe/GEBCO_E_AWARE_phase2.nc"
)

MODEL_PATH = Path(
    "models/eaware_phase2_logistic.joblib"
)

REF_PATH = Path(
    "models/eaware_phase2_reference_scores.npy"
)

META_PATH = Path(
    "models/eaware_phase2_model_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================

def download(
    dataset,
    variables,
    date_string,
    filename,
):

    with TemporaryDirectory() as td:

        response = copernicusmarine.subset(
            dataset_id=dataset,
            variables=list(variables),

            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,

            minimum_depth=0,
            maximum_depth=1,

            start_datetime=f"{date_string}T00:00:00",
            end_datetime=f"{date_string}T23:59:59",

            output_directory=td,
            output_filename=filename,
            disable_progress_bar=True,
        )

        path = Path(
            str(response.file_path)
        )

        if not path.exists():
            path = Path(td) / filename

        with xr.open_dataset(path) as ds:
            return ds.load()


def surface_frame(
    ds,
    variable,
    output_name,
):

    da = ds[variable]

    for dim in list(da.dims):

        if dim.lower() in (
            "depth",
            "deptht",
        ):
            da = da.isel(
                {
                    dim: 0
                }
            )

    frame = (
        da.to_dataframe(
            name=output_name
        )
        .reset_index()
    )

    lat_col = next(
        c
        for c in frame.columns
        if c.lower() in (
            "latitude",
            "lat",
        )
    )

    lon_col = next(
        c
        for c in frame.columns
        if c.lower() in (
            "longitude",
            "lon",
        )
    )

    frame["lat"] = pd.to_numeric(
        frame[lat_col],
        errors="coerce",
    )

    frame["lon"] = pd.to_numeric(
        frame[lon_col],
        errors="coerce",
    )

    frame["lat_key"] = (
        frame["lat"].round(3)
    )

    frame["lon_key"] = (
        frame["lon"].round(3)
    )

    return (
        frame[
            [
                "lat",
                "lon",
                "lat_key",
                "lon_key",
                output_name,
            ]
        ]
        .dropna()
        .drop_duplicates(
            [
                "lat_key",
                "lon_key",
            ]
        )
    )


def nearest_values(
    target,
    source,
    source_column,
):

    target_lat = target[
        "lat"
    ].to_numpy(float)

    target_lon = target[
        "lon"
    ].to_numpy(float)

    source_lat = source[
        "lat"
    ].to_numpy(float)

    source_lon = source[
        "lon"
    ].to_numpy(float)

    output = []

    for lat, lon in zip(
        target_lat,
        target_lon,
    ):

        d2 = (
            (source_lat - lat) ** 2
            +
            (source_lon - lon) ** 2
        )

        idx = int(
            np.nanargmin(d2)
        )

        output.append(
            float(
                source.iloc[idx][
                    source_column
                ]
            )
        )

    return np.asarray(
        output,
        dtype=float,
    )


def add_gebco_depth(
    frame,
):

    with xr.open_dataset(
        GEBCO
    ) as ds:

        elevation_name = (
            "elevation"
            if "elevation" in ds.data_vars
            else list(ds.data_vars)[0]
        )

        lat_name = next(
            c
            for c in ds.coords
            if c.lower()
            in (
                "lat",
                "latitude",
            )
        )

        lon_name = next(
            c
            for c in ds.coords
            if c.lower()
            in (
                "lon",
                "longitude",
            )
        )

        elevations = []

        for _, row in frame.iterrows():

            value = (
                ds[elevation_name]
                .sel(
                    {
                        lat_name:
                            float(row["lat"]),

                        lon_name:
                            float(row["lon"]),
                    },
                    method="nearest",
                )
                .item()
            )

            elevations.append(
                float(value)
            )

    elevations = np.asarray(
        elevations,
        dtype=float,
    )

    frame[
        "water_depth_m"
    ] = np.where(
        elevations < 0,
        -elevations,
        np.nan,
    )

    return frame


def hsi_class(value):

    if value >= 0.8:
        return "Very High"

    if value >= 0.6:
        return "High"

    if value >= 0.3:
        return "Moderate"

    return "Low"


# ============================================================
# LOAD FROZEN MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

reference_scores = np.sort(
    np.load(
        REF_PATH,
        allow_pickle=False,
    ).astype(float)
)

metadata = json.loads(
    META_PATH.read_text(
        encoding="utf-8"
    )
)

features = metadata[
    "features"
]

envelope = metadata[
    "training_envelope"
]


print("=" * 105)
print(
    "E-AWARE — CONTROLLED VALIDATED-ML PRIMARY-BRANCH TEST"
)
print("=" * 105)


for date_string in TEST_DATES:

    print(
        "\n"
        + "=" * 105
    )

    print(
        "TEST DATE:",
        date_string
    )

    print(
        "=" * 105
    )


    date = pd.Timestamp(
        date_string
    )


    # --------------------------------------------------------
    # LIVE/ANALYSIS SST
    # --------------------------------------------------------

    temp_ds = download(
        TEMP_DATASET,
        ["thetao"],
        date_string,
        f"temp_{date_string}.nc",
    )

    temp = surface_frame(
        temp_ds,
        "thetao",
        "temperature",
    )


    # --------------------------------------------------------
    # SALINITY
    # --------------------------------------------------------

    sal_ds = download(
        SAL_DATASET,
        ["so"],
        date_string,
        f"sal_{date_string}.nc",
    )

    sal = surface_frame(
        sal_ds,
        "so",
        "salinity",
    )


    # --------------------------------------------------------
    # SURFACE OXYGEN
    # --------------------------------------------------------

    bgc_ds = download(
        BGC_DATASET,
        ["o2"],
        date_string,
        f"o2_{date_string}.nc",
    )

    oxygen = surface_frame(
        bgc_ds,
        "o2",
        "oxygen_mmol_m3",
    )


    # --------------------------------------------------------
    # MONTHLY TRAINING-CONSISTENT CLIMATOLOGY
    # --------------------------------------------------------

    clim_date = (
        f"2004-{date.month:02d}-01"
    )

    clim_ds = download(
        CLIM_DATASET,
        ["thetao"],
        clim_date,
        f"clim_{date.month:02d}.nc",
    )

    climatology = surface_frame(
        clim_ds,
        "thetao",
        "climatology_sst",
    )


    # --------------------------------------------------------
    # MERGE PHYSICS GRID
    # --------------------------------------------------------

    frame = (
        temp.merge(
            sal[
                [
                    "lat_key",
                    "lon_key",
                    "salinity",
                ]
            ],
            on=[
                "lat_key",
                "lon_key",
            ],
            how="inner",
        )
        .merge(
            climatology[
                [
                    "lat_key",
                    "lon_key",
                    "climatology_sst",
                ]
            ],
            on=[
                "lat_key",
                "lon_key",
            ],
            how="inner",
        )
    )


    # Oxygen grid is coarser, so use nearest cell just like
    # the production application does conceptually.
    frame[
        "oxygen_mmol_m3"
    ] = nearest_values(
        frame,
        oxygen,
        "oxygen_mmol_m3",
    )


    frame = add_gebco_depth(
        frame
    )


    # --------------------------------------------------------
    # EXACT PRODUCTION FEATURES
    # --------------------------------------------------------

    frame[
        "deploy_sst_c"
    ] = frame[
        "temperature"
    ]


    frame[
        "deploy_sss_psu"
    ] = frame[
        "salinity"
    ]


    frame[
        "deploy_oxygen_mg_l"
    ] = (
        frame[
            "oxygen_mmol_m3"
        ]
        *
        0.032
    )


    frame[
        "deploy_log_bottom_depth"
    ] = np.log1p(
        frame[
            "water_depth_m"
        ]
    )


    frame[
        "sst_anomaly_c"
    ] = (
        frame[
            "temperature"
        ]
        -
        frame[
            "climatology_sst"
        ]
    )


    angle = (
        2.0
        *
        np.pi
        *
        (
            date.month - 1
        )
        /
        12.0
    )


    frame[
        "month_sin"
    ] = np.sin(
        angle
    )


    frame[
        "month_cos"
    ] = np.cos(
        angle
    )


    complete = (
        frame[
            features
        ]
        .notna()
        .all(axis=1)
    )


    frame = frame[
        complete
    ].copy()


    # --------------------------------------------------------
    # DOMAIN CHECK
    # --------------------------------------------------------

    outside_count = np.zeros(
        len(frame),
        dtype=int,
    )

    outside_summary = []


    for feature in features:

        lower = float(
            envelope[
                feature
            ][
                "min"
            ]
        )

        upper = float(
            envelope[
                feature
            ][
                "max"
            ]
        )

        values = frame[
            feature
        ].to_numpy(float)

        outside = (
            (values < lower)
            |
            (values > upper)
        )

        outside_count += (
            outside.astype(int)
        )

        outside_summary.append(
            {
                "feature":
                    feature,

                "training_min":
                    lower,

                "training_max":
                    upper,

                "live_min":
                    float(
                        np.min(values)
                    ),

                "live_max":
                    float(
                        np.max(values)
                    ),

                "cells_outside":
                    int(
                        outside.sum()
                    ),
            }
        )


    inside = (
        outside_count
        ==
        0
    )


    inside_pct = (
        100.0
        *
        inside.mean()
        if len(inside)
        else 0.0
    )


    # --------------------------------------------------------
    # MODEL SCORE + HISTORICAL HSI
    # --------------------------------------------------------

    X = frame[
        features
    ]


    raw = model.predict_proba(
        X
    )[:, 1]


    hsi = (
        np.searchsorted(
            reference_scores,
            raw,
            side="right",
        )
        /
        len(
            reference_scores
        )
    )


    frame[
        "validated_ml_hsi"
    ] = hsi


    # --------------------------------------------------------
    # PRIMARY-BRANCH DECISION
    # --------------------------------------------------------

    ml_primary_eligible = (
        len(frame) > 0
        and
        bool(
            inside.all()
        )
    )


    if ml_primary_eligible:

        operational_model = (
            "Validated ML HSI"
        )

    else:

        operational_model = (
            "Literature HSI fallback"
        )


    print(
        "\nCells scored:",
        len(frame)
    )

    print(
        "Cells inside ML domain:",
        int(
            inside.sum()
        ),
        "/",
        len(frame)
    )

    print(
        "Inside-domain percentage:",
        f"{inside_pct:.1f}%"
    )


    print(
        "\nOperational model decision:",
        operational_model
    )


    print(
        "\nML HSI range:",
        (
            f"{float(np.min(hsi)):.3f}"
            " → "
            f"{float(np.max(hsi)):.3f}"
        )
    )

    print(
        "Median ML HSI:",
        f"{float(np.median(hsi)):.3f}"
    )


    best_pos = int(
        np.argmax(hsi)
    )

    best = frame.iloc[
        best_pos
    ]


    print(
        "\nBest validated-ML cell:"
    )

    print(
        "lat:",
        round(
            float(best["lat"]),
            4
        )
    )

    print(
        "lon:",
        round(
            float(best["lon"]),
            4
        )
    )

    print(
        "HSI:",
        round(
            float(hsi[best_pos]),
            3
        )
    )

    print(
        "Class:",
        hsi_class(
            float(
                hsi[best_pos]
            )
        )
    )


    print(
        "\nDOMAIN TABLE"
    )


    table = pd.DataFrame(
        outside_summary
    )


    print(
        table
        .round(4)
        .to_string(
            index=False
        )
    )


    if ml_primary_eligible:

        print(
            "\n✅ ML-PRIMARY BRANCH PASSED"
        )

    else:

        print(
            "\n⚠️ This date remains outside the complete ML domain."
        )


print(
    "\n"
    + "=" * 105
)

print(
    "TEST COMPLETE"
)

print(
    "=" * 105
)
