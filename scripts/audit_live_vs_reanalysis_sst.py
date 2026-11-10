from pathlib import Path
from tempfile import TemporaryDirectory

import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr


MIN_LON, MAX_LON = -82.15, -81.20
MIN_LAT, MAX_LAT = -4.60, -4.36

LIVE = (
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
)

REANALYSIS = (
    "cmems_mod_glo_phy_my_0.083deg_P1D-m"
)

CLIMATOLOGY = (
    "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"
)


# Dates where BOTH products exist.
TEST_DATES = [
    "2024-11-10",
    "2025-03-15",
    "2025-11-10",
    "2026-03-15",
    "2026-09-15",
]


def download(
    dataset,
    variable,
    date_string,
    filename,
):

    with TemporaryDirectory() as td:

        response = copernicusmarine.subset(
            dataset_id=dataset,
            variables=[variable],

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
            path = (
                Path(td)
                / filename
            )

        with xr.open_dataset(path) as ds:
            return ds.load()


def surface_dataframe(
    ds,
    variable,
    output_name,
):

    da = ds[
        variable
    ]

    for dim in list(
        da.dims
    ):

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
        x
        for x in frame.columns
        if x.lower()
        in (
            "latitude",
            "lat",
        )
    )

    lon_col = next(
        x
        for x in frame.columns
        if x.lower()
        in (
            "longitude",
            "lon",
        )
    )


    frame[
        "lat_key"
    ] = (
        pd.to_numeric(
            frame[
                lat_col
            ]
        )
        .round(3)
    )

    frame[
        "lon_key"
    ] = (
        pd.to_numeric(
            frame[
                lon_col
            ]
        )
        .round(3)
    )


    return (
        frame[
            [
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


def load_climatology(
    month,
):

    date_string = (
        f"2004-{month:02d}-01"
    )


    ds = download(
        CLIMATOLOGY,
        "thetao",
        date_string,
        f"clim_{month:02d}.nc",
    )


    return surface_dataframe(
        ds,
        "thetao",
        "climatology_sst",
    )


rows = []


print("=" * 100)
print(
    "E-AWARE — LIVE ANALYSIS/FORECAST vs MULTIYEAR REANALYSIS SST"
)
print("=" * 100)


climatology_cache = {}


for date_string in TEST_DATES:

    date = pd.Timestamp(
        date_string
    )


    month = int(
        date.month
    )


    print(
        "\nChecking:",
        date_string
    )


    live_ds = download(
        LIVE,
        "thetao",
        date_string,
        f"live_{date_string}.nc",
    )


    reanalysis_ds = download(
        REANALYSIS,
        "thetao",
        date_string,
        f"reanalysis_{date_string}.nc",
    )


    live = surface_dataframe(
        live_ds,
        "thetao",
        "live_sst",
    )


    reanalysis = surface_dataframe(
        reanalysis_ds,
        "thetao",
        "reanalysis_sst",
    )


    if month not in climatology_cache:

        climatology_cache[
            month
        ] = load_climatology(
            month
        )


    climatology = (
        climatology_cache[
            month
        ]
    )


    merged = (
        live.merge(
            reanalysis,
            on=[
                "lat_key",
                "lon_key",
            ],
            how="inner",
        )
        .merge(
            climatology,
            on=[
                "lat_key",
                "lon_key",
            ],
            how="inner",
        )
    )


    if merged.empty:

        print(
            "No matching cells."
        )

        continue


    merged[
        "live_minus_reanalysis"
    ] = (
        merged[
            "live_sst"
        ]
        -
        merged[
            "reanalysis_sst"
        ]
    )


    merged[
        "live_anomaly"
    ] = (
        merged[
            "live_sst"
        ]
        -
        merged[
            "climatology_sst"
        ]
    )


    merged[
        "reanalysis_anomaly"
    ] = (
        merged[
            "reanalysis_sst"
        ]
        -
        merged[
            "climatology_sst"
        ]
    )


    rows.append(
        {
            "date":
                date_string,

            "cells":
                len(
                    merged
                ),

            "live_sst_median":
                merged[
                    "live_sst"
                ].median(),

            "reanalysis_sst_median":
                merged[
                    "reanalysis_sst"
                ].median(),

            "median_source_difference_c":
                merged[
                    "live_minus_reanalysis"
                ].median(),

            "p95_abs_source_difference_c":
                merged[
                    "live_minus_reanalysis"
                ]
                .abs()
                .quantile(
                    0.95
                ),

            "climatology_median":
                merged[
                    "climatology_sst"
                ].median(),

            "live_anomaly_median":
                merged[
                    "live_anomaly"
                ].median(),

            "reanalysis_anomaly_median":
                merged[
                    "reanalysis_anomaly"
                ].median(),
        }
    )


result = pd.DataFrame(
    rows
)


print("\n" + "=" * 100)
print("SOURCE SHIFT SUMMARY")
print("=" * 100)


print(
    result
    .round(4)
    .to_string(
        index=False
    )
)


if not result.empty:

    print(
        "\nOverall median live − reanalysis SST difference:",
        round(
            result[
                "median_source_difference_c"
            ].median(),
            4
        ),
        "°C"
    )

    print(
        "Largest median source difference:",
        round(
            result[
                "median_source_difference_c"
            ].abs().max(),
            4
        ),
        "°C"
    )


output = Path(
    "data/imarpe/phase2_live_vs_reanalysis_sst_audit.csv"
)


result.to_csv(
    output,
    index=False,
)


print("\nSaved:")
print(output)
