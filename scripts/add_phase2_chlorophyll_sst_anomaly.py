from pathlib import Path
import math
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("data/imarpe")

INPUT = ROOT / "E_AWARE_IMARPE_phase2_copernicus_filled.csv"

OUTPUT = ROOT / "E_AWARE_IMARPE_phase2_enriched.csv"

CACHE = ROOT / "copernicus_cache_phase2"

PHY_DAILY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

PHY_CLIM = (
    "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"
)

BGC_DAILY = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"


COPERNICUS_START = pd.Timestamp("1993-01-01")

MIN_LON = -82.70
MAX_LON = -80.50

MIN_LAT = -5.60
MAX_LAT = -3.60

MAX_PHY_DISTANCE_KM = 30.0
MAX_BGC_DISTANCE_KM = 60.0


def download_subset(
    dataset_id,
    variables,
    date,
    output_file,
):

    if output_file.exists():
        print("Using cached:", output_file.name)
        return

    cmd = [
        "copernicusmarine",
        "subset",
        "--dataset-id",
        dataset_id,
    ]

    for variable in variables:
        cmd.extend(
            [
                "--variable",
                variable,
            ]
        )

    cmd.extend(
        [
            "--start-datetime",
            date,
            "--end-datetime",
            date,
            "--minimum-longitude",
            str(MIN_LON),
            "--maximum-longitude",
            str(MAX_LON),
            "--minimum-latitude",
            str(MIN_LAT),
            "--maximum-latitude",
            str(MAX_LAT),
            "--minimum-depth",
            "0",
            "--maximum-depth",
            "10",
            "--output-directory",
            str(CACHE),
            "--output-filename",
            output_file.name,
        ]
    )

    print(
        "\nDownloading:",
        dataset_id,
        date,
    )

    subprocess.run(
        cmd,
        check=True,
    )


def coordinate_name(
    ds,
    options,
):

    for name in options:

        if name in ds.coords:
            return name

    raise KeyError(
        f"Could not find coordinate from {options}. "
        f"Available: {list(ds.coords)}"
    )


def surface_grid(
    ds,
    variable,
):

    da = ds[variable]

    for name in [
        "time",
        "TIME",
    ]:

        if name in da.dims:

            da = da.isel(
                {
                    name: 0
                }
            )

            break

    for name in [
        "depth",
        "DEPTH",
    ]:

        if name in da.dims:

            da = da.isel(
                {
                    name: 0
                }
            )

            break

    lat_name = coordinate_name(
        ds,
        [
            "latitude",
            "lat",
        ],
    )

    lon_name = coordinate_name(
        ds,
        [
            "longitude",
            "lon",
        ],
    )

    values = (
        np.asarray(
            da.values
        )
        .squeeze()
    )

    lats = np.asarray(
        ds[lat_name].values
    )

    lons = np.asarray(
        ds[lon_name].values
    )

    if values.ndim != 2:

        raise ValueError(
            f"{variable}: expected 2-D surface data, "
            f"got {values.shape}"
        )

    lat_grid, lon_grid = np.meshgrid(
        lats,
        lons,
        indexing="ij",
    )

    valid = np.isfinite(
        values
    )

    return {
        "values": values[valid].astype(float),
        "latitudes": lat_grid[valid].astype(float),
        "longitudes": lon_grid[valid].astype(float),
        "units": da.attrs.get(
            "units",
            "",
        ),
    }


def nearest_from_grid(
    grid,
    target_lat,
    target_lon,
    maximum_distance_km,
):

    lat2 = np.radians(
        grid["latitudes"]
    )

    lon2 = np.radians(
        grid["longitudes"]
    )

    lat1 = math.radians(
        float(target_lat)
    )

    lon1 = math.radians(
        float(target_lon)
    )

    dlat = (
        lat2
        - lat1
    )

    dlon = (
        lon2
        - lon1
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            dlon / 2
        ) ** 2
    )

    distance = (
        2
        * 6371.0088
        * np.arcsin(
            np.sqrt(a)
        )
    )

    if len(distance) == 0:
        return None

    idx = int(
        np.argmin(
            distance
        )
    )

    best_distance = float(
        distance[idx]
    )

    if (
        best_distance
        > maximum_distance_km
    ):

        return None

    return {
        "value": float(
            grid["values"][idx]
        ),
        "distance_km": best_distance,
        "latitude": float(
            grid["latitudes"][idx]
        ),
        "longitude": float(
            grid["longitudes"][idx]
        ),
    }


def chlorophyll_to_mg_m3(
    value,
    units,
):

    unit = (
        str(units)
        .lower()
        .replace(" ", "")
        .replace("^", "")
    )

    # Normal Copernicus chlorophyll unit.
    if "mg" in unit:

        return float(
            value
        )

    # Defensive conversion if a future dataset
    # supplies kg/m3.
    if "kg" in unit:

        return float(
            value
        ) * 1_000_000.0

    raise ValueError(
        "Unrecognised chlorophyll units: "
        + repr(units)
    )


def main():

    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    eligible = (
        dates.notna()
        & (
            dates
            >= COPERNICUS_START
        )
    )

    eligible_dates = sorted(
        dates.loc[
            eligible
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .unique()
    )

    print("=" * 90)
    print(
        "E-AWARE PHASE 2 — "
        "CHLOROPHYLL + SST ANOMALY"
    )
    print("=" * 90)

    print(
        "\nEligible records:",
        int(
            eligible.sum()
        ),
    )

    print(
        "Unique eligible dates:",
        len(
            eligible_dates
        ),
    )


    # ------------------------------------------------------
    # Download daily SST for every survey date.
    # Existing physics cache files are reused.
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("DAILY SST FILES")
    print("=" * 90)

    for date in eligible_dates:

        file = (
            CACHE
            / f"phy_{date}.nc"
        )

        download_subset(
            PHY_DAILY,
            [
                "thetao",
            ],
            date,
            file,
        )


    # ------------------------------------------------------
    # Download daily chlorophyll.
    # Separate filename because existing BGC files contain O2.
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("DAILY CHLOROPHYLL FILES")
    print("=" * 90)

    for date in eligible_dates:

        file = (
            CACHE
            / f"chl_{date}.nc"
        )

        download_subset(
            BGC_DAILY,
            [
                "chl",
            ],
            date,
            file,
        )


    # ------------------------------------------------------
    # Download 12 monthly SST climatology fields.
    #
    # Copernicus encodes this month-of-year climatology
    # using the 12 months of 2004.
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("MONTHLY SST CLIMATOLOGY")
    print("=" * 90)

    for month in range(
        1,
        13,
    ):

        climatology_date = (
            f"2004-{month:02d}-01"
        )

        file = (
            CACHE
            / f"phy_climatology_{month:02d}.nc"
        )

        download_subset(
            PHY_CLIM,
            [
                "thetao",
            ],
            climatology_date,
            file,
        )


    # ------------------------------------------------------
    # Add output columns.
    # ------------------------------------------------------

    columns = [
        "copernicus_daily_sst_for_anomaly_c",
        "copernicus_daily_sst_anomaly_distance_km",
        "sst_climatology_c",
        "sst_climatology_distance_km",
        "sst_anomaly_c",
        "cold_anomaly_strength_c",

        "chlorophyll_mg_m3",
        "chlorophyll_distance_km",
        "chlorophyll_grid_lat",
        "chlorophyll_grid_lon",
    ]

    for col in columns:

        if col not in df.columns:

            df[col] = np.nan

    df["sst_anomaly_source"] = None
    df["chlorophyll_source"] = None


    # ------------------------------------------------------
    # Load 12 climatology grids only once.
    # ------------------------------------------------------

    climatology = {}

    for month in range(
        1,
        13,
    ):

        path = (
            CACHE
            / f"phy_climatology_{month:02d}.nc"
        )

        with xr.open_dataset(
            path
        ) as ds:

            climatology[
                month
            ] = surface_grid(
                ds,
                "thetao",
            )


    # ------------------------------------------------------
    # Process one survey date at a time.
    # This is much faster than reopening datasets for
    # every individual acoustic record.
    # ------------------------------------------------------

    chlorophyll_units_seen = set()

    for date_string in eligible_dates:

        print(
            "\nExtracting features:",
            date_string,
        )

        date_timestamp = pd.Timestamp(
            date_string
        )

        row_indices = df.index[
            eligible
            & (
                dates.dt.normalize()
                == date_timestamp
            )
        ]

        phy_path = (
            CACHE
            / f"phy_{date_string}.nc"
        )

        chl_path = (
            CACHE
            / f"chl_{date_string}.nc"
        )

        with xr.open_dataset(
            phy_path
        ) as phy_ds:

            daily_sst_grid = (
                surface_grid(
                    phy_ds,
                    "thetao",
                )
            )

        with xr.open_dataset(
            chl_path
        ) as chl_ds:

            chl_grid = (
                surface_grid(
                    chl_ds,
                    "chl",
                )
            )

        chlorophyll_units_seen.add(
            chl_grid["units"]
        )

        month = (
            date_timestamp.month
        )

        climate_grid = (
            climatology[
                month
            ]
        )

        for idx in row_indices:

            lat = float(
                df.at[
                    idx,
                    "lat",
                ]
            )

            lon = float(
                df.at[
                    idx,
                    "lon",
                ]
            )


            # ----------------------------------------------
            # Daily Copernicus SST
            # ----------------------------------------------

            daily_sst = (
                nearest_from_grid(
                    daily_sst_grid,
                    lat,
                    lon,
                    MAX_PHY_DISTANCE_KM,
                )
            )


            # ----------------------------------------------
            # Monthly Copernicus climatological SST
            # ----------------------------------------------

            climate_sst = (
                nearest_from_grid(
                    climate_grid,
                    lat,
                    lon,
                    MAX_PHY_DISTANCE_KM,
                )
            )


            if (
                daily_sst is not None
                and climate_sst is not None
            ):

                anomaly = (
                    daily_sst["value"]
                    - climate_sst["value"]
                )

                df.at[
                    idx,
                    "copernicus_daily_sst_for_anomaly_c",
                ] = daily_sst[
                    "value"
                ]

                df.at[
                    idx,
                    "copernicus_daily_sst_anomaly_distance_km",
                ] = daily_sst[
                    "distance_km"
                ]

                df.at[
                    idx,
                    "sst_climatology_c",
                ] = climate_sst[
                    "value"
                ]

                df.at[
                    idx,
                    "sst_climatology_distance_km",
                ] = climate_sst[
                    "distance_km"
                ]

                df.at[
                    idx,
                    "sst_anomaly_c",
                ] = anomaly

                # Positive number only when the sea
                # is colder than climatology.
                df.at[
                    idx,
                    "cold_anomaly_strength_c",
                ] = max(
                    0.0,
                    -anomaly,
                )

                df.at[
                    idx,
                    "sst_anomaly_source",
                ] = (
                    "Copernicus GLORYS12V1 "
                    "daily minus monthly climatology"
                )


            # ----------------------------------------------
            # Chlorophyll-a
            # ----------------------------------------------

            chl = nearest_from_grid(
                chl_grid,
                lat,
                lon,
                MAX_BGC_DISTANCE_KM,
            )

            if chl is not None:

                chlorophyll_mg_m3 = (
                    chlorophyll_to_mg_m3(
                        chl["value"],
                        chl_grid["units"],
                    )
                )

                df.at[
                    idx,
                    "chlorophyll_mg_m3",
                ] = chlorophyll_mg_m3

                df.at[
                    idx,
                    "chlorophyll_distance_km",
                ] = chl[
                    "distance_km"
                ]

                df.at[
                    idx,
                    "chlorophyll_grid_lat",
                ] = chl[
                    "latitude"
                ]

                df.at[
                    idx,
                    "chlorophyll_grid_lon",
                ] = chl[
                    "longitude"
                ]

                df.at[
                    idx,
                    "chlorophyll_source",
                ] = (
                    "Copernicus "
                    "GLOBAL_MULTIYEAR_BGC"
                )


    # ------------------------------------------------------
    # Save.
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False,
    )


    # ------------------------------------------------------
    # Audit.
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("FEATURE COMPLETENESS")
    print("=" * 90)

    print(
        "\nChlorophyll units encountered:",
        chlorophyll_units_seen,
    )

    for col in [
        "sst_anomaly_c",
        "cold_anomaly_strength_c",
        "chlorophyll_mg_m3",
    ]:

        available = (
            eligible
            & df[
                col
            ].notna()
        )

        print(
            f"\n{col}:",
            int(
                available.sum()
            ),
            "/",
            int(
                eligible.sum()
            ),
            f"({available.sum()/eligible.sum()*100:.1f}%)",
        )

        if available.any():

            print(
                df.loc[
                    available,
                    col,
                ]
                .describe(
                    percentiles=[
                        0.01,
                        0.05,
                        0.25,
                        0.50,
                        0.75,
                        0.95,
                        0.99,
                    ]
                )
                .round(4)
                .to_string()
            )


    complete_extended = (
        eligible
        & df[
            [
                "sst_c",
                "sss_psu",
                "oxygen_mg_l",
                "bottom_depth_m",
                "sst_anomaly_c",
                "chlorophyll_mg_m3",
            ]
        ]
        .notna()
        .all(
            axis=1
        )
    )

    print("\n" + "=" * 90)
    print("EXTENDED FEATURE COMPLETENESS")
    print("=" * 90)

    print(
        "Core + anomaly + chlorophyll:",
        int(
            complete_extended.sum()
        ),
        "/",
        int(
            eligible.sum()
        ),
        f"({complete_extended.sum()/eligible.sum()*100:.1f}%)",
    )


    print("\n" + "=" * 90)
    print("PRESENCE VS ABSENCE")
    print("=" * 90)

    analysis = df.loc[
        complete_extended
    ]

    print(
        analysis.groupby(
            "presence"
        )[
            [
                "sst_anomaly_c",
                "cold_anomaly_strength_c",
                "chlorophyll_mg_m3",
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
            ]
        )
        .round(
            4
        )
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
