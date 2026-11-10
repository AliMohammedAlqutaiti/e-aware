from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_deployment_consistent.csv"
)

OUTPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_deployment_relative.csv"
)

CACHE = ROOT / "copernicus_cache_phase2"


# Actual Phase 2 E-AWARE prediction corridor.
MIN_LAT = -5.00
MAX_LAT = -4.20

MIN_LON = -82.00
MAX_LON = -81.10


def coordinate_name(ds, options):

    for name in options:

        if name in ds.coords:
            return name

    raise KeyError(
        f"Could not find coordinate from {options}. "
        f"Available: {list(ds.coords)}"
    )


def regional_surface_values(
    path,
    variable,
):

    with xr.open_dataset(path) as ds:

        da = ds[variable]

        for dim in [
            "time",
            "TIME",
        ]:

            if dim in da.dims:

                da = da.isel(
                    {
                        dim: 0
                    }
                )

                break

        for dim in [
            "depth",
            "DEPTH",
        ]:

            if dim in da.dims:

                da = da.isel(
                    {
                        dim: 0
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
            f"{variable}: expected 2-D field, "
            f"got {values.shape}"
        )

    lat_grid, lon_grid = np.meshgrid(
        lats,
        lons,
        indexing="ij",
    )

    mask = (
        np.isfinite(values)
        & (lat_grid >= MIN_LAT)
        & (lat_grid <= MAX_LAT)
        & (lon_grid >= MIN_LON)
        & (lon_grid <= MAX_LON)
    )

    return values[mask].astype(float)


def percentile_rank(
    regional_values,
    value,
):

    if (
        len(regional_values) == 0
        or not np.isfinite(value)
    ):

        return np.nan

    return float(
        np.mean(
            regional_values
            <= value
        )
    )


def main():

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )

    eligible = (
        df["date"].notna()
        & (
            df["date"]
            >= pd.Timestamp("1993-01-01")
        )
    )

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    unique_dates = sorted(
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
        "DEPLOYABLE SPATIAL-RELATIVE FEATURES"
    )
    print("=" * 90)

    print(
        "\nEligible records:",
        int(
            eligible.sum()
        )
    )

    print(
        "Unique dates:",
        len(
            unique_dates
        )
    )


    new_columns = [
        "daily_region_sst_median_c",
        "sst_spatial_anomaly_c",
        "cold_spatial_strength_c",
        "sst_spatial_percentile",

        "daily_region_chlorophyll_median_mg_m3",
        "chlorophyll_spatial_anomaly_mg_m3",
        "chlorophyll_spatial_ratio",
        "chlorophyll_spatial_percentile",
        "log_chlorophyll_spatial_anomaly",
    ]


    for col in new_columns:
        df[col] = np.nan


    for date in unique_dates:

        print(
            "Processing:",
            date
        )

        date_ts = pd.Timestamp(
            date
        )

        indices = df.index[
            eligible
            & (
                dates.dt.normalize()
                == date_ts
            )
        ]


        # --------------------------------------------------
        # Daily regional SST distribution.
        # --------------------------------------------------

        phy_path = (
            CACHE
            / f"phy_{date}.nc"
        )

        regional_sst = (
            regional_surface_values(
                phy_path,
                "thetao",
            )
        )


        # --------------------------------------------------
        # Daily regional chlorophyll distribution.
        # --------------------------------------------------

        chl_path = (
            CACHE
            / f"chl_{date}.nc"
        )

        regional_chl = (
            regional_surface_values(
                chl_path,
                "chl",
            )
        )


        if len(regional_sst) == 0:

            raise ValueError(
                f"No regional SST cells for {date}"
            )


        if len(regional_chl) == 0:

            raise ValueError(
                f"No regional chlorophyll cells for {date}"
            )


        sst_median = float(
            np.median(
                regional_sst
            )
        )


        chl_median = float(
            np.median(
                regional_chl
            )
        )


        regional_log_chl = np.log1p(
            regional_chl
        )

        log_chl_median = float(
            np.median(
                regional_log_chl
            )
        )


        # --------------------------------------------------
        # Convert each acoustic observation into a
        # same-day regional-relative feature.
        # --------------------------------------------------

        for idx in indices:

            sst = float(
                df.at[
                    idx,
                    "deploy_sst_c",
                ]
            )

            chl = float(
                df.at[
                    idx,
                    "deploy_chlorophyll_mg_m3",
                ]
            )


            # ---------------- SST ----------------

            sst_relative = (
                sst
                - sst_median
            )

            df.at[
                idx,
                "daily_region_sst_median_c",
            ] = sst_median

            df.at[
                idx,
                "sst_spatial_anomaly_c",
            ] = sst_relative

            # Positive only if the location is colder
            # than the surrounding regional median.
            df.at[
                idx,
                "cold_spatial_strength_c",
            ] = max(
                0.0,
                -sst_relative,
            )

            df.at[
                idx,
                "sst_spatial_percentile",
            ] = percentile_rank(
                regional_sst,
                sst,
            )


            # ---------------- CHLOROPHYLL ----------------

            df.at[
                idx,
                "daily_region_chlorophyll_median_mg_m3",
            ] = chl_median

            df.at[
                idx,
                "chlorophyll_spatial_anomaly_mg_m3",
            ] = (
                chl
                - chl_median
            )

            if chl_median > 0:

                df.at[
                    idx,
                    "chlorophyll_spatial_ratio",
                ] = (
                    chl
                    / chl_median
                )

            df.at[
                idx,
                "chlorophyll_spatial_percentile",
            ] = percentile_rank(
                regional_chl,
                chl,
            )

            df.at[
                idx,
                "log_chlorophyll_spatial_anomaly",
            ] = (
                np.log1p(
                    chl
                )
                - log_chl_median
            )


    # ------------------------------------------------------
    # Save.
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False,
    )


    relative_features = [
        "sst_spatial_anomaly_c",
        "cold_spatial_strength_c",
        "sst_spatial_percentile",
        "chlorophyll_spatial_anomaly_mg_m3",
        "chlorophyll_spatial_ratio",
        "chlorophyll_spatial_percentile",
        "log_chlorophyll_spatial_anomaly",
    ]


    complete = (
        eligible
        & df[
            relative_features
        ]
        .notna()
        .all(axis=1)
    )


    print("\n" + "=" * 90)
    print("SPATIAL FEATURE COMPLETENESS")
    print("=" * 90)

    for col in relative_features:

        available = (
            eligible
            & df[col].notna()
        )

        print(
            f"{col}:",
            int(
                available.sum()
            ),
            "/",
            int(
                eligible.sum()
            )
        )


    print(
        "\nAll spatial features complete:",
        int(
            complete.sum()
        ),
        "/",
        int(
            eligible.sum()
        )
    )


    print("\n" + "=" * 90)
    print("PRESENCE VS ABSENCE — SPATIAL FEATURES")
    print("=" * 90)

    print(
        df.loc[
            complete
        ]
        .groupby(
            "presence"
        )[
            relative_features
        ]
        .agg(
            [
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
