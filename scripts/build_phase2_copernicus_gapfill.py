from pathlib import Path
import math
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("data/imarpe")

INPUT = ROOT / "E_AWARE_IMARPE_phase2_environment_observed.csv"
OUTPUT = ROOT / "E_AWARE_IMARPE_phase2_copernicus_filled.csv"
CACHE = ROOT / "copernicus_cache_phase2"

PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"

COPERNICUS_START = pd.Timestamp("1993-01-01")

# Expanded around the Phase 2 E-AWARE corridor.
# The extra margin allows the nearest valid offshore grid cell
# to be found when coastal Copernicus cells are masked.
MIN_LON = -82.70
MAX_LON = -80.50
MIN_LAT = -5.60
MAX_LAT = -3.60

MAX_PHY_DISTANCE_KM = 30.0
MAX_BGC_DISTANCE_KM = 60.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2.0) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


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
        f"\nDownloading {dataset_id} "
        f"for {date}"
    )

    subprocess.run(
        cmd,
        check=True,
    )


def coord_name(ds, options):
    for name in options:
        if name in ds.coords:
            return name

    raise KeyError(
        f"Could not find coordinate from {options}. "
        f"Available coordinates: {list(ds.coords)}"
    )


def nearest_valid_surface_value(
    ds,
    variable,
    target_lat,
    target_lon,
    max_distance_km,
):

    da = ds[variable]

    time_name = None

    for candidate in ["time", "TIME"]:
        if candidate in da.dims:
            time_name = candidate
            break

    if time_name is not None:
        da = da.isel(
            {
                time_name: 0
            }
        )

    depth_name = None

    for candidate in [
        "depth",
        "DEPTH",
    ]:
        if candidate in da.dims:
            depth_name = candidate
            break

    if depth_name is not None:
        da = da.isel(
            {
                depth_name: 0
            }
        )

    lat_name = coord_name(
        ds,
        [
            "latitude",
            "lat",
        ],
    )

    lon_name = coord_name(
        ds,
        [
            "longitude",
            "lon",
        ],
    )

    values = np.asarray(
        da.values
    ).squeeze()

    lats = np.asarray(
        ds[lat_name].values
    )

    lons = np.asarray(
        ds[lon_name].values
    )

    if values.ndim != 2:
        raise ValueError(
            f"{variable}: expected 2D surface field, "
            f"got shape {values.shape}"
        )

    best = None

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):

            value = values[i, j]

            if not np.isfinite(value):
                continue

            distance = haversine_km(
                target_lat,
                target_lon,
                float(lat),
                float(lon),
            )

            if distance > max_distance_km:
                continue

            if (
                best is None
                or distance
                < best["distance_km"]
            ):
                best = {
                    "value": float(value),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "distance_km": float(distance),
                }

    return best


def unique_needed_dates(
    dates,
    eligible,
    missing_mask,
):

    mask = (
        eligible
        & missing_mask
        & dates.notna()
    )

    return sorted(
        dates.loc[mask]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )


def main():

    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT,
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

    print("=" * 90)
    print("E-AWARE PHASE 2 — COPERNICUS GAP FILL")
    print("=" * 90)

    print("\nTotal records:", len(df))

    print(
        "Dated records:",
        int(
            dates.notna().sum()
        ),
    )

    print(
        "Copernicus eligible:",
        int(
            eligible.sum()
        ),
    )

    print(
        "Pre-1993 dated records:",
        int(
            (
                dates.notna()
                & ~eligible
            ).sum()
        ),
    )

    print(
        "Unknown-date records:",
        int(
            dates.isna().sum()
        ),
    )

    # Preserve the observed values.
    for col in [
        "sst_c",
        "sss_psu",
        "oxygen_mg_l",
    ]:

        before = (
            f"{col}_before_copernicus"
        )

        if before not in df.columns:
            df[before] = df[col]

    # Copernicus provenance columns.
    new_columns = [
        "copernicus_sst_c",
        "copernicus_sst_distance_km",
        "copernicus_sst_grid_lat",
        "copernicus_sst_grid_lon",

        "copernicus_sss_psu",
        "copernicus_sss_distance_km",
        "copernicus_sss_grid_lat",
        "copernicus_sss_grid_lon",

        "copernicus_oxygen_mg_l",
        "copernicus_oxygen_distance_km",
        "copernicus_oxygen_grid_lat",
        "copernicus_oxygen_grid_lon",
    ]

    for col in new_columns:
        if col not in df.columns:
            df[col] = np.nan

    missing_sst = (
        df["sst_c"].isna()
    )

    missing_sss = (
        df["sss_psu"].isna()
    )

    missing_o2 = (
        df["oxygen_mg_l"].isna()
    )

    print("\n" + "=" * 90)
    print("GAPS BEFORE COPERNICUS")
    print("=" * 90)

    print(
        "Eligible SST gaps:",
        int(
            (
                eligible
                & missing_sst
            ).sum()
        ),
    )

    print(
        "Eligible salinity gaps:",
        int(
            (
                eligible
                & missing_sss
            ).sum()
        ),
    )

    print(
        "Eligible oxygen gaps:",
        int(
            (
                eligible
                & missing_o2
            ).sum()
        ),
    )

    phy_dates = unique_needed_dates(
        dates,
        eligible,
        (
            missing_sst
            | missing_sss
        ),
    )

    bgc_dates = unique_needed_dates(
        dates,
        eligible,
        missing_o2,
    )

    print(
        "\nPhysics dates needed:",
        len(phy_dates),
    )

    print(
        "BGC dates needed:",
        len(bgc_dates),
    )

    # ------------------------------------------------------
    # Download only the required dates.
    # ------------------------------------------------------

    phy_paths = {}

    for date in phy_dates:

        path = (
            CACHE
            / f"phy_{date}.nc"
        )

        download_subset(
            PHY_DATASET,
            [
                "thetao",
                "so",
            ],
            date,
            path,
        )

        phy_paths[date] = path

    bgc_paths = {}

    for date in bgc_dates:

        path = (
            CACHE
            / f"bgc_{date}.nc"
        )

        download_subset(
            BGC_DATASET,
            [
                "o2",
            ],
            date,
            path,
        )

        bgc_paths[date] = path

    # ------------------------------------------------------
    # Open downloaded datasets.
    # ------------------------------------------------------

    phy_ds = {
        date: xr.open_dataset(path)
        for date, path
        in phy_paths.items()
    }

    bgc_ds = {
        date: xr.open_dataset(path)
        for date, path
        in bgc_paths.items()
    }

    # ------------------------------------------------------
    # Fill only missing values.
    # ------------------------------------------------------

    for idx, row in df.iterrows():

        date_value = dates.iloc[idx]

        if (
            pd.isna(date_value)
            or date_value
            < COPERNICUS_START
        ):
            continue

        date = (
            date_value
            .strftime("%Y-%m-%d")
        )

        lat = float(
            row["lat"]
        )

        lon = float(
            row["lon"]
        )

        # ---------------- SST ----------------
        if (
            pd.isna(
                df.at[
                    idx,
                    "sst_c",
                ]
            )
            and date in phy_ds
        ):

            temp = (
                nearest_valid_surface_value(
                    phy_ds[date],
                    "thetao",
                    lat,
                    lon,
                    MAX_PHY_DISTANCE_KM,
                )
            )

            if temp is not None:

                df.at[
                    idx,
                    "copernicus_sst_c",
                ] = temp["value"]

                df.at[
                    idx,
                    "copernicus_sst_distance_km",
                ] = temp["distance_km"]

                df.at[
                    idx,
                    "copernicus_sst_grid_lat",
                ] = temp["latitude"]

                df.at[
                    idx,
                    "copernicus_sst_grid_lon",
                ] = temp["longitude"]

                df.at[
                    idx,
                    "sst_c",
                ] = temp["value"]

                df.at[
                    idx,
                    "sst_source",
                ] = (
                    "Copernicus GLORYS12V1"
                )

        # ---------------- SALINITY ----------------
        if (
            pd.isna(
                df.at[
                    idx,
                    "sss_psu",
                ]
            )
            and date in phy_ds
        ):

            sal = (
                nearest_valid_surface_value(
                    phy_ds[date],
                    "so",
                    lat,
                    lon,
                    MAX_PHY_DISTANCE_KM,
                )
            )

            if sal is not None:

                df.at[
                    idx,
                    "copernicus_sss_psu",
                ] = sal["value"]

                df.at[
                    idx,
                    "copernicus_sss_distance_km",
                ] = sal["distance_km"]

                df.at[
                    idx,
                    "copernicus_sss_grid_lat",
                ] = sal["latitude"]

                df.at[
                    idx,
                    "copernicus_sss_grid_lon",
                ] = sal["longitude"]

                df.at[
                    idx,
                    "sss_psu",
                ] = sal["value"]

                df.at[
                    idx,
                    "sss_source",
                ] = (
                    "Copernicus GLORYS12V1"
                )

        # ---------------- OXYGEN ----------------
        if (
            pd.isna(
                df.at[
                    idx,
                    "oxygen_mg_l",
                ]
            )
            and date in bgc_ds
        ):

            oxygen = (
                nearest_valid_surface_value(
                    bgc_ds[date],
                    "o2",
                    lat,
                    lon,
                    MAX_BGC_DISTANCE_KM,
                )
            )

            if oxygen is not None:

                # Dataset unit: mmol/m^3.
                # mmol/m^3 × 0.032 = mg/L.
                oxygen_mg_l = (
                    oxygen["value"]
                    * 0.032
                )

                df.at[
                    idx,
                    "copernicus_oxygen_mg_l",
                ] = oxygen_mg_l

                df.at[
                    idx,
                    "copernicus_oxygen_distance_km",
                ] = oxygen["distance_km"]

                df.at[
                    idx,
                    "copernicus_oxygen_grid_lat",
                ] = oxygen["latitude"]

                df.at[
                    idx,
                    "copernicus_oxygen_grid_lon",
                ] = oxygen["longitude"]

                df.at[
                    idx,
                    "oxygen_mg_l",
                ] = oxygen_mg_l

                df.at[
                    idx,
                    "oxygen_source",
                ] = (
                    "Copernicus "
                    "GLOBAL_MULTIYEAR_BGC"
                )

    # ------------------------------------------------------
    # Close datasets.
    # ------------------------------------------------------

    for ds in phy_ds.values():
        ds.close()

    for ds in bgc_ds.values():
        ds.close()

    # ------------------------------------------------------
    # Save.
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # ------------------------------------------------------
    # Final audit.
    # ------------------------------------------------------

    print("\n" + "=" * 90)
    print("COPERNICUS GAP FILL RESULTS")
    print("=" * 90)

    for col, label in [
        (
            "sst_c",
            "SST",
        ),
        (
            "sss_psu",
            "Salinity",
        ),
        (
            "oxygen_mg_l",
            "Oxygen",
        ),
    ]:

        before = (
            f"{col}_before_copernicus"
        )

        filled = (
            df[before].isna()
            & df[col].notna()
        )

        remaining_eligible = (
            eligible
            & df[col].isna()
        )

        print(
            f"\n{label}:"
        )

        print(
            "  Filled by Copernicus:",
            int(
                filled.sum()
            ),
        )

        print(
            "  Remaining eligible gaps:",
            int(
                remaining_eligible.sum()
            ),
        )

        print(
            "  Total available:",
            int(
                df[col]
                .notna()
                .sum()
            ),
            "/",
            len(df),
        )

    print("\n" + "=" * 90)
    print("SOURCE COUNTS")
    print("=" * 90)

    for col in [
        "sst_source",
        "sss_source",
        "oxygen_source",
    ]:

        print(
            f"\n{col}:"
        )

        print(
            df[col]
            .fillna("MISSING")
            .value_counts()
            .to_string()
        )

    core = (
        df["sst_c"].notna()
        & df["sss_psu"].notna()
        & df["oxygen_mg_l"].notna()
        & df["bottom_depth_m"].notna()
    )

    print("\n" + "=" * 90)
    print("CORE PREDICTOR COMPLETENESS")
    print("=" * 90)

    print(
        "All 3009 records:",
        int(
            core.sum()
        ),
        "/",
        len(df),
        f"({core.mean()*100:.1f}%)",
    )

    eligible_core = (
        core
        & eligible
    )

    print(
        "Copernicus-eligible records:",
        int(
            eligible_core.sum()
        ),
        "/",
        int(
            eligible.sum()
        ),
        f"({eligible_core.sum()/eligible.sum()*100:.1f}%)",
    )

    dated_core = (
        core
        & dates.notna()
    )

    print(
        "All dated records:",
        int(
            dated_core.sum()
        ),
        "/",
        int(
            dates.notna().sum()
        ),
        f"({dated_core.sum()/dates.notna().sum()*100:.1f}%)",
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
