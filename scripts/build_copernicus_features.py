from pathlib import Path
import math
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


INPUT = Path("data/imarpe/E_AWARE_IMARPE_master_phase1.csv")
OUTPUT = Path("data/imarpe/E_AWARE_IMARPE_copernicus_filled.csv")
CACHE = Path("data/imarpe/copernicus_cache")

PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"

# Wider than our E-AWARE study box so we can find the nearest
# valid offshore cell when a coastal Copernicus cell is masked.
MIN_LON = -82.0
MAX_LON = -81.0
MIN_LAT = -4.85
MAX_LAT = -4.10

MAX_PHY_DISTANCE_KM = 30
MAX_BGC_DISTANCE_KM = 60


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def download_subset(dataset_id, variables, date, output_file):
    if output_file.exists():
        print(f"Using cached file: {output_file}")
        return

    cmd = [
        "copernicusmarine",
        "subset",
        "--dataset-id",
        dataset_id,
    ]

    for variable in variables:
        cmd.extend(["--variable", variable])

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

    print(f"\nDownloading {dataset_id} for {date}")
    subprocess.run(cmd, check=True)


def nearest_valid_surface_value(
    ds,
    variable,
    target_lat,
    target_lon,
    max_distance_km,
):
    da = ds[variable]

    if "time" in da.dims:
        da = da.isel(time=0)

    if "depth" in da.dims:
        da = da.isel(depth=0)

    values = np.asarray(da.values)
    lats = np.asarray(ds["latitude"].values)
    lons = np.asarray(ds["longitude"].values)

    if values.ndim != 2:
        raise ValueError(
            f"{variable}: expected 2D surface field, got {values.shape}"
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

            if best is None or distance < best["distance_km"]:
                best = {
                    "value": float(value),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "distance_km": float(distance),
                }

    return best


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    print("===================================")
    print("E-AWARE COPERNICUS FEATURE BUILDER")
    print("===================================")
    print("Rows:", len(df))

    # Preserve original values before any Copernicus filling.
    for col in ["sst_c", "sss_psu", "oxygen_mg_l"]:
        original = f"{col}_before_copernicus"

        if original not in df.columns:
            df[original] = df[col]

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

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")

    unique_dates = sorted(
        parsed_dates.dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    print("Rows with exact date:", parsed_dates.notna().sum())
    print("Unique dates:", len(unique_dates))

    datasets = {}

    # Download one small regional file for each unique date.
    for date in unique_dates:
        phy_file = CACHE / f"phy_{date}.nc"
        bgc_file = CACHE / f"bgc_{date}.nc"

        download_subset(
            PHY_DATASET,
            ["thetao", "so"],
            date,
            phy_file,
        )

        download_subset(
            BGC_DATASET,
            ["o2"],
            date,
            bgc_file,
        )

        datasets[date] = {
            "phy": xr.open_dataset(phy_file),
            "bgc": xr.open_dataset(bgc_file),
        }

    for idx, row in df.iterrows():
        date_value = parsed_dates.iloc[idx]

        if pd.isna(date_value):
            continue

        date = date_value.strftime("%Y-%m-%d")

        lat = float(row["lat"])
        lon = float(row["lon"])

        phy = datasets[date]["phy"]
        bgc = datasets[date]["bgc"]

        temp = nearest_valid_surface_value(
            phy,
            "thetao",
            lat,
            lon,
            MAX_PHY_DISTANCE_KM,
        )

        sal = nearest_valid_surface_value(
            phy,
            "so",
            lat,
            lon,
            MAX_PHY_DISTANCE_KM,
        )

        oxygen = nearest_valid_surface_value(
            bgc,
            "o2",
            lat,
            lon,
            MAX_BGC_DISTANCE_KM,
        )

        if temp is not None:
            df.at[idx, "copernicus_sst_c"] = temp["value"]
            df.at[idx, "copernicus_sst_distance_km"] = temp["distance_km"]
            df.at[idx, "copernicus_sst_grid_lat"] = temp["latitude"]
            df.at[idx, "copernicus_sst_grid_lon"] = temp["longitude"]

            if pd.isna(row["sst_c"]):
                df.at[idx, "sst_c"] = temp["value"]
                df.at[idx, "sst_source"] = "Copernicus GLORYS12V1"

        if sal is not None:
            df.at[idx, "copernicus_sss_psu"] = sal["value"]
            df.at[idx, "copernicus_sss_distance_km"] = sal["distance_km"]
            df.at[idx, "copernicus_sss_grid_lat"] = sal["latitude"]
            df.at[idx, "copernicus_sss_grid_lon"] = sal["longitude"]

            if pd.isna(row["sss_psu"]):
                df.at[idx, "sss_psu"] = sal["value"]
                df.at[idx, "sss_source"] = "Copernicus GLORYS12V1"

        if oxygen is not None:
            # Copernicus oxygen is mmol/m^3.
            # 1 mmol O2 = 32 mg O2.
            # mmol/m^3 × 0.032 = mg/L.
            oxygen_mg_l = oxygen["value"] * 0.032

            df.at[idx, "copernicus_oxygen_mg_l"] = oxygen_mg_l
            df.at[idx, "copernicus_oxygen_distance_km"] = oxygen["distance_km"]
            df.at[idx, "copernicus_oxygen_grid_lat"] = oxygen["latitude"]
            df.at[idx, "copernicus_oxygen_grid_lon"] = oxygen["longitude"]

            if pd.isna(row["oxygen_mg_l"]):
                df.at[idx, "oxygen_mg_l"] = oxygen_mg_l
                df.at[idx, "oxygen_source"] = "Copernicus FREEBIORYS2V4"

    for date in datasets:
        datasets[date]["phy"].close()
        datasets[date]["bgc"].close()

    df.to_csv(OUTPUT, index=False)

    print("\n===================================")
    print("FEATURE COMPLETION FINISHED")
    print("===================================")

    print("\nOutput file:")
    print(OUTPUT)

    print("\nMissing values AFTER Copernicus:")

    for col in [
        "sst_c",
        "sss_psu",
        "oxygen_mg_l",
        "bottom_depth_m",
    ]:
        print(f"{col}: {df[col].isna().sum()}")

    print("\nCopernicus observations extracted:")
    print(
        "SST:",
        df["copernicus_sst_c"].notna().sum(),
    )
    print(
        "Salinity:",
        df["copernicus_sss_psu"].notna().sum(),
    )
    print(
        "Oxygen:",
        df["copernicus_oxygen_mg_l"].notna().sum(),
    )

    print("\nValues actually filled from Copernicus:")
    print(
        "SST:",
        (
            df["sst_c_before_copernicus"].isna()
            & df["sst_c"].notna()
        ).sum(),
    )
    print(
        "Salinity:",
        (
            df["sss_psu_before_copernicus"].isna()
            & df["sss_psu"].notna()
        ).sum(),
    )
    print(
        "Oxygen:",
        (
            df["oxygen_mg_l_before_copernicus"].isna()
            & df["oxygen_mg_l"].notna()
        ).sum(),
    )


if __name__ == "__main__":
    main()
