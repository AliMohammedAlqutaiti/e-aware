from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("data/imarpe")

INPUT = ROOT / "E_AWARE_IMARPE_phase2_acoustic_master.csv"
GEBCO = ROOT / "GEBCO_E_AWARE_phase2.nc"

OUTPUT = ROOT / "E_AWARE_IMARPE_phase2_with_depth.csv"


print("=" * 85)
print("E-AWARE PHASE 2 — GEBCO BATHYMETRY MATCHING")
print("=" * 85)

df = pd.read_csv(
    INPUT,
    parse_dates=["date"]
)

ds = xr.open_dataset(GEBCO)

print("\nAcoustic records:", len(df))
print(
    "GEBCO lon bounds:",
    float(ds.lon.min()),
    "to",
    float(ds.lon.max())
)
print(
    "GEBCO lat bounds:",
    float(ds.lat.min()),
    "to",
    float(ds.lat.max())
)


# ----------------------------------------------------------
# Vectorised nearest-grid lookup
# ----------------------------------------------------------

point = np.arange(len(df))

target_lat = xr.DataArray(
    df["lat"].to_numpy(),
    dims="point",
    coords={"point": point},
)

target_lon = xr.DataArray(
    df["lon"].to_numpy(),
    dims="point",
    coords={"point": point},
)

nearest = ds["elevation"].sel(
    lat=target_lat,
    lon=target_lon,
    method="nearest",
)

elevation = nearest.to_numpy()

df["gebco_elevation_m"] = elevation

# Negative elevation = ocean depth.
df["bottom_depth_m"] = np.where(
    df["gebco_elevation_m"] < 0,
    -df["gebco_elevation_m"],
    np.nan,
)

df["depth_source"] = np.where(
    df["bottom_depth_m"].notna(),
    "GEBCO 2026",
    "GEBCO land/non-ocean cell",
)

df["gebco_land_flag"] = (
    df["gebco_elevation_m"] >= 0
)

df["gebco_missing_flag"] = (
    df["gebco_elevation_m"].isna()
)


# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)


# ----------------------------------------------------------
# Audit
# ----------------------------------------------------------

print("\n" + "=" * 85)
print("DEPTH MATCH SUMMARY")
print("=" * 85)

print("Records:", len(df))

print(
    "Valid ocean depths:",
    int(df["bottom_depth_m"].notna().sum())
)

print(
    "Land/non-ocean nearest cells:",
    int(df["gebco_land_flag"].sum())
)

print(
    "Missing GEBCO values:",
    int(df["gebco_missing_flag"].sum())
)

if df["bottom_depth_m"].notna().any():

    print("\nDepth statistics (m):")

    print(
        df["bottom_depth_m"]
        .describe()
        .round(2)
    )


print("\n" + "=" * 85)
print("DEPTH BY ANCHOVETA PRESENCE")
print("=" * 85)

print(
    df.groupby("presence")["bottom_depth_m"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .round(2)
)


if df["gebco_land_flag"].any():

    print("\n" + "=" * 85)
    print("RECORDS MAPPING TO LAND / NON-OCEAN CELLS")
    print("=" * 85)

    print(
        df.loc[
            df["gebco_land_flag"],
            [
                "record_id",
                "lat",
                "lon",
                "source_group",
                "gebco_elevation_m",
            ]
        ]
        .to_string(index=False)
    )


print("\nSaved:")
print(OUTPUT)

ds.close()
