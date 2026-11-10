from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scipy.spatial import cKDTree


ROOT = Path("data/imarpe")

INPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_deployment_relative.csv"
)

GEBCO = (
    ROOT
    / "GEBCO_E_AWARE_phase2.nc"
)

OUTPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_final_features.csv"
)

EARTH_RADIUS_KM = 6371.0088


def coordinate_name(ds, options):

    for name in options:

        if name in ds.coords:
            return name

    raise KeyError(
        f"Could not find coordinate from {options}. "
        f"Available: {list(ds.coords)}"
    )


def latlon_to_xyz(lat, lon):

    lat = np.radians(
        np.asarray(lat, dtype=float)
    )

    lon = np.radians(
        np.asarray(lon, dtype=float)
    )

    x = (
        np.cos(lat)
        * np.cos(lon)
    )

    y = (
        np.cos(lat)
        * np.sin(lon)
    )

    z = np.sin(lat)

    return np.column_stack(
        [x, y, z]
    )


def chord_to_km(chord):

    chord = np.clip(
        chord,
        0.0,
        2.0
    )

    angle = (
        2.0
        * np.arcsin(
            chord / 2.0
        )
    )

    return (
        EARTH_RADIUS_KM
        * angle
    )


def build_coastline():

    with xr.open_dataset(
        GEBCO
    ) as ds:

        lat_name = coordinate_name(
            ds,
            [
                "lat",
                "latitude",
            ],
        )

        lon_name = coordinate_name(
            ds,
            [
                "lon",
                "longitude",
            ],
        )

        if "elevation" in ds.data_vars:

            elevation = (
                ds["elevation"]
                .values
            )

        else:

            variable = list(
                ds.data_vars
            )[0]

            print(
                "Using GEBCO variable:",
                variable
            )

            elevation = (
                ds[variable]
                .values
            )

        lats = np.asarray(
            ds[lat_name].values
        )

        lons = np.asarray(
            ds[lon_name].values
        )


    elevation = np.asarray(
        elevation
    ).squeeze()


    if elevation.ndim != 2:

        raise ValueError(
            "Expected 2-D GEBCO elevation grid, "
            f"got {elevation.shape}"
        )


    # GEBCO:
    # elevation >= 0 → land
    # elevation < 0  → ocean
    land = (
        np.isfinite(elevation)
        & (elevation >= 0)
    )

    ocean = (
        np.isfinite(elevation)
        & (elevation < 0)
    )


    # ------------------------------------------------------
    # Coastline approximation:
    # ocean cells touching a land cell.
    #
    # This is preferable to simply finding the nearest
    # arbitrary land cell.
    # ------------------------------------------------------

    adjacent_land = np.zeros_like(
        land,
        dtype=bool
    )


    # North / south
    adjacent_land[1:, :] |= (
        land[:-1, :]
    )

    adjacent_land[:-1, :] |= (
        land[1:, :]
    )


    # East / west
    adjacent_land[:, 1:] |= (
        land[:, :-1]
    )

    adjacent_land[:, :-1] |= (
        land[:, 1:]
    )


    coast_ocean = (
        ocean
        & adjacent_land
    )


    lat_grid, lon_grid = np.meshgrid(
        lats,
        lons,
        indexing="ij",
    )


    coast_lat = (
        lat_grid[
            coast_ocean
        ]
    )

    coast_lon = (
        lon_grid[
            coast_ocean
        ]
    )


    if len(coast_lat) == 0:

        raise ValueError(
            "No coastline cells detected."
        )


    print(
        "Detected GEBCO coastline cells:",
        len(coast_lat)
    )


    return (
        coast_lat,
        coast_lon
    )


def main():

    df = pd.read_csv(
        INPUT,
        parse_dates=["date"],
    )


    print("=" * 90)
    print(
        "E-AWARE PHASE 2 — "
        "COAST + SEASON FEATURES"
    )
    print("=" * 90)


    # ======================================================
    # DISTANCE TO COAST
    # ======================================================

    coast_lat, coast_lon = (
        build_coastline()
    )


    coast_xyz = latlon_to_xyz(
        coast_lat,
        coast_lon
    )


    tree = cKDTree(
        coast_xyz
    )


    point_xyz = latlon_to_xyz(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
    )


    chord_distance, _ = tree.query(
        point_xyz,
        k=1
    )


    df["distance_to_coast_km"] = (
        chord_to_km(
            chord_distance
        )
    )


    # ======================================================
    # TEMPORAL / SEASON FEATURES
    # ======================================================

    date = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    df["month"] = (
        date.dt.month
    )


    # Circular encoding avoids the artificial jump
    # between December (12) and January (1).
    df["month_sin"] = np.sin(
        2
        * np.pi
        * (
            df["month"] - 1
        )
        / 12
    )


    df["month_cos"] = np.cos(
        2
        * np.pi
        * (
            df["month"] - 1
        )
        / 12
    )


    # Human-readable austral seasons.
    season_map = {
        12: "summer",
        1: "summer",
        2: "summer",

        3: "autumn",
        4: "autumn",
        5: "autumn",

        6: "winter",
        7: "winter",
        8: "winter",

        9: "spring",
        10: "spring",
        11: "spring",
    }


    df["austral_season"] = (
        df["month"]
        .map(
            season_map
        )
    )


    # ======================================================
    # SAVE
    # ======================================================

    df.to_csv(
        OUTPUT,
        index=False
    )


    eligible = (
        date.notna()
        & (
            date
            >= pd.Timestamp(
                "1993-01-01"
            )
        )
    )


    print("\n" + "=" * 90)
    print("FEATURE COMPLETENESS")
    print("=" * 90)


    for col in [
        "distance_to_coast_km",
        "month",
        "month_sin",
        "month_cos",
    ]:

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


    print("\n" + "=" * 90)
    print("DISTANCE TO COAST")
    print("=" * 90)


    print(
        df.loc[
            eligible,
            "distance_to_coast_km"
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
        .round(3)
        .to_string()
    )


    print("\n" + "=" * 90)
    print("PRESENCE VS ABSENCE")
    print("=" * 90)


    print(
        df.loc[
            eligible
        ]
        .groupby(
            "presence"
        )[
            [
                "distance_to_coast_km",
                "month_sin",
                "month_cos",
            ]
        ]
        .agg(
            [
                "mean",
                "median",
            ]
        )
        .round(4)
        .to_string()
    )


    print("\n" + "=" * 90)
    print("DETECTION BY MONTH")
    print("=" * 90)


    monthly = (
        df.loc[
            eligible
        ]
        .groupby(
            "month"
        )
        .agg(
            records=(
                "presence",
                "size",
            ),
            detections=(
                "presence",
                "sum",
            ),
        )
    )


    monthly[
        "detection_rate"
    ] = (
        monthly["detections"]
        / monthly["records"]
    )


    print(
        monthly
        .round(4)
        .to_string()
    )


    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
