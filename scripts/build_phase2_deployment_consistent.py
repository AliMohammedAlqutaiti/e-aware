from pathlib import Path
import math
import subprocess

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("data/imarpe")

INPUT = ROOT / "E_AWARE_IMARPE_phase2_enriched.csv"

OUTPUT = (
    ROOT
    / "E_AWARE_IMARPE_phase2_deployment_consistent.csv"
)

CACHE = ROOT / "copernicus_cache_phase2"

PHY_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
BGC_DATASET = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"

COPERNICUS_START = pd.Timestamp("1993-01-01")

MIN_LON = -82.70
MAX_LON = -80.50
MIN_LAT = -5.60
MAX_LAT = -3.60

MAX_PHY_DISTANCE_KM = 30.0
MAX_BGC_DISTANCE_KM = 60.0


def download_subset(
    dataset_id,
    variable,
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
        "--variable",
        variable,
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

    print(
        f"\nDownloading {variable} for {date}"
    )

    subprocess.run(
        cmd,
        check=True,
    )


def has_variable(path, variable):

    if not path.exists():
        return False

    try:
        with xr.open_dataset(path) as ds:
            return variable in ds.data_vars

    except Exception:
        return False


def coordinate_name(ds, options):

    for name in options:

        if name in ds.coords:
            return name

    raise KeyError(
        f"Coordinate not found from {options}. "
        f"Available: {list(ds.coords)}"
    )


def surface_grid(
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

        units = da.attrs.get(
            "units",
            "",
        )

    if values.ndim != 2:

        raise ValueError(
            f"{variable} in {path.name}: "
            f"expected 2D field, "
            f"got {values.shape}"
        )

    lat_grid, lon_grid = np.meshgrid(
        lats,
        lons,
        indexing="ij",
    )

    valid = np.isfinite(values)

    return {
        "values": values[valid].astype(float),
        "latitudes": lat_grid[valid].astype(float),
        "longitudes": lon_grid[valid].astype(float),
        "units": units,
    }


def nearest(
    grid,
    target_lat,
    target_lon,
    max_distance_km,
):

    lat1 = math.radians(
        float(target_lat)
    )

    lon1 = math.radians(
        float(target_lon)
    )

    lat2 = np.radians(
        grid["latitudes"]
    )

    lon2 = np.radians(
        grid["longitudes"]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
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

    i = int(
        np.argmin(distance)
    )

    d = float(
        distance[i]
    )

    if d > max_distance_km:
        return None

    return {
        "value": float(
            grid["values"][i]
        ),
        "distance_km": d,
        "grid_lat": float(
            grid["latitudes"][i]
        ),
        "grid_lon": float(
            grid["longitudes"][i]
        ),
    }


def oxygen_to_mg_l(
    value,
    units,
):

    unit = (
        str(units)
        .lower()
        .replace(" ", "")
        .replace("^", "")
    )

    # Current Copernicus BGC format.
    if "mmol" in unit:

        return float(value) * 0.032

    if "mg" in unit and (
        "m-3" in unit
        or "m-3" in unit
    ):

        return float(value) / 1000.0

    if "mg/l" in unit or "mgl-1" in unit:

        return float(value)

    raise ValueError(
        "Unknown oxygen units: "
        + repr(units)
    )


def chlorophyll_to_mg_m3(
    value,
    units,
):

    unit = (
        str(units)
        .lower()
        .replace(" ", "")
    )

    if "mg" in unit:
        return float(value)

    if "kg" in unit:
        return float(value) * 1_000_000.0

    raise ValueError(
        "Unknown chlorophyll units: "
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
        "DEPLOYMENT-CONSISTENT COPERNICUS DATA"
    )
    print("=" * 90)

    print(
        "\nEligible records:",
        int(eligible.sum())
    )

    print(
        "Unique dates:",
        len(unique_dates)
    )


    # ------------------------------------------------------
    # Complete salinity and oxygen cache.
    # ------------------------------------------------------

    missing_so_dates = []
    missing_o2_dates = []

    for date in unique_dates:

        phy = CACHE / f"phy_{date}.nc"

        if not has_variable(
            phy,
            "so",
        ):

            missing_so_dates.append(
                date
            )

        bgc = CACHE / f"bgc_{date}.nc"

        if not has_variable(
            bgc,
            "o2",
        ):

            missing_o2_dates.append(
                date
            )


    print(
        "\nSalinity dates to download:",
        len(missing_so_dates)
    )

    print(
        "Oxygen dates to download:",
        len(missing_o2_dates)
    )


    for date in missing_so_dates:

        output = (
            CACHE
            / f"so_{date}.nc"
        )

        download_subset(
            PHY_DATASET,
            "so",
            date,
            output,
        )


    for date in missing_o2_dates:

        output = (
            CACHE
            / f"o2_{date}.nc"
        )

        download_subset(
            BGC_DATASET,
            "o2",
            date,
            output,
        )


    # ------------------------------------------------------
    # Deployment feature columns.
    # ------------------------------------------------------

    columns = [
        "deploy_sst_c",
        "deploy_sst_distance_km",

        "deploy_sss_psu",
        "deploy_sss_distance_km",

        "deploy_oxygen_mg_l",
        "deploy_oxygen_distance_km",

        "deploy_chlorophyll_mg_m3",
        "deploy_chlorophyll_distance_km",
    ]

    for col in columns:
        df[col] = np.nan

    df["deploy_bottom_depth_m"] = (
        df["bottom_depth_m"]
    )

    df["deployment_env_source"] = None


    # ------------------------------------------------------
    # Extract one date at a time.
    # ------------------------------------------------------

    oxygen_units = set()
    chl_units = set()

    for date in unique_dates:

        print(
            "\nExtracting deployment features:",
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


        # ---------------- SST ----------------

        phy_path = (
            CACHE
            / f"phy_{date}.nc"
        )

        sst_grid = surface_grid(
            phy_path,
            "thetao",
        )


        # ---------------- SALINITY ----------------

        if has_variable(
            phy_path,
            "so",
        ):

            so_path = phy_path

        else:

            so_path = (
                CACHE
                / f"so_{date}.nc"
            )

        sal_grid = surface_grid(
            so_path,
            "so",
        )


        # ---------------- OXYGEN ----------------

        bgc_path = (
            CACHE
            / f"bgc_{date}.nc"
        )

        if has_variable(
            bgc_path,
            "o2",
        ):

            o2_path = bgc_path

        else:

            o2_path = (
                CACHE
                / f"o2_{date}.nc"
            )

        o2_grid = surface_grid(
            o2_path,
            "o2",
        )

        oxygen_units.add(
            o2_grid["units"]
        )


        # ---------------- CHLOROPHYLL ----------------

        chl_path = (
            CACHE
            / f"chl_{date}.nc"
        )

        chl_grid = surface_grid(
            chl_path,
            "chl",
        )

        chl_units.add(
            chl_grid["units"]
        )


        # --------------------------------------------------
        # Match all acoustic points for this date.
        # --------------------------------------------------

        for idx in indices:

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


            sst = nearest(
                sst_grid,
                lat,
                lon,
                MAX_PHY_DISTANCE_KM,
            )

            sal = nearest(
                sal_grid,
                lat,
                lon,
                MAX_PHY_DISTANCE_KM,
            )

            oxygen = nearest(
                o2_grid,
                lat,
                lon,
                MAX_BGC_DISTANCE_KM,
            )

            chl = nearest(
                chl_grid,
                lat,
                lon,
                MAX_BGC_DISTANCE_KM,
            )


            if sst is not None:

                df.at[
                    idx,
                    "deploy_sst_c",
                ] = sst["value"]

                df.at[
                    idx,
                    "deploy_sst_distance_km",
                ] = sst["distance_km"]


            if sal is not None:

                df.at[
                    idx,
                    "deploy_sss_psu",
                ] = sal["value"]

                df.at[
                    idx,
                    "deploy_sss_distance_km",
                ] = sal["distance_km"]


            if oxygen is not None:

                df.at[
                    idx,
                    "deploy_oxygen_mg_l",
                ] = oxygen_to_mg_l(
                    oxygen["value"],
                    o2_grid["units"],
                )

                df.at[
                    idx,
                    "deploy_oxygen_distance_km",
                ] = oxygen[
                    "distance_km"
                ]


            if chl is not None:

                df.at[
                    idx,
                    "deploy_chlorophyll_mg_m3",
                ] = chlorophyll_to_mg_m3(
                    chl["value"],
                    chl_grid["units"],
                )

                df.at[
                    idx,
                    "deploy_chlorophyll_distance_km",
                ] = chl[
                    "distance_km"
                ]


            if (
                sst is not None
                and sal is not None
                and oxygen is not None
                and chl is not None
            ):

                df.at[
                    idx,
                    "deployment_env_source",
                ] = (
                    "Copernicus historical + GEBCO"
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

    deployment_features = [
        "deploy_sst_c",
        "deploy_sss_psu",
        "deploy_oxygen_mg_l",
        "deploy_chlorophyll_mg_m3",
        "deploy_bottom_depth_m",
    ]

    complete = (
        eligible
        & df[
            deployment_features
        ]
        .notna()
        .all(axis=1)
    )


    print("\n" + "=" * 90)
    print("DEPLOYMENT DATASET COMPLETENESS")
    print("=" * 90)

    for col in deployment_features:

        available = (
            eligible
            & df[col].notna()
        )

        print(
            f"{col}:",
            int(available.sum()),
            "/",
            int(eligible.sum()),
        )


    print(
        "\nComplete deployment records:",
        int(complete.sum()),
        "/",
        int(eligible.sum()),
        f"({complete.sum()/eligible.sum()*100:.1f}%)",
    )


    print(
        "\nOxygen units encountered:",
        oxygen_units,
    )

    print(
        "Chlorophyll units encountered:",
        chl_units,
    )


    print("\n" + "=" * 90)
    print("DEPLOYMENT FEATURE RANGES")
    print("=" * 90)

    print(
        df.loc[
            complete,
            deployment_features,
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


    print("\n" + "=" * 90)
    print("PRESENCE VS ABSENCE")
    print("=" * 90)

    print(
        df.loc[
            complete
        ]
        .groupby(
            "presence"
        )[
            deployment_features
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


    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
