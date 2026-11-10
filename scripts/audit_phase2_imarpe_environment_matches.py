from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("data/imarpe")

ACOUSTIC = ROOT / "E_AWARE_IMARPE_phase2_with_depth.csv"

CTD = (
    ROOT
    / "raw_environment"
    / "DATOS_CTD_1994_2010.xlsx"
)

BOTTLE = (
    ROOT
    / "raw_environment"
    / "DATOS_BOTELLA_1990_2010.xlsx"
)

OUTPUT = (
    ROOT
    / "phase2_imarpe_environment_match_audit.csv"
)

# Wider than E-AWARE acoustic corridor so nearby
# stations up to ~25 km offshore/alongshore are retained.
LAT_MIN, LAT_MAX = -5.35, -3.85
LON_MIN, LON_MAX = -82.35, -80.85


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


def build_profiles(path, sheet, source):

    print(f"\nReading {source}...")

    df = pd.read_excel(
        path,
        sheet_name=sheet,
    )

    df["FECHA"] = pd.to_datetime(
        df["FECHA"],
        errors="coerce",
    ).dt.normalize()

    for c in [
        "LATITUD",
        "LONGITUD",
        "PROFUNDIDAD",
        "TEMPERATURA",
        "SALINIDAD",
        "OXIGENO",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df[
        df["FECHA"].notna()
        & df["LATITUD"].between(
            LAT_MIN,
            LAT_MAX
        )
        & df["LONGITUD"].between(
            LON_MIN,
            LON_MAX
        )
    ].copy()

    print(
        "Rows in expanded E-AWARE region:",
        len(df)
    )

    group_cols = [
        "CRUCERO",
        "ESTACION",
        "FECHA",
        "LATITUD",
        "LONGITUD",
    ]

    # Full-profile diagnostics.
    full = (
        df.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            profile_max_depth_m=(
                "PROFUNDIDAD",
                "max",
            ),
            profile_points=(
                "PROFUNDIDAD",
                "size",
            ),
            oxygen_points=(
                "OXIGENO",
                lambda x: x.notna().sum(),
            ),
        )
        .reset_index()
    )

    # We define a surface predictor from measurements
    # within the upper 10 m only.
    surface = df[
        df["PROFUNDIDAD"].between(
            0,
            10,
        )
    ].copy()

    # Sorting lets groupby.first() choose the shallowest
    # available non-missing value for each variable.
    surface = surface.sort_values(
        "PROFUNDIDAD"
    )

    surf = (
        surface.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            surface_depth_m=(
                "PROFUNDIDAD",
                "min",
            ),
            surface_temp_c=(
                "TEMPERATURA",
                "first",
            ),
            surface_salinity_psu=(
                "SALINIDAD",
                "first",
            ),
            surface_oxygen_mg_l=(
                "OXIGENO",
                "first",
            ),
        )
        .reset_index()
    )

    profiles = full.merge(
        surf,
        on=group_cols,
        how="left",
    )

    profiles["env_source"] = source

    print(
        "Unique profiles/stations:",
        len(profiles)
    )

    print(
        "Surface temperature available:",
        profiles["surface_temp_c"]
        .notna()
        .sum()
    )

    print(
        "Surface salinity available:",
        profiles["surface_salinity_psu"]
        .notna()
        .sum()
    )

    print(
        "Surface oxygen available:",
        profiles["surface_oxygen_mg_l"]
        .notna()
        .sum()
    )

    return profiles


ctd = build_profiles(
    CTD,
    "CTD",
    "IMARPE_ctd",
)

bottle = build_profiles(
    BOTTLE,
    "data_pelagico",
    "IMARPE_bottle",
)


ac = pd.read_csv(
    ACOUSTIC,
    parse_dates=["date"],
)

ac["date"] = (
    pd.to_datetime(
        ac["date"],
        errors="coerce",
    )
    .dt.normalize()
)

dated = ac[
    ac["date"].notna()
].copy()


def nearest_same_day(
    acoustic,
    profiles,
    prefix,
):

    profiles_by_date = {
        d: g.reset_index(drop=True)
        for d, g in profiles.groupby("FECHA")
    }

    output = []

    for _, row in acoustic.iterrows():

        day = row["date"]

        result = {
            "record_id": row["record_id"],
        }

        if day not in profiles_by_date:
            result[f"{prefix}_distance_km"] = np.nan
            result[f"{prefix}_cruise"] = None
            result[f"{prefix}_station"] = None
            result[f"{prefix}_temp_c"] = np.nan
            result[f"{prefix}_salinity_psu"] = np.nan
            result[f"{prefix}_oxygen_mg_l"] = np.nan
            result[f"{prefix}_profile_max_depth_m"] = np.nan
            result[f"{prefix}_surface_depth_m"] = np.nan

            output.append(result)
            continue

        candidates = profiles_by_date[day]

        distances = haversine_km(
            float(row["lat"]),
            float(row["lon"]),
            candidates["LATITUD"].to_numpy(),
            candidates["LONGITUD"].to_numpy(),
        )

        idx = int(np.nanargmin(distances))
        best = candidates.iloc[idx]

        result[f"{prefix}_distance_km"] = (
            float(distances[idx])
        )

        result[f"{prefix}_cruise"] = (
            best["CRUCERO"]
        )

        result[f"{prefix}_station"] = (
            best["ESTACION"]
        )

        result[f"{prefix}_temp_c"] = (
            best["surface_temp_c"]
        )

        result[f"{prefix}_salinity_psu"] = (
            best["surface_salinity_psu"]
        )

        result[f"{prefix}_oxygen_mg_l"] = (
            best["surface_oxygen_mg_l"]
        )

        result[
            f"{prefix}_profile_max_depth_m"
        ] = best["profile_max_depth_m"]

        result[
            f"{prefix}_surface_depth_m"
        ] = best["surface_depth_m"]

        output.append(result)

    return pd.DataFrame(output)


print("\nMatching acoustic records to CTD...")
ctd_match = nearest_same_day(
    dated,
    ctd,
    "ctd",
)

print("Matching acoustic records to bottle...")
bottle_match = nearest_same_day(
    dated,
    bottle,
    "bottle",
)

audit = (
    dated[
        [
            "record_id",
            "date",
            "lat",
            "lon",
            "presence",
            "source_group",
        ]
    ]
    .merge(
        ctd_match,
        on="record_id",
        how="left",
    )
    .merge(
        bottle_match,
        on="record_id",
        how="left",
    )
)


def print_source_summary(prefix, label):

    distance = audit[
        f"{prefix}_distance_km"
    ]

    print("\n" + "=" * 85)
    print(label)
    print("=" * 85)

    print(
        "Same-day station exists:",
        int(distance.notna().sum()),
        "/",
        len(audit),
    )

    for limit in [10, 25]:

        m = (
            distance.notna()
            & (distance <= limit)
        )

        print(
            f"\nWithin {limit} km:",
            int(m.sum())
        )

        for variable in [
            "temp_c",
            "salinity_psu",
            "oxygen_mg_l",
        ]:

            valid = (
                m
                & audit[
                    f"{prefix}_{variable}"
                ].notna()
            )

            print(
                f"  {variable}:",
                int(valid.sum())
            )


print("\n" + "=" * 85)
print("PHASE 2 IMARPE ENVIRONMENTAL MATCH AUDIT")
print("=" * 85)

print("Dated acoustic records:", len(audit))
print(
    "Unique acoustic dates:",
    audit["date"].nunique()
)

print_source_summary(
    "ctd",
    "IMARPE CTD",
)

print_source_summary(
    "bottle",
    "IMARPE BOTTLE",
)


print("\n" + "=" * 85)
print("BEST OF CTD + BOTTLE")
print("=" * 85)

for limit in [10, 25]:

    print(f"\nWithin {limit} km:")

    for variable in [
        "temp_c",
        "salinity_psu",
        "oxygen_mg_l",
    ]:

        ctd_ok = (
            audit["ctd_distance_km"].le(limit)
            & audit[
                f"ctd_{variable}"
            ].notna()
        )

        bottle_ok = (
            audit["bottle_distance_km"].le(limit)
            & audit[
                f"bottle_{variable}"
            ].notna()
        )

        best = ctd_ok | bottle_ok

        print(
            f"  {variable}:",
            int(best.sum()),
            "/",
            len(audit),
            f"({best.mean()*100:.1f}%)"
        )


print("\n" + "=" * 85)
print("DATE COVERAGE")
print("=" * 85)

dates = pd.DataFrame({
    "date": sorted(
        audit["date"].dropna().unique()
    )
})

ctd_dates = set(
    ctd["FECHA"].dropna()
)

bottle_dates = set(
    bottle["FECHA"].dropna()
)

dates["ctd_same_day"] = (
    dates["date"].isin(ctd_dates)
)

dates["bottle_same_day"] = (
    dates["date"].isin(bottle_dates)
)

dates["either_same_day"] = (
    dates["ctd_same_day"]
    | dates["bottle_same_day"]
)

print(
    dates.to_string(index=False)
)

print("\nDates with IMARPE same-day data:")
print(
    int(
        dates["either_same_day"].sum()
    ),
    "/",
    len(dates)
)

audit.to_csv(
    OUTPUT,
    index=False,
)

print("\nSaved:")
print(OUTPUT)
