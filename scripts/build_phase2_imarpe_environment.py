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
    / "E_AWARE_IMARPE_phase2_environment_observed.csv"
)

LAT_MIN, LAT_MAX = -5.35, -3.85
LON_MIN, LON_MAX = -82.35, -80.85

MAX_DISTANCE_KM = 25.0


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


def load_environment(path, sheet, source):

    print("\nReading:", source)

    df = pd.read_excel(
        path,
        sheet_name=sheet,
    )

    df["FECHA"] = pd.to_datetime(
        df["FECHA"],
        errors="coerce",
    ).dt.normalize()

    numeric = [
        "LATITUD",
        "LONGITUD",
        "PROFUNDIDAD",
        "TEMPERATURA",
        "SALINIDAD",
        "OXIGENO",
    ]

    for c in numeric:
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

    # Surface environmental predictors only.
    df = df[
        df["PROFUNDIDAD"].between(
            0,
            10,
        )
    ].copy()

    df["env_source"] = source

    print(
        "Surface rows retained:",
        len(df)
    )

    return df


ctd = load_environment(
    CTD,
    "CTD",
    "IMARPE_ctd",
)

bottle = load_environment(
    BOTTLE,
    "data_pelagico",
    "IMARPE_bottle",
)

env = pd.concat(
    [ctd, bottle],
    ignore_index=True,
)


# ----------------------------------------------------------
# Create one shallowest valid measurement per station,
# date and environmental variable.
# ----------------------------------------------------------

station_cols = [
    "CRUCERO",
    "ESTACION",
    "FECHA",
    "LATITUD",
    "LONGITUD",
    "env_source",
]


def build_variable_table(variable):

    x = env[
        env[variable].notna()
    ].copy()

    x = x.sort_values(
        "PROFUNDIDAD"
    )

    x = (
        x.groupby(
            station_cols,
            dropna=False,
            as_index=False,
        )
        .first()
    )

    keep = station_cols + [
        "PROFUNDIDAD",
        variable,
    ]

    return x[keep].copy()


temp = build_variable_table(
    "TEMPERATURA"
)

sal = build_variable_table(
    "SALINIDAD"
)

oxy = build_variable_table(
    "OXIGENO"
)

print("\nAvailable station/day surface measurements:")
print("Temperature:", len(temp))
print("Salinity:   ", len(sal))
print("Oxygen:     ", len(oxy))


# ----------------------------------------------------------
# Prepare acoustic data.
# ----------------------------------------------------------

ac = pd.read_csv(
    ACOUSTIC,
    parse_dates=["date"],
)

ac["date"] = pd.to_datetime(
    ac["date"],
    errors="coerce",
).dt.normalize()


# ----------------------------------------------------------
# Find nearest valid IMARPE measurement independently
# for each environmental variable.
# ----------------------------------------------------------

def match_variable(
    acoustic,
    measurements,
    value_col,
    output_value,
    prefix,
):

    by_date = {
        d: g.reset_index(drop=True)
        for d, g in measurements.groupby("FECHA")
    }

    results = []

    for _, row in acoustic.iterrows():

        result = {
            "record_id": row["record_id"],
            output_value: np.nan,
            f"{prefix}_source": None,
            f"{prefix}_match_quality": None,
            f"{prefix}_match_distance_km": np.nan,
            f"{prefix}_measurement_depth_m": np.nan,
            f"{prefix}_match_cruise": None,
            f"{prefix}_match_station": None,
            f"{prefix}_match_date": pd.NaT,
        }

        day = row["date"]

        if pd.isna(day):
            results.append(result)
            continue

        candidates = by_date.get(day)

        if candidates is None or len(candidates) == 0:
            results.append(result)
            continue

        distances = haversine_km(
            float(row["lat"]),
            float(row["lon"]),
            candidates["LATITUD"].to_numpy(),
            candidates["LONGITUD"].to_numpy(),
        )

        candidates = candidates.copy()
        candidates["_distance_km"] = distances

        candidates = candidates[
            candidates["_distance_km"]
            <= MAX_DISTANCE_KM
        ].copy()

        if candidates.empty:
            results.append(result)
            continue

        # Prefer:
        # 1. shortest geographic distance
        # 2. shallowest measurement
        # 3. CTD if otherwise tied
        candidates["_source_priority"] = (
            candidates["env_source"]
            .map({
                "IMARPE_ctd": 0,
                "IMARPE_bottle": 1,
            })
            .fillna(9)
        )

        candidates = candidates.sort_values(
            [
                "_distance_km",
                "PROFUNDIDAD",
                "_source_priority",
            ]
        )

        best = candidates.iloc[0]

        distance = float(
            best["_distance_km"]
        )

        quality = (
            "A_exact_10km"
            if distance <= 10
            else "B_exact_25km"
        )

        result[output_value] = (
            best[value_col]
        )

        result[f"{prefix}_source"] = (
            best["env_source"]
        )

        result[
            f"{prefix}_match_quality"
        ] = quality

        result[
            f"{prefix}_match_distance_km"
        ] = distance

        result[
            f"{prefix}_measurement_depth_m"
        ] = best["PROFUNDIDAD"]

        result[
            f"{prefix}_match_cruise"
        ] = best["CRUCERO"]

        result[
            f"{prefix}_match_station"
        ] = best["ESTACION"]

        result[
            f"{prefix}_match_date"
        ] = best["FECHA"]

        results.append(result)

    return pd.DataFrame(results)


print("\nMatching temperature...")

m_temp = match_variable(
    ac,
    temp,
    "TEMPERATURA",
    "sst_c",
    "sst",
)

print("Matching salinity...")

m_sal = match_variable(
    ac,
    sal,
    "SALINIDAD",
    "sss_psu",
    "sss",
)

print("Matching oxygen...")

m_oxy = match_variable(
    ac,
    oxy,
    "OXIGENO",
    "oxygen_mg_l",
    "oxygen",
)


# ----------------------------------------------------------
# Merge into Phase 2 dataset.
# ----------------------------------------------------------

out = (
    ac
    .merge(
        m_temp,
        on="record_id",
        how="left",
    )
    .merge(
        m_sal,
        on="record_id",
        how="left",
    )
    .merge(
        m_oxy,
        on="record_id",
        how="left",
    )
)

out.to_csv(
    OUTPUT,
    index=False,
)


# ----------------------------------------------------------
# Audit output.
# ----------------------------------------------------------

print("\n" + "=" * 90)
print("PHASE 2 IMARPE OBSERVED ENVIRONMENT")
print("=" * 90)

print("\nTotal acoustic records:", len(out))
print(
    "Dated records:",
    int(out["date"].notna().sum())
)

variables = [
    ("sst_c", "sst", "Temperature"),
    ("sss_psu", "sss", "Salinity"),
    (
        "oxygen_mg_l",
        "oxygen",
        "Oxygen",
    ),
]

for value, prefix, label in variables:

    print("\n" + label)

    matched = out[value].notna()

    print(
        "Matched:",
        int(matched.sum()),
        "/",
        len(out)
    )

    dated = out["date"].notna()

    print(
        "Matched among dated:",
        int((matched & dated).sum()),
        "/",
        int(dated.sum()),
        f"({(matched & dated).sum()/dated.sum()*100:.1f}%)"
    )

    print("\nQuality:")
    print(
        out.loc[
            matched,
            f"{prefix}_match_quality"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSources:")
    print(
        out.loc[
            matched,
            f"{prefix}_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDistance median/max (km):",
        round(
            out.loc[
                matched,
                f"{prefix}_match_distance_km"
            ].median(),
            2,
        ),
        "/",
        round(
            out.loc[
                matched,
                f"{prefix}_match_distance_km"
            ].max(),
            2,
        ),
    )


print("\n" + "=" * 90)
print("COMPLETE OBSERVED CORE PREDICTORS")
print("=" * 90)

complete = (
    out["sst_c"].notna()
    & out["sss_psu"].notna()
    & out["oxygen_mg_l"].notna()
)

print(
    "SST + salinity + oxygen:",
    int(complete.sum()),
    "/",
    len(out)
)

print(
    "Among dated records:",
    int(
        (
            complete
            & out["date"].notna()
        ).sum()
    ),
    "/",
    int(out["date"].notna().sum())
)

print("\nSaved:")
print(OUTPUT)
