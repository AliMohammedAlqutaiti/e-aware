from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path("data/imarpe")
XLSX_ROOT = ROOT / "raw_acoustic"
INVENTORY = ROOT / "phase2_acoustic_inventory.csv"

OUTPUT = ROOT / "E_AWARE_IMARPE_phase2_acoustic_master.csv"
SUMMARY_OUTPUT = ROOT / "E_AWARE_IMARPE_phase2_acoustic_summary.csv"

LAT_MIN, LAT_MAX = -5.00, -4.20
LON_MIN, LON_MAX = -82.00, -81.10

OLD_CSVS = [
    "acoustic_9002-04.csv",
    "acoustic_9011-9102.csv",
    "acoustic_9201-03.csv",
    "acoustic_9301-03.csv",
    "acoustic_9502-04.csv",
    "acoustic_9602-04.csv",
]


def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalise_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
    )


def find_date_column(columns):
    # Prefer end-date because our acoustic coordinates
    # are commonly Lon_E / Lat_E in IMARPE workbooks.
    priority = [
        "datee",
        "fechae",
        "datem",
        "fecham",
        "dates",
        "fechas",
        "date",
        "fecha",
    ]

    normalized = {
        normalise_name(col): col
        for col in columns
    }

    for target in priority:
        if target in normalized:
            return normalized[target]

    # Fallback for less-standard labels.
    for col in columns:
        n = normalise_name(col)

        if (
            n.startswith("fecha")
            or n.startswith("date")
        ):
            return col

    return None


def find_time_column(columns):
    for col in columns:
        n = normalise_name(col)

        if n in {
            "time",
            "timee",
            "hora",
            "horae",
        }:
            return col

    return None


def parse_date_value(value):
    if pd.isna(value):
        return pd.NaT

    # Already date-like
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).normalize()

    raw = str(value).strip()

    if not raw:
        return pd.NaT

    # Remove Excel-style trailing .0
    if raw.endswith(".0"):
        raw = raw[:-2]

    digits = "".join(
        ch for ch in raw
        if ch.isdigit()
    )

    # YYYYMMDD
    if len(digits) == 8:
        try:
            return pd.Timestamp(
                datetime.strptime(
                    digits,
                    "%Y%m%d"
                )
            )
        except ValueError:
            pass

    # YYMMDD
    if len(digits) == 6:
        try:
            return pd.Timestamp(
                datetime.strptime(
                    digits,
                    "%y%m%d"
                )
            )
        except ValueError:
            pass

    # Excel serial date
    try:
        number = float(raw)

        if 20000 <= number <= 60000:
            return (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(
                    number,
                    unit="D"
                )
            ).normalize()

    except Exception:
        pass

    # Last resort for textual dates
    try:
        return pd.to_datetime(
            raw,
            dayfirst=True,
            errors="raise",
        ).normalize()

    except Exception:
        return pd.NaT


def parse_date_series(series):
    return series.apply(parse_date_value)


def correct_coordinates(lat, lon):

    if (
        lat.notna().any()
        and lat.dropna().between(0, 10).mean() > 0.8
    ):
        lat = -lat

    if (
        lon.notna().any()
        and lon.dropna().between(70, 90).mean() > 0.8
    ):
        lon = -lon

    return lat, lon


def collapse_duplicates(df):

    if df.empty:
        return df

    df = df.copy()

    # Acoustic-cell key.
    df["base_key"] = (
        df["lat"].round(5).astype(str)
        + "|"
        + df["lon"].round(5).astype(str)
        + "|"
        + df["anchoveta_acoustic_raw"].round(6).astype(str)
    )

    kept = []

    for (_, key), g in df.groupby(
        ["source_group", "base_key"],
        sort=False,
        dropna=False,
    ):

        g = g.copy()

        sheets = " | ".join(
            sorted(
                set(
                    g["source_sheet"]
                    .dropna()
                    .astype(str)
                )
            )
        )

        distinct_dates = (
            g["date"]
            .dropna()
            .dt.normalize()
            .drop_duplicates()
            .tolist()
        )

        # Usually identical records copied across worksheets.
        if len(distinct_dates) <= 1:

            g["metadata_score"] = (
                g["date"].notna().astype(int) * 2
                + g["time_raw"].notna().astype(int)
            )

            best = (
                g.sort_values(
                    "metadata_score",
                    ascending=False
                )
                .iloc[0]
                .copy()
            )

            best["duplicate_source_rows"] = len(g)
            best["source_sheets_all"] = sheets

            kept.append(best)

        # Same exact acoustic cell/value occurs on different
        # survey dates: retain one record per actual date.
        else:

            dated = g[g["date"].notna()].copy()

            for date, dg in dated.groupby(
                dated["date"].dt.normalize(),
                sort=False,
            ):

                dg["metadata_score"] = (
                    dg["time_raw"].notna().astype(int)
                )

                best = (
                    dg.sort_values(
                        "metadata_score",
                        ascending=False
                    )
                    .iloc[0]
                    .copy()
                )

                best["duplicate_source_rows"] = len(dg)
                best["source_sheets_all"] = sheets

                kept.append(best)

    out = pd.DataFrame(kept)

    drop_cols = [
        "base_key",
        "metadata_score",
    ]

    out = out.drop(
        columns=[
            c for c in drop_cols
            if c in out.columns
        ]
    )

    return out


# ============================================================
# 1. RAW XLSX ACOUSTIC ARCHIVE
# ============================================================

inventory = pd.read_csv(INVENTORY)

usable = inventory[
    (inventory["status"] == "OK")
    & (inventory["corridor_rows"].fillna(0) > 0)
].copy()

xlsx_records = []

print("=" * 90)
print("BUILDING PHASE 2 ACOUSTIC MASTER")
print("=" * 90)
print("\nReading XLSX archive...")


for file_name, group in usable.groupby("file"):

    path = XLSX_ROOT / file_name

    print(" -", file_name)

    xls = pd.ExcelFile(path)

    for _, spec in group.iterrows():

        sheet = spec["sheet"]
        header = int(spec["header_row"]) - 1

        df = pd.read_excel(
            xls,
            sheet_name=sheet,
            header=header,
        )

        lat_col = spec["lat_col"]
        lon_col = spec["lon_col"]
        anch_col = spec["anch_col"]

        if any(
            col not in df.columns
            for col in [
                lat_col,
                lon_col,
                anch_col,
            ]
        ):
            continue

        lat = numeric(df[lat_col])
        lon = numeric(df[lon_col])
        anch = numeric(df[anch_col])

        lat, lon = correct_coordinates(
            lat,
            lon,
        )

        corridor = (
            lat.between(LAT_MIN, LAT_MAX)
            & lon.between(LON_MIN, LON_MAX)
        )

        date_col = find_date_column(df.columns)
        time_col = find_time_column(df.columns)

        if date_col is not None:
            dates = parse_date_series(
                df[date_col]
            )
        else:
            dates = pd.Series(
                pd.NaT,
                index=df.index,
                dtype="datetime64[ns]",
            )

        if time_col is not None:
            time_raw = df[time_col].astype(object)
        else:
            time_raw = pd.Series(
                None,
                index=df.index,
                dtype=object,
            )

        source_row = (
            pd.Series(
                df.index,
                index=df.index
            )
            + header
            + 2
        )

        part = pd.DataFrame({
            "date": dates[corridor],
            "time_raw": time_raw[corridor],
            "lat": lat[corridor],
            "lon": lon[corridor],
            "anchoveta_acoustic_raw": anch[corridor],
            "source_row": source_row[corridor],
        })

        # Only rows with an actual target measurement.
        part = part[
            part["anchoveta_acoustic_raw"].notna()
        ].copy()

        if part.empty:
            continue

        part["source_type"] = "xlsx_acoustic"
        part["source_file"] = file_name
        part["source_sheet"] = sheet

        # Workbook is our safest independent survey group.
        part["source_group"] = (
            "xlsx::" + file_name
        )

        xlsx_records.append(part)


xlsx_raw = pd.concat(
    xlsx_records,
    ignore_index=True,
)

print(
    "\nRaw labelled XLSX corridor rows:",
    len(xlsx_raw)
)

xlsx_clean = collapse_duplicates(
    xlsx_raw
)

print(
    "XLSX rows after within-workbook deduplication:",
    len(xlsx_clean)
)


# ============================================================
# 2. OLDER PROCESSED CSV CRUISES
# ============================================================

csv_records = []

print("\nReading historical processed CSVs...")


for name in OLD_CSVS:

    path = ROOT / name

    if not path.exists():
        print(" MISSING:", name)
        continue

    print(" -", name)

    df = pd.read_csv(path)

    lat_col = "lat"
    lon_col = "long"
    anch_col = "anch"

    if any(
        c not in df.columns
        for c in [lat_col, lon_col, anch_col]
    ):
        print(
            "   skipped: expected columns missing"
        )
        continue

    lat = numeric(df[lat_col])
    lon = numeric(df[lon_col])
    anch = numeric(df[anch_col])

    lat, lon = correct_coordinates(
        lat,
        lon,
    )

    corridor = (
        lat.between(LAT_MIN, LAT_MAX)
        & lon.between(LON_MIN, LON_MAX)
    )

    date_col = find_date_column(df.columns)
    time_col = find_time_column(df.columns)

    if date_col is not None:
        dates = parse_date_series(
            df[date_col]
        )
    else:
        dates = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns]",
        )

    if time_col is not None:
        time_raw = df[time_col].astype(object)
    else:
        time_raw = pd.Series(
            None,
            index=df.index,
            dtype=object,
        )

    part = pd.DataFrame({
        "date": dates[corridor],
        "time_raw": time_raw[corridor],
        "lat": lat[corridor],
        "lon": lon[corridor],
        "anchoveta_acoustic_raw": anch[corridor],
        "source_row": (
            pd.Series(
                df.index,
                index=df.index
            )[corridor]
            + 2
        ),
    })

    part = part[
        part["anchoveta_acoustic_raw"].notna()
    ].copy()

    if part.empty:
        continue

    cruise = (
        Path(name)
        .stem
        .replace(
            "acoustic_",
            ""
        )
    )

    part["source_type"] = "csv_processed"
    part["source_file"] = name
    part["source_sheet"] = "csv"
    part["source_group"] = (
        "csv::" + cruise
    )

    csv_records.append(part)


csv_raw = pd.concat(
    csv_records,
    ignore_index=True,
)

print(
    "\nRaw labelled CSV corridor rows:",
    len(csv_raw)
)

csv_clean = collapse_duplicates(
    csv_raw
)

print(
    "CSV rows after within-file deduplication:",
    len(csv_clean)
)


# ============================================================
# 3. COMBINE
# ============================================================

master = pd.concat(
    [
        xlsx_clean,
        csv_clean,
    ],
    ignore_index=True,
)

master["anchoveta_acoustic"] = (
    master["anchoveta_acoustic_raw"]
    .clip(lower=0)
)

master["negative_acoustic_artifact"] = (
    master["anchoveta_acoustic_raw"] < 0
)

master["presence"] = (
    master["anchoveta_acoustic_raw"] > 0
).astype(int)

master["date_status"] = np.where(
    master["date"].notna(),
    "exact",
    "unknown",
)

# Copernicus GLOBAL_MULTIYEAR coverage begins in 1993.
master["copernicus_eligible"] = (
    master["date"].notna()
    & (master["date"] >= pd.Timestamp("1993-01-01"))
)

master = master.sort_values(
    [
        "source_group",
        "date",
        "lat",
        "lon",
    ],
    na_position="last",
).reset_index(drop=True)

master.insert(
    0,
    "record_id",
    [
        f"EA2-{i:05d}"
        for i in range(
            1,
            len(master) + 1
        )
    ],
)

master.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# 4. SUMMARY
# ============================================================

summary = (
    master
    .groupby(
        [
            "source_type",
            "source_group",
        ],
        dropna=False,
    )
    .agg(
        rows=("record_id", "size"),
        detections=("presence", "sum"),
        exact_dates=(
            "date",
            lambda x: x.notna().sum()
        ),
        date_min=("date", "min"),
        date_max=("date", "max"),
        negative_artifacts=(
            "negative_acoustic_artifact",
            "sum"
        ),
    )
    .reset_index()
)

summary["non_detections"] = (
    summary["rows"]
    - summary["detections"]
)

summary["detection_rate"] = (
    summary["detections"]
    / summary["rows"]
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print("\n" + "=" * 90)
print("PHASE 2 MASTER SUMMARY")
print("=" * 90)

print("Total records:", len(master))
print(
    "Detections:",
    int(master["presence"].sum())
)
print(
    "Non-detections:",
    int((master["presence"] == 0).sum())
)
print(
    "Detection rate:",
    round(
        master["presence"].mean() * 100,
        1
    ),
    "%"
)

print(
    "Records with exact date:",
    int(master["date"].notna().sum())
)

print(
    "Records with unknown date:",
    int(master["date"].isna().sum())
)

print(
    "Copernicus eligible:",
    int(master["copernicus_eligible"].sum())
)

print(
    "Negative acoustic artefacts:",
    int(
        master[
            "negative_acoustic_artifact"
        ].sum()
    )
)

print(
    "Independent source groups:",
    master["source_group"].nunique()
)

print("\n" + "=" * 90)
print("BY SURVEY / SOURCE GROUP")
print("=" * 90)

display = summary.copy()

display["detection_rate"] = (
    display["detection_rate"] * 100
).round(1)

print(
    display[
        [
            "source_group",
            "rows",
            "detections",
            "non_detections",
            "detection_rate",
            "exact_dates",
            "date_min",
            "date_max",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)
print(SUMMARY_OUTPUT)
