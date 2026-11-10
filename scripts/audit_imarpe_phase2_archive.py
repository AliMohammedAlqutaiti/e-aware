from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path("data/imarpe/raw_acoustic")
OUTPUT = Path("data/imarpe/phase2_acoustic_inventory.csv")

LAT_MIN = -5.00
LAT_MAX = -4.20
LON_MIN = -82.00
LON_MAX = -81.10


def norm(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    return re.sub(r"[^a-z0-9]+", "", text)


def is_lat(text):
    return (
        text in {
            "lat",
            "late",
            "latc",
            "latcorr",
        }
        or text.startswith("latitud")
        or text.startswith("latitude")
    )


def is_lon(text):
    return (
        text in {
            "lon",
            "lone",
            "lonc",
            "loncorr",
            "long",
        }
        or text.startswith("longitud")
        or text.startswith("longitude")
    )


def is_anch(text):
    return (
        text == "anc"
        or text == "anch"
        or text.startswith("anchov")
    )


def find_header(raw):
    for idx, row in raw.iterrows():
        values = [norm(v) for v in row.values]

        has_lat = any(is_lat(v) for v in values)
        has_lon = any(is_lon(v) for v in values)
        has_anch = any(is_anch(v) for v in values)

        if has_lat and has_lon and has_anch:
            return idx

    return None


def find_column(columns, test):
    candidates = []

    for col in columns:
        n = norm(col)

        if test(n):
            candidates.append(col)

    if not candidates:
        return None

    return candidates[0]


def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


results = []

files = sorted(ROOT.glob("*.xlsx"))

print("========================================")
print("E-AWARE IMARPE PHASE 2 ARCHIVE AUDIT")
print("========================================")
print("Workbooks:", len(files))
print()

for path in files:

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:
        results.append({
            "file": path.name,
            "sheet": "",
            "status": f"OPEN_ERROR: {exc}",
        })
        continue

    for sheet in xls.sheet_names:

        try:
            raw = pd.read_excel(
                path,
                sheet_name=sheet,
                header=None,
                nrows=50,
            )
        except Exception as exc:
            results.append({
                "file": path.name,
                "sheet": sheet,
                "status": f"READ_ERROR: {exc}",
            })
            continue

        header_row = find_header(raw)

        if header_row is None:
            results.append({
                "file": path.name,
                "sheet": sheet,
                "status": "NO_HEADER_MATCH",
            })
            continue

        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet,
                header=header_row,
            )
        except Exception as exc:
            results.append({
                "file": path.name,
                "sheet": sheet,
                "status": f"PARSE_ERROR: {exc}",
            })
            continue

        lat_col = find_column(df.columns, is_lat)
        lon_col = find_column(df.columns, is_lon)
        anch_col = find_column(df.columns, is_anch)

        if lat_col is None or lon_col is None or anch_col is None:
            results.append({
                "file": path.name,
                "sheet": sheet,
                "header_row": header_row + 1,
                "status": "COLUMN_MATCH_FAILED",
                "lat_col": lat_col,
                "lon_col": lon_col,
                "anch_col": anch_col,
            })
            continue

        lat = numeric(df[lat_col])
        lon = numeric(df[lon_col])
        anch = numeric(df[anch_col])

        # Some historical IMARPE files may store south/west
        # coordinates as positive magnitudes.
        if lat.dropna().between(0, 10).mean() > 0.8:
            lat = -lat

        if lon.dropna().between(70, 90).mean() > 0.8:
            lon = -lon

        valid_coordinates = lat.notna() & lon.notna()

        corridor = (
            valid_coordinates
            & lat.between(LAT_MIN, LAT_MAX)
            & lon.between(LON_MIN, LON_MAX)
        )

        corridor_anch = anch[corridor]

        detections = int((corridor_anch > 0).sum())
        zeros = int((corridor_anch == 0).sum())
        missing_anch = int(corridor_anch.isna().sum())

        results.append({
            "file": path.name,
            "sheet": sheet,
            "status": "OK",
            "header_row": header_row + 1,
            "lat_col": str(lat_col),
            "lon_col": str(lon_col),
            "anch_col": str(anch_col),
            "sheet_rows": len(df),
            "valid_coordinate_rows": int(valid_coordinates.sum()),
            "corridor_rows": int(corridor.sum()),
            "detections": detections,
            "zeros": zeros,
            "missing_anch": missing_anch,
        })


audit = pd.DataFrame(results)
audit.to_csv(OUTPUT, index=False)

print("Status counts:")
print(audit["status"].value_counts())

good = audit[
    (audit["status"] == "OK")
    & (audit["corridor_rows"].fillna(0) > 0)
].copy()

print("\n========================================")
print("PARSEABLE SHEETS WITH CORRIDOR DATA")
print("========================================")

if len(good) == 0:
    print("No parseable corridor sheets found.")
else:
    display_cols = [
        "file",
        "sheet",
        "corridor_rows",
        "detections",
        "zeros",
        "missing_anch",
    ]

    print(
        good[display_cols]
        .sort_values(
            ["detections", "corridor_rows"],
            ascending=False,
        )
        .to_string(index=False)
    )

print("\n========================================")
print("RAW CANDIDATE TOTALS")
print("========================================")

print("Parseable corridor sheets:", len(good))
print("Candidate rows:", int(good["corridor_rows"].sum()) if len(good) else 0)
print("Candidate detections:", int(good["detections"].sum()) if len(good) else 0)
print("Candidate zeros:", int(good["zeros"].sum()) if len(good) else 0)
print("Candidate missing anch:", int(good["missing_anch"].sum()) if len(good) else 0)

print("\nIMPORTANT:")
print(
    "These are audit totals only. Different worksheets may contain "
    "duplicate acoustic cells or frequencies, so they are NOT yet "
    "the final Phase 2 sample size."
)

print("\nSaved:")
print(OUTPUT)
