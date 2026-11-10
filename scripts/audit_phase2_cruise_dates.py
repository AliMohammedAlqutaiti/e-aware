from pathlib import Path
import pandas as pd
import re
import unicodedata

ROOT = Path("data/imarpe/raw_acoustic")
INV = Path("data/imarpe/phase2_acoustic_inventory.csv")
OUTPUT = Path("data/imarpe/phase2_cruise_date_audit.csv")


def norm(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    x = "".join(
        c for c in unicodedata.normalize("NFKD", x)
        if not unicodedata.combining(c)
    )

    return re.sub(r"[^a-z0-9]+", "", x)


def numeric(s):
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce"
    )


def find_date_column(columns):
    candidates = []

    for c in columns:
        n = norm(c)

        if (
            n in {"date", "datee", "fecha", "fechae"}
            or n.startswith("fecha")
            or n.startswith("date")
        ):
            candidates.append(c)

    return candidates[0] if candidates else None


def parse_dates(series):
    raw = series.astype(str).str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    # ---------------------------------
    # 1. Numeric date codes
    #    YYYYMMDD e.g. 20011007
    #    YYMMDD   e.g. 970421
    # ---------------------------------
    clean = (
        raw
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )

    mask8 = clean.str.fullmatch(r"\d{8}", na=False)

    result.loc[mask8] = pd.to_datetime(
        clean.loc[mask8],
        format="%Y%m%d",
        errors="coerce"
    )

    mask6 = (
        result.isna()
        & clean.str.fullmatch(r"\d{6}", na=False)
    )

    result.loc[mask6] = pd.to_datetime(
        clean.loc[mask6],
        format="%y%m%d",
        errors="coerce"
    )

    # ---------------------------------
    # 2. Normal textual dates
    # ---------------------------------
    remaining = result.isna()

    if remaining.any():
        result.loc[remaining] = pd.to_datetime(
            raw.loc[remaining],
            errors="coerce",
            dayfirst=True
        )

    # ---------------------------------
    # 3. Excel serial dates
    # e.g. ~35000-45000
    # ---------------------------------
    numeric_values = pd.to_numeric(
        series,
        errors="coerce"
    )

    excel_mask = (
        result.isna()
        & numeric_values.between(20000, 60000)
    )

    if excel_mask.any():
        result.loc[excel_mask] = pd.to_datetime(
            numeric_values.loc[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    return result


inv = pd.read_csv(INV)

good = inv[
    (inv["status"] == "OK")
    & (inv["corridor_rows"].fillna(0) > 0)
].copy()


rows = []

print("=" * 95)
print("E-AWARE PHASE 2 CRUISE / DATE AUDIT")
print("=" * 95)


for file_name, group in good.groupby("file"):

    path = ROOT / file_name
    xls = pd.ExcelFile(path)

    all_dates = []
    corridor_rows = 0
    positives = 0
    zeros = 0

    sheets_used = []

    for _, r in group.iterrows():

        sheet = r["sheet"]
        header = int(r["header_row"]) - 1

        df = pd.read_excel(
            xls,
            sheet_name=sheet,
            header=header
        )

        lat_col = r["lat_col"]
        lon_col = r["lon_col"]
        anch_col = r["anch_col"]

        if any(
            c not in df.columns
            for c in [lat_col, lon_col, anch_col]
        ):
            continue

        lat = numeric(df[lat_col])
        lon = numeric(df[lon_col])
        anch = numeric(df[anch_col])

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

        mask = (
            lat.between(-5.00, -4.20)
            & lon.between(-82.00, -81.10)
        )

        if not mask.any():
            continue

        sheets_used.append(sheet)

        corridor_rows += int(mask.sum())
        positives += int((anch[mask] > 0).sum())
        zeros += int((anch[mask] <= 0).sum())

        date_col = find_date_column(df.columns)

        if date_col is not None:

            dates = parse_dates(df.loc[mask, date_col])

            dates = dates[
                (dates.dt.year >= 1980)
                & (dates.dt.year <= 2020)
            ]

            all_dates.extend(
                dates.dropna().tolist()
            )

    if all_dates:

        min_date = min(all_dates)
        max_date = max(all_dates)

        unique_dates = len(
            pd.Series(all_dates).dt.date.unique()
        )

    else:

        min_date = pd.NaT
        max_date = pd.NaT
        unique_dates = 0

    rows.append({
        "file": file_name,
        "sheets_used": " | ".join(sheets_used),
        "raw_corridor_rows": corridor_rows,
        "positive": positives,
        "zero_or_negative": zeros,
        "date_min": min_date,
        "date_max": max_date,
        "unique_dates": unique_dates,
    })


out = pd.DataFrame(rows)

out = out.sort_values(
    ["date_min", "file"],
    na_position="last"
)

out.to_csv(
    OUTPUT,
    index=False
)


print(
    out[
        [
            "file",
            "raw_corridor_rows",
            "positive",
            "zero_or_negative",
            "date_min",
            "date_max",
            "unique_dates",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 95)
print("SUMMARY")
print("=" * 95)

print("Workbooks with corridor data:", len(out))
print(
    "Workbooks with usable dates:",
    out["date_min"].notna().sum()
)
print(
    "Workbooks without usable dates:",
    out["date_min"].isna().sum()
)

print("\nSaved:")
print(OUTPUT)
