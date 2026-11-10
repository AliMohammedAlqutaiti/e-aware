from pathlib import Path
from itertools import combinations

import pandas as pd


ROOT = Path("data/imarpe/raw_acoustic")
INV = Path("data/imarpe/phase2_acoustic_inventory.csv")

LAT_MIN, LAT_MAX = -5.00, -4.20
LON_MIN, LON_MAX = -82.00, -81.10


def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip(),
        errors="coerce",
    )


inv = pd.read_csv(INV)

good = inv[
    (inv["status"] == "OK")
    & (inv["corridor_rows"].fillna(0) > 0)
].copy()

datasets = {}
summary = []

for _, r in good.iterrows():

    path = ROOT / r["file"]
    sheet = r["sheet"]
    header = int(r["header_row"]) - 1

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=header,
    )

    lat_col = r["lat_col"]
    lon_col = r["lon_col"]
    anch_col = r["anch_col"]

    if (
        lat_col not in df.columns
        or lon_col not in df.columns
        or anch_col not in df.columns
    ):
        continue

    lat = numeric(df[lat_col])
    lon = numeric(df[lon_col])
    anch = numeric(df[anch_col])

    if len(lat.dropna()) and lat.dropna().between(0, 10).mean() > 0.8:
        lat = -lat

    if len(lon.dropna()) and lon.dropna().between(70, 90).mean() > 0.8:
        lon = -lon

    mask = (
        lat.between(LAT_MIN, LAT_MAX)
        & lon.between(LON_MIN, LON_MAX)
    )

    sub = pd.DataFrame({
        "lat": lat[mask],
        "lon": lon[mask],
        "anch": anch[mask],
    }).reset_index(drop=True)

    # Rounded key only for duplicate diagnostics.
    sub["key"] = (
        sub["lat"].round(5).astype(str)
        + "|"
        + sub["lon"].round(5).astype(str)
        + "|"
        + sub["anch"].round(6).astype(str)
    )

    key = (r["file"], sheet)
    datasets[key] = sub

    valid_anch = sub["anch"].notna().sum()
    positive = (sub["anch"] > 0).sum()
    zeros = (sub["anch"] == 0).sum()

    prevalence = (
        positive / valid_anch
        if valid_anch
        else float("nan")
    )

    summary.append({
        "file": r["file"],
        "sheet": sheet,
        "rows": len(sub),
        "anch_col": anch_col,
        "positive": int(positive),
        "zero": int(zeros),
        "missing": int(sub["anch"].isna().sum()),
        "positive_rate": prevalence,
        "unique_keys": sub["key"].nunique(),
    })


summary = pd.DataFrame(summary)

print("=" * 80)
print("PHASE 2 SHEET SUMMARY")
print("=" * 80)

print(
    summary.sort_values(
        ["positive_rate", "rows"],
        ascending=[False, False],
    ).to_string(
        index=False,
        formatters={
            "positive_rate": lambda x: f"{x:.1%}"
        }
    )
)


print("\n" + "=" * 80)
print("SUSPICIOUS / SPECIAL-CASE SHEETS")
print("=" * 80)

special = summary[
    (summary["positive_rate"] >= 0.95)
    | (summary["missing"] > 0)
]

if len(special):
    print(
        special.to_string(
            index=False,
            formatters={
                "positive_rate": lambda x: f"{x:.1%}"
            }
        )
    )
else:
    print("None")


print("\n" + "=" * 80)
print("HIGH-OVERLAP SHEETS WITHIN SAME WORKBOOK")
print("=" * 80)

found = False

for file_name, group in summary.groupby("file"):

    sheets = group["sheet"].tolist()

    if len(sheets) < 2:
        continue

    for s1, s2 in combinations(sheets, 2):

        a = set(datasets[(file_name, s1)]["key"])
        b = set(datasets[(file_name, s2)]["key"])

        if not a or not b:
            continue

        overlap = len(a & b)
        smaller = min(len(a), len(b))

        overlap_pct = overlap / smaller

        if overlap_pct >= 0.50:

            found = True

            print("\nFILE:", file_name)
            print("Sheet A:", s1, "unique rows:", len(a))
            print("Sheet B:", s2, "unique rows:", len(b))
            print("Exact overlapping rows:", overlap)
            print(
                "Overlap vs smaller sheet:",
                f"{overlap_pct:.1%}"
            )

if not found:
    print("No >=50% overlaps found.")


print("\n" + "=" * 80)
print("WORKBOOK-LEVEL UNIQUE-ROW ESTIMATE")
print("=" * 80)

total_raw = 0
total_unique = 0

for file_name, group in summary.groupby("file"):

    frames = []

    for sheet in group["sheet"]:
        frames.append(
            datasets[(file_name, sheet)][
                ["lat", "lon", "anch", "key"]
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    raw = len(combined)
    unique = combined["key"].nunique()

    total_raw += raw
    total_unique += unique

    print(
        f"{file_name[:58]:58s} "
        f"raw={raw:4d} "
        f"unique={unique:4d} "
        f"possible_duplicates={raw-unique:4d}"
    )

print("\nRaw rows:", total_raw)
print(
    "Workbook-deduplicated estimate:",
    total_unique
)
print(
    "Possible duplicate rows:",
    total_raw - total_unique
)
