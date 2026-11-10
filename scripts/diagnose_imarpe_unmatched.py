from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path("data/imarpe/raw_acoustic")
INVENTORY = Path("data/imarpe/phase2_acoustic_inventory.csv")
OUTPUT = Path("data/imarpe/phase2_unmatched_diagnosis.csv")


def norm(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    return re.sub(r"[^a-z0-9]+", "", text)


KEYWORDS = {
    "lat": [
        "lat",
        "latitud",
        "latitude",
        "latitudini",
    ],
    "lon": [
        "lon",
        "long",
        "longitud",
        "longitude",
    ],
    "anch": [
        "anc",
        "anch",
        "anchoveta",
        "anchov",
    ],
    "nasc": [
        "nasc",
        "sa",
        "ecointegracion",
        "ecointegracion",
    ],
    "date": [
        "fecha",
        "date",
        "dia",
    ],
    "depth": [
        "prof",
        "profundidad",
        "depth",
        "fondo",
    ],
}


def matches(text, options):
    return any(
        text == x or text.startswith(x)
        for x in options
    )


inventory = pd.read_csv(INVENTORY)

unmatched = inventory[
    inventory["status"] == "NO_HEADER_MATCH"
][["file", "sheet"]].drop_duplicates()

results = []

print("======================================")
print("IMARPE UNMATCHED SHEET DIAGNOSIS")
print("======================================")
print("Sheets:", len(unmatched))
print()

for _, item in unmatched.iterrows():
    path = ROOT / item["file"]
    sheet = item["sheet"]

    try:
        raw = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            nrows=40,
        )
    except Exception as exc:
        results.append({
            "file": item["file"],
            "sheet": sheet,
            "status": "READ_ERROR",
            "error": str(exc),
        })
        continue

    best_score = -1
    best_row = None
    best_found = {}
    best_values = ""

    if raw.empty:
        results.append({
            "file": item["file"],
            "sheet": sheet,
            "status": "EMPTY_SHEET",
            "best_row": None,
            "keyword_score": 0,
            "lat_found": False,
            "lon_found": False,
            "anch_found": False,
            "nasc_found": False,
            "date_found": False,
            "depth_found": False,
            "row_preview": "",
        })
        continue

    for idx, row in raw.iterrows():

        values = [
            norm(v)
            for v in row.values
            if not pd.isna(v)
        ]

        found = {}

        for category, options in KEYWORDS.items():
            found[category] = any(
                matches(v, options)
                for v in values
            )

        score = sum(found.values())

        if score > best_score:
            best_score = score
            best_row = idx + 1
            best_found = found

            original_values = [
                str(v).strip()
                for v in row.values
                if not pd.isna(v)
                and str(v).strip() != ""
            ]

            best_values = " | ".join(
                original_values[:18]
            )

    result = {
        "file": item["file"],
        "sheet": sheet,
        "best_row": best_row,
        "keyword_score": best_score,
        "lat_found": best_found.get("lat", False),
        "lon_found": best_found.get("lon", False),
        "anch_found": best_found.get("anch", False),
        "nasc_found": best_found.get("nasc", False),
        "date_found": best_found.get("date", False),
        "depth_found": best_found.get("depth", False),
        "row_preview": best_values,
    }

    results.append(result)


diagnosis = pd.DataFrame(results)

diagnosis.to_csv(
    OUTPUT,
    index=False
)

print("Keyword-score distribution:")
print(
    diagnosis["keyword_score"]
    .value_counts()
    .sort_index(ascending=False)
)

print("\n======================================")
print("MOST PROMISING UNMATCHED SHEETS")
print("======================================")

promising = diagnosis[
    diagnosis["keyword_score"] >= 2
].sort_values(
    ["keyword_score", "file"],
    ascending=[False, True],
)

if len(promising) == 0:
    print("No promising sheets detected.")
else:
    for _, row in promising.iterrows():
        print("\nFILE:", row["file"])
        print("SHEET:", row["sheet"])
        print("Candidate header row:", row["best_row"])
        print("Score:", row["keyword_score"])
        print("Preview:")
        print(row["row_preview"])

print("\nSaved:")
print(OUTPUT)
