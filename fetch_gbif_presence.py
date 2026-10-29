#!/usr/bin/env python3

"""
E-AWARE V2 Step 4B
Public anchoveta presence-data builder using the official GBIF API.

This script:
- downloads public Engraulis ringens occurrences from Peru
- keeps records from 1993 onward
- requires coordinates
- requires an exact usable calendar date
- removes obvious bad coordinates
- applies a coordinate-uncertainty QC threshold
- removes duplicate date/location records
- outputs presence-only data for E-AWARE

IMPORTANT:
These are PRESENCE observations.
They are not biological absence data.
"""

from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd


API = "https://api.gbif.org/v1/occurrence/search"

SCIENTIFIC_NAME = "Engraulis ringens"

START_YEAR = 1993
END_YEAR = 2026

# Peru / Peruvian Pacific screening box
MIN_LAT = -18.5
MAX_LAT = -3.0
MIN_LON = -86.0
MAX_LON = -70.0

# Keep records with <=10 km reported uncertainty.
# Missing uncertainty is retained but flagged.
MAX_UNCERTAINTY_M = 10000

PAGE_SIZE = 300

OUTDIR = Path("data/public_occurrences")
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / "gbif_engraulis_ringens_peru_raw.csv"
CLEAN_OUT = OUTDIR / "anchoveta_public_presences.csv"
REJECT_OUT = OUTDIR / "gbif_rejected_records.csv"
QC_OUT = OUTDIR / "gbif_presence_qc_summary.csv"


def fetch_all():
    rows = []
    offset = 0

    params = {
        "scientificName": SCIENTIFIC_NAME,
        "country": "PE",
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "year": f"{START_YEAR},{END_YEAR}",
        "limit": PAGE_SIZE,
    }

    print("E-AWARE Step 4B — GBIF public occurrence builder")
    print()
    print("Species:", SCIENTIFIC_NAME)
    print("Country: Peru")
    print(f"Years: {START_YEAR}–{END_YEAR}")
    print()

    while True:
        params["offset"] = offset

        response = requests.get(
            API,
            params=params,
            timeout=60,
            headers={
                "User-Agent":
                    "E-AWARE University of Portsmouth academic prototype"
            },
        )

        response.raise_for_status()

        payload = response.json()

        results = payload.get("results", [])

        if offset == 0:
            print(
                "GBIF records reported:",
                payload.get("count", "unknown")
            )

        if not results:
            break

        for r in results:
            rows.append(
                {
                    "gbif_id":
                        r.get("key"),

                    "occurrence_id":
                        r.get("occurrenceID"),

                    "dataset_key":
                        r.get("datasetKey"),

                    "dataset_name":
                        r.get("datasetTitle"),

                    "scientific_name":
                        r.get("scientificName"),

                    "accepted_scientific_name":
                        r.get("acceptedScientificName"),

                    "basis_of_record":
                        r.get("basisOfRecord"),

                    "occurrence_status":
                        r.get("occurrenceStatus"),

                    "event_date":
                        r.get("eventDate"),

                    "year":
                        r.get("year"),

                    "month":
                        r.get("month"),

                    "day":
                        r.get("day"),

                    "lat":
                        r.get("decimalLatitude"),

                    "lon":
                        r.get("decimalLongitude"),

                    "coordinate_uncertainty_m":
                        r.get("coordinateUncertaintyInMeters"),

                    "country_code":
                        r.get("countryCode"),

                    "state_province":
                        r.get("stateProvince"),

                    "locality":
                        r.get("locality"),

                    "recorded_by":
                        r.get("recordedBy"),

                    "institution_code":
                        r.get("institutionCode"),

                    "collection_code":
                        r.get("collectionCode"),

                    "catalog_number":
                        r.get("catalogNumber"),

                    "issues":
                        "|".join(
                            r.get("issues", [])
                        ),
                }
            )

        offset += len(results)

        print(
            f"Downloaded {offset} record(s)..."
        )

        if payload.get("endOfRecords", False):
            break

        if offset >= 100000:
            print(
                "Stopped at GBIF search API limit."
            )
            break

        time.sleep(0.15)

    return pd.DataFrame(rows)


def build_exact_date(df):
    year = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    month = pd.to_numeric(
        df["month"],
        errors="coerce"
    )

    day = pd.to_numeric(
        df["day"],
        errors="coerce"
    )

    return pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": day,
        },
        errors="coerce",
    )


def main():
    raw = fetch_all()

    if raw.empty:
        raise RuntimeError(
            "GBIF returned zero records."
        )

    raw.to_csv(
        RAW_OUT,
        index=False
    )

    print()
    print(
        f"Raw records saved: {len(raw)}"
    )

    df = raw.copy()

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    df["coordinate_uncertainty_m"] = pd.to_numeric(
        df["coordinate_uncertainty_m"],
        errors="coerce"
    )

    df["date"] = build_exact_date(df)

    # ---------- QC FLAGS ----------

    df["qc_valid_coordinates"] = (
        df["lat"].between(-90, 90)
        &
        df["lon"].between(-180, 180)
    )

    df["qc_inside_peru_coastal_box"] = (
        df["lat"].between(
            MIN_LAT,
            MAX_LAT
        )
        &
        df["lon"].between(
            MIN_LON,
            MAX_LON
        )
    )

    df["qc_exact_date"] = (
        df["date"].notna()
    )

    df["qc_after_1993"] = (
        df["date"]
        >=
        pd.Timestamp("1993-01-01")
    )

    df["qc_uncertainty_ok"] = (
        df["coordinate_uncertainty_m"].isna()
        |
        (
            df["coordinate_uncertainty_m"]
            <=
            MAX_UNCERTAINTY_M
        )
    )

    df["qc_presence"] = (
        df["occurrence_status"]
        .astype(str)
        .str.upper()
        .eq("PRESENT")
    )

    # Reject obvious zero coordinates
    df["qc_not_zero_coordinate"] = ~(
        df["lat"].fillna(0).eq(0)
        &
        df["lon"].fillna(0).eq(0)
    )

    qc_cols = [
        "qc_valid_coordinates",
        "qc_inside_peru_coastal_box",
        "qc_exact_date",
        "qc_after_1993",
        "qc_uncertainty_ok",
        "qc_presence",
        "qc_not_zero_coordinate",
    ]

    df["qc_pass"] = (
        df[qc_cols]
        .all(axis=1)
    )

    rejected = df[
        ~df["qc_pass"]
    ].copy()

    clean = df[
        df["qc_pass"]
    ].copy()

    # ---------- DUPLICATE CONTROL ----------

    # Prefer records with smaller reported uncertainty.
    clean["_uncertainty_sort"] = (
        clean["coordinate_uncertainty_m"]
        .fillna(1e12)
    )

    clean = clean.sort_values(
        [
            "_uncertainty_sort",
            "gbif_id",
        ]
    )

    clean["lat_key"] = (
        clean["lat"]
        .round(4)
    )

    clean["lon_key"] = (
        clean["lon"]
        .round(4)
    )

    before_dedup = len(clean)

    clean = clean.drop_duplicates(
        subset=[
            "date",
            "lat_key",
            "lon_key",
        ],
        keep="first",
    ).copy()

    duplicates_removed = (
        before_dedup
        -
        len(clean)
    )

    clean.drop(
        columns=[
            "_uncertainty_sort",
            "lat_key",
            "lon_key",
        ],
        inplace=True,
        errors="ignore",
    )

    # ---------- E-AWARE STANDARD FIELDS ----------

    clean["presence"] = 1

    clean["sample_type"] = "presence"

    clean["source"] = "GBIF"

    clean["absence_semantics_verified"] = False

    clean["within_eaware_study_area"] = (
        clean["lat"].between(
            -4.60,
            -4.36
        )
        &
        clean["lon"].between(
            -81.50,
            -81.20
        )
    )

    output_cols = [
        "date",
        "lat",
        "lon",
        "presence",
        "sample_type",
        "source",
        "gbif_id",
        "occurrence_id",
        "dataset_key",
        "dataset_name",
        "basis_of_record",
        "coordinate_uncertainty_m",
        "event_date",
        "year",
        "month",
        "day",
        "institution_code",
        "collection_code",
        "catalog_number",
        "locality",
        "state_province",
        "within_eaware_study_area",
        "absence_semantics_verified",
        "issues",
    ]

    clean = clean[
        [
            c
            for c in output_cols
            if c in clean.columns
        ]
    ].sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    clean.to_csv(
        CLEAN_OUT,
        index=False
    )

    rejected.to_csv(
        REJECT_OUT,
        index=False
    )

    # ---------- QC SUMMARY ----------

    summary = [
        ("GBIF raw records", len(raw)),
        (
            "Records rejected by QC",
            len(rejected),
        ),
        (
            "Records before deduplication",
            before_dedup,
        ),
        (
            "Duplicates removed",
            duplicates_removed,
        ),
        (
            "Final public presence records",
            len(clean),
        ),
        (
            "Inside E-AWARE study area",
            int(
                clean[
                    "within_eaware_study_area"
                ].sum()
            )
            if len(clean)
            else 0,
        ),
        (
            "Verified absence records",
            0,
        ),
    ]

    pd.DataFrame(
        summary,
        columns=[
            "Metric",
            "Value",
        ]
    ).to_csv(
        QC_OUT,
        index=False
    )

    print()
    print("===== STEP 4B RESULTS =====")

    for metric, value in summary:
        print(
            f"{metric}: {value}"
        )

    if len(clean):
        print()
        print(
            "Date range:",
            clean["date"].min(),
            "→",
            clean["date"].max(),
        )

        print()
        print("Basis of record:")
        print(
            clean[
                "basis_of_record"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print(
        "Clean file:",
        CLEAN_OUT
    )

    print(
        "Rejected records:",
        REJECT_OUT
    )

    print(
        "QC summary:",
        QC_OUT
    )

    print()
    print(
        "IMPORTANT: these are observed "
        "PRESENCES only."
    )

    print(
        "No absence or background "
        "points were created."
    )


if __name__ == "__main__":
    main()
