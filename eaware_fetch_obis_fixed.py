#!/usr/bin/env python3

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API = "https://api.obis.org/v3/occurrence"
SCIENTIFIC_NAME = "Engraulis ringens"
OUT_DIR = Path("obis_output")

STAGES = {
    "stageA_study_box": (-81.50, -81.20, -4.60, -4.36),
    "stageB_northern_peru": (-82.50, -80.00, -6.50, -3.40),
    "stageC_peru_wide": (-86.00, -70.00, -18.50, -3.40),
}

PAGE_SIZE = 5000

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "E-AWARE academic prototype - University of Portsmouth",
}


def make_polygon(lon_min, lon_max, lat_min, lat_max):
    return (
        f"POLYGON(("
        f"{lon_min} {lat_min},"
        f"{lon_max} {lat_min},"
        f"{lon_max} {lat_max},"
        f"{lon_min} {lat_max},"
        f"{lon_min} {lat_min}"
        f"))"
    )


def parse_absence(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "absent", "absence"}:
        return True

    if text in {"false", "0", "no", "present", "presence"}:
        return False

    return None


def presence_from_record(record):
    status = str(
        record.get("occurrenceStatus", "")
    ).strip().lower()

    if status in {"present", "presence"}:
        return 1

    if status in {"absent", "absence"}:
        return 0

    absence = parse_absence(
        record.get("absence")
    )

    if absence is True:
        return 0

    if absence is False:
        return 1

    return pd.NA


def fetch_stage(name, box):
    geometry = make_polygon(*box)

    params = {
        "scientificname": SCIENTIFIC_NAME,
        "geometry": geometry,
        "absence": "include",
        "size": PAGE_SIZE,
    }

    all_records = []
    cursor = None
    reported_total = None

    print()
    print("=" * 70)
    print(name)
    print(
        f"Longitude: {box[0]} to {box[1]} | "
        f"Latitude: {box[2]} to {box[3]}"
    )
    print("=" * 70)

    while True:
        if cursor:
            params["after"] = cursor

        response = requests.get(
            API,
            params=params,
            headers=HEADERS,
            timeout=90,
        )

        response.raise_for_status()

        payload = response.json()

        if reported_total is None:
            reported_total = payload.get("total")

        batch = payload.get("results", [])

        if not batch:
            break

        all_records.extend(batch)

        print(
            f"Downloaded {len(all_records)} records..."
        )

        new_cursor = batch[-1].get("id")

        if not new_cursor:
            break

        if new_cursor == cursor:
            print("Pagination cursor repeated. Stopping safely.")
            break

        cursor = new_cursor

        if len(batch) < PAGE_SIZE:
            break

        time.sleep(0.5)

    print(
        f"OBIS reported total: {reported_total}"
    )
    print(
        f"Actually downloaded: {len(all_records)}"
    )

    return all_records, geometry


def clean_records(records, stage_name):
    rows = []

    for record in records:
        event_date = record.get("eventDate")
        lat = record.get("decimalLatitude")
        lon = record.get("decimalLongitude")

        if not event_date:
            continue

        first_date = str(event_date).split("/")[0][:10]

        parsed_date = pd.to_datetime(
            first_date,
            errors="coerce",
        )

        if pd.isna(parsed_date):
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        if not (-90 <= lat <= 90):
            continue

        if not (-180 <= lon <= 180):
            continue

        rows.append(
            {
                "date": parsed_date.strftime("%Y-%m-%d"),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "presence": presence_from_record(record),
                "dataset_id": record.get("dataset_id"),
                "dataset_name": record.get("datasetName"),
                "event_id": record.get("eventID"),
                "occurrence_id": record.get("occurrenceID"),
                "obis_id": record.get("id"),
                "basis_of_record": record.get("basisOfRecord"),
                "coordinate_uncertainty_m": record.get(
                    "coordinateUncertaintyInMeters"
                ),
                "obis_stage": stage_name,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "latitude",
                "longitude",
                "presence",
                "dataset_id",
                "dataset_name",
                "event_id",
                "occurrence_id",
                "obis_id",
                "basis_of_record",
                "coordinate_uncertainty_m",
                "obis_stage",
            ]
        )

    df = pd.DataFrame(rows)

    df["presence"] = pd.array(
        df["presence"],
        dtype="Int64",
    )

    df = df.drop_duplicates(
        subset=[
            "date",
            "latitude",
            "longitude",
            "presence",
            "dataset_id",
        ]
    )

    df = df.sort_values(
        [
            "date",
            "latitude",
            "longitude",
        ]
    ).reset_index(drop=True)

    return df


def main():
    OUT_DIR.mkdir(
        exist_ok=True
    )

    log_lines = []

    def log(message):
        print(message)
        log_lines.append(str(message))

    log(
        f"E-AWARE OBIS fetch started: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

    log(
        f"Species: {SCIENTIFIC_NAME}"
    )

    all_absences = []

    for stage_name, box in STAGES.items():
        try:
            records, geometry = fetch_stage(
                stage_name,
                box,
            )

            clean = clean_records(
                records,
                stage_name,
            )

            presence_count = int(
                (clean["presence"] == 1).sum()
            )

            absence_count = int(
                (clean["presence"] == 0).sum()
            )

            unknown_count = int(
                clean["presence"].isna().sum()
            )

            log("")
            log(f"{stage_name}")
            log(f"Geometry: {geometry}")
            log(f"Clean records: {len(clean)}")
            log(f"Presence: {presence_count}")
            log(f"Absence: {absence_count}")
            log(f"Unknown target: {unknown_count}")

            if not clean.empty:
                log(
                    f"Date range: "
                    f"{clean['date'].min()} "
                    f"to "
                    f"{clean['date'].max()}"
                )

            output_path = (
                OUT_DIR
                / f"obis_{stage_name}.csv"
            )

            clean.to_csv(
                output_path,
                index=False,
            )

            log(
                f"Saved: {output_path}"
            )

            absence_rows = clean[
                clean["presence"] == 0
            ].copy()

            if not absence_rows.empty:
                all_absences.append(
                    absence_rows
                )

        except Exception as exc:
            log("")
            log(
                f"{stage_name} FAILED: {exc}"
            )

    if all_absences:
        absence_review = pd.concat(
            all_absences,
            ignore_index=True,
        ).drop_duplicates()

        absence_path = (
            OUT_DIR
            / "obis_absence_review.csv"
        )

        absence_review.to_csv(
            absence_path,
            index=False,
        )

        log("")
        log(
            "OBIS absence records were found."
        )
        log(
            "Do NOT use them for final model training "
            "until the source methodology is reviewed."
        )

    else:
        log("")
        log(
            "No explicit OBIS absences were found."
        )
        log(
            "Treat these records as presence-only pipeline data."
        )

    log_path = (
        OUT_DIR
        / "obis_fetch_log.txt"
    )

    log_path.write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    log("")
    log(
        f"Log saved to: {log_path}"
    )


if __name__ == "__main__":
    main()
