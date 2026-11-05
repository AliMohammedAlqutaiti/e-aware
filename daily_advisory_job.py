"""
E-AWARE scheduled advisory sender.

The Streamlit application generates the actual advisory.
This script sends that advisory to active Supabase subscribers.
"""

import html
import json
import os
import smtplib

from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from supabase import create_client


PERU_TZ = ZoneInfo("America/Lima")

ADVISORY_FILE = Path(
    "/tmp/eaware_advisory.json"
)


def required_env(name):
    value = os.environ.get(
        name,
        ""
    ).strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def already_sent_today(last_sent_at):
    if not last_sent_at:
        return False

    try:
        value = str(
            last_sent_at
        ).replace(
            "Z",
            "+00:00"
        )

        sent = datetime.fromisoformat(
            value
        )

        if sent.tzinfo is None:
            sent = sent.replace(
                tzinfo=timezone.utc
            )

        return (
            sent.astimezone(
                PERU_TZ
            ).date()
            ==
            datetime.now(
                PERU_TZ
            ).date()
        )

    except Exception:
        return False


def mask_email(address):
    try:
        local, domain = address.split(
            "@",
            1
        )

        if len(local) <= 2:
            masked = local[0] + "*"
        else:
            masked = (
                local[0]
                +
                "***"
                +
                local[-1]
            )

        return f"{masked}@{domain}"

    except Exception:
        return "***"


def build_html(advisory):
    safe = html.escape(
        advisory
    )

    return f"""
    <div style="
        font-family:Arial,sans-serif;
        max-width:700px;
        margin:auto;
        color:#222;
    ">

        <h1>
            E-AWARE Daily Fisheries Advisory
        </h1>

        <p style="color:#666;">
            El Niño Adaptive Fisheries Intelligence
        </p>

        <div style="
            background:#f4f4f4;
            border-radius:12px;
            padding:22px;
            margin-top:20px;
        ">

            <pre style="
                white-space:pre-wrap;
                font-family:Arial,sans-serif;
                line-height:1.6;
                margin:0;
            ">{safe}</pre>

        </div>

        <p style="
            margin-top:24px;
            font-size:13px;
            color:#666;
        ">
            E-AWARE provides habitat decision support only.
            It does not provide legal fishing permission,
            exact fish locations or navigation instructions.
            Current PRODUCE regulations must be verified
            before fishing.
        </p>

    </div>
    """


def send_email(
    smtp,
    sender,
    recipient,
    subject,
    advisory
):
    msg = EmailMessage()

    msg["Subject"] = subject
    msg["From"] = f"E-AWARE <{sender}>"
    msg["To"] = recipient

    msg.set_content(
        advisory
    )

    msg.add_alternative(
        build_html(
            advisory
        ),
        subtype="html"
    )

    smtp.send_message(
        msg
    )


def main():

    if not ADVISORY_FILE.exists():
        raise RuntimeError(
            "E-AWARE did not generate "
            "/tmp/eaware_advisory.json"
        )

    payload = json.loads(
        ADVISORY_FILE.read_text(
            encoding="utf-8"
        )
    )

    advisory = payload[
        "advisory"
    ]

    data_date = payload.get(
        "data_date",
        "Current"
    )

    supabase_url = required_env(
        "SUPABASE_URL"
    )

    supabase_secret = required_env(
        "SUPABASE_SECRET_KEY"
    )

    gmail_sender = required_env(
        "GMAIL_SENDER"
    )

    gmail_password = required_env(
        "GMAIL_APP_PASSWORD"
    ).replace(
        " ",
        ""
    )

    # Manual workflow tests may send again on the same day.
    force_send = (
        os.environ.get(
            "FORCE_SEND",
            "0"
        )
        ==
        "1"
    )

    db = create_client(
        supabase_url,
        supabase_secret
    )

    result = (
        db
        .table(
            "eaware_subscribers"
        )
        .select(
            "id,email,last_sent_at"
        )
        .eq(
            "active",
            True
        )
        .execute()
    )

    subscribers = (
        result.data
        or []
    )

    print(
        f"Active subscribers: {len(subscribers)}"
    )

    if not subscribers:
        print(
            "No active subscribers. Nothing to send."
        )
        return

    sent_count = 0
    skipped_count = 0
    failed_count = 0

    subject = (
        f"E-AWARE Daily Fisheries Advisory — "
        f"{data_date}"
    )

    with smtplib.SMTP(
        "smtp.gmail.com",
        587,
        timeout=30
    ) as smtp:

        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        smtp.login(
            gmail_sender,
            gmail_password
        )

        for subscriber in subscribers:

            recipient = str(
                subscriber[
                    "email"
                ]
            ).strip()

            if (
                not force_send
                and
                already_sent_today(
                    subscriber.get(
                        "last_sent_at"
                    )
                )
            ):

                print(
                    "SKIPPED: "
                    + mask_email(
                        recipient
                    )
                )

                skipped_count += 1
                continue

            try:

                send_email(
                    smtp,
                    gmail_sender,
                    recipient,
                    subject,
                    advisory
                )

                (
                    db
                    .table(
                        "eaware_subscribers"
                    )
                    .update(
                        {
                            "last_sent_at":
                                datetime.now(
                                    timezone.utc
                                ).isoformat()
                        }
                    )
                    .eq(
                        "id",
                        subscriber[
                            "id"
                        ]
                    )
                    .execute()
                )

                print(
                    "SENT: "
                    + mask_email(
                        recipient
                    )
                )

                sent_count += 1

            except Exception as exc:

                print(
                    "FAILED: "
                    + mask_email(
                        recipient
                    )
                    + " — "
                    + type(exc).__name__
                )

                failed_count += 1


    print()
    print("E-AWARE DAILY DELIVERY SUMMARY")
    print(f"Sent: {sent_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")

    if failed_count:
        raise RuntimeError(
            f"{failed_count} advisory email(s) failed."
        )


if __name__ == "__main__":
    main()
