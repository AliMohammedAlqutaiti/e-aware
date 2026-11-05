import os
import smtplib
import tempfile
import unittest

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import daily_advisory_job as job


TEST_ENV = {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SECRET_KEY": "test-secret",
    "GMAIL_SENDER": "test@example.com",
    "GMAIL_APP_PASSWORD": "test-password",
}


def make_advisory_file(folder):
    path = Path(folder) / "eaware_advisory.json"

    path.write_text(
        """
{
  "data_date": "2026-11-05",
  "advisory": "E-AWARE TEST ADVISORY"
}
""".strip(),
        encoding="utf-8",
    )

    return path


def fake_database(subscribers):
    db = MagicMock()

    result = SimpleNamespace(
        data=subscribers
    )

    (
        db.table.return_value
        .select.return_value
        .eq.return_value
        .execute.return_value
    ) = result

    return db


class FailureModeTests(unittest.TestCase):

    def test_01_missing_advisory_file_fails_safely(self):
        with tempfile.TemporaryDirectory() as tmp:

            missing = (
                Path(tmp)
                / "missing_advisory.json"
            )

            with patch.object(
                job,
                "ADVISORY_FILE",
                missing
            ):

                with self.assertRaisesRegex(
                    RuntimeError,
                    "did not generate"
                ):
                    job.main()


    def test_02_no_active_subscribers_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:

            advisory = make_advisory_file(
                tmp
            )

            db = fake_database(
                []
            )

            with (
                patch.object(
                    job,
                    "ADVISORY_FILE",
                    advisory
                ),
                patch.dict(
                    os.environ,
                    TEST_ENV,
                    clear=False
                ),
                patch(
                    "daily_advisory_job.create_client",
                    return_value=db
                ),
                patch(
                    "daily_advisory_job.smtplib.SMTP"
                ) as smtp_mock
            ):

                job.main()

                smtp_mock.assert_not_called()


    def test_03_supabase_failure_stops_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:

            advisory = make_advisory_file(
                tmp
            )

            with (
                patch.object(
                    job,
                    "ADVISORY_FILE",
                    advisory
                ),
                patch.dict(
                    os.environ,
                    TEST_ENV,
                    clear=False
                ),
                patch(
                    "daily_advisory_job.create_client",
                    side_effect=RuntimeError(
                        "simulated database outage"
                    )
                )
            ):

                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated database outage"
                ):
                    job.main()


    def test_04_gmail_authentication_failure_stops_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:

            advisory = make_advisory_file(
                tmp
            )

            db = fake_database(
                [
                    {
                        "id": 1,
                        "email":
                            "subscriber@example.com",
                        "last_sent_at":
                            None,
                    }
                ]
            )

            smtp_context = MagicMock()

            smtp_instance = (
                smtp_context.return_value
                .__enter__.return_value
            )

            smtp_instance.login.side_effect = (
                smtplib.SMTPAuthenticationError(
                    535,
                    b"simulated authentication failure"
                )
            )

            with (
                patch.object(
                    job,
                    "ADVISORY_FILE",
                    advisory
                ),
                patch.dict(
                    os.environ,
                    TEST_ENV,
                    clear=False
                ),
                patch(
                    "daily_advisory_job.create_client",
                    return_value=db
                ),
                patch(
                    "daily_advisory_job.smtplib.SMTP",
                    smtp_context
                )
            ):

                with self.assertRaises(
                    smtplib.SMTPAuthenticationError
                ):
                    job.main()


    def test_05_duplicate_daily_email_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:

            advisory = make_advisory_file(
                tmp
            )

            db = fake_database(
                [
                    {
                        "id": 1,
                        "email":
                            "subscriber@example.com",
                        "last_sent_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                    }
                ]
            )

            smtp_context = MagicMock()

            with (
                patch.object(
                    job,
                    "ADVISORY_FILE",
                    advisory
                ),
                patch.dict(
                    os.environ,
                    {
                        **TEST_ENV,
                        "FORCE_SEND": "0",
                    },
                    clear=False
                ),
                patch(
                    "daily_advisory_job.create_client",
                    return_value=db
                ),
                patch(
                    "daily_advisory_job.smtplib.SMTP",
                    smtp_context
                ),
                patch(
                    "daily_advisory_job.send_email"
                ) as send_mock
            ):

                job.main()

                send_mock.assert_not_called()


    def test_06_manual_test_can_force_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:

            advisory = make_advisory_file(
                tmp
            )

            db = fake_database(
                [
                    {
                        "id": 1,
                        "email":
                            "subscriber@example.com",
                        "last_sent_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                    }
                ]
            )

            smtp_context = MagicMock()

            with (
                patch.object(
                    job,
                    "ADVISORY_FILE",
                    advisory
                ),
                patch.dict(
                    os.environ,
                    {
                        **TEST_ENV,
                        "FORCE_SEND": "1",
                    },
                    clear=False
                ),
                patch(
                    "daily_advisory_job.create_client",
                    return_value=db
                ),
                patch(
                    "daily_advisory_job.smtplib.SMTP",
                    smtp_context
                ),
                patch(
                    "daily_advisory_job.send_email"
                ) as send_mock
            ):

                job.main()

                send_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
