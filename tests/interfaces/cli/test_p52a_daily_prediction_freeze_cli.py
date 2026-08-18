"""Focused tests for P52A daily Moneyline prospective prediction freeze CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.interfaces.cli.run_p52a_daily_prediction_freeze import (
    build_parser,
    main,
)
from tests.unit.test_p52a_daily_moneyline_prediction_freeze import (
    MockMLBOpener,
    _build_mock_pitcher_log_payload,
    _build_mock_schedule_payload,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class TestP52ADailyPredictionFreezeCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parser_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--target-date", "2026-08-18", "--json"])
        self.assertEqual(args.target_date, "2026-08-18")
        self.assertTrue(args.json)
        self.assertIsNone(args.as_of_utc)

    def test_cli_execution_standard_output(self) -> None:
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=5),
            201: _build_mock_pitcher_log_payload(201, start_count=4),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        with (
            patch(
                "match_analysis.application.use_cases.p52a_daily_prediction_freeze._default_opener",
                mock_opener,
            ),
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            exit_code = main(
                [
                    "--target-date",
                    target_date,
                    "--as-of-utc",
                    as_of_utc,
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--output-root",
                    str(self.output_root),
                ]
            )

        self.assertEqual(exit_code, 0)
        output = mock_stdout.getvalue()
        self.assertIn("p52a-daily-prediction-freeze=", output)
        self.assertIn("target_date=2026-08-18", output)
        self.assertIn("eligible=1", output)
        self.assertIn("pending_frozen=1", output)
        self.assertIn("settled_prediction_forward_sample_count=0", output)

    def test_cli_execution_json_output(self) -> None:
        as_of_utc = "2026-08-18T13:00:00Z"
        target_date = "2026-08-18"
        target_start = "2026-08-18T23:05:00Z"

        sched = _build_mock_schedule_payload(
            target_date=target_date,
            target_start_utc=target_start,
            home_starter_id=101,
            away_starter_id=201,
        )
        p_map = {
            101: _build_mock_pitcher_log_payload(101, start_count=5),
            201: _build_mock_pitcher_log_payload(201, start_count=4),
        }
        mock_opener = MockMLBOpener(sched, p_map)

        with (
            patch(
                "match_analysis.application.use_cases.p52a_daily_prediction_freeze._default_opener",
                mock_opener,
            ),
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            exit_code = main(
                [
                    "--target-date",
                    target_date,
                    "--as-of-utc",
                    as_of_utc,
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--output-root",
                    str(self.output_root),
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        parsed = json.loads(mock_stdout.getvalue())
        self.assertEqual(parsed["target_date"], "2026-08-18")
        self.assertEqual(parsed["eligible_predictions_count"], 1)
        self.assertEqual(parsed["exclusion_count"], 0)
        self.assertEqual(parsed["pending_count"], 1)
        self.assertEqual(parsed["settled_prediction_forward_sample_count"], 0)
        self.assertEqual(parsed["betting_forward_sample_count"], 0)
        self.assertTrue(parsed["run_id"].startswith("p50c_run_"))


if __name__ == "__main__":
    unittest.main()
