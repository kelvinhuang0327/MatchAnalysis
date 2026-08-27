"""CLI characterization tests for P65A daily Run Line prospective prediction freeze."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.run_p65a_daily_run_line_prediction_freeze import main as cli_main
from tests.unit.test_p65a_daily_run_line_prediction_freeze import _build_mock_schedule


class TestP65ADailyRunLinePredictionFreezeCLI(unittest.TestCase):
    def test_cli_freeze_human_readable_output(self) -> None:
        target_date = "2026-08-21"
        as_of_utc = "2026-08-21T01:30:25Z"
        sched_payload = _build_mock_schedule(target_date=target_date)

        def mock_fetch(url, **kwargs):
            return json.dumps(sched_payload).encode("utf-8"), url

        with tempfile.TemporaryDirectory(prefix="p65a-cli-") as tmpdir:
            tmp_root = Path(tmpdir)
            with patch(
                "match_analysis.application.use_cases.p65a_daily_run_line_prediction_freeze.fetch_json_bytes",
                side_effect=mock_fetch,
            ):
                stdout_capture = io.StringIO()
                with patch("sys.stdout", stdout_capture):
                    code = cli_main(
                        [
                            "--target-date", target_date,
                            "--as-of-utc", as_of_utc,
                            "--repository-root", str(tmp_root),
                            "--output-root", str(tmp_root),
                        ]
                    )

                self.assertEqual(code, 0)
                output = stdout_capture.getvalue()
                self.assertIn("p65a-daily-run-line-prediction-freeze=", output)
                self.assertIn(f"target_date={target_date}", output)
                self.assertIn("eligible=2", output)
                self.assertIn("freeze_status=FROZEN", output)

    def test_cli_freeze_json_output(self) -> None:
        target_date = "2026-08-21"
        as_of_utc = "2026-08-21T01:30:25Z"
        sched_payload = _build_mock_schedule(target_date=target_date)

        def mock_fetch(url, **kwargs):
            return json.dumps(sched_payload).encode("utf-8"), url

        with tempfile.TemporaryDirectory(prefix="p65a-cli-json-") as tmpdir:
            tmp_root = Path(tmpdir)
            with patch(
                "match_analysis.application.use_cases.p65a_daily_run_line_prediction_freeze.fetch_json_bytes",
                side_effect=mock_fetch,
            ):
                stdout_capture = io.StringIO()
                with patch("sys.stdout", stdout_capture):
                    code = cli_main(
                        [
                            "--target-date", target_date,
                            "--as-of-utc", as_of_utc,
                            "--repository-root", str(tmp_root),
                            "--output-root", str(tmp_root),
                            "--json",
                        ]
                    )

                self.assertEqual(code, 0)
                data = json.loads(stdout_capture.getvalue())
                self.assertEqual(data["target_date"], target_date)
                self.assertEqual(data["eligible_predictions_count"], 2)
                self.assertEqual(data["market_id"], "run_line")
                self.assertEqual(len(data["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
