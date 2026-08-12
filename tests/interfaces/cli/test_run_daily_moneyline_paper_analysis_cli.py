"""CLI surface checks for P33A."""

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.run_daily_moneyline_paper_analysis import (
    build_parser,
    main,
)


class RunDailyMoneylinePaperAnalysisCliTests(unittest.TestCase):
    def test_parser_supports_explicit_date_and_bundle_replay(self) -> None:
        args = build_parser().parse_args(
            [
                "--date",
                "2026-08-12",
                "--output-dir",
                "/tmp/matchanalysis-p33a-daily-paper-run/live",
            ]
        )
        self.assertEqual(args.date_value, "2026-08-12")
        self.assertIsNone(args.from_bundle)

        replay_args = build_parser().parse_args(
            [
                "--from-bundle",
                "/tmp/matchanalysis-p33a-daily-paper-run/bundles/run",
            ]
        )
        self.assertIsNotNone(replay_args.from_bundle)

    def test_date_and_replay_are_mutually_exclusive(self) -> None:
        self.assertEqual(
            main(
                [
                    "--date",
                    "2026-08-12",
                    "--from-bundle",
                    "/tmp/matchanalysis-p33a-daily-paper-run/bundles/run",
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
