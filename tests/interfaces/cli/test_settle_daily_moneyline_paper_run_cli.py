"""CLI integration checks for P34A daily settlement."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.settle_daily_moneyline_paper_run import (
    build_parser,
    main,
)


P33A_BUNDLE = Path(
    "/tmp/matchanalysis-p33a-daily-paper-run/live-smoke-20260812-r3/bundles/"
    "a646ec0081afde1469f978e20bf98fc90cf21587bcecaa9b5cccb280b46bd569"
)


class SettleDailyMoneylinePaperRunCliTests(unittest.TestCase):
    def test_parser_supports_frozen_p33a_and_replay_bundle(self) -> None:
        args = build_parser().parse_args(
            ["--p33a-bundle", str(P33A_BUNDLE), "--final-results", "/tmp/results.jsonl"]
        )
        self.assertEqual(args.p33a_bundle, P33A_BUNDLE)
        self.assertIsNone(args.from_bundle)

        replay_args = build_parser().parse_args(
            ["--p33a-bundle", str(P33A_BUNDLE), "--from-bundle", "/tmp/p34a"]
        )
        self.assertIsNotNone(replay_args.from_bundle)

    def test_cli_writes_truthful_empty_daily_sample_from_frozen_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p34a-cli-") as temporary:
            root = Path(temporary)
            final_results = root / "final_results.jsonl"
            output_dir = root / "report"
            final_results.write_bytes(b"")
            exit_code = main(
                [
                    "--p33a-bundle",
                    str(P33A_BUNDLE),
                    "--final-results",
                    str(final_results),
                    "--observed-at-utc",
                    "2026-08-12T03:00:00Z",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "admitted_observations.jsonl",
                    "attachments.jsonl",
                    "evaluations.jsonl",
                    "feedback.jsonl",
                    "final_results.jsonl",
                    "structural_rows.jsonl",
                    "prediction_results.jsonl",
                    "settled_predictions.jsonl",
                    "result_authority.json",
                    "report.md",
                    "summary.json",
                },
            )
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(summary["settleable_prediction_count"], 0)
            self.assertEqual(summary["structural_row_count"], 15)
            self.assertEqual(summary["settled_prediction_count"], 0)
            self.assertFalse(summary["claims"]["profitability_claim"])


if __name__ == "__main__":
    unittest.main()
