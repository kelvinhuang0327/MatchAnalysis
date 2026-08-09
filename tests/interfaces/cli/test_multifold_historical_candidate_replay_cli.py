"""CLI tests for the bounded P21B replay."""

from pathlib import Path
import json
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.multifold_historical_candidate_replay import main


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"


class MultifoldHistoricalCandidateReplayCliTests(unittest.TestCase):
    def test_cli_writes_only_the_five_p21b_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "p21b"
            exit_code = main(
                [
                    "--fold",
                    str(FIXTURE_ROOT / "fold_wf_003.json"),
                    "--fold",
                    str(FIXTURE_ROOT / "fold_wf_002.json"),
                    "--historical-results",
                    str(FIXTURE_ROOT / "final_results.jsonl"),
                    "--historical-provenance",
                    str(FIXTURE_ROOT / "provenance.json"),
                    "--reconstructed-models",
                    str(FIXTURE_ROOT / "reconstructed_models.json"),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "feedback.jsonl",
                    "assessments.jsonl",
                    "learning_candidates.jsonl",
                    "summary.json",
                    "report.md",
                },
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_fold_ids"], ["wf_002", "wf_003"])
            self.assertEqual(summary["candidate_count"], 1354)
            self.assertEqual(summary["p20b_historical_runtime_compliance"], "REMAINS_REFUTED")


if __name__ == "__main__":
    unittest.main()
