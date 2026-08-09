"""CLI characterization for the P22A dataset materializer."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"
CANDIDATE_ROOT = REPOSITORY_ROOT / "report/p21b_contiguous_multifold_historical_candidates"


class MaterializeMoneylineTrainingDatasetCliTests(unittest.TestCase):
    def run_cli(self, output_dir: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "match_analysis.interfaces.cli.materialize_moneyline_training_dataset",
                "--learning-candidates",
                str(CANDIDATE_ROOT / "learning_candidates.jsonl"),
                "--candidate-summary",
                str(CANDIDATE_ROOT / "summary.json"),
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
            ],
            cwd=REPOSITORY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_materializes_expected_artifacts_and_replays_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.run_cli(Path(first_dir))
            second = self.run_cli(Path(second_dir))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("games=677", first.stdout)
            self.assertIn("eligible_candidates=1354", first.stdout)
            self.assertIn("collapsed=677", first.stdout)
            self.assertIn("unmapped=0", first.stdout)
            self.assertEqual(
                (Path(first_dir) / "training_examples.jsonl").read_bytes(),
                (Path(second_dir) / "training_examples.jsonl").read_bytes(),
            )
            self.assertEqual(
                (Path(first_dir) / "summary.json").read_bytes(),
                (Path(second_dir) / "summary.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
