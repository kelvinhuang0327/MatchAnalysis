"""CLI tests for deterministic P19A Moneyline inference."""

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.moneyline_inference import main


FIXTURE_DIR = REPOSITORY_ROOT / "data" / "fixtures" / "p19a_moneyline_inference"


class MoneylineInferenceCliTests(unittest.TestCase):
    def _args(self, output_dir: Path) -> list[str]:
        return [
            "--feature-snapshots",
            str(FIXTURE_DIR / "feature_snapshots.jsonl"),
            "--model-artifact",
            str(FIXTURE_DIR / "model_artifact.json"),
            "--prediction-generated-at-utc",
            "2025-06-01T00:01:00Z",
            "--response-received-at-utc",
            "2025-06-01T00:01:01Z",
            "--ingested-at-utc",
            "2025-06-01T00:01:02Z",
            "--output-dir",
            str(output_dir),
        ]

    def test_cli_replay_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            self.assertEqual(main(self._args(first)), 0)
            self.assertEqual(main(self._args(second)), 0)
            for filename in (
                "predictions.jsonl",
                "admissions.jsonl",
                "summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

    def test_cli_missing_input_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = self._args(output_dir)
            args[1] = str(Path(temp_dir) / "missing.jsonl")
            self.assertEqual(main(args), 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
