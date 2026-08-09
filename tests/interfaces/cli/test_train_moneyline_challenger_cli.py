"""CLI tests for deterministic P22B challenger training."""

from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.train_moneyline_challenger import main


DATASET_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/training_examples.jsonl"
SUMMARY_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/summary.json"


class TrainMoneylineChallengerCliTests(unittest.TestCase):
    def args(self, output_dir: Path) -> list[str]:
        return [
            "--dataset",
            str(DATASET_PATH),
            "--summary",
            str(SUMMARY_PATH),
            "--fit-runtime",
            "/usr/bin/python3",
            "--source-repository",
            str(REPOSITORY_ROOT),
            "--output-dir",
            str(output_dir),
        ]

    def test_cli_materializes_byte_identical_outputs_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            self.assertEqual(main(self.args(first)), 0)
            self.assertEqual(main(self.args(second)), 0)
            self.assertEqual(
                (first / "model_artifact.json").read_bytes(),
                (second / "model_artifact.json").read_bytes(),
            )
            self.assertEqual(
                (first / "summary.json").read_bytes(),
                (second / "summary.json").read_bytes(),
            )

    def test_cli_missing_dataset_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            args = self.args(output_dir)
            args[1] = str(Path(temp_dir) / "missing.jsonl")
            self.assertEqual(main(args), 1)
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
