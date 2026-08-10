"""CLI acceptance tests for P23A artifact materialization."""

import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.evaluate_moneyline_challenger_oos import main


ROOT = Path(__file__).resolve().parents[3]


class MoneylineChallengerOOSCLITests(unittest.TestCase):
    def test_cli_writes_exact_three_deterministic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "p23a_strictly_future_oos"
            args = ["--repository-root", str(ROOT), "--output-dir", str(output)]
            self.assertEqual(main(args), 0)
            first = {
                path.name: path.read_bytes()
                for path in sorted(output.iterdir())
            }
            self.assertEqual(
                set(first),
                {"comparisons.jsonl", "summary.json", "incumbent_model_artifact.json"},
            )
            comparisons = first["comparisons.jsonl"].decode().splitlines()
            self.assertEqual(len(comparisons), 23)
            summary = json.loads(first["summary.json"])
            self.assertTrue(summary["deterministic_replay_verified"])
            self.assertEqual(summary["game_count"], 23)
            self.assertFalse(summary["model_promoted"])
            self.assertEqual(main(args), 0)
            second = {
                path.name: path.read_bytes()
                for path in sorted(output.iterdir())
            }
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
