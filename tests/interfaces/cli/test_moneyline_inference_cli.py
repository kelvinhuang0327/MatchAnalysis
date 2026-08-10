"""CLI tests for deterministic P19A Moneyline inference."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_model_artifact,
)
from match_analysis.interfaces.cli.moneyline_inference import main


FIXTURE_DIR = REPOSITORY_ROOT / "data" / "fixtures" / "p19a_moneyline_inference"
P22B_ARTIFACT_PATH = (
    REPOSITORY_ROOT / "report" / "p22b_moneyline_challenger" / "model_artifact.json"
)


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

    def _args_without_model_artifact(self, output_dir: Path) -> list[str]:
        args = self._args(output_dir)
        del args[2:4]
        return args

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

    def test_cli_defaults_to_frozen_p22b_challenger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "challenger"
            self.assertEqual(main(self._args_without_model_artifact(output_dir)), 0)
            summary = json.loads((output_dir / "summary.json").read_text())
            predictions = [
                json.loads(line)
                for line in (output_dir / "predictions.jsonl").read_text().splitlines()
            ]
            challenger_projection = json.loads(P22B_ARTIFACT_PATH.read_text())
            inference_artifact = load_moneyline_model_artifact(P22B_ARTIFACT_PATH)
            self.assertEqual(
                summary["model_artifact"]["model_id"],
                "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            )
            self.assertEqual(
                challenger_projection["artifact_fingerprint"],
                "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            )
            self.assertEqual(
                summary["model_artifact"],
                inference_artifact.to_projection(),
            )
            self.assertEqual(
                summary["model_artifact_fingerprint"],
                inference_artifact.fingerprint(),
            )
            self.assertEqual(
                {row["model_id"] for row in predictions},
                {"p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630"},
            )
            self.assertFalse(summary["claims"]["production_claim"])
            self.assertFalse(summary["claims"]["betting_claim"])

    def test_explicit_artifact_override_remains_in_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "incumbent"
            self.assertEqual(main(self._args(output_dir)), 0)
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                summary["model_artifact"]["model_id"],
                "p13_walk_forward_logistic_v1_fixture",
            )


if __name__ == "__main__":
    unittest.main()
