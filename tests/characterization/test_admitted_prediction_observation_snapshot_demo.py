"""Characterization tests for committed admitted prediction observation snapshot demo."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.admitted_prediction_observation_snapshot import main as cli_main


class AdmittedPredictionObservationSnapshotDemoCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results_path = (
            REPOSITORY_ROOT / "report" / "p15b_real_schedule_admission" / "results.jsonl"
        )
        self.summary_path = (
            REPOSITORY_ROOT / "report" / "p15b_real_schedule_admission" / "summary.json"
        )
        self.report_dir = (
            REPOSITORY_ROOT / "report" / "p15c_admitted_prediction_observation_snapshot"
        )

    def test_committed_demo_artifacts_exist_and_contain_required_structure(self) -> None:
        obs_file = self.report_dir / "admitted_observations.jsonl"
        summary_file = self.report_dir / "summary.json"
        report_file = self.report_dir / "report.md"

        self.assertTrue(obs_file.exists())
        self.assertTrue(summary_file.exists())
        self.assertTrue(report_file.exists())

        # Parse and validate JSONL
        obs_rows = [
            json.loads(line)
            for line in obs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(obs_rows), 1)

        # Each row has required fields
        for row in obs_rows:
            self.assertIn("prediction_observation_id", row)
            self.assertIn("source_result_row_fingerprint", row)
            self.assertIn("observation", row)
            self.assertIn("snapshot_row_fingerprint", row)

        # Rows are sorted by prediction_observation_id
        obs_ids = [row["prediction_observation_id"] for row in obs_rows]
        self.assertEqual(obs_ids, sorted(obs_ids))
        self.assertEqual(len(obs_ids), len(set(obs_ids)))

        # Parse and validate summary
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["schema_version"],
            "p15c.admitted_prediction_observation_snapshot.v1",
        )
        self.assertEqual(summary["snapshot_row_count"], len(obs_rows))
        self.assertEqual(summary["sorted_observation_ids"], obs_ids)

        # Verify JSONL SHA-256 matches summary
        obs_sha = hashlib.sha256(
            obs_file.read_bytes()
        ).hexdigest()
        self.assertEqual(
            summary["admitted_observations_jsonl_sha256"],
            obs_sha,
        )

        # Verify safety claims
        claims = summary["claims"]
        self.assertFalse(claims["legacy_rows_admitted"])
        self.assertFalse(claims["outcomes_attached"])
        self.assertFalse(claims["provider_called"])
        self.assertFalse(claims["network_called"])
        self.assertFalse(claims["db_written"])
        self.assertFalse(claims["deployed"])
        self.assertFalse(claims["betting_claim"])

        # Verify source fingerprint matches P15B1
        p15b_summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["source_result_set_fingerprint"],
            p15b_summary["result_set_fingerprint"],
        )

    def test_rerunning_cli_reproduces_committed_demo_artifacts_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            exit_code = cli_main([
                "--admission-results", str(self.results_path),
                "--admission-summary", str(self.summary_path),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(exit_code, 0)

            for artifact_name in (
                "admitted_observations.jsonl",
                "summary.json",
                "report.md",
            ):
                generated_bytes = (out_dir / artifact_name).read_bytes()
                committed_bytes = (self.report_dir / artifact_name).read_bytes()
                self.assertEqual(
                    generated_bytes,
                    committed_bytes,
                    f"Committed artifact {artifact_name} differs from freshly generated output",
                )


if __name__ == "__main__":
    unittest.main()
