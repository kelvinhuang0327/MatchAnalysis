"""Characterization test for committed P16A final result attachment demo."""

from hashlib import sha256
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.interfaces.cli.final_result_attachment import main as cli_main
from match_analysis.application.use_cases.attach_final_results_to_admitted_predictions import (
    attach_final_results_to_admitted_predictions,
)

P15C_SNAPSHOT_PATH = (
    REPOSITORY_ROOT
    / "report"
    / "p15c_admitted_prediction_observation_snapshot"
    / "admitted_observations.jsonl"
)
P15C_SUMMARY_PATH = (
    REPOSITORY_ROOT
    / "report"
    / "p15c_admitted_prediction_observation_snapshot"
    / "summary.json"
)
DEMO_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "examples"
    / "p16a_final_result_attachment"
    / "final_results.jsonl"
)

P16A_REPORT_DIR = REPOSITORY_ROOT / "report" / "p16a_final_result_attachment"


class FinalResultAttachmentDemoCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="p16a_demo_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_committed_demo_outputs_reproduce_byte_for_byte(self) -> None:
        out_dir = self.temp_dir / "replay_out"
        argv = [
            "--prediction-snapshot", str(P15C_SNAPSHOT_PATH),
            "--prediction-summary", str(P15C_SUMMARY_PATH),
            "--final-results", str(DEMO_RESULTS_PATH),
            "--output-dir", str(out_dir),
        ]
        self.assertEqual(cli_main(argv), 0)

        for filename in ["attachments.jsonl", "summary.json", "report.md"]:
            committed_bytes = (P16A_REPORT_DIR / filename).read_bytes()
            generated_bytes = (out_dir / filename).read_bytes()
            self.assertEqual(
                sha256(generated_bytes).hexdigest(),
                sha256(committed_bytes).hexdigest(),
                f"Byte mismatch in committed output artifact {filename}",
            )

    def test_shuffled_result_input_order_preserves_attachment_set_fingerprint(self) -> None:
        lines = DEMO_RESULTS_PATH.read_text(encoding="utf-8").splitlines()
        shuffled_lines = [lines[2], lines[0], lines[1]]
        shuffled_results_bytes = ("\n".join(shuffled_lines) + "\n").encode("utf-8")

        res_normal = attach_final_results_to_admitted_predictions(
            snapshot_bytes=P15C_SNAPSHOT_PATH.read_bytes(),
            summary_bytes=P15C_SUMMARY_PATH.read_bytes(),
            final_results_bytes=DEMO_RESULTS_PATH.read_bytes(),
        )
        res_shuffled = attach_final_results_to_admitted_predictions(
            snapshot_bytes=P15C_SNAPSHOT_PATH.read_bytes(),
            summary_bytes=P15C_SUMMARY_PATH.read_bytes(),
            final_results_bytes=shuffled_results_bytes,
        )

        self.assertEqual(
            res_normal.attachment_set_fingerprint,
            res_shuffled.attachment_set_fingerprint,
        )
        self.assertEqual(
            [r.prediction_observation_id for r in res_normal.attachment_rows],
            [r.prediction_observation_id for r in res_shuffled.attachment_rows],
        )

    def test_committed_demo_counts_and_safety_claims(self) -> None:
        res = attach_final_results_to_admitted_predictions(
            snapshot_bytes=P15C_SNAPSHOT_PATH.read_bytes(),
            summary_bytes=P15C_SUMMARY_PATH.read_bytes(),
            final_results_bytes=DEMO_RESULTS_PATH.read_bytes(),
        )

        self.assertEqual(res.source_prediction_count, 3)
        self.assertEqual(res.final_result_observation_count, 3)
        self.assertEqual(res.attached_count, 3)
        self.assertEqual(res.rejected_count, 0)
        self.assertEqual(res.correct_count, 2)
        self.assertEqual(res.incorrect_count, 1)

        # Check winners include both HOME and AWAY
        winners = {r.actual_winner for r in res.attachment_rows}
        self.assertEqual(winners, {"HOME", "AWAY"})

        # Check claims
        self.assertTrue(res.claims["synthetic_results"])
        self.assertFalse(res.claims["provider_called"])
        self.assertFalse(res.claims["network_called"])
        self.assertFalse(res.claims["db_written"])
        self.assertFalse(res.claims["odds_used"])
        self.assertFalse(res.claims["profitability_claim"])
        self.assertFalse(res.claims["deployed"])


if __name__ == "__main__":
    unittest.main()
