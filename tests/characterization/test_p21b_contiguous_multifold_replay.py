"""Characterization of the committed P21B historical candidate batch."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.multifold_historical_candidate_artifacts import (
    render_multifold_assessments_jsonl,
    render_multifold_candidates_jsonl,
    render_multifold_feedback_jsonl,
    render_multifold_report_markdown,
    write_multifold_historical_candidate_artifacts,
)
from match_analysis.application.use_cases.replay_multifold_historical_candidates import (
    load_multifold_folds,
    load_multifold_reconstructed_models,
    replay_multifold_historical_candidates,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"


def build_result():
    return replay_multifold_historical_candidates(
        folds=load_multifold_folds(
            [FIXTURE_ROOT / "fold_wf_002.json", FIXTURE_ROOT / "fold_wf_003.json"]
        ),
        historical_results_bytes=(FIXTURE_ROOT / "final_results.jsonl").read_bytes(),
        historical_provenance_bytes=(FIXTURE_ROOT / "provenance.json").read_bytes(),
        reconstructed_models=load_multifold_reconstructed_models(
            FIXTURE_ROOT / "reconstructed_models.json"
        ),
    )


class P21BContiguousMultifoldReplayCharacterizationTests(unittest.TestCase):
    def test_exact_fold_shape_parity_and_p15_to_p21_lineage(self) -> None:
        result = build_result()
        self.assertEqual(
            [(fold.fold_id, fold.validation_start, fold.validation_end, fold.prediction_row_count) for fold in result.folds],
            [
                ("wf_002", "2025-07-01", "2025-07-31", 319),
                ("wf_003", "2025-08-01", "2025-08-31", 358),
            ],
        )
        self.assertEqual(
            [fold.max_absolute_difference for fold in result.folds],
            ["4.992170296492576096638E-7", "4.974175138497870790439E-7"],
        )
        self.assertEqual(
            result.feedback_result.feedback_ledger_fingerprint,
            "8c4f00f5adc3c5207329be26e8f82b05f4d65e66bf214b57fc05dfb9beda9d9d",
        )
        self.assertEqual(
            result.assessment_semantic_fingerprint,
            "bc5ff8bed5c2db939d8da3035421e777e49e11da50dc40e434997d1548ede2e6",
        )
        self.assertEqual(
            result.candidate_semantic_fingerprint,
            "6cd935ea6d5ac5707726b25b2d1c42f8007a35a9b87f740792aa28b576424ac0",
        )

    def test_artifacts_are_deterministic_and_claims_remain_bounded(self) -> None:
        first = build_result()
        second = build_result()
        self.assertEqual(first.to_projection(), second.to_projection())
        self.assertEqual(
            hashlib.sha256(
                render_multifold_feedback_jsonl(first).encode("utf-8")
            ).hexdigest(),
            hashlib.sha256(
                render_multifold_feedback_jsonl(second).encode("utf-8")
            ).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "p21b"
            write_multifold_historical_candidate_artifacts(output_dir, first)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["p15c_admission_count"], 1354)
            self.assertEqual(summary["p16a_attachment_count"], 1354)
            self.assertEqual(summary["p16b_evaluation_count"], 1354)
            self.assertEqual(summary["p21a_assessed_count"], 1354)
            self.assertTrue(summary["claims"]["historical"])
            self.assertTrue(summary["claims"]["sample_limited"])
            self.assertFalse(summary["claims"]["synthetic_results"])
            self.assertFalse(summary["claims"]["training_authorized"])
            self.assertFalse(summary["claims"]["training_dataset_claim"])
            self.assertFalse(summary["claims"]["retraining_performed"])
            self.assertEqual(summary["p20b_historical_runtime_compliance"], "REMAINS_REFUTED")
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("P20B historical runtime compliance remains REFUTED", report)
            self.assertEqual(
                summary["feedback_jsonl_sha256"],
                hashlib.sha256((output_dir / "feedback.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["assessments_jsonl_sha256"],
                hashlib.sha256((output_dir / "assessments.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["learning_candidates_jsonl_sha256"],
                hashlib.sha256((output_dir / "learning_candidates.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                len([line for line in render_multifold_assessments_jsonl(first).splitlines() if line]),
                1354,
            )
            self.assertEqual(
                len([line for line in render_multifold_candidates_jsonl(first).splitlines() if line]),
                1354,
            )
            self.assertIn("sample_limited=true", render_multifold_report_markdown(first).lower())


if __name__ == "__main__":
    unittest.main()
