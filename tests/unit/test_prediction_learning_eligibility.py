"""Unit tests for the pure P21A fail-closed eligibility contract."""

import copy
import json
from pathlib import Path
import unittest

from match_analysis.baseball.domain.prediction_learning_eligibility import (
    ELIGIBLE,
    EXCLUDED,
    FEEDBACK_NOT_EVALUATED,
    INCOMPLETE_LINEAGE,
    INVALID_MODEL_PROBABILITY,
    MISSING_EVALUATION_EVIDENCE,
    MISSING_RESULT_EVIDENCE,
    RESULT_ATTACHMENT_REJECTED,
    SYNTHETIC_RESULT_EVIDENCE_EXCLUDED,
    UNSUPPORTED_MARKET,
    UNSUPPORTED_SELECTION,
    assess_prediction_learning_eligibility,
    compute_learning_candidate_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_PATH = REPOSITORY_ROOT / "report/p20b_first_non_synthetic_historical_feedback/feedback.jsonl"


class PredictionLearningEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()[0])
        self.ledger_fingerprint = "a" * 64

    def assess(self, **changes: object):
        row = copy.deepcopy(self.row)
        for key, value in changes.items():
            if key.startswith("observation_payload."):
                row["observation_payload"][key.split(".", 1)[1]] = value
            else:
                row[key] = value
        return assess_prediction_learning_eligibility(
            row,
            synthetic_results=False,
            source_feedback_ledger_fingerprint=self.ledger_fingerprint,
        )

    def test_complete_non_synthetic_row_is_eligible(self) -> None:
        assessment = self.assess()
        self.assertEqual(assessment.status, ELIGIBLE)
        self.assertEqual(assessment.exclusion_reasons, ())
        self.assertEqual(
            assessment.candidate_id,
            compute_learning_candidate_id(
                prediction_observation_id=self.row["prediction_observation_id"],
                feedback_row_fingerprint=self.row["feedback_row_fingerprint"],
            ),
        )

    def test_controlled_reasons_fail_closed(self) -> None:
        cases = (
            ({"synthetic_results": True}, SYNTHETIC_RESULT_EVIDENCE_EXCLUDED),
            ({"feedback_status": "RESULT_ATTACHMENT_REJECTED"}, FEEDBACK_NOT_EVALUATED),
            ({"attachment_status": "REJECTED"}, RESULT_ATTACHMENT_REJECTED),
            ({"result_observation_id": None}, MISSING_RESULT_EVIDENCE),
            ({"source_evaluation_row_fingerprint": None}, MISSING_EVALUATION_EVIDENCE),
            ({"market_id": "spread"}, UNSUPPORTED_MARKET),
            ({"selection": "DRAW"}, UNSUPPORTED_SELECTION),
            ({"model_probability": "1.01"}, INVALID_MODEL_PROBABILITY),
            ({"observation_payload.source_prediction_id": None}, INCOMPLETE_LINEAGE),
        )
        for changes, expected_reason in cases:
            with self.subTest(changes=changes):
                if "synthetic_results" in changes:
                    assessment = assess_prediction_learning_eligibility(
                        copy.deepcopy(self.row),
                        synthetic_results=True,
                        source_feedback_ledger_fingerprint=self.ledger_fingerprint,
                    )
                else:
                    assessment = self.assess(**changes)
                self.assertEqual(assessment.status, EXCLUDED)
                self.assertIn(expected_reason, assessment.exclusion_reasons)

    def test_reasons_are_unique_and_stably_sorted(self) -> None:
        assessment = self.assess(
            feedback_status="RESULT_ATTACHMENT_REJECTED",
            attachment_status="REJECTED",
            market_id="spread",
            selection="DRAW",
            model_probability="NaN",
        )
        self.assertEqual(
            assessment.exclusion_reasons,
            tuple(sorted(set(assessment.exclusion_reasons))),
        )
        self.assertIsNone(assessment.candidate_id)

    def test_candidate_identity_depends_on_immutable_source_identity_not_position(self) -> None:
        first = compute_learning_candidate_id(
            prediction_observation_id="a" * 64,
            feedback_row_fingerprint="b" * 64,
        )
        second = compute_learning_candidate_id(
            prediction_observation_id="c" * 64,
            feedback_row_fingerprint="b" * 64,
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
