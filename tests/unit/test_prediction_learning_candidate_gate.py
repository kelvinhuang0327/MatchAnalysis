"""Unit tests for strict P20B loading and P21A candidate assessment."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from match_analysis.application.use_cases.assess_prediction_learning_candidates import (
    NO_REAL_POSITIVE_PATH_STOP,
    PredictionLearningCandidateSourceError,
    SOURCE_ARTIFACT_INVALID_STOP,
    assess_prediction_learning_candidates,
)
from match_analysis.baseball.domain.prediction_feedback import (
    compute_feedback_ledger_fingerprint,
    compute_feedback_row_fingerprint,
)
from match_analysis.application.use_cases.prediction_learning_candidate_artifacts import (
    render_assessments_jsonl,
    render_learning_candidates_jsonl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P20B_ROOT = REPOSITORY_ROOT / "report/p20b_first_non_synthetic_historical_feedback"


def _row_fingerprint(row: dict[str, object]) -> str:
    return compute_feedback_row_fingerprint(
        prediction_observation_id=row["prediction_observation_id"],
        source_snapshot_row_fingerprint=row["source_snapshot_row_fingerprint"],
        source_attachment_row_fingerprint=row["source_attachment_row_fingerprint"],
        source_evaluation_row_fingerprint=row["source_evaluation_row_fingerprint"],
        provider_namespace=row["provider_namespace"],
        provider_game_id=row["provider_game_id"],
        game_number=row["game_number"],
        scheduled_start_utc=row["scheduled_start_utc"],
        model_id=row["model_id"],
        market_id=row["market_id"],
        selection=row["selection"],
        model_probability=Decimal(row["model_probability"]),
        result_observation_id=row["result_observation_id"],
        result_observed_at_utc=row["result_observed_at_utc"],
        home_score=row["home_score"],
        away_score=row["away_score"],
        actual_winner=row["actual_winner"],
        attachment_status=row["attachment_status"],
        attachment_rejection_reason=row["attachment_rejection_reason"],
        feedback_status=row["feedback_status"],
        is_correct=row["is_correct"],
        correctness_target=row["correctness_target"],
        brier_component=(
            Decimal(row["brier_component"])
            if row["brier_component"] is not None
            else None
        ),
    )


class PredictionLearningCandidateGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feedback_path = P20B_ROOT / "feedback.jsonl"
        self.summary_path = P20B_ROOT / "summary.json"
        self.feedback_bytes = self.feedback_path.read_bytes()
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.rows = [json.loads(line) for line in self.feedback_bytes.decode("utf-8").splitlines()]

    def _inputs(
        self,
        *,
        rows: list[dict[str, object]] | None = None,
        summary: dict[str, object] | None = None,
        reverse_rows: bool = False,
    ) -> tuple[bytes, bytes]:
        rows = deepcopy(rows if rows is not None else self.rows)
        if reverse_rows:
            rows.reverse()
        for row in rows:
            row["feedback_row_fingerprint"] = _row_fingerprint(row)
        feedback_bytes = (
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
        ).encode("utf-8")
        summary = deepcopy(summary if summary is not None else self.summary)
        summary["feedback_jsonl_sha256"] = hashlib.sha256(feedback_bytes).hexdigest()
        summary["feedback_ledger_fingerprint"] = compute_feedback_ledger_fingerprint(
            tuple(
                SimpleNamespace(
                    prediction_observation_id=row["prediction_observation_id"],
                    feedback_row_fingerprint=row["feedback_row_fingerprint"],
                )
                for row in sorted(rows, key=lambda item: item["prediction_observation_id"])
            )
        )
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
        return feedback_bytes, summary_bytes

    def test_committed_source_has_real_positive_path_and_preserves_values(self) -> None:
        result = assess_prediction_learning_candidates(
            feedback_bytes=self.feedback_bytes,
            feedback_summary_bytes=self.summary_path.read_bytes(),
        )
        self.assertEqual(len(result.assessments), 4)
        self.assertEqual(len(result.candidates), 4)
        self.assertTrue(all(assessment.status == "ELIGIBLE" for assessment in result.assessments))
        for candidate in result.candidates:
            source = next(
                row
                for row in self.rows
                if row["prediction_observation_id"] == candidate["prediction_observation_id"]
            )
            for key, value in source.items():
                self.assertEqual(candidate[key], value, key)
            self.assertEqual(candidate["source_feedback_fingerprint"], source["feedback_row_fingerprint"])

        self.assertTrue(result.claims["sample_limited"])
        for key in (
            "training_dataset_claim",
            "training_authorized",
            "retraining_performed",
            "model_promoted",
            "profitability_claim",
            "real_betting_recommendation",
        ):
            self.assertFalse(result.claims[key], key)

    def test_single_row_exclusion_is_export_filtered(self) -> None:
        rows = deepcopy(self.rows)
        rows[0]["market_id"] = "spread"
        feedback_bytes, summary_bytes = self._inputs(rows=rows)
        result = assess_prediction_learning_candidates(
            feedback_bytes=feedback_bytes,
            feedback_summary_bytes=summary_bytes,
        )
        self.assertEqual(len(result.assessments), 4)
        self.assertEqual(len(result.candidates), 3)
        excluded = [assessment for assessment in result.assessments if assessment.status == "EXCLUDED"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].exclusion_reasons, ("UNSUPPORTED_MARKET",))
        self.assertNotIn(excluded[0].prediction_observation_id, {
            candidate["prediction_observation_id"] for candidate in result.candidates
        })

    def test_structural_corruption_fails_complete_build(self) -> None:
        malformed = b"{not-json}\n"
        with self.assertRaisesRegex(PredictionLearningCandidateSourceError, SOURCE_ARTIFACT_INVALID_STOP):
            assess_prediction_learning_candidates(
                feedback_bytes=malformed,
                feedback_summary_bytes=self.summary_path.read_bytes(),
            )

        bad_rows = deepcopy(self.rows)
        bad_rows[0]["feedback_row_fingerprint"] = "0" * 64
        feedback_bytes, summary_bytes = self._inputs(rows=bad_rows)
        # _inputs refreshes the row fingerprint, so corrupt after the refresh.
        bad_feedback = feedback_bytes.replace(
            self.rows[0]["feedback_row_fingerprint"].encode("ascii"),
            ("0" * 64).encode("ascii"),
            1,
        )
        with self.assertRaisesRegex(PredictionLearningCandidateSourceError, SOURCE_ARTIFACT_INVALID_STOP):
            assess_prediction_learning_candidates(
                feedback_bytes=bad_feedback,
                feedback_summary_bytes=summary_bytes,
            )

        bad_summary = json.loads(summary_bytes.decode("utf-8"))
        bad_summary["feedback_ledger_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(PredictionLearningCandidateSourceError, SOURCE_ARTIFACT_INVALID_STOP):
            assess_prediction_learning_candidates(
                feedback_bytes=feedback_bytes,
                feedback_summary_bytes=(json.dumps(bad_summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )

    def test_no_real_positive_path_stops_after_all_rows_are_excluded(self) -> None:
        rows = deepcopy(self.rows)
        for row in rows:
            row["model_probability"] = "NaN"
        feedback_bytes, summary_bytes = self._inputs(rows=rows)
        with self.assertRaisesRegex(ValueError, NO_REAL_POSITIVE_PATH_STOP):
            assess_prediction_learning_candidates(
                feedback_bytes=feedback_bytes,
                feedback_summary_bytes=summary_bytes,
            )

    def test_row_order_does_not_change_semantic_results_or_rendered_rows(self) -> None:
        original = assess_prediction_learning_candidates(
            feedback_bytes=self.feedback_bytes,
            feedback_summary_bytes=self.summary_path.read_bytes(),
        )
        shuffled_feedback, shuffled_summary = self._inputs(reverse_rows=True)
        shuffled = assess_prediction_learning_candidates(
            feedback_bytes=shuffled_feedback,
            feedback_summary_bytes=shuffled_summary,
        )
        self.assertEqual(
            original.assessments_semantic_fingerprint,
            shuffled.assessments_semantic_fingerprint,
        )
        self.assertEqual(
            original.candidates_semantic_fingerprint,
            shuffled.candidates_semantic_fingerprint,
        )
        self.assertEqual(
            [candidate["candidate_id"] for candidate in original.candidates],
            [candidate["candidate_id"] for candidate in shuffled.candidates],
        )
        self.assertEqual(render_assessments_jsonl(original), render_assessments_jsonl(shuffled))
        self.assertEqual(
            render_learning_candidates_jsonl(original),
            render_learning_candidates_jsonl(shuffled),
        )


if __name__ == "__main__":
    unittest.main()
