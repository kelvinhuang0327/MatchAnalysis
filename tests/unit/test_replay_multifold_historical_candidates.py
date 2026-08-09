"""Unit coverage for the bounded P21B multifold adapter."""

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.replay_multifold_historical_candidates import (
    load_multifold_folds,
    load_multifold_reconstructed_models,
    replay_multifold_historical_candidates,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"


def replay(fold_names: tuple[str, ...] = ("fold_wf_002.json", "fold_wf_003.json")):
    return replay_multifold_historical_candidates(
        folds=load_multifold_folds([FIXTURE_ROOT / name for name in fold_names]),
        historical_results_bytes=(FIXTURE_ROOT / "final_results.jsonl").read_bytes(),
        historical_provenance_bytes=(FIXTURE_ROOT / "provenance.json").read_bytes(),
        reconstructed_models=load_multifold_reconstructed_models(
            FIXTURE_ROOT / "reconstructed_models.json"
        ),
    )


class ReplayMultifoldHistoricalCandidatesTests(unittest.TestCase):
    def test_contiguous_folds_are_sorted_and_point_in_time_safe(self) -> None:
        folds = load_multifold_folds(
            [FIXTURE_ROOT / "fold_wf_003.json", FIXTURE_ROOT / "fold_wf_002.json"]
        )
        self.assertEqual(tuple(fold.fold_id for fold in folds), ("wf_002", "wf_003"))
        self.assertTrue(all(fold.point_in_time_safe() for fold in folds))
        self.assertEqual(
            [(fold.train_as_of, fold.training_row_count, fold.prediction_row_count) for fold in folds],
            [("2025-06-30", 893, 319), ("2025-07-31", 1212, 358)],
        )

    def test_replay_preserves_lineage_and_non_synthetic_claims(self) -> None:
        result = replay()
        self.assertEqual(result.selected_fold_ids, ("wf_002", "wf_003"))
        self.assertEqual(result.prediction_row_count, 677)
        self.assertEqual(len(result.feedback_rows), 1354)
        self.assertEqual(len(result.assessments), 1354)
        self.assertEqual(len(result.candidates), 1354)
        self.assertEqual(result.p15c_admission_count, 1354)
        self.assertEqual(result.p16a_attachment_count, 1354)
        self.assertEqual(result.p16b_evaluation_count, 1354)
        self.assertEqual(result.p21a_eligible_count, 1354)
        self.assertEqual(result.p21a_excluded_count, 0)
        self.assertEqual(
            result.membership_sha256,
            "afba81ae0d9858905675b64717b59abd082bb62fd256820f933a7b845ed8d163",
        )
        self.assertEqual(
            result.result_rows_sha256,
            "79fd4f858bc6c70c2c2d044460503baf60cebaf2a9a58ce16ee0c78671c34064",
        )
        self.assertTrue(result.claims["historical"])
        self.assertTrue(result.claims["sample_limited"])
        self.assertFalse(result.claims["synthetic_results"])
        self.assertFalse(result.claims["training_dataset_claim"])

    def test_input_order_does_not_change_projection(self) -> None:
        ordered = replay()
        reversed_input = replay(("fold_wf_003.json", "fold_wf_002.json"))
        self.assertEqual(ordered.to_projection(), reversed_input.to_projection())

    def test_model_state_is_verified_and_parity_passes_for_every_row(self) -> None:
        result = replay()
        self.assertEqual(
            [(fold.fold_id, fold.model_fingerprint) for fold in result.folds],
            [
                (
                    "wf_002",
                    "2a38f6c83e960ac1795af7f774f834160656f9c5fabd13106650e3ee24fd2f2b",
                ),
                (
                    "wf_003",
                    "556d8f2ef51ec87aa7dd7e438437a77eb059d6cd39d7b8f15d92bed9fa9dec73",
                ),
            ],
        )
        self.assertTrue(all(fold.parity_passed for fold in result.folds))
        self.assertTrue(
            all(
                row["passed"]
                for fold in result.folds
                for row in fold.parity_rows
            )
        )


if __name__ == "__main__":
    unittest.main()
