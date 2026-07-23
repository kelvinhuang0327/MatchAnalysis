"""Unit tests for immutable quarantined prediction candidates."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.prediction import (
    DIAGNOSTIC_UNTIMED,
    MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF,
    LegacyPredictionCandidate,
)


def make_candidate(**overrides: object) -> LegacyPredictionCandidate:
    values: dict[str, object] = {
        "source_game_id": "mlb_2026_822733",
        "source_prediction_version": "p84b_diagnostic_baseline_v1",
        "predicted_side": "home",
        "sp_fip_delta": Decimal("-0.0295"),
    }
    values.update(overrides)
    return LegacyPredictionCandidate(**values)


class LegacyPredictionCandidateTests(unittest.TestCase):
    def test_candidate_is_immutable_and_exactly_quarantined(self) -> None:
        candidate = make_candidate()

        self.assertEqual(candidate.diagnostic_status, DIAGNOSTIC_UNTIMED)
        self.assertEqual(
            candidate.quarantine_reason,
            MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF,
        )
        with self.assertRaises(FrozenInstanceError):
            candidate.predicted_side = "away"

    def test_candidate_contains_no_identity_time_or_outcome_fields(self) -> None:
        fields = set(LegacyPredictionCandidate.__dataclass_fields__)

        self.assertTrue(
            {
                "result_home_score",
                "result_away_score",
                "actual_winner",
                "is_correct",
                "identity",
                "scheduled_start_utc",
                "prediction_as_of_utc",
            }.isdisjoint(fields)
        )

    def test_invalid_source_id_side_version_and_delta_are_rejected(self) -> None:
        invalid_cases = (
            {"source_game_id": "822733"},
            {"predicted_side": "draw"},
            {"source_prediction_version": ""},
            {"sp_fip_delta": Decimal("0")},
            {"sp_fip_delta": Decimal("NaN")},
            {"sp_fip_delta": "-0.0295"},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    make_candidate(**overrides)


if __name__ == "__main__":
    unittest.main()
