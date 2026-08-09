"""Unit tests for immutable P18A decision and settlement contracts."""

from dataclasses import FrozenInstanceError
import unittest

from match_analysis.baseball.domain.result_only_paper_decision import (
    DECISION_SCHEMA_VERSION,
    SETTLEMENT_LOST,
    SETTLEMENT_UNSETTLED,
    SETTLEMENT_WON,
    ResultOnlyDecisionSelection,
    ResultOnlyPaperDecision,
    ResultOnlyPaperSettlement,
    compute_decision_id,
    compute_decision_set_fingerprint,
    compute_settlement_row_fingerprint,
    settlement_status_for,
)


def make_decision(observation_id: str, selection: str = "HOME") -> ResultOnlyPaperDecision:
    row_fingerprint = "a" * 64
    decision_id = compute_decision_id(
        prediction_observation_id=observation_id,
        source_snapshot_row_fingerprint=row_fingerprint,
        provider_namespace="MLB_STATS_API",
        provider_game_id="888001",
        game_number=1,
        selection=selection,
    )
    return ResultOnlyPaperDecision(
        decision_id=decision_id,
        prediction_observation_id=observation_id,
        source_snapshot_row_fingerprint=row_fingerprint,
        provider_namespace="MLB_STATS_API",
        provider_game_id="888001",
        game_number=1,
        selection=selection,
        prediction_generated_at_utc="2026-04-05T11:00:00Z",
        scheduled_start_utc="2026-04-05T19:00:00Z",
    )


class ResultOnlyPaperDecisionContractTests(unittest.TestCase):
    def test_decision_is_frozen_and_id_is_prediction_time_only(self) -> None:
        decision = make_decision("prediction-1")
        self.assertEqual(decision.decision_id, decision.decision_id)
        with self.assertRaises(FrozenInstanceError):
            decision.selection = "AWAY"

    def test_settlement_status_is_result_only(self) -> None:
        self.assertEqual(settlement_status_for("HOME", "HOME"), SETTLEMENT_WON)
        self.assertEqual(settlement_status_for("HOME", "AWAY"), SETTLEMENT_LOST)

    def test_selection_freezes_sorted_decision_ids(self) -> None:
        first = make_decision("a")
        second = make_decision("b")
        ordered = tuple(sorted((first, second), key=lambda item: item.decision_id))
        selection = ResultOnlyDecisionSelection(
            schema_version=DECISION_SCHEMA_VERSION,
            source_snapshot_sha256="b" * 64,
            source_snapshot_summary_sha256="c" * 64,
            source_snapshot_fingerprint="d" * 64,
            excluded_row_count=0,
            decisions=ordered,
            decision_set_fingerprint=compute_decision_set_fingerprint(ordered),
        )
        self.assertEqual(
            tuple(item.decision_id for item in selection.decisions),
            tuple(item.decision_id for item in ordered),
        )

    def test_unsettled_settlement_cannot_carry_result_fields(self) -> None:
        decision = make_decision("prediction-1")
        fingerprint = compute_settlement_row_fingerprint(
            decision_id=decision.decision_id,
            prediction_observation_id=decision.prediction_observation_id,
            result_observation_id=None,
            provider_namespace=decision.provider_namespace,
            provider_game_id=decision.provider_game_id,
            game_number=decision.game_number,
            selection=decision.selection,
            settlement_status=SETTLEMENT_UNSETTLED,
            result_observed_at_utc=None,
            home_score=None,
            away_score=None,
            actual_winner=None,
        )
        settlement = ResultOnlyPaperSettlement(
            decision_id=decision.decision_id,
            prediction_observation_id=decision.prediction_observation_id,
            result_observation_id=None,
            provider_namespace=decision.provider_namespace,
            provider_game_id=decision.provider_game_id,
            game_number=decision.game_number,
            selection=decision.selection,
            settlement_status=SETTLEMENT_UNSETTLED,
            result_observed_at_utc=None,
            home_score=None,
            away_score=None,
            actual_winner=None,
            settlement_row_fingerprint=fingerprint,
        )
        self.assertEqual(settlement.settlement_status, SETTLEMENT_UNSETTLED)


if __name__ == "__main__":
    unittest.main()
