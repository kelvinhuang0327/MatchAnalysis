"""Focused contracts for the P39A market join and snapshot selection."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import unittest

from match_analysis.application.use_cases.join_p37_oos_market_snapshots import (
    P37OOSPrediction,
    build_p39a_market_join,
    load_p37_predictions,
)
from match_analysis.baseball.domain.moneyline_market_snapshot import (
    MoneylineMarketObservationCandidate,
)


SOURCE_MANIFEST = {
    "source_repository": "/legacy",
    "source_path": "/legacy/data/tsl_odds_history.jsonl",
    "source_relative_path": "data/tsl_odds_history.jsonl",
    "source_head": "a" * 40,
    "source_tree": "b" * 40,
    "source_blob_at_head": "c" * 40,
    "source_branch": "task",
    "source_status": "clean",
    "source_sha256": "d" * 64,
    "source_stable": True,
    "timestamp_semantics_trusted": True,
    "source_row_count": 4,
    "scoped_source_row_count": 2,
}


def _prediction() -> P37OOSPrediction:
    return P37OOSPrediction(
        fold_id="wf_004",
        evaluation_window_id="window_001_holdout_wf_004",
        comparison_row_id="comparison-row",
        provider_namespace="MLB_STATS_API",
        provider_game_id="824829",
        game_pk=824829,
        game_number=1,
        official_date="2026-06-08",
        scheduled_start_utc="2026-06-08T22:35:00Z",
        home_team="Baltimore Orioles",
        away_team="Seattle Mariners",
        home_team_code="BAL",
        away_team_code="SEA",
        challenger_home_probability=Decimal("0.5217"),
    )


def _candidate(
    *,
    fetched_at: str,
    row_index: int,
    source_match_id: str = "3473130.1",
    status: str = "VALID_PREGAME",
    reason: str | None = None,
) -> MoneylineMarketObservationCandidate:
    return MoneylineMarketObservationCandidate(
        source_row_index=row_index,
        source_row_fingerprint=f"{row_index:064x}",
        source_match_id=source_match_id,
        source_home_team_name="巴爾的摩金鶯",
        source_away_team_name="西雅圖水手",
        source_home_code="BAL",
        source_away_code="SEA",
        scheduled_start_utc="2026-06-08T22:35:00Z",
        market_observed_at_utc=fetched_at if status != "MISSING_OR_UNTRUSTED_TIMESTAMP" else None,
        local_fetched_at_utc=fetched_at if status != "MISSING_OR_UNTRUSTED_TIMESTAMP" else None,
        provider_observed_at_utc=None,
        is_pregame=True,
        market_code="MNL",
        market_status=status,
        rejection_reason=reason,
        home_decimal_price=Decimal("1.94") if status != "MISSING_OR_UNTRUSTED_TIMESTAMP" else None,
        away_decimal_price=Decimal("1.56") if status != "MISSING_OR_UNTRUSTED_TIMESTAMP" else None,
    )


class P39AMarketJoinTests(unittest.TestCase):
    def test_latest_valid_pregame_snapshot_is_selected_without_edge_optimization(self) -> None:
        prediction = _prediction()
        early = _candidate(fetched_at="2026-06-08T20:00:00Z", row_index=1)
        latest = _candidate(fetched_at="2026-06-08T21:00:00Z", row_index=2)

        result = build_p39a_market_join(
            [prediction],
            [latest, early],
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )

        self.assertEqual(result.summary["edge_ready_count"], 1)
        self.assertEqual(result.summary["conclusion"], "MARKET_JOIN_READY")
        self.assertEqual(
            result.join_rows[0].market_observed_at_utc,
            "2026-06-08T21:00:00Z",
        )
        self.assertEqual(result.join_rows[0].prediction.predicted_side, "HOME")
        self.assertNotIn("target_home_win", result.join_rows[0].to_projection())
        self.assertNotIn("actual_winner", result.join_rows[0].to_projection())

    def test_candidate_order_does_not_change_selection(self) -> None:
        prediction = _prediction()
        candidates = [
            _candidate(fetched_at="2026-06-08T20:00:00Z", row_index=1),
            _candidate(fetched_at="2026-06-08T21:00:00Z", row_index=2),
        ]
        first = build_p39a_market_join(
            [prediction],
            candidates,
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )
        second = build_p39a_market_join(
            [prediction],
            list(reversed(candidates)),
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )
        self.assertEqual(first.comparable_projection(), second.comparable_projection())

    def test_post_start_and_ambiguous_identity_fail_closed(self) -> None:
        prediction = _prediction()
        post_start = _candidate(
            fetched_at="2026-06-08T22:36:00Z",
            row_index=1,
            status="POST_START",
            reason="POST_START",
        )
        post_result = build_p39a_market_join(
            [prediction],
            [post_start],
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )
        self.assertEqual(post_result.summary["post_start_rejected_rows"], 1)
        self.assertEqual(post_result.join_rows[0].rejection_reason, "POST_START")

        ambiguous = [
            _candidate(
                fetched_at="2026-06-08T20:00:00Z",
                row_index=2,
                source_match_id="source-a",
            ),
            _candidate(
                fetched_at="2026-06-08T21:00:00Z",
                row_index=3,
                source_match_id="source-b",
            ),
        ]
        ambiguous_result = build_p39a_market_join(
            [prediction],
            ambiguous,
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )
        self.assertEqual(ambiguous_result.summary["ambiguous_rows"], 1)
        self.assertEqual(
            ambiguous_result.join_rows[0].rejection_reason,
            "AMBIGUOUS_SOURCE_GAME_IDENTITY",
        )

    def test_authoritative_p37_loader_is_exactly_65_rows(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        predictions, manifest = load_p37_predictions(repository_root)
        self.assertEqual(len(predictions), 65)
        self.assertEqual(
            {prediction.fold_id for prediction in predictions},
            {"wf_004", "wf_005", "wf_006"},
        )
        self.assertEqual(manifest["evaluable_row_count"], 65)
        self.assertEqual(
            manifest["fold_counts"],
            {"wf_004": 23, "wf_005": 17, "wf_006": 25},
        )


if __name__ == "__main__":
    unittest.main()
