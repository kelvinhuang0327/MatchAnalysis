"""Unit tests for pure paired P23A comparison metrics."""

from decimal import Decimal
import unittest

from match_analysis.baseball.domain.moneyline_oos_comparison import (
    aggregate_metrics,
    build_comparison_row,
)


class MoneylineOOSComparisonTests(unittest.TestCase):
    def _row(self, *, game_id: str, target: int, challenger: str, incumbent: str):
        return build_comparison_row(
            fold_id="wf_004",
            feature_row={
                "provider_game_id": game_id,
                "game_pk": int(game_id),
                "game_number": 1,
                "scheduled_start_utc": f"2026-06-08T22:{game_id[-2:]}:00Z",
                "feature_fingerprint": "a" * 64,
            },
            result_row={
                "provider_game_id": game_id,
                "scheduled_start_utc": f"2026-06-08T22:{game_id[-2:]}:00Z",
                "home_score": 2 if target else 1,
                "away_score": 1 if target else 2,
            },
            challenger_model_id="challenger",
            challenger_model_fingerprint="b" * 64,
            challenger_home_probability=Decimal(challenger),
            incumbent_model_id="incumbent",
            incumbent_model_fingerprint="c" * 64,
            incumbent_home_probability=Decimal(incumbent),
        )

    def test_brier_and_accuracy_metrics_are_paired(self) -> None:
        rows = (
            self._row(game_id="1001", target=1, challenger="0.8", incumbent="0.6"),
            self._row(game_id="1002", target=0, challenger="0.4", incumbent="0.7"),
        )
        metrics = aggregate_metrics(rows)
        self.assertEqual(metrics["game_count"], 2)
        self.assertEqual(metrics["challenger_mean_brier"], "0.10")
        self.assertEqual(metrics["incumbent_mean_brier"], "0.325")
        self.assertEqual(metrics["brier_delta"], "-0.225")
        self.assertEqual(metrics["challenger_accuracy"], "1")
        self.assertEqual(metrics["incumbent_accuracy"], "0.5")
        self.assertEqual(metrics["challenger_brier_better_count"], 2)
        self.assertEqual(metrics["incumbent_brier_better_count"], 0)
        self.assertEqual(metrics["equal_brier_count"], 0)

    def test_tied_final_score_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tied final scores"):
            build_comparison_row(
                fold_id="wf_004",
                feature_row={
                    "provider_game_id": "1003",
                    "game_pk": 1003,
                    "game_number": 1,
                    "scheduled_start_utc": "2026-06-08T22:03:00Z",
                    "feature_fingerprint": "a" * 64,
                },
                result_row={
                    "provider_game_id": "1003",
                    "scheduled_start_utc": "2026-06-08T22:03:00Z",
                    "home_score": 3,
                    "away_score": 3,
                },
                challenger_model_id="challenger",
                challenger_model_fingerprint="b" * 64,
                challenger_home_probability=Decimal("0.5"),
                incumbent_model_id="incumbent",
                incumbent_model_fingerprint="c" * 64,
                incumbent_home_probability=Decimal("0.5"),
            )


if __name__ == "__main__":
    unittest.main()
