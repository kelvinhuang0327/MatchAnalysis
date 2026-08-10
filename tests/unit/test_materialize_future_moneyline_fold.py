"""P23F2 PIT, identity, and result-separation tests."""

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.acquire_future_moneyline_history import load_normalized_rows
from match_analysis.application.use_cases.future_moneyline_fold_artifacts import canonical_json_bytes
from match_analysis.application.use_cases.materialize_future_moneyline_fold import materialize_future_moneyline_fold, materialize_from_normalized_dir
from match_analysis.application.use_cases.materialize_future_moneyline_fold import classify_future_feature_eligibility
from match_analysis.baseball.domain.future_evaluation_fold import (
    FEATURE_NAMES,
    TRAINING_INFORMATION_BOUNDARY_UTC,
    fingerprint_rows,
)


NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p23f2_official_2026_history/normalized"
P23B_NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p23b_future_folds/wf_005/normalized"
SOURCE_MANIFEST = REPOSITORY_ROOT / "report/p23f2_official_future_fold/source_manifest.json"


def source_fingerprint() -> str:
    value = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class MaterializeFutureMoneylineFoldTests(unittest.TestCase):
    def test_committed_fixture_materializes_strict_future_fold(self) -> None:
        fold = materialize_from_normalized_dir(
            NORMALIZED_ROOT,
            source_manifest_fingerprint=source_fingerprint(),
        )
        self.assertEqual(fold.fold_id, "wf_004")
        self.assertEqual(len(fold.feature_rows), 23)
        self.assertEqual(
            tuple(fold.feature_rows[0].projection()["features"]),
            FEATURE_NAMES,
        )
        self.assertTrue(
            all(row.scheduled_start_utc > TRAINING_INFORMATION_BOUNDARY_UTC for row in fold.feature_rows)
        )
        self.assertTrue(all(result.status == "Final" for result in fold.result_rows))

    def test_input_order_does_not_change_semantic_identity(self) -> None:
        schedule = load_normalized_rows(NORMALIZED_ROOT / "schedule.jsonl")
        boxes = load_normalized_rows(NORMALIZED_ROOT / "target_boxscores.jsonl")
        logs = load_normalized_rows(NORMALIZED_ROOT / "pitcher_game_logs.jsonl")
        first = materialize_future_moneyline_fold(
            schedule_rows=schedule,
            target_boxscore_rows=boxes,
            pitcher_game_log_rows=logs,
            source_manifest_fingerprint=source_fingerprint(),
        )
        second = materialize_future_moneyline_fold(
            schedule_rows=tuple(reversed(schedule)),
            target_boxscore_rows=tuple(reversed(boxes)),
            pitcher_game_log_rows=tuple(reversed(logs)),
            source_manifest_fingerprint=source_fingerprint(),
        )
        self.assertEqual(first.feature_fingerprint, second.feature_fingerprint)
        self.assertEqual(first.result_fingerprint, second.result_fingerprint)
        self.assertEqual(first.fold_fingerprint, second.fold_fingerprint)

    def test_result_replacement_does_not_change_features(self) -> None:
        fold = materialize_from_normalized_dir(
            NORMALIZED_ROOT,
            source_manifest_fingerprint=source_fingerprint(),
        )
        mutated = replace(fold.result_rows[0], home_score=0, away_score=99)
        mutated_fold = replace(
            fold,
            result_rows=(mutated, *fold.result_rows[1:]),
        )
        original_result_fingerprint = fingerprint_rows(
            tuple(row.projection() for row in fold.result_rows)
        )
        mutated_result_fingerprint = fingerprint_rows(
            tuple(row.projection() for row in mutated_fold.result_rows)
        )
        self.assertNotEqual(mutated.home_score, fold.result_rows[0].home_score)
        self.assertNotEqual(original_result_fingerprint, mutated_result_fingerprint)
        self.assertEqual(mutated_fold.feature_fingerprint, fold.feature_fingerprint)
        self.assertEqual(mutated_fold.feature_rows, fold.feature_rows)


class P23BFeatureEligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = load_normalized_rows(P23B_NORMALIZED_ROOT / "schedule.jsonl")
        self.history_schedule = load_normalized_rows(NORMALIZED_ROOT / "schedule.jsonl")
        self.boxes = load_normalized_rows(P23B_NORMALIZED_ROOT / "target_boxscores.jsonl")
        self.logs = load_normalized_rows(P23B_NORMALIZED_ROOT / "pitcher_game_logs.jsonl")

    def _eligibility(self, *, logs=None, schedule=None):
        return classify_future_feature_eligibility(
            schedule_rows=tuple(schedule or self.schedule),
            target_boxscore_rows=tuple(self.boxes),
            pitcher_game_log_rows=tuple(logs or self.logs),
            fold_id="wf_005",
            validation_start="2026-06-10",
            validation_end="2026-06-11",
        )

    def _logs_with_prior_starts(self, count: int):
        rows = [dict(row) for row in self.logs]
        prior = [
            row
            for row in rows
            if row["player_id"] == 702474 and row["date"] < "2026-06-10"
        ]
        self.assertGreaterEqual(len(prior), count)
        for index, row in enumerate(prior):
            row["games_started"] = 1 if index < count else 0
        return rows

    def test_zero_prior_same_season_starts_is_feature_unavailable(self) -> None:
        result = self._eligibility()
        row = next(row for row in result.feature_unavailable_rows if row["game_id"] == "824266")
        self.assertEqual(row["status"], "FEATURE_UNAVAILABLE")
        self.assertEqual(row["reason"], "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY")
        affected = next(starter for starter in row["affected_starters"] if starter["starter_id"] == 702474)
        self.assertEqual(affected["qualifying_prior_start_count"], 0)
        self.assertEqual(affected["required_prior_start_count"], 2)

    def test_one_prior_same_season_start_is_feature_unavailable(self) -> None:
        result = self._eligibility(logs=self._logs_with_prior_starts(1))
        row = next(row for row in result.feature_unavailable_rows if row["game_id"] == "824266")
        affected = next(starter for starter in row["affected_starters"] if starter["starter_id"] == 702474)
        self.assertEqual(affected["qualifying_prior_start_count"], 1)

    def test_two_prior_same_season_starts_materialize_normally(self) -> None:
        result = self._eligibility(logs=self._logs_with_prior_starts(2))
        self.assertIn("824266", result.evaluable_game_ids)
        self.assertNotIn("824266", {row["game_id"] for row in result.feature_unavailable_rows})
        fold = materialize_future_moneyline_fold(
            schedule_rows=tuple(self.history_schedule),
            target_boxscore_rows=tuple(self.boxes),
            pitcher_game_log_rows=tuple(self._logs_with_prior_starts(2)),
            source_manifest_fingerprint="a" * 64,
            fold_id="wf_005",
            validation_start="2026-06-10",
            validation_end="2026-06-11",
            evaluable_game_ids=frozenset(result.evaluable_game_ids),
            raw_game_ids=result.raw_game_ids,
            feature_unavailable_rows=result.feature_unavailable_rows,
        )
        materialized = next(row for row in fold.feature_rows if row.provider_game_id == "824266")
        self.assertTrue(all(value.is_finite() for value in (Decimal(materialized.recent_win_rate_delta), Decimal(materialized.starter_era_delta))))

    def test_unavailable_game_has_no_fallback_feature_row(self) -> None:
        result = self._eligibility()
        fold = materialize_future_moneyline_fold(
            schedule_rows=tuple(self.history_schedule),
            target_boxscore_rows=tuple(self.boxes),
            pitcher_game_log_rows=tuple(self.logs),
            source_manifest_fingerprint="a" * 64,
            fold_id="wf_005",
            validation_start="2026-06-10",
            validation_end="2026-06-11",
            evaluable_game_ids=frozenset(result.evaluable_game_ids),
            raw_game_ids=result.raw_game_ids,
            feature_unavailable_rows=result.feature_unavailable_rows,
        )
        self.assertNotIn("824266", {row.provider_game_id for row in fold.feature_rows})
        self.assertIn("824266", fold.raw_game_ids)
        self.assertEqual(len(fold.raw_game_ids), len(fold.feature_rows) + len(fold.feature_unavailable_rows))

    def test_outcome_mutation_does_not_change_eligibility(self) -> None:
        mutated_schedule = [dict(row) for row in self.schedule]
        for row in mutated_schedule:
            if row["provider_game_id"] == "824266":
                row["home_score"], row["away_score"] = 99, 0
        first = self._eligibility()
        second = self._eligibility(schedule=mutated_schedule)
        self.assertEqual(first, second)

    def test_input_order_does_not_change_eligibility_identities_or_counts(self) -> None:
        first = self._eligibility()
        second = self._eligibility(
            logs=tuple(reversed(self.logs)),
            schedule=tuple(reversed(self.schedule)),
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
