"""P23F2 PIT, identity, and result-separation tests."""

from dataclasses import replace
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
from match_analysis.baseball.domain.future_evaluation_fold import (
    FEATURE_NAMES,
    TRAINING_INFORMATION_BOUNDARY_UTC,
    fingerprint_rows,
)


NORMALIZED_ROOT = REPOSITORY_ROOT / "data/fixtures/p23f2_official_2026_history/normalized"
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


if __name__ == "__main__":
    unittest.main()
