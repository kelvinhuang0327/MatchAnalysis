"""Unit tests for the immutable P20A fold contract."""

import copy
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
)
from match_analysis.baseball.domain.moneyline_walk_forward_fold import (
    P20A_FEATURE_NAMES,
    MoneylineWalkForwardFold,
)


FIXTURE = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "p20a_p13_walk_forward"
    / "fold_wf_001.json"
)


class MoneylineWalkForwardFoldTests(unittest.TestCase):
    def test_bounded_fold_has_explicit_pit_boundary_and_multiple_rows(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        self.assertEqual(fold.fold_id, "wf_001")
        self.assertEqual(fold.training_row_count, 566)
        self.assertEqual(fold.prediction_row_count, 2)
        self.assertEqual(fold.feature_names, P20A_FEATURE_NAMES)
        self.assertTrue(fold.point_in_time_safe())
        self.assertEqual(fold.train_as_of, "2025-05-31")
        self.assertEqual(fold.validation_start, "2025-06-01")

    def test_future_training_row_is_rejected(self) -> None:
        projection = json.loads(FIXTURE.read_text(encoding="utf-8"))
        projection["training_rows"][-1]["date"] = "2025-06-01"
        with self.assertRaises(ValueError):
            MoneylineWalkForwardFold.from_projection(projection)

    def test_fold_fingerprint_is_byte_stable(self) -> None:
        first = load_moneyline_walk_forward_fold(FIXTURE)
        second = load_moneyline_walk_forward_fold(FIXTURE)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_prediction_outcome_is_not_required_as_training_state(self) -> None:
        projection = json.loads(FIXTURE.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(projection)
        mutated["prediction_rows"][0]["target_home_win"] = 1 - int(
            mutated["prediction_rows"][0]["target_home_win"]
        )
        original = MoneylineWalkForwardFold.from_projection(projection)
        changed = MoneylineWalkForwardFold.from_projection(mutated)
        self.assertEqual(
            tuple(row.to_projection() for row in original.training_rows),
            tuple(row.to_projection() for row in changed.training_rows),
        )


if __name__ == "__main__":
    unittest.main()
