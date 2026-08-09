"""Focused tests for deterministic P13 fold reconstruction."""

import importlib.util
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
    reconstruct_moneyline_walk_forward_model,
)


FIXTURE = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "p20a_p13_walk_forward"
    / "fold_wf_001.json"
)


def _legacy_fit_backend_available() -> bool:
    return (
        importlib.util.find_spec("numpy") is not None
        and importlib.util.find_spec("sklearn") is not None
    )


@unittest.skipUnless(
    _legacy_fit_backend_available(),
    "The exact legacy P13 fitting backend is not installed in this interpreter",
)
class ReconstructMoneylineWalkForwardModelTests(unittest.TestCase):
    def test_reconstruction_is_deterministic(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        first = reconstruct_moneyline_walk_forward_model(fold)
        second = reconstruct_moneyline_walk_forward_model(fold)
        self.assertEqual(first.to_projection(), second.to_projection())
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(
            first.fingerprint(),
            "5a0e7a0a90253c90eeed30a6f7672b578a76dc2a00a2fa3da7cf1196c0e80155",
        )

    def test_reconstruction_uses_only_training_rows(self) -> None:
        import dataclasses

        fold = load_moneyline_walk_forward_fold(FIXTURE)
        original = reconstruct_moneyline_walk_forward_model(fold)
        changed_row = dataclasses.replace(
            fold.prediction_rows[0],
            target_home_win=1 - int(fold.prediction_rows[0].target_home_win),
        )
        changed_fold = dataclasses.replace(
            fold,
            prediction_rows=(changed_row, *fold.prediction_rows[1:]),
        )
        changed = reconstruct_moneyline_walk_forward_model(changed_fold)
        self.assertEqual(original.to_projection(), changed.to_projection())


if __name__ == "__main__":
    unittest.main()
