"""Characterization of the bounded committed P13 replay slice."""

from decimal import Decimal
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
)
from match_analysis.application.use_cases.replay_historical_moneyline_predictions import (
    replay_historical_moneyline_predictions,
)
from match_analysis.baseball.domain.moneyline_walk_forward_fold import (
    ReconstructedWalkForwardModel,
)


FIXTURE = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "p20a_p13_walk_forward"
    / "fold_wf_001.json"
)


class P13WalkForwardReconstructionParityTests(unittest.TestCase):
    def test_selected_rows_match_reproducible_legacy_outputs(self) -> None:
        fold = load_moneyline_walk_forward_fold(FIXTURE)
        model = ReconstructedWalkForwardModel(
            fold_id=fold.fold_id,
            feature_names=fold.feature_names,
            coefficients=(0.2820639975011785, -0.09609616789072993),
            intercept=0.19592616199017746,
            scaler_means=(-0.004478903507172063, -0.09830887114456008),
            scaler_stds=(0.2364885946804932, 2.1424749536909156),
            train_size=fold.training_row_count,
        )
        result = replay_historical_moneyline_predictions(fold, model)
        self.assertEqual(
            tuple(row.game_id for row in result.parity_rows),
            ("2025-06-01_TEX_STL", "2025-06-01_ATL_BOS"),
        )
        self.assertEqual(
            tuple(row.expected_home_probability for row in result.parity_rows),
            (Decimal("0.469229"), Decimal("0.539322")),
        )
        self.assertTrue(result.parity_passed)


if __name__ == "__main__":
    unittest.main()
