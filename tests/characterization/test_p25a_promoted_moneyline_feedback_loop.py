"""Characterization of the committed P25A paper feedback artifacts."""

import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.paper_moneyline_feedback_artifacts import (
    render_paper_moneyline_feedback_artifacts,
)
from match_analysis.application.use_cases.settle_paper_moneyline_batch import (
    settle_paper_moneyline_batch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p25a_promoted_moneyline_feedback"


class P25APromotedMoneylineFeedbackLoopTests(unittest.TestCase):
    def test_committed_artifacts_reproduce_from_pinned_authority(self) -> None:
        result = settle_paper_moneyline_batch(REPOSITORY_ROOT)
        actual = render_paper_moneyline_feedback_artifacts(result)

        self.assertEqual(
            {path.name for path in REPORT_ROOT.iterdir()},
            set(actual),
        )
        for name, content in actual.items():
            self.assertEqual((REPORT_ROOT / name).read_bytes(), content, name)

        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["source_batch_id"], result.authority.batch_id)
        self.assertEqual(summary["raw_game_count"], 90)
        self.assertEqual(summary["prediction_count"], 79)
        self.assertEqual(summary["feature_unavailable_count"], 11)
        self.assertEqual(summary["settled_prediction_count"], 79)
        self.assertEqual(summary["evaluation_count"], 79)
        self.assertEqual(summary["feedback_row_count"], 79)
        self.assertTrue(summary["all_results_final"])
        self.assertTrue(summary["prediction_authority_verified"])
        self.assertTrue(summary["offline_settlement"])
        self.assertTrue(summary["model_promoted"])
        self.assertFalse(summary["challenger_retrained"])
        self.assertFalse(summary["deployment_performed"])
        self.assertFalse(summary["profitability_claim"])
        self.assertFalse(summary["production_ready"])
        self.assertFalse(summary["real_betting_recommendation"])
        self.assertTrue(summary["claims"]["all_results_final"])
        self.assertTrue(summary["claims"]["model_promoted"])
        self.assertEqual(summary["promotion_scope"], "paper_only")
        self.assertFalse(summary["claims"]["production_ready"])
        self.assertFalse(summary["claims"]["profitability_claim"])
        self.assertFalse(summary["claims"]["real_betting_recommendation"])
        self.assertEqual(
            summary["p20b_historical_runtime_compliance"],
            "REMAINS_REFUTED",
        )


if __name__ == "__main__":
    unittest.main()
