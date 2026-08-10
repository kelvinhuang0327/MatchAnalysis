"""Characterization of the committed P23F2 official future fold."""

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p23f2_official_future_fold"


class P23F2OfficialFutureFoldTests(unittest.TestCase):
    def test_summary_is_strict_future_and_not_model_evaluation(self) -> None:
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["fold_id"], "wf_004")
        self.assertEqual(summary["game_count"], 23)
        self.assertEqual(summary["feature_names"], ["recent_win_rate_delta", "starter_era_delta"])
        self.assertTrue(summary["strict_future"])
        self.assertTrue(summary["external_source"])
        self.assertTrue(summary["offline_replay_verified"])
        self.assertFalse(summary["model_evaluated"])
        self.assertFalse(summary["model_promoted"])
        self.assertFalse(summary["production_ready"])

    def test_sources_are_mlb_owned_and_results_are_separate(self) -> None:
        manifest = json.loads((REPORT_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_domains"], ["mlb.com"])
        self.assertTrue(manifest["records"])
        self.assertTrue(all("statsapi.mlb.com" in row["url"] for row in manifest["records"]))
        feature_ids = {
            json.loads(line)["provider_game_id"]
            for line in (REPORT_ROOT / "feature_rows.jsonl").read_text(encoding="utf-8").splitlines()
        }
        result_ids = {
            json.loads(line)["provider_game_id"]
            for line in (REPORT_ROOT / "results.jsonl").read_text(encoding="utf-8").splitlines()
        }
        self.assertEqual(feature_ids, result_ids)


if __name__ == "__main__":
    unittest.main()
