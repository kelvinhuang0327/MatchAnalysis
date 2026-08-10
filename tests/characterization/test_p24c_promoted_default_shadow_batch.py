"""Characterization of the committed P24C promoted-default paper batch."""

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p24c_promoted_moneyline_shadow_batch"

P22B_MODEL_ID = "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630"
P22B_ARTIFACT_FINGERPRINT = "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e"
P23_ARTIFACT_SHA256 = {
    "report/p23b_contiguous_multifold_oos/fold_comparisons.jsonl": "f5374cf2be5d5b3878db11b3945abb18b047ce59081c513e2fa7b934301a5109",
    "report/p23b_contiguous_multifold_oos/per_fold_summary.json": "b58023a86105e14c04b493f5f71f360cffc4fb0ec90d546d94fbb5dd800f4911",
    "report/p23b_contiguous_multifold_oos/summary.json": "875f08033d6fd083e82fa37d639f7d3d4ccfef74786cf17f8ca2533503dcde70",
    "report/p23a_strictly_future_oos/comparisons.jsonl": "c7efb4ed77b541a422ce366e2b8ae0bbac28373c622558f56a00af60a5f42f92",
    "report/p23a_strictly_future_oos/summary.json": "739c8976b1cf66070126df79b1d5ceddef92e58a170dd31fd72ce931f508b96b",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class P24CPromotedDefaultShadowBatchTests(unittest.TestCase):
    def test_summary_records_the_complete_paper_shadow_contract(self) -> None:
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["batch_id"], "a43aa88cef4df6a3acac7a2fbdf04f6bad0b3a44c6b4eb96a3538bbc0953264c")
        self.assertEqual(summary["window_start_date"], "2026-06-14")
        self.assertEqual(summary["window_end_date"], "2026-06-20")
        self.assertEqual(summary["raw_game_count"], 90)
        self.assertEqual(summary["evaluable_game_count"], 79)
        self.assertEqual(summary["feature_unavailable_count"], 11)
        self.assertEqual(summary["promoted_default_model_id"], P22B_MODEL_ID)
        self.assertEqual(summary["promoted_default_model_fingerprint"], P22B_ARTIFACT_FINGERPRINT)
        for key in (
            "default_explicit_equivalence_verified",
            "explicit_incumbent_override_verified",
            "result_mutation_isolation_verified",
            "offline_replay_verified",
            "historical_shadow",
            "paper_only",
            "model_promoted",
        ):
            self.assertTrue(summary[key], key)
        for key in (
            "challenger_retrained",
            "production_ready",
            "deployment_performed",
            "real_betting_recommendation",
            "profitability_claim",
        ):
            self.assertFalse(summary[key], key)
        self.assertEqual(summary["promotion_scope"], "paper_only")
        self.assertEqual(summary["p20b_historical_runtime_compliance"], "REMAINS_REFUTED")

    def test_ledgers_have_raw_accounting_and_no_outcome_columns(self) -> None:
        predictions = _jsonl(REPORT_ROOT / "predictions.jsonl")
        unavailable = _jsonl(REPORT_ROOT / "feature_unavailable.jsonl")
        self.assertEqual(len(predictions), 79)
        self.assertEqual(len(unavailable), 11)
        prediction_ids = {row["game_id"] for row in predictions}
        unavailable_ids = {row["game_id"] for row in unavailable}
        self.assertTrue(prediction_ids.isdisjoint(unavailable_ids))
        self.assertEqual(len(prediction_ids | unavailable_ids), 90)
        forbidden = {"home_score", "away_score", "outcome", "result", "runs"}
        for row in predictions + unavailable:
            self.assertTrue(forbidden.isdisjoint(row))
        self.assertTrue(all(row["model_id"] == P22B_MODEL_ID for row in predictions))
        self.assertTrue(all(row["model_fingerprint"] == P22B_ARTIFACT_FINGERPRINT for row in predictions))
        self.assertTrue(all(row["inference_mode"] == "PAPER_DEFAULT" for row in predictions))
        self.assertTrue(all(row["status"] == "FEATURE_UNAVAILABLE" for row in unavailable))

    def test_manifest_is_official_scoped_and_p23_artifacts_are_unchanged(self) -> None:
        manifest = json.loads((REPORT_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_authority"], "MLB_STATS_API")
        self.assertEqual(manifest["historical_date_scope"], {"start": "2026-06-14", "end": "2026-06-20"})
        self.assertEqual(manifest["source_domains"], ["mlb.com"])
        self.assertEqual(len(manifest["records"]), 246)
        self.assertTrue(all("statsapi.mlb.com" in row["url"] for row in manifest["records"]))
        allowed_raw_prefixes = (
            "data/fixtures/p23f2_official_2026_history/raw/",
            "data/fixtures/p24c_promoted_moneyline_shadow_batch/raw/",
        )
        self.assertTrue(
            all(row["path"].startswith(allowed_raw_prefixes) for row in manifest["records"])
        )
        for relative, expected in P23_ARTIFACT_SHA256.items():
            actual = hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


if __name__ == "__main__":
    unittest.main()
