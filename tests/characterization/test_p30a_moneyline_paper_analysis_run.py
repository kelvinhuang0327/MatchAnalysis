"""Characterization of the committed P30A paper-analysis artifact."""

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPOSITORY_ROOT / "report/p30a_moneyline_paper_analysis"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class P30AMoneylinePaperAnalysisTests(unittest.TestCase):
    def test_summary_records_complete_paper_only_contract(self) -> None:
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "p30a.moneyline_paper_analysis.v1")
        self.assertEqual(summary["operation"], "MONEYLINE_PAPER_ANALYSIS_RUN")
        self.assertEqual(summary["raw_game_count"], 16)
        self.assertEqual(summary["edge_available_count"], 9)
        self.assertEqual(summary["feature_unavailable_count"], 7)
        self.assertEqual(summary["price_unavailable_pre_cutoff_count"], 0)
        self.assertEqual(
            summary["p28ab_crosswalk_status_counts"],
            {
                "MATCHED_FINAL": 31,
                "NO_CANONICAL_TEAM_CODE": 8,
                "POSTPONED_OR_NON_FINAL": 7,
            },
        )
        self.assertEqual(
            summary["other_structural_exclusion_counts"],
            {"NO_CANONICAL_TEAM_CODE": 8, "POSTPONED_OR_NON_FINAL": 7},
        )
        self.assertTrue(summary["deterministic_replay_verified"])
        self.assertTrue(summary["outcome_isolation_verified"])
        self.assertTrue(summary["historical_shadow"])
        self.assertTrue(summary["paper_only"])
        self.assertTrue(summary["moneyline_model_promoted"])
        self.assertEqual(summary["moneyline_promotion_scope"], "paper_only")
        self.assertFalse(summary["decision_policy_used"])
        self.assertFalse(summary["staking_implemented"])
        self.assertFalse(summary["profitability_claim"])
        self.assertFalse(summary["real_betting_recommendation"])
        self.assertEqual(
            summary["run_line_migration_status"],
            "BLOCKED_NO_PIT_SAFE_AUTHORITY",
        )
        self.assertEqual(
            summary["total_migration_status"],
            "BLOCKED_NO_PIT_SAFE_AUTHORITY",
        )
        self.assertEqual(
            summary["legacy_decision_policy_status"],
            "BLOCKED_NO_PIT_SAFE_AUTHORITY",
        )

    def test_analysis_rows_are_unique_and_have_no_outcome_fields(self) -> None:
        rows = _jsonl(REPORT_ROOT / "analysis.jsonl")
        summary = json.loads((REPORT_ROOT / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row["game_id"] for row in rows}), 16)
        self.assertEqual(
            hashlib.sha256((REPORT_ROOT / "analysis.jsonl").read_bytes()).hexdigest(),
            summary["analysis_set_fingerprint"],
        )
        forbidden = {
            "away_score",
            "bankroll",
            "bet",
            "clv",
            "final",
            "home_score",
            "kelly",
            "result",
            "roi",
            "runs",
            "settlement",
            "stake",
            "winner",
        }
        for row in rows:
            self.assertTrue(forbidden.isdisjoint(row))
            self.assertEqual(row["run_id"], summary["run_id"])


if __name__ == "__main__":
    unittest.main()
