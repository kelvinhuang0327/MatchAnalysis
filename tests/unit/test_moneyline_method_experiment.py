from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.run_moneyline_method_experiment import run_moneyline_method_experiment


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "examples/method_loop_v1/protocol.json"


class MoneylineMethodExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = json.loads(PROTOCOL.read_bytes())
        for name in self.protocol["sources"]:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())

    def run_trial(self, audit=False):
        return run_moneyline_method_experiment(self.root, json.dumps(self.protocol).encode(), open_audit=audit)

    def change_rows(self, name, mutate):
        path = self.root/name
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        mutate(rows)
        raw = "".join(json.dumps(r)+"\n" for r in rows).encode()
        path.write_bytes(raw)
        self.protocol["sources"][name] = sha256(raw).hexdigest()

    def test_real_paired_baselines_denominators_and_failed_single_candidate(self):
        result = self.run_trial(audit=True)
        self.assertEqual(result["coverage"]["original_targets"], 75)
        self.assertEqual(result["coverage"]["model_evaluable"], 65)
        self.assertEqual(result["coverage"]["paired_scored"], 62)
        self.assertEqual(result["coverage"]["quote_excluded"], 3)
        self.assertEqual(len(result["model_excluded_records"]), 10)
        self.assertEqual({r["reason"] for r in result["model_excluded_records"]},
                         {"INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"})
        methods = result["scores"]["all_opened"]["methods"]
        self.assertEqual([methods[x]["correct"] for x in ("incumbent", "market", "candidate")], [42, 33, 40])
        self.assertAlmostEqual(methods["incumbent"]["brier"], 0.2294924498542968)
        self.assertEqual(result["decision"], "REJECT_CANDIDATE")
        self.assertEqual(result["trial_accounting"]["this_batch_candidates"], 1)
        self.assertEqual(result["scores"]["retrospective_audit"]["methods"]["candidate"]["n"], 25)

    def test_default_does_not_score_terminal_audit(self):
        first = self.run_trial()
        self.change_rows("report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
                         lambda rows: [r.update(target_home_win=1-r["target_home_win"])
                                       for r in rows if r["fold_id"] == "wf_006"])
        second = self.run_trial()
        self.assertEqual(first["scores"], second["scores"])
        self.assertNotIn("retrospective_audit", second["scores"])
        self.assertEqual(second["coverage"]["paired_scored"], 37)
        with self.assertRaisesRegex(ValueError, "official final score"):
            self.run_trial(audit=True)

    def test_frozen_hash_rejects_revised_source(self):
        path = self.root/"report/p39a_tsl_moneyline_market_join/market_join.jsonl"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_trial()

    def test_same_id_wrong_team_rejected_even_with_new_hash(self):
        self.change_rows("report/p39a_tsl_moneyline_market_join/market_join.jsonl",
                         lambda rows: rows[0].update(home_team_code="WRONG"))
        with self.assertRaisesRegex(ValueError, "team identity"):
            self.run_trial()

    def test_post_start_quote_rejected_even_when_snapshot_and_join_agree(self):
        name = "report/p39a_tsl_moneyline_market_join/market_join.jsonl"
        rows = [json.loads(s) for s in (self.root/name).read_text().splitlines()]
        row = rows[0]
        snapshot_id = row["market_snapshot_id"]
        stamp = row["scheduled_start_utc"]
        self.change_rows(name, lambda rows: rows[0].update(market_observed_at_utc=stamp))
        self.change_rows("report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl",
                         lambda rows: [r.update(market_observed_at_utc=stamp) for r in rows if r["snapshot_id"]==snapshot_id])
        with self.assertRaisesRegex(ValueError, "post-start quote"):
            self.run_trial()

    def test_duplicate_game_and_unregistered_method_fail(self):
        self.change_rows("report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
                         lambda rows: rows.append(deepcopy(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicate full"):
            self.run_trial()
        self.protocol["candidate"]["model_weight"] = 0.7
        with self.assertRaisesRegex(ValueError, "new trial"):
            self.run_trial()


if __name__ == "__main__":
    unittest.main()
