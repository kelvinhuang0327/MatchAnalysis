from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.interfaces.cli.replay_forward_moneyline_baseline import main


ROOT = Path(__file__).resolve().parents[2]


class ForwardBaselineReplayTests(unittest.TestCase):
    def test_portable_replay_preserves_market_denominator_and_rejects_membership_change(self):
        source = ROOT/"examples/method_loop_v1/forward_market_baseline_sources.json"
        manifest = json.loads(source.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in manifest["files"]:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT/name).read_bytes())
            output = root/"replay.json"
            args = ["--repository-root",str(root),"--sources",str(source),"--output",str(output)]
            self.assertEqual(main(args), 0)
            self.assertEqual(sha256(output.read_bytes()).hexdigest(), manifest["output_sha256"])
            report = json.loads(output.read_bytes())
            regular = report["game_type_cohorts"]["R"]
            self.assertEqual(regular["model_final_metrics"]["sample_count"], 41)
            self.assertEqual(regular["model_final_metrics"]["correct_count"], 24)
            self.assertEqual(regular["market_scored_count"], 0)
            self.assertEqual(report["input_counts"]["non_moneyline_forward_ledger_rows_excluded"], 42)
            self.assertIsNone(report["fixed_unit_paper_roi"]["net_roi"])
            self.assertEqual(report["fixed_unit_paper_roi"]["no_bet_count"], 4)
            extra = root/"report/moneyline_paper_decision_v1/extra/quote_snapshots.jsonl"
            extra.parent.mkdir()
            extra.write_text("")
            with self.assertRaisesRegex(ValueError, "membership changed"):
                main([*args[:-1],str(root/"new.json")])


if __name__ == "__main__":
    unittest.main()
