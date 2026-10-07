from copy import deepcopy
import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.audit_historical_price_readiness import audit_historical_price_readiness


ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/"examples/tsl_archive_eligibility_v1/readiness_protocol.json"


class HistoricalReadinessTests(unittest.TestCase):
    def test_real_102_universe_has_reference_scores_and_zero_champion_pairs(self):
        result=audit_historical_price_readiness(ROOT,PROTOCOL.read_bytes(),open_audit=True)
        self.assertEqual(result["denominators"]["price_eligible_games"],102)
        self.assertEqual(result["denominators"]["champion_market_paired_games"],0)
        self.assertEqual(result["denominators"]["development_games"],69)
        self.assertEqual(result["denominators"]["historical_audit_games"],33)
        self.assertTrue(all(n==102 for n in result["denominators"]["missing_evidence_counts"].values()))
        self.assertEqual(result["historical_reference_scores"]["all_opened"]["market"]["correct_count"],56)
        self.assertEqual(result["historical_reference_scores"]["all_opened"]["constant"]["correct_count"],57)
        self.assertIsNone(result["champion_vs_market"]["champion_metrics"]["accuracy"])
        self.assertEqual(result["optimization_trials_this_batch"],0)
        self.assertEqual(result,json.loads((PROTOCOL.parent/"readiness_and_reference_scores.json").read_bytes()))

    def test_actual_decision_missing_never_replaced_by_scheduled_start(self):
        result=audit_historical_price_readiness(ROOT,PROTOCOL.read_bytes())
        self.assertEqual(result["freshness"]["verified_decision_time_games"],0)
        self.assertEqual(result["freshness"]["decision_freshness_status"],"UNKNOWN")
        self.assertTrue(all(r["age_proxy_is_admission"] is False for r in result["rows"]))
        self.assertTrue(all(r["model_pair_status"]=="NOT_EVALUABLE" for r in result["rows"]))

    def test_default_does_not_score_last_historical_audit(self):
        result=audit_historical_price_readiness(ROOT,PROTOCOL.read_bytes())
        self.assertEqual(result["historical_reference_scores"]["all_opened"]["market"]["sample_count"],69)
        self.assertEqual(result["historical_reference_scores"]["historical_audit"]["market"]["sample_count"],0)
        self.assertTrue(all("actual_home_win" not in r for r in result["rows"] if r["role"]=="historical_audit"))

    def test_frozen_input_hash_and_full_population_cannot_silently_change(self):
        protocol=json.loads(PROTOCOL.read_bytes())
        changed=deepcopy(protocol);changed["sources"][next(iter(changed["sources"]))]="0"*64
        with self.assertRaisesRegex(ValueError,"source hash mismatch"):
            audit_historical_price_readiness(ROOT,json.dumps(changed).encode())
        changed=deepcopy(protocol);changed["population"]["expected_games"]=100
        with self.assertRaisesRegex(ValueError,"universe count"):
            audit_historical_price_readiness(ROOT,json.dumps(changed).encode())
        changed=deepcopy(protocol);changed["baseline_controls"]["constant_home_probability"]="0.7"
        with self.assertRaisesRegex(ValueError,"fixed reference"):
            audit_historical_price_readiness(ROOT,json.dumps(changed).encode())


if __name__=="__main__":unittest.main()
