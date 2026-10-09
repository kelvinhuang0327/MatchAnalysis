"""Hash-bound, read-only replay of the blocked data gate; no training or network."""
from hashlib import sha256
from pathlib import Path
import json
import sys
from data_gate import audit

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_bytes())
for name,expected in manifest['files'].items():
    if sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:'+name)
legacy_root=Path(sys.argv[1]) if len(sys.argv)>1 else None
result=audit(root,legacy_root)
rendered=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
if rendered!=(root/'data_gate_result.json').read_bytes():raise ValueError('REPLAY_NOT_BYTE_IDENTICAL')
assert result['candidate_fit_count']==result['prediction_count']==result['paired_model_market_games']==0
assert result['evaluation_population']==result['excluded_from_candidate_comparison']==102
assert result['paired_prestart_two_sided_market_records']==result['reusable_verified_target_outcomes']==102
assert result['efficacy_status']=='NOT_EVALUABLE'
assert all(m['sample_count']==0 and m['accuracy'] is None and m['brier'] is None and m['logloss'] is None
           for m in result['same_game_probability_metrics'].values())
assert result['near10_rule_rerun_count']==result['optimization_trials']==result['new_data_source_gets']==0
print(json.dumps({'byte_identical_data_gate_replay':True,'capsule_hashes_verified':len(manifest['files']),
    'training_population':2430,'verified_actual_completion_training_rows':0,'evaluation_population':102,
    'pregame_market_records':102,'paired_model_market_denominator':0,'fits':0,'source_gets':0},sort_keys=True))
