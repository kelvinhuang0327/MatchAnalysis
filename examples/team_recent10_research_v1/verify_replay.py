"""Read-only hash and byte-identical replay verification; makes no network requests."""
from pathlib import Path
from hashlib import sha256
import json
from evaluate import evaluate,table_csv

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_bytes())
for name,expected in manifest['files'].items():
    if sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:'+name)
result=evaluate(root)
rendered=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
if rendered!=(root/'results.json').read_bytes():raise ValueError('RESULT_REPLAY_NOT_BYTE_IDENTICAL')
if table_csv(result).encode()!=(root/'per_game.csv').read_bytes():raise ValueError('CSV_REPLAY_NOT_BYTE_IDENTICAL')
assert result['population_games']==102
assert result['uniquely_paired_with_usable_history_games']+result['excluded_games']==102
assert result['recent10_direction_games']+result['recent10_tie_games']==result['uniquely_paired_with_usable_history_games']
assert result['probability_mapping'] is None and result['brier'] is None and result['log_loss'] is None
assert result['optimization_trials']==result['bets_placed']==result['hypothesis_test_count']==0
print(json.dumps({'source_manifest_hashes_verified':len(manifest['files']),
    'json_byte_identical':True,'csv_byte_identical':True,'population':102,
    'included':result['uniquely_paired_with_usable_history_games'],'excluded':result['excluded_games'],
    'probability_mapping':None,'optimization_trials':0,'network_requests_during_replay':0},sort_keys=True))
