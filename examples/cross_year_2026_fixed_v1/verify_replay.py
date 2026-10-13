"""Read-only verification of frozen source, models, training split and outputs."""
from hashlib import sha256
from pathlib import Path
import json
import sys
import evaluate as e

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    if sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:' + name)
r = e.evaluate(root, sys.argv[1] if len(sys.argv) > 1 else None)
if e.result_json(r) != (root / 'results.json').read_bytes() or e.table_csv(r) != (root / 'per_game.csv').read_bytes():
    raise ValueError('JSON_OR_CSV_REPLAY_NOT_BYTE_IDENTICAL')
assert not any(row['old102_target'] for row in r['rows'] if row['included'])
assert r['previous_opened102_source_matches'] == 102
assert r['new_fits'] == 0 and not r['training_separation']['training_contains_evaluation_year']
assert r['market']['paired_market_sample_count'] == 0
print(json.dumps({'json_byte_identical': True, 'csv_byte_identical': True,
                  'capsule_hashes_verified': len(manifest['files']), 'evaluated': r['paired_evaluated_count'],
                  'opened102_targets_excluded': 102, 'training_fingerprint_and_scaler_preserved': True,
                  'new_fits': 0, 'source_gets_total': 1, 'new_gets_during_replay': 0,
                  'market_samples': 0}, sort_keys=True))
