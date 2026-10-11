"""Read-only byte replay; no training path and no new data retrieval."""
from hashlib import sha256
from pathlib import Path
import json
import sys
import source_gate

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest.json').read_bytes())
for relative, expected in manifest['capsule_files'].items():
    if sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:' + relative)
for relative, expected in manifest['sealed_dependencies'].items():
    if sha256((root.parent.parent / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('SEALED_DEPENDENCY_CHANGED:' + relative)
result = source_gate.audit(legacy_root=sys.argv[1] if len(sys.argv) > 1 else None)
if source_gate.result_json(result) != (root / 'source_gate_result.json').read_bytes():
    raise ValueError('JSON_REPLAY_DIFFERS')
if source_gate.table_csv(result).encode() != (root / 'per_game_gate.csv').read_bytes():
    raise ValueError('CSV_REPLAY_DIFFERS')
if any(result[k] != 0 for k in ['baseline_fit_count', 'candidate_fit_count', 'history_events_admitted']):
    raise ValueError('BLOCKED_SOURCE_CANNOT_BE_FITTED_OR_ADMITTED')
print(json.dumps({'json_byte_identical': True, 'csv_byte_identical': True,
                  'capsule_hashes_verified': len(manifest['capsule_files']),
                  'sealed_dependencies_verified': len(manifest['sealed_dependencies']),
                  'population': len(result['rows']), 'train_population': result['phases']['train']['population'],
                  'evaluation_population': result['phases']['evaluation']['population'],
                  'baseline_fits': 0, 'candidate_fits': 0, 'paired_admitted': 0,
                  'new_game_source_gets': 0}, sort_keys=True))
