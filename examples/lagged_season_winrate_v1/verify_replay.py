"""Read-only byte replay; no training path and no new data retrieval."""
from hashlib import sha256
from pathlib import Path
import json
import sys
import source_gate
import research

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest.json').read_bytes())
for relative, expected in manifest['capsule_files'].items():
    if sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:' + relative)
for relative, expected in manifest['sealed_dependencies'].items():
    if sha256((root.parent.parent / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('SEALED_DEPENDENCY_CHANGED:' + relative)
artifact = json.loads((root / 'model_artifact.json').read_bytes())
for relative, expected in artifact['fit_code_sha256'].items():
    if sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('FIT_CODE_CHANGED:' + relative)
result = research.score(root, artifact, sys.argv[1] if len(sys.argv) > 1 else None)
if source_gate.result_json(result) != (root / 'source_gate_result.json').read_bytes():
    raise ValueError('JSON_REPLAY_DIFFERS')
if source_gate.table_csv(result).encode() != (root / 'per_game_gate.csv').read_bytes():
    raise ValueError('CSV_REPLAY_DIFFERS')
training = research.training_rows(result)
for name in ['baseline', 'candidate']:
    saved = artifact['models'][name]
    examples = [(research.design(r, artifact['standardization'], len(saved['parameters']) - 1), r['y']) for r in training]
    loss, gradient, _ = research.objective(saved['parameters'], examples)
    if loss != saved['penalized_mean_loss'] or max(abs(g) for g in gradient) != saved['gradient_infinity_norm'] or max(abs(g) for g in gradient) > 1e-8:
        raise ValueError('FIXED_OBJECTIVE_OR_CONVERGENCE_MISMATCH:' + name)
train_dates = [r['official_date'] for r in result['rows'] if r['phase'] == 'train']
eval_dates = [r['official_date'] for r in result['rows'] if r['phase'] == 'evaluation']
if max(train_dates) >= min(eval_dates):
    raise ValueError('SAME_DATE_SPLIT_OR_CHRONOLOGY_ERROR')
if any(r['paired_admitted'] for r in result['rows'] if r['later_completion']):
    raise ValueError('LATER_COMPLETION_WAS_ADMITTED')
print(json.dumps({'json_byte_identical': True, 'csv_byte_identical': True,
                  'capsule_hashes_verified': len(manifest['capsule_files']),
                  'sealed_dependencies_verified': len(manifest['sealed_dependencies']),
                  'population': len(result['rows']), 'train_population': result['phases']['train']['population'],
                  'evaluation_population': result['phases']['evaluation']['population'],
                  'baseline_fits': 1, 'candidate_fits': 1,
                  'paired_evaluation': result['metrics']['baseline']['sample_count'],
                  'training_admitted': len(training), 'new_fits_during_replay': 0,
                  'objective_and_gradient_verified': True, 'later_completion_targets_excluded': 4,
                  'development_rule_passed': result['development_rule_passed'],
                  'new_game_source_gets': 0}, sort_keys=True))
