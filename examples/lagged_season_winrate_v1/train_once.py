"""Reserve exclusive fit artifact before either of the two single fits."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import source_gate
import research


def run(root, legacy_root=None):
    path = root / 'model_artifact.json'
    if path.exists():
        raise ValueError('MODEL_EXISTS_REFIT_REFUSED')
    prepared = source_gate.audit(root, legacy_root)
    training = research.training_rows(prepared)
    if not prepared['phases']['evaluation']['paired_admitted']:
        raise ValueError('NO_PAIRED_EVALUATION_FIT_REFUSED')
    scales = research.scaler(training)
    artifact = {'schema_version': 'lagged_season_winrate.model.v1', 'status': 'RESERVED',
                'protocol_sha256': prepared['protocol_sha256'],
                'training_rows_sha256': source_gate.annual.digest(training),
                'training_count': len(training), 'standardization': scales,
                'fit_attempts': {'baseline': 0, 'candidate': 0}, 'models': {},
                'fit_code_sha256': {name: sha256((root / name).read_bytes()).hexdigest()
                                    for name in ['research.py', 'train_once.py', 'source_gate.py']}}
    with path.open('xb') as f:
        f.write(source_gate.annual.encoded(artifact))
    for name, width in [('baseline', 1), ('candidate', 2)]:
        artifact['fit_attempts'][name] = 1
        path.write_bytes(source_gate.annual.encoded(artifact))
        examples = [(research.design(r, scales, width), r['y']) for r in training]
        artifact['models'][name] = research.fit(examples)
        path.write_bytes(source_gate.annual.encoded(artifact))
    artifact['status'] = 'COMPLETE'
    path.write_bytes(source_gate.annual.encoded(artifact))
    # Evaluation happens only after both fits are sealed; no tuning/retries.
    result = research.score(root, artifact, legacy_root)
    (root / 'source_gate_result.json').write_bytes(source_gate.result_json(result))
    (root / 'per_game_gate.csv').write_text(source_gate.table_csv(result))
    print(json.dumps({'training_admitted': len(training), 'phases': result['phases'],
                      'metrics': result['metrics'], 'candidate_minus_baseline': result['candidate_minus_baseline'],
                      'development_rule_passed': result['development_rule_passed'],
                      'direction_counts': result['direction_counts'], 'models': artifact['models']}, sort_keys=True))


if __name__ == '__main__':
    run(Path(__file__).resolve().parent, sys.argv[1] if len(sys.argv) > 1 else None)
