"""Two fixed logistic fits, then score-only replay on common admitted games."""
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import source_gate


def training_rows(prepared):
    return [{k: r[k] for k in ['key', 'y', 'prior_strength_difference', 'current_winrate_difference']}
            for r in prepared['rows'] if r['phase'] == 'train' and r['paired_admitted']]


def scaler(rows):
    if not rows or not 0 < sum(r['y'] for r in rows) < len(rows):
        raise ValueError('TRAINING_BOTH_CLASSES_REQUIRED')
    result = []
    for key in ['prior_strength_difference', 'current_winrate_difference']:
        mean = math.fsum(r[key] for r in rows) / len(rows)
        variance = math.fsum((r[key] - mean) ** 2 for r in rows) / len(rows)
        result.append({'key': key, 'mean': mean, 'scale': math.sqrt(variance) if variance else 1.0})
    return result


def design(row, scales, width):
    return [1.0] + [(row[s['key']] - s['mean']) / s['scale'] for s in scales[:width]]


def objective(parameters, examples):
    n = len(examples)
    size = len(parameters)
    loss = 0.0
    gradient = [0.0] * size
    hessian = [[0.0] * size for _ in range(size)]
    for x, y in examples:
        z = math.fsum(b * value for b, value in zip(parameters, x))
        p = source_gate.annual.sigmoid(z)
        w = p * (1 - p)
        loss += max(z, 0) + math.log1p(math.exp(-abs(z))) - y * z
        for i in range(size):
            gradient[i] += (p - y) * x[i]
            for j in range(size):
                hessian[i][j] += w * x[i] * x[j]
    for i in range(size):
        gradient[i] = gradient[i] / n + (parameters[i] if i else 0)
        for j in range(size):
            hessian[i][j] = hessian[i][j] / n + int(i == j and i > 0)
    return loss / n + 0.5 * math.fsum(b * b for b in parameters[1:]), gradient, hessian


def solve(matrix, rhs):
    """Small deterministic Gaussian elimination (only 2 or 3 parameters)."""
    n = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(rows[k][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        if abs(rows[i][i]) <= 1e-15:
            raise ValueError('SINGULAR_FIXED_HESSIAN')
        value = rows[i][i]
        rows[i] = [v / value for v in rows[i]]
        for j in range(n):
            if i == j:
                continue
            weight = rows[j][i]
            rows[j] = [v - weight * u for v, u in zip(rows[j], rows[i])]
    return [r[-1] for r in rows]


def fit(examples):
    parameters = [0.0] * len(examples[0][0])
    for iteration in range(101):
        loss, gradient, hessian = objective(parameters, examples)
        norm = max(abs(g) for g in gradient)
        if norm <= 1e-8:
            return {'parameters': parameters, 'iterations': iteration,
                    'gradient_infinity_norm': norm, 'penalized_mean_loss': loss, 'converged': True}
        if iteration == 100:
            break
        delta = solve(hessian, gradient)
        descent = math.fsum(g * d for g, d in zip(gradient, delta))
        if descent <= 0:
            raise ValueError('NON_DESCENT_FIXED_NEWTON')
        for backtrack in range(20):
            step = 2.0 ** (-backtrack)
            trial = [b - step * d for b, d in zip(parameters, delta)]
            if objective(trial, examples)[0] <= loss - 1e-4 * step * descent:
                parameters = trial
                break
        else:
            raise ValueError('FIXED_NEWTON_BACKTRACKING_FAILED')
    raise ValueError('FIXED_NEWTON_NOT_CONVERGED')


def predict(row, artifact, name):
    parameters = artifact['models'][name]['parameters']
    x = design(row, artifact['standardization'], len(parameters) - 1)
    contributions = [b * value for b, value in zip(parameters, x)]
    return source_gate.annual.sigmoid(math.fsum(contributions)), contributions


def metric(rows, key):
    n = len(rows)
    if not n:
        raise ValueError('EMPTY_EVALUATION')
    return {'sample_count': n, 'correct_count': sum((r[key] >= 0.5) == bool(r['y']) for r in rows),
            'accuracy': sum((r[key] >= 0.5) == bool(r['y']) for r in rows) / n,
            'brier': math.fsum((r[key] - r['y']) ** 2 for r in rows) / n,
            'logloss': -math.fsum(math.log(r[key]) if r['y'] else math.log1p(-r[key]) for r in rows) / n}


def score(root, artifact, legacy_root=None):
    prepared = source_gate.audit(root, legacy_root)
    training = training_rows(prepared)
    if artifact.get('status') != 'COMPLETE' or artifact['fit_attempts'] != {'baseline': 1, 'candidate': 1}:
        raise ValueError('TWO_SINGLE_FITS_NOT_COMPLETE')
    if artifact['protocol_sha256'] != prepared['protocol_sha256'] or artifact['training_rows_sha256'] != source_gate.annual.digest(training):
        raise ValueError('FROZEN_TRAINING_OR_PROTOCOL_MISMATCH')
    if scaler(training) != artifact['standardization']:
        raise ValueError('TRAINING_ONLY_SCALER_CHANGED')
    for r in prepared['rows']:
        if not r['paired_admitted']:
            continue
        for name in ['baseline', 'candidate']:
            p, contributions = predict(r, artifact, name)
            r[name + '_p_home'] = p
            r[name + '_intercept_contribution'] = contributions[0]
            r[name + '_prior_contribution'] = contributions[1]
            if name == 'candidate':
                r['candidate_current_contribution'] = contributions[2]
    selected = [r for r in prepared['rows'] if r['phase'] == 'evaluation' and r['paired_admitted']]
    metrics = {name: metric(selected, name + '_p_home') for name in ['baseline', 'candidate']}
    for r in selected:
        r['fixed_home_p_home'] = 0.5
    metrics['fixed_home_0_5'] = metric(selected, 'fixed_home_p_home')
    differences = {k: metrics['candidate'][k] - metrics['baseline'][k] for k in ['accuracy', 'brier', 'logloss']}
    passed = differences['accuracy'] > 0 and differences['brier'] <= 0 and differences['logloss'] <= 0
    contribution_summary = {}
    for name in ['baseline', 'candidate']:
        names = ['intercept', 'prior'] + (['current'] if name == 'candidate' else [])
        contribution_summary[name] = {
            term: {'mean': math.fsum(r[name + '_' + term + '_contribution'] for r in selected) / len(selected),
                   'mean_absolute': math.fsum(abs(r[name + '_' + term + '_contribution']) for r in selected) / len(selected),
                   'min': min(r[name + '_' + term + '_contribution'] for r in selected),
                   'max': max(r[name + '_' + term + '_contribution'] for r in selected)}
            for term in names}
    discordance = Counter()
    for r in selected:
        b = (r['baseline_p_home'] >= 0.5) == bool(r['y'])
        c = (r['candidate_p_home'] >= 0.5) == bool(r['y'])
        discordance['both_correct' if b and c else 'candidate_only_correct' if c else 'baseline_only_correct' if b else 'both_wrong'] += 1
    prepared.update(schema_version='lagged_season_winrate.result.v2', status='DEVELOPMENT_PASS' if passed else 'DEVELOPMENT_REJECT',
                    baseline_fit_count=1, candidate_fit_count=1, evaluation_performed=True,
                    metrics=metrics, candidate_minus_baseline=differences, development_rule_passed=passed,
                    coefficients_and_contributions={'models': artifact['models'], 'evaluation_logit_contribution_summary': contribution_summary},
                    new_fits_during_score_or_replay=0, paired_accuracy_discordance=dict(sorted(discordance.items())),
                    direction_counts={name: dict(Counter('HOME' if r[name + '_p_home'] >= 0.5 else 'AWAY' for r in selected)) for name in ['baseline', 'candidate']},
                    intercept_dominates_absolute_sum_on_evaluation={name: sum(abs(r[name + '_intercept_contribution']) > abs(r[name + '_prior_contribution']) + (abs(r['candidate_current_contribution']) if name == 'candidate' else 0) for r in selected) for name in ['baseline', 'candidate']},
                    new_same_game_market_validation_performed=False, accepted_beat_market_goal_established=False,
                    profitability_claim=False, training_admitted=len(training),
                    model_artifact_sha256=sha256((root / 'model_artifact.json').read_bytes()).hexdigest())
    return prepared


if __name__ == '__main__':
    import sys
    root = Path(__file__).resolve().parent
    artifact = json.loads((root / 'model_artifact.json').read_bytes())
    print(source_gate.result_json(score(root, artifact, sys.argv[1] if len(sys.argv) > 1 else None)).decode(), end='')
