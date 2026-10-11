"""Bounded offline admission audit. Never fits or reads opened 2026 outcomes."""
from collections import Counter
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
import csv
import importlib.util
import io
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'sealed_annual_strength', ROOT.parent / 'prior_season_strength_v1' / 'research.py')
annual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual)


def split_date(games, fraction=0.60):
    """Use dates/counts only; whole boundary date goes to training."""
    counts = Counter(g['official_date'] for g in games)
    cumulative = 0
    for day in sorted(counts):
        cumulative += counts[day]
        if cumulative >= math.ceil(len(games) * fraction):
            return day, cumulative, len(games) - cumulative
    raise ValueError('EMPTY_POPULATION')


def event_reasons(game, postponed):
    """postponed must be explicit verified source evidence, never absent->False."""
    reasons = []
    if not game.get('complete_game_source') or not game.get('independent_final'):
        reasons.append('COMPLETION_SOURCE_UNKNOWN')
    if game.get('later_completion'):
        reasons.append('LATER_DATE_COMPLETION')
    if game.get('forfeit') or game.get('protest'):
        reasons.append('FORFEIT_OR_PROTEST')
    if postponed is None:
        reasons.append('POSTPONEMENT_STATUS_UNKNOWN')
    elif postponed is True:
        reasons.append('POSTPONED_OR_RESCHEDULED')
    elif postponed is not False:
        raise ValueError('POSTPONEMENT_PROOF_NOT_BOOLEAN')
    return reasons


def lagged_history(target, admitted_history, lag_days=7, minimum=20):
    """Same-day rows never enter; only already admitted events can contribute."""
    boundary = date.fromisoformat(target['official_date']) - timedelta(days=lag_days)
    totals = {target[s]: {'games': 0, 'wins': 0, 'keys': []} for s in ['home_id', 'away_id']}
    for g in admitted_history:
        if date.fromisoformat(g['completion_date']) > boundary:
            continue
        for side in ['home_id', 'away_id']:
            tid = g[side]
            if tid not in totals:
                continue
            totals[tid]['games'] += 1
            totals[tid]['wins'] += int(bool(g['y']) == (side == 'home_id'))
            totals[tid]['keys'].append(g['key'])
    errors = ['HOME_HISTORY_BELOW_MINIMUM'] if totals[target['home_id']]['games'] < minimum else []
    if totals[target['away_id']]['games'] < minimum:
        errors.append('AWAY_HISTORY_BELOW_MINIMUM')
    x = None if errors else (
        totals[target['home_id']]['wins'] / totals[target['home_id']]['games']
        - totals[target['away_id']]['wins'] / totals[target['away_id']]['games'])
    return x, totals, errors


def audit(root=ROOT, legacy_root=None):
    policy = json.loads((root / 'protocol.json').read_bytes())
    repo = root.parent.parent
    prior_source = policy['sealed_prior_protocol']
    prior_path = repo / prior_source['relative_path']
    if sha256(prior_path.read_bytes()).hexdigest() != prior_source['sha256']:
        raise ValueError('SEALED_PRIOR_PROTOCOL_CHANGED')
    prior_policy = json.loads(prior_path.read_bytes())
    paths = {}
    for key, item in policy['sources'].items():
        path = Path(item['path'])
        if legacy_root is not None:
            path = Path(legacy_root) / path.name
        if sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('SOURCE_HASH_MISMATCH:' + key)
        paths[key] = path
    prior, _, prior_info = annual.season(
        2024, paths['earlier_csv'], paths['earlier_gamelog'], prior_policy)
    _, games, current_info = annual.season(
        2025, paths['train_csv'], paths['train_gamelog'], prior_policy)
    gap = annual.annual_chronology(prior_info['last_completion_date'], current_info['first_game_date'])
    boundary, train_count, eval_count = split_date(games)
    expected = policy['split']
    if (boundary, train_count, eval_count, len(games)) != (
            expected['train_last_date'], expected['raw_train_count'],
            expected['raw_evaluation_count'], expected['total_raw_count']):
        raise ValueError('FROZEN_SPLIT_MISMATCH')
    with paths['train_gamelog'].open() as f:
        raw = {'|'.join(r[i] for i in [0, 1, 3, 6]): r for r in csv.reader(f)}
    with paths['train_csv'].open() as f:
        csv_rows = list(csv.DictReader(f))
    # The independent CSV has no raw gameNumber; use a multiset of full score identity.
    finals = Counter((r['date'], r['away_team'], r['home_team'],
                      int(r['away_score']), int(r['home_score']))
                     for r in csv_rows if r['status'] == 'Final' and r['game_type'] == 'REG')
    names = prior_policy['feature_definition']['retrosheet_code_names']
    csv_fields = list(csv_rows[0])
    unavailable = ['original_scheduled_date', 'postponed', 'rescheduled_from']
    if any(k in csv_fields for k in unavailable):
        raise ValueError('SOURCE_CONTRACT_CHANGED_NEEDS_EXPLICIT_PROOF_IMPLEMENTATION')
    rows = []
    for g in games:
        r = raw[g['key']]
        final_identity = (g['official_date'], names[r[3]], names[r[6]], g['away_score'], g['home_score'])
        g.update(complete_game_source=r[160] == 'Y', independent_final=finals[final_identity] > 0,
                 forfeit=r[14], protest=r[15])
        reasons = event_reasons(g, postponed=None)
        rows.append({**g, 'phase': 'train' if g['official_date'] <= boundary else 'evaluation',
                     'event_admitted': not reasons, 'event_exclusions': reasons,
                     'postponed_status': None, 'field14_raw': r[13], 'field161_raw': r[160],
                     'prior_strength_difference': prior[g['home_id']]['strength'] - prior[g['away_id']]['strength']})
    admitted_history = [r for r in rows if r['event_admitted']]
    for r in rows:
        x, histories, errors = lagged_history(
            r, admitted_history, policy['features']['lag_calendar_days'],
            policy['features']['minimum_eligible_history_per_team'])
        r.update(current_winrate_difference=x,
                 home_history_games=histories[r['home_id']]['games'],
                 away_history_games=histories[r['away_id']]['games'],
                 history_exclusions=errors, paired_admitted=r['event_admitted'] and not errors)
    event_counts = Counter(reason for r in rows for reason in r['event_exclusions'])
    phases = {}
    for phase in ['train', 'evaluation']:
        subset = [r for r in rows if r['phase'] == phase]
        phases[phase] = {'population': len(subset), 'paired_admitted': sum(r['paired_admitted'] for r in subset),
                         'paired_excluded': sum(not r['paired_admitted'] for r in subset),
                         'coverage': sum(r['paired_admitted'] for r in subset) / len(subset),
                         'primary_event_exclusions': dict(Counter(r['event_exclusions'][0] for r in subset if r['event_exclusions']))}
    if admitted_history:
        raise ValueError('UNEXPECTED_ADMISSION_WITHOUT_POSTPONEMENT_SOURCE')
    result = {'schema_version': 'lagged_season_winrate.source_gate.v1',
              'protocol_sha256': sha256((root / 'protocol.json').read_bytes()).hexdigest(),
              'status': 'BLOCKED_POSTPONEMENT_PROOF_MISSING', 'baseline_fit_count': 0, 'candidate_fit_count': 0,
              'new_game_source_gets': 0, 'hyperparameter_searches': 0, 'evaluation_performed': False,
              'metrics': None, 'coefficients_and_contributions': None,
              'split': {'train_last_date': boundary, 'train_population': train_count, 'evaluation_population': eval_count},
              'phases': phases, 'history_event_population': len(rows), 'history_events_admitted': len(admitted_history),
              'event_exclusion_reason_counts_nonexclusive': dict(sorted(event_counts.items())),
              'non_later_completion_final_records': sum(r['complete_game_source'] and r['independent_final'] and not r['later_completion'] for r in rows),
              'known_later_completion_games': [{k: r[k] for k in ['key', 'official_date', 'completion_date', 'field14_raw']} for r in rows if r['later_completion']],
              'independent_csv_fields': csv_fields, 'missing_postponement_source_fields': unavailable,
              'source_hashes': {k: item['sha256'] for k, item in policy['sources'].items()},
              'annual_2024_to_2025_gap_days': gap, 'annual_csv_raw_multiset_and_ordinals_verified': True,
              'interpretation': policy['interpretation'],
              'blocker': 'Field14 proves later-date completion when populated. Blank field14 does not prove never-postponed. Existing as-played CSV has no original schedule or reschedule status. All 2430 statuses remain unknown under the explicit exclusion rule.',
              'next_requirement': 'An existing authorized full-identity 2025 scheduling/rescheduling evidence source, or an explicit scope decision allowing completed-as-played games irrespective of postponement before first pitch. Neither is assumed.',
              'rows': rows}
    return result


def table_csv(result):
    output = io.StringIO(newline='')
    fields = ['key', 'official_date', 'completion_date', 'phase', 'home_id', 'away_id',
              'field14_raw', 'field161_raw', 'postponed_status', 'event_admitted', 'paired_admitted',
              'home_history_games', 'away_history_games', 'prior_strength_difference',
              'current_winrate_difference', 'event_exclusions', 'history_exclusions']
    writer = csv.DictWriter(output, fields, lineterminator='\n')
    writer.writeheader()
    for r in result['rows']:
        selected = {k: r[k] for k in fields}
        for k in ['event_exclusions', 'history_exclusions']:
            selected[k] = '|'.join(selected[k])
        writer.writerow(selected)
    return output.getvalue()


def result_json(result):
    """Summary binds detailed CSV rather than duplicating thousands of rows."""
    summary = {k: v for k, v in result.items() if k != 'rows'}
    summary['per_game_csv_sha256'] = sha256(table_csv(result).encode()).hexdigest()
    summary['per_game_csv_rows'] = len(result['rows'])
    return annual.encoded(summary)


if __name__ == '__main__':
    result = audit(legacy_root=sys.argv[1] if len(sys.argv) > 1 else None)
    print(result_json(result).decode(), end='')
