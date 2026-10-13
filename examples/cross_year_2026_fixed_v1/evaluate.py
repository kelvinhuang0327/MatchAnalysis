"""One fixed-calendar cross-year replay; no fitting, network, ledger or orders."""
from collections import Counter, defaultdict
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
import csv
import io
import json
import math
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / 'examples/lagged_winrate_paper_v1'))
import paper


def payload(g):
    """Only actual identity/score/completion fields; unrelated metadata may differ."""
    return {'identity': paper.identity(g), 'status': g['status']['abstractGameState'],
            'home_score': g['teams']['home'].get('score'), 'away_score': g['teams']['away'].get('score'),
            'gameInfo': {k: g.get('gameInfo', {}).get(k) for k in ['firstPitch', 'gameDurationMinutes']}}


def canonical_game(versions):
    """Resolve explicit pre-start reschedule chain; never count one game twice."""
    if any(any(g.get(k) for k in ['resumeDate', 'resumeGameDate', 'resumedFrom', 'resumedFromDate']) for g in versions):
        return None, 'RESUMPTION_COMPLETION_UNSUPPORTED'
    unique = {paper.fingerprint(payload(g)): g for g in versions}
    if len(unique) == 1:
        return next(iter(unique.values())), None
    dates = {g['gameDate']: g for g in versions}
    outgoing = {g.get('rescheduledFrom') for g in versions if g.get('rescheduledFrom')}
    leaves = [g for g in versions if g.get('rescheduledFrom') and g['gameDate'] not in outgoing]
    if len(leaves) != 1:
        return None, 'PHYSICAL_GAME_IDENTITY_AMBIGUOUS'
    actual = leaves[0]; chain = {actual['gameDate']}; current = actual
    while current.get('rescheduledFrom'):
        earlier = current['rescheduledFrom']
        if earlier in chain or earlier not in dates:
            return None, 'RESCHEDULE_CHAIN_INCOMPLETE_OR_CYCLIC'
        chain.add(earlier); current = dates[earlier]
    if set(dates) != chain:
        return None, 'RESCHEDULE_CHAIN_HAS_UNEXPLAINED_VERSION'
    expected = payload(actual)
    for g in versions:
        p = payload(g)
        # A never-played original schedule representation can inherit Final
        # status/gameInfo while keeping NULL scores. NULL is not an outcome.
        if (p['identity'][:2] != expected['identity'][:2] or p['identity'][-2:] != expected['identity'][-2:]
                or any(p[k] is not None and p[k] != expected[k] for k in ['home_score', 'away_score'])
                or any(p['gameInfo'][k] is not None and p['gameInfo'][k] != expected['gameInfo'][k]
                       for k in ['firstPitch', 'gameDurationMinutes'])):
            return None, 'RESCHEDULE_SCORE_TEAM_OR_COMPLETION_CONFLICT'
    return actual, None


def completed(g, received):
    """Same firstPitch/duration bound as b6d31f4; never invented actual finish UTC."""
    if g['status']['abstractGameState'] != 'Final':
        return None, 'TARGET_OR_HISTORY_NOT_FINAL'
    hs = g['teams']['home'].get('score'); aws = g['teams']['away'].get('score')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in [hs, aws]) or hs == aws:
        return None, 'FINAL_SCORE_INVALID_OR_TIED'
    info = g.get('gameInfo', {})
    if not info.get('firstPitch') or isinstance(info.get('gameDurationMinutes'), bool):
        return None, 'COMPLETION_FIRST_PITCH_OR_DURATION_MISSING'
    first = paper.utc(info['firstPitch'])
    end = paper.completion_source.completion(g)
    start = paper.utc(g['gameDate'])
    if end is None:
        return None, 'COMPLETION_FIRST_PITCH_OR_DURATION_MISSING'
    if end > received or first < start - timedelta(hours=1) or end < start:
        return None, 'COMPLETION_TIME_INCONSISTENT_OR_NOT_YET_COMPLETE'
    if first.date() > date.fromisoformat(g['officialDate']) + timedelta(days=1):
        return None, 'UNMARKED_LATER_DATE_FIRST_PITCH'
    return end, None


def exact_paired(rows, left, right):
    counts = Counter()
    for r in rows:
        a = (r[left] >= .5) == bool(r['y']); b = (r[right] >= .5) == bool(r['y'])
        counts['both_correct' if a and b else 'left_only_correct' if a else 'right_only_correct' if b else 'both_wrong'] += 1
    n = counts['left_only_correct'] + counts['right_only_correct']
    numerator = min(2 * sum(math.comb(n, k) for k in range(min(counts['left_only_correct'], counts['right_only_correct']) + 1)), 2 ** n)
    return {'sample_count': len(rows), 'left': left, 'right': right, 'paired_counts': dict(sorted(counts.items())),
            'discordant_pairs': n, 'exact_mcnemar_two_sided_p': numerator / 2 ** n,
            'accuracy_difference': (counts['left_only_correct'] - counts['right_only_correct']) / len(rows) if rows else None,
            'limits': 'Nominal exact sign/binomial test assumes exchangeable independent discordances; shared teams/time and prior reuse limit confirmatory inference.'}


def market_audit():
    paths = ['report/moneyline_paper_decision_v1/p52a_f_identity_fix_v1/postgame_v1/quote_snapshots.jsonl',
             'report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl', 'data/authority/tsl/tsl_odds_history.jsonl']
    audits = []
    for name in paths:
        path = REPO / name
        raw = path.read_bytes(); rows = [json.loads(line) for line in raw.splitlines() if line]
        audits.append({'path': name, 'sha256': sha256(raw).hexdigest(), 'rows': len(rows),
                       'reason': 'POSTSEASON_F_NOT_R' if 'quote_snapshots' in name else 'P39A_HISTORICAL_PRICES_NOT_ADMITTED' if 'p39a' in name else 'RAW_PRICES_WITHOUT_VERIFIED_FULL_GAME_IDENTITY_AND_PREGAME_CAPTURE_AUTHORITY'})
    if any(json.loads(line).get('game_identity', {}).get('game_type') == 'R' for line in (REPO / paths[0]).read_text().splitlines() if line):
        raise ValueError('NEW_NORMALIZED_R_QUOTE_REQUIRES_AUTHORITY_REVIEW_NOT_SILENTLY_IGNORED')
    return {'paired_market_sample_count': 0, 'no_new_prices_acquired': True,
            'sources_reviewed': audits, 'opened102_prices': 'All102 excluded as targets; not substituted into this evaluation.',
            'market_metrics': None, 'market_superiority_established': False}


def evaluate(root=ROOT, legacy_root=None):
    policy = json.loads((root / 'protocol.json').read_bytes())
    for name, expected in policy['sealed_repository_files'].items():
        if sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise ValueError('SEALED_REPOSITORY_INPUT_CHANGED:' + name)
    source = json.loads((root / 'source_capture.json').read_bytes())
    if source['status'] != 'SUCCESS' or source['source_get_count'] != 1 or source['url'] != policy['source_url']:
        raise ValueError('SINGLE_BOUNDED_SOURCE_CAPTURE_UNAVAILABLE')
    raw = (root / source['path']).read_bytes()
    if sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('OFFICIAL_SOURCE_HASH_MISMATCH')
    received = paper.utc(source['received_at_utc'])
    if received > paper.utc(source['persisted_at_utc']):
        raise ValueError('SOURCE_TIME_ORDER_INVALID')
    body = json.loads(raw); raw_games = paper.games(body)
    if body.get('totalGames') != len(raw_games):
        raise ValueError('OFFICIAL_TOTAL_GAMES_INCONSISTENT')
    model = paper.saved_model()
    if sha256((paper.MODEL_ROOT / 'model_artifact.json').read_bytes()).hexdigest() != policy['model_sha256']:
        raise ValueError('FROZEN_MODEL_CHANGED')
    with (paper.MODEL_ROOT / 'per_game_gate.csv').open() as f:
        old_rows = list(csv.DictReader(f))
    training = [{k: (int(r[k]) if k == 'y' else float(r[k]) if k.endswith('difference') else r[k])
                 for k in ['key', 'y', 'prior_strength_difference', 'current_winrate_difference']}
                for r in old_rows if r['phase'] == 'train' and r['paired_admitted'] == 'True']
    if (len(training) != 1076 or paper.research.source_gate.annual.digest(training) != model['training_rows_sha256']
            or any(r['key'][:8] > '20250718' or not r['key'].startswith('2025') for r in training)
            or paper.research.scaler(training) != model['standardization']):
        raise ValueError('TRAINING_CHRONOLOGY_FINGERPRINT_OR_SCALER_CHANGED')
    annual_paths = {}
    for name, item in policy['sources'].items():
        path = Path(item['path']) if legacy_root is None else Path(legacy_root) / Path(item['path']).name
        if sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('PRIOR_ANNUAL_SOURCE_CHANGED')
        annual_paths[name] = path
    annual_policy = json.loads((REPO / 'examples/prior_season_strength_v1/protocol.json').read_bytes())
    strengths, _, annual_info = paper.research.source_gate.annual.season(2025, annual_paths['train_csv'], annual_paths['train_gamelog'], annual_policy)
    opened = [json.loads(line) for line in (REPO / 'examples/team_recent10_research_v1/population.jsonl').read_text().splitlines() if line]
    excluded102 = {r['game_pk'] for r in opened}
    if len(opened) != 102 or len(excluded102) != 102:
        raise ValueError('FIXED_UNIQUE_OPENED102_EXCLUSION_REQUIRED')
    grouped = defaultdict(list)
    for g in raw_games:
        if isinstance(g.get('gamePk'), bool) or not isinstance(g.get('gamePk'), int) or g['gamePk'] <= 0:
            raise ValueError('PROVIDER_PHYSICAL_GAME_PK_MISSING')
        grouped[g['gamePk']].append(g)
    history = []; rows = []; resolved_reschedules = []
    for pk, versions in sorted(grouped.items()):
        row = {'game_pk': pk, 'representation_count': len(versions), 'included': False,
               'old102_target': pk in excluded102, 'exclusions': [], 'official_date': None}
        if row['old102_target']:
            row['exclusions'].append('OPENED102_TARGET_EXCLUDED')
        try:
            g, reason = canonical_game(versions)
        except (KeyError, ValueError, TypeError):
            g, reason = None, 'FULL_IDENTITY_INCOMPLETE'
        if reason:
            row['exclusions'].append(reason); rows.append(row); continue
        try:
            full = paper.identity(g)
            start = paper.utc(g['gameDate']); official = date.fromisoformat(g['officialDate'])
            if (g['gameType'] != 'R' or not policy['range_start'] <= official.isoformat() <= policy['range_end']
                    or isinstance(g['gameNumber'], bool) or not isinstance(g['gameNumber'], int) or g['gameNumber'] <= 0
                    or any(g['teams'][s]['team']['id'] not in strengths
                           or g['teams'][s]['team']['name'] != strengths[g['teams'][s]['team']['id']]['name'] for s in ['home', 'away'])
                    or full[-2] == full[-1]):
                raise ValueError('FULL_GAME_IDENTITY_OR_PRIOR_TEAM_MISMATCH')
            row.update(official_date=official.isoformat(), scheduled_start_utc=g['gameDate'], game_number=g['gameNumber'],
                       home_id=full[-2], away_id=full[-1], identity_sha256=paper.fingerprint(full))
            end, reason = completed(g, received)
            if reason:
                row['exclusions'].append(reason); rows.append(row); continue
        except (KeyError, ValueError, TypeError) as exc:
            row['exclusions'].append('FULL_IDENTITY_OR_COMPLETION_INVALID:' + str(exc)); rows.append(row); continue
        if len(versions) > 1:
            resolved_reschedules.append({'game_pk': pk, 'representations': len(versions), 'actual_identity': list(full),
                                         'rule': 'Explicit rescheduledFrom chain; same teams and compatible non-NULL score/firstPitch/duration. Only actual leaf supplies outcome; count physical game once.'})
        first = paper.utc(g['gameInfo']['firstPitch'])
        if start > received or first > received:
            row['exclusions'].append('TARGET_NOT_STARTED_AT_SOURCE_CAPTURE'); rows.append(row); continue
        hs = g['teams']['home']['score']; aws = g['teams']['away']['score']
        history.append({'key': str(pk), 'official_date': g['officialDate'],
                        'completion_date': max(official, end.date()).isoformat(), 'home_id': full[-2], 'away_id': full[-1],
                        'y': int(hs > aws), 'completion_upper_bound_utc': paper.text_time(end)})
        row.update(target_first_pitch_utc=g['gameInfo']['firstPitch'], completion_upper_bound_utc=paper.text_time(end),
                   home_score=hs, away_score=aws, y=int(hs > aws))
        rows.append(row)
    first_current = min(g['officialDate'] for g in raw_games if g['gameType'] == 'R')
    year_gap = paper.research.source_gate.annual.annual_chronology(annual_info['last_completion_date'], first_current)
    by_pk = {int(g['key']): g for g in history}
    minimum_margin = None
    for r in rows:
        if r['exclusions']:
            continue
        delta, histories, reasons = paper.research.source_gate.lagged_history(r, history, policy['lag_calendar_days'], policy['minimum_history_each_team'])
        r.update(home_history_games=histories[r['home_id']]['games'], away_history_games=histories[r['away_id']]['games'],
                 home_history_wins=histories[r['home_id']]['wins'], away_history_wins=histories[r['away_id']]['wins'])
        if reasons:
            r['exclusions'].extend(reasons); continue
        ids = sorted({int(k) for h in histories.values() for k in h['keys']})
        target_start = min(paper.utc(r['scheduled_start_utc']), paper.utc(r['target_first_pitch_utc']))
        margin = min((target_start - paper.utc(by_pk[pk]['completion_upper_bound_utc'])).total_seconds() for pk in ids)
        if margin <= 0 or r['game_pk'] in ids:
            raise ValueError('SELF_OR_NOT_COMPLETED_HISTORY_ENTERED_TARGET')
        minimum_margin = margin if minimum_margin is None else min(margin, minimum_margin)
        r.update(prior_strength_difference=strengths[r['home_id']]['strength'] - strengths[r['away_id']]['strength'],
                 current_winrate_difference=delta, history_keys_sha256=paper.fingerprint(ids),
                 latest_completion_margin_seconds=margin, fixed_home_p_home=.5, included=True)
        for name in ['baseline', 'candidate']:
            p, contributions = paper.research.predict(r, model, name)
            r[name + '_p_home'] = p
            r[name + '_logit_contributions'] = contributions
    selected = [r for r in rows if r['included']]
    metrics = {name: paper.research.metric(selected, name + '_p_home') if selected else None for name in ['candidate', 'baseline', 'fixed_home']}
    cache_games = paper.games(json.loads((REPO / 'data/fixtures/p23f2_official_2026_history/raw/schedule_2026-03-25_2026-06-30.json').read_bytes()))
    seen_cache = {g['gamePk'] for g in cache_games}
    known_final_cache = {g['gamePk'] for g in cache_games if g['status']['abstractGameState'] == 'Final'
                         and all(isinstance(g['teams'][s].get('score'), int) for s in ['home', 'away'])
                         and g['teams']['home']['score'] != g['teams']['away']['score']}
    training_protocol = json.loads((paper.MODEL_ROOT / 'protocol.json').read_bytes())
    revision = training_protocol['method_revision']['authorized_at_utc']
    comparisons = {name: exact_paired(selected, 'candidate_p_home', other + '_p_home') for name, other in [('candidate_vs_baseline', 'baseline'), ('candidate_vs_fixed_home', 'fixed_home')]} if selected else {}
    return {'schema_version': 'cross_year_2026_fixed.result.v1', 'status': 'POSTHOC_FIXED_MODEL_EVALUATED' if selected else 'COMPLETION_OR_IDENTITY_GAP_NO_EVALUATION',
            'interpretation': policy['interpretation'], 'protocol_sha256': sha256((root / 'protocol.json').read_bytes()).hexdigest(),
            'model_sha256': policy['model_sha256'], 'new_fits': 0, 'new_source_gets': 1,
            'source_byte_count': source['byte_count'], 'source_sha256': source['sha256'],
            'source_received_at_utc': source['received_at_utc'], 'raw_representation_count': len(raw_games),
            'physical_game_population': len(rows), 'previous_opened102_source_matches': len(excluded102.intersection(grouped)),
            'previous_opened102_missing_from_source': sorted(excluded102 - set(grouped)),
            'physical_population_after_excluding_old102': sum(not r['old102_target'] for r in rows),
            'paired_evaluated_count': len(selected), 'excluded_total': len(rows) - len(selected),
            'coverage_of_full_physical_population': len(selected) / len(rows),
            'coverage_after_opened102_exclusion': len(selected) / sum(not r['old102_target'] for r in rows),
            'primary_exclusion_counts': dict(sorted(Counter(r['exclusions'][0] for r in rows if r['exclusions']).items())),
            'nonexclusive_exclusion_counts': dict(sorted(Counter(e for r in rows for e in r['exclusions']).items())),
            'admitted_history_physical_games': len(history), 'resolved_prestart_reschedules': resolved_reschedules,
            'minimum_completion_margin_seconds': minimum_margin, 'metrics': metrics, 'paired_uncertainty': comparisons,
            'candidate_minus_baseline': {k: metrics['candidate'][k] - metrics['baseline'][k] for k in ['accuracy', 'brier', 'logloss']} if selected else None,
            'candidate_minus_fixed_home': {k: metrics['candidate'][k] - metrics['fixed_home'][k] for k in ['accuracy', 'brier', 'logloss']} if selected else None,
            'direction_counts': {name: dict(Counter('HOME' if r[name + '_p_home'] >= .5 else 'AWAY' for r in selected)) for name in ['candidate', 'baseline']},
            'mean_absolute_logit_contributions': {name: [math.fsum(abs(r[name + '_logit_contributions'][i]) for r in selected) / len(selected)
                                                        for i in range(3 if name == 'candidate' else 2)] for name in ['candidate', 'baseline']} if selected else {},
            'intercept_dominates_absolute_signal_sum_count': {name: sum(abs(r[name + '_logit_contributions'][0]) > sum(abs(x) for x in r[name + '_logit_contributions'][1:]) for r in selected) for name in ['candidate', 'baseline']},
            'training_separation': {'training_year': 2025, 'admitted_training_rows': len(training), 'latest_training_key': max(r['key'] for r in training),
                                    'training_rows_sha256': model['training_rows_sha256'], 'training_scaler_matches_saved': True,
                                    'evaluation_year': 2026, 'training_contains_evaluation_year': False, '2025_annual_end_to_2026_start_days': year_gap,
                                    'model_training_method_revision_at_utc': revision,
                                    'all_evaluation_events_precede_model_method_revision': all(paper.utc(r['target_first_pitch_utc']) < paper.utc(revision) for r in selected),
                                    'model_not_available_for_these_historical_games': True},
            'previous_use': {'selected_pks_already_in_old_raw_cache': sum(r['game_pk'] in seen_cache for r in selected),
                             'selected_pks_with_final_label_in_old_raw_cache': sum(r['game_pk'] in known_final_cache for r in selected),
                             'selected_pks_not_in_that_one_cache': sum(r['game_pk'] not in seen_cache for r in selected),
                             'not_in_cache_does_not_mean_unseen': True, 'other_prior_use': policy['previous_use_summary']},
            'market': market_audit(), 'improvement_or_profitability_established': False, 'rows': rows}


def table_csv(result):
    fields = ['game_pk', 'official_date', 'game_number', 'scheduled_start_utc', 'home_id', 'away_id', 'included', 'old102_target',
              'representation_count', 'home_score', 'away_score', 'y', 'completion_upper_bound_utc',
              'home_history_games', 'away_history_games', 'home_history_wins', 'away_history_wins',
              'prior_strength_difference', 'current_winrate_difference', 'history_keys_sha256', 'latest_completion_margin_seconds',
              'candidate_p_home', 'baseline_p_home', 'fixed_home_p_home', 'candidate_logit_contributions', 'baseline_logit_contributions', 'exclusions']
    output = io.StringIO(newline=''); writer = csv.DictWriter(output, fields, lineterminator='\n'); writer.writeheader()
    for r in result['rows']:
        selected = {k: r.get(k) for k in fields}; selected['exclusions'] = '|'.join(r['exclusions'])
        writer.writerow(selected)
    return output.getvalue().encode()


def result_json(result):
    summary = {k: v for k, v in result.items() if k != 'rows'}
    summary['per_game_csv_sha256'] = sha256(table_csv(result)).hexdigest()
    return paper.encoded(summary)


if __name__ == '__main__':
    result = evaluate(legacy_root=sys.argv[1] if len(sys.argv) > 1 else None)
    print(result_json(result).decode(), end='')
