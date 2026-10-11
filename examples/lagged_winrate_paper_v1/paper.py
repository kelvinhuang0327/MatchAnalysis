"""Saved-parameter paper adapter; existing P50C/P32A engines, isolated outputs."""
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / 'examples/lagged_season_winrate_v1'
MODEL_HASH = 'ce77257860afe420b022c375d50f2c1c717787cf1fb1cafc3a45365de7c7adf0'
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(MODEL_ROOT))
import research
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    create_p50c_prediction_run, settle_p50c_prediction_run,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION as FORWARD,
    CLASSIFICATION_HISTORICAL_REHEARSAL as REHEARSAL,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import NormalizedPregameInput, NormalizedResultRecord
from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import P40APredictionRow
from match_analysis.application.use_cases.moneyline_paper_decision import load_frozen_prediction_run
from match_analysis.application.use_cases.prospective_pairing_capture import SourceCaptures, assess_pair
from match_analysis.application.use_cases.p32a_frozen_moneyline_quotes import quotes_from_acquisition
from match_analysis.application.use_cases.acquire_tsl_moneyline_snapshot import acquire_tsl_moneyline_snapshot
from match_analysis.application.use_cases.p50c_market_comparison import MarketComparisonPolicy, evaluate_p50c_market_comparison

spec = importlib.util.spec_from_file_location('saved_completion_bound', REPO / 'examples/team_recent10_research_v1/evaluate.py')
completion_source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion_source)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def fingerprint(value):
    return sha256(encoded(value)).hexdigest()


def now():
    return datetime.now(UTC)


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('TIMEZONE_AWARE_TIMESTAMP_REQUIRED')
    return parsed.astimezone(UTC)


def text_time(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('INPUT_PATH_ESCAPE')
    return path


def isolated(root):
    root = root.resolve()
    protected = [REPO, Path('/Users/kelvin/VibeCoding-WorkSpace/MatchAnalysis'),
                 Path('/Users/kelvin/Documents/Codex/2026-10-04/task/review/MatchAnalysis-mainline-20261005')]
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('ISOLATED_PAPER_OUTPUT_REQUIRED')
    return root


def saved_model():
    raw = (MODEL_ROOT / 'model_artifact.json').read_bytes()
    if sha256(raw).hexdigest() != MODEL_HASH:
        raise ValueError('SEALED_133D29F_MODEL_CHANGED')
    return json.loads(raw)


def captured(root, record, cutoff):
    if record['status'] != 'SUCCESS' or utc(record['received_at_utc']) > utc(record['persisted_at_utc']) or utc(record['persisted_at_utc']) > cutoff:
        raise ValueError('SOURCE_NOT_AVAILABLE_AT_CUTOFF')
    if not record['url'].startswith('https://statsapi.mlb.com/api/v1/schedule?'):
        raise ValueError('EXISTING_OFFICIAL_SCHEDULE_SOURCE_REQUIRED')
    raw = inside(root, record['path']).read_bytes()
    # Reading an already captured whole-season schedule is bounded separately
    # from live GET limits; full official fields can exceed four megabytes.
    if len(raw) > 16 * 1024 * 1024 or sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('SOURCE_HASH_OR_SIZE_MISMATCH')
    return json.loads(raw), raw


def games(payload):
    return [g for day in payload['dates'] for g in day['games']]


def identity(g):
    return (g['gamePk'], g['gameType'], g['officialDate'], g['gameNumber'], g['gameDate'],
            g['teams']['home']['team']['id'], g['teams']['away']['team']['id'])


def prepare(input_root, cutoff):
    input_root = input_root.resolve()
    manifest = json.loads((input_root / 'input_manifest.json').read_bytes())
    if manifest['schema_version'] != 'lagged_winrate.paper_input.v1':
        raise ValueError('INPUT_SCHEMA_UNSUPPORTED')
    records = manifest['source_captures']
    if len(records) != 2 or {r['name'] for r in records} != {'target_schedule', 'season_history'}:
        raise ValueError('TWO_EXACT_MODEL_CAPTURE_SOURCES_REQUIRED')
    by_name = {r['name']: r for r in records}
    target_payload, target_bytes = captured(input_root, by_name['target_schedule'], cutoff)
    target_versions = {fingerprint(g): g for g in games(target_payload) if g['gamePk'] == manifest['target_game_pk']}
    if len(target_versions) != 1:
        raise ValueError('TARGET_FULL_IDENTITY_NOT_UNIQUE')
    target = next(iter(target_versions.values()))
    if target['gameType'] != 'R' or target['status']['abstractGameState'] != 'Preview' or utc(target['gameDate']) <= cutoff:
        raise ValueError('TARGET_NOT_FUTURE_REGULAR_PREVIEW')
    target_date = date.fromisoformat(target['officialDate'])
    if not cutoff.date() <= target_date <= cutoff.date() + timedelta(days=7):
        raise ValueError('TARGET_OUTSIDE_BOUNDED_SEVEN_DAY_WINDOW')
    history_payload, history_bytes = captured(input_root, by_name['season_history'], cutoff)
    query = parse_qs(urlparse(by_name['season_history']['url']).query)
    boundary = target_date - timedelta(days=7)
    if (query.get('gameType') != ['R'] or query.get('sportId') != ['1'] or 'fields' in query
            or date.fromisoformat(query['startDate'][0]) > date(target_date.year, 1, 1)
            or date.fromisoformat(query['endDate'][0]) < boundary):
        raise ValueError('FULL_CURRENT_SEASON_THROUGH_LAG_BOUNDARY_REQUIRED')
    history_games = games(history_payload)
    if history_payload.get('totalGames') != len(history_games):
        raise ValueError('HISTORY_PROVIDER_TOTAL_OR_COMPLETENESS_MISMATCH')
    versions = defaultdict(dict)
    for g in history_games:
        versions[g['gamePk']][fingerprint(g)] = g
    admitted = []
    exclusions = []
    for pk, alternatives in sorted(versions.items()):
        g = next(iter(alternatives.values()))
        reason = None
        if len(alternatives) != 1 or any(g.get(k) for k in ['resumeDate', 'resumeGameDate', 'resumedFrom', 'resumedFromDate']):
            reason = 'HISTORY_AMBIGUOUS_OR_RESUMED'
        elif g['gameType'] != 'R' or date.fromisoformat(g['officialDate']).year != target_date.year:
            reason = 'HISTORY_WRONG_SEASON_OR_GAME_TYPE'
        elif g['status']['abstractGameState'] != 'Final':
            reason = 'HISTORY_NOT_FINAL'
        else:
            info = g.get('gameInfo', {})
            if info.get('firstPitch'):
                utc(info['firstPitch'])
            end = None if isinstance(info.get('gameDurationMinutes'), bool) else completion_source.completion(g)
            if end is None:
                reason = 'HISTORY_COMPLETION_EVIDENCE_MISSING'
            elif end > cutoff or end > utc(by_name['season_history']['received_at_utc']) or end.date() > boundary or date.fromisoformat(g['officialDate']) > boundary:
                reason = 'HISTORY_NOT_FINISHED_BY_SEVEN_DAY_BOUNDARY'
            elif utc(g['gameInfo']['firstPitch']) < utc(g['gameDate']) - timedelta(hours=1) or end < utc(g['gameDate']):
                reason = 'HISTORY_COMPLETION_TIME_INCONSISTENT'
        if reason:
            exclusions.append({'game_pk': pk, 'reason': reason})
            continue
        hs = g['teams']['home'].get('score'); aws = g['teams']['away'].get('score')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in [hs, aws]) or hs == aws:
            exclusions.append({'game_pk': pk, 'reason': 'HISTORY_FINAL_SCORE_INVALID'})
            continue
        admitted.append({'key': str(pk), 'official_date': g['officialDate'], 'completion_date': end.date().isoformat(),
                         'home_id': g['teams']['home']['team']['id'], 'away_id': g['teams']['away']['team']['id'],
                         'y': int(hs > aws), 'completion_upper_bound_utc': text_time(end),
                         'completion_semantics': 'OFFICIAL_FIRST_PITCH_PLUS_DURATION_PLUS_60_SECONDS_DERIVED'})
    annual = manifest['prior_annual']
    if annual['year'] != target_date.year - 1:
        raise ValueError('PREVIOUS_COMPLETE_SEASON_REQUIRED')
    if not manifest.get('synthetic'):
        sealed = json.loads((REPO / 'examples/prior_season_strength_v1/protocol.json').read_bytes())['sources']
        keys = {2024: ('earlier_csv', 'earlier_gamelog'), 2025: ('train_csv', 'train_gamelog')}
        if annual['year'] not in keys:
            raise ValueError('FULL_PRIOR_ANNUAL_SOURCE_VERSION_NOT_YET_ADMITTED:' + str(annual['year']))
        csv_key, raw_key = keys[annual['year']]
        if annual['csv']['sha256'] != sealed[csv_key]['sha256'] or annual['gamelog']['sha256'] != sealed[raw_key]['sha256']:
            raise ValueError('FULL_ANNUAL_SOURCE_MUST_MATCH_ALREADY_VERIFIED_VERSION')
    annual_bytes = {}
    for name in ['csv', 'gamelog']:
        item = annual[name]; raw = inside(input_root, item['path']).read_bytes()
        if sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('ANNUAL_SOURCE_HASH_MISMATCH')
        annual_bytes[name] = raw
    policy = json.loads((REPO / 'examples/prior_season_strength_v1/protocol.json').read_bytes())
    strengths, _, info = research.source_gate.annual.season(annual['year'], inside(input_root, annual['csv']['path']),
                                                          inside(input_root, annual['gamelog']['path']), policy)
    first_current = min(g['officialDate'] for g in history_games if g['gameType'] == 'R') if history_games else target['officialDate']
    research.source_gate.annual.annual_chronology(info['last_completion_date'], first_current)
    feature_row = {'official_date': target['officialDate'], 'home_id': target['teams']['home']['team']['id'],
                   'away_id': target['teams']['away']['team']['id']}
    delta, histories, errors = research.source_gate.lagged_history(feature_row, admitted)
    if errors:
        raise ValueError('INSUFFICIENT_LAGGED_HISTORY:' + '|'.join(errors))
    feature_row['prior_strength_difference'] = strengths[feature_row['home_id']]['strength'] - strengths[feature_row['away_id']]['strength']
    feature_row['current_winrate_difference'] = delta
    model = saved_model()
    p, contributions = research.predict(feature_row, model, 'candidate')
    baseline_p, _ = research.predict(feature_row, model, 'baseline')
    evidence = {'features': feature_row, 'histories': histories, 'accepted_history': admitted,
                'history_exclusions': exclusions, 'history_reason_counts': dict(Counter(e['reason'] for e in exclusions)),
                'annual_year': annual['year'], 'annual_source_info': info,
                'source_hashes': {r['name']: r['sha256'] for r in records},
                'annual_hashes': {name: annual[name]['sha256'] for name in ['csv', 'gamelog']},
                'model_sha256': MODEL_HASH, 'baseline_probability_diagnostic': baseline_p,
                'candidate_logit_contributions': contributions,
                'same_calendar_rule': 'Both official game date and conservative derived completion date <= target officialDate minus7 days; uncertain completion excluded.',
                'no_retraining': True}
    evidence = json.loads(encoded(evidence))
    return manifest, target, str(p), evidence, {'target_schedule.json': target_bytes, 'season_history.json': history_bytes,
                                               'prior.csv': annual_bytes['csv'], 'prior.gamelog': annual_bytes['gamelog']}


def freeze(input_root, output_root, *, clock=now, capture_market=False):
    out = isolated(output_root)
    if out.exists():
        raise ValueError('EXCLUSIVE_NEW_PAPER_OUTPUT_REQUIRED')
    cutoff = clock().astimezone(UTC)
    manifest, target, probability, evidence, snapshots = prepare(input_root, cutoff)
    if manifest.get('synthetic') and capture_market:
        raise ValueError('SYNTHETIC_INPUT_CANNOT_CAPTURE_LIVE_MARKET')
    classification = REHEARSAL if manifest.get('synthetic') else FORWARD
    out.mkdir(parents=True)
    for name, raw in snapshots.items():
        (out / name).write_bytes(raw)
    (out / 'model_artifact.json').write_bytes((MODEL_ROOT / 'model_artifact.json').read_bytes())
    (out / 'original_input_manifest.json').write_bytes((input_root / 'input_manifest.json').read_bytes())
    normalized_manifest = json.loads(encoded(manifest))
    for record in normalized_manifest['source_captures']:
        record['path'] = record['name'] + '.json'
    normalized_manifest['prior_annual']['csv']['path'] = 'prior.csv'
    normalized_manifest['prior_annual']['gamelog']['path'] = 'prior.gamelog'
    (out / 'input_manifest.json').write_bytes(encoded(normalized_manifest))
    # Local snapshot copies are real availability witnesses, never provider publication times.
    local_persisted = text_time(clock())
    inference = text_time(clock())
    if utc(inference) >= utc(target['gameDate']):
        raise ValueError('SNAPSHOT_COPY_CROSSED_TARGET_START')
    (out / 'input_evidence.json').write_bytes(encoded(evidence))
    probability = str(research.predict(evidence['features'], saved_model(), 'candidate')[0])
    input_hash = fingerprint(evidence)
    p = Decimal(probability)
    row = P40APredictionRow(p37_fold_id='saved-133d29f', p37_window='paper-candidate', p37_prediction_row_id=input_hash,
        provider_namespace='MLB_STATS_API', provider_game_id=str(target['gamePk']), game_pk=target['gamePk'],
        game_number=target['gameNumber'], game_type='R', scheduled_start_utc=target['gameDate'],
        champion_model_id='prior_strength_plus_lagged_current_winrate_v1', champion_model_fingerprint=MODEL_HASH,
        champion_home_probability=p, challenger_model_id='prior_strength_plus_lagged_current_winrate_v1',
        challenger_model_fingerprint=MODEL_HASH, challenger_home_probability=p)
    team_identity = {s + '_team': target['teams'][s]['team']['name'] for s in ['home', 'away']}
    team_identity.update({s + '_team_code': target['teams'][s]['team']['abbreviation'] for s in ['home', 'away']})
    bundle = NormalizedPregameInput(source_identity='ISOLATED_SAVED_CANDIDATE' if classification == FORWARD else 'SYNTHETIC_REHEARSAL',
        prediction_rows=(row,), market_rows=(), exclusion_rows=(),
        source_manifest={'official_dates_by_prediction_row_id': {input_hash: target['officialDate']},
                         'team_identities_by_prediction_row_id': {input_hash: team_identity},
                         'input_evidence_sha256': input_hash, 'model_role': 'RESEARCH_CANDIDATE'},
        authority_hashes={}, p37_summary={}, p39_summary={}, p39_source_manifest={})
    run = create_p50c_prediction_run(REPO, pregame_input=bundle, run_classification=classification,
                                    run_root=out / 'runs', created_at_utc=inference,
                                    target_date=target['officialDate'], model_role='RESEARCH_CANDIDATE')
    frozen_persisted = text_time(clock())
    settings = json.loads((REPO / 'report/moneyline_paper_decision_v1/paper_state.json').read_bytes())['policy_parameters']
    policy = MarketComparisonPolicy(**{k: settings[k] for k in ['min_net_ev', 'max_quote_age_seconds', 'paper_cost_per_unit']})
    (out / 'policy.json').write_bytes(encoded(policy.to_payload()))
    quote_records = []; quotes = (); issues = {}; error = None
    if capture_market:
        captures = SourceCaptures(out, clock=clock); captures.purpose = 'QUOTE_CAPTURE'
        try:
            acquisition = acquire_tsl_moneyline_snapshot(repository_root=REPO, runtime_root=out / 'p32a', game_type='R',
                target_date=target['officialDate'],
                minimum_qualified_games=1, clock=captures.now, schedule_opener=captures.fetch,
                tsl_fetcher=lambda url: captures.fetch(__import__('urllib.request', fromlist=['Request']).Request(url)))
            quotes, issues = quotes_from_acquisition(acquisition, load_frozen_prediction_run(run.run_dir, data_mode='PROSPECTIVE'), reject_ambiguous=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            error = str(exc)
        quote_records = captures.records
    quote_path = out / 'quote_snapshots.jsonl'
    quote_path.write_bytes(b''.join(encoded(q).replace(b'\n', b'') + b'\n' for q in quotes))
    decision = text_time(clock())
    source_records = [dict(r, purpose='MODEL_INPUT') for r in manifest['source_captures']]
    source_records += [{'status': 'SUCCESS', 'purpose': 'MODEL_INPUT', 'url': 'LOCAL_ANNUAL_AND_INPUT_SNAPSHOT',
                        'received_at_utc': local_persisted, 'persisted_at_utc': local_persisted}]
    source_records += quote_records
    pair = assess_pair(run.frozen_predictions[0], quotes[0] if len(quotes) == 1 else None,
                       source_records=source_records, freeze_persisted_at_utc=frozen_persisted, decision_at_utc=decision, policy=policy)
    if issues:
        pair.update(status=issues.get(target['gamePk'], 'QUOTE_AMBIGUOUS'), paired_pregame_eligible=False)
    elif len(quotes) > 1:
        pair.update(status='QUOTE_AMBIGUOUS', paired_pregame_eligible=False)
    # Retain raw normalized quotes separately; final comparison admits only the
    # quote that passed the ACTUAL recorded pregame decision time, not max(times).
    (out / 'paired_quote_snapshots.jsonl').write_bytes(quote_path.read_bytes() if pair['paired_pregame_eligible'] else b'')
    (out / 'freeze_manifest.json').write_bytes((run.run_dir / 'run_manifest.json').read_bytes())
    report = {'schema_version': 'lagged_winrate.paper_capture.v1', 'paper_only': True, 'model_role': 'RESEARCH_CANDIDATE',
              'run_relative': str(run.run_dir.relative_to(out)), 'run_classification': classification,
              'inference_at_utc': inference, 'freeze_persisted_at_utc': frozen_persisted, 'decision_at_utc': decision,
              'input_fingerprint': input_hash, 'candidate_probability': probability, 'pair': pair,
              'quote_error': error, 'quote_identity_issues': issues, 'source_records': source_records,
              'synthetic': bool(manifest.get('synthetic')), 'settled_count': 0, 'model_sha256': MODEL_HASH,
              'target_identity': list(identity(target)), 'new_fits': 0}
    if utc(frozen_persisted) >= utc(target['gameDate']):
        raise ValueError('FREEZE_PERSIST_CROSSED_TARGET_START_REJECTED')
    immutable = ['input_manifest.json', 'original_input_manifest.json', 'input_evidence.json', 'model_artifact.json', 'policy.json', 'freeze_manifest.json',
                 'quote_snapshots.jsonl', 'paired_quote_snapshots.jsonl', *snapshots,
                 str((run.run_dir / 'frozen_predictions.jsonl').relative_to(out))]
    immutable += [r['path'] for r in quote_records if r['status'] == 'SUCCESS']
    report['files'] = {n: sha256(inside(out, n).read_bytes()).hexdigest() for n in immutable}
    (out / 'paper_capture.json').write_bytes(encoded(report))
    return report


def replay(out):
    out = isolated(out)
    report = json.loads((out / 'paper_capture.json').read_bytes())
    if report['schema_version'] != 'lagged_winrate.paper_capture.v1' or report['model_sha256'] != MODEL_HASH:
        raise ValueError('PAPER_AUTHORITY_INVALID')
    for name, expected in report['files'].items():
        if sha256(inside(out, name).read_bytes()).hexdigest() != expected:
            raise ValueError('PAPER_CAPTURE_HASH_MISMATCH:' + name)
    saved_model()
    evidence = json.loads((out / 'input_evidence.json').read_bytes())
    _, _, reconstructed_p, reconstructed_evidence, _ = prepare(out, utc(report['inference_at_utc']))
    if reconstructed_evidence != evidence or reconstructed_p != report['candidate_probability']:
        raise ValueError('SOURCE_TO_FEATURE_REPLAY_CHANGED')
    p, _ = research.predict(evidence['features'], saved_model(), 'candidate')
    if str(p) != report['candidate_probability'] or fingerprint(evidence) != report['input_fingerprint']:
        raise ValueError('SAVED_PROBABILITY_OR_INPUT_REPLAY_MISMATCH')
    run = load_frozen_prediction_run(inside(out, report['run_relative']), data_mode='FIXTURE' if report['synthetic'] else 'PROSPECTIVE')
    if run.predictions[0]['model_identity']['model_role'] != 'RESEARCH_CANDIDATE' or run.predictions[0]['model_identity']['p_home'] != report['candidate_probability']:
        raise ValueError('CANDIDATE_ROLE_OR_PROBABILITY_CHANGED')
    quotes = [json.loads(line) for line in (out / 'quote_snapshots.jsonl').read_text().splitlines()]
    settings = json.loads((out / 'policy.json').read_bytes())
    policy = MarketComparisonPolicy(settings['minimum_net_ev'], settings['maximum_quote_age_seconds_at_decision'], settings['paper_cost_per_unit'])
    paired = assess_pair(run.predictions[0], quotes[0] if len(quotes) == 1 else None,
                        source_records=report['source_records'], freeze_persisted_at_utc=report['freeze_persisted_at_utc'],
                        decision_at_utc=report['decision_at_utc'], policy=policy)
    if report['quote_identity_issues']:
        paired.update(status=report['pair']['status'], paired_pregame_eligible=False)
    elif len(quotes) > 1:
        paired.update(status='QUOTE_AMBIGUOUS', paired_pregame_eligible=False)
    if paired != report['pair']:
        raise ValueError('PAIR_REPLAY_CHANGED')
    return report


def settle(out, result_root, *, clock=now):
    out = isolated(out); report = replay(out); current = clock().astimezone(UTC)
    manifest = json.loads((result_root / 'input_manifest.json').read_bytes())
    if len(manifest['source_captures']) != 1:
        raise ValueError('ONE_FINAL_SOURCE_REQUIRED')
    record = manifest['source_captures'][0]
    payload, raw = captured(result_root, record, current)
    expected = tuple(report['target_identity'])
    matching = {fingerprint(g): g for g in games(payload) if identity(g) == expected}
    if len(matching) != 1:
        raise ValueError('FINAL_FULL_IDENTITY_NOT_UNIQUE_OR_MISMATCHED')
    game = next(iter(matching.values()))
    if game['status']['abstractGameState'] != 'Final':
        return {'status': 'PENDING_OFFICIAL_FINAL', 'newly_settled_count': 0}
    if utc(record['received_at_utc']) <= utc(game['gameDate']):
        raise ValueError('RESULT_CAPTURE_NOT_POST_START')
    home = game['teams']['home']['score']; away = game['teams']['away']['score']
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in [home, away]) or home == away:
        raise ValueError('FINAL_SCORE_UNUSABLE')
    run_dir = inside(out, report['run_relative']); frozen_before = (run_dir / 'frozen_predictions.jsonl').read_bytes()
    result = NormalizedResultRecord(prediction_row_id=report['input_fingerprint'], provider_namespace='MLB_STATS_API',
        provider_game_id=str(game['gamePk']), game_number=game['gameNumber'], game_type='R', status='FINAL',
        home_score=home, away_score=away, result_observed_at_utc=record['received_at_utc'], source_identity='MLB_OFFICIAL_FINAL:' + record['sha256'])
    settled = settle_p50c_prediction_run(REPO, run_dir=run_dir, result_input=(result,), ledger_root=out / 'ledger', settled_at_utc=text_time(current))
    if frozen_before != (run_dir / 'frozen_predictions.jsonl').read_bytes():
        raise ValueError('FROZEN_PREDICTION_MUTATED')
    (out / ('final_source_' + record['sha256'] + '.json')).write_bytes(raw)
    settings = json.loads((out / 'policy.json').read_bytes())
    comparison = {'status': 'SYNTHETIC_REHEARSAL_NOT_MARKET_EVIDENCE', 'paired_count': 0} if report['synthetic'] else evaluate_p50c_market_comparison(
        repository_root=REPO, quote_snapshots=[out / 'paired_quote_snapshots.jsonl'], runs_root=out / 'runs', ledger_root=out / 'ledger',
        policy=MarketComparisonPolicy(settings['minimum_net_ev'], settings['maximum_quote_age_seconds_at_decision'], settings['paper_cost_per_unit']))
    return {'status': settled.lifecycle_state, 'newly_settled_count': settled.newly_settled_count,
            'prediction_summary': settled.summary, 'same_game_market_comparison': comparison, 'new_fits': 0}


def uncertainty():
    path = MODEL_ROOT / 'per_game_gate.csv'
    expected = json.loads((MODEL_ROOT / 'source_manifest.json').read_bytes())['capsule_files']['per_game_gate.csv']
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('SEALED_DEVELOPMENT_CSV_CHANGED')
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['phase'] == 'evaluation' and r['paired_admitted'] == 'True']
    counts = Counter()
    for r in rows:
        b = (float(r['baseline_p_home']) >= .5) == bool(int(r['y']))
        c = (float(r['candidate_p_home']) >= .5) == bool(int(r['y']))
        counts['both_correct' if b and c else 'candidate_only_correct' if c else 'baseline_only_correct' if b else 'both_wrong'] += 1
    b = counts['baseline_only_correct']; c = counts['candidate_only_correct']; n = b + c
    numerator = min(2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)), 2 ** n)
    return {'population': len(rows), 'paired_counts': dict(sorted(counts.items())), 'discordant_pairs': n,
            'accuracy_difference': (c - b) / len(rows), 'exact_mcnemar_two_sided_p': numerator / (2 ** n),
            'exact_binomial_numerator': numerator, 'exact_binomial_denominator': 2 ** n,
            'saved_csv_sha256': sha256(path.read_bytes()).hexdigest(), 'new_fits': 0,
            'interpretation': 'Plus3 remains compatible with sampling fluctuation. Development pass does not establish stable or market advantage. Exact test assumes exchangeable independent discordant signs under the null; team/season dependencies and prior reuse limit inference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'replay', 'settle', 'uncertainty'])
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--capture-market', action='store_true')
    args = parser.parse_args()
    if args.action in ('freeze', 'settle') and args.input_root is None:
        parser.error('--input-root is required for freeze/settle')
    if args.action != 'uncertainty' and args.output_root is None:
        parser.error('--output-root is required')
    try:
        if args.action == 'uncertainty':
            result = uncertainty()
        elif args.action == 'freeze':
            result = freeze(args.input_root, args.output_root, capture_market=args.capture_market)
        elif args.action == 'replay':
            result = replay(args.output_root)
        else:
            result = settle(args.output_root, args.input_root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(encoded({'status': 'BLOCKED', 'error': str(exc), 'new_fits': 0}).decode(), end='')
        raise SystemExit(1)
    print(encoded(result).decode(), end='')


if __name__ == '__main__':
    main()
