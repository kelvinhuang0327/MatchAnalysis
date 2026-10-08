"""Offline feature recovery, with field-level historical cutoffs and no model scoring."""
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import sys

def parse(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(UTC)

def analyze(root):
    plan = json.loads((root / 'plan.json').read_bytes())
    history = json.loads((root / 'required_history_plan.json').read_bytes())
    records = json.loads((root / 'captures.json').read_bytes())
    audit = json.loads((root / 'cache_locator_audit.json').read_bytes())
    if plan['target_ids'] != [824673, 824269] or history['model_targets'] != 2:
        raise ValueError('EXACT_TWO_FIXED_TARGETS_REQUIRED')
    successful = {r['path']: r for r in records if r['status'] == 'SUCCESS'}
    for path, record in successful.items():
        if sha256((root / path).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('RAW_HASH_MISMATCH')
        if not (parse(record['request_started_at_utc']) <= parse(record['received_at_utc'])
                <= parse(record['persisted_at_utc'])):
            raise ValueError('ACTUAL_RECEIPT_ORDER')
    sources = {}
    unavailable = {}
    for source in history['required_sources']:
        if source['path'] not in successful:
            unavailable[source['game_pk']] = {'reason': 'NOT_CAPTURED_AFTER_FEATURE_STOP', 'path': source['path']}
            continue
        data = json.loads((root / source['path']).read_bytes())
        gd = data['gameData']
        cutoff = parse(source['cutoff_utc'])
        version = datetime.strptime(data['metaData']['timeStamp'], '%Y%m%d_%H%M%S').replace(tzinfo=UTC)
        events = [parse(play['about'][key])
                  for play in data['liveData']['plays']['allPlays']
                  for key in ['startTime', 'endTime'] if play.get('about', {}).get(key)]
        if not (data['gamePk'] == gd['game']['pk'] == source['game_pk']
                and gd['game']['type'] == 'R' and gd['game']['season'] == '2026'
                and gd['status']['abstractGameState'] == 'Final'
                and version <= cutoff and events and max(events) <= cutoff):
            unavailable[source['game_pk']] = {'reason': 'PREQUOTE_FINAL_IDENTITY_OR_EVENT_FAILURE',
                'path': source['path'], 'returned_timecode': data['metaData']['timeStamp'],
                'status': gd['status']}
            continue
        for role in source['roles']:
            if role['feature'] == 'recent_10':
                expected = role['identity']
                if not (gd['datetime']['officialDate'] == expected['officialDate']
                        and gd['datetime']['dateTime'] == expected['gameDate']
                        and int(gd['game']['id'].rsplit('-', 1)[-1]) == expected['gameNumber']
                        and all(gd['teams'][side]['id'] == expected[side+'_id'] for side in ['home', 'away'])):
                    raise ValueError('RECENT_GAME_IDENTITY_MISMATCH')
            elif gd['datetime']['officialDate'] != role['locator_date']:
                raise ValueError('APPEARANCE_DATE_MISMATCH')
        sources[source['game_pk']] = data
    results = []
    for target in plan['targets']:
        ident = target['identity']
        pk = ident['game_pk']
        quote = parse(ident['market_observed_at_utc'])
        frozen = json.loads((root / f'raw/{pk}-feature-version-system.json').read_bytes())
        gd = frozen['gameData']
        target_version = datetime.strptime(frozen['metaData']['timeStamp'], '%Y%m%d_%H%M%S').replace(tzinfo=UTC)
        target_events = [parse(play['about'][key]) for play in frozen['liveData']['plays']['allPlays']
                         for key in ['startTime', 'endTime'] if play.get('about', {}).get(key)]
        if not (target_version <= quote and gd['status']['abstractGameState'] == 'Preview'
                and frozen['metaData']['timeStamp'] == target['archived_timecode']
                and all(event <= quote for event in target_events)
                and frozen['gamePk'] == gd['game']['pk'] == pk
                and gd['game']['type'] == ident['game_type']
                and gd['datetime']['officialDate'] == ident['official_date']
                and gd['datetime']['dateTime'] == ident['scheduled_start_utc']
                and int(gd['game']['id'].rsplit('-', 1)[-1]) == ident['game_number']
                and all(gd['teams'][s]['id'] == ident[s+'_team']['id'] for s in ['home', 'away'])):
            raise ValueError('TARGET_PREQUOTE_IDENTITY')
        recent = {}
        pitching = {}
        team_audit = next(x for x in audit['targets'] if x['game_pk'] == pk)['teams']
        leaked_records = {}
        for side in ['home', 'away']:
            tid = ident[side+'_team']['id']
            candidates = target['team_recent_game_candidates'][side]
            if len(candidates) != 10 or set(team_audit[side]['known_window_scheduled_ids']) != {g['gamePk'] for g in candidates}:
                raise ValueError('KNOWN_CACHE_WINDOW_INCOMPLETE')
            game_rows = []
            recent_missing = [g['gamePk'] for g in candidates if g['gamePk'] not in sources]
            for candidate in sorted(candidates, key=lambda g: g['gameDate']):
                if candidate['gamePk'] not in sources:
                    continue
                data = sources[candidate['gamePk']]
                historical_side = next(s for s in ['home', 'away'] if data['gameData']['teams'][s]['id'] == tid)
                score = data['liveData']['linescore']['teams']
                home, away = score['home']['runs'], score['away']['runs']
                if home == away or parse(candidate['gameDate']) >= min(quote, parse(ident['scheduled_start_utc'])):
                    raise ValueError('RECENT_GAME_NOT_USABLE')
                won = (home > away) if historical_side == 'home' else (away > home)
                game_rows.append({'game_pk': candidate['gamePk'], 'scheduled_start_utc': candidate['gameDate'],
                    'home_score': home, 'away_score': away, 'team_won': won,
                    'returned_timecode': data['metaData']['timeStamp']})
            wins = sum(x['team_won'] for x in game_rows)
            recent[side] = {'team_id': tid, 'verified_completed_games': 10, 'wins': wins,
                            'rate': str(Decimal(wins)/10), 'games': game_rows,
                            'status': 'PREQUOTE_FINAL_VALUES_RECOVERED_WITHIN_UNVERSIONED_OFFICIAL_INDEX'}
            if recent_missing:
                recent[side] = {'team_id': tid, 'verified_completed_games': len(game_rows),
                    'required_missing_game_pks': recent_missing, 'games': game_rows, 'rate': None,
                    'status': 'NOT_FULLY_RECOVERABLE_AVAILABLE_ARCHIVE_LACKS_PREQUOTE_FINAL'}
            pid = gd['probablePitchers'][side]['id']
            target_player = frozen['liveData']['boxscore']['teams'][side]['players']['ID'+str(pid)]
            if (target_player['person']['id'] != pid
                    or target_player['stats']['pitching']['gamesPlayed'] != 0):
                raise ValueError('TARGET_PLAYER_ALREADY_PLAYED_OR_ID_MISMATCH')
            logs = []
            pitcher_missing = []
            for source in history['required_sources']:
                roles = [r for r in source['roles'] if r.get('pitcher_id') == pid and r['target_game_pk'] == pk]
                if not roles: continue
                if source['game_pk'] not in sources:
                    pitcher_missing.append(source['game_pk'])
                    continue
                data = sources[source['game_pk']]
                if data['gameData']['datetime']['officialDate'] >= ident['official_date']:
                    raise ValueError('SAME_DAY_OR_FUTURE_APPEARANCE')
                hs = next(s for s in ['home', 'away'] if data['gameData']['teams'][s]['id'] == tid)
                p = data['liveData']['boxscore']['teams'][hs]['players']['ID'+str(pid)]
                stats = p['stats']['pitching']
                if not (p['person']['id'] == pid and stats['gamesPlayed'] == 1
                        and stats['gamesStarted'] in [0, 1]
                        and isinstance(stats['outs'], int) and stats['outs'] >= 0
                        and isinstance(stats['earnedRuns'], int) and stats['earnedRuns'] >= 0):
                    raise ValueError('HISTORICAL_APPEARANCE_ROLE_OR_COUNTS_INVALID')
                logs.append({'game_pk': source['game_pk'], 'date': data['gameData']['datetime']['officialDate'],
                             'games_started': stats['gamesStarted'], 'earned_runs': stats['earnedRuns'],
                             'outs': stats['outs'], 'returned_timecode': data['metaData']['timeStamp']})
            cumulative = target_player['seasonStats']['pitching']
            aggregates = {'gamesPlayed': len(logs), 'gamesStarted': sum(x['games_started'] for x in logs),
                          'earnedRuns': sum(x['earned_runs'] for x in logs), 'outs': sum(x['outs'] for x in logs)}
            reconcile = not pitcher_missing and all(aggregates[k] == cumulative[k] for k in aggregates)
            starts = [x for x in logs if x['games_started'] == 1]
            outs = sum(x['outs'] for x in starts)
            er = sum(x['earned_runs'] for x in starts)
            if not reconcile or len(starts) < 2 or outs <= 0:
                pitching[side] = {'pitcher_id': pid, 'verified_prior_appearance_count': len(logs),
                    'required_missing_game_pks': pitcher_missing, 'prior_appearances': logs,
                    'status': 'NOT_FULLY_RECOVERABLE_AVAILABLE_ARCHIVE_OR_RECONCILIATION_INCOMPLETE',
                    'starts_only_era': None, 'seasonStats_used_as_era_input': False}
            else:
                pitching[side] = {'pitcher_id': pid, 'prior_appearance_count': len(logs), 'prior_start_count': len(starts),
                    'excluded_relief_count': len(logs)-len(starts), 'starter_earned_runs': er, 'starter_outs': outs,
                    'starts_only_era': str(Decimal(er*27)/Decimal(outs)), 'era_numerator': er*27,
                    'era_denominator': outs, 'cumulative_reconciliation': aggregates, 'prior_appearances': logs,
                    'status': 'RECOVERED_FROM_PREQUOTE_FINAL_GAME_STATS_CUMULATIVE_COUNTS_RECONCILED',
                    'seasonStats_used_as_era_input': False}
            rec = gd['teams'][side]['record']
            post = team_audit[side]['target_postgame_league_record_for_leakage_audit_only']
            leaked_records[side] = {'archived_preview_record': rec,
                'cache_prior_completed_count': team_audit[side]['cache_prior_completed_count'],
                'record_matches_target_postgame_record': rec['wins'] == post['wins'] and rec['losses'] == post['losses'],
                'record_gamesPlayed_exceeds_prior_completed_by': rec['gamesPlayed']-team_audit[side]['cache_prior_completed_count'],
                'admitted_as_pregame_feature': False}
        values_ready = all(p.get('starts_only_era') is not None for p in pitching.values()) and all(r['rate'] is not None for r in recent.values())
        results.append({'game_pk': pk, 'quote_cutoff_utc': ident['market_observed_at_utc'],
            'target_returned_timecode': frozen['metaData']['timeStamp'], 'pitching': pitching, 'recent_10': recent,
            'recent_10_home_minus_away_candidate': str(Decimal(recent['home']['wins']-recent['away']['wins'])/10) if all(r['rate'] is not None for r in recent.values()) else None,
            'gameData_record_field_leakage_audit': leaked_records,
            'feature_values_reconstructed': values_ready,
            'full_input_cutoff_certification': 'UNCERTAIN_NO_PREQUOTE_VERSION_OF_COMPLETE_TEAM_GAME_INDEX' if values_ready else 'NOT_FULLY_RECOVERABLE_REQUIRED_HISTORY_MISSING_AT_CUTOFF',
            'production_or_frozen_ledger_admitted': False,
            'first_announcement_time_established': False})
    return {'schema_version': 'two_target_feature_cutoff.findings.v1',
        'attempt_count': len(records), 'successful_response_count': len(successful),
        'failed_attempt_count': len(records)-len(successful),
        'raw_response_bytes': sum(r['byte_count'] for r in successful.values()),
        'unique_history_games_verified': len(sources), 'prior_appearances_verified': sum(
            len(p['prior_appearances']) for row in results for p in row['pitching'].values()),
        'starts_only_era_recovered_pitchers': sum(p['starts_only_era'] is not None for row in results for p in row['pitching'].values()),
        'recent_10_value_reconstructed_team_windows': sum(r['rate'] is not None for row in results for r in row['recent_10'].values()),
        'feature_value_reconstructed_targets': sum(row['feature_values_reconstructed'] for row in results),
        'fully_certified_complete_historical_input_targets': 0,
        'uncertified_index_membership_targets': 2, 'prospective_samples_added': 0,
        'model_score_or_method_evaluation_count': 0, 'new_hypothesis_count': 0,
        'historical_provider_version_is_not_actual_local_receipt': True,
        'team_history_index_provenance': audit['source_path'], 'team_history_index_sha256': audit['source_sha256'],
        'limits': ['Official cached schedule is a postquote locator, without an immutable prequote universe version.',
                   'Each used score and pitching line is verified in a prequote Final feed; absent/omitted team games cannot be certified by leaked gameData.teams.record.',
                   'No current seasonStats replaces ERA. Historical cumulative counts only reconcile individually archived appearances.',
                   'No blind holdout, prospective sample, prediction, Brier, ROI or method finding is produced.'],
        'unavailable_history_sources': [{'game_pk': pk, **v} for pk, v in sorted(unavailable.items())],
        'unavailable_final_route_evidence': {
            '823135': {'version_list_path': 'raw/823135-version-availability.json',
                'last_listed_timecode': json.loads((root/'raw/823135-version-availability.json').read_bytes())[-1],
                'exact_listed_request_path': 'raw/823135-exact-listed-history.json',
                'exact_listed_request_returned_timecode': json.loads((root/'raw/823135-exact-listed-history.json').read_bytes())['metaData']['timeStamp'],
                'exact_listed_request_returned_status': json.loads((root/'raw/823135-exact-listed-history.json').read_bytes())['gameData']['status'],
                'interpretation': 'Both bounded requests fail to return a prequote Final. This does not prove every alternative official source lacks one.'},
            '824290': {'interpretation': 'Required historical feed returns Live; Montero ERA recovery stops without substituting cumulative ERA.'}},
        'rows': results}

if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(json.dumps(analyze(root), sort_keys=True, indent=2))
