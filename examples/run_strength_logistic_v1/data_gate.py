"""Hash-bound data feasibility audit only; never fits or scores any candidate."""
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import csv
import json
import re
import sys

def parse(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('UTC_INSTANT_REQUIRED')
    return parsed.astimezone(UTC)

def strict_completion_reason(row,cutoff):
    """Future data contract: require actual final-event evidence, never a derived clock."""
    if row.get('timing_is_derived') is not False or row.get('completion_source_kind')!='OFFICIAL_ACTUAL_FINAL_EVENT':
        return 'ACTUAL_COMPLETION_EVENT_PROVENANCE_REQUIRED'
    if not row.get('completed_at_utc') or not row.get('first_pitch_utc'):
        return 'ACTUAL_UTC_TIMING_MISSING'
    if not row.get('final') or row.get('game_type')!='R':return 'FINAL_REGULAR_GAME_REQUIRED'
    if not re.fullmatch(r'[0-9a-f]{64}',row.get('raw_sha256','')):
        return 'RAW_RESPONSE_HASH_REQUIRED'
    if row.get('suspended_resumed') and not row.get('resumption_final_event_resolved'):
        return 'RESUMPTION_FINAL_COMPLETION_UNRESOLVED'
    try:
        started=parse(row['first_pitch_utc']);finished=parse(row['completed_at_utc']);at=parse(cutoff)
    except (TypeError,ValueError):return 'UTC_INSTANT_REQUIRED'
    if finished<started:return 'COMPLETION_BEFORE_FIRST_PITCH'
    if finished>=at:return 'NOT_COMPLETED_STRICTLY_BEFORE_FEATURE_CUTOFF'
    return None

def audit_year(year,rows,gamelogs):
    valid_labels=0
    for row in rows:
        try:
            hs=int(row['home_score']);away=int(row['away_score']);label=Decimal(row['home_win'])
            if hs>=0 and away>=0 and hs!=away and label==int(hs>away):valid_labels+=1
        except (ValueError,ArithmeticError):pass
    unique={tuple(g[i] for i in [0,1,3,6]) for g in gamelogs}
    ambiguous=Counter((row['date'],row['away_team'],row['home_team']) for row in rows)
    actual_utc_fields=[field for field in (list(rows[0]) if rows else [])
                       if field.lower() in ['completed_at_utc','actual_final_event_at_utc','game_end_utc']]
    verified=sum(strict_completion_reason(row,'2026-01-01T00:00:00Z') is None for row in rows)
    completion_dates=[{'retrosheet_key':'|'.join(g[i] for i in [0,1,3,6]),
        'original_date':g[0],'completion_information_field14':g[13],
        'duration_minutes_field19':g[18],'actual_completion_utc_available':False}
        for g in gamelogs if g[13]]
    return {'season':year,'csv_rows':len(rows),'raw_gamelog_rows':len(gamelogs),'csv_fields':list(rows[0]) if rows else [],
        'date_min':min(row['date'] for row in rows),'date_max':max(row['date'] for row in rows),
        'valid_non_tie_score_labels':valid_labels,'invalid_or_conflicting_labels':len(rows)-valid_labels,
        'unique_retrosheet_date_number_away_home_keys':len(unique),
        'duplicate_csv_date_team_excess_rows':len(rows)-len(ambiguous),
        'ambiguous_csv_date_team_groups':sum(n>1 for n in ambiguous.values()),
        'raw_game_duration_minutes_present':sum(g[18].isdigit() for g in gamelogs),
        'display_EDT_start_time_rows':sum(bool(row.get('Start Time (EDT)')) for row in rows),
        'actual_completion_utc_fields':actual_utc_fields,'strict_verified_completion_rows':verified,
        'strict_feature_admitted_training_rows':0,'rows_missing_actual_completion_provenance':len(rows)-verified,
        'suspended_or_protested_later_completion_rows':len(completion_dates),'later_completion_examples':completion_dates,
        'prior_research_use':'DEVELOPMENT_ALREADY_REUSED','training_status':'BLOCKED_ACTUAL_COMPLETION_CHRONOLOGY_AND_CUTOFFS'}

def audit(root,legacy_data_root=None):
    protocol=json.loads((root/'protocol.json').read_bytes())
    interpretation=json.loads((root/'interpretation_policy.json').read_bytes())
    paths={}
    original_repo=Path(protocol['sources']['eval_population']['path']).parents[2]
    original_legacy=Path(protocol['sources']['train_csv']['path']).parent
    for key,source in protocol['sources'].items():
        path=Path(source['path'])
        if path.is_relative_to(original_repo):
            path=root.parent.parent/path.relative_to(original_repo)
        elif legacy_data_root is not None and path.is_relative_to(original_legacy):
            path=Path(legacy_data_root)/path.relative_to(original_legacy)
        if sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('SOURCE_HASH_MISMATCH:'+key)
        paths[key]=path
    seasons=[]
    for year,prefix in [(2024,'earlier'),(2025,'train')]:
        with paths[prefix+'_csv'].open() as stream:rows=list(csv.DictReader(stream))
        with paths[prefix+'_gamelog'].open() as stream:raw=list(csv.reader(stream))
        if any(len(row)!=161 for row in raw):raise ValueError('RETROSHEET_161_COLUMN_CONTRACT')
        seasons.append(audit_year(year,rows,raw))
    population=[json.loads(row) for row in paths['eval_population'].read_text().splitlines()]
    outcomes=[json.loads(row) for row in paths['eval_opened_outcomes'].read_text().splitlines()]
    schedule=[game for date in json.loads(paths['eval_schedule'].read_bytes())['dates'] for game in date['games']]
    timing=[game for date in json.loads(paths['eval_game_info'].read_bytes())['dates'] for game in date['games']]
    schedule_index=defaultdict(list);time_index=defaultdict(list);outcome_index=defaultdict(list)
    for game in schedule:schedule_index[game['gamePk']].append(game)
    for game in timing:time_index[game['gamePk']].append(game)
    for game in outcomes:outcome_index[int(game['provider_game_id'])].append(game)
    def matches(game,target):
        return (game['gamePk']==target['game_pk'] and game['gameType']==target['game_type']
            and game['officialDate']==target['official_date'] and game['gameNumber']==target['game_number']
            and game['gameDate']==target['scheduled_start_utc']
            and all(game['teams'][s]['team']['id']==target[s+'_team']['id'] for s in ['home','away']))
    paired_market=0;valid_outcomes=0;required_history_ids=set();eval_rows=[]
    prior_findings=json.loads(paths['verified_history_findings'].read_bytes())
    history_plan=json.loads(paths['verified_history_plan'].read_bytes())
    unavailable={row['game_pk'] for row in prior_findings['unavailable_history_sources']}
    archived_final_ids={source['game_pk'] for source in history_plan['required_sources'] if source['game_pk'] not in unavailable}
    for target in population:
        pk=target['game_pk'];errors=[]
        raw={json.dumps(g,sort_keys=True):g for g in schedule_index[pk] if matches(g,target)}
        info={json.dumps(g,sort_keys=True):g for g in time_index[pk] if matches(g,target)}
        if len(raw)!=1 or len(info)!=1:errors.append('UNIQUE_FULL_IDENTITY_OR_ACTUAL_START_UNAVAILABLE')
        if not errors:
            raw=next(iter(raw.values()));info=next(iter(info.values()))
            quote=parse(target['market_observed_at_utc'])
            first=info.get('gameInfo',{}).get('firstPitch')
            try:
                hp=Decimal(target['home_decimal_price']);ap=Decimal(target['away_decimal_price'])
                valid_quote=(first and quote<min(parse(target['scheduled_start_utc']),parse(target['source_game_time_utc']),parse(first))
                    and hp>1 and ap>1 and re.fullmatch(r'[0-9a-f]{64}',target.get('raw_record_sha256',''))
                    and abs(ap/(hp+ap)-Decimal(target['market_no_vig_home_probability']))<=Decimal('1e-25'))
            except (TypeError,ValueError,ArithmeticError):valid_quote=False
            if valid_quote:paired_market+=1
            else:errors.append('PAIRED_PRESTART_TWO_SIDED_QUOTE_NOT_VERIFIED')
            labels=outcome_index[pk]
            hs=raw['teams']['home'].get('score');away=raw['teams']['away'].get('score')
            if (len(labels)==1 and hs is not None and away is not None and hs!=away
                and labels[0]['official_date']==target['official_date'] and labels[0]['game_number']==target['game_number']
                and int(hs>away)==labels[0]['actual_home_win']):valid_outcomes+=1
            else:errors.append('TARGET_OUTCOME_MISSING_OR_CONFLICTING')
        q=parse(target['market_observed_at_utc']);teams={target[s+'_team']['id'] for s in ['home','away']}
        required={g['gamePk'] for g in schedule if g['gameType']=='R' and g['status']['abstractGameState']=='Final'
            and parse(g['gameDate'])<q and any(g['teams'][s]['team']['id'] in teams for s in ['home','away'])
            and all(g['teams'][s].get('score') is not None for s in ['home','away']) and g['teams']['home']['score']!=g['teams']['away']['score']}
        required_history_ids.update(required)
        missing=required-archived_final_ids
        errors.append('FULL_SEASON_TO_QUOTE_ACTUAL_COMPLETION_CHRONOLOGY_INCOMPLETE')
        eval_rows.append({'game_pk':pk,'quote_feature_cutoff_utc':target['market_observed_at_utc'],
            'possible_prior_scored_R_game_count':len(required),'existing_archived_final_subset_count':len(required & archived_final_ids),
            'no_archived_final_version_in_existing_subset_count':len(missing),'candidate_feature_admitted':False,'exclusions':errors})
    assert len(population)==len({g['game_pk'] for g in population})==102
    metrics={name:{'sample_count':0,'accuracy':None,'brier':None,'logloss':None}
             for name in ['candidate','market','fixed_home_0_5_reference']}
    return {'schema_version':'run_strength_logistic.data_gate.v1','protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
        'status':'STOPPED_AT_USER_REQUIRED_DATA_GATE','candidate_implemented_and_fitted':False,
        'candidate_fit_count':0,'prediction_count':0,'model_comparison_count':0,'hyperparameter_search_count':0,
        'optimization_trials':0,'new_data_source_gets':0,'earlier_seasons':seasons,
        'preferred_training_season':2025,'preferred_training_raw_label_population':2430,'strict_training_feature_rows':0,
        'evaluation_population':102,'unique_game_ids':102,'paired_prestart_two_sided_market_records':paired_market,
        'reusable_verified_target_outcomes':valid_outcomes,'strict_evaluation_feature_rows':0,'paired_model_market_games':0,
        'excluded_from_candidate_comparison':102,'same_game_probability_metrics':metrics,
        'season_to_quote_history_gap':{'unique_possible_prior_R_ids':len(required_history_ids),
            'covered_by_existing_archived_final_subset':len(required_history_ids & archived_final_ids),
            'missing_from_existing_archived_final_subset':len(required_history_ids-archived_final_ids),
            'archive_subset_is_not_full_season_history':True,
            'required_unresolved_game_pks':sorted(required_history_ids-archived_final_ids)},
        'blocking_requirements':['2025 training full game identity to actual first-pitch and actual completion UTC provenance for all potentially used prior R games.',
            '2026 complete season-to-fixed-quote team histories; a few selected Final feeds and firstPitch+duration projections do not establish the required actual completion timeline.'],
        'market_is_primary_blocker':paired_market!=102,'alternative_ranges_not_executed':[
            {'scope':'2025 regular-season magnitude research, 2430 verified score labels','status':'Requires actual UTC completion/start enrichment and warmup admissions before the single fit.'},
            {'scope':'2024 earlier alternative, 2429 verified score labels','status':'Same actual timing deficit, CSV lacks game numbers; raw logs supply unique doubleheader keys. Already reused development.'},
            {'scope':'Existing 2026 102 price records','status':'Market/outcomes reusable; complete season-to-quote magnitude features unavailable.'},
            {'scope':'Date-only, previous-day or firstPitch+minute-duration reconstruction','status':'Would relax the user-required true completion gate; not substituted or executed.'}],
        'success_priority':interpretation['user_accepted_priority'],'efficacy_status':'NOT_EVALUABLE',
        'numeric_success_threshold':interpretation['numeric_improvement_threshold'],
        'prior_use_evidence':{'2024':'p39_2024_holdout_summary.json; p41_cross_year_combined_summary.json',
            '2025':'p41_cross_year_combined_summary.json; p44_temporal_stability_summary.json; existing P22/P37 history',
            '2026':'Prior 102 opened price-only and independent recent-ten research; all future comparisons remain development.'},
        'rows':eval_rows,'profitability_claim':False,'near10_rule_rerun_count':0,
        'preserved':['Champion','main','ledger','frozen predictions','all prior sealed research artifacts']}

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    legacy_root=Path(sys.argv[2]) if len(sys.argv)>2 else None
    print(json.dumps(audit(root,legacy_root),sort_keys=True,indent=2))
