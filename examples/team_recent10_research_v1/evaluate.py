"""One frozen direction-only recent-ten research baseline on already-opened games."""
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import csv
import io
import sys

def parse(value): return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(UTC)
def lines(path): return [json.loads(s) for s in path.read_text().splitlines() if s]
def identity(g):
    return (int(g['gamePk']),g['gameType'],g['officialDate'],int(g['gameNumber']),g['gameDate'],
            int(g['teams']['home']['team']['id']),int(g['teams']['away']['team']['id']))
def price_identity(g):
    return (g['game_pk'],g['game_type'],g['official_date'],g['game_number'],g['scheduled_start_utc'],
            g['home_team']['id'],g['away_team']['id'])
def direction(home_wins,away_wins):
    return 'HOME' if home_wins>away_wins else 'AWAY' if home_wins<away_wins else None

def completion(g):
    """Historical duration estimate with one minute rounding margin; no immutable version claim."""
    info=g.get('gameInfo',{})
    if not info.get('firstPitch') or not isinstance(info.get('gameDurationMinutes'),int) or info['gameDurationMinutes']<=0:
        return None
    return parse(info['firstPitch'])+timedelta(minutes=info['gameDurationMinutes'],seconds=60)

def unique_matching(index,expected):
    # Identical duplicate representations collapse; conflicting complete identities do not.
    matches={json.dumps(g,sort_keys=True):g for g in index.get(expected[0],[]) if identity(g)==expected}
    return next(iter(matches.values())) if len(matches)==1 else None

def summarize(rows,which):
    selected=[r for r in rows if r['included'] and r[which] is not None]
    correct=sum(r[which]==r['actual_winner'] for r in selected)
    return {'direction_count':len(selected),'correct_count':correct,
            'accuracy':str(Decimal(correct)/len(selected)) if selected else None,
            'coverage_of_included':str(Decimal(len(selected))/len(rows)) if rows else None,
            'probability_mapping':None,'brier':None,'log_loss':None}

def table_csv(result):
    output=io.StringIO(newline='')
    fields=['game_pk','official_date','included','home_prior10_wins','away_prior10_wins',
            'recent10_direction','market_direction','always_home_direction','actual_winner','exclusions']
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in result['rows']:
        writer.writerow({'game_pk':row['game_pk'],'official_date':row['official_date'],
            'included':row['included'],'home_prior10_wins':row['histories'].get('home',{}).get('wins'),
            'away_prior10_wins':row['histories'].get('away',{}).get('wins'),
            'recent10_direction':row['recent10_direction'] or ('TIE_ABSTAIN' if row['included'] else 'EXCLUDED'),
            'market_direction':row['market_direction'],'always_home_direction':row['home_direction'],
            'actual_winner':row['actual_winner'],'exclusions':'|'.join(row['exclusions'])})
    return output.getvalue()

def evaluate(root):
    protocol=json.loads((root/'protocol.json').read_bytes())
    if protocol['probability_mapping'] is not None or protocol['expected_population']!=102:
        raise ValueError('FIXED_DIRECTION_ONLY_PROTOCOL_REQUIRED')
    population=lines(root/'population.jsonl')
    outcomes=lines(root/'opened_outcomes.jsonl')
    repo=root.parent.parent
    schedule_path=repo/'data/fixtures/p23f2_official_2026_history/raw/schedule_2026-03-25_2026-06-30.json'
    if sha256(schedule_path.read_bytes()).hexdigest()!=protocol['source_hashes']['raw_schedule']:
        raise ValueError('CACHE_SOURCE_HASH_MISMATCH')
    captures=json.loads((root/'source_captures.json').read_bytes())
    for capture in captures:
        if capture['status']=='SUCCESS' and sha256((root/capture['path']).read_bytes()).hexdigest()!=capture['sha256']:
            raise ValueError('TIMING_SOURCE_HASH_MISMATCH')
    if len(population)!=102 or len({price_identity(p) for p in population})!=102 or len({p['game_pk'] for p in population})!=102:
        raise ValueError('EXACT_UNIQUE_102_POPULATION_REQUIRED')
    cache=defaultdict(list); timing=defaultdict(list); outcome_index=defaultdict(list)
    for date in json.loads(schedule_path.read_bytes())['dates']:
        for g in date['games']:cache[g['gamePk']].append(g)
    if not (root/'completion_source.json').exists():
        timing_rows=[]
    else:
        timing_rows=[g for date in json.loads((root/'completion_source.json').read_bytes())['dates'] for g in date['games']]
    for g in timing_rows:timing[g['gamePk']].append(g)
    for g in outcomes:outcome_index[int(g['provider_game_id'])].append(g)
    locators={g['game_pk']:g for g in json.loads((root/'history_locators.json').read_bytes())}
    rows=[]
    for target in population:
        pk=target['game_pk']; expected=price_identity(target)
        row={'game_pk':pk,'official_date':target['official_date'],'identity':expected,
             'scheduled_start_utc':target['scheduled_start_utc'],'quote_at_utc':target['market_observed_at_utc'],
             'included':False,'exclusions':[],'recent10_direction':None,'market_direction':None,'home_direction':None,
             'actual_winner':None,'histories':{},'exploratory_already_opened':True}
        raw=unique_matching(cache,expected); current=unique_matching(timing,expected)
        if raw is None:row['exclusions'].append('TARGET_FULL_IDENTITY_NOT_UNIQUE_IN_CACHE')
        if current is None:row['exclusions'].append('TARGET_FULL_IDENTITY_OR_TIMING_NOT_UNIQUE_IN_OFFICIAL_PROJECTION')
        target_outcomes=outcome_index[pk]
        if len(target_outcomes)!=1 or target_outcomes[0]['official_date']!=target['official_date'] or target_outcomes[0]['game_number']!=target['game_number']:
            row['exclusions'].append('OPENED_OUTCOME_NOT_UNIQUE_FULL_GAME_MATCH')
        if row['exclusions']:
            rows.append(row);continue
        hs=raw['teams']['home'].get('score'); away_score=raw['teams']['away'].get('score')
        if raw['status']['abstractGameState']!='Final' or current['status']['abstractGameState']!='Final' or hs is None or away_score is None or hs==away_score:
            row['exclusions'].append('TARGET_NOT_FINAL_OR_SCORE_UNUSABLE')
        elif int(hs>away_score)!=target_outcomes[0]['actual_home_win']:
            row['exclusions'].append('OPENED_OUTCOME_CACHE_CONFLICT')
        first=current.get('gameInfo',{}).get('firstPitch')
        if not first or parse(target['market_observed_at_utc'])>=min(parse(target['scheduled_start_utc']),parse(target['source_game_time_utc']),parse(first)):
            row['exclusions'].append('QUOTE_NOT_ESTABLISHED_BEFORE_TARGET_STARTS')
        if target['game_type']!='R' or target['provider_namespace']!='MLB_STATS_API' or str(pk)!=target['provider_game_id']:
            row['exclusions'].append('TARGET_PROVIDER_OR_GAME_TYPE_MISMATCH')
        try:
            hp=Decimal(target['home_decimal_price']);ap=Decimal(target['away_decimal_price'])
            if hp<=1 or ap<=1:raise ValueError('INVALID_PRICES')
            market=ap/(hp+ap)
            if abs(market-Decimal(target['market_no_vig_home_probability']))>Decimal('1e-25'):
                raise ValueError('NO_VIG_MISMATCH')
        except (ValueError,ArithmeticError):
            row['exclusions'].append('MARKET_TWO_SIDED_PRICE_INVALID_OR_INCONSISTENT')
        if row['exclusions']:
            rows.append(row);continue
        cutoff=parse(target['scheduled_start_utc'])
        for side in ['home','away']:
            tid=target[side+'_team']['id']; ids=list(dict.fromkeys(locators[pk]['teams'][side])); candidates=[]
            for game_pk in ids:
                versions=cache[game_pk]
                # A suspended physical game has multiple starts; never count it twice.
                canonical=min(versions,key=lambda g:g['gameDate'])
                candidates.append(canonical)
            candidates.sort(key=lambda g:(parse(g['gameDate']),g['gamePk']),reverse=True)
            accepted=[];history_errors=[];not_completed=[]
            for candidate in candidates:
                if len(accepted)==10:break
                game_pk=candidate['gamePk']; hist=unique_matching(timing,identity(candidate))
                if hist is None:
                    history_errors.append({'game_pk':game_pk,'reason':'HISTORY_FULL_IDENTITY_OR_COMPLETION_NOT_UNIQUE'});break
                if any(any(g.get(key) for key in ['resumeDate','resumeGameDate','resumedFrom','resumedFromDate']) for g in cache[game_pk]):
                    history_errors.append({'game_pk':game_pk,'reason':'SUSPENDED_RESUMED_GAME_COMPLETION_CANNOT_BE_DERIVED_FROM_ORIGINAL_FIRST_PITCH'});break
                end=completion(hist)
                if end is None or hist['status']['abstractGameState']!='Final':
                    history_errors.append({'game_pk':game_pk,'reason':'HISTORY_COMPLETION_TIME_MISSING_OR_NOT_FINAL'});break
                if parse(hist['gameDate'])>=cutoff or end>=cutoff:
                    not_completed.append({'game_pk':game_pk,'derived_completion_upper_bound_utc':end.isoformat().replace('+00:00','Z')});continue
                if candidate['gameType']!='R' or game_pk==pk or tid not in [candidate['teams'][s]['team']['id'] for s in ['home','away']]:
                    history_errors.append({'game_pk':game_pk,'reason':'HISTORY_GAME_TYPE_TEAM_OR_SELF_MISMATCH'});break
                historical_side=next(s for s in ['home','away'] if candidate['teams'][s]['team']['id']==tid)
                home_score=candidate['teams']['home']['score'];ascore=candidate['teams']['away']['score']
                won=(home_score>ascore) if historical_side=='home' else (ascore>home_score)
                accepted.append({'game_pk':game_pk,'official_date':candidate['officialDate'],
                    'scheduled_start_utc':candidate['gameDate'],'first_pitch_utc':hist['gameInfo']['firstPitch'],
                    'game_duration_minutes':hist['gameInfo']['gameDurationMinutes'],
                    'derived_completion_upper_bound_utc':end.isoformat().replace('+00:00','Z'),
                    'team_won':won,'home_score':home_score,'away_score':ascore})
            if history_errors or len(accepted)<10:
                row['exclusions'].append(side.upper()+'_PRIOR10_COMPLETION_OR_ORDER_NOT_ESTABLISHED')
            row['histories'][side]={'team_id':tid,'completed_count':len(accepted),'wins':sum(g['team_won'] for g in accepted) if len(accepted)==10 and not history_errors else None,
                'games':accepted,'errors':history_errors,'known_uncompleted_before_cutoff':not_completed,
                'contemporaneous_source_version_verified':False,'completion_provenance':'OFFICIAL_HISTORICAL_FIRST_PITCH_PLUS_DURATION_PLUS_60_SECONDS_DERIVED'}
        if not row['exclusions']:
            row.update(included=True,actual_winner='HOME' if hs>away_score else 'AWAY',home_direction='HOME',
                market_direction=direction(market,Decimal('0.5')),
                market_no_vig_home_probability=str(market),
                recent10_direction=direction(row['histories']['home']['wins'],row['histories']['away']['wins']))
        rows.append(row)
    admitted=[r for r in rows if r['included']]
    recent_subset=[r for r in admitted if r['recent10_direction'] is not None]
    common=[r for r in recent_subset if r['market_direction'] is not None]
    tied=[r for r in admitted if r['recent10_direction'] is None]
    views={}
    for name,group in [('all_included',admitted),('recent10_direction_subset',recent_subset),('common_three_direction_subset',common),('recent10_ties',tied)]:
        views[name]={'game_count':len(group),**{label:summarize(group,key) for label,key in [('recent10','recent10_direction'),('market_favorite','market_direction'),('always_home','home_direction')]}}
    discordance=Counter()
    for row in common:
        recent=row['recent10_direction']==row['actual_winner'];market_correct=row['market_direction']==row['actual_winner']
        discordance['both_correct' if recent and market_correct else 'recent_only_correct' if recent else 'market_only_correct' if market_correct else 'both_wrong']+=1
    by_date=[]
    for date in sorted({r['official_date'] for r in rows}):
        group=[r for r in admitted if r['official_date']==date]
        by_date.append({'official_date':date,'included':len(group),'recent10':summarize(group,'recent10_direction'),
                        'market_favorite':summarize(group,'market_direction'),'always_home':summarize(group,'home_direction')})
    used_history=[(row,game) for row in admitted for history in row['histories'].values() for game in history['games']]
    margins=[(parse(row['scheduled_start_utc'])-parse(game['derived_completion_upper_bound_utc'])).total_seconds()
             for row,game in used_history]
    return {'schema_version':'team_recent10_research.result.v1','protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
        'population_games':102,'uniquely_paired_with_usable_history_games':len(admitted),'excluded_games':102-len(admitted),
        'excluded_reason_counts':dict(sorted(Counter(reason for row in rows for reason in row['exclusions']).items())),
        'excluded_rows':[{'game_pk':r['game_pk'],'reasons':r['exclusions']} for r in rows if not r['included']],
        'recent10_direction_games':len(recent_subset),'recent10_tie_games':len(tied),
        'recent10_coverage_of_population':str(Decimal(len(recent_subset))/102),
        'recent10_coverage_of_included':str(Decimal(len(recent_subset))/len(admitted)) if admitted else None,
        'views':views,'paired_market_discordance_descriptive':dict(sorted(discordance.items())),
        'by_official_date':by_date,'rows':rows,'source_get_count':len(captures),
        'completion_source_plan_sha256':sha256((root/'completion_source_plan.json').read_bytes()).hexdigest(),
        'timing_audit':{'minimum_derived_completion_margin_seconds':min(margins) if margins else None,
            'same_official_date_history_occurrences':sum(row['official_date']==game['official_date'] for row,game in used_history),
            'contemporaneous_completion_source_versions_verified':0},
        'successful_source_responses':sum(c['status']=='SUCCESS' for c in captures),
        'source_response_bytes':sum(c.get('byte_count',0) for c in captures),
        'direction_rule_trial_count':1,'optimization_trials':0,'hypothesis_test_count':0,
        'probability_mapping':None,'brier':None,'log_loss':None,'roi':None,'bets_placed':0,
        'interpretation':'EXPLORATORY_ON_PREVIOUSLY_OPENED_OUTCOMES; NO_EFFICACY_OR_PROFITABILITY_CLAIM',
        'limitations':['Schedule, scores and timing statistics were retrieved after targets; no contemporaneous full-universe snapshot.',
            'Completion times are derived from official firstPitch and minute duration with a 60-second rounding margin; conditional on duration representing elapsed play duration, not exact last-out observations.',
            'Explicit suspended/resumed histories are rejected because original firstPitch plus duration can imply a false early completion.',
            'This independent direction baseline does not restore Champion features or change any ledger, model or risk policy.']}

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    print(json.dumps(evaluate(root),sort_keys=True,indent=2))
