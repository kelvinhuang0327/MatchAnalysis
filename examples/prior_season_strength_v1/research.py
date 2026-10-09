"""Prior-year fixed run strength: one fit, score-only replay, no production writes."""
from collections import Counter, defaultdict
from datetime import date, datetime, UTC
from hashlib import sha256
from pathlib import Path
from decimal import Decimal
import csv
import io
import json
import math

def parse(value):return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(UTC)
def encoded(data):return (json.dumps(data,sort_keys=True,indent=2)+'\n').encode()
def digest(data):return sha256(encoded(data)).hexdigest()

def sources(root,legacy_root=None):
    policy=json.loads((root/'protocol.json').read_bytes())
    original_repo=Path(policy['sources']['eval_population']['path']).parents[2]
    original_legacy=Path(policy['sources']['train_csv']['path']).parent
    paths={}
    for key,source in policy['sources'].items():
        path=Path(source['path'])
        if path.is_relative_to(original_repo):path=root.parent.parent/path.relative_to(original_repo)
        elif legacy_root is not None and path.is_relative_to(original_legacy):path=Path(legacy_root)/path.relative_to(original_legacy)
        if sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('SOURCE_HASH_MISMATCH:'+key)
        paths[key]=path
    old=policy['old_candidate_source'];path=root.parent.parent/Path(old['path']).relative_to(original_repo)
    if sha256(path.read_bytes()).hexdigest()!=old['sha256'] or json.loads(path.read_bytes())['candidate_fit_count']!=0:
        raise ValueError('OLD_CANDIDATE_CHANGED_OR_FITTED')
    return policy,paths

def season(year,csv_path,raw_path,policy):
    with csv_path.open() as f:independent=list(csv.DictReader(f))
    with raw_path.open() as f:raw=list(csv.reader(f))
    names=policy['feature_definition']['retrosheet_code_names'];ids=policy['feature_definition']['identity_crosswalk']
    csv_tuples=[]
    for row in independent:
        hs=int(row['home_score']);away=int(row['away_score'])
        if hs<0 or away<0 or hs==away or Decimal(row['home_win'])!=int(hs>away):raise ValueError('INDEPENDENT_LABEL_INVALID')
        if row['home_team'] not in ids or row['away_team'] not in ids:raise ValueError('CSV_TEAM_IDENTITY_UNKNOWN')
        csv_tuples.append((row['date'],row['away_team'],row['home_team'],away,hs))
    totals=defaultdict(lambda:{'games':0,'runs_for':0,'runs_against':0,'ordinals':[]})
    games=[];raw_tuples=[];physical=set();late=[]
    for row in raw:
        if len(row)!=161:raise ValueError('RAW_161_FIELD_CONTRACT')
        key='|'.join(row[i] for i in [0,1,3,6])
        if key in physical:raise ValueError('DUPLICATE_PHYSICAL_GAME')
        physical.add(key)
        if row[3] not in names or row[6] not in names:raise ValueError('RAW_TEAM_IDENTITY_UNKNOWN')
        original=date.fromisoformat(f'{row[0][:4]}-{row[0][4:6]}-{row[0][6:]}')
        completed=row[13].split(',')[0] if row[13] else row[0]
        completion=date.fromisoformat(f'{completed[:4]}-{completed[4:6]}-{completed[6:]}')
        if original.year!=year or completion<original:raise ValueError('SOURCE_SEASON_OR_COMPLETION_DATE_INVALID')
        if row[14] or row[15]:raise ValueError('FORFEIT_OR_PROTEST_UNRESOLVED')
        away_name=names[row[3]];home_name=names[row[6]];away=int(row[9]);home=int(row[10])
        if home<0 or away<0 or home==away:raise ValueError('RAW_FINAL_SCORE_INVALID')
        away_id=ids[away_name];home_id=ids[home_name]
        for tid,rf,ra,ordinal in [(away_id,away,home,int(row[5])),(home_id,home,away,int(row[8]))]:
            totals[tid]['games']+=1;totals[tid]['runs_for']+=rf;totals[tid]['runs_against']+=ra;totals[tid]['ordinals'].append(ordinal)
        raw_tuples.append((original.isoformat(),away_name,home_name,away,home))
        games.append({'key':key,'official_date':original.isoformat(),'completion_date':completion.isoformat(),
            'home_id':home_id,'away_id':away_id,'home_score':home,'away_score':away,'y':int(home>away),
            'later_completion':bool(row[13])})
        if row[13]:late.append({'key':key,'original_date':original.isoformat(),'completion_date':completion.isoformat()})
    if Counter(csv_tuples)!=Counter(raw_tuples):raise ValueError('CSV_RAW_MULTISETS_DISAGREE')
    if set(totals)!=set(ids.values()) or len(totals)!=30:raise ValueError('FULL_THIRTY_TEAM_ANNUAL_SOURCE_REQUIRED')
    aggregate={}
    for tid,t in totals.items():
        if sorted(t['ordinals'])!=list(range(1,t['games']+1)):raise ValueError('TEAM_SEASON_ORDINAL_GAP_OR_DUPLICATE')
        aggregate[tid]={'team_id':tid,'name':next(name for name,team_id in ids.items() if team_id==tid),
            'games':t['games'],'runs_for':t['runs_for'],'runs_against':t['runs_against'],
            'runs_for_per_game':t['runs_for']/t['games'],'runs_against_per_game':t['runs_against']/t['games'],
            'strength':(t['runs_for']-t['runs_against'])/t['games']}
    if sum(t['runs_for'] for t in aggregate.values())!=sum(t['runs_against'] for t in aggregate.values()):
        raise ValueError('ANNUAL_RUN_CONSERVATION_FAILED')
    info={'season':year,'games':len(games),'teams':30,'first_game_date':min(g['official_date'] for g in games),
        'last_original_date':max(g['official_date'] for g in games),'last_completion_date':max(g['completion_date'] for g in games),
        'resolved_later_completion_games':late,'csv_raw_score_label_multiset_matches':True,
        'all_team_game_ordinals_contiguous':True,'no_contemporaneous_capture_claim':True,
        'aggregates':[aggregate[tid] for tid in sorted(aggregate)]}
    return aggregate,sorted(games,key=lambda g:g['key']),info

def annual_chronology(last_completed,next_first):
    if date.fromisoformat(last_completed)>=date.fromisoformat(next_first):raise ValueError('ANNUAL_SOURCE_NOT_FINISHED_BEFORE_NEXT_SEASON')
    return (date.fromisoformat(next_first)-date.fromisoformat(last_completed)).days

def prepare(root,legacy_root=None):
    policy,paths=sources(root,legacy_root)
    prior,games2024,info2024=season(2024,paths['earlier_csv'],paths['earlier_gamelog'],policy)
    newer,games2025,info2025=season(2025,paths['train_csv'],paths['train_gamelog'],policy)
    schedule=[g for day in json.loads(paths['eval_schedule'].read_bytes())['dates'] for g in day['games'] if g['gameType']=='R']
    first2026=min(g['officialDate'] for g in schedule)
    chronological={'2024_to_2025_gap_days':annual_chronology(info2024['last_completion_date'],info2025['first_game_date']),
        '2025_to_2026_gap_days':annual_chronology(info2025['last_completion_date'],first2026),
        'first_2026_R_date':first2026,'requires_per_game_final_utc':False,'source_version_claim':'CURRENT_HASH_BOUND_RECONSTRUCTION_NOT_CONTEMPORANEOUS'}
    training=[];train_exclusions=[]
    for g in games2025:
        if g['home_id'] not in prior or g['away_id'] not in prior:
            train_exclusions.append({'key':g['key'],'reason':'PRIOR_ANNUAL_TEAM_MISSING'});continue
        training.append({'key':g['key'],'home_id':g['home_id'],'away_id':g['away_id'],
            'x':prior[g['home_id']]['strength']-prior[g['away_id']]['strength'],'y':g['y']})
    population=[json.loads(line) for line in paths['eval_population'].read_text().splitlines()]
    outcomes=[json.loads(line) for line in paths['eval_opened_outcomes'].read_text().splitlines()]
    timing=[g for day in json.loads(paths['eval_game_info'].read_bytes())['dates'] for g in day['games']]
    if len(population)!=102 or len({g['game_pk'] for g in population})!=102:raise ValueError('UNIQUE_FIXED_102_POPULATION_REQUIRED')
    def match(g,t):return (g['gamePk']==t['game_pk'] and g['gameType']==t['game_type'] and g['officialDate']==t['official_date']
        and g['gameNumber']==t['game_number'] and g['gameDate']==t['scheduled_start_utc']
        and all(g['teams'][s]['team']['id']==t[s+'_team']['id'] for s in ['home','away']))
    evaluated=[]
    for t in population:
        pk=t['game_pk'];errors=[]
        official={json.dumps(g,sort_keys=True):g for g in schedule if match(g,t)}
        actual={json.dumps(g,sort_keys=True):g for g in timing if match(g,t)}
        labels=[g for g in outcomes if int(g['provider_game_id'])==pk and g['official_date']==t['official_date'] and g['game_number']==t['game_number']]
        row={'game_pk':pk,'official_date':t['official_date'],'home_id':t['home_team']['id'],'away_id':t['away_team']['id'],
            'market_observed_at_utc':t['market_observed_at_utc'],'quote_raw_record_sha256':t['raw_record_sha256'],
            'included':False,'exclusions':errors,'x':None,'y':None,'market_p_home':None}
        if len(official)!=1 or len(actual)!=1 or len(labels)!=1:errors.append('FULL_IDENTITY_OR_OUTCOME_NOT_UNIQUE')
        if not errors:
            g=next(iter(official.values()));actual_g=next(iter(actual.values()))
            hs=g['teams']['home'].get('score');away=g['teams']['away'].get('score')
            if g['status']['abstractGameState']!='Final' or hs is None or away is None or hs==away or int(hs>away)!=labels[0]['actual_home_win']:
                errors.append('FINAL_SCORE_LABEL_INVALID_OR_CONFLICTING')
            first=actual_g.get('gameInfo',{}).get('firstPitch')
            if not first or parse(t['market_observed_at_utc'])>=min(parse(t['scheduled_start_utc']),parse(t['source_game_time_utc']),parse(first)):
                errors.append('QUOTE_NOT_BEFORE_ORIGINAL_START')
            hp=Decimal(t['home_decimal_price']);ap=Decimal(t['away_decimal_price']);market=ap/(hp+ap)
            if hp<=1 or ap<=1 or abs(market-Decimal(t['market_no_vig_home_probability']))>Decimal('1e-25'):
                errors.append('TWO_SIDED_NO_VIG_PRICE_INVALID')
            if row['home_id'] not in newer or row['away_id'] not in newer:errors.append('PRIOR_ANNUAL_TEAM_MISSING')
            if not errors:
                row.update(included=True,x=newer[row['home_id']]['strength']-newer[row['away_id']]['strength'],
                    y=int(hs>away),market_p_home=float(market),quote_age_at_scheduled_start_seconds=(parse(t['scheduled_start_utc'])-parse(t['market_observed_at_utc'])).total_seconds())
        evaluated.append(row)
    provenance={'annual_chronology':chronological,'annual_sources':[info2024,info2025],
        'source_metadata':{key:json.loads(paths[key].read_bytes()) for key in ['earlier_metadata','train_metadata']},
        'training_population':2430,'training_admitted':len(training),'training_exclusions':train_exclusions,
        'evaluation_population':102,'evaluation_admitted':sum(row['included'] for row in evaluated),
        'previously_used_development_only':True}
    return policy,training,evaluated,provenance

def standardize(training):
    n=len(training)
    if not n or not 0<sum(r['y'] for r in training)<n:raise ValueError('TRAINING_BOTH_CLASSES_REQUIRED')
    mean=math.fsum(r['x'] for r in training)/n
    variance=math.fsum((r['x']-mean)**2 for r in training)/n
    scale=math.sqrt(variance) if variance>0 else 1.0
    return mean,scale,[((r['x']-mean)/scale,r['y']) for r in training]

def sigmoid(z):
    if z>=0:return 1/(1+math.exp(-z))
    e=math.exp(z);return e/(1+e)

def loss_grad_hessian(parameters,examples):
    intercept,beta=parameters;n=len(examples);loss=[];g0=[];g1=[];h00=[];h01=[];h11=[]
    for x,y in examples:
        z=intercept+beta*x;p=sigmoid(z);w=p*(1-p)
        loss.append(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
        g0.append(p-y);g1.append((p-y)*x);h00.append(w);h01.append(w*x);h11.append(w*x*x)
    return (math.fsum(loss)/n+0.5*beta*beta,
        (math.fsum(g0)/n,math.fsum(g1)/n+beta),
        (math.fsum(h00)/n,math.fsum(h01)/n,math.fsum(h11)/n+1))

def fit(examples):
    parameters=(0.0,0.0)
    for iteration in range(101):
        loss,gradient,hessian=loss_grad_hessian(parameters,examples)
        if max(abs(g) for g in gradient)<=1e-8:
            return {'intercept':parameters[0],'beta_strength':parameters[1],'iterations':iteration,
                'penalized_mean_loss':loss,'gradient_infinity_norm':max(abs(g) for g in gradient),'converged':True}
        if iteration==100:break
        h00,h01,h11=hessian;det=h00*h11-h01*h01
        if det<=0:raise ValueError('NEWTON_HESSIAN_NOT_POSITIVE_DEFINITE')
        d0=(h11*gradient[0]-h01*gradient[1])/det;d1=(-h01*gradient[0]+h00*gradient[1])/det
        for backtrack in range(20):
            step=2.0**(-backtrack);trial=(parameters[0]-step*d0,parameters[1]-step*d1)
            new_loss,_,_=loss_grad_hessian(trial,examples)
            if new_loss<=loss-1e-4*step*(gradient[0]*d0+gradient[1]*d1):parameters=trial;break
        else:raise ValueError('NEWTON_BACKTRACKING_FAILED')
    raise ValueError('FIXED_NEWTON_NOT_CONVERGED')

def metric(rows,key):
    selected=[r for r in rows if r['included']];n=len(selected)
    if not n:return {'sample_count':0,'correct_count':0,'accuracy':None,'brier':None,'logloss':None}
    correct=sum((r[key]>=0.5)==bool(r['y']) for r in selected)
    return {'sample_count':n,'correct_count':correct,'accuracy':correct/n,
        'brier':math.fsum((r[key]-r['y'])**2 for r in selected)/n,
        'logloss':-math.fsum(math.log(r[key]) if r['y'] else math.log1p(-r[key]) for r in selected)/n}

def predict(x,model):
    standardized=(x-model['standardization']['mean'])/model['standardization']['scale']
    return sigmoid(model['fit']['intercept']+model['fit']['beta_strength']*standardized)

def score(root,model,legacy_root=None):
    policy,training,rows,provenance=prepare(root,legacy_root)
    if model['protocol_sha256']!=sha256((root/'protocol.json').read_bytes()).hexdigest() or model['training_rows_sha256']!=digest(training):
        raise ValueError('MODEL_PROTOCOL_OR_TRAINING_INPUT_MISMATCH')
    mean,scale,_=standardize(training)
    if model['standardization']!={'mean':mean,'scale':scale}:raise ValueError('TRAINING_ONLY_SCALER_MISMATCH')
    for row in rows:
        if row['included']:
            row.update(candidate_p_home=predict(row['x'],model),fixed_home_p_home=0.5)
            row['candidate_direction']='HOME' if row['candidate_p_home']>=0.5 else 'AWAY'
            row['market_direction']='HOME' if row['market_p_home']>=0.5 else 'AWAY'
            row['actual_winner']='HOME' if row['y'] else 'AWAY'
    metrics={name:metric(rows,key) for name,key in [('candidate','candidate_p_home'),('market','market_p_home'),('fixed_home_0_5','fixed_home_p_home')]}
    candidate=metrics['candidate'];market=metrics['market'];n=candidate['sample_count']
    differences={key:candidate[key]-market[key] if n else None for key in ['accuracy','brier','logloss']}
    target_met=(candidate['accuracy']>market['accuracy'] and candidate['brier']<=market['brier'] and candidate['logloss']<=market['logloss']) if n else None
    discordance=Counter()
    for row in rows:
        if not row['included']:continue
        c=row['candidate_direction']==row['actual_winner'];m=row['market_direction']==row['actual_winner']
        discordance['both_correct' if c and m else 'candidate_only_correct' if c else 'market_only_correct' if m else 'both_wrong']+=1
    return {'schema_version':'prior_season_strength.result.v1','candidate_id':policy['candidate_id'],
        'protocol_sha256':model['protocol_sha256'],'model_artifact_sha256':sha256((root/'model_artifact.json').read_bytes()).hexdigest(),
        'real_fit_count':1,'new_fits_during_score_or_replay':0,'optimization_trials':0,'new_source_gets':0,
        'training_population':2430,'training_admitted':len(training),'training_excluded':2430-len(training),
        'evaluation_population':102,'paired_model_market_games':n,'evaluation_excluded':102-n,'coverage':n/102,
        'metrics':metrics,'candidate_minus_market':differences,'descriptive_user_priority_met_on_opened_batch':target_met,
        'probability_quality_checks':{'Brier_not_worse':candidate['brier']<=market['brier'] if n else None,
            'logloss_not_worse':candidate['logloss']<=market['logloss'] if n else None},
        'paired_accuracy_discordance':dict(sorted(discordance.items())),'source_audit':provenance,'rows':rows,
        'candidate_direction_counts':dict(sorted(Counter(row['candidate_direction'] for row in rows if row['included']).items())),
        'all_candidate_directions_equal_fixed_home':all(row['candidate_direction']=='HOME' for row in rows if row['included']) if n else None,
        'interpretation':'HISTORICAL_RECONSTRUCTION_DEVELOPMENT_ALREADY_OPENED; NOT_HOLDOUT_OR_PROSPECTIVE',
        'improvement_established':False,'profitability_claim':False,'bets_placed':0,
        'old_season_to_date_candidate_fit_count':0,
        'limits':['Full-year current snapshots can include later revisions and normalized identities; no claim these exact versions were available at forecast time.',
            '2024/2025 previously used in legacy research; 2026 102 outcomes already opened. This single fixed trial cannot establish out-of-sample improvement.',
            'Previous-year strength ignores offseason roster changes, current-season changes and neutral-venue effects.',
            'Market snapshots are fixed archived pregame references, without live decision/freshness or closing-price claims.']}

def table_csv(result):
    out=io.StringIO(newline='');fields=['game_pk','official_date','included','home_id','away_id','x',
        'candidate_p_home','market_p_home','fixed_home_p_home','candidate_direction','market_direction','actual_winner','exclusions']
    writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in result['rows']:
        cells={key:row.get(key) for key in fields};cells['exclusions']='|'.join(row['exclusions']);writer.writerow(cells)
    return out.getvalue()
