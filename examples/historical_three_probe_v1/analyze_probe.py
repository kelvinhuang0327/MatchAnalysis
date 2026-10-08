"""Offline, outcome-free replay of six fixed historical source probes."""
from datetime import UTC,datetime
from hashlib import sha256
from pathlib import Path
import json
import sys

def parse(value): return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(UTC)
def stamp(value): return datetime.strptime(value,'%Y%m%d_%H%M%S').replace(tzinfo=UTC)

def analyze(root):
 manifest=json.loads((root/'targets.json').read_bytes())
 plan=json.loads((root/'feed_plan.json').read_bytes())
 records=json.loads((root/'capture_records.json').read_bytes())
 if len(records)>manifest['maximum_free_gets'] or len(records)!=6:
  raise ValueError('SIX_GET_CONTRACT')
 if plan['target_manifest_sha256']!=sha256((root/'targets.json').read_bytes()).hexdigest():
  raise ValueError('FIXED_TARGETS_HASH_MISMATCH')
 for r in records:
  if r['status']!='SUCCESS': raise ValueError('INCOMPLETE_SOURCE_PROBE')
  if sha256((root/r['path']).read_bytes()).hexdigest()!=r['sha256']:
   raise ValueError('RAW_RESPONSE_HASH_MISMATCH')
 rows=[]
 for target,choice in zip(manifest['targets'],plan['plan']):
  pk=target['game_pk']
  if pk!=choice['game_pk']: raise ValueError('PLAN_IDENTITY_MISMATCH')
  data=json.loads((root/f'raw/{pk}-feed.json').read_bytes())
  versions=json.loads((root/f'raw/{pk}-timestamps.json').read_bytes())
  quote=parse(target['market_observed_at_utc'])
  gd=data['gameData']; actual=data['metaData']['timeStamp']; version=stamp(actual)
  game=gd['game']
  teams=gd['teams']
  identity_ok=(data['gamePk']==pk and game['pk']==pk and game['type']==target['game_type']
      and game['id'].rsplit('-',1)[-1]==str(target['game_number'])
      and gd['datetime']['officialDate']==target['official_date']
      and parse(gd['datetime']['dateTime'])==parse(target['scheduled_start_utc'])
      and all(teams[s]['id']==target[s+'_team']['id'] and teams[s]['name']==target[s+'_team']['name']
              and teams[s]['abbreviation']==target[s+'_team']['abbreviation'] for s in ['home','away']))
  probable=gd.get('probablePitchers',{})
  pitcher_ok=all(isinstance(probable.get(s,{}).get('id'),int) for s in ['home','away'])
  event_times=[parse(p['about'][k]) for p in data.get('liveData',{}).get('plays',{}).get('allPlays',[])
               for k in ['startTime','endTime'] if p.get('about',{}).get(k)]
  within_cutoff=version<=quote and all(t<=quote for t in event_times)
  pregame=gd['status']['abstractGameState']=='Preview'
  valid=identity_ok and pitcher_ok and pregame and within_cutoff and actual in versions
  season_fields={}
  for side,p in probable.items():
   player=data.get('liveData',{}).get('boxscore',{}).get('teams',{}).get(side,{}).get('players',{}).get('ID'+str(p['id']),{})
   season_fields[side]=sorted(player.get('seasonStats',{}).get('pitching',{}))
  rows.append({'game_pk':pk,'official_date':target['official_date'],'quote_at_utc':target['market_observed_at_utc'],
    'requested_timecode':choice['chosen_timecode'],'returned_timecode':actual,'earliest_available_timecode':versions[0],
    'listed_at_or_before_quote_count':choice['listed_at_or_before_quote_count'],'returned_minus_quote_seconds':(version-quote).total_seconds(),
    'identity_matches':identity_ok,'pregame_status':gd['status'],'probable_pitchers':probable,
    'returned_version_and_events_not_after_quote':within_cutoff,'starter_identity_recovery':'RECOVERABLE_OFFICIAL_PREQUOTE_VERSION' if valid else 'NOT_RECOVERABLE_AT_FIXED_QUOTE_FROM_AVAILABLE_VERSIONS',
    'season_pitching_fields_in_returned_version':season_fields,'season_stats_at_quote_recovery':'PARTIALLY_RECOVERABLE_VERSION_FIELDS' if valid else 'NOT_RECOVERABLE_AT_FIXED_QUOTE',
    'champion_exact_starts_only_era_recovery':'UNCERTAIN_NO_PRIOR_START_ONLY_LOG_OR_PROOF_ALL_APPEARANCES_ARE_STARTS' if valid else 'NOT_RECOVERABLE_AT_FIXED_QUOTE',
    'champion_recent_10_completed_games_recovery':'UNCERTAIN_NO_PREQUOTE_COMPLETION_VERSION_EVIDENCE_FOR_REQUIRED_HISTORY' if valid else 'NOT_RECOVERABLE_AT_FIXED_QUOTE',
    'overall_full_champion_input_status':'UNCERTAIN_PARTIAL_RECOVERY' if valid else 'NOT_RECOVERABLE_AT_FIXED_QUOTE',
    'historical_reconstruction_not_prospective':True,'first_announcement_time_established':False,
    'current_capture_is_not_historical_receipt':True,'method_scored':False,'new_hypothesis_proposed':False})
 return {'schema_version':'historical_three_probe.findings.v1','request_count':len(records),'target_manifest_sha256':plan['target_manifest_sha256'],
   'recoverable_prequote_starter_identity_count':sum(x['starter_identity_recovery'].startswith('RECOVERABLE') for x in rows),
   'verified_full_champion_input_count':0,'new_method_evaluation_count':0,'new_hypothesis_count':0,'rows':rows}

if __name__=='__main__':
 root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
 print(json.dumps(analyze(root),sort_keys=True,indent=2))
