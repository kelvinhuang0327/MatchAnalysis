from datetime import UTC,datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode
import json
from fetch_probe import ROOT,fetch

manifest=json.loads((ROOT/'targets.json').read_bytes())
plan=[]
for target in manifest['targets']:
 pk=target['game_pk']
 stamps=json.loads((ROOT/f'raw/{pk}-timestamps.json').read_bytes())
 quote=datetime.fromisoformat(target['market_observed_at_utc'].replace('Z','+00:00'))
 valid=[]
 for stamp in stamps:
  try:
   parsed=datetime.strptime(stamp,'%Y%m%d_%H%M%S').replace(tzinfo=UTC)
  except (ValueError,TypeError): continue
  if parsed<=quote: valid.append((parsed,stamp))
 chosen=max(valid)[1] if valid else quote.strftime('%Y%m%d_%H%M%S')
 plan.append({'game_pk':pk,'quote_at_utc':target['market_observed_at_utc'],'listed_at_or_before_quote_count':len(valid),'chosen_timecode':chosen,'choice':'LATEST_LISTED_NOT_AFTER_QUOTE' if valid else 'EXPLICIT_QUOTE_TIMECODE_NO_LISTED_EARLIER_VERSION','earliest_listed_timecode':stamps[0] if stamps else None})
raw=(json.dumps({'target_manifest_sha256':sha256((ROOT/'targets.json').read_bytes()).hexdigest(),'plan':plan},sort_keys=True,indent=2)+'\n').encode()
with (ROOT/'feed_plan.json').open('xb') as out: out.write(raw)
print('Fixed feed plan:',raw.decode(),flush=True)
fields='gamePk,metaData,timeStamp,gameData,game,pk,type,season,datetime,dateTime,officialDate,status,abstractGameState,detailedState,teams,home,away,id,name,abbreviation,probablePitchers,fullName,liveData,boxscore,players,person,seasonStats,pitching,gamesStarted,earnedRuns,outs,inningsPitched,era,pitchers,plays,allPlays,about,startTime,endTime'
for row in plan:
 pk=row['game_pk']
 url=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?'+urlencode({'timecode':row['chosen_timecode'],'fields':fields})
 data=fetch(url,f'raw/{pk}-feed.json')
 if isinstance(data,dict):
  gd=data.get('gameData',{})
  box=(data.get('liveData',{}).get('boxscore') or {}).get('teams',{})
  print(json.dumps({'game_pk':pk,'metaData':data.get('metaData'),'status':gd.get('status'),'probablePitchers':gd.get('probablePitchers'),'game':gd.get('game'),'datetime':gd.get('datetime'),'boxscore_team_keys':{s:list(v) for s,v in box.items()},'allPlays_count':len(data.get('liveData',{}).get('plays',{}).get('allPlays',[]))},indent=2),flush=True)
 else: print(pk,'feed unavailable',flush=True)
