from datetime import UTC,datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import json

ROOT=Path('/tmp/match-historical-three-probe-20261008')
CAPS=ROOT/'capture_records.json'

def now(): return datetime.now(UTC).isoformat().replace('+00:00','Z')

def fetch(url,relative):
 records=json.loads(CAPS.read_bytes()) if CAPS.exists() else []
 if len(records)>=6: raise ValueError('SIX_GET_LIMIT')
 if not url.startswith('https://statsapi.mlb.com/api/v1.1/game/'): raise ValueError('SOURCE_NOT_ALLOWED')
 rec={'url':url,'method':'GET','request_started_at_utc':now(),'status':'FAILED','attempt':len(records)+1,'provider_historical_time_is_not_current_receipt':True}
 raw=None
 try:
  with urlopen(Request(url,headers={'Accept':'application/json','User-Agent':'MatchAnalysis-bounded-history-probe/1.0'}),timeout=20) as res:
   raw=res.read(1024*1024+1)
   rec['http_status']=res.status
  received=now()
  if len(raw)>1024*1024: raise ValueError('ONE_MIB_LIMIT')
  json.loads(raw)
  rec['status']='SUCCESS'
 except HTTPError as exc:
  rec['http_status']=exc.code
  raw=exc.read(1024*1024)
  received=now()
  rec['error']=f'{type(exc).__name__}: {exc}'
 except Exception as exc:
  received=now()
  rec['error']=f'{type(exc).__name__}: {exc}'
 if raw is not None:
  p=ROOT/relative
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as out: out.write(raw)
  rec.update(received_at_utc=received,persisted_at_utc=now(),path=relative,byte_count=len(raw),sha256=sha256(raw).hexdigest())
 records.append(rec)
 with CAPS.open('w') as out: out.write(json.dumps(records,sort_keys=True,indent=2)+'\n')
 return json.loads(raw) if rec['status']=='SUCCESS' else None

if __name__=='__main__':
 targets=json.loads((ROOT/'targets.json').read_bytes())['targets']
 for t in targets:
  pk=t['game_pk']
  data=fetch(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live/timestamps',f'raw/{pk}-timestamps.json')
  if isinstance(data,list):
   print(pk,'timestamp entries',len(data),'first',data[:3],'last',data[-1:])
  else: print(pk,'timestamp response',str(data)[:150])
