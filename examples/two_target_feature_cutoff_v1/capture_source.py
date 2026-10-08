from datetime import UTC,datetime
from hashlib import sha256
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import json
ROOT=Path('/tmp/match-two-feature-cutoff-20261008')

def now(): return datetime.now(UTC).isoformat().replace('+00:00','Z')

def fetch(url,path,limit):
 record_file=ROOT/'captures.json'
 records=json.loads(record_file.read_bytes()) if record_file.exists() else []
 if len(records)>=limit: raise ValueError('FIXED_GET_LIMIT')
 if not url.startswith('https://statsapi.mlb.com/api/v1.1/game/'): raise ValueError('NON_OFFICIAL_URL')
 r={'url':url,'path':path,'request_started_at_utc':now(),'status':'FAILED','http_attempt_count':1}
 raw=None
 try:
  with urlopen(Request(url,headers={'Accept':'application/json','User-Agent':'MatchAnalysis-feature-cutoff/1.0'}),timeout=20) as res:
   raw=res.read(256*1024+1); r['http_status']=res.status
  received=now()
  if len(raw)>256*1024: raise ValueError('256_KIB_LIMIT')
  json.loads(raw); r['status']='SUCCESS'
 except HTTPError as e:
  raw=e.read(256*1024); received=now(); r.update(http_status=e.code,error=f'{type(e).__name__}: {e}')
 except Exception as e:
  received=now(); r['error']=f'{type(e).__name__}: {e}'
 if raw is not None:
  p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as out: out.write(raw)
  r.update(received_at_utc=received,persisted_at_utc=now(),sha256=sha256(raw).hexdigest(),byte_count=len(raw))
 records.append(r); record_file.write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
 return json.loads(raw) if r['status']=='SUCCESS' else None
