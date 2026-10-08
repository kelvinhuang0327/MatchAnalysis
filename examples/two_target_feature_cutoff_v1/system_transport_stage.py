from datetime import UTC,datetime
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('/tmp/match-two-feature-cutoff-20261008')
records=json.loads((ROOT/'captures.json').read_bytes())
fixed=[dict(r) for r in records]
if len(fixed)!=2 or not all(r['status']=='FAILED' and 'CERTIFICATE_VERIFY_FAILED' in r.get('error','') for r in fixed):
 raise ValueError('EXACT_TWO_TLS_FAILURES_REQUIRED')
plan={'planned_at_utc':datetime.now(UTC).isoformat().replace('+00:00','Z'),'extra_get_limit':2,'reason':'Existing system curl with normal TLS validation; no insecure flag, trust edits or HTTP fallback.','fixed_urls':[r['url'] for r in fixed]}
(ROOT/'system_transport_plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
for old in fixed:
 r={'url':old['url'],'path':old['path'].replace('.json','-system.json'),'status':'FAILED','request_started_at_utc':datetime.now(UTC).isoformat().replace('+00:00','Z'),'http_attempt_count':1,'transport':'SYSTEM_CURL_NORMAL_TLS'}
 try:
  res=subprocess.run(['/usr/bin/curl','--proto','=https','--fail','--silent','--show-error','--connect-timeout','5','--max-time','20','--max-filesize','262144',r['url']],check=True,capture_output=True,timeout=25)
  received=datetime.now(UTC).isoformat().replace('+00:00','Z')
  raw=res.stdout
  if len(raw)>262144: raise ValueError('256_KIB_LIMIT')
  d=json.loads(raw)
  p=ROOT/r['path']; p.parent.mkdir(exist_ok=True)
  with p.open('xb') as out: out.write(raw)
  r.update(status='SUCCESS',received_at_utc=received,persisted_at_utc=datetime.now(UTC).isoformat().replace('+00:00','Z'),sha256=sha256(raw).hexdigest(),byte_count=len(raw))
  gd=d['gameData']
  print('game',d['gamePk'],'version',d['metaData'],'records',{s:v.get('record') for s,v in gd['teams'].items()},flush=True)
  for side,pitcher in gd['probablePitchers'].items():
   p=d['liveData']['boxscore']['teams'][side]['players'].get('ID'+str(pitcher['id']),{})
   print(side,pitcher,'seasonStats',p.get('seasonStats',{}).get('pitching'),'currentGameStats',p.get('stats',{}).get('pitching'),flush=True)
 except subprocess.CalledProcessError as e:
  r.update(error=f'{type(e).__name__}: {e}',exit_status=e.returncode,stderr=e.stderr.decode(errors='replace')[:2000])
  print('FAILED',r['stderr'],flush=True)
 except Exception as e:
  r['error']=f'{type(e).__name__}: {e}'
 records.append(r)
 (ROOT/'captures.json').write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
