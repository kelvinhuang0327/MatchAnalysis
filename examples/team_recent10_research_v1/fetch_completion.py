"""One bounded free official schedule projection; never runs during replay."""
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
plan=json.loads((root/sys.argv[1]).read_bytes())
path=root/sys.argv[2]
if path.exists():raise ValueError('NO_REPEAT_OR_OVERWRITE')
url=plan['url']
if not url.startswith('https://statsapi.mlb.com/api/v1/schedule?'):
    raise ValueError('FREE_OFFICIAL_SCHEDULE_ONLY')
def now():return datetime.now(UTC).isoformat().replace('+00:00','Z')
record={'url':url,'path':path.name,'request_started_at_utc':now(),'status':'FAILED',
    'transport':'SYSTEM_CURL_NORMAL_TLS','http_attempt_count':1}
try:
    response=subprocess.run(['/usr/bin/curl','--proto','=https','--fail','--silent','--show-error',
        '--connect-timeout','5','--max-time','20','--max-filesize','262144',url],
        capture_output=True,check=True,timeout=25)
    received=now();raw=response.stdout
    if len(raw)>262144:raise ValueError('256_KIB_LIMIT')
    data=json.loads(raw)
    with path.open('xb') as output:output.write(raw)
    record.update(status='SUCCESS',received_at_utc=received,persisted_at_utc=now(),
        sha256=sha256(raw).hexdigest(),byte_count=len(raw))
    print('games',sum(len(d['games']) for d in data['dates']),'bytes',len(raw))
    print('gameInfo examples',[g.get('gameInfo') for d in data['dates'] for g in d['games']][:3])
except Exception as error:
    record['error']=f'{type(error).__name__}: {error}'
    if isinstance(error,subprocess.CalledProcessError):record['stderr']=error.stderr.decode(errors='replace')[:2000]
records_path=root/'source_captures.json'
records=json.loads(records_path.read_bytes()) if records_path.exists() else []
records.append(record)
records_path.write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
