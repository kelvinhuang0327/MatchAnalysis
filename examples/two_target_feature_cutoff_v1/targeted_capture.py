"""Execute one fixed, bounded public official source plan, with normal TLS."""
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys
from urllib.parse import urlparse

root = Path(__file__).resolve().parent
plan = json.loads((root / sys.argv[1]).read_bytes())
records = json.loads((root / 'captures.json').read_bytes())
if len(plan['requests']) > plan['maximum_gets']:
    raise ValueError('FIXED_PLAN_LIMIT')
if any(r['path'] in {x['path'] for x in records} for r in plan['requests']):
    raise ValueError('NO_REPEAT_REQUESTS')
def now(): return datetime.now(UTC).isoformat().replace('+00:00', 'Z')
blocked = set(plan.get('blocked_features', []))
for request in plan['requests']:
    feature_keys = set(request.get('feature_keys', []))
    if feature_keys and feature_keys <= blocked:
        continue
    url = request['url']
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.netloc != 'statsapi.mlb.com':
        raise ValueError('PUBLIC_OFFICIAL_HTTPS_ONLY')
    record = dict(request, status='FAILED', request_started_at_utc=now(), http_attempt_count=1,
                  transport='SYSTEM_CURL_NORMAL_TLS')
    try:
        response = subprocess.run(['/usr/bin/curl', '--proto', '=https', '--fail', '--silent',
            '--show-error', '--connect-timeout', '5', '--max-time', '20', '--max-filesize',
            '262144', url], capture_output=True, check=True, timeout=25)
        received = now()
        raw = response.stdout
        if len(raw) > 262144: raise ValueError('256_KIB_LIMIT')
        data = json.loads(raw)
        path = root / request['path']
        path.parent.mkdir(exist_ok=True)
        with path.open('xb') as output: output.write(raw)
        record.update(status='SUCCESS', received_at_utc=received, persisted_at_utc=now(),
                      byte_count=len(raw), sha256=sha256(raw).hexdigest())
        print(request['path'], len(raw), data.get('metaData', {}) if isinstance(data, dict) else 'VERSION_LIST', flush=True)
    except subprocess.CalledProcessError as error:
        record.update(error=str(error), exit_status=error.returncode,
                      stderr=error.stderr.decode(errors='replace')[:2000])
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    records.append(record)
    (root / 'captures.json').write_text(json.dumps(records, sort_keys=True, indent=2)+'\n')
    if plan.get('stop_on_first_failure') and record['status'] != 'SUCCESS':
        break
    if request.get('quote_cutoff_utc') and record['status'] == 'SUCCESS':
        version = datetime.strptime(data['metaData']['timeStamp'], '%Y%m%d_%H%M%S').replace(tzinfo=UTC)
        cutoff = datetime.fromisoformat(request['quote_cutoff_utc'].replace('Z', '+00:00'))
        if version > cutoff or data['gameData']['status']['abstractGameState'] != 'Final':
            print('STOP: REQUIRED_FINAL_VERSION_NOT_AVAILABLE_BEFORE_CUTOFF', flush=True)
            if plan.get('independent_feature_gates'):
                blocked.update(feature_keys)
                (root / 'blocked_features.json').write_text(json.dumps(sorted(blocked), indent=2)+'\n')
            else:
                break
