"""Read-only capsule hashes, deterministic replay and existing P52A feature equivalence."""
from pathlib import Path
from hashlib import sha256
from tempfile import TemporaryDirectory
import json
import shutil
import sys
from decimal import Decimal
from analyze_features import analyze, parse

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    if sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError('CAPSULE_HASH_MISMATCH:'+name)
result = analyze(root)
rendered = (json.dumps(result, sort_keys=True, indent=2)+'\n').encode()
if rendered != (root / 'findings.json').read_bytes():
    raise ValueError('BYTE_IDENTICAL_REPLAY_FAILED')
sys.path.insert(0, str(root.parent.parent / 'src'))
from match_analysis.application.use_cases.p52a_daily_prediction_freeze import compute_starter_era, compute_team_recent_win_rate_delta
checks = 0
plan = json.loads((root / 'plan.json').read_bytes())
for row in result['rows']:
    ident = next(t['identity'] for t in plan['targets'] if t['identity']['game_pk'] == row['game_pk'])
    for p in row['pitching'].values():
        if p['starts_only_era'] is None: continue
        logs = [{**x, 'game_type': 'R'} for x in p['prior_appearances']]
        value, error = compute_starter_era(p['pitcher_id'], ident['official_date'], {p['pitcher_id']: logs})
        if error is not None or value != Decimal(p['starts_only_era']):
            raise ValueError('EXISTING_P52A_STARTS_ONLY_ERA_MISMATCH')
        checks += 1
    if row['recent_10_home_minus_away_candidate'] is not None:
        rows = []
        for side, team in row['recent_10'].items():
            for g in team['games']:
                source = next(s for s in json.loads((root/'required_history_plan.json').read_bytes())['required_sources'] if s['game_pk']==g['game_pk'])
                raw = json.loads((root/source['path']).read_bytes())
                rows.append({**g, 'game_type':'R', 'final':True,
                    'home_team':{'id':raw['gameData']['teams']['home']['id']},
                    'away_team':{'id':raw['gameData']['teams']['away']['id']}})
        unique = {r['game_pk']:r for r in rows}
        value = compute_team_recent_win_rate_delta(ident['home_team']['id'],ident['away_team']['id'],list(unique.values()),
            parse(ident['market_observed_at_utc']),parse(ident['scheduled_start_utc']))
        if value != Decimal(row['recent_10_home_minus_away_candidate']):
            raise ValueError('EXISTING_P52A_RECENT10_MISMATCH')
        checks += 1
# A captured Live response cannot become a completed-game feature.
if any(x['game_pk']==824269 and x['recent_10']['away']['rate'] is not None for x in result['rows']):
    raise ValueError('LIVE_SOURCE_ADMITTED_AS_FINAL')
# Corrupted raw is rejected before computation, using only an isolated synthetic copy.
with TemporaryDirectory(prefix='match-feature-guard-') as tmp:
    scratch = Path(tmp)/'capsule'
    shutil.copytree(root,scratch,ignore=shutil.ignore_patterns('__pycache__'))
    records=json.loads((scratch/'captures.json').read_bytes())
    path=next(x['path'] for x in records if x['status']=='SUCCESS')
    (scratch/path).write_bytes((scratch/path).read_bytes()+b' ')
    try: analyze(scratch)
    except ValueError as error:
        if str(error)!='RAW_HASH_MISMATCH':raise
    else:raise ValueError('TAMPERED_RAW_ACCEPTED')
    (scratch/path).write_bytes((root/path).read_bytes())
    source=next(s for s in json.loads((scratch/'required_history_plan.json').read_bytes())['required_sources']
                if s['game_pk']==823243)
    historical=json.loads((scratch/source['path']).read_bytes())
    historical['metaData']['timeStamp']='20260606_000000'
    altered=(json.dumps(historical,sort_keys=True)+'\n').encode()
    (scratch/source['path']).write_bytes(altered)
    for record in records:
        if record['path']==source['path']:
            record['sha256']=sha256(altered).hexdigest()
            record['byte_count']=len(altered)
    (scratch/'captures.json').write_text(json.dumps(records))
    rejected=analyze(scratch)
    ray=next(row for row in rejected['rows'] if row['game_pk']==824673)['pitching']['away']
    if ray['starts_only_era'] is not None or 823243 not in ray['required_missing_game_pks']:
        raise ValueError('POSTQUOTE_VERSION_ACCEPTED')
print(json.dumps({'source_manifest_hashes_verified':len(manifest['files']), 'byte_identical_replay':True,
    'existing_P52A_feature_equivalence_checks':checks, 'live_final_rejection':True,
    'isolated_raw_tamper_rejection':True, 'isolated_postquote_version_rejection':True,
    'models_scored':0}, sort_keys=True))
