"""Saved-model replay, annual conservation and fixed objective; never a second fit."""
from pathlib import Path
from hashlib import sha256
import json
import sys
from research import prepare,standardize,score,encoded,table_csv,loss_grad_hessian

root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_bytes())
for name,expected in manifest['files'].items():
    if sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('CAPSULE_HASH_MISMATCH:'+name)
legacy_root=Path(sys.argv[1]) if len(sys.argv)>1 else None
model=json.loads((root/'model_artifact.json').read_bytes())
result=score(root,model,legacy_root)
if encoded(result)!=(root/'results.json').read_bytes():raise ValueError('JSON_REPLAY_NOT_IDENTICAL')
if table_csv(result).encode()!=(root/'per_game.csv').read_bytes():raise ValueError('CSV_REPLAY_NOT_IDENTICAL')
_,training,_,provenance=prepare(root,legacy_root)
if encoded(provenance)!=(root/'annual_source_audit.json').read_bytes():raise ValueError('ANNUAL_AUDIT_REPLAY_NOT_IDENTICAL')
_,_,examples=standardize(training)
loss,gradient,_=loss_grad_hessian((model['fit']['intercept'],model['fit']['beta_strength']),examples)
assert max(abs(g) for g in gradient)<=1e-8
assert loss==model['fit']['penalized_mean_loss']
assert result['training_admitted']+result['training_excluded']==2430
assert result['paired_model_market_games']+result['evaluation_excluded']==102
assert {m['sample_count'] for m in result['metrics'].values()}=={result['paired_model_market_games']}
assert model['real_fit_count']==result['real_fit_count']==1
assert result['new_fits_during_score_or_replay']==result['new_source_gets']==result['optimization_trials']==0
assert result['old_season_to_date_candidate_fit_count']==0
print(json.dumps({'capsule_hashes_verified':len(manifest['files']),'json_byte_identical':True,
    'csv_byte_identical':True,'annual_audit_byte_identical':True,'fixed_objective_gradient_verified':True,
    'training_admitted':len(training),'paired_games':result['paired_model_market_games'],
    'real_fits_total':1,'new_fits_during_replay':0,'old_candidate_fits':0},sort_keys=True))
