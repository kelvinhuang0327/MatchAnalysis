"""Exclusive single real fit; saved-model scoring is the only replay path."""
from pathlib import Path
from hashlib import sha256
import json
import sys
from research import prepare,standardize,fit,encoded,digest,score,table_csv

root=Path(__file__).resolve().parent
model_path=root/'model_artifact.json'
if model_path.exists():raise ValueError('REAL_FIT_ALREADY_FROZEN_USE_SAVED_MODEL_REPLAY')
legacy_root=Path(sys.argv[1]) if len(sys.argv)>1 else None
policy,training,rows,provenance=prepare(root,legacy_root)
mean,scale,examples=standardize(training)
parameters=fit(examples)
model={'schema_version':'prior_season_strength.model.v1','candidate_id':policy['candidate_id'],
    'protocol_sha256':sha256((root/'protocol.json').read_bytes()).hexdigest(),
    'training_rows_sha256':digest(training),'training_population':2430,'training_admitted':len(training),
    'standardization':{'mean':mean,'scale':scale},'fit':parameters,'l2_lambda':1,'intercept_penalized':False,
    'real_fit_count':1,'features':['previous_season_net_runs_per_game_home_minus_away'],
    'training_feature_season':2024,'training_label_season':2025,'evaluation_feature_season':2025,
    'annual_feature_update_rule':'Fixed for whole following season; never current-season updates.'}
with model_path.open('xb') as stream:stream.write(encoded(model))
(root/'annual_source_audit.json').write_bytes(encoded(provenance))
result=score(root,model,legacy_root)
(root/'results.json').write_bytes(encoded(result))
(root/'per_game.csv').write_text(table_csv(result))
print(json.dumps({'real_fit_count':1,'training_admitted':len(training),'converged':parameters['converged'],
    'iterations':parameters['iterations'],'paired_games':result['paired_model_market_games'],
    'metrics':result['metrics'],'priority_met_descriptive_only':result['descriptive_user_priority_met_on_opened_batch']},sort_keys=True))
