"""Score with frozen coefficients only; never calls fit or changes an artifact."""
from pathlib import Path
import json
import sys
from research import score

root=Path(__file__).resolve().parent
legacy_root=Path(sys.argv[1]) if len(sys.argv)>1 else None
model=json.loads((root/'model_artifact.json').read_bytes())
print(json.dumps(score(root,model,legacy_root),sort_keys=True,indent=2))
