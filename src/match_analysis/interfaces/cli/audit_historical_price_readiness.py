"""Replay a hash-bound historical readiness and reference-score report."""
import argparse
import json
from pathlib import Path

from ...application.use_cases.audit_historical_price_readiness import audit_historical_price_readiness


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root",type=Path,default=Path.cwd())
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--open-historical-audit",action="store_true")
    args=parser.parse_args(argv)
    result=audit_historical_price_readiness(args.repository_root,args.protocol.read_bytes(),open_audit=args.open_historical_audit)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("denominators","freshness","historical_reference_scores")},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":raise SystemExit(main())
