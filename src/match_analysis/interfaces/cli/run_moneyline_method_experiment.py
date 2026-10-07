"""Run the hash-bound small offline method trial, with an explicit audit opening."""
import argparse
import json
from pathlib import Path

from ...application.use_cases.run_moneyline_method_experiment import run_moneyline_method_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-retrospective-audit", action="store_true")
    args = parser.parse_args(argv)
    result = run_moneyline_method_experiment(args.repository_root, args.protocol.read_bytes(),
                                           open_audit=args.open_retrospective_audit)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
