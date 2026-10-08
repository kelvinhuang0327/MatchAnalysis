"""Verify a sealed one-shot capture offline without changing any ledger."""
import argparse
import json
from pathlib import Path
from ...application.use_cases.prospective_pairing_capture import replay_capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = replay_capture(args.capture_root)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
