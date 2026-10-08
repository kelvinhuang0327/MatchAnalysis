"""Capture ONE future game using public sources, isolated files and existing Champion."""
import argparse
import json
from pathlib import Path

from ...application.use_cases.prospective_pairing_capture import collect_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--target-date', required=True)
    parser.add_argument('--game-type', choices=('R', 'F', 'D', 'L', 'W'), default='R')
    parser.add_argument('--game-pk', type=int, required=True)
    parser.add_argument('--history-days', type=int, default=45)
    args = parser.parse_args(argv)
    result = collect_once(repository_root=args.repository_root, output_root=args.output_root,
        target_date=args.target_date, game_type=args.game_type, game_pk=args.game_pk,
        history_days=args.history_days)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 1 if result['status'] == 'CAPTURE_BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
