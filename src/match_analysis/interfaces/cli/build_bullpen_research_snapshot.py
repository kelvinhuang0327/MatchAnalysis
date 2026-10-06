"""Create one offline research snapshot from explicit local JSON evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from ...application.use_cases.build_bullpen_research_snapshot import build_bullpen_research_snapshot
from ...baseball.domain.bullpen_research_snapshot import canonical_bytes


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New file only; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        bundle = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant)
        snapshot = build_bullpen_research_snapshot(bundle)
        output = {
            "input_file_sha256": sha256(raw).hexdigest(),
            "snapshot_sha256": snapshot.fingerprint(),
            "snapshot": snapshot.to_projection(),
        }
        with args.output.open("xb") as handle:
            handle.write(canonical_bytes(output))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"bullpen-research-snapshot={snapshot.fingerprint()} output={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
