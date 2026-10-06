"""Offline legacy bullpen adapter; writes a new research result file only."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys

from ...application.use_cases.adapt_legacy_bullpen_evidence import (
    FORMATS, adapt_legacy_bullpen_evidence, load_json,
)
from ...baseball.domain.bullpen_research_snapshot import canonical_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--target", type=Path, required=True, help="Explicit research target object JSON.")
    parser.add_argument("--capture-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output file; never overwrite evidence.")
    args = parser.parse_args(argv)
    try:
        raw = args.source.read_bytes()
        target_raw = args.target.read_bytes()
        evidence_raw = args.capture_evidence.read_bytes() if args.capture_evidence else None
        result = adapt_legacy_bullpen_evidence(raw, source_format=args.format, target=load_json(target_raw),
                                               evidence=load_json(evidence_raw) if evidence_raw is not None else None)
        result["target_file_sha256"] = sha256(target_raw).hexdigest()
        result["capture_evidence_file_sha256"] = sha256(evidence_raw).hexdigest() if evidence_raw is not None else None
        with args.output.open("xb") as handle:
            handle.write(canonical_bytes(result))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"bullpen-legacy-adapter=rows:{result['source_row_count']} selected:{result['selected_relief_appearance_count']} snapshot:{result['snapshot_sha256']} output:{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
