"""CLI for P23F2 official acquisition and offline future-fold materialization."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ...application.use_cases.acquire_future_moneyline_history import acquire_official_history
from ...application.use_cases.future_moneyline_fold_artifacts import (
    render_source_manifest,
    write_future_fold_artifacts,
)
from ...application.use_cases.materialize_future_moneyline_fold import materialize_from_normalized_dir
from ...infrastructure.providers.mlb_official_historical_source import canonical_json_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquired-at", default="2026-08-10T00:00:00Z")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline:
        if args.source_manifest is None:
            raise SystemExit("--offline requires --source-manifest")
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        source_fingerprint = hashlib.sha256(canonical_json_bytes(source_manifest)).hexdigest()
    else:
        acquisition = acquire_official_history(
            raw_root=args.raw_root,
            normalized_root=args.normalized_root,
            acquired_at_utc=datetime.fromisoformat(args.acquired_at.replace("Z", "+00:00")).astimezone(UTC),
        )
        source_manifest = render_source_manifest(
            records=[asdict(record) for record in acquisition.source_records],
            normalized_hashes=acquisition.normalized_hashes,
        )
        source_fingerprint = hashlib.sha256(canonical_json_bytes(source_manifest)).hexdigest()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "source_manifest.json").write_bytes(
            json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
    fold = materialize_from_normalized_dir(
        args.normalized_root,
        source_manifest_fingerprint=source_fingerprint,
    )
    hashes = write_future_fold_artifacts(
        args.output_dir,
        fold,
        source_manifest=source_manifest,
        offline_replay_verified=args.offline,
    )
    print(f"fold_id={fold.fold_id} games={len(fold.feature_rows)}")
    print(f"feature_fingerprint={fold.feature_fingerprint}")
    print(f"result_fingerprint={fold.result_fingerprint}")
    print(f"fold_fingerprint={fold.fold_fingerprint}")
    print(f"artifacts={len(hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
