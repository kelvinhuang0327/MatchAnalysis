"""CLI for the deterministic P39A P37/TSL Moneyline market join."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

from ...application.use_cases.join_p37_oos_market_snapshots import (
    P39A_REPORT_RELATIVE_PATH,
    build_p39a_market_join,
    load_p37_predictions,
    mark_deterministic_rerun_verified,
    source_scope_keys,
)
from ...application.use_cases.p39a_market_join_artifacts import write_p39a_artifacts
from ...infrastructure.sources.p39a_tsl_market_snapshot import (
    P39A_TSL_SOURCE_RELATIVE_PATH,
    load_tsl_market_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--legacy-repository", type=Path, required=True)
    parser.add_argument("--legacy-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _legacy_metadata(
    legacy_repository: Path,
    *,
    expected_sha256: str,
    source_path: Path,
) -> dict[str, Any]:
    relative_path = P39A_TSL_SOURCE_RELATIVE_PATH
    return {
        "source_repository": str(legacy_repository.resolve()),
        "source_path": str(source_path.resolve()),
        "source_relative_path": relative_path,
        "source_head": _git(legacy_repository, "rev-parse", "HEAD"),
        "source_tree": _git(legacy_repository, "rev-parse", "HEAD^{tree}"),
        "source_blob_at_head": _git(
            legacy_repository,
            "rev-parse",
            f"HEAD:{relative_path}",
        ),
        "source_branch": _git(legacy_repository, "branch", "--show-current"),
        "source_status": _git(
            legacy_repository,
            "status",
            "--porcelain=v2",
            "--branch",
            "--",
            relative_path,
        ),
        "source_sha256": expected_sha256,
        "source_stable": True,
        "timestamp_semantics_trusted": True,
        "timestamp_semantics_basis": (
            "Legacy row fetched_at is the local fetch/market-observation time; "
            "legacy row game_time is the scheduled start; no provider-side "
            "observation timestamp is present."
        ),
    }


def _execute_once(
    repository_root: Path,
    legacy_repository: Path,
    expected_sha256: str,
) -> Any:
    source_path = legacy_repository / P39A_TSL_SOURCE_RELATIVE_PATH
    before = _legacy_metadata(
        legacy_repository,
        expected_sha256=expected_sha256,
        source_path=source_path,
    )
    predictions, p37_manifest = load_p37_predictions(repository_root)
    source = load_tsl_market_source(
        source_path,
        expected_sha256=expected_sha256,
        target_source_keys=source_scope_keys(predictions),
    )
    after = _legacy_metadata(
        legacy_repository,
        expected_sha256=expected_sha256,
        source_path=source_path,
    )
    for field_name in (
        "source_head",
        "source_tree",
        "source_blob_at_head",
        "source_status",
    ):
        if before[field_name] != after[field_name]:
            raise RuntimeError(
                "P39A_LEGACY_MARKET_SOURCE_UNSTABLE_STOP: "
                f"legacy metadata changed for {field_name}"
            )
    source_manifest = {
        **before,
        "source_sha256": source.raw_sha256,
        "source_row_count": source.source_row_count,
        "scoped_source_row_count": source.scoped_source_row_count,
        "scoped_status_counts": dict(source.scoped_status_counts),
    }
    return build_p39a_market_join(
        predictions,
        source.candidates,
        source_manifest=source_manifest,
        p37_manifest=p37_manifest,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    legacy_repository = args.legacy_repository.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repository_root / P39A_REPORT_RELATIVE_PATH).resolve()
    )
    expected_output_dir = (repository_root / P39A_REPORT_RELATIVE_PATH).resolve()
    if output_dir != expected_output_dir:
        raise ValueError(
            "ARTIFACT_OUTPUT_PATH_CONFLICT: P39A output must remain under the "
            f"packet-allowlisted path {expected_output_dir}"
        )

    try:
        first = _execute_once(
            repository_root,
            legacy_repository,
            args.legacy_source_sha256,
        )
        second = _execute_once(
            repository_root,
            legacy_repository,
            args.legacy_source_sha256,
        )
        if first.comparable_projection() != second.comparable_projection():
            raise RuntimeError("P39A deterministic rerun did not reproduce the same result")
        result = mark_deterministic_rerun_verified(first)
        write_p39a_artifacts(output_dir, result)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"p37_evaluable_target_count={summary['p37_evaluable_target_count']} "
        f"exact_identity_match_count={summary['exact_identity_match_count']} "
        f"usable_pregame_market_rows={summary['usable_pregame_market_rows']} "
        f"edge_ready_count={summary['edge_ready_count']} "
        f"no_market_rows={summary['no_market_rows']} "
        f"post_start_rejected_rows={summary['post_start_rejected_rows']} "
        f"ambiguous_rows={summary['ambiguous_rows']} "
        f"missing_or_untrusted_timestamp_rows={summary['missing_or_untrusted_timestamp_rows']} "
        f"malformed_or_incomplete_price_rows={summary['malformed_or_incomplete_price_rows']} "
        f"conclusion={summary['conclusion']} "
        f"deterministic_rerun_verified={summary['deterministic_rerun_verified']} "
        "bet_pass=NOT_RUN roi_profitability=NOT_RUN model_promotion=NOT_RUN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
