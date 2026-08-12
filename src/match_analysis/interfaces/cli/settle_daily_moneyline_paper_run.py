"""CLI for P34A daily Moneyline paper-result settlement and feedback."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

from ...application.use_cases.moneyline_paper_run_bundle import (
    read_json_object,
)
from ...application.use_cases.p34a_daily_moneyline_settlement_artifacts import (
    render_p34a_artifacts,
    write_p34a_artifacts,
)
from ...application.use_cases.settle_daily_moneyline_paper_run import (
    P34A_RESULT_AUTHORITY,
    build_official_result_authority,
    compute_result_authority_fingerprint,
    load_p33a_authority,
    replay_daily_moneyline_paper_settlement,
    settle_daily_moneyline_paper_run,
)
from ...baseball.domain.final_result_observation import (
    load_final_result_observations,
)
from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    fetch_json_bytes,
    normalize_schedule_payload,
)


DEFAULT_OUTPUT_DIR = Path("report/p34a_daily_moneyline_settlement_feedback")


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _authority_for_frozen_final_results(
    *,
    p33a_bundle: Path,
    final_results_bytes: bytes,
    observed_at_utc: str,
    authority_path: Path | None,
) -> dict[str, object]:
    p33a = load_p33a_authority(p33a_bundle)
    if authority_path is not None:
        authority = read_json_object(authority_path)
        authority["network_called"] = False
        return authority
    observations = load_final_result_observations(final_results_bytes)
    target_keys = {
        (str(row["provider_game_id"]), int(row["game_number"]))
        for row in p33a.schedule_rows
    }
    final_keys = {
        (observation.provider_game_id, observation.game_number)
        for observation in observations
    }
    return {
        "source": P34A_RESULT_AUTHORITY,
        "provider_namespace": P34A_RESULT_AUTHORITY,
        "source_url": "frozen-final-results-input",
        "observed_at_utc": observed_at_utc,
        "raw_payload_sha256": hashlib.sha256(final_results_bytes).hexdigest(),
        "network_called": False,
        "target_game_count": len(target_keys),
        "final_result_count": len(observations),
        "non_final_target_count": 0,
        "missing_target_count": len(target_keys - final_keys),
        "all_target_results_final": len(target_keys) == len(final_keys),
        "all_settleable_results_final": True,
        "result_authority_fingerprint": compute_result_authority_fingerprint(
            observations
        ),
    }


def _fetch_official_authority(
    p33a_bundle: Path,
) -> tuple[bytes, dict[str, object]]:
    p33a = load_p33a_authority(p33a_bundle)
    official_dates = sorted(str(row["official_date"]) for row in p33a.schedule_rows)
    if not official_dates:
        raise ValueError("P34A result source has no official target dates")
    raw, source_url = fetch_json_bytes(
        f"{MLB_STATS_API_BASE}/schedule",
        query={
            "sportId": "1",
            "startDate": official_dates[0],
            "endDate": official_dates[-1],
            "gameType": "R",
            "hydrate": "team",
        },
    )
    payload = json.loads(raw.decode("utf-8"))
    normalized_rows = normalize_schedule_payload(payload)
    return build_official_result_authority(
        normalized_schedule_rows=normalized_rows,
        target_game_ids=tuple(str(value) for value in p33a.run_manifest["target_game_ids"]),
        observed_at_utc=_now_utc(),
        source_url=source_url,
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
        network_called=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p33a-bundle",
        type=Path,
        required=True,
        help="exact frozen P33A bundle used as the pregame authority",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="MatchAnalysis repository root (retained for CLI consistency)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="P34A bundle/report output directory",
    )
    parser.add_argument(
        "--from-bundle",
        type=Path,
        help="replay an existing P34A bundle without network access",
    )
    parser.add_argument(
        "--final-results",
        type=Path,
        help="offline FINAL-result JSONL input; bypasses live acquisition",
    )
    parser.add_argument(
        "--result-authority",
        type=Path,
        help="frozen result_authority.json paired with --final-results",
    )
    parser.add_argument(
        "--observed-at-utc",
        help="observation timestamp for --final-results when no authority file is supplied",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_bundle is not None and (
        args.final_results is not None or args.result_authority is not None
    ):
        print(
            "ERROR: --from-bundle cannot be combined with frozen result inputs",
            file=sys.stderr,
        )
        return 1
    if args.result_authority is not None and args.final_results is None:
        print("ERROR: --result-authority requires --final-results", file=sys.stderr)
        return 1
    try:
        p33a_bundle = args.p33a_bundle.resolve()
        if args.from_bundle is not None:
            result = replay_daily_moneyline_paper_settlement(
                p33a_bundle_path=p33a_bundle,
                settlement_bundle_path=args.from_bundle.resolve(),
            )
            hashes = write_p34a_artifacts(args.output_dir, result)
            expected = render_p34a_artifacts(result)
            for name, content in expected.items():
                if (args.output_dir / name).read_bytes() != content:
                    raise ValueError(f"P34A offline replay artifact drift: {name}")
            print(
                f"run_id={result.p33a.run_manifest['run_id']} "
                f"settled={result.settled_count} unresolved={result.unresolved_count} "
                f"offline_replay=PASS feedback_fingerprint="
                f"{result.feedback_result.feedback_ledger_fingerprint} "
                f"output={args.output_dir} files={len(hashes)}"
            )
            return 0

        if args.final_results is not None:
            final_results_bytes = args.final_results.read_bytes()
            authority = _authority_for_frozen_final_results(
                p33a_bundle=p33a_bundle,
                final_results_bytes=final_results_bytes,
                observed_at_utc=args.observed_at_utc or _now_utc(),
                authority_path=(
                    args.result_authority.resolve()
                    if args.result_authority is not None
                    else None
                ),
            )
        else:
            final_results_bytes, authority = _fetch_official_authority(p33a_bundle)
        result = settle_daily_moneyline_paper_run(
            p33a_bundle_path=p33a_bundle,
            final_results_bytes=final_results_bytes,
            result_authority=authority,
        )
        hashes = write_p34a_artifacts(args.output_dir, result)
    except (OSError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"run_id={result.p33a.run_manifest['run_id']} "
        f"target_date={result.p33a.summary['target_date']} "
        f"settled={result.settled_count} correct={result.evaluation_result.correct_count} "
        f"incorrect={result.evaluation_result.incorrect_count} "
        f"unresolved={result.unresolved_count} "
        f"feedback_fingerprint={result.feedback_result.feedback_ledger_fingerprint} "
        f"output={args.output_dir} files={len(hashes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
