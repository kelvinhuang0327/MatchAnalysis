"""Acquire one bounded TSL Moneyline snapshot for the existing P30A path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.request import Request, urlopen

from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    fetch_json_bytes,
    normalize_schedule_payload,
)
from ...infrastructure.sources.tsl_moneyline_acquisition import (
    TSL_ACQUISITION_SCHEMA_VERSION,
    TSL_BLOB3RD_SOURCE_LABEL,
    TSL_LOCAL_TIMEZONE,
    TslBlob3rdClient,
    TslMoneylineHistory,
    TslNormalizationResult,
    build_tsl_moneyline_history,
)


P32A_OPERATION = "ACQUIRE_TSL_MONEYLINE_SNAPSHOT"
P32A_RUNTIME_ROOT = Path("/tmp/matchanalysis-p32a-tsl-acquisition")
P32A_DEFAULT_SOURCE_MANIFEST = Path(
    "data/fixtures/p28ab_tsl_aligned_moneyline_edge/source_manifest.json"
)
P32A_MAX_TARGET_DATE_OFFSET_DAYS = 7

STOP_P32A_NO_TARGET_SLATE = "STOP_MATCHANALYSIS_P32A_NO_TARGET_SLATE_AVAILABLE"
STOP_P32A_NO_QUALIFIED_OBSERVATIONS = (
    "STOP_MATCHANALYSIS_P32A_NO_QUALIFIED_TSL_OBSERVATIONS"
)
STOP_P32A_TARGET_COVERAGE = "STOP_MATCHANALYSIS_P32A_TARGET_SLATE_COVERAGE"

JsonOpener = Callable[[Request, int], bytes]
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class TslMoneylineSnapshotAcquisition:
    """Immutable result of one official-slate-first acquisition."""

    operation: str
    target_date: str
    selection_started_at_utc: str
    fetched_at_utc: str
    schedule_url: str
    schedule_rows: tuple[dict[str, Any], ...]
    target_schedule_rows: tuple[dict[str, Any], ...]
    requested_game_ids: tuple[str, ...]
    history: TslMoneylineHistory
    normalization: TslNormalizationResult
    source_payload_sha256: tuple[tuple[str, str], ...]
    runtime_capture_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TslMoneylinePaperInputs:
    """P30A-compatible inputs without mutating the canonical history."""

    tsl_rows: tuple[dict[str, Any], ...]
    tsl_raw_sha256: str
    schedule_rows: tuple[dict[str, Any], ...]
    target_boxscore_rows: tuple[dict[str, Any], ...]
    pitcher_game_log_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    requested_game_ids: tuple[str, ...]
    cohort_start_date: str
    cohort_end_date: str


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("acquisition clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    normalized = _utc(value)
    return normalized.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("official schedule timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _write_immutable_capture(runtime_root: Path, relative_path: str, raw: bytes) -> str:
    path = runtime_root / "capture" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"immutable runtime capture changed: {path}")
    else:
        path.write_bytes(raw)
    return str(path)


def _default_mlb_schedule_opener(request: Request, timeout: int) -> bytes:
    """Use the existing urllib transport with the same bounded curl fallback."""

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is validated by fetch_json_bytes
            return response.read()
    except Exception as original_error:
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-L",
                    "-sS",
                    "--max-time",
                    str(timeout),
                    "-X",
                    "GET",
                    request.full_url,
                    "-H",
                    "Accept: application/json",
                    "-H",
                    "User-Agent: MatchAnalysis/1.0",
                ],
                check=True,
                capture_output=True,
                timeout=timeout + 5,
            )
            return completed.stdout
        except Exception as curl_error:
            raise original_error from curl_error


def _select_target_slate(
    schedule_rows: tuple[dict[str, Any], ...],
    *,
    started_at_utc: datetime,
    requested_target_date: str | None = None,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Choose the earliest future local slate from official MLB timing only."""

    started = _utc(started_at_utc)
    local_today = started.astimezone(TSL_LOCAL_TIMEZONE).date()
    latest_date = local_today + timedelta(days=P32A_MAX_TARGET_DATE_OFFSET_DAYS)
    if requested_target_date is not None:
        try:
            requested_date = date.fromisoformat(requested_target_date)
        except ValueError as exc:
            raise ValueError("target_date must be YYYY-MM-DD") from exc
        if not local_today <= requested_date <= latest_date:
            raise RuntimeError(STOP_P32A_NO_TARGET_SLATE)
        requested_rows = tuple(
            sorted(
                (
                    dict(row)
                    for row in schedule_rows
                    if requested_date.isoformat()
                    == _parse_utc(str(row["scheduled_start_utc"]))
                    .astimezone(TSL_LOCAL_TIMEZONE)
                    .date()
                    .isoformat()
                    and started < _parse_utc(str(row["scheduled_start_utc"]))
                ),
                key=lambda row: (
                    str(row["scheduled_start_utc"]),
                    int(row["game_number"]),
                    int(row["game_pk"]),
                ),
            )
        )
        if len(requested_rows) < 2:
            raise RuntimeError(STOP_P32A_NO_TARGET_SLATE)
        return requested_date.isoformat(), requested_rows

    by_local_date: dict[date, list[dict[str, Any]]] = {}
    for row in schedule_rows:
        scheduled = _parse_utc(str(row["scheduled_start_utc"]))
        local_date = scheduled.astimezone(TSL_LOCAL_TIMEZONE).date()
        if not started < scheduled or not local_today <= local_date <= latest_date:
            continue
        by_local_date.setdefault(local_date, []).append(dict(row))
    if not by_local_date:
        raise RuntimeError(STOP_P32A_NO_TARGET_SLATE)
    target_date = min(by_local_date)
    target_rows = tuple(
        sorted(
            by_local_date[target_date],
            key=lambda row: (
                str(row["scheduled_start_utc"]),
                int(row["game_number"]),
                int(row["game_pk"]),
            ),
        )
    )
    if len(target_rows) < 2:
        raise RuntimeError(STOP_P32A_NO_TARGET_SLATE)
    return target_date.isoformat(), target_rows


def acquire_tsl_moneyline_snapshot(
    *,
    repository_root: str | Path,
    runtime_root: str | Path = P32A_RUNTIME_ROOT,
    target_date: str | None = None,
    clock: Clock = lambda: datetime.now(UTC),
    schedule_opener: JsonOpener | None = None,
    tsl_fetcher: Callable[[str], bytes] | None = None,
) -> TslMoneylineSnapshotAcquisition:
    """Freeze an official MLB slate, then acquire one bounded TSL snapshot.

    The official schedule request is always completed before the first TSL
    request.  ``clock`` and injected transports exist only for deterministic
    tests; the default path records the actual post-response UTC time.
    """

    runtime_path = Path(runtime_root)
    started_at = _utc(clock())
    schedule_start = started_at.astimezone(TSL_LOCAL_TIMEZONE).date()
    schedule_end = schedule_start + timedelta(days=P32A_MAX_TARGET_DATE_OFFSET_DAYS)
    schedule_kwargs: dict[str, Any] = {
        "query": {
            "sportId": "1",
            "startDate": started_at.date().isoformat(),
            "endDate": schedule_end.isoformat(),
            "gameType": "R",
            "hydrate": "team",
        }
    }
    schedule_kwargs["opener"] = schedule_opener or _default_mlb_schedule_opener
    raw_schedule, schedule_url = fetch_json_bytes(
        f"{MLB_STATS_API_BASE}/schedule",
        **schedule_kwargs,
    )
    _write_immutable_capture(runtime_path, "mlb_schedule.json", raw_schedule)
    schedule_payload = json.loads(raw_schedule.decode("utf-8"))
    schedule_rows = normalize_schedule_payload(schedule_payload)
    target_date, target_rows = _select_target_slate(
        schedule_rows,
        started_at_utc=started_at,
        requested_target_date=target_date,
    )

    client = (
        TslBlob3rdClient(fetcher=tsl_fetcher)
        if tsl_fetcher is not None
        else TslBlob3rdClient()
    )
    capture = client.fetch_modern_capture()
    capture_paths = [
        _write_immutable_capture(
            runtime_path,
            (
                "tsl_live_games.json"
                if url.endswith("/Live/Games.zh.json")
                else "tsl_sports.json"
                if url.endswith("/Pre/Sports.zh.json")
                else f"tsl_pre_games_{capture.sport_id}_{url.rsplit('.', 2)[-2]}.json"
            ),
            raw,
        )
        for url, raw in capture.payloads
    ]
    fetched_at = _format_utc(clock())
    history, normalization = build_tsl_moneyline_history(
        capture,
        fetched_at=fetched_at,
        target_date=target_date,
    )
    if not history.observations:
        raise RuntimeError(STOP_P32A_NO_QUALIFIED_OBSERVATIONS)

    # Resolve overlap through the existing P28AB crosswalk.  This is imported
    # only after acquisition so this adapter remains independent of P30A.
    from .generate_tsl_moneyline_edge_batch import _crosswalk

    crosswalk = _crosswalk(
        tsl_rows=history.rows,
        schedule_rows=target_rows,
        cohort_start_date=target_date,
        cohort_end_date=target_date,
    )
    requested_game_ids = tuple(
        sorted(
            {
                str(item.official["provider_game_id"])
                for item in crosswalk
                if item.official is not None
            }
        )
    )
    if len(requested_game_ids) < 2:
        raise RuntimeError(STOP_P32A_TARGET_COVERAGE)
    return TslMoneylineSnapshotAcquisition(
        operation=P32A_OPERATION,
        target_date=target_date,
        selection_started_at_utc=_format_utc(started_at),
        fetched_at_utc=fetched_at,
        schedule_url=schedule_url,
        schedule_rows=tuple(dict(row) for row in schedule_rows),
        target_schedule_rows=target_rows,
        requested_game_ids=requested_game_ids,
        history=history,
        normalization=normalization,
        source_payload_sha256=capture.payload_sha256,
        runtime_capture_paths=tuple(capture_paths),
    )


def build_p30a_paper_inputs(
    acquisition: TslMoneylineSnapshotAcquisition,
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path = P32A_DEFAULT_SOURCE_MANIFEST,
) -> TslMoneylinePaperInputs:
    """Adapt fresh rows to the established P30A source-manifest contract."""

    repository = Path(repository_root)
    manifest_path = Path(source_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repository / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("P30A source manifest must be an object")
    manifest = deepcopy(manifest)
    scope = manifest.get("cohort_scope")
    if not isinstance(scope, dict):
        raise ValueError("P30A source manifest has no cohort_scope")
    scope["game_time_start_date"] = acquisition.target_date
    scope["game_time_end_date"] = acquisition.target_date
    manifest["tsl_fixture_sha256"] = acquisition.history.selected_rows_sha256
    manifest["target_game_ids"] = list(acquisition.requested_game_ids)
    manifest["p32a_tsl_acquisition"] = {
        "schema_version": TSL_ACQUISITION_SCHEMA_VERSION,
        "operation": P32A_OPERATION,
        "source": TSL_BLOB3RD_SOURCE_LABEL,
        "target_date": acquisition.target_date,
        "selection_started_at_utc": acquisition.selection_started_at_utc,
        "fetched_at_utc": acquisition.fetched_at_utc,
        "source_payload_sha256": dict(acquisition.source_payload_sha256),
        "source_row_count": acquisition.normalization.source_row_count,
        "qualified_observation_count": len(acquisition.history.observations),
        "rejected_row_count": len(acquisition.normalization.rejected_games),
        "official_target_game_count": len(acquisition.target_schedule_rows),
        "official_overlap_game_count": len(acquisition.requested_game_ids),
        "qualified_non_overlap_count": max(
            0,
            len(acquisition.history.observations)
            - len(acquisition.requested_game_ids),
        ),
        "runtime_capture_paths": list(acquisition.runtime_capture_paths),
    }
    return TslMoneylinePaperInputs(
        tsl_rows=acquisition.history.rows,
        tsl_raw_sha256=acquisition.history.selected_rows_sha256,
        schedule_rows=acquisition.schedule_rows,
        target_boxscore_rows=(),
        pitcher_game_log_rows=(),
        source_manifest=manifest,
        requested_game_ids=acquisition.requested_game_ids,
        cohort_start_date=acquisition.target_date,
        cohort_end_date=acquisition.target_date,
    )


def run_tsl_moneyline_paper_smoke(
    acquisition: TslMoneylineSnapshotAcquisition,
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path = P32A_DEFAULT_SOURCE_MANIFEST,
) -> Any:
    """Run the existing P30A paper composition over the fresh runtime bundle."""

    inputs = build_p30a_paper_inputs(
        acquisition,
        repository_root=repository_root,
        source_manifest_path=source_manifest_path,
    )
    from .run_moneyline_paper_analysis import run_moneyline_paper_analysis

    return run_moneyline_paper_analysis(
        repository_root=repository_root,
        tsl_rows=inputs.tsl_rows,
        tsl_raw_sha256=inputs.tsl_raw_sha256,
        schedule_rows=inputs.schedule_rows,
        target_boxscore_rows=inputs.target_boxscore_rows,
        pitcher_game_log_rows=inputs.pitcher_game_log_rows,
        source_manifest=inputs.source_manifest,
        offline_replay_verified=True,
        cohort_start_date=inputs.cohort_start_date,
        cohort_end_date=inputs.cohort_end_date,
        requested_game_ids=inputs.requested_game_ids,
        allow_missing_starter_identity=True,
        allow_insufficient_evaluable=True,
    )


def write_p32a_summary(
    path: str | Path,
    *,
    acquisition: TslMoneylineSnapshotAcquisition,
    paper_result: Any | None = None,
) -> dict[str, Any]:
    """Write the minimal retained P32A report artifact."""

    report = {
        "schema_version": TSL_ACQUISITION_SCHEMA_VERSION,
        "operation": P32A_OPERATION,
        "source": TSL_BLOB3RD_SOURCE_LABEL,
        "target_date": acquisition.target_date,
        "selection_started_at_utc": acquisition.selection_started_at_utc,
        "fetched_at_utc": acquisition.fetched_at_utc,
        "schedule_url": acquisition.schedule_url,
        "source_payload_sha256": dict(acquisition.source_payload_sha256),
        "source_row_count": acquisition.normalization.source_row_count,
        "qualified_observation_count": len(acquisition.history.observations),
        "rejected_row_count": len(acquisition.normalization.rejected_games),
        "rejection_reason_counts": {
            reason: sum(
                rejection.reason == reason
                for rejection in acquisition.normalization.rejected_games
            )
            for reason in sorted(
                {rejection.reason for rejection in acquisition.normalization.rejected_games}
            )
        },
        "official_target_game_count": len(acquisition.target_schedule_rows),
        "official_overlap_game_count": len(acquisition.requested_game_ids),
        "qualified_non_overlap_count": max(
            0,
            len(acquisition.history.observations)
            - len(acquisition.requested_game_ids),
        ),
        "requested_game_ids": list(acquisition.requested_game_ids),
        "runtime_capture_paths": list(acquisition.runtime_capture_paths),
        "canonical_history_mutated": False,
        "paper_only": True,
        "diagnostic_only": True,
        "real_betting_recommendation": False,
        "staking_implemented": False,
        "profitability_claim": False,
        "paper_smoke": (
            {
                "run_id": paper_result.summary["run_id"],
                "raw_game_count": paper_result.summary["raw_game_count"],
                "structural_status_counts": paper_result.summary[
                    "structural_status_counts"
                ],
                "edge_available_count": paper_result.summary[
                    "edge_available_count"
                ],
                "feature_unavailable_count": paper_result.summary[
                    "feature_unavailable_count"
                ],
                "price_unavailable_pre_cutoff_count": paper_result.summary[
                    "price_unavailable_pre_cutoff_count"
                ],
                "paper_only": paper_result.summary["paper_only"],
                "real_betting_recommendation": paper_result.summary[
                    "real_betting_recommendation"
                ],
            }
            if paper_result is not None
            else None
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = (
    "P32A_DEFAULT_SOURCE_MANIFEST",
    "P32A_OPERATION",
    "P32A_RUNTIME_ROOT",
    "STOP_P32A_NO_QUALIFIED_OBSERVATIONS",
    "STOP_P32A_NO_TARGET_SLATE",
    "STOP_P32A_TARGET_COVERAGE",
    "TslMoneylinePaperInputs",
    "TslMoneylineSnapshotAcquisition",
    "acquire_tsl_moneyline_snapshot",
    "build_p30a_paper_inputs",
    "run_tsl_moneyline_paper_smoke",
    "write_p32a_summary",
)
