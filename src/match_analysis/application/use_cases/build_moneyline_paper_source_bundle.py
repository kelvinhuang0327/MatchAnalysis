"""Build the bounded native source bundle consumed by the P30A run."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...infrastructure.sources.tsl_moneyline_history import (
    STOP_TSL_LOADER_PARITY_FAILED,
    TSL_AUTHORITY_BLOB,
    TSL_AUTHORITY_PATH,
    TSL_AUTHORITY_RAW_SHA256,
    TSL_AUTHORITY_REF,
    TSL_AUTHORITY_TREE,
    TSL_LOCAL_TIMEZONE,
    TSL_MIGRATED_PATH,
    TslMoneylineHistory,
    load_tsl_moneyline_history,
)
from .acquire_future_moneyline_history import load_normalized_rows
from .generate_tsl_moneyline_edge_batch import (
    P28AB_COHORT_END_DATE,
    P28AB_COHORT_START_DATE,
)


MAX_DATE_SCOPE_DAYS = 7
P31A_PRECOHORT_SELECTION_RULE = (
    "NEAREST_SEVEN_CONSECUTIVE_DATES_WITH_QUALIFIED_TSL_AND_MLB_SCHEDULE_GAMES"
)
STOP_P31A_PRECOHORT_OVERLAP = "STOP_MATCHANALYSIS_P31A_PRECOHORT_OVERLAP"
STOP_P31A_NO_GENERALIZATION_WINDOW = (
    "STOP_MATCHANALYSIS_P31A_NO_GENERALIZATION_WINDOW_AVAILABLE"
)
STOP_P31A_SOURCE_SCOPE = "STOP_MATCHANALYSIS_P31A_SOURCE_SCOPE_INVALID"


@dataclass(frozen=True, slots=True)
class MoneylinePaperSourceBundle:
    """All deterministic inputs needed by the existing P30A/P28AB path."""

    scope_start_date: str
    scope_end_date: str
    tsl_history: TslMoneylineHistory
    schedule_rows: tuple[dict[str, Any], ...]
    target_boxscore_rows: tuple[dict[str, Any], ...]
    pitcher_game_log_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    tsl_authority_manifest: dict[str, Any]
    requested_game_ids: tuple[str, ...] | None
    parity_mode: bool
    selection_metadata: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class PreCohortWindow:
    """A frozen, source-only generalization window selected before inference."""

    p30a_cohort_start_date: str
    p30a_cohort_end_date: str
    selected_start_date: str
    selected_end_date: str
    selected_dates: tuple[str, ...]
    source_row_counts: dict[str, int]
    official_game_ids: tuple[str, ...]
    fallback_used: bool

    @property
    def length(self) -> int:
        return len(self.selected_dates)

    def metadata(self) -> dict[str, Any]:
        return {
            "selection_rule": P31A_PRECOHORT_SELECTION_RULE,
            "p30a_cohort_start_date": self.p30a_cohort_start_date,
            "p30a_cohort_end_date": self.p30a_cohort_end_date,
            "selected_start_date": self.selected_start_date,
            "selected_end_date": self.selected_end_date,
            "window_length": self.length,
            "selected_dates": list(self.selected_dates),
            "source_row_counts": dict(self.source_row_counts),
            "official_game_count": len(self.official_game_ids),
            "official_game_ids": list(self.official_game_ids),
            "fallback_used": self.fallback_used,
            "selection_stage": (
                "BEFORE_FEATURE_ELIGIBILITY_INFERENCE_PRICE_OUTCOME"
            ),
            "outcome_based_selection": False,
        }


def _date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc


def resolve_date_scope(
    *,
    date_value: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    default_start_date: str = P28AB_COHORT_START_DATE,
    default_end_date: str = P28AB_COHORT_END_DATE,
) -> tuple[str, str, bool]:
    """Resolve one bounded scope and report whether it was explicitly requested."""

    explicit = date_value is not None or start_date is not None or end_date is not None
    if date_value is not None and (start_date is not None or end_date is not None):
        raise ValueError("--date cannot be combined with --start-date or --end-date")
    if date_value is not None:
        start = end = _date(date_value, field_name="date")
    elif start_date is None and end_date is None:
        start = _date(default_start_date, field_name="start_date")
        end = _date(default_end_date, field_name="end_date")
    elif start_date is None or end_date is None:
        raise ValueError("--start-date and --end-date must be supplied together")
    else:
        start = _date(start_date, field_name="start_date")
        end = _date(end_date, field_name="end_date")
    if end < start:
        raise ValueError("end date must not precede start date")
    if (end - start).days + 1 > MAX_DATE_SCOPE_DAYS:
        raise ValueError("date scope must be at most seven consecutive calendar dates")
    return start.isoformat(), end.isoformat(), explicit


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("official schedule timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _requested_game_ids(
    schedule_rows: tuple[dict[str, Any], ...],
    *,
    start_date: str,
    end_date: str,
) -> tuple[str, ...]:
    start = _date(start_date, field_name="start_date")
    end = _date(end_date, field_name="end_date") + timedelta(days=1)
    start_utc = datetime.combine(start, time.min, tzinfo=TSL_LOCAL_TIMEZONE).astimezone(UTC)
    end_utc = datetime.combine(end, time.min, tzinfo=TSL_LOCAL_TIMEZONE).astimezone(UTC)
    selected: dict[str, dict[str, Any]] = {}
    for row in schedule_rows:
        game_id = str(row["provider_game_id"])
        scheduled = _parse_utc(str(row["scheduled_start_utc"]))
        if not start_utc <= scheduled < end_utc:
            continue
        if not (bool(row.get("final")) and row.get("status") == "Final"):
            continue
        selected[game_id] = row
    ordered = sorted(
        selected.values(),
        key=lambda row: (
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        ),
    )
    if len(ordered) < 2:
        raise RuntimeError(
            "STOP_MATCHANALYSIS_P31A_OFFICIAL_SLATE_INSUFFICIENT"
        )
    return tuple(str(row["provider_game_id"]) for row in ordered)


def _official_game_ids_by_date(
    schedule_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    by_date: dict[str, set[str]] = {}
    for row in schedule_rows:
        scheduled = _parse_utc(str(row["scheduled_start_utc"]))
        official_date = scheduled.astimezone(TSL_LOCAL_TIMEZONE).date().isoformat()
        by_date.setdefault(official_date, set()).add(str(row["provider_game_id"]))
    return {
        official_date: tuple(sorted(game_ids))
        for official_date, game_ids in by_date.items()
    }


def select_precohort_window(
    *,
    tsl_history_path: str | Path,
    tsl_authority_manifest_path: str | Path,
    schedule_rows: Sequence[Mapping[str, Any]],
    p30a_cohort_start_date: str = P28AB_COHORT_START_DATE,
    p30a_cohort_end_date: str = P28AB_COHORT_END_DATE,
    p30a_game_ids: Sequence[str] = (),
) -> PreCohortWindow:
    """Select the nearest usable source-only pre-cohort calendar window."""

    p30a_start = _date(p30a_cohort_start_date, field_name="p30a_cohort_start_date")
    p30a_end = _date(p30a_cohort_end_date, field_name="p30a_cohort_end_date")
    if p30a_end < p30a_start:
        raise ValueError("P30A cohort end must not precede its start")
    history = load_tsl_moneyline_history(
        tsl_history_path,
        start_date=date.min,
        end_date=p30a_start - timedelta(days=1),
    )
    _load_tsl_authority_manifest(tsl_authority_manifest_path, history=history)

    source_row_counts: dict[str, int] = {}
    for observation in history.observations:
        local_date = _parse_utc(observation.game_time).astimezone(
            TSL_LOCAL_TIMEZONE
        ).date().isoformat()
        source_row_counts[local_date] = source_row_counts.get(local_date, 0) + 1
    official_game_ids_by_date = _official_game_ids_by_date(schedule_rows)
    represented_dates = sorted(
        date_value
        for date_value, source_count in source_row_counts.items()
        if source_count > 0
        and date_value < p30a_start.isoformat()
        and len(official_game_ids_by_date.get(date_value, ())) >= 2
    )
    if not represented_dates:
        raise RuntimeError(STOP_P31A_NO_GENERALIZATION_WINDOW)

    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for date_value in represented_dates:
        parsed = _date(date_value, field_name="represented_date")
        if current:
            previous = _date(current[-1], field_name="represented_date")
            if parsed != previous + timedelta(days=1):
                runs.append(tuple(current))
                current = []
        current.append(date_value)
    if current:
        runs.append(tuple(current))

    immediately_preceding = max(runs, key=lambda run: (run[-1], len(run)))
    selected_dates = (
        immediately_preceding[-MAX_DATE_SCOPE_DAYS:]
        if len(immediately_preceding) >= MAX_DATE_SCOPE_DAYS
        else immediately_preceding
    )
    if not selected_dates:
        raise RuntimeError(STOP_P31A_NO_GENERALIZATION_WINDOW)
    selected_start = _date(selected_dates[0], field_name="selected_start_date")
    selected_end = _date(selected_dates[-1], field_name="selected_end_date")
    if not selected_end < p30a_start:
        raise RuntimeError(STOP_P31A_PRECOHORT_OVERLAP)

    selected_game_ids = tuple(
        sorted(
            {
                game_id
                for date_value in selected_dates
        for game_id in official_game_ids_by_date.get(date_value, ())
            }
        )
    )
    overlap = set(selected_game_ids).intersection(str(game_id) for game_id in p30a_game_ids)
    if overlap:
        raise RuntimeError(STOP_P31A_PRECOHORT_OVERLAP)
    return PreCohortWindow(
        p30a_cohort_start_date=p30a_start.isoformat(),
        p30a_cohort_end_date=p30a_end.isoformat(),
        selected_start_date=selected_start.isoformat(),
        selected_end_date=selected_end.isoformat(),
        selected_dates=tuple(selected_dates),
        source_row_counts={date_value: source_row_counts[date_value] for date_value in selected_dates},
        official_game_ids=selected_game_ids,
        fallback_used=len(immediately_preceding) < MAX_DATE_SCOPE_DAYS,
    )


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source manifest must be an object: {path}")
    return value


def _load_tsl_authority_manifest(
    path: str | Path,
    *,
    history: TslMoneylineHistory,
) -> dict[str, Any]:
    manifest = _load_object(path)
    expected = {
        "original_ref": TSL_AUTHORITY_REF,
        "original_tree": TSL_AUTHORITY_TREE,
        "original_path": TSL_AUTHORITY_PATH,
        "original_git_blob": TSL_AUTHORITY_BLOB,
        "source_label": "TSL_BLOB3RD",
        "raw_sha256": TSL_AUTHORITY_RAW_SHA256,
        "migrated_path": TSL_MIGRATED_PATH,
        "migration_task_id": "P31A",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"{STOP_TSL_LOADER_PARITY_FAILED}: source manifest provenance mismatch"
        )
    if history.raw_sha256 != manifest["raw_sha256"]:
        raise ValueError(
            f"{STOP_TSL_LOADER_PARITY_FAILED}: source manifest hash mismatch"
        )
    return manifest


def _source_manifest(
    path: str | Path,
    *,
    history: TslMoneylineHistory,
    start_date: str,
    end_date: str,
    parity_mode: bool,
    selection_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = _load_object(path)
    if parity_mode:
        return manifest
    manifest = deepcopy(manifest)
    scope = manifest.get("cohort_scope")
    if not isinstance(scope, dict):
        raise ValueError("P28AB source manifest has no cohort_scope")
    scope["game_time_start_date"] = start_date
    scope["game_time_end_date"] = end_date
    manifest["tsl_fixture_sha256"] = history.selected_rows_sha256
    counts = manifest.get("authority_reference_counts")
    if isinstance(counts, dict):
        counts["cohort_window_exact_two_way_pregame_rows"] = history.qualified_row_count
    manifest["p31a_tsl_source"] = {
        "migrated_path": TSL_MIGRATED_PATH,
        "raw_sha256": history.raw_sha256,
        "selected_rows_sha256": history.selected_rows_sha256,
        "scope_start_date": start_date,
        "scope_end_date": end_date,
    }
    if selection_metadata is not None:
        manifest["p31a_generalization_window"] = deepcopy(dict(selection_metadata))
    return manifest


def build_moneyline_paper_source_bundle(
    *,
    tsl_history_path: str | Path,
    tsl_authority_manifest_path: str | Path,
    source_manifest_path: str | Path,
    schedule_path: str | Path,
    target_boxscores_path: str | Path,
    pitcher_game_logs_path: str | Path,
    start_date: str,
    end_date: str,
    parity_mode: bool = False,
    selection_metadata: Mapping[str, Any] | None = None,
    p30a_game_ids: Sequence[str] = (),
) -> MoneylinePaperSourceBundle:
    """Assemble native TSL and existing official normalized inputs.

    ``parity_mode`` retains the committed P28AB source cohort membership.  An
    explicitly date-driven call instead uses the complete final official slate
    inside the requested time window as P30A raw-game membership.
    """

    start, end, _ = resolve_date_scope(
        start_date=start_date,
        end_date=end_date,
    )
    if parity_mode and (start, end) != (
        P28AB_COHORT_START_DATE,
        P28AB_COHORT_END_DATE,
    ):
        raise RuntimeError(STOP_P31A_SOURCE_SCOPE)
    if selection_metadata is not None:
        if selection_metadata.get("selected_start_date") != start or selection_metadata.get(
            "selected_end_date"
        ) != end:
            raise RuntimeError(STOP_P31A_SOURCE_SCOPE)
        if not end < P28AB_COHORT_START_DATE:
            raise RuntimeError(STOP_P31A_PRECOHORT_OVERLAP)
    history = load_tsl_moneyline_history(
        tsl_history_path,
        start_date=start,
        end_date=end,
    )
    tsl_authority_manifest = _load_tsl_authority_manifest(
        tsl_authority_manifest_path,
        history=history,
    )
    schedule_rows = load_normalized_rows(schedule_path)
    if parity_mode:
        requested_game_ids = None
    else:
        requested_game_ids = _requested_game_ids(
            schedule_rows,
            start_date=start,
            end_date=end,
        )
        overlap = set(requested_game_ids).intersection(
            str(game_id) for game_id in p30a_game_ids
        )
        if overlap:
            raise RuntimeError(STOP_P31A_PRECOHORT_OVERLAP)
    return MoneylinePaperSourceBundle(
        scope_start_date=start,
        scope_end_date=end,
        tsl_history=history,
        schedule_rows=schedule_rows,
        target_boxscore_rows=load_normalized_rows(target_boxscores_path),
        pitcher_game_log_rows=load_normalized_rows(pitcher_game_logs_path),
        source_manifest=_source_manifest(
            source_manifest_path,
            history=history,
            start_date=start,
            end_date=end,
            parity_mode=parity_mode,
            selection_metadata=selection_metadata,
        ),
        tsl_authority_manifest=tsl_authority_manifest,
        requested_game_ids=requested_game_ids,
        parity_mode=parity_mode,
        selection_metadata=(
            deepcopy(dict(selection_metadata))
            if selection_metadata is not None
            else None
        ),
    )


__all__ = (
    "MAX_DATE_SCOPE_DAYS",
    "MoneylinePaperSourceBundle",
    "P31A_PRECOHORT_SELECTION_RULE",
    "PreCohortWindow",
    "build_moneyline_paper_source_bundle",
    "resolve_date_scope",
    "select_precohort_window",
    "STOP_P31A_NO_GENERALIZATION_WINDOW",
    "STOP_P31A_PRECOHORT_OVERLAP",
)
