"""Acquire the minimum bounded MLB-owned inputs needed by P23F2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    RawSourceRecord,
    canonical_json_bytes,
    fetch_json_bytes,
    load_json,
    normalize_boxscore_payload,
    normalize_pitcher_game_log_payload,
    normalize_schedule_payload,
    sha256_bytes,
    write_normalized_json,
    write_raw_response,
)


START_DATE = "2026-03-25"
END_DATE = "2026-06-30"
# June 8-9 is the earliest contiguous 2026 interval with more than one game
# for which every official starter has a prior 2026 starting appearance.
VALIDATION_START = "2026-06-08"
VALIDATION_END = "2026-06-09"
STOP_SOURCE_INSUFFICIENT = "STOP_MATCHANALYSIS_P23F2_OFFICIAL_SOURCE_INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    schedule_rows: tuple[dict[str, Any], ...]
    target_boxscore_rows: tuple[dict[str, Any], ...]
    pitcher_game_log_rows: tuple[dict[str, Any], ...]
    source_records: tuple[RawSourceRecord, ...]
    normalized_hashes: dict[str, str]


def _fetch_or_load(
    *,
    raw_root: Path,
    relative_path: str,
    url: str,
    query: dict[str, str],
    acquired_at_utc: str,
    opener: Any,
    source_scope: str,
) -> tuple[dict[str, Any], RawSourceRecord]:
    path = raw_root / relative_path
    if path.exists():
        raw = path.read_bytes()
        request_url = url + "?" + "&".join(f"{key}={value}" for key, value in query.items())
    else:
        raw, request_url = fetch_json_bytes(url, query=query, opener=opener)
        write_raw_response(raw_root=raw_root, relative_path=relative_path, raw=raw)
    return load_json(path), RawSourceRecord(
        path=str(Path("data/fixtures/p23f2_official_2026_history") / "raw" / relative_path),
        url=request_url,
        scope=source_scope,
        acquired_at_utc=acquired_at_utc,
        sha256=sha256_bytes(raw),
    )


def acquire_official_history(
    *,
    raw_root: str | Path,
    normalized_root: str | Path,
    acquired_at_utc: datetime,
    opener: Any = None,
) -> AcquisitionResult:
    """Acquire once and materialize deterministic normalized source inputs."""

    raw_root = Path(raw_root)
    normalized_root = Path(normalized_root)
    acquired_at = acquired_at_utc.astimezone(UTC)
    opener = opener
    if opener is None:
        from ...infrastructure.providers.mlb_official_historical_source import _default_opener

        opener = _default_opener
    acquired_at_text = acquired_at.isoformat().replace("+00:00", "Z")
    source_records: list[RawSourceRecord] = []

    schedule_payload, schedule_record = _fetch_or_load(
        raw_root=raw_root,
        relative_path=f"schedule_{START_DATE}_{END_DATE}.json",
        url=f"{MLB_STATS_API_BASE}/schedule",
        query={
            "sportId": "1",
            "startDate": START_DATE,
            "endDate": END_DATE,
            "gameType": "R",
            "hydrate": "team",
        },
        acquired_at_utc=acquired_at_text,
        opener=opener,
        source_scope=f"regular-season schedule {START_DATE}..{END_DATE}",
    )
    source_records.append(schedule_record)
    schedule_rows = normalize_schedule_payload(schedule_payload)
    target_games = tuple(
        row for row in schedule_rows
        if VALIDATION_START <= row["official_date"] <= VALIDATION_END
    )
    if len(target_games) < 2 or any(not row["final"] for row in target_games):
        raise RuntimeError(STOP_SOURCE_INSUFFICIENT)

    boxscore_rows: list[dict[str, Any]] = []
    for game in target_games:
        relative = f"boxscores/{game['game_pk']}.json"
        payload, record = _fetch_or_load(
            raw_root=raw_root,
            relative_path=relative,
            url=f"{MLB_STATS_API_BASE}/game/{game['game_pk']}/boxscore",
            query={},
            acquired_at_utc=acquired_at_text,
            opener=opener,
            source_scope=f"official boxscore game {game['game_pk']}",
        )
        source_records.append(record)
        boxscore_rows.append(normalize_boxscore_payload(payload, game=game))

    starter_ids = sorted({
        starter["player_id"]
        for row in boxscore_rows
        for starter in (row["home_starter"], row["away_starter"])
    })
    log_rows: list[dict[str, Any]] = []
    for player_id in starter_ids:
        relative = f"pitcher_game_logs/{player_id}.json"
        payload, record = _fetch_or_load(
            raw_root=raw_root,
            relative_path=relative,
            url=f"{MLB_STATS_API_BASE}/people/{player_id}/stats",
            query={"stats": "gameLog", "group": "pitching", "season": "2026", "gameType": "R"},
            acquired_at_utc=acquired_at_text,
            opener=opener,
            source_scope=f"2026 regular-season pitching game log player {player_id}",
        )
        source_records.append(record)
        log_rows.extend(normalize_pitcher_game_log_payload(payload, player_id=player_id))

    normalized_hashes = {
        "schedule.jsonl": write_normalized_json(normalized_root / "schedule.jsonl", schedule_rows),
        "target_boxscores.jsonl": write_normalized_json(normalized_root / "target_boxscores.jsonl", boxscore_rows),
        "pitcher_game_logs.jsonl": write_normalized_json(
            normalized_root / "pitcher_game_logs.jsonl",
            sorted(log_rows, key=lambda row: (row["player_id"], row["date"], row["game_pk"])),
        ),
    }
    return AcquisitionResult(
        schedule_rows=schedule_rows,
        target_boxscore_rows=tuple(boxscore_rows),
        pitcher_game_log_rows=tuple(log_rows),
        source_records=tuple(source_records),
        normalized_hashes=normalized_hashes,
    )


def load_normalized_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"normalized row must be an object: {path}")
        rows.append(value)
    if not rows:
        raise ValueError(f"normalized file is empty: {path}")
    return tuple(rows)


def acquire_official_future_fold(
    *,
    repository_root: str | Path,
    fold_id: str,
    validation_start: str,
    validation_end: str,
    raw_root: str | Path,
    normalized_root: str | Path,
    acquired_at_utc: datetime,
    opener: Any = None,
) -> AcquisitionResult:
    """Acquire only new target sources while reusing the committed schedule history."""

    repository_root = Path(repository_root)
    raw_root = Path(raw_root)
    normalized_root = Path(normalized_root)
    acquired_at = acquired_at_utc.astimezone(UTC)
    opener = opener
    if opener is None:
        from ...infrastructure.providers.mlb_official_historical_source import _default_opener

        opener = _default_opener

    schedule_rows = load_normalized_rows(
        repository_root
        / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
    )
    target_games = tuple(
        sorted(
            (
                row
                for row in schedule_rows
                if validation_start <= row["official_date"] <= validation_end
            ),
            key=lambda row: (
                row["scheduled_start_utc"],
                row["game_number"],
                row["game_pk"],
            ),
        )
    )
    if len(target_games) < 2 or any(not row["final"] for row in target_games):
        raise RuntimeError(STOP_SOURCE_INSUFFICIENT)

    source_manifest = json.loads(
        (
            repository_root
            / "report/p23f2_official_future_fold/source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    schedule_record = next(
        record
        for record in source_manifest["records"]
        if record["path"].endswith("/raw/schedule_2026-03-25_2026-06-30.json")
    )
    source_records: list[RawSourceRecord] = [
        RawSourceRecord(
            path=str(schedule_record["path"]),
            url=str(schedule_record["url"]),
            scope=(
                f"reused official schedule rows {validation_start}..{validation_end}"
            ),
            acquired_at_utc=str(schedule_record["acquired_at_utc"]),
            sha256=str(schedule_record["sha256"]),
        )
    ]
    acquired_at_text = acquired_at.isoformat().replace("+00:00", "Z")

    def fetch_or_load(
        *,
        relative_path: str,
        url: str,
        query: dict[str, str],
        source_scope: str,
    ) -> dict[str, Any]:
        path = raw_root / relative_path
        if path.exists():
            raw = path.read_bytes()
            request_url = url + "?" + "&".join(
                f"{key}={value}" for key, value in query.items()
            )
        else:
            raw, request_url = fetch_json_bytes(url, query=query, opener=opener)
            write_raw_response(raw_root=raw_root, relative_path=relative_path, raw=raw)
        source_records.append(
            RawSourceRecord(
                path=str(
                    Path("data/fixtures/p23b_future_folds")
                    / fold_id
                    / "raw"
                    / relative_path
                ),
                url=request_url,
                scope=source_scope,
                acquired_at_utc=acquired_at_text,
                sha256=sha256_bytes(raw),
            )
        )
        return load_json(path)

    boxscore_rows: list[dict[str, Any]] = []
    for game in target_games:
        payload = fetch_or_load(
            relative_path=f"boxscores/{game['game_pk']}.json",
            url=f"{MLB_STATS_API_BASE}/game/{game['game_pk']}/boxscore",
            query={},
            source_scope=f"official boxscore game {game['game_pk']}",
        )
        boxscore_rows.append(normalize_boxscore_payload(payload, game=game))

    starter_ids = sorted(
        {
            starter["player_id"]
            for row in boxscore_rows
            for starter in (row["home_starter"], row["away_starter"])
        }
    )
    log_rows: list[dict[str, Any]] = []
    for player_id in starter_ids:
        payload = fetch_or_load(
            relative_path=f"pitcher_game_logs/{player_id}.json",
            url=f"{MLB_STATS_API_BASE}/people/{player_id}/stats",
            query={
                "stats": "gameLog",
                "group": "pitching",
                "season": "2026",
                "gameType": "R",
            },
            source_scope=f"2026 regular-season pitching game log player {player_id}",
        )
        log_rows.extend(
            normalize_pitcher_game_log_payload(payload, player_id=player_id)
        )

    def write_normalized_immutable(
        path: Path,
        rows: tuple[Mapping[str, Any], ...],
    ) -> str:
        raw = b"".join(canonical_json_bytes(row) for row in rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"immutable normalized source changed: {path}")
        if not path.exists():
            path.write_bytes(raw)
        return sha256_bytes(raw)

    normalized_hashes = {
        "schedule.jsonl": write_normalized_immutable(
            normalized_root / "schedule.jsonl", target_games
        ),
        "target_boxscores.jsonl": write_normalized_immutable(
            normalized_root / "target_boxscores.jsonl", tuple(boxscore_rows)
        ),
        "pitcher_game_logs.jsonl": write_normalized_immutable(
            normalized_root / "pitcher_game_logs.jsonl",
            tuple(
                sorted(
                    log_rows,
                    key=lambda row: (
                        row["player_id"],
                        row["date"],
                        row["game_pk"],
                    ),
                )
            ),
        ),
    }
    return AcquisitionResult(
        schedule_rows=schedule_rows,
        target_boxscore_rows=tuple(boxscore_rows),
        pitcher_game_log_rows=tuple(log_rows),
        source_records=tuple(source_records),
        normalized_hashes=normalized_hashes,
    )


__all__ = (
    "AcquisitionResult",
    "END_DATE",
    "START_DATE",
    "VALIDATION_END",
    "VALIDATION_START",
    "acquire_official_history",
    "acquire_official_future_fold",
    "load_normalized_rows",
)
