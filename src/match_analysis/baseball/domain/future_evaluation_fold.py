"""P23F2 future-only, point-in-time-safe evaluation fold contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Any, Mapping


FUTURE_FOLD_SCHEMA_VERSION = "p23f2.future_evaluation_fold.v1"
FEATURE_NAMES = ("recent_win_rate_delta", "starter_era_delta")
TRAINING_INFORMATION_BOUNDARY_UTC = "2026-03-12T06:29:35.016973Z"
RECENT_GAME_WINDOW = 15
MIN_HISTORY_MONTHS = 2
FOLD_ID = "wf_004"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _sha(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class FutureFeatureRow:
    provider_game_id: str
    game_pk: int
    game_number: int
    official_date: str
    scheduled_start_utc: str
    feature_as_of_utc: str
    home_team: str
    away_team: str
    home_starter_id: int
    home_starter_name: str
    away_starter_id: int
    away_starter_name: str
    recent_win_rate_delta: str
    starter_era_delta: str
    feature_fingerprint: str = ""

    def projection(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": FUTURE_FOLD_SCHEMA_VERSION,
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": self.provider_game_id,
            "game_pk": self.game_pk,
            "game_number": self.game_number,
            "official_date": self.official_date,
            "scheduled_start_utc": self.scheduled_start_utc,
            "feature_as_of_utc": self.feature_as_of_utc,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_starter": {"id": self.home_starter_id, "name": self.home_starter_name},
            "away_starter": {"id": self.away_starter_id, "name": self.away_starter_name},
            "features": {
                "recent_win_rate_delta": self.recent_win_rate_delta,
                "starter_era_delta": self.starter_era_delta,
            },
        }
        if include_fingerprint:
            value["feature_fingerprint"] = self.feature_fingerprint
        return value

    def with_fingerprint(self) -> "FutureFeatureRow":
        return replace(
            self,
            feature_fingerprint=_sha(self.projection(include_fingerprint=False)),
        )


@dataclass(frozen=True, slots=True)
class FutureResultRow:
    provider_game_id: str
    game_pk: int
    game_number: int
    scheduled_start_utc: str
    home_score: int
    away_score: int
    status: str
    source_result_id: str

    def projection(self) -> dict[str, Any]:
        return {
            "schema_version": FUTURE_FOLD_SCHEMA_VERSION,
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": self.provider_game_id,
            "game_pk": self.game_pk,
            "game_number": self.game_number,
            "scheduled_start_utc": self.scheduled_start_utc,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "status": self.status,
            "source_result_id": self.source_result_id,
        }


@dataclass(frozen=True, slots=True)
class FutureEvaluationFold:
    fold_id: str
    training_information_boundary_utc: str
    validation_start: str
    validation_end: str
    feature_rows: tuple[FutureFeatureRow, ...]
    result_rows: tuple[FutureResultRow, ...]
    source_manifest_fingerprint: str
    feature_fingerprint: str
    result_fingerprint: str
    fold_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.fold_id != FOLD_ID:
            raise ValueError("P23F2 must materialize exactly wf_004")
        if len(self.feature_rows) < 2:
            raise ValueError("future fold requires more than one feature row")
        feature_ids = [row.provider_game_id for row in self.feature_rows]
        result_ids = [row.provider_game_id for row in self.result_rows]
        feature_order = [
            (row.scheduled_start_utc, row.game_number, row.game_pk)
            for row in self.feature_rows
        ]
        if feature_order != sorted(feature_order):
            raise ValueError("feature rows must be in deterministic order")
        if set(feature_ids) != set(result_ids) or len(result_ids) != len(feature_ids):
            raise ValueError("results must match feature identities exactly")
        boundary = _parse(self.training_information_boundary_utc)
        for row in self.feature_rows:
            scheduled = _parse(row.scheduled_start_utc)
            if scheduled <= boundary or _parse(row.feature_as_of_utc) >= scheduled:
                raise ValueError("future feature row violates strict PIT boundary")
            if tuple(row.projection(include_fingerprint=False)["features"]) != FEATURE_NAMES:
                raise ValueError("feature schema must remain exactly P13")
        for result in self.result_rows:
            if result.status != "Final":
                raise ValueError("future fold results must be final")

    def manifest_projection(self) -> dict[str, Any]:
        return {
            "schema_version": FUTURE_FOLD_SCHEMA_VERSION,
            "fold_id": self.fold_id,
            "training_information_boundary_utc": self.training_information_boundary_utc,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "feature_names": list(FEATURE_NAMES),
            "game_count": len(self.feature_rows),
            "scheduled_start_range": [self.feature_rows[0].scheduled_start_utc, self.feature_rows[-1].scheduled_start_utc],
            "feature_as_of_range": [self.feature_rows[0].feature_as_of_utc, self.feature_rows[-1].feature_as_of_utc],
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "strict_future": True,
            "external_source": True,
            "model_evaluated": False,
            "model_promoted": False,
            "production_ready": False,
        }


def fingerprint_rows(rows: tuple[Mapping[str, Any], ...]) -> str:
    return sha256(b"".join(_canonical_bytes(row) for row in rows)).hexdigest()


def fingerprint_manifest(manifest: Mapping[str, Any]) -> str:
    return _sha(manifest)


__all__ = (
    "FEATURE_NAMES",
    "FOLD_ID",
    "FUTURE_FOLD_SCHEMA_VERSION",
    "FutureEvaluationFold",
    "FutureFeatureRow",
    "FutureResultRow",
    "MIN_HISTORY_MONTHS",
    "RECENT_GAME_WINDOW",
    "TRAINING_INFORMATION_BOUNDARY_UTC",
    "fingerprint_manifest",
    "fingerprint_rows",
)
