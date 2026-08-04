"""Immutable final result observation for deterministic result attachment.

A FinalResultObservation captures the explicit final score of a completed game.
It requires no inference — every field must be supplied explicitly.

Only status FINAL is attachable. Tied final scores fail closed with
TIED_FINAL_RESULT_UNSUPPORTED.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .canonical_utc import parse_canonical_utc


@dataclass(frozen=True, slots=True)
class FinalResultObservation:
    """Immutable final-result observation for a completed game."""

    result_observation_id: str
    source_result_id: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    status: str
    result_observed_at_utc: str
    home_score: int
    away_score: int

    def __post_init__(self) -> None:
        _validate_final_result_observation(self)


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    """Require an explicit, non-empty, trimmed string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")


def _validate_final_result_observation(obs: FinalResultObservation) -> None:
    """Validate all invariants of a FinalResultObservation."""
    # All identity strings must be explicit and non-empty
    _validate_non_empty_string(obs.result_observation_id, "result_observation_id")
    _validate_non_empty_string(obs.source_result_id, "source_result_id")
    _validate_non_empty_string(obs.provider_namespace, "provider_namespace")
    _validate_non_empty_string(obs.provider_game_id, "provider_game_id")

    # game_number must be a positive integer; bool is not accepted
    if isinstance(obs.game_number, bool) or not isinstance(obs.game_number, int):
        raise TypeError(
            f"game_number must be a positive integer, got {type(obs.game_number).__name__}"
        )
    if obs.game_number < 1:
        raise ValueError(f"game_number must be positive, got {obs.game_number}")

    # status must be explicit
    _validate_non_empty_string(obs.status, "status")
    if obs.status != "FINAL":
        raise ValueError(
            f"only status FINAL is attachable, got {obs.status!r}"
        )

    # result_observed_at_utc must use the canonical UTC parser
    _validate_non_empty_string(obs.result_observed_at_utc, "result_observed_at_utc")
    parse_canonical_utc(obs.result_observed_at_utc)

    # home_score and away_score must be integers >= 0; bool is not accepted
    for field_name, value in [("home_score", obs.home_score), ("away_score", obs.away_score)]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer >= 0, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")

    # Tied FINAL scores fail closed
    if obs.home_score == obs.away_score:
        raise ValueError("TIED_FINAL_RESULT_UNSUPPORTED")


def compute_final_result_observation_id(
    *,
    source_result_id: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    status: str,
    result_observed_at_utc: str,
    home_score: int,
    away_score: int,
) -> str:
    """Compute a deterministic observation ID for a final result."""
    canonical_payload = {
        "away_score": away_score,
        "game_number": game_number,
        "home_score": home_score,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "result_observed_at_utc": result_observed_at_utc,
        "source_result_id": source_result_id,
        "status": status,
    }
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_final_result_observations(
    raw_bytes: bytes,
) -> list[FinalResultObservation]:
    """Load and validate final result observations from JSONL bytes.

    Rejects duplicate JSON keys, duplicate identity keys, and any
    observation that fails validation.
    """
    text = raw_bytes.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    observations: list[FinalResultObservation] = []
    seen_keys: dict[tuple[str, str, int], int] = {}

    for i, line in enumerate(lines):
        row = _validate_json_no_duplicate_keys(line, i)
        obs = _parse_final_result_row(row, i)
        key = (obs.provider_namespace, obs.provider_game_id, obs.game_number)
        if key in seen_keys:
            raise ValueError(
                f"AMBIGUOUS_FINAL_RESULT_OBSERVATION: duplicate identity key "
                f"{key} at lines {seen_keys[key] + 1} and {i + 1}"
            )
        seen_keys[key] = i
        observations.append(obs)

    return observations


def _validate_json_no_duplicate_keys(
    raw_line: str, line_index: int
) -> dict[str, Any]:
    """Parse JSON rejecting duplicate keys at all nesting levels."""

    def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, int] = {}
        for key, _ in pairs:
            if key in seen:
                raise ValueError(
                    f"Duplicate JSON key {key!r} in final result row {line_index + 1}"
                )
            seen[key] = 1
        return dict(pairs)

    try:
        return json.loads(raw_line, object_pairs_hook=_object_pairs_hook)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON in final result row {line_index + 1}: {exc}"
        ) from exc


_REQUIRED_FIELDS = (
    "source_result_id",
    "provider_namespace",
    "provider_game_id",
    "game_number",
    "status",
    "result_observed_at_utc",
    "home_score",
    "away_score",
)


def _parse_final_result_row(
    row: dict[str, Any], line_index: int
) -> FinalResultObservation:
    """Parse a single final result row into a FinalResultObservation."""
    for field in _REQUIRED_FIELDS:
        if field not in row:
            raise ValueError(
                f"Missing required field {field!r} in final result row {line_index + 1}"
            )

    source_result_id = row["source_result_id"]
    provider_namespace = row["provider_namespace"]
    provider_game_id = row["provider_game_id"]
    game_number = row["game_number"]
    status = row["status"]
    result_observed_at_utc = row["result_observed_at_utc"]
    home_score = row["home_score"]
    away_score = row["away_score"]

    result_observation_id = compute_final_result_observation_id(
        source_result_id=source_result_id,
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        game_number=game_number,
        status=status,
        result_observed_at_utc=result_observed_at_utc,
        home_score=home_score,
        away_score=away_score,
    )

    return FinalResultObservation(
        result_observation_id=result_observation_id,
        source_result_id=source_result_id,
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        game_number=game_number,
        status=status,
        result_observed_at_utc=result_observed_at_utc,
        home_score=home_score,
        away_score=away_score,
    )
