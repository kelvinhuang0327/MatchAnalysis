"""Capture one provider schedule observation in memory."""

from datetime import datetime, timezone
from hashlib import sha256

from ...baseball.domain.schedule_observation import (
    ScheduleSourceObservation,
    compute_schedule_observation_id,
)
from ..ports.schedule_observation_source import ScheduleObservationSource


def _normalize_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def capture_schedule_observation(
    source: ScheduleObservationSource,
    previous_observation: ScheduleSourceObservation | None = None,
) -> ScheduleSourceObservation:
    """Validate one capture and return immutable append-only evidence."""

    capture = source.capture()
    if not isinstance(capture.raw_payload_bytes, bytes):
        raise TypeError("raw_payload_bytes must be bytes")
    if not capture.raw_payload_bytes:
        raise ValueError("raw_payload_bytes must not be empty")
    actual_raw_sha256 = sha256(capture.raw_payload_bytes).hexdigest()
    if capture.raw_payload_sha256 != actual_raw_sha256:
        raise ValueError("raw_payload_sha256 must match exact payload bytes")

    scheduled_start_utc = _normalize_utc(
        capture.scheduled_start_utc,
        "scheduled_start_utc",
    )
    response_received_at_utc = _normalize_utc(
        capture.response_received_at_utc,
        "response_received_at_utc",
    )
    ingested_at_utc = _normalize_utc(
        capture.ingested_at_utc,
        "ingested_at_utc",
    )
    if response_received_at_utc > ingested_at_utc:
        raise ValueError(
            "response_received_at_utc must not follow ingested_at_utc"
        )

    observation_id = compute_schedule_observation_id(
        provider_namespace=capture.provider_namespace,
        provider_game_id=capture.provider_game_id,
        scheduled_start_utc=scheduled_start_utc,
        official_local_date=capture.official_local_date,
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=capture.provider_status_code,
        provider_detailed_status=capture.provider_detailed_status,
        game_number=capture.game_number,
        home_provider_participant_id=(
            capture.home_provider_participant_id
        ),
        away_provider_participant_id=(
            capture.away_provider_participant_id
        ),
        endpoint_id=capture.endpoint_id,
        parser_version=capture.parser_version,
        schema_version=capture.schema_version,
        raw_payload_sha256=capture.raw_payload_sha256,
        supersedes_observation_id=capture.supersedes_observation_id,
    )
    observation = ScheduleSourceObservation(
        observation_id=observation_id,
        provider_namespace=capture.provider_namespace,
        provider_game_id=capture.provider_game_id,
        scheduled_start_utc=scheduled_start_utc,
        official_local_date=capture.official_local_date,
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=capture.provider_status_code,
        provider_detailed_status=capture.provider_detailed_status,
        game_number=capture.game_number,
        home_provider_participant_id=(
            capture.home_provider_participant_id
        ),
        away_provider_participant_id=(
            capture.away_provider_participant_id
        ),
        endpoint_id=capture.endpoint_id,
        parser_version=capture.parser_version,
        schema_version=capture.schema_version,
        raw_payload_bytes=capture.raw_payload_bytes,
        raw_payload_sha256=capture.raw_payload_sha256,
        supersedes_observation_id=capture.supersedes_observation_id,
    )
    _validate_revision(observation, previous_observation)
    return observation


def _validate_revision(
    observation: ScheduleSourceObservation,
    previous_observation: ScheduleSourceObservation | None,
) -> None:
    superseded_id = observation.supersedes_observation_id
    if superseded_id is None:
        if previous_observation is not None:
            raise ValueError(
                "a non-revision must not receive a previous observation"
            )
        return
    if previous_observation is None:
        raise ValueError("a revision requires the previous observation")
    if superseded_id != previous_observation.observation_id:
        raise ValueError(
            "supersedes_observation_id must equal the previous observation ID"
        )
    if observation.provider_namespace != previous_observation.provider_namespace:
        raise ValueError("revision provider namespace must remain unchanged")
    if observation.provider_game_id != previous_observation.provider_game_id:
        raise ValueError("revision provider game ID must remain unchanged")
    if observation.game_number != previous_observation.game_number:
        raise ValueError("revision game number must remain unchanged")
    if (
        observation.response_received_at_utc
        <= previous_observation.response_received_at_utc
    ):
        raise ValueError(
            "revision response-received time must be strictly later"
        )
    if observation.observation_id == previous_observation.observation_id:
        raise ValueError("a revision must produce a distinct observation ID")
