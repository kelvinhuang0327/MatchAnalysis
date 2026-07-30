"""Immutable canonical pregame-eligibility projection contracts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import re

from ...core.identity import MatchIdentity
from .participant_identity_resolution import (
    ResolvedScheduleIdentityCandidate,
    UnresolvedScheduleIdentityCandidate,
)
from .schedule_game_materialization import (
    ScheduleBaseballGameMaterialization,
)
from .schedule_observation import canonical_utc_timestamp
from .schedule_snapshot import ChainKey


SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION = (
    "schedule_pregame_eligibility_set_v1"
)

ELIGIBLE = "ELIGIBLE"
INELIGIBLE = "INELIGIBLE"

BEFORE_SCHEDULED_START = "BEFORE_SCHEDULED_START"
SCHEDULED_START_REACHED_OR_PASSED = (
    "SCHEDULED_START_REACHED_OR_PASSED"
)

_STATUS_REASON_PAIRS = {
    (ELIGIBLE, BEFORE_SCHEDULED_START),
    (INELIGIBLE, SCHEDULED_START_REACHED_OR_PASSED),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256"
        )


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_utc(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be normalized to UTC")


def _canonical_json_bytes(projection: dict[str, object]) -> bytes:
    return (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _match_identity_projection(
    identity: MatchIdentity,
) -> dict[str, object]:
    return {
        "sport": identity.sport,
        "league": identity.league,
        "season": identity.season,
        "canonical_game_id": identity.canonical_game_id,
        "home_participant": identity.home_participant,
        "away_participant": identity.away_participant,
        "game_discriminator": identity.game_discriminator,
    }


@dataclass(frozen=True, slots=True)
class SchedulePregameEligibilityDecision:
    """One exact P12 materialization and its time-only eligibility."""

    materialization: ScheduleBaseballGameMaterialization
    eligibility_status: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.materialization,
            ScheduleBaseballGameMaterialization,
        ):
            raise TypeError(
                "materialization must be a"
                " ScheduleBaseballGameMaterialization"
            )
        _require_explicit(self.eligibility_status, "eligibility_status")
        _require_explicit(self.reason, "reason")
        if (self.eligibility_status, self.reason) not in (
            _STATUS_REASON_PAIRS
        ):
            raise ValueError(
                "eligibility_status and reason must be one controlled pair"
            )


def _decision_projection(
    decision: SchedulePregameEligibilityDecision,
) -> dict[str, object]:
    materialization = decision.materialization
    return {
        "source_observation_id": materialization.source_observation_id,
        "match_identity": _match_identity_projection(
            materialization.match_identity
        ),
        "scheduled_start_utc": canonical_utc_timestamp(
            materialization.baseball_game.scheduled_start.value
        ),
        "eligibility_status": decision.eligibility_status,
        "reason": decision.reason,
    }


def _unresolved_candidate_projection(
    candidate: UnresolvedScheduleIdentityCandidate,
) -> dict[str, object]:
    return {
        "source_observation_id": candidate.source_observation_id,
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "reasons": list(candidate.reasons),
    }


def _resolved_candidate_projection(
    candidate: ResolvedScheduleIdentityCandidate,
) -> dict[str, object]:
    return {
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "scheduled_start_utc": canonical_utc_timestamp(
            candidate.scheduled_start_utc
        ),
        "official_local_date": candidate.official_local_date.isoformat(),
        "home_provider_participant_id": (
            candidate.home_provider_participant_id
        ),
        "away_provider_participant_id": (
            candidate.away_provider_participant_id
        ),
        "source_observation_id": candidate.source_observation_id,
        "source_raw_payload_sha256": (
            candidate.source_raw_payload_sha256
        ),
        "home_canonical_participant_id": (
            candidate.home_canonical_participant_id
        ),
        "away_canonical_participant_id": (
            candidate.away_canonical_participant_id
        ),
        "mapping_version": candidate.mapping_version,
    }


def compute_schedule_pregame_eligibility_set_fingerprint(
    *,
    as_of_utc: datetime,
    source_materialization_set_fingerprint: str,
    eligible_count: int,
    ineligible_count: int,
    unresolved_count: int,
    unavailable_count: int,
    authority_missing_count: int,
    eligible_decisions: tuple[
        SchedulePregameEligibilityDecision, ...
    ],
    ineligible_decisions: tuple[
        SchedulePregameEligibilityDecision, ...
    ],
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ],
    unavailable_chain_keys: tuple[ChainKey, ...],
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ],
) -> str:
    """Hash the exact public eligibility-set projection plus one final LF."""

    projection = {
        "schema_version": (
            SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION
        ),
        "as_of_utc": canonical_utc_timestamp(as_of_utc),
        "source_materialization_set_fingerprint": (
            source_materialization_set_fingerprint
        ),
        "eligible_count": eligible_count,
        "ineligible_count": ineligible_count,
        "unresolved_count": unresolved_count,
        "unavailable_count": unavailable_count,
        "authority_missing_count": authority_missing_count,
        "eligible_decisions": [
            _decision_projection(decision)
            for decision in eligible_decisions
        ],
        "ineligible_decisions": [
            _decision_projection(decision)
            for decision in ineligible_decisions
        ],
        "unresolved_candidates": [
            _unresolved_candidate_projection(candidate)
            for candidate in unresolved_candidates
        ],
        "unavailable_chain_keys": [
            {
                "provider_namespace": key[0],
                "provider_game_id": key[1],
                "game_number": key[2],
            }
            for key in unavailable_chain_keys
        ],
        "authority_missing_candidates": [
            _resolved_candidate_projection(candidate)
            for candidate in authority_missing_candidates
        ],
    }
    return sha256(_canonical_json_bytes(projection)).hexdigest()


def _candidate_key(
    candidate: (
        ResolvedScheduleIdentityCandidate
        | UnresolvedScheduleIdentityCandidate
    ),
) -> ChainKey:
    return (
        candidate.provider_namespace,
        candidate.provider_game_id,
        candidate.game_number,
    )


@dataclass(frozen=True, slots=True)
class SchedulePregameEligibilitySet:
    """Deterministic P12-to-pregame-eligibility result."""

    as_of_utc: datetime
    source_materialization_set_fingerprint: str
    eligible_decisions: tuple[
        SchedulePregameEligibilityDecision, ...
    ]
    ineligible_decisions: tuple[
        SchedulePregameEligibilityDecision, ...
    ]
    unresolved_candidates: tuple[
        UnresolvedScheduleIdentityCandidate, ...
    ]
    unavailable_chain_keys: tuple[ChainKey, ...]
    authority_missing_candidates: tuple[
        ResolvedScheduleIdentityCandidate, ...
    ]
    eligible_count: int
    ineligible_count: int
    unresolved_count: int
    unavailable_count: int
    authority_missing_count: int
    eligibility_set_fingerprint: str
    schema_version: str

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must be exactly"
                " schedule_pregame_eligibility_set_v1"
            )
        _require_utc(self.as_of_utc, "as_of_utc")
        _require_sha256(
            self.source_materialization_set_fingerprint,
            "source_materialization_set_fingerprint",
        )
        _require_sha256(
            self.eligibility_set_fingerprint,
            "eligibility_set_fingerprint",
        )

        decision_partitions = (
            ("eligible_decisions", ELIGIBLE),
            ("ineligible_decisions", INELIGIBLE),
        )
        all_decision_observation_ids: list[str] = []
        for field_name, expected_status in decision_partitions:
            decisions = getattr(self, field_name)
            if not isinstance(decisions, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(
                not isinstance(
                    decision,
                    SchedulePregameEligibilityDecision,
                )
                for decision in decisions
            ):
                raise TypeError(
                    f"every {field_name} value must be a"
                    " SchedulePregameEligibilityDecision"
                )
            if any(
                decision.eligibility_status != expected_status
                for decision in decisions
            ):
                raise ValueError(
                    f"every {field_name} value must have status"
                    f" {expected_status}"
                )
            for decision in decisions:
                scheduled_start = (
                    decision.materialization
                    .baseball_game.scheduled_start.value
                )
                is_before_start = self.as_of_utc < scheduled_start
                if (
                    expected_status == ELIGIBLE
                    and not is_before_start
                ) or (
                    expected_status == INELIGIBLE
                    and is_before_start
                ):
                    raise ValueError(
                        f"every {field_name} value must match as_of_utc"
                        " against its canonical scheduled start"
                    )
            observation_ids = [
                decision.materialization.source_observation_id
                for decision in decisions
            ]
            if observation_ids != sorted(observation_ids):
                raise ValueError(
                    f"{field_name} must preserve P12 materialization order"
                )
            all_decision_observation_ids.extend(observation_ids)

        candidate_partitions = (
            (
                "unresolved_candidates",
                UnresolvedScheduleIdentityCandidate,
            ),
            (
                "authority_missing_candidates",
                ResolvedScheduleIdentityCandidate,
            ),
        )
        candidate_observation_ids: list[str] = []
        for field_name, expected_type in candidate_partitions:
            candidates = getattr(self, field_name)
            if not isinstance(candidates, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(
                not isinstance(candidate, expected_type)
                for candidate in candidates
            ):
                raise TypeError(
                    f"every {field_name} value must be a"
                    f" {expected_type.__name__}"
                )
            keys = [_candidate_key(candidate) for candidate in candidates]
            if keys != sorted(keys):
                raise ValueError(
                    f"{field_name} must preserve P12 ordering"
                )
            candidate_observation_ids.extend(
                candidate.source_observation_id
                for candidate in candidates
            )

        if not isinstance(self.unavailable_chain_keys, tuple):
            raise TypeError("unavailable_chain_keys must be a tuple")
        for key in self.unavailable_chain_keys:
            if not isinstance(key, tuple) or len(key) != 3:
                raise TypeError("every unavailable chain key must be a tuple")
            _require_explicit(key[0], "provider_namespace")
            _require_explicit(key[1], "provider_game_id")
            _require_positive_integer(key[2], "game_number")
        if list(self.unavailable_chain_keys) != sorted(
            self.unavailable_chain_keys
        ):
            raise ValueError(
                "unavailable_chain_keys must preserve P12 ordering"
            )

        observation_ids = (
            all_decision_observation_ids + candidate_observation_ids
        )
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(
                "a source observation must not appear in more than one"
                " eligibility partition"
            )

        expected_counts = {
            "eligible_count": len(self.eligible_decisions),
            "ineligible_count": len(self.ineligible_decisions),
            "unresolved_count": len(self.unresolved_candidates),
            "unavailable_count": len(self.unavailable_chain_keys),
            "authority_missing_count": len(
                self.authority_missing_candidates
            ),
        }
        for field_name, expected in expected_counts.items():
            value = getattr(self, field_name)
            _require_non_negative_integer(value, field_name)
            if value != expected:
                raise ValueError(
                    f"{field_name} must match its result partition"
                )

        expected_fingerprint = (
            compute_schedule_pregame_eligibility_set_fingerprint(
                as_of_utc=self.as_of_utc,
                source_materialization_set_fingerprint=(
                    self.source_materialization_set_fingerprint
                ),
                eligible_count=self.eligible_count,
                ineligible_count=self.ineligible_count,
                unresolved_count=self.unresolved_count,
                unavailable_count=self.unavailable_count,
                authority_missing_count=self.authority_missing_count,
                eligible_decisions=self.eligible_decisions,
                ineligible_decisions=self.ineligible_decisions,
                unresolved_candidates=self.unresolved_candidates,
                unavailable_chain_keys=self.unavailable_chain_keys,
                authority_missing_candidates=(
                    self.authority_missing_candidates
                ),
            )
        )
        if self.eligibility_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "eligibility_set_fingerprint must match the canonical"
                " eligibility-set projection"
            )
