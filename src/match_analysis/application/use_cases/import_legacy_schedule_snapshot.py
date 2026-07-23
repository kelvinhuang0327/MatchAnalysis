"""Import a validated P84B schedule into an in-memory quarantine."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

from ...baseball.domain.schedule import (
    PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT,
    DateTeamCollisionGroup,
    LegacyDiagnosticScheduleCandidate,
    ProviderGameReference,
    ScheduleQuarantineReason,
    UNIVERSAL_QUARANTINE_REASONS,
)
from ..ports.legacy_schedule_source import (
    LegacyScheduleRow,
    LegacyScheduleSource,
)


PINNED_P84B_ARTIFACT_SHA256 = (
    "d970c6ce4c16d4ca7aaebcebd2aa5aaea7834934b2a92d59ebb281a90ee10b69"
)
PINNED_P84B_SEMANTIC_FINGERPRINT = (
    "4b219859cd1cd0fc19d75f1323684f4f8816115bf68677d1e25eb409f6ced077"
)
SEMANTIC_FINGERPRINT_FIELDS = (
    "provider_namespace",
    "provider_game_id",
    "source_game_id",
    "sport",
    "league",
    "season",
    "game_date",
    "source_home_team",
    "source_away_team",
    "legacy_collection_marker_utc",
    "diagnostic_status",
    "quarantine_reasons",
)
SEMANTIC_FINGERPRINT_ENCODING = "utf-8"
SEMANTIC_FINGERPRINT_SOURCE_ORDER = (
    "provider_namespace,provider_game_id,source_game_id"
)
SEMANTIC_FINGERPRINT_NEWLINE = "LF_AFTER_EVERY_PROJECTION"


@dataclass(frozen=True, slots=True)
class LegacyScheduleImportResult:
    """Immutable result of an in-memory, non-promoting schedule import."""

    artifact_sha256: str
    row_count: int
    unique_provider_reference_count: int
    provider_game_references: tuple[ProviderGameReference, ...]
    candidates: tuple[LegacyDiagnosticScheduleCandidate, ...]
    collision_groups: tuple[DateTeamCollisionGroup, ...]
    collision_affected_row_count: int
    semantic_fingerprint: str
    limitations: tuple[str, ...]
    quarantine_counts: tuple[tuple[ScheduleQuarantineReason, int], ...]
    match_identity_count: int
    trusted_schedule_observation_count: int
    baseball_game_count: int
    pregame_eligible_context_count: int


def candidate_projection(
    candidate: LegacyDiagnosticScheduleCandidate,
) -> dict[str, object]:
    """Return the exact authorized semantic projection."""

    reference = candidate.provider_reference
    return {
        "provider_namespace": reference.provider_namespace,
        "provider_game_id": reference.provider_game_id,
        "source_game_id": reference.source_game_id,
        "sport": candidate.sport,
        "league": candidate.league,
        "season": candidate.season,
        "game_date": candidate.game_date,
        "source_home_team": candidate.source_home_team,
        "source_away_team": candidate.source_away_team,
        "legacy_collection_marker_utc": (
            candidate.legacy_collection_marker_utc
        ),
        "diagnostic_status": candidate.diagnostic_status,
        "quarantine_reasons": [
            reason.value for reason in candidate.quarantine_reasons
        ],
    }


def semantic_projection_bytes(
    candidates: tuple[LegacyDiagnosticScheduleCandidate, ...],
) -> bytes:
    """Encode sorted projected JSONL with an LF after every row."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.provider_reference.provider_namespace,
            candidate.provider_reference.provider_game_id,
            candidate.provider_reference.source_game_id,
        ),
    )
    return "".join(
        json.dumps(
            candidate_projection(candidate),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for candidate in ordered
    ).encode(SEMANTIC_FINGERPRINT_ENCODING)


def _collision_key(row: LegacyScheduleRow) -> tuple[str, str, str]:
    return (
        row.game_date,
        row.source_home_team,
        row.source_away_team,
    )


def import_legacy_schedule_snapshot(
    source: LegacyScheduleSource,
) -> LegacyScheduleImportResult:
    """Create date-only diagnostic candidates without persistence or promotion."""

    snapshot = source.load()
    grouped_ids: defaultdict[tuple[str, str, str], list[str]] = defaultdict(
        list
    )
    for row in snapshot.rows:
        grouped_ids[_collision_key(row)].append(
            row.provider_reference.source_game_id
        )
    collision_keys = {
        key for key, source_ids in grouped_ids.items() if len(source_ids) > 1
    }
    collision_groups = tuple(
        DateTeamCollisionGroup(
            game_date=key[0],
            source_home_team=key[1],
            source_away_team=key[2],
            source_game_ids=tuple(sorted(grouped_ids[key])),
        )
        for key in sorted(collision_keys)
    )

    candidates = tuple(
        LegacyDiagnosticScheduleCandidate(
            provider_reference=row.provider_reference,
            season=row.season,
            game_date=row.game_date,
            source_home_team=row.source_home_team,
            source_away_team=row.source_away_team,
            legacy_collection_marker_utc=(
                row.legacy_collection_marker_utc
            ),
            quarantine_reasons=(
                (
                    *UNIVERSAL_QUARANTINE_REASONS,
                    ScheduleQuarantineReason.DATE_TEAM_COLLISION,
                )
                if _collision_key(row) in collision_keys
                else UNIVERSAL_QUARANTINE_REASONS
            ),
        )
        for row in sorted(
            snapshot.rows,
            key=lambda item: (
                item.provider_reference.provider_namespace,
                item.provider_reference.provider_game_id,
                item.provider_reference.source_game_id,
            ),
        )
    )
    references = tuple(
        candidate.provider_reference for candidate in candidates
    )
    unique_reference_count = len(set(references))
    if unique_reference_count != len(references):
        raise ValueError(
            "validated schedule source unexpectedly contains duplicate references"
        )
    collision_affected_row_count = sum(
        len(group.source_game_ids) for group in collision_groups
    )
    projection = semantic_projection_bytes(candidates)
    fingerprint = sha256(projection).hexdigest()
    quarantine_counts = Counter(
        reason
        for candidate in candidates
        for reason in candidate.quarantine_reasons
    )

    return LegacyScheduleImportResult(
        artifact_sha256=snapshot.artifact_sha256,
        row_count=len(candidates),
        unique_provider_reference_count=unique_reference_count,
        provider_game_references=references,
        candidates=candidates,
        collision_groups=collision_groups,
        collision_affected_row_count=collision_affected_row_count,
        semantic_fingerprint=fingerprint,
        limitations=(
            PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT,
            "DATE_ONLY_ROWS_ARE_NOT_CANONICAL_SCHEDULE_STATE",
        ),
        quarantine_counts=tuple(
            (reason, quarantine_counts[reason])
            for reason in ScheduleQuarantineReason
            if quarantine_counts[reason]
        ),
        match_identity_count=0,
        trusted_schedule_observation_count=0,
        baseball_game_count=0,
        pregame_eligible_context_count=0,
    )
