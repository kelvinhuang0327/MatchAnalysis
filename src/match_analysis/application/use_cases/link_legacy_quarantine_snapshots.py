"""Join existing P83E and P84B quarantine import results by exact source game ID.

Pure in-memory application logic: no file, path, or current-time dependency,
no persistence, and no promotion of any kind.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

from ...baseball.domain.quarantine_link import LegacyDiagnosticPredictionScheduleLink
from ...baseball.domain.schedule import ScheduleQuarantineReason
from ...core.provenance import ArtifactProvenance
from .import_legacy_prediction_snapshot import LegacyPredictionImportResult
from .import_legacy_schedule_snapshot import LegacyScheduleImportResult


JOINT_SCHEMA_VERSION = "p83e_p84b_quarantine_link_v1"

NO_PROMOTION_LIMITATION = (
    "This link performs no MatchIdentity resolution, trusted schedule "
    "promotion, or canonical prediction promotion; all five promotion "
    "counts are fixed at zero."
)
DIAGNOSTIC_ONLY_LIMITATION = (
    "Both nested candidates remain exactly as quarantined by their own "
    "importers; this join adds no new time, identity, or outcome evidence."
)


@dataclass(frozen=True, slots=True)
class LegacyQuarantineLinkResult:
    """Immutable result of joining quarantined prediction and schedule imports."""

    links: tuple[LegacyDiagnosticPredictionScheduleLink, ...]
    linked_count: int
    prediction_missing_schedule_ids: tuple[str, ...]
    schedule_only_source_ids: tuple[str, ...]
    collision_affected_linked_source_ids: tuple[str, ...]
    collision_affected_linked_count: int
    p83e_artifact_provenance: ArtifactProvenance
    p84b_artifact_sha256: str
    p83e_semantic_fingerprint: str
    p84b_semantic_fingerprint: str
    joint_semantic_fingerprint: str
    match_identity_count: int
    trusted_schedule_observation_count: int
    baseball_game_count: int
    canonical_prediction_count: int
    pregame_eligible_context_count: int
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.links, tuple) or not all(
            isinstance(link, LegacyDiagnosticPredictionScheduleLink)
            for link in self.links
        ):
            raise TypeError("links must be a tuple of LegacyDiagnosticPredictionScheduleLink")
        link_ids = tuple(link.source_game_id for link in self.links)
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("links must not contain duplicate source_game_id values")
        if link_ids != tuple(sorted(link_ids)):
            raise ValueError("links must be ordered by ascending source_game_id")
        if self.linked_count != len(self.links):
            raise ValueError("linked_count must equal len(links)")

        for field_name in (
            "prediction_missing_schedule_ids",
            "schedule_only_source_ids",
            "collision_affected_linked_source_ids",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) for item in value
            ):
                raise TypeError(f"{field_name} must be a tuple of strings")
            if value != tuple(sorted(value)):
                raise ValueError(f"{field_name} must be lexicographically sorted")
            if len(set(value)) != len(value):
                raise ValueError(f"{field_name} must not contain duplicates")

        if self.collision_affected_linked_count != len(
            self.collision_affected_linked_source_ids
        ):
            raise ValueError(
                "collision_affected_linked_count must equal "
                "len(collision_affected_linked_source_ids)"
            )
        if not set(self.collision_affected_linked_source_ids) <= set(link_ids):
            raise ValueError(
                "collision_affected_linked_source_ids must be a subset of linked ids"
            )

        if not isinstance(self.p83e_artifact_provenance, ArtifactProvenance):
            raise TypeError("p83e_artifact_provenance must be an ArtifactProvenance")

        for field_name in (
            "match_identity_count",
            "trusted_schedule_observation_count",
            "baseball_game_count",
            "canonical_prediction_count",
            "pregame_eligible_context_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise ValueError(f"{field_name} must be fixed at 0")

        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("limitations must be a non-empty tuple")


def _joint_projection(
    *,
    p83e_result: LegacyPredictionImportResult,
    p84b_result: LegacyScheduleImportResult,
    linked_ids: tuple[str, ...],
    missing_ids: tuple[str, ...],
    schedule_only_ids: tuple[str, ...],
    collision_linked_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": JOINT_SCHEMA_VERSION,
        "p83e_raw_sha256": p83e_result.provenance.input_fingerprint,
        "p83e_semantic_fingerprint": p83e_result.semantic_fingerprint,
        "p84b_raw_sha256": p84b_result.artifact_sha256,
        "p84b_semantic_fingerprint": p84b_result.semantic_fingerprint,
        "linked_source_ids": list(linked_ids),
        "prediction_missing_schedule_ids": list(missing_ids),
        "schedule_only_source_ids": list(schedule_only_ids),
        "collision_affected_linked_source_ids": list(collision_linked_ids),
        "linked_count": len(linked_ids),
        "prediction_missing_schedule_count": len(missing_ids),
        "schedule_only_count": len(schedule_only_ids),
        "collision_affected_linked_count": len(collision_linked_ids),
        "match_identity_count": 0,
        "trusted_schedule_observation_count": 0,
        "baseball_game_count": 0,
        "canonical_prediction_count": 0,
        "pregame_eligible_context_count": 0,
    }


def _joint_semantic_fingerprint(projection: dict[str, object]) -> str:
    encoded = (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def link_legacy_quarantine_snapshots(
    prediction_result: LegacyPredictionImportResult,
    schedule_result: LegacyScheduleImportResult,
) -> LegacyQuarantineLinkResult:
    """Join by exact source game ID only; no persistence, no promotion.

    Every prediction candidate is matched against schedule candidates by exact
    ``source_game_id`` equality. A prediction with no matching schedule row is
    never linked with fabricated or partial evidence: it fails closed into
    ``prediction_missing_schedule_ids`` instead. No date, team, or timestamp
    inference is ever used to join.
    """

    prediction_ids = tuple(
        candidate.source_game_id for candidate in prediction_result.candidates
    )
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(
            "prediction_result unexpectedly contains duplicate source_game_id values"
        )

    schedule_by_id: dict[str, object] = {}
    for candidate in schedule_result.candidates:
        source_id = candidate.provider_reference.source_game_id
        if source_id in schedule_by_id:
            raise ValueError(
                "schedule_result unexpectedly contains duplicate source_game_id values"
            )
        schedule_by_id[source_id] = candidate

    links = []
    missing_ids = []
    for prediction_candidate in prediction_result.candidates:
        schedule_candidate = schedule_by_id.get(prediction_candidate.source_game_id)
        if schedule_candidate is None:
            missing_ids.append(prediction_candidate.source_game_id)
            continue
        links.append(
            LegacyDiagnosticPredictionScheduleLink(
                source_game_id=prediction_candidate.source_game_id,
                provider_reference=schedule_candidate.provider_reference,
                prediction_candidate=prediction_candidate,
                schedule_candidate=schedule_candidate,
            )
        )

    links.sort(key=lambda link: link.source_game_id)
    links = tuple(links)
    linked_ids = tuple(link.source_game_id for link in links)
    schedule_only_ids = tuple(sorted(set(schedule_by_id) - set(linked_ids)))
    missing_ids = tuple(sorted(missing_ids))
    collision_linked_ids = tuple(
        sorted(link.source_game_id for link in links if link.schedule_collision_affected)
    )

    projection = _joint_projection(
        p83e_result=prediction_result,
        p84b_result=schedule_result,
        linked_ids=linked_ids,
        missing_ids=missing_ids,
        schedule_only_ids=schedule_only_ids,
        collision_linked_ids=collision_linked_ids,
    )
    joint_fingerprint = _joint_semantic_fingerprint(projection)

    return LegacyQuarantineLinkResult(
        links=links,
        linked_count=len(links),
        prediction_missing_schedule_ids=missing_ids,
        schedule_only_source_ids=schedule_only_ids,
        collision_affected_linked_source_ids=collision_linked_ids,
        collision_affected_linked_count=len(collision_linked_ids),
        p83e_artifact_provenance=prediction_result.provenance,
        p84b_artifact_sha256=schedule_result.artifact_sha256,
        p83e_semantic_fingerprint=prediction_result.semantic_fingerprint,
        p84b_semantic_fingerprint=schedule_result.semantic_fingerprint,
        joint_semantic_fingerprint=joint_fingerprint,
        match_identity_count=0,
        trusted_schedule_observation_count=0,
        baseball_game_count=0,
        canonical_prediction_count=0,
        pregame_eligible_context_count=0,
        limitations=(NO_PROMOTION_LIMITATION, DIAGNOSTIC_ONLY_LIMITATION),
    )
