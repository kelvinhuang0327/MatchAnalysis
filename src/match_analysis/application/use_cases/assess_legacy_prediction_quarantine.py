"""Assess every legacy P83E prediction row into an immutable quarantine set.

Pure in-memory application logic: consumes the existing P83E import result
and the existing P83E<->P84B diagnostic link exactly as produced by their own
use cases, plus optional exact P9 schedule-identity candidates. No file,
path, current-time, network, or persistence dependency. Constructs no
PredictionSourceObservation and performs no promotion; every row remains
quarantined. Joins are by exact key equality only -- never by date, teams,
time, participants, or list position.
"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

from ...baseball.domain.legacy_prediction_quarantine import (
    AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
    CONTROLLED_QUARANTINE_REASONS,
    MISSING_DIAGNOSTIC_SCHEDULE_LINK,
    MISSING_GAME_NUMBER,
    MISSING_SOURCE_OBSERVATION_ID,
    QUARANTINE_STATUS,
    UNIVERSAL_MISSING_OBSERVATION_REASONS,
    ZERO_DELTA_SELECTION_POLICY_UNRESOLVED,
    LegacyPredictionQuarantineAssessment,
)
from ...baseball.domain.schedule_identity_candidate import (
    ScheduleIdentityResolutionCandidate,
)
from .import_legacy_prediction_snapshot import LegacyPredictionImportResult
from .link_legacy_quarantine_snapshots import LegacyQuarantineLinkResult


ASSESSMENT_SET_SCHEMA_VERSION = "legacy_prediction_quarantine_assessment_set_v1"

NO_ADMISSION_LIMITATION = (
    "This assessment constructs no PredictionSourceObservation and admits or "
    "promotes no prediction; admitted_observation_count is fixed at zero."
)
DIAGNOSTIC_ONLY_LIMITATION = (
    "Every assessed row remains exactly as quarantined by its own P83E "
    "candidate and P3 diagnostic link; this pass adds no new time, identity, "
    "market, or outcome evidence beyond optional exact P9 game_number and "
    "source_observation_id enrichment."
)


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


@dataclass(frozen=True, slots=True)
class LegacyPredictionQuarantineAssessmentSet:
    """Immutable result of assessing every legacy P83E row for quarantine."""

    schema_version: str
    assessments: tuple[LegacyPredictionQuarantineAssessment, ...]
    row_count: int
    quarantined_count: int
    admitted_observation_count: int
    unique_enrichment_count: int
    missing_enrichment_count: int
    ambiguous_enrichment_count: int
    p83e_raw_sha256: str
    p83e_semantic_fingerprint: str
    p84b_artifact_sha256: str
    p84b_semantic_fingerprint: str
    joint_semantic_fingerprint: str
    assessment_set_fingerprint: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ASSESSMENT_SET_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSESSMENT_SET_SCHEMA_VERSION}"
            )
        if not isinstance(self.assessments, tuple) or not all(
            isinstance(item, LegacyPredictionQuarantineAssessment)
            for item in self.assessments
        ):
            raise TypeError(
                "assessments must be a tuple of LegacyPredictionQuarantineAssessment"
            )
        ids = tuple(item.source_game_id for item in self.assessments)
        if len(set(ids)) != len(ids):
            raise ValueError("assessments must not contain duplicate source_game_id")
        if ids != tuple(sorted(ids)):
            raise ValueError("assessments must be ordered by ascending source_game_id")

        for field_name in (
            "row_count",
            "quarantined_count",
            "admitted_observation_count",
            "unique_enrichment_count",
            "missing_enrichment_count",
            "ambiguous_enrichment_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        if self.row_count != len(self.assessments):
            raise ValueError("row_count must equal len(assessments)")
        if self.quarantined_count != self.row_count:
            raise ValueError(
                "quarantined_count must equal row_count: every row stays"
                " quarantined"
            )
        if any(
            item.quarantine_status != QUARANTINE_STATUS for item in self.assessments
        ):
            raise ValueError("every assessment must carry quarantine_status QUARANTINED")
        if self.admitted_observation_count != 0:
            raise ValueError("admitted_observation_count must be fixed at 0")

        actual_unique = sum(
            1 for item in self.assessments if item.enriched_game_number is not None
        )
        actual_ambiguous = sum(
            1
            for item in self.assessments
            if AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH in item.quarantine_reasons
        )
        if self.unique_enrichment_count != actual_unique:
            raise ValueError("unique_enrichment_count does not match the assessments")
        if self.ambiguous_enrichment_count != actual_ambiguous:
            raise ValueError(
                "ambiguous_enrichment_count does not match the assessments"
            )
        if (
            self.unique_enrichment_count
            + self.missing_enrichment_count
            + self.ambiguous_enrichment_count
            != self.row_count
        ):
            raise ValueError(
                "unique_enrichment_count, missing_enrichment_count, and"
                " ambiguous_enrichment_count must sum to row_count"
            )

        for field_name in (
            "p83e_raw_sha256",
            "p83e_semantic_fingerprint",
            "p84b_artifact_sha256",
            "p84b_semantic_fingerprint",
            "joint_semantic_fingerprint",
            "assessment_set_fingerprint",
        ):
            _require_sha256(getattr(self, field_name), field_name)

        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("limitations must be a non-empty tuple")

        expected_fingerprint = _assessment_set_fingerprint(
            assessments=self.assessments,
            row_count=self.row_count,
            quarantined_count=self.quarantined_count,
            admitted_observation_count=self.admitted_observation_count,
            unique_enrichment_count=self.unique_enrichment_count,
            missing_enrichment_count=self.missing_enrichment_count,
            ambiguous_enrichment_count=self.ambiguous_enrichment_count,
            p83e_raw_sha256=self.p83e_raw_sha256,
            p83e_semantic_fingerprint=self.p83e_semantic_fingerprint,
            p84b_artifact_sha256=self.p84b_artifact_sha256,
            p84b_semantic_fingerprint=self.p84b_semantic_fingerprint,
            joint_semantic_fingerprint=self.joint_semantic_fingerprint,
        )
        if self.assessment_set_fingerprint != expected_fingerprint:
            raise ValueError(
                "assessment_set_fingerprint does not match the canonical"
                " assessment-set projection"
            )


def _assessment_projection(
    assessment: LegacyPredictionQuarantineAssessment,
) -> dict[str, object]:
    return {
        "source_game_id": assessment.source_game_id,
        "quarantine_status": assessment.quarantine_status,
        "quarantine_reasons": list(assessment.quarantine_reasons),
        "enriched_game_number": assessment.enriched_game_number,
        "enriched_source_observation_id": assessment.enriched_source_observation_id,
    }


def _assessment_set_fingerprint(
    *,
    assessments: tuple[LegacyPredictionQuarantineAssessment, ...],
    row_count: int,
    quarantined_count: int,
    admitted_observation_count: int,
    unique_enrichment_count: int,
    missing_enrichment_count: int,
    ambiguous_enrichment_count: int,
    p83e_raw_sha256: str,
    p83e_semantic_fingerprint: str,
    p84b_artifact_sha256: str,
    p84b_semantic_fingerprint: str,
    joint_semantic_fingerprint: str,
) -> str:
    projection = {
        "schema_version": ASSESSMENT_SET_SCHEMA_VERSION,
        "row_count": row_count,
        "quarantined_count": quarantined_count,
        "admitted_observation_count": admitted_observation_count,
        "unique_enrichment_count": unique_enrichment_count,
        "missing_enrichment_count": missing_enrichment_count,
        "ambiguous_enrichment_count": ambiguous_enrichment_count,
        "p83e_raw_sha256": p83e_raw_sha256,
        "p83e_semantic_fingerprint": p83e_semantic_fingerprint,
        "p84b_artifact_sha256": p84b_artifact_sha256,
        "p84b_semantic_fingerprint": p84b_semantic_fingerprint,
        "joint_semantic_fingerprint": joint_semantic_fingerprint,
        "assessments": [_assessment_projection(item) for item in assessments],
    }
    payload = (
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def assess_legacy_prediction_quarantine(
    prediction_result: LegacyPredictionImportResult,
    quarantine_link: LegacyQuarantineLinkResult,
    schedule_identity_candidates: tuple[ScheduleIdentityResolutionCandidate, ...] = (),
) -> LegacyPredictionQuarantineAssessmentSet:
    """Assess every prediction candidate; every row remains quarantined.

    Optional P9 enrichment adds ``game_number``/``source_observation_id`` only
    when exactly one P9 candidate shares the linked row's exact provider
    namespace and provider game ID. Zero matches remain unenriched; multiple
    exact matches remain unenriched and receive
    ``AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH``. No date, team, time, participant,
    or list-position join is ever used.
    """

    if quarantine_link.p83e_artifact_provenance != prediction_result.provenance:
        raise ValueError(
            "quarantine_link must be derived from the supplied prediction_result"
            " (provenance mismatch)"
        )
    if quarantine_link.p83e_semantic_fingerprint != prediction_result.semantic_fingerprint:
        raise ValueError(
            "quarantine_link must be derived from the supplied prediction_result"
            " (semantic fingerprint mismatch)"
        )

    prediction_ids = tuple(
        candidate.source_game_id for candidate in prediction_result.candidates
    )
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(
            "prediction_result unexpectedly contains duplicate source_game_id"
            " values"
        )
    link_by_id = {link.source_game_id: link for link in quarantine_link.links}
    accounted_ids = set(link_by_id) | set(quarantine_link.prediction_missing_schedule_ids)
    if accounted_ids != set(prediction_ids):
        raise ValueError(
            "quarantine_link does not account for exactly the supplied"
            " prediction candidates"
        )

    candidates_by_key: dict[tuple[str, str], list[ScheduleIdentityResolutionCandidate]] = (
        defaultdict(list)
    )
    for candidate in schedule_identity_candidates:
        key = (candidate.provider_namespace, candidate.provider_game_id)
        candidates_by_key[key].append(candidate)

    assessments: list[LegacyPredictionQuarantineAssessment] = []
    unique_count = 0
    ambiguous_count = 0
    missing_count = 0

    for prediction_candidate in sorted(
        prediction_result.candidates, key=lambda candidate: candidate.source_game_id
    ):
        link = link_by_id.get(prediction_candidate.source_game_id)
        reasons = list(UNIVERSAL_MISSING_OBSERVATION_REASONS)
        enriched_game_number = None
        enriched_source_observation_id = None

        if link is None:
            reasons.append(MISSING_DIAGNOSTIC_SCHEDULE_LINK)
            reasons.append(MISSING_GAME_NUMBER)
            reasons.append(MISSING_SOURCE_OBSERVATION_ID)
            missing_count += 1
        else:
            key = (
                link.provider_reference.provider_namespace,
                link.provider_reference.provider_game_id,
            )
            matches = candidates_by_key.get(key, [])
            if len(matches) == 1:
                enriched_game_number = matches[0].game_number
                enriched_source_observation_id = matches[0].source_observation_id
                unique_count += 1
            elif len(matches) == 0:
                reasons.append(MISSING_GAME_NUMBER)
                reasons.append(MISSING_SOURCE_OBSERVATION_ID)
                missing_count += 1
            else:
                reasons.append(MISSING_GAME_NUMBER)
                reasons.append(MISSING_SOURCE_OBSERVATION_ID)
                reasons.append(AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH)
                ambiguous_count += 1

        if prediction_candidate.sp_fip_delta.is_zero():
            reasons.append(ZERO_DELTA_SELECTION_POLICY_UNRESOLVED)

        assessments.append(
            LegacyPredictionQuarantineAssessment(
                source_game_id=prediction_candidate.source_game_id,
                prediction_candidate=prediction_candidate,
                diagnostic_link=link,
                quarantine_reasons=tuple(sorted(set(reasons))),
                enriched_game_number=enriched_game_number,
                enriched_source_observation_id=enriched_source_observation_id,
            )
        )

    ordered_assessments = tuple(assessments)
    row_count = len(ordered_assessments)

    fingerprint = _assessment_set_fingerprint(
        assessments=ordered_assessments,
        row_count=row_count,
        quarantined_count=row_count,
        admitted_observation_count=0,
        unique_enrichment_count=unique_count,
        missing_enrichment_count=missing_count,
        ambiguous_enrichment_count=ambiguous_count,
        p83e_raw_sha256=prediction_result.provenance.input_fingerprint,
        p83e_semantic_fingerprint=prediction_result.semantic_fingerprint,
        p84b_artifact_sha256=quarantine_link.p84b_artifact_sha256,
        p84b_semantic_fingerprint=quarantine_link.p84b_semantic_fingerprint,
        joint_semantic_fingerprint=quarantine_link.joint_semantic_fingerprint,
    )

    return LegacyPredictionQuarantineAssessmentSet(
        schema_version=ASSESSMENT_SET_SCHEMA_VERSION,
        assessments=ordered_assessments,
        row_count=row_count,
        quarantined_count=row_count,
        admitted_observation_count=0,
        unique_enrichment_count=unique_count,
        missing_enrichment_count=missing_count,
        ambiguous_enrichment_count=ambiguous_count,
        p83e_raw_sha256=prediction_result.provenance.input_fingerprint,
        p83e_semantic_fingerprint=prediction_result.semantic_fingerprint,
        p84b_artifact_sha256=quarantine_link.p84b_artifact_sha256,
        p84b_semantic_fingerprint=quarantine_link.p84b_semantic_fingerprint,
        joint_semantic_fingerprint=quarantine_link.joint_semantic_fingerprint,
        assessment_set_fingerprint=fingerprint,
        limitations=(NO_ADMISSION_LIMITATION, DIAGNOSTIC_ONLY_LIMITATION),
    )
