"""Import a validated P83E snapshot into an in-memory quarantine."""

from dataclasses import dataclass
from hashlib import sha256
import json

from ...baseball.domain.prediction import (
    DIAGNOSTIC_UNTIMED,
    MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF,
    LegacyPredictionCandidate,
)
from ...core.provenance import ArtifactProvenance
from ..ports.legacy_prediction_source import (
    LegacyPredictionSource,
    NULL_OUTCOME_PLACEHOLDER_FIELDS,
    PINNED_SOURCE_PREDICTION_VERSION,
)


PINNED_LEGACY_COMMIT = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
PINNED_P83E_ARTIFACT_SHA256 = (
    "74c4a5498f80b2e7335b472d742bffac4313922aa4d0f35f89f2c9c220df73bb"
)
SEMANTIC_FINGERPRINT_FIELDS = (
    "diagnostic_status",
    "predicted_side",
    "quarantine_reason",
    "source_game_id",
    "source_prediction_version",
    "sp_fip_delta",
)
SEMANTIC_FINGERPRINT_ENCODING = "utf-8"
SEMANTIC_FINGERPRINT_JSON_KEY_ORDER = "LEXICOGRAPHIC"
SEMANTIC_FINGERPRINT_NEWLINE = "LF_AFTER_EVERY_PROJECTION"
SEMANTIC_FINGERPRINT_SOURCE_ORDER = "STRICT_SOURCE_GAME_ID_ASCENDING"
SEMANTIC_FINGERPRINT_DECIMAL_ENCODING = "FIXED_POINT_PRESERVE_SOURCE_SCALE"
SEMANTIC_FINGERPRINT_BOOLEAN_ENCODING = "NO_BOOLEAN_FIELDS_IN_PROJECTION"

NULL_PLACEHOLDER_LIMITATION = (
    "The source contains four validated always-null outcome placeholders and "
    "contains no observed outcome values."
)
UPSTREAM_READINESS_LIMITATION = (
    "Pinned P84B evidence remained operationally blocked; this quarantine "
    "import does not assert P84B operational readiness."
)


@dataclass(frozen=True, slots=True)
class LegacyPredictionImportResult:
    """Immutable result of an in-memory, non-promoting import."""

    provenance: ArtifactProvenance
    row_count: int
    unique_id_count: int
    validated_null_outcome_placeholder_fields: tuple[str, ...]
    validated_null_outcome_placeholder_count: int
    rows_with_observed_outcomes: int
    promoted_prediction_count: int
    candidates: tuple[LegacyPredictionCandidate, ...]
    semantic_fingerprint: str
    limitations: tuple[str, ...]
    quarantine_counts: tuple[tuple[str, int], ...]


def _candidate_projection(candidate: LegacyPredictionCandidate) -> dict[str, str]:
    return {
        "diagnostic_status": candidate.diagnostic_status,
        "predicted_side": candidate.predicted_side,
        "quarantine_reason": candidate.quarantine_reason,
        "source_game_id": candidate.source_game_id,
        "source_prediction_version": candidate.source_prediction_version,
        "sp_fip_delta": format(candidate.sp_fip_delta, "f"),
    }


def _semantic_fingerprint(
    candidates: tuple[LegacyPredictionCandidate, ...],
) -> str:
    encoded_rows = [
        json.dumps(
            _candidate_projection(candidate),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for candidate in candidates
    ]
    payload = "".join(f"{row}\n" for row in encoded_rows).encode(
        SEMANTIC_FINGERPRINT_ENCODING
    )
    return sha256(payload).hexdigest()


def import_legacy_prediction_snapshot(
    source: LegacyPredictionSource,
) -> LegacyPredictionImportResult:
    """Create outcome-free candidates without persistence or promotion."""

    snapshot = source.load()
    candidates = tuple(
        LegacyPredictionCandidate(
            source_game_id=row.source_game_id,
            source_prediction_version=row.source_prediction_version,
            predicted_side=row.predicted_side,
            sp_fip_delta=row.sp_fip_delta,
        )
        for row in snapshot.rows
    )
    fingerprint = _semantic_fingerprint(candidates)
    unique_id_count = len(
        {candidate.source_game_id for candidate in candidates}
    )
    if unique_id_count != len(candidates):
        raise ValueError("validated source unexpectedly contains duplicate IDs")

    provenance = ArtifactProvenance(
        schema_version="p83e_snapshot_quarantine_v1",
        source_repository="Betting-pool",
        source_commit=PINNED_LEGACY_COMMIT,
        producer_id="p83e_2026_canonical_prediction_row_producer",
        producer_version=PINNED_SOURCE_PREDICTION_VERSION,
        input_fingerprint=snapshot.artifact_sha256,
        content_fingerprint=fingerprint,
    )
    return LegacyPredictionImportResult(
        provenance=provenance,
        row_count=len(candidates),
        unique_id_count=unique_id_count,
        validated_null_outcome_placeholder_fields=(
            snapshot.validated_null_outcome_placeholder_fields
        ),
        validated_null_outcome_placeholder_count=len(
            NULL_OUTCOME_PLACEHOLDER_FIELDS
        ),
        rows_with_observed_outcomes=snapshot.rows_with_observed_outcomes,
        promoted_prediction_count=0,
        candidates=candidates,
        semantic_fingerprint=fingerprint,
        limitations=(
            NULL_PLACEHOLDER_LIMITATION,
            UPSTREAM_READINESS_LIMITATION,
        ),
        quarantine_counts=(
            (
                MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF,
                len(candidates),
            ),
        ),
    )
