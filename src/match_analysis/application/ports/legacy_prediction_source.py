"""Ports for validated, outcome-free legacy prediction snapshots."""

from dataclasses import InitVar, dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Protocol


NULL_OUTCOME_PLACEHOLDER_FIELDS = (
    "result_home_score",
    "result_away_score",
    "actual_winner",
    "is_correct",
)
PINNED_SOURCE_PREDICTION_VERSION = "p84b_diagnostic_baseline_v1"
LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "legacy_prediction_evidence_snapshot_v1"
)
LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION = "p83e_jsonl_parser_v1"
LEGACY_PREDICTION_RAW_ROW_BYTES_RULE = (
    "exact JSON row bytes excluding one terminal LF or CRLF delimiter"
)

_DIAGNOSTIC_STATUS = "DIAGNOSTIC_UNTIMED"
_QUARANTINE_REASON = "MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF"


def _raw_jsonl_rows(raw_artifact_bytes: bytes) -> tuple[bytes, ...]:
    """Apply the public raw-row delimiter rule without normalizing row bytes."""

    segments = raw_artifact_bytes.split(b"\n")
    has_terminal_lf = raw_artifact_bytes.endswith(b"\n")
    if has_terminal_lf:
        segments.pop()
    rows: list[bytes] = []
    for index, segment in enumerate(segments):
        followed_by_lf = index < len(segments) - 1 or has_terminal_lf
        if followed_by_lf and segment.endswith(b"\r"):
            segment = segment[:-1]
        rows.append(segment)
    return tuple(rows)


def legacy_prediction_evidence_semantic_fingerprint(
    rows: tuple["LegacyPredictionEvidenceRow", ...],
) -> str:
    """Reproduce the established P83E quarantine semantic projection."""

    encoded_rows = (
        json.dumps(
            {
                "diagnostic_status": _DIAGNOSTIC_STATUS,
                "predicted_side": row.predicted_side,
                "quarantine_reason": _QUARANTINE_REASON,
                "source_game_id": row.legacy_game_id,
                "source_prediction_version": row.source_prediction_version,
                "sp_fip_delta": format(row.sp_fip_delta, "f"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for row in rows
    )
    payload = "".join(f"{row}\n" for row in encoded_rows).encode("utf-8")
    return sha256(payload).hexdigest()


def legacy_prediction_evidence_snapshot_fingerprint(
    *,
    rows: tuple["LegacyPredictionEvidenceRow", ...],
    source_repository: str,
    source_ref: str,
    source_blob: str,
    raw_artifact_sha256: str,
    semantic_fingerprint: str,
    parser_version: str,
    schema_version: str,
    row_count: int,
) -> str:
    """Hash a canonical projection that binds provenance to ordered raw rows."""

    projection = {
        "parser_version": parser_version,
        "raw_artifact_sha256": raw_artifact_sha256,
        "raw_row_sha256": [row.raw_row_sha256 for row in rows],
        "row_count": row_count,
        "schema_version": schema_version,
        "semantic_fingerprint": semantic_fingerprint,
        "source_blob": source_blob,
        "source_ref": source_ref,
        "source_repository": source_repository,
    }
    payload = json.dumps(
        projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LegacyPredictionEvidenceRow:
    """Every directly parsed P83E field plus its exact source-row evidence.

    ``home_win_probability`` is explicitly ``P(home wins)``. Raw row bytes
    follow ``LEGACY_PREDICTION_RAW_ROW_BYTES_RULE``.
    """

    legacy_game_id: str
    game_date: str
    season: Decimal
    home_team: str
    away_team: str
    home_sp_fip: Decimal
    away_sp_fip: Decimal
    sp_fip_delta: Decimal
    abs_sp_fip_delta: Decimal
    home_win_probability: Decimal
    predicted_side: str
    source_prediction_version: str
    rule_primary_125_flag: bool
    rule_shadow_100_flag: bool
    tier_b_candidate_flag: bool
    tier_a_watchlist_flag: bool
    paper_only: bool
    diagnostic_only: bool
    odds_used: bool
    market_edge_evaluated: bool
    production_ready: bool
    result_home_score: Decimal | None
    result_away_score: Decimal | None
    actual_winner: str | None
    is_correct: bool | None
    raw_row_bytes: bytes
    raw_row_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw_row_bytes, bytes):
            raise TypeError("raw_row_bytes must be immutable bytes")
        if sha256(self.raw_row_bytes).hexdigest() != self.raw_row_sha256:
            raise ValueError("raw_row_sha256 must hash the exposed raw_row_bytes")


@dataclass(frozen=True, slots=True)
class LegacyPredictionEvidenceSnapshot:
    """Immutable ordered P83E rows bound to explicit source provenance."""

    rows: tuple[LegacyPredictionEvidenceRow, ...]
    source_repository: str
    source_ref: str
    source_blob: str
    raw_artifact_sha256: str
    semantic_fingerprint: str
    parser_version: str
    schema_version: str
    row_count: int
    snapshot_fingerprint: str
    raw_artifact_bytes: InitVar[bytes]

    def __post_init__(self, raw_artifact_bytes: bytes) -> None:
        if not isinstance(raw_artifact_bytes, bytes):
            raise TypeError("raw_artifact_bytes must be immutable bytes")
        if not isinstance(self.rows, tuple):
            raise TypeError("evidence rows must be an immutable tuple")
        if self.schema_version != LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unexpected evidence snapshot schema version")
        if self.parser_version != LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION:
            raise ValueError("unexpected evidence snapshot parser version")
        if (
            not self.source_repository
            or not self.source_ref
            or not self.source_blob
        ):
            raise ValueError("source repository, ref, and blob must be explicit")
        if self.row_count != len(self.rows):
            raise ValueError("row_count must match the ordered evidence rows")
        if len({row.legacy_game_id for row in self.rows}) != self.row_count:
            raise ValueError("legacy_game_id values must be unique")
        if tuple(row.raw_row_bytes for row in self.rows) != _raw_jsonl_rows(
            raw_artifact_bytes
        ):
            raise ValueError("raw row bytes or row order differ from the artifact")
        if sha256(raw_artifact_bytes).hexdigest() != self.raw_artifact_sha256:
            raise ValueError("raw artifact SHA-256 does not match its bytes")
        expected_semantic = legacy_prediction_evidence_semantic_fingerprint(
            self.rows
        )
        if self.semantic_fingerprint != expected_semantic:
            raise ValueError("semantic fingerprint does not match evidence rows")
        expected_snapshot = legacy_prediction_evidence_snapshot_fingerprint(
            rows=self.rows,
            source_repository=self.source_repository,
            source_ref=self.source_ref,
            source_blob=self.source_blob,
            raw_artifact_sha256=self.raw_artifact_sha256,
            semantic_fingerprint=self.semantic_fingerprint,
            parser_version=self.parser_version,
            schema_version=self.schema_version,
            row_count=self.row_count,
        )
        if self.snapshot_fingerprint != expected_snapshot:
            raise ValueError("snapshot fingerprint does not match its contents")


@dataclass(frozen=True, slots=True)
class LegacyPredictionRow:
    """Source semantics allowed to cross the legacy adapter boundary."""

    source_game_id: str
    source_prediction_version: str
    predicted_side: str
    sp_fip_delta: Decimal


@dataclass(frozen=True, slots=True)
class LegacyPredictionSnapshot:
    """Validated transport data without outcome or timestamp fields."""

    artifact_sha256: str
    rows: tuple[LegacyPredictionRow, ...]
    validated_null_outcome_placeholder_fields: tuple[str, ...]
    rows_with_observed_outcomes: int

    def __post_init__(self) -> None:
        if (
            self.validated_null_outcome_placeholder_fields
            != NULL_OUTCOME_PLACEHOLDER_FIELDS
        ):
            raise ValueError("the exact four null placeholders must be validated")
        if self.rows_with_observed_outcomes != 0:
            raise ValueError("legacy prediction snapshots must be outcome-free")


class LegacyPredictionSource(Protocol):
    """Loads one explicitly selected and hash-pinned snapshot."""

    def load(self) -> LegacyPredictionSnapshot:
        """Return validated source rows without performing writes."""


class LegacyPredictionEvidenceSource(Protocol):
    """Loads one complete snapshot with explicit immutable provenance."""

    def load_evidence(self) -> LegacyPredictionEvidenceSnapshot:
        """Return all artifact fields and exact source-row evidence."""
