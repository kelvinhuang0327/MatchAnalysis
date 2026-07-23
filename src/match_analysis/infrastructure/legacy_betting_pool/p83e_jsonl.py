"""Closed-schema, read-only adapter for the pinned P83E JSONL artifact."""

from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from typing import BinaryIO

from ...application.ports.legacy_prediction_source import (
    LegacyPredictionRow,
    LegacyPredictionSnapshot,
    NULL_OUTCOME_PLACEHOLDER_FIELDS,
    PINNED_SOURCE_PREDICTION_VERSION,
)


STOP_SCHEMA_MISMATCH = "STOP_MATCHANALYSIS_P83E_SCHEMA_MISMATCH"
STOP_OBSERVED_OUTCOME = (
    "STOP_MATCHANALYSIS_OBSERVED_OUTCOME_VALUE_PRESENT"
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_GAME_ID_PATTERN = re.compile(r"^mlb_2026_[0-9]+$")
_DATE_PATTERN = re.compile(r"^2026-[0-9]{2}-[0-9]{2}$")
_OUTCOME_LIKE_PATTERN = re.compile(
    r"(^|_)(outcome|score|result|correct|winner)($|_)"
)

_EXPECTED_FIELDS = frozenset(
    {
        "game_id",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "home_sp_fip",
        "away_sp_fip",
        "sp_fip_delta",
        "abs_sp_fip_delta",
        "model_probability",
        "predicted_side",
        "source_prediction_version",
        "rule_primary_125_flag",
        "rule_shadow_100_flag",
        "tier_b_candidate_flag",
        "tier_a_watchlist_flag",
        "paper_only",
        "diagnostic_only",
        "odds_used",
        "market_edge_evaluated",
        "production_ready",
        *NULL_OUTCOME_PLACEHOLDER_FIELDS,
    }
)

_NUMERIC_FIELDS = (
    "home_sp_fip",
    "away_sp_fip",
    "sp_fip_delta",
    "abs_sp_fip_delta",
    "model_probability",
)


class P83eSnapshotValidationError(ValueError):
    """A fail-closed rejection of untrusted snapshot bytes."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _schema_error(detail: str) -> P83eSnapshotValidationError:
    return P83eSnapshotValidationError(STOP_SCHEMA_MISMATCH, detail)


def _reject_json_constant(value: str) -> None:
    raise _schema_error(f"non-standard JSON numeric constant: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _schema_error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_explicit_string(row: dict[str, object], field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value or value != value.strip():
        raise _schema_error(f"{field} must be an explicit trimmed string")
    return value


def _require_decimal(row: dict[str, object], field: str) -> Decimal:
    value = row[field]
    if not isinstance(value, Decimal) or not value.is_finite():
        raise _schema_error(f"{field} must be a finite JSON number")
    return value


def _expected_rule_flags(
    predicted_side: str,
    absolute_delta: Decimal,
) -> dict[str, bool]:
    if predicted_side == "home":
        primary = absolute_delta >= Decimal("0.50")
        shadow = absolute_delta >= Decimal("0.50")
    else:
        primary = absolute_delta >= Decimal("1.25")
        shadow = absolute_delta >= Decimal("1.00")
    return {
        "rule_primary_125_flag": primary,
        "rule_shadow_100_flag": shadow,
        "tier_b_candidate_flag": (
            Decimal("0.25") <= absolute_delta < Decimal("0.50")
        ),
        "tier_a_watchlist_flag": absolute_delta < Decimal("0.25"),
    }


class P83eJsonlSnapshotSource:
    """Loads only an explicitly supplied path, bytes object, or byte stream."""

    def __init__(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        expected_sha256: str,
    ) -> None:
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a lowercase SHA-256")
        self._expected_sha256 = expected_sha256
        self._path: Path | None = None
        self._bytes: bytes | None = None

        if isinstance(source, (str, Path)):
            self._path = Path(source)
        elif isinstance(source, (bytes, bytearray)):
            self._bytes = bytes(source)
        elif isinstance(source, io.BufferedIOBase) or hasattr(source, "read"):
            stream_bytes = source.read()
            if not isinstance(stream_bytes, bytes):
                raise TypeError("byte stream must return bytes")
            self._bytes = stream_bytes
        else:
            raise TypeError("source must be an explicit path, bytes, or byte stream")

    def _read_bytes(self) -> bytes:
        if self._bytes is not None:
            return self._bytes
        if self._path is None:
            raise RuntimeError("snapshot source was not initialized")
        return self._path.read_bytes()

    def load(self) -> LegacyPredictionSnapshot:
        """Hash first, then parse and validate without writing."""

        raw = self._read_bytes()
        artifact_sha256 = sha256(raw).hexdigest()
        if artifact_sha256 != self._expected_sha256:
            raise P83eSnapshotValidationError(
                STOP_SCHEMA_MISMATCH,
                "raw artifact SHA-256 does not match the expected value",
            )
        if not raw:
            raise _schema_error("snapshot is empty")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _schema_error("snapshot is not valid UTF-8") from error

        lines = text.splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise _schema_error("snapshot contains a blank row")

        validated_rows: list[LegacyPredictionRow] = []
        seen_ids: set[str] = set()
        previous_id: str | None = None
        for line_number, line in enumerate(lines, start=1):
            try:
                parsed = json.loads(
                    line,
                    parse_float=Decimal,
                    parse_int=Decimal,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except json.JSONDecodeError as error:
                raise _schema_error(
                    f"line {line_number} is malformed JSON"
                ) from error
            if not isinstance(parsed, dict):
                raise _schema_error(f"line {line_number} must be a JSON object")

            row = parsed
            extra_outcome_fields = sorted(
                field
                for field in row
                if _OUTCOME_LIKE_PATTERN.search(field)
                and field not in NULL_OUTCOME_PLACEHOLDER_FIELDS
            )
            if extra_outcome_fields:
                raise _schema_error(
                    "additional outcome-like fields are forbidden: "
                    + ", ".join(extra_outcome_fields)
                )
            missing_placeholders = [
                field
                for field in NULL_OUTCOME_PLACEHOLDER_FIELDS
                if field not in row
            ]
            if missing_placeholders:
                raise _schema_error(
                    "missing null outcome placeholders: "
                    + ", ".join(missing_placeholders)
                )
            non_null_placeholders = [
                field
                for field in NULL_OUTCOME_PLACEHOLDER_FIELDS
                if row[field] is not None
            ]
            if non_null_placeholders:
                raise P83eSnapshotValidationError(
                    STOP_OBSERVED_OUTCOME,
                    "observed outcome value in: "
                    + ", ".join(non_null_placeholders),
                )
            if frozenset(row) != _EXPECTED_FIELDS:
                missing = sorted(_EXPECTED_FIELDS - frozenset(row))
                extra = sorted(frozenset(row) - _EXPECTED_FIELDS)
                raise _schema_error(
                    f"closed schema mismatch; missing={missing}, extra={extra}"
                )

            game_id = _require_explicit_string(row, "game_id")
            if _SOURCE_GAME_ID_PATTERN.fullmatch(game_id) is None:
                raise _schema_error("game_id does not match the pinned source format")
            if game_id in seen_ids:
                raise _schema_error(f"duplicate game_id: {game_id}")
            if previous_id is not None and game_id <= previous_id:
                raise _schema_error(
                    f"game_id order is not strictly increasing at {game_id}"
                )

            game_date = _require_explicit_string(row, "game_date")
            if _DATE_PATTERN.fullmatch(game_date) is None:
                raise _schema_error("game_date must use YYYY-MM-DD in 2026")
            if row["season"] != Decimal("2026"):
                raise _schema_error("season must be the JSON number 2026")
            home_team = _require_explicit_string(row, "home_team")
            away_team = _require_explicit_string(row, "away_team")
            if home_team == away_team:
                raise _schema_error("home_team and away_team must differ")

            numerics = {
                field: _require_decimal(row, field)
                for field in _NUMERIC_FIELDS
            }
            delta = numerics["sp_fip_delta"]
            absolute_delta = numerics["abs_sp_fip_delta"]
            if numerics["home_sp_fip"] - numerics["away_sp_fip"] != delta:
                raise _schema_error("sp_fip_delta is not the exact Decimal delta")
            if abs(delta) != absolute_delta:
                raise _schema_error(
                    "abs_sp_fip_delta is not the exact absolute Decimal delta"
                )
            if delta.is_zero():
                raise _schema_error("zero sp_fip_delta is not allowed")
            if not Decimal("0") <= numerics["model_probability"] <= Decimal("1"):
                raise _schema_error("model_probability must be within [0, 1]")

            predicted_side = _require_explicit_string(row, "predicted_side")
            expected_side = "away" if delta > 0 else "home"
            if predicted_side != expected_side:
                raise _schema_error(
                    "predicted_side conflicts with the corrected FIP mapping"
                )
            source_version = _require_explicit_string(
                row,
                "source_prediction_version",
            )
            if source_version != PINNED_SOURCE_PREDICTION_VERSION:
                raise _schema_error("unexpected source_prediction_version")

            expected_flags = _expected_rule_flags(
                predicted_side,
                absolute_delta,
            )
            for field, expected_value in expected_flags.items():
                if row[field] is not expected_value:
                    raise _schema_error(f"{field} violates the pinned rule contract")
            governance = {
                "paper_only": True,
                "diagnostic_only": True,
                "odds_used": False,
                "market_edge_evaluated": False,
                "production_ready": False,
            }
            for field, expected_value in governance.items():
                if row[field] is not expected_value:
                    raise _schema_error(f"{field} violates quarantine governance")

            validated_rows.append(
                LegacyPredictionRow(
                    source_game_id=game_id,
                    source_prediction_version=source_version,
                    predicted_side=predicted_side,
                    sp_fip_delta=delta,
                )
            )
            seen_ids.add(game_id)
            previous_id = game_id

        return LegacyPredictionSnapshot(
            artifact_sha256=artifact_sha256,
            rows=tuple(validated_rows),
            validated_null_outcome_placeholder_fields=(
                NULL_OUTCOME_PLACEHOLDER_FIELDS
            ),
            rows_with_observed_outcomes=0,
        )
