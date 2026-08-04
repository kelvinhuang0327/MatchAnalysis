"""Build an immutable admitted prediction observation snapshot from P15B1 results.

Reads verified P15B1 admission results and extracts only ADMITTED observation
payloads into a deterministic snapshot. Does not construct
PredictionSourceObservation, re-run admission, or call any provider.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


SNAPSHOT_SCHEMA_VERSION = "p15c.admitted_prediction_observation_snapshot.v1"


EXPLICIT_SNAPSHOT_CLAIMS = {
    "betting_claim": False,
    "db_written": False,
    "deployed": False,
    "legacy_rows_admitted": False,
    "network_called": False,
    "outcomes_attached": False,
    "provider_called": False,
}


@dataclass(frozen=True, slots=True)
class AdmittedPredictionObservationRow:
    """A single admitted observation row in the snapshot."""

    prediction_observation_id: str
    source_result_row_fingerprint: str
    observation: dict[str, Any]
    snapshot_row_fingerprint: str


@dataclass(frozen=True, slots=True)
class AdmittedPredictionObservationSnapshotResult:
    """Immutable result of building the admitted observation snapshot."""

    schema_version: str
    source_results_sha256: str
    source_summary_sha256: str
    source_result_set_fingerprint: str
    source_row_count: int
    source_admitted_count: int
    source_rejected_count: int
    snapshot_rows: tuple[AdmittedPredictionObservationRow, ...]
    snapshot_fingerprint: str
    claims: dict[str, bool]


def _compute_result_row_fingerprint(row_dict: dict[str, Any]) -> str:
    """Compute deterministic fingerprint for a single P15B1 result row."""
    canonical = json.dumps(row_dict, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_snapshot_row_fingerprint(
    prediction_observation_id: str,
    source_result_row_fingerprint: str,
    observation: dict[str, Any],
) -> str:
    """Compute deterministic fingerprint for a snapshot row."""
    canonical_payload = {
        "prediction_observation_id": prediction_observation_id,
        "source_result_row_fingerprint": source_result_row_fingerprint,
        "observation": observation,
    }
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_snapshot_fingerprint(
    rows: tuple[AdmittedPredictionObservationRow, ...],
) -> str:
    """Compute deterministic fingerprint over the entire snapshot."""
    parts = []
    for row in rows:
        parts.append(
            f"{row.prediction_observation_id}:{row.snapshot_row_fingerprint}\n"
        )
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _validate_json_no_duplicate_keys(raw_line: str, line_index: int) -> dict[str, Any]:
    """Parse JSON rejecting duplicate keys at all levels."""

    def _object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, int] = {}
        for key, _ in pairs:
            if key in seen:
                raise ValueError(
                    f"Duplicate JSON key {key!r} in result row {line_index + 1}"
                )
            seen[key] = 1
        return dict(pairs)

    return json.loads(raw_line, object_pairs_hook=_object_pairs_hook)


def load_and_validate_p15b1_results(
    results_bytes: bytes,
    summary_bytes: bytes,
    results_sha256: str,
    summary_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate P15B1 result and summary artifacts.

    Returns (result_rows, summary_dict).
    Raises ValueError on any validation failure.
    """
    # Verify SHA-256 hashes
    actual_results_sha256 = hashlib.sha256(results_bytes).hexdigest()
    if actual_results_sha256 != results_sha256:
        raise ValueError(
            f"P15B1 results SHA-256 mismatch: expected {results_sha256}, "
            f"got {actual_results_sha256}"
        )

    actual_summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    if actual_summary_sha256 != summary_sha256:
        raise ValueError(
            f"P15B1 summary SHA-256 mismatch: expected {summary_sha256}, "
            f"got {actual_summary_sha256}"
        )

    # Parse results JSONL rejecting duplicate keys
    results_text = results_bytes.decode("utf-8")
    raw_lines = [line for line in results_text.splitlines() if line.strip()]
    result_rows: list[dict[str, Any]] = []
    for i, line in enumerate(raw_lines):
        row = _validate_json_no_duplicate_keys(line, i)
        result_rows.append(row)

    # Parse summary rejecting duplicate keys
    summary_dict = json.loads(
        summary_bytes.decode("utf-8"),
        object_pairs_hook=lambda pairs: _check_duplicate_keys_recursive(pairs),
    )

    return result_rows, summary_dict


def _check_duplicate_keys_recursive(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Object pairs hook that rejects duplicate JSON keys."""
    seen: dict[str, int] = {}
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"Duplicate JSON key {key!r} in summary")
        seen[key] = 1
    return dict(pairs)


REQUIRED_OBSERVATION_FIELDS = (
    "prediction_observation_id",
    "source_prediction_id",
    "model_id",
    "market_id",
    "selection",
    "model_probability",
    "line_value",
    "push_policy",
    "provider_namespace",
    "provider_game_id",
    "game_number",
    "source_schedule_observation_id",
    "prediction_generated_at_utc",
    "response_received_at_utc",
    "ingested_at_utc",
    "scheduled_start_utc",
)


def build_admitted_prediction_observation_snapshot(
    *,
    results_bytes: bytes,
    summary_bytes: bytes,
) -> AdmittedPredictionObservationSnapshotResult:
    """Build a deterministic admitted observation snapshot from P15B1 artifacts.

    Validates source artifacts, extracts ADMITTED observations, produces
    deterministic snapshot rows sorted by prediction_observation_id.

    Raises ValueError on any validation failure.
    """
    results_sha256 = hashlib.sha256(results_bytes).hexdigest()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()

    result_rows, summary_dict = load_and_validate_p15b1_results(
        results_bytes, summary_bytes, results_sha256, summary_sha256,
    )

    # Verify summary schema expectations
    if "result_set_fingerprint" not in summary_dict:
        raise ValueError("P15B1 summary missing result_set_fingerprint")
    if "admitted_count" not in summary_dict:
        raise ValueError("P15B1 summary missing admitted_count")
    if "rejected_count" not in summary_dict:
        raise ValueError("P15B1 summary missing rejected_count")
    if "claims" not in summary_dict:
        raise ValueError("P15B1 summary missing claims")

    # Verify legacy_rows_admitted=false
    claims = summary_dict["claims"]
    if claims.get("legacy_rows_admitted") is not False:
        raise ValueError(
            "P15B1 summary claims.legacy_rows_admitted is not false"
        )

    source_result_set_fingerprint = summary_dict["result_set_fingerprint"]

    # Verify every row has valid admission status
    for i, row in enumerate(result_rows):
        status = row.get("admission_status")
        if status not in ("ADMITTED", "REJECTED"):
            raise ValueError(
                f"Invalid admission_status {status!r} at row index {i}"
            )

    # Verify row counts match summary
    admitted_rows = [r for r in result_rows if r.get("admission_status") == "ADMITTED"]
    rejected_rows = [r for r in result_rows if r.get("admission_status") == "REJECTED"]

    if len(admitted_rows) != summary_dict["admitted_count"]:
        raise ValueError(
            f"Admitted count mismatch: {len(admitted_rows)} rows vs "
            f"summary {summary_dict['admitted_count']}"
        )
    if len(rejected_rows) != summary_dict["rejected_count"]:
        raise ValueError(
            f"Rejected count mismatch: {len(rejected_rows)} rows vs "
            f"summary {summary_dict['rejected_count']}"
        )
    if len(result_rows) != summary_dict.get("request_count", -1):
        raise ValueError(
            f"Request count mismatch: {len(result_rows)} rows vs "
            f"summary {summary_dict.get('request_count')}"
        )

    # Verify every admitted row has a complete observation
    for i, row in enumerate(admitted_rows):
        if row.get("observation") is None:
            raise ValueError(
                f"ADMITTED row at index {i} has null observation"
            )
        obs = row["observation"]
        for field in REQUIRED_OBSERVATION_FIELDS:
            if field not in obs or obs[field] is None:
                raise ValueError(
                    f"ADMITTED row at index {i} observation missing or null field {field!r}"
                )

    # Verify every rejected row has observation=null
    for i, row in enumerate(rejected_rows):
        if row.get("observation") is not None:
            raise ValueError(
                f"REJECTED row at index {i} has non-null observation"
            )

    # Verify result-set fingerprint (in request_index order for shuffle invariance)
    fingerprint_parts = []
    canonical_row_order = sorted(
        result_rows, key=lambda r: r.get("request_index", 0)
    )
    for row in canonical_row_order:
        status = row["admission_status"]
        reason = row.get("reason") or ""
        obs_id = ""
        if row.get("observation") is not None:
            obs_id = row["observation"].get("prediction_observation_id", "")
        fingerprint_parts.append(f"{status}:{reason}:{obs_id}\n")
    computed_fingerprint = hashlib.sha256(
        "".join(fingerprint_parts).encode("utf-8")
    ).hexdigest()
    if computed_fingerprint != source_result_set_fingerprint:
        raise ValueError(
            f"Result-set fingerprint mismatch: computed {computed_fingerprint}, "
            f"expected {source_result_set_fingerprint}"
        )

    # Verify every result row deterministic fingerprint
    for i, row in enumerate(result_rows):
        _compute_result_row_fingerprint(row)

    # Check for duplicate observation IDs (fail closed, including byte-identical)
    obs_ids: list[str] = []
    for row in admitted_rows:
        obs_id = row["observation"]["prediction_observation_id"]
        if obs_id in obs_ids:
            raise ValueError(
                f"Duplicate prediction_observation_id: {obs_id}"
            )
        obs_ids.append(obs_id)

    # Build snapshot rows sorted by prediction_observation_id
    unsorted_rows: list[AdmittedPredictionObservationRow] = []
    for row in admitted_rows:
        obs = row["observation"]
        obs_id = obs["prediction_observation_id"]
        row_fingerprint = _compute_result_row_fingerprint(row)
        snapshot_row_fp = _compute_snapshot_row_fingerprint(
            obs_id, row_fingerprint, obs,
        )
        unsorted_rows.append(
            AdmittedPredictionObservationRow(
                prediction_observation_id=obs_id,
                source_result_row_fingerprint=row_fingerprint,
                observation=obs,
                snapshot_row_fingerprint=snapshot_row_fp,
            )
        )

    sorted_rows = tuple(
        sorted(unsorted_rows, key=lambda r: r.prediction_observation_id)
    )

    snapshot_fp = _compute_snapshot_fingerprint(sorted_rows)

    return AdmittedPredictionObservationSnapshotResult(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        source_results_sha256=results_sha256,
        source_summary_sha256=summary_sha256,
        source_result_set_fingerprint=source_result_set_fingerprint,
        source_row_count=len(result_rows),
        source_admitted_count=len(admitted_rows),
        source_rejected_count=len(rejected_rows),
        snapshot_rows=sorted_rows,
        snapshot_fingerprint=snapshot_fp,
        claims=dict(EXPLICIT_SNAPSHOT_CLAIMS),
    )
