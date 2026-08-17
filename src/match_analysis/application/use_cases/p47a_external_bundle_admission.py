"""P47A external P35A bundle admission and prospective import gate.

Validates an already-produced frozen external P35A bundle, computes deterministic
content-addressed fingerprints, binds immutable source bytes, and hands off safely
to the existing P46 adapter and P44/P45 prospective paper lifecycle without modifying
P35A or bypassing security/TLS boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from .p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    NormalizedPregameInput,
    parse_normalized_pregame_payload,
    reject_pregame_outcome_fields,
    write_normalized_pregame_input,
)
from .p46a_p35a_pregame_adapter import (
    P46A_ADAPTER_SOURCE_IDENTITY,
    P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    P46A_DEFAULT_NAMESPACE,
    P46A_DEFAULT_WINDOW,
    P46A_EXCLUSION_SCHEMA,
    P46A_PRODUCTIVE_STATUS,
    adapt_p35a_pregame,
)


P47A_TASK_ID = "P47A"
P47A_ADMISSION_RECORD_SCHEMA = "p47a.external_p35a_admission_record.v1"
P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY = "P35A_CONTRACT_REHEARSAL"
P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY = "P47A_ADMITTED_EXTERNAL_P35A_BUNDLE"
P47A_REPORT_RELATIVE_PATH = Path("report/p47a_external_p35a_bundle_admission")

REQUIRED_BUNDLE_FILES = frozenset(
    {
        "analysis.jsonl",
        "mlb_source_snapshot.jsonl",
        "source_manifest.json",
        "run_manifest.json",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path}") from exc
    value = json.loads(text, object_pairs_hook=_duplicate_rejecting_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped, object_pairs_hook=_duplicate_rejecting_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _extract_64_hex(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        candidate = parts[-1]
        if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate.lower()):
            return candidate.lower()
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text.lower()):
        return text.lower()
    return _sha256_text(text)


def _validate_probability(value: Any, *, field_name: str) -> Decimal:
    try:
        prob = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal probability") from exc
    if not prob.is_finite() or not (Decimal("0") < prob < Decimal("1")):
        raise ValueError(f"{field_name} must be strictly between 0 and 1, got {prob}")
    return prob


def _validate_odds(value: Any, *, field_name: str) -> Decimal:
    try:
        odds = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal price") from exc
    if not odds.is_finite() or odds <= Decimal("1"):
        raise ValueError(f"{field_name} must be strictly greater than 1.0, got {odds}")
    return odds


def _check_confinement(bundle_path: Path) -> Path:
    """Verify that the bundle path exists, is a directory, and contains no escaping symlinks."""
    resolved_root = bundle_path.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"bundle path must be an existing directory: {bundle_path}")

    # Check for path traversal / symlink escape inside the bundle root
    for item in resolved_root.rglob("*"):
        item_resolved = item.resolve()
        try:
            item_resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"P47A_PATH_TRAVERSAL_OR_SYMLINK_ESCAPE: bundle path {item} escapes {resolved_root}"
            ) from exc

    return resolved_root


def _snapshot_file_bytes(bundle_root: Path) -> dict[str, bytes]:
    """Read exact bytes of all required bundle files."""
    snapshots: dict[str, bytes] = {}
    for filename in sorted(REQUIRED_BUNDLE_FILES):
        file_path = bundle_root / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"required bundle file missing: {filename}")
        snapshots[filename] = file_path.read_bytes()
    return snapshots


def compute_deterministic_bundle_fingerprint(
    *,
    run_id: str,
    target_date: str,
    required_file_hashes: Mapping[str, str],
    source_manifest_fingerprint: str,
    analysis_fingerprint: str,
) -> str:
    """Compute canonical imported-bundle fingerprint binding all logical and raw attributes."""
    projection = {
        "schema_version": P47A_ADMISSION_RECORD_SCHEMA,
        "run_id": run_id,
        "target_date": target_date,
        "required_file_hashes": dict(sorted(required_file_hashes.items())),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "analysis_fingerprint": analysis_fingerprint,
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


@dataclass(frozen=True, slots=True)
class P47AAdmittedBundle:
    status: str
    admitted_bundle_id: str
    run_id: str
    target_date: str
    bundle_fingerprint: str
    source_identity: str
    imported_bundle_dir: Path
    normalized_pregame_path: Path
    admission_record_path: Path
    productive_row_count: int
    excluded_row_count: int
    admission_record: dict[str, Any]
    normalized_pregame_input: NormalizedPregameInput


def admit_external_p35a_bundle(
    external_bundle_path: str | Path,
    *,
    admission_root: str | Path | None = None,
    source_identity: str = P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    admitted_at_utc: str = "2026-08-17T12:00:00Z",
) -> P47AAdmittedBundle:
    """Validate, stably snapshot, admit, and adapt an external frozen P35A bundle."""

    # 0. Label guard
    if "live" in source_identity.lower():
        raise ValueError(f"rehearsal fixture must not be labeled live: {source_identity!r}")

    raw_path = Path(external_bundle_path)
    bundle_root = _check_confinement(raw_path)

    # 1. First snapshot read
    first_snapshot = _snapshot_file_bytes(bundle_root)

    # 2. Parse required bundle files
    try:
        run_manifest = json.loads(
            first_snapshot["run_manifest.json"].decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_pairs,
        )
    except Exception as exc:
        raise ValueError(f"malformed run_manifest.json: {exc}") from exc

    try:
        source_manifest = json.loads(
            first_snapshot["source_manifest.json"].decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_pairs,
        )
    except Exception as exc:
        raise ValueError(f"malformed source_manifest.json: {exc}") from exc

    if not isinstance(run_manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object")
    if not isinstance(source_manifest, dict):
        raise ValueError("source_manifest.json must contain a JSON object")

    analysis_rows = _read_jsonl_objects(bundle_root / "analysis.jsonl")
    schedule_rows = _read_jsonl_objects(bundle_root / "mlb_source_snapshot.jsonl")

    if not analysis_rows:
        raise ValueError("analysis.jsonl contains no analysis rows")

    # 3. Reject pregame outcome contamination across all files
    reject_pregame_outcome_fields(analysis_rows)
    reject_pregame_outcome_fields(source_manifest)
    reject_pregame_outcome_fields(run_manifest)
    for sched in schedule_rows:
        for forbidden in FORBIDDEN_PREGAME_FIELD_NAMES:
            if sched.get(forbidden) is not None:
                raise ValueError(
                    f"P44A_PREGAME_OUTCOME_FIELDS_REJECTED schedule row contains {forbidden}={sched[forbidden]}"
                )

    # 4. Validate run manifest & identity consistency
    raw_run_id = run_manifest.get("run_id") or run_manifest.get("run_fingerprint")
    if not raw_run_id or not str(raw_run_id).strip():
        raise ValueError("run_manifest.json is missing non-blank run_id")
    run_id = str(raw_run_id).strip()

    target_date = str(run_manifest.get("target_date") or "").strip()
    if not target_date:
        raise ValueError("run_manifest.json is missing non-blank target_date")

    # Check that analysis rows match run_id where present
    for idx, row in enumerate(analysis_rows, start=1):
        row_run_id = row.get("run_id")
        if row_run_id is not None and str(row_run_id).strip() != run_id:
            raise ValueError(
                f"analysis row {idx} run_id mismatch: expected {run_id}, got {row_run_id}"
            )

    # Check manifest source fingerprints / references if declared
    declared_source_manifest_fp = run_manifest.get("source_manifest_fingerprint")
    actual_source_manifest_fp = _sha256_bytes(
        _canonical_json_bytes(source_manifest)
    )
    if declared_source_manifest_fp and declared_source_manifest_fp != actual_source_manifest_fp:
        raise ValueError(
            f"source_manifest_fingerprint mismatch in run_manifest: "
            f"declared {declared_source_manifest_fp}, actual {actual_source_manifest_fp}"
        )

    # 5. Validate game universe, temporal fields, and probabilities
    seen_game_ids: set[str] = set()
    productive_count = 0
    excluded_count = 0

    schedule_by_game_id: dict[str, dict[str, Any]] = {}
    for sched in schedule_rows:
        g_id = str(sched.get("provider_game_id") or sched.get("game_pk") or "").strip()
        if not g_id:
            raise ValueError("schedule row missing provider_game_id")
        if g_id in schedule_by_game_id:
            raise ValueError(f"duplicate game_id in schedule snapshot: {g_id}")
        schedule_by_game_id[g_id] = sched

    for row in analysis_rows:
        raw_game_id = row.get("game_id") or row.get("provider_game_id")
        if not raw_game_id:
            raise ValueError("analysis row missing game_id")
        game_id = str(raw_game_id).strip()
        if not game_id:
            raise ValueError("analysis row has blank game_id")
        if game_id in seen_game_ids:
            raise ValueError(f"duplicate game_id in analysis: {game_id}")
        seen_game_ids.add(game_id)

        # Scheduled start
        sched = schedule_by_game_id.get(game_id)
        scheduled_start = str(
            row.get("scheduled_start")
            or row.get("scheduled_start_utc")
            or (sched.get("scheduled_start_utc") if sched else "")
        ).strip()
        if not scheduled_start:
            raise ValueError(f"game_id={game_id} missing scheduled start time")
        parse_canonical_utc(scheduled_start)

        if sched is not None:
            sched_start = str(sched.get("scheduled_start_utc", "")).strip()
            if sched_start and sched_start != scheduled_start:
                raise ValueError(
                    f"scheduled start mismatch between analysis ({scheduled_start}) "
                    f"and schedule ({sched_start}) for game_id={game_id}"
                )

        status = str(row.get("status") or row.get("structural_status") or "")
        is_productive = (
            status == P46A_PRODUCTIVE_STATUS
            and row.get("prediction_id") is not None
            and row.get("model_home_probability") is not None
            and row.get("home_decimal_odds") is not None
            and row.get("away_decimal_odds") is not None
        )

        if is_productive:
            productive_count += 1
            # Validate probabilities and odds
            _validate_probability(row["model_home_probability"], field_name="model_home_probability")
            _validate_odds(row["home_decimal_odds"], field_name="home_decimal_odds")
            _validate_odds(row["away_decimal_odds"], field_name="away_decimal_odds")

            raw_observed = row.get("price_observed_at") or row.get("market_observed_at_utc")
            if not raw_observed:
                raise ValueError(f"productive row for game_id={game_id} missing price_observed_at")
            observed_at = str(raw_observed).strip()
            parse_canonical_utc(observed_at)

            # Strict temporal guard
            if observed_at >= scheduled_start:
                raise ValueError(
                    f"market observation time ({observed_at}) is not strictly pregame for {scheduled_start}"
                )
        else:
            excluded_count += 1

    # 6. Second snapshot read (stable-source two-read check)
    second_snapshot = _snapshot_file_bytes(bundle_root)
    for filename, first_bytes in first_snapshot.items():
        second_bytes = second_snapshot[filename]
        if first_bytes != second_bytes:
            raise RuntimeError(
                f"P47A_EXTERNAL_BUNDLE_CHANGED_DURING_ADMISSION: {filename} changed between reads"
            )

    # 7. Compute deterministic fingerprints
    canonical_sorted_analysis = sorted(
        analysis_rows,
        key=lambda r: (
            str(r.get("scheduled_start") or r.get("scheduled_start_utc") or ""),
            str(r.get("game_id") or r.get("provider_game_id") or ""),
        ),
    )
    analysis_fp = _sha256_bytes(_canonical_json_bytes(canonical_sorted_analysis))

    required_file_hashes = {
        filename: _sha256_bytes(content)
        for filename, content in sorted(first_snapshot.items())
    }

    bundle_fp = compute_deterministic_bundle_fingerprint(
        run_id=run_id,
        target_date=target_date,
        required_file_hashes=required_file_hashes,
        source_manifest_fingerprint=actual_source_manifest_fp,
        analysis_fingerprint=analysis_fp,
    )

    admitted_bundle_id = f"p47a_bundle_{bundle_fp[:32]}"

    # 8. Determine admission storage location
    if admission_root is not None:
        target_root = Path(admission_root).resolve()
    else:
        # Default repository runtime location
        target_root = Path(__file__).resolve().parents[4] / P47A_REPORT_RELATIVE_PATH / "admitted"

    imported_bundle_dir = target_root / admitted_bundle_id
    record_path = imported_bundle_dir / "admission_record.json"
    normalized_pregame_path = imported_bundle_dir / "normalized_pregame_input.json"

    # 9. Invoke P46 adapter to obtain NormalizedPregameInput
    adapted_pregame = adapt_p35a_pregame(
        analysis_rows,
        schedule_input=schedule_rows,
        source_manifest_input=source_manifest,
        run_manifest_input=run_manifest,
        source_identity=source_identity,
    )

    admission_record = {
        "schema_version": P47A_ADMISSION_RECORD_SCHEMA,
        "task_id": P47A_TASK_ID,
        "admitted_bundle_id": admitted_bundle_id,
        "run_id": run_id,
        "target_date": target_date,
        "bundle_fingerprint": bundle_fp,
        "source_identity": source_identity,
        "admitted_at_utc": admitted_at_utc,
        "productive_row_count": productive_count,
        "excluded_row_count": excluded_count,
        "target_universe_count": len(analysis_rows),
        "required_file_hashes": required_file_hashes,
        "source_manifest_fingerprint": actual_source_manifest_fp,
        "analysis_fingerprint": analysis_fp,
        "external_source_path_metadata": str(bundle_root),
        "p46_compatibility_status": "PASS",
    }

    # 10. Idempotent check vs conflicting admission check
    if record_path.is_file():
        existing_record = _read_json_object(record_path)
        if (
            existing_record.get("bundle_fingerprint") == bundle_fp
            and existing_record.get("run_id") == run_id
            and existing_record.get("admitted_bundle_id") == admitted_bundle_id
        ):
            return P47AAdmittedBundle(
                status="RECOGNIZED_IDENTICAL",
                admitted_bundle_id=admitted_bundle_id,
                run_id=run_id,
                target_date=target_date,
                bundle_fingerprint=bundle_fp,
                source_identity=source_identity,
                imported_bundle_dir=imported_bundle_dir,
                normalized_pregame_path=normalized_pregame_path,
                admission_record_path=record_path,
                productive_row_count=productive_count,
                excluded_row_count=excluded_count,
                admission_record=existing_record,
                normalized_pregame_input=adapted_pregame,
            )
        raise RuntimeError(
            f"P47A_BUNDLE_AUTHORITY_CONFLICT: existing admission at {record_path} conflicts with incoming bundle"
        )

    # 11. Write immutable snapshot and admission record
    imported_bundle_dir.mkdir(parents=True, exist_ok=True)
    raw_bundle_dir = imported_bundle_dir / "raw_bundle"
    raw_bundle_dir.mkdir(parents=True, exist_ok=True)

    for filename, raw_bytes in first_snapshot.items():
        (raw_bundle_dir / filename).write_bytes(raw_bytes)

    record_path.write_bytes(
        (json.dumps(admission_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    write_normalized_pregame_input(normalized_pregame_path, adapted_pregame)

    return P47AAdmittedBundle(
        status="ADMITTED",
        admitted_bundle_id=admitted_bundle_id,
        run_id=run_id,
        target_date=target_date,
        bundle_fingerprint=bundle_fp,
        source_identity=source_identity,
        imported_bundle_dir=imported_bundle_dir,
        normalized_pregame_path=normalized_pregame_path,
        admission_record_path=record_path,
        productive_row_count=productive_count,
        excluded_row_count=excluded_count,
        admission_record=admission_record,
        normalized_pregame_input=adapted_pregame,
    )


__all__ = (
    "P47A_ADMISSION_RECORD_SCHEMA",
    "P47A_CONTRACT_REHEARSAL_SOURCE_IDENTITY",
    "P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY",
    "P47A_REPORT_RELATIVE_PATH",
    "P47A_TASK_ID",
    "REQUIRED_BUNDLE_FILES",
    "P47AAdmittedBundle",
    "admit_external_p35a_bundle",
    "compute_deterministic_bundle_fingerprint",
)
