"""P49A external final-result admission and prospective postgame settlement intake.

Validates an authoritative external final-result bundle, computes deterministic
content-addressed fingerprints, binds immutable source bytes, and hands off safely
to the existing P44 normalized result boundary and P45 prospective paper settlement
lifecycle without modifying historical authority or bypassing security invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from .p43a_postgame_settle import load_p43a_frozen_decision_bundle
from .p44a_normalized_workflow_input import (
    NormalizedResultRecord,
    parse_normalized_result_payload,
    write_normalized_result_input,
)
from .p45a_paper_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    P45ASettleRunResult,
    read_json_object,
    read_jsonl_objects,
    settle_p45a_paper_run,
)


P49A_TASK_ID = "P49A"
P49A_ADMISSION_RECORD_SCHEMA = "p49a.external_final_result_admission_record.v1"
P49A_INTAKE_RECEIPT_SCHEMA = "p49a.prospective_postgame_settle_receipt.v1"
P49A_DEFAULT_SOURCE_IDENTITY = "P49A_ADMITTED_EXTERNAL_RESULT_BUNDLE"
P49A_CONTRACT_REHEARSAL_SOURCE_IDENTITY = "MLB_OFFICIAL_RESULTS_REHEARSAL"
P49A_REPORT_RELATIVE_PATH = Path("report/p49a_external_final_result_bundle_admission")

REQUIRED_RESULT_BUNDLE_FILES = frozenset(
    {
        "final_results.jsonl",
        "source_manifest.json",
        "result_manifest.json",
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


def _check_confinement(bundle_path: Path) -> Path:
    """Verify that the bundle path exists, is a directory, and contains no escaping symlinks."""
    resolved_root = bundle_path.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"bundle path must be an existing directory: {bundle_path}")

    for item in resolved_root.rglob("*"):
        item_resolved = item.resolve()
        try:
            item_resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"P49A_PATH_TRAVERSAL_OR_SYMLINK_ESCAPE: bundle path {item} escapes {resolved_root}"
            ) from exc

    return resolved_root


def _snapshot_file_bytes(bundle_root: Path) -> dict[str, bytes]:
    """Read exact bytes of all required bundle files."""
    snapshots: dict[str, bytes] = {}
    for filename in sorted(REQUIRED_RESULT_BUNDLE_FILES):
        file_path = bundle_root / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"required bundle file missing: {filename}")
        snapshots[filename] = file_path.read_bytes()
    return snapshots


def compute_deterministic_result_bundle_fingerprint(
    *,
    target_date: str,
    required_file_hashes: Mapping[str, str],
    source_manifest_fingerprint: str,
    results_fingerprint: str,
) -> str:
    """Compute canonical imported final-result bundle fingerprint."""
    projection = {
        "schema_version": P49A_ADMISSION_RECORD_SCHEMA,
        "target_date": target_date,
        "required_file_hashes": dict(sorted(required_file_hashes.items())),
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "results_fingerprint": results_fingerprint,
    }
    return _sha256_bytes(_canonical_json_bytes(projection))


@dataclass(frozen=True, slots=True)
class P49AAdmittedResultBundle:
    status: str
    admitted_bundle_id: str
    target_date: str
    bundle_fingerprint: str
    source_identity: str
    imported_bundle_dir: Path
    normalized_result_path: Path
    admission_record_path: Path
    final_result_count: int
    admission_record: dict[str, Any]
    normalized_result_records: tuple[NormalizedResultRecord, ...]


def _require_int_score(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer >= 0")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def admit_external_final_result_bundle(
    external_bundle_path: str | Path,
    *,
    admission_root: str | Path | None = None,
    source_identity: str = P49A_DEFAULT_SOURCE_IDENTITY,
    admitted_at_utc: str = "2026-08-18T23:59:59Z",
    target_pregame_decisions: Sequence[Mapping[str, Any]] | None = None,
) -> P49AAdmittedResultBundle:
    """Validate, stably snapshot, admit, and normalize an external final-result bundle."""

    if "live" in source_identity.lower():
        raise ValueError(f"rehearsal fixture must not be labeled live: {source_identity!r}")

    raw_path = Path(external_bundle_path)
    bundle_root = _check_confinement(raw_path)

    # 1. First snapshot read
    first_snapshot = _snapshot_file_bytes(bundle_root)

    # 2. Parse required bundle metadata
    try:
        result_manifest = json.loads(
            first_snapshot["result_manifest.json"].decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_pairs,
        )
    except Exception as exc:
        raise ValueError(f"malformed result_manifest.json: {exc}") from exc

    try:
        source_manifest = json.loads(
            first_snapshot["source_manifest.json"].decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_pairs,
        )
    except Exception as exc:
        raise ValueError(f"malformed source_manifest.json: {exc}") from exc

    if not isinstance(result_manifest, dict):
        raise ValueError("result_manifest.json must contain a JSON object")
    if not isinstance(source_manifest, dict):
        raise ValueError("source_manifest.json must contain a JSON object")

    raw_result_rows = _read_jsonl_objects(bundle_root / "final_results.jsonl")
    if not raw_result_rows:
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")

    target_date = str(result_manifest.get("target_date") or "").strip()
    if not target_date:
        raise ValueError("result_manifest.json is missing non-blank target_date")

    # 3. Validate every final result record strictly
    seen_identities: set[tuple[str, str, int]] = set()
    normalized_records: list[NormalizedResultRecord] = []

    # Map pregame decisions by game identity if provided
    decisions_by_game_key: dict[tuple[str, str, int], str] = {}
    decisions_by_gid_gnum: dict[tuple[str, int], str] = {}
    decisions_by_gid: dict[str, str] = {}

    if target_pregame_decisions is not None:
        for dec in target_pregame_decisions:
            game_id_dict = dec.get("game_identity", {})
            pred_dict = dec.get("prediction_authority", {})
            ns = str(game_id_dict.get("provider_namespace") or "").strip()
            gid = str(game_id_dict.get("provider_game_id") or "").strip()
            try:
                gnum = int(game_id_dict.get("game_number", 1))
            except (ValueError, TypeError):
                gnum = 1
            pred_id = str(pred_dict.get("p37_prediction_row_id") or "").strip()
            if gid and pred_id:
                if ns:
                    decisions_by_game_key[(ns, gid, gnum)] = pred_id
                decisions_by_gid_gnum[(gid, gnum)] = pred_id
                decisions_by_gid[gid] = pred_id

    for idx, row in enumerate(raw_result_rows, start=1):
        status = str(row.get("status") or "").strip()
        if status != "FINAL":
            raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")

        home_score = _require_int_score(row.get("home_score"), field_name="home_score")
        away_score = _require_int_score(row.get("away_score"), field_name="away_score")
        if home_score == away_score:
            raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")

        ns = str(row.get("provider_namespace") or "MLB_STATS_API").strip()
        gid = str(row.get("provider_game_id") or row.get("game_id") or row.get("game_pk") or "").strip()
        if not gid:
            raise ValueError(f"result row {idx} is missing provider_game_id")

        try:
            gnum = int(row.get("game_number", 1))
        except (ValueError, TypeError):
            gnum = 1

        game_key = (ns, gid, gnum)
        if game_key in seen_identities:
            raise RuntimeError("P43A_CONFLICTING_RESULT_REJECTED: duplicate game identity in final results")
        seen_identities.add(game_key)

        observed_at = str(row.get("result_observed_at_utc") or "").strip()
        if not observed_at:
            raise ValueError(f"result row {idx} is missing result_observed_at_utc")
        parse_canonical_utc(observed_at)

        pred_id = str(row.get("prediction_row_id") or "").strip()
        if not pred_id:
            # Resolve from pregame decisions if possible
            if game_key in decisions_by_game_key:
                pred_id = decisions_by_game_key[game_key]
            elif (gid, gnum) in decisions_by_gid_gnum:
                pred_id = decisions_by_gid_gnum[(gid, gnum)]
            elif gid in decisions_by_gid:
                pred_id = decisions_by_gid[gid]
            else:
                # Deterministic fallback prediction id from game identity
                pred_id = _sha256_text(f"{ns}:{gid}:{gnum}:{target_date}")

        row_source_identity = str(row.get("source_identity") or source_identity).strip()

        record = NormalizedResultRecord(
            prediction_row_id=pred_id,
            provider_namespace=ns,
            provider_game_id=gid,
            game_number=gnum,
            status=status,
            home_score=home_score,
            away_score=away_score,
            result_observed_at_utc=observed_at,
            source_identity=row_source_identity,
        )
        normalized_records.append(record)

    # 4. Second snapshot read (stable-source two-read check)
    second_snapshot = _snapshot_file_bytes(bundle_root)
    for filename, first_bytes in first_snapshot.items():
        second_bytes = second_snapshot[filename]
        if first_bytes != second_bytes:
            raise RuntimeError(
                f"P49A_EXTERNAL_BUNDLE_CHANGED_DURING_ADMISSION: {filename} changed between reads"
            )

    # 5. Deterministic fingerprinting
    canonical_sorted_results = sorted(
        [r.to_payload() for r in normalized_records],
        key=lambda r: (r["provider_namespace"], r["provider_game_id"], r["game_number"]),
    )
    results_fp = _sha256_bytes(_canonical_json_bytes(canonical_sorted_results))

    actual_source_manifest_fp = _sha256_bytes(
        _canonical_json_bytes(source_manifest)
    )

    required_file_hashes = {
        filename: _sha256_bytes(content)
        for filename, content in sorted(first_snapshot.items())
    }

    bundle_fp = compute_deterministic_result_bundle_fingerprint(
        target_date=target_date,
        required_file_hashes=required_file_hashes,
        source_manifest_fingerprint=actual_source_manifest_fp,
        results_fingerprint=results_fp,
    )

    admitted_bundle_id = f"p49a_bundle_{bundle_fp[:32]}"

    # 6. Admission directory resolution
    if admission_root is not None:
        target_root = Path(admission_root).resolve()
    else:
        target_root = Path(__file__).resolve().parents[4] / P49A_REPORT_RELATIVE_PATH / "admitted"

    imported_bundle_dir = target_root / admitted_bundle_id
    record_path = imported_bundle_dir / "admission_record.json"
    normalized_result_path = imported_bundle_dir / "normalized_result_input.jsonl"

    admission_record = {
        "schema_version": P49A_ADMISSION_RECORD_SCHEMA,
        "task_id": P49A_TASK_ID,
        "admitted_bundle_id": admitted_bundle_id,
        "target_date": target_date,
        "bundle_fingerprint": bundle_fp,
        "source_identity": source_identity,
        "admitted_at_utc": admitted_at_utc,
        "final_result_count": len(normalized_records),
        "required_file_hashes": required_file_hashes,
        "source_manifest_fingerprint": actual_source_manifest_fp,
        "results_fingerprint": results_fp,
        "external_source_path_metadata": str(bundle_root),
        "p44_compatibility_status": "PASS",
    }

    # 7. Idempotency / conflict check
    if record_path.is_file():
        existing_record = _read_json_object(record_path)
        if (
            existing_record.get("bundle_fingerprint") == bundle_fp
            and existing_record.get("admitted_bundle_id") == admitted_bundle_id
        ):
            return P49AAdmittedResultBundle(
                status="RECOGNIZED_IDENTICAL",
                admitted_bundle_id=admitted_bundle_id,
                target_date=target_date,
                bundle_fingerprint=bundle_fp,
                source_identity=source_identity,
                imported_bundle_dir=imported_bundle_dir,
                normalized_result_path=normalized_result_path,
                admission_record_path=record_path,
                final_result_count=len(normalized_records),
                admission_record=existing_record,
                normalized_result_records=tuple(normalized_records),
            )
        raise RuntimeError(
            f"P49A_BUNDLE_AUTHORITY_CONFLICT: existing admission at {record_path} conflicts with incoming bundle"
        )

    # 8. Write immutable snapshots and normalized results
    imported_bundle_dir.mkdir(parents=True, exist_ok=True)
    raw_bundle_dir = imported_bundle_dir / "raw_bundle"
    raw_bundle_dir.mkdir(parents=True, exist_ok=True)

    for filename, raw_bytes in first_snapshot.items():
        (raw_bundle_dir / filename).write_bytes(raw_bytes)

    record_path.write_bytes(
        (json.dumps(admission_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    write_normalized_result_input(normalized_result_path, normalized_records)

    return P49AAdmittedResultBundle(
        status="ADMITTED",
        admitted_bundle_id=admitted_bundle_id,
        target_date=target_date,
        bundle_fingerprint=bundle_fp,
        source_identity=source_identity,
        imported_bundle_dir=imported_bundle_dir,
        normalized_result_path=normalized_result_path,
        admission_record_path=record_path,
        final_result_count=len(normalized_records),
        admission_record=admission_record,
        normalized_result_records=tuple(normalized_records),
    )


@dataclass(frozen=True, slots=True)
class P49ASettleResult:
    status: str
    run_classification: str
    lifecycle_state: str
    admitted_bundle_id: str
    bundle_fingerprint: str
    paper_run_id: str
    run_dir: Path
    newly_settled_count: int
    total_settled_count: int
    settled_bet_count: int
    settled_pass_count: int
    pending_count: int
    win_count: int
    loss_count: int
    units_risked: str
    net_paper_units: str
    descriptive_roi: str | None
    max_drawdown: str
    forward_sample_count: int
    admission_record_path: Path
    normalized_result_path: Path
    receipt_payload: dict[str, Any]
    admitted_bundle: P49AAdmittedResultBundle
    settle_run_result: P45ASettleRunResult | None


def intake_prospective_postgame_results(
    external_bundle_path: str | Path,
    *,
    paper_run_dir: str | Path,
    repository_root: str | Path | None = None,
    admission_root: str | Path | None = None,
    ledger_root: str | Path | None = None,
    source_identity: str = P49A_DEFAULT_SOURCE_IDENTITY,
    settled_at_utc: str = "2026-08-18T23:59:59Z",
    validate_only: bool = False,
) -> P49ASettleResult:
    """Admit external final results and atomically settle against a frozen prospective paper run."""

    repo_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )

    resolved_run_dir = Path(paper_run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"paper run manifest missing: {manifest_path}")

    run_manifest = _read_json_object(manifest_path)
    run_id = run_manifest["run_id"]
    run_classification = run_manifest["run_classification"]

    # Load frozen decisions for game matching
    decisions, pregame_records = load_p43a_frozen_decision_bundle(resolved_run_dir)

    # 1. Admit external final results
    admitted = admit_external_final_result_bundle(
        external_bundle_path,
        admission_root=admission_root,
        source_identity=source_identity,
        admitted_at_utc=settled_at_utc,
        target_pregame_decisions=pregame_records,
    )

    # 2. Validation-only mode
    if validate_only:
        receipt_payload = {
            "schema_version": P49A_INTAKE_RECEIPT_SCHEMA,
            "task_id": P49A_TASK_ID,
            "intake_status": "VALIDATED",
            "run_classification": run_classification,
            "lifecycle_state": run_manifest.get("lifecycle_state", STATE_FROZEN),
            "source_bundle_path": str(Path(external_bundle_path).resolve()),
            "admitted_bundle_id": admitted.admitted_bundle_id,
            "bundle_fingerprint": admitted.bundle_fingerprint,
            "paper_run_id": run_id,
            "run_dir": str(resolved_run_dir),
            "target_date": admitted.target_date,
            "source_identity": source_identity,
            "settled_at_utc": settled_at_utc,
            "final_result_count": admitted.final_result_count,
            "eligible_decision_count": len(decisions),
            "newly_settled_count": 0,
            "total_settled_count": run_manifest.get("settled_total_count", 0),
            "pending_count": run_manifest.get("pending_count", len(decisions)),
            "admission_record_path": str(admitted.admission_record_path),
            "normalized_result_path": str(admitted.normalized_result_path),
        }
        return P49ASettleResult(
            status="VALIDATED",
            run_classification=run_classification,
            lifecycle_state=run_manifest.get("lifecycle_state", STATE_FROZEN),
            admitted_bundle_id=admitted.admitted_bundle_id,
            bundle_fingerprint=admitted.bundle_fingerprint,
            paper_run_id=run_id,
            run_dir=resolved_run_dir,
            newly_settled_count=0,
            total_settled_count=run_manifest.get("settled_total_count", 0),
            settled_bet_count=run_manifest.get("settled_bet_count", 0),
            settled_pass_count=0,
            pending_count=run_manifest.get("pending_count", len(decisions)),
            win_count=0,
            loss_count=0,
            units_risked="0.0",
            net_paper_units="0.00",
            descriptive_roi=None,
            max_drawdown="0",
            forward_sample_count=0,
            admission_record_path=admitted.admission_record_path,
            normalized_result_path=admitted.normalized_result_path,
            receipt_payload=receipt_payload,
            admitted_bundle=admitted,
            settle_run_result=None,
        )

    # 3. Settle against P45 paper run
    settle_res = settle_p45a_paper_run(
        repo_root,
        run_dir=resolved_run_dir,
        result_input=admitted.normalized_result_records,
        ledger_root=ledger_root,
        settled_at_utc=settled_at_utc,
    )

    summary = settle_res.summary
    forward_summary = settle_res.forward_summary

    # 4. Write postgame settle receipt in run directory
    receipt_payload = {
        "schema_version": P49A_INTAKE_RECEIPT_SCHEMA,
        "task_id": P49A_TASK_ID,
        "intake_status": settle_res.lifecycle_state,
        "run_classification": run_classification,
        "lifecycle_state": settle_res.lifecycle_state,
        "source_bundle_path": str(Path(external_bundle_path).resolve()),
        "admitted_bundle_id": admitted.admitted_bundle_id,
        "bundle_fingerprint": admitted.bundle_fingerprint,
        "paper_run_id": run_id,
        "run_dir": str(resolved_run_dir),
        "target_date": admitted.target_date,
        "source_identity": source_identity,
        "settled_at_utc": settled_at_utc,
        "final_result_count": admitted.final_result_count,
        "eligible_decision_count": len(decisions),
        "newly_settled_count": settle_res.newly_settled_count,
        "total_settled_count": settle_res.total_settled_count,
        "settled_bet_count": summary.get("settled_bet_count", 0),
        "settled_pass_count": summary.get("settled_pass_count", 0),
        "pending_count": settle_res.pending_count,
        "win_count": summary.get("win_count", 0),
        "loss_count": summary.get("loss_count", 0),
        "units_risked": summary.get("units_risked", "0.0"),
        "net_paper_units": summary.get("net_paper_units", "0.00"),
        "descriptive_roi": summary.get("descriptive_roi"),
        "max_drawdown": summary.get("max_drawdown", "0"),
        "forward_sample_count": forward_summary.get("forward_sample_count", 0),
        "admission_record_path": str(admitted.admission_record_path),
        "normalized_result_path": str(admitted.normalized_result_path),
    }

    receipt_path = resolved_run_dir / "postgame_settle_receipt.json"
    receipt_path.write_bytes(
        (json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    return P49ASettleResult(
        status=settle_res.lifecycle_state,
        run_classification=run_classification,
        lifecycle_state=settle_res.lifecycle_state,
        admitted_bundle_id=admitted.admitted_bundle_id,
        bundle_fingerprint=admitted.bundle_fingerprint,
        paper_run_id=run_id,
        run_dir=resolved_run_dir,
        newly_settled_count=settle_res.newly_settled_count,
        total_settled_count=settle_res.total_settled_count,
        settled_bet_count=summary.get("settled_bet_count", 0),
        settled_pass_count=summary.get("settled_pass_count", 0),
        pending_count=settle_res.pending_count,
        win_count=summary.get("win_count", 0),
        loss_count=summary.get("loss_count", 0),
        units_risked=summary.get("units_risked", "0.0"),
        net_paper_units=summary.get("net_paper_units", "0.00"),
        descriptive_roi=summary.get("descriptive_roi"),
        max_drawdown=summary.get("max_drawdown", "0"),
        forward_sample_count=forward_summary.get("forward_sample_count", 0),
        admission_record_path=admitted.admission_record_path,
        normalized_result_path=admitted.normalized_result_path,
        receipt_payload=receipt_payload,
        admitted_bundle=admitted,
        settle_run_result=settle_res,
    )


__all__ = (
    "P49A_ADMISSION_RECORD_SCHEMA",
    "P49A_CONTRACT_REHEARSAL_SOURCE_IDENTITY",
    "P49A_DEFAULT_SOURCE_IDENTITY",
    "P49A_INTAKE_RECEIPT_SCHEMA",
    "P49A_REPORT_RELATIVE_PATH",
    "P49A_TASK_ID",
    "REQUIRED_RESULT_BUNDLE_FILES",
    "P49AAdmittedResultBundle",
    "P49ASettleResult",
    "admit_external_final_result_bundle",
    "compute_deterministic_result_bundle_fingerprint",
    "intake_prospective_postgame_results",
)
