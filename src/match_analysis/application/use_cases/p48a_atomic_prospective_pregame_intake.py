"""P48A atomic prospective pregame intake orchestration.

Composes the verified pipeline:
external frozen P35A bundle
  → P47 admission
  → P46 adapter
  → P44 normalized pregame input
  → P45 create-run (PROSPECTIVE_FORWARD_PAPER classification)
  → immutable frozen paper run.

Fails closed on any error, never downgrades prospective classification to historical,
rejects outcome contamination, enforces idempotency on retries, and preserves
strict outcome isolation without duplicating existing business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .p44a_normalized_workflow_input import NormalizedPregameInput
from .p45a_paper_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    P45ACreateRunResult,
    canonical_pregame_fingerprint,
    create_p45a_paper_run,
    validate_run_classification,
)
from .p47a_external_bundle_admission import (
    P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY,
    P47AAdmittedBundle,
    admit_external_p35a_bundle,
)


P48A_TASK_ID = "P48A"
P48A_INTAKE_RECEIPT_SCHEMA = "p48a.prospective_pregame_intake_receipt.v1"
P48A_DEFAULT_SOURCE_IDENTITY = P47A_EXTERNAL_ADMISSION_SOURCE_IDENTITY
P48A_REPORT_RELATIVE_PATH = Path("report/p48a_atomic_prospective_pregame_intake")


@dataclass(frozen=True, slots=True)
class P48AIntakeResult:
    status: str
    run_classification: str
    lifecycle_state: str
    admitted_bundle_id: str
    bundle_fingerprint: str
    source_run_id: str
    target_date: str
    paper_run_id: str | None
    run_dir: Path | None
    normalized_input_fingerprint: str
    decision_bundle_fingerprint: str | None
    target_universe_count: int
    eligible_decision_count: int
    bet_count: int
    pass_count: int
    exclusion_count: int
    admission_record_path: Path
    normalized_pregame_path: Path
    receipt_payload: dict[str, Any]
    admitted_bundle: P47AAdmittedBundle
    create_run_result: P45ACreateRunResult | None


def intake_prospective_pregame_bundle(
    external_bundle_path: str | Path,
    *,
    repository_root: str | Path | None = None,
    admission_root: str | Path | None = None,
    run_root: str | Path | None = None,
    source_identity: str = P48A_DEFAULT_SOURCE_IDENTITY,
    intake_timestamp_utc: str = "2026-08-17T12:00:00Z",
    validate_only: bool = False,
) -> P48AIntakeResult:
    """Admit an external P35A bundle and atomically create a prospective paper run."""

    repo_root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )

    # 1. P47 admission (validates bundle files, schemas, temporal bounds, and invokes P46 adapter)
    admitted = admit_external_p35a_bundle(
        external_bundle_path,
        admission_root=admission_root,
        source_identity=source_identity,
        admitted_at_utc=intake_timestamp_utc,
    )

    norm_fp = canonical_pregame_fingerprint(admitted.normalized_pregame_input)

    # 2. Validation-only mode: verify classification eligibility without creating P45 run
    if validate_only:
        validate_run_classification(
            CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
            pregame_input=admitted.normalized_pregame_input,
            created_at_utc=intake_timestamp_utc,
        )

        receipt_payload = {
            "schema_version": P48A_INTAKE_RECEIPT_SCHEMA,
            "task_id": P48A_TASK_ID,
            "intake_status": "VALIDATED",
            "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
            "lifecycle_state": "PRE_INTAKE",
            "source_bundle_path": str(Path(external_bundle_path).resolve()),
            "admitted_bundle_id": admitted.admitted_bundle_id,
            "bundle_fingerprint": admitted.bundle_fingerprint,
            "source_run_id": admitted.run_id,
            "target_date": admitted.target_date,
            "source_identity": source_identity,
            "intake_timestamp_utc": intake_timestamp_utc,
            "paper_run_id": None,
            "run_dir": None,
            "normalized_input_fingerprint": norm_fp,
            "decision_bundle_fingerprint": None,
            "target_universe_count": len(admitted.normalized_pregame_input.prediction_rows)
            + len(admitted.normalized_pregame_input.exclusion_rows),
            "eligible_decision_count": len(admitted.normalized_pregame_input.prediction_rows),
            "bet_count": 0,
            "pass_count": 0,
            "exclusion_count": len(admitted.normalized_pregame_input.exclusion_rows),
            "admission_record_path": str(admitted.admission_record_path),
            "normalized_pregame_path": str(admitted.normalized_pregame_path),
        }

        return P48AIntakeResult(
            status="VALIDATED",
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
            lifecycle_state="PRE_INTAKE",
            admitted_bundle_id=admitted.admitted_bundle_id,
            bundle_fingerprint=admitted.bundle_fingerprint,
            source_run_id=admitted.run_id,
            target_date=admitted.target_date,
            paper_run_id=None,
            run_dir=None,
            normalized_input_fingerprint=norm_fp,
            decision_bundle_fingerprint=None,
            target_universe_count=len(admitted.normalized_pregame_input.prediction_rows)
            + len(admitted.normalized_pregame_input.exclusion_rows),
            eligible_decision_count=len(admitted.normalized_pregame_input.prediction_rows),
            bet_count=0,
            pass_count=0,
            exclusion_count=len(admitted.normalized_pregame_input.exclusion_rows),
            admission_record_path=admitted.admission_record_path,
            normalized_pregame_path=admitted.normalized_pregame_path,
            receipt_payload=receipt_payload,
            admitted_bundle=admitted,
            create_run_result=None,
        )

    # 3. P45 prospective paper run creation (strictly enforces prospective classification)
    create_result = create_p45a_paper_run(
        repo_root,
        pregame_input=admitted.normalized_pregame_input,
        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
        run_root=run_root,
        created_at_utc=intake_timestamp_utc,
    )

    manifest = create_result.manifest
    actual_classification = manifest.get("run_classification")
    if actual_classification != CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER:
        raise RuntimeError(
            f"P48A_PROSPECTIVE_CLASSIFICATION_FAILED: run classification was {actual_classification!r}, "
            f"expected {CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER!r}"
        )

    # 4. Build and write minimal intake receipt
    receipt_payload = {
        "schema_version": P48A_INTAKE_RECEIPT_SCHEMA,
        "task_id": P48A_TASK_ID,
        "intake_status": create_result.status,
        "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
        "lifecycle_state": manifest.get("lifecycle_state", STATE_FROZEN),
        "source_bundle_path": str(Path(external_bundle_path).resolve()),
        "admitted_bundle_id": admitted.admitted_bundle_id,
        "bundle_fingerprint": admitted.bundle_fingerprint,
        "source_run_id": admitted.run_id,
        "target_date": admitted.target_date,
        "source_identity": source_identity,
        "intake_timestamp_utc": intake_timestamp_utc,
        "paper_run_id": create_result.run_id,
        "run_dir": str(create_result.run_dir),
        "normalized_input_fingerprint": manifest.get("normalized_input_fingerprint", norm_fp),
        "decision_bundle_fingerprint": manifest.get("decision_bundle_fingerprint"),
        "target_universe_count": manifest.get("target_universe_count", 0),
        "eligible_decision_count": manifest.get("eligible_decision_count", 0),
        "bet_count": manifest.get("bet_count", 0),
        "pass_count": manifest.get("pass_count", 0),
        "exclusion_count": manifest.get("exclusion_count", 0),
        "admission_record_path": str(admitted.admission_record_path),
        "normalized_pregame_path": str(admitted.normalized_pregame_path),
    }

    receipt_path = create_result.run_dir / "intake_receipt.json"
    if not receipt_path.is_file():
        receipt_path.write_bytes(
            (json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

    return P48AIntakeResult(
        status=create_result.status,
        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
        lifecycle_state=manifest.get("lifecycle_state", STATE_FROZEN),
        admitted_bundle_id=admitted.admitted_bundle_id,
        bundle_fingerprint=admitted.bundle_fingerprint,
        source_run_id=admitted.run_id,
        target_date=admitted.target_date,
        paper_run_id=create_result.run_id,
        run_dir=create_result.run_dir,
        normalized_input_fingerprint=manifest.get("normalized_input_fingerprint", norm_fp),
        decision_bundle_fingerprint=manifest.get("decision_bundle_fingerprint"),
        target_universe_count=manifest.get("target_universe_count", 0),
        eligible_decision_count=manifest.get("eligible_decision_count", 0),
        bet_count=manifest.get("bet_count", 0),
        pass_count=manifest.get("pass_count", 0),
        exclusion_count=manifest.get("exclusion_count", 0),
        admission_record_path=admitted.admission_record_path,
        normalized_pregame_path=admitted.normalized_pregame_path,
        receipt_payload=receipt_payload,
        admitted_bundle=admitted,
        create_run_result=create_result,
    )


__all__ = (
    "P48A_DEFAULT_SOURCE_IDENTITY",
    "P48A_INTAKE_RECEIPT_SCHEMA",
    "P48A_REPORT_RELATIVE_PATH",
    "P48A_TASK_ID",
    "P48AIntakeResult",
    "intake_prospective_pregame_bundle",
)
