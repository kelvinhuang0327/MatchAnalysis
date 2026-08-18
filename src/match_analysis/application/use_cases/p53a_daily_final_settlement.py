"""P53A daily Moneyline prospective prediction FINAL settlement use case.

Orchestrates the postgame daily settlement path:
  1. Locates and validates an existing immutable frozen P50C prospective prediction run.
  2. Acquires official MLB schedule and game finality status for frozen games only.
  3. Admits only authoritative FINAL games (with non-tied integer scores).
  4. Leaves scheduled, in-progress, postponed, and suspended games pending.
  5. Invokes existing P50C settlement and evaluation engine.
  6. Enforces byte-for-byte immutability of frozen predictions.
  7. Enforces complete separation of PREDICTION_FORWARD_SAMPLE_COUNT from betting FORWARD_SAMPLE_COUNT.
  8. Emits a postgame settlement receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    JsonOpener,
    _default_opener,
    fetch_json_bytes,
    format_utc,
    normalize_schedule_payload,
)
from .p44a_normalized_workflow_input import (
    NormalizedResultRecord,
    load_normalized_result_input,
)
from .p45a_paper_run_ledger import get_p45a_forward_summary
from .p50c_prediction_run_ledger import (
    P50C_REPORT_RELATIVE_PATH,
    STATE_FROZEN,
    STATE_PARTIALLY_SETTLED,
    STATE_SETTLED,
    get_p50c_forward_summary,
    get_p50c_run_status,
    read_json_object,
    read_jsonl_objects,
    settle_p50c_prediction_run,
)


P53A_TASK_ID = "P53A"
P53A_SOURCE_IDENTITY = "MLB_OFFICIAL_STATS_API_POSTGAME_FEED"
P53A_RECEIPT_SCHEMA = "p53a.daily_final_settlement_receipt.v1"


def _now_utc() -> str:
    return format_utc(datetime.now(UTC))


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class DailyFinalSettlementResult:
    run_id: str
    run_dir: Path
    target_date: str
    lifecycle_state: str
    eligible_prediction_count: int
    newly_settled_count: int
    total_settled_count: int
    pending_count: int
    final_results_discovered: int
    non_final_games_count: int
    accuracy: str | None
    brier_score: str | None
    log_loss: str | None
    expected_calibration_error: str | None
    prediction_forward_sample_count: int
    betting_forward_sample_count: int
    frozen_predictions_fingerprint_intact: bool
    status: str
    receipt_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": P53A_RECEIPT_SCHEMA,
            "task_id": P53A_TASK_ID,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "target_date": self.target_date,
            "lifecycle_state": self.lifecycle_state,
            "eligible_prediction_count": self.eligible_prediction_count,
            "newly_settled_count": self.newly_settled_count,
            "total_settled_count": self.total_settled_count,
            "pending_count": self.pending_count,
            "final_results_discovered": self.final_results_discovered,
            "non_final_games_count": self.non_final_games_count,
            "accuracy": self.accuracy,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "expected_calibration_error": self.expected_calibration_error,
            "prediction_forward_sample_count": self.prediction_forward_sample_count,
            "betting_forward_sample_count": self.betting_forward_sample_count,
            "frozen_predictions_fingerprint_intact": self.frozen_predictions_fingerprint_intact,
            "status": self.status,
            "receipt_path": str(self.receipt_path),
        }


def resolve_prediction_run(
    repository_root: str | Path,
    *,
    run: str | Path | None = None,
    target_date: str | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    """Resolve the directory path for a frozen P50C prediction run."""
    root = Path(repository_root).resolve()
    base_runs_dir = (
        Path(runs_root).resolve()
        if runs_root is not None
        else (root / P50C_REPORT_RELATIVE_PATH / "runs").resolve()
    )

    if run is not None:
        run_path = Path(run)
        if run_path.is_dir() and (run_path / "run_manifest.json").is_file():
            return run_path.resolve()
        candidate = base_runs_dir / str(run)
        if candidate.is_dir() and (candidate / "run_manifest.json").is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"prediction run directory not found for: {run}")

    if target_date is not None:
        clean_target = target_date.strip()
        if not base_runs_dir.is_dir():
            raise FileNotFoundError(f"runs root directory does not exist: {base_runs_dir}")

        matching_runs: list[Path] = []
        for child in sorted(base_runs_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_file = child / "run_manifest.json"
            pred_file = child / "frozen_predictions.jsonl"
            if not manifest_file.is_file() or not pred_file.is_file():
                continue
            try:
                preds = read_jsonl_objects(pred_file)
            except Exception:
                continue
            for p in preds:
                g_id = p.get("game_identity", {})
                off_date = str(g_id.get("official_date") or "").strip()
                sched_utc = str(g_id.get("scheduled_start_utc") or "").strip()
                if off_date == clean_target or sched_utc.startswith(clean_target):
                    matching_runs.append(child.resolve())
                    break

        if len(matching_runs) == 1:
            return matching_runs[0]
        if len(matching_runs) > 1:
            # If multiple match, sort by created_at_utc descending from manifest
            def _created_key(path: Path) -> str:
                try:
                    return str(read_json_object(path / "run_manifest.json").get("created_at_utc", ""))
                except Exception:
                    return ""
            sorted_runs = sorted(matching_runs, key=_created_key, reverse=True)
            return sorted_runs[0]

        raise FileNotFoundError(f"no frozen prediction run found for target date: {target_date}")

    # Fallback: check if there is exactly one prospective run in base_runs_dir
    if base_runs_dir.is_dir():
        all_runs = [
            child.resolve()
            for child in base_runs_dir.iterdir()
            if child.is_dir() and (child / "run_manifest.json").is_file()
        ]
        if len(all_runs) == 1:
            return all_runs[0]

    raise ValueError("either --run or --target-date is required to identify the prediction run")


def acquire_official_final_results_for_run(
    frozen_predictions: Sequence[Mapping[str, Any]],
    *,
    opener: JsonOpener | None = None,
    observed_at_utc: str | None = None,
    source_identity: str = P53A_SOURCE_IDENTITY,
) -> tuple[tuple[NormalizedResultRecord, ...], dict[str, int]]:
    """Fetch and filter official MLB game status for games in the frozen run only."""
    if not frozen_predictions:
        return (), {}

    actual_opener = opener if opener is not None else _default_opener
    actual_observed = observed_at_utc or _now_utc()

    # Collect official dates and game identities present in the frozen run
    official_dates: set[str] = set()
    target_by_game_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    target_by_pk: dict[int, Mapping[str, Any]] = {}

    for pred in frozen_predictions:
        game_id = pred["game_identity"]
        gpk = int(game_id["game_pk"])
        gid = str(game_id["provider_game_id"])
        gnum = int(game_id.get("game_number", 1))
        off_date = str(game_id.get("official_date") or "").strip()
        if off_date:
            official_dates.add(off_date)
        else:
            sched = str(game_id.get("scheduled_start_utc") or "")[:10]
            if sched:
                official_dates.add(sched)

        target_by_game_key[(gid, gnum)] = pred
        target_by_pk[gpk] = pred

    if not official_dates:
        return (), {}

    sorted_dates = sorted(official_dates)
    start_date = sorted_dates[0]
    end_date = sorted_dates[-1]

    # Fetch official schedule from MLB Stats API
    url = f"{MLB_STATS_API_BASE}/schedule"
    query = {
        "sportId": "1",
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team",
    }
    raw_bytes, req_url = fetch_json_bytes(url, query=query, opener=actual_opener, timeout=60)
    parsed_payload = json.loads(raw_bytes.decode("utf-8"))
    normalized_schedule_rows = normalize_schedule_payload(parsed_payload)

    final_records: list[NormalizedResultRecord] = []
    status_counts: dict[str, int] = {}

    for sched_row in normalized_schedule_rows:
        gpk = int(sched_row["game_pk"])
        gid = str(sched_row["provider_game_id"])
        gnum = int(sched_row["game_number"])

        matching_pred = target_by_game_key.get((gid, gnum))
        if matching_pred is None:
            matching_pred = target_by_pk.get(gpk)

        if matching_pred is None:
            # Game was not in frozen predictions; ignore
            continue

        status_name = str(sched_row.get("status") or "UNKNOWN")
        status_counts[status_name] = status_counts.get(status_name, 0) + 1

        is_final = sched_row.get("final") is True or status_name.lower() == "final"

        if is_final:
            home_score = sched_row.get("home_score")
            away_score = sched_row.get("away_score")
            if (
                not isinstance(home_score, int)
                or isinstance(home_score, bool)
                or not isinstance(away_score, int)
                or isinstance(away_score, bool)
            ):
                raise RuntimeError(
                    f"P53A_INVALID_FINAL_SCORE: final game {gid} has non-integer score: home={home_score} away={away_score}"
                )
            if home_score == away_score:
                raise RuntimeError(
                    f"P53A_TIED_FINAL_SCORE_REJECTED: final game {gid} tied {home_score}-{away_score}"
                )

            pred_auth = matching_pred.get("prediction_authority", {})
            pred_id = str(pred_auth.get("p37_prediction_row_id") or "")

            record = NormalizedResultRecord(
                prediction_row_id=pred_id,
                provider_namespace=str(matching_pred["game_identity"]["provider_namespace"]),
                provider_game_id=gid,
                game_number=gnum,
                status="FINAL",
                home_score=home_score,
                away_score=away_score,
                result_observed_at_utc=actual_observed,
                source_identity=source_identity,
            )
            final_records.append(record)

    return tuple(final_records), status_counts


def execute_daily_moneyline_final_settlement(
    *,
    run: str | Path | None = None,
    target_date: str | None = None,
    repository_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    ledger_root: str | Path | None = None,
    result_input: Sequence[NormalizedResultRecord] | str | Path | None = None,
    observed_at_utc: str | None = None,
    opener: JsonOpener | None = None,
) -> DailyFinalSettlementResult:
    """Execute daily Moneyline prospective prediction FINAL settlement."""
    repo_root = Path(repository_root or Path.cwd()).resolve()

    run_dir = resolve_prediction_run(
        repo_root,
        run=run,
        target_date=target_date,
        runs_root=runs_root,
    )

    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "frozen_predictions.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction run manifest missing: {manifest_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"frozen predictions file missing: {predictions_path}")

    # Snapshot pregame frozen predictions bytes to guarantee strict byte immutability
    frozen_bytes_before = predictions_path.read_bytes()
    frozen_fp_before = sha256(frozen_bytes_before).hexdigest()

    manifest = read_json_object(manifest_path)
    run_id = str(manifest["run_id"])
    eligible_count = int(manifest.get("eligible_prediction_count", 0))

    frozen_predictions = read_jsonl_objects(predictions_path)
    if not frozen_predictions:
        raise RuntimeError("P53A_EMPTY_FROZEN_PREDICTIONS: run contains no frozen predictions")

    # Inferred target date from frozen predictions
    dates = [
        str(p["game_identity"]["official_date"])
        for p in frozen_predictions
        if p.get("game_identity", {}).get("official_date")
    ]
    slate_target_date = target_date or (dates[0] if dates else "UNKNOWN")

    # Capture betting forward sample count before settlement
    p45a_summary_before = get_p45a_forward_summary(repo_root)
    betting_count_before = int(p45a_summary_before.get("forward_sample_count", 0))

    actual_observed = observed_at_utc or _now_utc()
    parse_canonical_utc(actual_observed)

    # Acquire results: either from supplied offline result_input or live MLB Stats API
    if result_input is not None:
        if isinstance(result_input, (str, Path)):
            all_input_results = load_normalized_result_input(result_input)
        else:
            all_input_results = tuple(result_input)

        # Filter only results that match frozen predictions in this run
        target_game_keys = {
            (
                str(p["game_identity"]["provider_namespace"]),
                str(p["game_identity"]["provider_game_id"]),
                int(p["game_identity"].get("game_number", 1)),
            ): p
            for p in frozen_predictions
        }
        target_pred_ids = {
            str(p.get("prediction_authority", {}).get("p37_prediction_row_id")): p
            for p in frozen_predictions
            if p.get("prediction_authority", {}).get("p37_prediction_row_id")
        }

        final_records: list[NormalizedResultRecord] = []
        for r in all_input_results:
            key = (r.provider_namespace, r.provider_game_id, r.game_number)
            if key in target_game_keys or r.prediction_row_id in target_pred_ids:
                final_records.append(r)

        non_final_count = eligible_count - len(final_records)
    else:
        final_tuple, status_counts = acquire_official_final_results_for_run(
            frozen_predictions,
            opener=opener,
            observed_at_utc=actual_observed,
        )
        final_records = list(final_tuple)
        non_final_count = eligible_count - len(final_records)

    # Invoke existing P50C settlement engine
    settle_res = settle_p50c_prediction_run(
        repo_root,
        run_dir=run_dir,
        result_input=tuple(final_records),
        ledger_root=ledger_root,
        settled_at_utc=actual_observed,
    )

    # Verify byte-for-byte immutability of frozen predictions
    frozen_bytes_after = predictions_path.read_bytes()
    frozen_fp_after = sha256(frozen_bytes_after).hexdigest()
    if frozen_fp_before != frozen_fp_after or frozen_bytes_before != frozen_bytes_after:
        raise RuntimeError("P53A_PREDICTION_FREEZE_MUTATION_DETECTED: frozen predictions were modified")

    # Verify betting forward count invariance
    p45a_summary_after = get_p45a_forward_summary(repo_root)
    betting_count_after = int(p45a_summary_after.get("forward_sample_count", 0))
    if betting_count_before != betting_count_after:
        raise RuntimeError(
            f"P53A_BETTING_FORWARD_COUNT_CORRUPTED: betting count changed from {betting_count_before} to {betting_count_after}"
        )

    summary = settle_res.summary
    forward_summary = settle_res.forward_summary

    status_str = (
        "NEWLY_SETTLED"
        if settle_res.newly_settled_count > 0
        else ("IDEMPOTENT_NO_CHANGE" if settle_res.total_settled_count > 0 else "ALL_PENDING")
    )

    receipt_payload = {
        "schema_version": P53A_RECEIPT_SCHEMA,
        "task_id": P53A_TASK_ID,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "target_date": slate_target_date,
        "lifecycle_state": settle_res.lifecycle_state,
        "settled_at_utc": actual_observed,
        "eligible_prediction_count": eligible_count,
        "newly_settled_count": settle_res.newly_settled_count,
        "total_settled_count": settle_res.total_settled_count,
        "pending_count": settle_res.pending_count,
        "final_results_discovered": len(final_records),
        "non_final_games_count": non_final_count,
        "accuracy": summary.get("accuracy"),
        "brier_score": summary.get("brier_score"),
        "log_loss": summary.get("log_loss"),
        "expected_calibration_error": summary.get("expected_calibration_error"),
        "prediction_forward_sample_count": forward_summary.get("PREDICTION_FORWARD_SAMPLE_COUNT", 0),
        "betting_forward_sample_count": betting_count_after,
        "frozen_predictions_fingerprint_intact": True,
        "status": status_str,
    }

    receipt_path = run_dir / "p53a_final_settlement_receipt.json"
    receipt_path.write_bytes(
        (json.dumps(receipt_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    return DailyFinalSettlementResult(
        run_id=run_id,
        run_dir=run_dir,
        target_date=slate_target_date,
        lifecycle_state=settle_res.lifecycle_state,
        eligible_prediction_count=eligible_count,
        newly_settled_count=settle_res.newly_settled_count,
        total_settled_count=settle_res.total_settled_count,
        pending_count=settle_res.pending_count,
        final_results_discovered=len(final_records),
        non_final_games_count=non_final_count,
        accuracy=summary.get("accuracy"),
        brier_score=summary.get("brier_score"),
        log_loss=summary.get("log_loss"),
        expected_calibration_error=summary.get("expected_calibration_error"),
        prediction_forward_sample_count=forward_summary.get("PREDICTION_FORWARD_SAMPLE_COUNT", 0),
        betting_forward_sample_count=betting_count_after,
        frozen_predictions_fingerprint_intact=True,
        status=status_str,
        receipt_path=receipt_path,
    )


__all__ = (
    "P53A_RECEIPT_SCHEMA",
    "P53A_SOURCE_IDENTITY",
    "P53A_TASK_ID",
    "DailyFinalSettlementResult",
    "acquire_official_final_results_for_run",
    "execute_daily_moneyline_final_settlement",
    "resolve_prediction_run",
)
