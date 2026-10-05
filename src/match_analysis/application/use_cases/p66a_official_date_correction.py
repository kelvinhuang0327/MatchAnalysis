"""P66A provider-backed, append-only official-date correction materialization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...infrastructure.providers.mlb_official_historical_source import (
    MLB_STATS_API_BASE,
    JsonOpener,
    fetch_json_bytes,
    sha256_bytes,
)
from .p50c_prediction_run_ledger import (
    P50C_REPORT_RELATIVE_PATH,
    P66A_OFFICIAL_DATE_CORRECTION_FILENAME,
    P66A_OFFICIAL_DATE_CORRECTION_SCHEMA,
    apply_official_date_corrections,
    compute_forward_prediction_summary,
    read_json_object,
    read_jsonl_objects,
    require_canonical_official_date,
)


P66A_TASK_ID = "P66A"
P66A_CORRECTION_VERSION = "v1"
P66A_RECEIPT_SCHEMA = "p66a.official_date_correction_receipt.v1"
P66A_RECEIPT_FILENAME = "p66a_official_date_correction_v1_receipt.json"
P66A_PROVIDER_SOURCE_IDENTITY = "MLB_STATS_API_OFFICIAL_SCHEDULE"
P66A_CORRECTION_REASON = "UTC_CALENDAR_DATE_REPLACED_BY_MLB_PROVIDER_OFFICIAL_DATE"
P66A_ROLLBACK_MODE = "IGNORE_OVERLAY_AND_REGENERATE_FROM_RAW_LEDGER"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class P66AOfficialDateCorrectionResult:
    raw_ledger_path: Path
    correction_path: Path
    receipt_path: Path
    summary_path: Path
    provider_resolution_count: int
    correction_count: int
    raw_row_count: int
    raw_ledger_sha256: str
    correction_artifact_sha256: str
    post_correction_summary_sha256: str
    effective_grouping: dict[str, int]
    corrected_ledger_ids: tuple[str, ...]


def _provider_query_window(raw_records: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    scheduled_dates: list[date] = []
    for row in raw_records:
        scheduled = str(row.get("game_identity", {}).get("scheduled_start_utc") or "")
        if len(scheduled) < 10:
            raise ValueError("P66A_SCHEDULED_START_REQUIRED: raw ledger row lacks scheduled_start_utc")
        try:
            scheduled_dates.append(date.fromisoformat(scheduled[:10]))
        except ValueError as exc:
            raise ValueError("P66A_SCHEDULED_START_INVALID: expected canonical UTC timestamp") from exc
    if not scheduled_dates:
        raise ValueError("P66A_EMPTY_RAW_LEDGER")
    # One day of padding keeps a rescheduled official bucket inside the bounded query.
    return (
        (min(scheduled_dates) - timedelta(days=1)).isoformat(),
        (max(scheduled_dates) + timedelta(days=1)).isoformat(),
    )


def resolve_provider_official_dates(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    opener: JsonOpener | None = None,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Resolve every raw official game PK exactly once from MLB schedule authority."""
    start_date, end_date = _provider_query_window(raw_records)
    query = {
        "sportId": "1",
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team",
    }
    if opener is None:
        raw_payload, request_url = fetch_json_bytes(
            f"{MLB_STATS_API_BASE}/schedule",
            query=query,
            timeout=60,
        )
    else:
        raw_payload, request_url = fetch_json_bytes(
            f"{MLB_STATS_API_BASE}/schedule",
            query=query,
            opener=opener,
            timeout=60,
        )
    payload = json.loads(raw_payload.decode("utf-8"))

    provider_matches: dict[int, list[dict[str, Any]]] = {}
    for bucket in payload.get("dates", []):
        for game in bucket.get("games", []):
            game_pk = game.get("gamePk")
            if isinstance(game_pk, bool) or not isinstance(game_pk, int):
                continue
            provider_matches.setdefault(game_pk, []).append(
                {
                    "official_date": game.get("officialDate"),
                    "bucket_date": bucket.get("date"),
                    "scheduled_start_utc": game.get("gameDate"),
                }
            )

    resolved: dict[int, str] = {}
    row_resolution_count = 0
    for raw_row in raw_records:
        game_pk = raw_row.get("game_identity", {}).get("game_pk")
        if isinstance(game_pk, bool) or not isinstance(game_pk, int):
            raise ValueError("P66A_OFFICIAL_GAME_PK_REQUIRED")
        matches = provider_matches.get(game_pk, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"P66A_PROVIDER_IDENTITY_AMBIGUOUS: game_pk {game_pk} resolved {len(matches)} times"
            )
        official_date = require_canonical_official_date(
            matches[0].get("official_date"),
            field_name=f"MLB game {game_pk} officialDate",
        )
        previous = resolved.get(game_pk)
        if previous is not None and previous != official_date:
            raise RuntimeError(f"P66A_PROVIDER_DATE_CONFLICT: game_pk {game_pk}")
        resolved[game_pk] = official_date
        row_resolution_count += 1

    source_evidence = {
        "source_identity": P66A_PROVIDER_SOURCE_IDENTITY,
        "request_url": request_url,
        "payload_sha256": sha256_bytes(raw_payload),
        "query_start_date": start_date,
        "query_end_date": end_date,
        "provider_game_count": sum(len(values) for values in provider_matches.values()),
        "resolved_row_count": row_resolution_count,
        "resolved_unique_game_pk_count": len(resolved),
    }
    return resolved, source_evidence


def build_official_date_corrections(
    raw_records: Sequence[Mapping[str, Any]],
    provider_dates: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Build one deterministic correction row for each provider-confirmed mismatch."""
    corrections: list[dict[str, Any]] = []
    seen_raw_ids: set[str] = set()
    for raw_row in raw_records:
        ledger_id = str(raw_row.get("ledger_record_id") or "").strip()
        if not ledger_id:
            raise ValueError("P66A_RAW_LEDGER_ID_REQUIRED")
        if ledger_id in seen_raw_ids:
            raise ValueError(f"P66A_DUPLICATE_RAW_LEDGER_ID: {ledger_id}")
        seen_raw_ids.add(ledger_id)

        game_identity = raw_row.get("game_identity", {})
        prediction_identity = raw_row.get("prediction_identity", {})
        game_pk = game_identity.get("game_pk")
        if game_pk not in provider_dates:
            raise RuntimeError(f"P66A_PROVIDER_IDENTITY_UNRESOLVED: game_pk {game_pk}")
        original_date = require_canonical_official_date(
            game_identity.get("official_date"),
            field_name=f"raw ledger {ledger_id} official_date",
        )
        corrected_date = require_canonical_official_date(
            provider_dates[game_pk],
            field_name=f"provider game {game_pk} officialDate",
        )
        if original_date == corrected_date:
            continue

        corrections.append(
            {
                "schema_version": P66A_OFFICIAL_DATE_CORRECTION_SCHEMA,
                "ledger_id": ledger_id,
                "run_id": raw_row.get("run_id"),
                "prediction_id": prediction_identity.get("workflow_prediction_id"),
                "official_game_pk": game_pk,
                "original_official_date": original_date,
                "corrected_official_date": corrected_date,
                "source_identity": P66A_PROVIDER_SOURCE_IDENTITY,
                "correction_reason": P66A_CORRECTION_REASON,
                "correction_version": P66A_CORRECTION_VERSION,
            }
        )
    return corrections


def effective_market_date_grouping(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(
        (
            str(row.get("market_id") or "moneyline"),
            str(row.get("game_identity", {}).get("official_date") or ""),
        )
        for row in records
    )
    return {
        f"{official_date}/{market}": count
        for (market, official_date), count in sorted(counts.items())
    }


def _identity_projection(record: Mapping[str, Any]) -> tuple[Any, ...]:
    prediction = record.get("prediction_identity", {})
    game = record.get("game_identity", {})
    return (
        record.get("ledger_record_id"),
        record.get("run_id"),
        prediction.get("workflow_prediction_id"),
        prediction.get("prediction_fingerprint"),
        game.get("game_pk"),
        game.get("provider_game_id"),
        game.get("game_number"),
    )


def execute_p66a_official_date_correction(
    repository_root: str | Path,
    *,
    opener: JsonOpener | None = None,
) -> P66AOfficialDateCorrectionResult:
    """Materialize the authorized overlay, receipt, and effective summary only."""
    root = Path(repository_root).resolve()
    report_root = root / P50C_REPORT_RELATIVE_PATH
    ledger_root = report_root / "ledger"
    correction_root = report_root / "corrections"
    raw_ledger_path = ledger_root / "prediction_forward_ledger.jsonl"
    summary_path = ledger_root / "forward_prediction_summary.json"
    correction_path = correction_root / P66A_OFFICIAL_DATE_CORRECTION_FILENAME
    receipt_path = correction_root / P66A_RECEIPT_FILENAME

    raw_ledger_bytes = raw_ledger_path.read_bytes()
    raw_records = read_jsonl_objects(raw_ledger_path)
    if not raw_records:
        raise RuntimeError("P66A_EMPTY_RAW_LEDGER")
    pre_summary_bytes = summary_path.read_bytes()
    pre_summary = read_json_object(summary_path)
    computed_raw_summary = compute_forward_prediction_summary(raw_records)
    if pre_summary != computed_raw_summary:
        raise RuntimeError(
            "P66A_PRE_CORRECTION_SUMMARY_DRIFT: existing summary does not match immutable raw ledger"
        )

    provider_dates, provider_evidence = resolve_provider_official_dates(
        raw_records,
        opener=opener,
    )
    corrections = build_official_date_corrections(raw_records, provider_dates)
    effective_records = apply_official_date_corrections(raw_records, corrections)

    if [_identity_projection(row) for row in raw_records] != [
        _identity_projection(row) for row in effective_records
    ]:
        raise RuntimeError("P66A_IDENTITY_INVARIANCE_FAILED")

    post_summary = compute_forward_prediction_summary(effective_records)
    def non_date_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
        # Date spans change in the headline and each independently grouped cohort.
        # All cohort identities, counts, and model metrics remain protected.
        date_fields = {"first_target_date", "last_target_date"}
        projection = {key: value for key, value in summary.items() if key not in date_fields}
        projection["game_type_cohorts"] = {
            game_type: {key: value for key, value in cohort.items() if key not in date_fields}
            for game_type, cohort in summary.get("game_type_cohorts", {}).items()
        }
        return projection

    if non_date_metrics(pre_summary) != non_date_metrics(post_summary):
        raise RuntimeError("P66A_NON_DATE_METRIC_INVARIANCE_FAILED")

    correction_bytes = _jsonl_bytes(corrections)
    post_summary_bytes = _json_bytes(post_summary)
    receipt = {
        "receipt_schema_version": P66A_RECEIPT_SCHEMA,
        "task_id": P66A_TASK_ID,
        "source_identity": P66A_PROVIDER_SOURCE_IDENTITY,
        "provider_source_identity": provider_evidence,
        "raw_ledger_sha256": _sha256_bytes(raw_ledger_bytes),
        "raw_ledger_row_count": len(raw_records),
        "correction_artifact_sha256": _sha256_bytes(correction_bytes),
        "correction_record_count": len(corrections),
        "pre_correction_summary_sha256": _sha256_bytes(pre_summary_bytes),
        "post_correction_summary_sha256": _sha256_bytes(post_summary_bytes),
        "rollback_mode": P66A_ROLLBACK_MODE,
        "corrected_ledger_ids": [row["ledger_id"] for row in corrections],
    }

    correction_root.mkdir(parents=True, exist_ok=True)
    correction_path.write_bytes(correction_bytes)
    summary_path.write_bytes(post_summary_bytes)
    receipt_path.write_bytes(_json_bytes(receipt))

    return P66AOfficialDateCorrectionResult(
        raw_ledger_path=raw_ledger_path,
        correction_path=correction_path,
        receipt_path=receipt_path,
        summary_path=summary_path,
        provider_resolution_count=int(provider_evidence["resolved_row_count"]),
        correction_count=len(corrections),
        raw_row_count=len(raw_records),
        raw_ledger_sha256=_sha256_bytes(raw_ledger_bytes),
        correction_artifact_sha256=_sha256_bytes(correction_bytes),
        post_correction_summary_sha256=_sha256_bytes(post_summary_bytes),
        effective_grouping=effective_market_date_grouping(effective_records),
        corrected_ledger_ids=tuple(row["ledger_id"] for row in corrections),
    )


__all__ = (
    "P66A_CORRECTION_REASON",
    "P66A_CORRECTION_VERSION",
    "P66A_PROVIDER_SOURCE_IDENTITY",
    "P66A_RECEIPT_FILENAME",
    "P66A_RECEIPT_SCHEMA",
    "P66A_ROLLBACK_MODE",
    "P66AOfficialDateCorrectionResult",
    "build_official_date_corrections",
    "effective_market_date_grouping",
    "execute_p66a_official_date_correction",
    "resolve_provider_official_dates",
)
