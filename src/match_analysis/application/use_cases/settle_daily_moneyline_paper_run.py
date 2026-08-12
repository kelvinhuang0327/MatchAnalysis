"""Settle one frozen P33A daily Moneyline paper run.

This use case is deliberately provider-free.  The CLI (or another boundary
adapter) acquires and freezes official result observations, then passes the
frozen JSONL and provenance here.  P34A adapts only complete P33A prediction
rows into the existing P15C/P16A/P16B/P17A contracts.  Structural and
feature-unavailable rows remain outside that lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.final_result_observation import (
    FinalResultObservation,
    load_final_result_observations,
)
from .admitted_prediction_observation_artifacts import (
    render_admitted_observations_jsonl,
    render_snapshot_report_markdown,
    render_snapshot_summary_json,
)
from .attach_final_results_to_admitted_predictions import (
    FinalResultAttachmentResult,
    attach_final_results_to_admitted_predictions,
)
from .build_admitted_prediction_observation_snapshot import (
    AdmittedPredictionObservationSnapshotResult,
    build_admitted_prediction_observation_snapshot,
)
from .build_prediction_evaluation_scorecard import (
    PredictionEvaluationScorecardResult,
    build_prediction_evaluation_scorecard,
)
from .build_prediction_feedback_ledger import (
    PredictionFeedbackLedgerResult,
    build_prediction_feedback_ledger,
)
from .final_result_attachment_artifacts import (
    render_attachment_report_markdown,
    render_attachment_summary_json,
    render_attachments_jsonl,
)
from .moneyline_paper_run_bundle import (
    FrozenMoneylinePaperRunInputs,
    jsonl_fingerprint,
    load_frozen_bundle_inputs,
    read_json_object,
    read_jsonl_objects,
)
from .prediction_evaluation_artifacts import (
    render_evaluation_report_markdown,
    render_evaluation_summary_json,
    render_evaluations_jsonl,
)
from .prediction_feedback_artifacts import render_feedback_jsonl


P34A_SCHEMA_VERSION = "p34a.daily_moneyline_settlement_feedback.v1"
P34A_OPERATION = "SETTLE_DAILY_MONEYLINE_PAPER_RUN"
P34A_SETTLED_ROW_SCHEMA_VERSION = "p34a.settled_prediction.v1"
P34A_PREDICTION_RESULT_ROW_SCHEMA_VERSION = "p34a.prediction_result.v1"
P34A_RESULT_AUTHORITY = "MLB_STATS_API"
P34A_MARKET_ID = "moneyline"
P34A_STOP_P33A_AUTHORITY_DRIFT = "STOP_MATCHANALYSIS_P34A_P33A_AUTHORITY_DRIFT"
P34A_STOP_RESULT_AUTHORITY_DRIFT = (
    "STOP_MATCHANALYSIS_P34A_RESULT_AUTHORITY_DRIFT"
)
P34A_STOP_RESULT_INPUT_INVALID = "STOP_MATCHANALYSIS_P34A_RESULT_INPUT_INVALID"

_OUTCOME_FIELDS = frozenset(
    {
        "actual_winner",
        "away_score",
        "brier_component",
        "correctness_target",
        "evaluation_status",
        "home_score",
        "is_correct",
        "result",
        "result_observation_id",
        "settlement_status",
        "winner",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _required_string(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.{field}")
    return value


def _required_decimal(
    row: Mapping[str, Any],
    field: str,
    *,
    context: str,
    lower: Decimal | None = None,
    upper: Decimal | None = None,
) -> Decimal:
    value = row.get(field)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.{field} is not decimal"
        ) from exc
    if not parsed.is_finite():
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.{field} is not finite"
        )
    if lower is not None and parsed < lower:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.{field} below lower bound"
        )
    if upper is not None and parsed > upper:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.{field} above upper bound"
        )
    return parsed


def _source_row_fingerprint(row: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(dict(row)))


def _result_identity_projection(
    observations: Sequence[FinalResultObservation],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "away_score": observation.away_score,
            "game_number": observation.game_number,
            "home_score": observation.home_score,
            "provider_game_id": observation.provider_game_id,
            "provider_namespace": observation.provider_namespace,
            "result_observed_at_utc": observation.result_observed_at_utc,
            "source_result_id": observation.source_result_id,
            "status": observation.status,
        }
        for observation in sorted(
            observations,
            key=lambda item: (
                item.provider_namespace,
                item.provider_game_id,
                item.game_number,
            ),
        )
    )


def compute_result_authority_fingerprint(
    observations: Sequence[FinalResultObservation],
) -> str:
    """Fingerprint the exact final observations that may be attached."""

    return _sha256(_canonical_jsonl_bytes(_result_identity_projection(observations)))


def build_official_result_authority(
    *,
    normalized_schedule_rows: Sequence[Mapping[str, Any]],
    target_game_ids: Sequence[str],
    observed_at_utc: str,
    source_url: str,
    raw_payload_sha256: str,
    network_called: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Project official normalized schedule rows into FINAL observations.

    Non-final games are recorded as unresolved authority rows and are not
    passed to P16A.  Duplicate target identities or a malformed FINAL row
    fail closed before settlement.
    """

    try:
        parse_canonical_utc(observed_at_utc)
    except ValueError as exc:
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: invalid observed-at timestamp"
        ) from exc
    if not isinstance(source_url, str) or not source_url:
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: source URL is missing"
        )
    if not _is_sha256(raw_payload_sha256):
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: raw source fingerprint is invalid"
        )

    target_ids = {str(value) for value in target_game_ids}
    target_rows = [
        dict(row)
        for row in normalized_schedule_rows
        if str(row.get("provider_game_id")) in target_ids
    ]
    if {str(row.get("provider_game_id")) for row in target_rows} != target_ids:
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: official target coverage is incomplete"
        )

    seen_keys: set[tuple[str, int, str]] = set()
    final_rows: list[dict[str, Any]] = []
    non_final_count = 0
    status_counts: dict[str, int] = {}
    for row in target_rows:
        provider_game_id = str(row.get("provider_game_id"))
        try:
            game_number = int(row["game_number"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: invalid official game number"
            ) from exc
        scheduled_start = str(row.get("scheduled_start_utc"))
        key = (provider_game_id, game_number, scheduled_start)
        if key in seen_keys:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: duplicate/conflicting official identity {key}"
            )
        seen_keys.add(key)
        status = row.get("status")
        status_counts[str(status)] = status_counts.get(str(status), 0) + 1
        if status != "Final" or row.get("final") is not True:
            non_final_count += 1
            continue
        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if (
            isinstance(home_score, bool)
            or not isinstance(home_score, int)
            or isinstance(away_score, bool)
            or not isinstance(away_score, int)
        ):
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: FINAL row has no integer scores {key}"
            )
        if home_score == away_score:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: tied FINAL result is unsupported {key}"
            )
        final_rows.append(
            {
                "source_result_id": (
                    f"mlb_stats_api:schedule:{raw_payload_sha256}:"
                    f"{provider_game_id}:{game_number}"
                ),
                "provider_namespace": P34A_RESULT_AUTHORITY,
                "provider_game_id": provider_game_id,
                "game_number": game_number,
                "status": "FINAL",
                "result_observed_at_utc": observed_at_utc,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    final_rows.sort(key=lambda row: (row["provider_game_id"], row["game_number"]))
    final_results_bytes = _canonical_jsonl_bytes(final_rows)
    final_observations = load_final_result_observations(final_results_bytes)
    result_authority_fingerprint = compute_result_authority_fingerprint(
        final_observations
    )
    authority = {
        "source": P34A_RESULT_AUTHORITY,
        "provider_namespace": P34A_RESULT_AUTHORITY,
        "source_url": source_url,
        "observed_at_utc": observed_at_utc,
        "raw_payload_sha256": raw_payload_sha256,
        "network_called": bool(network_called),
        "target_game_count": len(target_rows),
        "final_result_count": len(final_rows),
        "non_final_target_count": non_final_count,
        "missing_target_count": len(target_ids)
        - len({str(row["provider_game_id"]) for row in target_rows}),
        "finality_status_counts": dict(sorted(status_counts.items())),
        "all_target_results_final": non_final_count == 0,
        "all_settleable_results_final": True,
        "result_authority_fingerprint": result_authority_fingerprint,
    }
    return final_results_bytes, authority


@dataclass(frozen=True, slots=True)
class P33AAuthority:
    """Validated immutable P33A authority and its two accounting partitions."""

    bundle_root: Path
    run_manifest: dict[str, Any]
    summary: dict[str, Any]
    source_manifest: dict[str, Any]
    analysis_rows: tuple[dict[str, Any], ...]
    structural_rows: tuple[dict[str, Any], ...]
    prediction_rows: tuple[dict[str, Any], ...]
    schedule_rows: tuple[dict[str, Any], ...]
    analysis_jsonl_sha256: str
    summary_json_sha256: str
    analysis_set_fingerprint: str
    pregame_authority_fingerprint: str


@dataclass(frozen=True, slots=True)
class P34ASettlementResult:
    """Complete deterministic P34A settlement/feedback result."""

    p33a: P33AAuthority
    result_authority: dict[str, Any]
    final_results_bytes: bytes
    snapshot_result: AdmittedPredictionObservationSnapshotResult
    attachment_result: FinalResultAttachmentResult
    evaluation_result: PredictionEvaluationScorecardResult
    feedback_result: PredictionFeedbackLedgerResult
    prediction_result_rows: tuple[dict[str, Any], ...]
    settled_predictions: tuple[dict[str, Any], ...]
    structural_rows: tuple[dict[str, Any], ...]
    offline_replay_verified: bool
    network_called: bool

    @property
    def settled_count(self) -> int:
        return len(self.settled_predictions)

    @property
    def unresolved_count(self) -> int:
        return sum(
            row["settlement_status"] == "UNRESOLVED"
            for row in self.prediction_result_rows
        )

    @property
    def accuracy(self) -> float | None:
        if self.evaluation_result.evaluation_row_count == 0:
            return None
        return self.evaluation_result.accuracy

    @property
    def mean_selected_side_probability(self) -> float | None:
        if self.evaluation_result.evaluation_row_count == 0:
            return None
        return self.evaluation_result.mean_selected_side_probability

    @property
    def brier_score(self) -> float | None:
        if self.evaluation_result.evaluation_row_count == 0:
            return None
        return self.evaluation_result.brier_score


def _verify_p33a_manifest_fingerprint(run_manifest: Mapping[str, Any]) -> None:
    declared = run_manifest.get("run_manifest_fingerprint")
    if not _is_sha256(declared):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: run manifest fingerprint missing"
        )
    without_fingerprint = {
        key: value
        for key, value in run_manifest.items()
        if key != "run_manifest_fingerprint"
    }
    if _sha256(_canonical_json_bytes(without_fingerprint)) != declared:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: run manifest fingerprint mismatch"
        )


def _target_schedule_rows(
    inputs: FrozenMoneylinePaperRunInputs,
) -> tuple[dict[str, Any], ...]:
    target_ids = {str(value) for value in inputs.run_manifest.get("target_game_ids", [])}
    if not target_ids:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: target game IDs are missing"
        )
    selected = [
        dict(row)
        for row in inputs.schedule_rows
        if str(row.get("provider_game_id")) in target_ids
    ]
    if {str(row.get("provider_game_id")) for row in selected} != target_ids:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: target schedule coverage drift"
        )
    selected.sort(
        key=lambda row: (
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            str(row["provider_game_id"]),
        )
    )
    seen: set[tuple[str, int, str]] = set()
    for row in selected:
        key = (
            str(row["provider_game_id"]),
            int(row["game_number"]),
            str(row["scheduled_start_utc"]),
        )
        if key in seen:
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: duplicate target schedule identity {key}"
            )
        seen.add(key)
    return tuple(selected)


def load_p33a_authority(bundle_path: str | Path) -> P33AAuthority:
    """Load and verify a frozen P33A bundle without regenerating it."""

    root = Path(bundle_path).resolve()
    inputs = load_frozen_bundle_inputs(root)
    run_manifest = deepcopy(inputs.run_manifest)
    summary_path = root / "summary.json"
    analysis_path = root / "analysis.jsonl"
    summary = read_json_object(summary_path)
    analysis_rows = read_jsonl_objects(analysis_path)
    summary_raw = summary_path.read_bytes()
    analysis_raw = analysis_path.read_bytes()

    run_id = _required_string(run_manifest, "run_id", context="run_manifest")
    if root.name != run_id or run_manifest.get("run_fingerprint") != run_id:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A run identity does not match bundle path"
        )
    if summary.get("run_id") != run_id or summary.get("run_fingerprint") != run_id:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A summary run identity drift"
        )
    if summary.get("bundle_fingerprint") != run_manifest.get("bundle_fingerprint"):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A bundle fingerprint drift"
        )
    _verify_p33a_manifest_fingerprint(run_manifest)

    analysis_fingerprint = jsonl_fingerprint(analysis_rows)
    if summary.get("analysis_set_fingerprint") != analysis_fingerprint:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A analysis fingerprint drift"
        )
    analysis_sha256 = _sha256(analysis_raw)
    summary_sha256 = _sha256(summary_raw)
    analysis_metadata = run_manifest.get("analysis")
    if not isinstance(analysis_metadata, Mapping):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A analysis manifest is missing"
        )
    if analysis_metadata.get("analysis_jsonl_sha256") != analysis_sha256:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A analysis bytes drift"
        )
    if analysis_metadata.get("summary_json_sha256") != summary_sha256:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A summary bytes drift"
        )
    if analysis_metadata.get("p30a_run_id") != summary.get("analysis_run_id"):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P30A lineage drift"
        )

    structural_counts: dict[str, int] = {}
    for row in analysis_rows:
        status = _required_string(row, "status", context="analysis")
        structural_counts[status] = structural_counts.get(status, 0) + 1
        if _OUTCOME_FIELDS.intersection(row):
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A row contains outcome fields"
            )
        if row.get("run_id") != summary.get("analysis_run_id"):
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A analysis row run identity drift"
            )
    declared_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("analysis_terminal_state_counts", {})).items()
    }
    normalized_structural_counts = {
        key: structural_counts.get(key, 0)
        for key in sorted(set(structural_counts) | set(declared_counts))
    }
    normalized_declared_counts = {
        key: declared_counts.get(key, 0)
        for key in sorted(set(structural_counts) | set(declared_counts))
    }
    if normalized_structural_counts != normalized_declared_counts:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A structural counts drift"
        )
    if int(summary.get("official_raw_game_count", -1)) != len(analysis_rows):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A raw game count drift"
        )

    acquisition = run_manifest.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A acquisition manifest is missing"
        )
    for summary_key, acquisition_key in (
        ("source_records_received", "source_row_count"),
        ("observations_qualified", "qualified_observation_count"),
        ("observations_rejected", "rejected_row_count"),
    ):
        if int(summary.get(summary_key, -1)) != int(acquisition.get(acquisition_key, -2)):
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A {summary_key} drift"
            )
    claims = summary.get("claims")
    if not isinstance(claims, Mapping):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A claims are missing"
        )
    for claim in (
        "settlement_included",
        "staking_implemented",
        "profitability_claim",
        "real_betting_recommendation",
    ):
        if claims.get(claim) is not False:
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A {claim} is not false"
            )

    target_schedule = _target_schedule_rows(inputs)
    analysis_by_game = {str(row.get("game_id")): row for row in analysis_rows}
    if len(analysis_by_game) != len(analysis_rows):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: duplicate P33A game identity"
        )
    target_ids = {str(value) for value in run_manifest["target_game_ids"]}
    if set(analysis_by_game) != target_ids:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: analysis/target game coverage drift"
        )

    prediction_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    schedule_by_game = {
        str(row["provider_game_id"]): row for row in target_schedule
    }
    for row in analysis_rows:
        status = str(row["status"])
        has_prediction = row.get("prediction_id") is not None
        if status == "EDGE_AVAILABLE" and not has_prediction:
            raise ValueError(
                f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: EDGE_AVAILABLE row has no prediction"
            )
        if status == "EDGE_AVAILABLE":
            prediction_id = row.get("prediction_id")
            if not _is_sha256(prediction_id):
                raise ValueError(
                    f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: invalid P33A prediction ID"
                )
            scheduled = _required_string(row, "scheduled_start", context="analysis")
            try:
                parse_canonical_utc(scheduled)
            except ValueError as exc:
                raise ValueError(
                    f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: invalid P33A scheduled start"
                ) from exc
            if str(schedule_by_game[str(row["game_id"])] ["scheduled_start_utc"]) != scheduled:
                raise ValueError(
                    f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: P33A schedule identity drift"
                )
            prediction_rows.append(dict(row))
        else:
            structural_rows.append(dict(row))

    prediction_ids = [str(row["prediction_id"]) for row in prediction_rows]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: duplicate P33A prediction identity"
        )
    pregame_fingerprint = _sha256(_canonical_jsonl_bytes(analysis_rows))
    return P33AAuthority(
        bundle_root=root,
        run_manifest=run_manifest,
        summary=summary,
        source_manifest=deepcopy(inputs.source_manifest),
        analysis_rows=tuple(dict(row) for row in analysis_rows),
        structural_rows=tuple(structural_rows),
        prediction_rows=tuple(sorted(prediction_rows, key=lambda row: str(row["prediction_id"]))),
        schedule_rows=target_schedule,
        analysis_jsonl_sha256=analysis_sha256,
        summary_json_sha256=summary_sha256,
        analysis_set_fingerprint=analysis_fingerprint,
        pregame_authority_fingerprint=pregame_fingerprint,
    )


def _schedule_row_for_prediction(
    p33a: P33AAuthority,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    game_id = _required_string(row, "game_id", context="prediction")
    candidates = [
        schedule
        for schedule in p33a.schedule_rows
        if str(schedule.get("provider_game_id")) == game_id
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: prediction schedule identity is ambiguous"
        )
    schedule = candidates[0]
    if str(schedule.get("scheduled_start_utc")) != str(row.get("scheduled_start")):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: prediction scheduled start drift"
        )
    home = schedule.get("home_team")
    away = schedule.get("away_team")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: schedule team identity is missing"
        )
    if row.get("home_team") != home.get("name") or row.get("away_team") != away.get("name"):
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: prediction team identity drift"
        )
    return schedule


def _prediction_observation(
    p33a: P33AAuthority,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"prediction:{row.get('prediction_id')}"
    prediction_id = _required_string(row, "prediction_id", context=context)
    if not _is_sha256(prediction_id):
        raise ValueError(f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context}.prediction_id")
    model_id = _required_string(row, "model_id", context=context)
    model_fingerprint = _required_string(row, "model_fingerprint", context=context)
    scheduled_start = _required_string(row, "scheduled_start", context=context)
    price_observed_at = _required_string(row, "price_observed_at", context=context)
    try:
        scheduled_time = parse_canonical_utc(scheduled_start)
        price_time = parse_canonical_utc(price_observed_at)
    except ValueError as exc:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context} timestamp"
        ) from exc
    if price_time > scheduled_time:
        raise ValueError(
            f"{P34A_STOP_P33A_AUTHORITY_DRIFT}: {context} price is postgame"
        )
    home_probability = _required_decimal(
        row,
        "model_home_probability",
        context=context,
        lower=Decimal("0"),
        upper=Decimal("1"),
    )
    away_probability = Decimal("1") - home_probability
    selection = "HOME" if home_probability >= Decimal("0.5") else "AWAY"
    model_probability = (
        home_probability if selection == "HOME" else away_probability
    )
    home_price = _required_string(row, "home_decimal_odds", context=context)
    away_price = _required_string(row, "away_decimal_odds", context=context)
    home_edge = _required_string(row, "home_edge", context=context)
    away_edge = _required_string(row, "away_edge", context=context)
    market_price_id = _required_string(row, "market_price_id", context=context)
    schedule = _schedule_row_for_prediction(p33a, row)
    source_row_fp = _source_row_fingerprint(row)
    schedule_identity = _sha256(
        _canonical_json_bytes(
            {
                "p33a_run_id": p33a.run_manifest["run_id"],
                "provider_game_id": str(row["game_id"]),
                "game_number": int(schedule["game_number"]),
                "scheduled_start_utc": scheduled_start,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
            }
        )
    )
    return {
        "prediction_observation_id": prediction_id,
        "source_prediction_id": prediction_id,
        "p33a_run_id": p33a.run_manifest["run_id"],
        "p33a_analysis_set_fingerprint": p33a.analysis_set_fingerprint,
        "p33a_source_row_fingerprint": source_row_fp,
        "prediction_id": prediction_id,
        "game_id": str(row["game_id"]),
        "model_id": model_id,
        "model_fingerprint": model_fingerprint,
        "market_id": P34A_MARKET_ID,
        "selection": selection,
        "model_probability": str(model_probability),
        "model_home_probability": str(home_probability),
        "model_away_probability": str(away_probability),
        "predicted_side": selection,
        "home_decimal_odds": home_price,
        "away_decimal_odds": away_price,
        "home_edge": home_edge,
        "away_edge": away_edge,
        "market_price_id": market_price_id,
        "line_value": "0",
        "push_policy": "NO_PUSH",
        "feature_eligibility": "ELIGIBLE",
        "provider_namespace": P34A_RESULT_AUTHORITY,
        "provider_game_id": str(row["game_id"]),
        "game_number": int(schedule["game_number"]),
        "source_schedule_observation_id": schedule_identity,
        "prediction_generated_at_utc": price_observed_at,
        "response_received_at_utc": price_observed_at,
        "ingested_at_utc": price_observed_at,
        "scheduled_start_utc": scheduled_start,
        "source_p33a_analysis_row": deepcopy(dict(row)),
    }


def _make_admission_inputs(
    observations: Sequence[Mapping[str, Any]],
    p33a: P33AAuthority,
) -> tuple[bytes, bytes]:
    rows = [
        {
            "request_index": index,
            "admission_status": "ADMITTED",
            "reason": None,
            "observation": dict(observation),
        }
        for index, observation in enumerate(observations, start=1)
    ]
    result_set_fingerprint = _sha256(
        "".join(
            f"ADMITTED::{row['observation']['prediction_observation_id']}\n"
            for row in rows
        ).encode("utf-8")
    )
    summary = {
        "schema_version": "p34a.p33a_prediction_adapter.v1",
        "schedule_as_of_utc": p33a.summary.get("target_date"),
        "request_count": len(rows),
        "admitted_count": len(rows),
        "rejected_count": 0,
        "rejection_reasons_breakdown": {},
        "result_set_fingerprint": result_set_fingerprint,
        "input_hashes": {
            "p33a_bundle_fingerprint": p33a.summary.get("bundle_fingerprint"),
            "p33a_analysis_jsonl_sha256": p33a.analysis_jsonl_sha256,
        },
        "claims": {
            "provider_called": False,
            "db_written": False,
            "legacy_rows_admitted": False,
            "deployed": False,
            "betting_claim": False,
        },
    }
    return _canonical_jsonl_bytes(rows), _canonical_json_bytes(summary)


def _adjust_result_claims(result: Any, *, network_called: bool, official: bool) -> Any:
    claims = dict(result.claims)
    claims.update(
        {
            "network_called": network_called,
            "provider_called": network_called,
            "synthetic_results": not official,
        }
    )
    return replace(result, claims=claims)


def _validate_result_authority(
    *,
    p33a: P33AAuthority,
    final_results_bytes: bytes,
    result_authority: Mapping[str, Any],
) -> tuple[list[FinalResultObservation], dict[str, Any]]:
    if not isinstance(final_results_bytes, bytes):
        raise TypeError(f"{P34A_STOP_RESULT_INPUT_INVALID}: final results must be bytes")
    try:
        observations = load_final_result_observations(final_results_bytes)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{P34A_STOP_RESULT_INPUT_INVALID}: {exc}") from exc

    target_keys = {
        (
            P34A_RESULT_AUTHORITY,
            str(row["provider_game_id"]),
            int(row["game_number"]),
        )
        for row in p33a.schedule_rows
    }
    seen: set[tuple[str, str, int]] = set()
    for observation in observations:
        key = (
            observation.provider_namespace,
            observation.provider_game_id,
            observation.game_number,
        )
        if key not in target_keys:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: result is outside frozen target scope"
            )
        if key in seen:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: duplicate/conflicting result identity"
            )
        seen.add(key)

    authority = deepcopy(dict(result_authority))
    source = authority.get("source", P34A_RESULT_AUTHORITY)
    if source != P34A_RESULT_AUTHORITY:
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: result source is not official MLB"
        )
    authority["source"] = P34A_RESULT_AUTHORITY
    authority["provider_namespace"] = P34A_RESULT_AUTHORITY
    authority["final_result_count"] = len(observations)
    authority["result_authority_fingerprint"] = compute_result_authority_fingerprint(
        observations
    )
    declared_fingerprint = result_authority.get("result_authority_fingerprint")
    if declared_fingerprint is not None and declared_fingerprint != authority[
        "result_authority_fingerprint"
    ]:
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: result fingerprint mismatch"
        )
    if "observed_at_utc" in authority:
        try:
            parse_canonical_utc(str(authority["observed_at_utc"]))
        except ValueError as exc:
            raise ValueError(
                f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: invalid observed-at timestamp"
            ) from exc
    authority["target_game_count"] = len(p33a.schedule_rows)
    authority["settleable_prediction_count"] = len(p33a.prediction_rows)
    settleable_keys = {
        (
            str(row["game_id"]),
            int(
                next(
                    schedule["game_number"]
                    for schedule in p33a.schedule_rows
                    if str(schedule["provider_game_id"]) == str(row["game_id"])
                )
            ),
        )
        for row in p33a.prediction_rows
    }
    observed_result_keys = {
        (observation.provider_game_id, observation.game_number)
        for observation in observations
    }
    missing_settleable_count = len(settleable_keys - observed_result_keys)
    authority["missing_settleable_result_count"] = missing_settleable_count
    authority["missing_settleable_result_check"] = (
        "PASS" if missing_settleable_count == 0 else "FAIL_CLOSED"
    )
    authority["duplicate_result_identity_check"] = "PASS"
    authority["conflicting_result_check"] = "PASS"
    authority["non_final_result_check"] = "PASS"
    authority["all_settleable_results_final"] = True
    return observations, authority


def _prediction_result_rows(
    *,
    p33a: P33AAuthority,
    snapshot_result: AdmittedPredictionObservationSnapshotResult,
    attachment_result: FinalResultAttachmentResult,
    evaluation_result: PredictionEvaluationScorecardResult,
) -> tuple[dict[str, Any], ...]:
    snapshot_by_id = {
        row.prediction_observation_id: row.observation
        for row in snapshot_result.snapshot_rows
    }
    attachment_by_id = {
        row.prediction_observation_id: row for row in attachment_result.attachment_rows
    }
    evaluation_by_id = {
        row.prediction_observation_id: row for row in evaluation_result.evaluation_rows
    }
    source_by_id = {
        str(row["prediction_id"]): row for row in p33a.prediction_rows
    }
    rows: list[dict[str, Any]] = []
    for prediction_id in sorted(source_by_id):
        source = source_by_id[prediction_id]
        observation = snapshot_by_id[prediction_id]
        attachment = attachment_by_id[prediction_id]
        evaluation = evaluation_by_id.get(prediction_id)
        selection = str(observation["selection"])
        selected_price = source[
            "home_decimal_odds" if selection == "HOME" else "away_decimal_odds"
        ]
        selected_edge = source["home_edge" if selection == "HOME" else "away_edge"]
        attached = attachment.attachment_status == "ATTACHED"
        row = {
            "schema_version": P34A_PREDICTION_RESULT_ROW_SCHEMA_VERSION,
            "p33a_run_id": p33a.run_manifest["run_id"],
            "p33a_analysis_set_fingerprint": p33a.analysis_set_fingerprint,
            "p33a_source_row_fingerprint": _source_row_fingerprint(source),
            "prediction_id": prediction_id,
            "prediction_observation_id": prediction_id,
            "game_id": source["game_id"],
            "provider_namespace": observation["provider_namespace"],
            "provider_game_id": observation["provider_game_id"],
            "game_number": observation["game_number"],
            "home_team": source["home_team"],
            "away_team": source["away_team"],
            "scheduled_start": source["scheduled_start"],
            "predicted_side": observation["predicted_side"],
            "model_id": source["model_id"],
            "model_fingerprint": source["model_fingerprint"],
            "model_home_probability": source["model_home_probability"],
            "model_probability": observation["model_probability"],
            "market_price_id": source["market_price_id"],
            "price_observed_at": source["price_observed_at"],
            "market_price": selected_price,
            "market_edge": selected_edge,
            "home_decimal_odds": source["home_decimal_odds"],
            "away_decimal_odds": source["away_decimal_odds"],
            "home_edge": source["home_edge"],
            "away_edge": source["away_edge"],
            "settlement_status": "SETTLED" if attached else "UNRESOLVED",
            "evaluation_status": "EVALUATED" if evaluation is not None else "NOT_EVALUATED",
            "result_observation_id": attachment.result_observation_id,
            "result_observed_at_utc": attachment.result_observed_at_utc,
            "home_score": attachment.home_score,
            "away_score": attachment.away_score,
            "actual_winner": attachment.actual_winner,
            "is_correct": evaluation.is_correct if evaluation else None,
            "correctness_target": evaluation.correctness_target if evaluation else None,
            "brier_component": str(evaluation.brier_component) if evaluation else None,
            "unresolved_reason": attachment.rejection_reason,
            "source_p33a_analysis_row": deepcopy(dict(source)),
            "source_observation": deepcopy(dict(observation)),
        }
        rows.append(row)
    return tuple(rows)


def settle_daily_moneyline_paper_run(
    *,
    p33a_bundle_path: str | Path,
    final_results_bytes: bytes,
    result_authority: Mapping[str, Any],
    offline_replay_verified: bool = False,
) -> P34ASettlementResult:
    """Run P34A from frozen P33A and frozen official result authority."""

    p33a = load_p33a_authority(p33a_bundle_path)
    _final_observations, authority = _validate_result_authority(
        p33a=p33a,
        final_results_bytes=final_results_bytes,
        result_authority=result_authority,
    )
    prediction_observations = tuple(
        _prediction_observation(p33a, row) for row in p33a.prediction_rows
    )
    admission_bytes, admission_summary_bytes = _make_admission_inputs(
        prediction_observations,
        p33a,
    )
    snapshot_result = build_admitted_prediction_observation_snapshot(
        results_bytes=admission_bytes,
        summary_bytes=admission_summary_bytes,
    )
    snapshot_bytes = render_admitted_observations_jsonl(snapshot_result).encode("utf-8")
    snapshot_report_bytes = render_snapshot_report_markdown(snapshot_result).encode(
        "utf-8"
    )
    snapshot_summary_bytes = render_snapshot_summary_json(
        snapshot_result,
        _sha256(snapshot_bytes),
        _sha256(snapshot_report_bytes),
    ).encode("utf-8")

    attachment_result = attach_final_results_to_admitted_predictions(
        snapshot_bytes=snapshot_bytes,
        summary_bytes=snapshot_summary_bytes,
        final_results_bytes=final_results_bytes,
    )
    network_called = bool(authority.get("network_called", False))
    attachment_result = _adjust_result_claims(
        attachment_result,
        network_called=network_called,
        official=True,
    )
    attachments_bytes = render_attachments_jsonl(attachment_result).encode("utf-8")
    attachment_report_bytes = render_attachment_report_markdown(
        attachment_result
    ).encode("utf-8")
    attachment_summary_bytes = render_attachment_summary_json(
        attachment_result,
        _sha256(attachments_bytes),
        _sha256(attachment_report_bytes),
    ).encode("utf-8")

    evaluation_result = build_prediction_evaluation_scorecard(
        attachments_bytes=attachments_bytes,
        attachment_summary_bytes=attachment_summary_bytes,
        snapshot_bytes=snapshot_bytes,
        snapshot_summary_bytes=snapshot_summary_bytes,
    )
    evaluation_result = _adjust_result_claims(
        evaluation_result,
        network_called=network_called,
        official=True,
    )
    evaluations_bytes = render_evaluations_jsonl(evaluation_result).encode("utf-8")
    evaluation_report_bytes = render_evaluation_report_markdown(
        evaluation_result
    ).encode("utf-8")
    evaluation_summary_bytes = render_evaluation_summary_json(
        evaluation_result,
        _sha256(evaluations_bytes),
        _sha256(evaluation_report_bytes),
    ).encode("utf-8")

    feedback_result = build_prediction_feedback_ledger(
        snapshot_bytes=snapshot_bytes,
        snapshot_summary_bytes=snapshot_summary_bytes,
        attachments_bytes=attachments_bytes,
        attachment_summary_bytes=attachment_summary_bytes,
        evaluations_bytes=evaluations_bytes,
        evaluation_summary_bytes=evaluation_summary_bytes,
    )
    feedback_result = _adjust_result_claims(
        feedback_result,
        network_called=network_called,
        official=True,
    )
    feedback_bytes = render_feedback_jsonl(feedback_result).encode("utf-8")
    prediction_result_rows = _prediction_result_rows(
        p33a=p33a,
        snapshot_result=snapshot_result,
        attachment_result=attachment_result,
        evaluation_result=evaluation_result,
    )
    settled_predictions = tuple(
        {
            **row,
            "schema_version": P34A_SETTLED_ROW_SCHEMA_VERSION,
        }
        for row in prediction_result_rows
        if row["settlement_status"] == "SETTLED"
    )
    authority["final_results_jsonl_sha256"] = _sha256(final_results_bytes)
    authority["attachment_result_count"] = attachment_result.attached_count
    authority["unresolved_result_count"] = len(prediction_result_rows) - len(
        settled_predictions
    )
    authority["all_settleable_results_final"] = (
        len(prediction_result_rows) == len(settled_predictions)
    )
    authority["settlement_status"] = (
        "NO_SETTLEABLE_PREDICTIONS"
        if not prediction_result_rows
        else "SETTLED"
        if len(prediction_result_rows) == len(settled_predictions)
        else "PARTIAL_UNRESOLVED"
    )
    # Keep the local variable alive for explicit evidence that the feedback
    # renderer was exercised even when the daily sample is empty.
    del feedback_bytes
    return P34ASettlementResult(
        p33a=p33a,
        result_authority=authority,
        final_results_bytes=final_results_bytes,
        snapshot_result=snapshot_result,
        attachment_result=attachment_result,
        evaluation_result=evaluation_result,
        feedback_result=feedback_result,
        prediction_result_rows=prediction_result_rows,
        settled_predictions=settled_predictions,
        structural_rows=p33a.structural_rows,
        offline_replay_verified=offline_replay_verified,
        network_called=network_called,
    )


def replay_daily_moneyline_paper_settlement(
    *,
    p33a_bundle_path: str | Path,
    settlement_bundle_path: str | Path,
) -> P34ASettlementResult:
    """Replay a committed/frozen P34A result authority without network access."""

    root = Path(settlement_bundle_path).resolve()
    final_results_path = root / "final_results.jsonl"
    authority_path = root / "result_authority.json"
    if not final_results_path.is_file() or not authority_path.is_file():
        raise ValueError(
            f"{P34A_STOP_RESULT_AUTHORITY_DRIFT}: frozen result authority is incomplete"
        )
    authority = read_json_object(authority_path)
    authority["network_called"] = False
    return settle_daily_moneyline_paper_run(
        p33a_bundle_path=p33a_bundle_path,
        final_results_bytes=final_results_path.read_bytes(),
        result_authority=authority,
        offline_replay_verified=True,
    )


__all__ = (
    "P33AAuthority",
    "P34A_OPERATION",
    "P34A_PREDICTION_RESULT_ROW_SCHEMA_VERSION",
    "P34A_RESULT_AUTHORITY",
    "P34A_SCHEMA_VERSION",
    "P34A_SETTLED_ROW_SCHEMA_VERSION",
    "P34ASettlementResult",
    "P34A_STOP_P33A_AUTHORITY_DRIFT",
    "P34A_STOP_RESULT_AUTHORITY_DRIFT",
    "P34A_STOP_RESULT_INPUT_INVALID",
    "build_official_result_authority",
    "compute_result_authority_fingerprint",
    "load_p33a_authority",
    "replay_daily_moneyline_paper_settlement",
    "settle_daily_moneyline_paper_run",
)
