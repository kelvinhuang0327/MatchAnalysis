"""P45A prospective paper run ledger and multi-run lifecycle.

Orchestrates immutable paper runs across pregame freeze, partial/full postgame
settlement, and append-only forward-paper ledger tracking. Strict isolation
ensures historical rehearsals never increment prospective forward sample counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    P40A_POLICY_ID,
    SETTLEMENT_LOST,
    SETTLEMENT_PASS,
    SETTLEMENT_WON,
    PaperMoneylineDecision,
    PaperMoneylineSettlement,
    settle_paper_moneyline_decision,
)
from .p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
    P40AMarketRow,
    P40APredictionRow,
)
from .p42a_offline_end_to_end_paper_workflow import (
    P42A_EVALUATION_BET_LOST,
    P42A_EVALUATION_BET_WON,
    P42A_EVALUATION_PASS_NO_WAGER,
)
from .p43a_postgame_settle import (
    load_p43a_frozen_decision_bundle,
    reconstruct_frozen_decision,
    verify_p43a_pregame_record,
)
from .p43a_pregame_freeze import (
    P43A_CLAIMS,
    P43A_HUMAN_LABEL,
    build_p43a_exclusion_row,
    build_p43a_pregame_record,
    freeze_p43a_pregame_decisions,
)
from .p44a_normalized_workflow_input import (
    NormalizedPregameInput,
    NormalizedResultRecord,
    load_normalized_pregame_input,
    load_normalized_result_input,
    reject_pregame_outcome_fields,
)


P45A_TASK_ID = "P45A"
P45A_RUN_MANIFEST_SCHEMA = "p45a.paper_run_manifest.v1"
P45A_SETTLEMENT_SUMMARY_SCHEMA = "p45a.paper_run_settlement_summary.v1"
P45A_LEDGER_RECORD_SCHEMA = "p45a.forward_paper_ledger_record.v1"
P45A_FORWARD_SUMMARY_SCHEMA = "p45a.cumulative_forward_paper_summary.v1"
P45A_REPORT_RELATIVE_PATH = Path("report/p45a_prospective_paper_run_ledger")

CLASSIFICATION_HISTORICAL_REHEARSAL = "HISTORICAL_REHEARSAL"
CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER = "PROSPECTIVE_FORWARD_PAPER"
VALID_CLASSIFICATIONS = frozenset(
    {
        CLASSIFICATION_HISTORICAL_REHEARSAL,
        CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    }
)

STATE_FROZEN = "FROZEN"
STATE_PARTIALLY_SETTLED = "PARTIALLY_SETTLED"
STATE_SETTLED = "SETTLED"
VALID_LIFECYCLE_STATES = frozenset(
    {
        STATE_FROZEN,
        STATE_PARTIALLY_SETTLED,
        STATE_SETTLED,
    }
)


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


def _sha256_projection(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _decimal_text(value: Any) -> str | None:
    if value is None:
        return None
    return format(value, "f") if hasattr(value, "is_finite") else str(value)


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_rejecting_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, object_pairs_hook=_duplicate_rejecting_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    return rows


def _evaluation_projection(settlement: PaperMoneylineSettlement) -> dict[str, Any]:
    if settlement.decision.decision == DECISION_PASS:
        if settlement.settlement_status != SETTLEMENT_PASS:
            raise ValueError("PASS settlement status drifted")
        return {
            "evaluation_status": P42A_EVALUATION_PASS_NO_WAGER,
            "is_correct": None,
            "correctness_label": "NO_WAGER",
            "uses_p40_settlement_status": settlement.settlement_status,
        }
    if settlement.settlement_status == SETTLEMENT_WON:
        return {
            "evaluation_status": P42A_EVALUATION_BET_WON,
            "is_correct": True,
            "correctness_label": "WIN",
            "uses_p40_settlement_status": settlement.settlement_status,
        }
    if settlement.settlement_status == SETTLEMENT_LOST:
        return {
            "evaluation_status": P42A_EVALUATION_BET_LOST,
            "is_correct": False,
            "correctness_label": "LOSS",
            "uses_p40_settlement_status": settlement.settlement_status,
        }
    raise ValueError(f"unexpected settlement status {settlement.settlement_status}")


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == Decimal("0"):
        raise ValueError("ratio denominator must not be zero")
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def _feedback_identity(settlement: PaperMoneylineSettlement) -> str:
    evaluation = _evaluation_projection(settlement)
    return _sha256_projection(
        {
            "decision_id": settlement.decision.decision_id,
            "settlement_row_fingerprint": settlement.settlement_row_fingerprint,
            "outcome_authority_row_id": settlement.outcome_authority_row_id,
            "evaluation_status": evaluation["evaluation_status"],
            "is_correct": evaluation["is_correct"],
            "correctness_label": evaluation["correctness_label"],
        }
    )


def validate_run_classification(
    classification: str,
    *,
    pregame_input: NormalizedPregameInput,
    created_at_utc: str,
) -> None:
    """Enforce classification rules and strict temporal verification."""

    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"unknown run classification: {classification!r}, "
            f"must be one of {sorted(VALID_CLASSIFICATIONS)}"
        )

    # Standard pregame observation guards for all classifications
    for market in pregame_input.market_rows:
        if market.market_observed_at_utc >= market.scheduled_start_utc:
            raise ValueError("market observation is not strictly pregame")
        if market.local_fetched_at_utc >= market.scheduled_start_utc:
            raise ValueError("market local fetch is not strictly pregame")

    if classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER:
        # Check source identity - cannot be historical rehearsal
        if (
            "historical" in pregame_input.source_identity.lower()
            or "rehearsal" in pregame_input.source_identity.lower()
        ):
            raise RuntimeError(
                "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: "
                f"historical source identity {pregame_input.source_identity!r} "
                "cannot be classified as prospective forward paper"
            )

        # Freeze creation timestamp must be strictly before earliest scheduled start
        parse_canonical_utc(created_at_utc)
        earliest_start = min(row.scheduled_start_utc for row in pregame_input.prediction_rows)
        parse_canonical_utc(earliest_start)
        if created_at_utc >= earliest_start:
            raise RuntimeError(
                "P45A_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: "
                f"prospective run creation time {created_at_utc} is not strictly before "
                f"earliest scheduled start {earliest_start}"
            )


def compute_deterministic_run_id(
    *,
    run_classification: str,
    normalized_input_fingerprint: str,
    decision_bundle_fingerprint: str,
    target_universe: Sequence[Mapping[str, Any]],
) -> str:
    """Derive deterministic logical run identity from immutable pregame authority."""

    payload = {
        "schema_version": P45A_RUN_MANIFEST_SCHEMA,
        "run_classification": run_classification,
        "policy_id": P40A_POLICY_ID,
        "model_role": P40A_CHAMPION_ROLE,
        "normalized_input_fingerprint": normalized_input_fingerprint,
        "decision_bundle_fingerprint": decision_bundle_fingerprint,
        "target_universe": sorted(
            [
                {
                    "game_pk": row["game_pk"],
                    "game_number": row["game_number"],
                    "provider_game_id": str(row["provider_game_id"]),
                    "p37_prediction_row_id": str(row["p37_prediction_row_id"]),
                }
                for row in target_universe
            ],
            key=lambda item: (item["game_pk"], item["game_number"], item["provider_game_id"]),
        ),
    }
    digest = sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"p45a_run_{digest[:32]}"


@dataclass(frozen=True, slots=True)
class P45ACreateRunResult:
    status: str
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    pregame_decisions: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]


def canonical_pregame_fingerprint(bundle: NormalizedPregameInput) -> str:
    """Compute canonical order-independent fingerprint of normalized pregame input."""
    from .p44a_normalized_workflow_input import (
        market_row_to_payload,
        prediction_row_to_payload,
    )

    sorted_predictions = sorted(
        [prediction_row_to_payload(row) for row in bundle.prediction_rows],
        key=lambda r: (r["scheduled_start_utc"], r["game_number"], r["provider_game_id"]),
    )
    sorted_markets = sorted(
        [market_row_to_payload(row) for row in bundle.market_rows],
        key=lambda r: (r["scheduled_start_utc"], r["game_number"], r["provider_game_id"]),
    )
    payload = {
        "source_identity": bundle.source_identity,
        "predictions": sorted_predictions,
        "markets": sorted_markets,
        "exclusions": sorted(
            [dict(r) for r in bundle.exclusion_rows],
            key=lambda r: (str(r.get("scheduled_start_utc", "")), str(r.get("provider_game_id", ""))),
        ),
        "source_manifest": bundle.source_manifest,
        "authority_hashes": bundle.authority_hashes,
    }
    return _sha256_projection(payload)


def create_p45a_paper_run(
    repository_root: str | Path,
    *,
    pregame_input: NormalizedPregameInput | str | Path,
    run_classification: str = CLASSIFICATION_HISTORICAL_REHEARSAL,
    run_root: str | Path | None = None,
    created_at_utc: str = "2026-08-17T12:00:00Z",
) -> P45ACreateRunResult:
    """Create and freeze one immutable paper run before game outcomes occur."""

    root = Path(repository_root).resolve()
    if isinstance(pregame_input, NormalizedPregameInput):
        bundle = pregame_input
    else:
        bundle = load_normalized_pregame_input(pregame_input)

    # Validate classification and temporal rules
    validate_run_classification(
        run_classification,
        pregame_input=bundle,
        created_at_utc=created_at_utc,
    )

    # Freeze Champion BET/PASS decisions using unchanged P40 zero-EV rule
    authority = bundle.to_authority(root)
    champion_decisions = freeze_p43a_pregame_decisions(authority)

    pregame_records = tuple(
        build_p43a_pregame_record(decision) for decision in champion_decisions
    )
    exclusion_records = tuple(
        build_p43a_exclusion_row(row) for row in bundle.exclusion_rows
    )

    normalized_input_fingerprint = canonical_pregame_fingerprint(bundle)
    decision_fingerprints = [row["decision_fingerprint"] for row in pregame_records]
    decision_bundle_fingerprint = _sha256_projection(decision_fingerprints)

    target_universe = [
        {
            "game_pk": row.game_pk,
            "game_number": row.game_number,
            "provider_game_id": row.provider_game_id,
            "p37_prediction_row_id": row.p37_prediction_row_id,
        }
        for row in bundle.prediction_rows
    ]

    run_id = compute_deterministic_run_id(
        run_classification=run_classification,
        normalized_input_fingerprint=normalized_input_fingerprint,
        decision_bundle_fingerprint=decision_bundle_fingerprint,
        target_universe=target_universe,
    )

    if run_root is not None:
        destination_root = Path(run_root).resolve()
    else:
        destination_root = (root / P45A_REPORT_RELATIVE_PATH / "runs").resolve()

    run_dir = destination_root / run_id
    manifest_path = run_dir / "run_manifest.json"
    decisions_path = run_dir / "pregame_decisions.jsonl"
    exclusions_path = run_dir / "exclusions.jsonl"

    bets = tuple(row for row in pregame_records if row["bet_or_pass"] == DECISION_BET)
    passes = tuple(row for row in pregame_records if row["bet_or_pass"] == DECISION_PASS)

    manifest_payload = {
        "schema_version": P45A_RUN_MANIFEST_SCHEMA,
        "task_id": P45A_TASK_ID,
        "run_id": run_id,
        "run_classification": run_classification,
        "lifecycle_state": STATE_FROZEN,
        "created_at_utc": created_at_utc,
        "policy_id": P40A_POLICY_ID,
        "model_role": P40A_CHAMPION_ROLE,
        "target_universe_count": len(bundle.prediction_rows) + len(bundle.exclusion_rows),
        "eligible_decision_count": len(pregame_records),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "settled_bet_count": 0,
        "settled_total_count": 0,
        "pending_count": len(pregame_records),
        "exclusion_count": len(exclusion_records),
        "normalized_input_fingerprint": normalized_input_fingerprint,
        "decision_bundle_fingerprint": decision_bundle_fingerprint,
        "decision_fingerprints": decision_fingerprints,
        "claims": dict(P43A_CLAIMS),
    }

    # Idempotency / conflict check
    if manifest_path.is_file():
        existing_manifest = read_json_object(manifest_path)
        # Check if identical authority
        if (
            existing_manifest.get("run_id") == run_id
            and existing_manifest.get("decision_bundle_fingerprint") == decision_bundle_fingerprint
            and existing_manifest.get("normalized_input_fingerprint") == normalized_input_fingerprint
            and existing_manifest.get("run_classification") == run_classification
        ):
            # Verify existing pregame decisions
            if decisions_path.is_file() and decisions_path.read_bytes() == _jsonl_bytes(pregame_records):
                return P45ACreateRunResult(
                    status="RECOGNIZED_IDENTICAL",
                    run_id=run_id,
                    run_dir=run_dir,
                    manifest=existing_manifest,
                    pregame_decisions=pregame_records,
                    exclusions=exclusion_records,
                )
        raise RuntimeError(
            "P45A_RUN_AUTHORITY_CONFLICT: existing run manifest conflicts with incoming authority"
        )

    # First write
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json_bytes(manifest_payload))
    decisions_path.write_bytes(_jsonl_bytes(pregame_records))
    exclusions_path.write_bytes(_jsonl_bytes(exclusion_records))

    return P45ACreateRunResult(
        status="CREATED",
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest_payload,
        pregame_decisions=pregame_records,
        exclusions=exclusion_records,
    )


def build_p45a_ledger_record(
    *,
    run_id: str,
    run_classification: str,
    decision: PaperMoneylineDecision,
    pregame_record: Mapping[str, Any],
    result_record: NormalizedResultRecord,
    settlement: PaperMoneylineSettlement,
    manifest: Mapping[str, Any],
    settled_at_utc: str,
) -> dict[str, Any]:
    """Project one settled decision into an authoritative append-only ledger record."""

    evaluation = _evaluation_projection(settlement)
    selected_side = decision.candidate_side if decision.decision == DECISION_BET else "NONE"

    payload_without_id = {
        "schema_version": P45A_LEDGER_RECORD_SCHEMA,
        "run_id": run_id,
        "run_classification": run_classification,
        "decision_identity": {
            "workflow_decision_id": pregame_record.get("workflow_decision_id"),
            "decision_fingerprint": decision.decision_id,
            "p37_prediction_row_id": decision.p37_prediction_row_id,
        },
        "game_identity": {
            "provider_namespace": decision.provider_namespace,
            "provider_game_id": decision.provider_game_id,
            "game_pk": decision.game_pk,
            "game_number": decision.game_number,
            "scheduled_start_utc": decision.scheduled_start_utc,
            "official_date": decision.official_date,
            "home_team": decision.home_team,
            "away_team": decision.away_team,
            "home_team_code": decision.home_team_code,
            "away_team_code": decision.away_team_code,
        },
        "model_identity": {
            "model_role": decision.model_role,
            "model_id": decision.model_id,
            "model_fingerprint": decision.model_fingerprint,
            "model_probability_source": decision.model_probability_source,
            "p_home": _decimal_text(decision.p_home),
            "p_away": _decimal_text(decision.p_away),
        },
        "market_authority": {
            "market_snapshot_id": decision.market_snapshot_id,
            "market_observed_at_utc": decision.market_observed_at_utc,
            "local_fetched_at_utc": decision.local_fetched_at_utc,
            "source_match_id": decision.source_match_id,
            "home_decimal_odds": _decimal_text(decision.home_decimal_odds),
            "away_decimal_odds": _decimal_text(decision.away_decimal_odds),
            "ev_home": _decimal_text(decision.ev_home),
            "ev_away": _decimal_text(decision.ev_away),
        },
        "pregame_freeze_authority": {
            "policy_id": P40A_POLICY_ID,
            "decision": decision.decision,
            "selected_side": selected_side,
            "candidate_side": decision.candidate_side,
            "paper_stake_units": _decimal_text(decision.paper_stake_units),
            "paper_stake_convention": decision.paper_stake_convention,
            "created_at_utc": manifest.get("created_at_utc"),
            "normalized_input_fingerprint": manifest.get("normalized_input_fingerprint"),
            "decision_bundle_fingerprint": manifest.get("decision_bundle_fingerprint"),
        },
        "final_result_authority": {
            "status": result_record.status,
            "home_score": result_record.home_score,
            "away_score": result_record.away_score,
            "actual_winner": result_record.actual_winner,
            "target_home_win": result_record.target_home_win,
            "result_observed_at_utc": result_record.result_observed_at_utc,
            "source_identity": result_record.source_identity,
        },
        "settlement": {
            "settlement_status": settlement.settlement_status,
            "paper_units_risked": _decimal_text(decision.paper_stake_units),
            "gross_return_units": _decimal_text(settlement.gross_return_units),
            "net_paper_units": _decimal_text(settlement.net_paper_units),
            "settlement_row_fingerprint": settlement.settlement_row_fingerprint,
            "settled_at_utc": settled_at_utc,
        },
        "evaluation": {
            "evaluation_status": evaluation["evaluation_status"],
            "correctness_label": evaluation["correctness_label"],
            "is_correct": evaluation["is_correct"],
        },
        "feedback": {
            "feedback_identity": _feedback_identity(settlement),
        },
    }

    record_id = _sha256_projection(payload_without_id)
    return {
        "ledger_record_id": record_id,
        **payload_without_id,
    }


@dataclass(frozen=True, slots=True)
class P45ASettleRunResult:
    run_id: str
    run_classification: str
    lifecycle_state: str
    newly_settled_count: int
    total_settled_count: int
    pending_count: int
    settled_decisions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    forward_summary: dict[str, Any]


def _calculate_max_drawdown(settled_bets: Sequence[Mapping[str, Any]]) -> str:
    """Calculate maximum paper drawdown across settled BET records."""

    peak = Decimal("0")
    running = Decimal("0")
    max_dd = Decimal("0")
    for row in settled_bets:
        net = Decimal(str(row["settlement"]["net_paper_units"]))
        running += net
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    return format(max_dd, "f") if max_dd != Decimal("0") else "0"


def compute_forward_paper_summary(
    forward_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute cumulative descriptive statistics over strictly prospective records."""

    prospective_rows = [
        row for row in forward_records
        if row.get("run_classification") == CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER
    ]

    distinct_runs = sorted({row["run_id"] for row in prospective_rows})
    bets = [row for row in prospective_rows if row["pregame_freeze_authority"]["decision"] == DECISION_BET]
    passes = [row for row in prospective_rows if row["pregame_freeze_authority"]["decision"] == DECISION_PASS]

    settled_bets = [
        row for row in bets
        if row["settlement"]["settlement_status"] in (SETTLEMENT_WON, SETTLEMENT_LOST)
    ]
    wins = [row for row in settled_bets if row["settlement"]["settlement_status"] == SETTLEMENT_WON]
    losses = [row for row in settled_bets if row["settlement"]["settlement_status"] == SETTLEMENT_LOST]

    total_risked = sum((Decimal(str(row["settlement"]["paper_units_risked"])) for row in settled_bets), Decimal("0"))
    total_net = sum((Decimal(str(row["settlement"]["net_paper_units"])) for row in prospective_rows), Decimal("0"))

    roi = None
    if total_risked > Decimal("0"):
        roi = _decimal_text(_ratio(total_net, total_risked))

    dates = [
        row["game_identity"]["official_date"]
        for row in prospective_rows
        if row.get("game_identity", {}).get("official_date")
    ]
    first_date = min(dates) if dates else None
    last_date = max(dates) if dates else None

    return {
        "schema_version": P45A_FORWARD_SUMMARY_SCHEMA,
        "run_count": len(distinct_runs),
        "run_ids": distinct_runs,
        "frozen_decision_count": len(prospective_rows),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "settled_bet_count": len(settled_bets),
        "pending_count": 0,
        "wins": len(wins),
        "losses": len(losses),
        "pushes": 0,
        "paper_units_risked": format(total_risked, "f") if total_risked != Decimal("0") else "0.0",
        "net_paper_units": format(total_net, "f") if total_net != Decimal("0") else "0.00",
        "descriptive_roi": roi,
        "max_drawdown": _calculate_max_drawdown(settled_bets),
        "feedback_count": len(prospective_rows),
        "first_target_date": first_date,
        "last_target_date": last_date,
        "forward_sample_count": len(prospective_rows),
    }


def settle_p45a_paper_run(
    repository_root: str | Path,
    *,
    run_dir: str | Path,
    result_input: Sequence[NormalizedResultRecord] | str | Path,
    ledger_root: str | Path | None = None,
    settled_at_utc: str = "2026-08-17T23:59:59Z",
) -> P45ASettleRunResult:
    """Settle available game results against an immutable frozen paper run."""

    root = Path(repository_root).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")

    manifest = read_json_object(manifest_path)
    run_id = manifest["run_id"]
    run_classification = manifest["run_classification"]

    # Verify decision bundle integrity
    decisions, pregame_records = load_p43a_frozen_decision_bundle(resolved_run_dir)
    pregame_records_by_id = {
        row["prediction_authority"]["p37_prediction_row_id"]: row
        for row in pregame_records
    }

    # Load result records
    if isinstance(result_input, (str, Path)):
        result_records = load_normalized_result_input(result_input)
    else:
        result_records = tuple(result_input)

    result_by_id: dict[str, NormalizedResultRecord] = {}
    for res in result_records:
        if res.prediction_row_id in result_by_id:
            raise RuntimeError("P45A_CONFLICTING_RESULT_REJECTED: duplicate result for prediction")
        result_by_id[res.prediction_row_id] = res

    # Load existing settlement state in run_dir
    settlements_file = resolved_run_dir / "settled_decisions.jsonl"
    existing_settled_records = read_jsonl_objects(settlements_file)
    existing_settled_by_id = {
        row["decision_identity"]["p37_prediction_row_id"]: row
        for row in existing_settled_records
    }

    newly_settled: list[dict[str, Any]] = []
    all_settled: list[dict[str, Any]] = list(existing_settled_records)

    for decision in decisions:
        pred_id = decision.p37_prediction_row_id
        existing = existing_settled_by_id.get(pred_id)
        incoming_res = result_by_id.get(pred_id)

        if existing is not None:
            # Already settled in a prior pass
            if incoming_res is not None:
                # Check for conflicting result
                prev_winner = existing["final_result_authority"]["actual_winner"]
                prev_home = existing["final_result_authority"]["home_score"]
                prev_away = existing["final_result_authority"]["away_score"]
                if (
                    incoming_res.actual_winner != prev_winner
                    or incoming_res.home_score != prev_home
                    or incoming_res.away_score != prev_away
                ):
                    raise RuntimeError(
                        "P45A_CONFLICTING_RESULT_REJECTED: conflicting result authority for already-settled decision"
                    )
            continue

        # Not yet settled
        if incoming_res is not None:
            if incoming_res.provider_game_id != decision.provider_game_id:
                raise RuntimeError(
                    "P45A_CONFLICTING_RESULT_REJECTED: outcome identity mismatch"
                )

            # Settle using domain model
            settlement = settle_paper_moneyline_decision(
                decision,
                final_game_outcome=incoming_res.actual_winner,
                target_home_win=incoming_res.target_home_win,
                outcome_authority_row_id=decision.p37_prediction_row_id,
                outcome_authority=incoming_res.source_identity,
            )

            pregame_record = pregame_records_by_id[pred_id]
            ledger_record = build_p45a_ledger_record(
                run_id=run_id,
                run_classification=run_classification,
                decision=decision,
                pregame_record=pregame_record,
                result_record=incoming_res,
                settlement=settlement,
                manifest=manifest,
                settled_at_utc=settled_at_utc,
            )

            newly_settled.append(ledger_record)
            all_settled.append(ledger_record)

    # Determine lifecycle state
    total_eligible = len(decisions)
    total_settled = len(all_settled)
    pending_count = total_eligible - total_settled

    if total_settled == 0:
        new_state = STATE_FROZEN
    elif total_settled < total_eligible:
        new_state = STATE_PARTIALLY_SETTLED
    else:
        new_state = STATE_SETTLED

    # Update settlements in run directory
    settlements_file.write_bytes(_jsonl_bytes(all_settled))

    settled_bets = [
        row for row in all_settled
        if row["pregame_freeze_authority"]["decision"] == DECISION_BET
    ]
    settled_passes = [
        row for row in all_settled
        if row["pregame_freeze_authority"]["decision"] == DECISION_PASS
    ]
    wins = [row for row in settled_bets if row["settlement"]["settlement_status"] == SETTLEMENT_WON]
    losses = [row for row in settled_bets if row["settlement"]["settlement_status"] == SETTLEMENT_LOST]
    units_risked = sum((Decimal(str(row["settlement"]["paper_units_risked"])) for row in settled_bets), Decimal("0"))
    net_units = sum((Decimal(str(row["settlement"]["net_paper_units"])) for row in all_settled), Decimal("0"))

    roi = None
    if units_risked > Decimal("0"):
        roi = _decimal_text(_ratio(net_units, units_risked))

    settlement_summary = {
        "schema_version": P45A_SETTLEMENT_SUMMARY_SCHEMA,
        "run_id": run_id,
        "run_classification": run_classification,
        "lifecycle_state": new_state,
        "target_universe_count": manifest.get("target_universe_count", total_eligible),
        "eligible_decision_count": total_eligible,
        "settled_total_count": total_settled,
        "settled_bet_count": len(settled_bets),
        "settled_pass_count": len(settled_passes),
        "pending_count": pending_count,
        "win_count": len(wins),
        "loss_count": len(losses),
        "push_count": 0,
        "units_risked": format(units_risked, "f"),
        "net_paper_units": format(net_units, "f"),
        "descriptive_roi": roi,
        "max_drawdown": _calculate_max_drawdown(settled_bets),
        "feedback_row_count": total_settled,
    }
    (resolved_run_dir / "settlement_summary.json").write_bytes(_json_bytes(settlement_summary))

    # Update run manifest
    updated_manifest = dict(manifest)
    updated_manifest["lifecycle_state"] = new_state
    updated_manifest["settled_total_count"] = total_settled
    updated_manifest["settled_bet_count"] = len(settled_bets)
    updated_manifest["pending_count"] = pending_count
    manifest_path.write_bytes(_json_bytes(updated_manifest))

    # Update ledger
    if ledger_root is not None:
        resolved_ledger_root = Path(ledger_root).resolve()
    else:
        resolved_ledger_root = (root / P45A_REPORT_RELATIVE_PATH / "ledger").resolve()

    resolved_ledger_root.mkdir(parents=True, exist_ok=True)

    if run_classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER:
        target_ledger_file = resolved_ledger_root / "forward_paper_ledger.jsonl"
    else:
        target_ledger_file = resolved_ledger_root / "rehearsal_ledger.jsonl"

    existing_ledger_rows = read_jsonl_objects(target_ledger_file)
    existing_ledger_keys = {
        (row["run_id"], row["decision_identity"]["decision_fingerprint"]): row
        for row in existing_ledger_rows
    }

    rows_to_append: list[dict[str, Any]] = []
    for record in newly_settled:
        key = (record["run_id"], record["decision_identity"]["decision_fingerprint"])
        if key in existing_ledger_keys:
            existing_row = existing_ledger_keys[key]
            if existing_row != record:
                raise RuntimeError("P45A_CONFLICTING_RESULT_REJECTED: ledger record conflict")
            continue
        rows_to_append.append(record)
        existing_ledger_keys[key] = record

    if rows_to_append:
        with target_ledger_file.open("ab") as f:
            f.write(_jsonl_bytes(rows_to_append))

    # Update forward cumulative summary
    forward_ledger_file = resolved_ledger_root / "forward_paper_ledger.jsonl"
    forward_rows = read_jsonl_objects(forward_ledger_file)
    forward_summary = compute_forward_paper_summary(forward_rows)
    (resolved_ledger_root / "forward_summary.json").write_bytes(_json_bytes(forward_summary))

    return P45ASettleRunResult(
        run_id=run_id,
        run_classification=run_classification,
        lifecycle_state=new_state,
        newly_settled_count=len(newly_settled),
        total_settled_count=total_settled,
        pending_count=pending_count,
        settled_decisions=tuple(all_settled),
        summary=settlement_summary,
        forward_summary=forward_summary,
    )


def get_p45a_run_status(
    repository_root: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Return status and summary of an existing paper run."""

    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
    manifest = read_json_object(manifest_path)
    summary_path = resolved_run_dir / "settlement_summary.json"
    summary = read_json_object(summary_path) if summary_path.is_file() else {}
    return {
        "run_id": manifest["run_id"],
        "run_classification": manifest["run_classification"],
        "lifecycle_state": manifest["lifecycle_state"],
        "created_at_utc": manifest.get("created_at_utc"),
        "eligible_decision_count": manifest["eligible_decision_count"],
        "bet_count": manifest["bet_count"],
        "pass_count": manifest["pass_count"],
        "settled_total_count": manifest.get("settled_total_count", 0),
        "settled_bet_count": manifest.get("settled_bet_count", 0),
        "pending_count": manifest.get("pending_count", manifest["eligible_decision_count"]),
        "exclusion_count": manifest.get("exclusion_count", 0),
        "settlement_summary": summary,
    }


def get_p45a_forward_summary(
    repository_root: str | Path,
    *,
    ledger_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return cumulative forward paper summary."""

    root = Path(repository_root).resolve()
    if ledger_root is not None:
        resolved_ledger_root = Path(ledger_root).resolve()
    else:
        resolved_ledger_root = (root / P45A_REPORT_RELATIVE_PATH / "ledger").resolve()

    forward_ledger_file = resolved_ledger_root / "forward_paper_ledger.jsonl"
    forward_rows = read_jsonl_objects(forward_ledger_file)
    return compute_forward_paper_summary(forward_rows)


__all__ = (
    "CLASSIFICATION_HISTORICAL_REHEARSAL",
    "CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER",
    "P45A_FORWARD_SUMMARY_SCHEMA",
    "P45A_LEDGER_RECORD_SCHEMA",
    "P45A_REPORT_RELATIVE_PATH",
    "P45A_RUN_MANIFEST_SCHEMA",
    "P45A_SETTLEMENT_SUMMARY_SCHEMA",
    "P45A_TASK_ID",
    "P45ACreateRunResult",
    "P45ASettleRunResult",
    "STATE_FROZEN",
    "STATE_PARTIALLY_SETTLED",
    "STATE_SETTLED",
    "VALID_CLASSIFICATIONS",
    "VALID_LIFECYCLE_STATES",
    "build_p45a_ledger_record",
    "compute_deterministic_run_id",
    "compute_forward_paper_summary",
    "create_p45a_paper_run",
    "get_p45a_forward_summary",
    "get_p45a_run_status",
    "settle_p45a_paper_run",
    "validate_run_classification",
)
