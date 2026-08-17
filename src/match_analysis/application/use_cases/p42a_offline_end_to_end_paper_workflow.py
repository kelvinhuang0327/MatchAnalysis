"""P42A offline historical paper-workflow rehearsal.

Composes already-committed P37/P39/P40 authorities into one coherent path:

frozen Champion probability
→ frozen P39 pregame market
→ frozen P40 zero-EV BET/PASS
→ authoritative HOME/AWAY result
→ existing P40 settlement
→ evaluation/feedback lineage
→ append-only workflow ledger

This module does not re-derive EV, change the P40 rule, call a network, or
claim prospective/live/production status.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    P40A_POLICY_ID,
    SETTLEMENT_LOST,
    SETTLEMENT_PASS,
    SETTLEMENT_WON,
    PaperMoneylineDecision,
    PaperMoneylineSettlement,
    aggregate_paper_settlements,
    settle_paper_moneyline_decision,
)
from .p40a_moneyline_paper_bet_pass import (
    P37A_REPORT_RELATIVE_PATH,
    P38A_REPORT_RELATIVE_PATH,
    P39A_REPORT_RELATIVE_PATH,
    P40A_CHAMPION_ROLE,
    P40A_OUTCOME_AUTHORITY,
    P40A_REPORT_RELATIVE_PATH,
    P40AAuthority,
    P40AOutcomeRow,
    build_p40a_decisions,
    load_p40a_authority,
)


P42A_TASK_ID = "P42A"
P42A_REPORT_RELATIVE_PATH = Path("report/p42a_offline_end_to_end_paper_workflow")
P42A_ARTIFACT_FILES = (
    "source_manifest.json",
    "workflow_ledger.jsonl",
    "exclusions.jsonl",
    "summary.json",
    "report.md",
)
P42A_LEDGER_SCHEMA_VERSION = "p42a.offline_end_to_end_paper_workflow_ledger.v1"
P42A_SUMMARY_SCHEMA_VERSION = "p42a.offline_end_to_end_paper_workflow_summary.v1"
P42A_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "p42a.offline_end_to_end_paper_workflow_source_manifest.v1"
)
P42A_WORKFLOW_LABEL = "OFFLINE_HISTORICAL_PAPER_REHEARSAL"
P42A_EVALUATION_BET_WON = "BET_WON"
P42A_EVALUATION_BET_LOST = "BET_LOST"
P42A_EVALUATION_PASS_NO_WAGER = "PASS_NO_WAGER"
P41A_REPORT_RELATIVE_PATH = Path("report/p41a_walk_forward_ev_margin_policy")
P42A_EXPECTED_P37_TARGET_COUNT = 65
P42A_EXPECTED_P39_EDGE_READY_COUNT = 62
P42A_EXPECTED_P39_NO_MARKET_COUNT = 3
P42A_EXPECTED_CHAMPION = {
    "edge_ready_rows": 62,
    "bet_count": 22,
    "pass_count": 40,
    "win_count": 14,
    "loss_count": 8,
    "push_count": 0,
    "total_paper_units_risked": "22.0",
    "net_paper_units": "5.90",
    "maximum_paper_drawdown": "2",
    "descriptive_paper_roi": (
        "0.26818181818181818181818181818181818181818181818182"
    ),
}

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


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_projection(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_rejecting_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path} contains a blank row at line {line_number}")
        value = json.loads(line, object_pairs_hook=_duplicate_rejecting_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _protected_authority_hashes(root: Path) -> dict[str, str]:
    paths = {
        "p37_comparisons": root / P37A_REPORT_RELATIVE_PATH / "comparisons.jsonl",
        "p37_summary": root / P37A_REPORT_RELATIVE_PATH / "summary.json",
        "p38_comparisons": root / P38A_REPORT_RELATIVE_PATH / "comparisons.jsonl",
        "p38_summary": root / P38A_REPORT_RELATIVE_PATH / "summary.json",
        "p39_market_join": root / P39A_REPORT_RELATIVE_PATH / "market_join.jsonl",
        "p39_market_snapshots": root / P39A_REPORT_RELATIVE_PATH / "market_snapshots.jsonl",
        "p39_summary": root / P39A_REPORT_RELATIVE_PATH / "summary.json",
        "p39_source_manifest": root / P39A_REPORT_RELATIVE_PATH / "source_manifest.json",
        "p40_decisions": root / P40A_REPORT_RELATIVE_PATH / "decisions.jsonl",
        "p40_settlements": root / P40A_REPORT_RELATIVE_PATH / "settlements.jsonl",
        "p40_summary": root / P40A_REPORT_RELATIVE_PATH / "summary.json",
        "p40_source_manifest": root / P40A_REPORT_RELATIVE_PATH / "source_manifest.json",
        "p41_summary": root / P41A_REPORT_RELATIVE_PATH / "summary.json",
        "p41_policy_evaluations": root
        / P41A_REPORT_RELATIVE_PATH
        / "policy_evaluations.jsonl",
    }
    return {name: _sha256_path(path) for name, path in paths.items()}


def load_p39_no_market_exclusions(repository_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Return the committed P39 NO_MARKET rows; they never become decisions."""

    path = Path(repository_root).resolve() / P39A_REPORT_RELATIVE_PATH / "market_join.jsonl"
    rows = [
        row
        for row in _read_jsonl(path)
        if row.get("market_snapshot_status") == "NO_MARKET"
    ]
    if len(rows) != P42A_EXPECTED_P39_NO_MARKET_COUNT:
        raise ValueError(
            f"P42A expected {P42A_EXPECTED_P39_NO_MARKET_COUNT} NO_MARKET rows, got {len(rows)}"
        )
    return tuple(rows)


def freeze_champion_decisions(
    authority: P40AAuthority,
    *,
    market_rows: Sequence[Any] | None = None,
    prediction_rows: Sequence[Any] | None = None,
) -> tuple[PaperMoneylineDecision, ...]:
    """Freeze Champion-primary decisions from pregame inputs only."""

    decisions = build_p40a_decisions(
        authority,
        market_rows=market_rows,
        prediction_rows=prediction_rows,
    )
    champion = tuple(
        row for row in decisions if row.model_role == P40A_CHAMPION_ROLE
    )
    if len(champion) != P42A_EXPECTED_P39_EDGE_READY_COUNT:
        raise ValueError("P42A Champion decision universe drifted from 62 edge-ready rows")
    if any("final_game_outcome" in row.to_projection() for row in champion):
        raise RuntimeError("P42A decision freeze included outcome fields")
    return champion


def _validate_outcome_attachment_universe(
    decisions: Sequence[PaperMoneylineDecision],
    outcome_rows: Sequence[P40AOutcomeRow],
) -> dict[str, P40AOutcomeRow]:
    """Fail closed on missing, duplicate, or non-final outcomes."""

    by_id: dict[str, P40AOutcomeRow] = {}
    for outcome in outcome_rows:
        if outcome.p37_prediction_row_id in by_id:
            raise RuntimeError("P42A_DUPLICATE_RESULT_REJECTED_STOP")
        if outcome.actual_winner not in ("HOME", "AWAY"):
            raise RuntimeError("P42A_NON_FINAL_RESULT_FAIL_CLOSED_STOP")
        by_id[outcome.p37_prediction_row_id] = outcome
    for decision in decisions:
        outcome = by_id.get(decision.p37_prediction_row_id)
        if outcome is None:
            raise RuntimeError("P42A_MISSING_RESULT_FAIL_CLOSED_STOP")
        if outcome.provider_game_id != decision.provider_game_id:
            raise RuntimeError("P42A_OUTCOME_IDENTITY_MISMATCH_STOP")
    return by_id


def attach_and_settle_frozen_decisions(
    decisions: Sequence[PaperMoneylineDecision],
    outcome_rows: Sequence[P40AOutcomeRow],
) -> tuple[PaperMoneylineSettlement, ...]:
    """Attach exactly one HOME/AWAY result after the decision fingerprint exists."""

    if any("final_game_outcome" in row.to_projection() for row in decisions):
        raise RuntimeError("P42A decision was not frozen before outcome attachment")
    by_id = _validate_outcome_attachment_universe(decisions, outcome_rows)
    settlements: list[PaperMoneylineSettlement] = []
    for decision in decisions:
        outcome = by_id[decision.p37_prediction_row_id]
        settlements.append(
            settle_paper_moneyline_decision(
                decision,
                final_game_outcome=outcome.actual_winner,
                target_home_win=outcome.target_home_win,
                outcome_authority_row_id=outcome.p37_prediction_row_id,
                outcome_authority=P40A_OUTCOME_AUTHORITY,
            )
        )
    if len(settlements) != len(decisions):
        raise RuntimeError("P42A settlement count drifted from frozen decisions")
    return tuple(settlements)


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


def build_workflow_ledger_row(
    settlement: PaperMoneylineSettlement,
    *,
    workflow_execution_id: str,
) -> dict[str, Any]:
    """Project one Champion settlement into the P42A rehearsal ledger."""

    decision = settlement.decision
    evaluation = _evaluation_projection(settlement)
    selected_side = (
        decision.candidate_side if decision.decision == DECISION_BET else "NONE"
    )
    return {
        "schema_version": P42A_LEDGER_SCHEMA_VERSION,
        "workflow_label": P42A_WORKFLOW_LABEL,
        "workflow_kind": "OFFLINE_HISTORICAL_PAPER_REHEARSAL",
        "prospective": False,
        "live": False,
        "production": False,
        "forward_real": False,
        "real_betting_history": False,
        "workflow_execution_id": workflow_execution_id,
        "game_identity": {
            "provider_namespace": decision.provider_namespace,
            "provider_game_id": decision.provider_game_id,
            "game_pk": decision.game_pk,
            "game_number": decision.game_number,
        },
        "p37_window": decision.p37_window,
        "p37_fold_id": decision.p37_fold_id,
        "scheduled_start": decision.scheduled_start_utc,
        "model_identity": {
            "model_role": decision.model_role,
            "model_id": decision.model_id,
            "model_fingerprint": decision.model_fingerprint,
        },
        "prediction_probability": _decimal_text(decision.p_home),
        "p_home": _decimal_text(decision.p_home),
        "p_away": _decimal_text(decision.p_away),
        "market_observation_identity": decision.market_snapshot_id,
        "market_observation_time": decision.market_observed_at_utc,
        "home_decimal_odds": _decimal_text(decision.home_decimal_odds),
        "away_decimal_odds": _decimal_text(decision.away_decimal_odds),
        "ev_home": _decimal_text(decision.ev_home),
        "ev_away": _decimal_text(decision.ev_away),
        "bet_or_pass": decision.decision,
        "selected_side": selected_side,
        "candidate_side": decision.candidate_side,
        "paper_stake_convention": decision.paper_stake_convention,
        "paper_stake_units": _decimal_text(decision.paper_stake_units),
        "decision_fingerprint": decision.decision_id,
        "decision_id": decision.decision_id,
        "final_score": None,
        "actual_winner": settlement.final_game_outcome,
        "target_home_win": settlement.target_home_win,
        "settlement_status": settlement.settlement_status,
        "net_paper_units": _decimal_text(settlement.net_paper_units),
        "gross_return_units": _decimal_text(settlement.gross_return_units),
        "evaluation_status": evaluation["evaluation_status"],
        "evaluation_correctness_status": evaluation["correctness_label"],
        "evaluation_is_correct": evaluation["is_correct"],
        "feedback_identity": _feedback_identity(settlement),
        "upstream_authority_identifiers": {
            "p37_prediction_row_id": decision.p37_prediction_row_id,
            "p39_market_snapshot_id": decision.market_snapshot_id,
            "p39_source_match_id": decision.source_match_id,
            "p40_policy_id": P40A_POLICY_ID,
            "p40_decision_id": decision.decision_id,
            "p40_settlement_row_fingerprint": settlement.settlement_row_fingerprint,
            "outcome_authority": settlement.outcome_authority,
            "outcome_authority_row_id": settlement.outcome_authority_row_id,
        },
    }


def _exclusion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "p42a.offline_end_to_end_paper_workflow_exclusion.v1",
        "workflow_label": P42A_WORKFLOW_LABEL,
        "exclusion_reason": "NO_MARKET",
        "market_snapshot_status": row.get("market_snapshot_status"),
        "p37_prediction_row_id": row.get("p37_prediction_row_id"),
        "p37_window": row.get("p37_window"),
        "p37_fold_id": row.get("p37_fold_id"),
        "provider_namespace": row.get("provider_namespace"),
        "provider_game_id": row.get("provider_game_id"),
        "scheduled_start_utc": row.get("scheduled_start_utc"),
        "became_bet": False,
    }


def _workflow_execution_id(
    *,
    authority_hashes: Mapping[str, str],
    decisions: Sequence[PaperMoneylineDecision],
) -> str:
    return _sha256_projection(
        {
            "schema_version": P42A_LEDGER_SCHEMA_VERSION,
            "workflow_label": P42A_WORKFLOW_LABEL,
            "policy_id": P40A_POLICY_ID,
            "model_role": P40A_CHAMPION_ROLE,
            "authority_hashes": dict(authority_hashes),
            "decision_fingerprints": [row.decision_id for row in decisions],
        }
    )


def reconcile_with_p40_champion(
    settlements: Sequence[PaperMoneylineSettlement],
    committed_p40_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare P42A Champion aggregates to committed P40 authority."""

    aggregate = aggregate_paper_settlements(
        settlements,
        edge_ready_rows=P42A_EXPECTED_P39_EDGE_READY_COUNT,
        model_role=P40A_CHAMPION_ROLE,
    )
    committed = committed_p40_summary["models"]["champion_primary"]
    compared = {
        "edge_ready_rows": aggregate["edge_ready_rows"],
        "bet_count": aggregate["bet_count"],
        "pass_count": aggregate["pass_count"],
        "win_count": aggregate["win_count"],
        "loss_count": aggregate["loss_count"],
        "push_count": aggregate["push_count"],
        "total_paper_units_risked": aggregate["total_paper_units_risked"],
        "net_paper_units": aggregate["net_paper_units"],
        "maximum_paper_drawdown": aggregate["maximum_paper_drawdown"],
        "descriptive_paper_roi": aggregate["descriptive_paper_roi"],
    }
    expected = {
        key: committed[key] if key != "edge_ready_rows" else committed["edge_ready_rows"]
        for key in compared
    }
    if compared != expected or compared != P42A_EXPECTED_CHAMPION:
        raise RuntimeError(
            "P42A_P40_RECONCILIATION_FAILURE_STOP "
            f"computed={compared} committed={expected} packet={P42A_EXPECTED_CHAMPION}"
        )
    return {
        "status": "RECONCILED",
        "computed": compared,
        "committed_p40_champion": expected,
    }


def _assert_pass_semantics(settlements: Sequence[PaperMoneylineSettlement]) -> None:
    for row in settlements:
        if row.decision.decision != DECISION_PASS:
            continue
        if row.decision.paper_stake_units != Decimal("0"):
            raise ValueError("PASS must have zero stake")
        if row.net_paper_units != Decimal("0") or row.gross_return_units != Decimal("0"):
            raise ValueError("PASS must have zero net and gross units")
        if row.settlement_status != SETTLEMENT_PASS:
            raise ValueError("PASS must not receive a fake win/loss settlement")


def _assert_pregame_inputs(decisions: Sequence[PaperMoneylineDecision]) -> None:
    for row in decisions:
        if row.market_observed_at_utc >= row.scheduled_start_utc:
            raise ValueError("P42A market observation is not strictly pregame")
        if row.local_fetched_at_utc >= row.scheduled_start_utc:
            raise ValueError("P42A local fetch is not strictly pregame")


def _build_summary(
    *,
    p37_target_count: int,
    no_market_rows: Sequence[Mapping[str, Any]],
    settlements: Sequence[PaperMoneylineSettlement],
    ledger_rows: Sequence[Mapping[str, Any]],
    reconciliation: Mapping[str, Any],
    deterministic_rerun_verified: bool,
    authority_hashes: Mapping[str, str],
) -> dict[str, Any]:
    bets = tuple(row for row in settlements if row.decision.decision == DECISION_BET)
    passes = tuple(row for row in settlements if row.decision.decision == DECISION_PASS)
    wins = tuple(row for row in bets if row.settlement_status == SETTLEMENT_WON)
    losses = tuple(row for row in bets if row.settlement_status == SETTLEMENT_LOST)
    champion = reconciliation["computed"]
    exclusion_counts = {"NO_MARKET": len(no_market_rows)}
    return {
        "schema_version": P42A_SUMMARY_SCHEMA_VERSION,
        "task_id": P42A_TASK_ID,
        "workflow_label": P42A_WORKFLOW_LABEL,
        "workflow_kind": "OFFLINE_HISTORICAL_PAPER_REHEARSAL",
        "labels": ["OFFLINE", "HISTORICAL", "PAPER_REHEARSAL"],
        "prospective": False,
        "live": False,
        "production": False,
        "forward_real": False,
        "real_betting_history": False,
        "descriptive_only": True,
        "roi_label": "DESCRIPTIVE_PAPER_ONLY",
        "p37_target_count": p37_target_count,
        "p39_edge_ready_count": P42A_EXPECTED_P39_EDGE_READY_COUNT,
        "workflow_decision_count": len(settlements),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "settled_bet_count": len(wins) + len(losses),
        "unresolved_result_count": 0,
        "feedback_row_count": len(ledger_rows),
        "win_count": len(wins),
        "loss_count": len(losses),
        "push_count": 0,
        "units_risked": champion["total_paper_units_risked"],
        "net_paper_units": champion["net_paper_units"],
        "descriptive_historical_paper_roi": champion["descriptive_paper_roi"],
        "maximum_drawdown": champion["maximum_paper_drawdown"],
        "workflow_completeness": {
            "frozen_decisions": len(settlements),
            "settlements": len(settlements),
            "evaluations": len(ledger_rows),
            "feedback_rows": len(ledger_rows),
            "one_to_one_lineage": len({row["decision_fingerprint"] for row in ledger_rows})
            == len(ledger_rows)
            == len(settlements),
        },
        "exclusion_reasons": exclusion_counts,
        "p39_no_market_count": len(no_market_rows),
        "p40_reconciliation": reconciliation,
        "deterministic_rerun_verified": deterministic_rerun_verified,
        "outcome_isolation_verified": True,
        "network_required": False,
        "protected_authority_hashes": dict(authority_hashes),
        "decision_rule": {
            "policy_id": P40A_POLICY_ID,
            "consumed_not_rederived": True,
            "additional_threshold": "NONE",
        },
        "claims": {
            "real_betting": False,
            "profitability_claim": False,
            "expected_future_roi_claim": False,
            "threshold_optimization": False,
            "staking_optimization": False,
            "kelly": False,
            "model_promotion": False,
            "calibration": False,
            "training": False,
            "live_acquisition": False,
            "prospective_forward_sample": False,
        },
    }


@dataclass(frozen=True, slots=True)
class P42AResult:
    authority: P40AAuthority
    decisions: tuple[PaperMoneylineDecision, ...]
    settlements: tuple[PaperMoneylineSettlement, ...]
    ledger_rows: tuple[dict[str, Any], ...]
    exclusion_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    summary: dict[str, Any]
    authority_hashes_before: dict[str, str]
    authority_hashes_after: dict[str, str]


def _build_once(
    authority: P40AAuthority,
    *,
    no_market_rows: Sequence[Mapping[str, Any]],
    authority_hashes: Mapping[str, str],
    committed_p40_summary: Mapping[str, Any],
) -> tuple[
    tuple[PaperMoneylineDecision, ...],
    tuple[PaperMoneylineSettlement, ...],
    tuple[dict[str, Any], ...],
]:
    # Phase boundary: outcomes are not passed into decision freeze.
    decisions = freeze_champion_decisions(authority)
    _assert_pregame_inputs(decisions)
    settlements = attach_and_settle_frozen_decisions(
        decisions,
        authority.outcome_rows,
    )
    _assert_pass_semantics(settlements)
    reconcile_with_p40_champion(settlements, committed_p40_summary)
    workflow_execution_id = _workflow_execution_id(
        authority_hashes=authority_hashes,
        decisions=decisions,
    )
    ledger_rows = tuple(
        build_workflow_ledger_row(
            settlement,
            workflow_execution_id=workflow_execution_id,
        )
        for settlement in settlements
    )
    if len(no_market_rows) != P42A_EXPECTED_P39_NO_MARKET_COUNT:
        raise ValueError("P42A NO_MARKET accounting drifted")
    no_market_ids = {row["p37_prediction_row_id"] for row in no_market_rows}
    if no_market_ids & {row.p37_prediction_row_id for row in decisions}:
        raise ValueError("P42A admitted a NO_MARKET row into the decision universe")
    return decisions, settlements, ledger_rows


def run_p42a_offline_end_to_end_paper_workflow(
    repository_root: str | Path,
) -> P42AResult:
    """Run the offline rehearsal twice from frozen authorities."""

    root = Path(repository_root).resolve()
    hashes_before = _protected_authority_hashes(root)
    authority = load_p40a_authority(root)
    no_market_rows = load_p39_no_market_exclusions(root)
    committed_p40_summary = _read_json(root / P40A_REPORT_RELATIVE_PATH / "summary.json")
    if len(authority.prediction_rows) != P42A_EXPECTED_P37_TARGET_COUNT:
        raise ValueError("P42A P37 target count drifted")

    first = _build_once(
        authority,
        no_market_rows=no_market_rows,
        authority_hashes=hashes_before,
        committed_p40_summary=committed_p40_summary,
    )
    second = _build_once(
        authority,
        no_market_rows=no_market_rows,
        authority_hashes=hashes_before,
        committed_p40_summary=committed_p40_summary,
    )
    if tuple(row.to_projection() for row in first[0]) != tuple(
        row.to_projection() for row in second[0]
    ):
        raise RuntimeError("P42A deterministic decision replay differed")
    if tuple(row.to_projection() for row in first[1]) != tuple(
        row.to_projection() for row in second[1]
    ):
        raise RuntimeError("P42A deterministic settlement replay differed")
    if first[2] != second[2]:
        raise RuntimeError("P42A deterministic ledger replay differed")

    decisions, settlements, ledger_rows = first
    reconciliation = reconcile_with_p40_champion(settlements, committed_p40_summary)
    exclusion_rows = tuple(_exclusion_row(row) for row in no_market_rows)
    source_manifest = {
        "schema_version": P42A_SOURCE_MANIFEST_SCHEMA_VERSION,
        "task_id": P42A_TASK_ID,
        "workflow_label": P42A_WORKFLOW_LABEL,
        "workflow_kind": "OFFLINE_HISTORICAL_PAPER_REHEARSAL",
        "consumed_authorities": {
            "p37a": str(P37A_REPORT_RELATIVE_PATH),
            "p38a_read_only_unused_for_decisions": str(P38A_REPORT_RELATIVE_PATH),
            "p39a": str(P39A_REPORT_RELATIVE_PATH),
            "p40a": str(P40A_REPORT_RELATIVE_PATH),
            "p41a_research_only": str(P41A_REPORT_RELATIVE_PATH),
        },
        "p40_policy_id": P40A_POLICY_ID,
        "p40_rule_changed": False,
        "network_required": False,
        "protected_authority_hashes": hashes_before,
    }
    summary = _build_summary(
        p37_target_count=len(authority.prediction_rows),
        no_market_rows=no_market_rows,
        settlements=settlements,
        ledger_rows=ledger_rows,
        reconciliation=reconciliation,
        deterministic_rerun_verified=True,
        authority_hashes=hashes_before,
    )
    hashes_after = _protected_authority_hashes(root)
    if hashes_after != hashes_before:
        raise RuntimeError("P42A mutated a protected P37/P38/P39/P40/P41 authority")
    return P42AResult(
        authority=authority,
        decisions=decisions,
        settlements=settlements,
        ledger_rows=ledger_rows,
        exclusion_rows=exclusion_rows,
        source_manifest=source_manifest,
        summary=summary,
        authority_hashes_before=hashes_before,
        authority_hashes_after=hashes_after,
    )


def render_p42a_report(result: P42AResult) -> str:
    summary = result.summary
    return "\n".join(
        [
            "# P42A Offline End-to-End Paper Workflow Rehearsal",
            "",
            "This artifact is an **OFFLINE / HISTORICAL PAPER REHEARSAL**.",
            "It is not prospective, live, production, forward-real, or real betting history.",
            "",
            "## Composition",
            "",
            "- Frozen P37 Champion probability (read-only).",
            "- Frozen P39 trusted pregame Moneyline market (read-only).",
            "- Frozen P40 zero-EV BET/PASS rule, consumed not re-derived.",
            "- Authoritative P37 HOME/AWAY final result attached only after decision freeze.",
            "- Existing P40 settlement semantics.",
            "- Evaluation/feedback lineage projected from that settlement.",
            "",
            "## Counts",
            "",
            f"- P37 target count: `{summary['p37_target_count']}`.",
            f"- P39 edge-ready count: `{summary['p39_edge_ready_count']}`.",
            f"- Workflow decisions: `{summary['workflow_decision_count']}`.",
            f"- BET `{summary['bet_count']}` / PASS `{summary['pass_count']}`.",
            f"- Settled BET `{summary['settled_bet_count']}`; unresolved results `{summary['unresolved_result_count']}`.",
            f"- Feedback rows: `{summary['feedback_row_count']}`.",
            f"- Wins `{summary['win_count']}` / losses `{summary['loss_count']}` / pushes `{summary['push_count']}`.",
            f"- Units risked `{summary['units_risked']}`; net paper units `{summary['net_paper_units']}`.",
            f"- DESCRIPTIVE_PAPER_ONLY ROI `{summary['descriptive_historical_paper_roi']}`.",
            f"- Maximum drawdown `{summary['maximum_drawdown']}`.",
            f"- NO_MARKET exclusions: `{summary['p39_no_market_count']}` (never became bets).",
            f"- P40 reconciliation: `{summary['p40_reconciliation']['status']}`.",
            f"- Deterministic rerun: `{summary['deterministic_rerun_verified']}`.",
            "",
            "## Safety boundary",
            "",
            "- Historical paper rehearsal only. Not a profitability or promotion claim.",
            "- Real betting: `NOT RUN`.",
            "- Threshold / staking / Kelly: `NOT RUN`.",
            "- Live TSL / MLB / Fortinet / P35A: `NOT RUN`.",
            "",
        ]
    )


def render_p42a_artifacts(result: P42AResult) -> dict[str, bytes]:
    return {
        "source_manifest.json": _json_bytes(result.source_manifest),
        "workflow_ledger.jsonl": _jsonl_bytes(result.ledger_rows),
        "exclusions.jsonl": _jsonl_bytes(result.exclusion_rows),
        "summary.json": _json_bytes(result.summary),
        "report.md": render_p42a_report(result).encode("utf-8"),
    }


def write_p42a_artifacts(output_dir: str | Path, result: P42AResult) -> dict[str, str]:
    """Write only the allowlisted repository-native P42A artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_p42a_artifacts(result)
    hashes: dict[str, str] = {}
    for name, content in rendered.items():
        path = directory / name
        path.write_bytes(content)
        hashes[name] = _sha256_bytes(content)
    return hashes


__all__ = (
    "P42A_ARTIFACT_FILES",
    "P42A_EXPECTED_CHAMPION",
    "P42A_REPORT_RELATIVE_PATH",
    "P42A_TASK_ID",
    "P42A_WORKFLOW_LABEL",
    "P42AResult",
    "attach_and_settle_frozen_decisions",
    "build_workflow_ledger_row",
    "freeze_champion_decisions",
    "load_p39_no_market_exclusions",
    "reconcile_with_p40_champion",
    "render_p42a_artifacts",
    "run_p42a_offline_end_to_end_paper_workflow",
    "write_p42a_artifacts",
)
