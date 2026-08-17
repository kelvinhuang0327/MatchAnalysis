"""P43A Phase 2: postgame settlement of an already-frozen decision bundle.

Consumes the immutable Phase 1 bundle, verifies fingerprints, attaches exactly
one authoritative final result, and projects settlement → evaluation →
feedback → ledger. This module never rebuilds BET/PASS from market or
prediction authority.
"""

from __future__ import annotations

from dataclasses import dataclass
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
)
from .p40a_moneyline_paper_bet_pass import (
    P40A_CHAMPION_ROLE,
    P40AOutcomeRow,
    _read_json,
)
from .p42a_offline_end_to_end_paper_workflow import (
    P42A_EXPECTED_CHAMPION,
    P42A_REPORT_RELATIVE_PATH,
    attach_and_settle_frozen_decisions,
    build_workflow_ledger_row,
    reconcile_with_p40_champion,
)
from .p43a_pregame_freeze import (
    P43A_CLAIMS,
    P43A_HUMAN_LABEL,
    P43A_REPORT_RELATIVE_PATH,
    P43A_TASK_ID,
    P43A_WORKFLOW_KIND,
    P43A_WORKFLOW_LABEL,
    _json_bytes,
    _jsonl_bytes,
    _sha256_projection,
    read_json_object,
    read_jsonl_objects,
    write_bytes_idempotent,
)


P43A_LEDGER_SCHEMA = "p43a.two_phase_paper_workflow_ledger.v1"
P43A_POSTGAME_SUMMARY_SCHEMA = "p43a.postgame_summary.v1"
P43A_POSTGAME_ARTIFACT_FILES = (
    "workflow_ledger.jsonl",
    "postgame_summary.json",
    "report.md",
)
P43A_P42_RECONCILIATION_KEYS = (
    "eligible_universe",
    "bet_count",
    "pass_count",
    "settled_bet_count",
    "wins",
    "losses",
    "pushes",
    "units_risked",
    "net_paper_units",
    "descriptive_roi",
    "feedback_count",
)


def reconstruct_frozen_decision(projection: Mapping[str, Any]) -> PaperMoneylineDecision:
    """Rebuild the frozen P40 decision and reject any payload drift."""

    try:
        reconstructed = PaperMoneylineDecision.create(
            model_role=projection["model_role"],
            model_id=projection["model_id"],
            model_fingerprint=projection["model_fingerprint"],
            p37_fold_id=projection["p37_fold_id"],
            p37_window=projection["p37_window"],
            p37_prediction_row_id=projection["p37_prediction_row_id"],
            provider_namespace=projection["provider_namespace"],
            provider_game_id=projection["provider_game_id"],
            game_pk=projection["game_pk"],
            game_number=projection["game_number"],
            official_date=projection["official_date"],
            scheduled_start_utc=projection["scheduled_start_utc"],
            home_team=projection["home_team"],
            away_team=projection["away_team"],
            home_team_code=projection["home_team_code"],
            away_team_code=projection["away_team_code"],
            market_snapshot_id=projection["market_snapshot_id"],
            market_observed_at_utc=projection["market_observed_at_utc"],
            local_fetched_at_utc=projection["local_fetched_at_utc"],
            source_match_id=projection["source_match_id"],
            market_source_sha256=projection["market_source_sha256"],
            p37_comparisons_sha256=projection["p37_comparisons_sha256"],
            model_probability_source=projection["model_probability_source"],
            p_home=projection["p_home"],
            home_decimal_odds=projection["home_decimal_odds"],
            away_decimal_odds=projection["away_decimal_odds"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED") from exc
    if reconstructed.to_projection() != dict(projection):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    return reconstructed


def verify_p43a_pregame_record(row: Mapping[str, Any]) -> PaperMoneylineDecision:
    """Verify envelope + P40 fingerprints; never regenerate a new decision."""

    payload = {key: value for key, value in row.items() if key != "workflow_decision_id"}
    expected_id = _sha256_projection(payload)
    if row.get("workflow_decision_id") != expected_id:
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    p40_decision = row.get("p40_decision")
    if not isinstance(p40_decision, Mapping):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    decision = reconstruct_frozen_decision(p40_decision)
    if decision.decision_id != row.get("decision_fingerprint"):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    if decision.decision != row.get("bet_or_pass"):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    selected_side = decision.candidate_side if decision.decision == DECISION_BET else "NONE"
    if selected_side != row.get("selected_side"):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    if format(decision.paper_stake_units, "f") != row.get("paper_stake_units"):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    if format(decision.ev_home, "f") != row.get("ev_home") or format(decision.ev_away, "f") != row.get(
        "ev_away"
    ):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    return decision


def load_p43a_frozen_decision_bundle(
    output_dir: str | Path,
) -> tuple[tuple[PaperMoneylineDecision, ...], tuple[dict[str, Any], ...]]:
    path = Path(output_dir) / "pregame_decisions.jsonl"
    if not path.is_file():
        raise RuntimeError("P43A_DECISION_BUNDLE_MISSING")
    rows = read_jsonl_objects(path)
    decisions = tuple(verify_p43a_pregame_record(row) for row in rows)
    if len(decisions) != len({row.decision_id for row in decisions}):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    return decisions, tuple(rows)


def load_p43a_final_result_authority(
    result_input: str | Path,
) -> tuple[P40AOutcomeRow, ...]:
    """Load independent normalized finals after the decision bundle already exists."""

    from .p44a_normalized_workflow_input import (
        load_normalized_result_input,
        project_normalized_results,
    )

    return project_normalized_results(load_normalized_result_input(result_input))


def settle_p43a_frozen_decisions(
    decisions: Sequence[PaperMoneylineDecision],
    outcome_rows: Sequence[P40AOutcomeRow],
) -> tuple[PaperMoneylineSettlement, ...]:
    """Attach results to the existing frozen decisions; do not re-freeze."""

    try:
        return attach_and_settle_frozen_decisions(decisions, outcome_rows)
    except RuntimeError as exc:
        message = str(exc)
        mapping = {
            "P42A_MISSING_RESULT_FAIL_CLOSED_STOP": "P43A_MISSING_RESULT_FAIL_CLOSED",
            "P42A_NON_FINAL_RESULT_FAIL_CLOSED_STOP": "P43A_NON_FINAL_RESULT_FAIL_CLOSED",
            "P42A_DUPLICATE_RESULT_REJECTED_STOP": "P43A_CONFLICTING_RESULT_REJECTED",
            "P42A_OUTCOME_IDENTITY_MISMATCH_STOP": "P43A_CONFLICTING_RESULT_REJECTED",
        }
        for original, renamed in mapping.items():
            if original in message:
                raise RuntimeError(renamed) from exc
        raise


def _workflow_execution_id(
    *,
    authority_hashes: Mapping[str, str],
    decisions: Sequence[PaperMoneylineDecision],
) -> str:
    return _sha256_projection(
        {
            "schema_version": P43A_LEDGER_SCHEMA,
            "workflow_label": P43A_WORKFLOW_LABEL,
            "policy_id": P40A_POLICY_ID,
            "model_role": P40A_CHAMPION_ROLE,
            "authority_hashes": dict(authority_hashes),
            "decision_fingerprints": [row.decision_id for row in decisions],
        }
    )


def build_p43a_ledger_row(
    settlement: PaperMoneylineSettlement,
    *,
    workflow_execution_id: str,
) -> dict[str, Any]:
    row = build_workflow_ledger_row(
        settlement,
        workflow_execution_id=workflow_execution_id,
    )
    return {
        **row,
        "schema_version": P43A_LEDGER_SCHEMA,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "workflow_kind": P43A_WORKFLOW_KIND,
        "human_label": P43A_HUMAN_LABEL,
        "phase": "POSTGAME_SETTLE",
        "pregame_decision_fingerprint": row["decision_fingerprint"],
        "upstream_decision_id_unchanged": True,
    }


def p43a_reconciliation_metrics(
    *,
    decisions: Sequence[PaperMoneylineDecision],
    settlements: Sequence[PaperMoneylineSettlement],
    ledger_rows: Sequence[Mapping[str, Any]],
    champion_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    bets = tuple(row for row in settlements if row.decision.decision == DECISION_BET)
    passes = tuple(row for row in settlements if row.decision.decision == DECISION_PASS)
    wins = tuple(row for row in bets if row.settlement_status == SETTLEMENT_WON)
    losses = tuple(row for row in bets if row.settlement_status == SETTLEMENT_LOST)
    return {
        "eligible_universe": len(decisions),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "settled_bet_count": len(wins) + len(losses),
        "wins": len(wins),
        "losses": len(losses),
        "pushes": int(champion_aggregate["push_count"]),
        "units_risked": champion_aggregate["total_paper_units_risked"],
        "net_paper_units": champion_aggregate["net_paper_units"],
        "descriptive_roi": champion_aggregate["descriptive_paper_roi"],
        "feedback_count": len(ledger_rows),
    }


def expected_p42_reconciliation_metrics(committed_p42_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eligible_universe": committed_p42_summary["workflow_decision_count"],
        "bet_count": committed_p42_summary["bet_count"],
        "pass_count": committed_p42_summary["pass_count"],
        "settled_bet_count": committed_p42_summary["settled_bet_count"],
        "wins": committed_p42_summary["win_count"],
        "losses": committed_p42_summary["loss_count"],
        "pushes": committed_p42_summary["push_count"],
        "units_risked": committed_p42_summary["units_risked"],
        "net_paper_units": committed_p42_summary["net_paper_units"],
        "descriptive_roi": committed_p42_summary["descriptive_historical_paper_roi"],
        "feedback_count": committed_p42_summary["feedback_row_count"],
    }


def reconcile_p43a_with_p42(
    computed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    for key in P43A_P42_RECONCILIATION_KEYS:
        if computed.get(key) != expected.get(key):
            raise RuntimeError(
                "P43A_P42_RECONCILIATION_FAILURE_STOP "
                f"first_divergence={key} p43a={computed.get(key)!r} p42={expected.get(key)!r}"
            )
    packet_expected = {
        "eligible_universe": P42A_EXPECTED_CHAMPION["edge_ready_rows"],
        "bet_count": P42A_EXPECTED_CHAMPION["bet_count"],
        "pass_count": P42A_EXPECTED_CHAMPION["pass_count"],
        "settled_bet_count": P42A_EXPECTED_CHAMPION["win_count"]
        + P42A_EXPECTED_CHAMPION["loss_count"],
        "wins": P42A_EXPECTED_CHAMPION["win_count"],
        "losses": P42A_EXPECTED_CHAMPION["loss_count"],
        "pushes": P42A_EXPECTED_CHAMPION["push_count"],
        "units_risked": P42A_EXPECTED_CHAMPION["total_paper_units_risked"],
        "net_paper_units": P42A_EXPECTED_CHAMPION["net_paper_units"],
        "descriptive_roi": P42A_EXPECTED_CHAMPION["descriptive_paper_roi"],
        "feedback_count": P42A_EXPECTED_CHAMPION["edge_ready_rows"],
    }
    for key in P43A_P42_RECONCILIATION_KEYS:
        if computed.get(key) != packet_expected[key]:
            raise RuntimeError(
                "P43A_P42_RECONCILIATION_FAILURE_STOP "
                f"first_divergence={key} p43a={computed.get(key)!r} p42={packet_expected[key]!r}"
            )
    return {
        "status": "RECONCILED",
        "computed": dict(computed),
        "committed_p42": dict(expected),
    }


def build_p43a_postgame_summary(
    *,
    pregame_summary: Mapping[str, Any],
    settlements: Sequence[PaperMoneylineSettlement],
    ledger_rows: Sequence[Mapping[str, Any]],
    reconciliation: Mapping[str, Any],
    p40_reconciliation: Mapping[str, Any],
    deterministic_rerun_verified: bool,
    authority_hashes: Mapping[str, str],
) -> dict[str, Any]:
    computed = reconciliation["computed"]
    return {
        "schema_version": P43A_POSTGAME_SUMMARY_SCHEMA,
        "task_id": P43A_TASK_ID,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "workflow_kind": P43A_WORKFLOW_KIND,
        "human_label": P43A_HUMAN_LABEL,
        "labels": ["OFFLINE", "HISTORICAL", "TWO_PHASE", "PAPER_REHEARSAL"],
        "phase": "POSTGAME_SETTLE",
        "prospective": False,
        "live": False,
        "production": False,
        "forward_real": False,
        "real_betting_history": False,
        "descriptive_only": True,
        "roi_label": "DESCRIPTIVE_PAPER_ONLY",
        "p37_target_count": pregame_summary["p37_target_count"],
        "p39_edge_ready_count": pregame_summary["p39_edge_ready_count"],
        "workflow_decision_count": computed["eligible_universe"],
        "bet_count": computed["bet_count"],
        "pass_count": computed["pass_count"],
        "settled_bet_count": computed["settled_bet_count"],
        "unresolved_result_count": 0,
        "feedback_row_count": computed["feedback_count"],
        "win_count": computed["wins"],
        "loss_count": computed["losses"],
        "push_count": computed["pushes"],
        "units_risked": computed["units_risked"],
        "net_paper_units": computed["net_paper_units"],
        "descriptive_historical_paper_roi": computed["descriptive_roi"],
        "maximum_drawdown": p40_reconciliation["computed"]["maximum_paper_drawdown"],
        "workflow_completeness": {
            "frozen_decisions": len(settlements),
            "settlements": len(settlements),
            "evaluations": len(ledger_rows),
            "feedback_rows": len(ledger_rows),
            "one_to_one_lineage": len({row["decision_fingerprint"] for row in ledger_rows})
            == len(ledger_rows)
            == len(settlements),
        },
        "p39_no_market_count": pregame_summary["p39_no_market_count"],
        "exclusion_reasons": dict(pregame_summary["exclusion_reasons"]),
        "p40_reconciliation": p40_reconciliation,
        "p42_reconciliation": reconciliation,
        "pregame_bundle_fingerprint": pregame_summary["bundle_fingerprint"],
        "pregame_freeze_status": pregame_summary["freeze_status"],
        "deterministic_rerun_verified": deterministic_rerun_verified,
        "network_required": False,
        "protected_authority_hashes": dict(authority_hashes),
        "decision_rule": {
            "policy_id": P40A_POLICY_ID,
            "consumed_not_rederived": True,
            "additional_threshold": "NONE",
        },
        "claims": dict(P43A_CLAIMS),
    }


def render_p43a_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# P43A Two-Phase Paper Workflow",
            "",
            f"This artifact is an **{P43A_HUMAN_LABEL}**.",
            "It is not prospective, live, production, forward-real, or real betting history.",
            "Historical rehearsal rows are not new prospective bets.",
            "",
            "## Phase 1 — Pregame freeze",
            "",
            "- Frozen P37 Champion probability (prediction columns only).",
            "- Frozen P39 trusted pregame Moneyline market.",
            "- Unchanged P40 zero-EV BET/PASS rule, consumed not re-derived.",
            "- Immutable decision bundle written; no settlement.",
            "",
            "## Phase 2 — Postgame settlement",
            "",
            "- Existing frozen decision bundle verified by fingerprint.",
            "- Authoritative HOME/AWAY final attached after freeze.",
            "- Existing P40/P42 settlement, evaluation, and feedback lineage.",
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
            f"- P42 reconciliation: `{summary['p42_reconciliation']['status']}`.",
            f"- Deterministic rerun: `{summary['deterministic_rerun_verified']}`.",
            "",
            "## Safety boundary",
            "",
            "- Historical two-phase paper rehearsal only. Not a profitability or promotion claim.",
            "- Real betting: `NOT RUN`.",
            "- Threshold / staking / Kelly: `NOT RUN`.",
            "- Live TSL / MLB / Fortinet / P35A: `NOT RUN`.",
            "",
        ]
    )


@dataclass(frozen=True, slots=True)
class P43APostgameResult:
    decisions: tuple[PaperMoneylineDecision, ...]
    settlements: tuple[PaperMoneylineSettlement, ...]
    ledger_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    authority_hashes_before: dict[str, str]
    authority_hashes_after: dict[str, str]
    write_status: dict[str, str]


def run_p43a_postgame_settle(
    repository_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    result_input: str | Path | None = None,
    persist: bool = True,
    outcome_rows: Sequence[P40AOutcomeRow] | None = None,
) -> P43APostgameResult:
    """Settle an already-frozen bundle. Never recomputes Phase 1 fields."""

    root = Path(repository_root).resolve()
    directory = Path(output_dir or (root / P43A_REPORT_RELATIVE_PATH))
    decisions, frozen_rows = load_p43a_frozen_decision_bundle(directory)
    pregame_summary = read_json_object(directory / "pregame_summary.json")
    hashes_before = dict(pregame_summary.get("protected_authority_hashes") or {})
    if pregame_summary.get("bundle_fingerprint") != _sha256_projection(
        [row["decision_fingerprint"] for row in frozen_rows]
    ):
        raise RuntimeError("P43A_DECISION_BUNDLE_TAMPERED")
    if outcome_rows is not None:
        outcomes = tuple(outcome_rows)
    elif result_input is not None:
        outcomes = load_p43a_final_result_authority(result_input)
    else:
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")
    first_settlements = settle_p43a_frozen_decisions(decisions, outcomes)
    second_settlements = settle_p43a_frozen_decisions(decisions, outcomes)
    if tuple(row.to_projection() for row in first_settlements) != tuple(
        row.to_projection() for row in second_settlements
    ):
        raise RuntimeError("P43A deterministic postgame replay differed")
    if [row.decision.decision_id for row in first_settlements] != [
        row.decision_id for row in decisions
    ]:
        raise RuntimeError("P43A Phase 2 changed an upstream decision identity")
    committed_p40_summary = _read_json(root / "report/p40a_moneyline_paper_bet_pass/summary.json")
    p40_reconciliation = reconcile_with_p40_champion(first_settlements, committed_p40_summary)
    workflow_execution_id = _workflow_execution_id(
        authority_hashes=hashes_before,
        decisions=decisions,
    )
    ledger_rows = tuple(
        build_p43a_ledger_row(row, workflow_execution_id=workflow_execution_id)
        for row in first_settlements
    )
    computed = p43a_reconciliation_metrics(
        decisions=decisions,
        settlements=first_settlements,
        ledger_rows=ledger_rows,
        champion_aggregate=p40_reconciliation["computed"],
    )
    committed_p42 = _read_json(root / P42A_REPORT_RELATIVE_PATH / "summary.json")
    reconciliation = reconcile_p43a_with_p42(
        computed,
        expected_p42_reconciliation_metrics(committed_p42),
    )
    hashes_after = dict(hashes_before)
    summary = build_p43a_postgame_summary(
        pregame_summary=pregame_summary,
        settlements=first_settlements,
        ledger_rows=ledger_rows,
        reconciliation=reconciliation,
        p40_reconciliation=p40_reconciliation,
        deterministic_rerun_verified=True,
        authority_hashes=hashes_before,
    )
    write_status: dict[str, str] = {}
    if persist:
        write_status = write_p43a_postgame_artifacts(
            directory,
            ledger_rows=ledger_rows,
            summary=summary,
        )
    return P43APostgameResult(
        decisions=decisions,
        settlements=first_settlements,
        ledger_rows=ledger_rows,
        summary=summary,
        authority_hashes_before=hashes_before,
        authority_hashes_after=hashes_after,
        write_status=write_status,
    )


def write_p43a_postgame_artifacts(
    output_dir: str | Path,
    *,
    ledger_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = {
        "workflow_ledger.jsonl": _jsonl_bytes(ledger_rows),
        "postgame_summary.json": _json_bytes(summary),
        "report.md": render_p43a_report(summary).encode("utf-8"),
    }
    return {name: write_bytes_idempotent(directory / name, content) for name, content in rendered.items()}


__all__ = (
    "P43A_POSTGAME_ARTIFACT_FILES",
    "P43APostgameResult",
    "expected_p42_reconciliation_metrics",
    "load_p43a_final_result_authority",
    "load_p43a_frozen_decision_bundle",
    "reconcile_p43a_with_p42",
    "reconstruct_frozen_decision",
    "run_p43a_postgame_settle",
    "settle_p43a_frozen_decisions",
    "verify_p43a_pregame_record",
    "write_p43a_postgame_artifacts",
)
