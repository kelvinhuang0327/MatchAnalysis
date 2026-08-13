"""P41A leakage-safe chronological EV-margin policy evaluation.

This use case reuses the immutable P40A Champion decisions and settlement
arithmetic.  A threshold is selected from strictly prior edge-ready rows,
frozen, and only then applied to the next target window.  Target outcomes are
attached in a later phase and are never read by threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
    PaperMoneylineDecision,
    PaperMoneylineSettlement,
    aggregate_paper_settlements,
    settle_paper_moneyline_decision,
)
from .p40a_moneyline_paper_bet_pass import (
    P40A_ARTIFACT_FILES,
    P40A_CHAMPION_ROLE,
    P40A_REPORT_RELATIVE_PATH,
    P40AOutcomeRow,
    P40AResult,
    render_p40a_artifacts,
    run_p40a_moneyline_paper_bet_pass,
)


P41A_TASK_ID = "P41A"
P41A_POLICY_ID = "P41A_WALK_FORWARD_EV_MARGIN_V1"
P41A_CHAMPION_ROLE = P40A_CHAMPION_ROLE
P41A_REPORT_RELATIVE_PATH = Path("report/p41a_walk_forward_ev_margin_policy")
P41A_ARTIFACT_FILES = (
    "source_manifest.json",
    "policy_evaluations.jsonl",
    "summary.json",
    "report.md",
)
P41A_CANDIDATE_THRESHOLDS = (
    Decimal("0.00"),
    Decimal("0.01"),
    Decimal("0.02"),
    Decimal("0.03"),
    Decimal("0.05"),
)
P41A_ZERO_EV_THRESHOLD = P41A_CANDIDATE_THRESHOLDS[0]
P41A_MIN_VALID_TARGET_WINDOWS = 2
P41A_OUTCOME_AUTHORITY = "P37A_COMMITTED_TRUE_OOS_COMPARISON_ACTUAL_WINNER"
P41A_WINDOW_SCHEMA_VERSION = "p41a.walk_forward_ev_margin_policy_window.v1"
P41A_SUMMARY_SCHEMA_VERSION = "p41a.walk_forward_ev_margin_policy_summary.v1"
P41A_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "p41a.walk_forward_ev_margin_policy_source_manifest.v1"
)
_ZERO_DECISION_ID = "0" * 64


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


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _decision_sort_key(decision: PaperMoneylineDecision) -> tuple[str, int, str]:
    return (
        decision.scheduled_start_utc,
        decision.game_number,
        decision.provider_game_id,
    )


def _metric_projection(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Project the shared P40A aggregate into the P41A report vocabulary."""

    return {
        "row_count": aggregate["row_count"],
        "edge_ready_rows": aggregate["edge_ready_rows"],
        "bet_count": aggregate["bet_count"],
        "pass_count": aggregate["pass_count"],
        "win_count": aggregate["win_count"],
        "loss_count": aggregate["loss_count"],
        "push_count": aggregate["push_count"],
        "total_paper_units_risked": aggregate["total_paper_units_risked"],
        "net_paper_units": aggregate["net_paper_units"],
        "descriptive_paper_roi": aggregate["descriptive_paper_roi"],
        "maximum_paper_drawdown": aggregate["maximum_paper_drawdown"],
    }


@dataclass(frozen=True, slots=True)
class P41AThresholdSelection:
    """Candidate metrics and the threshold selected from prior outcomes."""

    prior_threshold_metrics: tuple[dict[str, Any], ...]
    selected_threshold: Decimal
    tie_break_reason: str


@dataclass(frozen=True, slots=True)
class P41AWindowEvaluation:
    """One frozen-threshold true-OOS target-window record."""

    target_window: str
    target_fold_id: str
    target_window_order: int
    prior_policy_training_windows: tuple[str, ...]
    prior_eligible_row_count: int
    prior_threshold_metrics: tuple[dict[str, Any], ...]
    selected_threshold: Decimal
    tie_break_reason: str
    target_prediction_row_ids: tuple[str, ...]
    selected_policy_target_metrics: dict[str, Any]
    zero_ev_baseline_target_metrics: dict[str, Any]

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": P41A_WINDOW_SCHEMA_VERSION,
            "target_window": self.target_window,
            "target_fold_id": self.target_fold_id,
            "target_window_order": self.target_window_order,
            "prior_policy_training_windows": list(self.prior_policy_training_windows),
            "prior_eligible_row_count": self.prior_eligible_row_count,
            "candidate_thresholds": [
                _decimal_text(threshold) for threshold in P41A_CANDIDATE_THRESHOLDS
            ],
            "prior_threshold_metrics": list(self.prior_threshold_metrics),
            "selected_threshold": _decimal_text(self.selected_threshold),
            "tie_break_reason": self.tie_break_reason,
            "target_prediction_row_ids": list(self.target_prediction_row_ids),
            "target_row_count": len(self.target_prediction_row_ids),
            "selected_policy_target_metrics": self.selected_policy_target_metrics,
            "zero_ev_baseline_target_metrics": self.zero_ev_baseline_target_metrics,
            "identical_target_universe_verified": True,
            "target_outcomes_attached_after_threshold_freeze": True,
        }


@dataclass(frozen=True, slots=True)
class P41AAuthority:
    """Immutable P40A authority and the outcome-free Champion decision rows."""

    repository_root: Path
    p40a_result: P40AResult
    champion_decisions: tuple[PaperMoneylineDecision, ...]
    outcome_rows: tuple[P40AOutcomeRow, ...]
    p40a_artifact_hashes: dict[str, str]
    source_manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class P41AResult:
    authority: P41AAuthority
    window_evaluations: tuple[P41AWindowEvaluation, ...]
    summary: dict[str, Any]


def _validate_threshold(threshold: Decimal) -> Decimal:
    if not isinstance(threshold, Decimal):
        raise TypeError("threshold must be a Decimal")
    if not threshold.is_finite() or threshold < Decimal("0"):
        raise ValueError("threshold must be a finite non-negative Decimal")
    return threshold


def _outcome_map(
    outcome_rows: Mapping[str, P40AOutcomeRow] | Sequence[P40AOutcomeRow],
) -> dict[str, P40AOutcomeRow]:
    if isinstance(outcome_rows, Mapping):
        result = dict(outcome_rows)
    else:
        result = {}
        for row in outcome_rows:
            if row.p37_prediction_row_id in result:
                raise ValueError("duplicate P41A outcome authority row")
            result[row.p37_prediction_row_id] = row
    if not result:
        raise ValueError("P41A outcome authority must not be empty")
    return result


def _outcomes_for_rows(
    rows: Sequence[PaperMoneylineDecision],
    outcome_rows: Mapping[str, P40AOutcomeRow] | Sequence[P40AOutcomeRow],
) -> dict[str, P40AOutcomeRow]:
    """Project outcome authority to exactly the rows being settled."""

    available = _outcome_map(outcome_rows)
    projected: dict[str, P40AOutcomeRow] = {}
    for row in rows:
        outcome = available.get(row.p37_prediction_row_id)
        if outcome is None or outcome.provider_game_id != row.provider_game_id:
            raise RuntimeError("P41A_OUTCOME_AUTHORITY_UNRESOLVED_STOP")
        projected[row.p37_prediction_row_id] = outcome
    return projected


def _threshold_decision(
    decision: PaperMoneylineDecision,
    threshold: Decimal,
) -> PaperMoneylineDecision:
    """Apply a frozen policy threshold while preserving P40A EV arithmetic."""

    _validate_threshold(threshold)
    should_bet = decision.candidate_ev > threshold
    return replace(
        decision,
        decision_id=_ZERO_DECISION_ID,
        decision=DECISION_BET if should_bet else DECISION_PASS,
        paper_stake_units=Decimal("1.0") if should_bet else Decimal("0"),
    )


def _settle_threshold_rows(
    rows: Sequence[PaperMoneylineDecision],
    threshold: Decimal,
    outcome_rows: Mapping[str, P40AOutcomeRow] | Sequence[P40AOutcomeRow],
) -> tuple[PaperMoneylineSettlement, ...]:
    """Attach outcomes to already-frozen threshold decisions."""

    if not rows:
        raise ValueError("cannot settle an empty P41A policy cohort")
    outcome_by_id = _outcome_map(outcome_rows)
    settlements: list[PaperMoneylineSettlement] = []
    for base_decision in sorted(rows, key=_decision_sort_key):
        outcome = outcome_by_id.get(base_decision.p37_prediction_row_id)
        if outcome is None or outcome.provider_game_id != base_decision.provider_game_id:
            raise RuntimeError("P41A_OUTCOME_AUTHORITY_UNRESOLVED_STOP")
        threshold_decision = _threshold_decision(base_decision, threshold)
        settlements.append(
            settle_paper_moneyline_decision(
                threshold_decision,
                final_game_outcome=outcome.actual_winner,
                target_home_win=outcome.target_home_win,
                outcome_authority_row_id=outcome.p37_prediction_row_id,
                outcome_authority=P41A_OUTCOME_AUTHORITY,
            )
        )
    return tuple(settlements)


def select_threshold_from_prior_rows(
    prior_rows: Sequence[PaperMoneylineDecision],
    outcome_rows: Mapping[str, P40AOutcomeRow] | Sequence[P40AOutcomeRow],
) -> P41AThresholdSelection:
    """Select T using only the supplied prior rows and their outcomes.

    Target rows are intentionally not accepted by this function.  The caller
    must pass only completed prior-window rows, so target outcomes cannot
    influence candidate scoring or selection.
    """

    if not prior_rows:
        raise ValueError("P41A threshold selection requires prior eligible rows")
    if any(row.model_role != P40A_CHAMPION_ROLE for row in prior_rows):
        raise ValueError("P41A threshold selection requires Champion rows")
    unique_ids = {row.p37_prediction_row_id for row in prior_rows}
    if len(unique_ids) != len(prior_rows):
        raise ValueError("P41A prior rows must have unique prediction identities")

    metrics: list[dict[str, Any]] = []
    for threshold in P41A_CANDIDATE_THRESHOLDS:
        settlements = _settle_threshold_rows(prior_rows, threshold, outcome_rows)
        aggregate = aggregate_paper_settlements(
            settlements,
            edge_ready_rows=len(prior_rows),
            model_role=P40A_CHAMPION_ROLE,
        )
        metrics.append(
            {
                "threshold": _decimal_text(threshold),
                **_metric_projection(aggregate),
            }
        )

    best_net = max(Decimal(row["net_paper_units"]) for row in metrics)
    tied_thresholds = tuple(
        Decimal(row["threshold"])
        for row in metrics
        if Decimal(row["net_paper_units"]) == best_net
    )
    selected_threshold = max(tied_thresholds)
    tie_break_reason = (
        "LARGER_EV_THRESHOLD_ON_EQUAL_PRIOR_NET_UNITS"
        if len(tied_thresholds) > 1
        else "HIGHEST_PRIOR_CUMULATIVE_NET_PAPER_UNITS"
    )
    return P41AThresholdSelection(
        prior_threshold_metrics=tuple(metrics),
        selected_threshold=selected_threshold,
        tie_break_reason=tie_break_reason,
    )


def _verify_p40a_artifacts(
    repository_root: Path,
    p40a_result: P40AResult,
) -> dict[str, str]:
    """Verify the committed P40A files against the deterministic P40A render."""

    rendered = render_p40a_artifacts(p40a_result)
    report = repository_root / P40A_REPORT_RELATIVE_PATH
    hashes: dict[str, str] = {}
    for name in P40A_ARTIFACT_FILES:
        path = report / name
        actual = path.read_bytes()
        expected = rendered.get(name)
        if expected is None or actual != expected:
            raise RuntimeError(f"P41A_P40A_ARTIFACT_DRIFT: {path}")
        hashes[name] = _sha256_bytes(actual)
    return hashes


def _build_source_manifest(
    repository_root: Path,
    p40a_result: P40AResult,
    p40a_artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    p40a_source = p40a_result.authority.source_manifest
    return {
        "schema_version": P41A_SOURCE_MANIFEST_SCHEMA_VERSION,
        "task_id": P41A_TASK_ID,
        "p40a": {
            "report_path": str(P40A_REPORT_RELATIVE_PATH),
            "artifact_sha256": dict(p40a_artifact_hashes),
            "p37_comparisons_sha256": p40a_source["p37a"]["p37_comparisons_sha256"],
            "p37_summary_sha256": p40a_source["p37a"]["p37_summary_sha256"],
            "p39_market_join_sha256": p40a_source["p39a"]["p39_market_join_sha256"],
            "p39_summary_sha256": p40a_source["p39a"]["p39_summary_sha256"],
            "p39_source_manifest_sha256": p40a_source["p39a"][
                "p39_source_manifest_sha256"
            ],
            "p38_summary_sha256": p40a_source["p38a"]["p38_summary_sha256"],
            "p38_comparisons_sha256": p40a_source["p38a"]["p38_comparisons_sha256"],
        },
        "repository_root": str(repository_root),
        "champion_policy": {
            "model_role": P40A_CHAMPION_ROLE,
            "model_id": p40a_source["p37a"]["champion_model_id"],
            "model_fingerprint": p40a_source["p37a"]["champion_model_fingerprint"],
            "policy_id": P41A_POLICY_ID,
        },
        "candidate_thresholds": [
            _decimal_text(threshold) for threshold in P41A_CANDIDATE_THRESHOLDS
        ],
        "selection_objective": "HIGHEST_CUMULATIVE_PRIOR_NET_PAPER_UNITS",
        "selection_tie_break": "LARGER_EV_THRESHOLD",
        "target_outcome_isolation": (
            "TARGET_OUTCOMES_ATTACHED_ONLY_AFTER_SELECTED_THRESHOLD_IS_FROZEN"
        ),
        "settlement_rule": p40a_source["settlement_rule"],
        "claims": {
            "real_betting": False,
            "staking_optimization": False,
            "kelly": False,
            "bankroll_management": False,
            "model_promotion": False,
            "training": False,
            "calibration": False,
            "external_market_acquisition": False,
            "expected_future_roi": False,
        },
    }


def load_p41a_authority(repository_root: str | Path) -> P41AAuthority:
    """Load P40A and verify that its committed artifacts remain unchanged."""

    root = Path(repository_root).resolve()
    p40a_result = run_p40a_moneyline_paper_bet_pass(root)
    p40a_artifact_hashes = _verify_p40a_artifacts(root, p40a_result)
    champion_decisions = tuple(
        sorted(
            (
                decision
                for decision in p40a_result.decisions
                if decision.model_role == P40A_CHAMPION_ROLE
            ),
            key=_decision_sort_key,
        )
    )
    if len(champion_decisions) != 62:
        raise ValueError("P41A requires exactly 62 committed P40A Champion rows")
    return P41AAuthority(
        repository_root=root,
        p40a_result=p40a_result,
        champion_decisions=champion_decisions,
        outcome_rows=p40a_result.authority.outcome_rows,
        p40a_artifact_hashes=p40a_artifact_hashes,
        source_manifest=_build_source_manifest(
            root,
            p40a_result,
            p40a_artifact_hashes,
        ),
    )


def _chronological_windows(authority: P41AAuthority) -> tuple[dict[str, Any], ...]:
    raw_windows = authority.p40a_result.authority.p37_summary.get("evaluation_windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("P41A P37 chronology authority is missing")
    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    for raw in raw_windows:
        if not isinstance(raw, Mapping):
            raise ValueError("P41A P37 chronology entry is invalid")
        window_id = raw.get("evaluation_window_id")
        fold_id = raw.get("holdout_fold_id")
        order = raw.get("evaluation_window_order")
        if (
            not isinstance(window_id, str)
            or not window_id
            or not isinstance(fold_id, str)
            or not fold_id
            or isinstance(order, bool)
            or not isinstance(order, int)
        ):
            raise ValueError("P41A P37 chronology fields are invalid")
        if window_id in seen_ids or order in seen_orders:
            raise ValueError("P41A P37 chronology is not unique")
        seen_ids.add(window_id)
        seen_orders.add(order)
        windows.append(
            {
                "window_id": window_id,
                "fold_id": fold_id,
                "order": order,
            }
        )
    windows.sort(key=lambda row: row["order"])
    if [row["order"] for row in windows] != sorted(seen_orders):
        raise ValueError("P41A P37 chronology could not be sorted")
    observed_windows = {row.p37_window for row in authority.champion_decisions}
    if not observed_windows.issubset(seen_ids):
        raise ValueError("P41A Champion rows contain an unknown chronology window")
    return tuple(windows)


def _aggregate_metrics(
    settlements: Sequence[PaperMoneylineSettlement],
    *,
    edge_ready_rows: int,
) -> dict[str, Any]:
    return _metric_projection(
        aggregate_paper_settlements(
            settlements,
            edge_ready_rows=edge_ready_rows,
            model_role=P40A_CHAMPION_ROLE,
        )
    )


def _conclusion(
    selected_policy: Mapping[str, Any],
    zero_ev_baseline: Mapping[str, Any],
) -> str:
    selected_net = Decimal(selected_policy["net_paper_units"])
    baseline_net = Decimal(zero_ev_baseline["net_paper_units"])
    selected_drawdown = Decimal(selected_policy["maximum_paper_drawdown"])
    baseline_drawdown = Decimal(zero_ev_baseline["maximum_paper_drawdown"])
    if selected_net > baseline_net and selected_drawdown <= baseline_drawdown:
        return "EV_MARGIN_POLICY_IMPROVED"
    if baseline_net >= selected_net and baseline_drawdown <= selected_drawdown:
        return "ZERO_EV_BASELINE_RETAINS"
    return "MIXED_OR_INCONCLUSIVE"


def evaluate_p41a_authority(authority: P41AAuthority) -> P41AResult:
    """Run one P41A evaluation pass against an already-loaded authority."""

    windows = _chronological_windows(authority)
    decisions_by_window: dict[str, tuple[PaperMoneylineDecision, ...]] = {}
    for window in windows:
        rows = tuple(
            sorted(
                (
                    decision
                    for decision in authority.champion_decisions
                    if decision.p37_window == window["window_id"]
                ),
                key=_decision_sort_key,
            )
        )
        if rows:
            decisions_by_window[window["window_id"]] = rows
    outcome_by_id = _outcome_map(authority.outcome_rows)

    selected_specs: list[
        tuple[dict[str, Any], tuple[str, ...], tuple[PaperMoneylineDecision, ...], P41AThresholdSelection]
    ] = []
    for window in windows:
        target_rows = decisions_by_window.get(window["window_id"], ())
        if not target_rows:
            continue
        prior_windows = tuple(
            prior["window_id"]
            for prior in windows
            if prior["order"] < window["order"]
            and decisions_by_window.get(prior["window_id"])
        )
        prior_rows = tuple(
            row
            for prior_window in prior_windows
            for row in decisions_by_window[prior_window]
        )
        if not prior_rows:
            continue
        selection = select_threshold_from_prior_rows(
            prior_rows,
            _outcomes_for_rows(prior_rows, outcome_by_id),
        )
        selected_specs.append((window, prior_windows, target_rows, selection))

    if len(selected_specs) < P41A_MIN_VALID_TARGET_WINDOWS:
        raise RuntimeError("P41A_INSUFFICIENT_POLICY_OOS_WINDOWS_STOP")

    window_evaluations: list[P41AWindowEvaluation] = []
    selected_settlements: list[PaperMoneylineSettlement] = []
    zero_ev_settlements: list[PaperMoneylineSettlement] = []
    threshold_counts = {
        _decimal_text(threshold): 0 for threshold in P41A_CANDIDATE_THRESHOLDS
    }

    # All selected thresholds are frozen before any target-window settlement.
    for window, prior_windows, target_rows, selection in selected_specs:
        threshold_counts[_decimal_text(selection.selected_threshold)] += 1

    # Only after the preceding freeze are target outcomes attached.
    for window, prior_windows, target_rows, selection in selected_specs:
        selected_window_settlements = _settle_threshold_rows(
            target_rows,
            selection.selected_threshold,
            outcome_by_id,
        )
        zero_ev_window_settlements = _settle_threshold_rows(
            target_rows,
            P41A_ZERO_EV_THRESHOLD,
            outcome_by_id,
        )
        selected_ids = {
            row.decision.p37_prediction_row_id for row in selected_window_settlements
        }
        baseline_ids = {
            row.decision.p37_prediction_row_id for row in zero_ev_window_settlements
        }
        if selected_ids != baseline_ids:
            raise RuntimeError("P41A_IDENTICAL_TARGET_UNIVERSE_STOP")
        selected_settlements.extend(selected_window_settlements)
        zero_ev_settlements.extend(zero_ev_window_settlements)
        window_evaluations.append(
            P41AWindowEvaluation(
                target_window=window["window_id"],
                target_fold_id=window["fold_id"],
                target_window_order=window["order"],
                prior_policy_training_windows=prior_windows,
                prior_eligible_row_count=sum(
                    len(decisions_by_window[prior_window])
                    for prior_window in prior_windows
                ),
                prior_threshold_metrics=selection.prior_threshold_metrics,
                selected_threshold=selection.selected_threshold,
                tie_break_reason=selection.tie_break_reason,
                target_prediction_row_ids=tuple(
                    row.p37_prediction_row_id for row in target_rows
                ),
                selected_policy_target_metrics=_aggregate_metrics(
                    selected_window_settlements,
                    edge_ready_rows=len(target_rows),
                ),
                zero_ev_baseline_target_metrics=_aggregate_metrics(
                    zero_ev_window_settlements,
                    edge_ready_rows=len(target_rows),
                ),
            )
        )

    total_target_rows = sum(
        len(evaluation.target_prediction_row_ids) for evaluation in window_evaluations
    )
    selected_policy = _aggregate_metrics(
        selected_settlements,
        edge_ready_rows=total_target_rows,
    )
    zero_ev_baseline = _aggregate_metrics(
        zero_ev_settlements,
        edge_ready_rows=total_target_rows,
    )
    if {
        row.decision.p37_prediction_row_id for row in selected_settlements
    } != {
        row.decision.p37_prediction_row_id for row in zero_ev_settlements
    }:
        raise RuntimeError("P41A_IDENTICAL_TARGET_UNIVERSE_STOP")

    summary = {
        "schema_version": P41A_SUMMARY_SCHEMA_VERSION,
        "task_id": P41A_TASK_ID,
        "candidate_thresholds": [
            _decimal_text(threshold) for threshold in P41A_CANDIDATE_THRESHOLDS
        ],
        "policy_oos_target_rows": total_target_rows,
        "total_target_windows": len(window_evaluations),
        "target_windows": [
            evaluation.target_window for evaluation in window_evaluations
        ],
        "selected_threshold_per_target_window": {
            evaluation.target_window: _decimal_text(evaluation.selected_threshold)
            for evaluation in window_evaluations
        },
        "threshold_selection_counts": threshold_counts,
        "selected_policy": selected_policy,
        "zero_ev_baseline": zero_ev_baseline,
        "conclusion": _conclusion(selected_policy, zero_ev_baseline),
        "conclusion_rule": (
            "EV_MARGIN_POLICY_IMPROVED iff selected net units > zero-EV net units "
            "and selected maximum drawdown <= zero-EV maximum drawdown; "
            "ZERO_EV_BASELINE_RETAINS iff zero-EV net units >= selected net units "
            "and zero-EV maximum drawdown <= selected maximum drawdown; otherwise "
            "MIXED_OR_INCONCLUSIVE."
        ),
        "selection_objective": "HIGHEST_CUMULATIVE_PRIOR_NET_PAPER_UNITS",
        "selection_tie_break": "LARGER_EV_THRESHOLD",
        "skipped_windows_without_prior_policy_training": [
            window["window_id"]
            for window in windows
            if window["window_id"] in decisions_by_window
            and window["window_id"]
            not in {evaluation.target_window for evaluation in window_evaluations}
        ],
        "true_oos_verified": True,
        "target_outcome_isolation_verified": True,
        "identical_target_universe_verified": True,
        "p37_p38_p39_p40_inputs_read_only": True,
        "deterministic_rerun_verified": False,
        "claims": authority.source_manifest["claims"],
    }
    return P41AResult(
        authority=authority,
        window_evaluations=tuple(window_evaluations),
        summary=summary,
    )


def run_p41a_walk_forward_ev_margin_policy(
    repository_root: str | Path,
) -> P41AResult:
    """Run P41A twice in memory and return the deterministic result."""

    authority = load_p41a_authority(repository_root)
    first = evaluate_p41a_authority(authority)
    second = evaluate_p41a_authority(authority)
    if tuple(
        evaluation.to_projection() for evaluation in first.window_evaluations
    ) != tuple(evaluation.to_projection() for evaluation in second.window_evaluations):
        raise RuntimeError("P41A deterministic window evaluation rerun differed")
    first_summary = {
        key: value
        for key, value in first.summary.items()
        if key != "deterministic_rerun_verified"
    }
    second_summary = {
        key: value
        for key, value in second.summary.items()
        if key != "deterministic_rerun_verified"
    }
    if first_summary != second_summary:
        raise RuntimeError("P41A deterministic summary rerun differed")
    summary = dict(first.summary)
    summary["deterministic_rerun_verified"] = True
    return P41AResult(
        authority=authority,
        window_evaluations=first.window_evaluations,
        summary=summary,
    )


def _metric_row(label: str, metrics: Mapping[str, Any]) -> str:
    return "| " + " | ".join(
        (
            label,
            str(metrics["bet_count"]),
            str(metrics["pass_count"]),
            str(metrics["win_count"]),
            str(metrics["loss_count"]),
            str(metrics["push_count"]),
            str(metrics["total_paper_units_risked"]),
            str(metrics["net_paper_units"]),
            str(metrics["descriptive_paper_roi"]),
            str(metrics["maximum_paper_drawdown"]),
        )
    ) + " |"


def render_p41a_report(result: P41AResult) -> str:
    summary = result.summary
    lines = [
        "# P41A Leakage-Safe Walk-Forward EV-Margin Policy",
        "",
        "This is a deterministic historical true-OOS paper-only comparison. It is",
        "not a real betting recommendation, profitability claim, staking strategy,",
        "model-promotion decision, or future-performance claim.",
        "",
        "## Frozen candidate thresholds",
        "",
        "- Candidate thresholds: `0.00`, `0.01`, `0.02`, `0.03`, `0.05`.",
        "- Selection objective: highest cumulative prior net paper units.",
        "- Tie-break: larger EV threshold.",
        "- Target outcomes are attached only after each target threshold is frozen.",
        "",
        "## Chronological target windows",
        "",
        "| Target window | Prior policy windows | Prior rows | Selected T | Tie-break reason | Target rows |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for evaluation in result.window_evaluations:
        lines.append(
            "| `{window}` | `{prior}` | {prior_rows} | `{threshold}` | `{reason}` | {target_rows} |".format(
                window=evaluation.target_window,
                prior=", ".join(evaluation.prior_policy_training_windows),
                prior_rows=evaluation.prior_eligible_row_count,
                threshold=_decimal_text(evaluation.selected_threshold),
                reason=evaluation.tie_break_reason,
                target_rows=len(evaluation.target_prediction_row_ids),
            )
        )
    for evaluation in result.window_evaluations:
        lines.extend(
            [
                "",
                f"### `{evaluation.target_window}` prior candidate metrics",
                "",
                "| Threshold | BET | PASS | Wins | Losses | Units risked | Net units | ROI | Max drawdown |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in evaluation.prior_threshold_metrics:
            lines.append(
                "| `{threshold}` | {bet} | {pass_count} | {wins} | {losses} | {risk} | {net} | {roi} | {drawdown} |".format(
                    threshold=candidate["threshold"],
                    bet=candidate["bet_count"],
                    pass_count=candidate["pass_count"],
                    wins=candidate["win_count"],
                    losses=candidate["loss_count"],
                    risk=candidate["total_paper_units_risked"],
                    net=candidate["net_paper_units"],
                    roi=candidate["descriptive_paper_roi"],
                    drawdown=candidate["maximum_paper_drawdown"],
                )
            )
        lines.extend(
            [
                "",
                "| Target policy | BET | PASS | Wins | Losses | Pushes | Units risked | Net units | ROI | Max drawdown |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                _metric_row(
                    f"Selected T={_decimal_text(evaluation.selected_threshold)}",
                    evaluation.selected_policy_target_metrics,
                ),
                _metric_row("Fixed zero-EV T=0.00", evaluation.zero_ev_baseline_target_metrics),
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregate true-OOS target comparison",
            "",
            "| Policy | BET | PASS | Wins | Losses | Pushes | Units risked | Net units | ROI | Max drawdown |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _metric_row("Walk-forward selected margin", summary["selected_policy"]),
            _metric_row("Fixed zero-EV baseline", summary["zero_ev_baseline"]),
            "",
            f"- Target windows: `{summary['total_target_windows']}`; target rows: `{summary['policy_oos_target_rows']}`.",
            f"- Selected threshold counts: `{summary['threshold_selection_counts']}`.",
            "",
            "## Conclusion",
            "",
            f"- `CONCLUSION: {summary['conclusion']}`.",
            "",
            "## Safety boundary",
            "",
            "- P37/P38/P39/P40 authority is read-only and remains unchanged.",
            "- Champion probabilities and model authority were not changed.",
            "- Real betting: `NOT RUN`.",
            "- Staking/Kelly and bankroll optimization: `NOT RUN`.",
            "- Model promotion, retraining, calibration, and deployment: `NOT RUN`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_p41a_artifacts(result: P41AResult) -> dict[str, bytes]:
    window_rows = [evaluation.to_projection() for evaluation in result.window_evaluations]
    return {
        "source_manifest.json": _json_bytes(result.authority.source_manifest),
        "policy_evaluations.jsonl": _jsonl_bytes(window_rows),
        "summary.json": _json_bytes(result.summary),
        "report.md": render_p41a_report(result).encode("utf-8"),
    }


def write_p41a_artifacts(output_dir: str | Path, result: P41AResult) -> dict[str, str]:
    """Write only the repository-native P41A artifact set."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_p41a_artifacts(result)
    hashes: dict[str, str] = {}
    for name in P41A_ARTIFACT_FILES:
        content = rendered[name]
        path = directory / name
        path.write_bytes(content)
        hashes[name] = _sha256_bytes(content)
    return hashes


__all__ = (
    "P41A_ARTIFACT_FILES",
    "P41A_CANDIDATE_THRESHOLDS",
    "P41A_CHAMPION_ROLE",
    "P41A_POLICY_ID",
    "P41A_REPORT_RELATIVE_PATH",
    "P41AResult",
    "P41AAuthority",
    "P41AThresholdSelection",
    "P41AWindowEvaluation",
    "evaluate_p41a_authority",
    "load_p41a_authority",
    "render_p41a_artifacts",
    "render_p41a_report",
    "run_p41a_walk_forward_ev_margin_policy",
    "select_threshold_from_prior_rows",
    "write_p41a_artifacts",
)
