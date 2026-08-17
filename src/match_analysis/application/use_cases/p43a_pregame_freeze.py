"""P43A Phase 1: pregame freeze of Champion paper BET/PASS decisions.

Consumes only information available before scheduled first pitch: frozen P37
Champion identity/probability, game identity, scheduled start, and frozen P39
trusted pregame market. Applies the unchanged P40 zero-EV rule and writes an
immutable decision bundle. This module does not load a postgame payload.
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
    PAPER_STAKE_CONVENTION,
    PaperMoneylineDecision,
)
from .p40a_moneyline_paper_bet_pass import (
    P37A_REPORT_RELATIVE_PATH,
    P38A_REPORT_RELATIVE_PATH,
    P39A_REPORT_RELATIVE_PATH,
    P40A_CHAMPION_ROLE,
    P40A_EXPECTED_P37_COMPARISONS_SHA256,
    P40A_EXPECTED_P37_TARGET_COUNT,
    P40A_EXPECTED_P39_EDGE_READY_COUNT,
    P40A_REPORT_RELATIVE_PATH,
    P40AAuthority,
    P40APredictionRow,
    _decimal,
    _load_p39_market_rows,
    _positive_int,
    _read_json,
    _read_jsonl,
    _sha,
    _sha256_path,
    _text,
    _validate_p39_authority,
)
from .p42a_offline_end_to_end_paper_workflow import (
    P41A_REPORT_RELATIVE_PATH,
    freeze_champion_decisions,
    load_p39_no_market_exclusions,
)


P43A_TASK_ID = "P43A"
P43A_REPORT_RELATIVE_PATH = Path("report/p43a_two_phase_paper_workflow")
P43A_WORKFLOW_LABEL = "OFFLINE_HISTORICAL_TWO_PHASE_PAPER_REHEARSAL"
P43A_WORKFLOW_KIND = "OFFLINE_HISTORICAL_TWO_PHASE_PAPER_REHEARSAL"
P43A_HUMAN_LABEL = "OFFLINE / HISTORICAL TWO-PHASE PAPER REHEARSAL"
P43A_PREGAME_DECISION_SCHEMA = "p43a.pregame_decision.v1"
P43A_PREGAME_SUMMARY_SCHEMA = "p43a.pregame_summary.v1"
P43A_SOURCE_MANIFEST_SCHEMA = "p43a.two_phase_paper_workflow_source_manifest.v1"
P43A_EXCLUSION_SCHEMA = "p43a.two_phase_paper_workflow_exclusion.v1"
P43A_PREGAME_ARTIFACT_FILES = (
    "source_manifest.json",
    "pregame_decisions.jsonl",
    "pregame_summary.json",
    "exclusions.jsonl",
)
P43A_PREDICTION_KEYS = (
    "fold_id",
    "evaluation_window_id",
    "comparison_row_id",
    "provider_namespace",
    "provider_game_id",
    "game_pk",
    "game_number",
    "scheduled_start_utc",
    "incumbent_model_id",
    "incumbent_model_fingerprint",
    "incumbent_home_probability",
    "challenger_model_id",
    "challenger_model_fingerprint",
    "challenger_home_probability",
    "true_oos_verified",
)
P43A_CLAIMS = {
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


def _sha256_projection(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


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


def repo_relative_path(root: Path, path: Path) -> str:
    """Persist in-repo paths as portable posix relatives, never a worktree absolute."""

    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def write_bytes_idempotent(path: Path, content: bytes) -> str:
    """Write content, or recognize an already-identical freeze artifact."""

    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return "RECOGNIZED_IDENTICAL"
        raise RuntimeError(
            "P43A_CONFLICTING_EXISTING_ARTIFACT "
            f"{path.name} already exists with a different payload"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return "WRITTEN"


def protected_authority_hashes(root: Path) -> dict[str, str]:
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
        "p42_summary": root / "report/p42a_offline_end_to_end_paper_workflow/summary.json",
        "p42_ledger": root
        / "report/p42a_offline_end_to_end_paper_workflow/workflow_ledger.jsonl",
    }
    return {name: _sha256_path(path) for name, path in paths.items()}


def _prediction_view(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in P43A_PREDICTION_KEYS if key not in row]
    if missing:
        raise ValueError(f"P43A pregame prediction row missing keys: {missing}")
    return {key: row[key] for key in P43A_PREDICTION_KEYS}


def _load_p37_pregame_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    champion_model_id: str,
    champion_fingerprint: str,
) -> tuple[P40APredictionRow, ...]:
    predictions: list[P40APredictionRow] = []
    seen_ids: set[str] = set()
    seen_games: set[str] = set()
    fold_counts: dict[str, int] = {}
    for raw in rows:
        row = _prediction_view(raw)
        fold = _text(row.get("fold_id"), field_name="fold_id")
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
        row_id = _sha(row.get("comparison_row_id"), field_name="comparison_row_id")
        if row_id in seen_ids:
            raise ValueError("P43A P37 prediction identities are not unique")
        seen_ids.add(row_id)
        provider_game_id = _text(row.get("provider_game_id"), field_name="provider_game_id")
        if provider_game_id in seen_games:
            raise ValueError("P43A P37 provider game identities are not unique")
        seen_games.add(provider_game_id)
        if row.get("true_oos_verified") is not True:
            raise ValueError("P43A requires every P37 prediction row to be true-OOS verified")
        if (
            row.get("incumbent_model_id") != champion_model_id
            or row.get("incumbent_model_fingerprint") != champion_fingerprint
        ):
            raise ValueError("P43A Champion authority drift in P37 predictions")
        for field_name in ("incumbent_home_probability", "challenger_home_probability"):
            probability = _decimal(row.get(field_name), field_name=field_name)
            if not Decimal("0") < probability < Decimal("1"):
                raise ValueError(f"P43A {field_name} is outside the probability domain")
        predictions.append(
            P40APredictionRow(
                p37_fold_id=fold,
                p37_window=_text(
                    row.get("evaluation_window_id"), field_name="evaluation_window_id"
                ),
                p37_prediction_row_id=row_id,
                provider_namespace=_text(
                    row.get("provider_namespace"), field_name="provider_namespace"
                ),
                provider_game_id=provider_game_id,
                game_pk=_positive_int(row.get("game_pk"), field_name="game_pk"),
                game_number=_positive_int(row.get("game_number"), field_name="game_number"),
                scheduled_start_utc=_text(
                    row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
                ),
                champion_model_id=_text(
                    row.get("incumbent_model_id"), field_name="incumbent_model_id"
                ),
                champion_model_fingerprint=_sha(
                    row.get("incumbent_model_fingerprint"),
                    field_name="incumbent_model_fingerprint",
                ),
                champion_home_probability=_decimal(
                    row.get("incumbent_home_probability"),
                    field_name="incumbent_home_probability",
                ),
                challenger_model_id=_text(
                    row.get("challenger_model_id"), field_name="challenger_model_id"
                ),
                challenger_model_fingerprint=_sha(
                    row.get("challenger_model_fingerprint"),
                    field_name="challenger_model_fingerprint",
                ),
                challenger_home_probability=_decimal(
                    row.get("challenger_home_probability"),
                    field_name="challenger_home_probability",
                ),
            )
        )
    if len(predictions) != P40A_EXPECTED_P37_TARGET_COUNT:
        raise ValueError("P43A P37 predictions must contain 65 rows")
    if fold_counts != {"wf_004": 23, "wf_005": 17, "wf_006": 25}:
        raise ValueError(f"P43A P37 fold row counts drifted: {fold_counts}")
    return tuple(sorted(predictions, key=lambda row: row.p37_prediction_row_id))


def _load_p37_pregame_summary(root: Path) -> dict[str, Any]:
    summary = _read_json(root / P37A_REPORT_RELATIVE_PATH / "summary.json")
    aggregate = summary.get("aggregate")
    authority = summary.get("authority")
    if not isinstance(aggregate, Mapping) or not isinstance(authority, Mapping):
        raise ValueError("P43A P37 summary authority is incomplete")
    if (
        aggregate.get("raw_row_count") != 75
        or aggregate.get("evaluable_row_count") != 65
        or aggregate.get("excluded_row_count") != 10
    ):
        raise ValueError("P43A P37 aggregate row authority drift")
    if summary.get("admitted_evaluation_fold_ids") != ["wf_004", "wf_005", "wf_006"]:
        raise ValueError("P43A P37 fold authority drift")
    _text(authority.get("current_champion_model_id"), field_name="current_champion_model_id")
    _sha(
        authority.get("current_champion_artifact_fingerprint"),
        field_name="current_champion_artifact_fingerprint",
    )
    return summary


def load_p43a_pregame_authority(
    repository_root: str | Path,
    *,
    comparisons_path: str | Path | None = None,
) -> P40AAuthority:
    """Load prediction + market authority without materializing a postgame payload."""

    root = Path(repository_root).resolve()
    p39_summary, p39_source, p39_raw_rows, p39_hashes = _validate_p39_authority(root)
    p37_summary = _load_p37_pregame_summary(root)
    default_comparisons = root / P37A_REPORT_RELATIVE_PATH / "comparisons.jsonl"
    path = Path(comparisons_path).resolve() if comparisons_path is not None else default_comparisons
    if not path.is_file():
        raise FileNotFoundError(f"P43A pregame prediction authority is missing: {path}")
    raw_rows = _read_jsonl(path)
    comparisons_sha256 = _sha256_path(path)
    if comparisons_path is None and comparisons_sha256 != P40A_EXPECTED_P37_COMPARISONS_SHA256:
        raise ValueError("P43A P37 comparisons SHA-256 authority drift")
    champion_model_id = p37_summary["authority"]["current_champion_model_id"]
    champion_fingerprint = p37_summary["authority"]["current_champion_artifact_fingerprint"]
    prediction_rows = _load_p37_pregame_predictions(
        raw_rows,
        champion_model_id=champion_model_id,
        champion_fingerprint=champion_fingerprint,
    )
    market_rows = _load_p39_market_rows(p39_raw_rows)
    prediction_by_id = {row.p37_prediction_row_id: row for row in prediction_rows}
    for market in market_rows:
        prediction = prediction_by_id.get(market.p37_prediction_row_id)
        if prediction is None:
            raise ValueError(
                "P43A P39 market row has no P37 prediction authority: "
                f"{market.p37_prediction_row_id}"
            )
        for field_name in (
            "p37_fold_id",
            "p37_window",
            "provider_namespace",
            "provider_game_id",
            "game_pk",
            "game_number",
            "scheduled_start_utc",
        ):
            if getattr(market, field_name) != getattr(prediction, field_name):
                raise ValueError(f"P43A P37/P39 identity mismatch in {field_name}")
    p38_report = root / P38A_REPORT_RELATIVE_PATH
    source_manifest = {
        "schema_version": P43A_SOURCE_MANIFEST_SCHEMA,
        "task_id": P43A_TASK_ID,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "workflow_kind": P43A_WORKFLOW_KIND,
        "human_label": P43A_HUMAN_LABEL,
        "p39a": {
            "report_path": str(P39A_REPORT_RELATIVE_PATH),
            "legacy_source_sha256": p39_source["source_sha256"],
            **p39_hashes,
        },
        "p37a": {
            "report_path": str(P37A_REPORT_RELATIVE_PATH),
            "champion_model_id": champion_model_id,
            "champion_model_fingerprint": champion_fingerprint,
            "p37_summary_sha256": _sha256_path(root / P37A_REPORT_RELATIVE_PATH / "summary.json"),
            "p37_comparisons_sha256": comparisons_sha256,
            "prediction_source_path": repo_relative_path(root, path),
        },
        "p38a": {
            "report_path": str(P38A_REPORT_RELATIVE_PATH),
            "used_for_decisions": False,
            "p38_summary_sha256": _sha256_path(p38_report / "summary.json"),
            "p38_comparisons_sha256": _sha256_path(p38_report / "comparisons.jsonl"),
        },
        "p40_policy_id": P40A_POLICY_ID,
        "p40_rule_changed": False,
        "network_required": False,
        "claims": dict(P43A_CLAIMS),
    }
    return P40AAuthority(
        repository_root=root,
        p39_summary=p39_summary,
        p39_source_manifest=p39_source,
        p37_summary=p37_summary,
        market_rows=market_rows,
        prediction_rows=prediction_rows,
        outcome_rows=(),
        source_manifest=source_manifest,
    )


def freeze_p43a_pregame_decisions(
    authority: P40AAuthority,
    *,
    market_rows: Sequence[Any] | None = None,
    prediction_rows: Sequence[Any] | None = None,
) -> tuple[PaperMoneylineDecision, ...]:
    """Freeze Champion BET/PASS from pregame inputs only."""

    if authority.outcome_rows:
        raise RuntimeError("P43A pregame freeze received a postgame payload")
    return freeze_champion_decisions(
        authority,
        market_rows=market_rows,
        prediction_rows=prediction_rows,
    )


def _selected_side(decision: PaperMoneylineDecision) -> str:
    return decision.candidate_side if decision.decision == DECISION_BET else "NONE"


def build_p43a_pregame_record(decision: PaperMoneylineDecision) -> dict[str, Any]:
    """Project one frozen Champion decision into the Phase 1 bundle row."""

    selected_side = _selected_side(decision)
    payload = {
        "schema_version": P43A_PREGAME_DECISION_SCHEMA,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "workflow_kind": P43A_WORKFLOW_KIND,
        "human_label": P43A_HUMAN_LABEL,
        "phase": "PREGAME_FREEZE",
        "prospective": False,
        "live": False,
        "production": False,
        "forward_real": False,
        "real_betting_history": False,
        "decision_fingerprint": decision.decision_id,
        "game_identity": {
            "provider_namespace": decision.provider_namespace,
            "provider_game_id": decision.provider_game_id,
            "game_pk": decision.game_pk,
            "game_number": decision.game_number,
        },
        "prediction_authority": {
            "p37_prediction_row_id": decision.p37_prediction_row_id,
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
            "source_match_id": decision.source_match_id,
            "home_decimal_odds": _decimal_text(decision.home_decimal_odds),
            "away_decimal_odds": _decimal_text(decision.away_decimal_odds),
        },
        "scheduled_start": decision.scheduled_start_utc,
        "ev_home": _decimal_text(decision.ev_home),
        "ev_away": _decimal_text(decision.ev_away),
        "candidate_side": decision.candidate_side,
        "bet_or_pass": decision.decision,
        "selected_side": selected_side,
        "paper_stake_convention": decision.paper_stake_convention,
        "paper_stake_units": _decimal_text(decision.paper_stake_units),
        "p40_decision": decision.to_projection(),
        "freeze_metadata": {
            "policy_id": P40A_POLICY_ID,
            "consumed_not_rederived": True,
            "additional_threshold": "NONE",
            "network_required": False,
            "paper_stake_convention": PAPER_STAKE_CONVENTION,
        },
    }
    return {
        "workflow_decision_id": _sha256_projection(payload),
        **payload,
    }


def build_p43a_exclusion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": P43A_EXCLUSION_SCHEMA,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "human_label": P43A_HUMAN_LABEL,
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


def _assert_pregame_inputs(decisions: Sequence[PaperMoneylineDecision]) -> None:
    for row in decisions:
        if row.market_observed_at_utc >= row.scheduled_start_utc:
            raise ValueError("P43A market observation is not strictly pregame")
        if row.local_fetched_at_utc >= row.scheduled_start_utc:
            raise ValueError("P43A local fetch is not strictly pregame")
        if row.model_role != P40A_CHAMPION_ROLE:
            raise ValueError("P43A pregame freeze is Champion-primary only")
        if row.decision == DECISION_PASS and row.paper_stake_units != 0:
            raise ValueError("PASS must have zero stake")
        if row.decision == DECISION_BET and row.paper_stake_units != 1:
            raise ValueError("BET must use the flat one-unit paper stake")


def build_p43a_pregame_summary(
    *,
    p37_target_count: int,
    no_market_count: int,
    records: Sequence[Mapping[str, Any]],
    authority_hashes: Mapping[str, str],
    freeze_status: str,
    deterministic_rerun_verified: bool,
) -> dict[str, Any]:
    bets = tuple(row for row in records if row["bet_or_pass"] == DECISION_BET)
    passes = tuple(row for row in records if row["bet_or_pass"] == DECISION_PASS)
    fingerprints = [row["decision_fingerprint"] for row in records]
    return {
        "schema_version": P43A_PREGAME_SUMMARY_SCHEMA,
        "task_id": P43A_TASK_ID,
        "workflow_label": P43A_WORKFLOW_LABEL,
        "workflow_kind": P43A_WORKFLOW_KIND,
        "human_label": P43A_HUMAN_LABEL,
        "labels": ["OFFLINE", "HISTORICAL", "TWO_PHASE", "PAPER_REHEARSAL"],
        "phase": "PREGAME_FREEZE",
        "prospective": False,
        "live": False,
        "production": False,
        "forward_real": False,
        "real_betting_history": False,
        "descriptive_only": True,
        "p37_target_count": p37_target_count,
        "p39_edge_ready_count": P40A_EXPECTED_P39_EDGE_READY_COUNT,
        "workflow_decision_count": len(records),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "settled_bet_count": 0,
        "unresolved_result_count": len(records),
        "feedback_row_count": 0,
        "p39_no_market_count": no_market_count,
        "exclusion_reasons": {"NO_MARKET": no_market_count},
        "freeze_status": freeze_status,
        "bundle_fingerprint": _sha256_projection(fingerprints),
        "decision_fingerprints": fingerprints,
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


@dataclass(frozen=True, slots=True)
class P43APregameResult:
    authority: P40AAuthority
    decisions: tuple[PaperMoneylineDecision, ...]
    records: tuple[dict[str, Any], ...]
    exclusion_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    summary: dict[str, Any]
    authority_hashes_before: dict[str, str]
    authority_hashes_after: dict[str, str]
    write_status: dict[str, str]


def run_p43a_pregame_freeze(
    repository_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    comparisons_path: str | Path | None = None,
    persist: bool = True,
) -> P43APregameResult:
    """Freeze the Champion paper decision bundle and stop without a final result."""

    root = Path(repository_root).resolve()
    hashes_before = protected_authority_hashes(root)
    first_authority = load_p43a_pregame_authority(root, comparisons_path=comparisons_path)
    no_market_rows = load_p39_no_market_exclusions(root)
    first_decisions = freeze_p43a_pregame_decisions(first_authority)
    _assert_pregame_inputs(first_decisions)
    second_authority = load_p43a_pregame_authority(root, comparisons_path=comparisons_path)
    second_decisions = freeze_p43a_pregame_decisions(
        second_authority,
        market_rows=tuple(reversed(second_authority.market_rows)),
        prediction_rows=tuple(reversed(second_authority.prediction_rows)),
    )
    if tuple(row.to_projection() for row in first_decisions) != tuple(
        row.to_projection() for row in second_decisions
    ):
        raise RuntimeError("P43A deterministic pregame replay differed")
    records = tuple(build_p43a_pregame_record(row) for row in first_decisions)
    exclusion_rows = tuple(build_p43a_exclusion_row(row) for row in no_market_rows)
    no_market_ids = {row["p37_prediction_row_id"] for row in no_market_rows}
    if no_market_ids & {row.p37_prediction_row_id for row in first_decisions}:
        raise ValueError("P43A admitted a NO_MARKET row into the decision universe")
    source_manifest = {
        **first_authority.source_manifest,
        "protected_authority_hashes": hashes_before,
        "consumed_authorities": {
            "p37a": str(P37A_REPORT_RELATIVE_PATH),
            "p38a_read_only_unused_for_decisions": str(P38A_REPORT_RELATIVE_PATH),
            "p39a": str(P39A_REPORT_RELATIVE_PATH),
            "p40a_rule": P40A_POLICY_ID,
            "p41a_research_only": str(P41A_REPORT_RELATIVE_PATH),
        },
    }
    summary = build_p43a_pregame_summary(
        p37_target_count=len(first_authority.prediction_rows),
        no_market_count=len(no_market_rows),
        records=records,
        authority_hashes=hashes_before,
        freeze_status="FROZEN",
        deterministic_rerun_verified=True,
    )
    hashes_after = protected_authority_hashes(root)
    if hashes_after != hashes_before:
        raise RuntimeError("P43A mutated a protected P37/P38/P39/P40/P41/P42 authority")
    write_status: dict[str, str] = {}
    if persist:
        directory = Path(output_dir or (root / P43A_REPORT_RELATIVE_PATH))
        write_status = write_p43a_pregame_artifacts(
            directory,
            records=records,
            exclusion_rows=exclusion_rows,
            source_manifest=source_manifest,
            summary=summary,
        )
        if all(status == "RECOGNIZED_IDENTICAL" for status in write_status.values()):
            summary = {**summary, "freeze_status": "RECOGNIZED_IDENTICAL"}
    return P43APregameResult(
        authority=first_authority,
        decisions=first_decisions,
        records=records,
        exclusion_rows=exclusion_rows,
        source_manifest=source_manifest,
        summary=summary,
        authority_hashes_before=hashes_before,
        authority_hashes_after=hashes_after,
        write_status=write_status,
    )


def render_p43a_pregame_artifacts(
    *,
    records: Sequence[Mapping[str, Any]],
    exclusion_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, bytes]:
    return {
        "source_manifest.json": _json_bytes(source_manifest),
        "pregame_decisions.jsonl": _jsonl_bytes(records),
        "pregame_summary.json": _json_bytes(summary),
        "exclusions.jsonl": _jsonl_bytes(exclusion_rows),
    }


def write_p43a_pregame_artifacts(
    output_dir: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    exclusion_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_p43a_pregame_artifacts(
        records=records,
        exclusion_rows=exclusion_rows,
        source_manifest=source_manifest,
        summary=summary,
    )
    return {name: write_bytes_idempotent(directory / name, content) for name, content in rendered.items()}


__all__ = (
    "P43A_CLAIMS",
    "P43A_HUMAN_LABEL",
    "P43A_PREGAME_ARTIFACT_FILES",
    "P43A_PREDICTION_KEYS",
    "P43A_REPORT_RELATIVE_PATH",
    "P43A_TASK_ID",
    "P43A_WORKFLOW_KIND",
    "P43A_WORKFLOW_LABEL",
    "P43APregameResult",
    "build_p43a_pregame_record",
    "freeze_p43a_pregame_decisions",
    "load_p43a_pregame_authority",
    "protected_authority_hashes",
    "repo_relative_path",
    "read_json_object",
    "read_jsonl_objects",
    "run_p43a_pregame_freeze",
    "write_bytes_idempotent",
    "write_p43a_pregame_artifacts",
)
