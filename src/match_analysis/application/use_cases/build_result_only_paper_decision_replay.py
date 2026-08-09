"""Build a deterministic result-only paper-decision replay (P18A).

The use case has two explicit phases.  ``select_result_only_paper_decisions``
reads only the P15C prediction snapshot and freezes decision IDs/order.
``settle_result_only_paper_decisions`` may then attach P16A-style final-result
observations.  The outcome bytes never participate in selection.
"""

from hashlib import sha256
import json
from typing import Any

from ...baseball.domain.final_result_observation import (
    FinalResultObservation,
    load_final_result_observations,
)
from ...baseball.domain.result_only_paper_decision import (
    DECISION_SCHEMA_VERSION,
    SETTLEMENT_LOST,
    SETTLEMENT_SCHEMA_VERSION,
    SETTLEMENT_UNSETTLED,
    SETTLEMENT_WON,
    ResultOnlyDecisionSelection,
    ResultOnlyPaperDecision,
    ResultOnlyPaperDecisionReplay,
    ResultOnlyPaperSettlement,
    compute_decision_id,
    compute_decision_set_fingerprint,
    compute_settlement_row_fingerprint,
    compute_settlement_set_fingerprint,
    settlement_status_for,
)
from .build_admitted_prediction_observation_snapshot import (
    AdmittedPredictionObservationRow,
    _compute_snapshot_fingerprint,
)


EXPLICIT_REPLAY_CLAIMS = {
    "db_written": False,
    "deployed": False,
    "network_called": False,
    "odds_used": False,
    "pnl_computed": False,
    "profitability_claim": False,
    "provider_called": False,
    "training_performed": False,
    "outcomes_used_for_selection": False,
    "result_only_settlement": True,
}

_OUTCOME_FIELDS = frozenset(
    {
        "actual_winner",
        "away_score",
        "home_score",
        "is_correct",
        "result_observation_id",
        "result_observed_at_utc",
        "settlement_status",
    }
)


def _parse_object(raw_text: str, context: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {context}")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw_text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {context}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{context} must be a JSON object")
    return parsed


def _parse_summary(summary_bytes: bytes) -> dict[str, Any]:
    return _parse_object(summary_bytes.decode("utf-8"), "P15C summary")


def _load_snapshot_rows(
    snapshot_bytes: bytes,
    summary_bytes: bytes,
) -> tuple[tuple[AdmittedPredictionObservationRow, ...], str, str, str]:
    """Validate P15C snapshot bytes and return canonical ordered rows."""

    summary = _parse_summary(summary_bytes)
    snapshot_sha256 = sha256(snapshot_bytes).hexdigest()
    summary_sha256 = sha256(summary_bytes).hexdigest()
    expected_snapshot_sha256 = summary.get("admitted_observations_jsonl_sha256")
    if snapshot_sha256 != expected_snapshot_sha256:
        raise ValueError(
            "P15C snapshot SHA-256 mismatch: "
            f"computed {snapshot_sha256}, expected {expected_snapshot_sha256}"
        )
    expected_snapshot_fingerprint = summary.get("snapshot_fingerprint")
    if not isinstance(expected_snapshot_fingerprint, str):
        raise ValueError("P15C summary missing snapshot_fingerprint")

    rows: list[AdmittedPredictionObservationRow] = []
    for line_number, raw_line in enumerate(
        snapshot_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        row = _parse_object(raw_line, f"P15C snapshot row {line_number}")
        required = (
            "prediction_observation_id",
            "source_result_row_fingerprint",
            "observation",
            "snapshot_row_fingerprint",
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                f"P15C snapshot row {line_number} missing fields: {missing}"
            )
        observation = row["observation"]
        if not isinstance(observation, dict):
            raise ValueError(f"P15C snapshot row {line_number} observation must be an object")
        forbidden = sorted(_OUTCOME_FIELDS.intersection(observation))
        if forbidden:
            raise ValueError(
                "P18A prediction snapshot contains outcome fields: "
                + ", ".join(forbidden)
            )
        rows.append(
            AdmittedPredictionObservationRow(
                prediction_observation_id=row["prediction_observation_id"],
                source_result_row_fingerprint=row["source_result_row_fingerprint"],
                observation=observation,
                snapshot_row_fingerprint=row["snapshot_row_fingerprint"],
            )
        )

    ordered_rows = tuple(sorted(rows, key=lambda item: item.prediction_observation_id))
    ids = tuple(item.prediction_observation_id for item in ordered_rows)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction_observation_id in P15C snapshot")
    if tuple(item.prediction_observation_id for item in rows) != ids:
        raise ValueError("P15C snapshot rows must be ordered by prediction_observation_id")
    expected_row_count = summary.get("snapshot_row_count")
    if expected_row_count != len(ordered_rows):
        raise ValueError(
            f"P15C snapshot row count mismatch: {len(ordered_rows)} vs {expected_row_count}"
        )
    actual_fingerprint = _compute_snapshot_fingerprint(ordered_rows)
    if actual_fingerprint != expected_snapshot_fingerprint:
        raise ValueError(
            "P15C snapshot fingerprint mismatch: "
            f"computed {actual_fingerprint}, expected {expected_snapshot_fingerprint}"
        )
    return ordered_rows, snapshot_sha256, summary_sha256, actual_fingerprint


def select_result_only_paper_decisions(
    *,
    snapshot_bytes: bytes,
    snapshot_summary_bytes: bytes,
) -> ResultOnlyDecisionSelection:
    """Freeze moneyline side decisions from prediction-time snapshot fields.

    ``line_value`` and ``model_probability`` are intentionally not read when
    constructing a decision.  Rows outside the existing HOME/AWAY moneyline
    contract are excluded deterministically and are never settled.
    """

    rows, snapshot_sha256, summary_sha256, snapshot_fingerprint = _load_snapshot_rows(
        snapshot_bytes,
        snapshot_summary_bytes,
    )
    decisions: list[ResultOnlyPaperDecision] = []
    for row in rows:
        observation = row.observation
        if observation.get("market_id") != "moneyline":
            continue
        selection = observation.get("selection")
        if selection not in ("HOME", "AWAY"):
            continue
        decision_id = compute_decision_id(
            prediction_observation_id=row.prediction_observation_id,
            source_snapshot_row_fingerprint=row.snapshot_row_fingerprint,
            provider_namespace=observation["provider_namespace"],
            provider_game_id=observation["provider_game_id"],
            game_number=observation["game_number"],
            selection=selection,
        )
        decisions.append(
            ResultOnlyPaperDecision(
                decision_id=decision_id,
                prediction_observation_id=row.prediction_observation_id,
                source_snapshot_row_fingerprint=row.snapshot_row_fingerprint,
                provider_namespace=observation["provider_namespace"],
                provider_game_id=observation["provider_game_id"],
                game_number=observation["game_number"],
                selection=selection,
                prediction_generated_at_utc=observation["prediction_generated_at_utc"],
                scheduled_start_utc=observation["scheduled_start_utc"],
            )
        )

    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.decision_id))
    return ResultOnlyDecisionSelection(
        schema_version=DECISION_SCHEMA_VERSION,
        source_snapshot_sha256=snapshot_sha256,
        source_snapshot_summary_sha256=summary_sha256,
        source_snapshot_fingerprint=snapshot_fingerprint,
        excluded_row_count=len(rows) - len(ordered_decisions),
        decisions=ordered_decisions,
        decision_set_fingerprint=compute_decision_set_fingerprint(ordered_decisions),
    )


def _actual_winner(result: FinalResultObservation) -> str:
    return "HOME" if result.home_score > result.away_score else "AWAY"


def settle_result_only_paper_decisions(
    *,
    selection: ResultOnlyDecisionSelection,
    final_results_bytes: bytes,
) -> ResultOnlyPaperDecisionReplay:
    """Attach final results to an already-frozen decision selection."""

    final_results_sha256 = sha256(final_results_bytes).hexdigest()
    final_results = load_final_result_observations(final_results_bytes)
    result_by_identity = {
        (item.provider_namespace, item.provider_game_id, item.game_number): item
        for item in final_results
    }

    settlements: list[ResultOnlyPaperSettlement] = []
    for decision in selection.decisions:
        result = result_by_identity.get(
            (
                decision.provider_namespace,
                decision.provider_game_id,
                decision.game_number,
            )
        )
        if result is None:
            status = SETTLEMENT_UNSETTLED
            result_observation_id = None
            result_observed_at_utc = None
            home_score = None
            away_score = None
            actual_winner = None
        else:
            result_observation_id = result.result_observation_id
            result_observed_at_utc = result.result_observed_at_utc
            home_score = result.home_score
            away_score = result.away_score
            actual_winner = _actual_winner(result)
            status = settlement_status_for(decision.selection, actual_winner)
        row_fingerprint = compute_settlement_row_fingerprint(
            decision_id=decision.decision_id,
            prediction_observation_id=decision.prediction_observation_id,
            result_observation_id=result_observation_id,
            provider_namespace=decision.provider_namespace,
            provider_game_id=decision.provider_game_id,
            game_number=decision.game_number,
            selection=decision.selection,
            settlement_status=status,
            result_observed_at_utc=result_observed_at_utc,
            home_score=home_score,
            away_score=away_score,
            actual_winner=actual_winner,
        )
        settlements.append(
            ResultOnlyPaperSettlement(
                decision_id=decision.decision_id,
                prediction_observation_id=decision.prediction_observation_id,
                result_observation_id=result_observation_id,
                provider_namespace=decision.provider_namespace,
                provider_game_id=decision.provider_game_id,
                game_number=decision.game_number,
                selection=decision.selection,
                settlement_status=status,
                result_observed_at_utc=result_observed_at_utc,
                home_score=home_score,
                away_score=away_score,
                actual_winner=actual_winner,
                settlement_row_fingerprint=row_fingerprint,
            )
        )

    ordered_settlements = tuple(settlements)
    settled_count = sum(
        1 for item in ordered_settlements if item.settlement_status != SETTLEMENT_UNSETTLED
    )
    unsettled_count = len(ordered_settlements) - settled_count
    won_count = sum(
        1 for item in ordered_settlements if item.settlement_status == SETTLEMENT_WON
    )
    lost_count = sum(
        1 for item in ordered_settlements if item.settlement_status == SETTLEMENT_LOST
    )
    return ResultOnlyPaperDecisionReplay(
        schema_version=DECISION_SCHEMA_VERSION,
        settlement_schema_version=SETTLEMENT_SCHEMA_VERSION,
        source_snapshot_sha256=selection.source_snapshot_sha256,
        source_snapshot_summary_sha256=selection.source_snapshot_summary_sha256,
        source_snapshot_fingerprint=selection.source_snapshot_fingerprint,
        final_results_sha256=final_results_sha256,
        selection=selection,
        settlements=ordered_settlements,
        settled_count=settled_count,
        unsettled_count=unsettled_count,
        won_count=won_count,
        lost_count=lost_count,
        settlement_set_fingerprint=compute_settlement_set_fingerprint(
            ordered_settlements
        ),
        claims=dict(EXPLICIT_REPLAY_CLAIMS),
    )


def build_result_only_paper_decision_replay(
    *,
    snapshot_bytes: bytes,
    snapshot_summary_bytes: bytes,
    final_results_bytes: bytes,
) -> ResultOnlyPaperDecisionReplay:
    """Select decisions first, then attach final results deterministically."""

    selection = select_result_only_paper_decisions(
        snapshot_bytes=snapshot_bytes,
        snapshot_summary_bytes=snapshot_summary_bytes,
    )
    return settle_result_only_paper_decisions(
        selection=selection,
        final_results_bytes=final_results_bytes,
    )
