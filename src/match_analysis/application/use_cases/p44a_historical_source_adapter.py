"""Historical P37/P39/P42/P43 adapter into the P44A normalized input boundary.

This is rehearsal/test infrastructure for committed historical authority.
It is the only P44A module allowed to open those historical report paths.
The workflow core consumes the objects this adapter emits.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .p40a_moneyline_paper_bet_pass import (
    P37A_REPORT_RELATIVE_PATH,
    P38A_REPORT_RELATIVE_PATH,
    P39A_REPORT_RELATIVE_PATH,
    P40A_EXPECTED_P37_COMPARISONS_SHA256,
    P40A_EXPECTED_P37_TARGET_COUNT,
    P40A_OUTCOME_AUTHORITY,
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
    P42A_EXPECTED_P39_NO_MARKET_COUNT,
    load_p39_no_market_exclusions,
)
from .p43a_pregame_freeze import (
    P43A_CLAIMS,
    P43A_HUMAN_LABEL,
    P43A_PREDICTION_KEYS,
    P43A_SOURCE_MANIFEST_SCHEMA,
    P43A_TASK_ID,
    P43A_WORKFLOW_KIND,
    P43A_WORKFLOW_LABEL,
)
from .p44a_normalized_workflow_input import (
    NormalizedPregameInput,
    NormalizedResultRecord,
    parse_normalized_pregame_payload,
)


P44A_HISTORICAL_SOURCE_IDENTITY = "P44A_HISTORICAL_P37_P39_ADAPTER"
P44A_HISTORICAL_RESULT_SOURCE_IDENTITY = P40A_OUTCOME_AUTHORITY


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
    rows: list[Mapping[str, Any]],
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


def adapt_historical_pregame(
    repository_root: str | Path,
    *,
    comparisons_path: str | Path | None = None,
) -> NormalizedPregameInput:
    """Convert committed P37/P39 historical authority into normalized pregame input."""

    root = Path(repository_root).resolve()
    hashes = protected_authority_hashes(root)
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
    no_market_rows = load_p39_no_market_exclusions(root)
    if len(no_market_rows) != P42A_EXPECTED_P39_NO_MARKET_COUNT:
        raise ValueError("P43A P39 no-market exclusion count drifted")
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
            "prediction_source_path": str(path),
        },
        "p38a": {
            "report_path": str(P38A_REPORT_RELATIVE_PATH),
            "used_for_decisions": False,
            "p38_summary_sha256": _sha256_path(p38_report / "summary.json"),
            "p38_comparisons_sha256": _sha256_path(p38_report / "comparisons.jsonl"),
        },
        "p40_policy_id": "P40A_ZERO_EV_MONEYLINE_BET_PASS_V1",
        "p40_rule_changed": False,
        "network_required": False,
        "claims": dict(P43A_CLAIMS),
        "protected_authority_hashes": hashes,
        "consumed_authorities": {
            "p37a": str(P37A_REPORT_RELATIVE_PATH),
            "p38a_read_only_unused_for_decisions": str(P38A_REPORT_RELATIVE_PATH),
            "p39a": str(P39A_REPORT_RELATIVE_PATH),
            "p40a_rule": "P40A_ZERO_EV_MONEYLINE_BET_PASS_V1",
            "p41a_research_only": str(P41A_REPORT_RELATIVE_PATH),
        },
    }
    pregame = NormalizedPregameInput(
        source_identity=P44A_HISTORICAL_SOURCE_IDENTITY,
        prediction_rows=prediction_rows,
        market_rows=market_rows,
        exclusion_rows=tuple(dict(row) for row in no_market_rows),
        source_manifest=source_manifest,
        authority_hashes=hashes,
        p37_summary=p37_summary,
        p39_summary=p39_summary,
        p39_source_manifest=p39_source,
    )
    return parse_normalized_pregame_payload(pregame.to_payload())


def adapt_historical_pregame_authority(
    repository_root: str | Path,
    *,
    comparisons_path: str | Path | None = None,
) -> P40AAuthority:
    """Adapter convenience: historical authority as the existing P40 object."""

    return adapt_historical_pregame(
        repository_root, comparisons_path=comparisons_path
    ).to_authority(repository_root)


def _scores_from_winner(winner: str) -> tuple[int, int]:
    if winner == "HOME":
        return 1, 0
    if winner == "AWAY":
        return 0, 1
    raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")


def adapt_historical_results(
    repository_root: str | Path,
    *,
    result_source: str | Path | None = None,
) -> tuple[NormalizedResultRecord, ...]:
    """Project committed P37 HOME/AWAY finals into the normalized result boundary.

    Historical comparisons carry winner/target only. Scores are the unique
    winner-preserving 1-0 / 0-1 projection; settlement still uses winner only.
    Observation time is the comparison row's scheduled start, the only committed
    timestamp on that authority row.
    """

    root = Path(repository_root).resolve()
    path = (
        Path(result_source).resolve()
        if result_source is not None
        else root / P37A_REPORT_RELATIVE_PATH / "comparisons.jsonl"
    )
    if not path.is_file():
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")
    rows = _read_jsonl(path)
    records: list[NormalizedResultRecord] = []
    seen: dict[str, NormalizedResultRecord] = {}
    for row in rows:
        prediction_id = _sha(row.get("comparison_row_id"), field_name="comparison_row_id")
        winner = row.get("actual_winner")
        if winner not in ("HOME", "AWAY"):
            raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")
        target = row.get("target_home_win")
        if target not in (0, 1):
            raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")
        expected_target = 1 if winner == "HOME" else 0
        if target != expected_target:
            raise RuntimeError("P43A_CONFLICTING_RESULT_REJECTED")
        home_score, away_score = _scores_from_winner(winner)
        record = NormalizedResultRecord(
            prediction_row_id=prediction_id,
            provider_namespace=_text(
                row.get("provider_namespace"), field_name="provider_namespace"
            ),
            provider_game_id=_text(
                row.get("provider_game_id"), field_name="provider_game_id"
            ),
            game_number=_positive_int(row.get("game_number"), field_name="game_number"),
            status="FINAL",
            home_score=home_score,
            away_score=away_score,
            result_observed_at_utc=_text(
                row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
            ),
            source_identity=P44A_HISTORICAL_RESULT_SOURCE_IDENTITY,
        )
        existing = seen.get(prediction_id)
        if existing is not None:
            raise RuntimeError("P43A_CONFLICTING_RESULT_REJECTED")
        seen[prediction_id] = record
        records.append(record)
    if not records:
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")
    return tuple(records)


__all__ = (
    "P44A_HISTORICAL_RESULT_SOURCE_IDENTITY",
    "P44A_HISTORICAL_SOURCE_IDENTITY",
    "adapt_historical_pregame",
    "adapt_historical_pregame_authority",
    "adapt_historical_results",
    "protected_authority_hashes",
)
