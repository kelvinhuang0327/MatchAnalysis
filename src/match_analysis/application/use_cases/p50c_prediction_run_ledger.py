"""P50C prospective prediction shadow ledger and lifecycle.

Implements a TSL-independent, market-agnostic, prediction-only forward workflow:
  MLB pregame authority
  → unchanged Champion probability
  → immutable prediction freeze
  → authoritative MLB FINAL result
  → prediction evaluation
  → append-only prediction-forward ledger
  → cumulative prediction-forward metrics.

Strictly enforces:
- Pre-first-pitch prediction freeze (timestamp strictly before scheduled start).
- Immutable order-independent fingerprints.
- Complete absence of result fields at freeze time.
- Complete absence of odds, EV, BET/PASS, stake, Kelly, bankroll or market data.
- Duplicate freeze and settlement idempotency.
- Conflicting-FINAL outcome and tamper rejection.
- Append-only ledger accounting.
- Separation of PREDICTION_FORWARD_SAMPLE_COUNT from betting FORWARD_SAMPLE_COUNT.
- Historical rehearsal and synthetic predictions do not increment forward sample count.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from .p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from .p44a_normalized_workflow_input import (
    NormalizedPregameInput,
    NormalizedResultRecord,
    load_normalized_pregame_input,
    load_normalized_result_input,
    prediction_row_to_payload,
)



P50C_TASK_ID = "P50C"
P50C_RUN_MANIFEST_SCHEMA = "p50c.prediction_run_manifest.v1"
P50C_SETTLEMENT_SUMMARY_SCHEMA = "p50c.prediction_run_settlement_summary.v1"
P50C_LEDGER_RECORD_SCHEMA = "p50c.forward_prediction_ledger_record.v1"
P50C_FORWARD_SUMMARY_SCHEMA = "p50c.cumulative_forward_prediction_summary.v1"
P50C_REPORT_RELATIVE_PATH = Path("report/p50c_prospective_prediction_shadow_ledger")

CLASSIFICATION_HISTORICAL_REHEARSAL = "HISTORICAL_REHEARSAL"
CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION = "PROSPECTIVE_FORWARD_PREDICTION"
VALID_CLASSIFICATIONS = frozenset(
    {
        CLASSIFICATION_HISTORICAL_REHEARSAL,
        CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
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

P50C_CLAIMS = {
    "real_betting": False,
    "odds_or_market_dependent": False,
    "tsl_access": False,
    "ev_or_bet_pass_selection": False,
    "model_retraining": False,
    "calibration_fitting": False,
    "model_promotion": False,
    "threshold_optimization": False,
    "kelly": False,
}

FORBIDDEN_PREGAME_RESULT_FIELDS = frozenset(
    {
        "actual_winner",
        "home_score",
        "away_score",
        "result_observed_at_utc",
        "is_correct",
        "correctness_target",
        "brier_component",
        "log_loss_component",
        "evaluated_at_utc",
        "feedback_identity",
    }
)

FORBIDDEN_PREGAME_BETTING_FIELDS = frozenset(
    {
        "home_decimal_odds",
        "away_decimal_odds",
        "ev_home",
        "ev_away",
        "bet_or_pass",
        "candidate_side",
        "paper_stake_units",
        "paper_stake_convention",
        "bankroll",
        "kelly_fraction",
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
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path}") from exc
    value = json.loads(text, object_pairs_hook=_duplicate_rejecting_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped, object_pairs_hook=_duplicate_rejecting_pairs)
        if not isinstance(value, dict):
            raise ValueError(f"{path} row {line_number} must be an object")
        rows.append(value)
    return rows


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == Decimal("0"):
        raise ValueError("ratio denominator must not be zero")
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def reject_pregame_contamination(row: Mapping[str, Any]) -> None:
    """Ensure pregame row has neither result nor betting/odds fields."""
    found_result = FORBIDDEN_PREGAME_RESULT_FIELDS.intersection(row.keys())
    if found_result:
        raise RuntimeError(
            f"P50C_PREGAME_RESULT_CONTAMINATION_REJECTED: found forbidden result fields: {sorted(found_result)}"
        )
    found_betting = FORBIDDEN_PREGAME_BETTING_FIELDS.intersection(row.keys())
    if found_betting:
        raise RuntimeError(
            f"P50C_PREGAME_BETTING_CONTAMINATION_REJECTED: found forbidden betting fields: {sorted(found_betting)}"
        )


def validate_prediction_run_classification(
    classification: str,
    *,
    prediction_rows: Sequence[NormalizedPredictionRow],
    source_identity: str,
    created_at_utc: str,
) -> None:
    """Enforce temporal rules and classification invariants on prediction freeze."""
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"unknown run classification: {classification!r}, "
            f"must be one of {sorted(VALID_CLASSIFICATIONS)}"
        )

    if not prediction_rows:
        raise ValueError("prediction_rows must not be empty")

    parse_canonical_utc(created_at_utc)

    if classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION:
        if (
            "historical" in source_identity.lower()
            or "rehearsal" in source_identity.lower()
            or "synthetic" in source_identity.lower()
        ):
            raise RuntimeError(
                "P50C_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: "
                f"historical/rehearsal source identity {source_identity!r} "
                "cannot be classified as prospective forward prediction"
            )

        earliest_start = min(row.scheduled_start_utc for row in prediction_rows)
        parse_canonical_utc(earliest_start)
        if created_at_utc >= earliest_start:
            raise RuntimeError(
                "P50C_PROSPECTIVE_TEMPORAL_AUTHORITY_INVALID: "
                f"prospective prediction run creation time {created_at_utc} is not strictly before "
                f"earliest scheduled start {earliest_start}"
            )


def canonical_prediction_fingerprint(
    *,
    source_identity: str,
    prediction_rows: Sequence[NormalizedPredictionRow],
    exclusion_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
) -> str:
    """Compute canonical order-independent fingerprint of pregame predictions."""
    sorted_predictions = sorted(
        [prediction_row_to_payload(row) for row in prediction_rows],
        key=lambda r: (r["scheduled_start_utc"], r["game_number"], r["provider_game_id"]),
    )
    payload = {
        "source_identity": source_identity,
        "predictions": sorted_predictions,
        "exclusions": sorted(
            [dict(r) for r in exclusion_rows],
            key=lambda r: (str(r.get("scheduled_start_utc", "")), str(r.get("provider_game_id", ""))),
        ),
        "source_manifest": dict(source_manifest),
    }
    return _sha256_projection(payload)


def compute_prediction_row_fingerprint(
    row: P40APredictionRow,
    *,
    official_date: str = "",
    home_team: str = "",
    away_team: str = "",
    home_team_code: str = "",
    away_team_code: str = "",
) -> str:
    """Compute deterministic SHA-256 fingerprint for one frozen prediction."""
    p_home = row.champion_home_probability
    p_away = Decimal("1") - p_home
    selection = "HOME" if p_home >= Decimal("0.5") else "AWAY"
    payload = {
        "p37_prediction_row_id": row.p37_prediction_row_id,
        "provider_namespace": row.provider_namespace,
        "provider_game_id": row.provider_game_id,
        "game_pk": row.game_pk,
        "game_number": row.game_number,
        "scheduled_start_utc": row.scheduled_start_utc,
        "official_date": official_date or row.scheduled_start_utc[:10],
        "home_team": home_team or "HOME",
        "away_team": away_team or "AWAY",
        "home_team_code": home_team_code or "HOME",
        "away_team_code": away_team_code or "AWAY",
        "model_role": "CHAMPION",
        "model_id": row.champion_model_id,
        "model_fingerprint": row.champion_model_fingerprint,
        "model_probability_source": "calibrated_p_home",
        "p_home": _decimal_text(p_home),
        "p_away": _decimal_text(p_away),
        "selection": selection,
    }
    return _sha256_projection(payload)


def build_frozen_prediction_record(
    row: P40APredictionRow,
    *,
    run_id: str,
    run_classification: str,
    created_at_utc: str,
    official_date: str = "",
    home_team: str = "",
    away_team: str = "",
    home_team_code: str = "",
    away_team_code: str = "",
) -> dict[str, Any]:
    """Construct one immutable pregame prediction record with zero outcome/market fields."""
    row_fp = compute_prediction_row_fingerprint(
        row,
        official_date=official_date,
        home_team=home_team,
        away_team=away_team,
        home_team_code=home_team_code,
        away_team_code=away_team_code,
    )
    p_home = row.champion_home_probability
    p_away = Decimal("1") - p_home
    selection = "HOME" if p_home >= Decimal("0.5") else "AWAY"
    confidence = p_home if selection == "HOME" else p_away
    ht = home_team or "HOME"
    at = away_team or "AWAY"
    pred_winner = ht if selection == "HOME" else at

    record = {
        "workflow_prediction_id": f"p50c_pred_{row_fp[:32]}",
        "prediction_fingerprint": row_fp,
        "game_identity": {
            "provider_namespace": row.provider_namespace,
            "provider_game_id": row.provider_game_id,
            "game_pk": row.game_pk,
            "game_number": row.game_number,
            "scheduled_start_utc": row.scheduled_start_utc,
            "official_date": official_date or row.scheduled_start_utc[:10],
            "home_team": ht,
            "away_team": at,
            "home_team_code": home_team_code or "HOME",
            "away_team_code": away_team_code or "AWAY",
        },
        "model_identity": {
            "model_role": "CHAMPION",
            "model_id": row.champion_model_id,
            "model_fingerprint": row.champion_model_fingerprint,
            "model_probability_source": "calibrated_p_home",
            "p_home": _decimal_text(p_home),
            "p_away": _decimal_text(p_away),
            "selection": selection,
            "predicted_winner": pred_winner,
            "confidence_probability": _decimal_text(confidence),
        },
        "prediction_authority": {
            "p37_prediction_row_id": row.p37_prediction_row_id,
            "created_at_utc": created_at_utc,
            "run_id": run_id,
            "run_classification": run_classification,
        },
    }
    reject_pregame_contamination(record)
    return record


def compute_deterministic_prediction_run_id(
    *,
    run_classification: str,
    normalized_input_fingerprint: str,
    prediction_bundle_fingerprint: str,
    target_universe: Sequence[Mapping[str, Any]],
) -> str:
    """Derive deterministic logical run identity from immutable pregame prediction authority."""
    payload = {
        "schema_version": P50C_RUN_MANIFEST_SCHEMA,
        "run_classification": run_classification,
        "model_role": "CHAMPION",
        "normalized_input_fingerprint": normalized_input_fingerprint,
        "prediction_bundle_fingerprint": prediction_bundle_fingerprint,
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
    return f"p50c_run_{digest[:32]}"


@dataclass(frozen=True, slots=True)
class P50CCreateRunResult:
    status: str
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    frozen_predictions: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]


def _parse_prediction_row_dict(row: Mapping[str, Any]) -> P40APredictionRow:
    p_home = Decimal(str(row.get("champion_home_probability") or row.get("p_home") or "0.5"))
    p_challenger = Decimal(str(row.get("challenger_home_probability") or row.get("p_home") or "0.5"))
    return P40APredictionRow(
        p37_fold_id=str(row.get("p37_fold_id", "fold_0")),
        p37_window=str(row.get("p37_window", "window_0")),
        p37_prediction_row_id=str(row.get("p37_prediction_row_id") or row.get("prediction_row_id") or "0" * 64),
        provider_namespace=str(row.get("provider_namespace", "mlb_official")),
        provider_game_id=str(row.get("provider_game_id", "")),
        game_pk=int(row.get("game_pk", 1)),
        game_number=int(row.get("game_number", 1)),
        scheduled_start_utc=str(row.get("scheduled_start_utc", "")),
        champion_model_id=str(row.get("champion_model_id") or row.get("model_id") or "champion_model"),
        champion_model_fingerprint=str(row.get("champion_model_fingerprint") or row.get("model_fingerprint") or "0" * 64),
        champion_home_probability=p_home,
        challenger_model_id=str(row.get("challenger_model_id") or row.get("champion_model_id") or "challenger_model"),
        challenger_model_fingerprint=str(row.get("challenger_model_fingerprint") or row.get("champion_model_fingerprint") or "0" * 64),
        challenger_home_probability=p_challenger,
    )


def load_p50c_pregame_input(source: NormalizedPregameInput | str | Path) -> NormalizedPregameInput:
    """Load normalized pregame input or prediction-only pregame bundle."""
    if isinstance(source, NormalizedPregameInput):
        return source
    path = Path(source).resolve()
    try:
        return load_normalized_pregame_input(path)
    except Exception:
        payload = read_json_object(path)
        raw_predictions = payload.get("predictions", [])
        if not raw_predictions:
            raise ValueError(f"no predictions found in pregame input {path}")

        parsed_predictions = []
        for r in raw_predictions:
            reject_pregame_contamination(r)
            parsed_predictions.append(_parse_prediction_row_dict(r))

        raw_exclusions = payload.get("exclusions", [])
        exclusions = [dict(x) for x in raw_exclusions] if isinstance(raw_exclusions, list) else []
        for x in exclusions:
            reject_pregame_contamination(x)

        return NormalizedPregameInput(
            source_identity=str(payload.get("source_identity", "P50C_PREDICTION_SOURCE")),
            prediction_rows=tuple(parsed_predictions),
            market_rows=(),
            exclusion_rows=tuple(exclusions),
            source_manifest=dict(payload.get("source_manifest", {})),
            authority_hashes={},
            p37_summary={},
            p39_summary={},
            p39_source_manifest={},
        )


def create_p50c_prediction_run(
    repository_root: str | Path,
    *,
    pregame_input: NormalizedPregameInput | str | Path,
    run_classification: str = CLASSIFICATION_HISTORICAL_REHEARSAL,
    run_root: str | Path | None = None,
    created_at_utc: str = "2026-08-17T12:00:00Z",
) -> P50CCreateRunResult:
    """Create and freeze an immutable prediction run strictly before scheduled game start."""
    root = Path(repository_root).resolve()
    bundle = load_p50c_pregame_input(pregame_input)


    validate_prediction_run_classification(
        run_classification,
        prediction_rows=bundle.prediction_rows,
        source_identity=bundle.source_identity,
        created_at_utc=created_at_utc,
    )

    norm_fp = canonical_prediction_fingerprint(
        source_identity=bundle.source_identity,
        prediction_rows=bundle.prediction_rows,
        exclusion_rows=bundle.exclusion_rows,
        source_manifest=bundle.source_manifest,
    )

    target_universe = [
        {
            "game_pk": row.game_pk,
            "game_number": row.game_number,
            "provider_game_id": row.provider_game_id,
            "p37_prediction_row_id": row.p37_prediction_row_id,
        }
        for row in bundle.prediction_rows
    ]

    market_by_pred_id = {m.p37_prediction_row_id: m for m in bundle.market_rows}

    pred_fps = [
        compute_prediction_row_fingerprint(
            row,
            official_date=market_by_pred_id[row.p37_prediction_row_id].official_date if row.p37_prediction_row_id in market_by_pred_id else "",
            home_team=market_by_pred_id[row.p37_prediction_row_id].home_team if row.p37_prediction_row_id in market_by_pred_id else "",
            away_team=market_by_pred_id[row.p37_prediction_row_id].away_team if row.p37_prediction_row_id in market_by_pred_id else "",
            home_team_code=market_by_pred_id[row.p37_prediction_row_id].home_team_code if row.p37_prediction_row_id in market_by_pred_id else "",
            away_team_code=market_by_pred_id[row.p37_prediction_row_id].away_team_code if row.p37_prediction_row_id in market_by_pred_id else "",
        )
        for row in bundle.prediction_rows
    ]
    pred_bundle_fp = _sha256_projection(pred_fps)

    run_id = compute_deterministic_prediction_run_id(
        run_classification=run_classification,
        normalized_input_fingerprint=norm_fp,
        prediction_bundle_fingerprint=pred_bundle_fp,
        target_universe=target_universe,
    )

    if run_root is not None:
        destination_root = Path(run_root).resolve()
    else:
        destination_root = (root / P50C_REPORT_RELATIVE_PATH / "runs").resolve()

    run_dir = destination_root / run_id
    manifest_path = run_dir / "run_manifest.json"
    predictions_path = run_dir / "frozen_predictions.jsonl"
    exclusions_path = run_dir / "exclusions.jsonl"

    frozen_records = tuple(
        build_frozen_prediction_record(
            row,
            run_id=run_id,
            run_classification=run_classification,
            created_at_utc=created_at_utc,
            official_date=market_by_pred_id[row.p37_prediction_row_id].official_date if row.p37_prediction_row_id in market_by_pred_id else "",
            home_team=market_by_pred_id[row.p37_prediction_row_id].home_team if row.p37_prediction_row_id in market_by_pred_id else "",
            away_team=market_by_pred_id[row.p37_prediction_row_id].away_team if row.p37_prediction_row_id in market_by_pred_id else "",
            home_team_code=market_by_pred_id[row.p37_prediction_row_id].home_team_code if row.p37_prediction_row_id in market_by_pred_id else "",
            away_team_code=market_by_pred_id[row.p37_prediction_row_id].away_team_code if row.p37_prediction_row_id in market_by_pred_id else "",
        )
        for row in bundle.prediction_rows
    )
    exclusion_records = tuple(dict(r) for r in bundle.exclusion_rows)


    manifest_payload = {
        "schema_version": P50C_RUN_MANIFEST_SCHEMA,
        "task_id": P50C_TASK_ID,
        "run_id": run_id,
        "run_classification": run_classification,
        "lifecycle_state": STATE_FROZEN,
        "created_at_utc": created_at_utc,
        "model_role": "CHAMPION",
        "target_universe_count": len(bundle.prediction_rows) + len(bundle.exclusion_rows),
        "eligible_prediction_count": len(frozen_records),
        "settled_total_count": 0,
        "pending_count": len(frozen_records),
        "exclusion_count": len(exclusion_records),
        "normalized_input_fingerprint": norm_fp,
        "prediction_bundle_fingerprint": pred_bundle_fp,
        "prediction_fingerprints": pred_fps,
        "claims": dict(P50C_CLAIMS),
    }

    # Idempotency and conflict check
    if manifest_path.is_file():
        existing_manifest = read_json_object(manifest_path)
        if (
            existing_manifest.get("run_id") == run_id
            and existing_manifest.get("prediction_bundle_fingerprint") == pred_bundle_fp
            and existing_manifest.get("normalized_input_fingerprint") == norm_fp
            and existing_manifest.get("run_classification") == run_classification
        ):
            if predictions_path.is_file() and predictions_path.read_bytes() == _jsonl_bytes(frozen_records):
                return P50CCreateRunResult(
                    status="RECOGNIZED_IDENTICAL",
                    run_id=run_id,
                    run_dir=run_dir,
                    manifest=existing_manifest,
                    frozen_predictions=frozen_records,
                    exclusions=exclusion_records,
                )
        raise RuntimeError(
            "P50C_RUN_AUTHORITY_CONFLICT: existing prediction run manifest conflicts with incoming authority"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json_bytes(manifest_payload))
    predictions_path.write_bytes(_jsonl_bytes(frozen_records))
    exclusions_path.write_bytes(_jsonl_bytes(exclusion_records))

    return P50CCreateRunResult(
        status="CREATED",
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest_payload,
        frozen_predictions=frozen_records,
        exclusions=exclusion_records,
    )


# --- Evaluation Arithmetic ---


def calculate_log_loss(probability: Decimal, target: int) -> Decimal:
    """Compute binary log loss with probability clipped to [1e-15, 1 - 1e-15]."""
    p_float = float(probability)
    p_clipped = max(min(p_float, 1.0 - 1e-15), 1e-15)
    if target == 1:
        loss = -math.log(p_clipped)
    else:
        loss = -math.log(1.0 - p_clipped)
    return Decimal(str(round(loss, 12)))


def compute_expected_calibration_error(
    predicted_probabilities: Sequence[Decimal],
    actual_outcomes: Sequence[int],
    num_bins: int = 10,
) -> Decimal:
    """Calculate Expected Calibration Error (ECE) with equal-width probability bins."""
    if len(predicted_probabilities) != len(actual_outcomes):
        raise ValueError("predicted_probabilities and actual_outcomes must have identical length")

    n = len(predicted_probabilities)
    if n == 0:
        return Decimal("0.000000")

    bin_counts = [0] * num_bins
    bin_correct = [0] * num_bins
    bin_prob_sums = [Decimal("0")] * num_bins

    for prob, outcome in zip(predicted_probabilities, actual_outcomes):
        bin_idx = min(int(prob * Decimal(num_bins)), num_bins - 1)
        bin_counts[bin_idx] += 1
        bin_correct[bin_idx] += outcome
        bin_prob_sums[bin_idx] += prob

    total_ece = Decimal("0")
    for i in range(num_bins):
        count = bin_counts[i]
        if count > 0:
            avg_acc = Decimal(bin_correct[i]) / Decimal(count)
            avg_conf = bin_prob_sums[i] / Decimal(count)
            bin_error = abs(avg_acc - avg_conf)
            weight = Decimal(count) / Decimal(n)
            total_ece += weight * bin_error

    return Decimal(str(round(float(total_ece), 8)))


def build_p50c_prediction_ledger_record(
    *,
    run_id: str,
    run_classification: str,
    frozen_prediction: Mapping[str, Any],
    result_record: NormalizedResultRecord,
    evaluated_at_utc: str,
) -> dict[str, Any]:
    """Construct one immutable prediction evaluation ledger record."""
    model_id_info = frozen_prediction["model_identity"]
    game_id_info = frozen_prediction["game_identity"]
    pred_auth = frozen_prediction["prediction_authority"]

    selection = model_id_info["selection"]
    actual_winner = result_record.actual_winner
    is_correct = selection == actual_winner
    correctness_target = 1 if is_correct else 0

    p_home = Decimal(str(model_id_info["p_home"]))
    p_away = Decimal(str(model_id_info["p_away"]))
    model_prob = p_home if selection == "HOME" else p_away

    brier_component = (model_prob - Decimal(correctness_target)) ** 2
    log_loss_comp = calculate_log_loss(model_prob, correctness_target)

    feedback_identity = _sha256_projection(
        {
            "workflow_prediction_id": frozen_prediction["workflow_prediction_id"],
            "prediction_fingerprint": frozen_prediction["prediction_fingerprint"],
            "actual_winner": actual_winner,
            "is_correct": is_correct,
            "brier_component": str(brier_component),
            "log_loss_component": str(log_loss_comp),
        }
    )

    payload_without_id = {
        "schema_version": P50C_LEDGER_RECORD_SCHEMA,
        "run_id": run_id,
        "run_classification": run_classification,
        "prediction_identity": {
            "workflow_prediction_id": frozen_prediction["workflow_prediction_id"],
            "prediction_fingerprint": frozen_prediction["prediction_fingerprint"],
            "p37_prediction_row_id": pred_auth.get("p37_prediction_row_id"),
        },
        "game_identity": {
            "provider_namespace": game_id_info["provider_namespace"],
            "provider_game_id": game_id_info["provider_game_id"],
            "game_pk": game_id_info["game_pk"],
            "game_number": game_id_info["game_number"],
            "scheduled_start_utc": game_id_info["scheduled_start_utc"],
            "official_date": game_id_info["official_date"],
            "home_team": game_id_info["home_team"],
            "away_team": game_id_info["away_team"],
            "home_team_code": game_id_info["home_team_code"],
            "away_team_code": game_id_info["away_team_code"],
        },
        "model_identity": {
            "model_role": model_id_info["model_role"],
            "model_id": model_id_info["model_id"],
            "model_fingerprint": model_id_info["model_fingerprint"],
            "model_probability_source": model_id_info["model_probability_source"],
            "p_home": _decimal_text(p_home),
            "p_away": _decimal_text(p_away),
            "selection": selection,
            "model_probability": _decimal_text(model_prob),
        },
        "pregame_freeze_authority": {
            "created_at_utc": pred_auth.get("created_at_utc"),
            "prediction_fingerprint": frozen_prediction["prediction_fingerprint"],
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
        "evaluation": {
            "selection": selection,
            "actual_winner": actual_winner,
            "is_correct": is_correct,
            "correctness_target": correctness_target,
            "brier_component": _decimal_text(brier_component),
            "log_loss_component": _decimal_text(log_loss_comp),
            "evaluated_at_utc": evaluated_at_utc,
        },
        "feedback": {
            "feedback_identity": feedback_identity,
        },
    }

    record_id = _sha256_projection(payload_without_id)
    return {
        "ledger_record_id": record_id,
        **payload_without_id,
    }


@dataclass(frozen=True, slots=True)
class P50CSettleRunResult:
    run_id: str
    run_classification: str
    lifecycle_state: str
    newly_settled_count: int
    total_settled_count: int
    pending_count: int
    settled_predictions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    forward_summary: dict[str, Any]


def compute_forward_prediction_summary(
    forward_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute cumulative forward evaluation metrics strictly over prospective prediction records."""
    prospective_rows = [
        row for row in forward_records
        if row.get("run_classification") == CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION
    ]

    distinct_runs = sorted({row["run_id"] for row in prospective_rows})
    total_settled = len(prospective_rows)

    if total_settled == 0:
        return {
            "schema_version": P50C_FORWARD_SUMMARY_SCHEMA,
            "run_count": 0,
            "run_ids": [],
            "frozen_prediction_count": 0,
            "settled_prediction_count": 0,
            "pending_count": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "accuracy": None,
            "brier_score": None,
            "log_loss": None,
            "expected_calibration_error": None,
            "first_target_date": None,
            "last_target_date": None,
            "PREDICTION_FORWARD_SAMPLE_COUNT": 0,
        }

    correct_rows = [row for row in prospective_rows if row["evaluation"]["is_correct"] is True]
    correct_count = len(correct_rows)
    incorrect_count = total_settled - correct_count

    accuracy = Decimal(correct_count) / Decimal(total_settled)
    brier_sum = sum(Decimal(str(r["evaluation"]["brier_component"])) for r in prospective_rows)
    mean_brier = brier_sum / Decimal(total_settled)

    log_loss_sum = sum(Decimal(str(r["evaluation"]["log_loss_component"])) for r in prospective_rows)
    mean_log_loss = log_loss_sum / Decimal(total_settled)

    probs = [Decimal(str(r["model_identity"]["model_probability"])) for r in prospective_rows]
    targets = [int(r["evaluation"]["correctness_target"]) for r in prospective_rows]
    ece = compute_expected_calibration_error(probs, targets)

    dates = [
        row["game_identity"]["official_date"]
        for row in prospective_rows
        if row.get("game_identity", {}).get("official_date")
    ]
    first_date = min(dates) if dates else None
    last_date = max(dates) if dates else None

    return {
        "schema_version": P50C_FORWARD_SUMMARY_SCHEMA,
        "run_count": len(distinct_runs),
        "run_ids": distinct_runs,
        "frozen_prediction_count": total_settled,
        "settled_prediction_count": total_settled,
        "pending_count": 0,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": _decimal_text(accuracy),
        "brier_score": _decimal_text(mean_brier),
        "log_loss": _decimal_text(mean_log_loss),
        "expected_calibration_error": _decimal_text(ece),
        "first_target_date": first_date,
        "last_target_date": last_date,
        "PREDICTION_FORWARD_SAMPLE_COUNT": total_settled,
    }


def settle_p50c_prediction_run(
    repository_root: str | Path,
    *,
    run_dir: str | Path,
    result_input: Sequence[NormalizedResultRecord] | str | Path,
    ledger_root: str | Path | None = None,
    settled_at_utc: str = "2026-08-17T23:59:59Z",
) -> P50CSettleRunResult:
    """Evaluate and settle MLB FINAL results against an immutable frozen prediction run."""
    root = Path(repository_root).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction run manifest is missing: {manifest_path}")

    manifest = read_json_object(manifest_path)
    run_id = manifest["run_id"]
    run_classification = manifest["run_classification"]

    predictions_path = resolved_run_dir / "frozen_predictions.jsonl"
    frozen_predictions = read_jsonl_objects(predictions_path)
    if not frozen_predictions:
        raise RuntimeError("P50C_MISSING_FROZEN_PREDICTIONS: run contains no frozen predictions")

    if isinstance(result_input, (str, Path)):
        result_records = load_normalized_result_input(result_input)
    else:
        result_records = tuple(result_input)

    # Index results by prediction_row_id and (provider_namespace, provider_game_id, game_number)
    results_by_pred_id: dict[str, NormalizedResultRecord] = {}
    results_by_game_key: dict[tuple[str, str, int], NormalizedResultRecord] = {}

    for res in result_records:
        if res.status != "FINAL":
            raise RuntimeError("P50C_NON_FINAL_RESULT_REJECTED: result status must be FINAL")
        if res.home_score == res.away_score:
            raise RuntimeError("P50C_TIE_RESULT_REJECTED: tie games not supported in MLB decision")

        if res.prediction_row_id:
            if res.prediction_row_id in results_by_pred_id:
                raise RuntimeError("P50C_CONFLICTING_RESULT_REJECTED: duplicate result for prediction")
            results_by_pred_id[res.prediction_row_id] = res

        game_key = (res.provider_namespace, res.provider_game_id, res.game_number)
        if game_key in results_by_game_key:
            raise RuntimeError("P50C_CONFLICTING_RESULT_REJECTED: duplicate result for game identity")
        results_by_game_key[game_key] = res

    # Read existing settlements
    settlements_file = resolved_run_dir / "settled_predictions.jsonl"
    existing_settled_records = read_jsonl_objects(settlements_file)
    existing_settled_by_pred_fp = {
        row["prediction_identity"]["prediction_fingerprint"]: row
        for row in existing_settled_records
    }

    newly_settled: list[dict[str, Any]] = []
    all_settled: list[dict[str, Any]] = list(existing_settled_records)

    for frozen_pred in frozen_predictions:
        pred_fp = frozen_pred["prediction_fingerprint"]
        pred_auth_id = frozen_pred.get("prediction_authority", {}).get("p37_prediction_row_id", "")
        game_id = frozen_pred["game_identity"]
        game_key = (
            game_id["provider_namespace"],
            game_id["provider_game_id"],
            game_id["game_number"],
        )

        existing = existing_settled_by_pred_fp.get(pred_fp)

        # Match incoming result
        incoming_res = results_by_pred_id.get(pred_auth_id)
        if incoming_res is None:
            incoming_res = results_by_game_key.get(game_key)

        if existing is not None:
            if incoming_res is not None:
                prev_winner = existing["final_result_authority"]["actual_winner"]
                prev_home = existing["final_result_authority"]["home_score"]
                prev_away = existing["final_result_authority"]["away_score"]
                if (
                    incoming_res.actual_winner != prev_winner
                    or incoming_res.home_score != prev_home
                    or incoming_res.away_score != prev_away
                ):
                    raise RuntimeError(
                        "P50C_CONFLICTING_RESULT_REJECTED: conflicting result authority for already-settled prediction"
                    )
            continue

        if incoming_res is not None:
            if (
                incoming_res.provider_game_id != game_id["provider_game_id"]
                or incoming_res.game_number != game_id["game_number"]
            ):
                raise RuntimeError("P50C_CONFLICTING_RESULT_REJECTED: outcome game identity mismatch")

            ledger_rec = build_p50c_prediction_ledger_record(
                run_id=run_id,
                run_classification=run_classification,
                frozen_prediction=frozen_pred,
                result_record=incoming_res,
                evaluated_at_utc=settled_at_utc,
            )
            newly_settled.append(ledger_rec)
            all_settled.append(ledger_rec)

    total_eligible = len(frozen_predictions)
    total_settled = len(all_settled)
    pending_count = total_eligible - total_settled

    if total_settled == 0:
        new_state = STATE_FROZEN
    elif total_settled < total_eligible:
        new_state = STATE_PARTIALLY_SETTLED
    else:
        new_state = STATE_SETTLED

    # Write settlements in run_dir
    settlements_file.write_bytes(_jsonl_bytes(all_settled))

    correct_count = sum(1 for r in all_settled if r["evaluation"]["is_correct"] is True)
    incorrect_count = total_settled - correct_count
    accuracy = Decimal(correct_count) / Decimal(total_settled) if total_settled > 0 else Decimal("0")

    brier_sum = sum(Decimal(str(r["evaluation"]["brier_component"])) for r in all_settled) if total_settled > 0 else Decimal("0")
    mean_brier = brier_sum / Decimal(total_settled) if total_settled > 0 else Decimal("0")

    log_loss_sum = sum(Decimal(str(r["evaluation"]["log_loss_component"])) for r in all_settled) if total_settled > 0 else Decimal("0")
    mean_log_loss = log_loss_sum / Decimal(total_settled) if total_settled > 0 else Decimal("0")

    probs = [Decimal(str(r["model_identity"]["model_probability"])) for r in all_settled]
    targets = [int(r["evaluation"]["correctness_target"]) for r in all_settled]
    ece = compute_expected_calibration_error(probs, targets)

    settlement_summary = {
        "schema_version": P50C_SETTLEMENT_SUMMARY_SCHEMA,
        "run_id": run_id,
        "run_classification": run_classification,
        "lifecycle_state": new_state,
        "target_universe_count": manifest.get("target_universe_count", total_eligible),
        "eligible_prediction_count": total_eligible,
        "settled_total_count": total_settled,
        "pending_count": pending_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": _decimal_text(accuracy) if total_settled > 0 else None,
        "brier_score": _decimal_text(mean_brier) if total_settled > 0 else None,
        "log_loss": _decimal_text(mean_log_loss) if total_settled > 0 else None,
        "expected_calibration_error": _decimal_text(ece) if total_settled > 0 else None,
        "feedback_row_count": total_settled,
    }
    (resolved_run_dir / "settlement_summary.json").write_bytes(_json_bytes(settlement_summary))

    updated_manifest = dict(manifest)
    updated_manifest["lifecycle_state"] = new_state
    updated_manifest["settled_total_count"] = total_settled
    updated_manifest["pending_count"] = pending_count
    manifest_path.write_bytes(_json_bytes(updated_manifest))

    # Update ledger
    if ledger_root is not None:
        resolved_ledger_root = Path(ledger_root).resolve()
    else:
        resolved_ledger_root = (root / P50C_REPORT_RELATIVE_PATH / "ledger").resolve()

    resolved_ledger_root.mkdir(parents=True, exist_ok=True)

    if run_classification == CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION:
        target_ledger_file = resolved_ledger_root / "prediction_forward_ledger.jsonl"
    else:
        target_ledger_file = resolved_ledger_root / "rehearsal_prediction_ledger.jsonl"

    existing_ledger_rows = read_jsonl_objects(target_ledger_file)
    existing_ledger_keys = {
        (row["run_id"], row["prediction_identity"]["prediction_fingerprint"]): row
        for row in existing_ledger_rows
    }

    rows_to_append: list[dict[str, Any]] = []
    for record in newly_settled:
        key = (record["run_id"], record["prediction_identity"]["prediction_fingerprint"])
        if key in existing_ledger_keys:
            existing_row = existing_ledger_keys[key]
            if existing_row != record:
                raise RuntimeError("P50C_CONFLICTING_RESULT_REJECTED: prediction ledger record conflict")
            continue
        rows_to_append.append(record)
        existing_ledger_keys[key] = record

    if rows_to_append:
        with target_ledger_file.open("ab") as f:
            f.write(_jsonl_bytes(rows_to_append))

    # Update forward cumulative summary
    forward_ledger_file = resolved_ledger_root / "prediction_forward_ledger.jsonl"
    forward_rows = read_jsonl_objects(forward_ledger_file)
    forward_summary = compute_forward_prediction_summary(forward_rows)
    (resolved_ledger_root / "forward_prediction_summary.json").write_bytes(_json_bytes(forward_summary))

    return P50CSettleRunResult(
        run_id=run_id,
        run_classification=run_classification,
        lifecycle_state=new_state,
        newly_settled_count=len(newly_settled),
        total_settled_count=total_settled,
        pending_count=pending_count,
        settled_predictions=tuple(all_settled),
        summary=settlement_summary,
        forward_summary=forward_summary,
    )


def get_p50c_run_status(
    repository_root: str | Path,
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Return status and summary of an existing prediction run."""
    resolved_run_dir = Path(run_dir).resolve()
    manifest_path = resolved_run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prediction run manifest is missing: {manifest_path}")
    manifest = read_json_object(manifest_path)
    summary_path = resolved_run_dir / "settlement_summary.json"
    summary = read_json_object(summary_path) if summary_path.is_file() else {}
    return {
        "run_id": manifest["run_id"],
        "run_classification": manifest["run_classification"],
        "lifecycle_state": manifest["lifecycle_state"],
        "created_at_utc": manifest.get("created_at_utc"),
        "eligible_prediction_count": manifest["eligible_prediction_count"],
        "settled_total_count": manifest.get("settled_total_count", 0),
        "pending_count": manifest.get("pending_count", manifest["eligible_prediction_count"]),
        "exclusion_count": manifest.get("exclusion_count", 0),
        "settlement_summary": summary,
    }


def get_p50c_forward_summary(
    repository_root: str | Path,
    *,
    ledger_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return cumulative forward prediction summary."""
    root = Path(repository_root).resolve()
    if ledger_root is not None:
        resolved_ledger_root = Path(ledger_root).resolve()
    else:
        resolved_ledger_root = (root / P50C_REPORT_RELATIVE_PATH / "ledger").resolve()

    forward_ledger_file = resolved_ledger_root / "prediction_forward_ledger.jsonl"
    forward_rows = read_jsonl_objects(forward_ledger_file)
    return compute_forward_prediction_summary(forward_rows)


__all__ = (
    "CLASSIFICATION_HISTORICAL_REHEARSAL",
    "CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION",
    "P50C_CLAIMS",
    "P50C_FORWARD_SUMMARY_SCHEMA",
    "P50C_LEDGER_RECORD_SCHEMA",
    "P50C_REPORT_RELATIVE_PATH",
    "P50C_RUN_MANIFEST_SCHEMA",
    "P50C_SETTLEMENT_SUMMARY_SCHEMA",
    "P50C_TASK_ID",
    "STATE_FROZEN",
    "STATE_PARTIALLY_SETTLED",
    "STATE_SETTLED",
    "VALID_CLASSIFICATIONS",
    "VALID_LIFECYCLE_STATES",
    "P50CCreateRunResult",
    "P50CSettleRunResult",
    "build_frozen_prediction_record",
    "build_p50c_prediction_ledger_record",
    "calculate_log_loss",
    "canonical_prediction_fingerprint",
    "compute_deterministic_prediction_run_id",
    "compute_expected_calibration_error",
    "compute_forward_prediction_summary",
    "compute_prediction_row_fingerprint",
    "create_p50c_prediction_run",
    "get_p50c_forward_summary",
    "get_p50c_run_status",
    "settle_p50c_prediction_run",
    "validate_prediction_run_classification",
)

