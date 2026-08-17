"""P46A deterministic P35A-to-Normalized Pregame Adapter.

Converts serialized P35A pregame output (analysis rows, schedule rows, and manifests)
into the source-independent P44 normalized pregame boundary accepted by P45.
This module is a translation boundary only: it does not perform live network calls,
does not recompute predictions or odds, does not manufacture timestamps, and preserves
all point-in-time and security invariants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ...baseball.domain.canonical_utc import parse_canonical_utc
from .p40a_moneyline_paper_bet_pass import (
    P40AMarketRow,
    P40APredictionRow,
    _decimal,
    _positive_int,
    _sha,
    _text,
)
from .p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    NormalizedPregameInput,
    parse_normalized_pregame_payload,
    reject_pregame_outcome_fields,
    write_normalized_pregame_input,
)


P46A_TASK_ID = "P46A"
P46A_ADAPTER_SOURCE_IDENTITY = "P46A_P35A_PREGAME_ADAPTER"
P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY = "P35A_CONTRACT_REHEARSAL"
P46A_EXCLUSION_SCHEMA = "p43a.two_phase_paper_workflow_exclusion.v1"
P46A_WORKFLOW_LABEL = "P46A_P35A_NORMALIZED_PREGAME_ADAPTER"
P46A_HUMAN_LABEL = "P35A PREGAME NORMALIZED ADAPTER"
P46A_DEFAULT_WINDOW = "daily_pregame"
P46A_DEFAULT_NAMESPACE = "MLB_STATS_API"
P46A_PRODUCTIVE_STATUS = "EDGE_AVAILABLE"

_SCHEDULE_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "provider_game_id",
        "game_pk",
        "game_number",
        "official_date",
        "scheduled_start_utc",
        "provider_namespace",
        "home_team",
        "away_team",
        "home_team_code",
        "away_team_code",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _extract_64_hex(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        candidate = parts[-1]
        if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate.lower()):
            return candidate.lower()
    if len(text) == 64 and all(c in "0123456789abcdef" for c in text.lower()):
        return text.lower()
    return _sha256_text(text)


def _validate_probability(value: Any, *, field_name: str) -> Decimal:
    try:
        prob = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal probability") from exc
    if not prob.is_finite() or not (Decimal("0") < prob < Decimal("1")):
        raise ValueError(f"{field_name} must be strictly between 0 and 1, got {prob}")
    return prob


def _validate_odds(value: Any, *, field_name: str) -> Decimal:
    try:
        odds = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal price") from exc
    if not odds.is_finite() or odds <= Decimal("1"):
        raise ValueError(f"{field_name} must be strictly greater than 1.0, got {odds}")
    return odds


def _is_productive_p35a_row(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status") or row.get("structural_status") or "")
    if status != P46A_PRODUCTIVE_STATUS:
        return False
    if not row.get("prediction_id"):
        return False
    if row.get("model_home_probability") is None:
        return False
    if row.get("home_decimal_odds") is None or row.get("away_decimal_odds") is None:
        return False
    if not row.get("scheduled_start") and not row.get("scheduled_start_utc"):
        return False
    if not row.get("price_observed_at") and not row.get("market_observed_at_utc"):
        return False
    return True


def _clean_schedule_row(row: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if key in _SCHEDULE_ALLOWED_KEYS:
            cleaned[key] = deepcopy(value)
    return cleaned


def adapt_p35a_pregame(
    p35a_input: str | Path | Sequence[Mapping[str, Any]],
    *,
    schedule_input: str | Path | Sequence[Mapping[str, Any]] | None = None,
    source_manifest_input: str | Path | Mapping[str, Any] | None = None,
    run_manifest_input: str | Path | Mapping[str, Any] | None = None,
    source_identity: str = P46A_ADAPTER_SOURCE_IDENTITY,
) -> NormalizedPregameInput:
    """Convert serialized P35A pregame output into NormalizedPregameInput."""

    # 1. Resolve input rows and metadata
    analysis_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}
    run_manifest: dict[str, Any] = {}

    if isinstance(p35a_input, (str, Path)):
        resolved_path = Path(p35a_input).resolve()
        if resolved_path.is_dir():
            analysis_file = resolved_path / "analysis.jsonl"
            schedule_file = resolved_path / "mlb_source_snapshot.jsonl"
            source_manifest_file = resolved_path / "source_manifest.json"
            run_manifest_file = resolved_path / "run_manifest.json"

            if not analysis_file.is_file():
                raise FileNotFoundError(f"analysis.jsonl missing in bundle directory: {resolved_path}")
            analysis_rows = _read_jsonl_objects(analysis_file)

            if schedule_file.is_file():
                schedule_rows = _read_jsonl_objects(schedule_file)
            if source_manifest_file.is_file():
                source_manifest = _read_json_object(source_manifest_file)
            if run_manifest_file.is_file():
                run_manifest = _read_json_object(run_manifest_file)
        elif resolved_path.is_file():
            analysis_rows = _read_jsonl_objects(resolved_path)
            sibling_schedule = resolved_path.parent / "mlb_source_snapshot.jsonl"
            if sibling_schedule.is_file() and schedule_input is None:
                schedule_rows = _read_jsonl_objects(sibling_schedule)
            sibling_source_manifest = resolved_path.parent / "source_manifest.json"
            if sibling_source_manifest.is_file() and source_manifest_input is None:
                source_manifest = _read_json_object(sibling_source_manifest)
            sibling_run_manifest = resolved_path.parent / "run_manifest.json"
            if sibling_run_manifest.is_file() and run_manifest_input is None:
                run_manifest = _read_json_object(sibling_run_manifest)
        else:
            raise FileNotFoundError(f"P35A input path does not exist: {resolved_path}")
    else:
        analysis_rows = [dict(r) for r in p35a_input]

    if schedule_input is not None:
        if isinstance(schedule_input, (str, Path)):
            sched_path = Path(schedule_input).resolve()
            if not sched_path.is_file():
                raise FileNotFoundError(f"schedule input file missing: {sched_path}")
            schedule_rows = _read_jsonl_objects(sched_path)
        else:
            schedule_rows = [dict(r) for r in schedule_input]

    if source_manifest_input is not None:
        if isinstance(source_manifest_input, (str, Path)):
            sm_path = Path(source_manifest_input).resolve()
            if sm_path.is_file():
                source_manifest = _read_json_object(sm_path)
        else:
            source_manifest = dict(source_manifest_input)

    if run_manifest_input is not None:
        if isinstance(run_manifest_input, (str, Path)):
            rm_path = Path(run_manifest_input).resolve()
            if rm_path.is_file():
                run_manifest = _read_json_object(rm_path)
        else:
            run_manifest = dict(run_manifest_input)

    if not analysis_rows:
        raise ValueError("P35A input contains no analysis rows")

    # 2. Reject any outcome fields in analysis and manifests immediately
    reject_pregame_outcome_fields(analysis_rows)
    reject_pregame_outcome_fields(source_manifest)
    reject_pregame_outcome_fields(run_manifest)

    # Check if schedule input has non-null outcome values
    for sched in schedule_rows:
        for outcome_key in ("home_score", "away_score", "actual_winner", "final_score", "settlement"):
            if sched.get(outcome_key) is not None:
                raise ValueError(
                    f"P44A_PREGAME_OUTCOME_FIELDS_REJECTED schedule row contains {outcome_key}={sched[outcome_key]}"
                )

    # 3. Index schedule by provider_game_id if provided
    schedule_by_game_id: dict[str, dict[str, Any]] = {}
    seen_sched_keys: set[tuple[str, int, str]] = set()
    for raw_sched in schedule_rows:
        sched = _clean_schedule_row(raw_sched)
        game_id = str(sched.get("provider_game_id") or sched.get("game_pk") or "")
        if not game_id:
            raise ValueError("schedule row missing provider_game_id")
        game_num = int(sched.get("game_number", 1))
        start_utc = str(sched.get("scheduled_start_utc") or "")
        key = (game_id, game_num, start_utc)
        if key in seen_sched_keys:
            raise ValueError(f"duplicate schedule identity {key}")
        seen_sched_keys.add(key)
        if game_id in schedule_by_game_id:
            raise ValueError(f"ambiguous doubleheader or duplicate schedule for game_id={game_id}")
        schedule_by_game_id[game_id] = sched

    # 4. Map analysis rows into P40APredictionRow, P40AMarketRow, and exclusion rows
    predictions: list[P40APredictionRow] = []
    markets: list[P40AMarketRow] = []
    exclusions: list[dict[str, Any]] = []

    seen_game_ids: set[str] = set()
    seen_prediction_ids: set[str] = set()
    seen_market_snapshot_ids: set[str] = set()

    for row in analysis_rows:
        raw_game_id = row.get("game_id") or row.get("provider_game_id")
        if not raw_game_id:
            raise ValueError("P35A row is missing game identity (game_id)")
        game_id = str(raw_game_id).strip()
        if not game_id:
            raise ValueError("P35A row has blank game_id")

        if game_id in seen_game_ids:
            raise ValueError(f"duplicate game_id in P35A analysis: {game_id}")
        seen_game_ids.add(game_id)

        sched = schedule_by_game_id.get(game_id)

        # Scheduled start
        scheduled_start = str(
            row.get("scheduled_start")
            or row.get("scheduled_start_utc")
            or (sched.get("scheduled_start_utc") if sched else "")
        ).strip()
        if not scheduled_start:
            raise ValueError(f"game_id={game_id} is missing scheduled start time")
        parse_canonical_utc(scheduled_start)

        if sched is not None:
            sched_start = str(sched.get("scheduled_start_utc", "")).strip()
            if sched_start and sched_start != scheduled_start:
                raise ValueError(
                    f"scheduled start mismatch between analysis ({scheduled_start}) "
                    f"and schedule ({sched_start}) for game_id={game_id}"
                )

        # Game numbers and PK
        try:
            game_pk = int(sched.get("game_pk")) if sched and sched.get("game_pk") is not None else int(game_id)
        except (ValueError, TypeError):
            game_pk = int(_sha256_text(game_id)[:8], 16)

        try:
            game_number = int(sched.get("game_number")) if sched and sched.get("game_number") is not None else int(row.get("game_number", 1))
        except (ValueError, TypeError):
            game_number = 1

        official_date = str(
            (sched.get("official_date") if sched else None)
            or row.get("official_date")
            or scheduled_start[:10]
        )

        # Teams
        home_team = str(
            (sched.get("home_team", {}).get("name") if sched and isinstance(sched.get("home_team"), Mapping) else None)
            or row.get("home_team")
            or ""
        ).strip()
        away_team = str(
            (sched.get("away_team", {}).get("name") if sched and isinstance(sched.get("away_team"), Mapping) else None)
            or row.get("away_team")
            or ""
        ).strip()

        home_team_code = str(
            (sched.get("home_team", {}).get("abbreviation") if sched and isinstance(sched.get("home_team"), Mapping) else None)
            or row.get("home_team_code")
            or (home_team[:3].upper() if home_team else "HOM")
        ).strip()
        away_team_code = str(
            (sched.get("away_team", {}).get("abbreviation") if sched and isinstance(sched.get("away_team"), Mapping) else None)
            or row.get("away_team_code")
            or (away_team[:3].upper() if away_team else "AWY")
        ).strip()

        # Fold ID and Window
        fold_id = str(row.get("p37_fold_id") or row.get("fold_id") or f"p35a_{official_date.replace('-', '')}")
        window = str(row.get("p37_window") or row.get("window") or P46A_DEFAULT_WINDOW)
        namespace = str(
            row.get("provider_namespace")
            or (sched.get("provider_namespace") if sched else None)
            or P46A_DEFAULT_NAMESPACE
        )

        status = str(row.get("status") or row.get("structural_status") or "")

        # Check if productive
        if _is_productive_p35a_row(row):
            raw_pred_id = row.get("prediction_id")
            if not raw_pred_id:
                raise ValueError(f"productive row for game_id={game_id} is missing prediction_id")
            pred_id = _extract_64_hex(raw_pred_id, field_name="prediction_id")

            if pred_id in seen_prediction_ids:
                raise ValueError(f"duplicate prediction_id: {pred_id}")
            seen_prediction_ids.add(pred_id)

            model_id = _text(row.get("model_id"), field_name="model_id")
            model_fp = _extract_64_hex(row.get("model_fingerprint"), field_name="model_fingerprint")

            home_prob = _validate_probability(row.get("model_home_probability"), field_name="model_home_probability")
            challenger_prob = (
                _validate_probability(row.get("challenger_home_probability"), field_name="challenger_home_probability")
                if row.get("challenger_home_probability") is not None
                else home_prob
            )
            challenger_model_id = str(row.get("challenger_model_id") or model_id)
            challenger_model_fp = (
                _extract_64_hex(row.get("challenger_model_fingerprint"), field_name="challenger_model_fingerprint")
                if row.get("challenger_model_fingerprint")
                else model_fp
            )

            # Market prices and observations
            home_odds = _validate_odds(row.get("home_decimal_odds"), field_name="home_decimal_odds")
            away_odds = _validate_odds(row.get("away_decimal_odds"), field_name="away_decimal_odds")

            raw_observed_at = row.get("price_observed_at") or row.get("market_observed_at_utc")
            if not raw_observed_at:
                raise ValueError(f"productive row for game_id={game_id} missing price_observed_at")
            market_observed_at = str(raw_observed_at).strip()
            parse_canonical_utc(market_observed_at)

            raw_fetched_at = row.get("local_fetched_at_utc") or market_observed_at
            local_fetched_at = str(raw_fetched_at).strip()
            parse_canonical_utc(local_fetched_at)

            # Strict temporal guard
            if market_observed_at >= scheduled_start:
                raise ValueError(
                    f"market_observed_at ({market_observed_at}) is not strictly pregame "
                    f"for scheduled start ({scheduled_start})"
                )
            if local_fetched_at >= scheduled_start:
                raise ValueError(
                    f"local_fetched_at ({local_fetched_at}) is not strictly pregame "
                    f"for scheduled start ({scheduled_start})"
                )

            market_price_id = str(row.get("market_price_id") or f"p35a_market_{game_id}")
            market_snapshot_id = _extract_64_hex(
                row.get("market_snapshot_id") or market_price_id,
                field_name="market_snapshot_id",
            )
            if market_snapshot_id in seen_market_snapshot_ids:
                raise ValueError(f"duplicate market_snapshot_id: {market_snapshot_id}")
            seen_market_snapshot_ids.add(market_snapshot_id)

            pred_row = P40APredictionRow(
                p37_fold_id=fold_id,
                p37_window=window,
                p37_prediction_row_id=pred_id,
                provider_namespace=namespace,
                provider_game_id=game_id,
                game_pk=game_pk,
                game_number=game_number,
                scheduled_start_utc=scheduled_start,
                champion_model_id=model_id,
                champion_model_fingerprint=model_fp,
                champion_home_probability=home_prob,
                challenger_model_id=challenger_model_id,
                challenger_model_fingerprint=challenger_model_fp,
                challenger_home_probability=challenger_prob,
            )
            predictions.append(pred_row)

            mkt_row = P40AMarketRow(
                p37_fold_id=fold_id,
                p37_window=window,
                p37_prediction_row_id=pred_id,
                provider_namespace=namespace,
                provider_game_id=game_id,
                game_pk=game_pk,
                game_number=game_number,
                official_date=official_date,
                scheduled_start_utc=scheduled_start,
                home_team=home_team or "Home Team",
                away_team=away_team or "Away Team",
                home_team_code=home_team_code,
                away_team_code=away_team_code,
                market_snapshot_id=market_snapshot_id,
                market_observed_at_utc=market_observed_at,
                local_fetched_at_utc=local_fetched_at,
                source_match_id=market_price_id,
                home_decimal_odds=home_odds,
                away_decimal_odds=away_odds,
            )
            markets.append(mkt_row)
        else:
            # Non-productive / unavailable / rejected row -> must become an exclusion
            reason = str(
                row.get("controlled_unavailable_reason")
                or row.get("reason")
                or status
                or "FEATURE_UNAVAILABLE"
            )
            raw_pred_id = row.get("prediction_id")
            pred_id = (
                _extract_64_hex(raw_pred_id, field_name="prediction_id")
                if raw_pred_id
                else _sha256_text(f"exclusion:{game_id}:{scheduled_start}")
            )

            exclusion_row = {
                "schema_version": P46A_EXCLUSION_SCHEMA,
                "workflow_label": P46A_WORKFLOW_LABEL,
                "human_label": P46A_HUMAN_LABEL,
                "exclusion_reason": reason,
                "market_snapshot_status": status or reason,
                "p37_prediction_row_id": pred_id,
                "p37_window": window,
                "p37_fold_id": fold_id,
                "provider_namespace": namespace,
                "provider_game_id": game_id,
                "scheduled_start_utc": scheduled_start,
                "became_bet": False,
            }
            exclusions.append(exclusion_row)

    # 5. Deterministic canonical sorting
    predictions.sort(key=lambda row: row.p37_prediction_row_id)
    markets.sort(key=lambda row: row.p37_prediction_row_id)
    exclusions.sort(
        key=lambda r: (
            str(r.get("scheduled_start_utc", "")),
            str(r.get("provider_game_id", "")),
            str(r.get("p37_prediction_row_id", "")),
        )
    )

    # 6. Source manifest with required opaque p37a / p39a identity hashes
    # Sort analysis rows canonically so raw fingerprint is input-order independent
    canonical_sorted_analysis = sorted(
        analysis_rows,
        key=lambda r: (
            str(r.get("scheduled_start") or r.get("scheduled_start_utc") or ""),
            str(r.get("game_id") or r.get("provider_game_id") or ""),
        ),
    )
    p35a_raw_fp = _sha256_bytes(_canonical_json_bytes(canonical_sorted_analysis))
    source_manifest_payload = deepcopy(source_manifest)
    source_manifest_payload.update(
        {
            "schema_version": "p46a.p35a_normalized_pregame_adapter.v1",
            "task_id": P46A_TASK_ID,
            "adapter_label": P46A_WORKFLOW_LABEL,
            "human_label": P46A_HUMAN_LABEL,
            "p35a_run_id": run_manifest.get("run_id") or run_manifest.get("run_fingerprint") or f"p35a_{p35a_raw_fp[:16]}",
            "p35a_raw_analysis_fingerprint": p35a_raw_fp,
            "p37a": {
                "p37_comparisons_sha256": _sha256_text(f"p35a_predictions_{p35a_raw_fp}"),
                "champion_model_id": predictions[0].champion_model_id if predictions else "p22b_moneyline_logistic_challenger_v1",
                "champion_model_fingerprint": predictions[0].champion_model_fingerprint if predictions else _sha256_text("champion_model"),
            },
            "p39a": {
                "legacy_source_sha256": _sha256_text(f"p35a_tsl_market_{p35a_raw_fp}"),
            },
        }
    )

    authority_hashes = {
        "p35a_analysis": p35a_raw_fp,
        "p35a_source_manifest": _sha256_bytes(_canonical_json_bytes(source_manifest)),
        "p35a_run_manifest": _sha256_bytes(_canonical_json_bytes(run_manifest)),
    }

    pregame_bundle = NormalizedPregameInput(
        source_identity=source_identity,
        prediction_rows=tuple(predictions),
        market_rows=tuple(markets),
        exclusion_rows=tuple(exclusions),
        source_manifest=source_manifest_payload,
        authority_hashes=authority_hashes,
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )

    # Validate complete payload against P44 schema and outcome rejection
    return parse_normalized_pregame_payload(pregame_bundle.to_payload())


def adapt_p35a_pregame_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schedule_path: str | Path | None = None,
    source_identity: str = P46A_ADAPTER_SOURCE_IDENTITY,
) -> Path:
    """Read serialized P35A pregame inputs, adapt to normalized boundary, and write output."""

    adapted = adapt_p35a_pregame(
        input_path,
        schedule_input=schedule_path,
        source_identity=source_identity,
    )
    return write_normalized_pregame_input(output_path, adapted)


__all__ = (
    "P46A_ADAPTER_SOURCE_IDENTITY",
    "P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY",
    "P46A_DEFAULT_NAMESPACE",
    "P46A_DEFAULT_WINDOW",
    "P46A_EXCLUSION_SCHEMA",
    "P46A_HUMAN_LABEL",
    "P46A_PRODUCTIVE_STATUS",
    "P46A_TASK_ID",
    "P46A_WORKFLOW_LABEL",
    "adapt_p35a_pregame",
    "adapt_p35a_pregame_file",
)
