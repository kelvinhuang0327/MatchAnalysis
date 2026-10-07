"""Offline market scoring and fixed-unit paper replay over existing P50C rows.

This is a read-only evaluator. It consumes the P50C frozen runs and settled
forward ledger plus the normalized two-sided quote snapshots produced by the
existing P32A/P28AB path. It never acquires prices or mutates a P50C artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_LEDGER_RECORD_SCHEMA,
    P50C_REPORT_RELATIVE_PATH,
    P50C_TASK_ID,
    calculate_log_loss,
    load_effective_forward_records,
    read_json_object,
    read_jsonl_objects,
)


QUOTE_SCHEMA = "moneyline_paper_decision_v1.quote_snapshot.v1"
TSL_SOURCE = "TSL_BLOB3RD"
P28AB_CROSSWALK = "P28AB_NEAREST_TEAM_AND_START_WITHIN_3600_SECONDS"
IDENTITY_FIELDS = (
    "provider_namespace",
    "provider_game_id",
    "game_pk",
    "game_type",
    "game_number",
    "official_date",
    "scheduled_start_utc",
    "home_team",
    "away_team",
    "home_team_code",
    "away_team_code",
)
LEGACY_GAME_TYPE = "R"


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return result


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return parsed.astimezone(UTC)


def _effective_game_type(row: Mapping[str, Any]) -> str:
    identity = row.get("game_identity")
    nested = identity.get("game_type") if isinstance(identity, Mapping) else None
    top = row.get("game_type")
    if top is not None and nested is not None and top != nested:
        raise ValueError("P50C_GAME_TYPE_MISMATCH")
    value = top if top is not None else nested
    # P50C's backward-compatible contract treats an omitted type as regular season.
    value = LEGACY_GAME_TYPE if value is None else value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("P50C_INVALID_GAME_TYPE")
    return value


def _fingerprint_list(values: Sequence[str]) -> str:
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256((canonical + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MarketComparisonPolicy:
    """Required paper-selection settings; no strategy threshold is inferred."""

    min_net_ev: Decimal | str
    max_quote_age_seconds: int
    paper_cost_per_unit: Decimal | str

    def __post_init__(self) -> None:
        minimum = _decimal(self.min_net_ev, "min_net_ev")
        cost = _decimal(self.paper_cost_per_unit, "paper_cost_per_unit")
        if minimum <= 0:
            raise ValueError("min_net_ev must be positive")
        if cost < 0:
            raise ValueError("paper_cost_per_unit must be non-negative")
        if (
            isinstance(self.max_quote_age_seconds, bool)
            or not isinstance(self.max_quote_age_seconds, int)
            or self.max_quote_age_seconds <= 0
        ):
            raise ValueError("max_quote_age_seconds must be a positive integer")
        object.__setattr__(self, "min_net_ev", minimum)
        object.__setattr__(self, "paper_cost_per_unit", cost)

    def to_payload(self) -> dict[str, Any]:
        return {
            "selection_rule": "BET the frozen P50C selected side iff p_selected * selected_decimal_odds - 1 - cost_per_unit >= min_net_ev",
            "minimum_net_ev": _decimal_text(self.min_net_ev),
            "maximum_quote_age_seconds_at_decision": self.max_quote_age_seconds,
            "paper_cost_per_unit": _decimal_text(self.paper_cost_per_unit),
            "stake_units_per_bet": "1",
            "roi_denominator": "units staked on BET rows only",
            "cost_application": "deducted once per staked unit on wins and losses",
            "quote_selection": "latest valid P32A quote by observed_at_utc within the configured pregame window",
        }


def _game_identity(row: Mapping[str, Any], *, quote: bool = False) -> dict[str, Any] | None:
    identity = row.get("game_identity")
    if not isinstance(identity, Mapping):
        return None
    result = dict(identity)
    if not quote:
        try:
            game_type = _effective_game_type(row)
        except ValueError:
            return None
        if "game_type" in result and result["game_type"] != game_type:
            return None
        result["game_type"] = game_type
    elif "game_type" not in result:
        return None
    for field in IDENTITY_FIELDS:
        value = result.get(field)
        if field in ("game_pk", "game_number"):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return None
        elif not isinstance(value, str) or not value.strip() or value != value.strip():
            return None
    if result["home_team_code"] == result["away_team_code"]:
        return None
    try:
        result["scheduled_start_utc"] = _timestamp(
            result["scheduled_start_utc"], "game_identity.scheduled_start_utc"
        )
    except ValueError:
        return None
    return result


def _identity_matches(prediction: Mapping[str, Any], quote: Mapping[str, Any]) -> bool:
    expected = _game_identity(prediction)
    actual = _game_identity(quote, quote=True)
    if expected is None or actual is None:
        return False
    return all(
        expected[field] == actual[field]
        for field in IDENTITY_FIELDS
        if field != "scheduled_start_utc"
    ) and expected["scheduled_start_utc"] == actual["scheduled_start_utc"]


def _quote_error(
    quote: Mapping[str, Any],
    *,
    prediction: Mapping[str, Any],
    policy: MarketComparisonPolicy,
) -> tuple[str | None, datetime | None, Decimal | None, Decimal | None]:
    if quote.get("_paper_state_origin") is True:
        data_modes = quote.get("_quote_data_modes")
        if not isinstance(data_modes, list):
            data_modes = [quote.get("_quote_data_mode")]
        if not data_modes or any(mode != "PROSPECTIVE" for mode in data_modes):
            return "QUOTE_PAPER_STATE_NOT_PROSPECTIVE", None, None, None
        authorities = quote.get("_paper_decision_authorities")
        if not isinstance(authorities, list) or not authorities:
            return "QUOTE_PAPER_STATE_PREDICTION_AUTHORITY_MISSING", None, None, None
        prediction_authority = prediction.get("prediction_authority")
        if not isinstance(prediction_authority, Mapping):
            return "PREDICTION_AUTHORITY_MISSING", None, None, None
        expected = (
            prediction_authority.get("run_id"),
            prediction.get("prediction_fingerprint"),
        )
        if not any(
            isinstance(authority, Mapping)
            and (authority.get("prediction_run_id"), authority.get("prediction_fingerprint"))
            == expected
            for authority in authorities
        ):
            return "QUOTE_P50C_PREDICTION_IDENTITY_MISMATCH", None, None, None
    if quote.get("schema_version") != QUOTE_SCHEMA:
        return "QUOTE_SCHEMA_UNSUPPORTED", None, None, None
    if quote.get("quote_source") != TSL_SOURCE:
        return "QUOTE_SOURCE_NOT_P32A_TSL", None, None, None
    snapshot_identity = quote.get("snapshot_identity")
    if not isinstance(snapshot_identity, str) or not snapshot_identity.strip():
        return "QUOTE_SNAPSHOT_IDENTITY_MISSING", None, None, None
    source = quote.get("source_identity")
    if not isinstance(source, Mapping) or not all(
        isinstance(source.get(field), str) and str(source.get(field)).strip()
        for field in ("source_match_id", "source_row_fingerprint")
    ):
        return "QUOTE_PROVENANCE_MISSING", None, None, None
    source_fingerprint = str(source["source_row_fingerprint"])
    if len(source_fingerprint) != 64:
        return "QUOTE_PROVENANCE_MISSING", None, None, None
    try:
        bytes.fromhex(source_fingerprint)
    except ValueError:
        return "QUOTE_PROVENANCE_MISSING", None, None, None
    if source.get("crosswalk_authority") != P28AB_CROSSWALK:
        return "QUOTE_P28AB_CROSSWALK_MISSING", None, None, None
    if quote.get("market") != "MONEYLINE" or quote.get("market_code") not in (
        "MNL",
        "MONEYLINE",
    ):
        return "QUOTE_MARKET_NOT_FULL_GAME_MONEYLINE", None, None, None

    try:
        observed = _timestamp(quote.get("observed_at_utc"), "quote.observed_at_utc")
    except ValueError:
        return "QUOTE_TIMESTAMP_INVALID", None, None, None
    identity = _game_identity(prediction)
    if identity is None:
        return "PREDICTION_IDENTITY_INCOMPLETE", None, None, None
    start: datetime = identity["scheduled_start_utc"]
    source_game_time_raw = source.get("source_game_time")
    if source_game_time_raw is None:
        return "QUOTE_SOURCE_GAME_TIME_MISSING", None, None, None
    try:
        source_game_time = _timestamp(source_game_time_raw, "source_identity.source_game_time")
    except ValueError:
        return "QUOTE_SOURCE_GAME_TIME_INVALID", None, None, None
    delta = (source_game_time - start).total_seconds()
    if abs(delta) > 3600:
        return "QUOTE_P28AB_TIME_CROSSWALK_OUT_OF_RANGE", None, None, None
    if observed >= start or observed >= source_game_time:
        return "POSTGAME_OR_POST_START_QUOTE", None, None, None
    authority = prediction.get("prediction_authority")
    if not isinstance(authority, Mapping):
        return "PREDICTION_FREEZE_TIMESTAMP_INVALID", None, None, None
    try:
        frozen_at = _timestamp(authority.get("created_at_utc"), "prediction.created_at_utc")
    except ValueError:
        return "PREDICTION_FREEZE_TIMESTAMP_INVALID", None, None, None
    decision_time = max(frozen_at, observed)
    if (decision_time - observed).total_seconds() > policy.max_quote_age_seconds:
        return "QUOTE_STALE_BEFORE_START", observed, None, None

    home_raw = quote.get("home_decimal_odds")
    away_raw = quote.get("away_decimal_odds")
    if home_raw is None or away_raw is None:
        return "TWO_SIDED_PRICE_MISSING", observed, None, None
    try:
        home_odds = _decimal(home_raw, "home_decimal_odds")
        away_odds = _decimal(away_raw, "away_decimal_odds")
    except ValueError:
        return "QUOTE_ODDS_INVALID", observed, None, None
    if home_odds <= 1 or away_odds <= 1:
        return "QUOTE_ODDS_INVALID", observed, None, None
    return None, observed, home_odds, away_odds


def _quote_source_files(inputs: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    requested = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    if not requested:
        raise ValueError("at least one quote snapshot file or capture directory is required")
    files: set[Path] = set()
    for item in requested:
        path = Path(item).resolve()
        if path.is_dir():
            found = {
                candidate.resolve()
                for pattern in ("quote_snapshots.jsonl", "paper_state.json")
                for candidate in path.rglob(pattern)
                if candidate.is_file()
            }
            if not found:
                raise FileNotFoundError(
                    f"no quote_snapshots.jsonl or paper_state.json found under {path}"
                )
            files.update(found)
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"quote snapshot input does not exist: {path}")
    return tuple(sorted(files, key=str))


def _paper_state_quote_rows(path: Path) -> list[dict[str, Any]]:
    state = read_json_object(path)
    if state.get("schema_version") != "moneyline_paper_decision_v1.state.v1":
        raise ValueError(f"unsupported paper decision state schema: {path}")
    decisions = state.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError(f"paper decision state has no decisions array: {path}")
    result: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions, 1):
        if not isinstance(decision, Mapping):
            raise ValueError(f"paper decision state row {index} is not an object: {path}")
        snapshot = decision.get("quote_snapshot")
        if snapshot is None:
            continue
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"paper decision quote snapshot row {index} is invalid: {path}")
        row = dict(snapshot)
        row.setdefault("schema_version", QUOTE_SCHEMA)
        row.setdefault("quote_source", row.get("source"))
        row.pop("source", None)
        authority = decision.get("decision_identity")
        if not isinstance(authority, Mapping):
            authority = {}
        result.append(
            {
                **row,
                "_paper_state_origin": True,
                "_quote_data_mode": decision.get("data_mode", state.get("data_mode")),
                "_paper_decision_authorities": [
                    {
                        "prediction_run_id": authority.get("prediction_run_id"),
                        "prediction_fingerprint": authority.get("prediction_fingerprint"),
                    }
                ]
                if authority.get("prediction_run_id")
                and authority.get("prediction_fingerprint")
                else [],
            }
        )
    return result


def _load_quote_batch(
    inputs: str | Path | Sequence[str | Path],
) -> tuple[list[dict[str, Any]], tuple[str, ...], int, int]:
    """Load raw P32A quote rows or their existing paper_state projections.

    Repeated captures of the same immutable snapshot are collapsed by its
    identity. Reusing an identity for different quote content fails closed.
    """
    source_files = _quote_source_files(inputs)
    rows_by_snapshot: dict[str, dict[str, Any]] = {}
    rows_without_identity: list[dict[str, Any]] = []
    row_count = 0
    duplicate_count = 0
    for path in source_files:
        if path.name == "paper_state.json":
            rows = _paper_state_quote_rows(path)
        elif path.suffix.lower() == ".jsonl":
            rows = read_jsonl_objects(path)
        else:
            raise ValueError(
                f"quote input must be JSONL or a supported paper_state.json: {path}"
            )
        row_count += len(rows)
        for row in rows:
            snapshot_id = row.get("snapshot_identity")
            if not isinstance(snapshot_id, str) or not snapshot_id.strip():
                rows_without_identity.append(row)
                continue
            core = {key: value for key, value in row.items() if not key.startswith("_")}
            existing = rows_by_snapshot.get(snapshot_id)
            if existing is None:
                rows_by_snapshot[snapshot_id] = row
                continue
            existing_core = {
                key: value for key, value in existing.items() if not key.startswith("_")
            }
            if existing_core != core:
                raise ValueError(
                    f"QUOTE_SNAPSHOT_IDENTITY_CONFLICT: snapshot {snapshot_id} has different content"
                )
            duplicate_count += 1
            if row.get("_paper_state_origin") is True:
                existing["_paper_state_origin"] = True
                modes = set(existing.get("_quote_data_modes", ()))
                if existing.get("_quote_data_mode") is not None:
                    modes.add(str(existing["_quote_data_mode"]))
                if row.get("_quote_data_mode") is not None:
                    modes.add(str(row["_quote_data_mode"]))
                existing["_quote_data_modes"] = sorted(
                    modes
                )
                authorities = list(existing.get("_paper_decision_authorities", ()))
                authorities.extend(row.get("_paper_decision_authorities", ()))
                unique_authorities = {
                    (item.get("prediction_run_id"), item.get("prediction_fingerprint"))
                    for item in authorities
                    if isinstance(item, Mapping)
                }
                existing["_paper_decision_authorities"] = [
                    {"prediction_run_id": run_id, "prediction_fingerprint": fingerprint}
                    for run_id, fingerprint in sorted(unique_authorities)
                ]
    return (
        [rows_by_snapshot[key] for key in sorted(rows_by_snapshot)] + rows_without_identity,
        tuple(str(path) for path in source_files),
        row_count,
        duplicate_count,
    )


def _select_quote(
    prediction: Mapping[str, Any],
    quotes: Sequence[Mapping[str, Any]],
    *,
    policy: MarketComparisonPolicy,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    identity = _game_identity(prediction)
    if identity is None:
        return None, {"status": "PREDICTION_IDENTITY_INCOMPLETE", "rejections": []}
    same_pk = [
        row
        for row in quotes
        if isinstance(row.get("game_identity"), Mapping)
        and row["game_identity"].get("game_pk") == identity["game_pk"]
    ]
    exact = [row for row in same_pk if _identity_matches(prediction, row)]
    if not exact:
        reason = "QUOTE_IDENTITY_MISMATCH" if same_pk else "QUOTE_MISSING"
        return None, {"status": reason, "rejections": []}

    accepted: list[tuple[datetime, Mapping[str, Any], Decimal, Decimal]] = []
    rejections: Counter[str] = Counter()
    for quote in exact:
        error, observed, home_odds, away_odds = _quote_error(
            quote, prediction=prediction, policy=policy
        )
        if error is not None:
            rejections[error] += 1
        elif observed is not None and home_odds is not None and away_odds is not None:
            accepted.append((observed, quote, home_odds, away_odds))

    if not accepted:
        if rejections:
            priority = (
                "POSTGAME_OR_POST_START_QUOTE",
                "QUOTE_STALE_BEFORE_START",
                "TWO_SIDED_PRICE_MISSING",
                "QUOTE_ODDS_INVALID",
                "QUOTE_MARKET_NOT_FULL_GAME_MONEYLINE",
                "QUOTE_SOURCE_NOT_P32A_TSL",
                "QUOTE_SCHEMA_UNSUPPORTED",
                "QUOTE_PROVENANCE_MISSING",
                "QUOTE_P28AB_CROSSWALK_MISSING",
                "QUOTE_P28AB_TIME_CROSSWALK_OUT_OF_RANGE",
                "QUOTE_SOURCE_GAME_TIME_MISSING",
                "QUOTE_SOURCE_GAME_TIME_INVALID",
                "QUOTE_TIMESTAMP_INVALID",
            )
            reason = next((name for name in priority if rejections[name]), sorted(rejections)[0])
        else:
            reason = "QUOTE_MISSING"
        return None, {
            "status": reason,
            "rejections": [{"reason": key, "count": rejections[key]} for key in sorted(rejections)],
        }

    accepted.sort(key=lambda item: item[0], reverse=True)
    latest_time = accepted[0][0]
    latest = [item for item in accepted if item[0] == latest_time]
    price_pairs = {(item[2], item[3]) for item in latest}
    if len(price_pairs) > 1:
        return None, {
            "status": "QUOTE_AMBIGUOUS_SAME_TIMESTAMP",
            "rejections": [{"reason": "QUOTE_AMBIGUOUS_SAME_TIMESTAMP", "count": len(latest)}],
        }
    selected = sorted(
        latest,
        key=lambda item: str(item[1].get("snapshot_identity") or ""),
    )[0]
    return selected[1], {
        "status": "MATCHED",
        "rejections": [{"reason": key, "count": rejections[key]} for key in sorted(rejections)],
    }


def _metrics(rows: Sequence[tuple[Decimal, int]]) -> dict[str, Any]:
    count = len(rows)
    if count == 0:
        return {"sample_count": 0, "correct_count": 0, "accuracy": None,
                "accuracy_wilson95": None, "brier_score": None, "log_loss": None,
                "classification_rule": "HOME iff p_home >= 0.5; AWAY otherwise"}
    brier = sum((p - Decimal(target)) ** 2 for p, target in rows) / Decimal(count)
    log_loss = sum(calculate_log_loss(p, target) for p, target in rows) / Decimal(count)
    correct = sum(int(p >= Decimal("0.5")) == target for p, target in rows)
    accuracy = Decimal(correct) / Decimal(count)
    z = Decimal("1.959963984540054")
    n = Decimal(count)
    center = (accuracy + z*z/(2*n))/(1+z*z/n)
    half = z*(accuracy*(1-accuracy)/n + z*z/(4*n*n)).sqrt()/(1+z*z/n)
    return {
        "sample_count": count,
        "correct_count": correct,
        "accuracy": _decimal_text(accuracy),
        "accuracy_wilson95": [_decimal_text(center-half), _decimal_text(center+half)],
        "classification_rule": "HOME iff p_home >= 0.5; AWAY otherwise",
        "brier_score": _decimal_text(brier),
        "log_loss": _decimal_text(log_loss),
    }


def _read_p50c_predictions(runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.is_dir():
        return []
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        manifest = read_json_object(manifest_path)
        if manifest.get("run_classification") != CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION:
            continue
        # P65A Run Line runs share the report tree but are not P50C Moneyline
        # probability rows; the P50C prediction-forward count intentionally
        # excludes them.
        if manifest.get("task_id") != P50C_TASK_ID:
            continue
        if manifest.get("market_id") not in (None, "moneyline", "MONEYLINE"):
            continue
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"P50C run manifest has no run id: {manifest_path}")
        rows = read_jsonl_objects(manifest_path.parent / "frozen_predictions.jsonl")
        expected_count = manifest.get("eligible_prediction_count")
        if expected_count is not None and expected_count != len(rows):
            raise ValueError(f"P50C frozen prediction count disagrees with manifest: {manifest_path.parent}")
        row_fingerprints: list[str] = []
        for row in rows:
            authority = row.get("prediction_authority")
            fingerprint = row.get("prediction_fingerprint")
            if (
                not isinstance(authority, Mapping)
                or authority.get("run_id") != run_id
                or authority.get("run_classification")
                != CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION
                or not isinstance(fingerprint, str)
                or not fingerprint
            ):
                raise ValueError(f"P50C frozen prediction authority is invalid: {manifest_path.parent}")
            key = (run_id, fingerprint)
            if key in seen:
                raise ValueError(f"duplicate P50C frozen prediction: {run_id}/{fingerprint}")
            row_game_type = _effective_game_type(row)
            manifest_game_type = manifest.get("game_type")
            if manifest_game_type is not None and row_game_type != manifest_game_type:
                raise ValueError(f"P50C frozen game type disagrees with manifest: {manifest_path.parent}")
            seen.add(key)
            row_fingerprints.append(fingerprint)
            result.append(row)
        expected_fingerprints = manifest.get("prediction_fingerprints")
        if expected_fingerprints is not None:
            if expected_fingerprints != row_fingerprints:
                raise ValueError(f"P50C frozen fingerprints disagree with manifest: {manifest_path.parent}")
            expected_bundle = manifest.get("prediction_bundle_fingerprint")
            if expected_bundle is not None and expected_bundle != _fingerprint_list(row_fingerprints):
                raise ValueError(f"P50C frozen bundle fingerprint disagrees with manifest: {manifest_path.parent}")
    return result


def _prediction_from_settlement(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the frozen P50C projection embedded in an existing ledger row."""
    identity = row.get("prediction_identity")
    game = row.get("game_identity")
    model = row.get("model_identity")
    freeze = row.get("pregame_freeze_authority")
    if not all(isinstance(value, Mapping) for value in (identity, game, model, freeze)):
        raise ValueError("P50C settled row lacks the frozen prediction projection")
    fingerprint = identity.get("prediction_fingerprint")
    workflow_id = identity.get("workflow_prediction_id")
    created_at = freeze.get("created_at_utc")
    if not all(isinstance(value, str) and value for value in (fingerprint, workflow_id, created_at)):
        raise ValueError("P50C settled row lacks frozen prediction authority")
    return {
        "workflow_prediction_id": workflow_id,
        "prediction_fingerprint": fingerprint,
        "game_identity": dict(game),
        "model_identity": dict(model),
        "prediction_authority": {
            "run_id": row.get("run_id"),
            "run_classification": row.get("run_classification"),
            "p37_prediction_row_id": identity.get("p37_prediction_row_id"),
            "created_at_utc": created_at,
        },
        "_recovered_from_settled_p50c_ledger": True,
    }


def evaluate_p50c_market_comparison(
    *,
    repository_root: str | Path,
    quote_snapshots: str | Path | Sequence[str | Path],
    policy: MarketComparisonPolicy,
    runs_root: str | Path | None = None,
    ledger_root: str | Path | None = None,
) -> dict[str, Any]:
    """Score actual P32A quotes against frozen, settled prospective P50C games.

    Quote age is measured when both the frozen model and quote are available:
    decision time is the later of P50C creation and P32A observation. A
    missing/invalid quote keeps its reason and never falls
    back to historical P39A values, file timestamps, or generated timestamps.
    """
    root = Path(repository_root).resolve()
    resolved_runs = (
        Path(runs_root).resolve()
        if runs_root is not None
        else (root / P50C_REPORT_RELATIVE_PATH / "runs").resolve()
    )
    resolved_ledger = (
        Path(ledger_root).resolve()
        if ledger_root is not None
        else (root / P50C_REPORT_RELATIVE_PATH / "ledger").resolve()
    )
    quote_rows, quote_files, quote_rows_read, duplicate_quote_count = _load_quote_batch(
        quote_snapshots
    )
    predictions = _read_p50c_predictions(resolved_runs)
    if not predictions:
        raise ValueError(f"no prospective P50C frozen predictions found under {resolved_runs}")
    settled_rows = load_effective_forward_records(resolved_ledger)
    settlement_by_prediction: dict[tuple[str, str], dict[str, Any]] = {}
    excluded_non_moneyline_settlement_count = 0
    prospective_p50c_ledger_row_count = 0
    for row in settled_rows:
        if (
            row.get("schema_version") != P50C_LEDGER_RECORD_SCHEMA
            or row.get("run_classification") != CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION
        ):
            continue
        prospective_p50c_ledger_row_count += 1
        model_identity = row.get("model_identity")
        if not isinstance(model_identity, Mapping) or model_identity.get(
            "model_probability_source"
        ) != "calibrated_p_home":
            # P65A Run Line records share the append-only report tree/schema,
            # but a Run Line outcome is not comparable to a Moneyline quote.
            excluded_non_moneyline_settlement_count += 1
            continue
        p_identity = row.get("prediction_identity")
        if not isinstance(p_identity, Mapping):
            continue
        key = (str(row.get("run_id") or ""), str(p_identity.get("prediction_fingerprint") or ""))
        if key in settlement_by_prediction:
            raise ValueError(f"duplicate P50C settlement for prediction {key}")
        settlement_by_prediction[key] = row

    prediction_by_key = {
        (str(row["prediction_authority"]["run_id"]), str(row["prediction_fingerprint"])): row
        for row in predictions
    }
    recovered_prediction_count = 0
    for key, settlement in settlement_by_prediction.items():
        if key not in prediction_by_key:
            recovered = _prediction_from_settlement(settlement)
            prediction_by_key[key] = recovered
            recovered_prediction_count += 1
    predictions = list(prediction_by_key.values())

    by_game_pk: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for quote in quote_rows:
        game = quote.get("game_identity")
        if isinstance(game, Mapping) and isinstance(game.get("game_pk"), int) and not isinstance(game.get("game_pk"), bool):
            by_game_pk[game["game_pk"]].append(quote)

    detail_rows: list[dict[str, Any]] = []
    model_baseline: list[tuple[Decimal, int]] = []
    comparison_model: list[tuple[Decimal, int]] = []
    comparison_market: list[tuple[Decimal, int]] = []
    reason_counts: Counter[str] = Counter()
    cohort: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "prediction_count": 0,
            "final_count": 0,
            "market_scored_count": 0,
            "quote_status_counts": Counter(),
            "paper_bet_count": 0,
            "paper_no_bet_count": 0,
            "paper_staked_units": Decimal("0"),
            "paper_net_profit_units": Decimal("0"),
            "paper_cost_units": Decimal("0"),
            "model_final_rows": [],
        }
    )
    total_bets = 0
    total_no_bets = 0
    total_staked = Decimal("0")
    total_net_profit = Decimal("0")
    total_cost = Decimal("0")

    for prediction in predictions:
        authority = prediction["prediction_authority"]
        run_id = str(authority["run_id"])
        fingerprint = str(prediction["prediction_fingerprint"])
        settlement = settlement_by_prediction.get((run_id, fingerprint))
        game = _game_identity(prediction)
        game_type = game["game_type"] if game is not None else "UNKNOWN"
        cohort_row = cohort[game_type]
        cohort_row["prediction_count"] += 1
        entry: dict[str, Any] = {
            "run_id": run_id,
            "prediction_fingerprint": fingerprint,
            "game_identity": {
                key: (
                    value.isoformat().replace("+00:00", "Z")
                    if isinstance(value, datetime)
                    else value
                )
                for key, value in (game or {}).items()
            },
            "result_status": "PENDING" if settlement is None else None,
            "market_status": None,
            "market_rejections": [],
            "decision": "NOT_EVALUABLE",
            "decision_reason": None,
            "clv": {"status": "UNKNOWN_NO_CLOSING_QUOTE", "value": None},
        }
        model = prediction.get("model_identity")
        if not isinstance(model, Mapping):
            entry["market_status"] = "PREDICTION_PROBABILITY_INVALID"
            entry["decision_reason"] = "PREDICTION_PROBABILITY_INVALID"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        try:
            p_home = _decimal(model.get("p_home"), "model_identity.p_home")
            p_away = _decimal(model.get("p_away"), "model_identity.p_away")
        except ValueError:
            p_home = p_away = Decimal("NaN")
        if (
            not p_home.is_finite()
            or not p_away.is_finite()
            or p_home < 0
            or p_home > 1
            or p_away != Decimal("1") - p_home
        ):
            entry["market_status"] = "PREDICTION_PROBABILITY_INVALID"
            entry["decision_reason"] = "PREDICTION_PROBABILITY_INVALID"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        selection = model.get("selection")
        expected_selection = "HOME" if p_home >= Decimal("0.5") else "AWAY"
        if selection != expected_selection:
            entry["market_status"] = "PREDICTION_SELECTION_INVALID"
            entry["decision_reason"] = "PREDICTION_SELECTION_INVALID"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue

        if settlement is None:
            entry["market_status"] = "PREDICTION_NOT_FINAL"
            entry["decision_reason"] = "PREDICTION_NOT_FINAL"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        final = settlement.get("final_result_authority")
        if not isinstance(final, Mapping) or final.get("status") != "FINAL":
            entry["result_status"] = str(final.get("status") if isinstance(final, Mapping) else "UNKNOWN")
            entry["market_status"] = "RESULT_NOT_FINAL"
            entry["decision_reason"] = "RESULT_NOT_FINAL"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        settled_game = _game_identity(settlement)
        if game is None or settled_game is None or any(
            game[field] != settled_game[field]
            for field in IDENTITY_FIELDS
            if field != "official_date"
        ):
            entry["market_status"] = "P50C_SETTLEMENT_IDENTITY_MISMATCH"
            entry["decision_reason"] = "P50C_SETTLEMENT_IDENTITY_MISMATCH"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        target_home = final.get("target_home_win")
        if isinstance(target_home, bool) or target_home not in (0, 1):
            entry["market_status"] = "FINAL_HOME_WIN_TARGET_MISSING"
            entry["decision_reason"] = "FINAL_HOME_WIN_TARGET_MISSING"
            reason_counts[entry["market_status"]] += 1
            cohort_row["quote_status_counts"][entry["market_status"]] += 1
            detail_rows.append(entry)
            continue
        target = int(target_home)
        cohort_row["final_count"] += 1
        model_baseline.append((p_home, target))
        cohort_row["model_final_rows"].append((p_home, target))
        quote_target = dict(prediction)
        quote_target["game_identity"] = dict(settlement.get("game_identity", {}))
        if "game_type" in settlement:
            quote_target["game_type"] = settlement["game_type"]
        entry["result_status"] = "FINAL"
        entry["target_home_win"] = target
        entry["model_probability_home"] = _decimal_text(p_home)

        quote, selected_status = _select_quote(
            quote_target,
            by_game_pk.get(game["game_pk"], []) if game is not None else [],
            policy=policy,
        )
        entry["market_status"] = selected_status["status"]
        entry["market_rejections"] = selected_status["rejections"]
        reason_counts[entry["market_status"]] += 1
        cohort_row["quote_status_counts"][entry["market_status"]] += 1
        if quote is None:
            entry["decision_reason"] = entry["market_status"]
            detail_rows.append(entry)
            continue

        observed = _timestamp(quote["observed_at_utc"], "quote.observed_at_utc")
        home_odds = _decimal(quote["home_decimal_odds"], "home_decimal_odds")
        away_odds = _decimal(quote["away_decimal_odds"], "away_decimal_odds")
        with localcontext() as context:
            context.prec = 50
            inverse_home = Decimal("1") / home_odds
            inverse_away = Decimal("1") / away_odds
            p_market_home = inverse_home / (inverse_home + inverse_away)
        comparison_model.append((p_home, target))
        comparison_market.append((p_market_home, target))
        cohort_row["market_scored_count"] += 1
        entry.update(
            {
                "market_probability_home_no_vig": _decimal_text(p_market_home),
                "quote_observed_at_utc": observed.isoformat().replace("+00:00", "Z"),
                "home_decimal_odds": _decimal_text(home_odds),
                "away_decimal_odds": _decimal_text(away_odds),
                "market_overround": _decimal_text(inverse_home + inverse_away - Decimal("1")),
            }
        )

        selected_odds = home_odds if selection == "HOME" else away_odds
        selected_probability = p_home if selection == "HOME" else p_away
        net_ev = selected_probability * selected_odds - Decimal("1") - policy.paper_cost_per_unit
        entry["estimated_net_ev"] = _decimal_text(net_ev)
        if net_ev < policy.min_net_ev:
            entry["decision"] = "NO_BET"
            entry["decision_reason"] = "EV_BELOW_MIN_NET_EV"
            total_no_bets += 1
            cohort_row["paper_no_bet_count"] += 1
        else:
            total_bets += 1
            cohort_row["paper_bet_count"] += 1
            total_staked += Decimal("1")
            cohort_row["paper_staked_units"] += Decimal("1")
            is_win = (selection == "HOME" and target == 1) or (selection == "AWAY" and target == 0)
            gross_profit = selected_odds - Decimal("1") if is_win else Decimal("-1")
            net_profit = gross_profit - policy.paper_cost_per_unit
            total_cost += policy.paper_cost_per_unit
            total_net_profit += net_profit
            cohort_row["paper_cost_units"] += policy.paper_cost_per_unit
            cohort_row["paper_net_profit_units"] += net_profit
            entry.update(
                {
                    "decision": "BET",
                    "decision_reason": "MIN_NET_EV_MET",
                    "paper_stake_units": "1",
                    "paper_gross_profit_units": _decimal_text(gross_profit),
                    "paper_net_profit_units": _decimal_text(net_profit),
                }
            )
        detail_rows.append(entry)

    def render_cohort(record: dict[str, Any]) -> dict[str, Any]:
        staked = record["paper_staked_units"]
        profit = record["paper_net_profit_units"]
        return {
            "prediction_count": record["prediction_count"],
            "final_count": record["final_count"],
            "model_final_metrics": _metrics(record["model_final_rows"]),
            "market_scored_count": record["market_scored_count"],
            "quote_status_counts": dict(sorted(record["quote_status_counts"].items())),
            "paper_bet_count": record["paper_bet_count"],
            "paper_no_bet_count": record["paper_no_bet_count"],
            "paper_staked_units": _decimal_text(staked),
            "paper_cost_units": _decimal_text(record["paper_cost_units"]),
            "paper_net_profit_units": _decimal_text(profit),
            "paper_net_roi": _decimal_text(profit / staked) if staked else None,
        }

    detail_rows.sort(
        key=lambda row: (
            str(row.get("game_identity", {}).get("scheduled_start_utc") or ""),
            str(row.get("game_identity", {}).get("game_pk") or ""),
            str(row.get("run_id") or ""),
        )
    )
    comparison_model_metrics = _metrics(comparison_model)
    comparison_market_metrics = _metrics(comparison_market)
    if comparison_model_metrics["brier_score"] is not None:
        brier_delta = Decimal(comparison_model_metrics["brier_score"]) - Decimal(comparison_market_metrics["brier_score"])
        log_delta = Decimal(comparison_model_metrics["log_loss"]) - Decimal(comparison_market_metrics["log_loss"])
    else:
        brier_delta = log_delta = None

    return {
        "schema_version": "p50c.market_comparison_report.v1",
        "read_only": True,
        "paper_only": True,
        "real_betting_recommendation": False,
        "profitability_claim": False,
        "quote_source_requirement": "P32A TSL_BLOB3RD plus P28AB crosswalk provenance; P39A and generated timestamps are not accepted",
        "clv": {"status": "UNKNOWN_NO_CLOSING_QUOTE", "value": None},
        "policy": policy.to_payload(),
        "input_counts": {
            "prospective_frozen_prediction_count": len(predictions),
            "prospective_p50c_forward_ledger_rows_scanned": prospective_p50c_ledger_row_count,
            "settled_moneyline_forward_ledger_row_count": len(settlement_by_prediction),
            "frozen_prediction_recovered_from_ledger_count": recovered_prediction_count,
            "non_moneyline_forward_ledger_rows_excluded": excluded_non_moneyline_settlement_count,
            "quote_snapshot_count": len(quote_rows),
            "quote_snapshot_rows_read": quote_rows_read,
            "duplicate_quote_snapshot_rows_collapsed": duplicate_quote_count,
            "quote_snapshot_source_file_count": len(quote_files),
        },
        "quote_input_files": list(quote_files),
        "model_final_baseline": _metrics(model_baseline),
        "matched_market_comparison": {
            "sample_count": len(comparison_model),
            "model": comparison_model_metrics,
            "market_no_vig": comparison_market_metrics,
            "delta_definition": "model metric minus market metric; negative favors model",
            "model_minus_market_brier": _decimal_text(brier_delta),
            "model_minus_market_log_loss": _decimal_text(log_delta),
        },
        "quote_status_counts": dict(sorted(reason_counts.items())),
        "fixed_unit_paper_roi": {
            "bet_count": total_bets,
            "no_bet_count": total_no_bets,
            "staked_units": _decimal_text(total_staked),
            "cost_units": _decimal_text(total_cost),
            "net_profit_units": _decimal_text(total_net_profit),
            "net_roi": _decimal_text(total_net_profit / total_staked) if total_staked else None,
        },
        "game_type_cohorts": {
            game_type: render_cohort(record)
            for game_type, record in sorted(cohort.items())
        },
        "predictions": detail_rows,
    }


__all__ = (
    "MarketComparisonPolicy",
    "evaluate_p50c_market_comparison",
)
