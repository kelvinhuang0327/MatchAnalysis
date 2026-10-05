"""Capture prospective MLB starting-pitcher model reprices and market edges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from ...baseball.domain.information_edge_observation import (
    EDGE_OBSERVATION,
    INFORMATION_EDGE_SCHEMA_VERSION,
    InformationEdgeObservation,
    SIGNAL_BASELINE,
    STARTING_PITCHER_CHANGE,
)
from ...infrastructure.sources.mlb_starting_pitcher_signal import (
    StartingPitcherScheduleObservation,
)
from ...infrastructure.sources.tsl_moneyline_history import TslMoneylineObservation
from ..ports.reference_market_quote import (
    ReferenceMarketQuote,
    ReferenceMarketQuoteObservation,
)
from .moneyline_paper_decision import DATA_MODE_PROSPECTIVE, load_frozen_prediction_run
from .p52a_daily_prediction_freeze import (
    execute_daily_moneyline_prediction_freeze,
    load_champion_model_parameters,
)
from .p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_REPORT_RELATIVE_PATH,
    create_p50c_prediction_run,
    read_json_object,
    read_jsonl_objects,
)


MODEL_REPRICE_NOT_AVAILABLE = "MODEL_REPRICE_NOT_AVAILABLE"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTO_REPRICE_SCHEMA = "auto_model_reprice_on_signal_v1"
_AUTO_REPRICE_INTAKE_PREFIX = "auto_model_reprice_on_signal_v1_"


@dataclass(frozen=True, slots=True)
class TslMoneylineQuote:
    """P32A-normalized, explicitly joined two-sided TSL observation."""

    snapshot_id: str
    game_identity_json: str
    scheduled_start_utc: str
    observed_at_utc: str
    home_decimal_odds: Decimal
    away_decimal_odds: Decimal

    @classmethod
    def from_p32a_observation(
        cls,
        observation: TslMoneylineObservation,
        *,
        provider_game_id: str,
        game_identity: Mapping[str, Any],
    ) -> "TslMoneylineQuote":
        if game_identity.get("provider_game_id") != provider_game_id:
            raise ValueError("TSL_CANONICAL_GAME_IDENTITY_MISMATCH")
        home_code = observation.row.get("home_code")
        away_code = observation.row.get("away_code")
        if (
            not isinstance(home_code, str)
            or not home_code
            or not isinstance(away_code, str)
            or not away_code
            or home_code == away_code
            or home_code != game_identity.get("home_team_code")
            or away_code != game_identity.get("away_team_code")
        ):
            raise ValueError("TSL_CANONICAL_GAME_IDENTITY_MISMATCH")
        markets = observation.row.get("markets")
        if (
            observation.market_code != "MNL"
            or observation.row.get("is_pregame") is not True
            or not isinstance(markets, list)
            or len(markets) != 1
            or not isinstance(markets[0], Mapping)
            or markets[0].get("marketCode") != "MNL"
            or not isinstance(markets[0].get("outcomes"), list)
            or len(markets[0]["outcomes"]) != 2
        ):
            raise ValueError("TSL_TWO_SIDED_PREGAME_MONEYLINE_REQUIRED")
        scheduled_start = format_canonical_utc(
            parse_canonical_utc(observation.game_time)
        )
        observed_at = format_canonical_utc(parse_canonical_utc(observation.fetched_at))
        if parse_canonical_utc(observed_at) >= parse_canonical_utc(scheduled_start):
            raise ValueError("TSL_POST_START_OBSERVATION_REJECTED")
        identity_json = json.dumps(
            dict(game_identity), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return cls(
            snapshot_id=observation.source_row_fingerprint,
            game_identity_json=identity_json,
            scheduled_start_utc=scheduled_start,
            observed_at_utc=observed_at,
            home_decimal_odds=_odds(observation.home_decimal_odds, "tsl_home_decimal_odds"),
            away_decimal_odds=_odds(observation.away_decimal_odds, "tsl_away_decimal_odds"),
        )

    @property
    def game_identity(self) -> dict[str, Any]:
        return json.loads(self.game_identity_json)


@dataclass(frozen=True, slots=True)
class InformationEdgeCaptureResult:
    status: str
    observation: InformationEdgeObservation | None
    after_run_id: str | None = None
    intake_path: Path | None = None
    reprice_as_of_utc: str | None = None


def _now_utc() -> str:
    return format_canonical_utc(datetime.now(UTC))


def _odds(value: object, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be valid decimal odds") from error
    if not result.is_finite() or result <= Decimal("1"):
        raise ValueError(f"{field_name} must be finite decimal odds greater than 1")
    return result


def _probability(value: object, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a valid probability") from error
    if not result.is_finite() or result < 0 or result > 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _same_json(left: object, right: object) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _schedule_key(identity: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        identity.get("provider_namespace"),
        identity.get("provider_game_id"),
    )


def _provider_namespace_matches(
    prediction_namespace: object, schedule_namespace: str
) -> bool:
    return prediction_namespace == schedule_namespace or (
        prediction_namespace,
        schedule_namespace,
    ) == ("MLB_STATS_API", "mlb_official")


def _assert_schedule_matches_prediction(
    schedule: StartingPitcherScheduleObservation,
    game_identity: Mapping[str, Any],
) -> None:
    expected = {
        "provider_game_id": schedule.provider_game_id,
        "official_date": schedule.official_date,
        "game_type": schedule.game_type,
        "game_number": schedule.game_number,
        "scheduled_start_utc": schedule.scheduled_start_utc,
    }
    if not _provider_namespace_matches(
        game_identity.get("provider_namespace"), schedule.provider_namespace
    ):
        raise ValueError("SCHEDULE_PREDICTION_IDENTITY_MISMATCH")
    if any(game_identity.get(key) != value for key, value in expected.items()):
        raise ValueError("SCHEDULE_PREDICTION_IDENTITY_MISMATCH")
    for side in ("home", "away"):
        schedule_team = getattr(schedule, f"{side}_team")
        if game_identity.get(f"{side}_team") != schedule_team.get("name"):
            raise ValueError("SCHEDULE_PREDICTION_TEAM_IDENTITY_MISMATCH")
        code_key = f"{side}_team_code"
        if game_identity.get(code_key) != schedule_team.get("abbreviation"):
            raise ValueError("SCHEDULE_PREDICTION_TEAM_IDENTITY_MISMATCH")


def _load_prediction(
    *,
    run_root: Path,
    run_id: str,
    schedule: StartingPitcherScheduleObservation,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: invalid prediction run ID")
    run_dir = run_root / run_id
    try:
        frozen_run = load_frozen_prediction_run(
            run_dir,
            data_mode=DATA_MODE_PROSPECTIVE,
        )
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation) as error:
        raise RuntimeError(
            f"{MODEL_REPRICE_NOT_AVAILABLE}: cannot validate frozen run {run_id}"
        ) from error
    if (
        frozen_run.run_id != run_id
        or frozen_run.run_classification != CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION
    ):
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: run is not a prospective frozen run")
    matches: list[dict[str, Any]] = []
    for record in frozen_run.predictions:
        identity = record.get("game_identity")
        if not isinstance(identity, Mapping):
            continue
        if (
            _provider_namespace_matches(
                identity.get("provider_namespace"), schedule.provider_namespace
            )
            and identity.get("provider_game_id") == schedule.provider_game_id
            and identity.get("game_number") == schedule.game_number
        ):
            matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: run has no unique frozen game")
    record = matches[0]
    try:
        _assert_schedule_matches_prediction(schedule, record["game_identity"])
        created = record["prediction_authority"]["created_at_utc"]
        parse_canonical_utc(created)
        _probability(record["model_identity"]["p_home"], "model_identity.p_home")
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: invalid frozen prediction") from error
    return record


def _append_only(path: Path, observation: InformationEdgeObservation, existing: list[dict[str, Any]]) -> None:
    if any(row.get("observation_id") == observation.observation_id for row in existing):
        raise RuntimeError("OBSERVATION_DUPLICATE: identical canonical observation already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, observation.to_jsonl_bytes())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_observations(path: Path) -> list[dict[str, Any]]:
    try:
        return read_jsonl_objects(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"EDGE_OBSERVATION_LOG_INVALID: {path}") from error


def _state_fields(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state = row.get("probable_pitcher_state")
    return state if isinstance(state, Mapping) else None


def _state_for(schedule: StartingPitcherScheduleObservation) -> dict[str, str | None]:
    return schedule.probable_pitcher_state()


def _find_latest_game_record(
    records: list[dict[str, Any]], schedule: StartingPitcherScheduleObservation
) -> dict[str, Any] | None:
    target_key = (schedule.provider_namespace, schedule.provider_game_id)
    for row in reversed(records):
        identity = row.get("schedule_identity")
        if isinstance(identity, Mapping) and _schedule_key(identity) == target_key:
            return row
    return None


def _validate_schedule_continuity(
    prior: Mapping[str, Any], schedule: StartingPitcherScheduleObservation
) -> None:
    prior_identity = prior.get("schedule_identity")
    if not isinstance(prior_identity, Mapping) or not _same_json(
        prior_identity, schedule.schedule_identity()
    ):
        raise RuntimeError("SIGNAL_GAME_IDENTITY_CHANGED: schedule identity no longer matches baseline")


def _baseline_observation(
    schedule: StartingPitcherScheduleObservation, captured_at_utc: str
) -> InformationEdgeObservation:
    return InformationEdgeObservation.create(
        {
            "schema_version": INFORMATION_EDGE_SCHEMA_VERSION,
            "record_type": SIGNAL_BASELINE,
            "schedule_identity": schedule.schedule_identity(),
            "signal_observed_at_utc": schedule.signal_observed_at_utc,
            "probable_pitcher_state": _state_for(schedule),
            "captured_at_utc": captured_at_utc,
        }
    )


def _previous_edge_for_current_state(
    records: list[dict[str, Any]],
    schedule: StartingPitcherScheduleObservation,
) -> dict[str, Any] | None:
    current_state = _state_for(schedule)
    identity = schedule.schedule_identity()
    for row in reversed(records):
        if row.get("record_type") != EDGE_OBSERVATION:
            continue
        if not _same_json(row.get("schedule_identity"), identity):
            continue
        if _same_json(row.get("probable_pitcher_state"), current_state):
            return row
    return None


def _minutes_to_start(start: datetime, signal: datetime) -> Decimal:
    delta = start - signal
    microseconds = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    return Decimal(microseconds) / Decimal(60_000_000)


def _game_identity_from_schedule(
    schedule: StartingPitcherScheduleObservation,
) -> dict[str, Any]:
    try:
        game_pk = int(schedule.provider_game_id)
    except (TypeError, ValueError) as error:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: invalid canonical MLB game ID") from error
    if game_pk <= 0:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: invalid canonical MLB game ID")
    return {
        "provider_namespace": "MLB_STATS_API",
        "provider_game_id": schedule.provider_game_id,
        "game_pk": game_pk,
        "game_number": schedule.game_number,
        "game_type": schedule.game_type,
        "scheduled_start_utc": schedule.scheduled_start_utc,
        "official_date": schedule.official_date,
        "home_team": schedule.home_team["name"],
        "away_team": schedule.away_team["name"],
        "home_team_code": schedule.home_team["abbreviation"],
        "away_team_code": schedule.away_team["abbreviation"],
    }


def _signal_identity(
    *,
    game_identity: Mapping[str, Any],
    signal_observed_at_utc: str,
    side: str,
    previous_pitcher_id: str,
    new_pitcher_id: str,
) -> dict[str, Any]:
    return {
        "game_identity": dict(game_identity),
        "signal_observed_at_utc": format_canonical_utc(
            parse_canonical_utc(signal_observed_at_utc)
        ),
        "changed_side": side,
        "previous_pitcher_player_id": str(previous_pitcher_id),
        "new_pitcher_player_id": str(new_pitcher_id),
    }


def _signal_key(signal_identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(signal_identity), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _signal_intake_path(output_root: Path, signal_key: str) -> Path:
    return (
        output_root
        / P50C_REPORT_RELATIVE_PATH
        / "intake"
        / f"{_AUTO_REPRICE_INTAKE_PREFIX}{signal_key}.json"
    )


def _read_signal_intake(
    path: Path,
    *,
    expected_signal_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake is not a regular file")
    try:
        payload = read_json_object(path)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake is invalid") from error
    source_manifest = payload.get("source_manifest")
    metadata = source_manifest.get(_AUTO_REPRICE_SCHEMA) if isinstance(source_manifest, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake metadata is missing")
    signal_identity = metadata.get("signal_identity")
    signal_key = metadata.get("signal_key")
    probable_pitcher_state = metadata.get("probable_pitcher_state")
    if (
        metadata.get("schema_version") != _AUTO_REPRICE_SCHEMA
        or not isinstance(signal_identity, Mapping)
        or not isinstance(signal_key, str)
        or not isinstance(signal_identity.get("game_identity"), Mapping)
        or signal_identity.get("changed_side") not in ("HOME", "AWAY")
        or not isinstance(signal_identity.get("previous_pitcher_player_id"), str)
        or not isinstance(signal_identity.get("new_pitcher_player_id"), str)
        or signal_identity.get("previous_pitcher_player_id")
        == signal_identity.get("new_pitcher_player_id")
        or not isinstance(probable_pitcher_state, Mapping)
        or set(probable_pitcher_state) != {"HOME", "AWAY"}
        or any(not isinstance(probable_pitcher_state.get(side), str) for side in ("HOME", "AWAY"))
        or _signal_key(signal_identity) != signal_key
        or path.name != f"{_AUTO_REPRICE_INTAKE_PREFIX}{signal_key}.json"
    ):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake identity conflicts")
    if expected_signal_identity is not None and not _same_json(
        signal_identity, expected_signal_identity
    ):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: existing signal-keyed intake conflicts")
    as_of = metadata.get("reprice_as_of_utc")
    try:
        canonical_as_of = format_canonical_utc(parse_canonical_utc(as_of))
        signal_time = parse_canonical_utc(signal_identity["signal_observed_at_utc"])
        start_time = parse_canonical_utc(
            signal_identity["game_identity"]["scheduled_start_utc"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake timestamps are invalid") from error
    if (
        as_of != canonical_as_of
        or source_manifest.get("fetched_at_utc") != canonical_as_of
        or source_manifest.get("target_date")
        != signal_identity["game_identity"].get("official_date")
        or not signal_time <= parse_canonical_utc(canonical_as_of) < start_time
    ):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake time is invalid")
    if not isinstance(metadata.get("before_run_id"), str) or _RUN_ID.fullmatch(
        metadata["before_run_id"]
    ) is None:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake has no before authority")
    return payload, dict(metadata)


def _latest_signal_intake(
    *,
    output_root: Path,
    game_identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    intake_dir = output_root / P50C_REPORT_RELATIVE_PATH / "intake"
    if intake_dir.is_symlink():
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: canonical intake directory is a symlink")
    if not intake_dir.exists():
        return None
    if not intake_dir.is_dir():
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: canonical intake path is not a directory")
    matches: list[tuple[datetime, Path, dict[str, Any], dict[str, Any]]] = []
    for path in sorted(intake_dir.glob(f"{_AUTO_REPRICE_INTAKE_PREFIX}*.json")):
        payload, metadata = _read_signal_intake(path)
        signal_identity = metadata["signal_identity"]
        if not _same_json(signal_identity.get("game_identity"), game_identity):
            continue
        when = parse_canonical_utc(signal_identity["signal_observed_at_utc"])
        matches.append((when, path, payload, metadata))
    if not matches:
        return None
    newest_time = max(item[0] for item in matches)
    newest = [item for item in matches if item[0] == newest_time]
    if len(newest) != 1:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: ambiguous signal-keyed intake history")
    _, path, payload, metadata = newest[0]
    return path, payload, metadata


def _create_or_reuse_after_run(
    *,
    schedule: StartingPitcherScheduleObservation,
    before: Mapping[str, Any],
    before_run_id: str,
    side: str,
    previous_pitcher_id: str,
    new_pitcher_id: str,
    signal_observed_at_utc: str,
    output_root: Path,
    repository_root: Path,
    run_root: Path,
    as_of_utc: str,
    opener: Callable[..., bytes] | None,
) -> tuple[str, Path, str]:
    game_identity = before["game_identity"]
    if not isinstance(game_identity, Mapping):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: before authority has no game identity")
    signal_identity = _signal_identity(
        game_identity=game_identity,
        signal_observed_at_utc=signal_observed_at_utc,
        side=side,
        previous_pitcher_id=previous_pitcher_id,
        new_pitcher_id=new_pitcher_id,
    )
    key = _signal_key(signal_identity)
    intake_path = _signal_intake_path(output_root, key)
    start_time = parse_canonical_utc(schedule.scheduled_start_utc)
    signal_time = parse_canonical_utc(signal_identity["signal_observed_at_utc"])
    requested_as_of = format_canonical_utc(parse_canonical_utc(as_of_utc))
    if not signal_time <= parse_canonical_utc(requested_as_of) < start_time:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: reprice must be prospective")

    model_id, model_fingerprint, *_ = load_champion_model_parameters(repository_root)
    before_model = before.get("model_identity")
    if (
        not isinstance(before_model, Mapping)
        or before_model.get("model_role") != "CHAMPION"
        or before_model.get("model_id") != model_id
        or before_model.get("model_fingerprint") != model_fingerprint
    ):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: current Champion differs from before authority")

    if intake_path.exists() or intake_path.is_symlink():
        payload, metadata = _read_signal_intake(
            intake_path, expected_signal_identity=signal_identity
        )
        if metadata.get("before_run_id") != before_run_id:
            raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: existing intake has a different before authority")
        if not _same_json(metadata.get("probable_pitcher_state"), _state_for(schedule)):
            raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: existing intake has a different starter state")
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: signal-keyed intake has no eligible prediction")
        created = create_p50c_prediction_run(
            repository_root,
            pregame_input=intake_path,
            run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
            run_root=run_root,
            created_at_utc=metadata["reprice_as_of_utc"],
            target_date=schedule.official_date,
        )
        return created.run_id, intake_path, metadata["reprice_as_of_utc"]

    metadata = {
        "schema_version": _AUTO_REPRICE_SCHEMA,
        "signal_key": key,
        "signal_identity": signal_identity,
        "probable_pitcher_state": _state_for(schedule),
        "reprice_as_of_utc": requested_as_of,
        "before_run_id": before_run_id,
    }
    try:
        result = execute_daily_moneyline_prediction_freeze(
            target_date=schedule.official_date,
            as_of_utc=requested_as_of,
            repository_root=repository_root,
            output_root=output_root,
            game_type=schedule.game_type,
            intake_path=intake_path,
            target_game_identity=game_identity,
            expected_probable_pitcher_state=_state_for(schedule),
            extra_source_manifest={_AUTO_REPRICE_SCHEMA: metadata},
            opener=opener,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if str(error).startswith(MODEL_REPRICE_NOT_AVAILABLE):
            raise
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: {error}") from error
    _read_signal_intake(intake_path, expected_signal_identity=signal_identity)
    if (
        result.target_games_count != 1
        or result.eligible_predictions_count != 1
        or result.run_id is None
    ):
        reasons = [str(row.get("reason")) for row in result.exclusions]
        detail = ",".join(reasons) if reasons else "selected game was not eligible"
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: {detail}")
    return result.run_id, intake_path, requested_as_of


def capture_information_edge_v1(
    *,
    schedule: StartingPitcherScheduleObservation,
    prediction_run_root: str | Path,
    observation_path: str | Path,
    before_run_id: str | None = None,
    after_run_id: str | None = None,
    tsl_quote: TslMoneylineQuote | None = None,
    reference_market_quote: ReferenceMarketQuote | None = None,
    captured_at_utc: str | None = None,
    repository_root: str | Path | None = None,
    output_root: str | Path | None = None,
    opener: Callable[..., bytes] | None = None,
) -> InformationEdgeCaptureResult:
    """Reprice one confirmed starter change, then record a complete market edge if available."""

    run_root = Path(prediction_run_root).resolve()
    repo_root = Path(repository_root or Path.cwd()).resolve()
    if output_root is not None:
        out_root = Path(output_root).resolve()
    elif run_root.name == "runs" and run_root.parent.name == P50C_REPORT_RELATIVE_PATH.name:
        out_root = run_root.parent.parent.parent
    else:
        out_root = run_root.parent
    path = Path(observation_path)
    captured = format_canonical_utc(
        parse_canonical_utc(captured_at_utc or _now_utc())
    )
    captured_time = parse_canonical_utc(captured)
    records = _read_observations(path)
    prior = _find_latest_game_record(records, schedule)
    if prior is None:
        baseline = _baseline_observation(schedule, captured)
        _append_only(path, baseline, records)
        return InformationEdgeCaptureResult("BASELINE_RECORDED", baseline)
    _validate_schedule_continuity(prior, schedule)
    prior_state = _state_fields(prior)
    if prior_state is None:
        raise RuntimeError("SIGNAL_BASELINE_INVALID: prior pitcher state is missing")

    current_state = _state_for(schedule)
    canonical_game_identity = _game_identity_from_schedule(schedule)
    latest_intake = _latest_signal_intake(
        output_root=out_root,
        game_identity=canonical_game_identity,
    )
    prior_signal_time = parse_canonical_utc(str(prior["signal_observed_at_utc"]))
    latest_metadata = latest_intake[2] if latest_intake is not None else None
    latest_signal_identity = (
        latest_metadata.get("signal_identity")
        if isinstance(latest_metadata, Mapping)
        else None
    )
    intake_state_is_latest = False
    if isinstance(latest_signal_identity, Mapping):
        intake_signal_time = parse_canonical_utc(
            str(latest_signal_identity["signal_observed_at_utc"])
        )
        intake_state_is_latest = intake_signal_time >= prior_signal_time
    effective_prior_state = (
        latest_metadata.get("probable_pitcher_state")
        if intake_state_is_latest and latest_metadata is not None
        else prior_state
    )
    effective_prior_signal_time = (
        parse_canonical_utc(str(latest_signal_identity["signal_observed_at_utc"]))
        if intake_state_is_latest and isinstance(latest_signal_identity, Mapping)
        else prior_signal_time
    )
    state_changed_sides = [
        side
        for side in ("HOME", "AWAY")
        if effective_prior_state.get(side) != current_state.get(side)
    ]
    if any(
        effective_prior_state.get(side) is None or current_state.get(side) is None
        for side in state_changed_sides
    ):
        # A partial ID transition cannot prove a known-to-known one-side change.
        baseline = _baseline_observation(schedule, captured)
        _append_only(path, baseline, records)
        return InformationEdgeCaptureResult("BASELINE_REFRESHED", baseline)
    changed_sides = state_changed_sides
    replay_edge: dict[str, Any] | None = None
    replay_intake: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
    signal_observed_at_utc: str
    if not changed_sides:
        replay_edge = _previous_edge_for_current_state(records, schedule)
        if replay_edge is not None:
            side = replay_edge.get("pitcher_side")
            previous_pitcher_id = replay_edge.get("previous_pitcher_player_id")
            new_pitcher_id = replay_edge.get("new_pitcher_player_id")
            signal_observed_at_utc = str(replay_edge.get("signal_observed_at_utc"))
            if before_run_id is None:
                before_prediction = replay_edge.get("before_prediction_identity")
                if isinstance(before_prediction, Mapping):
                    before_run_id = before_prediction.get("run_id")
        elif (
            intake_state_is_latest
            and latest_intake is not None
            and _same_json(latest_metadata.get("probable_pitcher_state"), current_state)
        ):
            replay_intake = latest_intake
            stored_signal = latest_metadata["signal_identity"]
            side = stored_signal.get("changed_side")
            previous_pitcher_id = stored_signal.get("previous_pitcher_player_id")
            new_pitcher_id = stored_signal.get("new_pitcher_player_id")
            signal_observed_at_utc = stored_signal.get("signal_observed_at_utc")
            if before_run_id is None:
                before_run_id = latest_metadata.get("before_run_id")
        else:
            # A missing-to-known or known-to-missing ID transition updates the
            # baseline, but cannot establish a real pitcher-for-pitcher change.
            if not _same_json(effective_prior_state, current_state):
                baseline = _baseline_observation(schedule, captured)
                _append_only(path, baseline, records)
                return InformationEdgeCaptureResult("BASELINE_REFRESHED", baseline)
            return InformationEdgeCaptureResult("NO_CHANGE", None)
    else:
        if len(changed_sides) != 1:
            raise RuntimeError("AMBIGUOUS_STARTING_PITCHER_CHANGE: exactly one side may change")
        side = changed_sides[0]
        previous_pitcher_id = effective_prior_state[side]
        new_pitcher_id = current_state[side]
        signal_observed_at_utc = schedule.signal_observed_at_utc
        if effective_prior_signal_time >= parse_canonical_utc(signal_observed_at_utc):
            raise RuntimeError("SIGNAL_TIME_ORDER_INVALID: new schedule response is not later than baseline")

    if (
        side not in ("HOME", "AWAY")
        or not previous_pitcher_id
        or not new_pitcher_id
        or str(previous_pitcher_id) == str(new_pitcher_id)
    ):
        raise RuntimeError("STARTING_PITCHER_CHANGE_INVALID: both official player IDs are required")
    if not before_run_id:
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: before run ID is required")

    before = _load_prediction(run_root=run_root, run_id=before_run_id, schedule=schedule)
    before_identity = before["game_identity"]
    signal_time = parse_canonical_utc(str(signal_observed_at_utc))
    start_time = parse_canonical_utc(schedule.scheduled_start_utc)
    before_created = parse_canonical_utc(str(before["prediction_authority"]["created_at_utc"]))
    if not before_created < signal_time:
        raise RuntimeError("NO_LOOKAHEAD_BEFORE_RUN: before prediction must precede signal")
    if signal_time > captured_time:
        raise RuntimeError("SIGNAL_AFTER_REPRICE_DECISION: signal must precede reprice decision")
    if captured_time >= start_time:
        raise RuntimeError("POST_START_OBSERVATION_REJECTED")

    auto_reprice = after_run_id is None
    intake_path: Path | None = None
    reprice_as_of_utc: str | None = None
    if auto_reprice:
        try:
            after_run_id, intake_path, reprice_as_of_utc = _create_or_reuse_after_run(
                schedule=schedule,
                before=before,
                before_run_id=before_run_id,
                side=str(side),
                previous_pitcher_id=str(previous_pitcher_id),
                new_pitcher_id=str(new_pitcher_id),
                signal_observed_at_utc=str(signal_observed_at_utc),
                output_root=out_root,
                repository_root=repo_root,
                run_root=run_root,
                as_of_utc=captured,
                opener=opener,
            )
        except RuntimeError:
            raise
    if not after_run_id:
        raise RuntimeError(f"{MODEL_REPRICE_NOT_AVAILABLE}: after run ID is unavailable")
    after = _load_prediction(run_root=run_root, run_id=after_run_id, schedule=schedule)
    after_identity = after["game_identity"]
    if not _same_json(before_identity, after_identity):
        raise RuntimeError("PREDICTION_GAME_IDENTITY_MISMATCH: before and after run identities differ")
    before_model = before.get("model_identity")
    after_model = after.get("model_identity")
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping) or any(
        before_model.get(field) != after_model.get(field)
        for field in ("model_role", "model_id", "model_fingerprint", "model_probability_source")
    ):
        raise RuntimeError("PREDICTION_MODEL_IDENTITY_MISMATCH: before and after Champion differ")
    after_created = parse_canonical_utc(
        str(after["prediction_authority"]["created_at_utc"])
    )
    if after_created < signal_time:
        raise RuntimeError("NO_LOOKAHEAD_AFTER_RUN: after prediction must be at or after signal")
    if after_created >= start_time:
        raise RuntimeError("POST_START_OBSERVATION_REJECTED")
    if auto_reprice and reprice_as_of_utc is not None and not signal_time <= after_created:
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: after authority is not prospective")
    if auto_reprice and reprice_as_of_utc is not None and after_created != parse_canonical_utc(
        reprice_as_of_utc
    ):
        raise RuntimeError("MODEL_REPRICE_NOT_AVAILABLE: after authority timestamp differs from intake")
    if tsl_quote is None and reference_market_quote is None:
        return InformationEdgeCaptureResult(
            "MODEL_REPRICED",
            None,
            after_run_id=after_run_id,
            intake_path=intake_path,
            reprice_as_of_utc=reprice_as_of_utc or format_canonical_utc(after_created),
        )
    if tsl_quote is None or reference_market_quote is None:
        raise RuntimeError("EDGE_MARKET_OBSERVATIONS_REQUIRED: TSL and reference quotes are required")
    if not _same_json(tsl_quote.game_identity, after_identity):
        raise RuntimeError("TSL_CANONICAL_GAME_IDENTITY_MISMATCH")
    reference = reference_market_quote.get_quote(dict(after_identity))
    if not isinstance(reference, ReferenceMarketQuoteObservation):
        raise RuntimeError("REFERENCE_MARKET_QUOTE_INVALID: port returned an unsupported quote")
    if not _same_json(reference.game_identity, after_identity):
        raise RuntimeError("REFERENCE_MARKET_GAME_IDENTITY_MISMATCH")

    signal_time = parse_canonical_utc(str(signal_observed_at_utc))
    start_time = parse_canonical_utc(str(after_identity["scheduled_start_utc"]))
    before_created = parse_canonical_utc(str(before["prediction_authority"]["created_at_utc"]))
    after_created = parse_canonical_utc(str(after["prediction_authority"]["created_at_utc"]))
    tsl_time = parse_canonical_utc(tsl_quote.observed_at_utc)
    reference_time = parse_canonical_utc(
        reference.retrieved_at_utc or reference.observed_at_utc
    )
    if reference.retrieved_at_utc is not None and reference_time != parse_canonical_utc(
        reference.observed_at_utc
    ):
        raise RuntimeError("REFERENCE_MARKET_RETRIEVAL_TIMESTAMP_MISMATCH")
    provider_reference_time = (
        parse_canonical_utc(reference.provider_observed_at_utc)
        if reference.provider_observed_at_utc is not None
        else None
    )
    captured_time = parse_canonical_utc(captured)
    if not before_created < signal_time:
        raise RuntimeError("NO_LOOKAHEAD_BEFORE_RUN: before prediction must precede signal")
    if after_created < signal_time:
        raise RuntimeError("NO_LOOKAHEAD_AFTER_RUN: after prediction must be at or after signal")
    if any(
        instant >= start_time
        for instant in (
            signal_time,
            before_created,
            after_created,
            tsl_time,
            reference_time,
            captured_time,
            *([provider_reference_time] if provider_reference_time is not None else []),
        )
    ):
        raise RuntimeError("POST_START_OBSERVATION_REJECTED")
    if parse_canonical_utc(tsl_quote.scheduled_start_utc) != start_time:
        raise RuntimeError("TSL_SCHEDULED_START_MISMATCH")

    p_home_before = _probability(before["model_identity"]["p_home"], "p_home_before")
    p_home_after = _probability(after["model_identity"]["p_home"], "p_home_after")
    tsl_home = _odds(tsl_quote.home_decimal_odds, "tsl_home_decimal_odds")
    tsl_away = _odds(tsl_quote.away_decimal_odds, "tsl_away_decimal_odds")
    reference_home = _odds(reference.home_decimal_odds, "reference_home_decimal_odds")
    reference_away = _odds(reference.away_decimal_odds, "reference_away_decimal_odds")
    reference_home_implied = Decimal(1) / reference_home
    reference_away_implied = Decimal(1) / reference_away
    reference_home_probability = reference_home_implied / (
        reference_home_implied + reference_away_implied
    )
    model_reprice_delta = p_home_after - p_home_before
    home_gross_ev = p_home_after * tsl_home - Decimal(1)
    away_gross_ev = (Decimal(1) - p_home_after) * tsl_away - Decimal(1)
    minutes_to_start = _minutes_to_start(start_time, signal_time)

    before_authority = before["prediction_authority"]
    after_authority = after["prediction_authority"]
    reference_market_observation = {
        "snapshot_id": reference.snapshot_id,
        "observed_at_utc": format_canonical_utc(reference_time),
        "home_decimal_odds": _decimal_text(reference_home),
        "away_decimal_odds": _decimal_text(reference_away),
        "reference_home_probability": _decimal_text(reference_home_probability),
    }
    for field_name in (
        "source_provider",
        "source_event_id",
        "bookmaker_key",
        "market_key",
        "provider_observed_at_utc",
        "retrieved_at_utc",
        "raw_response_sha256",
    ):
        field_value = getattr(reference, field_name, None)
        if field_value is not None:
            reference_market_observation[field_name] = field_value

    observation = InformationEdgeObservation.create(
        {
            "schema_version": INFORMATION_EDGE_SCHEMA_VERSION,
            "record_type": EDGE_OBSERVATION,
            "game_identity": dict(after_identity),
            "schedule_identity": schedule.schedule_identity(),
            "signal_type": STARTING_PITCHER_CHANGE,
            "signal_observed_at_utc": format_canonical_utc(signal_time),
            "pitcher_side": side,
            "previous_pitcher_player_id": str(previous_pitcher_id),
            "new_pitcher_player_id": str(new_pitcher_id),
            "probable_pitcher_state": current_state,
            "model_probability_before": _decimal_text(p_home_before),
            "model_probability_after": _decimal_text(p_home_after),
            "p_home_before": _decimal_text(p_home_before),
            "p_home_after": _decimal_text(p_home_after),
            "selected_probabilities": {
                "before": {
                    "HOME": _decimal_text(p_home_before),
                    "AWAY": _decimal_text(Decimal(1) - p_home_before),
                },
                "after": {
                    "HOME": _decimal_text(p_home_after),
                    "AWAY": _decimal_text(Decimal(1) - p_home_after),
                },
            },
            "before_prediction_identity": {
                "run_id": before_authority["run_id"],
                "prediction_fingerprint": before["prediction_fingerprint"],
                "created_at_utc": before_authority["created_at_utc"],
            },
            "after_prediction_identity": {
                "run_id": after_authority["run_id"],
                "prediction_fingerprint": after["prediction_fingerprint"],
                "created_at_utc": after_authority["created_at_utc"],
            },
            "model_repriced_at_utc": format_canonical_utc(after_created),
            "tsl_observation": {
                "snapshot_id": tsl_quote.snapshot_id,
                "observed_at_utc": format_canonical_utc(tsl_time),
                "home_decimal_odds": _decimal_text(tsl_home),
                "away_decimal_odds": _decimal_text(tsl_away),
            },
            "reference_market_observation": reference_market_observation,
            "metrics": {
                "model_reprice_delta": _decimal_text(model_reprice_delta),
                "model_vs_reference_home_edge": _decimal_text(
                    p_home_after - reference_home_probability
                ),
                "tsl_home_gross_ev": _decimal_text(home_gross_ev),
                "tsl_away_gross_ev": _decimal_text(away_gross_ev),
                "minutes_to_start": _decimal_text(minutes_to_start),
            },
            "captured_at_utc": captured,
        }
    )
    try:
        _append_only(path, observation, records)
    except RuntimeError as error:
        if auto_reprice and str(error).startswith("OBSERVATION_DUPLICATE:"):
            existing = next(
                row for row in records if row.get("observation_id") == observation.observation_id
            )
            return InformationEdgeCaptureResult(
                "EDGE_REUSED",
                InformationEdgeObservation.create(existing),
                after_run_id=after_run_id,
                intake_path=intake_path,
                reprice_as_of_utc=reprice_as_of_utc or format_canonical_utc(after_created),
            )
        raise

    # Advance the baseline only after a complete edge was appended. The edge
    # record itself remains available to replay the same signal with a later
    # distinct market snapshot.
    refreshed_records = [*records, observation.to_dict()]
    updated_baseline = _baseline_observation(schedule, captured)
    try:
        _append_only(path, updated_baseline, refreshed_records)
    except RuntimeError as error:
        # A concurrent or prior identical baseline is harmless after the edge
        # append; duplicate edge detection remains the authoritative boundary.
        if not str(error).startswith("OBSERVATION_DUPLICATE:"):
            raise
    return InformationEdgeCaptureResult(
        "EDGE_RECORDED",
        observation,
        after_run_id=after_run_id,
        intake_path=intake_path,
        reprice_as_of_utc=reprice_as_of_utc or format_canonical_utc(after_created),
    )


__all__ = (
    "InformationEdgeCaptureResult",
    "MODEL_REPRICE_NOT_AVAILABLE",
    "TslMoneylineQuote",
    "capture_information_edge_v1",
)
