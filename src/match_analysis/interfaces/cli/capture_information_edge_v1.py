"""Local-input CLI for INFORMATION_EDGE_CAPTURE_V1."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from ...application.ports.reference_market_quote import (
    ReferenceMarketQuoteObservation,
)
from ...application.use_cases.information_edge_capture_v1 import (
    TslMoneylineQuote,
    capture_information_edge_v1,
)
from ...application.use_cases.p50c_prediction_run_ledger import (
    P50C_REPORT_RELATIVE_PATH,
    read_json_object,
)
from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc
from ...infrastructure.sources.mlb_starting_pitcher_signal import (
    normalize_starting_pitcher_schedule,
)
from ...infrastructure.sources.tsl_moneyline_acquisition import (
    normalize_tsl_moneyline_games,
)
from ...infrastructure.sources.tsl_moneyline_history import TSL_LOCAL_TIMEZONE


def _load_object(path: Path) -> dict[str, Any]:
    value = read_json_object(path)
    return value


class _LocalReferenceMarketQuote:
    def __init__(self, quote: ReferenceMarketQuoteObservation) -> None:
        self._quote = quote

    def get_quote(self, game_identity: Mapping[str, Any]) -> ReferenceMarketQuoteObservation:
        if self._quote.game_identity != dict(game_identity):
            raise ValueError("REFERENCE_MARKET_GAME_IDENTITY_MISMATCH")
        return self._quote


def _load_tsl_quote(path: Path, schedule_observation: Any) -> TslMoneylineQuote:
    payload = _load_object(path)
    provider_game_id = payload.get("provider_game_id")
    game_identity = payload.get("game_identity")
    game = payload.get("game")
    fetched_at = payload.get("fetched_at")
    if not isinstance(game_identity, Mapping) or not isinstance(game, Mapping):
        raise ValueError("TSL_LOCAL_INPUT_INVALID: game_identity and game objects are required")
    if provider_game_id != schedule_observation.provider_game_id:
        raise ValueError("TSL_CANONICAL_GAME_IDENTITY_MISMATCH")
    if game_identity.get("provider_game_id") != provider_game_id:
        raise ValueError("TSL_CANONICAL_GAME_IDENTITY_MISMATCH")
    if not isinstance(fetched_at, str):
        raise ValueError("TSL_LOCAL_INPUT_INVALID: fetched_at is required")
    schedule_start = parse_canonical_utc(schedule_observation.scheduled_start_utc)
    target_date = schedule_start.astimezone(TSL_LOCAL_TIMEZONE).date()
    normalized = normalize_tsl_moneyline_games(
        [game], fetched_at=fetched_at, target_date=target_date
    )
    if len(normalized.observations) != 1:
        raise ValueError("TSL_LOCAL_INPUT_INVALID: expected exactly one normalized Moneyline")
    return TslMoneylineQuote.from_p32a_observation(
        normalized.observations[0],
        provider_game_id=provider_game_id,
        game_identity=game_identity,
    )


def _load_reference_quote(path: Path) -> _LocalReferenceMarketQuote:
    payload = _load_object(path)
    game_identity = payload.get("game_identity")
    if not isinstance(game_identity, Mapping):
        raise ValueError("REFERENCE_MARKET_LOCAL_INPUT_INVALID: game_identity is required")
    quote = ReferenceMarketQuoteObservation.create(
        snapshot_id=str(payload.get("snapshot_id", "")),
        game_identity=game_identity,
        observed_at_utc=str(payload.get("observed_at_utc", "")),
        home_decimal_odds=payload.get("home_decimal_odds"),
        away_decimal_odds=payload.get("away_decimal_odds"),
    )
    return _LocalReferenceMarketQuote(quote)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a prospective MLB starting-pitcher information edge from local inputs."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--schedule-json", type=Path, required=True)
    parser.add_argument("--provider-game-id", required=True)
    parser.add_argument("--game-number", type=int)
    parser.add_argument("--before-run-id")
    parser.add_argument("--after-run-id")
    parser.add_argument("--tsl-quote-json", type=Path)
    parser.add_argument("--reference-quote-json", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    read_clock = clock or (lambda: datetime.now(UTC))
    try:
        schedule_bytes = args.schedule_json.read_bytes()
        # This is the local receipt time for the supplied schedule response.
        received_at = format_canonical_utc(read_clock())
        schedule_payload = json.loads(schedule_bytes.decode("utf-8"))
        if not isinstance(schedule_payload, Mapping):
            raise ValueError("MLB_SCHEDULE_LOCAL_INPUT_INVALID: expected a JSON object")
        schedule_observation = normalize_starting_pitcher_schedule(
            schedule_payload,
            provider_game_id=args.provider_game_id,
            game_number=args.game_number,
            signal_observed_at_utc=received_at,
        )
        if args.after_run_id is not None and args.before_run_id is None:
            raise ValueError("EDGE_INPUTS_REQUIRED: an after run requires a before run")
        if (args.tsl_quote_json is None) != (args.reference_quote_json is None):
            raise ValueError("EDGE_INPUTS_REQUIRED: provide both market quote files")
        if args.tsl_quote_json is not None and args.before_run_id is None:
            raise ValueError("EDGE_INPUTS_REQUIRED: market observations require a before run")
        tsl_quote = (
            _load_tsl_quote(args.tsl_quote_json, schedule_observation)
            if args.tsl_quote_json is not None
            else None
        )
        reference_quote = (
            _load_reference_quote(args.reference_quote_json)
            if args.reference_quote_json is not None
            else None
        )
        prediction_run_root = (
            repository_root / P50C_REPORT_RELATIVE_PATH / "runs"
        )
        observation_path = (
            repository_root / "report" / "information_edge_capture_v1" / "edge_observations.jsonl"
        )
        result = capture_information_edge_v1(
            schedule=schedule_observation,
            prediction_run_root=prediction_run_root,
            observation_path=observation_path,
            before_run_id=args.before_run_id,
            after_run_id=args.after_run_id,
            tsl_quote=tsl_quote,
            reference_market_quote=reference_quote,
            captured_at_utc=format_canonical_utc(read_clock()),
            repository_root=repository_root,
            output_root=repository_root,
        )
        output: dict[str, Any] = {"status": result.status}
        if result.after_run_id is not None:
            output["after_run_id"] = result.after_run_id
        if result.intake_path is not None:
            output["intake_path"] = str(result.intake_path)
        if result.reprice_as_of_utc is not None:
            output["reprice_as_of_utc"] = result.reprice_as_of_utc
        if result.observation is not None:
            output["observation"] = result.observation.to_dict()
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI test
    raise SystemExit(main())


__all__ = ("build_parser", "main")
