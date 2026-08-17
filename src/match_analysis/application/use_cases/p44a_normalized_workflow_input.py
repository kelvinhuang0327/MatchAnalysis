"""Source-independent P43/P44 paper-workflow input boundary.

The workflow core consumes these normalized objects. It does not open
historical P37 prediction or P39 market artifact paths. Existing P40 row
types carry pregame semantics; the result record is the smallest extra
surface needed for scores, status, time, and source identity, then
projects onto P40AOutcomeRow without changing settlement math.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import parse_canonical_utc
from .p40a_moneyline_paper_bet_pass import (
    P40AAuthority,
    P40AMarketRow,
    P40AOutcomeRow,
    P40APredictionRow,
    _decimal,
    _positive_int,
    _sha,
    _text,
)


P44A_PREGAME_INPUT_SCHEMA = "p44a.normalized_pregame_input.v1"
P44A_RESULT_INPUT_SCHEMA = "p44a.normalized_result_input.v1"
P44A_REPORT_RELATIVE_PATH = Path("report/p44a_source_agnostic_prospective_input")

FORBIDDEN_PREGAME_FIELD_NAMES = frozenset(
    {
        "actual_winner",
        "target_home_win",
        "final_score",
        "final_winner",
        "final_game_outcome",
        "home_score",
        "away_score",
        "boxscore",
        "settlement",
        "settlement_status",
        "evaluation",
        "feedback",
        "incumbent_correct",
        "challenger_correct",
        "incumbent_brier_contribution",
        "challenger_brier_contribution",
        "incumbent_log_loss_contribution",
        "challenger_log_loss_contribution",
        "champion_log_loss_contribution",
        "paired_brier_delta",
        "net_paper_units",
        "descriptive_roi",
    }
)

_PREDICTION_REQUIRED = (
    "p37_fold_id",
    "p37_window",
    "p37_prediction_row_id",
    "provider_namespace",
    "provider_game_id",
    "game_pk",
    "game_number",
    "scheduled_start_utc",
    "champion_model_id",
    "champion_model_fingerprint",
    "champion_home_probability",
    "challenger_model_id",
    "challenger_model_fingerprint",
    "challenger_home_probability",
)
_MARKET_REQUIRED = (
    "p37_fold_id",
    "p37_window",
    "p37_prediction_row_id",
    "provider_namespace",
    "provider_game_id",
    "game_pk",
    "game_number",
    "official_date",
    "scheduled_start_utc",
    "home_team",
    "away_team",
    "home_team_code",
    "away_team_code",
    "market_snapshot_id",
    "market_observed_at_utc",
    "local_fetched_at_utc",
    "source_match_id",
    "home_decimal_odds",
    "away_decimal_odds",
)
_RESULT_REQUIRED = (
    "prediction_row_id",
    "provider_namespace",
    "provider_game_id",
    "game_number",
    "status",
    "home_score",
    "away_score",
    "result_observed_at_utc",
    "source_identity",
)


def _duplicate_rejecting_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicate_rejecting_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
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


def collect_forbidden_pregame_fields(value: Any) -> list[str]:
    """Return outcome/settlement/evaluation keys found anywhere in a payload."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_PREGAME_FIELD_NAMES:
                found.append(str(key))
            found.extend(collect_forbidden_pregame_fields(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.extend(collect_forbidden_pregame_fields(item))
    return found


def reject_pregame_outcome_fields(value: Any) -> None:
    forbidden = collect_forbidden_pregame_fields(value)
    if forbidden:
        unique = ", ".join(sorted(set(forbidden)))
        raise ValueError(
            "P44A_PREGAME_OUTCOME_FIELDS_REJECTED "
            f"normalized pregame input must not contain {unique}"
        )


def _require_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _require_object_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty JSON array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be a JSON object")
        rows.append(item)
    return rows


def _parse_prediction_row(row: Mapping[str, Any]) -> P40APredictionRow:
    missing = [key for key in _PREDICTION_REQUIRED if key not in row]
    if missing:
        raise ValueError(f"normalized pregame prediction missing keys: {missing}")
    probability = _decimal(
        row.get("champion_home_probability"), field_name="champion_home_probability"
    )
    if not Decimal("0") < probability < Decimal("1"):
        raise ValueError("champion_home_probability is outside the probability domain")
    challenger_probability = _decimal(
        row.get("challenger_home_probability"), field_name="challenger_home_probability"
    )
    if not Decimal("0") < challenger_probability < Decimal("1"):
        raise ValueError("challenger_home_probability is outside the probability domain")
    scheduled_start_utc = _text(
        row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
    )
    parse_canonical_utc(scheduled_start_utc)
    return P40APredictionRow(
        p37_fold_id=_text(row.get("p37_fold_id"), field_name="p37_fold_id"),
        p37_window=_text(row.get("p37_window"), field_name="p37_window"),
        p37_prediction_row_id=_sha(
            row.get("p37_prediction_row_id"), field_name="p37_prediction_row_id"
        ),
        provider_namespace=_text(
            row.get("provider_namespace"), field_name="provider_namespace"
        ),
        provider_game_id=_text(
            row.get("provider_game_id"), field_name="provider_game_id"
        ),
        game_pk=_positive_int(row.get("game_pk"), field_name="game_pk"),
        game_number=_positive_int(row.get("game_number"), field_name="game_number"),
        scheduled_start_utc=scheduled_start_utc,
        champion_model_id=_text(
            row.get("champion_model_id"), field_name="champion_model_id"
        ),
        champion_model_fingerprint=_sha(
            row.get("champion_model_fingerprint"),
            field_name="champion_model_fingerprint",
        ),
        champion_home_probability=probability,
        challenger_model_id=_text(
            row.get("challenger_model_id"), field_name="challenger_model_id"
        ),
        challenger_model_fingerprint=_sha(
            row.get("challenger_model_fingerprint"),
            field_name="challenger_model_fingerprint",
        ),
        challenger_home_probability=challenger_probability,
    )


def _assert_strictly_pregame(*, observed_at: str, fetched_at: str, scheduled_start: str) -> None:
    parse_canonical_utc(observed_at)
    parse_canonical_utc(fetched_at)
    parse_canonical_utc(scheduled_start)
    if observed_at >= scheduled_start:
        raise ValueError("P43A market observation is not strictly pregame")
    if fetched_at >= scheduled_start:
        raise ValueError("P43A local fetch is not strictly pregame")


def _parse_market_row(row: Mapping[str, Any]) -> P40AMarketRow:
    missing = [key for key in _MARKET_REQUIRED if key not in row]
    if missing:
        raise ValueError(f"normalized pregame market missing keys: {missing}")
    scheduled_start_utc = _text(
        row.get("scheduled_start_utc"), field_name="scheduled_start_utc"
    )
    market_observed_at_utc = _text(
        row.get("market_observed_at_utc"), field_name="market_observed_at_utc"
    )
    local_fetched_at_utc = _text(
        row.get("local_fetched_at_utc"), field_name="local_fetched_at_utc"
    )
    _assert_strictly_pregame(
        observed_at=market_observed_at_utc,
        fetched_at=local_fetched_at_utc,
        scheduled_start=scheduled_start_utc,
    )
    home_odds = _decimal(row.get("home_decimal_odds"), field_name="home_decimal_odds")
    away_odds = _decimal(row.get("away_decimal_odds"), field_name="away_decimal_odds")
    if home_odds <= Decimal("1") or away_odds <= Decimal("1"):
        raise ValueError("normalized pregame market contains a malformed decimal price")
    return P40AMarketRow(
        p37_fold_id=_text(row.get("p37_fold_id"), field_name="p37_fold_id"),
        p37_window=_text(row.get("p37_window"), field_name="p37_window"),
        p37_prediction_row_id=_sha(
            row.get("p37_prediction_row_id"), field_name="p37_prediction_row_id"
        ),
        provider_namespace=_text(
            row.get("provider_namespace"), field_name="provider_namespace"
        ),
        provider_game_id=_text(
            row.get("provider_game_id"), field_name="provider_game_id"
        ),
        game_pk=_positive_int(row.get("game_pk"), field_name="game_pk"),
        game_number=_positive_int(row.get("game_number"), field_name="game_number"),
        official_date=_text(row.get("official_date"), field_name="official_date"),
        scheduled_start_utc=scheduled_start_utc,
        home_team=_text(row.get("home_team"), field_name="home_team"),
        away_team=_text(row.get("away_team"), field_name="away_team"),
        home_team_code=_text(row.get("home_team_code"), field_name="home_team_code"),
        away_team_code=_text(row.get("away_team_code"), field_name="away_team_code"),
        market_snapshot_id=_sha(
            row.get("market_snapshot_id"), field_name="market_snapshot_id"
        ),
        market_observed_at_utc=market_observed_at_utc,
        local_fetched_at_utc=local_fetched_at_utc,
        source_match_id=_text(row.get("source_match_id"), field_name="source_match_id"),
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
    )


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class NormalizedPregameInput:
    """Source-independent pregame prediction + market bundle."""

    source_identity: str
    prediction_rows: tuple[P40APredictionRow, ...]
    market_rows: tuple[P40AMarketRow, ...]
    exclusion_rows: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    authority_hashes: dict[str, str]
    p37_summary: dict[str, Any]
    p39_summary: dict[str, Any]
    p39_source_manifest: dict[str, Any]

    def to_authority(self, repository_root: str | Path) -> P40AAuthority:
        if self.outcome_payload_present():
            raise RuntimeError("P43A pregame freeze received a postgame payload")
        return P40AAuthority(
            repository_root=Path(repository_root).resolve(),
            p39_summary=dict(self.p39_summary),
            p39_source_manifest=dict(self.p39_source_manifest),
            p37_summary=dict(self.p37_summary),
            market_rows=self.market_rows,
            prediction_rows=self.prediction_rows,
            outcome_rows=(),
            source_manifest=dict(self.source_manifest),
        )

    def outcome_payload_present(self) -> bool:
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": P44A_PREGAME_INPUT_SCHEMA,
            "source_identity": self.source_identity,
            "predictions": [prediction_row_to_payload(row) for row in self.prediction_rows],
            "markets": [market_row_to_payload(row) for row in self.market_rows],
            "exclusions": [dict(row) for row in self.exclusion_rows],
            "source_manifest": dict(self.source_manifest),
            "authority_hashes": dict(self.authority_hashes),
        }


def prediction_row_to_payload(row: P40APredictionRow) -> dict[str, Any]:
    return {
        "p37_fold_id": row.p37_fold_id,
        "p37_window": row.p37_window,
        "p37_prediction_row_id": row.p37_prediction_row_id,
        "provider_namespace": row.provider_namespace,
        "provider_game_id": row.provider_game_id,
        "game_pk": row.game_pk,
        "game_number": row.game_number,
        "scheduled_start_utc": row.scheduled_start_utc,
        "champion_model_id": row.champion_model_id,
        "champion_model_fingerprint": row.champion_model_fingerprint,
        "champion_home_probability": _decimal_text(row.champion_home_probability),
        "challenger_model_id": row.challenger_model_id,
        "challenger_model_fingerprint": row.challenger_model_fingerprint,
        "challenger_home_probability": _decimal_text(row.challenger_home_probability),
    }


def market_row_to_payload(row: P40AMarketRow) -> dict[str, Any]:
    return {
        "p37_fold_id": row.p37_fold_id,
        "p37_window": row.p37_window,
        "p37_prediction_row_id": row.p37_prediction_row_id,
        "provider_namespace": row.provider_namespace,
        "provider_game_id": row.provider_game_id,
        "game_pk": row.game_pk,
        "game_number": row.game_number,
        "official_date": row.official_date,
        "scheduled_start_utc": row.scheduled_start_utc,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "home_team_code": row.home_team_code,
        "away_team_code": row.away_team_code,
        "market_snapshot_id": row.market_snapshot_id,
        "market_observed_at_utc": row.market_observed_at_utc,
        "local_fetched_at_utc": row.local_fetched_at_utc,
        "source_match_id": row.source_match_id,
        "home_decimal_odds": _decimal_text(row.home_decimal_odds),
        "away_decimal_odds": _decimal_text(row.away_decimal_odds),
    }


def parse_normalized_pregame_payload(payload: Mapping[str, Any]) -> NormalizedPregameInput:
    reject_pregame_outcome_fields(payload.get("predictions"))
    reject_pregame_outcome_fields(payload.get("markets"))
    reject_pregame_outcome_fields(payload.get("exclusions"))
    top_level_forbidden = sorted(
        set(payload).intersection(FORBIDDEN_PREGAME_FIELD_NAMES)
    )
    if top_level_forbidden:
        raise ValueError(
            "P44A_PREGAME_OUTCOME_FIELDS_REJECTED "
            f"normalized pregame input must not contain {', '.join(top_level_forbidden)}"
        )
    schema = payload.get("schema_version")
    if schema != P44A_PREGAME_INPUT_SCHEMA:
        raise ValueError(
            f"normalized pregame schema must be {P44A_PREGAME_INPUT_SCHEMA}, got {schema!r}"
        )
    source_identity = _text(payload.get("source_identity"), field_name="source_identity")
    if source_identity.lower() == "live":
        raise ValueError("normalized pregame input must not be labeled live")
    predictions = tuple(
        _parse_prediction_row(row)
        for row in _require_object_list(payload.get("predictions"), field_name="predictions")
    )
    markets = tuple(
        _parse_market_row(row)
        for row in _require_object_list(payload.get("markets"), field_name="markets")
    )
    raw_exclusions = payload.get("exclusions", [])
    if raw_exclusions is None:
        raw_exclusions = []
    if not isinstance(raw_exclusions, list):
        raise ValueError("exclusions must be a JSON array")
    exclusions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_exclusions):
        if not isinstance(item, dict):
            raise ValueError(f"exclusions[{index}] must be a JSON object")
        reject_pregame_outcome_fields(item)
        exclusions.append(dict(item))
    source_manifest = _require_mapping(
        payload.get("source_manifest"), field_name="source_manifest"
    )
    p37a = source_manifest.get("p37a")
    p39a = source_manifest.get("p39a")
    if not isinstance(p37a, Mapping) or not isinstance(p39a, Mapping):
        raise ValueError("source_manifest must carry opaque p37a/p39a identity hashes")
    _sha(p37a.get("p37_comparisons_sha256"), field_name="p37_comparisons_sha256")
    _sha(p39a.get("legacy_source_sha256"), field_name="legacy_source_sha256")
    authority_hashes = payload.get("authority_hashes", {})
    if authority_hashes is None:
        authority_hashes = {}
    if not isinstance(authority_hashes, dict):
        raise ValueError("authority_hashes must be a JSON object")
    prediction_by_id = {row.p37_prediction_row_id: row for row in predictions}
    seen_markets: set[str] = set()
    for market in markets:
        if market.p37_prediction_row_id in seen_markets:
            raise ValueError("normalized pregame market identities are not unique")
        seen_markets.add(market.p37_prediction_row_id)
        prediction = prediction_by_id.get(market.p37_prediction_row_id)
        if prediction is None:
            raise ValueError(
                "normalized market row has no prediction authority: "
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
                raise ValueError(f"normalized prediction/market identity mismatch in {field_name}")
    return NormalizedPregameInput(
        source_identity=source_identity,
        prediction_rows=predictions,
        market_rows=markets,
        exclusion_rows=tuple(exclusions),
        source_manifest=dict(source_manifest),
        authority_hashes={
            _text(key, field_name="authority_hash_key"): _text(
                value, field_name="authority_hash_value"
            )
            for key, value in authority_hashes.items()
        },
        p37_summary={},
        p39_summary={},
        p39_source_manifest={},
    )


def resolve_normalized_pregame_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        candidate = resolved / "pregame_input.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"normalized pregame directory has no pregame_input.json: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"normalized pregame input is missing: {resolved}")
    return resolved


def load_normalized_pregame_input(path: str | Path) -> NormalizedPregameInput:
    """Load a source-independent pregame bundle from JSON."""

    payload = _read_json_object(resolve_normalized_pregame_path(path))
    return parse_normalized_pregame_payload(payload)


def write_normalized_pregame_input(
    path: str | Path,
    pregame: NormalizedPregameInput,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination / "pregame_input.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != _json_bytes(pregame.to_payload()):
        raise RuntimeError(
            "P43A_CONFLICTING_EXISTING_ARTIFACT "
            f"{destination.name} already exists with a different payload"
        )
    if not destination.exists():
        destination.write_bytes(_json_bytes(pregame.to_payload()))
    return destination


@dataclass(frozen=True, slots=True)
class NormalizedResultRecord:
    """Independent final-result observation used only after freeze."""

    prediction_row_id: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    status: str
    home_score: int
    away_score: int
    result_observed_at_utc: str
    source_identity: str

    @property
    def actual_winner(self) -> str:
        if self.home_score > self.away_score:
            return "HOME"
        return "AWAY"

    @property
    def target_home_win(self) -> int:
        return 1 if self.actual_winner == "HOME" else 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": P44A_RESULT_INPUT_SCHEMA,
            "prediction_row_id": self.prediction_row_id,
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "game_number": self.game_number,
            "status": self.status,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "result_observed_at_utc": self.result_observed_at_utc,
            "source_identity": self.source_identity,
        }

    def to_p40_outcome(self) -> P40AOutcomeRow:
        return P40AOutcomeRow(
            p37_prediction_row_id=self.prediction_row_id,
            provider_game_id=self.provider_game_id,
            actual_winner=self.actual_winner,
            target_home_win=self.target_home_win,
        )


def _require_score(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer >= 0")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def parse_normalized_result_payload(row: Mapping[str, Any]) -> NormalizedResultRecord:
    missing = [key for key in _RESULT_REQUIRED if key not in row]
    if missing:
        raise ValueError(f"normalized result row missing keys: {missing}")
    schema = row.get("schema_version")
    if schema not in (None, P44A_RESULT_INPUT_SCHEMA):
        raise ValueError(
            f"normalized result schema must be {P44A_RESULT_INPUT_SCHEMA}, got {schema!r}"
        )
    status = _text(row.get("status"), field_name="status")
    if status != "FINAL":
        raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")
    home_score = _require_score(row.get("home_score"), field_name="home_score")
    away_score = _require_score(row.get("away_score"), field_name="away_score")
    if home_score == away_score:
        raise RuntimeError("P43A_NON_FINAL_RESULT_FAIL_CLOSED")
    observed_at = _text(
        row.get("result_observed_at_utc"), field_name="result_observed_at_utc"
    )
    parse_canonical_utc(observed_at)
    return NormalizedResultRecord(
        prediction_row_id=_sha(row.get("prediction_row_id"), field_name="prediction_row_id"),
        provider_namespace=_text(
            row.get("provider_namespace"), field_name="provider_namespace"
        ),
        provider_game_id=_text(
            row.get("provider_game_id"), field_name="provider_game_id"
        ),
        game_number=_positive_int(row.get("game_number"), field_name="game_number"),
        status=status,
        home_score=home_score,
        away_score=away_score,
        result_observed_at_utc=observed_at,
        source_identity=_text(row.get("source_identity"), field_name="source_identity"),
    )


def load_normalized_result_input(path: str | Path) -> tuple[NormalizedResultRecord, ...]:
    """Load independent final-result observations from JSONL."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")
    try:
        rows = _read_jsonl_objects(resolved)
    except ValueError as exc:
        message = str(exc)
        if "contains no rows" in message or "is missing" in message:
            raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED") from exc
        raise
    if not rows:
        raise RuntimeError("P43A_MISSING_RESULT_FAIL_CLOSED")
    records: list[NormalizedResultRecord] = []
    seen: dict[str, NormalizedResultRecord] = {}
    for row in rows:
        record = parse_normalized_result_payload(row)
        existing = seen.get(record.prediction_row_id)
        if existing is not None:
            raise RuntimeError("P43A_CONFLICTING_RESULT_REJECTED")
        seen[record.prediction_row_id] = record
        records.append(record)
    return tuple(records)


def project_normalized_results(
    records: Sequence[NormalizedResultRecord],
) -> tuple[P40AOutcomeRow, ...]:
    return tuple(record.to_p40_outcome() for record in records)


def write_normalized_result_input(
    path: str | Path,
    records: Sequence[NormalizedResultRecord],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _jsonl_bytes([record.to_payload() for record in records])
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError(
            "P43A_CONFLICTING_EXISTING_ARTIFACT "
            f"{destination.name} already exists with a different payload"
        )
    if not destination.exists():
        destination.write_bytes(content)
    return destination


__all__ = (
    "FORBIDDEN_PREGAME_FIELD_NAMES",
    "NormalizedPregameInput",
    "NormalizedResultRecord",
    "P44A_PREGAME_INPUT_SCHEMA",
    "P44A_REPORT_RELATIVE_PATH",
    "P44A_RESULT_INPUT_SCHEMA",
    "load_normalized_pregame_input",
    "load_normalized_result_input",
    "parse_normalized_pregame_payload",
    "parse_normalized_result_payload",
    "project_normalized_results",
    "reject_pregame_outcome_fields",
    "write_normalized_pregame_input",
    "write_normalized_result_input",
)
