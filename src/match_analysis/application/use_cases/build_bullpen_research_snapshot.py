"""Offline bullpen workloads using explicitly available pre-cutoff evidence.

No API request, ledger write, model feature-vector change, or fatigue coefficient
is performed here. Coverage is a caller-supplied source assertion, never inferred
from an empty list of appearances.
"""

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, Mapping

from ...baseball.domain.bullpen_research_snapshot import (
    BULLPEN_RESEARCH_SCHEMA_VERSION, BullpenResearchSnapshot, canonical_bytes,
)
from ...baseball.domain.canonical_utc import parse_canonical_utc, format_canonical_utc
from ...core.identity import MatchIdentity


INPUT_SCHEMA_VERSION = "bullpen.research_input.v1"
PROVENANCE_FIELDS = {"source_id", "source_kind", "source_fingerprint"}
TARGET_FIELDS = {
    "identity", "provider_namespace", "provider_game_id", "game_number",
    "official_date", "scheduled_start_utc", "as_of_utc", "home_team_id",
    "away_team_id", "source_schedule_observation_id",
}
APPEARANCE_FIELDS = {
    "provider_namespace", "provider_game_id", "game_number", "official_date",
    "team_id", "team_code", "pitcher_id", "role", "started_at_utc", "ended_at_utc",
    "available_at_utc", "usage_date", "usage_date_basis", "outs", "pitches", "earned_runs", "high_leverage",
} | PROVENANCE_FIELDS
COVERAGE_FIELDS = {
    "team_id", "team_code", "start_date", "through_utc", "available_at_utc", "complete",
} | PROVENANCE_FIELDS


def _fields(record: Mapping[str, Any], allowed: set[str], label: str) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label}: expected an object")
    if set(record) - allowed:
        raise ValueError(f"{label}: unexpected fields {sorted(set(record) - allowed)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field}: expected explicit trimmed text")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field}: expected integer >= {minimum}")
    return value


def _hash(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field}: expected lowercase SHA-256")
    return text


def _date(value: Any) -> date:
    text = _text(value, "official date")
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError("date must be canonical YYYY-MM-DD")
    return parsed


def _provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": _text(row.get("source_id"), "source_id"),
        "source_kind": _text(row.get("source_kind"), "source_kind"),
        "source_fingerprint": _hash(row.get("source_fingerprint"), "source_fingerprint"),
        "source_fingerprint_verification": "CALLER_SUPPLIED_NOT_RAW_SOURCE_VERIFIED",
    }


def build_bullpen_research_snapshot(bundle: Mapping[str, Any]) -> BullpenResearchSnapshot:
    """Build v1 research evidence from two-team, per-pitcher pitching records.

    Missing or unavailable historical records yield explicit exclusions and
    unknown affected windows. Conflicting identities/duplicates fail closed.
    Exact duplicate appearances collapse once; starters never count as relief.
    """
    _fields(bundle, {"schema_version", "target", "coverage", "appearances"}, "bundle")
    if bundle.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unexpected bullpen input schema")
    target = bundle["target"]
    _fields(target, TARGET_FIELDS, "target")
    identity = MatchIdentity(**target["identity"])
    if identity.sport != "baseball" or identity.league != "MLB":
        raise ValueError("bullpen research requires explicit MLB identity")
    namespace = _text(target.get("provider_namespace"), "provider_namespace")
    target_id = _text(target.get("provider_game_id"), "provider_game_id")
    game_number = _integer(target.get("game_number"), "game_number", 1)
    official_date = _date(target.get("official_date"))
    if identity.season != official_date.year:
        raise ValueError("target official date and season disagree")
    cutoff = parse_canonical_utc(target.get("as_of_utc"))
    start = parse_canonical_utc(target.get("scheduled_start_utc"))
    if cutoff >= start:
        raise ValueError("as_of_utc must be strictly before scheduled start")
    _hash(target.get("source_schedule_observation_id"), "source_schedule_observation_id")
    home_id = _integer(target.get("home_team_id"), "home_team_id", 1)
    away_id = _integer(target.get("away_team_id"), "away_team_id", 1)
    if home_id == away_id:
        raise ValueError("home/away team ids must differ")
    teams = {home_id: identity.home_participant, away_id: identity.away_participant}
    coverage_rows = bundle.get("coverage", [])
    appearance_rows = bundle.get("appearances", [])
    if not isinstance(coverage_rows, list) or not isinstance(appearance_rows, list):
        raise ValueError("coverage and appearances must be arrays")

    coverage: dict[int, dict[str, Any]] = {}
    for row in coverage_rows:
        _fields(row, COVERAGE_FIELDS, "coverage")
        team_id = _integer(row.get("team_id"), "coverage team_id", 1)
        if teams.get(team_id) != row.get("team_code"):
            raise ValueError("coverage team identity mismatch")
        if team_id in coverage:
            raise ValueError("duplicate team coverage declaration")
        if not isinstance(row.get("complete"), bool):
            raise ValueError("coverage complete must be explicit boolean")
        first_date = _date(row.get("start_date"))
        through = parse_canonical_utc(row.get("through_utc"))
        available = parse_canonical_utc(row.get("available_at_utc"))
        if available > cutoff or through > available or first_date > official_date:
            raise ValueError("coverage unavailable/future at cutoff")
        coverage[team_id] = {
            **dict(row), **_provenance(row),
            "available_at_utc": format_canonical_utc(available),
            "through_utc": format_canonical_utc(through),
        }

    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, int], bytes] = {}
    game_identities: dict[str, tuple[int, str]] = {}
    blocked_dates: dict[int, list[date | None]] = defaultdict(list)
    duplicate_count = 0

    for index, row in enumerate(appearance_rows, 1):
        _fields(row, APPEARANCE_FIELDS, "appearance")
        team_id = _integer(row.get("team_id"), "team_id", 1)
        if teams.get(team_id) != row.get("team_code") or row.get("provider_namespace") != namespace:
            raise ValueError(f"appearance {index}: provider/team identity mismatch")
        pitcher_id = _integer(row.get("pitcher_id"), "pitcher_id", 1)
        historical_id = _text(row.get("provider_game_id"), "historical provider_game_id")
        historical_number = _integer(row.get("game_number"), "historical game_number", 1)
        historical_date = _date(row.get("official_date"))
        key = (namespace, historical_id, pitcher_id)
        raw_bytes = canonical_bytes(row)
        if key in seen:
            if seen[key] != raw_bytes:
                raise ValueError(f"appearance {index}: conflicting duplicate pitcher/game identity")
            duplicate_count += 1
            continue
        seen[key] = raw_bytes
        game_key = (historical_number, historical_date.isoformat())
        if historical_id in game_identities and game_identities[historical_id] != game_key:
            raise ValueError("historical game id has conflicting date/game number")
        game_identities[historical_id] = game_key

        reason = None
        usage_day = None
        if historical_id == target_id:
            if historical_number != game_number or historical_date != official_date:
                raise ValueError("target provider game identity conflict")
            reason = "TARGET_GAME_EXCLUDED"
        elif historical_date > official_date:
            reason = "FUTURE_OFFICIAL_DATE"
        else:
            try:
                provenance = _provenance(row)
                usage_day = _date(row.get("usage_date"))
                if usage_day < historical_date or row.get("usage_date_basis") != "SOURCE_CONFIRMED_SINGLE_DAY":
                    raise ValueError("require source-confirmed single-day pitching usage date >= official date")
                started = parse_canonical_utc(row.get("started_at_utc"))
                ended = parse_canonical_utc(row.get("ended_at_utc"))
                available = parse_canonical_utc(row.get("available_at_utc"))
                if not started < ended <= available:
                    raise ValueError("require started_at_utc < ended_at_utc <= available_at_utc")
                _integer(row.get("outs"), "outs")
                for optional in ("pitches", "earned_runs"):
                    if row.get(optional) is not None:
                        _integer(row[optional], optional)
                if row.get("high_leverage") is not None and not isinstance(row["high_leverage"], bool):
                    raise ValueError("high_leverage must be boolean or null")
                if row.get("role") not in {"RELIEF", "STARTER"}:
                    raise ValueError("role must be RELIEF or STARTER")
                if usage_day > official_date:
                    reason = "FUTURE_USAGE_DATE"
                elif ended >= cutoff or available > cutoff:
                    reason = "NOT_AVAILABLE_BEFORE_CUTOFF"
                elif row["role"] == "STARTER":
                    reason = "STARTER_NOT_BULLPEN"
                elif usage_day < official_date - timedelta(days=14):
                    reason = "OUTSIDE_14_DAY_HISTORY"
                else:
                    accepted.append({
                        **dict(row), **provenance,
                        "started_at_utc": format_canonical_utc(started),
                        "ended_at_utc": format_canonical_utc(ended),
                        "available_at_utc": format_canonical_utc(available),
                    })
            except (TypeError, ValueError) as exc:
                reason = f"MISSING_OR_INVALID_HISTORY_EVIDENCE: {exc}"
        if reason is not None:
            exclusions.append({"input_row_index": index, "reason": reason,
                               "input_record_sha256": sha256(raw_bytes).hexdigest()})
            if reason.startswith("MISSING_OR_INVALID") or reason == "NOT_AVAILABLE_BEFORE_CUTOFF":
                blocked_dates[team_id].append(usage_day)

    accepted.sort(key=lambda row: (row["usage_date"], parse_canonical_utc(row["ended_at_utc"]),
                                   row["provider_game_id"], row["pitcher_id"]))
    team_features: dict[str, Any] = {}
    for team_id, team_code in teams.items():
        history = [row for row in accepted if row["team_id"] == team_id]
        declared = coverage.get(team_id)

        def completeness(days: int, *, same_day: bool = False) -> str | None:
            first = official_date if same_day else official_date - timedelta(days=days)
            last = official_date if same_day else official_date - timedelta(days=1)
            if declared is None or declared["complete"] is not True:
                return "NO_COMPLETE_COVERAGE_DECLARATION"
            if _date(declared["start_date"]) > first or parse_canonical_utc(declared["through_utc"]) < cutoff:
                return "COVERAGE_DOES_NOT_SPAN_WINDOW_THROUGH_CUTOFF"
            if any(day is None or first <= day <= last for day in blocked_dates[team_id]):
                return "REJECTED_OR_UNAVAILABLE_RECORD_IN_WINDOW"
            return None

        def select(days: int, *, same_day: bool = False) -> list[dict[str, Any]]:
            return [row for row in history if (
                _date(row["usage_date"]) == official_date if same_day else
                official_date - timedelta(days=days) <= _date(row["usage_date"]) < official_date
            )]

        def load(days: int, *, same_day: bool = False) -> dict[str, Any]:
            rows = select(days, same_day=same_day)
            unknown = completeness(days, same_day=same_day)
            pitch_missing = any(row.get("pitches") is None for row in rows)
            return {
                "outs": sum(row["outs"] for row in rows) if unknown is None else None,
                "relief_appearances": len(rows) if unknown is None else None,
                "pitches": (sum(row["pitches"] for row in rows)
                            if unknown is None and not pitch_missing else None),
                "status": unknown or "COMPLETE_AS_DECLARED",
                "pitches_status": unknown or ("MISSING_PITCH_COUNT" if pitch_missing else "COMPLETE_AS_DECLARED"),
            }

        recent14 = select(14)
        unknown14 = completeness(14)
        er_missing = any(row.get("earned_runs") is None for row in recent14)
        total_outs = sum(row["outs"] for row in recent14)
        era = (str(Decimal(27) * sum(row["earned_runs"] for row in recent14) / Decimal(total_outs))
               if unknown14 is None and not er_missing and total_outs > 0 else None)
        recent7 = select(7)
        leverage_missing = any(row.get("high_leverage") is None for row in recent7)
        leverage = (str(Decimal(sum(row["high_leverage"] for row in recent7)) / Decimal(len(recent7)))
                    if completeness(7) is None and recent7 and not leverage_missing else None)
        yesterday_ids = {row["pitcher_id"] for row in select(2)
                         if _date(row["usage_date"]) == official_date - timedelta(days=1)}
        before_ids = {row["pitcher_id"] for row in select(2)
                      if _date(row["usage_date"]) == official_date - timedelta(days=2)}
        recent_pitchers: dict[int, dict[str, Any]] = {}
        for row in history:
            recent_pitchers[row["pitcher_id"]] = row
        rest = [{
            "pitcher_id": pitcher_id, "last_observed_relief_date": row["usage_date"],
            "last_observed_relief_game_id": row["provider_game_id"],
            "full_calendar_rest_days": (max(0, (official_date - _date(row["usage_date"])).days - 1)
                                        if completeness(14) is None and completeness(0, same_day=True) is None else None),
        } for pitcher_id, row in sorted(recent_pitchers.items())]
        team_features[team_code] = {
            "team_id": team_id,
            "previous_3_calendar_days": load(3),
            "previous_7_calendar_days": load(7),
            "prior_same_day": load(0, same_day=True),
            "relievers_worked_both_previous_days": (len(yesterday_ids & before_ids)
                                                    if completeness(2) is None else None),
            "back_to_back_status": completeness(2) or "COMPLETE_AS_DECLARED",
            "era_previous_14_calendar_days": era,
            "era_status": unknown14 or ("MISSING_EARNED_RUNS" if er_missing else
                                         "COMPLETE_AS_DECLARED" if total_outs else "ZERO_OUTS_DENOMINATOR"),
            "high_leverage_appearance_fraction_previous_7_days": leverage,
            "leverage_status": completeness(7) or ("MISSING_LEVERAGE_FLAG" if leverage_missing else
                                                   "COMPLETE_AS_DECLARED" if recent7 else "NO_APPEARANCES_DENOMINATOR"),
            "recent_observed_relief_pitchers": rest,
            "rest_scope": "OBSERVED_TEAM_RELIEF_HISTORY_ONLY;NOT_CURRENT_ROSTER_OR_CROSS_TEAM_REST",
        }
    normalized_target = {**dict(target), "as_of_utc": format_canonical_utc(cutoff),
                         "scheduled_start_utc": format_canonical_utc(start), "game_number": game_number}
    return BullpenResearchSnapshot.from_projection({
        "schema_version": BULLPEN_RESEARCH_SCHEMA_VERSION,
        "research_only": True,
        "target": normalized_target,
        "teams": team_features,
        "window_semantics": "PREVIOUS_N_SOURCE_CONFIRMED_USAGE_DATES_EXCLUDING_TARGET_OFFICIAL_DATE;PRIOR_SAME_DAY_SEPARATE",
        "availability_rule": "started_at_utc < ended_at_utc < as_of_utc; ended_at_utc <= available_at_utc <= as_of_utc < scheduled_start_utc",
        "coverage_semantics": "SOURCE_ASSERTION_NOT_INDEPENDENT_COMPLETENESS_PROOF",
        "identity_semantics": "EXPLICIT_PROVIDER_GAME_TEAM_PITCHER_AND_CANONICAL_TEAM_MATCH;NO_NAME_INFERENCE",
        "input_canonical_sha256": sha256(canonical_bytes(bundle)).hexdigest(),
        "coverage_evidence": [coverage[key] for key in sorted(coverage)],
        "accepted_relief_evidence": accepted,
        "excluded_records": exclusions,
        "exact_duplicates_collapsed": duplicate_count,
        "model_integration": "NONE;P19A_MONEYLINE_FEATURE_VECTOR_UNCHANGED",
    })
