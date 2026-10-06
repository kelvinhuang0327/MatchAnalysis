"""Bind legacy relief rows to explicit archived capture evidence, without dates-as-times."""

from collections import Counter, defaultdict
from datetime import date
from hashlib import sha256
import json
from typing import Any, Mapping

from .build_bullpen_research_snapshot import build_bullpen_research_snapshot, INPUT_SCHEMA_VERSION
from ...baseball.domain.bullpen_research_snapshot import canonical_bytes
from ...baseball.domain.canonical_utc import parse_canonical_utc, format_canonical_utc


ADAPTER_SCHEMA = "bullpen.legacy_adapter.v1"
EVIDENCE_SCHEMA = "bullpen.legacy_capture_evidence.v1"
FORMATS = ("phase56-context", "phase63-jsonl", "three-day-aggregate-jsonl")
CAPTURE_FIELDS = {
    "source_row_index", "raw_record_sha256", "provider_namespace", "provider_game_id",
    "game_number", "pitcher_id", "game_date", "usage_date", "usage_date_basis",
    "started_at_utc", "ended_at_utc", "response_received_at_utc", "persisted_at_utc",
    "source_id", "source_kind",
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field}: require integer >= {minimum}")
    return value


def _source_rows(raw: bytes, source_format: str) -> list[dict[str, Any]]:
    result = []
    if source_format == "phase56-context":
        context = load_json(raw)
        if not isinstance(context, dict) or set(context) - {"home_schedule", "away_schedule"}:
            raise ValueError("phase56-context requires only home_schedule/away_schedule")
        for side in ("home", "away"):
            rows = context.get(f"{side}_schedule", [])
            if not isinstance(rows, list):
                raise ValueError("legacy schedule must be an array")
            for row in rows:
                result.append({"index": len(result) + 1, "row": row, "side": side,
                               "hash": sha256(canonical_bytes({"side": side, "row": row})).hexdigest()})
    else:
        for index, line in enumerate(raw.splitlines(keepends=True), 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at physical line {index}")
            result.append({"index": index, "row": load_json(line), "side": None,
                           "hash": sha256(line).hexdigest()})
    if any(not isinstance(item["row"], dict) for item in result):
        raise ValueError("legacy source rows must be objects")
    return result


def adapt_legacy_bullpen_evidence(
    source_bytes: bytes, *, source_format: str, target: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a research snapshot, normalized input and complete row decisions.

    Capture metadata is explicit caller evidence, not independently authenticated.
    Its source-file/physical-row hashes are verified. File mtimes, generated dates,
    PIT flags, composite legacy game names and estimated league averages are not
    used to invent identity, pitching dates or historical availability.
    """
    if not isinstance(source_bytes, bytes) or source_format not in FORMATS:
        raise ValueError("explicit bytes and supported source_format required")
    source_hash = sha256(source_bytes).hexdigest()
    records = _source_rows(source_bytes, source_format)
    normalized = {"schema_version": INPUT_SCHEMA_VERSION, "target": dict(target),
                  "appearances": [], "coverage": []}
    build_bullpen_research_snapshot(normalized)  # Validate target before examining history.
    cutoff = parse_canonical_utc(target["as_of_utc"])
    namespace = target["provider_namespace"]
    teams = {target["home_team_id"]: target["identity"]["home_participant"],
             target["away_team_id"]: target["identity"]["away_participant"]}
    evidence = dict(evidence) if evidence is not None else {}
    if evidence:
        if set(evidence) - {"schema_version", "source_sha256", "team_mappings", "record_evidence", "coverage"}:
            raise ValueError("unexpected capture evidence fields")
        if evidence.get("schema_version") != EVIDENCE_SCHEMA or evidence.get("source_sha256") != source_hash:
            raise ValueError("capture evidence schema/source-file SHA mismatch")
    for field in ("team_mappings", "record_evidence", "coverage"):
        if not isinstance(evidence.get(field, []), list):
            raise ValueError(f"{field} must be an array")
    mappings = {}
    for mapping in evidence.get("team_mappings", []):
        if not isinstance(mapping, dict) or set(mapping) != {"legacy_team_name", "provider_namespace", "team_id", "team_code"}:
            raise ValueError("explicit complete team mapping required")
        name = mapping["legacy_team_name"]
        if not isinstance(name, str) or not name or name in mappings:
            raise ValueError("duplicate/empty legacy team mapping")
        if mapping["provider_namespace"] != namespace or teams.get(mapping["team_id"]) != mapping["team_code"]:
            raise ValueError("legacy team mapping conflicts with target identity")
        _integer(mapping["team_id"], "mapped team_id", 1)
        mappings[name] = mapping
    bindings = {}
    for binding in evidence.get("record_evidence", []):
        if not isinstance(binding, dict) or set(binding) - CAPTURE_FIELDS:
            raise ValueError("unexpected record capture evidence fields")
        index = _integer(binding.get("source_row_index"), "source_row_index", 1)
        if index > len(records) or index in bindings:
            raise ValueError("duplicate/out-of-range capture binding index")
        if binding.get("raw_record_sha256") != records[index - 1]["hash"]:
            raise ValueError("capture evidence raw record hash mismatch")
        bindings[index] = binding
    decisions = []
    candidates = defaultdict(list)
    tainted = defaultdict(list)
    for item in records:
        row, index = item["row"], item["index"]
        decision = {"source_row_index": index, "raw_record_sha256": item["hash"], "reason": None}
        decisions.append(decision)
        team_id = None
        if source_format == "phase56-context":
            team_id = target[f"{item['side']}_team_id"]
            decision["reason"] = "DATE_ONLY_TEAM_AGGREGATE_NO_PITCHER_OR_CAPTURE_EVIDENCE"
        elif source_format == "three-day-aggregate-jsonl":
            decision["reason"] = "ROLLING_TEAM_AGGREGATE_CANNOT_RECOVER_INDIVIDUAL_APPEARANCES"
        elif ("estimated" in row and row["estimated"] is not False) or row.get("source") == "schedule_proxy_fallback":
            decision["reason"] = "ESTIMATED_SCHEDULE_PROXY_NOT_OBSERVED_USAGE"
        elif index not in bindings:
            decision["reason"] = "MISSING_CAPTURE_EVIDENCE"
        elif row.get("team") not in mappings:
            decision["reason"] = "MISSING_EXPLICIT_TEAM_IDENTITY_MAPPING"
        else:
            binding = bindings[index]
            mapping = mappings[row["team"]]
            team_id = mapping["team_id"]
            try:
                missing = CAPTURE_FIELDS - set(binding)
                if missing:
                    raise ValueError(f"missing capture fields: {sorted(missing)}")
                if row.get("source") != "mlb_stats_api_boxscore":
                    raise ValueError("unsupported legacy source label")
                game_id = row.get("source_game_id")
                pitcher_id = _integer(row.get("pitcher_id"), "pitcher_id", 1)
                _integer(binding.get("pitcher_id"), "capture pitcher_id", 1)
                if (not isinstance(game_id, str) or not game_id or binding.get("provider_game_id") != game_id
                        or binding.get("pitcher_id") != pitcher_id or binding.get("provider_namespace") != namespace
                        or binding.get("game_date") != row.get("game_date")):
                    raise ValueError("capture/game/pitcher/date identity mismatch")
                order = _integer(row.get("appeared_order"), "appeared_order", 1)
                flags = [row.get(field) for field in ("starter_flag", "opener_flag", "reliever_flag")]
                if any(not isinstance(flag, bool) for flag in flags):
                    raise ValueError("explicit role flags required")
                if (flags[0] and (flags[1] or flags[2])) or (order > 1 and (flags[0] or flags[1] or not flags[2])):
                    raise ValueError("conflicting pitcher order/role flags")
                if order == 1:
                    decision["reason"] = "FIRST_PITCHER_OR_OPENER_NOT_BULLPEN"
                    continue
                outs = _integer(row.get("outs_recorded"), "outs_recorded")
                for optional in ("pitches_thrown", "earned_runs"):
                    if row.get(optional) is not None:
                        _integer(row[optional], optional)
                if row.get("high_leverage") is not None and not isinstance(row["high_leverage"], bool):
                    raise ValueError("invalid raw leverage flag")
                received = parse_canonical_utc(binding.get("response_received_at_utc"))
                persisted = parse_canonical_utc(binding.get("persisted_at_utc"))
                started = parse_canonical_utc(binding.get("started_at_utc"))
                ended = parse_canonical_utc(binding.get("ended_at_utc"))
                if not started < ended <= received <= persisted:
                    raise ValueError("capture chronology inconsistent with completed appearance")
                if binding.get("source_kind") not in {"SYNTHETIC_FIXTURE", "ARCHIVED_LOCAL_PUBLIC_CAPTURE"}:
                    raise ValueError("explicit archived/synthetic source kind required")
                usage_date = date.fromisoformat(binding.get("usage_date"))
                if usage_date < date.fromisoformat(row["game_date"]) or binding.get("usage_date_basis") != "SOURCE_CONFIRMED_SINGLE_DAY":
                    raise ValueError("actual single-day pitching usage evidence required")
                appearance = {
                    "provider_namespace": namespace, "provider_game_id": game_id,
                    "game_number": _integer(binding.get("game_number"), "game_number", 1),
                    "official_date": row["game_date"], "team_id": team_id, "team_code": mapping["team_code"],
                    "pitcher_id": pitcher_id, "role": "RELIEF", "usage_date": binding["usage_date"],
                    "usage_date_basis": binding["usage_date_basis"],
                    "started_at_utc": format_canonical_utc(started), "ended_at_utc": format_canonical_utc(ended),
                    "available_at_utc": format_canonical_utc(persisted), "outs": outs,
                    "pitches": row.get("pitches_thrown"), "earned_runs": row.get("earned_runs"),
                    "high_leverage": row.get("high_leverage"),
                    "source_id": binding.get("source_id"), "source_kind": binding["source_kind"],
                    "source_fingerprint": item["hash"],
                }
                # Reuse the strict normalized contract; timestamps are explicit, not defaulted.
                single = build_bullpen_research_snapshot({**normalized, "appearances": [appearance]})
                rejection = single.to_projection()["excluded_records"]
                reason = rejection[0]["reason"] if rejection else None
                identity_key = (namespace, game_id, pitcher_id)
                identity_shape = (team_id, appearance["game_number"], row["game_date"], binding["usage_date"])
                candidates[identity_key].append({"appearance": appearance, "available": persisted,
                                                 "decision": decision, "shape": identity_shape,
                                                 "reason": reason})
                decision["reason"] = reason or "CANDIDATE_PRE_CUTOFF"
                continue
            except (ValueError, TypeError, KeyError) as exc:
                decision["reason"] = f"MISSING_OR_INVALID_CAPTURE_OR_USAGE_EVIDENCE: {exc}"
        if decision["reason"] is not None:
            affected = [team_id] if team_id is not None else list(teams)
            for affected_team in affected:
                tainted[affected_team].append(index)

    # Select immutable captured versions as-of; a later revision cannot amend past snapshots.
    for group in candidates.values():
        if len({item["shape"] for item in group}) != 1:
            raise ValueError("conflicting revision game/team/usage identity")
        eligible = [item for item in group if item["reason"] is None]
        if not eligible:
            for item in group:
                if item["reason"] not in {"TARGET_GAME_EXCLUDED", "FUTURE_OFFICIAL_DATE", "FUTURE_USAGE_DATE", "OUTSIDE_14_DAY_HISTORY"}:
                    tainted[item["appearance"]["team_id"]].append(item["decision"]["source_row_index"])
            continue
        latest_time = max(item["available"] for item in eligible)
        latest = [item for item in eligible if item["available"] == latest_time]
        if len({canonical_bytes({key: value for key, value in item["appearance"].items()
                                if key not in {"source_id", "source_kind", "source_fingerprint"}})
                for item in latest}) > 1:
            for item in group:
                item["decision"]["reason"] = "AMBIGUOUS_SAME_TIMESTAMP_REVISION"
                tainted[item["appearance"]["team_id"]].append(item["decision"]["source_row_index"])
            continue
        chosen = latest[0]
        normalized["appearances"].append(chosen["appearance"])
        for item in group:
            if item is chosen:
                item["decision"]["reason"] = "ADMITTED"
            elif item["reason"] is None:
                item["decision"]["reason"] = "DUPLICATE_CAPTURE_COLLAPSED" if item["available"] == latest_time else "SUPERSEDED_PRE_CUTOFF"
            elif item["reason"] == "NOT_AVAILABLE_BEFORE_CUTOFF":
                item["decision"]["reason"] = "POST_CUTOFF_REVISION_IGNORED_USING_ARCHIVED_PRE_CUTOFF"

    coverage_diagnostics = []
    for original in evidence.get("coverage", []):
        if not isinstance(original, dict):
            raise ValueError("coverage records must be objects")
        row = dict(original)
        indexes = sorted(set(tainted.get(row.get("team_id"), [])))
        unsupported = source_format != "phase63-jsonl"
        if indexes or unsupported:
            row["complete"] = False
        normalized["coverage"].append(row)
        coverage_diagnostics.append({"team_id": row.get("team_id"), "source_declared_complete": original.get("complete"),
                                     "effective_complete": row.get("complete"), "blocking_source_row_indexes": indexes,
                                     "reason": "LEGACY_FORMAT_CANNOT_PROVIDE_RELIEF_COMPLETENESS" if unsupported else
                                               "REJECTED_SOURCE_ROWS_REQUIRE_UNKNOWN_COVERAGE" if indexes else
                                               "UNCHANGED_CALLER_COVERAGE_ASSERTION_NOT_INDEPENDENTLY_VERIFIED"})
    covered = {item["team_id"] for item in coverage_diagnostics}
    for team_id in sorted(set(teams) - covered):
        coverage_diagnostics.append({"team_id": team_id, "source_declared_complete": None,
                                     "effective_complete": False,
                                     "blocking_source_row_indexes": sorted(set(tainted.get(team_id, []))),
                                     "reason": "NO_COVERAGE_EVIDENCE_SUPPLIED"})
    snapshot = build_bullpen_research_snapshot(normalized)
    counts = dict(sorted(Counter(item["reason"] for item in decisions).items()))
    return {
        "schema_version": ADAPTER_SCHEMA, "research_only": True,
        "source_format": source_format, "source_file_sha256": source_hash,
        "raw_record_hash_basis": "CANONICAL_CONTEXT_SIDE_AND_ROW_WITH_LF" if source_format == "phase56-context" else "PHYSICAL_JSONL_LINE_INCLUDING_LINE_ENDING",
        "capture_metadata_semantics": "EXPLICIT_CALLER_ARCHIVE_ASSERTIONS;HASH_BINDINGS_VERIFIED;TIMESTAMP_AUTHENTICITY_NOT_INDEPENDENTLY_VERIFIED",
        "evidence_canonical_sha256": sha256(canonical_bytes(evidence)).hexdigest(),
        "capture_metadata_assertions": evidence,
        "source_row_count": len(records), "selected_relief_appearance_count": len(normalized["appearances"]),
        "decision_counts": counts, "row_decisions": decisions, "coverage_diagnostics": coverage_diagnostics,
        "normalized_input": normalized, "snapshot_sha256": snapshot.fingerprint(), "snapshot": snapshot.to_projection(),
    }
