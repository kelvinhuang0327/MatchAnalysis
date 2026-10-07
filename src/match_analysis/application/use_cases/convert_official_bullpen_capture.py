"""Project a hash-bound official final feed into the existing legacy adapter contract."""

from hashlib import sha256
import json
import re
from zoneinfo import ZoneInfo

from .adapt_legacy_bullpen_evidence import EVIDENCE_SCHEMA, load_json
from ...baseball.domain.canonical_utc import format_canonical_utc, parse_canonical_utc


def baseball_innings_to_outs(value):
    # Baseball .1/.2 are one/two outs, not tenths. Floats are ambiguous evidence.
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.[012]", value):
        raise ValueError("explicit baseball innings notation required")
    innings, outs = value.split(".")
    return 3*int(innings)+int(outs)


def convert_official_bullpen_capture(raw: bytes, capture: dict, schedule: dict):
    fingerprint = sha256(raw).hexdigest()
    if capture.get("sha256") != fingerprint or capture.get("bytes") != len(raw):
        raise ValueError("official capture byte hash/length mismatch")
    feed = load_json(raw)
    pk = feed["gamePk"]
    if not capture["url"].startswith(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?"):
        raise ValueError("capture URL/provider game identity mismatch")
    gd = feed["gameData"]
    if (str(pk) != schedule["provider_game_id"] or pk != schedule["game_pk"]
            or gd["game"]["pk"] != pk or gd["game"]["type"] != "R"
            or gd["datetime"]["officialDate"] != schedule["official_date"]
            or gd["datetime"]["dateTime"] != schedule["scheduled_start_utc"]
            or gd["status"]["abstractGameState"] != "Final"
            or gd["status"]["detailedState"] != "Final" or schedule["final"] is not True
            or type(schedule["game_number"]) is not int or schedule["game_number"] < 1):
        raise ValueError("final regular-season schedule identity required")
    zone = ZoneInfo(gd["venue"]["timeZone"]["id"])
    plays = feed["liveData"]["plays"]["allPlays"]
    if not plays:
        raise ValueError("actual completed play timestamps required")
    times = []
    for play in plays:
        start = parse_canonical_utc(play["about"]["startTime"])
        end = parse_canonical_utc(play["about"]["endTime"])
        if start > end:
            raise ValueError("play chronology inconsistent")
        times.extend((start, end))
    days = {t.astimezone(zone).date().isoformat() for t in times}
    if len(days) != 1:
        raise ValueError("multi-day/suspended game cannot infer individual pitching usage dates")
    usage_date = next(iter(days))
    received = parse_canonical_utc(capture["response_received_at_utc"])
    persisted = parse_canonical_utc(capture["persisted_at_utc"])
    first, last = min(times), max(times)
    if not first < last <= received <= persisted:
        raise ValueError("final plays must precede actual capture availability")
    rows, bindings, mappings, totals = [], [], [], {}
    for side in ("home", "away"):
        team = gd["teams"][side]
        expected = schedule[f"{side}_team"]
        if any(team[key] != expected[key] for key in ("id", "name", "abbreviation")):
            raise ValueError("official feed/schedule team identity mismatch")
        mappings.append({"legacy_team_name": team["name"], "provider_namespace": "MLB_STATS_API",
                         "team_id": team["id"], "team_code": team["abbreviation"]})
        box = feed["liveData"]["boxscore"]["teams"][side]
        if box["team"]["id"] != team["id"] or len(set(box["pitchers"])) != len(box["pitchers"]):
            raise ValueError("boxscore team/pitcher order conflict")
        totals[side] = {"pitchers": len(box["pitchers"]), "relief_outs": 0, "relief_pitches": 0}
        for order, pitcher in enumerate(box["pitchers"], 1):
            player = box["players"][f"ID{pitcher}"]
            stats = player["stats"]["pitching"]  # Never use seasonStats.
            if player["person"]["id"] != pitcher:
                raise ValueError("pitcher identity mismatch")
            outs = baseball_innings_to_outs(stats["inningsPitched"])
            for field in ("numberOfPitches", "earnedRuns", "gamesStarted"):
                if type(stats[field]) is not int or stats[field] < 0:
                    raise ValueError("integer official pitching count required")
            if stats["gamesStarted"] != int(order == 1):
                raise ValueError("boxscore starter disagrees with pitcher order")
            row = {"source_game_id": str(pk), "game_date": schedule["official_date"],
                   "team": team["name"], "pitcher_id": pitcher, "appeared_order": order,
                   "starter_flag": order == 1, "opener_flag": False, "reliever_flag": order > 1,
                   "source": "mlb_stats_api_boxscore", "outs_recorded": outs,
                   "pitches_thrown": stats["numberOfPitches"], "earned_runs": stats["earnedRuns"],
                   "high_leverage": None}
            line = (json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n").encode()
            rows.append(line)
            bindings.append({"source_row_index": len(rows), "raw_record_sha256": sha256(line).hexdigest(),
                             "provider_namespace": "MLB_STATS_API", "provider_game_id": str(pk),
                             "game_number": schedule["game_number"], "pitcher_id": pitcher,
                             "game_date": schedule["official_date"], "usage_date": usage_date,
                             "usage_date_basis": "SOURCE_CONFIRMED_SINGLE_DAY",
                             "started_at_utc": format_canonical_utc(first), "ended_at_utc": format_canonical_utc(last),
                             "response_received_at_utc": capture["response_received_at_utc"],
                             "persisted_at_utc": capture["persisted_at_utc"],
                             "source_id": capture["url"], "source_kind": "ARCHIVED_LOCAL_PUBLIC_CAPTURE"})
            if order > 1:
                totals[side]["relief_outs"] += outs
                totals[side]["relief_pitches"] += stats["numberOfPitches"]
    source = b"".join(rows)
    return {"source_bytes": source,
            "capture_evidence": {"schema_version": EVIDENCE_SCHEMA, "source_sha256": sha256(source).hexdigest(),
                                 "team_mappings": mappings, "record_evidence": bindings, "coverage": []},
            "diagnostics": {"raw_official_sha256": fingerprint, "capture": capture,
                            "schedule_record_sha256": sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
                            "usage_date": usage_date, "venue_timezone": zone.key, "team_totals": totals,
                            "timestamp_semantics": "Conservative whole-game completed-play bounds, not individual appearance times; all observed plays on one venue-local day.",
                            "coverage_status": "SINGLE_GAME_ONLY; NO_COMPLETE_3_7_14_DAY_COVERAGE_ASSERTION",
                            "leverage_status": "UNKNOWN; NO_INVENTED_HIGH_LEVERAGE_FLAG"}}
