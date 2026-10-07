"""Outcome-blind, full-archive historical price eligibility; no forecast or betting."""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from .adapt_legacy_bullpen_evidence import load_json
from .generate_tsl_moneyline_edge_batch import TSL_TO_MLB_TEAM_ABBREVIATION
from ...baseball.domain.canonical_utc import parse_canonical_utc, format_canonical_utc
from ...infrastructure.sources.p39a_tsl_market_snapshot import _moneyline_prices
from ...infrastructure.sources.tsl_moneyline_acquisition import MLB_ZH_TO_TSL_CODE


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return sha256_file(handle)


def sha256_file(handle):
    digest = sha256()
    while block := handle.read(1024*1024):
        digest.update(block)
    return digest.hexdigest()


def _canonical_code(code):
    return TSL_TO_MLB_TEAM_ABBREVIATION.get(code)


def _official_games(raw):
    games = []
    for day in load_json(raw)["dates"]:
        for g in day["games"]:
            pk, number = g["gamePk"], g["gameNumber"]
            if type(pk) is not int or pk <= 0 or type(number) is not int or number <= 0:
                raise ValueError("positive official game ID/number required")
            home, away = (g["teams"][s]["team"] for s in ("home", "away"))
            if type(home["id"]) is not int or type(away["id"]) is not int or home["id"] == away["id"]:
                raise ValueError("distinct official team IDs required")
            for team in (home, away):
                if not all(isinstance(team[k], str) and team[k] for k in ("abbreviation", "name")):
                    raise ValueError("official team name/code required")
            official_date = date.fromisoformat(g["officialDate"])
            if str(official_date.year) != str(g["season"]):
                raise ValueError("official season/date conflict")
            start = parse_canonical_utc(g["gameDate"])
            games.append({"provider_namespace": "MLB_STATS_API", "provider_game_id": str(pk),
                          "game_pk": pk, "game_number": number, "season": official_date.year,
                          "official_date": str(official_date), "scheduled_start_utc": format_canonical_utc(start),
                          "game_type": g["gameType"],
                          "home_team": {k:home[k] for k in ("id", "name", "abbreviation")},
                          "away_team": {k:away[k] for k in ("id", "name", "abbreviation")},
                          "final": g["status"]["abstractGameState"] == "Final"
                                   and g["status"]["detailedState"] == "Final"})
    by_id = defaultdict(list)
    for game in games:
        if game not in by_id[game["provider_game_id"]]:
            by_id[game["provider_game_id"]].append(game)
    result = []
    for variants in by_id.values():
        for game in variants:
            game["official_capture_conflict"] = len(variants) > 1
            result.append(game)
    return result  # Never project score, winner or model prediction fields.


def audit_tsl_archive_eligibility(source: Path, *, expected_source_sha256: str,
                                  schedule_raw: bytes, expected_schedule_sha256: str,
                                  prior_game_ids=()):
    if sha256(schedule_raw).hexdigest() != expected_schedule_sha256:
        raise ValueError("official schedule hash mismatch")
    if _file_hash(source) != expected_source_sha256:
        raise ValueError("TSL source hash mismatch")
    games = _official_games(schedule_raw)
    prior_game_ids = set(prior_game_ids)
    by_pair = defaultdict(list)
    for g in games:
        by_pair[(g["home_team"]["abbreviation"], g["away_team"]["abbreviation"])].append(g)
    decisions, candidates = [], []
    digest = sha256()
    with Path(source).open("rb") as handle:
        for index, line in enumerate(handle, 1):
            digest.update(line)
            r = load_json(line)
            d = {"source_row_index": index, "reason": None}
            decisions.append(d)
            home_name, away_name = r.get("home_team_name"), r.get("away_team_name")
            if home_name not in MLB_ZH_TO_TSL_CODE or away_name not in MLB_ZH_TO_TSL_CODE:
                d["reason"] = "OUTSIDE_EXPLICIT_MLB_TEAM_NAMES"; continue
            if r.get("source") != "TSL_BLOB3RD" or r.get("sport_league") != "MLB":
                d["reason"] = "UNSUPPORTED_SOURCE_OR_MISSING_MLB_LEAGUE"; continue
            if r.get("is_pregame") is not True:
                d["reason"] = "SOURCE_NOT_PREGAME"; continue
            try:
                source_start = parse_canonical_utc(r.get("game_time"))
                observed = parse_canonical_utc(r.get("fetched_at"))
            except (ValueError, TypeError):
                d["reason"] = "MISSING_OR_INVALID_SOURCE_TIMESTAMP"; continue
            if observed >= source_start:
                d["reason"] = "POST_SOURCE_START"; continue
            codes = tuple(_canonical_code(r.get(f"{s}_code")) for s in ("home", "away"))
            named_codes = tuple(_canonical_code(MLB_ZH_TO_TSL_CODE[n]) for n in (home_name, away_name))
            if None in codes or codes != named_codes or codes[0] == codes[1]:
                d["reason"] = "SOURCE_TEAM_CODE_NAME_CONFLICT_OR_MISSING_CODE"; continue
            matches = [g for g in by_pair[codes]
                       if abs((parse_canonical_utc(g["scheduled_start_utc"])-source_start).total_seconds()) <= 3600]
            if not matches:
                d["reason"] = "NO_OFFICIAL_SAME_TEAMS_START_WITHIN_3600_SECONDS"; continue
            if any(g["official_capture_conflict"] for g in matches):
                d["reason"] = "OFFICIAL_CAPTURE_GAME_REVISION_AMBIGUITY"; continue
            if len(matches) != 1:
                d["reason"] = "AMBIGUOUS_OFFICIAL_GAME_WITHIN_TIME_WINDOW"; continue
            g = matches[0]
            d["provider_game_id"] = g["provider_game_id"]
            if (any(str(r[k]) != str(g[k]) for k in ("provider_game_id", "game_pk", "game_number", "official_date", "game_type") if k in r)
                    or not isinstance(r.get("match_id"), str) or not r["match_id"].strip()):
                d["reason"] = "SOURCE_EXPLICIT_GAME_IDENTITY_CONFLICT"; continue
            if g["game_type"] != "R":
                d["reason"] = "OFFICIAL_NOT_REGULAR_SEASON"; continue
            if not g["final"]:
                d["reason"] = "OFFICIAL_NOT_FINAL"; continue
            official_start = parse_canonical_utc(g["scheduled_start_utc"])
            if observed >= official_start:
                d["reason"] = "POST_OFFICIAL_START"; continue
            status, home, away, price_reason = _moneyline_prices(r)
            if price_reason:
                d["reason"] = price_reason; continue
            # Legacy producer collapses two-way and three-way markets to MNL.
            # Reuse P39's exact two-named-side rule; never attach a tie market.
            markets = [m for m in r["markets"] if isinstance(m, dict) and m.get("marketCode") == "MNL"
                       and isinstance(m.get("outcomes"), list) and len(m["outcomes"]) == 2
                       and {o.get("outcomeName") for o in m["outcomes"] if isinstance(o, dict)} == {home_name, away_name}]
            if len(markets) != 1 or any(o.get("specialBetValue") not in (None, "") for o in markets[0]["outcomes"]):
                d["reason"] = "AMBIGUOUS_OR_NONSTANDARD_MNL_LINE"; continue
            fingerprint = sha256(line).hexdigest()
            d.update(reason="ADMISSIBLE_OBSERVATION", raw_record_sha256=fingerprint)
            candidates.append({**g, "source_row_index": index, "raw_record_sha256": fingerprint,
                               "source_match_id": r["match_id"], "source_game_time_utc": format_canonical_utc(source_start),
                               "market_observed_at_utc": format_canonical_utc(observed), "provider_observed_at_utc": None,
                               "home_decimal_price": str(home), "away_decimal_price": str(away),
                               "market_no_vig_home_probability": str((1/home)/(1/home+1/away)),
                               "start_crosswalk_delta_seconds": abs((official_start-source_start).total_seconds()),
                               "quote_age_seconds_at_official_start": (official_start-observed).total_seconds(),
                               "prior_p37_game": g["provider_game_id"] in prior_game_ids})
    if digest.hexdigest() != expected_source_sha256 or _file_hash(source) != expected_source_sha256:
        raise ValueError("TSL source changed during audit")
    by_id, by_game, by_time = defaultdict(set), defaultdict(list), defaultdict(set)
    for q in candidates:
        key = (q["provider_game_id"], q["game_number"])
        by_id[q["source_match_id"]].add(key)
        by_game[key].append(q)
        by_time[(key, q["market_observed_at_utc"])].add((Decimal(q["home_decimal_price"]), Decimal(q["away_decimal_price"])))
    conflicts = {key for keys in by_id.values() if len(keys)>1 for key in keys}
    conflicts.update(key for (key, stamp), prices in by_time.items() if len(prices)>1)
    selected = []
    for key, quotes in by_game.items():
        if key in conflicts:
            for q in quotes: decisions[q["source_row_index"]-1]["reason"] = "GAME_SOURCE_ID_OR_SAME_TIME_PRICE_CONFLICT"
            continue
        latest = max(parse_canonical_utc(q["market_observed_at_utc"]) for q in quotes)
        q = min((q for q in quotes if parse_canonical_utc(q["market_observed_at_utc"]) == latest),
                key=lambda q: q["raw_record_sha256"])
        selected.append(q)
        for item in quotes:
            decisions[item["source_row_index"]-1]["reason"] = ("SELECTED_LATEST_VALID_OBSERVATION" if item is q
                                                               else "VALID_NONSELECTED_OBSERVATION")
    selected.sort(key=lambda q: (q["scheduled_start_utc"], q["provider_game_id"], q["game_number"]))
    all_dates = [q["official_date"] for q in selected]
    new_dates = [q["official_date"] for q in selected if q["official_date"] > "2026-06-13"]
    return {"schema_version": "tsl.archive_price_eligibility.v1", "source_sha256": expected_source_sha256,
            "schedule_sha256": expected_schedule_sha256,
            "rules": {"population": "ALL_SOURCE_ROWS; NO_OUTCOME_OR_MODEL_BASED_SELECTION",
                      "crosswalk": "EXACT_MAPPED_HOME_AWAY_CODES_AND_UNIQUE_OFFICIAL_GAME_WITHIN_3600_SECONDS; ALL_AMBIGUITY_REJECTED",
                      "quote": "UNIQUE_EXACT_TWO_NAMED_SIDE_MNL; THREE_WAY_MNL_VARIANTS_IGNORED; BOTH_PRICES_SAME_RAW_RECORD; FETCH_STRICTLY_BEFORE_BOTH_STARTS",
                      "conflicts": "REJECT_WHOLE_GAME_IF_ONE_SOURCE_MATCH_ID_MAPS_TO_MULTIPLE_GAMES_OR_SAME_TIME_PRICES_CONFLICT",
                      "selection": "LATEST_VALID_TIMESTAMP_PER_FULL_GAME_IDENTITY; IDENTICAL_PRICE_TIES_BY_RAW_RECORD_SHA256"},
            "limitations": ["LOCAL_ARCHIVE_FETCH_TIMESTAMP; NO_PROVIDER_PUBLISH_TIMESTAMP_OR_INDEPENDENT_AUTHENTICATION",
                            "OFFICIAL_SCHEDULE_CAPTURE_IS_RETROSPECTIVE; NOT_A_HISTORICAL_PREGAME_SCHEDULE_CAPTURE",
                            "PRICE_ELIGIBILITY_ONLY; NO_CHAMPION_FREEZE_OR_FEATURE_AVAILABILITY_PROOF; NOT_UNSEEN",
                            "LEGACY_MNL_CODE_COLLAPSES_THREE_WAY_VARIANTS; NO_PERIOD_OR_CLOSING_METADATA_INVENTED",
                            "NO_DECISION_TIME_FRESHNESS_OR_CLOSING_QUOTE_ASSERTION; NO_ROI_OR_BETTING"],
            "summary": {"source_rows": len(decisions), "official_games_in_capture": len(games),
                        "selected_regular_final_games": len(selected), "reason_counts": dict(Counter(d["reason"] for d in decisions)),
                        "official_date_range": [min(all_dates), max(all_dates)] if all_dates else None,
                        "games_by_official_month": dict(Counter(d[:7] for d in all_dates)),
                        "overlap_p37_game_count": sum(q["prior_p37_game"] for q in selected),
                        "after_existing_p37_window": {"fixed_boundary_exclusive": "2026-06-13", "game_count": len(new_dates),
                                                     "date_range": [min(new_dates), max(new_dates)] if new_dates else None}},
            "games": selected, "row_decisions": decisions}
