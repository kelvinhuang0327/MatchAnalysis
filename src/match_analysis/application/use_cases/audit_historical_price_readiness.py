"""Readiness and explicitly labelled reference scores over a fixed historical universe."""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from .adapt_legacy_bullpen_evidence import load_json
from .p50c_market_comparison import _metrics
from .run_moneyline_method_experiment import _paired
from ...baseball.domain.canonical_utc import parse_canonical_utc


def audit_historical_price_readiness(root: Path, protocol_raw: bytes, *, open_audit=False):
    protocol = load_json(protocol_raw)
    if protocol["schema_version"] != "historical_price_readiness_protocol.v1":
        raise ValueError("unknown readiness protocol")
    if protocol["baseline_controls"]["constant_home_probability"] != "0.5" or protocol["optimization_trials_this_batch"] != 0:
        raise ValueError("fixed reference control or trial accounting changed")
    raw = {}
    for name, fingerprint in protocol["sources"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source outside repository")
        blob = path.read_bytes()
        if sha256(blob).hexdigest() != fingerprint:
            raise ValueError(f"frozen readiness source hash mismatch: {name}")
        raw[name] = blob
    quotes = [load_json(line) for line in raw["examples/tsl_archive_eligibility_v1/eligible_games.jsonl"].splitlines()]
    bounds = protocol["population"]
    quotes = [q for q in quotes if bounds["start_official_date"] <= q["official_date"] <= bounds["end_official_date"]]
    if len(quotes) != bounds["expected_games"] or len({(q["provider_game_id"],q["game_number"]) for q in quotes}) != len(quotes):
        raise ValueError("fixed price universe count/identity changed")
    schedule_path = "data/fixtures/p23f2_official_2026_history/raw/schedule_2026-03-25_2026-06-30.json"
    selected_ids = {q["game_pk"] for q in quotes}
    official = {}
    for day in load_json(raw[schedule_path])["dates"]:
        for game in day["games"]:
            if game["gamePk"] in selected_ids:
                if game["gamePk"] in official:
                    raise ValueError("selected game has conflicting schedule versions")
                official[game["gamePk"]] = game
    capture_manifest = load_json(raw["report/p23f2_official_future_fold/source_manifest.json"])
    records = [r for r in capture_manifest["records"] if r["path"] == schedule_path]
    if len(records) != 1 or records[0]["sha256"] != protocol["sources"][schedule_path]:
        raise ValueError("schedule acquisition provenance mismatch")
    captured = parse_canonical_utc(records[0]["acquired_at_utc"])
    policy = load_json(raw["report/moneyline_paper_decision_v1/paper_state.json"])["policy_parameters"]
    model = protocol["sole_model_for_readiness"]
    artifact = load_json(raw["report/p22b_moneyline_challenger/model_artifact.json"])
    if artifact["model_id"] != model["model_id"] or artifact["artifact_fingerprint"] != model["fingerprint"]:
        raise ValueError("frozen Champion artifact identity mismatch")
    freezes = []
    for name, blob in raw.items():
        if name.endswith("/frozen_predictions.jsonl"):
            for line in blob.splitlines():
                row = load_json(line)
                identity = row.get("model_identity", {})
                if (identity.get("model_id") == model["model_id"]
                        and identity.get("model_fingerprint") == model["fingerprint"]
                        and identity.get("model_probability_source") == "calibrated_p_home"):
                    freezes.append(row)
    rows = []
    scores = {role:{"market":[],"constant":[]} for role in protocol["split"]}
    for q in quotes:
        game = official[q["game_pk"]]
        if (str(game["gamePk"]) != q["provider_game_id"] or game["gameNumber"] != q["game_number"]
                or game["officialDate"] != q["official_date"] or game["gameDate"] != q["scheduled_start_utc"]
                or game["gameType"] != "R" or game["status"]["detailedState"] != "Final"
                or any(game["teams"][side]["team"]["id"] != q[f"{side}_team"]["id"] for side in ("home","away"))):
            raise ValueError("price/official final game identity mismatch")
        roles = [role for role,(start,end) in protocol["split"].items() if start <= q["official_date"] <= end]
        if len(roles) != 1:
            raise ValueError("unregistered or overlapping time split")
        role = roles[0]
        start = parse_canonical_utc(q["scheduled_start_utc"])
        matching = []
        for frozen in freezes:
            identity = frozen["game_identity"]
            if (identity.get("provider_namespace") == q["provider_namespace"]
                    and identity.get("provider_game_id") == q["provider_game_id"]
                    and identity.get("game_number") == q["game_number"]
                    and identity.get("official_date") == q["official_date"]
                    and identity.get("scheduled_start_utc") == q["scheduled_start_utc"]
                    and identity.get("game_type","R") == "R"
                    and all(identity.get(f"{side}_team_code") == q[f"{side}_team"]["abbreviation"] for side in ("home","away"))
                    and parse_canonical_utc(frozen["prediction_authority"]["created_at_utc"]) < start):
                matching.append(frozen)
        blockers = []
        if not matching: blockers.append("CHAMPION_FULL_IDENTITY_PREGAME_FREEZE_NOT_FOUND")
        if not all(game["teams"][side].get("probablePitcher",{}).get("id") for side in ("home","away")):
            blockers.append("STARTER_IDENTITIES_ABSENT_FROM_CACHED_SCHEDULE")
        if captured >= start: blockers.append("OFFICIAL_SCHEDULE_CAPTURE_AFTER_TARGET")
        # The fixed input contract contains no feature-availability evidence.
        # Missing evidence stays unknown even if a late freeze were discovered.
        blockers.extend(["CHAMPION_FEATURE_CAPTURE_BEFORE_CUTOFF_NOT_ESTABLISHED",
                         "ACTUAL_MODEL_DECISION_TIME_NOT_ESTABLISHED"])
        row = {"provider_game_id":q["provider_game_id"],"game_number":q["game_number"],
               "official_date":q["official_date"],"role":role,"model_pair_status":"NOT_EVALUABLE",
               "blockers":blockers,"verified_pregame_freeze_matches":len(matching),
               "decision_freshness":"UNKNOWN_NO_VERIFIED_MODEL_DECISION_TIME",
               "quote_age_at_scheduled_start_seconds":q["quote_age_seconds_at_official_start"],
               "within_existing_age_limit_at_start_proxy":q["quote_age_seconds_at_official_start"] <= policy["max_quote_age_seconds"],
               "age_proxy_is_admission":False,"market_reference_scored":False}
        if role != "historical_audit" or open_audit:
            home,away = (game["teams"][side].get("score") for side in ("home","away"))
            if type(home) is not int or type(away) is not int or home == away:
                raise ValueError("unambiguous official final score required for reference scoring")
            y = int(home > away)
            scores[role]["market"].append((Decimal(q["market_no_vig_home_probability"]),y))
            scores[role]["constant"].append((Decimal("0.5"),y))
            row.update(market_reference_scored=True,actual_home_win=y)
        rows.append(row)
    combined = {method:[v for score in scores.values() for v in score[method]] for method in ("market","constant")}
    market_by_id = {q["provider_game_id"]:float(q["market_no_vig_home_probability"]) for q in quotes}
    reference_pairs = [{"candidate":market_by_id[row["provider_game_id"]],"constant":0.5,
                        "y":row["actual_home_win"],"official_date":row["official_date"]}
                       for row in rows if row["market_reference_scored"]]
    return {"schema_version":"historical_price_readiness_result.v1","protocol_sha256":sha256(protocol_raw).hexdigest(),
            "data_status":protocol["interpretation"],"audit_opened":open_audit,
            "denominators":{"price_eligible_games":len(quotes),"champion_market_paired_games":0,
                            "development_games":sum(r["role"] == "development" for r in rows),
                            "historical_audit_games":sum(r["role"] == "historical_audit" for r in rows),
                            "missing_evidence_counts":dict(Counter(reason for row in rows for reason in row["blockers"]))},
            "existing_policy":policy,"freshness":{"verified_decision_time_games":0,
                "decision_freshness_status":"UNKNOWN","within_age_limit_at_scheduled_start_proxy_games":sum(r["within_existing_age_limit_at_start_proxy"] for r in rows),
                "warning":"Scheduled-start proxy is descriptive only; it never replaces actual freeze/decision timestamps."},
            "champion_vs_market":{"sample_count":0,"champion_metrics":_metrics([]),"market_metrics":_metrics([])},
            "historical_reference_scores":{**{role:{m:_metrics(v) for m,v in value.items()} for role,value in scores.items()},
                                           "all_opened":{m:_metrics(v) for m,v in combined.items()}},
            "constant_role":"UNINFORMATIVE_0.5_REFERENCE; NOT_CHAMPION_OR_NEW_FITTED_MODEL",
            "market_minus_constant_reference":_paired(reference_pairs,"constant"),
            "optimization_trials_this_batch":0,"profitability_claim":False,"bet_count":0,"clv":"UNKNOWN",
            "rows":rows}
