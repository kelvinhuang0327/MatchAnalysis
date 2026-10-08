"""One bounded, synchronous capture using P52A/P50C/P32A; no scheduler or orders."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping
from urllib.request import Request, urlopen

from .acquire_tsl_moneyline_snapshot import acquire_tsl_moneyline_snapshot
from .moneyline_paper_decision import DATA_MODE_PROSPECTIVE, load_frozen_prediction_run
from .p32a_frozen_moneyline_quotes import quotes_from_acquisition
from .p50c_market_comparison import MarketComparisonPolicy, _identity_matches, _quote_error
from .p52a_daily_prediction_freeze import (
    CHAMPION_MODEL_RELATIVE_PATH, _p52a_game_identity, compute_starter_era,
    compute_team_recent_win_rate_delta, execute_daily_moneyline_prediction_freeze,
    fetch_mlb_schedule_data, fetch_pitcher_game_logs, parse_schedule_rows,
)
from ...infrastructure.providers.mlb_official_historical_source import format_utc, parse_utc

SCHEMA = "prospective_pairing_capture.v1"
SCHEDULE_FIELDS = "dates,date,games,gamePk,gameNumber,gameType,officialDate,gameDate,status,abstractGameState,detailedState,teams,home,away,team,id,name,abbreviation,score,probablePitcher,fullName"
Clock = Callable[[], datetime]


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def bounded_public_get(request: Request, timeout: int) -> bytes:
    # Mac's system curl uses system trust; Homebrew Python can lack TSL's chain.
    # This is ONE verified HTTPS GET, no urllib retry, -k, credentials or cert edits.
    if sys.platform == "darwin" and request.full_url.startswith("https://blob3rd.sportslottery.com.tw/apidata/"):
        result = subprocess.run(["/usr/bin/curl", "--proto", "=https", "--fail", "--silent", "--show-error",
            "--connect-timeout", "5", "--max-time", str(min(timeout, 20)),
            "--max-filesize", str(2 * 1024 * 1024), request.full_url],
            check=True, capture_output=True, timeout=25)
        if len(result.stdout) > 2 * 1024 * 1024:
            raise ValueError("SOURCE_RESPONSE_SIZE_LIMIT")
        return result.stdout
    # Other sources use one urllib request with the same response ceiling.
    with urlopen(request, timeout=min(timeout, 20)) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("SOURCE_RESPONSE_SIZE_LIMIT")
    return raw


class SourceCaptures:
    def __init__(self, root: Path, *, clock: Clock, transport=bounded_public_get):
        self.root = root
        self.clock = clock
        self.transport = transport
        self.records = []
        self.by_url = {}
        self.purpose = "MODEL_INPUT"

    def now(self):
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("CAPTURE_CLOCK_MUST_BE_AWARE")
        return value.astimezone(UTC)

    def fetch(self, request: Request, timeout=20):
        url = request.full_url
        if not url.startswith(("https://statsapi.mlb.com/api/v1/",
                               "https://blob3rd.sportslottery.com.tw/apidata/")):
            raise ValueError("SOURCE_URL_NOT_ALLOWLISTED")
        if len(self.records) >= 12:
            raise ValueError("SOURCE_REQUEST_LIMIT")
        if url in self.by_url:
            raise ValueError("DUPLICATE_SOURCE_FETCH")
        started = self.now()
        record = {"url": url, "request_started_at_utc": format_utc(started),
                  "provider_published_at_utc": None, "status": "FAILED", "purpose": self.purpose}
        self.records.append(record)
        try:
            raw = self.transport(request, timeout)
            received = self.now()
            if received < started:
                raise ValueError("CAPTURE_CLOCK_REVERSED")
            if not raw or len(raw) > 2 * 1024 * 1024:
                raise ValueError("SOURCE_RESPONSE_SIZE_LIMIT")
            json.loads(raw)
            relative = f"source_captures/{len(self.records):02d}-{sha256(url.encode()).hexdigest()[:12]}.json"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(raw)
            persisted = self.now()
            if persisted < received:
                raise ValueError("CAPTURE_CLOCK_REVERSED")
            record.update(status="SUCCESS", received_at_utc=format_utc(received),
                          persisted_at_utc=format_utc(persisted), path=relative,
                          sha256=sha256(raw).hexdigest(), byte_count=len(raw))
            self.by_url[url] = (path, record)
            return raw
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def replay(self, request: Request, timeout=20):
        # P52A is allowed only inputs captured by THIS invocation, never old caches.
        path, record = self.by_url[request.full_url]
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("SOURCE_CAPTURE_HASH_MISMATCH")
        return raw


def assess_pair(prediction: Mapping, quote: Mapping | None, *, source_records,
                freeze_persisted_at_utc: str, decision_at_utc: str,
                policy: MarketComparisonPolicy):
    """Outcome-blind eligibility at an actually recorded decision, not inferred max()."""
    result = {"game_identity": dict(prediction["game_identity"]),
              "prediction_fingerprint": prediction.get("prediction_fingerprint"),
              "decision_at_utc": decision_at_utc, "market": "MONEYLINE",
              "clv": None, "clv_status": "CLOSING_QUOTE_UNKNOWN",
              "final_sample_eligible": False}
    start = parse_utc(prediction["game_identity"]["scheduled_start_utc"])
    frozen = parse_utc(prediction["prediction_authority"]["created_at_utc"])
    persisted = parse_utc(freeze_persisted_at_utc)
    decision = parse_utc(decision_at_utc)
    reason = None
    if persisted < frozen or decision < persisted:
        reason = "FREEZE_OR_DECISION_TIME_ORDER_INVALID"
    elif persisted >= start or decision >= start:
        reason = "FREEZE_OR_DECISION_NOT_PREGAME"
    elif not source_records:
        reason = "SOURCE_CAPTURE_EVIDENCE_MISSING"
    else:
        for record in source_records:
            if record.get("status") != "SUCCESS":
                continue  # failed optional quote source is reported separately
            try:
                received = parse_utc(record["received_at_utc"])
                saved = parse_utc(record["persisted_at_utc"])
                if received > saved or saved > decision:
                    reason = "SOURCE_NOT_AVAILABLE_AT_DECISION"
                # Model sources, unlike quote sources, must precede inference cutoff.
                if record.get("purpose") == "MODEL_INPUT" and saved > frozen:
                    reason = "MODEL_SOURCE_NOT_AVAILABLE_AT_FREEZE"
            except (KeyError, ValueError):
                reason = "SOURCE_CAPTURE_TIME_INVALID"
    if reason is None and not any(r.get("purpose") == "MODEL_INPUT" and r.get("status") == "SUCCESS" for r in source_records):
        reason = "MODEL_CAPTURE_EVIDENCE_MISSING"
    if reason is None and any(r.get("purpose") == "MODEL_INPUT" and r.get("status") != "SUCCESS" for r in source_records):
        reason = "MODEL_SOURCE_CAPTURE_FAILED"
    if reason is None and quote is not None:
        try:
            parse_utc(quote["observed_at_utc"])
        except (KeyError, ValueError):
            reason = "QUOTE_TIMESTAMP_INVALID"
    if reason is None and quote is not None:
        quote_evidence = [r for r in source_records if r.get("purpose") == "QUOTE_CAPTURE"
                          and r.get("status") == "SUCCESS"
                          and r["url"].startswith("https://blob3rd.sportslottery.com.tw/")]
        if not quote_evidence:
            reason = "QUOTE_CAPTURE_EVIDENCE_MISSING"
        elif any(parse_utc(r["persisted_at_utc"]) > parse_utc(quote["observed_at_utc"])
                 for r in quote_evidence):
            reason = "QUOTE_OBSERVATION_PRECEDES_CAPTURE"
    if reason is None and quote is None:
        reason = "QUOTE_MISSING"
    if reason is None and not _identity_matches(prediction, quote):
        reason = "QUOTE_IDENTITY_MISMATCH"
    home_odds = away_odds = None
    if reason is None:
        # Reuse P50C quote validation, with the ACTUAL decision for freshness.
        decision_projection = dict(prediction)
        decision_projection["prediction_authority"] = {
            **prediction["prediction_authority"], "created_at_utc": decision_at_utc}
        reason, observed, home_odds, away_odds = _quote_error(
            quote, prediction=decision_projection, policy=policy)
        if observed is not None and observed > decision:
            reason = "QUOTE_NOT_AVAILABLE_AT_DECISION"
    result["status"] = reason or "PAIRED_PENDING_FINAL"
    result["paired_pregame_eligible"] = reason is None
    result["paper_selection"] = "NOT_EVALUABLE"
    if reason is None:
        model = prediction["model_identity"]
        p_home = Decimal(model["p_home"])
        selection = model["selection"]
        p_selected = p_home if selection == "HOME" else 1 - p_home
        odds = home_odds if selection == "HOME" else away_odds
        ev = p_selected * odds - 1 - policy.paper_cost_per_unit
        result.update(model_home_probability=str(p_home),
                      market_home_probability=str((1 / home_odds) / (1 / home_odds + 1 / away_odds)),
                      paper_selection="BET" if ev >= policy.min_net_ev else "NO_BET",
                      estimated_net_ev=str(ev), paper_stake_units="1" if ev >= policy.min_net_ev else "0")
    return result


def collect_once(*, repository_root: Path, output_root: Path, target_date: str,
                 game_type: str, game_pk: int, history_days=45,
                 clock: Clock = lambda: datetime.now(UTC), transport=bounded_public_get):
    repo = repository_root.resolve()
    out = output_root.resolve()
    if out == repo or out.is_relative_to(repo):
        raise ValueError("ISOLATED_OUTPUT_OUTSIDE_REPOSITORY_REQUIRED")
    if not 1 <= history_days <= 45:
        raise ValueError("HISTORY_WINDOW_LIMIT")
    target = date.fromisoformat(target_date)
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("CAPTURE_CLOCK_MUST_BE_AWARE")
    if not now.date() <= target <= now.date() + timedelta(days=7):
        raise ValueError("LIVE_TARGET_DATE_OUT_OF_RANGE")
    out.mkdir(parents=True, exist_ok=False)
    captures = SourceCaptures(out, clock=clock, transport=transport)
    report = {"schema_version": SCHEMA, "target_date": target_date, "game_type": game_type,
              "game_pk": game_pk, "history_days": history_days, "settled_same_sample_count": 0, "accuracy": None,
              "brier": None, "logloss": None, "paper_roi": None, "pair_rows": []}
    policy_path = repo / "report/moneyline_paper_decision_v1/paper_state.json"
    policy_bytes = policy_path.read_bytes()
    settings = json.loads(policy_bytes)["policy_parameters"]
    policy = MarketComparisonPolicy(min_net_ev=settings["min_net_ev"],
        max_quote_age_seconds=settings["max_quote_age_seconds"],
        paper_cost_per_unit=settings["paper_cost_per_unit"])
    report["policy"] = policy.to_payload()
    (out / "policy_source.json").write_bytes(policy_bytes)
    (out / "champion_model_artifact.json").write_bytes((repo / CHAMPION_MODEL_RELATIVE_PATH).read_bytes())
    hist_start = (target - timedelta(days=history_days)).isoformat()
    try:
        historical, _, _ = fetch_mlb_schedule_data(hist_start, target_date, fields=SCHEDULE_FIELDS, opener=captures.fetch)
        slate = historical if game_type == "R" else fetch_mlb_schedule_data(
            target_date, target_date, game_type=game_type, fields=SCHEDULE_FIELDS, opener=captures.fetch)[0]
        games = [g for g in parse_schedule_rows(slate)
                 if g["game_pk"] == game_pk and g["game_type"] == game_type
                 and g["official_date"] == target_date]
        if len(games) != 1:
            raise ValueError("TARGET_GAME_IDENTITY_NOT_UNIQUE")
        game = games[0]
        report["schedule_game_identity"] = _p52a_game_identity(game)
        if game["final"] or parse_utc(game["scheduled_start_utc"]) <= captures.now():
            raise ValueError("TARGET_GAME_NOT_FUTURE")
        pitcher_ids = [g["player_id"] for g in (game["home_starter"], game["away_starter"]) if g]
        logs, _ = fetch_pitcher_game_logs(pitcher_ids, target_date[:4], opener=captures.fetch)
        # All model input bytes are persisted BEFORE the actual inference cutoff.
        cutoff = format_utc(captures.now())
        h_era, h_error = compute_starter_era(game["home_starter"]["player_id"], target_date, logs) if game["home_starter"] else (None, "STARTER_MISSING")
        a_era, a_error = compute_starter_era(game["away_starter"]["player_id"], target_date, logs) if game["away_starter"] else (None, "STARTER_MISSING")
        win_delta = compute_team_recent_win_rate_delta(game["home_team"]["id"], game["away_team"]["id"],
            parse_schedule_rows(historical), parse_utc(cutoff), parse_utc(game["scheduled_start_utc"]))
        report["feature_evidence"] = {"home_starter_id": game["home_starter"]["player_id"] if game["home_starter"] else None,
            "away_starter_id": game["away_starter"]["player_id"] if game["away_starter"] else None,
            "home_starter_era": str(h_era) if h_era is not None else None,
            "away_starter_era": str(a_era) if a_era is not None else None,
            "starter_era_delta": str(h_era - a_era) if h_era is not None and a_era is not None else None,
            "recent_win_rate_delta": str(win_delta) if win_delta is not None else None,
            "starter_exclusions": [r for r in (h_error, a_error) if r], "inference_cutoff_utc": cutoff}
        frozen = execute_daily_moneyline_prediction_freeze(target_date=target_date,
            as_of_utc=cutoff, repository_root=repo, output_root=out,
            history_start_date=hist_start, schedule_fields=SCHEDULE_FIELDS, game_type=game_type,
            target_game_identity=_p52a_game_identity(game), opener=captures.replay,
            extra_source_manifest={"source_capture_evidence": list(captures.records),
                                   "capture_semantics": "ACTUAL_RECEIPT_AND_PERSIST_BEFORE_INFERENCE"})
        persisted = format_utc(captures.now())
        report["freeze_result"] = frozen.to_dict()
        report["model_eligible_count"] = frozen.eligible_predictions_count
        report["model_excluded_count"] = frozen.exclusion_count
        report["freeze_persisted_at_utc"] = persisted
        acquisition = None
        captures.purpose = "QUOTE_CAPTURE"
        try:
            acquisition = acquire_tsl_moneyline_snapshot(repository_root=repo,
                runtime_root=out / "p32a", game_type=game_type, minimum_qualified_games=1,
                clock=captures.now, schedule_opener=captures.fetch,
                tsl_fetcher=lambda url: captures.fetch(Request(url)))
            report["quote_crosswalk_game_ids"] = list(acquisition.requested_game_ids)
            report["quote_observed_at_utc"] = acquisition.fetched_at_utc
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            report["quote_acquisition_error"] = f"{type(exc).__name__}: {exc}"
        report["quote_availability_denominator"] = 1
        report["quote_crosswalk_present_count"] = int(str(game_pk) in report.get("quote_crosswalk_game_ids", []))
        report["quote_crosswalk_semantics"] = "P32A_PRICE_OBSERVATION_ONLY_NOT_CHAMPION_PAIR"
        if frozen.run_dir is None:
            report["status"] = "MODEL_EXCLUDED"
        else:
            run = load_frozen_prediction_run(frozen.run_dir, data_mode=DATA_MODE_PROSPECTIVE)
            quotes, issues = (), {}
            if acquisition is not None:
                quotes, issues = quotes_from_acquisition(acquisition, run, reject_ambiguous=True)
            with (out / "quote_snapshots.jsonl").open("x") as handle:
                handle.write("".join(json.dumps(q, sort_keys=True) + "\n" for q in quotes))
            report["quote_identity_issues"] = {str(k): v for k, v in issues.items()}
            decision = format_utc(captures.now())
            report["decision_at_utc"] = decision
            for prediction in run.predictions:
                matched = [q for q in quotes if _identity_matches(prediction, q)]
                row = assess_pair(prediction, matched[0] if len(matched) == 1 else None,
                    source_records=captures.records, freeze_persisted_at_utc=persisted,
                    decision_at_utc=decision, policy=policy)
                if issues.get(game_pk) or len(matched) > 1:
                    row.update(status=issues.get(game_pk, "QUOTE_AMBIGUOUS"),
                               paired_pregame_eligible=False, paper_selection="NOT_EVALUABLE", paper_stake_units="0")
                report["pair_rows"].append(row)
            report["status"] = "CAPTURE_COMPLETED_PENDING_FINAL"
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        report["status"] = "CAPTURE_BLOCKED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["source_captures"] = captures.records
    report["paired_pending_count"] = sum(r["paired_pregame_eligible"] for r in report["pair_rows"])
    report["quote_missing_or_rejected_count"] = len(report["pair_rows"]) - report["paired_pending_count"]
    # The file-hash envelope binds raw captures, model intake, frozen run and policy.
    report["files"] = {str(p.relative_to(out)): sha256(p.read_bytes()).hexdigest()
                       for p in sorted(out.rglob("*")) if p.is_file()}
    with (out / "capture_report.json").open("xb") as handle:
        handle.write(_json(report))
    return report


def replay_capture(output_root: Path):
    """Read-only replay; refuse altered files, path escapes and changed pair decisions."""
    root = output_root.resolve()
    report = json.loads((root / "capture_report.json").read_bytes())
    if report["schema_version"] != SCHEMA:
        raise ValueError("CAPTURE_SCHEMA_UNSUPPORTED")
    for name, expected in report["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("CAPTURE_FILE_HASH_MISMATCH")
    policy_source = json.loads((root / "policy_source.json").read_bytes())["policy_parameters"]
    policy = MarketComparisonPolicy(min_net_ev=policy_source["min_net_ev"],
        max_quote_age_seconds=policy_source["max_quote_age_seconds"],
        paper_cost_per_unit=policy_source["paper_cost_per_unit"])
    if policy.to_payload() != report["policy"]:
        raise ValueError("CAPTURE_POLICY_MISMATCH")
    feature_verified = False
    inference_verified = None
    if "feature_evidence" in report:
        model_records = [r for r in report["source_captures"]
                         if r.get("purpose") == "MODEL_INPUT" and r.get("status") == "SUCCESS"]
        by_url = {r["url"]: r for r in model_records}
        def captured_opener(request, timeout):
            return (root / by_url[request.full_url]["path"]).read_bytes()
        regular = next(json.loads((root/r["path"]).read_bytes()) for r in model_records
                       if "/schedule?" in r["url"] and "gameType=R" in r["url"])
        slate = regular if report["game_type"] == "R" else next(
            json.loads((root/r["path"]).read_bytes()) for r in model_records
            if "/schedule?" in r["url"] and f"gameType={report['game_type']}" in r["url"])
        games = [g for g in parse_schedule_rows(slate) if g["game_pk"] == report["game_pk"]]
        if len(games) != 1 or _p52a_game_identity(games[0]) != report["schedule_game_identity"]:
            raise ValueError("CAPTURE_SCHEDULE_REPLAY_MISMATCH")
        game = games[0]
        features = report["feature_evidence"]
        cutoff = parse_utc(features["inference_cutoff_utc"])
        if any(parse_utc(r["persisted_at_utc"]) > cutoff for r in model_records):
            raise ValueError("CAPTURE_MODEL_SOURCE_AFTER_CUTOFF")
        ids = [p["player_id"] for p in (game["home_starter"],game["away_starter"]) if p]
        logs, _ = fetch_pitcher_game_logs(ids, report["target_date"][:4], opener=captured_opener)
        home, he = compute_starter_era(game["home_starter"]["player_id"], report["target_date"], logs) if game["home_starter"] else (None, "STARTER_MISSING")
        away, ae = compute_starter_era(game["away_starter"]["player_id"], report["target_date"], logs) if game["away_starter"] else (None, "STARTER_MISSING")
        wins = compute_team_recent_win_rate_delta(game["home_team"]["id"], game["away_team"]["id"],
            parse_schedule_rows(regular), cutoff, parse_utc(game["scheduled_start_utc"]))
        reproduced = {"home_starter_id": game["home_starter"]["player_id"] if game["home_starter"] else None,
            "away_starter_id": game["away_starter"]["player_id"] if game["away_starter"] else None,
            "home_starter_era": str(home) if home is not None else None,
            "away_starter_era": str(away) if away is not None else None,
            "starter_era_delta": str(home-away) if home is not None and away is not None else None,
            "recent_win_rate_delta": str(wins) if wins is not None else None,
            "starter_exclusions": [e for e in (he,ae) if e], "inference_cutoff_utc": features["inference_cutoff_utc"]}
        if reproduced != features:
            raise ValueError("CAPTURE_FEATURE_REPLAY_MISMATCH")
        feature_verified = True
        predictions = report["freeze_result"]["predictions"]
        if predictions:
            artifact = json.loads((root/"champion_model_artifact.json").read_bytes())
            values = [wins, home-away]
            z = [(v - Decimal(str(m))) / Decimal(str(sd))
                 for v,m,sd in zip(values, artifact["scaler_means"], artifact["scaler_stds"])]
            logit = (Decimal(str(artifact["intercept"])) + Decimal(str(artifact["coefficients"][0])) * z[0]
                     + Decimal(str(artifact["coefficients"][1])) * z[1])
            probability = Decimal(1) / (Decimal(1)+Decimal(str(math.exp(float(-logit)))))
            probability = min(Decimal("0.999999"), max(Decimal("0.000001"), probability))
            if len(predictions) != 1 or Decimal(predictions[0]["champion_home_probability"]) != probability:
                raise ValueError("CAPTURE_CHAMPION_INFERENCE_REPLAY_MISMATCH")
            inference_verified = True
        elif not he and not ae and wins is not None:
            raise ValueError("CAPTURE_MODEL_EXCLUSION_REPLAY_MISMATCH")
    result = []
    if report["pair_rows"]:
        freeze = report["freeze_result"]
        run_dir = root / "report/p50c_prospective_prediction_shadow_ledger/runs" / freeze["run_id"]
        run = load_frozen_prediction_run(run_dir, data_mode=DATA_MODE_PROSPECTIVE)
        quotes = [json.loads(line) for line in (root / "quote_snapshots.jsonl").read_text().splitlines()]
        for prediction in run.predictions:
            matched = [q for q in quotes if _identity_matches(prediction, q)]
            row = assess_pair(prediction, matched[0] if len(matched) == 1 else None,
                source_records=report["source_captures"],
                freeze_persisted_at_utc=report["freeze_persisted_at_utc"],
                decision_at_utc=report["decision_at_utc"], policy=policy)
            issue = report.get("quote_identity_issues", {}).get(str(prediction["game_identity"]["game_pk"]))
            if issue or len(matched) > 1:
                row.update(status=issue or "QUOTE_AMBIGUOUS", paired_pregame_eligible=False,
                           paper_selection="NOT_EVALUABLE", paper_stake_units="0")
            result.append(row)
    if result != report["pair_rows"]:
        raise ValueError("CAPTURE_PAIR_REPLAY_MISMATCH")
    return {"schema_version": SCHEMA, "status": report["status"], "pair_rows": result,
            "paired_pending_count": report["paired_pending_count"], "settled_same_sample_count": 0,
            "verified_file_count": len(report["files"]),
            "feature_replay_verified": feature_verified, "champion_inference_replay_verified": inference_verified}
