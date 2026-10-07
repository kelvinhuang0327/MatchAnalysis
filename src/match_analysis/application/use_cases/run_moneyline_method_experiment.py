"""Small, immutable retrospective experiment; never trains or promotes a model."""

from collections import Counter
from datetime import date
from hashlib import sha256
import math
from pathlib import Path
import random
from statistics import mean

from .adapt_legacy_bullpen_evidence import load_json
from ...baseball.domain.canonical_utc import parse_canonical_utc


def _rows(raw):
    return [load_json(line) for line in raw.splitlines() if line.strip()]


def _key(row):
    if (row["provider_namespace"] != "MLB_STATS_API"
            or not isinstance(row["provider_game_id"], str)
            or not row["provider_game_id"].isdigit() or int(row["provider_game_id"]) <= 0
            or type(row["game_number"]) is not int or row["game_number"] <= 0):
        raise ValueError("explicit MLB provider identity and positive game number required")
    return (row["provider_namespace"], row["provider_game_id"], row["game_number"])


def _unique(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError("duplicate full provider/game/number identity")
        result[key] = row
    return result


def _probability(value):
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or not 0 < result < 1:
        raise ValueError("finite interior probability required")
    return result


def _metrics(rows, method):
    n = len(rows)
    if not n:
        return {"n": 0, "correct": 0, "accuracy": None, "brier": None, "logloss": None}
    p = [row[method] for row in rows]
    y = [row["y"] for row in rows]
    k = sum((v >= 0.5) == bool(t) for v, t in zip(p, y))
    z = 1.959963984540054
    center = (k/n + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n))/(1+z*z/n)
    return {"n": n, "correct": k, "accuracy": k/n,
            "accuracy_wilson95": [center-half, center+half],
            "brier": mean((v-t)**2 for v, t in zip(p, y)),
            "logloss": mean(-math.log(v if t else 1-v) for v, t in zip(p, y))}


def _paired(rows, baseline):
    if not rows:
        return {"n": 0, "brier_delta": None, "logloss_delta": None}
    diffs = [(row["candidate"]-row["y"])**2-(row[baseline]-row["y"])**2 for row in rows]
    ll = [-math.log(row["candidate"] if row["y"] else 1-row["candidate"])
          + math.log(row[baseline] if row["y"] else 1-row[baseline]) for row in rows]
    # Resample complete official dates, preserving games on a shared day. This is
    # descriptive uncertainty for very few blocks, not a promotion significance test.
    groups = {}
    for row, delta in zip(rows, diffs):
        groups.setdefault(row["official_date"], []).append(delta)
    blocks = [groups[d] for d in sorted(groups)]
    rng = random.Random(20261007)
    draws = sorted(mean(v for block in rng.choices(blocks, k=len(blocks)) for v in block)
                   for _ in range(1000))
    return {"n": len(rows), "brier_delta": mean(diffs), "logloss_delta": mean(ll),
            "official_date_blocks": len(blocks),
            "brier_date_block_bootstrap95": [draws[24], draws[974]],
            "bootstrap_seed": 20261007, "bootstrap_replicates": 1000,
            "uncertainty_limit": "Descriptive only: few dates, correlated teams and prior trials; no efficacy inference."}


def run_moneyline_method_experiment(root: Path, protocol_raw: bytes, *, open_audit=False):
    protocol = load_json(protocol_raw)
    if (protocol.get("schema_version") != "moneyline.method_experiment.v1"
            or protocol.get("candidate", {}).get("model_weight") != 0.5
            or protocol.get("candidate", {}).get("market_weight") != 0.5
            or protocol.get("split_roles") != {"wf_004": "development", "wf_005": "validation", "wf_006": "retrospective_audit"}):
        raise ValueError("only frozen equal-pool protocol supported; a new method needs a new trial")
    blobs = {}
    for name, fingerprint in protocol["sources"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source outside repository")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != fingerprint:
            raise ValueError(f"frozen source hash mismatch: {name}")
        blobs[name] = raw
    comparisons = _rows(blobs["report/p37a_rolling_walk_forward_oos/comparisons.jsonl"])
    joins = _unique(_rows(blobs["report/p39a_tsl_moneyline_market_join/market_join.jsonl"]))
    indexed = _unique(comparisons)
    if set(indexed) != set(joins):
        raise ValueError("model and join population identities differ")
    snapshots = {}
    for row in _rows(blobs["report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl"]):
        if row["snapshot_id"] in snapshots:
            raise ValueError("duplicate quote snapshot")
        snapshots[row["snapshot_id"]] = row
    schedules = {}
    for name, raw in blobs.items():
        if name.endswith("normalized/schedule.jsonl"):
            for row in _rows(raw):
                key = ("MLB_STATS_API", row["provider_game_id"], row["game_number"])
                if key not in indexed:
                    continue  # Unrelated historical reschedules are outside this frozen population.
                if key in schedules and schedules[key] != row:
                    raise ValueError("conflicting official schedule revisions")
                schedules[key] = row
    admitted, excluded = [], []
    for key, model in indexed.items():
        quote = joins[key]
        if model["fold_id"] not in protocol["split_roles"]:
            raise ValueError("unregistered evaluation fold")
        role = protocol["split_roles"][model["fold_id"]]
        if role == "retrospective_audit" and not open_audit:
            continue  # No terminal audit outcomes scored by default.
        schedule = schedules[key]
        start = parse_canonical_utc(model["scheduled_start_utc"])
        if (quote["scheduled_start_utc"] != model["scheduled_start_utc"]
                or schedule["scheduled_start_utc"] != model["scheduled_start_utc"]
                or quote["p37_fold_id"] != model["fold_id"]
                or quote["official_date"] != schedule["official_date"]
                or str(model["game_pk"]) != key[1]
                or model["true_oos_verified"] is not True
                or date.fromisoformat(model["challenger_training_cutoff_date"]) >= date.fromisoformat(schedule["official_date"])):
            raise ValueError("identity or training/target chronology mismatch")
        for side in ("home", "away"):
            if (quote[f"{side}_team_code"] != schedule[f"{side}_team"]["abbreviation"]
                    or quote[f"{side}_team"] != schedule[f"{side}_team"]["name"]):
                raise ValueError("team identity mismatch")
        if (schedule["final"] is not True or schedule["home_score"] == schedule["away_score"]):
            raise ValueError("nonfinal or tied game cannot score")
        y = int(schedule["home_score"] > schedule["away_score"])
        if type(model["target_home_win"]) is not int or model["target_home_win"] != y:
            raise ValueError("model outcome differs from official final score")
        if quote["market_snapshot_status"] != "USABLE_PREGAME":
            excluded.append({"identity": list(key), "fold": model["fold_id"], "reason": quote["rejection_reason"]})
            continue
        snapshot = snapshots[quote["market_snapshot_id"]]
        if (_key(snapshot) != key or snapshot["market_code"] != "MNL"
                or snapshot["scheduled_start_utc"] != quote["scheduled_start_utc"]
                or snapshot["market_observed_at_utc"] != quote["market_observed_at_utc"]
                or snapshot["identity"]["canonical_game_id"] != key[1]
                or snapshot["identity"]["game_discriminator"] != str(key[2])
                or snapshot["identity"]["league"] != "MLB"
                or snapshot["identity"]["season"] != date.fromisoformat(schedule["official_date"]).year
                or not parse_canonical_utc(quote["market_observed_at_utc"]) < start
                or not parse_canonical_utc(quote["local_fetched_at_utc"]) < start):
            raise ValueError("wrong market or post-start quote")
        prices = []
        for side in ("home", "away"):
            if (snapshot[f"{side}_team_code"] != quote[f"{side}_team_code"]
                    or snapshot["identity"][f"{side}_participant"] != quote[f"{side}_team"]
                    or snapshot[f"{side}_decimal_price"] != quote[f"{side}_decimal_price"]):
                raise ValueError("quote snapshot identity/price mismatch")
            price = float(quote[f"{side}_decimal_price"])
            if not math.isfinite(price) or price <= 1:
                raise ValueError("both valid decimal prices required")
            prices.append(price)
        market = (1/prices[0]) / (1/prices[0] + 1/prices[1])
        incumbent = _probability(model["incumbent_home_probability"])
        admitted.append({"identity": list(key), "fold": model["fold_id"], "role": role,
                         "official_date": schedule["official_date"], "scheduled_start_utc": model["scheduled_start_utc"],
                         "market_observed_at_utc": quote["market_observed_at_utc"],
                         "snapshot_id": quote["market_snapshot_id"], "y": y,
                         "incumbent": incumbent, "market": market, "candidate": (incumbent+market)/2})
    admitted.sort(key=lambda x: (x["scheduled_start_utc"], x["identity"]))
    groups = {role: [x for x in admitted if x["role"] == role] for role in protocol["split_roles"].values()
              if role != "retrospective_audit" or open_audit}
    groups["all_opened"] = admitted
    scores = {role: {"methods": {m: _metrics(rows, m) for m in ("incumbent", "market", "candidate")},
                     "paired_candidate_minus": {m: _paired(rows, m) for m in ("incumbent", "market")}}
              for role, rows in groups.items()}
    passed = all(scores[role]["paired_candidate_minus"][base][metric] is not None
                 and scores[role]["paired_candidate_minus"][base][metric] < 0
                 for role in ("validation", "retrospective_audit") if role in scores
                 for base in ("incumbent", "market") for metric in ("brier_delta", "logloss_delta"))
    source_summary = load_json(blobs["report/p37a_rolling_walk_forward_oos/summary.json"])
    original = source_summary["aggregate"]
    holdouts = {w["holdout_fold_id"]: w["holdout"] for w in source_summary["evaluation_windows"]}
    model_exclusions = [{"fold": fold, "provider_game_id": row["game_id"], "reason": row["reason"]}
                        for fold, holdout in holdouts.items() for row in holdout["excluded_rows"]]
    return {"schema_version": "moneyline.method_experiment_result.v1", "protocol_sha256": sha256(protocol_raw).hexdigest(),
            "trial_accounting": protocol["prior_trials"], "audit_opened": open_audit,
            "data_status": "RETROSPECTIVE_RECONSTRUCTION; OFFICIAL_FEATURE_CAPTURES_AFTER_TARGETS; NOT_PROSPECTIVE",
            "coverage": {"original_targets": original["raw_row_count"], "model_evaluable": len(indexed),
                         "model_excluded": original["excluded_row_count"], "opened_model_targets": len(admitted)+len(excluded),
                         "paired_scored": len(admitted), "quote_excluded": len(excluded),
                         "quote_exclusion_reasons": dict(Counter(x["reason"] for x in excluded))},
            "fold_coverage": {fold: {"model_targets": sum(x["fold_id"] == fold for x in comparisons),
                                     "original_targets": holdouts[fold]["raw_row_count"],
                                     "model_excluded": holdouts[fold]["excluded_row_count"],
                                     "paired_scored": sum(x["fold"] == fold for x in admitted),
                                     "quote_excluded": sum(x["fold"] == fold for x in excluded),
                                     "opened": role != "retrospective_audit" or open_audit}
                              for fold, role in protocol["split_roles"].items()},
            "model_excluded_records": model_exclusions,
            "scores": scores, "rows": admitted, "excluded": excluded,
            "decision": "REJECT_CANDIDATE" if not passed else ("SHADOW_ONLY_UNPROVEN" if open_audit else "AWAIT_TERMINAL_AUDIT"),
            "holdout_limit": protocol["audit_status"], "future_shadow": protocol["future_shadow"],
            "financial_claims": protocol["financial_claims"]}
