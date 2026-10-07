"""Replay the current same-market P50C baseline with frozen inputs and existing policy."""
import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from ...application.use_cases.adapt_legacy_bullpen_evidence import load_json
from ...application.use_cases.p50c_market_comparison import MarketComparisonPolicy, evaluate_p50c_market_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    manifest = load_json(args.sources.read_bytes())
    if manifest["schema_version"] != "method_loop.forward_readonly_sources.v1":
        raise ValueError("unknown source manifest")
    for name, fingerprint in manifest["files"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError(f"frozen forward source hash mismatch: {name}")
    # Also freeze membership: newly added captures/runs/ledger revisions require
    # a separately versioned baseline, never silently alter the old population.
    ledger = root/"report/p50c_prospective_prediction_shadow_ledger"
    quotes = root/"report/moneyline_paper_decision_v1"
    actual = {str(p.relative_to(root)) for p in [*ledger.glob("runs/*/frozen_predictions.jsonl"),
              *ledger.glob("runs/*/run_manifest.json"), ledger/"ledger/prediction_forward_ledger.jsonl",
              ledger/"corrections/p66a_official_date_correction_v1.jsonl",
              *quotes.glob("**/quote_snapshots.jsonl"), *quotes.glob("**/paper_state.json")]}
    if actual != set(manifest["files"]):
        raise ValueError("forward source membership changed; create a new baseline version")
    existing = load_json((quotes/"paper_state.json").read_bytes())["policy_parameters"]
    if existing != manifest["policy"]:
        raise ValueError("existing policy differs from frozen policy")
    result = evaluate_p50c_market_comparison(repository_root=root, quote_snapshots=quotes,
        policy=MarketComparisonPolicy(min_net_ev=Decimal(existing["min_net_ev"]),
                                      max_quote_age_seconds=existing["max_quote_age_seconds"],
                                      paper_cost_per_unit=Decimal(existing["paper_cost_per_unit"])))
    result["quote_input_files"] = [str(Path(p).relative_to(root)) for p in result["quote_input_files"]]
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
