"""Read-only P50C model-versus-market comparison CLI."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import sys

from ...application.use_cases.p50c_market_comparison import (
    MarketComparisonPolicy,
    evaluate_p50c_market_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_p50c_market_comparison",
        description=(
            "Offline, read-only scoring of P50C prospective predictions against "
            "existing normalized P32A/P28AB quote snapshots."
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--quote-snapshots",
        type=Path,
        action="append",
        required=True,
        help=(
            "Existing quote JSONL, moneyline paper_state.json, or capture directory; "
            "repeat for multiple inputs. Directories scan both supported artifacts. No network access."
        ),
    )
    parser.add_argument("--runs-root", type=Path, help="Override the existing P50C runs directory")
    parser.add_argument("--ledger-root", type=Path, help="Override the existing P50C ledger directory")
    parser.add_argument(
        "--min-net-ev",
        type=Decimal,
        required=True,
        help="Existing approved minimum net EV per unit; no default is inferred",
    )
    parser.add_argument(
        "--max-quote-age-seconds",
        type=int,
        required=True,
        help="Maximum age at decision time (later of P50C freeze and quote observation)",
    )
    parser.add_argument(
        "--paper-cost-per-unit",
        type=Decimal,
        required=True,
        help="Explicit cost deducted once per one-unit paper wager, win or loss",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_p50c_market_comparison(
            repository_root=args.repository_root,
            quote_snapshots=args.quote_snapshots,
            runs_root=args.runs_root,
            ledger_root=args.ledger_root,
            policy=MarketComparisonPolicy(
                min_net_ev=args.min_net_ev,
                max_quote_age_seconds=args.max_quote_age_seconds,
                paper_cost_per_unit=args.paper_cost_per_unit,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
