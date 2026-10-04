"""Synthetic-only checks for P50C market comparison and paper ROI."""

from __future__ import annotations

import json
import io
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.p50c_market_comparison import (
    MarketComparisonPolicy,
    _load_quote_batch,
    evaluate_p50c_market_comparison,
)
from match_analysis.interfaces.cli.evaluate_p50c_market_comparison import main as cli_main


RUN_ID = "synthetic-market-run"
RUN_CLASS = "PROSPECTIVE_FORWARD_PREDICTION"
SCHEDULED = "2026-10-01T20:00:00Z"


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction(index: int, p_home: str = "0.60") -> dict[str, object]:
    game_pk = 900000 + index
    identity = {
        "provider_namespace": "MLB_STATS_API",
        "provider_game_id": str(game_pk),
        "game_pk": game_pk,
        "game_type": "R",
        "game_number": 1,
        "official_date": "2026-10-01",
        "scheduled_start_utc": SCHEDULED,
        "home_team": f"Home {index}",
        "away_team": f"Away {index}",
        "home_team_code": f"H{index:03}",
        "away_team_code": f"A{index:03}",
    }
    probability = float(p_home)
    # Keep the serialized test values exact and complementary.
    p_away = format(1 - probability, ".2f")
    return {
        "workflow_prediction_id": f"prediction-{index}",
        "prediction_fingerprint": f"fingerprint-{index}",
        "game_identity": identity,
        "model_identity": {
            "model_id": "synthetic-model",
            "model_fingerprint": "a" * 64,
            "model_probability_source": "calibrated_p_home",
            "p_home": p_home,
            "p_away": p_away,
            "selection": "HOME" if probability >= 0.5 else "AWAY",
        },
        "prediction_authority": {
            "run_id": RUN_ID,
            "run_classification": RUN_CLASS,
            "p37_prediction_row_id": f"source-row-{index}",
            "created_at_utc": "2026-10-01T16:00:00Z",
        },
    }


def _quote(
    prediction: dict[str, object],
    *,
    observed_at: str = "2026-10-01T19:30:00Z",
    home_odds: object = "1.90",
    away_odds: object = "2.00",
) -> dict[str, object]:
    game = dict(prediction["game_identity"])
    return {
        "schema_version": "moneyline_paper_decision_v1.quote_snapshot.v1",
        "snapshot_identity": f"quote-{game['game_pk']}",
        "quote_source": "TSL_BLOB3RD",
        "market": "MONEYLINE",
        "market_code": "MNL",
        "observed_at_utc": observed_at,
        "game_identity": game,
        "home_decimal_odds": home_odds,
        "away_decimal_odds": away_odds,
        "source_identity": {
            "source_match_id": f"tsl-{game['game_pk']}",
            "source_game_time": SCHEDULED,
            "source_row_fingerprint": "b" * 64,
            "crosswalk_authority": "P28AB_NEAREST_TEAM_AND_START_WITHIN_3600_SECONDS",
        },
    }


def _settlement(prediction: dict[str, object], status: str = "FINAL") -> dict[str, object]:
    game = dict(prediction["game_identity"])
    authority = dict(prediction["prediction_authority"])
    return {
        "schema_version": "p50c.forward_prediction_ledger_record.v1",
        "ledger_record_id": f"ledger-{prediction['prediction_fingerprint']}",
        "run_id": authority["run_id"],
        "run_classification": authority["run_classification"],
        "prediction_identity": {
            "workflow_prediction_id": prediction["workflow_prediction_id"],
            "prediction_fingerprint": prediction["prediction_fingerprint"],
            "p37_prediction_row_id": authority["p37_prediction_row_id"],
        },
        "game_identity": game,
        "model_identity": prediction["model_identity"],
        "pregame_freeze_authority": {
            "created_at_utc": authority["created_at_utc"],
        },
        "final_result_authority": {
            "status": status,
            "target_home_win": 1 if status == "FINAL" else None,
            "actual_winner": game["home_team"] if status == "FINAL" else None,
        },
        "evaluation": {},
    }


class P50CMarketComparisonTests(unittest.TestCase):
    def test_quote_batch_rejects_conflicting_snapshot_ids_and_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p50c-quote-batch-error-test-") as temp:
            root = Path(temp)
            empty = root / "empty-capture"
            empty.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "no quote_snapshots.jsonl"):
                _load_quote_batch(empty)
            cli_error = io.StringIO()
            with redirect_stderr(cli_error):
                cli_status = cli_main(
                    [
                        "--repository-root",
                        str(root),
                        "--quote-snapshots",
                        str(empty),
                        "--min-net-ev",
                        "0.05",
                        "--max-quote-age-seconds",
                        "120",
                        "--paper-cost-per-unit",
                        "0.02",
                    ]
                )
            self.assertEqual(cli_status, 1)
            self.assertIn("no quote_snapshots.jsonl or paper_state.json", cli_error.getvalue())

            prediction = _prediction(11)
            first = _quote(prediction)
            conflicting = dict(first)
            conflicting["home_decimal_odds"] = "9.99"
            path_a = root / "a.jsonl"
            path_b = root / "b.jsonl"
            _jsonl(path_a, [first])
            _jsonl(path_b, [conflicting])
            with self.assertRaisesRegex(ValueError, "QUOTE_SNAPSHOT_IDENTITY_CONFLICT"):
                _load_quote_batch([path_a, path_b])

    def test_full_identity_quote_status_scoring_and_fixed_unit_roi(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p50c-market-eval-test-") as temp:
            root = Path(temp)
            runs_root = root / "runs"
            run_dir = runs_root / RUN_ID
            run_dir.mkdir(parents=True)
            ledger_root = root / "ledger"
            ledger_root.mkdir()
            predictions = [
                _prediction(1),  # valid match, win, and BET
                _prediction(2),  # same gamePk but incorrect team identity
                _prediction(3),  # postgame quote
                _prediction(4),  # one side missing
                _prediction(5, "0.55"),  # quote matches but edge is below threshold
                _prediction(6),  # final game with no quote row at all
                _prediction(7),  # stale pregame quote
                _prediction(8),  # frozen prediction without a FINAL result
                _prediction(9),  # a recorded non-FINAL result
            ]
            orphan_prediction = _prediction(10)
            orphan_prediction["prediction_authority"]["run_id"] = "old-run-without-archive"
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": RUN_ID,
                        "task_id": "P50C",
                        "run_classification": RUN_CLASS,
                        "lifecycle_state": "SETTLED",
                    }
                ),
                encoding="utf-8",
            )
            _jsonl(run_dir / "frozen_predictions.jsonl", predictions)

            quotes = [_quote(predictions[0])]
            wrong_identity = _quote(predictions[1])
            wrong_identity["game_identity"]["away_team_code"] = "OTHER"
            quotes.append(wrong_identity)
            quotes.append(
                _quote(predictions[2], observed_at="2026-10-01T20:01:00Z")
            )
            quotes.append(_quote(predictions[3], away_odds=None))
            quotes.append(_quote(predictions[4], home_odds="1.40", away_odds="2.90"))
            quotes.append(_quote(predictions[6], observed_at="2026-10-01T15:55:00Z"))
            captures_root = root / "captures"
            jsonl_capture = captures_root / "capture-a" / "quote_snapshots.jsonl"
            jsonl_capture.parent.mkdir(parents=True)
            _jsonl(jsonl_capture, quotes)
            state_quote = {
                key: value
                for key, value in quotes[0].items()
                if key not in ("schema_version", "quote_source")
            }
            state_quote["source"] = quotes[0]["quote_source"]
            state = {
                "schema_version": "moneyline_paper_decision_v1.state.v1",
                "data_mode": "PROSPECTIVE",
                "decisions": [
                    {
                        "data_mode": "PROSPECTIVE",
                        "decision_identity": {
                            "prediction_run_id": RUN_ID,
                            "prediction_fingerprint": predictions[0]["prediction_fingerprint"],
                        },
                        "quote_snapshot": state_quote,
                    }
                ],
            }
            state_path = captures_root / "capture-b" / "paper_state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(json.dumps(state), encoding="utf-8")
            settlements = [
                _settlement(prediction)
                for prediction in (*predictions[:7], predictions[8])
            ]
            settlements[-1] = _settlement(predictions[8], status="InProgress")
            settlements.append(_settlement(orphan_prediction))
            _jsonl(ledger_root / "prediction_forward_ledger.jsonl", settlements)

            report = evaluate_p50c_market_comparison(
                repository_root=root,
                runs_root=runs_root,
                ledger_root=ledger_root,
                quote_snapshots=captures_root,
                policy=MarketComparisonPolicy(
                    min_net_ev="0.05",
                    max_quote_age_seconds=120,
                    paper_cost_per_unit="0.05",
                ),
            )
            cli_output = io.StringIO()
            with redirect_stdout(cli_output):
                cli_status = cli_main(
                    [
                        "--repository-root",
                        str(root),
                        "--runs-root",
                        str(runs_root),
                        "--ledger-root",
                        str(ledger_root),
                        "--quote-snapshots",
                        str(captures_root),
                        "--min-net-ev",
                        "0.05",
                        "--max-quote-age-seconds",
                        "120",
                        "--paper-cost-per-unit",
                        "0.05",
                    ]
                )
            cli_report = json.loads(cli_output.getvalue())

        self.assertEqual(cli_status, 0)
        self.assertEqual(cli_report, report)
        self.assertEqual(cli_report["matched_market_comparison"]["sample_count"], 2)
        self.assertEqual(report["input_counts"]["quote_snapshot_rows_read"], 7)
        self.assertEqual(report["input_counts"]["duplicate_quote_snapshot_rows_collapsed"], 1)
        self.assertEqual(report["input_counts"]["quote_snapshot_source_file_count"], 2)
        self.assertEqual(report["input_counts"]["quote_snapshot_count"], 6)
        self.assertEqual(report["input_counts"]["prospective_frozen_prediction_count"], 10)
        self.assertEqual(report["input_counts"]["frozen_prediction_recovered_from_ledger_count"], 1)
        self.assertEqual(report["model_final_baseline"]["sample_count"], 8)
        self.assertEqual(report["matched_market_comparison"]["sample_count"], 2)
        self.assertEqual(
            report["matched_market_comparison"]["model"]["brier_score"],
            "0.18125",
        )
        self.assertEqual(report["quote_status_counts"]["MATCHED"], 2)
        self.assertEqual(report["quote_status_counts"]["QUOTE_IDENTITY_MISMATCH"], 1)
        self.assertEqual(report["quote_status_counts"]["POSTGAME_OR_POST_START_QUOTE"], 1)
        self.assertEqual(report["quote_status_counts"]["TWO_SIDED_PRICE_MISSING"], 1)
        self.assertEqual(report["quote_status_counts"]["QUOTE_MISSING"], 2)
        self.assertEqual(report["quote_status_counts"]["QUOTE_STALE_BEFORE_START"], 1)
        self.assertEqual(report["quote_status_counts"]["PREDICTION_NOT_FINAL"], 1)
        self.assertEqual(report["quote_status_counts"]["RESULT_NOT_FINAL"], 1)
        self.assertEqual(report["fixed_unit_paper_roi"]["bet_count"], 1)
        self.assertEqual(report["fixed_unit_paper_roi"]["no_bet_count"], 1)
        self.assertEqual(report["fixed_unit_paper_roi"]["staked_units"], "1")
        self.assertEqual(report["fixed_unit_paper_roi"]["cost_units"], "0.05")
        self.assertEqual(report["fixed_unit_paper_roi"]["net_profit_units"], "0.85")
        self.assertEqual(report["fixed_unit_paper_roi"]["net_roi"], "0.85")
        self.assertEqual(report["clv"], {"status": "UNKNOWN_NO_CLOSING_QUOTE", "value": None})
        by_pk = {row["game_identity"]["game_pk"]: row for row in report["predictions"]}
        self.assertEqual(by_pk[900001]["decision"], "BET")
        self.assertEqual(by_pk[900005]["decision"], "NO_BET")
        self.assertEqual(by_pk[900002]["decision"], "NOT_EVALUABLE")
        self.assertEqual(by_pk[900006]["market_status"], "QUOTE_MISSING")
        self.assertAlmostEqual(
            float(Decimal(by_pk[900001]["market_probability_home_no_vig"])),
            0.5128205128205128,
        )


if __name__ == "__main__":
    unittest.main()
