from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from match_analysis.application.use_cases.generate_moneyline_market_movement import (
    CLOSING_PRICE_AVAILABLE,
    CLOSING_PRICE_UNAVAILABLE,
    generate_moneyline_market_movement,
)
from match_analysis.infrastructure.legacy_betting_pool.tsl_odds_history import (
    load_tsl_odds_history,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P28AB_REPORT_ROOT = REPOSITORY_ROOT / "report/p28ab_tsl_aligned_moneyline_edge"
TSL_FIXTURE = (
    REPOSITORY_ROOT
    / "data/fixtures/p28ab_tsl_aligned_moneyline_edge/tsl_odds_history.jsonl"
)
RESULT_FIELDS = {
    "away_score",
    "final",
    "game_status",
    "home_score",
    "result",
    "winner",
}


def _jsonl(path: Path) -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def _p28ab_inputs() -> dict:
    return {
        "p28ab_raw_cohort": _jsonl(P28AB_REPORT_ROOT / "raw_cohort.jsonl"),
        "p28ab_prices": _jsonl(P28AB_REPORT_ROOT / "prices.jsonl"),
        "p28ab_predictions": _jsonl(P28AB_REPORT_ROOT / "predictions.jsonl"),
        "p28ab_summary": json.loads(
            (P28AB_REPORT_ROOT / "summary.json").read_text(encoding="utf-8")
        ),
        "p28ab_source_manifest": json.loads(
            (P28AB_REPORT_ROOT / "source_manifest.json").read_text(encoding="utf-8")
        ),
    }


def _actual_result(*, tsl_rows=None):
    inputs = _p28ab_inputs()
    snapshot = load_tsl_odds_history(TSL_FIXTURE)
    return generate_moneyline_market_movement(
        **inputs,
        tsl_rows=snapshot.rows if tsl_rows is None else tsl_rows,
        tsl_raw_sha256=snapshot.raw_sha256 if tsl_rows is None else None,
    )


def _synthetic_closing_rows() -> tuple[tuple[dict, ...], dict]:
    inputs = _p28ab_inputs()
    snapshot = load_tsl_odds_history(TSL_FIXTURE)
    prediction = inputs["p28ab_predictions"][0]
    price = next(row for row in inputs["p28ab_prices"] if row["game_id"] == prediction["game_id"])
    raw_source = next(
        row
        for row in inputs["p28ab_raw_cohort"]
        if row["source_row_fingerprint"] == price["source_row_fingerprint"]
    )
    entry_row = next(
        row
        for row in snapshot.rows
        if row["match_id"] == raw_source["tsl_match_id"]
        and row["fetched_at"] == raw_source["tsl_fetched_at"]
    )
    closing = deepcopy(entry_row)
    closing["fetched_at"] = "2026-05-17T18:00:00Z"
    closing["markets"][0]["outcomes"][0]["odds"] = "1.50"
    closing["markets"][0]["outcomes"][1]["odds"] = "2.10"
    postgame = deepcopy(closing)
    postgame["fetched_at"] = "2026-05-17T18:11:00Z"
    postgame["is_pregame"] = False
    postgame["markets"][0]["outcomes"][0]["odds"] = "1.01"
    postgame["markets"][0]["outcomes"][1]["odds"] = "9.00"
    rows = tuple(snapshot.rows) + (closing, postgame)
    return rows, inputs


class GenerateMoneylineMarketMovementTests(unittest.TestCase):
    def test_committed_p28ab_replay_accounts_for_missing_closes(self) -> None:
        result = _actual_result()

        self.assertEqual(result.summary["p28ab_raw_source_row_count"], 46)
        self.assertEqual(result.summary["p28ab_selected_price_count"], 16)
        self.assertEqual(result.summary["p28ab_evaluable_prediction_count"], 9)
        self.assertEqual(result.summary["paired_p28ab_game_count"], 9)
        self.assertEqual(result.summary["closing_price_available_count"], 0)
        self.assertEqual(result.summary["closing_price_unavailable_count"], 9)
        self.assertEqual(result.summary["market_movement_row_count"], 0)
        self.assertTrue(result.summary["deterministic_replay_verified"])
        self.assertTrue(result.summary["input_order_invariance_verified"])
        self.assertTrue(all(row["closing_status"] == CLOSING_PRICE_UNAVAILABLE for row in result.closing_prices))
        self.assertFalse(any(RESULT_FIELDS.intersection(row) for row in result.closing_prices))

    def test_later_pregame_close_is_selected_and_both_sides_are_preserved(self) -> None:
        tsl_rows, inputs = _synthetic_closing_rows()
        result = generate_moneyline_market_movement(
            **inputs,
            tsl_rows=tsl_rows,
        )

        available = [
            row for row in result.closing_prices if row["closing_status"] == CLOSING_PRICE_AVAILABLE
        ]
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["closing_observed_at_utc"], "2026-05-17T18:00:00Z")
        self.assertEqual(len(result.market_movement), 1)
        movement = result.market_movement[0]
        self.assertEqual(movement["selection"], "HOME_AND_AWAY")
        self.assertEqual(movement["metric_name"], "CLV")
        self.assertEqual(
            Decimal(movement["home_clv_value"]),
            Decimal(available[0]["closing_home_raw_implied_probability"])
            - Decimal(available[0]["entry_home_raw_implied_probability"]),
        )
        self.assertEqual(
            Decimal(movement["away_clv_value"]),
            Decimal(available[0]["closing_away_raw_implied_probability"])
            - Decimal(available[0]["entry_away_raw_implied_probability"]),
        )
        self.assertEqual(movement["predicted_side"], available[0]["predicted_side"])
        self.assertFalse(RESULT_FIELDS.intersection(movement))

    def test_postgame_and_impossible_chronology_do_not_create_a_close(self) -> None:
        tsl_rows, inputs = _synthetic_closing_rows()
        prediction = inputs["p28ab_predictions"][0]
        price = next(row for row in inputs["p28ab_prices"] if row["game_id"] == prediction["game_id"])
        raw_source = next(
            row
            for row in inputs["p28ab_raw_cohort"]
            if row["source_row_fingerprint"] == price["source_row_fingerprint"]
        )
        match_id = raw_source["tsl_match_id"]
        only_postgame = tuple(row for row in tsl_rows if row["match_id"] != match_id) + (
            next(row for row in tsl_rows if row["match_id"] == match_id and not row["is_pregame"]),
        )
        postgame_result = generate_moneyline_market_movement(**inputs, tsl_rows=only_postgame)
        self.assertEqual(postgame_result.summary["closing_price_available_count"], 0)

        before_entry = deepcopy(next(row for row in tsl_rows if row["match_id"] == match_id))
        before_entry["fetched_at"] = "2026-05-17T02:04:35.317591Z"
        chronology_rows = tuple(
            row for row in tsl_rows if row["match_id"] != match_id
        ) + (before_entry,)
        chronology_result = generate_moneyline_market_movement(**inputs, tsl_rows=chronology_rows)
        self.assertEqual(chronology_result.summary["closing_price_available_count"], 0)

    def test_closing_mutation_changes_only_downstream_movement(self) -> None:
        tsl_rows, inputs = _synthetic_closing_rows()
        first = generate_moneyline_market_movement(**inputs, tsl_rows=tsl_rows)
        mutated_rows = deepcopy(tsl_rows)
        target = next(
            row
            for row in mutated_rows
            if row["fetched_at"] == "2026-05-17T18:00:00Z"
        )
        target["markets"][0]["outcomes"][0]["odds"] = "1.25"
        second = generate_moneyline_market_movement(**inputs, tsl_rows=mutated_rows)
        for field in (
            "prediction_id",
            "game_id",
            "entry_source_row_fingerprint",
            "entry_observed_at_utc",
            "entry_home_raw_implied_probability",
            "entry_away_raw_implied_probability",
            "home_win_probability",
            "away_win_probability",
            "predicted_side",
        ):
            self.assertEqual(first.closing_prices[0][field], second.closing_prices[0][field])
        self.assertNotEqual(
            first.closing_prices[0]["closing_away_raw_implied_probability"],
            second.closing_prices[0]["closing_away_raw_implied_probability"],
        )
        self.assertNotEqual(first.market_movement, second.market_movement)

    def test_result_fields_and_input_order_do_not_change_output(self) -> None:
        tsl_rows, inputs = _synthetic_closing_rows()
        baseline = generate_moneyline_market_movement(**inputs, tsl_rows=tsl_rows)
        mutated_inputs = deepcopy(inputs)
        for collection_name in ("p28ab_raw_cohort", "p28ab_prices", "p28ab_predictions"):
            for row in mutated_inputs[collection_name]:
                row["home_score"] = 999
                row["away_score"] = 0
                row["winner"] = "HOME"
        mutated_tsl = deepcopy(tsl_rows)
        for row in mutated_tsl:
            row["home_score"] = 999
            row["away_score"] = 0
            row["winner"] = "HOME"
        mutated = generate_moneyline_market_movement(
            **mutated_inputs,
            tsl_rows=mutated_tsl,
        )
        self.assertEqual(baseline.closing_prices, mutated.closing_prices)
        self.assertEqual(baseline.market_movement, mutated.market_movement)
        reversed_result = generate_moneyline_market_movement(
            p28ab_raw_cohort=tuple(reversed(inputs["p28ab_raw_cohort"])),
            p28ab_prices=tuple(reversed(inputs["p28ab_prices"])),
            p28ab_predictions=tuple(reversed(inputs["p28ab_predictions"])),
            p28ab_summary=inputs["p28ab_summary"],
            p28ab_source_manifest=inputs["p28ab_source_manifest"],
            tsl_rows=tuple(reversed(tsl_rows)),
        )
        self.assertEqual(baseline.closing_prices, reversed_result.closing_prices)
        self.assertEqual(baseline.market_movement, reversed_result.market_movement)


if __name__ == "__main__":
    unittest.main()
