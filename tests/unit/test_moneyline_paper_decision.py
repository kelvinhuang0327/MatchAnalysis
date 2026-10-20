from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request
import unittest

from match_analysis.application.use_cases.moneyline_paper_decision import (
    DATA_MODE_FIXTURE,
    PAPER_POLICY_VERSION,
    PaperPolicyConfig,
    load_frozen_prediction_run,
    run_moneyline_paper_decision_from_paths,
)
from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    NormalizedResultRecord,
    parse_normalized_result_payload,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    build_frozen_prediction_record,
    compute_prediction_row_fingerprint,
)
from match_analysis.interfaces.cli.moneyline_paper_decision import (
    _live_tsl_quotes,
    main,
)


DECISION_AT = "2026-08-01T18:00:00Z"
SETTLED_AT = "2026-08-02T01:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction(
    game_pk: int,
    *,
    p_home: str = "0.75",
    p_away: str | None = None,
    game_type: str = "R",
    scheduled_start: str = "2026-08-01T19:00:00Z",
    created_at: str = "2026-08-01T17:00:00Z",
) -> dict[str, object]:
    parsed_home = Decimal(p_home)
    expected_away = Decimal("1") - parsed_home
    if p_away is not None and Decimal(p_away) != expected_away:
        raise ValueError("fixture P50C p_away must equal one minus p_home")
    team_home = f"Fixture Home {game_pk}"
    team_away = f"Fixture Away {game_pk}"
    row = P40APredictionRow(
        p37_fold_id="fixture-fold",
        p37_window="fixture-window",
        p37_prediction_row_id=f"{game_pk:064x}",
        provider_namespace="MLB_STATS_API",
        provider_game_id=str(game_pk),
        game_pk=game_pk,
        game_number=1,
        game_type=game_type,
        scheduled_start_utc=scheduled_start,
        champion_model_id="fixture-model-v1",
        champion_model_fingerprint="a" * 64,
        champion_home_probability=parsed_home,
        challenger_model_id="fixture-model-v1",
        challenger_model_fingerprint="a" * 64,
        challenger_home_probability=parsed_home,
    )
    return build_frozen_prediction_record(
        row,
        run_id="fixture-run-1",
        run_classification="HISTORICAL_REHEARSAL",
        created_at_utc=created_at,
        official_date="2026-08-01",
        home_team=team_home,
        away_team=team_away,
        home_team_code="SEA",
        away_team_code="TEX",
    )


def _prediction_run(root: Path, rows: list[dict[str, object]]) -> Path:
    run_dir = root / "fixture-run-1"
    fingerprints = [str(row["prediction_fingerprint"]) for row in rows]
    bundle_fingerprint = sha256(
        (json.dumps(fingerprints, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "fixture-run-1",
            "run_classification": "HISTORICAL_REHEARSAL",
            "lifecycle_state": "FROZEN",
            "game_type": rows[0]["game_identity"].get("game_type", "R"),
            "target_universe_count": len(rows),
            "prediction_fingerprints": fingerprints,
            "prediction_bundle_fingerprint": bundle_fingerprint,
        },
    )
    _write_jsonl(run_dir / "frozen_predictions.jsonl", rows)
    return run_dir


def _quote(prediction: dict[str, object], *, home: str = "1.60", away: str = "2.30") -> dict[str, object]:
    game = prediction["game_identity"]
    assert isinstance(game, dict)
    game_pk = int(game["game_pk"])
    return {
        "schema_version": "moneyline_paper_decision_v1.quote_snapshot.v1",
        "snapshot_identity": f"fixture-quote-{game_pk}-one",
        "quote_source": "FIXTURE",
        "market": "MONEYLINE",
        "market_code": "MNL",
        "observed_at_utc": "2026-08-01T17:59:30Z",
        "game_identity": dict(game),
        "home_decimal_odds": home,
        "away_decimal_odds": away,
        "source_identity": {"fixture_id": f"quote-{game_pk}", "data_mode": "FIXTURE"},
    }


def _result(prediction: dict[str, object], *, home_score: int, away_score: int) -> dict[str, object]:
    game = prediction["game_identity"]
    authority = prediction["prediction_authority"]
    assert isinstance(game, dict)
    assert isinstance(authority, dict)
    return NormalizedResultRecord(
        prediction_row_id=str(authority["p37_prediction_row_id"]),
        provider_namespace=str(game["provider_namespace"]),
        provider_game_id=str(game["provider_game_id"]),
        game_number=int(game["game_number"]),
        status="FINAL",
        home_score=home_score,
        away_score=away_score,
        result_observed_at_utc="2026-08-02T00:30:00Z",
        source_identity="FIXTURE_OFFICIAL_RESULT",
        game_type=str(game.get("game_type") or "R"),
    ).to_payload()


class MoneylinePaperDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ml-paper-v1-test-")
        self.root = Path(self.temporary.name)
        self.config = PaperPolicyConfig(
            min_net_ev=Decimal("0.05"),
            max_quote_age_seconds=600,
            paper_cost_per_unit=Decimal("0.02"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(
        self,
        *,
        predictions: list[dict[str, object]],
        quotes: list[dict[str, object]] | None,
        decision_at: str = DECISION_AT,
        settled_at: str = SETTLED_AT,
        config: PaperPolicyConfig | None = None,
        results: list[dict[str, object]] | None = None,
        output_name: str = "out",
    ) -> dict[str, object]:
        run_dir = _prediction_run(self.root, predictions)
        quote_path = self.root / f"{output_name}-quotes.jsonl" if quotes is not None else None
        if quote_path is not None:
            _write_jsonl(quote_path, quotes)
        result_path = self.root / f"{output_name}-results.jsonl" if results is not None else None
        if result_path is not None:
            _write_jsonl(result_path, results)
        return run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=self.root / output_name,
            config=config or self.config,
            decision_at_utc=decision_at,
            settled_at_utc=settled_at,
            data_mode=DATA_MODE_FIXTURE,
            final_results_path=result_path,
        )

    def test_fixture_cli_end_to_end_bet_no_bet_settlement_metrics_and_isolation(self) -> None:
        first = _prediction(101)
        second = _prediction(102, p_home="0.60", scheduled_start="2026-08-01T19:30:00Z")
        run_dir = _prediction_run(self.root, [first, second])
        quote_path = self.root / "fixture-quotes.jsonl"
        results_path = self.root / "fixture-results.jsonl"
        _write_jsonl(quote_path, [_quote(first), _quote(second, home="1.50")])
        _write_jsonl(
            results_path,
            [_result(first, home_score=5, away_score=3), _result(second, home_score=2, away_score=4)],
        )
        output_dir = self.root / "paper-output"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--prediction-run",
                    str(run_dir),
                    "--data-mode",
                    "FIXTURE",
                    "--quote-snapshots",
                    str(quote_path),
                    "--final-results",
                    str(results_path),
                    "--output-dir",
                    str(output_dir),
                    "--min-net-ev",
                    "0.05",
                    "--max-quote-age-seconds",
                    "600",
                    "--paper-cost-per-unit",
                    "0.02",
                    "--decision-at-utc",
                    DECISION_AT,
                    "--settled-at-utc",
                    SETTLED_AT,
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, stdout.getvalue())
        state = json.loads((output_dir / "paper_state.json").read_text(encoding="utf-8"))
        by_game = {row["decision_identity"]["game_pk"]: row for row in state["decisions"]}
        self.assertEqual((by_game[101]["decision"], by_game[101]["reason_code"]), ("BET", "EV_THRESHOLD_MET"))
        self.assertEqual((by_game[102]["decision"], by_game[102]["reason_code"]), ("NO_BET", "EV_BELOW_MIN_NET_EV"))
        self.assertEqual(by_game[101]["estimated_net_ev"], "0.18")
        self.assertTrue(by_game[101]["paper_only"])
        self.assertTrue(by_game[101]["not_a_real_fill"])
        self.assertEqual(len(state["positions"]), 1)
        position = state["positions"][0]
        self.assertEqual(position["stake_units"], "1")
        self.assertEqual(position["decimal_odds"], "1.6")
        self.assertEqual(len(state["settlements"]), 1)
        settlement = state["settlements"][0]
        self.assertEqual(settlement["status"], "WIN")
        self.assertEqual(settlement["net_paper_units"], "0.58")
        self.assertEqual(settlement["quote_snapshot_identity"], by_game[101]["quote_snapshot"]["snapshot_identity"])
        metrics = state["metrics"]
        self.assertEqual(metrics["data_mode"], "FIXTURE")
        self.assertEqual(metrics["unique_candidate_games"]["value"], 2)
        self.assertEqual(metrics["decisions"]["bet"], 1)
        self.assertEqual(metrics["decisions"]["no_bet"], 1)
        self.assertEqual(metrics["paper_positions"]["win"], 1)
        self.assertEqual(metrics["paper_positions"]["loss"], 0)
        self.assertEqual(metrics["realized_paper_net_pnl"]["value"], "0.58")
        self.assertEqual(metrics["settled_non_void_stake_units"]["value"], "1")
        self.assertEqual(metrics["paper_roi"]["value"], "0.58")
        self.assertEqual(metrics["win_rate"]["value"], "1")
        self.assertEqual(metrics["brier_all_candidates"]["value"], "0.21125")
        self.assertEqual(metrics["brier_all_candidates"]["denominator"], 2)
        self.assertEqual(metrics["brier_bet_subset"]["value"], "0.0625")
        self.assertEqual(metrics["brier_bet_subset"]["denominator"], 1)
        self.assertEqual((output_dir / "paper_state.json").parent, output_dir)
        self.assertFalse((output_dir / "p50c_prospective_prediction_shadow_ledger").exists())

    def test_cli_rejects_output_inside_canonical_prediction_or_paper_ledger(self) -> None:
        prediction = _prediction(111)
        run_dir = _prediction_run(self.root, [prediction])
        quote_path = self.root / "guard-quotes.jsonl"
        _write_jsonl(quote_path, [_quote(prediction)])
        protected_output = (
            self.root
            / "report"
            / "p50c_prospective_prediction_shadow_ledger"
            / "ledger"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--prediction-run",
                    str(run_dir),
                    "--data-mode",
                    "FIXTURE",
                    "--quote-snapshots",
                    str(quote_path),
                    "--output-dir",
                    str(protected_output),
                    "--repository-root",
                    str(self.root),
                    "--min-net-ev",
                    "0.05",
                    "--max-quote-age-seconds",
                    "600",
                    "--paper-cost-per-unit",
                    "0.02",
                ]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("overlaps a read-only prediction or paper ledger", stderr.getvalue())
        self.assertFalse(protected_output.exists())

    def test_low_threshold_and_negative_ev_are_no_bet(self) -> None:
        prediction = _prediction(201)
        quote = _quote(prediction)
        below = self._run(
            predictions=[prediction],
            quotes=[quote],
            config=PaperPolicyConfig(Decimal("0.19"), 600, Decimal("0.02")),
            output_name="threshold-output",
        )
        self.assertEqual(below["decisions"][0]["decision"], "NO_BET")
        self.assertEqual(below["decisions"][0]["reason_code"], "EV_BELOW_MIN_NET_EV")

        low_value = _prediction(202, p_home="0.60", scheduled_start="2026-08-01T19:30:00Z")
        negative = self._run(
            predictions=[low_value],
            quotes=[_quote(low_value, home="1.50")],
            output_name="negative-output",
        )
        self.assertEqual(Decimal(negative["decisions"][0]["estimated_net_ev"]), Decimal("-0.12"))
        self.assertEqual(negative["decisions"][0]["reason_code"], "EV_BELOW_MIN_NET_EV")

    def test_missing_quote_or_policy_records_no_bet(self) -> None:
        prediction = _prediction(301)
        missing_quote = self._run(
            predictions=[prediction], quotes=None, output_name="no-source-output"
        )
        self.assertEqual(missing_quote["decisions"][0]["reason_code"], "ODDS_SOURCE_NOT_CONFIGURED")
        self.assertEqual(missing_quote["positions"], [])

        no_policy = self._run(
            predictions=[prediction],
            quotes=[_quote(prediction)],
            config=PaperPolicyConfig(),
            output_name="no-config-output",
        )
        self.assertEqual(no_policy["decisions"][0]["reason_code"], "CONFIG_MISSING")
        self.assertIsNone(no_policy["decisions"][0]["estimated_net_ev"])
        self.assertEqual(no_policy["positions"], [])

    def test_stale_future_wrong_identity_and_started_game_fail_closed(self) -> None:
        prediction = _prediction(401)
        variants = (
            (_quote(prediction) | {"observed_at_utc": "2026-08-01T17:30:00Z"}, DECISION_AT, "QUOTE_EXPIRED"),
            (_quote(prediction) | {"observed_at_utc": "2026-08-01T18:00:01Z"}, DECISION_AT, "QUOTE_FROM_FUTURE"),
            (
                _quote(prediction)
                | {"game_identity": dict(prediction["game_identity"], provider_game_id="999")},
                DECISION_AT,
                "QUOTE_IDENTITY_MISMATCH",
            ),
            (_quote(prediction) | {"market": "SPREAD"}, DECISION_AT, "QUOTE_MARKET_MISMATCH"),
            (_quote(prediction) | {"source_identity": {}}, DECISION_AT, "QUOTE_PROVENANCE_MISSING"),
            (_quote(prediction), "2026-08-01T19:00:00Z", "GAME_ALREADY_STARTED"),
        )
        for index, (quote, decision_at, reason) in enumerate(variants):
            with self.subTest(reason=reason):
                state = self._run(
                    predictions=[prediction],
                    quotes=[quote],
                    decision_at=decision_at,
                    output_name=f"time-output-{index}",
                )
                self.assertEqual(state["decisions"][0]["decision"], "NO_BET")
                self.assertEqual(state["decisions"][0]["reason_code"], reason)
                self.assertEqual(state["positions"], [])

        not_preexisting = _prediction(402, created_at=DECISION_AT)
        state = self._run(
            predictions=[not_preexisting],
            quotes=[_quote(not_preexisting)],
            output_name="not-preexisting-output",
        )
        self.assertEqual(state["decisions"][0]["reason_code"], "PREDICTION_NOT_PREEXISTING")

    def test_reruns_are_idempotent_and_quote_update_cannot_add_a_second_position(self) -> None:
        prediction = _prediction(501)
        quote = _quote(prediction)
        run_dir = _prediction_run(self.root, [prediction])
        quote_path = self.root / "idempotent-quotes.jsonl"
        result_path = self.root / "idempotent-results.jsonl"
        output = self.root / "idempotent-output"
        _write_jsonl(quote_path, [quote])
        _write_jsonl(result_path, [_result(prediction, home_score=4, away_score=1)])

        first = run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=output,
            config=self.config,
            decision_at_utc=DECISION_AT,
            settled_at_utc=SETTLED_AT,
            data_mode=DATA_MODE_FIXTURE,
        )
        second = run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=output,
            config=self.config,
            decision_at_utc=DECISION_AT,
            settled_at_utc=SETTLED_AT,
            data_mode=DATA_MODE_FIXTURE,
        )
        self.assertEqual((len(first["decisions"]), len(first["positions"])), (1, 1))
        self.assertEqual((len(second["decisions"]), len(second["positions"])), (1, 1))
        self.assertEqual(second["metrics"]["paper_positions"]["pending"], 1)

        settled = run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=output,
            config=self.config,
            decision_at_utc=DECISION_AT,
            settled_at_utc=SETTLED_AT,
            data_mode=DATA_MODE_FIXTURE,
            final_results_path=result_path,
        )
        repeated = run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=output,
            config=self.config,
            decision_at_utc=DECISION_AT,
            settled_at_utc=SETTLED_AT,
            data_mode=DATA_MODE_FIXTURE,
            final_results_path=result_path,
        )
        self.assertEqual((len(settled["settlements"]), len(repeated["settlements"])), (1, 1))

        updated_quote = _quote(prediction) | {
            "snapshot_identity": "fixture-quote-501-updated",
            "observed_at_utc": "2026-08-01T18:04:30Z",
        }
        _write_jsonl(quote_path, [updated_quote])
        updated = run_moneyline_paper_decision_from_paths(
            run_dir=run_dir,
            quote_path=quote_path,
            output_dir=output,
            config=self.config,
            decision_at_utc="2026-08-01T18:05:00Z",
            settled_at_utc=SETTLED_AT,
            data_mode=DATA_MODE_FIXTURE,
        )
        self.assertEqual(len(updated["positions"]), 1)
        self.assertEqual(len(updated["settlements"]), 1)
        self.assertIn(
            "POSITION_ALREADY_EXISTS",
            {row["reason_code"] for row in updated["decisions"]},
        )

    def test_loss_cost_and_zero_sample_metrics(self) -> None:
        prediction = _prediction(601)
        loss = self._run(
            predictions=[prediction],
            quotes=[_quote(prediction)],
            results=[_result(prediction, home_score=1, away_score=3)],
            output_name="loss-output",
        )
        self.assertEqual(loss["settlements"][0]["status"], "LOSS")
        self.assertEqual(loss["settlements"][0]["net_paper_units"], "-1.02")

        empty = self._run(
            predictions=[prediction], quotes=None, output_name="empty-metrics-output"
        )
        self.assertIsNone(empty["metrics"]["paper_roi"]["value"])
        self.assertEqual(empty["metrics"]["paper_roi"]["status"], "NOT_ENOUGH_DATA")
        self.assertIsNone(empty["metrics"]["win_rate"]["value"])
        self.assertEqual(empty["metrics"]["brier_all_candidates"]["denominator"], 0)
        self.assertIsNone(empty["metrics"]["brier_bet_subset"]["value"])

    def test_game_type_identity_and_metrics_are_kept_in_separate_cohorts(self) -> None:
        regular = _prediction(611)
        postseason = _prediction(
            612,
            game_type="P",
            scheduled_start="2026-08-01T19:30:00Z",
        )
        for prediction, home_score, away_score in ((regular, 5, 3), (postseason, 1, 4)):
            state = self._run(
                predictions=[prediction],
                quotes=[_quote(prediction)],
                results=[_result(prediction, home_score=home_score, away_score=away_score)],
                output_name="game-type-cohorts",
            )

        self.assertEqual(state["metrics"]["aggregation_scope"], "ALL_GAME_TYPES")
        self.assertEqual(state["metrics"]["brier_all_candidates"]["denominator"], 2)
        cohorts = state["metrics"]["game_type_cohorts"]
        self.assertEqual(set(cohorts), {"R", "P"})
        self.assertEqual(cohorts["R"]["game_type"], "R")
        self.assertEqual(cohorts["R"]["brier_all_candidates"]["denominator"], 1)
        self.assertEqual(cohorts["P"]["game_type"], "P")
        self.assertEqual(cohorts["P"]["brier_all_candidates"]["denominator"], 1)
        self.assertEqual(cohorts["R"]["paper_positions"]["win"], 1)
        self.assertEqual(cohorts["P"]["paper_positions"]["loss"], 1)
        by_type = {row["game_type"]: row for row in state["decisions"]}
        self.assertEqual(by_type["P"]["model_domain_status"], "POSTSEASON_OR_NON_R_DIAGNOSTIC")
        settled_types = {row["result_authority"]["game_type"] for row in state["settlements"]}
        self.assertEqual(settled_types, {"R", "P"})

    def test_prediction_id_result_with_wrong_game_type_is_rejected(self) -> None:
        postseason = _prediction(613, game_type="P")
        wrong_result = _result(postseason, home_score=4, away_score=2) | {"game_type": "R"}
        run_dir = _prediction_run(self.root, [postseason])
        quote_path = self.root / "wrong-game-type-quotes.jsonl"
        results_path = self.root / "wrong-game-type-results.jsonl"
        _write_jsonl(quote_path, [_quote(postseason)])
        _write_jsonl(results_path, [wrong_result])

        with self.assertRaisesRegex(RuntimeError, "PAPER_GAME_TYPE_MISMATCH"):
            run_moneyline_paper_decision_from_paths(
                run_dir=run_dir,
                quote_path=quote_path,
                output_dir=self.root / "wrong-game-type-output",
                config=self.config,
                decision_at_utc=DECISION_AT,
                settled_at_utc=SETTLED_AT,
                data_mode=DATA_MODE_FIXTURE,
                final_results_path=results_path,
            )

    def test_invalid_probability_is_excluded_and_current_contract_does_not_infer_void(self) -> None:
        prediction = _prediction(701, p_home="1.2", p_away="-0.2")
        state = self._run(
            predictions=[prediction],
            quotes=[_quote(prediction)],
            results=[_result(prediction, home_score=3, away_score=2)],
            output_name="invalid-prob-output",
        )
        self.assertEqual(state["decisions"][0]["reason_code"], "PREDICTION_PROBABILITY_INVALID")
        self.assertEqual(state["metrics"]["brier_all_candidates"]["denominator"], 0)
        self.assertEqual(state["metrics"]["paper_positions"]["void"], 0)
        self.assertEqual(
            state["metrics"]["void_support"],
            "NOT_SUPPORTED_BY_CURRENT_OFFICIAL_RESULT_CONTRACT",
        )
        with self.assertRaisesRegex(RuntimeError, "P43A_NON_FINAL_RESULT_FAIL_CLOSED"):
            parse_normalized_result_payload(
                _result(prediction, home_score=0, away_score=0) | {"status": "VOID"}
            )

    def test_prospective_mode_rejects_fixture_prediction_run(self) -> None:
        run_dir = _prediction_run(self.root, [_prediction(801)])
        with self.assertRaisesRegex(ValueError, "prospective P50C prediction run"):
            load_frozen_prediction_run(run_dir, data_mode="PROSPECTIVE")

    def test_live_adapter_bridge_records_calls_and_maps_official_identity(self) -> None:
        for game_type in ("R", "F", "P"):
            with self.subTest(game_type=game_type):
                self._assert_live_adapter_bridge(game_type)

    def _assert_live_adapter_bridge(self, game_type: str) -> None:
        prediction = _prediction(901, game_type=game_type)
        run_dir = _prediction_run(self.root, [prediction])
        run = load_frozen_prediction_run(run_dir, data_mode=DATA_MODE_FIXTURE)
        game = prediction["game_identity"]
        assert isinstance(game, dict)
        output_dir = self.root / "report" / "moneyline_paper_decision_v1"
        captured: dict[str, object] = {}
        observation = SimpleNamespace(
            official={
                "game_type": game_type,
                "provider_game_id": game["provider_game_id"],
                "game_number": game["game_number"],
                "official_date": game["official_date"],
                "scheduled_start_utc": game["scheduled_start_utc"],
                "home_team": {"abbreviation": game["home_team_code"]},
                "away_team": {"abbreviation": game["away_team_code"]},
            },
            row={
                "source": "TSL_BLOB3RD",
                "fetched_at": "2026-08-01T17:59:30Z",
                "match_id": "fixture-match-901",
                "game_time": "2026-08-01T19:00:00Z",
            },
            home_decimal_odds=Decimal("1.60"),
            away_decimal_odds=Decimal("2.30"),
            source_row_index=4,
            source_row_fingerprint="f" * 64,
            crosswalk_time_delta_seconds=30,
        )
        acquisition = SimpleNamespace(
            history=SimpleNamespace(rows=({"fixture": True},)),
            target_schedule_rows=({"fixture": True},),
            target_date="2026-08-01",
            source_payload_sha256=(("https://fixture.test/odds", "a" * 64),),
            runtime_capture_paths=(str(output_dir / "acquisition" / "capture.json"),),
        )

        def acquire_fixture(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            schedule_opener = kwargs["schedule_opener"]
            tsl_fetcher = kwargs["tsl_fetcher"]
            assert callable(schedule_opener) and callable(tsl_fetcher)
            schedule_opener(Request("https://fixture.test/schedule"), 10)
            tsl_fetcher("https://fixture.test/odds")
            return acquisition

        with (
            patch(
                "match_analysis.interfaces.cli.moneyline_paper_decision.acquire_tsl_moneyline_snapshot",
                side_effect=acquire_fixture,
            ),
            patch(
                "match_analysis.interfaces.cli.moneyline_paper_decision._default_mlb_schedule_opener",
                return_value=b"schedule fixture",
            ),
            patch(
                "match_analysis.interfaces.cli.moneyline_paper_decision._default_fetcher",
                return_value=b"odds fixture",
            ),
            patch(
                "match_analysis.interfaces.cli.moneyline_paper_decision._crosswalk",
                return_value=(observation,),
            ),
        ):
            quotes, issues, operation = _live_tsl_quotes(
                repository_root=self.root,
                output_dir=output_dir,
                run=run,
            )

        self.assertEqual(issues, {})
        self.assertEqual(captured["game_type"], game_type)
        self.assertEqual(operation["game_type"], game_type)
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0]["game_identity"], game)
        self.assertEqual(quotes[0]["home_decimal_odds"], "1.60")
        self.assertEqual(quotes[0]["source_identity"]["source_row_index"], 4)
        self.assertEqual(operation["status"], "SUCCESS")
        self.assertEqual(operation["adapter_call_count"], 2)
        self.assertEqual(operation["http_attempt_count"], "UNKNOWN (the existing urllib/curl fallback does not expose attempts)")
        self.assertEqual(Path(captured["runtime_root"]).parent, output_dir / "acquisition")
        self.assertEqual(
            [call["source"] for call in operation["adapter_calls_observed"]],
            ["MLB_STATS_API", "TSL_BLOB3RD"],
        )
        state = self._run(
            predictions=[prediction], quotes=list(quotes),
            config=PaperPolicyConfig(Decimal("0.02"), 120, Decimal("0.02")),
            output_name=f"bridge-{game_type}",
        )
        self.assertEqual(state["decisions"][0]["decision"], "BET")
        self.assertEqual(Decimal(state["decisions"][0]["estimated_net_ev"]), Decimal("0.1800"))

    def test_f_quote_reaches_ordinary_bet_and_economic_no_bet_logic(self) -> None:
        for game_pk, probability, decision in ((902, "0.75", "BET"), (903, "0.60", "NO_BET")):
            with self.subTest(decision=decision):
                prediction = _prediction(game_pk, game_type="F", p_home=probability)
                state = self._run(
                    predictions=[prediction], quotes=[_quote(prediction, home="1.60")],
                    config=PaperPolicyConfig(Decimal("0.02"), 120, Decimal("0.02")),
                    output_name=f"f-{game_pk}",
                )
                row = state["decisions"][0]
                self.assertEqual(row["decision"], decision)
                self.assertIsNotNone(row["estimated_net_ev"])
                if decision == "NO_BET":
                    self.assertEqual(row["reason_code"], "EV_BELOW_MIN_NET_EV")

    def test_manifest_game_type_mismatch_and_fresh_missing_type_reject(self) -> None:
        prediction = _prediction(904, game_type="F")
        run_dir = _prediction_run(self.root, [prediction])
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        manifest["game_type"] = "R"
        _write_json(run_dir / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "P50C_GAME_TYPE_MISMATCH"):
            load_frozen_prediction_run(run_dir, data_mode=DATA_MODE_FIXTURE)
        manifest["game_type"] = "F"
        _write_json(run_dir / "run_manifest.json", manifest)
        prediction["game_identity"].pop("game_type")
        _write_jsonl(run_dir / "frozen_predictions.jsonl", [prediction])
        with self.assertRaisesRegex(ValueError, "P50C_FRESH_GAME_TYPE_REQUIRED"):
            load_frozen_prediction_run(run_dir, data_mode=DATA_MODE_FIXTURE)

    def test_placeholder_frozen_fixture_remains_identity_invalid(self) -> None:
        prediction = _prediction(905, game_type="F")
        game = prediction["game_identity"]
        game["home_team_code"] = "HOME"
        game["away_team_code"] = "AWAY"
        model = prediction["model_identity"]
        authority = prediction["prediction_authority"]
        row = P40APredictionRow(
            p37_fold_id="fixture-fold", p37_window="fixture-window",
            p37_prediction_row_id=authority["p37_prediction_row_id"],
            provider_namespace=game["provider_namespace"], provider_game_id=game["provider_game_id"],
            game_pk=game["game_pk"], game_number=game["game_number"], game_type="F",
            scheduled_start_utc=game["scheduled_start_utc"],
            champion_model_id=model["model_id"], champion_model_fingerprint=model["model_fingerprint"],
            champion_home_probability=Decimal(model["p_home"]),
            challenger_model_id=model["model_id"], challenger_model_fingerprint=model["model_fingerprint"],
            challenger_home_probability=Decimal(model["p_home"]),
        )
        prediction["prediction_fingerprint"] = compute_prediction_row_fingerprint(
            row, **{key: game[key] for key in ("official_date", "home_team", "away_team", "home_team_code", "away_team_code")}
        )
        state = self._run(predictions=[prediction], quotes=[_quote(prediction)], output_name="bad-identity")
        self.assertEqual(state["decisions"][0]["reason_code"], "PREDICTION_IDENTITY_INVALID")
        self.assertEqual(state["positions"], [])


if __name__ == "__main__":
    unittest.main()
