"""Focused P34A daily settlement and feedback contracts."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from match_analysis.application.use_cases.moneyline_paper_run_bundle import (
    jsonl_fingerprint,
)
from match_analysis.application.use_cases.p34a_daily_moneyline_settlement_artifacts import (
    render_p34a_artifacts,
    render_p34a_report,
)
from match_analysis.application.use_cases.settle_daily_moneyline_paper_run import (
    P33AAuthority,
    P34A_RESULT_AUTHORITY,
    P34A_STOP_RESULT_INPUT_INVALID,
    build_official_result_authority,
    compute_result_authority_fingerprint,
    load_p33a_authority,
    replay_daily_moneyline_paper_settlement,
    settle_daily_moneyline_paper_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P33A_BUNDLE = Path(
    "/tmp/matchanalysis-p33a-daily-paper-run/live-smoke-20260812-r3/bundles/"
    "a646ec0081afde1469f978e20bf98fc90cf21587bcecaa9b5cccb280b46bd569"
)


def _sha(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _p33a_fixture() -> P33AAuthority:
    run_id = _sha("p34a-fixture-run")
    model_fingerprint = _sha("p34a-fixture-model")
    prediction_id = _sha("p34a-fixture-prediction")
    analysis_row: dict[str, object] = {
        "schema_version": "p30a.moneyline_paper_analysis.v1",
        "run_id": _sha("p34a-fixture-p30a-run"),
        "game_id": "888001",
        "scheduled_start": "2026-04-05T20:10:00Z",
        "home_team": "Home Team",
        "away_team": "Away Team",
        "structural_status": "EDGE_AVAILABLE",
        "status": "EDGE_AVAILABLE",
        "prediction_id": prediction_id,
        "model_id": "p34a_fixture_model",
        "model_fingerprint": model_fingerprint,
        "model_home_probability": "0.65",
        "market_price_id": f"p28ab:{_sha('p34a-price')}",
        "price_observed_at": "2026-04-05T18:00:00Z",
        "home_decimal_odds": "1.80",
        "away_decimal_odds": "2.10",
        "home_no_vig_probability": "0.5384615384615385",
        "away_no_vig_probability": "0.4615384615384615",
        "home_edge": "0.1115384615384615",
        "away_edge": "-0.1115384615384615",
        "controlled_unavailable_reason": None,
    }
    schedule_row = {
        "schema_version": "p23f2.mlb_official_normalized.v1",
        "provider_game_id": "888001",
        "game_pk": 888001,
        "game_number": 1,
        "official_date": "2026-04-05",
        "scheduled_start_utc": "2026-04-05T20:10:00Z",
        "status": "Scheduled",
        "final": False,
        "home_score": None,
        "away_score": None,
        "home_team": {"id": 118, "abbreviation": "HOM", "name": "Home Team"},
        "away_team": {"id": 109, "abbreviation": "AWY", "name": "Away Team"},
    }
    summary = {
        "schema_version": "p33a.daily_moneyline_paper_run.v1",
        "run_id": run_id,
        "run_fingerprint": run_id,
        "target_date": "2026-04-05",
        "bundle_fingerprint": _sha("p34a-fixture-bundle"),
        "analysis_run_id": analysis_row["run_id"],
        "analysis_set_fingerprint": jsonl_fingerprint((analysis_row,)),
        "analysis_terminal_state_counts": {"EDGE_AVAILABLE": 1},
        "official_raw_game_count": 1,
        "source_records_received": 1,
        "observations_qualified": 1,
        "observations_rejected": 0,
        "claims": {
            "settlement_included": False,
            "staking_implemented": False,
            "profitability_claim": False,
            "real_betting_recommendation": False,
        },
    }
    manifest = {
        "run_id": run_id,
        "run_fingerprint": run_id,
        "target_game_ids": ["888001"],
    }
    return P33AAuthority(
        bundle_root=Path("/tmp/p34a-fixture-bundle"),
        run_manifest=manifest,
        summary=summary,
        source_manifest={"schema_version": "p34a.fixture.source.v1"},
        analysis_rows=(analysis_row,),
        structural_rows=(),
        prediction_rows=(analysis_row,),
        schedule_rows=(schedule_row,),
        analysis_jsonl_sha256=_sha("p34a-analysis-bytes"),
        summary_json_sha256=_sha("p34a-summary-bytes"),
        analysis_set_fingerprint=summary["analysis_set_fingerprint"],
        pregame_authority_fingerprint=_sha("p34a-pregame-authority"),
    )


def _result_rows(*, home_score: int = 5, away_score: int = 3) -> bytes:
    return _canonical_jsonl(
        [
            {
                "source_result_id": "mlb-fixture-result-888001",
                "provider_namespace": P34A_RESULT_AUTHORITY,
                "provider_game_id": "888001",
                "game_number": 1,
                "status": "FINAL",
                "result_observed_at_utc": "2026-04-05T22:15:00Z",
                "home_score": home_score,
                "away_score": away_score,
            }
        ]
    )


def _authority(final_results: bytes) -> dict[str, object]:
    from match_analysis.baseball.domain.final_result_observation import (
        load_final_result_observations,
    )

    observations = load_final_result_observations(final_results)
    return {
        "source": P34A_RESULT_AUTHORITY,
        "provider_namespace": P34A_RESULT_AUTHORITY,
        "source_url": "https://statsapi.mlb.com/api/v1/game/888001/feed/live",
        "observed_at_utc": "2026-04-06T00:00:00Z",
        "raw_payload_sha256": sha256(final_results).hexdigest(),
        "network_called": False,
        "target_game_count": 1,
        "final_result_count": len(observations),
        "non_final_target_count": 0,
        "missing_target_count": 0,
        "all_target_results_final": True,
        "all_settleable_results_final": True,
        "result_authority_fingerprint": compute_result_authority_fingerprint(
            observations
        ),
    }


class P34ADailySettlementTests(unittest.TestCase):
    def test_frozen_p33a_authority_is_loaded_without_regeneration(self) -> None:
        authority = load_p33a_authority(P33A_BUNDLE)

        self.assertEqual(authority.run_manifest["run_id"], P33A_BUNDLE.name)
        self.assertEqual(len(authority.analysis_rows), 15)
        self.assertEqual(len(authority.prediction_rows), 0)
        self.assertEqual(len(authority.structural_rows), 15)
        self.assertEqual(authority.summary["official_raw_game_count"], 15)
        self.assertEqual(authority.summary["source_records_received"], 14)
        self.assertEqual(authority.summary["observations_qualified"], 4)
        self.assertEqual(authority.summary["observations_rejected"], 10)

    def test_official_authority_keeps_non_final_rows_out_of_final_results(self) -> None:
        rows = [
            {
                "provider_game_id": "888001",
                "game_number": 1,
                "scheduled_start_utc": "2026-04-05T20:10:00Z",
                "status": "Scheduled",
                "final": False,
                "home_score": None,
                "away_score": None,
            }
        ]
        final_results, authority = build_official_result_authority(
            normalized_schedule_rows=rows,
            target_game_ids=("888001",),
            observed_at_utc="2026-04-05T21:00:00Z",
            source_url="https://statsapi.mlb.com/api/v1/schedule",
            raw_payload_sha256=_sha("raw-official-schedule"),
            network_called=True,
        )

        self.assertEqual(final_results, b"")
        self.assertEqual(authority["final_result_count"], 0)
        self.assertEqual(authority["non_final_target_count"], 1)
        self.assertFalse(authority["all_target_results_final"])

    def test_one_final_result_reuses_attachment_evaluation_and_feedback_contracts(
        self,
    ) -> None:
        p33a = _p33a_fixture()
        final_results = _result_rows()
        with patch(
            "match_analysis.application.use_cases.settle_daily_moneyline_paper_run.load_p33a_authority",
            return_value=p33a,
        ):
            result = settle_daily_moneyline_paper_run(
                p33a_bundle_path=p33a.bundle_root,
                final_results_bytes=final_results,
                result_authority=_authority(final_results),
            )

        self.assertEqual(result.settled_count, 1)
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(result.evaluation_result.correct_count, 1)
        self.assertEqual(result.evaluation_result.incorrect_count, 0)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.brier_score, 0.1225)
        row = result.prediction_result_rows[0]
        self.assertEqual(row["predicted_side"], "HOME")
        self.assertEqual(row["market_price"], "1.80")
        self.assertEqual(row["market_edge"], "0.1115384615384615")
        self.assertEqual(row["home_score"], 5)
        self.assertEqual(row["away_score"], 3)
        self.assertTrue(row["is_correct"])
        self.assertEqual(result.feedback_result.evaluated_row_count, 1)

    def test_missing_result_is_unresolved_without_fake_evaluation(self) -> None:
        p33a = _p33a_fixture()
        final_results = b""
        authority = _authority(final_results)
        with patch(
            "match_analysis.application.use_cases.settle_daily_moneyline_paper_run.load_p33a_authority",
            return_value=p33a,
        ):
            result = settle_daily_moneyline_paper_run(
                p33a_bundle_path=p33a.bundle_root,
                final_results_bytes=final_results,
                result_authority=authority,
            )

        self.assertEqual(result.settled_count, 0)
        self.assertEqual(result.unresolved_count, 1)
        self.assertEqual(result.evaluation_result.evaluation_row_count, 0)
        self.assertEqual(result.feedback_result.evaluated_row_count, 0)
        self.assertEqual(result.feedback_result.non_evaluated_row_count, 1)
        self.assertEqual(
            result.prediction_result_rows[0]["unresolved_reason"],
            "MISSING_FINAL_RESULT_OBSERVATION",
        )

    def test_duplicate_or_non_final_result_fails_closed(self) -> None:
        p33a = _p33a_fixture()
        duplicate = _result_rows() + _result_rows(home_score=6, away_score=2)
        with patch(
            "match_analysis.application.use_cases.settle_daily_moneyline_paper_run.load_p33a_authority",
            return_value=p33a,
        ):
            with self.assertRaisesRegex(ValueError, P34A_STOP_RESULT_INPUT_INVALID):
                settle_daily_moneyline_paper_run(
                    p33a_bundle_path=p33a.bundle_root,
                    final_results_bytes=duplicate,
                    result_authority=_authority(_result_rows()),
                )

            non_final = _result_rows().replace(b'"FINAL"', b'"IN_PROGRESS"')
            with self.assertRaisesRegex(ValueError, P34A_STOP_RESULT_INPUT_INVALID):
                settle_daily_moneyline_paper_run(
                    p33a_bundle_path=p33a.bundle_root,
                    final_results_bytes=non_final,
                    result_authority=_authority(_result_rows()),
                )

    def test_artifacts_are_byte_identical_on_replay(self) -> None:
        p33a = _p33a_fixture()
        final_results = _result_rows()
        with patch(
            "match_analysis.application.use_cases.settle_daily_moneyline_paper_run.load_p33a_authority",
            return_value=p33a,
        ):
            first = settle_daily_moneyline_paper_run(
                p33a_bundle_path=p33a.bundle_root,
                final_results_bytes=final_results,
                result_authority=_authority(final_results),
                offline_replay_verified=True,
            )
            second = settle_daily_moneyline_paper_run(
                p33a_bundle_path=p33a.bundle_root,
                final_results_bytes=final_results,
                result_authority=_authority(final_results),
                offline_replay_verified=True,
            )

        first_artifacts = render_p34a_artifacts(first)
        second_artifacts = render_p34a_artifacts(second)
        self.assertEqual(first_artifacts, second_artifacts)
        self.assertEqual(
            first.feedback_result.feedback_ledger_fingerprint,
            second.feedback_result.feedback_ledger_fingerprint,
        )
        self.assertIn("small", render_p34a_report(first).decode("utf-8"))

    def test_actual_bundle_replay_materializes_zero_prediction_feedback(self) -> None:
        final_results = b""
        authority = _authority(final_results)
        authority.update(
            {
                "target_game_count": 15,
                "missing_target_count": 15,
                "source_url": "frozen-final-results-input",
            }
        )
        with tempfile.TemporaryDirectory(prefix="p34a-test-") as temporary:
            bundle = Path(temporary) / "p34a"
            bundle.mkdir()
            (bundle / "final_results.jsonl").write_bytes(final_results)
            (bundle / "result_authority.json").write_text(
                json.dumps(authority, sort_keys=True),
                encoding="utf-8",
            )
            with patch(
                "match_analysis.application.use_cases.settle_daily_moneyline_paper_run.load_p33a_authority",
                return_value=load_p33a_authority(P33A_BUNDLE),
            ):
                result = replay_daily_moneyline_paper_settlement(
                    p33a_bundle_path=P33A_BUNDLE,
                    settlement_bundle_path=bundle,
                )
        self.assertEqual(result.settled_count, 0)
        self.assertEqual(result.structural_rows.__len__(), 15)
        self.assertEqual(result.evaluation_result.evaluation_row_count, 0)


if __name__ == "__main__":
    unittest.main()
