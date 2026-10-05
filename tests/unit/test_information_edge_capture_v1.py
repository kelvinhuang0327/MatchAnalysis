from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.ports.reference_market_quote import (
    ReferenceMarketQuoteObservation,
)
from match_analysis.application.use_cases.information_edge_capture_v1 import (
    MODEL_REPRICE_NOT_AVAILABLE,
    TslMoneylineQuote,
    capture_information_edge_v1,
)
from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    build_frozen_prediction_record,
)
from match_analysis.baseball.domain.information_edge_observation import (
    EDGE_OBSERVATION,
    SIGNAL_BASELINE,
)
from match_analysis.infrastructure.sources.mlb_starting_pitcher_signal import (
    fetch_starting_pitcher_schedule,
    normalize_starting_pitcher_schedule,
)
from match_analysis.infrastructure.sources.tsl_moneyline_acquisition import (
    normalize_tsl_moneyline_games,
)


START = "2026-10-02T18:00:00Z"
IDENTITY = {
    "provider_namespace": "mlb_official",
    "provider_game_id": "9001",
    "game_pk": 9001,
    "game_number": 1,
    "game_type": "R",
    "scheduled_start_utc": START,
    "official_date": "2026-10-02",
    "home_team": "Detroit Tigers",
    "away_team": "Cleveland Guardians",
    "home_team_code": "DET",
    "away_team_code": "CLE",
}


def _schedule_payload(
    home_pitcher: int | None,
    *,
    away_pitcher: int | None = 202,
    game_type: str = "R",
) -> dict:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 9001,
                        "gameNumber": 1,
                        "gameType": game_type,
                        "gameDate": START,
                        "officialDate": "2026-10-02",
                        "teams": {
                            "home": {
                                "team": {"id": 11, "name": "Detroit Tigers", "abbreviation": "DET"},
                                "probablePitcher": (
                                    {"id": home_pitcher, "fullName": "Home Pitcher"}
                                    if home_pitcher is not None
                                    else None
                                ),
                            },
                            "away": {
                                "team": {"id": 22, "name": "Cleveland Guardians", "abbreviation": "CLE"},
                                "probablePitcher": (
                                    {"id": away_pitcher, "fullName": "Away Pitcher"}
                                    if away_pitcher is not None
                                    else None
                                ),
                            },
                        },
                    }
                ]
            }
        ]
    }


def _schedule(
    home_pitcher: int | None,
    received: str,
    *,
    away_pitcher: int | None = 202,
    game_type: str = "R",
):
    payload = _schedule_payload(
        home_pitcher,
        away_pitcher=away_pitcher,
        game_type=game_type,
    )
    return normalize_starting_pitcher_schedule(
        payload,
        provider_game_id="9001",
        signal_observed_at_utc=received,
    )


def _write_run(root: Path, run_id: str, created_at: str, p_home: str, *, game_type: str = "R") -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    row = P40APredictionRow(
        p37_fold_id="fold-1",
        p37_window="window-1",
        p37_prediction_row_id=f"p37-{run_id}",
        provider_namespace=IDENTITY["provider_namespace"],
        provider_game_id=IDENTITY["provider_game_id"],
        game_pk=IDENTITY["game_pk"],
        game_number=IDENTITY["game_number"],
        scheduled_start_utc=IDENTITY["scheduled_start_utc"],
        champion_model_id="model-1",
        champion_model_fingerprint="model-fingerprint",
        champion_home_probability=Decimal(p_home),
        challenger_model_id="model-1",
        challenger_model_fingerprint="model-fingerprint",
        challenger_home_probability=Decimal(p_home),
        game_type=game_type,
    )
    prediction = build_frozen_prediction_record(
        row,
        run_id=run_id,
        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
        created_at_utc=created_at,
        official_date=IDENTITY["official_date"],
        home_team=IDENTITY["home_team"],
        away_team=IDENTITY["away_team"],
        home_team_code=IDENTITY["home_team_code"],
        away_team_code=IDENTITY["away_team_code"],
    )
    fingerprint = prediction["prediction_fingerprint"]
    manifest = {
        "run_id": run_id,
        "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
        "game_type": game_type,
        "prediction_fingerprints": [fingerprint],
        "prediction_bundle_fingerprint": sha256(
            (json.dumps([fingerprint], sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "frozen_predictions.jsonl").write_text(
        json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _tsl_quote(
    *,
    fetched_at: str = "2026-10-02T12:06:00Z",
    home_odds: str = "1.90",
    game_identity: dict | None = None,
) -> TslMoneylineQuote:
    game = {
        "san": "BSB",
        "id": "tsl-game-1",
        "an": "克里夫蘭守護者",
        "hn": "底特律老虎",
        "kt": "2026-10-03T02:00:00+08:00",
        "ms": [
            {
                "name": "不讓分",
                "cs": [
                    {"name": "克里夫蘭守護者", "pd": "100", "pu": "105", "hv": None},
                    {
                        "name": "底特律老虎",
                        "pd": "100",
                        "pu": str(int((Decimal(home_odds) - Decimal(1)) * Decimal(100))),
                        "hv": None,
                    },
                ],
            }
        ],
    }
    normalized = normalize_tsl_moneyline_games(
        [game], fetched_at=fetched_at, target_date="2026-10-03"
    )
    assert len(normalized.observations) == 1
    return TslMoneylineQuote.from_p32a_observation(
        normalized.observations[0],
        provider_game_id="9001",
        game_identity=game_identity or IDENTITY,
    )


class _ReferenceQuote:
    def __init__(self, *, snapshot_id: str = "ref-1", observed_at: str = "2026-10-02T12:07:00Z", home: str = "1.80", away: str = "2.10", identity: dict | None = None):
        self.quote = ReferenceMarketQuoteObservation.create(
            snapshot_id=snapshot_id,
            game_identity=identity or IDENTITY,
            observed_at_utc=observed_at,
            home_decimal_odds=home,
            away_decimal_odds=away,
        )

    def get_quote(self, game_identity):
        return self.quote


class InformationEdgeCaptureV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_root = self.root / "runs"
        self.log = self.root / "report" / "information_edge_capture_v1" / "edge_observations.jsonl"
        _write_run(self.run_root, "before", "2026-10-02T11:55:00Z", "0.55")
        _write_run(self.run_root, "after", "2026-10-02T12:02:00Z", "0.62")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _baseline(
        self,
        *,
        home_pitcher: int | None = 101,
        away_pitcher: int | None = 202,
        game_type: str = "R",
        path: Path | None = None,
    ) -> bytes:
        observation_path = path or self.log
        result = capture_information_edge_v1(
            schedule=_schedule(
                home_pitcher,
                "2026-10-02T11:50:00Z",
                away_pitcher=away_pitcher,
                game_type=game_type,
            ),
            prediction_run_root=self.run_root,
            observation_path=observation_path,
            captured_at_utc="2026-10-02T11:51:00Z",
        )
        self.assertEqual(result.status, "BASELINE_RECORDED")
        self.assertEqual(result.observation.record_type, SIGNAL_BASELINE)
        record = result.observation.to_dict()
        self.assertNotIn("metrics", record)
        self.assertNotIn("signal_type", record)
        return observation_path.read_bytes()

    def _capture(
        self,
        *,
        schedule=None,
        tsl=None,
        reference=None,
        captured_at_utc: str = "2026-10-02T12:20:00Z",
    ):
        return capture_information_edge_v1(
            schedule=schedule or _schedule(303, "2026-10-02T12:00:00Z"),
            prediction_run_root=self.run_root,
            observation_path=self.log,
            before_run_id="before",
            after_run_id="after",
            tsl_quote=tsl or _tsl_quote(),
            reference_market_quote=reference or _ReferenceQuote(),
            captured_at_utc=captured_at_utc,
        )

    def test_schedule_fetch_uses_local_response_receipt_time(self) -> None:
        events: list[str] = []

        def opener(request, timeout: int) -> bytes:
            events.append("response")
            return json.dumps(_schedule_payload(101)).encode("utf-8")

        def clock() -> datetime:
            events.append("clock")
            return datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

        schedule = fetch_starting_pitcher_schedule(
            target_date="2026-10-02",
            provider_game_id="9001",
            opener=opener,
            clock=clock,
        )
        self.assertEqual(events, ["response", "clock"])
        self.assertEqual(schedule.signal_observed_at_utc, "2026-10-02T12:00:00Z")
        self.assertNotIn("as_of", schedule.schedule_identity())

    def test_baseline_then_real_signal_preserves_r_identity_and_computes_edges(self) -> None:
        baseline_bytes = self._baseline()
        result = self._capture()
        self.assertEqual(result.status, "EDGE_RECORDED")
        edge = result.observation.to_dict()
        self.assertEqual(edge["record_type"], EDGE_OBSERVATION)
        self.assertEqual(edge["signal_type"], "STARTING_PITCHER_CHANGE")
        self.assertEqual(edge["previous_pitcher_player_id"], "101")
        self.assertEqual(edge["new_pitcher_player_id"], "303")
        self.assertEqual(edge["pitcher_side"], "HOME")
        self.assertEqual(edge["game_identity"], IDENTITY)
        self.assertEqual(
            edge["schedule_identity"],
            _schedule(303, "2026-10-02T12:00:00Z").schedule_identity(),
        )
        self.assertEqual(edge["model_probability_before"], "0.55")
        self.assertEqual(edge["model_probability_after"], "0.62")
        self.assertEqual(edge["model_repriced_at_utc"], "2026-10-02T12:02:00Z")
        self.assertEqual(edge["metrics"]["model_reprice_delta"], "0.07")
        reference_home = Decimal("2.10") / Decimal("3.90")
        self.assertAlmostEqual(
            float(edge["reference_market_observation"]["reference_home_probability"]),
            float(reference_home),
            places=12,
        )
        self.assertAlmostEqual(float(edge["metrics"]["tsl_home_gross_ev"]), 0.178, places=12)
        self.assertAlmostEqual(float(edge["metrics"]["tsl_away_gross_ev"]), -0.221, places=12)
        self.assertEqual(edge["selected_probabilities"]["after"]["HOME"], "0.62")
        self.assertEqual(edge["selected_probabilities"]["after"]["AWAY"], "0.38")
        self.assertEqual(edge["selected_probabilities"]["before"]["HOME"], "0.55")
        self.assertEqual(edge["selected_probabilities"]["before"]["AWAY"], "0.45")
        self.assertEqual(edge["signal_observed_at_utc"], "2026-10-02T12:00:00Z")
        self.assertNotEqual(edge["captured_at_utc"], edge["signal_observed_at_utc"])
        self.assertTrue(self.log.read_bytes().startswith(baseline_bytes))
        self.assertEqual(edge["game_identity"]["game_type"], "R")

    def test_f_identity_is_preserved_without_reconstruction(self) -> None:
        _write_run(self.run_root, "before-f", "2026-10-02T11:55:00Z", "0.51", game_type="F")
        _write_run(self.run_root, "after-f", "2026-10-02T12:02:00Z", "0.59", game_type="F")
        self._baseline(game_type="F")
        result = capture_information_edge_v1(
            schedule=_schedule(303, "2026-10-02T12:00:00Z", game_type="F"),
            prediction_run_root=self.run_root,
            observation_path=self.log,
            before_run_id="before-f",
            after_run_id="after-f",
            tsl_quote=_tsl_quote(game_identity={**IDENTITY, "game_type": "F"}),
            reference_market_quote=_ReferenceQuote(identity={**IDENTITY, "game_type": "F"}),
            captured_at_utc="2026-10-02T12:20:00Z",
        )
        self.assertEqual(result.observation.to_dict()["game_identity"]["game_type"], "F")

    def test_tsl_quote_requires_p39a_home_away_identity_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "TSL_CANONICAL_GAME_IDENTITY_MISMATCH"):
            _tsl_quote(game_identity={**IDENTITY, "home_team_code": "NYA"})

    def test_unknown_probable_pitcher_transitions_refresh_baseline_only(self) -> None:
        path = self.root / "unknown-pitcher-baselines.jsonl"
        first = capture_information_edge_v1(
            schedule=_schedule(None, "2026-10-02T11:40:00Z"),
            prediction_run_root=self.run_root,
            observation_path=path,
            captured_at_utc="2026-10-02T11:41:00Z",
        )
        self.assertEqual(first.status, "BASELINE_RECORDED")
        for pitcher_id, received_at in (
            (101, "2026-10-02T11:50:00Z"),
            (None, "2026-10-02T11:55:00Z"),
            (303, "2026-10-02T12:00:00Z"),
        ):
            result = capture_information_edge_v1(
                schedule=_schedule(pitcher_id, received_at),
                prediction_run_root=self.run_root,
                observation_path=path,
                captured_at_utc="2026-10-02T12:01:00Z",
            )
            self.assertEqual(result.status, "BASELINE_REFRESHED")
            self.assertEqual(result.observation.record_type, SIGNAL_BASELINE)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(
            [record["record_type"] for record in records],
            [SIGNAL_BASELINE] * 4,
        )

    def test_away_pitcher_change_preserves_side_and_home_based_probability_semantics(self) -> None:
        self._baseline()
        result = self._capture(
            schedule=_schedule(
                101,
                "2026-10-02T12:00:00Z",
                away_pitcher=404,
            )
        )
        edge = result.observation.to_dict()
        self.assertEqual(edge["pitcher_side"], "AWAY")
        self.assertEqual(edge["previous_pitcher_player_id"], "202")
        self.assertEqual(edge["new_pitcher_player_id"], "404")
        self.assertEqual(edge["selected_probabilities"]["after"]["HOME"], "0.62")
        self.assertEqual(edge["selected_probabilities"]["after"]["AWAY"], "0.38")
        self.assertAlmostEqual(float(edge["metrics"]["tsl_home_gross_ev"]), 0.178, places=12)
        self.assertAlmostEqual(float(edge["metrics"]["tsl_away_gross_ev"]), -0.221, places=12)

    def test_model_reprice_not_available_appends_no_completed_edge(self) -> None:
        self._baseline()
        before_bytes = self.log.read_bytes()
        with self.assertRaisesRegex(RuntimeError, MODEL_REPRICE_NOT_AVAILABLE):
            capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=self.log,
                before_run_id="before",
                after_run_id="missing-after",
                tsl_quote=_tsl_quote(),
                reference_market_quote=_ReferenceQuote(),
                captured_at_utc="2026-10-02T12:20:00Z",
            )
        self.assertEqual(self.log.read_bytes(), before_bytes)

    def test_no_lookahead_and_post_start_are_rejected(self) -> None:
        before_path = self.root / "before-no-lookahead.jsonl"
        self._baseline(path=before_path)
        _write_run(self.run_root, "late-before", "2026-10-02T12:01:00Z", "0.55")
        with self.assertRaisesRegex(RuntimeError, "NO_LOOKAHEAD_BEFORE_RUN"):
            capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=before_path,
                before_run_id="late-before",
                after_run_id="after",
                tsl_quote=_tsl_quote(),
                reference_market_quote=_ReferenceQuote(),
                captured_at_utc="2026-10-02T12:20:00Z",
            )

        after_path = self.root / "after-no-lookahead.jsonl"
        self._baseline(path=after_path)
        _write_run(self.run_root, "early-after", "2026-10-02T11:59:00Z", "0.62")
        with self.assertRaisesRegex(RuntimeError, "NO_LOOKAHEAD_AFTER_RUN"):
            capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=after_path,
                before_run_id="before",
                after_run_id="early-after",
                tsl_quote=_tsl_quote(),
                reference_market_quote=_ReferenceQuote(),
                captured_at_utc="2026-10-02T12:20:00Z",
            )
        post_start_path = self.root / "post-start.jsonl"
        self._baseline(path=post_start_path)
        _write_run(self.run_root, "post-start", "2026-10-02T18:01:00Z", "0.62")
        with self.assertRaisesRegex(RuntimeError, "POST_START_OBSERVATION_REJECTED"):
            capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=post_start_path,
                before_run_id="before",
                after_run_id="post-start",
                tsl_quote=_tsl_quote(),
                reference_market_quote=_ReferenceQuote(),
                captured_at_utc="2026-10-02T18:02:00Z",
            )
        late_capture_path = self.root / "late-capture.jsonl"
        self._baseline(path=late_capture_path)
        with self.assertRaisesRegex(RuntimeError, "POST_START_OBSERVATION_REJECTED"):
            capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=late_capture_path,
                before_run_id="before",
                after_run_id="after",
                tsl_quote=_tsl_quote(),
                reference_market_quote=_ReferenceQuote(),
                captured_at_utc="2026-10-02T18:01:00Z",
            )

    def test_duplicate_rejected_and_later_distinct_market_snapshot_is_accepted(self) -> None:
        self._baseline()
        first = self._capture()
        first_id = first.observation.observation_id
        before_retry = self.log.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "OBSERVATION_DUPLICATE"):
            self._capture(captured_at_utc="2026-10-02T12:21:00Z")
        self.assertEqual(self.log.read_bytes(), before_retry)
        later_tsl = _tsl_quote(fetched_at="2026-10-02T12:15:00Z", home_odds="1.95")
        later_reference = _ReferenceQuote(
            snapshot_id="ref-2", observed_at="2026-10-02T12:16:00Z", home="1.85", away="2.05"
        )
        second = self._capture(tsl=later_tsl, reference=later_reference)
        self.assertNotEqual(second.observation.observation_id, first_id)
        self.assertEqual(second.status, "EDGE_RECORDED")

    def test_model_and_market_timestamps_remain_independent(self) -> None:
        self._baseline()
        edge = self._capture().observation.to_dict()
        self.assertEqual(edge["signal_observed_at_utc"], "2026-10-02T12:00:00Z")
        self.assertEqual(edge["model_repriced_at_utc"], "2026-10-02T12:02:00Z")
        self.assertEqual(edge["reference_market_observation"]["observed_at_utc"], "2026-10-02T12:07:00Z")
        self.assertEqual(edge["tsl_observation"]["observed_at_utc"], "2026-10-02T12:06:00Z")
        self.assertEqual(edge["captured_at_utc"], "2026-10-02T12:20:00Z")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
