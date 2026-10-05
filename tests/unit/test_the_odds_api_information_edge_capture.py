from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.information_edge_capture_v1 import (
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
from match_analysis.baseball.domain.information_edge_observation import SIGNAL_BASELINE
from match_analysis.infrastructure.providers.the_odds_api_reference_market_quote import (
    TheOddsApiReferenceMarketQuote,
)
from match_analysis.infrastructure.sources.mlb_starting_pitcher_signal import (
    StartingPitcherScheduleObservation,
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
API_KEY = "integration-secret-must-not-persist"


def _schedule(home_pitcher: int, observed_at: str) -> StartingPitcherScheduleObservation:
    return StartingPitcherScheduleObservation(
        provider_namespace="mlb_official",
        provider_game_id="9001",
        official_date="2026-10-02",
        game_type="R",
        game_number=1,
        scheduled_start_utc=START,
        home_team={"id": 11, "name": "Detroit Tigers", "abbreviation": "DET"},
        away_team={"id": 22, "name": "Cleveland Guardians", "abbreviation": "CLE"},
        home_probable_pitcher_player_id=str(home_pitcher),
        away_probable_pitcher_player_id="202",
        signal_observed_at_utc=observed_at,
    )


def _write_run(root: Path, run_id: str, created_at: str, p_home: str) -> None:
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
        scheduled_start_utc=START,
        champion_model_id="model-1",
        champion_model_fingerprint="model-fingerprint",
        champion_home_probability=Decimal(p_home),
        challenger_model_id="model-1",
        challenger_model_fingerprint="model-fingerprint",
        challenger_home_probability=Decimal(p_home),
        game_type="R",
    )
    prediction = build_frozen_prediction_record(
        row,
        run_id=run_id,
        run_classification=CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
        created_at_utc=created_at,
        official_date="2026-10-02",
        home_team="Detroit Tigers",
        away_team="Cleveland Guardians",
        home_team_code="DET",
        away_team_code="CLE",
    )
    fingerprint = prediction["prediction_fingerprint"]
    manifest = {
        "run_id": run_id,
        "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
        "game_type": "R",
        "prediction_fingerprints": [fingerprint],
        "prediction_bundle_fingerprint": sha256(
            (json.dumps([fingerprint], sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "frozen_predictions.jsonl").write_text(
        json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class TheOddsApiInformationEdgeCaptureTests(unittest.TestCase):
    def test_live_quote_is_consumed_and_both_timestamp_authorities_are_persisted(self) -> None:
        response = json.dumps(
            [
                {
                    "id": "source-event-77",
                    "sport_key": "baseball_mlb",
                    "commence_time": START,
                    "home_team": "Detroit Tigers",
                    "away_team": "Cleveland Guardians",
                    "bookmakers": [
                        {
                            "key": "fanduel",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "last_update": "2026-10-02T12:07:00Z",
                                    "outcomes": [
                                        {"name": "Cleveland Guardians", "price": 2.22},
                                        {"name": "Detroit Tigers", "price": 1.72},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            _write_run(run_root, "before", "2026-10-02T11:55:00Z", "0.55")
            _write_run(run_root, "after", "2026-10-02T12:02:00Z", "0.62")
            observations = root / "edge_observations.jsonl"
            baseline = capture_information_edge_v1(
                schedule=_schedule(101, "2026-10-02T11:50:00Z"),
                prediction_run_root=run_root,
                observation_path=observations,
                captured_at_utc="2026-10-02T11:51:00Z",
            )
            self.assertEqual(baseline.observation.record_type, SIGNAL_BASELINE)

            adapter = TheOddsApiReferenceMarketQuote(
                api_key=API_KEY,
                http_get=lambda _url: response,
                clock=lambda: datetime.fromisoformat("2026-10-02T12:08:03+00:00"),
            )
            tsl = TslMoneylineQuote(
                snapshot_id="tsl-1",
                game_identity_json=json.dumps(IDENTITY, sort_keys=True, separators=(",", ":")),
                scheduled_start_utc=START,
                observed_at_utc="2026-10-02T12:06:00Z",
                home_decimal_odds=Decimal("1.90"),
                away_decimal_odds=Decimal("2.10"),
            )
            result = capture_information_edge_v1(
                schedule=_schedule(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=run_root,
                observation_path=observations,
                before_run_id="before",
                after_run_id="after",
                tsl_quote=tsl,
                reference_market_quote=adapter,
                captured_at_utc="2026-10-02T12:10:00Z",
            )

            self.assertEqual(result.status, "EDGE_RECORDED")
            edge = result.observation.to_dict()
            market = edge["reference_market_observation"]
            self.assertEqual(market["source_provider"], "the_odds_api_v4")
            self.assertEqual(market["source_event_id"], "source-event-77")
            self.assertEqual(market["bookmaker_key"], "fanduel")
            self.assertEqual(market["market_key"], "h2h")
            self.assertEqual(market["provider_observed_at_utc"], "2026-10-02T12:07:00Z")
            self.assertEqual(market["retrieved_at_utc"], "2026-10-02T12:08:03Z")
            self.assertEqual(market["observed_at_utc"], "2026-10-02T12:08:03Z")
            self.assertEqual(market["home_decimal_odds"], "1.72")
            self.assertEqual(market["away_decimal_odds"], "2.22")
            self.assertEqual(edge["game_identity"]["provider_game_id"], "9001")
            self.assertNotEqual(market["source_event_id"], edge["game_identity"]["provider_game_id"])
            persisted = observations.read_text(encoding="utf-8")
            self.assertIn(market["raw_response_sha256"], persisted)
            self.assertNotIn(API_KEY, persisted)
            self.assertNotIn(API_KEY, json.dumps(edge, default=str))


if __name__ == "__main__":
    unittest.main()
