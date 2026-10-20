from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from typing import Callable

from match_analysis.application.use_cases.p40a_moneyline_paper_bet_pass import (
    P40APredictionRow,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    P50C_REPORT_RELATIVE_PATH,
    build_frozen_prediction_record,
)
from match_analysis.interfaces.cli.capture_information_edge_v1 import main


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


def _schedule_payload(home_pitcher: int) -> dict:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 9001,
                        "gameNumber": 1,
                        "gameType": "R",
                        "gameDate": START,
                        "officialDate": "2026-10-02",
                        "teams": {
                            "home": {
                                "team": {
                                    "id": 11,
                                    "name": "Detroit Tigers",
                                    "abbreviation": "DET",
                                },
                                "probablePitcher": {
                                    "id": home_pitcher,
                                    "fullName": "Home Pitcher",
                                },
                            },
                            "away": {
                                "team": {
                                    "id": 22,
                                    "name": "Cleveland Guardians",
                                    "abbreviation": "CLE",
                                },
                                "probablePitcher": {
                                    "id": 202,
                                    "fullName": "Away Pitcher",
                                },
                            },
                        },
                    }
                ]
            }
        ]
    }


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
        scheduled_start_utc=IDENTITY["scheduled_start_utc"],
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
        "game_type": "R",
        "prediction_fingerprints": [fingerprint],
        "prediction_bundle_fingerprint": sha256(
            (json.dumps([fingerprint], sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "frozen_predictions.jsonl").write_text(
        json.dumps(prediction, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _tsl_quote_payload() -> dict:
    return {
        "provider_game_id": "9001",
        "game_identity": IDENTITY,
        "fetched_at": "2026-10-02T12:06:00Z",
        "game": {
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
                        {"name": "底特律老虎", "pd": "100", "pu": "90", "hv": None},
                    ],
                }
            ],
        },
    }


def _clock_values(*values: tuple[int, int]) -> Callable[[], datetime]:
    return iter(datetime(2026, 10, 2, hour, minute, tzinfo=UTC) for hour, minute in values).__next__


class CaptureInformationEdgeV1CliTests(unittest.TestCase):
    def test_local_cli_smoke_records_baseline_then_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = Path(temp_dir)
            run_root = repository / P50C_REPORT_RELATIVE_PATH / "runs"
            _write_run(run_root, "before", "2026-10-02T11:55:00Z", "0.55")
            _write_run(run_root, "after", "2026-10-02T12:02:00Z", "0.62")
            schedule_path = repository / "schedule.json"
            tsl_path = repository / "tsl.json"
            reference_path = repository / "reference.json"
            schedule_path.write_text(json.dumps(_schedule_payload(101)), encoding="utf-8")
            tsl_path.write_text(json.dumps(_tsl_quote_payload()), encoding="utf-8")
            reference_path.write_text(
                json.dumps(
                    {
                        "snapshot_id": "ref-1",
                        "game_identity": IDENTITY,
                        "observed_at_utc": "2026-10-02T12:07:00Z",
                        "home_decimal_odds": "1.80",
                        "away_decimal_odds": "2.10",
                    }
                ),
                encoding="utf-8",
            )

            common_args = [
                "--repository-root",
                str(repository),
                "--schedule-json",
                str(schedule_path),
                "--provider-game-id",
                "9001",
            ]
            baseline_stdout = StringIO()
            baseline_stderr = StringIO()
            with redirect_stdout(baseline_stdout), redirect_stderr(baseline_stderr):
                baseline_status = main(
                    common_args,
                    clock=_clock_values((11, 50), (11, 51)),
                )
            self.assertEqual(baseline_status, 0, baseline_stderr.getvalue())
            self.assertEqual(json.loads(baseline_stdout.getvalue())["status"], "BASELINE_RECORDED")

            log_path = repository / "report/information_edge_capture_v1/edge_observations.jsonl"
            baseline_bytes = log_path.read_bytes()
            schedule_path.write_text(json.dumps(_schedule_payload(303)), encoding="utf-8")
            edge_stdout = StringIO()
            edge_stderr = StringIO()
            with redirect_stdout(edge_stdout), redirect_stderr(edge_stderr):
                edge_status = main(
                    [
                        *common_args,
                        "--before-run-id",
                        "before",
                        "--after-run-id",
                        "after",
                        "--tsl-quote-json",
                        str(tsl_path),
                        "--reference-quote-json",
                        str(reference_path),
                    ],
                    clock=_clock_values((12, 0), (12, 20)),
                )
            self.assertEqual(edge_status, 0, edge_stderr.getvalue())
            result = json.loads(edge_stdout.getvalue())
            self.assertEqual(result["status"], "EDGE_RECORDED")
            self.assertEqual(result["observation"]["signal_observed_at_utc"], "2026-10-02T12:00:00Z")
            self.assertEqual(result["observation"]["model_repriced_at_utc"], "2026-10-02T12:02:00Z")
            self.assertTrue(log_path.read_bytes().startswith(baseline_bytes))
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(
                [record["record_type"] for record in records],
                ["SIGNAL_BASELINE", "EDGE_OBSERVATION", "SIGNAL_BASELINE"],
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
