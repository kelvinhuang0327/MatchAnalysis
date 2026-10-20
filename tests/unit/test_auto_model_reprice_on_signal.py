from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlparse
from urllib.request import Request
from unittest.mock import patch

from match_analysis.application.use_cases.information_edge_capture_v1 import (
    InformationEdgeCaptureResult,
    capture_information_edge_v1,
)
from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    P50C_REPORT_RELATIVE_PATH,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p52a_daily_prediction_freeze import (
    execute_daily_moneyline_prediction_freeze,
)
from match_analysis.infrastructure.sources.mlb_starting_pitcher_signal import (
    normalize_starting_pitcher_schedule,
)
import match_analysis.interfaces.cli.capture_information_edge_v1 as information_edge_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TARGET_DATE = "2026-10-02"
TARGET_START = "2026-10-02T18:00:00Z"
P50C_RUNS = P50C_REPORT_RELATIVE_PATH / "runs"


def _schedule_payload(home_pitcher: int, away_pitcher: int = 202) -> dict:
    games = []
    first_day = datetime(2026, 9, 20, tzinfo=UTC)
    for index in range(10):
        day = first_day + timedelta(days=index)
        official_date = day.date().isoformat()
        games.append(
            {
                "gamePk": 8000 + index,
                "gameNumber": 1,
                "gameType": "R",
                "gameDate": f"{official_date}T19:00:00Z",
                "officialDate": official_date,
                "status": {"abstractGameState": "Final", "detailedState": "Final"},
                "teams": {
                    "home": {
                        "team": {"id": 11, "name": "Detroit Tigers", "abbreviation": "DET"},
                        "score": 4,
                    },
                    "away": {
                        "team": {"id": 22, "name": "Cleveland Guardians", "abbreviation": "CLE"},
                        "score": 2,
                    },
                },
            }
        )
    games.append(
        {
            "gamePk": 9001,
            "gameNumber": 1,
            "gameType": "R",
            "gameDate": TARGET_START,
            "officialDate": TARGET_DATE,
            "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            "teams": {
                "home": {
                    "team": {"id": 11, "name": "Detroit Tigers", "abbreviation": "DET"},
                    "probablePitcher": {"id": home_pitcher, "fullName": "Home Pitcher"},
                },
                "away": {
                    "team": {"id": 22, "name": "Cleveland Guardians", "abbreviation": "CLE"},
                    "probablePitcher": {"id": away_pitcher, "fullName": "Away Pitcher"},
                },
            },
        }
    )
    return {"dates": [{"date": TARGET_DATE, "games": games}]}


def _pitcher_logs(player_id: int, start_count: int) -> dict:
    earned_runs = {101: 1, 202: 2, 303: 5, 404: 3}.get(player_id, 2)
    return {
        "stats": [
            {
                "splits": [
                    {
                        "date": f"2026-08-{index + 1:02d}",
                        "gameType": "R",
                        "stat": {
                            "gamesStarted": 1,
                            "outs": 18,
                            "earnedRuns": earned_runs,
                        },
                    }
                    for index in range(start_count)
                ]
            }
        ]
    }


class MockP52AOpener:
    def __init__(self, home_pitcher: int, away_pitcher: int = 202, starts: dict[int, int] | None = None):
        self.schedule = _schedule_payload(home_pitcher, away_pitcher)
        self.starts = starts or {}
        self.calls: list[str] = []

    def __call__(self, request: Request, timeout: int = 30) -> bytes:
        self.calls.append(request.full_url)
        path = urlparse(request.full_url).path
        if path.endswith("/schedule"):
            return json.dumps(self.schedule).encode("utf-8")
        if "/people/" in path and path.endswith("/stats"):
            player_id = int(path.split("/people/")[1].split("/")[0])
            return json.dumps(
                _pitcher_logs(player_id, self.starts.get(player_id, 5))
            ).encode("utf-8")
        raise ValueError(f"unexpected MLB request: {request.full_url}")


def _signal(home_pitcher: int, received_at: str, away_pitcher: int = 202):
    return normalize_starting_pitcher_schedule(
        _schedule_payload(home_pitcher, away_pitcher),
        provider_game_id="9001",
        signal_observed_at_utc=received_at,
    )


def _frozen_run_bytes(run_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(run_dir.iterdir()) if path.is_file()}


class AutoModelRepriceOnSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.run_root = self.root / P50C_RUNS
        self.edge_log = self.root / "report/information_edge_capture_v1/edge_observations.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _before_run(self, opener: MockP52AOpener) -> str:
        result = execute_daily_moneyline_prediction_freeze(
            target_date=TARGET_DATE,
            as_of_utc="2026-10-02T11:30:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=opener,
        )
        self.assertEqual(result.target_games_count, 1)
        self.assertEqual(result.eligible_predictions_count, 1)
        self.assertIsNotNone(result.run_id)
        return result.run_id

    def _record_baseline(self, before_run_id: str) -> bytes:
        result = capture_information_edge_v1(
            schedule=_signal(101, "2026-10-02T11:40:00Z"),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            captured_at_utc="2026-10-02T11:41:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
        )
        self.assertEqual(result.status, "BASELINE_RECORDED")
        return self.edge_log.read_bytes()

    def test_auto_reprice_retry_and_distinct_signal_use_immutable_runs(self) -> None:
        before_run_id = self._before_run(MockP52AOpener(101))
        edge_log_bytes = self._record_baseline(before_run_id)
        before_dir = self.run_root / before_run_id
        before_bytes = _frozen_run_bytes(before_dir)
        before_record = read_jsonl_objects(before_dir / "frozen_predictions.jsonl")[0]

        first_opener = MockP52AOpener(303)
        first = capture_information_edge_v1(
            schedule=_signal(303, "2026-10-02T12:00:00Z"),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            before_run_id=before_run_id,
            captured_at_utc="2026-10-02T12:05:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=first_opener,
        )
        self.assertEqual(first.status, "MODEL_REPRICED")
        self.assertIsNone(first.observation)
        self.assertIsNotNone(first.after_run_id)
        self.assertEqual(first.reprice_as_of_utc, "2026-10-02T12:05:00Z")
        self.assertTrue(first.intake_path.is_file())
        self.assertTrue(any("/people/303/stats" in call for call in first_opener.calls))
        intake = read_json_object(first.intake_path)
        metadata = intake["source_manifest"]["auto_model_reprice_on_signal_v1"]
        self.assertEqual(metadata["before_run_id"], before_run_id)
        self.assertEqual(metadata["signal_identity"]["signal_observed_at_utc"], "2026-10-02T12:00:00Z")
        self.assertEqual(intake["source_manifest"]["fetched_at_utc"], first.reprice_as_of_utc)

        after_dir = self.run_root / first.after_run_id
        after_manifest = read_json_object(after_dir / "run_manifest.json")
        after_record = read_jsonl_objects(after_dir / "frozen_predictions.jsonl")[0]
        self.assertEqual(after_manifest["created_at_utc"], first.reprice_as_of_utc)
        self.assertEqual(after_record["game_identity"], before_record["game_identity"])
        self.assertEqual(
            (after_record["model_identity"]["model_role"], after_record["model_identity"]["model_id"], after_record["model_identity"]["model_fingerprint"]),
            (before_record["model_identity"]["model_role"], before_record["model_identity"]["model_id"], before_record["model_identity"]["model_fingerprint"]),
        )
        self.assertLess(
            read_json_object(before_dir / "run_manifest.json")["created_at_utc"],
            metadata["signal_identity"]["signal_observed_at_utc"],
        )
        self.assertLessEqual(metadata["signal_identity"]["signal_observed_at_utc"], after_manifest["created_at_utc"])
        self.assertLess(after_manifest["created_at_utc"], TARGET_START)
        self.assertEqual(_frozen_run_bytes(before_dir), before_bytes)
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)

        retry_opener = MockP52AOpener(303)
        retry = capture_information_edge_v1(
            schedule=_signal(303, "2026-10-02T12:10:00Z"),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            captured_at_utc="2026-10-02T12:11:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=retry_opener,
        )
        self.assertEqual(retry.after_run_id, first.after_run_id)
        self.assertEqual(retry.intake_path, first.intake_path)
        self.assertEqual(retry.reprice_as_of_utc, first.reprice_as_of_utc)
        self.assertEqual(retry_opener.calls, [])
        self.assertEqual(_frozen_run_bytes(before_dir), before_bytes)
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)

        second_signal = capture_information_edge_v1(
            schedule=_signal(303, "2026-10-02T12:20:00Z", away_pitcher=404),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            before_run_id=first.after_run_id,
            captured_at_utc="2026-10-02T12:25:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=MockP52AOpener(303, away_pitcher=404),
        )
        self.assertEqual(second_signal.status, "MODEL_REPRICED")
        self.assertNotEqual(second_signal.after_run_id, first.after_run_id)
        self.assertNotEqual(second_signal.intake_path, first.intake_path)
        self.assertEqual(
            read_json_object(second_signal.intake_path)["source_manifest"]["auto_model_reprice_on_signal_v1"]["signal_identity"]["changed_side"],
            "AWAY",
        )
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)

    def test_insufficient_changed_starter_history_fails_without_edge_or_after_run(self) -> None:
        before_run_id = self._before_run(MockP52AOpener(101))
        edge_log_bytes = self._record_baseline(before_run_id)
        runs_before = {path.name for path in self.run_root.iterdir() if path.is_dir()}

        with self.assertRaisesRegex(RuntimeError, "MODEL_REPRICE_NOT_AVAILABLE.*INSUFFICIENT_SAME_SEASON_STARTER_HISTORY"):
            capture_information_edge_v1(
                schedule=_signal(303, "2026-10-02T12:00:00Z"),
                prediction_run_root=self.run_root,
                observation_path=self.edge_log,
                before_run_id=before_run_id,
                captured_at_utc="2026-10-02T12:05:00Z",
                repository_root=REPOSITORY_ROOT,
                output_root=self.root,
                opener=MockP52AOpener(303, starts={303: 1}),
            )

        runs_after = {path.name for path in self.run_root.iterdir() if path.is_dir()}
        self.assertEqual(runs_after, runs_before)
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)
        intakes = list((self.root / P50C_REPORT_RELATIVE_PATH / "intake").glob("auto_model_reprice_on_signal_v1_*.json"))
        self.assertEqual(len(intakes), 1)
        intake = read_json_object(intakes[0])
        self.assertEqual(intake["predictions"], [])
        self.assertTrue(
            any(
                "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY" in row.get("reason", "")
                for row in intake["exclusions"]
            )
        )

    def test_existing_mismatching_signal_intake_is_rejected_without_overwrite(self) -> None:
        before_run_id = self._before_run(MockP52AOpener(101))
        edge_log_bytes = self._record_baseline(before_run_id)
        first = capture_information_edge_v1(
            schedule=_signal(303, "2026-10-02T12:00:00Z"),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            before_run_id=before_run_id,
            captured_at_utc="2026-10-02T12:05:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=MockP52AOpener(303),
        )
        payload = read_json_object(first.intake_path)
        payload["source_manifest"]["auto_model_reprice_on_signal_v1"]["before_run_id"] = "different-before"
        first.intake_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        conflicting_bytes = first.intake_path.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "MODEL_REPRICE_NOT_AVAILABLE.*different before authority"):
            capture_information_edge_v1(
                schedule=_signal(303, "2026-10-02T12:10:00Z"),
                prediction_run_root=self.run_root,
                observation_path=self.edge_log,
                before_run_id=before_run_id,
                captured_at_utc="2026-10-02T12:11:00Z",
                repository_root=REPOSITORY_ROOT,
                output_root=self.root,
            )

        self.assertEqual(first.intake_path.read_bytes(), conflicting_bytes)
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)

    def test_ambiguous_and_missing_id_transitions_do_not_reprice(self) -> None:
        before_run_id = self._before_run(MockP52AOpener(101))
        edge_log_bytes = self._record_baseline(before_run_id)
        runs_before = {path.name for path in self.run_root.iterdir() if path.is_dir()}
        ambiguous_opener = MockP52AOpener(303, away_pitcher=404)

        with self.assertRaisesRegex(RuntimeError, "AMBIGUOUS_STARTING_PITCHER_CHANGE"):
            capture_information_edge_v1(
                schedule=_signal(303, "2026-10-02T12:00:00Z", away_pitcher=404),
                prediction_run_root=self.run_root,
                observation_path=self.edge_log,
                before_run_id=before_run_id,
                captured_at_utc="2026-10-02T12:05:00Z",
                repository_root=REPOSITORY_ROOT,
                output_root=self.root,
                opener=ambiguous_opener,
            )
        self.assertEqual(ambiguous_opener.calls, [])
        self.assertEqual(self.edge_log.read_bytes(), edge_log_bytes)

        missing_id = capture_information_edge_v1(
            schedule=_signal(303, "2026-10-02T12:10:00Z", away_pitcher=None),
            prediction_run_root=self.run_root,
            observation_path=self.edge_log,
            captured_at_utc="2026-10-02T12:11:00Z",
            repository_root=REPOSITORY_ROOT,
            output_root=self.root,
            opener=MockP52AOpener(303, away_pitcher=202),
        )
        self.assertEqual(missing_id.status, "BASELINE_REFRESHED")
        self.assertIsNone(missing_id.after_run_id)
        self.assertEqual(
            {path.name for path in self.run_root.iterdir() if path.is_dir()}, runs_before
        )
        self.assertEqual(
            list((self.root / P50C_REPORT_RELATIVE_PATH / "intake").glob("auto_model_reprice_on_signal_v1_*.json")),
            [],
        )

    def test_cli_starts_automatic_reprice_without_after_run_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_path = root / "schedule.json"
            schedule_path.write_text(json.dumps(_schedule_payload(101)), encoding="utf-8")
            intake_path = root / "signal-intake.json"
            response = InformationEdgeCaptureResult(
                "MODEL_REPRICED",
                None,
                after_run_id="p50c_run_auto",
                intake_path=intake_path,
                reprice_as_of_utc="2026-10-02T12:05:00Z",
            )
            clock_values = iter(
                (
                    datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
                    datetime(2026, 10, 2, 12, 5, tzinfo=UTC),
                )
            )
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO

            stdout, stderr = StringIO(), StringIO()
            with patch.object(
                information_edge_cli, "capture_information_edge_v1", return_value=response
            ) as capture, redirect_stdout(stdout), redirect_stderr(stderr):
                status = information_edge_cli.main(
                    [
                        "--repository-root",
                        str(root),
                        "--schedule-json",
                        str(schedule_path),
                        "--provider-game-id",
                        "9001",
                        "--before-run-id",
                        "before",
                    ],
                    clock=lambda: next(clock_values),
                )

            self.assertEqual(status, 0, stderr.getvalue())
            self.assertIsNone(capture.call_args.kwargs["after_run_id"])
            self.assertEqual(capture.call_args.kwargs["repository_root"], root.resolve())
            self.assertEqual(capture.call_args.kwargs["output_root"], root.resolve())
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["after_run_id"], "p50c_run_auto")
            self.assertEqual(output["reprice_as_of_utc"], "2026-10-02T12:05:00Z")


if __name__ == "__main__":
    unittest.main()
