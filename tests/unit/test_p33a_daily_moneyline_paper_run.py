"""Focused P33A frozen-bundle and replay contracts."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.acquire_tsl_moneyline_snapshot import (
    TslMoneylineSnapshotAcquisition,
    _select_target_slate,
)
from match_analysis.application.use_cases.moneyline_paper_run_bundle import (
    jsonl_fingerprint,
    target_game_membership,
)
from match_analysis.application.use_cases.run_daily_moneyline_paper_analysis import (
    P33A_RUNTIME_ROOT,
    replay_daily_moneyline_paper_analysis,
    run_daily_moneyline_paper_analysis,
)
from match_analysis.infrastructure.sources.tsl_moneyline_acquisition import (
    TslBlob3rdRawCapture,
    build_tsl_moneyline_history,
)


P32A_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/p32a_tsl_acquisition/tsl_pre_games_v1.json"
)


def _fixture_game(index: int, *, away: str, home: str) -> dict[str, object]:
    game = json.loads(P32A_FIXTURE.read_text(encoding="utf-8"))[0]
    game["id"] = f"fixture-{index}"
    game["an"] = away
    game["hn"] = home
    game["ms"][0]["cs"] = [
        {"name": away, "pd": "50", "pu": "37", "hv": None},
        {"name": home, "pd": "25", "pu": "19", "hv": None},
    ]
    return game


def _schedule_row(
    *,
    game_id: str,
    game_pk: int,
    away_code: str,
    away_name: str,
    home_code: str,
    home_name: str,
) -> dict[str, object]:
    return {
        "schema_version": "p23f2.mlb_official_normalized.v1",
        "provider_game_id": game_id,
        "game_pk": game_pk,
        "game_number": 1,
        "official_date": "2026-08-12",
        "scheduled_start_utc": "2026-08-12T22:40:00Z",
        "status": "Scheduled",
        "final": False,
        "home_team": {"id": game_pk + 10, "abbreviation": home_code, "name": home_name},
        "away_team": {"id": game_pk + 20, "abbreviation": away_code, "name": away_name},
        "home_score": None,
        "away_score": None,
    }


def _acquisition() -> TslMoneylineSnapshotAcquisition:
    games = (
        _fixture_game(1, away="克里夫蘭守護者", home="底特律老虎"),
        _fixture_game(2, away="匹茲堡海盜", home="邁阿密馬林魚"),
    )
    payload = json.dumps(games, ensure_ascii=False).encode("utf-8")
    capture = TslBlob3rdRawCapture(
        sport_id="34731.1",
        games=games,
        payloads=(("fixture", payload),),
    )
    history, normalization = build_tsl_moneyline_history(
        capture,
        fetched_at="2026-08-11T15:00:00Z",
        target_date="2026-08-12",
    )
    schedule = (
        _schedule_row(
            game_id="9001",
            game_pk=9001,
            away_code="CLE",
            away_name="Cleveland Guardians",
            home_code="DET",
            home_name="Detroit Tigers",
        ),
        _schedule_row(
            game_id="9002",
            game_pk=9002,
            away_code="PIT",
            away_name="Pittsburgh Pirates",
            home_code="MIA",
            home_name="Miami Marlins",
        ),
    )
    return TslMoneylineSnapshotAcquisition(
        operation="ACQUIRE_TSL_MONEYLINE_SNAPSHOT",
        target_date="2026-08-12",
        selection_started_at_utc="2026-08-11T14:59:00Z",
        fetched_at_utc="2026-08-11T15:00:00Z",
        schedule_url="fixture://mlb/schedule",
        schedule_rows=schedule,
        target_schedule_rows=schedule,
        requested_game_ids=("9001", "9002"),
        history=history,
        normalization=normalization,
        source_payload_sha256=(("fixture", sha256(payload).hexdigest()),),
        runtime_capture_paths=(),
    )


def _fake_p30a_result(*, repository_root: Path, **_: object) -> SimpleNamespace:
    del repository_root
    analysis = (
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "p30a-frozen-test-run",
            "game_id": "9001",
            "scheduled_start": "2026-08-12T22:40:00Z",
            "structural_status": "FEATURE_UNAVAILABLE",
            "status": "FEATURE_UNAVAILABLE",
        },
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "p30a-frozen-test-run",
            "game_id": "9002",
            "scheduled_start": "2026-08-12T22:40:00Z",
            "structural_status": "FEATURE_UNAVAILABLE",
            "status": "FEATURE_UNAVAILABLE",
        },
    )
    summary = {
        "schema_version": "p30a.moneyline_paper_analysis.v1",
        "operation": "MONEYLINE_PAPER_ANALYSIS_RUN",
        "run_id": "p30a-frozen-test-run",
        "analysis_set_fingerprint": jsonl_fingerprint(analysis),
        "raw_game_count": 2,
        "edge_available_count": 0,
        "feature_unavailable_count": 2,
        "price_unavailable_pre_cutoff_count": 0,
        "structural_status_counts": {
            "EDGE_AVAILABLE": 0,
            "FEATURE_UNAVAILABLE": 2,
            "PRICE_UNAVAILABLE_PRE_CUTOFF": 0,
            "CROSSWALK_UNRESOLVED": 0,
        },
    }
    return SimpleNamespace(analysis=analysis, summary=summary)


class P33ADailyMoneylinePaperRunTests(unittest.TestCase):
    def test_explicit_target_date_uses_official_timing(self) -> None:
        rows = (
            {
                "provider_game_id": "1",
                "scheduled_start_utc": "2026-08-11T22:40:00Z",
                "game_number": 1,
                "game_pk": 1,
            },
            {
                "provider_game_id": "2",
                "scheduled_start_utc": "2026-08-11T23:00:00Z",
                "game_number": 1,
                "game_pk": 2,
            },
        )
        target_date, target_rows = _select_target_slate(
            rows,
            started_at_utc=datetime(2026, 8, 11, 15, 0, tzinfo=UTC),
            requested_target_date="2026-08-12",
        )
        self.assertEqual(target_date, "2026-08-12")
        self.assertEqual([row["provider_game_id"] for row in target_rows], ["1", "2"])

    def test_game_identity_excludes_result_fields(self) -> None:
        rows = (_schedule_row(
            game_id="9001",
            game_pk=9001,
            away_code="CLE",
            away_name="Cleveland Guardians",
            home_code="DET",
            home_name="Detroit Tigers",
        ),)
        changed = deepcopy(rows[0])
        changed["status"] = "Final"
        changed["final"] = True
        changed["home_score"] = 10
        changed["away_score"] = 0
        self.assertEqual(
            target_game_membership(rows),
            target_game_membership((changed,)),
        )

    def test_one_acquisition_is_frozen_and_two_replays_are_offline(self) -> None:
        acquisition = _acquisition()
        calls = {"acquisition": 0, "analysis": 0}

        def acquire_once(**kwargs: object) -> TslMoneylineSnapshotAcquisition:
            calls["acquisition"] += 1
            self.assertEqual(kwargs["target_date"], "2026-08-12")
            capture_root = Path(kwargs["runtime_root"]) / "capture"
            capture_root.mkdir(parents=True, exist_ok=True)
            capture_path = capture_root / "tsl_live_games.json"
            capture_path.write_bytes(b"frozen raw TSL response")
            return replace(
                acquisition,
                runtime_capture_paths=(str(capture_path),),
            )

        def analysis_once(**kwargs: object) -> SimpleNamespace:
            calls["analysis"] += 1
            return _fake_p30a_result(**kwargs)

        P33A_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="p33a-unit-",
            dir=P33A_RUNTIME_ROOT,
        ) as temporary:
            runtime_root = Path(temporary)
            live = run_daily_moneyline_paper_analysis(
                repository_root=REPOSITORY_ROOT,
                runtime_root=runtime_root,
                date_value="2026-08-12",
                acquisition_factory=acquire_once,
                analysis_runner=analysis_once,
            )
            self.assertEqual(calls["acquisition"], 1)
            self.assertEqual(calls["analysis"], 1)
            self.assertTrue(live.run_manifest["frozen_before_analysis"])
            self.assertEqual(live.summary["target_date"], "2026-08-12")
            self.assertEqual(live.summary["official_target_game_count"], 2)
            self.assertEqual(live.summary["official_overlap_game_count"], 2)
            self.assertEqual(live.summary["raw_game_count"], 2)
            self.assertEqual(
                live.summary["analysis_terminal_state_counts"]["FEATURE_UNAVAILABLE"],
                2,
            )
            self.assertFalse(live.summary["betting_decision_generated"])
            self.assertFalse(live.summary["staking_implemented"])
            self.assertFalse(live.summary["profitability_claim"])
            self.assertTrue((live.bundle_root / "run_manifest.json").is_file())
            self.assertTrue((live.bundle_root / "tsl_source_snapshot.jsonl").is_file())
            self.assertTrue((live.bundle_root / "mlb_source_snapshot.jsonl").is_file())
            self.assertEqual(
                (live.bundle_root / "capture/tsl_live_games.json").read_bytes(),
                b"frozen raw TSL response",
            )

            with patch(
                "match_analysis.application.use_cases.run_daily_moneyline_paper_analysis.acquire_tsl_moneyline_snapshot",
                side_effect=AssertionError("replay attempted acquisition"),
            ):
                first = replay_daily_moneyline_paper_analysis(
                    repository_root=REPOSITORY_ROOT,
                    bundle_path=live.bundle_root,
                    runtime_root=runtime_root,
                    output_dir=runtime_root / "replay-1",
                    analysis_runner=analysis_once,
                )
                second = replay_daily_moneyline_paper_analysis(
                    repository_root=REPOSITORY_ROOT,
                    bundle_path=live.bundle_root,
                    runtime_root=runtime_root,
                    output_dir=runtime_root / "replay-2",
                    analysis_runner=analysis_once,
                )
            self.assertTrue(first.offline_replay_equal)
            self.assertTrue(second.offline_replay_equal)
            self.assertEqual(first.analysis, live.analysis)
            self.assertEqual(second.analysis, live.analysis)
            self.assertEqual(first.summary, live.summary)
            self.assertEqual(second.summary, live.summary)
            for name in ("analysis.jsonl", "summary.json", "run_manifest.json"):
                self.assertEqual(
                    (first.bundle_root / name).read_bytes(),
                    (second.bundle_root / name).read_bytes(),
                )
            self.assertEqual(calls["acquisition"], 1)
            self.assertEqual(calls["analysis"], 3)


if __name__ == "__main__":
    unittest.main()
