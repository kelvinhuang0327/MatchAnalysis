"""Focused P31A native-source and bounded date-scope contracts."""

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.acquire_future_moneyline_history import (
    load_normalized_rows,
)
from match_analysis.application.use_cases.build_moneyline_paper_source_bundle import (
    build_moneyline_paper_source_bundle,
    resolve_date_scope,
    select_precohort_window,
)
from match_analysis.application.use_cases.run_moneyline_paper_analysis import (
    run_moneyline_paper_analysis,
)
from match_analysis.infrastructure.sources.tsl_moneyline_history import (
    TSL_AUTHORITY_RAW_SHA256,
    load_tsl_moneyline_history,
)


AUTHORITY_PATH = REPOSITORY_ROOT / "data/authority/tsl/tsl_odds_history.jsonl"
AUTHORITY_MANIFEST_PATH = REPOSITORY_ROOT / "data/authority/tsl/source_manifest.json"
P28AB_ROOT = REPOSITORY_ROOT / "data/fixtures/p28ab_tsl_aligned_moneyline_edge"
P28AB_MANIFEST_PATH = P28AB_ROOT / "source_manifest.json"
SCHEDULE_PATH = (
    REPOSITORY_ROOT
    / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
)
BOX_PATH = P28AB_ROOT / "normalized/target_boxscores.jsonl"
LOG_PATH = P28AB_ROOT / "normalized/pitcher_game_logs.jsonl"
TARGET_IDS = {
    "822738",
    "824192",
    "823708",
    "823138",
    "824114",
    "824680",
    "823865",
    "823548",
    "823704",
    "824112",
    "824677",
    "822977",
    "824273",
    "824518",
    "824674",
    "823862",
}


class P31ANativeTslSourceBundleTests(unittest.TestCase):
    def test_native_loader_matches_frozen_p28ab_qualification(self) -> None:
        native = load_tsl_moneyline_history(
            AUTHORITY_PATH,
            start_date="2026-05-18",
            end_date="2026-05-24",
        )
        fixture_rows = tuple(
            json.loads(line)
            for line in (P28AB_ROOT / "tsl_odds_history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
        self.assertEqual(native.raw_sha256, TSL_AUTHORITY_RAW_SHA256)
        self.assertEqual(native.selected_rows_sha256, "8415b8769885207f7baa6a064efec7223ed3dc8cf2a6842bd67c1ecf8ff62107")
        self.assertEqual(native.qualified_row_count, 46)
        self.assertEqual(
            Counter((row["match_id"], row["fetched_at"]) for row in native.rows),
            Counter((row["match_id"], row["fetched_at"]) for row in fixture_rows),
        )
        for observation in native.observations:
            self.assertEqual(observation.market_code, "MNL")
            self.assertEqual(observation.provider_source, "TSL_BLOB3RD")
            self.assertEqual(observation.source_identifier, observation.row["match_id"])

    def test_native_parity_bundle_preserves_committed_p30a_identity(self) -> None:
        bundle = build_moneyline_paper_source_bundle(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            source_manifest_path=P28AB_MANIFEST_PATH,
            schedule_path=SCHEDULE_PATH,
            target_boxscores_path=BOX_PATH,
            pitcher_game_logs_path=LOG_PATH,
            start_date="2026-05-18",
            end_date="2026-05-24",
            parity_mode=True,
        )
        result = run_moneyline_paper_analysis(
            repository_root=REPOSITORY_ROOT,
            tsl_rows=bundle.tsl_history.rows,
            tsl_raw_sha256=bundle.tsl_history.selected_rows_sha256,
            schedule_rows=bundle.schedule_rows,
            target_boxscore_rows=bundle.target_boxscore_rows,
            pitcher_game_log_rows=bundle.pitcher_game_log_rows,
            source_manifest=bundle.source_manifest,
            offline_replay_verified=True,
            cohort_start_date=bundle.scope_start_date,
            cohort_end_date=bundle.scope_end_date,
            requested_game_ids=bundle.requested_game_ids,
        )
        self.assertIsNone(bundle.requested_game_ids)
        self.assertEqual(
            result.summary["run_id"],
            "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
        )
        self.assertEqual(result.summary["raw_game_count"], 16)
        self.assertEqual(result.summary["edge_available_count"], 9)
        self.assertEqual(result.summary["feature_unavailable_count"], 7)
        committed_analysis = tuple(
            json.loads(line)
            for line in (REPOSITORY_ROOT / "report/p30a_moneyline_paper_analysis/analysis.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
        committed_summary = json.loads(
            (REPOSITORY_ROOT / "report/p30a_moneyline_paper_analysis/summary.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(result.analysis, committed_analysis)
        self.assertEqual(result.summary, committed_summary)

    def test_precohort_window_is_nearest_seven_and_non_overlapping(self) -> None:
        schedule_rows = load_normalized_rows(SCHEDULE_PATH)
        window = select_precohort_window(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            schedule_rows=schedule_rows,
            p30a_game_ids=TARGET_IDS,
        )
        self.assertEqual(window.selected_start_date, "2026-05-11")
        self.assertEqual(window.selected_end_date, "2026-05-17")
        self.assertEqual(window.length, 7)
        self.assertFalse(window.fallback_used)
        self.assertEqual(set(window.official_game_ids).intersection(TARGET_IDS), set())
        self.assertTrue(window.selected_end_date < window.p30a_cohort_start_date)
        self.assertFalse(window.metadata()["outcome_based_selection"])
        outcome_mutated_schedule = tuple(
            {
                **row,
                "home_score": 999,
                "away_score": 0,
            }
            for row in schedule_rows
        )
        mutated_window = select_precohort_window(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            schedule_rows=outcome_mutated_schedule,
            p30a_game_ids=TARGET_IDS,
        )
        self.assertEqual(window.metadata(), mutated_window.metadata())
        with self.assertRaisesRegex(
            RuntimeError, "STOP_MATCHANALYSIS_P31A_PRECOHORT_OVERLAP"
        ):
            select_precohort_window(
                tsl_history_path=AUTHORITY_PATH,
                tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
                schedule_rows=schedule_rows,
                p30a_game_ids=(window.official_game_ids[0],),
            )

    def test_precohort_run_accounts_for_every_official_game(self) -> None:
        schedule_rows = load_normalized_rows(SCHEDULE_PATH)
        window = select_precohort_window(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            schedule_rows=schedule_rows,
            p30a_game_ids=TARGET_IDS,
        )
        bundle = build_moneyline_paper_source_bundle(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            source_manifest_path=P28AB_MANIFEST_PATH,
            schedule_path=SCHEDULE_PATH,
            target_boxscores_path=BOX_PATH,
            pitcher_game_logs_path=LOG_PATH,
            start_date=window.selected_start_date,
            end_date=window.selected_end_date,
            parity_mode=False,
            selection_metadata=window.metadata(),
            p30a_game_ids=TARGET_IDS,
        )
        result = run_moneyline_paper_analysis(
            repository_root=REPOSITORY_ROOT,
            tsl_rows=bundle.tsl_history.rows,
            tsl_raw_sha256=bundle.tsl_history.selected_rows_sha256,
            schedule_rows=bundle.schedule_rows,
            target_boxscore_rows=bundle.target_boxscore_rows,
            pitcher_game_log_rows=bundle.pitcher_game_log_rows,
            source_manifest=bundle.source_manifest,
            offline_replay_verified=True,
            cohort_start_date=bundle.scope_start_date,
            cohort_end_date=bundle.scope_end_date,
            requested_game_ids=bundle.requested_game_ids,
            allow_missing_starter_identity=True,
            allow_insufficient_evaluable=True,
        )
        self.assertEqual(result.summary["run_id"], "e663bcc9758ba2c179beca72bb7fec29202838a38c45b24a53644d21c4e84f01")
        self.assertEqual(result.summary["raw_game_count"], 92)
        self.assertEqual(result.summary["edge_available_count"], 0)
        self.assertEqual(result.summary["feature_unavailable_count"], 92)
        self.assertEqual(result.summary["p28ab_raw_source_row_count"], 98)
        self.assertEqual(result.summary["p28ab_selected_price_count"], 42)
        self.assertEqual(result.summary["p28ab_descriptive_edge_row_count"], 0)
        self.assertEqual(
            Counter(row["structural_status"] for row in result.analysis),
            Counter({"FEATURE_UNAVAILABLE": 92}),
        )
        self.assertTrue(result.summary["deterministic_replay_verified"])
        self.assertTrue(result.summary["outcome_isolation_verified"])

    def test_precohort_run_is_invariant_to_input_order(self) -> None:
        schedule_rows = load_normalized_rows(SCHEDULE_PATH)
        window = select_precohort_window(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            schedule_rows=schedule_rows,
            p30a_game_ids=TARGET_IDS,
        )
        bundle = build_moneyline_paper_source_bundle(
            tsl_history_path=AUTHORITY_PATH,
            tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
            source_manifest_path=P28AB_MANIFEST_PATH,
            schedule_path=SCHEDULE_PATH,
            target_boxscores_path=BOX_PATH,
            pitcher_game_logs_path=LOG_PATH,
            start_date=window.selected_start_date,
            end_date=window.selected_end_date,
            parity_mode=False,
            selection_metadata=window.metadata(),
            p30a_game_ids=TARGET_IDS,
        )
        inputs = {
            "repository_root": REPOSITORY_ROOT,
            "tsl_rows": bundle.tsl_history.rows,
            "tsl_raw_sha256": bundle.tsl_history.selected_rows_sha256,
            "schedule_rows": bundle.schedule_rows,
            "target_boxscore_rows": bundle.target_boxscore_rows,
            "pitcher_game_log_rows": bundle.pitcher_game_log_rows,
            "source_manifest": bundle.source_manifest,
            "offline_replay_verified": True,
            "cohort_start_date": bundle.scope_start_date,
            "cohort_end_date": bundle.scope_end_date,
            "requested_game_ids": bundle.requested_game_ids,
            "allow_missing_starter_identity": True,
            "allow_insufficient_evaluable": True,
        }
        first = run_moneyline_paper_analysis(**inputs)
        reversed_inputs = dict(inputs)
        for key in ("tsl_rows", "schedule_rows", "target_boxscore_rows", "pitcher_game_log_rows"):
            reversed_inputs[key] = tuple(reversed(reversed_inputs[key]))
        second = run_moneyline_paper_analysis(**reversed_inputs)
        self.assertEqual(first.analysis, second.analysis)
        self.assertEqual(first.summary, second.summary)

    def test_explicit_bounded_range_drives_requested_game_membership(self) -> None:
        schedule_rows = load_normalized_rows(SCHEDULE_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            schedule_path = Path(temporary) / "schedule.jsonl"
            selected = [
                row
                for row in schedule_rows
                if str(row["provider_game_id"]) in TARGET_IDS
                or str(row["official_date"]) < "2026-05-17"
            ]
            schedule_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in selected
                ),
                encoding="utf-8",
            )
            bundle = build_moneyline_paper_source_bundle(
                tsl_history_path=AUTHORITY_PATH,
                tsl_authority_manifest_path=AUTHORITY_MANIFEST_PATH,
                source_manifest_path=P28AB_MANIFEST_PATH,
                schedule_path=schedule_path,
                target_boxscores_path=BOX_PATH,
                pitcher_game_logs_path=LOG_PATH,
                start_date="2026-05-18",
                end_date="2026-05-24",
                parity_mode=False,
            )
            result = run_moneyline_paper_analysis(
                repository_root=REPOSITORY_ROOT,
                tsl_rows=bundle.tsl_history.rows,
                tsl_raw_sha256=bundle.tsl_history.selected_rows_sha256,
                schedule_rows=bundle.schedule_rows,
                target_boxscore_rows=bundle.target_boxscore_rows,
                pitcher_game_log_rows=bundle.pitcher_game_log_rows,
                source_manifest=bundle.source_manifest,
                offline_replay_verified=True,
                cohort_start_date=bundle.scope_start_date,
                cohort_end_date=bundle.scope_end_date,
                requested_game_ids=bundle.requested_game_ids,
            )
        self.assertEqual(len(bundle.requested_game_ids or ()), 16)
        self.assertEqual(result.summary["raw_game_count"], 16)
        self.assertEqual(result.summary["raw_game_count"], sum(result.summary["structural_status_counts"].values()))
        self.assertEqual(result.summary["edge_available_count"], 9)
        self.assertEqual(result.summary["feature_unavailable_count"], 7)
        self.assertIn("p31a_tsl_source", bundle.source_manifest)

    def test_date_scope_is_bounded_and_unambiguous(self) -> None:
        self.assertEqual(
            resolve_date_scope(date_value="2026-05-18"),
            ("2026-05-18", "2026-05-18", True),
        )
        with self.assertRaisesRegex(ValueError, "at most seven"):
            resolve_date_scope(start_date="2026-05-18", end_date="2026-05-25")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_date_scope(date_value="2026-05-18", start_date="2026-05-18", end_date="2026-05-18")


if __name__ == "__main__":
    unittest.main()
