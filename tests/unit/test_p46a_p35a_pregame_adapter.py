"""Focused unit and integration tests for P46A P35A-to-Normalized Pregame Adapter."""

from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.p44a_historical_source_adapter import (
    protected_authority_hashes,
)
from match_analysis.application.use_cases.p44a_normalized_workflow_input import (
    FORBIDDEN_PREGAME_FIELD_NAMES,
    NormalizedPregameInput,
    load_normalized_pregame_input,
    parse_normalized_pregame_payload,
    write_normalized_pregame_input,
)
from match_analysis.application.use_cases.p45a_paper_run_ledger import (
    CLASSIFICATION_HISTORICAL_REHEARSAL,
    CLASSIFICATION_PROSPECTIVE_FORWARD_PAPER,
    STATE_FROZEN,
    create_p45a_paper_run,
    get_p45a_forward_summary,
)
from match_analysis.application.use_cases.p46a_p35a_pregame_adapter import (
    P46A_ADAPTER_SOURCE_IDENTITY,
    P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
    P46A_EXCLUSION_SCHEMA,
    P46A_PRODUCTIVE_STATUS,
    P46A_TASK_ID,
    adapt_p35a_pregame,
    adapt_p35a_pregame_file,
)
from match_analysis.baseball.domain.paper_moneyline_bet_pass import (
    DECISION_BET,
    DECISION_PASS,
)
from match_analysis.interfaces.cli.adapt_p35a_pregame_input import (
    main as cli_main,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PATHS = (
    REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/comparisons.jsonl",
    REPOSITORY_ROOT / "report/p37a_rolling_walk_forward_oos/summary.json",
    REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/comparisons.jsonl",
    REPOSITORY_ROOT / "report/p38a_rolling_probability_calibration/summary.json",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_join.jsonl",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/market_snapshots.jsonl",
    REPOSITORY_ROOT / "report/p39a_tsl_moneyline_market_join/summary.json",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/decisions.jsonl",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/settlements.jsonl",
    REPOSITORY_ROOT / "report/p40a_moneyline_paper_bet_pass/summary.json",
    REPOSITORY_ROOT / "report/p41a_walk_forward_ev_margin_policy/summary.json",
    REPOSITORY_ROOT / "report/p41a_walk_forward_ev_margin_policy/policy_evaluations.jsonl",
    REPOSITORY_ROOT / "report/p42a_offline_end_to_end_paper_workflow/summary.json",
    REPOSITORY_ROOT / "report/p42a_offline_end_to_end_paper_workflow/workflow_ledger.jsonl",
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/pregame_decisions.jsonl",
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/pregame_summary.json",
    REPOSITORY_ROOT / "report/p43a_two_phase_paper_workflow/postgame_summary.json",
)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def _sample_p35a_bundle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return minimal truthful P35A contract rehearsal fixture."""

    analysis_rows = [
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "824192",
            "scheduled_start": "2026-05-17T18:10:00Z",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "5de824e18c72b10c7d77ea843c4ee3a7787fbf2e789c86feac209206b16b39c5",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5861848091915969598190982782",
            "market_price_id": "p28ab:6b6f72c2b12b80b8c84b434105976a0f57d5c6f13f26e87bf7f935a856779bf5",
            "price_observed_at": "2026-05-17T02:04:35.317592Z",
            "home_decimal_odds": "1.85",
            "away_decimal_odds": "1.65",
            "home_no_vig_probability": "0.4714285714285714285714285712",
            "away_no_vig_probability": "0.5285714285714285714285714284",
            "home_edge": "0.1147562377630255312476697070",
            "away_edge": "-0.1147562377630255312476697066",
            "controlled_unavailable_reason": None,
        },
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "824114",
            "scheduled_start": "2026-05-18T23:10:00Z",
            "home_team": "Kansas City Royals",
            "away_team": "Boston Red Sox",
            "structural_status": "EDGE_AVAILABLE",
            "status": "EDGE_AVAILABLE",
            "prediction_id": "55e5ec556f198337f0f9a2ad090b052f3910c70b7c57d8fd829c0b3a94caa337",
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": "0.5303627085439037814014483218",
            "market_price_id": "p28ab:fff29ab049b29e9c47ca097af08b1df1048a85933d4719606a54d4b847da68c2",
            "price_observed_at": "2026-05-18T01:21:04.535534Z",
            "home_decimal_odds": "1.82",
            "away_decimal_odds": "1.68",
            "home_no_vig_probability": "0.4799999999999999999999999999",
            "away_no_vig_probability": "0.5199999999999999999999999998",
            "home_edge": "0.0503627085439037814014483219",
            "away_edge": "-0.0503627085439037814014483216",
            "controlled_unavailable_reason": None,
        },
        {
            "schema_version": "p30a.moneyline_paper_analysis.v1",
            "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
            "game_id": "822738",
            "scheduled_start": "2026-05-17T17:35:00Z",
            "home_team": "Washington Nationals",
            "away_team": "Baltimore Orioles",
            "structural_status": "FEATURE_UNAVAILABLE",
            "status": "FEATURE_UNAVAILABLE",
            "prediction_id": None,
            "model_id": "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630",
            "model_fingerprint": "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e",
            "model_home_probability": None,
            "market_price_id": "p28ab:28ab9f198d81aad8d36b7d4f38b3a175962d6ec9f4af50904475accd2e298bb5",
            "price_observed_at": "2026-05-17T02:04:35.317592Z",
            "home_decimal_odds": "1.90",
            "away_decimal_odds": "1.60",
            "home_no_vig_probability": "0.4571428571428571428571428570",
            "away_no_vig_probability": "0.5428571428571428571428571426",
            "home_edge": None,
            "away_edge": None,
            "controlled_unavailable_reason": "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY",
        },
    ]

    schedule_rows = [
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "824192",
            "game_pk": 824192,
            "game_number": 1,
            "official_date": "2026-05-17",
            "scheduled_start_utc": "2026-05-17T18:10:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 117, "abbreviation": "HOU", "name": "Houston Astros"},
            "away_team": {"id": 140, "abbreviation": "TEX", "name": "Texas Rangers"},
            "home_score": None,
            "away_score": None,
        },
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "824114",
            "game_pk": 824114,
            "game_number": 1,
            "official_date": "2026-05-18",
            "scheduled_start_utc": "2026-05-18T23:10:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 118, "abbreviation": "KC", "name": "Kansas City Royals"},
            "away_team": {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
            "home_score": None,
            "away_score": None,
        },
        {
            "schema_version": "p23f2.mlb_official_normalized.v1",
            "provider_game_id": "822738",
            "game_pk": 822738,
            "game_number": 1,
            "official_date": "2026-05-17",
            "scheduled_start_utc": "2026-05-17T17:35:00Z",
            "status": "Scheduled",
            "final": False,
            "home_team": {"id": 120, "abbreviation": "WSH", "name": "Washington Nationals"},
            "away_team": {"id": 110, "abbreviation": "BAL", "name": "Baltimore Orioles"},
            "home_score": None,
            "away_score": None,
        },
    ]

    source_manifest = {
        "schema_version": "p30a.source_manifest.v1",
        "tsl_authority": {
            "authority_label": "TSL_BLOB3RD",
            "legacy_repository": "Betting-pool",
        },
    }

    run_manifest = {
        "run_id": "779baaf06ec68624167f51979a634fd9e6a4089cd347df6cb859d997e2a81e33",
        "target_date": "2026-05-17",
    }

    return analysis_rows, schedule_rows, source_manifest, run_manifest


class P46AP35APregameAdapterTests(unittest.TestCase):
    def test_positive_path_adapts_p35a_into_normalized_pregame(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        adapted = adapt_p35a_pregame(
            analysis_rows,
            schedule_input=schedule_rows,
            source_manifest_input=source_manifest,
            run_manifest_input=run_manifest,
            source_identity=P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
        )

        self.assertEqual(adapted.source_identity, P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY)
        self.assertEqual(len(adapted.prediction_rows), 2)
        self.assertEqual(len(adapted.market_rows), 2)
        self.assertEqual(len(adapted.exclusion_rows), 1)

        # Check prediction row mapping
        first_pred = adapted.prediction_rows[0]
        self.assertEqual(first_pred.provider_game_id, "824114")
        self.assertEqual(first_pred.champion_home_probability, Decimal("0.5303627085439037814014483218"))
        self.assertEqual(first_pred.champion_model_id, "p22b_moneyline_logistic_challenger_v1_05f9b31c608e1630")
        self.assertEqual(first_pred.champion_model_fingerprint, "2e260f323e39880335f8d849ee8b83586b91e7bd9d4fa44127f530d6a931bf2e")

        # Check market row mapping
        first_mkt = adapted.market_rows[0]
        self.assertEqual(first_mkt.provider_game_id, "824114")
        self.assertEqual(first_mkt.home_decimal_odds, Decimal("1.82"))
        self.assertEqual(first_mkt.away_decimal_odds, Decimal("1.68"))
        self.assertEqual(first_mkt.home_team, "Kansas City Royals")
        self.assertEqual(first_mkt.away_team, "Boston Red Sox")
        self.assertEqual(first_mkt.market_observed_at_utc, "2026-05-18T01:21:04.535534Z")

        # Check exclusion row mapping
        exclusion = adapted.exclusion_rows[0]
        self.assertEqual(exclusion["provider_game_id"], "822738")
        self.assertEqual(exclusion["exclusion_reason"], "INSUFFICIENT_SAME_SEASON_STARTER_HISTORY")
        self.assertEqual(exclusion["market_snapshot_status"], "FEATURE_UNAVAILABLE")
        self.assertFalse(exclusion["became_bet"])

        # Check that payload roundtrip is valid
        payload = adapted.to_payload()
        reparsed = parse_normalized_pregame_payload(payload)
        self.assertEqual(len(reparsed.prediction_rows), 2)
        self.assertEqual(len(reparsed.market_rows), 2)
        self.assertEqual(len(reparsed.exclusion_rows), 1)

    def test_bundle_directory_and_file_adapter(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "p35a_bundle"
            bundle_dir.mkdir()

            (bundle_dir / "analysis.jsonl").write_text(
                "\n".join(json.dumps(r) for r in analysis_rows) + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "mlb_source_snapshot.jsonl").write_text(
                "\n".join(json.dumps(r) for r in schedule_rows) + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "source_manifest.json").write_text(
                json.dumps(source_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "run_manifest.json").write_text(
                json.dumps(run_manifest, indent=2) + "\n",
                encoding="utf-8",
            )

            # Adapt directly from bundle directory
            adapted_from_dir = adapt_p35a_pregame(bundle_dir)
            self.assertEqual(len(adapted_from_dir.prediction_rows), 2)
            self.assertEqual(len(adapted_from_dir.exclusion_rows), 1)

            # Adapt using adapt_p35a_pregame_file
            out_file = Path(raw_dir) / "output_pregame.json"
            written_path = adapt_p35a_pregame_file(bundle_dir, out_file)
            self.assertTrue(written_path.is_file())

            loaded = load_normalized_pregame_input(written_path)
            self.assertEqual(len(loaded.prediction_rows), 2)
            self.assertEqual(len(loaded.exclusion_rows), 1)

    def test_p45_create_run_compatibility_with_forward_sample_count_zero(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        adapted = adapt_p35a_pregame(
            analysis_rows,
            schedule_input=schedule_rows,
            source_manifest_input=source_manifest,
            run_manifest_input=run_manifest,
            source_identity=P46A_CONTRACT_REHEARSAL_SOURCE_IDENTITY,
        )

        with tempfile.TemporaryDirectory() as raw_dir:
            run_root = Path(raw_dir) / "runs"
            result = create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=adapted,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
                created_at_utc="2026-05-17T12:00:00Z",
            )

            self.assertEqual(result.status, "CREATED")
            self.assertEqual(result.manifest["run_classification"], CLASSIFICATION_HISTORICAL_REHEARSAL)
            self.assertEqual(result.manifest["lifecycle_state"], STATE_FROZEN)
            self.assertEqual(result.manifest["target_universe_count"], 3)
            self.assertEqual(result.manifest["eligible_decision_count"], 2)
            self.assertEqual(result.manifest["exclusion_count"], 1)

            # Cumulative forward summary MUST have forward_sample_count == 0
            forward_summary = get_p45a_forward_summary(REPOSITORY_ROOT)
            self.assertEqual(forward_summary["forward_sample_count"], 0)

    def test_rejection_of_outcome_fields(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        for forbidden_name in ("actual_winner", "home_score", "away_score", "final_score", "settlement"):
            tainted_analysis = deepcopy(analysis_rows)
            tainted_analysis[0][forbidden_name] = "HOME"
            with self.assertRaisesRegex(ValueError, "P44A_PREGAME_OUTCOME_FIELDS_REJECTED"):
                adapt_p35a_pregame(
                    tainted_analysis,
                    schedule_input=schedule_rows,
                    source_manifest_input=source_manifest,
                    run_manifest_input=run_manifest,
                )

    def test_rejection_of_live_label_for_contract_rehearsal(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        with self.assertRaisesRegex(ValueError, "must not be labeled live"):
            adapt_p35a_pregame(
                analysis_rows,
                schedule_input=schedule_rows,
                source_manifest_input=source_manifest,
                run_manifest_input=run_manifest,
                source_identity="LIVE",
            )

    def test_temporal_guard_rejects_postgame_market_observation(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        # Make market observation equal to scheduled start (not strictly pregame)
        late_analysis = deepcopy(analysis_rows)
        late_analysis[0]["price_observed_at"] = late_analysis[0]["scheduled_start"]

        with self.assertRaisesRegex(ValueError, "not strictly pregame"):
            adapt_p35a_pregame(
                late_analysis,
                schedule_input=schedule_rows,
                source_manifest_input=source_manifest,
                run_manifest_input=run_manifest,
            )

    def test_failure_on_missing_or_blank_game_identity(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        # Missing game_id
        missing_id = deepcopy(analysis_rows)
        del missing_id[0]["game_id"]
        with self.assertRaisesRegex(ValueError, "missing game identity"):
            adapt_p35a_pregame(missing_id, schedule_input=schedule_rows)

        # Blank game_id
        blank_id = deepcopy(analysis_rows)
        blank_id[0]["game_id"] = "   "
        with self.assertRaisesRegex(ValueError, "blank game_id"):
            adapt_p35a_pregame(blank_id, schedule_input=schedule_rows)

    def test_failure_on_duplicate_game_identity(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        duplicate_rows = deepcopy(analysis_rows)
        duplicate_rows.append(deepcopy(analysis_rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate game_id"):
            adapt_p35a_pregame(duplicate_rows, schedule_input=schedule_rows)

    def test_failure_on_missing_scheduled_start(self) -> None:
        analysis_rows, _, _, _ = _sample_p35a_bundle()
        missing_start = deepcopy(analysis_rows)
        del missing_start[0]["scheduled_start"]
        with self.assertRaisesRegex(ValueError, "missing scheduled start"):
            adapt_p35a_pregame(missing_start, schedule_input=())

    def test_failure_on_invalid_probabilities(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()

        for invalid_prob in ("0.0", "1.0", "-0.1", "1.5", "invalid"):
            bad_prob = deepcopy(analysis_rows)
            bad_prob[0]["model_home_probability"] = invalid_prob
            with self.assertRaises(ValueError):
                adapt_p35a_pregame(bad_prob, schedule_input=schedule_rows)

    def test_failure_on_malformed_or_nonpositive_odds(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()

        for invalid_odds in ("1.0", "0.95", "-1.50", "0", "abc"):
            bad_home = deepcopy(analysis_rows)
            bad_home[0]["home_decimal_odds"] = invalid_odds
            with self.assertRaises(ValueError):
                adapt_p35a_pregame(bad_home, schedule_input=schedule_rows)

            bad_away = deepcopy(analysis_rows)
            bad_away[0]["away_decimal_odds"] = invalid_odds
            with self.assertRaises(ValueError):
                adapt_p35a_pregame(bad_away, schedule_input=schedule_rows)

    def test_failure_on_missing_prices_on_productive_rows(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()

        # Missing home price with status EDGE_AVAILABLE
        no_home = deepcopy(analysis_rows)
        no_home[0]["home_decimal_odds"] = None
        adapted = adapt_p35a_pregame(no_home, schedule_input=schedule_rows)
        # Treated as non-productive exclusion
        self.assertEqual(len(adapted.prediction_rows), 1)
        self.assertEqual(len(adapted.exclusion_rows), 2)

        # Missing away price with status EDGE_AVAILABLE
        no_away = deepcopy(analysis_rows)
        no_away[0]["away_decimal_odds"] = None
        adapted_away = adapt_p35a_pregame(no_away, schedule_input=schedule_rows)
        self.assertEqual(len(adapted_away.prediction_rows), 1)
        self.assertEqual(len(adapted_away.exclusion_rows), 2)

    def test_failure_on_missing_probability_on_productive_rows(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()
        no_prob = deepcopy(analysis_rows)
        no_prob[0]["model_home_probability"] = None
        adapted = adapt_p35a_pregame(no_prob, schedule_input=schedule_rows)
        self.assertEqual(len(adapted.prediction_rows), 1)
        self.assertEqual(len(adapted.exclusion_rows), 2)

    def test_failure_on_missing_market_timestamp(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()
        no_time = deepcopy(analysis_rows)
        no_time[0]["price_observed_at"] = None
        adapted = adapt_p35a_pregame(no_time, schedule_input=schedule_rows)
        self.assertEqual(len(adapted.prediction_rows), 1)
        self.assertEqual(len(adapted.exclusion_rows), 2)

    def test_failure_on_duplicate_prediction_identity(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()
        dup_pred = deepcopy(analysis_rows)
        dup_pred[1]["prediction_id"] = dup_pred[0]["prediction_id"]
        with self.assertRaisesRegex(ValueError, "duplicate prediction_id"):
            adapt_p35a_pregame(dup_pred, schedule_input=schedule_rows)

    def test_failure_on_duplicate_market_snapshot_identity(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()
        dup_mkt = deepcopy(analysis_rows)
        dup_mkt[1]["market_price_id"] = dup_mkt[0]["market_price_id"]
        with self.assertRaisesRegex(ValueError, "duplicate market_snapshot_id"):
            adapt_p35a_pregame(dup_mkt, schedule_input=schedule_rows)

    def test_failure_on_schedule_start_mismatch(self) -> None:
        analysis_rows, schedule_rows, _, _ = _sample_p35a_bundle()
        mismatch_sched = deepcopy(schedule_rows)
        mismatch_sched[0]["scheduled_start_utc"] = "2026-05-17T20:00:00Z"
        with self.assertRaisesRegex(ValueError, "scheduled start mismatch"):
            adapt_p35a_pregame(analysis_rows, schedule_input=mismatch_sched)

    def test_deterministic_output_and_shuffled_rows_invariance(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        first = adapt_p35a_pregame(
            analysis_rows,
            schedule_input=schedule_rows,
            source_manifest_input=source_manifest,
            run_manifest_input=run_manifest,
        )

        shuffled_analysis = list(reversed(analysis_rows))
        shuffled_schedule = list(reversed(schedule_rows))

        second = adapt_p35a_pregame(
            shuffled_analysis,
            schedule_input=shuffled_schedule,
            source_manifest_input=source_manifest,
            run_manifest_input=run_manifest,
        )

        self.assertEqual(first.to_payload(), second.to_payload())

    def test_cli_execution_end_to_end(self) -> None:
        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()

        with tempfile.TemporaryDirectory() as raw_dir:
            bundle_dir = Path(raw_dir) / "bundle"
            bundle_dir.mkdir()

            (bundle_dir / "analysis.jsonl").write_text(
                "\n".join(json.dumps(r) for r in analysis_rows) + "\n",
                encoding="utf-8",
            )
            (bundle_dir / "mlb_source_snapshot.jsonl").write_text(
                "\n".join(json.dumps(r) for r in schedule_rows) + "\n",
                encoding="utf-8",
            )

            output_file = Path(raw_dir) / "normalized_pregame.json"

            exit_code = cli_main(
                [
                    "--p35a-input",
                    str(bundle_dir),
                    "--output",
                    str(output_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_file.is_file())

            loaded = load_normalized_pregame_input(output_file)
            self.assertEqual(len(loaded.prediction_rows), 2)
            self.assertEqual(len(loaded.exclusion_rows), 1)

    def test_no_network_imports(self) -> None:
        adapter_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/application/use_cases/p46a_p35a_pregame_adapter.py"
        )
        cli_path = (
            REPOSITORY_ROOT
            / "src/match_analysis/interfaces/cli/adapt_p35a_pregame_input.py"
        )

        for path in (adapter_path, cli_path):
            imported = _imported_roots(path)
            for forbidden in ("urllib", "requests", "http", "socket", "ssl", "aiohttp"):
                self.assertNotIn(forbidden, imported)
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("urlopen", source)
            self.assertNotIn("verify=False", source)

    def test_protected_authorities_remain_invariant(self) -> None:
        before = {path: path.read_bytes() for path in PROTECTED_PATHS}
        hashed = protected_authority_hashes(REPOSITORY_ROOT)

        analysis_rows, schedule_rows, source_manifest, run_manifest = _sample_p35a_bundle()
        adapted = adapt_p35a_pregame(
            analysis_rows,
            schedule_input=schedule_rows,
            source_manifest_input=source_manifest,
            run_manifest_input=run_manifest,
        )

        with tempfile.TemporaryDirectory() as raw_dir:
            out_file = Path(raw_dir) / "output.json"
            write_normalized_pregame_input(out_file, adapted)
            run_root = Path(raw_dir) / "runs"
            create_p45a_paper_run(
                REPOSITORY_ROOT,
                pregame_input=adapted,
                run_classification=CLASSIFICATION_HISTORICAL_REHEARSAL,
                run_root=run_root,
            )

        after = {path: path.read_bytes() for path in PROTECTED_PATHS}
        self.assertEqual(before, after)
        self.assertEqual(hashed, protected_authority_hashes(REPOSITORY_ROOT))


if __name__ == "__main__":
    unittest.main()
