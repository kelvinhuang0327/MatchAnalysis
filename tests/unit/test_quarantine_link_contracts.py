"""Unit tests for the immutable P83E/P84B quarantine referential link."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
import ast
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.prediction import LegacyPredictionCandidate
from match_analysis.baseball.domain.quarantine_link import (
    DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED,
    LegacyDiagnosticPredictionScheduleLink,
)
from match_analysis.baseball.domain.schedule import (
    PROVIDER_NAMESPACE,
    LegacyDiagnosticScheduleCandidate,
    ProviderGameReference,
    ScheduleQuarantineReason,
    UNIVERSAL_QUARANTINE_REASONS,
)
from match_analysis.application.use_cases.import_legacy_prediction_snapshot import (
    LegacyPredictionImportResult,
)
from match_analysis.application.use_cases.import_legacy_schedule_snapshot import (
    LegacyScheduleImportResult,
)
from match_analysis.application.use_cases.link_legacy_quarantine_snapshots import (
    LegacyQuarantineLinkResult,
    link_legacy_quarantine_snapshots,
)
from match_analysis.core.provenance import ArtifactProvenance


def make_prediction_candidate(
    game_id: str = "mlb_2026_100",
    **overrides: object,
) -> LegacyPredictionCandidate:
    values: dict[str, object] = {
        "source_game_id": game_id,
        "source_prediction_version": "p84b_diagnostic_baseline_v1",
        "predicted_side": "home",
        "sp_fip_delta": Decimal("-0.25"),
    }
    values.update(overrides)
    return LegacyPredictionCandidate(**values)


def make_reference(
    game_id: str = "mlb_2026_100",
) -> ProviderGameReference:
    provider_game_id = game_id.rsplit("_", 1)[-1]
    return ProviderGameReference(
        provider_namespace=PROVIDER_NAMESPACE,
        provider_game_id=provider_game_id,
        source_game_id=game_id,
    )


def make_schedule_candidate(
    game_id: str = "mlb_2026_100",
    *,
    quarantine_reasons: tuple[ScheduleQuarantineReason, ...] = UNIVERSAL_QUARANTINE_REASONS,
    **overrides: object,
) -> LegacyDiagnosticScheduleCandidate:
    values: dict[str, object] = {
        "provider_reference": make_reference(game_id),
        "season": 2026,
        "game_date": "2026-04-01",
        "source_home_team": "Team A",
        "source_away_team": "Team B",
        "legacy_collection_marker_utc": "2026-04-01T00:00:00Z",
        "quarantine_reasons": quarantine_reasons,
    }
    values.update(overrides)
    return LegacyDiagnosticScheduleCandidate(**values)


def make_link(
    game_id: str = "mlb_2026_100",
    *,
    prediction_candidate: LegacyPredictionCandidate | None = None,
    schedule_candidate: LegacyDiagnosticScheduleCandidate | None = None,
    provider_reference: ProviderGameReference | None = None,
    source_game_id: str | None = None,
) -> LegacyDiagnosticPredictionScheduleLink:
    schedule_candidate = schedule_candidate or make_schedule_candidate(game_id)
    return LegacyDiagnosticPredictionScheduleLink(
        source_game_id=source_game_id if source_game_id is not None else game_id,
        provider_reference=provider_reference or schedule_candidate.provider_reference,
        prediction_candidate=prediction_candidate or make_prediction_candidate(game_id),
        schedule_candidate=schedule_candidate,
    )


def make_prediction_result(
    candidates: tuple[LegacyPredictionCandidate, ...],
) -> LegacyPredictionImportResult:
    return LegacyPredictionImportResult(
        provenance=ArtifactProvenance(
            schema_version="p83e_snapshot_quarantine_v1",
            source_repository="Betting-pool",
            source_commit="0" * 40,
            producer_id="unit-test",
            producer_version="1",
            input_fingerprint="a" * 64,
            content_fingerprint="b" * 64,
        ),
        row_count=len(candidates),
        unique_id_count=len(candidates),
        validated_null_outcome_placeholder_fields=(
            "result_home_score",
            "result_away_score",
            "actual_winner",
            "is_correct",
        ),
        validated_null_outcome_placeholder_count=4,
        rows_with_observed_outcomes=0,
        promoted_prediction_count=0,
        candidates=candidates,
        semantic_fingerprint="c" * 64,
        limitations=("unit-test fixture",),
        quarantine_counts=(),
    )


def make_schedule_result(
    candidates: tuple[LegacyDiagnosticScheduleCandidate, ...],
) -> LegacyScheduleImportResult:
    references = tuple(c.provider_reference for c in candidates)
    return LegacyScheduleImportResult(
        artifact_sha256="d" * 64,
        row_count=len(candidates),
        unique_provider_reference_count=len(set(references)),
        provider_game_references=references,
        candidates=candidates,
        collision_groups=(),
        collision_affected_row_count=0,
        semantic_fingerprint="e" * 64,
        limitations=("unit-test fixture",),
        quarantine_counts=(),
        match_identity_count=0,
        trusted_schedule_observation_count=0,
        baseball_game_count=0,
        pregame_eligible_context_count=0,
    )


class LegacyDiagnosticPredictionScheduleLinkTests(unittest.TestCase):
    def test_link_is_immutable(self) -> None:
        link = make_link()

        with self.assertRaises(FrozenInstanceError):
            link.source_game_id = "mlb_2026_999"
        with self.assertRaises(FrozenInstanceError):
            link.prediction_candidate = make_prediction_candidate("mlb_2026_999")

    def test_diagnostic_status_is_exact(self) -> None:
        link = make_link()

        self.assertEqual(
            link.diagnostic_status,
            DIAGNOSTIC_LINKED_UNTIMED_UNRESOLVED,
        )
        with self.assertRaises(ValueError):
            LegacyDiagnosticPredictionScheduleLink(
                source_game_id="mlb_2026_100",
                provider_reference=make_reference("mlb_2026_100"),
                prediction_candidate=make_prediction_candidate("mlb_2026_100"),
                schedule_candidate=make_schedule_candidate("mlb_2026_100"),
                diagnostic_status="DIAGNOSTIC_LINKED_RESOLVED",
            )

    def test_source_id_agrees_with_both_candidates(self) -> None:
        link = make_link("mlb_2026_100")

        self.assertEqual(link.source_game_id, "mlb_2026_100")
        self.assertEqual(link.prediction_candidate.source_game_id, "mlb_2026_100")
        self.assertEqual(
            link.schedule_candidate.provider_reference.source_game_id,
            "mlb_2026_100",
        )

    def test_mismatched_source_ids_are_rejected(self) -> None:
        prediction_candidate = make_prediction_candidate("mlb_2026_100")

        with self.assertRaisesRegex(ValueError, "prediction candidate"):
            LegacyDiagnosticPredictionScheduleLink(
                source_game_id="mlb_2026_200",
                provider_reference=make_reference("mlb_2026_200"),
                prediction_candidate=prediction_candidate,
                schedule_candidate=make_schedule_candidate("mlb_2026_200"),
            )
        with self.assertRaisesRegex(ValueError, "schedule candidate"):
            LegacyDiagnosticPredictionScheduleLink(
                source_game_id="mlb_2026_100",
                provider_reference=make_reference("mlb_2026_100"),
                prediction_candidate=make_prediction_candidate("mlb_2026_100"),
                schedule_candidate=make_schedule_candidate("mlb_2026_300"),
            )

    def test_provider_reference_must_agree_with_schedule_candidate(self) -> None:
        schedule_candidate = make_schedule_candidate("mlb_2026_100")

        with self.assertRaisesRegex(ValueError, "provider_reference"):
            LegacyDiagnosticPredictionScheduleLink(
                source_game_id="mlb_2026_100",
                provider_reference=make_reference("mlb_2026_200"),
                prediction_candidate=make_prediction_candidate("mlb_2026_100"),
                schedule_candidate=schedule_candidate,
            )

    def test_decoded_provider_id_agrees_with_source_game_id(self) -> None:
        link = make_link("mlb_2026_88823")

        self.assertEqual(link.provider_reference.provider_game_id, "88823")
        self.assertEqual(
            link.source_game_id.rsplit("_", 1)[-1],
            link.provider_reference.provider_game_id,
        )

    def test_prediction_and_schedule_reason_collections_remain_separate(
        self,
    ) -> None:
        prediction_candidate = make_prediction_candidate("mlb_2026_100")
        schedule_candidate = make_schedule_candidate(
            "mlb_2026_100",
            quarantine_reasons=(
                *UNIVERSAL_QUARANTINE_REASONS,
                ScheduleQuarantineReason.DATE_TEAM_COLLISION,
            ),
        )
        link = make_link(
            "mlb_2026_100",
            prediction_candidate=prediction_candidate,
            schedule_candidate=schedule_candidate,
        )

        self.assertEqual(
            link.prediction_quarantine_reasons,
            (prediction_candidate.quarantine_reason,),
        )
        self.assertEqual(
            link.schedule_quarantine_reasons,
            schedule_candidate.quarantine_reasons,
        )
        self.assertNotEqual(
            set(link.prediction_quarantine_reasons),
            set(link.schedule_quarantine_reasons),
        )

    def test_collision_flag_is_derived_correctly(self) -> None:
        collided = make_link(
            "mlb_2026_100",
            schedule_candidate=make_schedule_candidate(
                "mlb_2026_100",
                quarantine_reasons=(
                    *UNIVERSAL_QUARANTINE_REASONS,
                    ScheduleQuarantineReason.DATE_TEAM_COLLISION,
                ),
            ),
        )
        not_collided = make_link("mlb_2026_101")

        self.assertTrue(collided.schedule_collision_affected)
        self.assertFalse(not_collided.schedule_collision_affected)


class LinkLegacyQuarantineSnapshotsTests(unittest.TestCase):
    def test_all_matched_predictions_are_linked(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_101"),
            )
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_101"),
                make_schedule_candidate("mlb_2026_102"),
            )
        )

        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(result.linked_count, 2)
        self.assertEqual(
            tuple(link.source_game_id for link in result.links),
            ("mlb_2026_100", "mlb_2026_101"),
        )
        self.assertEqual(result.prediction_missing_schedule_ids, ())
        self.assertEqual(result.schedule_only_source_ids, ("mlb_2026_102",))

    def test_prediction_missing_schedule_fails_closed(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_999"),
            )
        )
        schedule_result = make_schedule_result(
            (make_schedule_candidate("mlb_2026_100"),)
        )

        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(result.prediction_missing_schedule_ids, ("mlb_2026_999",))
        self.assertEqual(
            tuple(link.source_game_id for link in result.links),
            ("mlb_2026_100",),
        )
        self.assertEqual(result.linked_count, 1)
        self.assertTrue(
            all(
                link.source_game_id != "mlb_2026_999"
                for link in result.links
            )
        )

    def test_schedule_only_ids_are_retained_in_result_summary(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_200"),
                make_schedule_candidate("mlb_2026_300"),
            )
        )

        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(
            result.schedule_only_source_ids,
            ("mlb_2026_200", "mlb_2026_300"),
        )

    def test_duplicate_prediction_ids_are_rejected(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_100"),
            )
        )
        schedule_result = make_schedule_result(
            (make_schedule_candidate("mlb_2026_100"),)
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            link_legacy_quarantine_snapshots(prediction_result, schedule_result)

    def test_duplicate_schedule_ids_are_rejected(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_100"),
            )
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            link_legacy_quarantine_snapshots(prediction_result, schedule_result)

    def test_output_ordering_is_deterministic(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_300"),
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_200"),
            )
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_300"),
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_200"),
            )
        )

        first = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        second = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(
            tuple(link.source_game_id for link in first.links),
            ("mlb_2026_100", "mlb_2026_200", "mlb_2026_300"),
        )
        self.assertEqual(first.links, second.links)
        self.assertEqual(
            first.joint_semantic_fingerprint,
            second.joint_semantic_fingerprint,
        )

    def test_all_five_promotion_counts_remain_zero(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result(
            (make_schedule_candidate("mlb_2026_100"),)
        )

        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(
            (
                result.match_identity_count,
                result.trusted_schedule_observation_count,
                result.baseball_game_count,
                result.canonical_prediction_count,
                result.pregame_eligible_context_count,
            ),
            (0, 0, 0, 0, 0),
        )

    def test_result_is_immutable(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result(
            (make_schedule_candidate("mlb_2026_100"),)
        )
        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        with self.assertRaises(FrozenInstanceError):
            result.linked_count = 99

    def test_joint_fingerprint_is_deterministic_and_reproducible_independently(
        self,
    ) -> None:
        import json
        from hashlib import sha256

        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_999"),
            )
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_200"),
            )
        )

        result = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        independent_payload = {
            "schema_version": "p83e_p84b_quarantine_link_v1",
            "p83e_raw_sha256": prediction_result.provenance.input_fingerprint,
            "p83e_semantic_fingerprint": prediction_result.semantic_fingerprint,
            "p84b_raw_sha256": schedule_result.artifact_sha256,
            "p84b_semantic_fingerprint": schedule_result.semantic_fingerprint,
            "linked_source_ids": ["mlb_2026_100"],
            "prediction_missing_schedule_ids": ["mlb_2026_999"],
            "schedule_only_source_ids": ["mlb_2026_200"],
            "collision_affected_linked_source_ids": [],
            "linked_count": 1,
            "prediction_missing_schedule_count": 1,
            "schedule_only_count": 1,
            "collision_affected_linked_count": 0,
            "match_identity_count": 0,
            "trusted_schedule_observation_count": 0,
            "baseball_game_count": 0,
            "canonical_prediction_count": 0,
            "pregame_eligible_context_count": 0,
        }
        independent_encoded = (
            json.dumps(
                independent_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        independent_fingerprint = sha256(independent_encoded).hexdigest()

        self.assertEqual(result.joint_semantic_fingerprint, independent_fingerprint)

        second = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        self.assertEqual(
            result.joint_semantic_fingerprint,
            second.joint_semantic_fingerprint,
        )

    def test_use_case_has_no_file_path_or_time_dependency(self) -> None:
        module_path = (
            REPOSITORY_ROOT
            / "src"
            / "match_analysis"
            / "application"
            / "use_cases"
            / "link_legacy_quarantine_snapshots.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[0])

        self.assertTrue(
            {"pathlib", "os", "io", "datetime", "time"}.isdisjoint(imported_names)
        )

        source_text = module_path.read_text(encoding="utf-8")
        for forbidden_token in ("open(", "Path(", "datetime.now(", "time.time("):
            self.assertNotIn(forbidden_token, source_text)


if __name__ == "__main__":
    unittest.main()
