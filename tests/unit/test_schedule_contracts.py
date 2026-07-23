"""Unit tests for immutable quarantined schedule contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.schedule import (
    DIAGNOSTIC_DATE_ONLY,
    PROVIDER_NAMESPACE,
    DateTeamCollisionGroup,
    LegacyDiagnosticScheduleCandidate,
    ProviderGameReference,
    ScheduleQuarantineReason,
    UNIVERSAL_QUARANTINE_REASONS,
)


def make_reference(
    provider_game_id: str = "823244",
) -> ProviderGameReference:
    return ProviderGameReference(
        provider_namespace=PROVIDER_NAMESPACE,
        provider_game_id=provider_game_id,
        source_game_id=f"mlb_2026_{provider_game_id}",
    )


def make_candidate(
    **overrides: object,
) -> LegacyDiagnosticScheduleCandidate:
    values: dict[str, object] = {
        "provider_reference": make_reference(),
        "season": 2026,
        "game_date": "2026-03-25",
        "source_home_team": "San Francisco Giants",
        "source_away_team": "New York Yankees",
        "legacy_collection_marker_utc": "2026-05-27T03:33:32.410129Z",
        "quarantine_reasons": UNIVERSAL_QUARANTINE_REASONS,
    }
    values.update(overrides)
    return LegacyDiagnosticScheduleCandidate(**values)


class ProviderGameReferenceTests(unittest.TestCase):
    def test_reference_is_immutable_and_losslessly_wraps_provider_id(
        self,
    ) -> None:
        reference = make_reference()

        self.assertEqual(reference.provider_game_id, "823244")
        self.assertEqual(reference.source_game_id, "mlb_2026_823244")
        with self.assertRaises(FrozenInstanceError):
            reference.provider_game_id = "1"

    def test_invalid_namespace_or_non_lossless_ids_are_rejected(self) -> None:
        invalid_cases = (
            {
                "provider_namespace": "OTHER",
                "provider_game_id": "823244",
                "source_game_id": "mlb_2026_823244",
            },
            {
                "provider_namespace": PROVIDER_NAMESPACE,
                "provider_game_id": "0823244",
                "source_game_id": "mlb_2026_0823244",
            },
            {
                "provider_namespace": PROVIDER_NAMESPACE,
                "provider_game_id": "823244",
                "source_game_id": "mlb_2026_823245",
            },
        )
        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    ProviderGameReference(**values)


class LegacyDiagnosticScheduleCandidateTests(unittest.TestCase):
    def test_candidate_is_immutable_and_has_six_ordered_reasons(self) -> None:
        candidate = make_candidate()

        self.assertEqual(candidate.diagnostic_status, DIAGNOSTIC_DATE_ONLY)
        self.assertEqual(
            candidate.quarantine_reasons,
            UNIVERSAL_QUARANTINE_REASONS,
        )
        self.assertEqual(len(candidate.quarantine_reasons), 6)
        with self.assertRaises(FrozenInstanceError):
            candidate.season = 2027

    def test_collision_reason_is_allowed_only_after_universal_reasons(
        self,
    ) -> None:
        collision_reasons = (
            *UNIVERSAL_QUARANTINE_REASONS,
            ScheduleQuarantineReason.DATE_TEAM_COLLISION,
        )

        self.assertEqual(
            make_candidate(
                quarantine_reasons=collision_reasons
            ).quarantine_reasons,
            collision_reasons,
        )
        with self.assertRaises(ValueError):
            make_candidate(
                quarantine_reasons=(
                    ScheduleQuarantineReason.DATE_TEAM_COLLISION,
                    *UNIVERSAL_QUARANTINE_REASONS,
                )
            )

    def test_candidate_exposes_no_trusted_schedule_or_identity_fields(
        self,
    ) -> None:
        fields = set(LegacyDiagnosticScheduleCandidate.__dataclass_fields__)

        self.assertTrue(
            {
                "source_trace",
                "collected_at_utc",
                "scheduled_start_utc",
                "provider_observed_at_utc",
                "schedule_status",
                "game_discriminator",
                "match_identity",
                "baseball_game",
                "pregame_eligible",
            }.isdisjoint(fields)
        )

    def test_invalid_candidate_values_are_rejected(self) -> None:
        invalid_cases = (
            {"season": 0},
            {"season": True},
            {"game_date": "2026-02-30"},
            {"source_home_team": ""},
            {"source_away_team": "San Francisco Giants"},
            {"legacy_collection_marker_utc": "2026-05-27T03:33:32"},
            {"legacy_collection_marker_utc": "not-a-time"},
            {"diagnostic_status": "TRUSTED"},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises((TypeError, ValueError)):
                    make_candidate(**overrides)


class DateTeamCollisionGroupTests(unittest.TestCase):
    def test_group_is_immutable_sorted_and_requires_multiple_rows(self) -> None:
        group = DateTeamCollisionGroup(
            game_date="2026-04-01",
            source_home_team="Team A",
            source_away_team="Team B",
            source_game_ids=(
                "mlb_2026_100",
                "mlb_2026_101",
            ),
        )

        with self.assertRaises(FrozenInstanceError):
            group.game_date = "2026-04-02"
        for source_ids in (
            ("mlb_2026_100",),
            ("mlb_2026_101", "mlb_2026_100"),
            ("mlb_2026_100", "mlb_2026_100"),
        ):
            with self.subTest(source_ids=source_ids):
                with self.assertRaises(ValueError):
                    DateTeamCollisionGroup(
                        game_date="2026-04-01",
                        source_home_team="Team A",
                        source_away_team="Team B",
                        source_game_ids=source_ids,
                    )


if __name__ == "__main__":
    unittest.main()
