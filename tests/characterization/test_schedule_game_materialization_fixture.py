"""Characterize canonical game materialization across schedule cutoffs."""

from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.construct_match_identities import (
    construct_match_identities,
)
from match_analysis.application.use_cases.materialize_schedule_baseball_games import (
    materialize_schedule_baseball_games,
)
from match_analysis.baseball.domain.match_identity_authority import (
    build_match_identity_authority_catalog,
)
from tests.characterization.test_participant_identity_resolution_fixture import (
    AFTER_POSTPONED_AS_OF,
    BEFORE_ALL_AS_OF,
    EXACT_BOUNDARY_AS_OF,
    MID_CUTOFF_AS_OF,
    load_fixture_observations,
    resolve_at,
)
from tests.unit.test_construct_match_identities import (
    fixture_authority_entries,
    fixture_catalog,
)


BEFORE_ALL_MATERIALIZATION_FINGERPRINT = (
    "fe6079d9a947115c352678e27550000bc053c00fa8e2510bd7b86a37f6d4e23f"
)
MID_CUTOFF_MATERIALIZATION_FINGERPRINT = (
    "473d9e0f9cdb984428d26f067e680145f12ca558817ec291a532e9f8beb9bd6a"
)
AFTER_POSTPONED_MATERIALIZATION_FINGERPRINT = (
    "4cc83e9019ae050f074893f2107730edd0dbb6bcf40fbf38962e32cad613090b"
)
EXACT_BOUNDARY_MATERIALIZATION_FINGERPRINT = (
    "895bdc43c62d970998f4f8b14d8fe8ba2e79e50896db1af470b4327ac0f52f09"
)


def materialize_at(observations, cutoff, *, mappings=None, catalog=None):
    _, resolution = resolve_at(observations, cutoff, mappings)
    construction = construct_match_identities(
        resolution,
        fixture_catalog() if catalog is None else catalog,
    )
    materialization = materialize_schedule_baseball_games(
        construction,
        resolution,
    )
    return resolution, construction, materialization


class ScheduleGameMaterializationFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.by_key, cls.observations = load_fixture_observations()

    def test_four_cutoffs_match_pre_edit_references(self) -> None:
        cases = (
            (
                BEFORE_ALL_AS_OF,
                (0, 0, 2, 0),
                BEFORE_ALL_MATERIALIZATION_FINGERPRINT,
            ),
            (
                MID_CUTOFF_AS_OF,
                (2, 0, 0, 0),
                MID_CUTOFF_MATERIALIZATION_FINGERPRINT,
            ),
            (
                AFTER_POSTPONED_AS_OF,
                (2, 0, 0, 0),
                AFTER_POSTPONED_MATERIALIZATION_FINGERPRINT,
            ),
            (
                EXACT_BOUNDARY_AS_OF,
                (1, 0, 1, 0),
                EXACT_BOUNDARY_MATERIALIZATION_FINGERPRINT,
            ),
        )

        for cutoff, expected_counts, expected_fingerprint in cases:
            with self.subTest(cutoff=cutoff):
                _, _, result = materialize_at(
                    self.observations,
                    cutoff,
                )
                self.assertEqual(
                    (
                        result.materialized_count,
                        result.unresolved_count,
                        result.unavailable_count,
                        result.authority_missing_count,
                    ),
                    expected_counts,
                )
                self.assertEqual(
                    result.materialization_set_fingerprint,
                    expected_fingerprint,
                )

    def test_exact_identities_and_source_provenance_are_unchanged(
        self,
    ) -> None:
        resolution, construction, result = materialize_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )
        resolved_by_id = {
            value.source_observation_id: value
            for value in resolution.resolved_candidates
        }
        constructed_by_id = {
            value.source_observation_id: value
            for value in construction.constructed_identities
        }

        for materialization in result.game_materializations:
            source_id = materialization.source_observation_id
            resolved = resolved_by_id[source_id]
            constructed = constructed_by_id[source_id]
            with self.subTest(source_observation_id=source_id):
                self.assertIs(
                    materialization.match_identity,
                    constructed.match_identity,
                )
                self.assertIs(
                    materialization.baseball_game.identity,
                    constructed.match_identity,
                )
                self.assertEqual(
                    materialization.baseball_game.scheduled_start.value,
                    resolved.scheduled_start_utc,
                )
                self.assertEqual(
                    materialization.source_raw_payload_sha256,
                    resolved.source_raw_payload_sha256,
                )
                self.assertEqual(
                    materialization.source_resolution_set_fingerprint,
                    resolution.resolution_set_fingerprint,
                )
                self.assertEqual(
                    materialization.authority_catalog_fingerprint,
                    construction.authority_catalog_fingerprint,
                )
                self.assertEqual(
                    materialization.source_construction_set_fingerprint,
                    construction.construction_set_fingerprint,
                )

    def test_postponed_revision_updates_only_explicit_schedule_evidence(
        self,
    ) -> None:
        _, _, mid = materialize_at(
            self.observations,
            MID_CUTOFF_AS_OF,
        )
        _, _, after = materialize_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )
        mid_by_game = {
            value.match_identity.canonical_game_id: value
            for value in mid.game_materializations
        }
        after_by_game = {
            value.match_identity.canonical_game_id: value
            for value in after.game_materializations
        }
        game_id = "FIXTURE_CANONICAL_MLB_GAME_777001"

        self.assertIs(
            mid_by_game[game_id].match_identity,
            mid_by_game[game_id].baseball_game.identity,
        )
        self.assertEqual(
            mid_by_game[game_id].baseball_game.scheduled_start.to_iso8601(),
            "2026-04-04T18:10:00Z",
        )
        self.assertEqual(
            after_by_game[
                game_id
            ].baseball_game.scheduled_start.to_iso8601(),
            "2026-04-05T20:10:00Z",
        )
        self.assertNotEqual(
            mid_by_game[game_id].source_observation_id,
            after_by_game[game_id].source_observation_id,
        )
        self.assertNotEqual(
            mid_by_game[game_id].source_raw_payload_sha256,
            after_by_game[game_id].source_raw_payload_sha256,
        )

    def test_missing_unresolved_unavailable_and_authority_missing_make_no_game(
        self,
    ) -> None:
        _, _, before = materialize_at(
            self.observations,
            BEFORE_ALL_AS_OF,
        )
        self.assertEqual(before.game_materializations, ())
        self.assertEqual(before.unavailable_count, 2)

        _, _, unresolved = materialize_at(
            self.observations,
            MID_CUTOFF_AS_OF,
            mappings=(),
        )
        self.assertEqual(unresolved.game_materializations, ())
        self.assertEqual(unresolved.unresolved_count, 2)

        empty_catalog = build_match_identity_authority_catalog(())
        _, _, authority_missing = materialize_at(
            self.observations,
            MID_CUTOFF_AS_OF,
            catalog=empty_catalog,
        )
        self.assertEqual(authority_missing.game_materializations, ())
        self.assertEqual(authority_missing.authority_missing_count, 2)

    def test_shuffled_and_repeated_input_is_identical(self) -> None:
        shuffled_observations = list(self.observations)
        random.Random(2026).shuffle(shuffled_observations)
        shuffled_authority = list(fixture_authority_entries())
        random.Random(2031).shuffle(shuffled_authority)
        shuffled_catalog = build_match_identity_authority_catalog(
            tuple(shuffled_authority)
        )

        _, _, ordered = materialize_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )
        _, _, shuffled = materialize_at(
            tuple(shuffled_observations),
            AFTER_POSTPONED_AS_OF,
            catalog=shuffled_catalog,
        )
        _, _, repeated = materialize_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )

        self.assertEqual(ordered, shuffled)
        self.assertEqual(ordered, repeated)
        self.assertEqual(
            ordered.materialization_set_fingerprint,
            AFTER_POSTPONED_MATERIALIZATION_FINGERPRINT,
        )


if __name__ == "__main__":
    unittest.main()
