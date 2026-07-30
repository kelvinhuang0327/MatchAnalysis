"""Unit tests for canonical schedule baseball-game materialization."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
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
from match_analysis.baseball.domain.game import BaseballGame
from match_analysis.baseball.domain.match_identity_authority import (
    SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION,
    ConstructedScheduleMatchIdentity,
    ScheduleMatchIdentityConstructionSet,
    build_match_identity_authority_catalog,
    compute_schedule_match_identity_construction_set_fingerprint,
)
from match_analysis.baseball.domain.schedule_game_materialization import (
    SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION,
    ScheduleBaseballGameMaterialization,
    ScheduleBaseballGameMaterializationSet,
)
from match_analysis.core.identity import MatchIdentity
from match_analysis.core.time import UtcTimestamp
from tests.unit.test_construct_match_identities import (
    authority_entry,
    make_resolved_candidate,
    make_resolution_set,
    make_unresolved_candidate,
)


def construction_for(resolution, authority_entries):
    return construct_match_identities(
        resolution,
        build_match_identity_authority_catalog(tuple(authority_entries)),
    )


def rebuild_construction(
    original: ScheduleMatchIdentityConstructionSet,
    *,
    constructed_identities=None,
):
    constructed = (
        original.constructed_identities
        if constructed_identities is None
        else tuple(constructed_identities)
    )
    fingerprint = (
        compute_schedule_match_identity_construction_set_fingerprint(
            as_of_utc=original.as_of_utc,
            source_resolution_set_fingerprint=(
                original.source_resolution_set_fingerprint
            ),
            authority_catalog_fingerprint=(
                original.authority_catalog_fingerprint
            ),
            constructed_count=len(constructed),
            unresolved_count=original.unresolved_count,
            unavailable_count=original.unavailable_count,
            authority_missing_count=original.authority_missing_count,
            constructed_identities=constructed,
            unresolved_candidates=original.unresolved_candidates,
            unavailable_chain_keys=original.unavailable_chain_keys,
            authority_missing_candidates=(
                original.authority_missing_candidates
            ),
        )
    )
    return ScheduleMatchIdentityConstructionSet(
        as_of_utc=original.as_of_utc,
        source_resolution_set_fingerprint=(
            original.source_resolution_set_fingerprint
        ),
        authority_catalog_fingerprint=(
            original.authority_catalog_fingerprint
        ),
        constructed_identities=constructed,
        unresolved_candidates=original.unresolved_candidates,
        unavailable_chain_keys=original.unavailable_chain_keys,
        authority_missing_candidates=(
            original.authority_missing_candidates
        ),
        constructed_count=len(constructed),
        unresolved_count=original.unresolved_count,
        unavailable_count=original.unavailable_count,
        authority_missing_count=original.authority_missing_count,
        construction_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION
        ),
    )


class ScheduleBaseballGameMaterializationTests(unittest.TestCase):
    def test_existing_game_and_exact_p11b_identity_are_preserved(self) -> None:
        resolved = make_resolved_candidate()
        resolution = make_resolution_set(
            resolved_candidates=(resolved,)
        )
        construction = construction_for(
            resolution,
            (
                authority_entry(
                    resolved.provider_game_id,
                    resolved.game_number,
                    "EXPLICIT_CANONICAL_GAME",
                ),
            ),
        )

        result = materialize_schedule_baseball_games(
            construction,
            resolution,
        )

        materialization = result.game_materializations[0]
        constructed = construction.constructed_identities[0]
        self.assertIsInstance(
            materialization,
            ScheduleBaseballGameMaterialization,
        )
        self.assertIsInstance(materialization.baseball_game, BaseballGame)
        self.assertIsInstance(
            materialization.baseball_game.scheduled_start,
            UtcTimestamp,
        )
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
        self.assertEqual(result.as_of_utc.utcoffset().total_seconds(), 0)
        self.assertEqual(
            (
                materialization.source_observation_id,
                materialization.source_raw_payload_sha256,
                materialization.source_resolution_set_fingerprint,
                materialization.authority_catalog_fingerprint,
                materialization.source_construction_set_fingerprint,
            ),
            (
                resolved.source_observation_id,
                resolved.source_raw_payload_sha256,
                resolution.resolution_set_fingerprint,
                construction.authority_catalog_fingerprint,
                construction.construction_set_fingerprint,
            ),
        )

    def test_result_contract_is_exact_immutable_and_fingerprint_bound(
        self,
    ) -> None:
        resolved = make_resolved_candidate()
        resolution = make_resolution_set(
            resolved_candidates=(resolved,)
        )
        construction = construction_for(
            resolution,
            (
                authority_entry(
                    resolved.provider_game_id,
                    resolved.game_number,
                    "EXPLICIT_CANONICAL_GAME",
                ),
            ),
        )
        result = materialize_schedule_baseball_games(
            construction,
            resolution,
        )

        self.assertEqual(
            set(
                ScheduleBaseballGameMaterialization.__dataclass_fields__
            ),
            {
                "baseball_game",
                "match_identity",
                "source_observation_id",
                "source_raw_payload_sha256",
                "source_resolution_set_fingerprint",
                "authority_catalog_fingerprint",
                "source_construction_set_fingerprint",
            },
        )
        self.assertEqual(
            set(
                ScheduleBaseballGameMaterializationSet.__dataclass_fields__
            ),
            {
                "as_of_utc",
                "source_resolution_set_fingerprint",
                "authority_catalog_fingerprint",
                "source_construction_set_fingerprint",
                "game_materializations",
                "unresolved_candidates",
                "unavailable_chain_keys",
                "authority_missing_candidates",
                "materialized_count",
                "unresolved_count",
                "unavailable_count",
                "authority_missing_count",
                "materialization_set_fingerprint",
                "schema_version",
            },
        )
        self.assertEqual(
            result.schema_version,
            SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION,
        )
        with self.assertRaises(FrozenInstanceError):
            result.materialized_count = 0
        with self.assertRaises(ValueError):
            replace(result, materialization_set_fingerprint="0" * 64)
        with self.assertRaises(ValueError):
            replace(result, materialized_count=0)

        materialization = result.game_materializations[0]
        equal_but_distinct_identity = replace(
            materialization.match_identity
        )
        with self.assertRaises(ValueError):
            replace(
                materialization,
                match_identity=equal_but_distinct_identity,
            )

    def test_unresolved_unavailable_and_authority_missing_create_no_game(
        self,
    ) -> None:
        resolved = make_resolved_candidate()
        unresolved = make_unresolved_candidate()
        unavailable = ("MLB_STATS_API", "unavailable-provider-game", 3)
        resolution = make_resolution_set(
            resolved_candidates=(resolved,),
            unresolved_candidates=(unresolved,),
            unavailable_chain_keys=(unavailable,),
        )
        construction = construction_for(resolution, ())

        result = materialize_schedule_baseball_games(
            construction,
            resolution,
        )

        self.assertEqual(result.game_materializations, ())
        self.assertEqual(
            (
                result.materialized_count,
                result.unresolved_count,
                result.unavailable_count,
                result.authority_missing_count,
            ),
            (0, 1, 1, 1),
        )
        self.assertIs(
            result.unresolved_candidates,
            construction.unresolved_candidates,
        )
        self.assertIs(
            result.unavailable_chain_keys,
            construction.unavailable_chain_keys,
        )
        self.assertIs(
            result.authority_missing_candidates,
            construction.authority_missing_candidates,
        )

    def test_mismatched_or_conflicting_join_evidence_fails_closed(
        self,
    ) -> None:
        resolved = make_resolved_candidate()
        resolution = make_resolution_set(
            resolved_candidates=(resolved,)
        )
        construction = construction_for(
            resolution,
            (
                authority_entry(
                    resolved.provider_game_id,
                    resolved.game_number,
                    "EXPLICIT_CANONICAL_GAME",
                ),
            ),
        )
        other_resolution = make_resolution_set(
            resolved_candidates=(
                replace(
                    resolved,
                    scheduled_start_utc=resolved.scheduled_start_utc.replace(
                        hour=13
                    ),
                ),
            )
        )
        with self.assertRaises(ValueError):
            materialize_schedule_baseball_games(
                construction,
                other_resolution,
            )

        conflicting = replace(
            construction.constructed_identities[0],
            source_raw_payload_sha256="f" * 64,
        )
        conflicting_construction = rebuild_construction(
            construction,
            constructed_identities=(conflicting,),
        )
        with self.assertRaises(ValueError):
            materialize_schedule_baseball_games(
                conflicting_construction,
                resolution,
            )

    def test_duplicate_observation_join_fails_closed(self) -> None:
        first = make_resolved_candidate("777001", 1)
        second = replace(
            make_resolved_candidate("777002", 2),
            source_observation_id=first.source_observation_id,
        )
        resolution = make_resolution_set(
            resolved_candidates=(first, second)
        )
        identity = MatchIdentity(
            sport="baseball",
            league="MLB",
            season=2031,
            canonical_game_id="EXPLICIT_CANONICAL_GAME",
            home_participant=first.home_canonical_participant_id,
            away_participant=first.away_canonical_participant_id,
        )
        constructed = ConstructedScheduleMatchIdentity(
            match_identity=identity,
            source_observation_id=first.source_observation_id,
            source_raw_payload_sha256=first.source_raw_payload_sha256,
            source_candidate_set_fingerprint=(
                resolution.source_candidate_set_fingerprint
            ),
            source_resolution_set_fingerprint=(
                resolution.resolution_set_fingerprint
            ),
            mapping_set_fingerprint=resolution.mapping_set_fingerprint,
        )
        authority_fingerprint = sha256(
            b"explicit-authority-catalog\n"
        ).hexdigest()
        construction_fingerprint = (
            compute_schedule_match_identity_construction_set_fingerprint(
                as_of_utc=resolution.as_of_utc,
                source_resolution_set_fingerprint=(
                    resolution.resolution_set_fingerprint
                ),
                authority_catalog_fingerprint=authority_fingerprint,
                constructed_count=1,
                unresolved_count=0,
                unavailable_count=0,
                authority_missing_count=0,
                constructed_identities=(constructed,),
                unresolved_candidates=(),
                unavailable_chain_keys=(),
                authority_missing_candidates=(),
            )
        )
        construction = ScheduleMatchIdentityConstructionSet(
            as_of_utc=resolution.as_of_utc,
            source_resolution_set_fingerprint=(
                resolution.resolution_set_fingerprint
            ),
            authority_catalog_fingerprint=authority_fingerprint,
            constructed_identities=(constructed,),
            unresolved_candidates=(),
            unavailable_chain_keys=(),
            authority_missing_candidates=(),
            constructed_count=1,
            unresolved_count=0,
            unavailable_count=0,
            authority_missing_count=0,
            construction_set_fingerprint=construction_fingerprint,
            schema_version=(
                SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION
            ),
        )

        with self.assertRaises(ValueError):
            materialize_schedule_baseball_games(
                construction,
                resolution,
            )

    def test_repeated_inputs_produce_identical_order_and_fingerprint(
        self,
    ) -> None:
        first = make_resolved_candidate("777001", 1)
        second = make_resolved_candidate("777002", 2)
        resolution = make_resolution_set(
            resolved_candidates=(first, second)
        )
        construction = construction_for(
            resolution,
            (
                authority_entry(
                    "777002",
                    2,
                    "EXPLICIT_CANONICAL_GAME_TWO",
                    game_discriminator="doubleheader_game_2",
                ),
                authority_entry(
                    "777001",
                    1,
                    "EXPLICIT_CANONICAL_GAME_ONE",
                ),
            ),
        )

        first_result = materialize_schedule_baseball_games(
            construction,
            resolution,
        )
        second_result = materialize_schedule_baseball_games(
            construction,
            resolution,
        )

        self.assertEqual(first_result, second_result)
        self.assertEqual(
            first_result.materialization_set_fingerprint,
            second_result.materialization_set_fingerprint,
        )
        self.assertEqual(
            [
                value.source_observation_id
                for value in first_result.game_materializations
            ],
            sorted(
                value.source_observation_id
                for value in first_result.game_materializations
            ),
        )

    def test_non_contract_inputs_are_rejected(self) -> None:
        resolution = make_resolution_set()
        construction = construction_for(resolution, ())

        with self.assertRaises(TypeError):
            materialize_schedule_baseball_games(object(), resolution)
        with self.assertRaises(TypeError):
            materialize_schedule_baseball_games(construction, object())


if __name__ == "__main__":
    unittest.main()
