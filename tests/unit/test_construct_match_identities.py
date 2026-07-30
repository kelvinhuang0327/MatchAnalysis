"""Unit tests for explicit P10-to-P1 match-identity construction."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.construct_match_identities import (
    BASEBALL_MATCH_IDENTITY_SPORT,
    construct_match_identities,
)
from match_analysis.baseball.domain.match_identity_authority import (
    SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION,
    ConstructedScheduleMatchIdentity,
    MatchIdentityAuthorityEntry,
    ScheduleMatchIdentityConstructionSet,
    build_match_identity_authority_catalog,
)
from match_analysis.baseball.domain.participant_identity_resolution import (
    MISSING_HOME_PARTICIPANT_MAPPING,
    ResolvedScheduleIdentityCandidate,
    ScheduleParticipantIdentityResolutionSet,
    UnresolvedScheduleIdentityCandidate,
    compute_schedule_participant_identity_resolution_set_fingerprint,
)
from match_analysis.core.identity import MatchIdentity
from tests.characterization.test_participant_identity_resolution_fixture import (
    AFTER_POSTPONED_AS_OF,
    BEFORE_ALL_AS_OF,
    EXACT_BOUNDARY_AS_OF,
    MID_CUTOFF_AS_OF,
    load_fixture_observations,
    resolve_at,
)


AUTHORITY_CATALOG_FINGERPRINT = (
    "8e90640fa4c10eb71009fc556c8d8d6cb9bde444fe98a0097a480801fdf6a9dd"
)
BEFORE_ALL_CONSTRUCTION_FINGERPRINT = (
    "d04b54d3466f16caf5f14c6a77f642eb17eb96e02dc4678142b229c4259df194"
)
MID_CUTOFF_CONSTRUCTION_FINGERPRINT = (
    "672568c6a10272c9e8a0d8e03eac51334feffa9d21ca0fa0756d0fd8486a9935"
)
AFTER_POSTPONED_CONSTRUCTION_FINGERPRINT = (
    "78d7476f34d1af72beeab46165f19b0ad3579f866a46113372dbd61fd090f3db"
)
EXACT_BOUNDARY_CONSTRUCTION_FINGERPRINT = (
    "a3c861bd3cfff99265c255c9fdb2c0afbdf3c62138cffa05c2d8c6520a9d35a4"
)


def authority_entry(
    provider_game_id: str,
    game_number: int,
    canonical_game_id: str,
    *,
    league: str = "MLB",
    season: int = 2026,
    game_discriminator: str | None = None,
) -> MatchIdentityAuthorityEntry:
    return MatchIdentityAuthorityEntry(
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_number=game_number,
        league=league,
        season=season,
        canonical_game_id=canonical_game_id,
        game_discriminator=game_discriminator,
        authority_version="fixture_match_identity_authority_v1",
    )


def fixture_authority_entries():
    return (
        authority_entry(
            "777001",
            1,
            "FIXTURE_CANONICAL_MLB_GAME_777001",
        ),
        authority_entry(
            "777002",
            2,
            "FIXTURE_CANONICAL_MLB_GAME_777002",
            game_discriminator="doubleheader_game_2",
        ),
    )


def fixture_catalog():
    return build_match_identity_authority_catalog(
        fixture_authority_entries()
    )


def make_resolved_candidate(
    provider_game_id: str = "777001",
    game_number: int = 1,
) -> ResolvedScheduleIdentityCandidate:
    source_tag = f"{provider_game_id}:{game_number}"
    return ResolvedScheduleIdentityCandidate(
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_number=game_number,
        scheduled_start_utc=datetime(
            2031, 7, game_number, 12, tzinfo=timezone.utc
        ),
        official_local_date=date(2031, 7, game_number),
        home_provider_participant_id="provider-home",
        away_provider_participant_id="provider-away",
        source_observation_id=sha256(
            f"observation:{source_tag}".encode("utf-8")
        ).hexdigest(),
        source_raw_payload_sha256=sha256(
            f"payload:{source_tag}".encode("utf-8")
        ).hexdigest(),
        home_canonical_participant_id="EXPLICIT_CANONICAL_HOME",
        away_canonical_participant_id="EXPLICIT_CANONICAL_AWAY",
        mapping_version="explicit_participant_map_v1",
    )


def make_unresolved_candidate(
    provider_game_id: str = "unresolved-provider-game",
    game_number: int = 1,
) -> UnresolvedScheduleIdentityCandidate:
    return UnresolvedScheduleIdentityCandidate(
        source_observation_id=sha256(
            f"unresolved:{provider_game_id}:{game_number}".encode("utf-8")
        ).hexdigest(),
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_number=game_number,
        reasons=(MISSING_HOME_PARTICIPANT_MAPPING,),
    )


def make_resolution_set(
    *,
    resolved_candidates=(),
    unresolved_candidates=(),
    unavailable_chain_keys=(),
) -> ScheduleParticipantIdentityResolutionSet:
    resolved_candidates = tuple(
        sorted(
            resolved_candidates,
            key=lambda candidate: (
                candidate.provider_namespace,
                candidate.provider_game_id,
                candidate.game_number,
            ),
        )
    )
    unresolved_candidates = tuple(
        sorted(
            unresolved_candidates,
            key=lambda candidate: (
                candidate.provider_namespace,
                candidate.provider_game_id,
                candidate.game_number,
            ),
        )
    )
    unavailable_chain_keys = tuple(sorted(unavailable_chain_keys))
    source_candidate_set_fingerprint = "a" * 64
    mapping_set_fingerprint = "b" * 64
    as_of_utc = datetime(2031, 7, 1, tzinfo=timezone.utc)
    resolution_set_fingerprint = (
        compute_schedule_participant_identity_resolution_set_fingerprint(
            as_of_utc=as_of_utc,
            source_candidate_set_fingerprint=(
                source_candidate_set_fingerprint
            ),
            mapping_set_fingerprint=mapping_set_fingerprint,
            resolved_count=len(resolved_candidates),
            unresolved_count=len(unresolved_candidates),
            unavailable_count=len(unavailable_chain_keys),
            resolved_candidates=resolved_candidates,
            unresolved_candidates=unresolved_candidates,
            unavailable_chain_keys=unavailable_chain_keys,
        )
    )
    return ScheduleParticipantIdentityResolutionSet(
        as_of_utc=as_of_utc,
        source_candidate_set_fingerprint=(
            source_candidate_set_fingerprint
        ),
        mapping_set_fingerprint=mapping_set_fingerprint,
        resolved_candidates=resolved_candidates,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        resolved_count=len(resolved_candidates),
        unresolved_count=len(unresolved_candidates),
        unavailable_count=len(unavailable_chain_keys),
        resolution_set_fingerprint=resolution_set_fingerprint,
        schema_version=(
            "schedule_participant_identity_resolution_set_v1"
        ),
    )


class FixtureConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.observations = load_fixture_observations()

    def construct_at(self, cutoff):
        _, resolution = resolve_at(self.observations, cutoff)
        return resolution, construct_match_identities(
            resolution,
            fixture_catalog(),
        )

    def test_catalog_and_four_pre_edit_construction_references(self) -> None:
        self.assertEqual(
            fixture_catalog().catalog_fingerprint,
            AUTHORITY_CATALOG_FINGERPRINT,
        )
        cases = (
            (
                BEFORE_ALL_AS_OF,
                (0, 0, 2, 0),
                BEFORE_ALL_CONSTRUCTION_FINGERPRINT,
            ),
            (
                MID_CUTOFF_AS_OF,
                (2, 0, 0, 0),
                MID_CUTOFF_CONSTRUCTION_FINGERPRINT,
            ),
            (
                AFTER_POSTPONED_AS_OF,
                (2, 0, 0, 0),
                AFTER_POSTPONED_CONSTRUCTION_FINGERPRINT,
            ),
            (
                EXACT_BOUNDARY_AS_OF,
                (1, 0, 1, 0),
                EXACT_BOUNDARY_CONSTRUCTION_FINGERPRINT,
            ),
        )

        for cutoff, expected_counts, expected_fingerprint in cases:
            with self.subTest(cutoff=cutoff):
                _, result = self.construct_at(cutoff)
                self.assertEqual(
                    (
                        result.constructed_count,
                        result.unresolved_count,
                        result.unavailable_count,
                        result.authority_missing_count,
                    ),
                    expected_counts,
                )
                self.assertEqual(
                    result.construction_set_fingerprint,
                    expected_fingerprint,
                )

    def test_existing_p1_identity_and_explicit_sources_are_preserved(
        self,
    ) -> None:
        resolution, result = self.construct_at(AFTER_POSTPONED_AS_OF)
        first, second = result.constructed_identities

        self.assertIsInstance(first, ConstructedScheduleMatchIdentity)
        self.assertIsInstance(first.match_identity, MatchIdentity)
        self.assertEqual(
            (
                first.match_identity.sport,
                first.match_identity.league,
                first.match_identity.season,
                first.match_identity.canonical_game_id,
                first.match_identity.home_participant,
                first.match_identity.away_participant,
                first.match_identity.game_discriminator,
            ),
            (
                "baseball",
                "MLB",
                2026,
                "FIXTURE_CANONICAL_MLB_GAME_777001",
                "FIXTURE_CANONICAL_MLB_TEAM_118",
                "FIXTURE_CANONICAL_MLB_TEAM_109",
                None,
            ),
        )
        self.assertEqual(
            (
                second.match_identity.canonical_game_id,
                second.match_identity.game_discriminator,
            ),
            (
                "FIXTURE_CANONICAL_MLB_GAME_777002",
                "doubleheader_game_2",
            ),
        )
        for resolved, constructed in zip(
            resolution.resolved_candidates,
            result.constructed_identities,
            strict=True,
        ):
            self.assertEqual(
                (
                    constructed.source_observation_id,
                    constructed.source_raw_payload_sha256,
                    constructed.source_candidate_set_fingerprint,
                    constructed.source_resolution_set_fingerprint,
                    constructed.mapping_set_fingerprint,
                ),
                (
                    resolved.source_observation_id,
                    resolved.source_raw_payload_sha256,
                    resolution.source_candidate_set_fingerprint,
                    resolution.resolution_set_fingerprint,
                    resolution.mapping_set_fingerprint,
                ),
            )

    def test_missing_authority_fails_closed_for_only_that_candidate(
        self,
    ) -> None:
        _, resolution = resolve_at(
            self.observations,
            MID_CUTOFF_AS_OF,
        )
        partial_catalog = build_match_identity_authority_catalog(
            (fixture_authority_entries()[0],)
        )

        result = construct_match_identities(
            resolution,
            partial_catalog,
        )

        self.assertEqual(result.constructed_count, 1)
        self.assertEqual(result.authority_missing_count, 1)
        self.assertEqual(
            (
                result.authority_missing_candidates[0].provider_game_id,
                result.authority_missing_candidates[0].game_number,
            ),
            ("777002", 2),
        )
        self.assertEqual(
            result.constructed_identities[0].match_identity.canonical_game_id,
            "FIXTURE_CANONICAL_MLB_GAME_777001",
        )

    def test_shuffled_repeated_authority_inputs_are_identical(self) -> None:
        _, resolution = resolve_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )
        shuffled = list(fixture_authority_entries())
        random.Random(2026).shuffle(shuffled)
        repeated = (shuffled[0], shuffled[1], shuffled[0])

        first = construct_match_identities(
            resolution,
            build_match_identity_authority_catalog(repeated),
        )
        second = construct_match_identities(
            resolution,
            fixture_catalog(),
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first.construction_set_fingerprint,
            AFTER_POSTPONED_CONSTRUCTION_FINGERPRINT,
        )


class FailClosedConstructionTests(unittest.TestCase):
    def test_unresolved_and_unavailable_inputs_are_never_materialized(
        self,
    ) -> None:
        unresolved = make_unresolved_candidate()
        unavailable = ("MLB_STATS_API", "unavailable-provider-game", 3)
        resolution = make_resolution_set(
            unresolved_candidates=(unresolved,),
            unavailable_chain_keys=(unavailable,),
        )

        result = construct_match_identities(
            resolution,
            build_match_identity_authority_catalog(()),
        )

        self.assertEqual(result.constructed_count, 0)
        self.assertEqual(result.authority_missing_count, 0)
        self.assertEqual(result.unresolved_candidates, (unresolved,))
        self.assertEqual(result.unavailable_chain_keys, (unavailable,))

    def test_two_game_numbers_need_independently_explicit_entries(self) -> None:
        game_one = make_resolved_candidate("shared-provider-game", 1)
        game_two = make_resolved_candidate("shared-provider-game", 2)
        resolution = make_resolution_set(
            resolved_candidates=(game_one, game_two)
        )
        game_one_authority = authority_entry(
            "shared-provider-game",
            1,
            "EXPLICIT_CANONICAL_GAME_ONE",
        )

        partial = construct_match_identities(
            resolution,
            build_match_identity_authority_catalog(
                (game_one_authority,)
            ),
        )

        self.assertEqual(partial.constructed_count, 1)
        self.assertEqual(partial.authority_missing_count, 1)
        self.assertEqual(
            partial.authority_missing_candidates[0].game_number,
            2,
        )

        complete = construct_match_identities(
            resolution,
            build_match_identity_authority_catalog(
                (
                    game_one_authority,
                    authority_entry(
                        "shared-provider-game",
                        2,
                        "EXPLICIT_CANONICAL_GAME_TWO",
                        game_discriminator="EXPLICIT_GAME_TWO_DISC",
                    ),
                )
            ),
        )
        self.assertEqual(
            [
                (
                    value.match_identity.canonical_game_id,
                    value.match_identity.game_discriminator,
                )
                for value in complete.constructed_identities
            ],
            [
                ("EXPLICIT_CANONICAL_GAME_ONE", None),
                (
                    "EXPLICIT_CANONICAL_GAME_TWO",
                    "EXPLICIT_GAME_TWO_DISC",
                ),
            ],
        )

    def test_lookup_evidence_is_never_transformed_into_identity_fields(
        self,
    ) -> None:
        resolved = make_resolved_candidate(
            provider_game_id="provider-raw-999",
            game_number=9,
        )
        resolution = make_resolution_set(
            resolved_candidates=(resolved,)
        )
        catalog = build_match_identity_authority_catalog(
            (
                authority_entry(
                    "provider-raw-999",
                    9,
                    "OPAQUE_EXPLICIT_CANONICAL_ID",
                    league="EXPLICIT_LEAGUE",
                    season=2042,
                    game_discriminator="OPAQUE_EXPLICIT_DISC",
                ),
            )
        )

        identity = construct_match_identities(
            resolution,
            catalog,
        ).constructed_identities[0].match_identity

        self.assertEqual(
            (
                identity.sport,
                identity.league,
                identity.season,
                identity.canonical_game_id,
                identity.home_participant,
                identity.away_participant,
                identity.game_discriminator,
            ),
            (
                BASEBALL_MATCH_IDENTITY_SPORT,
                "EXPLICIT_LEAGUE",
                2042,
                "OPAQUE_EXPLICIT_CANONICAL_ID",
                "EXPLICIT_CANONICAL_HOME",
                "EXPLICIT_CANONICAL_AWAY",
                "OPAQUE_EXPLICIT_DISC",
            ),
        )
        self.assertNotIn("provider-raw-999", identity.canonical_game_id)
        self.assertNotIn("2031", identity.canonical_game_id)
        self.assertNotEqual(identity.game_discriminator, "9")

    def test_conflicting_authority_never_reaches_construction(self) -> None:
        first = authority_entry(
            "777001",
            1,
            "EXPLICIT_CANONICAL_ONE",
        )
        conflict = replace(
            first,
            canonical_game_id="CONFLICTING_CANONICAL_ID",
        )

        with self.assertRaises(ValueError):
            build_match_identity_authority_catalog((first, conflict))


class ConstructionSetContractTests(unittest.TestCase):
    def test_result_contract_is_exact_immutable_and_fingerprint_bound(
        self,
    ) -> None:
        resolution = make_resolution_set(
            resolved_candidates=(make_resolved_candidate(),)
        )
        result = construct_match_identities(
            resolution,
            build_match_identity_authority_catalog(
                (
                    authority_entry(
                        "777001",
                        1,
                        "EXPLICIT_CANONICAL_GAME",
                    ),
                )
            ),
        )

        self.assertEqual(
            set(ScheduleMatchIdentityConstructionSet.__dataclass_fields__),
            {
                "as_of_utc",
                "source_resolution_set_fingerprint",
                "authority_catalog_fingerprint",
                "constructed_identities",
                "unresolved_candidates",
                "unavailable_chain_keys",
                "authority_missing_candidates",
                "constructed_count",
                "unresolved_count",
                "unavailable_count",
                "authority_missing_count",
                "construction_set_fingerprint",
                "schema_version",
            },
        )
        self.assertEqual(
            result.schema_version,
            SCHEDULE_MATCH_IDENTITY_CONSTRUCTION_SET_SCHEMA_VERSION,
        )
        with self.assertRaises(FrozenInstanceError):
            result.constructed_count = 2
        with self.assertRaises(ValueError):
            replace(result, construction_set_fingerprint="0" * 64)
        with self.assertRaises(ValueError):
            replace(result, constructed_count=0)

    def test_non_contract_inputs_are_rejected(self) -> None:
        catalog = build_match_identity_authority_catalog(())
        resolution = make_resolution_set()

        with self.assertRaises(TypeError):
            construct_match_identities(object(), catalog)
        with self.assertRaises(TypeError):
            construct_match_identities(resolution, object())


if __name__ == "__main__":
    unittest.main()
