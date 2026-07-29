"""Unit tests for explicit provider-participant identity resolution."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.resolve_schedule_participant_identities import (
    resolve_schedule_participant_identities,
)
from match_analysis.baseball.domain.participant_identity_resolution import (
    CONFLICTING_AWAY_PARTICIPANT_MAPPING,
    CONFLICTING_HOME_PARTICIPANT_MAPPING,
    MAPPING_VERSION_MISMATCH,
    MISSING_AWAY_PARTICIPANT_MAPPING,
    MISSING_HOME_PARTICIPANT_MAPPING,
    RESOLVED_PARTICIPANTS_NOT_DISTINCT,
    ProviderParticipantIdentityMapping,
    ResolvedScheduleIdentityCandidate,
    ScheduleParticipantIdentityResolutionSet,
    UnresolvedScheduleIdentityCandidate,
    compute_provider_participant_identity_mapping_set_fingerprint,
)
from match_analysis.baseball.domain.schedule_identity_candidate import (
    SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION,
    ScheduleIdentityResolutionCandidate,
    ScheduleIdentityResolutionCandidateSet,
    compute_schedule_identity_resolution_candidate_set_fingerprint,
)


AS_OF = datetime(2026, 3, 15, tzinfo=timezone.utc)
MAPPING_VERSION = "provider_participant_identity_map_v1"


def make_candidate(
    *,
    provider_game_id: str = "777001",
    game_number: int = 1,
    home_provider_participant_id: str = "118",
    away_provider_participant_id: str = "109",
) -> ScheduleIdentityResolutionCandidate:
    source_tag = f"{provider_game_id}:{game_number}"
    return ScheduleIdentityResolutionCandidate(
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_number=game_number,
        scheduled_start_utc=datetime(
            2026, 4, 4, 18 + game_number, 10, tzinfo=timezone.utc
        ),
        official_local_date=date(2026, 4, 4),
        home_provider_participant_id=home_provider_participant_id,
        away_provider_participant_id=away_provider_participant_id,
        source_observation_id=sha256(
            f"observation:{source_tag}".encode("utf-8")
        ).hexdigest(),
        source_raw_payload_sha256=sha256(
            f"payload:{source_tag}".encode("utf-8")
        ).hexdigest(),
    )


def make_candidate_set(
    *candidates: ScheduleIdentityResolutionCandidate,
    unavailable_chain_keys=(),
) -> ScheduleIdentityResolutionCandidateSet:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.provider_namespace,
                candidate.provider_game_id,
                candidate.game_number,
            ),
        )
    )
    unavailable_chain_keys = tuple(sorted(unavailable_chain_keys))
    fingerprint = (
        compute_schedule_identity_resolution_candidate_set_fingerprint(
            as_of_utc=AS_OF,
            source_snapshot_fingerprint="1" * 64,
            candidate_count=len(ordered),
            unavailable_count=len(unavailable_chain_keys),
            candidates=ordered,
            unavailable_chain_keys=unavailable_chain_keys,
        )
    )
    return ScheduleIdentityResolutionCandidateSet(
        as_of_utc=AS_OF,
        source_snapshot_fingerprint="1" * 64,
        candidates=ordered,
        unavailable_chain_keys=unavailable_chain_keys,
        candidate_count=len(ordered),
        unavailable_count=len(unavailable_chain_keys),
        candidate_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_IDENTITY_RESOLUTION_CANDIDATE_SET_SCHEMA_VERSION
        ),
    )


def mapping(
    provider_participant_id: str,
    canonical_participant_id: str,
    *,
    mapping_version: str = MAPPING_VERSION,
) -> ProviderParticipantIdentityMapping:
    return ProviderParticipantIdentityMapping(
        provider_namespace="MLB_STATS_API",
        provider_participant_id=provider_participant_id,
        canonical_participant_id=canonical_participant_id,
        mapping_version=mapping_version,
    )


def complete_mappings():
    return (
        mapping("118", "CANONICAL_118"),
        mapping("109", "CANONICAL_109"),
    )


class MappingContractTests(unittest.TestCase):
    def test_mapping_is_exact_immutable_and_rejects_blank_fields(self) -> None:
        value = complete_mappings()[0]

        self.assertEqual(
            set(value.__dataclass_fields__),
            {
                "provider_namespace",
                "provider_participant_id",
                "canonical_participant_id",
                "mapping_version",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            value.canonical_participant_id = "OTHER"
        for field_name in value.__dataclass_fields__:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    replace(value, **{field_name: " "})

    def test_mapping_fingerprint_is_order_independent_and_duplicates_are_idempotent(
        self,
    ) -> None:
        mappings = complete_mappings()
        repeated = (mappings[1], mappings[0], mappings[0])

        self.assertEqual(
            compute_provider_participant_identity_mapping_set_fingerprint(
                mappings
            ),
            compute_provider_participant_identity_mapping_set_fingerprint(
                repeated
            ),
        )


class ResolutionUseCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = make_candidate()
        self.candidate_set = make_candidate_set(self.candidate)

    def test_exact_provider_keys_resolve_and_copy_all_p9_evidence(
        self,
    ) -> None:
        result = resolve_schedule_participant_identities(
            self.candidate_set, complete_mappings()
        )
        resolved = result.resolved_candidates[0]

        self.assertEqual(
            (
                resolved.provider_namespace,
                resolved.provider_game_id,
                resolved.game_number,
                resolved.scheduled_start_utc,
                resolved.official_local_date,
                resolved.home_provider_participant_id,
                resolved.away_provider_participant_id,
                resolved.source_observation_id,
                resolved.source_raw_payload_sha256,
            ),
            (
                self.candidate.provider_namespace,
                self.candidate.provider_game_id,
                self.candidate.game_number,
                self.candidate.scheduled_start_utc,
                self.candidate.official_local_date,
                self.candidate.home_provider_participant_id,
                self.candidate.away_provider_participant_id,
                self.candidate.source_observation_id,
                self.candidate.source_raw_payload_sha256,
            ),
        )
        self.assertEqual(
            (
                resolved.home_canonical_participant_id,
                resolved.away_canonical_participant_id,
                resolved.mapping_version,
            ),
            ("CANONICAL_118", "CANONICAL_109", MAPPING_VERSION),
        )
        self.assertEqual(result.resolved_count, 1)
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(
            result.schema_version,
            "schedule_participant_identity_resolution_set_v1",
        )

    def test_missing_home_and_away_mappings_fail_closed(self) -> None:
        cases = (
            (
                (complete_mappings()[1],),
                (MISSING_HOME_PARTICIPANT_MAPPING,),
            ),
            (
                (complete_mappings()[0],),
                (MISSING_AWAY_PARTICIPANT_MAPPING,),
            ),
            (
                (),
                (
                    MISSING_HOME_PARTICIPANT_MAPPING,
                    MISSING_AWAY_PARTICIPANT_MAPPING,
                ),
            ),
        )
        for mappings, expected_reasons in cases:
            with self.subTest(expected_reasons=expected_reasons):
                result = resolve_schedule_participant_identities(
                    self.candidate_set, mappings
                )
                self.assertEqual(result.resolved_count, 0)
                self.assertEqual(
                    result.unresolved_candidates[0].reasons,
                    expected_reasons,
                )

    def test_conflicting_home_and_away_mappings_fail_closed(self) -> None:
        home_conflict = (
            *complete_mappings(),
            mapping("118", "DIFFERENT_HOME"),
        )
        away_conflict = (
            *complete_mappings(),
            mapping("109", "DIFFERENT_AWAY"),
        )
        both_conflict = (*home_conflict, mapping("109", "DIFFERENT_AWAY"))
        cases = (
            (
                home_conflict,
                (CONFLICTING_HOME_PARTICIPANT_MAPPING,),
            ),
            (
                away_conflict,
                (CONFLICTING_AWAY_PARTICIPANT_MAPPING,),
            ),
            (
                both_conflict,
                (
                    CONFLICTING_HOME_PARTICIPANT_MAPPING,
                    CONFLICTING_AWAY_PARTICIPANT_MAPPING,
                ),
            ),
        )
        for mappings, expected_reasons in cases:
            with self.subTest(expected_reasons=expected_reasons):
                result = resolve_schedule_participant_identities(
                    self.candidate_set, mappings
                )
                self.assertEqual(
                    result.unresolved_candidates[0].reasons,
                    expected_reasons,
                )

    def test_non_distinct_and_version_mismatch_reasons_are_controlled(
        self,
    ) -> None:
        cases = (
            (
                (
                    mapping("118", "SAME"),
                    mapping("109", "SAME"),
                ),
                (RESOLVED_PARTICIPANTS_NOT_DISTINCT,),
            ),
            (
                (
                    mapping("118", "CANONICAL_118"),
                    mapping(
                        "109",
                        "CANONICAL_109",
                        mapping_version="provider_map_v2",
                    ),
                ),
                (MAPPING_VERSION_MISMATCH,),
            ),
        )
        for mappings, expected_reasons in cases:
            with self.subTest(expected_reasons=expected_reasons):
                result = resolve_schedule_participant_identities(
                    self.candidate_set, mappings
                )
                self.assertEqual(
                    result.unresolved_candidates[0].reasons,
                    expected_reasons,
                )

    def test_shuffled_mappings_and_repeated_runs_are_identical(self) -> None:
        mappings = list(complete_mappings())
        random.Random(2026).shuffle(mappings)

        first = resolve_schedule_participant_identities(
            self.candidate_set, tuple(mappings)
        )
        second = resolve_schedule_participant_identities(
            self.candidate_set, complete_mappings()
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first.resolution_set_fingerprint,
            second.resolution_set_fingerprint,
        )

    def test_unavailable_keys_are_inherited_without_resolution(self) -> None:
        candidate_set = make_candidate_set(
            unavailable_chain_keys=(
                ("MLB_STATS_API", "777002", 2),
                ("MLB_STATS_API", "777001", 1),
            )
        )

        result = resolve_schedule_participant_identities(
            candidate_set, complete_mappings()
        )

        self.assertEqual(result.resolved_count, 0)
        self.assertEqual(result.unresolved_count, 0)
        self.assertEqual(result.unavailable_count, 2)
        self.assertEqual(
            result.unavailable_chain_keys,
            candidate_set.unavailable_chain_keys,
        )

    def test_result_contract_is_immutable_and_has_no_match_or_game(self) -> None:
        result = resolve_schedule_participant_identities(
            self.candidate_set, complete_mappings()
        )
        result_fields = set(
            ScheduleParticipantIdentityResolutionSet.__dataclass_fields__
        )
        resolved_fields = set(
            ResolvedScheduleIdentityCandidate.__dataclass_fields__
        )
        unresolved_fields = set(
            UnresolvedScheduleIdentityCandidate.__dataclass_fields__
        )

        self.assertTrue(
            {
                "match_identity",
                "baseball_game",
                "prediction",
                "team_name",
                "alias",
            }.isdisjoint(
                result_fields | resolved_fields | unresolved_fields
            )
        )
        with self.assertRaises(FrozenInstanceError):
            result.resolved_count = 2
        with self.assertRaises(ValueError):
            replace(result, resolution_set_fingerprint="0" * 64)

    def test_non_candidate_set_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            resolve_schedule_participant_identities(
                object(), complete_mappings()
            )


if __name__ == "__main__":
    unittest.main()
