"""Characterize P9 participant resolution against synthetic mappings."""

from datetime import date, datetime
import json
from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.schedule_observation_source import (
    ScheduleObservationCapture,
)
from match_analysis.application.use_cases.build_schedule_observation_revision_chains import (
    build_schedule_observation_revision_chains,
)
from match_analysis.application.use_cases.capture_schedule_observation import (
    capture_schedule_observation,
)
from match_analysis.application.use_cases.project_schedule_identity_candidates import (
    project_schedule_identity_candidates,
)
from match_analysis.application.use_cases.resolve_schedule_participant_identities import (
    resolve_schedule_participant_identities,
)
from match_analysis.application.use_cases.select_schedule_observations_as_of import (
    select_schedule_observations_as_of,
)
from match_analysis.baseball.domain.participant_identity_resolution import (
    CONFLICTING_HOME_PARTICIPANT_MAPPING,
    MISSING_AWAY_PARTICIPANT_MAPPING,
    MISSING_HOME_PARTICIPANT_MAPPING,
    ProviderParticipantIdentityMapping,
    compute_provider_participant_identity_mapping_set_fingerprint,
)


OBSERVATION_FIXTURE_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "mlb_schedule_observation_v1.json"
)
MAPPING_FIXTURE_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "provider_participant_identity_map_v1.json"
)

BEFORE_ALL_AS_OF = "2026-01-01T00:00:00Z"
MID_CUTOFF_AS_OF = "2026-03-15T00:00:00Z"
AFTER_POSTPONED_AS_OF = "2026-05-01T00:00:00Z"
EXACT_BOUNDARY_AS_OF = "2026-03-01T12:00:01.500Z"

MAPPING_SET_FINGERPRINT = (
    "ba3110e558910d8373548f36ba4f31e9e87766c6af2432d7d80fbf3fddd3ae1f"
)
BEFORE_ALL_RESOLUTION_FINGERPRINT = (
    "c8fb1cc41257034c7e27373b06ad2f03eb3d4e45ee6f61b19e0fa8652f97948e"
)
MID_CUTOFF_RESOLUTION_FINGERPRINT = (
    "58257c472038f486e4b3ecb9f882bf0356abfb9b80234367cc4c5b7b717a6fba"
)
AFTER_POSTPONED_RESOLUTION_FINGERPRINT = (
    "500bff4d698f5f881969f37a4ce72026b133ac75f08ceb9e9fe8d44f44fc6fd9"
)
EXACT_BOUNDARY_RESOLUTION_FINGERPRINT = (
    "9775d5e6562ba36b40da6b6979a89fa4cb6c0c1329c79f2a054edd859c80cb53"
)


class FixtureSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_fixture_observations():
    fixture = json.loads(OBSERVATION_FIXTURE_PATH.read_bytes())
    observations_by_key = {}
    ordered = []
    for item in fixture["observations"]:
        previous = observations_by_key.get(
            item["supersedes_fixture_key"]
        )
        raw_payload = item["raw_payload_utf8"].encode("utf-8")
        capture = ScheduleObservationCapture(
            provider_namespace=item["provider_namespace"],
            provider_game_id=item["provider_game_id"],
            scheduled_start_utc=parse_timestamp(
                item["scheduled_start_utc"]
            ),
            official_local_date=date.fromisoformat(
                item["official_local_date"]
            ),
            response_received_at_utc=parse_timestamp(
                item["response_received_at_utc"]
            ),
            ingested_at_utc=parse_timestamp(item["ingested_at_utc"]),
            provider_status_code=item["provider_status_code"],
            provider_detailed_status=item["provider_detailed_status"],
            game_number=item["game_number"],
            home_provider_participant_id=(
                item["home_provider_participant_id"]
            ),
            away_provider_participant_id=(
                item["away_provider_participant_id"]
            ),
            endpoint_id=item["endpoint_id"],
            parser_version=item["parser_version"],
            schema_version=item["schema_version"],
            raw_payload_bytes=raw_payload,
            raw_payload_sha256=item["raw_payload_sha256"],
            supersedes_observation_id=(
                previous.observation_id if previous is not None else None
            ),
        )
        observation = capture_schedule_observation(
            FixtureSource(capture), previous
        )
        observations_by_key[item["fixture_key"]] = observation
        ordered.append(observation)
    return observations_by_key, tuple(ordered)


def load_fixture_mappings():
    fixture = json.loads(MAPPING_FIXTURE_PATH.read_bytes())
    if (
        fixture["fixture_schema"]
        != "provider_participant_identity_map_fixture_v1"
    ):
        raise ValueError("unexpected mapping fixture schema")
    return tuple(
        ProviderParticipantIdentityMapping(**item)
        for item in fixture["mappings"]
    )


def project_at(observations, cutoff):
    revision_set = build_schedule_observation_revision_chains(observations)
    snapshot = select_schedule_observations_as_of(
        revision_set, parse_timestamp(cutoff)
    )
    return project_schedule_identity_candidates(snapshot)


def resolve_at(observations, cutoff, mappings=None):
    candidate_set = project_at(observations, cutoff)
    result = resolve_schedule_participant_identities(
        candidate_set,
        load_fixture_mappings() if mappings is None else mappings,
    )
    return candidate_set, result


class ParticipantIdentityResolutionFixtureTests(unittest.TestCase):
    def test_mapping_fixture_is_exact_synthetic_authority(self) -> None:
        mappings = load_fixture_mappings()

        self.assertEqual(
            {
                (
                    mapping.provider_namespace,
                    mapping.provider_participant_id,
                    mapping.canonical_participant_id,
                    mapping.mapping_version,
                )
                for mapping in mappings
            },
            {
                (
                    "MLB_STATS_API",
                    "118",
                    "FIXTURE_CANONICAL_MLB_TEAM_118",
                    "provider_participant_identity_map_v1",
                ),
                (
                    "MLB_STATS_API",
                    "109",
                    "FIXTURE_CANONICAL_MLB_TEAM_109",
                    "provider_participant_identity_map_v1",
                ),
            },
        )
        self.assertEqual(
            compute_provider_participant_identity_mapping_set_fingerprint(
                mappings
            ),
            MAPPING_SET_FINGERPRINT,
        )

    def test_four_cutoffs_match_pre_edit_resolution_references(self) -> None:
        _, observations = load_fixture_observations()
        cases = (
            (
                BEFORE_ALL_AS_OF,
                (0, 0, 2),
                BEFORE_ALL_RESOLUTION_FINGERPRINT,
            ),
            (
                MID_CUTOFF_AS_OF,
                (2, 0, 0),
                MID_CUTOFF_RESOLUTION_FINGERPRINT,
            ),
            (
                AFTER_POSTPONED_AS_OF,
                (2, 0, 0),
                AFTER_POSTPONED_RESOLUTION_FINGERPRINT,
            ),
            (
                EXACT_BOUNDARY_AS_OF,
                (1, 0, 1),
                EXACT_BOUNDARY_RESOLUTION_FINGERPRINT,
            ),
        )

        for cutoff, counts, fingerprint in cases:
            with self.subTest(cutoff=cutoff):
                _, result = resolve_at(observations, cutoff)
                self.assertEqual(
                    (
                        result.resolved_count,
                        result.unresolved_count,
                        result.unavailable_count,
                    ),
                    counts,
                )
                self.assertEqual(
                    result.resolution_set_fingerprint, fingerprint
                )

    def test_canonical_ids_and_p9_source_provenance_are_preserved(
        self,
    ) -> None:
        _, observations = load_fixture_observations()
        candidate_set, result = resolve_at(
            observations, AFTER_POSTPONED_AS_OF
        )

        for candidate, resolved in zip(
            candidate_set.candidates,
            result.resolved_candidates,
            strict=True,
        ):
            with self.subTest(
                provider_game_id=candidate.provider_game_id
            ):
                self.assertEqual(
                    resolved.home_provider_participant_id, "118"
                )
                self.assertEqual(
                    resolved.home_canonical_participant_id,
                    "FIXTURE_CANONICAL_MLB_TEAM_118",
                )
                self.assertEqual(
                    resolved.away_provider_participant_id, "109"
                )
                self.assertEqual(
                    resolved.away_canonical_participant_id,
                    "FIXTURE_CANONICAL_MLB_TEAM_109",
                )
                self.assertEqual(
                    resolved.source_observation_id,
                    candidate.source_observation_id,
                )
                self.assertEqual(
                    resolved.source_raw_payload_sha256,
                    candidate.source_raw_payload_sha256,
                )

    def test_missing_and_conflicting_fixture_mappings_fail_closed(
        self,
    ) -> None:
        _, observations = load_fixture_observations()
        mappings = load_fixture_mappings()
        home = next(
            mapping
            for mapping in mappings
            if mapping.provider_participant_id == "118"
        )
        away = next(
            mapping
            for mapping in mappings
            if mapping.provider_participant_id == "109"
        )
        conflicting_home = ProviderParticipantIdentityMapping(
            provider_namespace=home.provider_namespace,
            provider_participant_id=home.provider_participant_id,
            canonical_participant_id="FIXTURE_CONFLICTING_HOME",
            mapping_version=home.mapping_version,
        )
        cases = (
            ((away,), (MISSING_HOME_PARTICIPANT_MAPPING,)),
            ((home,), (MISSING_AWAY_PARTICIPANT_MAPPING,)),
            (
                (home, away, conflicting_home),
                (CONFLICTING_HOME_PARTICIPANT_MAPPING,),
            ),
        )

        for provided, reasons in cases:
            with self.subTest(reasons=reasons):
                _, result = resolve_at(
                    observations, MID_CUTOFF_AS_OF, provided
                )
                self.assertEqual(result.resolved_count, 0)
                self.assertEqual(result.unresolved_count, 2)
                self.assertTrue(
                    all(
                        unresolved.reasons == reasons
                        for unresolved in result.unresolved_candidates
                    )
                )

    def test_shuffled_mappings_and_repeated_runs_are_identical(self) -> None:
        _, observations = load_fixture_observations()
        mappings = list(load_fixture_mappings())
        random.Random(2026).shuffle(mappings)

        _, shuffled = resolve_at(
            observations, AFTER_POSTPONED_AS_OF, tuple(mappings)
        )
        _, first = resolve_at(observations, AFTER_POSTPONED_AS_OF)
        _, second = resolve_at(observations, AFTER_POSTPONED_AS_OF)

        self.assertEqual(shuffled, first)
        self.assertEqual(first, second)
        self.assertEqual(
            first.resolution_set_fingerprint,
            AFTER_POSTPONED_RESOLUTION_FINGERPRINT,
        )


if __name__ == "__main__":
    unittest.main()
