"""Unit tests for schedule observation revision chains."""

import copy
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
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
from match_analysis.baseball.domain.schedule_revision import (
    ScheduleObservationRevisionChain,
    ScheduleObservationRevisionSet,
)


class StubScheduleObservationSource:
    def __init__(self, capture: ScheduleObservationCapture) -> None:
        self._capture = capture

    def capture(self) -> ScheduleObservationCapture:
        return self._capture


_BASE_TIME = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_observation(
    *,
    provider_namespace: str = "MLB_STATS_API",
    provider_game_id: str = "777001",
    game_number: int = 1,
    response_received_at_utc: datetime | None = None,
    provider_status_code: str = "S",
    provider_detailed_status: str = "Scheduled",
    payload_tag: str = "opening",
):
    """Build a standalone, validly hash-chained root observation.

    Every observation this helper returns is captured as its own root
    (``supersedes_observation_id=None``); tests that need an explicit
    revision relationship either capture a real successor against a real
    predecessor (``make_revision``) or rewrite the frozen field directly to
    simulate an already-existing, independently-sourced observation whose
    declared supersession this consumer must validate on its own.
    """

    response_received_at_utc = response_received_at_utc or _BASE_TIME
    ingested_at_utc = response_received_at_utc + timedelta(seconds=1)
    raw_payload_bytes = (
        f'{{"tag":"{payload_tag}","game_id":"{provider_game_id}",'
        f'"game_number":{game_number}}}'
    ).encode("utf-8")
    capture = ScheduleObservationCapture(
        provider_namespace=provider_namespace,
        provider_game_id=provider_game_id,
        scheduled_start_utc=_BASE_TIME + timedelta(days=1),
        official_local_date=date(2026, 4, 4),
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=provider_status_code,
        provider_detailed_status=provider_detailed_status,
        game_number=game_number,
        home_provider_participant_id="118",
        away_provider_participant_id="109",
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_schedule_parser_v1",
        schema_version="schedule_source_observation_v1",
        raw_payload_bytes=raw_payload_bytes,
        raw_payload_sha256=sha256(raw_payload_bytes).hexdigest(),
        supersedes_observation_id=None,
    )
    return capture_schedule_observation(StubScheduleObservationSource(capture))


def make_revision(previous, **overrides):
    """Capture a genuine, validly hash-chained successor to ``previous``."""

    values = {
        "provider_namespace": previous.provider_namespace,
        "provider_game_id": previous.provider_game_id,
        "game_number": previous.game_number,
        "response_received_at_utc": previous.response_received_at_utc
        + timedelta(hours=1),
        "provider_status_code": "P",
        "provider_detailed_status": "Postponed",
        "payload_tag": "revision",
    }
    values.update(overrides)
    response_received_at_utc = values["response_received_at_utc"]
    ingested_at_utc = response_received_at_utc + timedelta(seconds=1)
    raw_payload_bytes = (
        f'{{"tag":"{values["payload_tag"]}",'
        f'"game_id":"{values["provider_game_id"]}",'
        f'"game_number":{values["game_number"]}}}'
    ).encode("utf-8")
    capture = ScheduleObservationCapture(
        provider_namespace=values["provider_namespace"],
        provider_game_id=values["provider_game_id"],
        scheduled_start_utc=_BASE_TIME + timedelta(days=1),
        official_local_date=date(2026, 4, 5),
        response_received_at_utc=response_received_at_utc,
        ingested_at_utc=ingested_at_utc,
        provider_status_code=values["provider_status_code"],
        provider_detailed_status=values["provider_detailed_status"],
        game_number=values["game_number"],
        home_provider_participant_id="118",
        away_provider_participant_id="109",
        endpoint_id="mlb_schedule_v1",
        parser_version="matchanalysis_schedule_parser_v1",
        schema_version="schedule_source_observation_v1",
        raw_payload_bytes=raw_payload_bytes,
        raw_payload_sha256=sha256(raw_payload_bytes).hexdigest(),
        supersedes_observation_id=previous.observation_id,
    )
    return capture_schedule_observation(
        StubScheduleObservationSource(capture), previous
    )


def rewire_supersedes(observation, supersedes_observation_id):
    """Simulate an already-existing observation with a declared, unvetted edge.

    Bypasses the frozen dataclass's own constructor validation (which would
    otherwise recompute and check the hash-derived ``observation_id``) so
    tests can hand the use case an adversarial or malformed edge exactly as
    it would arrive from an untrusted upstream collection, independent of
    the domain object's own capture-time invariants (covered separately in
    ``test_schedule_observation_contracts.py``).
    """

    mutated = copy.deepcopy(observation)
    object.__setattr__(
        mutated, "supersedes_observation_id", supersedes_observation_id
    )
    return mutated


class BuildScheduleObservationRevisionChainsTests(unittest.TestCase):
    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains([])

    def test_singleton_chain(self) -> None:
        opening = make_observation()

        revision_set = build_schedule_observation_revision_chains([opening])

        self.assertEqual(len(revision_set.chains), 1)
        chain = revision_set.chains[0]
        self.assertEqual(chain.observation_count, 1)
        self.assertEqual(chain.root_observation_id, opening.observation_id)
        self.assertEqual(chain.head_observation_id, opening.observation_id)
        self.assertEqual(revision_set.unique_observation_count, 1)
        self.assertEqual(revision_set.idempotent_duplicate_count, 0)

    def test_valid_root_plus_revision(self) -> None:
        opening = make_observation()
        postponed = make_revision(opening)

        revision_set = build_schedule_observation_revision_chains(
            [opening, postponed]
        )

        self.assertEqual(len(revision_set.chains), 1)
        chain = revision_set.chains[0]
        self.assertEqual(chain.observation_count, 2)
        self.assertEqual(chain.root_observation_id, opening.observation_id)
        self.assertEqual(chain.head_observation_id, postponed.observation_id)
        self.assertEqual(
            tuple(obs.observation_id for obs in chain.observations),
            (opening.observation_id, postponed.observation_id),
        )

    def test_multiple_provider_games_partition_independently(self) -> None:
        game_one = make_observation(provider_game_id="777001", game_number=1)
        game_other_provider = make_observation(
            provider_namespace="OTHER_PROVIDER",
            provider_game_id="777001",
            game_number=1,
            payload_tag="other-provider",
        )

        revision_set = build_schedule_observation_revision_chains(
            [game_one, game_other_provider]
        )

        self.assertEqual(len(revision_set.chains), 2)
        self.assertEqual(
            {chain.provider_namespace for chain in revision_set.chains},
            {"MLB_STATS_API", "OTHER_PROVIDER"},
        )

    def test_doubleheader_partitions_by_game_number_alone(self) -> None:
        game_one = make_observation(
            provider_game_id="777001", game_number=1, payload_tag="game-1"
        )
        game_two = make_observation(
            provider_game_id="777001", game_number=2, payload_tag="game-2"
        )

        revision_set = build_schedule_observation_revision_chains(
            [game_one, game_two]
        )

        self.assertEqual(len(revision_set.chains), 2)
        self.assertEqual(
            {chain.game_number for chain in revision_set.chains}, {1, 2}
        )
        for chain in revision_set.chains:
            self.assertEqual(chain.observation_count, 1)

    def test_shuffled_input_produces_identical_fingerprint(self) -> None:
        opening = make_observation(provider_game_id="777001", game_number=1)
        postponed = make_revision(opening)
        game_two = make_observation(
            provider_game_id="777002", game_number=2, payload_tag="game-2"
        )
        observations = [opening, postponed, game_two]

        first = build_schedule_observation_revision_chains(observations)

        shuffled = observations[:]
        random.Random(99).shuffle(shuffled)
        second = build_schedule_observation_revision_chains(shuffled)

        self.assertEqual(
            first.revision_set_fingerprint, second.revision_set_fingerprint
        )
        self.assertEqual(
            tuple(chain.root_observation_id for chain in first.chains),
            tuple(chain.root_observation_id for chain in second.chains),
        )

    def test_exact_duplicate_observations_are_idempotent(self) -> None:
        opening = make_observation()

        revision_set = build_schedule_observation_revision_chains(
            [opening, opening, opening]
        )

        self.assertEqual(revision_set.unique_observation_count, 1)
        self.assertEqual(revision_set.idempotent_duplicate_count, 2)
        self.assertEqual(len(revision_set.chains), 1)
        self.assertEqual(revision_set.chains[0].observation_count, 1)

    def test_same_id_unequal_observations_are_rejected(self) -> None:
        opening = make_observation()
        conflicting = copy.deepcopy(opening)
        object.__setattr__(
            conflicting, "provider_detailed_status", "Something-Else"
        )
        self.assertEqual(conflicting.observation_id, opening.observation_id)
        self.assertNotEqual(conflicting, opening)

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains([opening, conflicting])

    def test_orphan_revision_is_rejected(self) -> None:
        opening = make_observation()
        orphan = rewire_supersedes(make_observation(payload_tag="orphan"), "0" * 64)

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains([opening, orphan])

    def test_fork_is_rejected(self) -> None:
        opening = make_observation()
        branch_a = rewire_supersedes(
            make_observation(payload_tag="branch-a"), opening.observation_id
        )
        branch_b = rewire_supersedes(
            make_observation(payload_tag="branch-b"), opening.observation_id
        )

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains(
                [opening, branch_a, branch_b]
            )

    def test_multiple_roots_are_rejected(self) -> None:
        root_a = make_observation(payload_tag="root-a")
        root_b = make_observation(payload_tag="root-b")

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains([root_a, root_b])

    def test_cross_provider_supersession_is_rejected(self) -> None:
        other_provider_root = make_observation(
            provider_namespace="OTHER_PROVIDER", payload_tag="other-root"
        )
        opening = make_observation()
        forged_revision = rewire_supersedes(
            make_observation(payload_tag="forged"),
            other_provider_root.observation_id,
        )

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains(
                [other_provider_root, opening, forged_revision]
            )

    def test_cross_game_number_supersession_is_rejected(self) -> None:
        game_two_root = make_observation(
            game_number=2, payload_tag="game-two-root"
        )
        opening = make_observation(game_number=1, payload_tag="game-one-root")
        forged_revision = rewire_supersedes(
            make_observation(game_number=1, payload_tag="forged"),
            game_two_root.observation_id,
        )

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains(
                [game_two_root, opening, forged_revision]
            )

    def test_non_increasing_response_time_is_rejected(self) -> None:
        opening = make_observation()
        backdated_revision = rewire_supersedes(
            make_observation(
                response_received_at_utc=opening.response_received_at_utc
                - timedelta(seconds=1),
                payload_tag="backdated",
            ),
            opening.observation_id,
        )

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains(
                [opening, backdated_revision]
            )

    def test_disconnected_partition_with_no_reachable_root_is_rejected(
        self,
    ) -> None:
        first = make_observation(payload_tag="cycle-a")
        second = make_observation(payload_tag="cycle-b")
        first = rewire_supersedes(first, second.observation_id)
        second = rewire_supersedes(second, first.observation_id)

        with self.assertRaises(ValueError):
            build_schedule_observation_revision_chains([first, second])

    def test_deterministic_fingerprint_matches_pre_edit_reference(self) -> None:
        opening = make_observation(provider_game_id="777001", game_number=1)
        postponed = make_revision(opening)
        game_two = make_observation(
            provider_game_id="777002", game_number=2, payload_tag="game-2"
        )

        first = build_schedule_observation_revision_chains(
            [opening, postponed, game_two]
        )
        second = build_schedule_observation_revision_chains(
            [opening, postponed, game_two]
        )

        self.assertEqual(
            first.revision_set_fingerprint, second.revision_set_fingerprint
        )

    def test_changed_ordered_ids_change_the_fingerprint(self) -> None:
        opening = make_observation()
        first = build_schedule_observation_revision_chains([opening])

        different_opening = make_observation(payload_tag="different")
        second = build_schedule_observation_revision_chains(
            [different_opening]
        )

        self.assertNotEqual(
            first.revision_set_fingerprint, second.revision_set_fingerprint
        )

    def test_chain_and_set_are_immutable(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        with self.assertRaises(FrozenInstanceError):
            revision_set.unique_observation_count = 99
        with self.assertRaises(FrozenInstanceError):
            revision_set.chains[0].observation_count = 99

    def test_schema_version_is_exact(self) -> None:
        opening = make_observation()
        revision_set = build_schedule_observation_revision_chains([opening])

        self.assertEqual(
            revision_set.schema_version,
            "schedule_observation_revision_set_v1",
        )

    def test_chain_rejects_construction_with_mismatched_root(self) -> None:
        opening = make_observation()

        with self.assertRaises(ValueError):
            ScheduleObservationRevisionChain(
                provider_namespace=opening.provider_namespace,
                provider_game_id=opening.provider_game_id,
                game_number=opening.game_number,
                observations=(opening,),
                root_observation_id="0" * 64,
                head_observation_id=opening.observation_id,
                observation_count=1,
            )

    def test_revision_set_rejects_unsorted_chains(self) -> None:
        game_one = make_observation(provider_game_id="777001", game_number=1)
        game_two = make_observation(
            provider_game_id="777002", game_number=2, payload_tag="game-2"
        )
        ordered_set = build_schedule_observation_revision_chains(
            [game_one, game_two]
        )
        reversed_chains = tuple(reversed(ordered_set.chains))

        with self.assertRaises(ValueError):
            ScheduleObservationRevisionSet(
                chains=reversed_chains,
                unique_observation_count=ordered_set.unique_observation_count,
                idempotent_duplicate_count=(
                    ordered_set.idempotent_duplicate_count
                ),
                revision_set_fingerprint=ordered_set.revision_set_fingerprint,
                schema_version=ordered_set.schema_version,
            )


if __name__ == "__main__":
    unittest.main()
