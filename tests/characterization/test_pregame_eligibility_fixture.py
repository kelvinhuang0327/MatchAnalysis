"""Characterize canonical pregame eligibility across schedule cutoffs."""

from datetime import timedelta
from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from match_analysis.baseball.domain.pregame_eligibility import (
    BEFORE_SCHEDULED_START,
    ELIGIBLE,
    INELIGIBLE,
    SCHEDULED_START_REACHED_OR_PASSED,
)
from tests.characterization.test_participant_identity_resolution_fixture import (
    AFTER_POSTPONED_AS_OF,
    BEFORE_ALL_AS_OF,
    EXACT_BOUNDARY_AS_OF,
    MID_CUTOFF_AS_OF,
    load_fixture_observations,
)
from tests.characterization.test_schedule_game_materialization_fixture import (
    materialize_at,
)
from tests.unit.test_pregame_eligibility_contracts import (
    replace_materialization_as_of,
)


BEFORE_ALL_ELIGIBILITY_FINGERPRINT = (
    "a3da0d3cb58d1f8fc1babcc2749b63a6d6e6faaef0eed1f0176d6f83b705dc56"
)
MID_CUTOFF_ELIGIBILITY_FINGERPRINT = (
    "eb78ce4c4ddfe2bd27350fb627d0c38b0ad6cb82ab34016cc0795aa8577cfc9f"
)
AFTER_POSTPONED_ELIGIBILITY_FINGERPRINT = (
    "2814df0e51fdfb9343d1bc6b73e71ddb33bbd7ab56ee10b749317c8869f2ffc8"
)
EXACT_BOUNDARY_ELIGIBILITY_FINGERPRINT = (
    "e2e8ba0071413b62444f6b8330ca69ace9124c1458161ed04f2cf42e7cade42d"
)


class SchedulePregameEligibilityFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.observations = load_fixture_observations()

    def evaluate_at(self, observations, cutoff):
        _, _, materialization_set = materialize_at(
            observations,
            cutoff,
        )
        return (
            materialization_set,
            evaluate_schedule_pregame_eligibility(materialization_set),
        )

    def test_four_cutoffs_match_pre_edit_references(self) -> None:
        cases = (
            (
                BEFORE_ALL_AS_OF,
                (0, 0, 0, 2, 0),
                BEFORE_ALL_ELIGIBILITY_FINGERPRINT,
            ),
            (
                MID_CUTOFF_AS_OF,
                (2, 0, 0, 0, 0),
                MID_CUTOFF_ELIGIBILITY_FINGERPRINT,
            ),
            (
                AFTER_POSTPONED_AS_OF,
                (0, 2, 0, 0, 0),
                AFTER_POSTPONED_ELIGIBILITY_FINGERPRINT,
            ),
            (
                EXACT_BOUNDARY_AS_OF,
                (1, 0, 0, 1, 0),
                EXACT_BOUNDARY_ELIGIBILITY_FINGERPRINT,
            ),
        )

        for cutoff, expected_counts, expected_fingerprint in cases:
            with self.subTest(cutoff=cutoff):
                _, result = self.evaluate_at(
                    self.observations,
                    cutoff,
                )
                self.assertEqual(
                    (
                        result.eligible_count,
                        result.ineligible_count,
                        result.unresolved_count,
                        result.unavailable_count,
                        result.authority_missing_count,
                    ),
                    expected_counts,
                )
                self.assertEqual(
                    result.eligibility_set_fingerprint,
                    expected_fingerprint,
                )

    def test_exact_start_is_ineligible_and_one_microsecond_before_is_eligible(
        self,
    ) -> None:
        source, _ = self.evaluate_at(
            self.observations,
            MID_CUTOFF_AS_OF,
        )
        target_materialization = min(
            source.game_materializations,
            key=lambda value: value.baseball_game.scheduled_start.value,
        )
        scheduled_start = (
            target_materialization.baseball_game.scheduled_start.value
        )

        exact = evaluate_schedule_pregame_eligibility(
            replace_materialization_as_of(source, scheduled_start)
        )
        before = evaluate_schedule_pregame_eligibility(
            replace_materialization_as_of(
                source,
                scheduled_start - timedelta(microseconds=1),
            )
        )

        exact_decision = next(
            decision
            for decision in exact.ineligible_decisions
            if decision.materialization.source_observation_id
            == target_materialization.source_observation_id
        )
        self.assertEqual(
            (
                exact_decision.eligibility_status,
                exact_decision.reason,
            ),
            (
                INELIGIBLE,
                SCHEDULED_START_REACHED_OR_PASSED,
            ),
        )
        before_decision = next(
            decision
            for decision in before.eligible_decisions
            if decision.materialization.source_observation_id
            == target_materialization.source_observation_id
        )
        self.assertEqual(
            (
                before_decision.eligibility_status,
                before_decision.reason,
            ),
            (ELIGIBLE, BEFORE_SCHEDULED_START),
        )

    def test_p12_materializations_and_all_provenance_are_unchanged(
        self,
    ) -> None:
        source, result = self.evaluate_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )

        self.assertEqual(
            result.source_materialization_set_fingerprint,
            source.materialization_set_fingerprint,
        )
        self.assertEqual(result.as_of_utc, source.as_of_utc)
        self.assertIs(
            result.unresolved_candidates,
            source.unresolved_candidates,
        )
        self.assertIs(
            result.unavailable_chain_keys,
            source.unavailable_chain_keys,
        )
        self.assertIs(
            result.authority_missing_candidates,
            source.authority_missing_candidates,
        )
        self.assertEqual(
            tuple(
                decision.materialization
                for decision in result.ineligible_decisions
            ),
            source.game_materializations,
        )
        for decision, materialization in zip(
            result.ineligible_decisions,
            source.game_materializations,
            strict=True,
        ):
            with self.subTest(
                source_observation_id=(
                    materialization.source_observation_id
                )
            ):
                self.assertIs(
                    decision.materialization,
                    materialization,
                )
                self.assertIs(
                    decision.materialization.baseball_game,
                    materialization.baseball_game,
                )
                self.assertIs(
                    decision.materialization.match_identity,
                    materialization.match_identity,
                )

    def test_shuffled_and_repeated_inputs_have_identical_fingerprints(
        self,
    ) -> None:
        shuffled = list(self.observations)
        random.Random(2026).shuffle(shuffled)

        _, ordered_result = self.evaluate_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )
        _, shuffled_result = self.evaluate_at(
            tuple(shuffled),
            AFTER_POSTPONED_AS_OF,
        )
        _, repeated_result = self.evaluate_at(
            self.observations,
            AFTER_POSTPONED_AS_OF,
        )

        self.assertEqual(ordered_result, shuffled_result)
        self.assertEqual(ordered_result, repeated_result)
        self.assertEqual(
            ordered_result.eligibility_set_fingerprint,
            AFTER_POSTPONED_ELIGIBILITY_FINGERPRINT,
        )


if __name__ == "__main__":
    unittest.main()
