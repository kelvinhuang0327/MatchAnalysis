"""Unit tests for canonical schedule pregame eligibility."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from match_analysis.application.use_cases.materialize_schedule_baseball_games import (
    materialize_schedule_baseball_games,
)
from match_analysis.baseball.domain.pregame_eligibility import (
    BEFORE_SCHEDULED_START,
    ELIGIBLE,
    INELIGIBLE,
    SCHEDULED_START_REACHED_OR_PASSED,
    SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION,
    SchedulePregameEligibilityDecision,
    SchedulePregameEligibilitySet,
    compute_schedule_pregame_eligibility_set_fingerprint,
)
from match_analysis.baseball.domain.schedule_game_materialization import (
    SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION,
    ScheduleBaseballGameMaterializationSet,
    compute_schedule_baseball_game_materialization_set_fingerprint,
)
from tests.unit.test_construct_match_identities import (
    authority_entry,
    make_resolved_candidate,
    make_resolution_set,
    make_unresolved_candidate,
)
from tests.unit.test_schedule_game_materialization_contracts import (
    construction_for,
)


def materialization_set_for(
    *,
    resolved_candidates=(),
    unresolved_candidates=(),
    unavailable_chain_keys=(),
    authority_entries=(),
) -> ScheduleBaseballGameMaterializationSet:
    resolution = make_resolution_set(
        resolved_candidates=resolved_candidates,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
    )
    construction = construction_for(resolution, authority_entries)
    return materialize_schedule_baseball_games(
        construction,
        resolution,
    )


def replace_materialization_as_of(
    original: ScheduleBaseballGameMaterializationSet,
    as_of_utc,
) -> ScheduleBaseballGameMaterializationSet:
    fingerprint = (
        compute_schedule_baseball_game_materialization_set_fingerprint(
            as_of_utc=as_of_utc,
            source_resolution_set_fingerprint=(
                original.source_resolution_set_fingerprint
            ),
            authority_catalog_fingerprint=(
                original.authority_catalog_fingerprint
            ),
            source_construction_set_fingerprint=(
                original.source_construction_set_fingerprint
            ),
            materialized_count=original.materialized_count,
            unresolved_count=original.unresolved_count,
            unavailable_count=original.unavailable_count,
            authority_missing_count=original.authority_missing_count,
            game_materializations=original.game_materializations,
            unresolved_candidates=original.unresolved_candidates,
            unavailable_chain_keys=original.unavailable_chain_keys,
            authority_missing_candidates=(
                original.authority_missing_candidates
            ),
        )
    )
    return ScheduleBaseballGameMaterializationSet(
        as_of_utc=as_of_utc,
        source_resolution_set_fingerprint=(
            original.source_resolution_set_fingerprint
        ),
        authority_catalog_fingerprint=(
            original.authority_catalog_fingerprint
        ),
        source_construction_set_fingerprint=(
            original.source_construction_set_fingerprint
        ),
        game_materializations=original.game_materializations,
        unresolved_candidates=original.unresolved_candidates,
        unavailable_chain_keys=original.unavailable_chain_keys,
        authority_missing_candidates=(
            original.authority_missing_candidates
        ),
        materialized_count=original.materialized_count,
        unresolved_count=original.unresolved_count,
        unavailable_count=original.unavailable_count,
        authority_missing_count=original.authority_missing_count,
        materialization_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_BASEBALL_GAME_MATERIALIZATION_SET_SCHEMA_VERSION
        ),
    )


def one_game_materialization_set():
    resolved = make_resolved_candidate()
    return materialization_set_for(
        resolved_candidates=(resolved,),
        authority_entries=(
            authority_entry(
                resolved.provider_game_id,
                resolved.game_number,
                "EXPLICIT_CANONICAL_GAME",
            ),
        ),
    )


class SchedulePregameEligibilityTests(unittest.TestCase):
    def test_only_explicit_canonical_time_values_determine_eligibility(
        self,
    ) -> None:
        source = one_game_materialization_set()
        scheduled_start = (
            source.game_materializations[0]
            .baseball_game.scheduled_start.value
        )

        before = evaluate_schedule_pregame_eligibility(
            replace_materialization_as_of(
                source,
                scheduled_start - timedelta(microseconds=1),
            )
        )
        exact = evaluate_schedule_pregame_eligibility(
            replace_materialization_as_of(source, scheduled_start)
        )
        after = evaluate_schedule_pregame_eligibility(
            replace_materialization_as_of(
                source,
                scheduled_start + timedelta(microseconds=1),
            )
        )

        self.assertEqual(
            (
                before.eligible_count,
                before.ineligible_count,
                before.eligible_decisions[0].eligibility_status,
                before.eligible_decisions[0].reason,
            ),
            (1, 0, ELIGIBLE, BEFORE_SCHEDULED_START),
        )
        for result in (exact, after):
            with self.subTest(as_of=result.as_of_utc):
                self.assertEqual(
                    (
                        result.eligible_count,
                        result.ineligible_count,
                        result.ineligible_decisions[
                            0
                        ].eligibility_status,
                        result.ineligible_decisions[0].reason,
                    ),
                    (
                        0,
                        1,
                        INELIGIBLE,
                        SCHEDULED_START_REACHED_OR_PASSED,
                    ),
                )

    def test_decision_preserves_exact_p12_materialization_and_provenance(
        self,
    ) -> None:
        source = one_game_materialization_set()
        materialization = source.game_materializations[0]

        result = evaluate_schedule_pregame_eligibility(source)
        decision = result.eligible_decisions[0]

        self.assertIs(decision.materialization, materialization)
        self.assertIs(
            decision.materialization.baseball_game,
            materialization.baseball_game,
        )
        self.assertIs(
            decision.materialization.match_identity,
            materialization.match_identity,
        )
        self.assertEqual(
            decision.materialization,
            materialization,
        )
        self.assertEqual(
            result.source_materialization_set_fingerprint,
            source.materialization_set_fingerprint,
        )

    def test_non_materialized_partitions_remain_unchanged_and_undecided(
        self,
    ) -> None:
        resolved = make_resolved_candidate()
        unresolved = make_unresolved_candidate()
        unavailable = ("MLB_STATS_API", "unavailable-provider-game", 3)
        source = materialization_set_for(
            resolved_candidates=(resolved,),
            unresolved_candidates=(unresolved,),
            unavailable_chain_keys=(unavailable,),
            authority_entries=(),
        )

        result = evaluate_schedule_pregame_eligibility(source)

        self.assertEqual(result.eligible_decisions, ())
        self.assertEqual(result.ineligible_decisions, ())
        self.assertEqual(
            (
                result.eligible_count,
                result.ineligible_count,
                result.unresolved_count,
                result.unavailable_count,
                result.authority_missing_count,
            ),
            (0, 0, 1, 1, 1),
        )
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

    def test_contracts_are_exact_immutable_and_fingerprint_bound(
        self,
    ) -> None:
        result = evaluate_schedule_pregame_eligibility(
            one_game_materialization_set()
        )

        self.assertEqual(
            set(SchedulePregameEligibilityDecision.__dataclass_fields__),
            {
                "materialization",
                "eligibility_status",
                "reason",
            },
        )
        self.assertEqual(
            set(SchedulePregameEligibilitySet.__dataclass_fields__),
            {
                "as_of_utc",
                "source_materialization_set_fingerprint",
                "eligible_decisions",
                "ineligible_decisions",
                "unresolved_candidates",
                "unavailable_chain_keys",
                "authority_missing_candidates",
                "eligible_count",
                "ineligible_count",
                "unresolved_count",
                "unavailable_count",
                "authority_missing_count",
                "eligibility_set_fingerprint",
                "schema_version",
            },
        )
        self.assertEqual(
            result.schema_version,
            SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION,
        )
        with self.assertRaises(FrozenInstanceError):
            result.eligible_count = 0
        with self.assertRaises(ValueError):
            replace(result, eligibility_set_fingerprint="0" * 64)
        with self.assertRaises(ValueError):
            replace(result, eligible_count=0)

    def test_decision_rejects_uncontrolled_status_reason_pairs(
        self,
    ) -> None:
        materialization = (
            one_game_materialization_set().game_materializations[0]
        )

        for status, reason in (
            (ELIGIBLE, SCHEDULED_START_REACHED_OR_PASSED),
            (INELIGIBLE, BEFORE_SCHEDULED_START),
            ("UNKNOWN", BEFORE_SCHEDULED_START),
            (ELIGIBLE, "UNKNOWN"),
        ):
            with self.subTest(status=status, reason=reason):
                with self.assertRaises(ValueError):
                    SchedulePregameEligibilityDecision(
                        materialization=materialization,
                        eligibility_status=status,
                        reason=reason,
                    )

    def test_set_rejects_decision_that_conflicts_with_canonical_time(
        self,
    ) -> None:
        source = one_game_materialization_set()
        eligible = evaluate_schedule_pregame_eligibility(
            source
        ).eligible_decisions
        as_of_utc = (
            source.game_materializations[0]
            .baseball_game.scheduled_start.value
        )
        fingerprint = (
            compute_schedule_pregame_eligibility_set_fingerprint(
                as_of_utc=as_of_utc,
                source_materialization_set_fingerprint=(
                    source.materialization_set_fingerprint
                ),
                eligible_count=1,
                ineligible_count=0,
                unresolved_count=0,
                unavailable_count=0,
                authority_missing_count=0,
                eligible_decisions=eligible,
                ineligible_decisions=(),
                unresolved_candidates=(),
                unavailable_chain_keys=(),
                authority_missing_candidates=(),
            )
        )

        with self.assertRaises(ValueError):
            SchedulePregameEligibilitySet(
                as_of_utc=as_of_utc,
                source_materialization_set_fingerprint=(
                    source.materialization_set_fingerprint
                ),
                eligible_decisions=eligible,
                ineligible_decisions=(),
                unresolved_candidates=(),
                unavailable_chain_keys=(),
                authority_missing_candidates=(),
                eligible_count=1,
                ineligible_count=0,
                unresolved_count=0,
                unavailable_count=0,
                authority_missing_count=0,
                eligibility_set_fingerprint=fingerprint,
                schema_version=(
                    SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION
                ),
            )

    def test_repeated_inputs_produce_identical_result(self) -> None:
        source = one_game_materialization_set()

        first = evaluate_schedule_pregame_eligibility(source)
        second = evaluate_schedule_pregame_eligibility(source)

        self.assertEqual(first, second)
        self.assertEqual(
            first.eligibility_set_fingerprint,
            second.eligibility_set_fingerprint,
        )

    def test_non_contract_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            evaluate_schedule_pregame_eligibility(object())


if __name__ == "__main__":
    unittest.main()
