"""Evaluate canonical schedule games for explicit as-of pregame eligibility."""

from ...baseball.domain.pregame_eligibility import (
    BEFORE_SCHEDULED_START,
    ELIGIBLE,
    INELIGIBLE,
    SCHEDULED_START_REACHED_OR_PASSED,
    SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION,
    SchedulePregameEligibilityDecision,
    SchedulePregameEligibilitySet,
    compute_schedule_pregame_eligibility_set_fingerprint,
)
from ...baseball.domain.schedule_game_materialization import (
    ScheduleBaseballGameMaterializationSet,
)


def evaluate_schedule_pregame_eligibility(
    materialization_set: ScheduleBaseballGameMaterializationSet,
) -> SchedulePregameEligibilitySet:
    """Classify each materialized game using only canonical time values."""

    if not isinstance(
        materialization_set,
        ScheduleBaseballGameMaterializationSet,
    ):
        raise TypeError(
            "materialization_set must be a"
            " ScheduleBaseballGameMaterializationSet"
        )

    eligible_decisions = []
    ineligible_decisions = []
    for materialization in materialization_set.game_materializations:
        if (
            materialization_set.as_of_utc
            < materialization.baseball_game.scheduled_start.value
        ):
            decision = SchedulePregameEligibilityDecision(
                materialization=materialization,
                eligibility_status=ELIGIBLE,
                reason=BEFORE_SCHEDULED_START,
            )
            eligible_decisions.append(decision)
        else:
            decision = SchedulePregameEligibilityDecision(
                materialization=materialization,
                eligibility_status=INELIGIBLE,
                reason=SCHEDULED_START_REACHED_OR_PASSED,
            )
            ineligible_decisions.append(decision)

    eligible_tuple = tuple(eligible_decisions)
    ineligible_tuple = tuple(ineligible_decisions)
    unresolved_candidates = materialization_set.unresolved_candidates
    unavailable_chain_keys = materialization_set.unavailable_chain_keys
    authority_missing_candidates = (
        materialization_set.authority_missing_candidates
    )
    eligible_count = len(eligible_tuple)
    ineligible_count = len(ineligible_tuple)
    unresolved_count = len(unresolved_candidates)
    unavailable_count = len(unavailable_chain_keys)
    authority_missing_count = len(authority_missing_candidates)
    fingerprint = compute_schedule_pregame_eligibility_set_fingerprint(
        as_of_utc=materialization_set.as_of_utc,
        source_materialization_set_fingerprint=(
            materialization_set.materialization_set_fingerprint
        ),
        eligible_count=eligible_count,
        ineligible_count=ineligible_count,
        unresolved_count=unresolved_count,
        unavailable_count=unavailable_count,
        authority_missing_count=authority_missing_count,
        eligible_decisions=eligible_tuple,
        ineligible_decisions=ineligible_tuple,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        authority_missing_candidates=authority_missing_candidates,
    )

    return SchedulePregameEligibilitySet(
        as_of_utc=materialization_set.as_of_utc,
        source_materialization_set_fingerprint=(
            materialization_set.materialization_set_fingerprint
        ),
        eligible_decisions=eligible_tuple,
        ineligible_decisions=ineligible_tuple,
        unresolved_candidates=unresolved_candidates,
        unavailable_chain_keys=unavailable_chain_keys,
        authority_missing_candidates=authority_missing_candidates,
        eligible_count=eligible_count,
        ineligible_count=ineligible_count,
        unresolved_count=unresolved_count,
        unavailable_count=unavailable_count,
        authority_missing_count=authority_missing_count,
        eligibility_set_fingerprint=fingerprint,
        schema_version=(
            SCHEDULE_PREGAME_ELIGIBILITY_SET_SCHEMA_VERSION
        ),
    )
