"""Unit tests for exact schedule-candidate resolution and prediction admission."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.evaluate_schedule_pregame_eligibility import (
    evaluate_schedule_pregame_eligibility,
)
from match_analysis.baseball.domain.prediction_admission import (
    ADMITTED,
    AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
    EXACT_IDENTITY_MISMATCH,
    INVALID_CANONICAL_UTC,
    INVALID_PREDICTION_TIMESTAMP_ORDER,
    MISSING_REQUIRED_PREDICTION_EVIDENCE,
    MISSING_SCHEDULE_CANDIDATE_MATCH,
    PREDICTION_NOT_BEFORE_SCHEDULED_START,
    REJECTED,
    SCHEDULE_NOT_PREGAME_ELIGIBLE,
    SCHEDULE_OBSERVATION_ID_MISMATCH,
    PredictionAdmissionResult,
    ProspectivePredictionCandidate,
    ScheduleCandidateProjection,
    admit_prospective_prediction,
    resolve_exact_schedule_candidate,
)
from match_analysis.baseball.domain.prediction_source_observation import (
    compute_prediction_observation_id,
)
from match_analysis.baseball.domain.schedule_game_materialization import (
    compute_schedule_baseball_game_materialization_set_fingerprint,
)
from tests.unit.test_construct_match_identities import make_resolved_candidate
from tests.unit.test_pregame_eligibility_contracts import (
    one_game_materialization_set,
)


def eligibility_set_for_one_game():
    materialization_set = one_game_materialization_set()
    return materialization_set, evaluate_schedule_pregame_eligibility(
        materialization_set
    )


def schedule_candidate_from(materialization_set, eligibility_set):
    materialization = materialization_set.game_materializations[0]
    resolved = make_resolved_candidate()
    return ScheduleCandidateProjection(
        provider_namespace=resolved.provider_namespace,
        provider_game_id=resolved.provider_game_id,
        game_number=resolved.game_number,
        source_schedule_observation_id=materialization.source_observation_id,
        schedule_as_of_utc=materialization_set.as_of_utc,
        scheduled_start_utc=(
            materialization.baseball_game.scheduled_start.value
        ),
    )


def evidence_fields(candidate: ScheduleCandidateProjection) -> dict:
    generated = candidate.scheduled_start_utc - timedelta(days=30)
    received = generated
    ingested = generated
    return {
        "source_prediction_id": "prediction-0001",
        "model_id": "model-v1",
        "market_id": "MONEYLINE",
        "selection": "home",
        "model_probability": Decimal("0.6321"),
        "line_value": Decimal("-1.5"),
        "push_policy": "NO_PUSH",
        "provider_namespace": candidate.provider_namespace,
        "provider_game_id": candidate.provider_game_id,
        "game_number": candidate.game_number,
        "source_schedule_observation_id": (
            candidate.source_schedule_observation_id
        ),
        "prediction_generated_at_utc": generated,
        "response_received_at_utc": received,
        "ingested_at_utc": ingested,
        "scheduled_start_utc": candidate.scheduled_start_utc,
    }


def raw_candidate_for(candidate: ScheduleCandidateProjection, **overrides):
    evidence = evidence_fields(candidate)
    evidence.update(overrides)
    scheduled_start_utc = evidence.pop("scheduled_start_utc")
    prediction_observation_id = compute_prediction_observation_id(
        scheduled_start_utc=scheduled_start_utc, **evidence
    )
    raw = {
        key: (
            value.isoformat()
            if key.endswith("_utc") and isinstance(value, datetime)
            else value
        )
        for key, value in evidence.items()
    }
    raw["prediction_observation_id"] = prediction_observation_id
    return ProspectivePredictionCandidate(**raw)


class ExactScheduleCandidateResolutionTests(unittest.TestCase):
    def test_zero_matches_is_missing_schedule_candidate_match(self) -> None:
        _, resolved_reason = resolve_exact_schedule_candidate(
            provider_namespace="MLB_STATS_API",
            provider_game_id="does-not-exist",
            game_number=1,
            schedule_candidates=(),
        )
        self.assertEqual(resolved_reason, MISSING_SCHEDULE_CANDIDATE_MATCH)

    def test_exact_unique_match_resolves(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)

        resolved, reason = resolve_exact_schedule_candidate(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number,
            schedule_candidates=(candidate,),
        )
        self.assertIsNone(reason)
        self.assertIs(resolved, candidate)

    def test_multiple_matches_is_ambiguous(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        duplicate = ScheduleCandidateProjection(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number,
            source_schedule_observation_id=(
                candidate.source_schedule_observation_id
            ),
            schedule_as_of_utc=candidate.schedule_as_of_utc,
            scheduled_start_utc=candidate.scheduled_start_utc,
        )

        _, reason = resolve_exact_schedule_candidate(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number,
            schedule_candidates=(candidate, duplicate),
        )
        self.assertEqual(reason, AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH)

    def test_same_provider_game_with_different_game_number_does_not_match(
        self,
    ) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)

        _, reason = resolve_exact_schedule_candidate(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number + 1,
            schedule_candidates=(candidate,),
        )
        self.assertEqual(reason, MISSING_SCHEDULE_CANDIDATE_MATCH)

    def test_resolution_never_matches_by_date_or_row_order(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        unrelated = ScheduleCandidateProjection(
            provider_namespace="OTHER_PROVIDER",
            provider_game_id="unrelated-game",
            game_number=1,
            source_schedule_observation_id=sha256(
                b"unrelated-observation"
            ).hexdigest(),
            schedule_as_of_utc=candidate.schedule_as_of_utc,
            scheduled_start_utc=candidate.scheduled_start_utc,
        )

        resolved, reason = resolve_exact_schedule_candidate(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number,
            schedule_candidates=(unrelated, candidate),
        )
        self.assertIsNone(reason)
        self.assertIs(resolved, candidate)


class PredictionAdmissionEvaluatorTests(unittest.TestCase):
    def test_one_fully_explicit_synthetic_record_is_admitted(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate)

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )

        self.assertIsInstance(result, PredictionAdmissionResult)
        self.assertEqual(result.admission_status, ADMITTED)
        self.assertIsNone(result.reason)
        self.assertEqual(
            result.observation.provider_namespace, candidate.provider_namespace
        )
        self.assertEqual(result.observation.game_number, candidate.game_number)
        self.assertEqual(
            result.observation.scheduled_start_utc, candidate.scheduled_start_utc
        )

    def test_repeated_admission_is_deterministic(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate)

        first = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        second = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(first, second)

    def test_missing_required_prediction_evidence(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate, source_prediction_id="")

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.admission_status, REJECTED)
        self.assertEqual(result.reason, MISSING_REQUIRED_PREDICTION_EVIDENCE)

    def test_naive_malformed_and_whitespace_timestamps_are_invalid_canonical_utc(
        self,
    ) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)

        for bad_value in (
            "2031-06-01T00:00:00",
            "not-a-timestamp",
            " 2031-06-01T00:00:00Z",
            "2031-06-01T00:00:00Z ",
        ):
            with self.subTest(value=bad_value):
                evidence = evidence_fields(candidate)
                scheduled_start_utc = evidence.pop("scheduled_start_utc")
                evidence["prediction_generated_at_utc"] = bad_value
                prediction_observation_id = "0" * 64
                raw = ProspectivePredictionCandidate(
                    prediction_observation_id=prediction_observation_id,
                    source_prediction_id=evidence["source_prediction_id"],
                    model_id=evidence["model_id"],
                    market_id=evidence["market_id"],
                    selection=evidence["selection"],
                    model_probability=evidence["model_probability"],
                    line_value=evidence["line_value"],
                    push_policy=evidence["push_policy"],
                    provider_namespace=evidence["provider_namespace"],
                    provider_game_id=evidence["provider_game_id"],
                    game_number=evidence["game_number"],
                    source_schedule_observation_id=(
                        evidence["source_schedule_observation_id"]
                    ),
                    prediction_generated_at_utc=bad_value,
                    response_received_at_utc=(
                        evidence["response_received_at_utc"].isoformat()
                    ),
                    ingested_at_utc=evidence["ingested_at_utc"].isoformat(),
                )

                result = admit_prospective_prediction(
                    raw,
                    schedule_candidates=(candidate,),
                    schedule_pregame_eligibility=eligibility_set,
                )
                self.assertEqual(result.admission_status, REJECTED)
                self.assertEqual(result.reason, INVALID_CANONICAL_UTC)

    def test_explicit_offset_is_accepted_and_normalized(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        evidence = evidence_fields(candidate)
        offset_moment = evidence["prediction_generated_at_utc"].astimezone(
            timezone(timedelta(hours=2))
        )
        raw = raw_candidate_for(
            candidate,
            prediction_generated_at_utc=evidence["prediction_generated_at_utc"],
        )
        # Rebuild using an explicit +02:00 string for the same instant.
        raw = ProspectivePredictionCandidate(
            **{
                **{
                    field: getattr(raw, field)
                    for field in raw.__dataclass_fields__
                },
                "prediction_generated_at_utc": offset_moment.isoformat(),
            }
        )

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.admission_status, ADMITTED)

    def test_timestamp_order_boundaries(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        scheduled_start = candidate.scheduled_start_utc
        earlier = scheduled_start - timedelta(days=30)
        later = scheduled_start - timedelta(days=29)

        with self.subTest(case="generated_after_received"):
            raw = raw_candidate_for(
                candidate,
                prediction_generated_at_utc=later,
                response_received_at_utc=earlier,
                ingested_at_utc=earlier,
            )
            result = admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=eligibility_set,
            )
            self.assertEqual(result.reason, INVALID_PREDICTION_TIMESTAMP_ORDER)

        with self.subTest(case="received_after_ingested"):
            raw = raw_candidate_for(
                candidate,
                prediction_generated_at_utc=earlier,
                response_received_at_utc=later,
                ingested_at_utc=earlier,
            )
            result = admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=eligibility_set,
            )
            self.assertEqual(result.reason, INVALID_PREDICTION_TIMESTAMP_ORDER)

        with self.subTest(case="ingested_equals_scheduled_start"):
            raw = raw_candidate_for(
                candidate,
                prediction_generated_at_utc=earlier,
                response_received_at_utc=earlier,
                ingested_at_utc=scheduled_start,
            )
            result = admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=eligibility_set,
            )
            self.assertEqual(
                result.reason, PREDICTION_NOT_BEFORE_SCHEDULED_START
            )

        with self.subTest(case="ingested_after_scheduled_start"):
            raw = raw_candidate_for(
                candidate,
                prediction_generated_at_utc=earlier,
                response_received_at_utc=earlier,
                ingested_at_utc=scheduled_start + timedelta(days=1),
            )
            result = admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=eligibility_set,
            )
            self.assertEqual(
                result.reason, PREDICTION_NOT_BEFORE_SCHEDULED_START
            )

        with self.subTest(case="generated_equals_received_equals_ingested"):
            raw = raw_candidate_for(
                candidate,
                prediction_generated_at_utc=earlier,
                response_received_at_utc=earlier,
                ingested_at_utc=earlier,
            )
            result = admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=eligibility_set,
            )
            self.assertEqual(result.admission_status, ADMITTED)

    def test_zero_schedule_candidates_is_missing_match(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate)

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.reason, MISSING_SCHEDULE_CANDIDATE_MATCH)

    def test_multiple_schedule_candidates_is_ambiguous(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        duplicate = ScheduleCandidateProjection(
            provider_namespace=candidate.provider_namespace,
            provider_game_id=candidate.provider_game_id,
            game_number=candidate.game_number,
            source_schedule_observation_id=(
                candidate.source_schedule_observation_id
            ),
            schedule_as_of_utc=candidate.schedule_as_of_utc,
            scheduled_start_utc=candidate.scheduled_start_utc,
        )
        raw = raw_candidate_for(candidate)

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate, duplicate),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.reason, AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH)

    def test_same_provider_game_different_game_number_is_missing_match(
        self,
    ) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate, game_number=candidate.game_number + 1)

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.reason, MISSING_SCHEDULE_CANDIDATE_MATCH)

    def test_missing_or_invalid_game_number(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)

        for bad_value in (0, -1, True):
            with self.subTest(value=bad_value):
                evidence = evidence_fields(candidate)
                evidence.pop("scheduled_start_utc")
                evidence["game_number"] = bad_value
                raw = ProspectivePredictionCandidate(
                    prediction_observation_id="0" * 64,
                    **{
                        key: (
                            value.isoformat()
                            if key.endswith("_utc")
                            else value
                        )
                        for key, value in evidence.items()
                    },
                )
                result = admit_prospective_prediction(
                    raw,
                    schedule_candidates=(candidate,),
                    schedule_pregame_eligibility=eligibility_set,
                )
                self.assertEqual(
                    result.reason, MISSING_REQUIRED_PREDICTION_EVIDENCE
                )

    def test_schedule_observation_id_mismatch(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(
            candidate,
            source_schedule_observation_id=sha256(b"different").hexdigest(),
        )

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.reason, SCHEDULE_OBSERVATION_ID_MISMATCH)

    def test_schedule_not_pregame_eligible(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate)

        # Re-evaluate the same P12 materialization as-of its own scheduled
        # start: the game becomes INELIGIBLE, so eligible_decisions is empty.
        at_start_fingerprint = (
            compute_schedule_baseball_game_materialization_set_fingerprint(
                as_of_utc=candidate.scheduled_start_utc,
                source_resolution_set_fingerprint=(
                    materialization_set.source_resolution_set_fingerprint
                ),
                authority_catalog_fingerprint=(
                    materialization_set.authority_catalog_fingerprint
                ),
                source_construction_set_fingerprint=(
                    materialization_set.source_construction_set_fingerprint
                ),
                materialized_count=materialization_set.materialized_count,
                unresolved_count=materialization_set.unresolved_count,
                unavailable_count=materialization_set.unavailable_count,
                authority_missing_count=(
                    materialization_set.authority_missing_count
                ),
                game_materializations=(
                    materialization_set.game_materializations
                ),
                unresolved_candidates=(
                    materialization_set.unresolved_candidates
                ),
                unavailable_chain_keys=(
                    materialization_set.unavailable_chain_keys
                ),
                authority_missing_candidates=(
                    materialization_set.authority_missing_candidates
                ),
            )
        )
        at_start_materialization_set = replace(
            materialization_set,
            as_of_utc=candidate.scheduled_start_utc,
            materialization_set_fingerprint=at_start_fingerprint,
        )
        at_start_eligibility = evaluate_schedule_pregame_eligibility(
            at_start_materialization_set
        )

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=at_start_eligibility,
        )
        self.assertEqual(result.reason, SCHEDULE_NOT_PREGAME_ELIGIBLE)

    def test_exact_identity_mismatch_when_declared_id_is_wrong(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        evidence = evidence_fields(candidate)
        scheduled_start_utc = evidence.pop("scheduled_start_utc")
        raw = ProspectivePredictionCandidate(
            prediction_observation_id=sha256(b"wrong-declared-id").hexdigest(),
            **{
                key: (
                    value.isoformat() if key.endswith("_utc") else value
                )
                for key, value in evidence.items()
            },
        )

        result = admit_prospective_prediction(
            raw,
            schedule_candidates=(candidate,),
            schedule_pregame_eligibility=eligibility_set,
        )
        self.assertEqual(result.reason, EXACT_IDENTITY_MISMATCH)

    def test_non_contract_input_is_rejected(self) -> None:
        materialization_set, eligibility_set = eligibility_set_for_one_game()
        with self.assertRaises(TypeError):
            admit_prospective_prediction(
                object(),
                schedule_candidates=(),
                schedule_pregame_eligibility=eligibility_set,
            )
        candidate = schedule_candidate_from(materialization_set, eligibility_set)
        raw = raw_candidate_for(candidate)
        with self.assertRaises(TypeError):
            admit_prospective_prediction(
                raw,
                schedule_candidates=(candidate,),
                schedule_pregame_eligibility=object(),
            )
        with self.assertRaises(TypeError):
            resolve_exact_schedule_candidate(
                provider_namespace="MLB_STATS_API",
                provider_game_id="777001",
                game_number=1,
                schedule_candidates=(object(),),
            )


if __name__ == "__main__":
    unittest.main()
