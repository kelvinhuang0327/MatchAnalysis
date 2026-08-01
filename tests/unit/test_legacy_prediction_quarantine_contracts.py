"""Unit tests for the immutable legacy prediction quarantine assessment."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from datetime import date, datetime, timezone
from pathlib import Path
import ast
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.legacy_prediction_quarantine import (
    AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
    CONTROLLED_QUARANTINE_REASONS,
    MISSING_DIAGNOSTIC_SCHEDULE_LINK,
    MISSING_GAME_NUMBER,
    MISSING_SOURCE_OBSERVATION_ID,
    QUARANTINE_STATUS,
    UNIVERSAL_MISSING_OBSERVATION_REASONS,
    ZERO_DELTA_SELECTION_POLICY_UNRESOLVED,
    LegacyPredictionQuarantineAssessment,
)
from match_analysis.baseball.domain.prediction import LegacyPredictionCandidate
from match_analysis.baseball.domain.quarantine_link import (
    LegacyDiagnosticPredictionScheduleLink,
)
from match_analysis.baseball.domain.schedule import (
    PROVIDER_NAMESPACE,
    LegacyDiagnosticScheduleCandidate,
    ProviderGameReference,
    UNIVERSAL_QUARANTINE_REASONS,
)
from match_analysis.baseball.domain.schedule_identity_candidate import (
    ScheduleIdentityResolutionCandidate,
)
from match_analysis.application.use_cases.assess_legacy_prediction_quarantine import (
    ASSESSMENT_SET_SCHEMA_VERSION,
    LegacyPredictionQuarantineAssessmentSet,
    assess_legacy_prediction_quarantine,
)
from match_analysis.application.use_cases.import_legacy_prediction_snapshot import (
    LegacyPredictionImportResult,
)
from match_analysis.application.use_cases.import_legacy_schedule_snapshot import (
    LegacyScheduleImportResult,
)
from match_analysis.application.use_cases.link_legacy_quarantine_snapshots import (
    link_legacy_quarantine_snapshots,
)
from match_analysis.core.provenance import ArtifactProvenance


ALL_MISSING_ENRICHMENT_REASONS = (MISSING_GAME_NUMBER, MISSING_SOURCE_OBSERVATION_ID)


def make_prediction_candidate(
    game_id: str = "mlb_2026_100",
    *,
    sp_fip_delta: Decimal = Decimal("-0.25"),
) -> LegacyPredictionCandidate:
    return LegacyPredictionCandidate(
        source_game_id=game_id,
        source_prediction_version="p84b_diagnostic_baseline_v1",
        predicted_side="home" if sp_fip_delta < 0 else "away",
        sp_fip_delta=sp_fip_delta,
    )


def make_reference(game_id: str = "mlb_2026_100") -> ProviderGameReference:
    provider_game_id = game_id.rsplit("_", 1)[-1]
    return ProviderGameReference(
        provider_namespace=PROVIDER_NAMESPACE,
        provider_game_id=provider_game_id,
        source_game_id=game_id,
    )


def make_schedule_candidate(
    game_id: str = "mlb_2026_100",
) -> LegacyDiagnosticScheduleCandidate:
    return LegacyDiagnosticScheduleCandidate(
        provider_reference=make_reference(game_id),
        season=2026,
        game_date="2026-04-01",
        source_home_team="Team A",
        source_away_team="Team B",
        legacy_collection_marker_utc="2026-04-01T00:00:00Z",
        quarantine_reasons=UNIVERSAL_QUARANTINE_REASONS,
    )


def make_link(
    game_id: str = "mlb_2026_100",
    *,
    prediction_candidate: LegacyPredictionCandidate | None = None,
) -> LegacyDiagnosticPredictionScheduleLink:
    schedule_candidate = make_schedule_candidate(game_id)
    return LegacyDiagnosticPredictionScheduleLink(
        source_game_id=game_id,
        provider_reference=schedule_candidate.provider_reference,
        prediction_candidate=prediction_candidate or make_prediction_candidate(game_id),
        schedule_candidate=schedule_candidate,
    )


def make_unenriched_reasons(*, linked: bool) -> tuple[str, ...]:
    extra = (MISSING_GAME_NUMBER, MISSING_SOURCE_OBSERVATION_ID)
    if not linked:
        extra = (MISSING_DIAGNOSTIC_SCHEDULE_LINK, *extra)
    return tuple(sorted({*UNIVERSAL_MISSING_OBSERVATION_REASONS, *extra}))


def make_assessment(
    game_id: str = "mlb_2026_100",
    *,
    prediction_candidate: LegacyPredictionCandidate | None = None,
    diagnostic_link: LegacyDiagnosticPredictionScheduleLink | None = None,
    quarantine_reasons: tuple[str, ...] | None = None,
    enriched_game_number: int | None = None,
    enriched_source_observation_id: str | None = None,
    linked: bool = True,
) -> LegacyPredictionQuarantineAssessment:
    prediction_candidate = prediction_candidate or make_prediction_candidate(game_id)
    if diagnostic_link is None and linked:
        diagnostic_link = make_link(game_id, prediction_candidate=prediction_candidate)
    if quarantine_reasons is None:
        if enriched_game_number is not None:
            quarantine_reasons = tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS))
        else:
            quarantine_reasons = make_unenriched_reasons(linked=diagnostic_link is not None)
    return LegacyPredictionQuarantineAssessment(
        source_game_id=game_id,
        prediction_candidate=prediction_candidate,
        diagnostic_link=diagnostic_link,
        quarantine_reasons=quarantine_reasons,
        enriched_game_number=enriched_game_number,
        enriched_source_observation_id=enriched_source_observation_id,
    )


def make_prediction_result(
    candidates: tuple[LegacyPredictionCandidate, ...],
) -> LegacyPredictionImportResult:
    return LegacyPredictionImportResult(
        provenance=ArtifactProvenance(
            schema_version="p83e_snapshot_quarantine_v1",
            source_repository="Betting-pool",
            source_commit="0" * 40,
            producer_id="unit-test",
            producer_version="1",
            input_fingerprint="a" * 64,
            content_fingerprint="b" * 64,
        ),
        row_count=len(candidates),
        unique_id_count=len(candidates),
        validated_null_outcome_placeholder_fields=(
            "result_home_score",
            "result_away_score",
            "actual_winner",
            "is_correct",
        ),
        validated_null_outcome_placeholder_count=4,
        rows_with_observed_outcomes=0,
        promoted_prediction_count=0,
        candidates=candidates,
        semantic_fingerprint="c" * 64,
        limitations=("unit-test fixture",),
        quarantine_counts=(),
    )


def make_schedule_result(
    candidates: tuple[LegacyDiagnosticScheduleCandidate, ...],
) -> LegacyScheduleImportResult:
    references = tuple(candidate.provider_reference for candidate in candidates)
    return LegacyScheduleImportResult(
        artifact_sha256="d" * 64,
        row_count=len(candidates),
        unique_provider_reference_count=len(set(references)),
        provider_game_references=references,
        candidates=candidates,
        collision_groups=(),
        collision_affected_row_count=0,
        semantic_fingerprint="e" * 64,
        limitations=("unit-test fixture",),
        quarantine_counts=(),
        match_identity_count=0,
        trusted_schedule_observation_count=0,
        baseball_game_count=0,
        pregame_eligible_context_count=0,
    )


def make_p9_candidate(
    game_id: str = "mlb_2026_100",
    *,
    game_number: int = 1,
    source_observation_id: str = "1" * 64,
) -> ScheduleIdentityResolutionCandidate:
    provider_game_id = game_id.rsplit("_", 1)[-1]
    return ScheduleIdentityResolutionCandidate(
        provider_namespace=PROVIDER_NAMESPACE,
        provider_game_id=provider_game_id,
        game_number=game_number,
        scheduled_start_utc=datetime(2026, 4, 1, 17, 5, tzinfo=timezone.utc),
        official_local_date=date(2026, 4, 1),
        home_provider_participant_id="provider-home",
        away_provider_participant_id="provider-away",
        source_observation_id=source_observation_id,
        source_raw_payload_sha256="2" * 64,
    )


class LegacyPredictionQuarantineAssessmentTests(unittest.TestCase):
    def test_assessment_is_immutable(self) -> None:
        assessment = make_assessment()

        with self.assertRaises(FrozenInstanceError):
            assessment.source_game_id = "mlb_2026_999"
        with self.assertRaises(FrozenInstanceError):
            assessment.quarantine_reasons = ()

    def test_quarantine_status_is_fixed(self) -> None:
        assessment = make_assessment()

        self.assertEqual(assessment.quarantine_status, QUARANTINE_STATUS)
        with self.assertRaisesRegex(ValueError, "quarantine_status"):
            LegacyPredictionQuarantineAssessment(
                source_game_id="mlb_2026_100",
                prediction_candidate=make_prediction_candidate("mlb_2026_100"),
                diagnostic_link=make_link("mlb_2026_100"),
                quarantine_reasons=make_unenriched_reasons(linked=True),
                quarantine_status="ADMITTED",
            )

    def test_source_game_id_must_match_prediction_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "prediction candidate"):
            LegacyPredictionQuarantineAssessment(
                source_game_id="mlb_2026_200",
                prediction_candidate=make_prediction_candidate("mlb_2026_100"),
                diagnostic_link=None,
                quarantine_reasons=make_unenriched_reasons(linked=False),
            )

    def test_diagnostic_link_must_agree_with_source_game_id_and_candidate(self) -> None:
        prediction_candidate = make_prediction_candidate("mlb_2026_100")

        with self.assertRaisesRegex(ValueError, "diagnostic link"):
            LegacyPredictionQuarantineAssessment(
                source_game_id="mlb_2026_100",
                prediction_candidate=prediction_candidate,
                diagnostic_link=make_link("mlb_2026_200"),
                quarantine_reasons=make_unenriched_reasons(linked=True),
            )
        with self.assertRaisesRegex(ValueError, "same prediction candidate"):
            LegacyPredictionQuarantineAssessment(
                source_game_id="mlb_2026_100",
                prediction_candidate=prediction_candidate,
                diagnostic_link=make_link(
                    "mlb_2026_100",
                    prediction_candidate=make_prediction_candidate(
                        "mlb_2026_100", sp_fip_delta=Decimal("0.75")
                    ),
                ),
                quarantine_reasons=make_unenriched_reasons(linked=True),
            )

    def test_quarantine_reasons_must_be_sorted_unique_and_controlled(self) -> None:
        base = make_unenriched_reasons(linked=False)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            make_assessment(linked=False, quarantine_reasons=())
        with self.assertRaisesRegex(ValueError, "not repeat"):
            make_assessment(linked=False, quarantine_reasons=(*base, base[0]))
        with self.assertRaisesRegex(ValueError, "sorted ascending"):
            make_assessment(linked=False, quarantine_reasons=tuple(reversed(base)))
        with self.assertRaisesRegex(ValueError, "uncontrolled"):
            make_assessment(
                linked=False,
                quarantine_reasons=tuple(sorted({*base, "NOT_A_REAL_REASON"})),
            )

    def test_every_row_requires_all_universal_missing_observation_reasons(self) -> None:
        incomplete = tuple(
            reason
            for reason in make_unenriched_reasons(linked=False)
            if reason != UNIVERSAL_MISSING_OBSERVATION_REASONS[2]
        )
        with self.assertRaisesRegex(ValueError, "universal missing-observation"):
            make_assessment(linked=False, quarantine_reasons=incomplete)

    def test_missing_diagnostic_schedule_link_reason_tracks_the_link(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be present"):
            make_assessment(
                linked=True,
                quarantine_reasons=make_unenriched_reasons(linked=False),
            )
        with self.assertRaisesRegex(ValueError, "must be present"):
            make_assessment(
                linked=False,
                quarantine_reasons=make_unenriched_reasons(linked=True),
            )

    def test_ambiguous_reason_requires_a_diagnostic_link(self) -> None:
        reasons = tuple(
            sorted({*make_unenriched_reasons(linked=False), AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH})
        )
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH"):
            make_assessment(linked=False, quarantine_reasons=reasons)

    def test_enrichment_fields_must_be_set_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "set or unset together"):
            make_assessment(linked=True, enriched_game_number=1)
        with self.assertRaisesRegex(ValueError, "set or unset together"):
            make_assessment(
                linked=True,
                quarantine_reasons=tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS)),
                enriched_source_observation_id="1" * 64,
            )

    def test_enrichment_requires_a_diagnostic_link(self) -> None:
        with self.assertRaisesRegex(ValueError, "enrichment requires"):
            make_assessment(
                linked=False,
                quarantine_reasons=tuple(
                    sorted(
                        {
                            *UNIVERSAL_MISSING_OBSERVATION_REASONS,
                            MISSING_DIAGNOSTIC_SCHEDULE_LINK,
                        }
                    )
                ),
                enriched_game_number=1,
                enriched_source_observation_id="1" * 64,
            )

    def test_enriched_game_number_must_be_a_positive_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_assessment(
                linked=True,
                quarantine_reasons=tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS)),
                enriched_game_number=0,
                enriched_source_observation_id="1" * 64,
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_assessment(
                linked=True,
                quarantine_reasons=tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS)),
                enriched_game_number=True,
                enriched_source_observation_id="1" * 64,
            )

    def test_enriched_source_observation_id_must_be_a_sha256(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            make_assessment(
                linked=True,
                quarantine_reasons=tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS)),
                enriched_game_number=1,
                enriched_source_observation_id="not-a-hash",
            )

    def test_enriched_row_must_not_carry_missing_enrichment_reasons(self) -> None:
        for reason in (
            MISSING_GAME_NUMBER,
            MISSING_SOURCE_OBSERVATION_ID,
            AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH,
        ):
            reasons = tuple(sorted({*UNIVERSAL_MISSING_OBSERVATION_REASONS, reason}))
            with self.assertRaisesRegex(ValueError, "must not be present on an enriched row"):
                make_assessment(
                    linked=True,
                    quarantine_reasons=reasons,
                    enriched_game_number=1,
                    enriched_source_observation_id="1" * 64,
                )

    def test_unenriched_row_must_carry_missing_enrichment_reasons(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be present on an unenriched row"):
            make_assessment(
                linked=True,
                quarantine_reasons=tuple(sorted(UNIVERSAL_MISSING_OBSERVATION_REASONS)),
            )

    def test_zero_delta_is_structurally_unreachable_but_defended(self) -> None:
        # LegacyPredictionCandidate itself forbids a zero delta, so this
        # defensive branch can never fire for any real P83E-sourced row.
        with self.assertRaisesRegex(ValueError, "non-zero"):
            make_prediction_candidate("mlb_2026_100", sp_fip_delta=Decimal("0"))

        # A non-zero delta must never carry the zero-delta reason.
        with self.assertRaisesRegex(ValueError, "must not be present"):
            make_assessment(
                linked=False,
                quarantine_reasons=tuple(
                    sorted(
                        {
                            *make_unenriched_reasons(linked=False),
                            ZERO_DELTA_SELECTION_POLICY_UNRESOLVED,
                        }
                    )
                ),
            )


class AssessLegacyPredictionQuarantineTests(unittest.TestCase):
    def test_all_rows_remain_quarantined_and_none_are_admitted(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_101"),
                make_prediction_candidate("mlb_2026_999"),
            )
        )
        schedule_result = make_schedule_result(
            (
                make_schedule_candidate("mlb_2026_100"),
                make_schedule_candidate("mlb_2026_101"),
            )
        )
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        result = assess_legacy_prediction_quarantine(prediction_result, link)

        self.assertEqual(result.row_count, 3)
        self.assertEqual(result.quarantined_count, 3)
        self.assertEqual(result.admitted_observation_count, 0)
        self.assertTrue(
            all(
                assessment.quarantine_status == QUARANTINE_STATUS
                for assessment in result.assessments
            )
        )

    def test_missing_schedule_link_yields_missing_diagnostic_schedule_link(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_999"),)
        )
        schedule_result = make_schedule_result(())
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        result = assess_legacy_prediction_quarantine(prediction_result, link)

        (assessment,) = result.assessments
        self.assertIsNone(assessment.diagnostic_link)
        self.assertIn(MISSING_DIAGNOSTIC_SCHEDULE_LINK, assessment.quarantine_reasons)
        self.assertEqual(result.missing_enrichment_count, 1)
        self.assertEqual(result.unique_enrichment_count, 0)
        self.assertEqual(result.ambiguous_enrichment_count, 0)

    def test_unique_p9_match_enriches_game_number_and_observation_id(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        p9 = (make_p9_candidate("mlb_2026_100", game_number=2, source_observation_id="9" * 64),)

        result = assess_legacy_prediction_quarantine(prediction_result, link, p9)

        (assessment,) = result.assessments
        self.assertEqual(assessment.enriched_game_number, 2)
        self.assertEqual(assessment.enriched_source_observation_id, "9" * 64)
        self.assertNotIn(MISSING_GAME_NUMBER, assessment.quarantine_reasons)
        self.assertNotIn(MISSING_SOURCE_OBSERVATION_ID, assessment.quarantine_reasons)
        self.assertEqual(result.unique_enrichment_count, 1)

    def test_zero_p9_matches_remain_unenriched(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        p9 = (make_p9_candidate("mlb_2026_200"),)  # different provider_game_id

        result = assess_legacy_prediction_quarantine(prediction_result, link, p9)

        (assessment,) = result.assessments
        self.assertIsNone(assessment.enriched_game_number)
        self.assertIn(MISSING_GAME_NUMBER, assessment.quarantine_reasons)
        self.assertNotIn(AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH, assessment.quarantine_reasons)
        self.assertEqual(result.missing_enrichment_count, 1)

    def test_multiple_p9_matches_are_ambiguous_and_remain_unenriched(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        p9 = (
            make_p9_candidate("mlb_2026_100", game_number=1, source_observation_id="1" * 64),
            make_p9_candidate("mlb_2026_100", game_number=2, source_observation_id="2" * 64),
        )

        result = assess_legacy_prediction_quarantine(prediction_result, link, p9)

        (assessment,) = result.assessments
        self.assertIsNone(assessment.enriched_game_number)
        self.assertIn(AMBIGUOUS_SCHEDULE_CANDIDATE_MATCH, assessment.quarantine_reasons)
        self.assertIn(MISSING_GAME_NUMBER, assessment.quarantine_reasons)
        self.assertEqual(result.ambiguous_enrichment_count, 1)
        self.assertEqual(result.unique_enrichment_count, 0)

    def test_never_joins_by_date_or_team_only_by_exact_provider_key(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        # Same game_date/home/away as the schedule fixture, but a different
        # provider_game_id: must never match.
        mismatched_key_candidate = make_p9_candidate("mlb_2026_999")

        result = assess_legacy_prediction_quarantine(
            prediction_result, link, (mismatched_key_candidate,)
        )

        (assessment,) = result.assessments
        self.assertIsNone(assessment.enriched_game_number)

    def test_mismatched_provenance_is_rejected(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        other_prediction_result = LegacyPredictionImportResult(
            provenance=ArtifactProvenance(
                schema_version="p83e_snapshot_quarantine_v1",
                source_repository="Betting-pool",
                source_commit="1" * 40,
                producer_id="unit-test",
                producer_version="1",
                input_fingerprint="f" * 64,
                content_fingerprint="f" * 64,
            ),
            row_count=prediction_result.row_count,
            unique_id_count=prediction_result.unique_id_count,
            validated_null_outcome_placeholder_fields=(
                prediction_result.validated_null_outcome_placeholder_fields
            ),
            validated_null_outcome_placeholder_count=(
                prediction_result.validated_null_outcome_placeholder_count
            ),
            rows_with_observed_outcomes=0,
            promoted_prediction_count=0,
            candidates=prediction_result.candidates,
            semantic_fingerprint=prediction_result.semantic_fingerprint,
            limitations=prediction_result.limitations,
            quarantine_counts=(),
        )

        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            assess_legacy_prediction_quarantine(other_prediction_result, link)

    def test_quarantine_link_must_account_for_exactly_the_supplied_candidates(self) -> None:
        prediction_result = make_prediction_result(
            (
                make_prediction_candidate("mlb_2026_100"),
                make_prediction_candidate("mlb_2026_200"),
            )
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        narrower_prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )

        with self.assertRaisesRegex(ValueError, "does not account for"):
            assess_legacy_prediction_quarantine(narrower_prediction_result, link)

    def test_result_is_immutable(self) -> None:
        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        result = assess_legacy_prediction_quarantine(prediction_result, link)

        with self.assertRaises(FrozenInstanceError):
            result.row_count = 99

    def test_output_is_deterministic_under_shuffled_equivalent_inputs(self) -> None:
        candidates = (
            make_prediction_candidate("mlb_2026_300"),
            make_prediction_candidate("mlb_2026_100"),
            make_prediction_candidate("mlb_2026_200"),
        )
        schedule_candidates = (
            make_schedule_candidate("mlb_2026_300"),
            make_schedule_candidate("mlb_2026_100"),
            make_schedule_candidate("mlb_2026_200"),
        )
        p9 = (
            make_p9_candidate("mlb_2026_100", game_number=1, source_observation_id="1" * 64),
            make_p9_candidate("mlb_2026_200", game_number=2, source_observation_id="2" * 64),
        )

        prediction_result = make_prediction_result(candidates)
        schedule_result = make_schedule_result(schedule_candidates)
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        first = assess_legacy_prediction_quarantine(prediction_result, link, p9)

        shuffled_prediction_result = make_prediction_result(tuple(reversed(candidates)))
        shuffled_schedule_result = make_schedule_result(tuple(reversed(schedule_candidates)))
        shuffled_link = link_legacy_quarantine_snapshots(
            shuffled_prediction_result, shuffled_schedule_result
        )
        second = assess_legacy_prediction_quarantine(
            shuffled_prediction_result, shuffled_link, tuple(reversed(p9))
        )

        self.assertEqual(
            tuple(a.source_game_id for a in first.assessments),
            ("mlb_2026_100", "mlb_2026_200", "mlb_2026_300"),
        )
        self.assertEqual(first.assessments, second.assessments)
        self.assertEqual(
            first.assessment_set_fingerprint, second.assessment_set_fingerprint
        )

    def test_enrichment_counts_sum_to_row_count(self) -> None:
        candidates = (
            make_prediction_candidate("mlb_2026_100"),
            make_prediction_candidate("mlb_2026_200"),
            make_prediction_candidate("mlb_2026_300"),
        )
        schedule_candidates = (
            make_schedule_candidate("mlb_2026_100"),
            make_schedule_candidate("mlb_2026_200"),
        )
        prediction_result = make_prediction_result(candidates)
        schedule_result = make_schedule_result(schedule_candidates)
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        p9 = (
            make_p9_candidate("mlb_2026_100"),
            make_p9_candidate("mlb_2026_200", game_number=9, source_observation_id="3" * 64),
            make_p9_candidate("mlb_2026_200", game_number=10, source_observation_id="4" * 64),
        )

        result = assess_legacy_prediction_quarantine(prediction_result, link, p9)

        self.assertEqual(
            result.unique_enrichment_count
            + result.missing_enrichment_count
            + result.ambiguous_enrichment_count,
            result.row_count,
        )
        self.assertEqual(result.unique_enrichment_count, 1)
        self.assertEqual(result.ambiguous_enrichment_count, 1)
        self.assertEqual(result.missing_enrichment_count, 1)

    def test_fingerprint_is_reproducible_independently(self) -> None:
        import json
        from hashlib import sha256

        prediction_result = make_prediction_result(
            (make_prediction_candidate("mlb_2026_100"),)
        )
        schedule_result = make_schedule_result((make_schedule_candidate("mlb_2026_100"),))
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        result = assess_legacy_prediction_quarantine(prediction_result, link)

        independent_payload = {
            "schema_version": ASSESSMENT_SET_SCHEMA_VERSION,
            "row_count": 1,
            "quarantined_count": 1,
            "admitted_observation_count": 0,
            "unique_enrichment_count": 0,
            "missing_enrichment_count": 1,
            "ambiguous_enrichment_count": 0,
            "p83e_raw_sha256": prediction_result.provenance.input_fingerprint,
            "p83e_semantic_fingerprint": prediction_result.semantic_fingerprint,
            "p84b_artifact_sha256": schedule_result.artifact_sha256,
            "p84b_semantic_fingerprint": schedule_result.semantic_fingerprint,
            "joint_semantic_fingerprint": link.joint_semantic_fingerprint,
            "assessments": [
                {
                    "source_game_id": "mlb_2026_100",
                    "quarantine_status": QUARANTINE_STATUS,
                    "quarantine_reasons": list(
                        make_unenriched_reasons(linked=True)
                    ),
                    "enriched_game_number": None,
                    "enriched_source_observation_id": None,
                }
            ],
        }
        independent_encoded = (
            json.dumps(
                independent_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        independent_fingerprint = sha256(independent_encoded).hexdigest()

        self.assertEqual(result.assessment_set_fingerprint, independent_fingerprint)

    def test_use_case_has_no_file_path_or_time_dependency(self) -> None:
        module_path = (
            REPOSITORY_ROOT
            / "src"
            / "match_analysis"
            / "application"
            / "use_cases"
            / "assess_legacy_prediction_quarantine.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[0])

        self.assertTrue(
            {"pathlib", "os", "io", "datetime", "time"}.isdisjoint(imported_names)
        )

        source_text = module_path.read_text(encoding="utf-8")
        for forbidden_token in (
            "open(",
            "Path(",
            "datetime.now(",
            "time.time(",
            "PredictionSourceObservation(",
        ):
            self.assertNotIn(forbidden_token, source_text)


if __name__ == "__main__":
    unittest.main()
