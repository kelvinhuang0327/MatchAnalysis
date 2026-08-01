"""Characterize the full-fixture legacy prediction quarantine assessment.

Requires both MATCHANALYSIS_P83E_SNAPSHOT and MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT
to point at the pinned legacy fixture bytes; neither fixture-dependent test may
skip during acceptance.
"""

from pathlib import Path
import os
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.assess_legacy_prediction_quarantine import (
    ASSESSMENT_SET_SCHEMA_VERSION,
    assess_legacy_prediction_quarantine,
)
from match_analysis.application.use_cases.import_legacy_prediction_snapshot import (
    PINNED_P83E_ARTIFACT_SHA256,
    import_legacy_prediction_snapshot,
)
from match_analysis.application.use_cases.import_legacy_schedule_snapshot import (
    PINNED_P84B_ARTIFACT_SHA256,
    PINNED_P84B_SEMANTIC_FINGERPRINT,
    import_legacy_schedule_snapshot,
)
from match_analysis.application.use_cases.link_legacy_quarantine_snapshots import (
    link_legacy_quarantine_snapshots,
)
from match_analysis.infrastructure.legacy_betting_pool.p83e_jsonl import (
    P83eJsonlSnapshotSource,
)
from match_analysis.infrastructure.legacy_betting_pool.p84b_schedule_jsonl import (
    P84bScheduleJsonlSource,
)


# Independently established against the pinned P83E/P84B fixture bytes before
# this file was written: one full run of the committed importer, linker, and
# assessment use cases end to end (see the P14B2 handoff evidence).
PINNED_P83E_SEMANTIC_FINGERPRINT = (
    "662dde0f0d7467c29217583f824fa26bbe02ecb43e303c0826e109b742c215ab"
)
REFERENCE_ROW_COUNT = 828
REFERENCE_QUARANTINED_COUNT = 828
REFERENCE_ADMITTED_OBSERVATION_COUNT = 0
REFERENCE_UNIQUE_ENRICHMENT_COUNT = 0
REFERENCE_MISSING_ENRICHMENT_COUNT = 828
REFERENCE_AMBIGUOUS_ENRICHMENT_COUNT = 0
REFERENCE_JOINT_SEMANTIC_FINGERPRINT = (
    "271f24a1c560dca24c2ba2749596731d526e5e87f4e20489d0786faf1536b679"
)
REFERENCE_ASSESSMENT_SET_FINGERPRINT = (
    "d1416bdec1aee09f937e34120f0a4d9f0061bc191b70c36b455570dac705ae3c"
)


def _both_fixtures_present() -> bool:
    return bool(
        os.environ.get("MATCHANALYSIS_P83E_SNAPSHOT")
        and os.environ.get("MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT")
    )


class PinnedLegacyPredictionQuarantineAssessmentCharacterizationTests(
    unittest.TestCase
):
    @unittest.skipUnless(
        _both_fixtures_present(),
        "MATCHANALYSIS_P83E_SNAPSHOT and MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT "
        "are both required for pinned characterization",
    )
    def test_all_828_pinned_rows_remain_quarantined_with_zero_admissions(
        self,
    ) -> None:
        p83e_path = Path(os.environ["MATCHANALYSIS_P83E_SNAPSHOT"])
        p84b_path = Path(os.environ["MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT"])

        prediction_result = import_legacy_prediction_snapshot(
            P83eJsonlSnapshotSource(
                p83e_path, expected_sha256=PINNED_P83E_ARTIFACT_SHA256
            )
        )
        schedule_result = import_legacy_schedule_snapshot(
            P84bScheduleJsonlSource(
                p84b_path, expected_sha256=PINNED_P84B_ARTIFACT_SHA256
            )
        )

        # Pinned upstream identities must reproduce before this pass runs.
        self.assertEqual(
            prediction_result.provenance.input_fingerprint,
            PINNED_P83E_ARTIFACT_SHA256,
        )
        self.assertEqual(
            prediction_result.semantic_fingerprint,
            PINNED_P83E_SEMANTIC_FINGERPRINT,
        )
        self.assertEqual(schedule_result.artifact_sha256, PINNED_P84B_ARTIFACT_SHA256)
        self.assertEqual(
            schedule_result.semantic_fingerprint,
            PINNED_P84B_SEMANTIC_FINGERPRINT,
        )

        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        self.assertEqual(
            link.joint_semantic_fingerprint, REFERENCE_JOINT_SEMANTIC_FINGERPRINT
        )

        first = assess_legacy_prediction_quarantine(prediction_result, link)
        second = assess_legacy_prediction_quarantine(prediction_result, link)

        # Every current row remains quarantined; nothing is ever admitted.
        self.assertEqual(first.row_count, REFERENCE_ROW_COUNT)
        self.assertEqual(first.quarantined_count, REFERENCE_QUARANTINED_COUNT)
        self.assertEqual(
            first.admitted_observation_count,
            REFERENCE_ADMITTED_OBSERVATION_COUNT,
        )
        self.assertEqual(len(first.assessments), REFERENCE_ROW_COUNT)
        self.assertTrue(
            all(
                assessment.quarantine_status == "QUARANTINED"
                for assessment in first.assessments
            )
        )

        # No P9 candidates were supplied: every linked row stays unenriched,
        # never ambiguous, since there is nothing to match against.
        self.assertEqual(
            first.unique_enrichment_count, REFERENCE_UNIQUE_ENRICHMENT_COUNT
        )
        self.assertEqual(
            first.missing_enrichment_count, REFERENCE_MISSING_ENRICHMENT_COUNT
        )
        self.assertEqual(
            first.ambiguous_enrichment_count, REFERENCE_AMBIGUOUS_ENRICHMENT_COUNT
        )

        # Pinned upstream fingerprints pass through this stage unchanged.
        self.assertEqual(first.p83e_raw_sha256, PINNED_P83E_ARTIFACT_SHA256)
        self.assertEqual(
            first.p83e_semantic_fingerprint, PINNED_P83E_SEMANTIC_FINGERPRINT
        )
        self.assertEqual(first.p84b_artifact_sha256, PINNED_P84B_ARTIFACT_SHA256)
        self.assertEqual(
            first.p84b_semantic_fingerprint, PINNED_P84B_SEMANTIC_FINGERPRINT
        )
        self.assertEqual(
            first.joint_semantic_fingerprint, REFERENCE_JOINT_SEMANTIC_FINGERPRINT
        )

        self.assertEqual(first.schema_version, ASSESSMENT_SET_SCHEMA_VERSION)
        self.assertEqual(
            first.assessment_set_fingerprint, REFERENCE_ASSESSMENT_SET_FINGERPRINT
        )

        # Two independent runs on the same pinned inputs are byte-identical.
        self.assertEqual(first, second)
        self.assertEqual(first.assessments, second.assessments)
        self.assertEqual(
            first.assessment_set_fingerprint, second.assessment_set_fingerprint
        )

    @unittest.skipUnless(
        _both_fixtures_present(),
        "MATCHANALYSIS_P83E_SNAPSHOT and MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT "
        "are both required for pinned characterization",
    )
    def test_shuffled_equivalent_inputs_reproduce_the_same_fingerprint(
        self,
    ) -> None:
        p83e_path = Path(os.environ["MATCHANALYSIS_P83E_SNAPSHOT"])
        p84b_path = Path(os.environ["MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT"])

        prediction_result = import_legacy_prediction_snapshot(
            P83eJsonlSnapshotSource(
                p83e_path, expected_sha256=PINNED_P83E_ARTIFACT_SHA256
            )
        )
        schedule_result = import_legacy_schedule_snapshot(
            P84bScheduleJsonlSource(
                p84b_path, expected_sha256=PINNED_P84B_ARTIFACT_SHA256
            )
        )
        link = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        baseline = assess_legacy_prediction_quarantine(prediction_result, link)

        # Re-deriving the same link from candidates in reverse order must not
        # change the assessment outcome or its fingerprint.
        reordered_candidates = tuple(reversed(prediction_result.candidates))
        reordered_prediction_result = type(prediction_result)(
            provenance=prediction_result.provenance,
            row_count=prediction_result.row_count,
            unique_id_count=prediction_result.unique_id_count,
            validated_null_outcome_placeholder_fields=(
                prediction_result.validated_null_outcome_placeholder_fields
            ),
            validated_null_outcome_placeholder_count=(
                prediction_result.validated_null_outcome_placeholder_count
            ),
            rows_with_observed_outcomes=prediction_result.rows_with_observed_outcomes,
            promoted_prediction_count=prediction_result.promoted_prediction_count,
            candidates=reordered_candidates,
            semantic_fingerprint=prediction_result.semantic_fingerprint,
            limitations=prediction_result.limitations,
            quarantine_counts=prediction_result.quarantine_counts,
        )
        reordered_link = link_legacy_quarantine_snapshots(
            reordered_prediction_result, schedule_result
        )
        reordered = assess_legacy_prediction_quarantine(
            reordered_prediction_result, reordered_link
        )

        self.assertEqual(
            baseline.assessment_set_fingerprint,
            reordered.assessment_set_fingerprint,
        )
        self.assertEqual(baseline.assessments, reordered.assessments)


if __name__ == "__main__":
    unittest.main()
