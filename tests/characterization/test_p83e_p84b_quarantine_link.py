"""Characterize the full-fixture P83E/P84B quarantine referential link.

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


# Independently established by the Pre-Edit Independent Reference Gate: two
# separate calculators (one via the committed importer contracts, one a
# from-scratch JSONL parser sharing no code with the first) agreed on every
# count and on both ID lists byte-for-byte before this file was written.
PINNED_P83E_SEMANTIC_FINGERPRINT = (
    "662dde0f0d7467c29217583f824fa26bbe02ecb43e303c0826e109b742c215ab"
)
REFERENCE_PREDICTION_CANDIDATE_COUNT = 828
REFERENCE_SCHEDULE_CANDIDATE_COUNT = 2430
REFERENCE_LINKED_COUNT = 828
REFERENCE_PREDICTION_MISSING_SCHEDULE_COUNT = 0
REFERENCE_SCHEDULE_ONLY_COUNT = 1602
REFERENCE_COLLISION_AFFECTED_LINKED_COUNT = 14
REFERENCE_JOINT_SEMANTIC_FINGERPRINT = (
    "271f24a1c560dca24c2ba2749596731d526e5e87f4e20489d0786faf1536b679"
)


def _both_fixtures_present() -> bool:
    return bool(
        os.environ.get("MATCHANALYSIS_P83E_SNAPSHOT")
        and os.environ.get("MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT")
    )


class PinnedP83eP84bQuarantineLinkCharacterizationTests(unittest.TestCase):
    @unittest.skipUnless(
        _both_fixtures_present(),
        "MATCHANALYSIS_P83E_SNAPSHOT and MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT "
        "are both required for pinned characterization",
    )
    def test_full_pinned_link_matches_the_independent_reference(self) -> None:
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

        self.assertEqual(
            prediction_result.provenance.input_fingerprint,
            PINNED_P83E_ARTIFACT_SHA256,
        )
        self.assertEqual(schedule_result.artifact_sha256, PINNED_P84B_ARTIFACT_SHA256)
        self.assertEqual(
            prediction_result.semantic_fingerprint,
            PINNED_P83E_SEMANTIC_FINGERPRINT,
        )
        self.assertEqual(
            schedule_result.semantic_fingerprint,
            PINNED_P84B_SEMANTIC_FINGERPRINT,
        )
        self.assertEqual(
            len(prediction_result.candidates),
            REFERENCE_PREDICTION_CANDIDATE_COUNT,
        )
        self.assertEqual(
            len(schedule_result.candidates),
            REFERENCE_SCHEDULE_CANDIDATE_COUNT,
        )

        first = link_legacy_quarantine_snapshots(prediction_result, schedule_result)
        second = link_legacy_quarantine_snapshots(prediction_result, schedule_result)

        self.assertEqual(first.linked_count, REFERENCE_LINKED_COUNT)
        self.assertEqual(
            len(first.prediction_missing_schedule_ids),
            REFERENCE_PREDICTION_MISSING_SCHEDULE_COUNT,
        )
        self.assertEqual(
            len(first.schedule_only_source_ids),
            REFERENCE_SCHEDULE_ONLY_COUNT,
        )
        self.assertEqual(
            first.collision_affected_linked_count,
            REFERENCE_COLLISION_AFFECTED_LINKED_COUNT,
        )
        self.assertEqual(
            first.joint_semantic_fingerprint,
            REFERENCE_JOINT_SEMANTIC_FINGERPRINT,
        )

        # Two independent use-case runs on the same inputs are identical.
        self.assertEqual(first, second)
        self.assertEqual(first.links, second.links)
        self.assertEqual(
            first.joint_semantic_fingerprint,
            second.joint_semantic_fingerprint,
        )

        # All five promotion counts are fixed at zero.
        self.assertEqual(
            (
                first.match_identity_count,
                first.trusted_schedule_observation_count,
                first.baseball_game_count,
                first.canonical_prediction_count,
                first.pregame_eligible_context_count,
            ),
            (0, 0, 0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
