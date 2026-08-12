"""Focused artifact-rendering and CLI output contract tests for P39A."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from match_analysis.application.use_cases.join_p37_oos_market_snapshots import (
    P37OOSPrediction,
    build_p39a_market_join,
    mark_deterministic_rerun_verified,
)
from match_analysis.application.use_cases.p39a_market_join_artifacts import (
    P39A_ARTIFACT_FILES,
    write_p39a_artifacts,
)

from tests.unit.test_p39a_market_join import SOURCE_MANIFEST, _candidate, _prediction


class P39AArtifactTests(unittest.TestCase):
    def test_writer_emits_exact_deterministic_artifact_set(self) -> None:
        result = build_p39a_market_join(
            [_prediction()],
            [_candidate(fetched_at="2026-06-08T21:00:00Z", row_index=1)],
            source_manifest=SOURCE_MANIFEST,
            p37_manifest={"comparisons_sha256": "e" * 64},
        )
        result = mark_deterministic_rerun_verified(result)
        first = tempfile.TemporaryDirectory()
        second = tempfile.TemporaryDirectory()
        self.addCleanup(first.cleanup)
        self.addCleanup(second.cleanup)

        write_p39a_artifacts(first.name, result)
        write_p39a_artifacts(second.name, result)

        self.assertEqual(
            sorted(path.name for path in Path(first.name).iterdir()),
            sorted(P39A_ARTIFACT_FILES),
        )
        for filename in P39A_ARTIFACT_FILES:
            self.assertEqual(
                (Path(first.name) / filename).read_bytes(),
                (Path(second.name) / filename).read_bytes(),
            )
        self.assertIn('"bet_pass": "NOT_RUN"', (Path(first.name) / "summary.json").read_text())


if __name__ == "__main__":
    unittest.main()
