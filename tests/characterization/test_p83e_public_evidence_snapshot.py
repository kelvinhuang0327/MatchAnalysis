"""Characterize the complete public evidence view of the pinned P83E artifact."""

from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.legacy_prediction_source import (
    LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION,
    LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION,
)
from match_analysis.application.use_cases.import_legacy_prediction_snapshot import (
    PINNED_P83E_ARTIFACT_SHA256,
)
from match_analysis.infrastructure.legacy_betting_pool.p83e_jsonl import (
    P83eJsonlSnapshotSource,
)


PINNED_SOURCE_REPOSITORY = "Betting-pool"
PINNED_SOURCE_REF = "03b2fcf4de1a13ee9929afcef803d61955c9f41b"
PINNED_SOURCE_BLOB = "bc06f353160656b79d21a27555758791535ea823"
PINNED_SEMANTIC_FINGERPRINT = (
    "662dde0f0d7467c29217583f824fa26bbe02ecb43e303c0826e109b742c215ab"
)
PINNED_ROW_COUNT = 828


def evidence_source(path: Path) -> P83eJsonlSnapshotSource:
    return P83eJsonlSnapshotSource(
        path,
        expected_sha256=PINNED_P83E_ARTIFACT_SHA256,
        expected_row_count=PINNED_ROW_COUNT,
        expected_semantic_fingerprint=PINNED_SEMANTIC_FINGERPRINT,
        source_repository=PINNED_SOURCE_REPOSITORY,
        source_ref=PINNED_SOURCE_REF,
        source_blob=PINNED_SOURCE_BLOB,
    )


def independent_snapshot_fingerprint(snapshot: object) -> str:
    projection = {
        "parser_version": snapshot.parser_version,
        "raw_artifact_sha256": snapshot.raw_artifact_sha256,
        "raw_row_sha256": [row.raw_row_sha256 for row in snapshot.rows],
        "row_count": snapshot.row_count,
        "schema_version": snapshot.schema_version,
        "semantic_fingerprint": snapshot.semantic_fingerprint,
        "source_blob": snapshot.source_blob,
        "source_ref": snapshot.source_ref,
        "source_repository": snapshot.source_repository,
    }
    return sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class P83ePublicEvidenceSnapshotTests(unittest.TestCase):
    def test_evidence_loading_requires_explicit_source_provenance(self) -> None:
        raw = b"{}"
        source = P83eJsonlSnapshotSource(
            raw,
            expected_sha256=sha256(raw).hexdigest(),
        )

        with self.assertRaisesRegex(ValueError, "load_evidence requires explicit"):
            source.load_evidence()

    @unittest.skipUnless(
        os.environ.get("MATCHANALYSIS_P83E_SNAPSHOT"),
        "MATCHANALYSIS_P83E_SNAPSHOT is required for pinned characterization",
    )
    def test_pinned_snapshot_exposes_complete_ordered_public_evidence(
        self,
    ) -> None:
        path = Path(os.environ["MATCHANALYSIS_P83E_SNAPSHOT"])
        raw = path.read_bytes()
        first = evidence_source(path).load_evidence()
        second = evidence_source(path).load_evidence()

        self.assertEqual(first, second)
        self.assertEqual(first.row_count, PINNED_ROW_COUNT)
        self.assertEqual(first.raw_artifact_sha256, PINNED_P83E_ARTIFACT_SHA256)
        self.assertEqual(first.semantic_fingerprint, PINNED_SEMANTIC_FINGERPRINT)
        self.assertEqual(first.source_repository, PINNED_SOURCE_REPOSITORY)
        self.assertEqual(first.source_ref, PINNED_SOURCE_REF)
        self.assertEqual(first.source_blob, PINNED_SOURCE_BLOB)
        self.assertEqual(
            first.parser_version,
            LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION,
        )
        self.assertEqual(
            first.schema_version,
            LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            first.snapshot_fingerprint,
            independent_snapshot_fingerprint(first),
        )
        self.assertEqual(
            len({row.legacy_game_id for row in first.rows}),
            PINNED_ROW_COUNT,
        )

        first_raw_row = raw.split(b"\n", 1)[0]
        row = first.rows[0]
        self.assertEqual(row.legacy_game_id, "mlb_2026_822733")
        self.assertEqual(row.game_date, "2026-05-21")
        self.assertEqual(row.season, Decimal("2026"))
        self.assertEqual(row.home_team, "Washington Nationals")
        self.assertEqual(row.away_team, "New York Mets")
        self.assertEqual(row.home_sp_fip, Decimal("3.033"))
        self.assertEqual(row.away_sp_fip, Decimal("3.0625"))
        self.assertEqual(row.sp_fip_delta, Decimal("-0.0295"))
        self.assertEqual(row.abs_sp_fip_delta, Decimal("0.0295"))
        self.assertEqual(row.home_win_probability, Decimal("0.504425"))
        self.assertEqual(row.predicted_side, "home")
        self.assertEqual(
            row.source_prediction_version,
            "p84b_diagnostic_baseline_v1",
        )
        self.assertEqual(
            (
                row.rule_primary_125_flag,
                row.rule_shadow_100_flag,
                row.tier_b_candidate_flag,
                row.tier_a_watchlist_flag,
                row.paper_only,
                row.diagnostic_only,
                row.odds_used,
                row.market_edge_evaluated,
                row.production_ready,
            ),
            (False, False, False, True, True, True, False, False, False),
        )
        self.assertEqual(
            (
                row.result_home_score,
                row.result_away_score,
                row.actual_winner,
                row.is_correct,
            ),
            (None, None, None, None),
        )
        self.assertEqual(row.raw_row_bytes, first_raw_row)
        self.assertEqual(row.raw_row_sha256, sha256(first_raw_row).hexdigest())
        self.assertTrue(
            all(
                candidate.raw_row_sha256
                == sha256(candidate.raw_row_bytes).hexdigest()
                for candidate in first.rows
            )
        )

    @unittest.skipUnless(
        os.environ.get("MATCHANALYSIS_P83E_SNAPSHOT"),
        "MATCHANALYSIS_P83E_SNAPSHOT is required for pinned characterization",
    )
    def test_reduced_api_remains_field_for_field_identical(self) -> None:
        path = Path(os.environ["MATCHANALYSIS_P83E_SNAPSHOT"])
        evidence = evidence_source(path).load_evidence()
        reduced = P83eJsonlSnapshotSource(
            path,
            expected_sha256=PINNED_P83E_ARTIFACT_SHA256,
        ).load()

        self.assertEqual(reduced.artifact_sha256, evidence.raw_artifact_sha256)
        self.assertEqual(
            tuple(
                (
                    row.source_game_id,
                    row.source_prediction_version,
                    row.predicted_side,
                    row.sp_fip_delta,
                )
                for row in reduced.rows
            ),
            tuple(
                (
                    row.legacy_game_id,
                    row.source_prediction_version,
                    row.predicted_side,
                    row.sp_fip_delta,
                )
                for row in evidence.rows
            ),
        )
        self.assertEqual(reduced.rows_with_observed_outcomes, 0)


if __name__ == "__main__":
    unittest.main()
