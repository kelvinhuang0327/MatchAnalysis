"""Unit tests for immutable public legacy prediction evidence contracts."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.legacy_prediction_source import (
    LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION,
    LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION,
    LEGACY_PREDICTION_RAW_ROW_BYTES_RULE,
    LegacyPredictionEvidenceRow,
    LegacyPredictionEvidenceSnapshot,
    legacy_prediction_evidence_semantic_fingerprint,
    legacy_prediction_evidence_snapshot_fingerprint,
)


def evidence_row(
    game_id: str = "mlb_2026_822733",
    raw_row_bytes: bytes = b'{"game_id":"mlb_2026_822733"}',
) -> LegacyPredictionEvidenceRow:
    return LegacyPredictionEvidenceRow(
        legacy_game_id=game_id,
        game_date="2026-05-21",
        season=Decimal("2026"),
        home_team="Washington Nationals",
        away_team="New York Mets",
        home_sp_fip=Decimal("3.033"),
        away_sp_fip=Decimal("3.0625"),
        sp_fip_delta=Decimal("-0.0295"),
        abs_sp_fip_delta=Decimal("0.0295"),
        home_win_probability=Decimal("0.504425"),
        predicted_side="home",
        source_prediction_version="p84b_diagnostic_baseline_v1",
        rule_primary_125_flag=False,
        rule_shadow_100_flag=False,
        tier_b_candidate_flag=False,
        tier_a_watchlist_flag=True,
        paper_only=True,
        diagnostic_only=True,
        odds_used=False,
        market_edge_evaluated=False,
        production_ready=False,
        result_home_score=None,
        result_away_score=None,
        actual_winner=None,
        is_correct=None,
        raw_row_bytes=raw_row_bytes,
        raw_row_sha256=sha256(raw_row_bytes).hexdigest(),
    )


def evidence_snapshot(
    rows: tuple[LegacyPredictionEvidenceRow, ...],
    raw_artifact_bytes: bytes,
    **overrides: object,
) -> LegacyPredictionEvidenceSnapshot:
    values: dict[str, object] = {
        "rows": rows,
        "source_repository": "Betting-pool",
        "source_ref": "03b2fcf4de1a13ee9929afcef803d61955c9f41b",
        "source_blob": "bc06f353160656b79d21a27555758791535ea823",
        "raw_artifact_sha256": sha256(raw_artifact_bytes).hexdigest(),
        "semantic_fingerprint": (
            legacy_prediction_evidence_semantic_fingerprint(rows)
        ),
        "parser_version": LEGACY_PREDICTION_EVIDENCE_PARSER_VERSION,
        "schema_version": LEGACY_PREDICTION_EVIDENCE_SCHEMA_VERSION,
        "row_count": len(rows),
    }
    values.update(overrides)
    values["snapshot_fingerprint"] = (
        legacy_prediction_evidence_snapshot_fingerprint(
            rows=values["rows"],
            source_repository=values["source_repository"],
            source_ref=values["source_ref"],
            source_blob=values["source_blob"],
            raw_artifact_sha256=values["raw_artifact_sha256"],
            semantic_fingerprint=values["semantic_fingerprint"],
            parser_version=values["parser_version"],
            schema_version=values["schema_version"],
            row_count=values["row_count"],
        )
    )
    return LegacyPredictionEvidenceSnapshot(
        **values,
        raw_artifact_bytes=raw_artifact_bytes,
    )


class LegacyPredictionEvidenceContractTests(unittest.TestCase):
    def test_row_exposes_exact_values_bytes_hash_and_home_probability(self) -> None:
        row = evidence_row()

        self.assertEqual(row.season, Decimal("2026"))
        self.assertEqual(row.home_win_probability, Decimal("0.504425"))
        self.assertEqual(
            (
                row.result_home_score,
                row.result_away_score,
                row.actual_winner,
                row.is_correct,
            ),
            (None, None, None, None),
        )
        self.assertEqual(
            row.raw_row_sha256,
            sha256(row.raw_row_bytes).hexdigest(),
        )
        self.assertIn("P(home wins)", LegacyPredictionEvidenceRow.__doc__)
        self.assertIn("LF or CRLF", LEGACY_PREDICTION_RAW_ROW_BYTES_RULE)

    def test_row_and_snapshot_are_frozen(self) -> None:
        row = evidence_row()
        snapshot = evidence_snapshot((row,), row.raw_row_bytes + b"\n")

        with self.assertRaises(FrozenInstanceError):
            row.predicted_side = "away"
        with self.assertRaises(FrozenInstanceError):
            snapshot.source_ref = "latest"

    def test_snapshot_accepts_ordered_rows_and_explicit_provenance(self) -> None:
        first = evidence_row()
        second_raw = b'{"game_id":"mlb_2026_822734"}'
        second = replace(
            evidence_row("mlb_2026_822734", second_raw),
            predicted_side="away",
            sp_fip_delta=Decimal("0.7472"),
        )
        raw = first.raw_row_bytes + b"\r\n" + second.raw_row_bytes + b"\n"

        snapshot = evidence_snapshot((first, second), raw)

        self.assertEqual(snapshot.rows, (first, second))
        self.assertEqual(snapshot.row_count, 2)
        self.assertEqual(snapshot.source_repository, "Betting-pool")
        self.assertEqual(
            snapshot.schema_version,
            "legacy_prediction_evidence_snapshot_v1",
        )

    def test_row_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw_row_sha256"):
            replace(evidence_row(), raw_row_sha256="0" * 64)

    def test_snapshot_rejects_count_duplicates_and_row_order_drift(self) -> None:
        row = evidence_row()
        raw = row.raw_row_bytes + b"\n"
        with self.assertRaisesRegex(ValueError, "row_count"):
            evidence_snapshot((row,), raw, row_count=2)

        duplicate_raw = row.raw_row_bytes + b"\n" + row.raw_row_bytes + b"\n"
        with self.assertRaisesRegex(ValueError, "unique"):
            evidence_snapshot((row, row), duplicate_raw)

        second_raw = b'{"game_id":"mlb_2026_822734"}'
        second = evidence_row("mlb_2026_822734", second_raw)
        ordered_raw = row.raw_row_bytes + b"\n" + second.raw_row_bytes + b"\n"
        with self.assertRaisesRegex(ValueError, "row order"):
            evidence_snapshot((second, row), ordered_raw)

    def test_snapshot_rejects_artifact_semantic_and_snapshot_hash_drift(
        self,
    ) -> None:
        row = evidence_row()
        raw = row.raw_row_bytes + b"\n"
        with self.assertRaisesRegex(ValueError, "raw artifact"):
            evidence_snapshot((row,), raw, raw_artifact_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "semantic fingerprint"):
            evidence_snapshot((row,), raw, semantic_fingerprint="0" * 64)

        valid = evidence_snapshot((row,), raw)
        with self.assertRaisesRegex(ValueError, "snapshot fingerprint"):
            replace(
                valid,
                snapshot_fingerprint="0" * 64,
                raw_artifact_bytes=raw,
            )


if __name__ == "__main__":
    unittest.main()
