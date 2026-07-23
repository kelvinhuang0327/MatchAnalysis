"""Characterize the pinned P83E snapshot quarantine boundary."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.ports.legacy_prediction_source import (
    NULL_OUTCOME_PLACEHOLDER_FIELDS,
)
from match_analysis.application.use_cases.import_legacy_prediction_snapshot import (
    NULL_PLACEHOLDER_LIMITATION,
    PINNED_P83E_ARTIFACT_SHA256,
    SEMANTIC_FINGERPRINT_FIELDS,
    SEMANTIC_FINGERPRINT_BOOLEAN_ENCODING,
    import_legacy_prediction_snapshot,
)
from match_analysis.infrastructure.legacy_betting_pool.p83e_jsonl import (
    P83eJsonlSnapshotSource,
    P83eSnapshotValidationError,
    STOP_OBSERVED_OUTCOME,
)


def valid_row(game_id: str = "mlb_2026_822733") -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_date": "2026-05-21",
        "season": 2026,
        "home_team": "Washington Nationals",
        "away_team": "New York Mets",
        "home_sp_fip": 3.033,
        "away_sp_fip": 3.0625,
        "sp_fip_delta": -0.0295,
        "abs_sp_fip_delta": 0.0295,
        "model_probability": 0.504425,
        "predicted_side": "home",
        "source_prediction_version": "p84b_diagnostic_baseline_v1",
        "rule_primary_125_flag": False,
        "rule_shadow_100_flag": False,
        "tier_b_candidate_flag": False,
        "tier_a_watchlist_flag": True,
        "paper_only": True,
        "diagnostic_only": True,
        "odds_used": False,
        "market_edge_evaluated": False,
        "production_ready": False,
        "result_home_score": None,
        "result_away_score": None,
        "actual_winner": None,
        "is_correct": None,
    }


def encode_rows(*rows: dict[str, object]) -> bytes:
    return "\n".join(
        json.dumps(row, separators=(",", ":")) for row in rows
    ).encode("utf-8")


def source_for(raw: bytes) -> P83eJsonlSnapshotSource:
    return P83eJsonlSnapshotSource(
        raw,
        expected_sha256=sha256(raw).hexdigest(),
    )


class P83eSnapshotAdapterTests(unittest.TestCase):
    def test_valid_row_records_exact_null_placeholder_policy(self) -> None:
        result = import_legacy_prediction_snapshot(
            source_for(encode_rows(valid_row()))
        )

        self.assertEqual(
            result.validated_null_outcome_placeholder_fields,
            NULL_OUTCOME_PLACEHOLDER_FIELDS,
        )
        self.assertEqual(result.validated_null_outcome_placeholder_count, 4)
        self.assertEqual(result.rows_with_observed_outcomes, 0)
        self.assertEqual(result.promoted_prediction_count, 0)
        self.assertEqual(result.limitations[0], NULL_PLACEHOLDER_LIMITATION)

    def test_explicit_byte_stream_is_supported(self) -> None:
        raw = encode_rows(valid_row())
        result = import_legacy_prediction_snapshot(
            P83eJsonlSnapshotSource(
                BytesIO(raw),
                expected_sha256=sha256(raw).hexdigest(),
            )
        )

        self.assertEqual(result.row_count, 1)

    def test_each_missing_placeholder_fails_closed(self) -> None:
        for field in NULL_OUTCOME_PLACEHOLDER_FIELDS:
            row = valid_row()
            del row[field]
            with self.subTest(field=field):
                with self.assertRaises(P83eSnapshotValidationError):
                    source_for(encode_rows(row)).load()

    def test_each_non_null_placeholder_fails_as_observed_outcome(self) -> None:
        values = {
            "result_home_score": 1,
            "result_away_score": 2,
            "actual_winner": "home",
            "is_correct": True,
        }
        for field, value in values.items():
            row = valid_row()
            row[field] = value
            with self.subTest(field=field):
                with self.assertRaises(P83eSnapshotValidationError) as raised:
                    source_for(encode_rows(row)).load()
                self.assertEqual(raised.exception.code, STOP_OBSERVED_OUTCOME)

    def test_additional_outcome_like_field_is_rejected(self) -> None:
        row = valid_row()
        row["outcome_available"] = None

        with self.assertRaisesRegex(
            P83eSnapshotValidationError,
            "additional outcome-like",
        ):
            source_for(encode_rows(row)).load()

    def test_placeholders_are_omitted_from_candidate_and_fingerprint(self) -> None:
        result = import_legacy_prediction_snapshot(
            source_for(encode_rows(valid_row()))
        )
        candidate_fields = set(
            result.candidates[0].__dataclass_fields__
        )

        self.assertTrue(
            set(NULL_OUTCOME_PLACEHOLDER_FIELDS).isdisjoint(candidate_fields)
        )
        self.assertTrue(
            set(NULL_OUTCOME_PLACEHOLDER_FIELDS).isdisjoint(
                SEMANTIC_FINGERPRINT_FIELDS
            )
        )
        self.assertEqual(
            SEMANTIC_FINGERPRINT_BOOLEAN_ENCODING,
            "NO_BOOLEAN_FIELDS_IN_PROJECTION",
        )

    def test_raw_hash_covers_full_bytes_but_semantics_ignore_formatting(self) -> None:
        compact = encode_rows(valid_row())
        spaced = json.dumps(valid_row()).encode("utf-8")
        compact_result = import_legacy_prediction_snapshot(source_for(compact))
        spaced_result = import_legacy_prediction_snapshot(source_for(spaced))

        self.assertNotEqual(
            compact_result.provenance.input_fingerprint,
            spaced_result.provenance.input_fingerprint,
        )
        self.assertEqual(
            compact_result.semantic_fingerprint,
            spaced_result.semantic_fingerprint,
        )

    def test_candidate_and_result_are_immutable(self) -> None:
        result = import_legacy_prediction_snapshot(
            source_for(encode_rows(valid_row()))
        )

        with self.assertRaises(FrozenInstanceError):
            result.promoted_prediction_count = 1
        with self.assertRaises(FrozenInstanceError):
            result.candidates[0].predicted_side = "away"

    def test_invalid_source_side_version_and_governance_fail_closed(self) -> None:
        invalid_cases = (
            ("game_id", "822733"),
            ("predicted_side", "away"),
            ("source_prediction_version", "latest"),
            ("paper_only", False),
            ("diagnostic_only", False),
            ("odds_used", True),
            ("market_edge_evaluated", True),
            ("production_ready", True),
            ("rule_primary_125_flag", True),
        )
        for field, value in invalid_cases:
            row = valid_row()
            row[field] = value
            with self.subTest(field=field):
                with self.assertRaises(P83eSnapshotValidationError):
                    source_for(encode_rows(row)).load()

    def test_sha_mismatch_is_rejected_before_parsing(self) -> None:
        raw = b"not-json"
        source = P83eJsonlSnapshotSource(
            raw,
            expected_sha256="0" * 64,
        )

        with self.assertRaisesRegex(
            P83eSnapshotValidationError,
            "SHA-256",
        ):
            source.load()

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = encode_rows(valid_row())
        duplicate = raw[:-1] + b',\"game_id\":\"mlb_2026_822734\"}'

        with self.assertRaisesRegex(
            P83eSnapshotValidationError,
            "duplicate JSON key",
        ):
            source_for(duplicate).load()

    def test_malformed_and_blank_rows_are_rejected(self) -> None:
        cases = (
            b"{",
            encode_rows(valid_row()) + b"\n\n" + encode_rows(valid_row()),
        )
        for raw in cases:
            with self.subTest(raw=raw[:20]):
                with self.assertRaises(P83eSnapshotValidationError):
                    source_for(raw).load()

    def test_duplicate_id_and_unexpected_order_are_rejected(self) -> None:
        first = valid_row("mlb_2026_822733")
        duplicate = valid_row("mlb_2026_822733")
        earlier = valid_row("mlb_2026_822732")
        for rows in ((first, duplicate), (first, earlier)):
            with self.subTest(ids=[row["game_id"] for row in rows]):
                with self.assertRaises(P83eSnapshotValidationError):
                    source_for(encode_rows(*rows)).load()

    def test_decimal_scale_is_preserved_in_candidate_and_projection(self) -> None:
        raw = encode_rows(valid_row()).replace(
            b'"home_sp_fip":3.033',
            b'"home_sp_fip":3.0000',
        ).replace(
            b'"away_sp_fip":3.0625',
            b'"away_sp_fip":3.5000',
        ).replace(
            b'"sp_fip_delta":-0.0295',
            b'"sp_fip_delta":-0.5000',
        ).replace(
            b'"abs_sp_fip_delta":0.0295',
            b'"abs_sp_fip_delta":0.5000',
        ).replace(
            b'"rule_primary_125_flag":false',
            b'"rule_primary_125_flag":true',
        ).replace(
            b'"rule_shadow_100_flag":false',
            b'"rule_shadow_100_flag":true',
        ).replace(
            b'"tier_a_watchlist_flag":true',
            b'"tier_a_watchlist_flag":false',
        )
        result = import_legacy_prediction_snapshot(source_for(raw))

        self.assertEqual(
            result.candidates[0].sp_fip_delta.as_tuple().exponent,
            -4,
        )
        expected_projection_fragment = sha256(
            (
                '{"diagnostic_status":"DIAGNOSTIC_UNTIMED",'
                '"predicted_side":"home",'
                '"quarantine_reason":'
                '"MISSING_SCHEDULED_START_AND_PREDICTION_AS_OF",'
                '"source_game_id":"mlb_2026_822733",'
                '"source_prediction_version":'
                '"p84b_diagnostic_baseline_v1",'
                '"sp_fip_delta":"-0.5000"}\n'
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            result.semantic_fingerprint,
            expected_projection_fragment,
        )

    def test_two_independent_imports_are_deterministic(self) -> None:
        raw = encode_rows(valid_row())
        first = import_legacy_prediction_snapshot(source_for(raw))
        second = import_legacy_prediction_snapshot(source_for(raw))

        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(
            first.semantic_fingerprint,
            second.semantic_fingerprint,
        )
        self.assertEqual(first.limitations, second.limitations)
        self.assertEqual(
            first.validated_null_outcome_placeholder_fields,
            second.validated_null_outcome_placeholder_fields,
        )
        self.assertEqual(first.promoted_prediction_count, 0)


class PinnedP83eSnapshotCharacterizationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MATCHANALYSIS_P83E_SNAPSHOT"),
        "MATCHANALYSIS_P83E_SNAPSHOT is required for pinned characterization",
    )
    def test_full_pinned_snapshot_is_quarantined_without_promotion(self) -> None:
        snapshot_path = Path(os.environ["MATCHANALYSIS_P83E_SNAPSHOT"])
        first = import_legacy_prediction_snapshot(
            P83eJsonlSnapshotSource(
                snapshot_path,
                expected_sha256=PINNED_P83E_ARTIFACT_SHA256,
            )
        )
        second = import_legacy_prediction_snapshot(
            P83eJsonlSnapshotSource(
                snapshot_path,
                expected_sha256=PINNED_P83E_ARTIFACT_SHA256,
            )
        )

        self.assertEqual(first.provenance.input_fingerprint, PINNED_P83E_ARTIFACT_SHA256)
        self.assertEqual(first.row_count, 828)
        self.assertEqual(first.unique_id_count, 828)
        self.assertEqual(first.validated_null_outcome_placeholder_count, 4)
        self.assertEqual(first.rows_with_observed_outcomes, 0)
        self.assertEqual(first.promoted_prediction_count, 0)
        self.assertEqual(len(first.candidates), 828)
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.semantic_fingerprint, second.semantic_fingerprint)
        self.assertEqual(first.limitations, second.limitations)
        self.assertEqual(first.quarantine_counts[0][1], 828)


if __name__ == "__main__":
    unittest.main()
