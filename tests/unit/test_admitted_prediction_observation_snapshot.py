"""Unit tests for admitted prediction observation snapshot builder."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.build_admitted_prediction_observation_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    EXPLICIT_SNAPSHOT_CLAIMS,
    AdmittedPredictionObservationRow,
    AdmittedPredictionObservationSnapshotResult,
    build_admitted_prediction_observation_snapshot,
    _compute_result_row_fingerprint,
    _compute_snapshot_row_fingerprint,
    _compute_snapshot_fingerprint,
)


class SnapshotSchemaTests(unittest.TestCase):
    def test_schema_version_is_p15c(self) -> None:
        self.assertEqual(
            SNAPSHOT_SCHEMA_VERSION,
            "p15c.admitted_prediction_observation_snapshot.v1",
        )

    def test_explicit_claims_contain_all_required_safety_claims(self) -> None:
        required_claims = {
            "betting_claim",
            "db_written",
            "deployed",
            "legacy_rows_admitted",
            "network_called",
            "outcomes_attached",
            "provider_called",
        }
        self.assertEqual(set(EXPLICIT_SNAPSHOT_CLAIMS.keys()), required_claims)
        for key, value in EXPLICIT_SNAPSHOT_CLAIMS.items():
            self.assertIs(value, False, f"Claim {key} should be False")


class ResultRowFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic(self) -> None:
        row = {"admission_status": "ADMITTED", "reason": None, "observation": {"id": "abc"}}
        fp1 = _compute_result_row_fingerprint(row)
        fp2 = _compute_result_row_fingerprint(row)
        self.assertEqual(fp1, fp2)

    def test_fingerprint_is_sha256_hex(self) -> None:
        row = {"admission_status": "ADMITTED", "reason": None}
        fp = _compute_result_row_fingerprint(row)
        self.assertEqual(len(fp), 64)
        int(fp, 16)  # Should not raise

    def test_different_rows_produce_different_fingerprints(self) -> None:
        row1 = {"admission_status": "ADMITTED", "reason": None}
        row2 = {"admission_status": "REJECTED", "reason": "MISSING"}
        self.assertNotEqual(
            _compute_result_row_fingerprint(row1),
            _compute_result_row_fingerprint(row2),
        )

    def test_key_order_does_not_affect_fingerprint(self) -> None:
        row1 = {"a": 1, "b": 2}
        row2 = {"b": 2, "a": 1}
        self.assertEqual(
            _compute_result_row_fingerprint(row1),
            _compute_result_row_fingerprint(row2),
        )


class SnapshotRowFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic(self) -> None:
        fp1 = _compute_snapshot_row_fingerprint("id1", "fp1", {"key": "val"})
        fp2 = _compute_snapshot_row_fingerprint("id1", "fp1", {"key": "val"})
        self.assertEqual(fp1, fp2)

    def test_different_inputs_produce_different_fingerprints(self) -> None:
        fp1 = _compute_snapshot_row_fingerprint("id1", "fp1", {"key": "val"})
        fp2 = _compute_snapshot_row_fingerprint("id2", "fp1", {"key": "val"})
        self.assertNotEqual(fp1, fp2)


class SnapshotFingerprintTests(unittest.TestCase):
    def test_empty_rows_have_deterministic_fingerprint(self) -> None:
        fp1 = _compute_snapshot_fingerprint(())
        fp2 = _compute_snapshot_fingerprint(())
        self.assertEqual(fp1, fp2)

    def test_fingerprint_depends_on_row_content(self) -> None:
        row1 = AdmittedPredictionObservationRow(
            prediction_observation_id="a",
            source_result_row_fingerprint="fp1",
            observation={"key": "val"},
            snapshot_row_fingerprint="sfp1",
        )
        row2 = AdmittedPredictionObservationRow(
            prediction_observation_id="b",
            source_result_row_fingerprint="fp2",
            observation={"key": "val2"},
            snapshot_row_fingerprint="sfp2",
        )
        fp_single = _compute_snapshot_fingerprint((row1,))
        fp_both = _compute_snapshot_fingerprint((row1, row2))
        self.assertNotEqual(fp_single, fp_both)


def _make_admitted_row(
    request_index: int,
    obs_id: str,
    source_prediction_id: str = "PRED_001",
    provider_game_id: str = "888001",
    game_number: int = 1,
) -> dict:
    """Helper to create an ADMITTED result row dict."""
    return {
        "request_index": request_index,
        "admission_status": "ADMITTED",
        "reason": None,
        "observation": {
            "prediction_observation_id": obs_id,
            "source_prediction_id": source_prediction_id,
            "model_id": "model_v1",
            "market_id": "moneyline",
            "selection": "HOME",
            "model_probability": "0.58",
            "line_value": "-110",
            "push_policy": "PUSH_VOID",
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": provider_game_id,
            "game_number": game_number,
            "source_schedule_observation_id": "aaaa",
            "prediction_generated_at_utc": "2026-04-05T11:00:00Z",
            "response_received_at_utc": "2026-04-05T11:00:01Z",
            "ingested_at_utc": "2026-04-05T11:00:02Z",
            "scheduled_start_utc": "2026-04-05T19:00:00Z",
        },
    }


def _make_rejected_row(request_index: int, reason: str) -> dict:
    """Helper to create a REJECTED result row dict."""
    return {
        "request_index": request_index,
        "admission_status": "REJECTED",
        "reason": reason,
        "observation": None,
    }


def _compute_result_set_fingerprint(rows: list[dict]) -> str:
    """Compute the result-set fingerprint matching P15B1 logic."""
    parts = []
    for row in rows:
        status = row["admission_status"]
        reason = row.get("reason") or ""
        obs_id = ""
        if row.get("observation") is not None:
            obs_id = row["observation"].get("prediction_observation_id", "")
        parts.append(f"{status}:{reason}:{obs_id}\n")
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _make_source_artifacts(
    admitted_rows: list[dict],
    rejected_rows: list[dict],
) -> tuple[bytes, bytes]:
    """Create valid P15B1 results.jsonl and summary.json bytes."""
    all_rows = admitted_rows + rejected_rows
    fingerprint = _compute_result_set_fingerprint(all_rows)

    results_lines = [
        json.dumps(row, separators=(",", ":")) for row in all_rows
    ]
    results_text = "\n".join(results_lines) + ("\n" if results_lines else "")
    results_bytes = results_text.encode("utf-8")

    summary = {
        "request_count": len(all_rows),
        "admitted_count": len(admitted_rows),
        "rejected_count": len(rejected_rows),
        "result_set_fingerprint": fingerprint,
        "claims": {
            "legacy_rows_admitted": False,
            "provider_called": False,
            "db_written": False,
            "deployed": False,
            "betting_claim": False,
        },
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_bytes = summary_text.encode("utf-8")

    return results_bytes, summary_bytes


class BuildSnapshotTests(unittest.TestCase):
    def test_basic_admitted_snapshot(self) -> None:
        admitted = [
            _make_admitted_row(1, "obs_id_b"),
            _make_admitted_row(2, "obs_id_a"),
        ]
        rejected = [_make_rejected_row(3, "MISSING")]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, rejected)

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        self.assertEqual(result.schema_version, SNAPSHOT_SCHEMA_VERSION)
        self.assertEqual(len(result.snapshot_rows), 2)
        self.assertEqual(result.source_row_count, 3)
        self.assertEqual(result.source_admitted_count, 2)
        self.assertEqual(result.source_rejected_count, 1)
        # Snapshot rows are sorted by prediction_observation_id
        self.assertEqual(
            result.snapshot_rows[0].prediction_observation_id, "obs_id_a"
        )
        self.assertEqual(
            result.snapshot_rows[1].prediction_observation_id, "obs_id_b"
        )

    def test_snapshot_preserves_observation_fields_exactly(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, [])

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        obs = result.snapshot_rows[0].observation
        self.assertEqual(obs["prediction_observation_id"], "obs_id_1")
        self.assertEqual(obs["model_probability"], "0.58")
        self.assertEqual(obs["line_value"], "-110")
        self.assertEqual(obs["game_number"], 1)
        self.assertEqual(obs["provider_namespace"], "MLB_STATS_API")

    def test_rejected_rows_never_appear_in_snapshot(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        rejected = [_make_rejected_row(2, "MISSING"), _make_rejected_row(3, "BAD")]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, rejected)

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        self.assertEqual(len(result.snapshot_rows), 1)
        self.assertEqual(
            result.snapshot_rows[0].prediction_observation_id, "obs_id_1"
        )

    def test_duplicate_observation_ids_fail_closed(self) -> None:
        admitted = [
            _make_admitted_row(1, "same_id"),
            _make_admitted_row(2, "same_id"),
        ]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, [])

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("Duplicate prediction_observation_id", str(ctx.exception))

    def test_legacy_rows_admitted_true_fails(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, _ = _make_source_artifacts(admitted, [])

        summary = {
            "request_count": 1,
            "admitted_count": 1,
            "rejected_count": 0,
            "result_set_fingerprint": _compute_result_set_fingerprint(admitted),
            "claims": {"legacy_rows_admitted": True},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("legacy_rows_admitted", str(ctx.exception))

    def test_admitted_count_mismatch_fails(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, _ = _make_source_artifacts(admitted, [])

        summary = {
            "request_count": 1,
            "admitted_count": 99,  # Wrong
            "rejected_count": 0,
            "result_set_fingerprint": _compute_result_set_fingerprint(admitted),
            "claims": {"legacy_rows_admitted": False},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("Admitted count mismatch", str(ctx.exception))

    def test_result_set_fingerprint_mismatch_fails(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, _ = _make_source_artifacts(admitted, [])

        summary = {
            "request_count": 1,
            "admitted_count": 1,
            "rejected_count": 0,
            "result_set_fingerprint": "wrong_fingerprint",
            "claims": {"legacy_rows_admitted": False},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("Result-set fingerprint mismatch", str(ctx.exception))

    def test_admitted_row_with_null_observation_fails(self) -> None:
        row = {
            "request_index": 1,
            "admission_status": "ADMITTED",
            "reason": None,
            "observation": None,
        }
        fingerprint = _compute_result_set_fingerprint([row])
        results_bytes = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")

        summary = {
            "request_count": 1,
            "admitted_count": 1,
            "rejected_count": 0,
            "result_set_fingerprint": fingerprint,
            "claims": {"legacy_rows_admitted": False},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("null observation", str(ctx.exception))

    def test_rejected_row_with_non_null_observation_fails(self) -> None:
        admitted = []
        rejected_with_obs = {
            "request_index": 1,
            "admission_status": "REJECTED",
            "reason": "MISSING",
            "observation": {"prediction_observation_id": "leak"},
        }
        fingerprint = _compute_result_set_fingerprint([rejected_with_obs])
        results_bytes = (json.dumps(rejected_with_obs, separators=(",", ":")) + "\n").encode("utf-8")

        summary = {
            "request_count": 1,
            "admitted_count": 0,
            "rejected_count": 1,
            "result_set_fingerprint": fingerprint,
            "claims": {"legacy_rows_admitted": False},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("non-null observation", str(ctx.exception))

    def test_missing_summary_fields_fail(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, _ = _make_source_artifacts(admitted, [])

        for missing_field in ("result_set_fingerprint", "admitted_count", "rejected_count", "claims"):
            summary = {
                "request_count": 1,
                "admitted_count": 1,
                "rejected_count": 0,
                "result_set_fingerprint": _compute_result_set_fingerprint(admitted),
                "claims": {"legacy_rows_admitted": False},
            }
            del summary[missing_field]
            summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

            with self.assertRaises(ValueError) as ctx:
                build_admitted_prediction_observation_snapshot(
                    results_bytes=results_bytes,
                    summary_bytes=summary_bytes,
                )
            self.assertIn(f"missing {missing_field}", str(ctx.exception).lower())

    def test_snapshot_fingerprint_is_deterministic_across_source_row_order(self) -> None:
        row_a = _make_admitted_row(1, "obs_a")
        row_b = _make_admitted_row(2, "obs_b")
        rej = _make_rejected_row(3, "MISSING")

        # Order 1: A, B, rejected
        results_bytes_1, summary_bytes_1 = _make_source_artifacts([row_a, row_b], [rej])
        result_1 = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes_1,
            summary_bytes=summary_bytes_1,
        )

        # Order 2: B, A, rejected (shuffled admitted order)
        all_rows_2 = [row_b, row_a, rej]
        fp_1 = _compute_result_set_fingerprint([row_a, row_b, rej])
        results_lines_2 = [json.dumps(r, separators=(",", ":")) for r in all_rows_2]
        results_bytes_2 = ("\n".join(results_lines_2) + "\n").encode("utf-8")
        summary_2 = {
            "request_count": 3,
            "admitted_count": 2,
            "rejected_count": 1,
            "result_set_fingerprint": fp_1,
            "claims": {"legacy_rows_admitted": False, "provider_called": False, "db_written": False, "deployed": False, "betting_claim": False},
        }
        summary_bytes_2 = (json.dumps(summary_2, indent=2, sort_keys=True) + "\n").encode("utf-8")
        result_2 = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes_2,
            summary_bytes=summary_bytes_2,
        )

        # Snapshot fingerprint and sorted IDs should be identical
        self.assertEqual(result_1.snapshot_fingerprint, result_2.snapshot_fingerprint)
        self.assertEqual(
            [r.prediction_observation_id for r in result_1.snapshot_rows],
            [r.prediction_observation_id for r in result_2.snapshot_rows],
        )

    def test_snapshot_observation_ids_are_unique_and_sorted(self) -> None:
        admitted = [
            _make_admitted_row(1, "zzz"),
            _make_admitted_row(2, "aaa"),
            _make_admitted_row(3, "mmm"),
        ]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, [])

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        ids = [r.prediction_observation_id for r in result.snapshot_rows]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_claims_match_explicit_snapshot_claims(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, [])

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        self.assertEqual(result.claims, EXPLICIT_SNAPSHOT_CLAIMS)

    def test_duplicate_json_keys_in_results_rejected(self) -> None:
        # Manually craft a JSONL line with duplicate keys
        bad_line = '{"admission_status":"ADMITTED","admission_status":"REJECTED","reason":null,"observation":null}'
        results_bytes = (bad_line + "\n").encode("utf-8")

        summary = {
            "request_count": 1,
            "admitted_count": 1,
            "rejected_count": 0,
            "result_set_fingerprint": "any",
            "claims": {"legacy_rows_admitted": False},
        }
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")

        with self.assertRaises(ValueError) as ctx:
            build_admitted_prediction_observation_snapshot(
                results_bytes=results_bytes,
                summary_bytes=summary_bytes,
            )
        self.assertIn("Duplicate JSON key", str(ctx.exception))

    def test_source_result_set_fingerprint_preserved_in_output(self) -> None:
        admitted = [_make_admitted_row(1, "obs_id_1")]
        results_bytes, summary_bytes = _make_source_artifacts(admitted, [])

        result = build_admitted_prediction_observation_snapshot(
            results_bytes=results_bytes,
            summary_bytes=summary_bytes,
        )

        expected_fp = _compute_result_set_fingerprint(admitted)
        self.assertEqual(result.source_result_set_fingerprint, expected_fp)


class SnapshotRowDataclassTests(unittest.TestCase):
    def test_snapshot_row_is_frozen(self) -> None:
        row = AdmittedPredictionObservationRow(
            prediction_observation_id="id1",
            source_result_row_fingerprint="fp1",
            observation={"key": "val"},
            snapshot_row_fingerprint="sfp1",
        )
        with self.assertRaises(AttributeError):
            row.prediction_observation_id = "new_id"  # type: ignore[misc]

    def test_snapshot_result_is_frozen(self) -> None:
        result = AdmittedPredictionObservationSnapshotResult(
            schema_version="v1",
            source_results_sha256="a",
            source_summary_sha256="b",
            source_result_set_fingerprint="c",
            source_row_count=0,
            source_admitted_count=0,
            source_rejected_count=0,
            snapshot_rows=(),
            snapshot_fingerprint="d",
            claims={},
        )
        with self.assertRaises(AttributeError):
            result.schema_version = "new"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
