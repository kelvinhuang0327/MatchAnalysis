"""Focused tests for the P66A versioned official-date correction materialization."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

from match_analysis.application.use_cases.p50c_prediction_run_ledger import (
    CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
    apply_official_date_corrections,
    compute_forward_prediction_summary,
    read_json_object,
    read_jsonl_objects,
)
from match_analysis.application.use_cases.p66a_official_date_correction import (
    P66A_RECEIPT_SCHEMA,
    P66A_ROLLBACK_MODE,
    execute_p66a_official_date_correction,
    resolve_provider_official_dates,
)


def _raw_record(
    *,
    ledger_id: str,
    game_pk: int,
    scheduled_start_utc: str,
    official_date: str,
) -> dict:
    return {
        "ledger_record_id": ledger_id,
        "schema_version": "p50c.forward_prediction_ledger_record.v1",
        "run_id": "run_001",
        "run_classification": CLASSIFICATION_PROSPECTIVE_FORWARD_PREDICTION,
        "market_id": "moneyline",
        "prediction_identity": {
            "workflow_prediction_id": f"prediction_{game_pk}",
            "prediction_fingerprint": f"fingerprint_{game_pk}",
        },
        "game_identity": {
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": str(game_pk),
            "game_pk": game_pk,
            "game_number": 1,
            "scheduled_start_utc": scheduled_start_utc,
            "official_date": official_date,
        },
        "model_identity": {
            "model_probability": "0.6",
        },
        "evaluation": {
            "is_correct": True,
            "correctness_target": 1,
            "brier_component": "0.16",
            "log_loss_component": "0.510825623766",
        },
    }


class P66AOfficialDateCorrectionTests(unittest.TestCase):
    def test_default_provider_opener_is_used_when_no_override_is_supplied(self) -> None:
        raw_records = [
            _raw_record(
                ledger_id="ledger_001",
                game_pk=824154,
                scheduled_start_utc="2026-08-19T00:10:00Z",
                official_date="2026-08-19",
            )
        ]
        provider_payload = {
            "dates": [
                {
                    "date": "2026-08-18",
                    "games": [
                        {
                            "gamePk": 824154,
                            "officialDate": "2026-08-18",
                            "gameDate": "2026-08-19T00:10:00Z",
                        }
                    ],
                }
            ]
        }

        def fetch_without_opener(url: str, *, query: dict, timeout: int):
            return json.dumps(provider_payload).encode("utf-8"), url

        with patch(
            "match_analysis.application.use_cases.p66a_official_date_correction.fetch_json_bytes",
            side_effect=fetch_without_opener,
        ):
            resolved, evidence = resolve_provider_official_dates(raw_records)

        self.assertEqual(resolved, {824154: "2026-08-18"})
        self.assertEqual(evidence["resolved_row_count"], 1)

    def test_materialization_binds_raw_authority_and_preserves_metrics(self) -> None:
        raw_records = [
            _raw_record(
                ledger_id="ledger_001",
                game_pk=824154,
                scheduled_start_utc="2026-08-19T00:10:00Z",
                official_date="2026-08-19",
            ),
            _raw_record(
                ledger_id="ledger_002",
                game_pk=824155,
                scheduled_start_utc="2026-08-18T19:00:00Z",
                official_date="2026-08-18",
            ),
        ]
        provider_payload = {
            "dates": [
                {
                    "date": "2026-08-18",
                    "games": [
                        {
                            "gamePk": 824154,
                            "officialDate": "2026-08-18",
                            "gameDate": "2026-08-19T00:10:00Z",
                        },
                        {
                            "gamePk": 824155,
                            "officialDate": "2026-08-18",
                            "gameDate": "2026-08-18T19:00:00Z",
                        },
                    ],
                }
            ]
        }

        def opener(request: Request, timeout: int) -> bytes:
            return json.dumps(provider_payload).encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="p66a-correction-") as tmpdir:
            root = Path(tmpdir)
            ledger_root = (
                root
                / "report"
                / "p50c_prospective_prediction_shadow_ledger"
                / "ledger"
            )
            ledger_root.mkdir(parents=True)
            raw_path = ledger_root / "prediction_forward_ledger.jsonl"
            raw_bytes = b"".join(
                (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
                for row in raw_records
            )
            raw_path.write_bytes(raw_bytes)
            summary = compute_forward_prediction_summary(raw_records)
            (ledger_root / "forward_prediction_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            calls = 0

            def compute_with_cohort_metric_drift(records):
                nonlocal calls
                calls += 1
                computed = compute_forward_prediction_summary(records)
                if calls == 2:
                    computed["game_type_cohorts"]["R"]["brier_score"] = "0.99"
                return computed

            with patch(
                "match_analysis.application.use_cases.p66a_official_date_correction.compute_forward_prediction_summary",
                side_effect=compute_with_cohort_metric_drift,
            ):
                with self.assertRaisesRegex(RuntimeError, "P66A_NON_DATE_METRIC_INVARIANCE_FAILED"):
                    execute_p66a_official_date_correction(root, opener=opener)
            self.assertEqual(raw_path.read_bytes(), raw_bytes)
            self.assertFalse((ledger_root.parent / "corrections").exists())

            result = execute_p66a_official_date_correction(root, opener=opener)

            self.assertEqual(result.provider_resolution_count, 2)
            self.assertEqual(result.correction_count, 1)
            self.assertEqual(result.raw_row_count, 2)
            self.assertEqual(result.raw_ledger_sha256, sha256(raw_bytes).hexdigest())
            self.assertEqual(raw_path.read_bytes(), raw_bytes)
            self.assertEqual(result.effective_grouping, {"2026-08-18/moneyline": 2})

            corrections = read_jsonl_objects(result.correction_path)
            self.assertEqual(corrections[0]["ledger_id"], "ledger_001")
            self.assertEqual(corrections[0]["original_official_date"], "2026-08-19")
            self.assertEqual(corrections[0]["corrected_official_date"], "2026-08-18")
            self.assertEqual(
                apply_official_date_corrections(raw_records, corrections),
                apply_official_date_corrections(raw_records, corrections),
            )

            receipt = read_json_object(result.receipt_path)
            self.assertEqual(receipt["receipt_schema_version"], P66A_RECEIPT_SCHEMA)
            self.assertEqual(receipt["raw_ledger_row_count"], 2)
            self.assertEqual(receipt["correction_record_count"], 1)
            self.assertEqual(receipt["rollback_mode"], P66A_ROLLBACK_MODE)

            effective_summary = read_json_object(result.summary_path)
            self.assertEqual(effective_summary["first_target_date"], "2026-08-18")
            self.assertEqual(effective_summary["last_target_date"], "2026-08-18")
            self.assertEqual(summary["game_type_cohorts"]["R"]["last_target_date"], "2026-08-19")
            self.assertEqual(effective_summary["game_type_cohorts"]["R"]["last_target_date"], "2026-08-18")
            date_fields = {"first_target_date", "last_target_date"}
            self.assertEqual(
                {key: value for key, value in summary["game_type_cohorts"]["R"].items() if key not in date_fields},
                {key: value for key, value in effective_summary["game_type_cohorts"]["R"].items() if key not in date_fields},
            )
            for metric in (
                "accuracy",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
                "correct_count",
                "incorrect_count",
                "PREDICTION_FORWARD_SAMPLE_COUNT",
            ):
                self.assertEqual(effective_summary[metric], summary[metric])


if __name__ == "__main__":
    unittest.main()
