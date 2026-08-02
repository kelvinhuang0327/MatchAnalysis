"""Unit tests for the immutable PredictionSourceObservation contract."""

import dataclasses
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.prediction_source_observation import (
    PredictionSourceObservation,
    compute_prediction_observation_id,
)


def _schedule_observation_id(tag: str) -> str:
    return sha256(f"schedule-observation:{tag}".encode("utf-8")).hexdigest()


def evidence_fields() -> dict:
    return {
        "source_prediction_id": "prediction-0001",
        "model_id": "model-v1",
        "market_id": "MONEYLINE",
        "selection": "home",
        "model_probability": Decimal("0.6321"),
        "line_value": Decimal("-1.5"),
        "push_policy": "NO_PUSH",
        "provider_namespace": "MLB_STATS_API",
        "provider_game_id": "777001",
        "game_number": 1,
        "source_schedule_observation_id": _schedule_observation_id("777001:1"),
        "prediction_generated_at_utc": datetime(
            2031, 6, 1, tzinfo=timezone.utc
        ),
        "response_received_at_utc": datetime(2031, 6, 1, tzinfo=timezone.utc),
        "ingested_at_utc": datetime(2031, 6, 1, tzinfo=timezone.utc),
        "scheduled_start_utc": datetime(2031, 7, 1, 12, tzinfo=timezone.utc),
    }


def base_fields() -> dict:
    fields = evidence_fields()
    fields["prediction_observation_id"] = compute_prediction_observation_id(
        **fields
    )
    return fields


def id_for(fields: dict) -> str:
    evidence = {
        key: value
        for key, value in fields.items()
        if key != "prediction_observation_id"
    }
    return compute_prediction_observation_id(**evidence)


def build(**overrides) -> PredictionSourceObservation:
    fields = base_fields()
    fields.update(overrides)
    return PredictionSourceObservation(**fields)


class PredictionSourceObservationContractTests(unittest.TestCase):
    def test_one_fully_explicit_synthetic_record_is_constructed(self) -> None:
        observation = build()

        self.assertEqual(observation.source_prediction_id, "prediction-0001")
        self.assertEqual(observation.game_number, 1)
        self.assertEqual(observation.model_probability, Decimal("0.6321"))
        self.assertEqual(
            observation.prediction_observation_id,
            id_for(base_fields()),
        )

    def test_observation_is_immutable(self) -> None:
        observation = build()
        with self.assertRaises(FrozenInstanceError):
            observation.game_number = 2

    def test_repeated_construction_is_deterministic(self) -> None:
        first = build()
        second = build()
        self.assertEqual(first, second)
        self.assertEqual(
            first.prediction_observation_id, second.prediction_observation_id
        )

    def test_prediction_observation_id_must_match_canonical_projection(
        self,
    ) -> None:
        fields = base_fields()
        fields["prediction_observation_id"] = sha256(b"wrong").hexdigest()
        with self.assertRaises(ValueError):
            PredictionSourceObservation(**fields)

    def test_identity_token_fields_reject_empty_and_untrimmed_values(
        self,
    ) -> None:
        token_fields = (
            "source_prediction_id",
            "model_id",
            "market_id",
            "selection",
            "push_policy",
            "provider_namespace",
            "provider_game_id",
        )
        for field_name in token_fields:
            for bad_value in ("", "   ", " padded", "padded "):
                with self.subTest(field=field_name, value=repr(bad_value)):
                    fields = base_fields()
                    fields[field_name] = bad_value
                    fields["prediction_observation_id"] = "0" * 64
                    with self.assertRaises(ValueError):
                        PredictionSourceObservation(**fields)

    def test_game_number_is_required_positive_integer_with_no_default(
        self,
    ) -> None:
        game_number_field = PredictionSourceObservation.__dataclass_fields__[
            "game_number"
        ]
        self.assertIs(game_number_field.default, dataclasses.MISSING)
        self.assertIs(game_number_field.default_factory, dataclasses.MISSING)

        for bad_value in (0, -1, True, 1.0, "1"):
            with self.subTest(value=bad_value):
                fields = base_fields()
                fields["game_number"] = bad_value
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises((TypeError, ValueError)):
                    PredictionSourceObservation(**fields)

    def test_model_probability_must_be_a_decimal_within_zero_and_one(
        self,
    ) -> None:
        for bad_value in (
            Decimal("-0.0001"),
            Decimal("1.0001"),
            0.5,
            "0.5",
        ):
            with self.subTest(value=bad_value):
                fields = base_fields()
                fields["model_probability"] = bad_value
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises((TypeError, ValueError)):
                    PredictionSourceObservation(**fields)

        for boundary_value in (Decimal("0"), Decimal("1")):
            with self.subTest(value=boundary_value):
                fields = base_fields()
                fields["model_probability"] = boundary_value
                fields["prediction_observation_id"] = id_for(fields)
                observation = PredictionSourceObservation(**fields)
                self.assertEqual(observation.model_probability, boundary_value)

    def test_line_value_must_be_a_finite_decimal(self) -> None:
        for bad_value in (Decimal("NaN"), Decimal("Infinity"), 1.5, "1.5"):
            with self.subTest(value=bad_value):
                fields = base_fields()
                fields["line_value"] = bad_value
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises((TypeError, ValueError)):
                    PredictionSourceObservation(**fields)

    def test_source_schedule_observation_id_must_be_lowercase_sha256(
        self,
    ) -> None:
        for bad_value in ("not-a-hash", "0" * 63, ("0" * 63) + "G"):
            with self.subTest(value=bad_value):
                fields = base_fields()
                fields["source_schedule_observation_id"] = bad_value
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises(ValueError):
                    PredictionSourceObservation(**fields)

    def test_timestamps_must_be_timezone_aware_utc(self) -> None:
        naive = datetime(2031, 6, 1)
        for field_name in (
            "prediction_generated_at_utc",
            "response_received_at_utc",
            "ingested_at_utc",
            "scheduled_start_utc",
        ):
            with self.subTest(field=field_name):
                fields = base_fields()
                fields[field_name] = naive
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises((TypeError, ValueError)):
                    PredictionSourceObservation(**fields)

    def test_generated_received_ingested_may_be_equal(self) -> None:
        moment = datetime(2031, 6, 15, tzinfo=timezone.utc)
        fields = base_fields()
        fields["prediction_generated_at_utc"] = moment
        fields["response_received_at_utc"] = moment
        fields["ingested_at_utc"] = moment
        fields["prediction_observation_id"] = id_for(fields)
        observation = PredictionSourceObservation(**fields)
        self.assertEqual(observation.ingested_at_utc, moment)

    def test_timestamp_order_violations_are_rejected(self) -> None:
        scheduled_start = datetime(2031, 7, 1, 12, tzinfo=timezone.utc)
        earlier = datetime(2031, 6, 1, tzinfo=timezone.utc)
        later = datetime(2031, 6, 2, tzinfo=timezone.utc)

        out_of_order_cases = {
            "generated_after_received": {
                "prediction_generated_at_utc": later,
                "response_received_at_utc": earlier,
                "ingested_at_utc": earlier,
            },
            "received_after_ingested": {
                "prediction_generated_at_utc": earlier,
                "response_received_at_utc": later,
                "ingested_at_utc": earlier,
            },
            "ingested_equals_scheduled_start": {
                "prediction_generated_at_utc": earlier,
                "response_received_at_utc": earlier,
                "ingested_at_utc": scheduled_start,
            },
            "ingested_after_scheduled_start": {
                "prediction_generated_at_utc": earlier,
                "response_received_at_utc": earlier,
                "ingested_at_utc": scheduled_start + (later - earlier),
            },
        }
        for case_name, overrides in out_of_order_cases.items():
            with self.subTest(case=case_name):
                fields = base_fields()
                fields.update(overrides)
                fields["scheduled_start_utc"] = scheduled_start
                fields["prediction_observation_id"] = "0" * 64
                with self.assertRaises(ValueError):
                    PredictionSourceObservation(**fields)


if __name__ == "__main__":
    unittest.main()
