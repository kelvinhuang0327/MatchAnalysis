"""Unit tests for the immutable P19A Moneyline feature snapshot."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from match_analysis.core.identity import MatchIdentity


AS_OF = datetime(2026, 4, 5, 10, tzinfo=timezone.utc)
START = datetime(2026, 4, 5, 12, tzinfo=timezone.utc)
IDENTITY = MatchIdentity(
    sport="baseball",
    league="MLB",
    season=2026,
    canonical_game_id="P19A_CANONICAL_GAME_0001",
    home_participant="HOME_TEAM",
    away_participant="AWAY_TEAM",
)


def provenance() -> tuple[MoneylineFeatureProvenance, ...]:
    return (
        MoneylineFeatureProvenance(
            field_name="recent_win_rate_delta",
            source_id="recent-source",
            source_kind="fixture",
            observed_as_of_utc=datetime(2026, 4, 5, 9, 30, tzinfo=timezone.utc),
            source_fingerprint=sha256(b"recent").hexdigest(),
        ),
        MoneylineFeatureProvenance(
            field_name="starter_era_delta",
            source_id="starter-source",
            source_kind="fixture",
            observed_as_of_utc=datetime(2026, 4, 5, 9, 45, tzinfo=timezone.utc),
            source_fingerprint=sha256(b"starter").hexdigest(),
        ),
    )


def make_snapshot(**overrides: object) -> MoneylineFeatureSnapshot:
    values: dict[str, object] = {
        "identity": IDENTITY,
        "provider_namespace": "MLB_STATS_API",
        "provider_game_id": "P19A-0001",
        "game_number": 1,
        "source_schedule_observation_id": "c" * 64,
        "as_of_utc": AS_OF,
        "scheduled_start_utc": START,
        "recent_win_rate_delta": Decimal("0.12"),
        "starter_era_delta": Decimal("-0.45"),
        "feature_provenance": provenance(),
    }
    values.update(overrides)
    return MoneylineFeatureSnapshot(**values)


class MoneylineFeatureSnapshotTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_serialization_is_stable(self) -> None:
        first = make_snapshot()
        second = make_snapshot()
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(
            first.feature_vector(),
            (Decimal("0.12"), Decimal("-0.45")),
        )
        with self.assertRaises(FrozenInstanceError):
            first.recent_win_rate_delta = Decimal("0")

    def test_record_outcome_mutation_does_not_change_snapshot(self) -> None:
        record = {
            "recent_win_rate_delta": "0.12",
            "starter_era_delta": "-0.45",
            "home_win": 1,
            "Home Score": 5,
            "Away Score": 2,
        }
        first = MoneylineFeatureSnapshot.from_record(
            record,
            identity=IDENTITY,
            provider_namespace="MLB_STATS_API",
            provider_game_id="P19A-0001",
            game_number=1,
            source_schedule_observation_id="c" * 64,
            as_of_utc=AS_OF,
            scheduled_start_utc=START,
            feature_provenance=provenance(),
        )
        record.update({"home_win": 0, "Home Score": 1, "Away Score": 8})
        second = MoneylineFeatureSnapshot.from_record(
            record,
            identity=IDENTITY,
            provider_namespace="MLB_STATS_API",
            provider_game_id="P19A-0001",
            game_number=1,
            source_schedule_observation_id="c" * 64,
            as_of_utc=AS_OF,
            scheduled_start_utc=START,
            feature_provenance=provenance(),
        )
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_missing_invalid_and_post_start_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MoneylineFeatureSnapshot.from_record(
                {"recent_win_rate_delta": "0.1"},
                identity=IDENTITY,
                provider_namespace="MLB_STATS_API",
                provider_game_id="P19A-0001",
                game_number=1,
                source_schedule_observation_id="c" * 64,
                as_of_utc=AS_OF,
                scheduled_start_utc=START,
                feature_provenance=provenance(),
            )
        with self.assertRaises(ValueError):
            make_snapshot(as_of_utc=START)
        with self.assertRaises(ValueError):
            make_snapshot(
                feature_provenance=(
                    provenance()[0],
                    MoneylineFeatureProvenance(
                        field_name="starter_era_delta",
                        source_id="late-source",
                        source_kind="fixture",
                        observed_as_of_utc=START,
                        source_fingerprint=sha256(b"late").hexdigest(),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
