"""Tests for the initial immutable contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain import BaseballGame
from match_analysis.core import ArtifactProvenance, MatchIdentity, UtcTimestamp


def make_identity(**overrides: object) -> MatchIdentity:
    values: dict[str, object] = {
        "sport": "baseball",
        "league": "WBC",
        "season": 2026,
        "canonical_game_id": "wbc-2026-001",
        "home_participant": "Taiwan",
        "away_participant": "Japan",
    }
    values.update(overrides)
    return MatchIdentity(**values)


def make_provenance(**overrides: str) -> ArtifactProvenance:
    values = {
        "schema_version": "1",
        "source_repository": "MatchAnalysis",
        "source_commit": "abc123",
        "producer_id": "unit-test",
        "producer_version": "1.0",
        "input_fingerprint": "a" * 64,
        "content_fingerprint": "b" * 64,
    }
    values.update(overrides)
    return ArtifactProvenance(**values)


class MatchIdentityTests(unittest.TestCase):
    def test_valid_identity(self) -> None:
        identity = make_identity()

        self.assertEqual(identity.canonical_game_id, "wbc-2026-001")

    def test_same_participant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_identity(away_participant="Taiwan")

    def test_blank_required_fields_are_rejected(self) -> None:
        for field_name in (
            "sport",
            "league",
            "canonical_game_id",
            "home_participant",
            "away_participant",
        ):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    make_identity(**{field_name: ""})

    def test_whitespace_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_identity(league=" WBC")
        with self.assertRaises(ValueError):
            make_identity(league="   ")

    def test_invalid_seasons_are_rejected(self) -> None:
        for season in (0, -1):
            with self.subTest(season=season):
                with self.assertRaises(ValueError):
                    make_identity(season=season)

    def test_game_discriminator_is_preserved(self) -> None:
        identity = make_identity(game_discriminator="doubleheader-2")

        self.assertEqual(identity.game_discriminator, "doubleheader-2")


class UtcTimestampTests(unittest.TestCase):
    def test_naive_datetime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UtcTimestamp(datetime(2026, 3, 5, 12, 0))

    def test_non_utc_datetime_is_normalized(self) -> None:
        source = datetime(
            2026,
            3,
            5,
            20,
            0,
            tzinfo=timezone(timedelta(hours=8)),
        )

        timestamp = UtcTimestamp(source)

        self.assertEqual(
            timestamp.value,
            datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc),
        )

    def test_serialization_uses_explicit_z_suffix(self) -> None:
        timestamp = UtcTimestamp(
            datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
        )

        self.assertEqual(timestamp.to_iso8601(), "2026-03-05T12:00:00Z")

    def test_serialization_preserves_subseconds(self) -> None:
        timestamp = UtcTimestamp(
            datetime(2026, 3, 5, 12, 0, 0, 123456, tzinfo=timezone.utc)
        )

        self.assertEqual(
            timestamp.to_iso8601(),
            "2026-03-05T12:00:00.123456Z",
        )


class ArtifactProvenanceTests(unittest.TestCase):
    def test_valid_provenance(self) -> None:
        provenance = make_provenance()

        self.assertEqual(provenance.schema_version, "1")

    def test_invalid_sha256_is_rejected(self) -> None:
        for fingerprint in ("a" * 63, "A" * 64, "g" * 64):
            with self.subTest(fingerprint=fingerprint):
                with self.assertRaises(ValueError):
                    make_provenance(input_fingerprint=fingerprint)


class BaseballGameTests(unittest.TestCase):
    def test_baseball_sport_is_enforced(self) -> None:
        scheduled_start = UtcTimestamp(
            datetime(2026, 3, 5, 12, 0, tzinfo=timezone.utc)
        )

        with self.assertRaises(ValueError):
            BaseballGame(
                identity=make_identity(sport="football"),
                scheduled_start=scheduled_start,
            )

        game = BaseballGame(
            identity=make_identity(),
            scheduled_start=scheduled_start,
        )
        self.assertEqual(game.identity.sport, "baseball")


if __name__ == "__main__":
    unittest.main()
