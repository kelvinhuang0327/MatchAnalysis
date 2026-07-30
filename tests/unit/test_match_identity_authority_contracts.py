"""Unit tests for explicit match-identity authority contracts."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.baseball.domain.match_identity_authority import (
    MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION,
    MatchIdentityAuthorityCatalog,
    MatchIdentityAuthorityEntry,
    build_match_identity_authority_catalog,
    compute_match_identity_authority_catalog_fingerprint,
)


AUTHORITY_CATALOG_FINGERPRINT = (
    "8e90640fa4c10eb71009fc556c8d8d6cb9bde444fe98a0097a480801fdf6a9dd"
)


def authority_entry(
    provider_game_id: str = "777001",
    game_number: int = 1,
    *,
    league: str = "MLB",
    season: int = 2026,
    canonical_game_id: str = "FIXTURE_CANONICAL_MLB_GAME_777001",
    game_discriminator: str | None = None,
    authority_version: str = "fixture_match_identity_authority_v1",
) -> MatchIdentityAuthorityEntry:
    return MatchIdentityAuthorityEntry(
        provider_namespace="MLB_STATS_API",
        provider_game_id=provider_game_id,
        game_number=game_number,
        league=league,
        season=season,
        canonical_game_id=canonical_game_id,
        game_discriminator=game_discriminator,
        authority_version=authority_version,
    )


def fixture_entries() -> tuple[MatchIdentityAuthorityEntry, ...]:
    return (
        authority_entry(),
        authority_entry(
            "777002",
            2,
            canonical_game_id="FIXTURE_CANONICAL_MLB_GAME_777002",
            game_discriminator="doubleheader_game_2",
        ),
    )


class MatchIdentityAuthorityEntryTests(unittest.TestCase):
    def test_entry_has_exact_fields_and_is_immutable(self) -> None:
        entry = authority_entry()

        self.assertEqual(
            set(entry.__dataclass_fields__),
            {
                "provider_namespace",
                "provider_game_id",
                "game_number",
                "league",
                "season",
                "canonical_game_id",
                "game_discriminator",
                "authority_version",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            entry.canonical_game_id = "OTHER"

    def test_required_strings_must_be_explicit_and_trimmed(self) -> None:
        entry = authority_entry()
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "league",
            "canonical_game_id",
            "authority_version",
        ):
            for invalid in ("", " ", " padded"):
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        replace(entry, **{field_name: invalid})
            with self.subTest(field_name=field_name, invalid=None):
                with self.assertRaises(TypeError):
                    replace(entry, **{field_name: None})

    def test_positive_non_boolean_integers_are_required(self) -> None:
        entry = authority_entry()
        for field_name in ("game_number", "season"):
            for invalid in (True, 1.5, "1"):
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises(TypeError):
                        replace(entry, **{field_name: invalid})
            for invalid in (0, -1):
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        replace(entry, **{field_name: invalid})

    def test_discriminator_is_explicit_or_none(self) -> None:
        self.assertIsNone(authority_entry().game_discriminator)
        self.assertEqual(
            authority_entry(
                game_discriminator="explicit_doubleheader_2"
            ).game_discriminator,
            "explicit_doubleheader_2",
        )
        for invalid in ("", " ", " padded"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    authority_entry(game_discriminator=invalid)


class MatchIdentityAuthorityCatalogTests(unittest.TestCase):
    def test_fixture_catalog_matches_pre_edit_fingerprint(self) -> None:
        catalog = build_match_identity_authority_catalog(
            fixture_entries()
        )

        self.assertEqual(
            set(MatchIdentityAuthorityCatalog.__dataclass_fields__),
            {
                "entries",
                "entry_count",
                "catalog_fingerprint",
                "schema_version",
            },
        )
        self.assertEqual(catalog.entry_count, 2)
        self.assertEqual(
            catalog.schema_version,
            MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION,
        )
        self.assertEqual(
            catalog.catalog_fingerprint,
            AUTHORITY_CATALOG_FINGERPRINT,
        )

    def test_order_and_exact_duplicates_are_idempotent(self) -> None:
        entries = list(fixture_entries())
        random.Random(2026).shuffle(entries)
        repeated = (entries[0], entries[1], entries[0])

        first = build_match_identity_authority_catalog(fixture_entries())
        second = build_match_identity_authority_catalog(repeated)

        self.assertEqual(first, second)
        self.assertEqual(
            compute_match_identity_authority_catalog_fingerprint(
                fixture_entries()
            ),
            compute_match_identity_authority_catalog_fingerprint(repeated),
        )

    def test_same_key_with_unequal_authority_fails_closed(self) -> None:
        conflict = replace(
            fixture_entries()[0],
            canonical_game_id="CONFLICTING_CANONICAL_GAME",
        )

        with self.assertRaises(ValueError):
            build_match_identity_authority_catalog(
                (*fixture_entries(), conflict)
            )

    def test_two_game_numbers_remain_independently_explicit_keys(self) -> None:
        first = authority_entry(
            provider_game_id="shared-provider-game",
            game_number=1,
            canonical_game_id="EXPLICIT_GAME_ONE",
        )
        second = authority_entry(
            provider_game_id="shared-provider-game",
            game_number=2,
            canonical_game_id="EXPLICIT_GAME_TWO",
            game_discriminator=None,
        )

        catalog = build_match_identity_authority_catalog((second, first))

        self.assertEqual(
            [
                (
                    entry.provider_game_id,
                    entry.game_number,
                    entry.canonical_game_id,
                    entry.game_discriminator,
                )
                for entry in catalog.entries
            ],
            [
                (
                    "shared-provider-game",
                    1,
                    "EXPLICIT_GAME_ONE",
                    None,
                ),
                (
                    "shared-provider-game",
                    2,
                    "EXPLICIT_GAME_TWO",
                    None,
                ),
            ],
        )

    def test_catalog_rejects_inconsistent_count_or_fingerprint(self) -> None:
        entries = fixture_entries()
        fingerprint = (
            compute_match_identity_authority_catalog_fingerprint(entries)
        )

        with self.assertRaises(ValueError):
            MatchIdentityAuthorityCatalog(
                entries=entries,
                entry_count=1,
                catalog_fingerprint=fingerprint,
                schema_version=(
                    MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION
                ),
            )
        with self.assertRaises(ValueError):
            MatchIdentityAuthorityCatalog(
                entries=entries,
                entry_count=2,
                catalog_fingerprint="0" * 64,
                schema_version=(
                    MATCH_IDENTITY_AUTHORITY_CATALOG_SCHEMA_VERSION
                ),
            )


if __name__ == "__main__":
    unittest.main()
