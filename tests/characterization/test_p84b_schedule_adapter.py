"""Characterize the pinned P84B date-only schedule quarantine boundary."""

from dataclasses import FrozenInstanceError
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.import_legacy_schedule_snapshot import (
    PINNED_P84B_ARTIFACT_SHA256,
    PINNED_P84B_SEMANTIC_FINGERPRINT,
    SEMANTIC_FINGERPRINT_FIELDS,
    candidate_projection,
    import_legacy_schedule_snapshot,
)
from match_analysis.baseball.domain.schedule import (
    PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT,
    PROVIDER_NAMESPACE,
    ScheduleQuarantineReason,
    UNIVERSAL_QUARANTINE_REASONS,
)
from match_analysis.infrastructure.legacy_betting_pool.p84b_schedule_jsonl import (
    P84bScheduleJsonlSource,
    P84bScheduleValidationError,
)


def valid_row(game_id: str = "mlb_2026_823244") -> dict[str, object]:
    return {
        "game_id": game_id,
        "game_date": "2026-03-25",
        "season": 2026,
        "home_team": "San Francisco Giants",
        "away_team": "New York Yankees",
        "source_trace": PROVIDER_NAMESPACE,
        "collected_at_utc": "2026-05-27T03:33:32.410129+00:00",
    }


def encode_rows(*rows: dict[str, object]) -> bytes:
    return "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ).encode("utf-8")


def source_for(raw: bytes) -> P84bScheduleJsonlSource:
    return P84bScheduleJsonlSource(
        raw,
        expected_sha256=sha256(raw).hexdigest(),
    )


class P84bScheduleAdapterTests(unittest.TestCase):
    def test_valid_row_is_normalized_and_quarantined_without_promotion(
        self,
    ) -> None:
        row = valid_row()
        row["collected_at_utc"] = "2026-05-27T11:33:32.4100+08:00"

        result = import_legacy_schedule_snapshot(
            source_for(encode_rows(row))
        )
        candidate = result.candidates[0]

        self.assertEqual(
            candidate.legacy_collection_marker_utc,
            "2026-05-27T03:33:32.4100Z",
        )
        self.assertEqual(
            candidate.quarantine_reasons,
            UNIVERSAL_QUARANTINE_REASONS,
        )
        self.assertEqual(
            result.limitations[0],
            PRE_REQUEST_COLLECTION_MARKER_NOT_PROVIDER_OBSERVED_AT,
        )
        self.assertFalse(hasattr(candidate, "provider_observed_at_utc"))
        self.assertEqual(
            (
                result.match_identity_count,
                result.trusted_schedule_observation_count,
                result.baseball_game_count,
                result.pregame_eligible_context_count,
            ),
            (0, 0, 0, 0),
        )
        with self.assertRaises(FrozenInstanceError):
            result.row_count = 2

    def test_date_team_collisions_receive_only_the_appended_reason(
        self,
    ) -> None:
        first = valid_row("mlb_2026_100")
        second = valid_row("mlb_2026_101")
        third = valid_row("mlb_2026_102")
        third["away_team"] = "Boston Red Sox"

        result = import_legacy_schedule_snapshot(
            source_for(encode_rows(third, second, first))
        )
        reasons_by_id = {
            candidate.provider_reference.source_game_id: (
                candidate.quarantine_reasons
            )
            for candidate in result.candidates
        }

        self.assertEqual(len(result.collision_groups), 1)
        self.assertEqual(result.collision_affected_row_count, 2)
        self.assertEqual(
            reasons_by_id["mlb_2026_100"],
            (
                *UNIVERSAL_QUARANTINE_REASONS,
                ScheduleQuarantineReason.DATE_TEAM_COLLISION,
            ),
        )
        self.assertEqual(
            reasons_by_id["mlb_2026_102"],
            UNIVERSAL_QUARANTINE_REASONS,
        )

    def test_source_trace_is_exact_transport_evidence_not_projection(
        self,
    ) -> None:
        result = import_legacy_schedule_snapshot(
            source_for(encode_rows(valid_row()))
        )
        projection = candidate_projection(result.candidates[0])

        self.assertNotIn("source_trace", SEMANTIC_FINGERPRINT_FIELDS)
        self.assertNotIn("source_trace", projection)
        self.assertEqual(
            result.provider_game_references[0].provider_namespace,
            PROVIDER_NAMESPACE,
        )
        row = valid_row()
        row["source_trace"] = "MLB_STATS_API"
        with self.assertRaisesRegex(
            P84bScheduleValidationError,
            "source_trace",
        ):
            source_for(encode_rows(row)).load()

    def test_raw_sha_mismatch_is_rejected_before_parsing(self) -> None:
        source = P84bScheduleJsonlSource(
            b"not-json",
            expected_sha256="0" * 64,
        )

        with self.assertRaisesRegex(
            P84bScheduleValidationError,
            "SHA-256",
        ):
            source.load()

    def test_duplicate_json_keys_and_source_ids_are_rejected(self) -> None:
        raw = encode_rows(valid_row())
        duplicate_key = (
            raw[:-1] + b',\"game_id\":\"mlb_2026_823245\"}'
        )
        duplicate_id = encode_rows(valid_row(), valid_row())

        for invalid in (duplicate_key, duplicate_id):
            with self.subTest(invalid=invalid[:40]):
                with self.assertRaises(P84bScheduleValidationError):
                    source_for(invalid).load()

    def test_malformed_blank_and_closed_schema_rows_are_rejected(self) -> None:
        missing = valid_row()
        del missing["source_trace"]
        extra = valid_row()
        extra["status"] = "Scheduled"
        invalid_rows = (
            b"{",
            encode_rows(valid_row()) + b"\n\n" + encode_rows(valid_row()),
            encode_rows(missing),
            encode_rows(extra),
            b"[]",
        )

        for raw in invalid_rows:
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(P84bScheduleValidationError):
                    source_for(raw).load()

    def test_invalid_wrapper_date_season_teams_and_marker_are_rejected(
        self,
    ) -> None:
        invalid_cases = (
            ("game_id", "823244"),
            ("game_id", "mlb_2026_0823244"),
            ("game_date", "2026-02-30"),
            ("season", 0),
            ("season", True),
            ("home_team", ""),
            ("away_team", "San Francisco Giants"),
            ("collected_at_utc", "2026-05-27T03:33:32"),
            ("collected_at_utc", "not-a-time"),
        )
        for field, value in invalid_cases:
            row = valid_row()
            row[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(P84bScheduleValidationError):
                    source_for(encode_rows(row)).load()

    def test_unsorted_input_uses_required_python_string_order(self) -> None:
        rows = (
            valid_row("mlb_2026_9"),
            valid_row("mlb_2026_100"),
            valid_row("mlb_2026_10"),
        )

        first = import_legacy_schedule_snapshot(
            source_for(encode_rows(*rows))
        )
        second = import_legacy_schedule_snapshot(
            source_for(encode_rows(*reversed(rows)))
        )
        ordered_ids = tuple(
            reference.provider_game_id
            for reference in first.provider_game_references
        )

        self.assertEqual(ordered_ids, ("10", "100", "9"))
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(
            first.semantic_fingerprint,
            second.semantic_fingerprint,
        )

    def test_constructor_accepts_only_explicit_path_or_bytes(self) -> None:
        with self.assertRaises(TypeError):
            P84bScheduleJsonlSource(
                BytesIO(encode_rows(valid_row())),
                expected_sha256="0" * 64,
            )


class PinnedP84bScheduleCharacterizationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT"),
        "MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT is required",
    )
    def test_full_pinned_schedule_matches_independent_golden_projection(
        self,
    ) -> None:
        snapshot_path = Path(
            os.environ["MATCHANALYSIS_P84B_SCHEDULE_SNAPSHOT"]
        )
        first = import_legacy_schedule_snapshot(
            P84bScheduleJsonlSource(
                snapshot_path,
                expected_sha256=PINNED_P84B_ARTIFACT_SHA256,
            )
        )
        second = import_legacy_schedule_snapshot(
            P84bScheduleJsonlSource(
                snapshot_path,
                expected_sha256=PINNED_P84B_ARTIFACT_SHA256,
            )
        )
        independently_projected = "".join(
            json.dumps(
                {
                    "provider_namespace": (
                        candidate.provider_reference.provider_namespace
                    ),
                    "provider_game_id": (
                        candidate.provider_reference.provider_game_id
                    ),
                    "source_game_id": (
                        candidate.provider_reference.source_game_id
                    ),
                    "sport": candidate.sport,
                    "league": candidate.league,
                    "season": candidate.season,
                    "game_date": candidate.game_date,
                    "source_home_team": candidate.source_home_team,
                    "source_away_team": candidate.source_away_team,
                    "legacy_collection_marker_utc": (
                        candidate.legacy_collection_marker_utc
                    ),
                    "diagnostic_status": candidate.diagnostic_status,
                    "quarantine_reasons": [
                        reason.value
                        for reason in candidate.quarantine_reasons
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for candidate in sorted(
                first.candidates,
                key=lambda item: (
                    item.provider_reference.provider_namespace,
                    item.provider_reference.provider_game_id,
                    item.provider_reference.source_game_id,
                ),
            )
        ).encode("utf-8")

        self.assertEqual(
            first.artifact_sha256,
            PINNED_P84B_ARTIFACT_SHA256,
        )
        self.assertEqual(first.row_count, 2430)
        self.assertEqual(first.unique_provider_reference_count, 2430)
        self.assertEqual(len(first.provider_game_references), 2430)
        self.assertEqual(len(first.candidates), 2430)
        self.assertEqual(len(first.collision_groups), 11)
        self.assertEqual(first.collision_affected_row_count, 22)
        self.assertEqual(len(independently_projected), 1425112)
        self.assertEqual(
            sha256(independently_projected).hexdigest(),
            "4b219859cd1cd0fc19d75f1323684f4f8816115bf68677d1e25eb409f6ced077",
        )
        self.assertEqual(
            first.semantic_fingerprint,
            PINNED_P84B_SEMANTIC_FINGERPRINT,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            (
                first.match_identity_count,
                first.trusted_schedule_observation_count,
                first.baseball_game_count,
                first.pregame_eligible_context_count,
            ),
            (0, 0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
