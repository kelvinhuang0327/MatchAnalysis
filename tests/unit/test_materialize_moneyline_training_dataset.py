"""Unit and invariant tests for P22A dataset materialization."""

import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.materialize_moneyline_training_dataset import (  # noqa: E402
    P22A_STOP_SELECTION_PAIR_INCONSISTENT,
    materialize_moneyline_training_dataset,
)
from match_analysis.application.use_cases.replay_multifold_historical_candidates import (  # noqa: E402
    load_multifold_folds,
    load_multifold_reconstructed_models,
)
from match_analysis.application.use_cases.moneyline_training_dataset_artifacts import (  # noqa: E402
    render_training_examples_jsonl,
)


FIXTURE_ROOT = REPOSITORY_ROOT / "data/fixtures/p21b_multifold_historical"
CANDIDATE_ROOT = REPOSITORY_ROOT / "report/p21b_contiguous_multifold_historical_candidates"


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def _load_dataset(
    *,
    candidate_bytes: bytes | None = None,
    candidate_summary_bytes: bytes | None = None,
    results_bytes: bytes | None = None,
):
    return materialize_moneyline_training_dataset(
        candidate_bytes=(
            CANDIDATE_ROOT / "learning_candidates.jsonl"
        ).read_bytes()
        if candidate_bytes is None
        else candidate_bytes,
        candidate_summary_bytes=(
            CANDIDATE_ROOT / "summary.json"
        ).read_bytes()
        if candidate_summary_bytes is None
        else candidate_summary_bytes,
        folds=load_multifold_folds(
            [FIXTURE_ROOT / "fold_wf_003.json", FIXTURE_ROOT / "fold_wf_002.json"]
        ),
        historical_results_bytes=(FIXTURE_ROOT / "final_results.jsonl").read_bytes()
        if results_bytes is None
        else results_bytes,
        historical_provenance_bytes=(FIXTURE_ROOT / "provenance.json").read_bytes(),
        reconstructed_models=load_multifold_reconstructed_models(
            FIXTURE_ROOT / "reconstructed_models.json"
        ),
    )


def _candidate_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (CANDIDATE_ROOT / "learning_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


class MaterializeMoneylineTrainingDatasetTests(unittest.TestCase):
    def test_materializes_one_game_level_example_per_eligible_candidate_group(self) -> None:
        dataset = _load_dataset()
        self.assertEqual(dataset.eligible_candidate_count, 1354)
        self.assertEqual(dataset.training_example_count, 677)
        self.assertEqual(dataset.candidates_collapsed_count, 677)
        self.assertEqual(dataset.unmapped_candidate_count, 0)
        self.assertEqual(
            {example.fold_id for example in dataset.examples}, {"wf_002", "wf_003"}
        )
        self.assertEqual(
            {example.provider_game_id for example in dataset.examples},
            {row["provider_game_id"] for row in _candidate_rows()},
        )
        self.assertTrue(
            all(
                example.source_candidates[0].selection
                in {"HOME", "AWAY"}
                for example in dataset.examples
            )
        )
        self.assertTrue(all(len(example.source_candidates) == 2 for example in dataset.examples))

    def test_dataset_summary_has_required_claims_and_label_distribution(self) -> None:
        dataset = _load_dataset()
        summary = dataset.to_summary(
            training_examples_jsonl_sha256="a" * 64
        )
        self.assertEqual(summary["label_distribution"], {"0": 309, "1": 368})
        self.assertEqual(summary["date_range"]["start_utc"], "2025-07-01T12:00:00Z")
        self.assertEqual(summary["date_range"]["end_utc"], "2025-08-31T17:57:00Z")
        self.assertEqual(summary["feature_names"], ["recent_win_rate_delta", "starter_era_delta"])
        self.assertTrue(summary["training_dataset_claim"])
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["retraining_performed"])
        self.assertFalse(summary["model_promoted"])
        self.assertTrue(summary["sample_limited"])
        self.assertFalse(summary["profitability_claim"])
        self.assertFalse(summary["production_ready"])
        self.assertEqual(summary["p20b_historical_runtime_compliance"], "REMAINS_REFUTED")

    def test_source_order_does_not_change_dataset_bytes_or_fingerprint(self) -> None:
        candidate_rows = _candidate_rows()
        candidate_bytes = _jsonl(list(reversed(candidate_rows)))
        results_rows = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "final_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        results_bytes = _jsonl(list(reversed(results_rows)))
        original = _load_dataset()
        reordered = _load_dataset(
            candidate_bytes=candidate_bytes,
            results_bytes=results_bytes,
        )
        self.assertEqual(original.dataset_fingerprint, reordered.dataset_fingerprint)
        self.assertEqual(
            render_training_examples_jsonl(original.examples),
            render_training_examples_jsonl(reordered.examples),
        )

    def test_inconsistent_selection_pair_fails_closed(self) -> None:
        rows = _candidate_rows()
        first_game = rows[0]["provider_game_id"]
        pair = [row for row in rows if row["provider_game_id"] == first_game]
        self.assertEqual(len(pair), 2)
        pair[1]["observation_payload"]["source_schedule_observation_id"] = "0" * 64
        modified_rows = [row for row in rows if row["provider_game_id"] != first_game] + pair
        summary = json.loads((CANDIDATE_ROOT / "summary.json").read_text(encoding="utf-8"))
        canonical_rows = sorted(modified_rows, key=lambda row: row["candidate_id"])
        candidate_bytes = _jsonl(canonical_rows)
        candidate_fingerprint = __import__("hashlib").sha256(candidate_bytes).hexdigest()
        summary["candidate_semantic_fingerprint"] = candidate_fingerprint
        summary["aggregate_candidate_fingerprint"] = candidate_fingerprint
        summary["learning_candidates_jsonl_sha256"] = candidate_fingerprint
        with self.assertRaisesRegex(ValueError, P22A_STOP_SELECTION_PAIR_INCONSISTENT):
            _load_dataset(
                candidate_bytes=candidate_bytes,
                candidate_summary_bytes=(
                    json.dumps(summary, sort_keys=True, indent=2) + "\n"
                ).encode("utf-8"),
            )

    def test_outcome_mutation_changes_only_target_and_result_fields_not_example_identity(self) -> None:
        original = _load_dataset()
        result_rows = [
            json.loads(line)
            for line in (FIXTURE_ROOT / "final_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        mutated_game = result_rows[0]["provider_game_id"]
        for row in result_rows:
            if row["provider_game_id"] == mutated_game:
                row["home_score"], row["away_score"] = 0, 9
                break
        mutated = _load_dataset(results_bytes=_jsonl(result_rows))
        original_by_game = {row.provider_game_id: row for row in original.examples}
        mutated_by_game = {row.provider_game_id: row for row in mutated.examples}
        self.assertEqual(
            [row.training_example_id for row in original.examples],
            [row.training_example_id for row in mutated.examples],
        )
        self.assertEqual(
            original_by_game[mutated_game].feature_values,
            mutated_by_game[mutated_game].feature_values,
        )
        self.assertEqual(
            original_by_game[mutated_game].feature_snapshot_fingerprint,
            mutated_by_game[mutated_game].feature_snapshot_fingerprint,
        )
        self.assertNotEqual(
            original_by_game[mutated_game].target_home_win,
            mutated_by_game[mutated_game].target_home_win,
        )
        self.assertNotEqual(original.dataset_fingerprint, mutated.dataset_fingerprint)


if __name__ == "__main__":
    unittest.main()
