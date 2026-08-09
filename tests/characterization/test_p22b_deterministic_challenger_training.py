"""Characterization of the committed P22B challenger artifact contract."""

from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from match_analysis.application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_model_artifact,
)
from match_analysis.application.use_cases.train_moneyline_challenger import (
    P22B_DATASET_FINGERPRINT,
    P22B_DEFAULT_FIT_RUNTIME,
    train_moneyline_challenger,
)
from match_analysis.baseball.domain.canonical_utc import parse_canonical_utc
from match_analysis.baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from match_analysis.core.identity import MatchIdentity


DATASET_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/training_examples.jsonl"
SUMMARY_PATH = REPOSITORY_ROOT / "report/p22a_game_level_training_dataset/summary.json"
ARTIFACT_DIR = REPOSITORY_ROOT / "report/p22b_moneyline_challenger"
AUTHORITY_REPOSITORY = "/Users/kelvin/VibeCoding-WorkSpace/MatchAnalysis"


def _snapshot_from_p22a_row(row: dict[str, object]) -> MoneylineFeatureSnapshot:
    feature_names = tuple(str(item) for item in row["feature_names"])
    feature_values = tuple(str(item) for item in row["feature_values"])
    lineage = tuple(row["feature_lineage"])
    identity = MatchIdentity(
        sport="baseball",
        league="MLB",
        season=2025,
        canonical_game_id=str(row["provider_game_id"]),
        home_participant=str(row["home_participant"]),
        away_participant=str(row["away_participant"]),
    )
    provenance = tuple(
        MoneylineFeatureProvenance(
            field_name=str(item["field_name"]),
            source_id=str(item["source_id"]),
            source_kind=str(item["source_kind"]),
            observed_as_of_utc=parse_canonical_utc(str(item["observed_as_of_utc"])),
            source_fingerprint=str(item["source_fingerprint"]),
        )
        for item in lineage
    )
    return MoneylineFeatureSnapshot.from_record(
        dict(zip(feature_names, feature_values, strict=True)),
        identity=identity,
        provider_namespace=str(row["provider_namespace"]),
        provider_game_id=str(row["provider_game_id"]),
        game_number=int(row["game_number"]),
        source_schedule_observation_id=str(row["source_schedule_observation_id"]),
        as_of_utc=parse_canonical_utc(str(row["feature_as_of_utc"])),
        scheduled_start_utc=parse_canonical_utc(str(row["scheduled_start_utc"])),
        feature_provenance=provenance,
    )


class P22BDeterministicChallengerTests(unittest.TestCase):
    def test_committed_artifact_has_complete_challenger_provenance(self) -> None:
        artifact = load_moneyline_model_artifact(ARTIFACT_DIR / "model_artifact.json")
        projection = json.loads(
            (ARTIFACT_DIR / "model_artifact.json").read_text(encoding="utf-8")
        )
        summary = json.loads((ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(projection["model_role"], "CHALLENGER")
        self.assertEqual(projection["source_dataset_fingerprint"], P22B_DATASET_FINGERPRINT)
        self.assertEqual(projection["training_example_count"], 677)
        self.assertEqual(projection["feature_names"], ["recent_win_rate_delta", "starter_era_delta"])
        self.assertEqual(projection["fit_configuration"], {
            "max_iter": 1000,
            "model_type": "logistic_regression",
            "solver": "lbfgs",
        })
        self.assertEqual(summary["artifact_fingerprint"], projection["artifact_fingerprint"])
        self.assertEqual(artifact.model_id, projection["model_id"])
        self.assertEqual(artifact.feature_names, ("recent_win_rate_delta", "starter_era_delta"))
        self.assertEqual(artifact.fingerprint(), load_moneyline_model_artifact(
            ARTIFACT_DIR / "model_artifact.json"
        ).fingerprint())

    def test_p19a_inference_accepts_a_p22a_feature_row(self) -> None:
        row = json.loads(DATASET_PATH.read_text(encoding="utf-8").splitlines()[0])
        snapshot = _snapshot_from_p22a_row(row)
        artifact = load_moneyline_model_artifact(ARTIFACT_DIR / "model_artifact.json")
        first = artifact.predict_home_probability(snapshot)
        second = artifact.predict_home_probability(snapshot)
        self.assertEqual(first, second)
        self.assertIsInstance(first, Decimal)
        self.assertGreater(first, Decimal("0"))
        self.assertLess(first, Decimal("1"))

    def test_training_reproduces_committed_artifact(self) -> None:
        artifact = train_moneyline_challenger(
            DATASET_PATH,
            SUMMARY_PATH,
            fit_runtime=P22B_DEFAULT_FIT_RUNTIME,
            source_repository=AUTHORITY_REPOSITORY,
        )
        committed = json.loads(
            (ARTIFACT_DIR / "model_artifact.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact.to_projection(), committed)


if __name__ == "__main__":
    unittest.main()
