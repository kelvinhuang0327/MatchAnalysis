"""Materialize deterministic game-level P22A supervised examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc
from ...baseball.domain.moneyline_feature_snapshot import (
    MONEYLINE_FEATURE_NAMES,
)
from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)
from ...baseball.domain.supervised_training_example import (
    CandidateLineage,
    FeatureLineage,
    SupervisedTrainingExample,
    compute_training_example_id,
)
from .moneyline_walk_forward_artifacts import build_moneyline_model_artifact
from .replay_historical_moneyline_predictions import _snapshot_for_row


P22A_DATASET_SCHEMA_VERSION = "p22a.game_level_supervised_training_dataset.v1"
P22A_DATASET_CONTRACT_VERSION = "p22a.game_level_supervised_training_dataset_contract.v1"
P22A_STOP_SELECTION_PAIR_INCONSISTENT = (
    "STOP_MATCHANALYSIS_P22A_SELECTION_PAIR_INCONSISTENT"
)
P22A_STOP_TARGET_SEMANTICS_UNRESOLVED = (
    "STOP_MATCHANALYSIS_P22A_TARGET_SEMANTICS_UNRESOLVED"
)
P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT = (
    "STOP_MATCHANALYSIS_P22A_COMMITTED_LINEAGE_INSUFFICIENT"
)

P22A_CLAIMS = {
    "training_dataset_claim": True,
    "training_authorized": False,
    "retraining_performed": False,
    "model_promoted": False,
    "sample_limited": True,
    "profitability_claim": False,
    "production_ready": False,
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _parse_json_object(raw: bytes, context: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _parse_jsonl(raw: bytes, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{context} contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{context} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{context} contains no rows")
    return rows


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        _canonical_json_bytes(row) for row in rows
    )


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _semantic_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: str(row["candidate_id"]))
    return _sha256(_render_jsonl(ordered))


def _candidate_row_fingerprint(row: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(row))


def _result_row_fingerprint(row: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(row))


def _result_rows_fingerprint(
    *,
    folds: Sequence[MoneylineWalkForwardFold],
    result_by_game: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> str:
    """Reproduce the committed P21B result-row identity, without replaying it."""

    provenance_by_game = {
        row["canonical_game_id"]: row for row in provenance["rows"]
    }
    serialized: list[str] = []
    for fold in sorted(folds, key=lambda item: item.fold_id):
        for prediction in fold.prediction_rows:
            source = provenance_by_game[prediction.game_id]
            result = result_by_game[prediction.game_id]
            _, home_team_code, away_team_code = prediction.game_id.split("_", 2)
            serialized.append(
                f"{prediction.game_id}\t{prediction.date}\t"
                f"{source['game_number']}\t{home_team_code}\t"
                f"{result['home_score']}\t{away_team_code}\t"
                f"{result['away_score']}\n"
            )
    return _sha256("".join(serialized).encode("utf-8"))


def _target_from_result(result: Mapping[str, Any]) -> int:
    home_score = result.get("home_score")
    away_score = result.get("away_score")
    if (
        isinstance(home_score, bool)
        or not isinstance(home_score, int)
        or isinstance(away_score, bool)
        or not isinstance(away_score, int)
        or home_score == away_score
    ):
        raise ValueError(f"{P22A_STOP_TARGET_SEMANTICS_UNRESOLVED}: invalid final score")
    return int(home_score > away_score)


def _feature_snapshot_fingerprint(source_prediction_id: str, selection: str) -> str:
    prefix = "p19a:"
    suffix = f":{selection.lower()}"
    if not source_prediction_id.startswith(prefix) or not source_prediction_id.endswith(
        suffix
    ):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: invalid source prediction lineage"
        )
    fingerprint = source_prediction_id[len(prefix) : -len(suffix)]
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: invalid feature snapshot fingerprint"
        )
    return fingerprint


def _validate_candidate_summary(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    candidate_summary_bytes: bytes,
) -> tuple[str, str]:
    ordered = sorted(candidate_rows, key=lambda row: str(row.get("candidate_id", "")))
    canonical_candidates = _render_jsonl(ordered)
    semantic_fingerprint = _sha256(canonical_candidates)
    if summary.get("learning_candidates_jsonl_sha256") != _sha256(canonical_candidates):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P21B candidate artifact fingerprint mismatch"
        )
    if summary.get("candidate_semantic_fingerprint") != semantic_fingerprint:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P21B candidate semantic fingerprint mismatch"
        )
    if summary.get("aggregate_candidate_fingerprint") != semantic_fingerprint:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P21B aggregate candidate fingerprint mismatch"
        )
    if summary.get("candidate_count") != len(candidate_rows):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P21B candidate count mismatch"
        )
    if summary.get("p21a_eligible_count") != len(candidate_rows):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P21A eligible count mismatch"
        )
    if summary.get("p21a_excluded_count") != 0:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: excluded candidates cannot enter P22A"
        )
    if summary.get("p20b_historical_runtime_compliance") != "REMAINS_REFUTED":
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: P20B historical verdict changed"
        )
    if not candidate_summary_bytes:
        raise ValueError("P21B candidate summary must not be empty")
    return semantic_fingerprint, _sha256(candidate_summary_bytes)


def _validate_results(
    *,
    folds: Sequence[MoneylineWalkForwardFold],
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    provenance = _parse_json_object(
        historical_provenance_bytes,
        "P21B historical provenance",
    )
    claims = provenance.get("claims")
    if not isinstance(claims, dict) or claims.get("historical") is not True:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: historical provenance claim missing"
        )
    if claims.get("non_synthetic") is not True:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: synthetic result source"
        )
    rows = _parse_jsonl(historical_results_bytes, "P21B historical results")
    result_by_game: dict[str, dict[str, Any]] = {}
    for row in rows:
        game_id = row.get("provider_game_id")
        if not isinstance(game_id, str) or game_id in result_by_game:
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: duplicate result identity"
            )
        if (
            row.get("provider_namespace") != "MLB_STATS_API"
            or row.get("game_number") != 1
            or row.get("status") != "FINAL"
            or not isinstance(row.get("source_result_id"), str)
            or not row["source_result_id"].strip()
        ):
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: invalid final-result authority"
            )
        if _target_from_result(row) not in (0, 1):
            raise ValueError(
                f"{P22A_STOP_TARGET_SEMANTICS_UNRESOLVED}: invalid target"
            )
        result_by_game[game_id] = row

    expected_game_ids = {
        prediction.game_id
        for fold in folds
        for prediction in fold.prediction_rows
    }
    if set(result_by_game) != expected_game_ids:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: result identities do not match P20A games"
        )
    provenance_rows = provenance.get("rows")
    if not isinstance(provenance_rows, list):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: provenance rows missing"
        )
    provenance_by_game = {
        row.get("canonical_game_id"): row
        for row in provenance_rows
        if isinstance(row, dict)
    }
    if set(provenance_by_game) != expected_game_ids:
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: provenance identities do not match P20A games"
        )
    return (
        result_by_game,
        provenance,
        _result_rows_fingerprint(
            folds=folds,
            result_by_game=result_by_game,
            provenance=provenance,
        ),
    )


def _validate_pair(
    *,
    rows: Sequence[Mapping[str, Any]],
    fold: MoneylineWalkForwardFold,
    model_artifact: Any,
    feature_row: Any,
    result: Mapping[str, Any],
) -> tuple[str, str]:
    if len(rows) not in (1, 2):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: expected one or two selections"
        )
    selections = [str(row.get("selection")) for row in rows]
    if len(rows) == 2 and set(selections) != {"HOME", "AWAY"}:
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: selections are not complementary"
        )
    if len(set(selections)) != len(selections):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: duplicate selection"
        )

    common_fields = (
        "provider_namespace",
        "provider_game_id",
        "game_number",
        "model_id",
        "scheduled_start_utc",
        "result_observation_id",
        "result_observed_at_utc",
        "home_score",
        "away_score",
        "actual_winner",
    )
    first = rows[0]
    for row in rows:
        for field_name in common_fields:
            if row.get(field_name) != first.get(field_name):
                raise ValueError(
                    f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: {field_name} differs"
                )
        payload = row.get("observation_payload")
        first_payload = first.get("observation_payload")
        if not isinstance(payload, dict) or not isinstance(first_payload, dict):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: prediction payload missing"
            )
        for field_name in (
            "provider_namespace",
            "provider_game_id",
            "game_number",
            "model_id",
            "prediction_generated_at_utc",
            "response_received_at_utc",
            "ingested_at_utc",
            "scheduled_start_utc",
            "source_schedule_observation_id",
            "market_id",
            "line_value",
            "push_policy",
        ):
            if payload.get(field_name) != first_payload.get(field_name):
                raise ValueError(
                    f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: prediction lineage differs"
                )
        if payload.get("selection") != row.get("selection"):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: selection payload mismatch"
            )
        if payload.get("model_probability") != row.get("model_probability"):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: probability payload mismatch"
            )
        if row.get("market_id") != "moneyline":
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: unsupported market"
            )
        if row.get("provider_game_id") != feature_row.game_id:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: feature game mismatch"
            )
        if row.get("scheduled_start_utc") != feature_row.scheduled_start_utc:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: scheduled start mismatch"
            )
        if payload.get("source_schedule_observation_id") != feature_row.source_schedule_observation_id:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: feature schedule lineage mismatch"
            )
        if row.get("model_id") != model_artifact.model_id:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: model identity mismatch"
            )
        if row.get("result_observed_at_utc") != result.get("result_observed_at_utc"):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: result timestamp mismatch"
            )
        if row.get("source_evaluation_row_fingerprint") is None:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: evaluation lineage missing"
            )

    source_prediction_ids = [
        str(row["observation_payload"]["source_prediction_id"]) for row in rows
    ]
    feature_fingerprints = [
        _feature_snapshot_fingerprint(source_prediction_id, selection)
        for source_prediction_id, selection in zip(source_prediction_ids, selections)
    ]
    if len(set(feature_fingerprints)) != 1:
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: feature snapshot identity differs"
        )
    if feature_fingerprints[0] != _snapshot_for_row(fold, feature_row).fingerprint():
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: feature snapshot fingerprint mismatch"
        )
    if len(rows) == 2:
        probabilities = [Decimal(str(row["model_probability"])) for row in rows]
        if any(probability <= 0 or probability >= 1 for probability in probabilities):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: invalid probability"
            )
        if probabilities[0] + probabilities[1] != Decimal("1"):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: paired probabilities are not complementary"
            )
    return feature_fingerprints[0], str(first["result_observation_id"])


@dataclass(frozen=True, slots=True)
class MoneylineTrainingDataset:
    """The validated dataset and immutable source identities used to build it."""

    examples: tuple[SupervisedTrainingExample, ...]
    eligible_candidate_count: int
    source_candidate_artifact_fingerprint: str
    source_candidate_artifact_sha256: str
    source_candidate_summary_sha256: str
    source_assessed_count: int
    source_excluded_count: int
    source_historical_results_sha256: str
    source_historical_provenance_sha256: str
    source_result_rows_fingerprint: str
    source_fold_artifacts: tuple[dict[str, str], ...]
    p20b_historical_runtime_compliance: str

    @property
    def training_example_count(self) -> int:
        return len(self.examples)

    @property
    def candidates_collapsed_count(self) -> int:
        return self.eligible_candidate_count - self.training_example_count

    @property
    def unmapped_candidate_count(self) -> int:
        return 0

    @property
    def ordered_examples(self) -> tuple[SupervisedTrainingExample, ...]:
        return tuple(sorted(self.examples, key=lambda example: example.training_example_id))

    @property
    def dataset_fingerprint(self) -> str:
        return _sha256(
            _canonical_json_bytes(
                {
                    "contract_version": P22A_DATASET_CONTRACT_VERSION,
                    "schema_version": P22A_DATASET_SCHEMA_VERSION,
                    "training_examples": [
                        example.to_projection() for example in self.ordered_examples
                    ],
                }
            )
        )

    def to_summary(self, *, training_examples_jsonl_sha256: str) -> dict[str, Any]:
        starts = [example.scheduled_start_utc for example in self.ordered_examples]
        labels = Counter(str(example.target_home_win) for example in self.examples)
        fold_counts: dict[str, list[SupervisedTrainingExample]] = defaultdict(list)
        for example in self.examples:
            fold_counts[example.fold_id].append(example)
        fold_summary = []
        for fold_id in sorted(fold_counts):
            fold_examples = sorted(
                fold_counts[fold_id], key=lambda example: example.training_example_id
            )
            fold_summary.append(
                {
                    "date_range": {
                        "end_utc": max(item.scheduled_start_utc for item in fold_examples),
                        "start_utc": min(item.scheduled_start_utc for item in fold_examples),
                    },
                    "fold_id": fold_id,
                    "label_distribution": dict(
                        sorted(Counter(str(item.target_home_win) for item in fold_examples).items())
                    ),
                    "training_example_count": len(fold_examples),
                }
            )
        return {
            "claims": dict(P22A_CLAIMS),
            "candidates_collapsed_count": self.candidates_collapsed_count,
            "contract_version": P22A_DATASET_CONTRACT_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "date_range": {"end_utc": max(starts), "start_utc": min(starts)},
            "eligible_candidate_count": self.eligible_candidate_count,
            "feature_names": list(MONEYLINE_FEATURE_NAMES),
            "fold_count": len(fold_counts),
            "folds": fold_summary,
            "label_distribution": dict(sorted(labels.items())),
            "model_promoted": False,
            "p20b_historical_runtime_compliance": self.p20b_historical_runtime_compliance,
            "production_ready": False,
            "profitability_claim": False,
            "schema_version": P22A_DATASET_SCHEMA_VERSION,
            "retraining_performed": False,
            "sample_limited": True,
            "source_assessed_count": self.source_assessed_count,
            "source_candidate_artifact_fingerprint": self.source_candidate_artifact_fingerprint,
            "source_candidate_artifact_sha256": self.source_candidate_artifact_sha256,
            "source_candidate_summary_sha256": self.source_candidate_summary_sha256,
            "source_excluded_count": self.source_excluded_count,
            "source_fold_artifacts": list(self.source_fold_artifacts),
            "source_historical_provenance_sha256": self.source_historical_provenance_sha256,
            "source_historical_results_sha256": self.source_historical_results_sha256,
            "source_result_rows_fingerprint": self.source_result_rows_fingerprint,
            "target_label_semantics": (
                "target_home_win=1 iff committed FINAL home_score is greater than "
                "away_score; target_home_win=0 iff away_score is greater"
            ),
            "training_authorized": False,
            "training_dataset_claim": True,
            "training_example_count": self.training_example_count,
            "training_examples_jsonl_sha256": training_examples_jsonl_sha256,
            "training_example_id_includes_target": False,
            "training_example_id_is_path_independent": True,
            "unmapped_candidate_count": self.unmapped_candidate_count,
        }


def materialize_moneyline_training_dataset(
    *,
    candidate_bytes: bytes,
    candidate_summary_bytes: bytes,
    folds: Sequence[MoneylineWalkForwardFold],
    historical_results_bytes: bytes,
    historical_provenance_bytes: bytes,
    reconstructed_models: Mapping[str, ReconstructedWalkForwardModel],
) -> MoneylineTrainingDataset:
    """Collapse committed P21B candidates into one example per unique game."""

    candidate_rows = _parse_jsonl(candidate_bytes, "P21B learning candidates")
    candidate_summary = _parse_json_object(candidate_summary_bytes, "P21B candidate summary")
    source_candidate_fingerprint, source_candidate_summary_sha256 = (
        _validate_candidate_summary(
            candidate_rows=candidate_rows,
            summary=candidate_summary,
            candidate_summary_bytes=candidate_summary_bytes,
        )
    )
    ordered_folds = tuple(sorted(folds, key=lambda fold: fold.fold_id))
    fold_by_id = {fold.fold_id: fold for fold in ordered_folds}
    if len(fold_by_id) != len(ordered_folds):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: duplicate fold identity"
        )
    if tuple(candidate_summary.get("selected_fold_ids", ())) != tuple(fold_by_id):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: selected fold identity mismatch"
        )
    result_by_game, provenance, result_rows_fingerprint = _validate_results(
        folds=ordered_folds,
        historical_results_bytes=historical_results_bytes,
        historical_provenance_bytes=historical_provenance_bytes,
    )

    feature_rows: dict[str, tuple[MoneylineWalkForwardFold, Any]] = {}
    for fold in ordered_folds:
        for row in fold.prediction_rows:
            if row.game_id in feature_rows:
                raise ValueError(
                    f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: duplicate game feature lineage"
                )
            feature_rows[row.game_id] = (fold, row)

    candidate_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for row in candidate_rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidate_ids:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: duplicate candidate identity"
            )
        candidate_ids.add(candidate_id)
        if row.get("assessment_status") != "ELIGIBLE":
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: non-ELIGIBLE candidate entered P22A"
            )
        game_id = row.get("provider_game_id")
        if not isinstance(game_id, str):
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: candidate game identity missing"
            )
        candidate_groups[game_id].append(row)

    if len(candidate_ids) != len(candidate_rows):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: candidate identity is not unique"
        )

    examples: list[SupervisedTrainingExample] = []
    for game_id in sorted(candidate_groups):
        rows = candidate_groups[game_id]
        model_ids = {str(row.get("model_id")) for row in rows}
        if len(model_ids) != 1:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: multiple model lineages for one game"
            )
        feature_context = feature_rows.get(game_id)
        if feature_context is None:
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: candidate has no P20A feature row"
            )
        fold, feature_row = feature_context
        expected_model_id = f"p13_walk_forward_logistic_v1_{fold.fold_id}"
        if model_ids != {expected_model_id}:
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: candidate fold/model lineage mismatch"
            )
        model = reconstructed_models.get(fold.fold_id)
        if model is None:
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: reconstructed model missing"
            )
        model_artifact = build_moneyline_model_artifact(fold, model)
        if model_artifact.model_id != expected_model_id:
            raise ValueError(
                f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: model artifact identity mismatch"
            )
        result = result_by_game[game_id]
        feature_snapshot = _snapshot_for_row(fold, feature_row)
        feature_snapshot_fp, result_observation_id = _validate_pair(
            rows=rows,
            fold=fold,
            model_artifact=model_artifact,
            feature_row=feature_row,
            result=result,
        )
        if feature_snapshot_fp != feature_snapshot.fingerprint():
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: feature snapshot identity mismatch"
            )
        target_home_win = _target_from_result(result)
        if result_observation_id != str(rows[0]["result_observation_id"]):
            raise ValueError(
                f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: result observation identity mismatch"
            )
        feature_lineage = tuple(
            FeatureLineage(
                field_name=provenance_item.field_name,
                value=getattr(feature_snapshot, provenance_item.field_name),
                source_id=provenance_item.source_id,
                source_kind=provenance_item.source_kind,
                observed_as_of_utc=format_canonical_utc(
                    provenance_item.observed_as_of_utc
                ),
                source_fingerprint=provenance_item.source_fingerprint,
            )
            for provenance_item in feature_snapshot.feature_provenance
        )
        source_candidates = tuple(
            sorted(
                (
                    CandidateLineage(
                        candidate_id=str(row["candidate_id"]),
                        candidate_row_fingerprint=_candidate_row_fingerprint(row),
                        selection=str(row["selection"]),
                        source_snapshot_row_fingerprint=str(
                            row["source_snapshot_row_fingerprint"]
                        ),
                        source_evaluation_row_fingerprint=str(
                            row["source_evaluation_row_fingerprint"]
                        ),
                    )
                    for row in rows
                ),
                key=lambda item: item.candidate_id,
            )
        )
        training_example_id = compute_training_example_id(
            schema_version="p22a.supervised_training_example.v1",
            provider_namespace=feature_snapshot.provider_namespace,
            provider_game_id=feature_snapshot.provider_game_id,
            game_number=feature_snapshot.game_number,
            scheduled_start_utc=format_canonical_utc(
                feature_snapshot.scheduled_start_utc
            ),
            feature_as_of_utc=format_canonical_utc(feature_snapshot.as_of_utc),
            fold_id=fold.fold_id,
            fold_fingerprint=fold.fingerprint(),
            model_id=model_artifact.model_id,
            model_fingerprint=model.fingerprint(),
            feature_snapshot_fingerprint=feature_snapshot_fp,
            source_schedule_observation_id=feature_snapshot.source_schedule_observation_id,
        )
        examples.append(
            SupervisedTrainingExample(
                training_example_id=training_example_id,
                provider_namespace=feature_snapshot.provider_namespace,
                provider_game_id=feature_snapshot.provider_game_id,
                game_number=feature_snapshot.game_number,
                home_participant=feature_snapshot.identity.home_participant,
                away_participant=feature_snapshot.identity.away_participant,
                scheduled_start_utc=format_canonical_utc(
                    feature_snapshot.scheduled_start_utc
                ),
                feature_as_of_utc=format_canonical_utc(feature_snapshot.as_of_utc),
                fold_id=fold.fold_id,
                fold_fingerprint=fold.fingerprint(),
                model_id=model_artifact.model_id,
                model_fingerprint=model.fingerprint(),
                model_artifact_fingerprint=model_artifact.fingerprint(),
                feature_snapshot_id=f"p19a:{feature_snapshot_fp}",
                feature_snapshot_fingerprint=feature_snapshot_fp,
                feature_snapshot_schema_version=feature_snapshot.schema_version,
                source_schedule_observation_id=feature_snapshot.source_schedule_observation_id,
                feature_names=MONEYLINE_FEATURE_NAMES,
                feature_values=feature_snapshot.feature_vector(),
                feature_lineage=feature_lineage,
                target_home_win=target_home_win,
                historical_result_source_id=str(result["source_result_id"]),
                historical_result_observation_id=result_observation_id,
                historical_result_observed_at_utc=str(result["result_observed_at_utc"]),
                historical_home_score=int(result["home_score"]),
                historical_away_score=int(result["away_score"]),
                historical_result_row_fingerprint=_result_row_fingerprint(result),
                source_candidates=source_candidates,
            )
        )

    if sum(len(rows) for rows in candidate_groups.values()) != len(candidate_rows):
        raise ValueError(
            f"{P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT}: candidate accounting mismatch"
        )
    if len({example.provider_game_id for example in examples}) != len(examples):
        raise ValueError(
            f"{P22A_STOP_SELECTION_PAIR_INCONSISTENT}: duplicate game training example"
        )
    source_fold_artifacts = tuple(
        {
            "fold_id": fold.fold_id,
            "fold_fingerprint": fold.fingerprint(),
            "model_artifact_fingerprint": build_moneyline_model_artifact(
                fold, reconstructed_models[fold.fold_id]
            ).fingerprint(),
            "model_fingerprint": reconstructed_models[fold.fold_id].fingerprint(),
        }
        for fold in ordered_folds
    )
    return MoneylineTrainingDataset(
        examples=tuple(examples),
        eligible_candidate_count=len(candidate_rows),
        source_candidate_artifact_fingerprint=source_candidate_fingerprint,
        source_candidate_artifact_sha256=str(
            candidate_summary["learning_candidates_jsonl_sha256"]
        ),
        source_candidate_summary_sha256=source_candidate_summary_sha256,
        source_assessed_count=int(candidate_summary["p21a_assessed_count"]),
        source_excluded_count=int(candidate_summary["p21a_excluded_count"]),
        source_historical_results_sha256=str(
            candidate_summary["historical_results_sha256"]
        ),
        source_historical_provenance_sha256=str(
            candidate_summary["historical_provenance_sha256"]
        ),
        source_result_rows_fingerprint=str(result_rows_fingerprint),
        source_fold_artifacts=source_fold_artifacts,
        p20b_historical_runtime_compliance=str(
            candidate_summary["p20b_historical_runtime_compliance"]
        ),
    )


__all__ = (
    "MoneylineTrainingDataset",
    "P22A_CLAIMS",
    "P22A_DATASET_CONTRACT_VERSION",
    "P22A_DATASET_SCHEMA_VERSION",
    "P22A_STOP_COMMITTED_LINEAGE_INSUFFICIENT",
    "P22A_STOP_SELECTION_PAIR_INCONSISTENT",
    "P22A_STOP_TARGET_SEMANTICS_UNRESOLVED",
    "materialize_moneyline_training_dataset",
)
