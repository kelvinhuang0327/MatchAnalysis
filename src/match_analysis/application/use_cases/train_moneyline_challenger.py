"""Train exactly one deterministic P22B Moneyline challenger model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from ...baseball.domain.moneyline_feature_snapshot import MONEYLINE_FEATURE_NAMES
from ...baseball.domain.moneyline_walk_forward_fold import (
    P20A_MAX_ITER,
    P20A_MODEL_TYPE,
    P20A_SOLVER,
)
from ...baseball.domain.supervised_training_example import (
    SupervisedTrainingExample,
)
from .moneyline_challenger_artifacts import (
    MoneylineChallengerArtifact,
    build_moneyline_challenger_artifact,
)


P22B_DATASET_FINGERPRINT = (
    "05f9b31c608e1630a40b2369ac45ada8b103b2c1131b0cedcaa2c7fc91ba7750"
)
P22B_FEATURE_NAMES = MONEYLINE_FEATURE_NAMES
P22B_DATASET_SCHEMA_VERSION = "p22a.game_level_supervised_training_dataset.v1"
P22B_DATASET_CONTRACT_VERSION = (
    "p22a.game_level_supervised_training_dataset_contract.v1"
)
P22B_LABEL_SEMANTICS = (
    "target_home_win=1 iff committed FINAL home_score is greater than away_score; "
    "target_home_win=0 iff away_score is greater"
)
P22B_FIT_CONFIGURATION = {
    "max_iter": P20A_MAX_ITER,
    "model_type": P20A_MODEL_TYPE,
    "solver": P20A_SOLVER,
}
P22B_DEFAULT_FIT_RUNTIME = "/usr/bin/python3"
P22B_DEFAULT_SOURCE_COMMIT = "01d4ac2544710280a03c617ba1016b2450a2414b"
P22B_DEFAULT_SOURCE_TREE = "050552b94ca582547a8c84f9fa8a479d9abaa9f0"
P22B_STOP_FIT_RUNTIME_UNAVAILABLE = "STOP_MATCHANALYSIS_P22B_FIT_RUNTIME_UNAVAILABLE"
P22B_TRAINING_CODE_CONTRACT = "p22b.deterministic_moneyline_challenger_training.v1"

_P13_FIT_SCRIPT = r'''
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

payload = json.load(sys.stdin)
X = np.asarray(payload["feature_rows"], dtype=float)
y = np.asarray(payload["labels"], dtype=int)
means = X.mean(axis=0)
raw_stds = X.std(axis=0)
stds = np.where(raw_stds < 1e-8, 1.0, raw_stds)
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit((X - means) / stds, y)
json.dump(
    {
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "scaler_means": [float(value) for value in means.tolist()],
        "scaler_stds": [float(value) for value in stds.tolist()],
        "runtime": {
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "sklearn_version": __import__("sklearn").__version__,
        },
    },
    sys.stdout,
    separators=(",", ":"),
    sort_keys=True,
)
'''


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _read_jsonl(path: Path) -> tuple[SupervisedTrainingExample, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unable to read P22A dataset: {path}") from exc
    if not lines:
        raise ValueError("P22B dataset must contain at least one row")
    examples: list[SupervisedTrainingExample] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"P22B dataset contains a blank line at {line_number}")
        try:
            projection = json.loads(line)
            examples.append(SupervisedTrainingExample.from_projection(projection))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid P22A training example at line {line_number}") from exc
    return tuple(examples)


def _dataset_fingerprint(examples: tuple[SupervisedTrainingExample, ...]) -> str:
    ordered = sorted(examples, key=lambda example: example.training_example_id)
    return _sha256(
        _canonical_json_bytes(
            {
                "contract_version": P22B_DATASET_CONTRACT_VERSION,
                "schema_version": P22B_DATASET_SCHEMA_VERSION,
                "training_examples": [
                    example.to_projection() for example in ordered
                ],
            }
        )
    )


@dataclass(frozen=True, slots=True)
class P22ATrainingDataset:
    """Validated P22A examples and their committed source identities."""

    examples: tuple[SupervisedTrainingExample, ...]
    dataset_fingerprint: str
    training_examples_jsonl_sha256: str
    label_distribution: dict[str, int]

    @property
    def ordered_examples(self) -> tuple[SupervisedTrainingExample, ...]:
        return tuple(sorted(self.examples, key=lambda example: example.training_example_id))


def load_p22a_training_dataset(
    dataset_path: str | Path,
    summary_path: str | Path,
) -> P22ATrainingDataset:
    """Load and verify the exact committed P22A dataset authority."""

    dataset_file = Path(dataset_path)
    summary_file = Path(summary_path)
    if not dataset_file.is_file() or not summary_file.is_file():
        raise ValueError("P22A dataset and summary must both be existing files")
    raw_dataset = dataset_file.read_bytes()
    examples = _read_jsonl(dataset_file)
    ordered = tuple(sorted(examples, key=lambda example: example.training_example_id))
    if len(ordered) != 677:
        raise ValueError("P22B requires exactly 677 P22A training examples")
    if len({example.training_example_id for example in ordered}) != len(ordered):
        raise ValueError("P22A training example IDs must be unique")
    if len({example.provider_game_id for example in ordered}) != len(ordered):
        raise ValueError("P22A examples must remain one row per game")
    if any(example.feature_names != MONEYLINE_FEATURE_NAMES for example in ordered):
        raise ValueError("P22A feature schema/order does not match P13")
    label_distribution = dict(
        sorted(Counter(str(example.target_home_win) for example in ordered).items())
    )
    if label_distribution != {"0": 309, "1": 368}:
        raise ValueError("P22A label distribution does not match the committed authority")
    dataset_fingerprint = _dataset_fingerprint(ordered)
    if dataset_fingerprint != P22B_DATASET_FINGERPRINT:
        raise ValueError("P22B dataset fingerprint does not match committed authority")
    training_examples_jsonl_sha256 = _sha256(raw_dataset)
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read P22A summary") from exc
    expected_summary = {
        "dataset_fingerprint": P22B_DATASET_FINGERPRINT,
        "feature_names": list(MONEYLINE_FEATURE_NAMES),
        "label_distribution": label_distribution,
        "training_example_count": 677,
        "training_examples_jsonl_sha256": training_examples_jsonl_sha256,
        "unmapped_candidate_count": 0,
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
    }
    for field_name, expected in expected_summary.items():
        if summary.get(field_name) != expected:
            raise ValueError(f"P22A summary authority mismatch: {field_name}")
    if summary.get("training_example_id_includes_target") is not False:
        raise ValueError("P22A training example identity must exclude the target")
    return P22ATrainingDataset(
        examples=ordered,
        dataset_fingerprint=dataset_fingerprint,
        training_examples_jsonl_sha256=training_examples_jsonl_sha256,
        label_distribution=label_distribution,
    )


@dataclass(frozen=True, slots=True)
class FittedChallengerState:
    """The exact fitted state returned by the migrated P13 settings."""

    coefficients: tuple[float, ...]
    intercept: float
    scaler_means: tuple[float, ...]
    scaler_stds: tuple[float, ...]
    runtime: dict[str, str]

    def to_projection(self) -> dict[str, Any]:
        return {
            "coefficients": [str(Decimal(repr(value))) for value in self.coefficients],
            "intercept": str(Decimal(repr(self.intercept))),
            "scaler_means": [str(Decimal(repr(value))) for value in self.scaler_means],
            "scaler_stds": [str(Decimal(repr(value))) for value in self.scaler_stds],
        }


def fit_moneyline_challenger(
    dataset: P22ATrainingDataset,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
) -> FittedChallengerState:
    """Fit one P13-standardized logistic model in an existing local runtime."""

    runtime = Path(fit_runtime)
    if not runtime.is_file():
        raise RuntimeError(f"{P22B_STOP_FIT_RUNTIME_UNAVAILABLE}: {runtime}")
    payload = {
        "feature_rows": [
            [str(value) for value in example.feature_values]
            for example in dataset.ordered_examples
        ],
        "labels": [example.target_home_win for example in dataset.ordered_examples],
    }
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [str(runtime), "-c", _P13_FIT_SCRIPT],
            input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{P22B_STOP_FIT_RUNTIME_UNAVAILABLE}: {runtime}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "fit runtime returned a non-zero status"
        raise RuntimeError(f"{P22B_STOP_FIT_RUNTIME_UNAVAILABLE}: {detail}")
    try:
        projection = json.loads(result.stdout)
        runtime_projection = projection["runtime"]
        state = FittedChallengerState(
            coefficients=tuple(float(value) for value in projection["coefficients"]),
            intercept=float(projection["intercept"]),
            scaler_means=tuple(float(value) for value in projection["scaler_means"]),
            scaler_stds=tuple(float(value) for value in projection["scaler_stds"]),
            runtime={
                "executable": str(runtime),
                "numpy_version": str(runtime_projection["numpy_version"]),
                "python_version": str(runtime_projection["python_version"]),
                "sklearn_version": str(runtime_projection["sklearn_version"]),
            },
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{P22B_STOP_FIT_RUNTIME_UNAVAILABLE}: invalid fit output") from exc
    if (
        len(state.coefficients) != len(MONEYLINE_FEATURE_NAMES)
        or len(state.scaler_means) != len(MONEYLINE_FEATURE_NAMES)
        or len(state.scaler_stds) != len(MONEYLINE_FEATURE_NAMES)
        or any(value == 0 for value in state.scaler_stds)
    ):
        raise RuntimeError("P22B fit produced an invalid fitted state")
    return state


def train_moneyline_challenger(
    dataset_path: str | Path,
    summary_path: str | Path,
    *,
    fit_runtime: str | Path = P22B_DEFAULT_FIT_RUNTIME,
    source_repository: str,
    source_commit: str = P22B_DEFAULT_SOURCE_COMMIT,
    source_tree: str = P22B_DEFAULT_SOURCE_TREE,
) -> MoneylineChallengerArtifact:
    """Load, fit, and materialize exactly one deterministic challenger."""

    dataset = load_p22a_training_dataset(dataset_path, summary_path)
    fitted_state = fit_moneyline_challenger(dataset, fit_runtime=fit_runtime)
    return build_moneyline_challenger_artifact(
        dataset=dataset,
        fitted_state=fitted_state,
        source_repository=source_repository,
        source_commit=source_commit,
        source_tree=source_tree,
    )


__all__ = (
    "FittedChallengerState",
    "P22B_DATASET_CONTRACT_VERSION",
    "P22B_DATASET_FINGERPRINT",
    "P22B_DATASET_SCHEMA_VERSION",
    "P22B_DEFAULT_FIT_RUNTIME",
    "P22B_DEFAULT_SOURCE_COMMIT",
    "P22B_DEFAULT_SOURCE_TREE",
    "P22B_FIT_CONFIGURATION",
    "P22B_LABEL_SEMANTICS",
    "P22B_STOP_FIT_RUNTIME_UNAVAILABLE",
    "P22B_TRAINING_CODE_CONTRACT",
    "P22ATrainingDataset",
    "fit_moneyline_challenger",
    "load_p22a_training_dataset",
    "train_moneyline_challenger",
)
