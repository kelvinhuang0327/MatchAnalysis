"""Reconstruct the bounded committed P13 Moneyline fold model."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Union

from ...baseball.domain.moneyline_walk_forward_fold import (
    P20A_MAX_ITER,
    P20A_SOLVER,
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)


STOP_MATCHANALYSIS_P20A_DEPENDENCY_REQUIREMENT = (
    "STOP_MATCHANALYSIS_P20A_DEPENDENCY_REQUIREMENT"
)


class P20ADependencyRequirement(RuntimeError):
    """Raised when the exact committed P13 fitting backend is unavailable."""


def load_moneyline_walk_forward_fold(
    path: Union[str, Path],
) -> MoneylineWalkForwardFold:
    """Load one explicit committed bounded fold without network or database access."""

    fixture_path = Path(path)
    projection = json.loads(fixture_path.read_text(encoding="utf-8"))
    return MoneylineWalkForwardFold.from_projection(projection)


def reconstruct_moneyline_walk_forward_model(
    fold: MoneylineWalkForwardFold,
) -> ReconstructedWalkForwardModel:
    """Fit exactly the legacy P13 standardized logistic model for one fold."""

    if not isinstance(fold, MoneylineWalkForwardFold):
        raise TypeError("fold must be a MoneylineWalkForwardFold")
    if not fold.point_in_time_safe():
        raise ValueError("fold is not point-in-time safe")

    try:
        np = importlib.import_module("numpy")
        LogisticRegression = importlib.import_module(
            "sklearn.linear_model"
        ).LogisticRegression
    except Exception as exc:  # pragma: no cover - exercised by environment gate
        raise P20ADependencyRequirement(
            f"{STOP_MATCHANALYSIS_P20A_DEPENDENCY_REQUIREMENT}: "
            "NumPy and scikit-learn are required by the committed P13 fit"
        ) from exc

    feature_count = len(fold.feature_names)
    X = np.array(
        [row.feature_vector for row in fold.training_rows],
        dtype=float,
    ).reshape((-1, feature_count))
    y = np.array(
        [int(row.target_home_win) for row in fold.training_rows],
        dtype=int,
    )

    # These operations intentionally mirror the committed P13 implementation:
    # fold-only mean/std standardization, constant-column std fallback, then
    # LogisticRegression(max_iter=1000, solver="lbfgs") with default penalty/C.
    means = X.mean(axis=0)
    stds = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
    model = LogisticRegression(max_iter=P20A_MAX_ITER, solver=P20A_SOLVER)
    model.fit((X - means) / stds, y)

    return ReconstructedWalkForwardModel(
        fold_id=fold.fold_id,
        feature_names=fold.feature_names,
        coefficients=tuple(float(value) for value in model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
        scaler_means=tuple(float(value) for value in means.tolist()),
        scaler_stds=tuple(float(value) for value in stds.tolist()),
        train_size=len(fold.training_rows),
    )


def reconstruct_moneyline_walk_forward_model_from_path(
    path: Union[str, Path],
) -> tuple[MoneylineWalkForwardFold, ReconstructedWalkForwardModel]:
    """Load and fit the bounded fold in one deterministic operation."""

    fold = load_moneyline_walk_forward_fold(path)
    return fold, reconstruct_moneyline_walk_forward_model(fold)
