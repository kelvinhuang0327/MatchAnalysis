"""Deterministic, dependency-free probability calibration for Moneyline OOS rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any


P38A_CALIBRATION_METHOD = "PLATT_LOGISTIC_RAW_PROBABILITY_LOGIT"
P38A_CALIBRATION_METHOD_VERSION = "p38a.platt_probability_calibrator.v1"
P38A_PROBABILITY_EPSILON = Decimal("0.000001")
P38A_L2_PENALTY = Decimal("0.0001")
P38A_MAX_ITERATIONS = 100
P38A_CONVERGENCE_TOLERANCE = Decimal("1e-24")


def _finite_decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{field_name} must be a finite Decimal")
    return value


def _strict_probability(value: Decimal) -> Decimal:
    value = _finite_decimal(value, "probability")
    if not Decimal("0") < value < Decimal("1"):
        raise ValueError("probability must be strictly between zero and one")
    return value


def _sigmoid(logit: Decimal) -> Decimal:
    """Evaluate sigmoid without allowing a large exponent to overflow."""

    if logit >= Decimal("50"):
        return Decimal("1")
    if logit <= Decimal("-50"):
        return Decimal("0")
    if logit >= Decimal("0"):
        return Decimal("1") / (Decimal("1") + (-logit).exp())
    exponent = logit.exp()
    return exponent / (Decimal("1") + exponent)


def _logit(probability: Decimal) -> Decimal:
    probability = _strict_probability(probability)
    return probability.ln() - (Decimal("1") - probability).ln()


@dataclass(frozen=True, slots=True)
class PlattProbabilityCalibrator:
    """A fixed two-parameter logistic map over raw probability logits."""

    intercept: Decimal
    slope: Decimal
    sample_count: int
    fit_iterations: int
    method: str = P38A_CALIBRATION_METHOD
    method_version: str = P38A_CALIBRATION_METHOD_VERSION
    probability_epsilon: Decimal = P38A_PROBABILITY_EPSILON
    l2_penalty: Decimal = P38A_L2_PENALTY

    def __post_init__(self) -> None:
        _finite_decimal(self.intercept, "intercept")
        _finite_decimal(self.slope, "slope")
        _finite_decimal(self.probability_epsilon, "probability_epsilon")
        _finite_decimal(self.l2_penalty, "l2_penalty")
        if self.method != P38A_CALIBRATION_METHOD:
            raise ValueError("unexpected P38A calibration method")
        if self.method_version != P38A_CALIBRATION_METHOD_VERSION:
            raise ValueError("unexpected P38A calibration method version")
        if self.sample_count < 2:
            raise ValueError("calibrator sample_count must be at least two")
        if self.fit_iterations < 1:
            raise ValueError("calibrator fit_iterations must be positive")
        if not Decimal("0") < self.probability_epsilon < Decimal("0.5"):
            raise ValueError("probability_epsilon must be between zero and one half")
        if self.l2_penalty <= Decimal("0"):
            raise ValueError("l2_penalty must be positive")

    def apply(self, raw_probability: Decimal) -> Decimal:
        """Map one raw probability to a bounded calibrated probability."""

        with localcontext() as context:
            context.prec = 50
            value = _sigmoid(
                self.intercept + self.slope * _logit(raw_probability)
            )
            return min(
                Decimal("1") - self.probability_epsilon,
                max(self.probability_epsilon, value),
            )

    def to_projection(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_version": self.method_version,
            "intercept": str(self.intercept),
            "slope": str(self.slope),
            "sample_count": self.sample_count,
            "fit_iterations": self.fit_iterations,
            "probability_epsilon": str(self.probability_epsilon),
            "l2_penalty": str(self.l2_penalty),
        }


def fit_platt_calibrator(
    raw_probabilities: Sequence[Decimal],
    targets: Sequence[int],
) -> PlattProbabilityCalibrator:
    """Fit one pre-specified Platt map with deterministic Newton updates."""

    if not raw_probabilities or len(raw_probabilities) != len(targets):
        raise ValueError("calibration probabilities and targets must be aligned")
    if len(raw_probabilities) < 2:
        raise ValueError("calibration requires at least two observations")
    normalized_targets = tuple(int(target) for target in targets)
    if any(target not in (0, 1) for target in normalized_targets):
        raise ValueError("calibration targets must be binary")
    if set(normalized_targets) != {0, 1}:
        raise ValueError("calibration requires both target classes")

    with localcontext() as context:
        context.prec = 50
        logits = tuple(_logit(value) for value in raw_probabilities)
        target_values = tuple(Decimal(target) for target in normalized_targets)
        smoothed_rate = (
            sum(target_values, Decimal("0")) + Decimal("0.5")
        ) / Decimal(len(target_values) + 1)
        intercept = _logit(smoothed_rate)
        slope = Decimal("1")
        converged = False
        fit_iterations = 0

        for iteration in range(1, P38A_MAX_ITERATIONS + 1):
            gradient_intercept = P38A_L2_PENALTY * intercept
            gradient_slope = P38A_L2_PENALTY * slope
            hessian_ii = P38A_L2_PENALTY
            hessian_is = Decimal("0")
            hessian_ss = P38A_L2_PENALTY
            for logit_value, target in zip(logits, target_values, strict=True):
                fitted = _sigmoid(intercept + slope * logit_value)
                residual = fitted - target
                weight = fitted * (Decimal("1") - fitted)
                gradient_intercept += residual
                gradient_slope += residual * logit_value
                hessian_ii += weight
                hessian_is += weight * logit_value
                hessian_ss += weight * logit_value * logit_value

            determinant = hessian_ii * hessian_ss - hessian_is * hessian_is
            if determinant == Decimal("0"):
                raise ValueError("P38A calibrator Hessian is singular")
            delta_intercept = (
                gradient_intercept * hessian_ss
                - gradient_slope * hessian_is
            ) / determinant
            delta_slope = (
                gradient_slope * hessian_ii
                - gradient_intercept * hessian_is
            ) / determinant
            intercept -= delta_intercept
            slope -= delta_slope
            fit_iterations = iteration
            if max(abs(delta_intercept), abs(delta_slope)) <= P38A_CONVERGENCE_TOLERANCE:
                converged = True
                break

        if not converged:
            raise ValueError("P38A calibrator did not converge")
        return PlattProbabilityCalibrator(
            intercept=intercept,
            slope=slope,
            sample_count=len(raw_probabilities),
            fit_iterations=fit_iterations,
        )


def _log_loss_component(probability: Decimal, target: int) -> Decimal:
    probability = _strict_probability(probability)
    if target not in (0, 1):
        raise ValueError("target must be binary")
    return -(probability.ln() if target else (Decimal("1") - probability).ln())


def _calibration_summary(
    probabilities: Sequence[Decimal],
    targets: Sequence[int],
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("metric probabilities and targets must be aligned")
    bins = (
        ("0.00-0.25", Decimal("0"), Decimal("0.25"), False),
        ("0.25-0.50", Decimal("0.25"), Decimal("0.50"), False),
        ("0.50-0.75", Decimal("0.50"), Decimal("0.75"), False),
        ("0.75-1.00", Decimal("0.75"), Decimal("1.00"), True),
    )
    projections: list[dict[str, Any]] = []
    weighted_gap = Decimal("0")
    total = Decimal(len(probabilities))
    for label, lower, upper, include_upper in bins:
        selected = [
            (probability, target)
            for probability, target in zip(probabilities, targets, strict=True)
            if lower <= probability <= upper
            if include_upper or probability < upper
        ]
        if not selected:
            projections.append(
                {
                    "bin": label,
                    "count": 0,
                    "mean_predicted_probability": None,
                    "observed_home_win_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        count = Decimal(len(selected))
        mean_probability = sum(
            (probability for probability, _target in selected), Decimal("0")
        ) / count
        observed_rate = sum(
            (Decimal(target) for _probability, target in selected), Decimal("0")
        ) / count
        gap = abs(mean_probability - observed_rate)
        weighted_gap += (count / total) * gap
        projections.append(
            {
                "bin": label,
                "count": len(selected),
                "mean_predicted_probability": str(mean_probability),
                "observed_home_win_rate": str(observed_rate),
                "absolute_gap": str(gap),
            }
        )
    return {
        "bin_count": len(projections),
        "expected_calibration_error": str(weighted_gap),
        "bins": projections,
    }


def probability_metrics(
    probabilities: Sequence[Decimal],
    targets: Sequence[int],
    *,
    raw_row_count: int,
) -> dict[str, Any]:
    """Calculate the P37A-compatible metrics for one exact row population."""

    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("metric probabilities and targets must be aligned")
    if raw_row_count < len(probabilities):
        raise ValueError("raw_row_count cannot be smaller than evaluable rows")
    normalized_targets = tuple(int(target) for target in targets)
    if any(target not in (0, 1) for target in normalized_targets):
        raise ValueError("metric targets must be binary")
    normalized_probabilities = tuple(
        _strict_probability(probability) for probability in probabilities
    )
    count = Decimal(len(normalized_probabilities))
    correct = sum(
        (probability >= Decimal("0.5")) == bool(target)
        for probability, target in zip(
            normalized_probabilities,
            normalized_targets,
            strict=True,
        )
    )
    brier = sum(
        (
            (probability - Decimal(target)) ** 2
            for probability, target in zip(
                normalized_probabilities,
                normalized_targets,
                strict=True,
            )
        ),
        Decimal("0"),
    ) / count
    log_loss = sum(
        (
            _log_loss_component(probability, target)
            for probability, target in zip(
                normalized_probabilities,
                normalized_targets,
                strict=True,
            )
        ),
        Decimal("0"),
    ) / count
    return {
        "row_count": len(normalized_probabilities),
        "accuracy": str(Decimal(correct) / count),
        "brier_score": str(brier),
        "log_loss": str(log_loss),
        "coverage": str(Decimal(len(normalized_probabilities)) / Decimal(raw_row_count)),
        "calibration": _calibration_summary(
            normalized_probabilities,
            normalized_targets,
        ),
    }


__all__ = (
    "P38A_CALIBRATION_METHOD",
    "P38A_CALIBRATION_METHOD_VERSION",
    "P38A_CONVERGENCE_TOLERANCE",
    "P38A_L2_PENALTY",
    "P38A_MAX_ITERATIONS",
    "P38A_PROBABILITY_EPSILON",
    "PlattProbabilityCalibrator",
    "fit_platt_calibrator",
    "probability_metrics",
)
