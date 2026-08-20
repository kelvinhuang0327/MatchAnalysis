"""Deterministic Total result grading, push settlement, and scoring metrics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TotalEvaluationResult:
    """Evaluation of one Total prediction against authoritative FINAL scores."""

    actual_result: str  # "OVER", "UNDER", or "PUSH"
    is_push: bool
    is_correct: bool | None
    correctness_target: int | None
    brier_component: float | None
    log_loss_component: float | None
    selected_side: str
    selected_total_probability: float


@dataclass(frozen=True, slots=True)
class TotalMetricsSummary:
    """Aggregated evaluation metrics for a set of settled Total predictions."""

    total_count: int
    push_count: int
    decided_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float | None
    brier_score: float | None
    log_loss: float | None


def settle_total(
    home_score: int,
    away_score: int,
    total_line: float | Decimal,
) -> str:
    """Grade actual Total outcome against total_line."""

    actual_total = float(home_score + away_score)
    line_f = float(total_line)

    if actual_total > line_f:
        return "OVER"
    elif actual_total < line_f:
        return "UNDER"
    else:
        return "PUSH"


def evaluate_total_prediction(
    predicted_side: str,
    p_over: float,
    p_under: float,
    total_line: float | Decimal,
    home_score: int,
    away_score: int,
) -> TotalEvaluationResult:
    """Evaluate one frozen Total prediction against final scores."""

    if predicted_side not in ("OVER", "UNDER"):
        raise ValueError("predicted_side must be 'OVER' or 'UNDER'")

    actual = settle_total(home_score, away_score, total_line)
    selected_prob = (
        p_over if predicted_side == "OVER" else p_under
    )

    if actual == "PUSH":
        return TotalEvaluationResult(
            actual_result="PUSH",
            is_push=True,
            is_correct=None,
            correctness_target=None,
            brier_component=None,
            log_loss_component=None,
            selected_side=predicted_side,
            selected_total_probability=selected_prob,
        )

    is_correct = (predicted_side == actual)
    target = 1 if is_correct else 0
    p = max(1e-12, min(1.0 - 1e-12, selected_prob))
    brier = (selected_prob - float(target)) ** 2
    log_loss = -math.log(p if is_correct else (1.0 - p))

    return TotalEvaluationResult(
        actual_result=actual,
        is_push=False,
        is_correct=is_correct,
        correctness_target=target,
        brier_component=brier,
        log_loss_component=log_loss,
        selected_side=predicted_side,
        selected_total_probability=selected_prob,
    )


def compute_total_metrics(
    evaluations: Sequence[TotalEvaluationResult],
) -> TotalMetricsSummary:
    """Compute aggregate accuracy, Brier, and log-loss with pushes excluded from denominators."""

    total_count = len(evaluations)
    push_count = sum(1 for e in evaluations if e.is_push)
    decided_evals = [e for e in evaluations if not e.is_push]
    decided_count = len(decided_evals)

    if decided_count == 0:
        return TotalMetricsSummary(
            total_count=total_count,
            push_count=push_count,
            decided_count=0,
            correct_count=0,
            incorrect_count=0,
            accuracy=None,
            brier_score=None,
            log_loss=None,
        )

    correct_count = sum(1 for e in decided_evals if e.is_correct is True)
    incorrect_count = decided_count - correct_count
    accuracy = correct_count / float(decided_count)
    brier_score = sum(
        e.brier_component for e in decided_evals if e.brier_component is not None
    ) / float(decided_count)
    log_loss = sum(
        e.log_loss_component
        for e in decided_evals
        if e.log_loss_component is not None
    ) / float(decided_count)

    return TotalMetricsSummary(
        total_count=total_count,
        push_count=push_count,
        decided_count=decided_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
        brier_score=brier_score,
        log_loss=log_loss,
    )
