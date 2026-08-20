"""Deterministic Run Line result grading, push settlement, and scoring metrics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RunLineEvaluationResult:
    """Evaluation of one Run Line prediction against authoritative FINAL scores."""

    actual_result: str  # "HOME", "AWAY", or "PUSH"
    is_push: bool
    is_correct: bool | None
    correctness_target: int | None
    brier_component: float | None
    log_loss_component: float | None
    selected_side: str
    selected_cover_probability: float


@dataclass(frozen=True, slots=True)
class RunLineMetricsSummary:
    """Aggregated evaluation metrics for a set of settled Run Line predictions."""

    total_count: int
    push_count: int
    decided_count: int
    correct_count: int
    incorrect_count: int
    accuracy: float | None
    brier_score: float | None
    log_loss: float | None


def settle_run_line(
    home_score: int,
    away_score: int,
    spread_home: float | Decimal,
) -> str:
    """Grade actual Run Line outcome against spread_home."""

    diff = float(home_score - away_score)
    threshold = -float(spread_home)

    if diff > threshold:
        return "HOME"
    elif diff < threshold:
        return "AWAY"
    else:
        return "PUSH"


def evaluate_run_line_prediction(
    predicted_side: str,
    p_home_covers: float,
    p_away_covers: float,
    spread_home: float | Decimal,
    home_score: int,
    away_score: int,
) -> RunLineEvaluationResult:
    """Evaluate one frozen Run Line prediction against final scores."""

    if predicted_side not in ("HOME", "AWAY"):
        raise ValueError("predicted_side must be 'HOME' or 'AWAY'")

    actual = settle_run_line(home_score, away_score, spread_home)
    selected_prob = (
        p_home_covers if predicted_side == "HOME" else p_away_covers
    )

    if actual == "PUSH":
        return RunLineEvaluationResult(
            actual_result="PUSH",
            is_push=True,
            is_correct=None,
            correctness_target=None,
            brier_component=None,
            log_loss_component=None,
            selected_side=predicted_side,
            selected_cover_probability=selected_prob,
        )

    is_correct = (predicted_side == actual)
    target = 1 if is_correct else 0
    p = max(1e-12, min(1.0 - 1e-12, selected_prob))
    brier = (selected_prob - float(target)) ** 2
    log_loss = -math.log(p if is_correct else (1.0 - p))

    return RunLineEvaluationResult(
        actual_result=actual,
        is_push=False,
        is_correct=is_correct,
        correctness_target=target,
        brier_component=brier,
        log_loss_component=log_loss,
        selected_side=predicted_side,
        selected_cover_probability=selected_prob,
    )


def compute_run_line_metrics(
    evaluations: Sequence[RunLineEvaluationResult],
) -> RunLineMetricsSummary:
    """Compute aggregate accuracy, Brier, and log-loss with pushes excluded from denominators."""

    total_count = len(evaluations)
    push_count = sum(1 for e in evaluations if e.is_push)
    decided_evals = [e for e in evaluations if not e.is_push]
    decided_count = len(decided_evals)

    if decided_count == 0:
        return RunLineMetricsSummary(
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

    return RunLineMetricsSummary(
        total_count=total_count,
        push_count=push_count,
        decided_count=decided_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
        brier_score=brier_score,
        log_loss=log_loss,
    )
