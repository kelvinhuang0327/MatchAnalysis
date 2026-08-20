"""Deterministic Total Poisson team-rate model and point-in-time state.

Implements the raw P226 Total model semantics:
- Model: poisson_team_rate_model
- Version: p226a.raw.v1
- Market: total
- Point-in-time pseudo-count shrinkage of team runs-for and runs-against rates (reused from P61B).
- Discrete Poisson 2D convolution (bounded 0..40 runs) for Total (Over/Under/Push) probabilities.
- Structural support for half-run and integer lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .run_line_model import (
    DEFAULT_LEAGUE_AVG_RUNS,
    FROZEN_HOME_ADVANTAGE,
    POISSON_MAX_RUNS,
    RUN_SMOOTH_K,
    PriorGameRecord,
    TeamRateState,
    compute_poisson_lambdas,
    compute_team_rate_state,
    compute_team_rates,
    poisson_pmf_array,
)


MODEL_ID = "poisson_team_rate_model"
MODEL_VERSION = "p226a.raw.v1"
MARKET_ID = "total"


@dataclass(frozen=True, slots=True)
class TotalProbabilities:
    """Computed Over/Under/Push probabilities and predicted side for one Total contest."""

    p_over: float
    p_under: float
    p_push: float
    predicted_side: str
    selected_total_probability: float
    total_line: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_over <= 1.000001):
            raise ValueError("p_over out of bounds")
        if not (0.0 <= self.p_under <= 1.000001):
            raise ValueError("p_under out of bounds")
        if not (0.0 <= self.p_push <= 1.000001):
            raise ValueError("p_push out of bounds")
        if self.predicted_side not in ("OVER", "UNDER"):
            raise ValueError("predicted_side must be 'OVER' or 'UNDER'")


def compute_total_probabilities(
    lambda_home: float,
    lambda_away: float,
    total_line: float | Decimal,
    n_max: int = POISSON_MAX_RUNS,
) -> TotalProbabilities:
    """Compute discrete Poisson 2D convolution for Total (Over/Under/Push) probabilities."""

    line_f = float(total_line)
    home_pmf = poisson_pmf_array(lambda_home, n_max)
    away_pmf = poisson_pmf_array(lambda_away, n_max)

    p_over = 0.0
    p_under = 0.0
    p_push = 0.0

    for h in range(n_max + 1):
        ph = home_pmf[h]
        if ph == 0.0:
            continue
        for a in range(n_max + 1):
            p = ph * away_pmf[a]
            tot = float(h + a)
            if tot > line_f:
                p_over += p
            elif tot < line_f:
                p_under += p
            else:
                p_push += p

    p_over_clipped = max(0.0, min(1.0, p_over))
    p_under_clipped = max(0.0, min(1.0, p_under))
    p_push_clipped = max(0.0, min(1.0, p_push))

    predicted_side = "OVER" if p_over_clipped >= 0.5 else "UNDER"
    selected_total_prob = (
        p_over_clipped if predicted_side == "OVER" else p_under_clipped
    )

    return TotalProbabilities(
        p_over=p_over_clipped,
        p_under=p_under_clipped,
        p_push=p_push_clipped,
        predicted_side=predicted_side,
        selected_total_probability=selected_total_prob,
        total_line=line_f,
    )
