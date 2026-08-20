"""Deterministic Run Line Poisson team-rate model and point-in-time state.

Implements the raw P226 Run Line model semantics:
- Model: poisson_team_rate_model
- Version: p226a.raw.v1
- Point-in-time pseudo-count shrinkage of team runs-for and runs-against rates.
- Skellam discrete Poisson convolution (bounded 0..40 runs) for run line cover probabilities.
- Structural support for half-run and integer lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from .canonical_utc import format_canonical_utc, parse_canonical_utc


MODEL_ID = "poisson_team_rate_model"
MODEL_VERSION = "p226a.raw.v1"
MARKET_ID = "run_line"

RUN_SMOOTH_K: float = 10.0
DEFAULT_LEAGUE_AVG_RUNS: float = 4.3
POISSON_MAX_RUNS: int = 40
FROZEN_HOME_ADVANTAGE: float = 1.0115984108834084


@dataclass(frozen=True, slots=True)
class TeamRateState:
    """Cumulative point-in-time scoring history for one team."""

    runs_for: float
    runs_against: float
    games_played: int


@dataclass(frozen=True, slots=True)
class PriorGameRecord:
    """Immutable record of a prior completed game used for point-in-time rate state."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    completed_at_utc: datetime

    def __post_init__(self) -> None:
        if not self.home_team or not self.away_team:
            raise ValueError("home_team and away_team must be non-empty strings")
        if self.home_score < 0 or self.away_score < 0:
            raise ValueError("scores must be non-negative integers")
        if self.completed_at_utc.tzinfo is None:
            raise ValueError("completed_at_utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class RunLineProbabilities:
    """Computed cover probabilities and predicted side for one run line contest."""

    p_home_covers: float
    p_away_covers: float
    p_push: float
    predicted_side: str
    selected_cover_probability: float
    spread_home: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_home_covers <= 1.000001):
            raise ValueError("p_home_covers out of bounds")
        if not (0.0 <= self.p_away_covers <= 1.000001):
            raise ValueError("p_away_covers out of bounds")
        if not (0.0 <= self.p_push <= 1.000001):
            raise ValueError("p_push out of bounds")
        if self.predicted_side not in ("HOME", "AWAY"):
            raise ValueError("predicted_side must be 'HOME' or 'AWAY'")


def compute_team_rate_state(
    prior_games: Sequence[PriorGameRecord],
    cutoff_utc: datetime,
) -> tuple[dict[str, TeamRateState], float]:
    """Compute point-in-time team rate state from prior FINAL games strictly before cutoff."""

    if cutoff_utc.tzinfo is None:
        raise ValueError("cutoff_utc must be timezone-aware")

    filtered = [
        g for g in prior_games if g.completed_at_utc <= cutoff_utc
    ]
    if not filtered:
        return {}, DEFAULT_LEAGUE_AVG_RUNS

    total_runs = sum(g.home_score + g.away_score for g in filtered)
    league_avg = total_runs / (2.0 * len(filtered))
    if league_avg <= 0.0:
        league_avg = DEFAULT_LEAGUE_AVG_RUNS

    stats: dict[str, list[float]] = {}
    for g in filtered:
        if g.home_team not in stats:
            stats[g.home_team] = [0.0, 0.0, 0.0]
        stats[g.home_team][0] += float(g.home_score)
        stats[g.home_team][1] += float(g.away_score)
        stats[g.home_team][2] += 1.0

        if g.away_team not in stats:
            stats[g.away_team] = [0.0, 0.0, 0.0]
        stats[g.away_team][0] += float(g.away_score)
        stats[g.away_team][1] += float(g.home_score)
        stats[g.away_team][2] += 1.0

    team_states = {
        team: TeamRateState(
            runs_for=vals[0],
            runs_against=vals[1],
            games_played=int(vals[2]),
        )
        for team, vals in stats.items()
    }
    return team_states, league_avg


def compute_team_rates(
    team_state: TeamRateState | None,
    league_avg: float,
    k: float = RUN_SMOOTH_K,
) -> tuple[float, float]:
    """Compute shrinkage-smoothed offensive and defensive scoring rates."""

    if team_state is None or team_state.games_played == 0:
        rf = 0.0
        ra = 0.0
        gp = 0.0
    else:
        rf = float(team_state.runs_for)
        ra = float(team_state.runs_against)
        gp = float(team_state.games_played)

    off = (rf + k * league_avg) / (gp + k)
    def_rate = (ra + k * league_avg) / (gp + k)
    return off, def_rate


def compute_poisson_lambdas(
    home_team: str,
    away_team: str,
    team_states: Mapping[str, TeamRateState],
    league_avg: float,
    frozen_home_advantage: float = FROZEN_HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Compute expected scoring Poisson lambdas for home and away teams."""

    off_home, def_home = compute_team_rates(team_states.get(home_team), league_avg)
    off_away, def_away = compute_team_rates(team_states.get(away_team), league_avg)

    lambda_home_raw = (off_home * def_away) / league_avg
    lambda_away_raw = (off_away * def_home) / league_avg

    lambda_home = lambda_home_raw * frozen_home_advantage
    lambda_away = lambda_away_raw
    return lambda_home, lambda_away


def poisson_pmf_array(lam: float, n_max: int = POISSON_MAX_RUNS) -> list[float]:
    """Compute recursive Poisson PMF array for 0..n_max."""

    arr = [0.0] * (n_max + 1)
    if lam <= 0.0:
        arr[0] = 1.0
        return arr
    arr[0] = math.exp(-lam)
    for k in range(1, n_max + 1):
        arr[k] = arr[k - 1] * lam / float(k)
    return arr


def compute_run_line_probabilities(
    lambda_home: float,
    lambda_away: float,
    spread_home: float | Decimal,
    n_max: int = POISSON_MAX_RUNS,
) -> RunLineProbabilities:
    """Compute discrete Poisson/Skellam convolution cover probabilities."""

    spread_f = float(spread_home)
    threshold = -spread_f
    home_pmf = poisson_pmf_array(lambda_home, n_max)
    away_pmf = poisson_pmf_array(lambda_away, n_max)

    p_home = 0.0
    p_away = 0.0
    p_push = 0.0

    for h in range(n_max + 1):
        ph = home_pmf[h]
        if ph == 0.0:
            continue
        for a in range(n_max + 1):
            p = ph * away_pmf[a]
            d = float(h - a)
            if d > threshold:
                p_home += p
            elif d < threshold:
                p_away += p
            else:
                p_push += p

    p_home_clipped = max(0.0, min(1.0, p_home))
    p_away_clipped = max(0.0, min(1.0, p_away))
    p_push_clipped = max(0.0, min(1.0, p_push))

    predicted_side = "HOME" if p_home_clipped >= 0.5 else "AWAY"
    selected_cover_prob = (
        p_home_clipped if predicted_side == "HOME" else p_away_clipped
    )

    return RunLineProbabilities(
        p_home_covers=p_home_clipped,
        p_away_covers=p_away_clipped,
        p_push=p_push_clipped,
        predicted_side=predicted_side,
        selected_cover_probability=selected_cover_prob,
        spread_home=spread_f,
    )
