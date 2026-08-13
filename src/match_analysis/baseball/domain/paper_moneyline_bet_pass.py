"""Pure P40A paper Moneyline BET/PASS decisions and settlement metrics.

The module intentionally contains no model fitting, market acquisition, or
outcome lookup.  A decision is created from a frozen pregame probability and
the two offered decimal prices.  A later settlement attaches one already
authoritative final outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
import json
from typing import Any, Iterable


P40A_DECISION_SCHEMA_VERSION = "p40a.moneyline_paper_decision.v1"
P40A_SETTLEMENT_SCHEMA_VERSION = "p40a.moneyline_paper_settlement.v1"
P40A_POLICY_ID = "P40A_ZERO_EV_MONEYLINE_BET_PASS_V1"
PAPER_STAKE_CONVENTION = "1.0_PAPER_UNIT_PER_BET"

DECISION_BET = "BET"
DECISION_PASS = "PASS"
SETTLEMENT_WON = "BET_WON"
SETTLEMENT_LOST = "BET_LOST"
SETTLEMENT_PASS = "PASS"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256_projection(value: Any) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _decimal(value: Decimal | str | int | float, *, field_name: str) -> Decimal:
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _require_text(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed non-empty string")


def _require_sha256(value: Any, *, field_name: str) -> None:
    _require_text(value, field_name=field_name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _require_positive_int(value: Any, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == Decimal("0"):
        raise ValueError("ratio denominator must not be zero")
    with localcontext() as context:
        context.prec = 50
        return numerator / denominator


def _validate_probability(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or not Decimal("0") < value < Decimal("1"):
        raise ValueError(f"{field_name} must be strictly between zero and one")


def _validate_price(value: Decimal, *, field_name: str) -> None:
    if not value.is_finite() or value <= Decimal("1"):
        raise ValueError(f"{field_name} must be a finite decimal price greater than one")


@dataclass(frozen=True, slots=True)
class PaperMoneylineDecision:
    """One immutable pre-outcome P40A decision record."""

    decision_id: str
    model_role: str
    model_id: str
    model_fingerprint: str
    p37_fold_id: str
    p37_window: str
    p37_prediction_row_id: str
    provider_namespace: str
    provider_game_id: str
    game_pk: int
    game_number: int
    official_date: str
    scheduled_start_utc: str
    home_team: str
    away_team: str
    home_team_code: str
    away_team_code: str
    market_snapshot_id: str
    market_observed_at_utc: str
    local_fetched_at_utc: str
    source_match_id: str
    market_source_sha256: str
    p37_comparisons_sha256: str
    model_probability_source: str
    p_home: Decimal
    p_away: Decimal
    home_decimal_odds: Decimal
    away_decimal_odds: Decimal
    ev_home: Decimal
    ev_away: Decimal
    candidate_side: str
    candidate_ev: Decimal
    decision: str
    paper_stake_units: Decimal
    paper_stake_convention: str
    home_raw_implied_probability: Decimal
    away_raw_implied_probability: Decimal
    bookmaker_overround: Decimal
    home_no_vig_probability: Decimal
    away_no_vig_probability: Decimal
    model_home_vs_no_vig_edge: Decimal
    model_away_vs_no_vig_edge: Decimal

    @classmethod
    def create(
        cls,
        *,
        model_role: str,
        model_id: str,
        model_fingerprint: str,
        p37_fold_id: str,
        p37_window: str,
        p37_prediction_row_id: str,
        provider_namespace: str,
        provider_game_id: str,
        game_pk: int,
        game_number: int,
        official_date: str,
        scheduled_start_utc: str,
        home_team: str,
        away_team: str,
        home_team_code: str,
        away_team_code: str,
        market_snapshot_id: str,
        market_observed_at_utc: str,
        local_fetched_at_utc: str,
        source_match_id: str,
        market_source_sha256: str,
        p37_comparisons_sha256: str,
        model_probability_source: str,
        p_home: Decimal | str,
        home_decimal_odds: Decimal | str,
        away_decimal_odds: Decimal | str,
    ) -> "PaperMoneylineDecision":
        probability = _decimal(p_home, field_name="p_home")
        home_price = _decimal(home_decimal_odds, field_name="home_decimal_odds")
        away_price = _decimal(away_decimal_odds, field_name="away_decimal_odds")
        _validate_probability(probability, field_name="p_home")
        _validate_price(home_price, field_name="home_decimal_odds")
        _validate_price(away_price, field_name="away_decimal_odds")

        away_probability = Decimal("1") - probability
        ev_home = probability * home_price - Decimal("1")
        ev_away = away_probability * away_price - Decimal("1")
        maximum_ev = max(ev_home, ev_away)
        if ev_home > ev_away:
            candidate_side = "HOME"
            candidate_ev = ev_home
        elif ev_away > ev_home:
            candidate_side = "AWAY"
            candidate_ev = ev_away
        else:
            candidate_side = "NONE"
            candidate_ev = ev_home
            if maximum_ev > Decimal("0"):
                raise ValueError("positive EV tie has no unique candidate side")

        decision = DECISION_BET if maximum_ev > Decimal("0") else DECISION_PASS
        stake = Decimal("1.0") if decision == DECISION_BET else Decimal("0")
        raw_home = Decimal("1") / home_price
        raw_away = Decimal("1") / away_price
        overround = raw_home + raw_away - Decimal("1")
        no_vig_total = raw_home + raw_away
        home_no_vig = raw_home / no_vig_total
        away_no_vig = raw_away / no_vig_total

        provisional = cls(
            decision_id="0" * 64,
            model_role=model_role,
            model_id=model_id,
            model_fingerprint=model_fingerprint,
            p37_fold_id=p37_fold_id,
            p37_window=p37_window,
            p37_prediction_row_id=p37_prediction_row_id,
            provider_namespace=provider_namespace,
            provider_game_id=provider_game_id,
            game_pk=game_pk,
            game_number=game_number,
            official_date=official_date,
            scheduled_start_utc=scheduled_start_utc,
            home_team=home_team,
            away_team=away_team,
            home_team_code=home_team_code,
            away_team_code=away_team_code,
            market_snapshot_id=market_snapshot_id,
            market_observed_at_utc=market_observed_at_utc,
            local_fetched_at_utc=local_fetched_at_utc,
            source_match_id=source_match_id,
            market_source_sha256=market_source_sha256,
            p37_comparisons_sha256=p37_comparisons_sha256,
            model_probability_source=model_probability_source,
            p_home=probability,
            p_away=away_probability,
            home_decimal_odds=home_price,
            away_decimal_odds=away_price,
            ev_home=ev_home,
            ev_away=ev_away,
            candidate_side=candidate_side,
            candidate_ev=candidate_ev,
            decision=decision,
            paper_stake_units=stake,
            paper_stake_convention=PAPER_STAKE_CONVENTION,
            home_raw_implied_probability=raw_home,
            away_raw_implied_probability=raw_away,
            bookmaker_overround=overround,
            home_no_vig_probability=home_no_vig,
            away_no_vig_probability=away_no_vig,
            model_home_vs_no_vig_edge=probability - home_no_vig,
            model_away_vs_no_vig_edge=away_probability - away_no_vig,
        )
        decision_id = _sha256_projection(provisional._projection_without_id())
        return replace(provisional, decision_id=decision_id)

    def __post_init__(self) -> None:
        _require_sha256(self.decision_id, field_name="decision_id")
        _require_sha256(self.model_fingerprint, field_name="model_fingerprint")
        _require_sha256(self.market_source_sha256, field_name="market_source_sha256")
        _require_sha256(self.p37_comparisons_sha256, field_name="p37_comparisons_sha256")
        for field_name in (
            "model_role",
            "model_id",
            "p37_fold_id",
            "p37_window",
            "p37_prediction_row_id",
            "provider_namespace",
            "provider_game_id",
            "official_date",
            "scheduled_start_utc",
            "home_team",
            "away_team",
            "home_team_code",
            "away_team_code",
            "market_snapshot_id",
            "market_observed_at_utc",
            "local_fetched_at_utc",
            "source_match_id",
            "model_probability_source",
            "paper_stake_convention",
        ):
            _require_text(getattr(self, field_name), field_name=field_name)
        _require_positive_int(self.game_pk, field_name="game_pk")
        _require_positive_int(self.game_number, field_name="game_number")
        for field_name in ("p_home", "p_away"):
            value = _decimal(getattr(self, field_name), field_name=field_name)
            _validate_probability(value, field_name=field_name)
        if self.p_home + self.p_away != Decimal("1"):
            raise ValueError("p_home and p_away must sum exactly to one")
        for field_name in ("home_decimal_odds", "away_decimal_odds"):
            _validate_price(
                _decimal(getattr(self, field_name), field_name=field_name),
                field_name=field_name,
            )
        for field_name in (
            "ev_home",
            "ev_away",
            "candidate_ev",
            "paper_stake_units",
            "home_raw_implied_probability",
            "away_raw_implied_probability",
            "bookmaker_overround",
            "home_no_vig_probability",
            "away_no_vig_probability",
            "model_home_vs_no_vig_edge",
            "model_away_vs_no_vig_edge",
        ):
            _decimal(getattr(self, field_name), field_name=field_name)
        if self.candidate_side not in ("HOME", "AWAY", "NONE"):
            raise ValueError("candidate_side must be HOME, AWAY, or NONE")
        if self.decision not in (DECISION_BET, DECISION_PASS):
            raise ValueError("decision must be BET or PASS")
        if self.decision == DECISION_BET:
            if self.candidate_side not in ("HOME", "AWAY"):
                raise ValueError("BET requires a unique candidate side")
            if self.candidate_ev <= Decimal("0") or self.paper_stake_units != Decimal("1.0"):
                raise ValueError("BET requires positive EV and a one-unit stake")
        elif self.paper_stake_units != Decimal("0"):
            raise ValueError("PASS must have zero paper stake")
        expected_home_ev = self.p_home * self.home_decimal_odds - Decimal("1")
        expected_away_ev = self.p_away * self.away_decimal_odds - Decimal("1")
        if self.ev_home != expected_home_ev or self.ev_away != expected_away_ev:
            raise ValueError("expected-value arithmetic does not match inputs")
        if self.decision_id != "0" * 64 and self.decision_id != _sha256_projection(
            self._projection_without_id()
        ):
            raise ValueError("decision_id does not match decision-time fields")

    def _projection_without_id(self) -> dict[str, Any]:
        return {
            "schema_version": P40A_DECISION_SCHEMA_VERSION,
            "policy_id": P40A_POLICY_ID,
            "model_role": self.model_role,
            "model_id": self.model_id,
            "model_fingerprint": self.model_fingerprint,
            "p37_fold_id": self.p37_fold_id,
            "p37_window": self.p37_window,
            "p37_prediction_row_id": self.p37_prediction_row_id,
            "provider_namespace": self.provider_namespace,
            "provider_game_id": self.provider_game_id,
            "game_pk": self.game_pk,
            "game_number": self.game_number,
            "official_date": self.official_date,
            "scheduled_start_utc": self.scheduled_start_utc,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_team_code": self.home_team_code,
            "away_team_code": self.away_team_code,
            "market_snapshot_id": self.market_snapshot_id,
            "market_observed_at_utc": self.market_observed_at_utc,
            "local_fetched_at_utc": self.local_fetched_at_utc,
            "source_match_id": self.source_match_id,
            "market_source_sha256": self.market_source_sha256,
            "p37_comparisons_sha256": self.p37_comparisons_sha256,
            "model_probability_source": self.model_probability_source,
            "p_home": _decimal_text(self.p_home),
            "p_away": _decimal_text(self.p_away),
            "home_decimal_odds": _decimal_text(self.home_decimal_odds),
            "away_decimal_odds": _decimal_text(self.away_decimal_odds),
            "ev_home": _decimal_text(self.ev_home),
            "ev_away": _decimal_text(self.ev_away),
            "candidate_side": self.candidate_side,
            "candidate_ev": _decimal_text(self.candidate_ev),
            "decision": self.decision,
            "paper_stake_units": _decimal_text(self.paper_stake_units),
            "paper_stake_convention": self.paper_stake_convention,
            "home_raw_implied_probability": _decimal_text(
                self.home_raw_implied_probability
            ),
            "away_raw_implied_probability": _decimal_text(
                self.away_raw_implied_probability
            ),
            "bookmaker_overround": _decimal_text(self.bookmaker_overround),
            "home_no_vig_probability": _decimal_text(self.home_no_vig_probability),
            "away_no_vig_probability": _decimal_text(self.away_no_vig_probability),
            "model_home_vs_no_vig_edge": _decimal_text(
                self.model_home_vs_no_vig_edge
            ),
            "model_away_vs_no_vig_edge": _decimal_text(
                self.model_away_vs_no_vig_edge
            ),
        }

    def to_projection(self) -> dict[str, Any]:
        return {"decision_id": self.decision_id, **self._projection_without_id()}


@dataclass(frozen=True, slots=True)
class PaperMoneylineSettlement:
    """One P40A decision with a later authoritative final outcome attached."""

    decision: PaperMoneylineDecision
    final_game_outcome: str
    target_home_win: int
    outcome_authority_row_id: str
    outcome_authority: str
    settlement_status: str
    gross_return_units: Decimal
    net_paper_units: Decimal
    settlement_row_fingerprint: str

    @classmethod
    def create(
        cls,
        decision: PaperMoneylineDecision,
        *,
        final_game_outcome: str,
        target_home_win: int,
        outcome_authority_row_id: str,
        outcome_authority: str,
    ) -> "PaperMoneylineSettlement":
        if final_game_outcome not in ("HOME", "AWAY"):
            raise ValueError(
                "P40A final outcomes must be HOME or AWAY under the existing final-result contract"
            )
        if target_home_win not in (0, 1):
            raise ValueError("target_home_win must be zero or one")
        expected_target = 1 if final_game_outcome == "HOME" else 0
        if target_home_win != expected_target:
            raise ValueError("target_home_win disagrees with final_game_outcome")
        _require_text(outcome_authority_row_id, field_name="outcome_authority_row_id")
        _require_text(outcome_authority, field_name="outcome_authority")

        if decision.decision == DECISION_PASS:
            status = SETTLEMENT_PASS
            gross = Decimal("0")
            net = Decimal("0")
        elif decision.candidate_side == final_game_outcome:
            status = SETTLEMENT_WON
            gross = (
                decision.home_decimal_odds
                if final_game_outcome == "HOME"
                else decision.away_decimal_odds
            )
            net = gross - Decimal("1")
        else:
            status = SETTLEMENT_LOST
            gross = Decimal("0")
            net = Decimal("-1")

        provisional = cls(
            decision=decision,
            final_game_outcome=final_game_outcome,
            target_home_win=target_home_win,
            outcome_authority_row_id=outcome_authority_row_id,
            outcome_authority=outcome_authority,
            settlement_status=status,
            gross_return_units=gross,
            net_paper_units=net,
            settlement_row_fingerprint="0" * 64,
        )
        fingerprint = _sha256_projection(provisional._projection_without_fingerprint())
        return replace(provisional, settlement_row_fingerprint=fingerprint)

    def __post_init__(self) -> None:
        if self.final_game_outcome not in ("HOME", "AWAY"):
            raise ValueError("final_game_outcome must be HOME or AWAY")
        if self.target_home_win not in (0, 1):
            raise ValueError("target_home_win must be zero or one")
        if self.target_home_win != (1 if self.final_game_outcome == "HOME" else 0):
            raise ValueError("target_home_win disagrees with final_game_outcome")
        _require_text(self.outcome_authority_row_id, field_name="outcome_authority_row_id")
        _require_text(self.outcome_authority, field_name="outcome_authority")
        _require_sha256(
            self.settlement_row_fingerprint,
            field_name="settlement_row_fingerprint",
        )
        if self.settlement_status not in (SETTLEMENT_WON, SETTLEMENT_LOST, SETTLEMENT_PASS):
            raise ValueError("unexpected P40A settlement status")
        if self.settlement_status == SETTLEMENT_WON:
            if self.decision.decision != DECISION_BET:
                raise ValueError("only BET decisions can win")
            expected_gross = (
                self.decision.home_decimal_odds
                if self.final_game_outcome == "HOME"
                else self.decision.away_decimal_odds
            )
            if self.gross_return_units != expected_gross:
                raise ValueError("winning gross return does not match offered odds")
            if self.net_paper_units != expected_gross - Decimal("1"):
                raise ValueError("winning net units do not match offered odds")
        elif self.settlement_status == SETTLEMENT_LOST:
            if self.decision.decision != DECISION_BET:
                raise ValueError("only BET decisions can lose")
            if self.gross_return_units != Decimal("0") or self.net_paper_units != Decimal("-1"):
                raise ValueError("losing settlement must return zero gross and lose one unit")
        elif (
            self.decision.decision != DECISION_PASS
            or self.gross_return_units != Decimal("0")
            or self.net_paper_units != Decimal("0")
        ):
            raise ValueError("PASS settlement must return zero gross and zero net units")
        if self.settlement_row_fingerprint != "0" * 64 and self.settlement_row_fingerprint != _sha256_projection(
            self._projection_without_fingerprint()
        ):
            raise ValueError("settlement_row_fingerprint does not match row fields")

    def _projection_without_fingerprint(self) -> dict[str, Any]:
        return {
            "settlement_schema_version": P40A_SETTLEMENT_SCHEMA_VERSION,
            "decision_id": self.decision.decision_id,
            "final_game_outcome": self.final_game_outcome,
            "target_home_win": self.target_home_win,
            "outcome_authority_row_id": self.outcome_authority_row_id,
            "outcome_authority": self.outcome_authority,
            "settlement_status": self.settlement_status,
            "gross_return_units": _decimal_text(self.gross_return_units),
            "net_paper_units": _decimal_text(self.net_paper_units),
        }

    def to_projection(self) -> dict[str, Any]:
        return {
            **self.decision.to_projection(),
            "settlement_schema_version": P40A_SETTLEMENT_SCHEMA_VERSION,
            "final_game_outcome": self.final_game_outcome,
            "target_home_win": self.target_home_win,
            "outcome_authority_row_id": self.outcome_authority_row_id,
            "outcome_authority": self.outcome_authority,
            "settlement_status": self.settlement_status,
            "gross_return_units": _decimal_text(self.gross_return_units),
            "net_paper_units": _decimal_text(self.net_paper_units),
            "settlement_row_fingerprint": self.settlement_row_fingerprint,
        }


def settle_paper_moneyline_decision(
    decision: PaperMoneylineDecision,
    *,
    final_game_outcome: str,
    target_home_win: int,
    outcome_authority_row_id: str,
    outcome_authority: str,
) -> PaperMoneylineSettlement:
    """Attach one final outcome after the decision is already frozen."""

    return PaperMoneylineSettlement.create(
        decision,
        final_game_outcome=final_game_outcome,
        target_home_win=target_home_win,
        outcome_authority_row_id=outcome_authority_row_id,
        outcome_authority=outcome_authority,
    )


def aggregate_paper_settlements(
    settlements: Iterable[PaperMoneylineSettlement],
    *,
    edge_ready_rows: int,
    model_role: str,
) -> dict[str, Any]:
    """Return deterministic descriptive metrics for one model policy."""

    rows = tuple(settlements)
    if not rows:
        raise ValueError("cannot aggregate an empty P40A model cohort")
    if edge_ready_rows < 1:
        raise ValueError("edge_ready_rows must be positive")
    if any(row.decision.model_role != model_role for row in rows):
        raise ValueError("settlement model role does not match aggregation role")

    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.decision.scheduled_start_utc,
                row.decision.game_number,
                row.decision.provider_game_id,
                row.decision.decision_id,
            ),
        )
    )
    bets = tuple(row for row in rows if row.decision.decision == DECISION_BET)
    passes = tuple(row for row in rows if row.decision.decision == DECISION_PASS)
    wins = tuple(row for row in bets if row.settlement_status == SETTLEMENT_WON)
    losses = tuple(row for row in bets if row.settlement_status == SETTLEMENT_LOST)
    pushes = tuple(row for row in bets if row.settlement_status not in {SETTLEMENT_WON, SETTLEMENT_LOST})
    risked = sum((row.decision.paper_stake_units for row in rows), Decimal("0"))
    gross = sum((row.gross_return_units for row in rows), Decimal("0"))
    net = sum((row.net_paper_units for row in rows), Decimal("0"))

    equity = Decimal("0")
    peak = Decimal("0")
    maximum_drawdown = Decimal("0")
    for row in ordered:
        equity += row.net_paper_units
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)

    average_ev = (
        _ratio(
            sum((row.decision.candidate_ev for row in bets), Decimal("0")),
            Decimal(len(bets)),
        )
        if bets
        else None
    )
    settled_bets = len(wins) + len(losses)
    return {
        "model_role": model_role,
        "edge_ready_rows": edge_ready_rows,
        "sample_size": edge_ready_rows,
        "row_count": len(rows),
        "bet_count": len(bets),
        "pass_count": len(passes),
        "bet_coverage": _decimal_text(_ratio(Decimal(len(bets)), Decimal(edge_ready_rows))),
        "win_count": len(wins),
        "loss_count": len(losses),
        "push_count": len(pushes),
        "observed_hit_rate": (
            _decimal_text(_ratio(Decimal(len(wins)), Decimal(settled_bets)))
            if settled_bets
            else None
        ),
        "total_paper_units_risked": _decimal_text(risked),
        "gross_return_units": _decimal_text(gross),
        "net_paper_units": _decimal_text(net),
        "descriptive_paper_roi": (
            _decimal_text(_ratio(net, risked)) if risked > Decimal("0") else None
        ),
        "maximum_paper_drawdown": _decimal_text(maximum_drawdown),
        "average_predicted_ev_of_bet_rows": _decimal_text(average_ev),
        "drawdown_order": "SCHEDULED_START_ASCENDING_GAME_NUMBER_PROVIDER_GAME_ID",
        "gross_return_definition": (
            "sum of returned decimal odds for winning BET rows; PASS and losing rows return zero"
        ),
        "roi_label": "DESCRIPTIVE_PAPER_ONLY",
    }


__all__ = (
    "DECISION_BET",
    "DECISION_PASS",
    "P40A_DECISION_SCHEMA_VERSION",
    "P40A_POLICY_ID",
    "P40A_SETTLEMENT_SCHEMA_VERSION",
    "PAPER_STAKE_CONVENTION",
    "PaperMoneylineDecision",
    "PaperMoneylineSettlement",
    "SETTLEMENT_LOST",
    "SETTLEMENT_PASS",
    "SETTLEMENT_WON",
    "aggregate_paper_settlements",
    "settle_paper_moneyline_decision",
)
