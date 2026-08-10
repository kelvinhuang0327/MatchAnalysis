"""Pure paired out-of-sample Moneyline comparison contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


COMPARISON_SCHEMA_VERSION = "p23a.moneyline_strictly_future_oos_comparison.v1"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def cohort_fingerprint(feature_rows: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the authoritative cohort independent of input row order."""

    ordered = sorted(
        feature_rows,
        key=lambda row: (
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        ),
    )
    rows = (
        f"{row['provider_game_id']}\t{row['game_pk']}\t{row['game_number']}\t"
        f"{row['scheduled_start_utc']}\t{row['feature_fingerprint']}\n"
        for row in ordered
    )
    return sha256("".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PairedMoneylineComparison:
    """One frozen-feature, paired-result comparison row."""

    comparison_row_id: str
    fold_id: str
    provider_game_id: str
    game_pk: int
    game_number: int
    scheduled_start_utc: str
    feature_fingerprint: str
    challenger_model_id: str
    challenger_model_fingerprint: str
    challenger_home_probability: Decimal
    incumbent_model_id: str
    incumbent_model_fingerprint: str
    incumbent_home_probability: Decimal
    target_home_win: int
    actual_winner: str
    challenger_correct: bool
    incumbent_correct: bool
    challenger_brier_contribution: Decimal
    incumbent_brier_contribution: Decimal
    paired_brier_delta: Decimal

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "comparison_row_id": self.comparison_row_id,
            "fold_id": self.fold_id,
            "provider_namespace": "MLB_STATS_API",
            "provider_game_id": self.provider_game_id,
            "game_pk": self.game_pk,
            "game_number": self.game_number,
            "scheduled_start_utc": self.scheduled_start_utc,
            "feature_fingerprint": self.feature_fingerprint,
            "challenger_model_id": self.challenger_model_id,
            "challenger_model_fingerprint": self.challenger_model_fingerprint,
            "challenger_home_probability": str(self.challenger_home_probability),
            "incumbent_model_id": self.incumbent_model_id,
            "incumbent_model_fingerprint": self.incumbent_model_fingerprint,
            "incumbent_home_probability": str(self.incumbent_home_probability),
            "target_home_win": self.target_home_win,
            "actual_winner": self.actual_winner,
            "challenger_correct": self.challenger_correct,
            "incumbent_correct": self.incumbent_correct,
            "challenger_brier_contribution": str(
                self.challenger_brier_contribution
            ),
            "incumbent_brier_contribution": str(self.incumbent_brier_contribution),
            "paired_brier_delta": str(self.paired_brier_delta),
        }


def build_comparison_row(
    *,
    fold_id: str,
    feature_row: Mapping[str, Any],
    result_row: Mapping[str, Any],
    challenger_model_id: str,
    challenger_model_fingerprint: str,
    challenger_home_probability: Decimal,
    incumbent_model_id: str,
    incumbent_model_fingerprint: str,
    incumbent_home_probability: Decimal,
) -> PairedMoneylineComparison:
    """Join one frozen prediction pair to one authoritative final result."""

    if feature_row["provider_game_id"] != result_row["provider_game_id"]:
        raise ValueError("feature/result game identities must match")
    if feature_row["scheduled_start_utc"] != result_row["scheduled_start_utc"]:
        raise ValueError("feature/result scheduled starts must match")
    home_score = result_row["home_score"]
    away_score = result_row["away_score"]
    if not isinstance(home_score, int) or not isinstance(away_score, int):
        raise ValueError("final scores must be integers")
    if home_score == away_score:
        raise ValueError("tied final scores cannot produce a home-win target")
    target = int(home_score > away_score)
    actual_winner = "HOME" if target else "AWAY"
    challenger_correct = (challenger_home_probability >= Decimal("0.5")) == bool(target)
    incumbent_correct = (incumbent_home_probability >= Decimal("0.5")) == bool(target)
    challenger_brier = (challenger_home_probability - Decimal(target)) ** 2
    incumbent_brier = (incumbent_home_probability - Decimal(target)) ** 2
    identity = {
        "fold_id": fold_id,
        "provider_game_id": feature_row["provider_game_id"],
        "game_pk": int(feature_row["game_pk"]),
        "game_number": int(feature_row["game_number"]),
        "scheduled_start_utc": feature_row["scheduled_start_utc"],
        "feature_fingerprint": feature_row["feature_fingerprint"],
        "challenger_model_id": challenger_model_id,
        "challenger_model_fingerprint": challenger_model_fingerprint,
        "challenger_home_probability": str(challenger_home_probability),
        "incumbent_model_id": incumbent_model_id,
        "incumbent_model_fingerprint": incumbent_model_fingerprint,
        "incumbent_home_probability": str(incumbent_home_probability),
    }
    return PairedMoneylineComparison(
        comparison_row_id=sha256_bytes(canonical_json_bytes(identity)),
        fold_id=fold_id,
        provider_game_id=str(feature_row["provider_game_id"]),
        game_pk=int(feature_row["game_pk"]),
        game_number=int(feature_row["game_number"]),
        scheduled_start_utc=str(feature_row["scheduled_start_utc"]),
        feature_fingerprint=str(feature_row["feature_fingerprint"]),
        challenger_model_id=challenger_model_id,
        challenger_model_fingerprint=challenger_model_fingerprint,
        challenger_home_probability=challenger_home_probability,
        incumbent_model_id=incumbent_model_id,
        incumbent_model_fingerprint=incumbent_model_fingerprint,
        incumbent_home_probability=incumbent_home_probability,
        target_home_win=target,
        actual_winner=actual_winner,
        challenger_correct=challenger_correct,
        incumbent_correct=incumbent_correct,
        challenger_brier_contribution=challenger_brier,
        incumbent_brier_contribution=incumbent_brier,
        paired_brier_delta=challenger_brier - incumbent_brier,
    )


def comparison_set_fingerprint(rows: Sequence[PairedMoneylineComparison]) -> str:
    ordered = sorted(rows, key=lambda row: row.comparison_row_id)
    return sha256(
        b"".join(canonical_json_bytes(row.to_projection()) for row in ordered)
    ).hexdigest()


def aggregate_metrics(rows: Sequence[PairedMoneylineComparison]) -> dict[str, Any]:
    if not rows:
        raise ValueError("comparison rows must not be empty")
    count = Decimal(len(rows))
    challenger_brier = sum(
        (row.challenger_brier_contribution for row in rows), Decimal("0")
    ) / count
    incumbent_brier = sum(
        (row.incumbent_brier_contribution for row in rows), Decimal("0")
    ) / count
    challenger_accuracy = Decimal(
        sum(row.challenger_correct for row in rows)
    ) / count
    incumbent_accuracy = Decimal(sum(row.incumbent_correct for row in rows)) / count
    return {
        "game_count": len(rows),
        "challenger_mean_brier": str(challenger_brier),
        "incumbent_mean_brier": str(incumbent_brier),
        "brier_delta": str(challenger_brier - incumbent_brier),
        "challenger_accuracy": str(challenger_accuracy),
        "incumbent_accuracy": str(incumbent_accuracy),
        "accuracy_delta": str(challenger_accuracy - incumbent_accuracy),
        "challenger_brier_better_count": sum(
            row.challenger_brier_contribution < row.incumbent_brier_contribution
            for row in rows
        ),
        "incumbent_brier_better_count": sum(
            row.incumbent_brier_contribution < row.challenger_brier_contribution
            for row in rows
        ),
        "equal_brier_count": sum(
            row.challenger_brier_contribution == row.incumbent_brier_contribution
            for row in rows
        ),
    }


__all__ = (
    "COMPARISON_SCHEMA_VERSION",
    "PairedMoneylineComparison",
    "aggregate_metrics",
    "build_comparison_row",
    "canonical_json_bytes",
    "cohort_fingerprint",
    "comparison_set_fingerprint",
    "sha256_bytes",
)
