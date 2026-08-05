"""Pure prediction evaluation contracts and deterministic scorecard computation.

Evaluates ATTACHED prospective predictions against exact final outcomes.
Does not construct PredictionSourceObservation or FinalResultObservation,
run admission or result attachment, calculate odds/ROI/EV, retrain models,
or touch external networks or databases.
"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import re


SCHEMA_VERSION = "p16b.prediction_evaluation_scorecard.v1"
EVALUATION_ROW_SCHEMA_VERSION = "p16b.prediction_evaluation_row.v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be explicit and trimmed")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_finite_decimal(value: Decimal, field_name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")


def compute_evaluation_row_fingerprint(
    *,
    prediction_observation_id: str,
    source_attachment_row_fingerprint: str,
    model_id: str,
    market_id: str,
    selection: str,
    provider_namespace: str,
    provider_game_id: str,
    game_number: int,
    model_probability: Decimal,
    actual_winner: str,
    is_correct: bool,
    correctness_target: int,
    brier_component: Decimal,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for an evaluation row."""
    payload = {
        "actual_winner": actual_winner,
        "brier_component": str(brier_component),
        "correctness_target": correctness_target,
        "game_number": game_number,
        "is_correct": is_correct,
        "market_id": market_id,
        "model_id": model_id,
        "model_probability": str(model_probability),
        "prediction_observation_id": prediction_observation_id,
        "provider_game_id": provider_game_id,
        "provider_namespace": provider_namespace,
        "selection": selection,
        "source_attachment_row_fingerprint": source_attachment_row_fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_evaluation_set_fingerprint(
    rows: tuple["PredictionEvaluationRow", ...],
) -> str:
    """Compute deterministic SHA-256 fingerprint over ordered evaluation rows."""
    parts = [
        f"{row.prediction_observation_id}:{row.evaluation_row_fingerprint}\n"
        for row in rows
    ]
    combined = "".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionEvaluationRow:
    """Immutable evaluation row for one ATTACHED prediction."""

    prediction_observation_id: str
    source_attachment_row_fingerprint: str
    model_id: str
    market_id: str
    selection: str
    provider_namespace: str
    provider_game_id: str
    game_number: int
    model_probability: Decimal
    actual_winner: str
    is_correct: bool
    correctness_target: int
    brier_component: Decimal
    evaluation_row_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "prediction_observation_id",
            "source_attachment_row_fingerprint",
            "model_id",
            "market_id",
            "selection",
            "provider_namespace",
            "provider_game_id",
            "actual_winner",
            "evaluation_row_fingerprint",
        ):
            _require_explicit(getattr(self, field_name), field_name)

        _require_sha256(
            self.prediction_observation_id,
            "prediction_observation_id",
        )
        _require_sha256(
            self.source_attachment_row_fingerprint,
            "source_attachment_row_fingerprint",
        )
        _require_sha256(
            self.evaluation_row_fingerprint,
            "evaluation_row_fingerprint",
        )

        _require_positive_integer(self.game_number, "game_number")

        if self.market_id != "moneyline":
            raise ValueError("market_id must be 'moneyline'")

        if self.selection not in ("HOME", "AWAY"):
            raise ValueError("selection must be 'HOME' or 'AWAY'")

        if self.actual_winner not in ("HOME", "AWAY"):
            raise ValueError("actual_winner must be 'HOME' or 'AWAY'")

        if not isinstance(self.is_correct, bool):
            raise TypeError("is_correct must be a boolean")

        expected_target = 1 if self.is_correct else 0
        if isinstance(self.correctness_target, bool) or not isinstance(
            self.correctness_target, int
        ):
            raise TypeError("correctness_target must be an integer")
        if self.correctness_target != expected_target:
            raise ValueError(
                f"correctness_target must be {expected_target} for is_correct={self.is_correct}"
            )

        _require_finite_decimal(self.model_probability, "model_probability")
        if not (Decimal("0") <= self.model_probability <= Decimal("1")):
            raise ValueError("model_probability must be within [0, 1]")

        _require_finite_decimal(self.brier_component, "brier_component")
        expected_brier = (
            self.model_probability - Decimal(self.correctness_target)
        ) ** 2
        if self.brier_component != expected_brier:
            raise ValueError(
                f"brier_component must equal (model_probability - correctness_target)^2: "
                f"expected {expected_brier}, got {self.brier_component}"
            )

        expected_fp = compute_evaluation_row_fingerprint(
            prediction_observation_id=self.prediction_observation_id,
            source_attachment_row_fingerprint=self.source_attachment_row_fingerprint,
            model_id=self.model_id,
            market_id=self.market_id,
            selection=self.selection,
            provider_namespace=self.provider_namespace,
            provider_game_id=self.provider_game_id,
            game_number=self.game_number,
            model_probability=self.model_probability,
            actual_winner=self.actual_winner,
            is_correct=self.is_correct,
            correctness_target=self.correctness_target,
            brier_component=self.brier_component,
        )
        if self.evaluation_row_fingerprint != expected_fp:
            raise ValueError(
                f"evaluation_row_fingerprint mismatch: computed {expected_fp}, "
                f"got {self.evaluation_row_fingerprint}"
            )


@dataclass(frozen=True, slots=True)
class BreakdownMetrics:
    """Aggregate evaluation metrics for a subgroup."""

    row_count: int
    correct_count: int
    incorrect_count: int
    accuracy: Decimal
    mean_selected_side_probability: Decimal
    brier_score: Decimal


@dataclass(frozen=True, slots=True)
class PredictionEvaluationScorecard:
    """Immutable deterministic evaluation scorecard."""

    evaluation_row_count: int
    correct_count: int
    incorrect_count: int
    accuracy: Decimal
    mean_selected_side_probability: Decimal
    brier_score: Decimal
    breakdown_by_model_id: dict[str, BreakdownMetrics]
    breakdown_by_market_id: dict[str, BreakdownMetrics]
    breakdown_by_selection: dict[str, BreakdownMetrics]
    source_attachment_set_fingerprint: str
    evaluation_set_fingerprint: str


def _compute_group_breakdown(
    group_rows: list[PredictionEvaluationRow],
) -> BreakdownMetrics:
    """Compute aggregate metrics for a single subgroup of evaluation rows."""
    count = len(group_rows)
    if count == 0:
        return BreakdownMetrics(
            row_count=0,
            correct_count=0,
            incorrect_count=0,
            accuracy=Decimal("0"),
            mean_selected_side_probability=Decimal("0"),
            brier_score=Decimal("0"),
        )
    correct = sum(1 for r in group_rows if r.is_correct)
    incorrect = count - correct
    accuracy = Decimal(correct) / Decimal(count)
    mean_prob = sum((r.model_probability for r in group_rows), Decimal("0")) / Decimal(count)
    brier = sum((r.brier_component for r in group_rows), Decimal("0")) / Decimal(count)
    return BreakdownMetrics(
        row_count=count,
        correct_count=correct,
        incorrect_count=incorrect,
        accuracy=accuracy,
        mean_selected_side_probability=mean_prob,
        brier_score=brier,
    )


def build_scorecard(
    rows: tuple[PredictionEvaluationRow, ...],
    source_attachment_set_fingerprint: str,
) -> PredictionEvaluationScorecard:
    """Build a deterministic PredictionEvaluationScorecard from evaluation rows.

    Requires rows to be sorted by prediction_observation_id with no duplicates.
    """
    _require_sha256(
        source_attachment_set_fingerprint,
        "source_attachment_set_fingerprint",
    )

    # Check sorting and duplicates
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.prediction_observation_id in seen_ids:
            raise ValueError(
                f"duplicate prediction_observation_id: {row.prediction_observation_id}"
            )
        seen_ids.add(row.prediction_observation_id)
        if index > 0 and rows[index - 1].prediction_observation_id > row.prediction_observation_id:
            raise ValueError("rows must be sorted by prediction_observation_id")

    eval_set_fp = compute_evaluation_set_fingerprint(rows)

    total_count = len(rows)
    if total_count == 0:
        return PredictionEvaluationScorecard(
            evaluation_row_count=0,
            correct_count=0,
            incorrect_count=0,
            accuracy=Decimal("0"),
            mean_selected_side_probability=Decimal("0"),
            brier_score=Decimal("0"),
            breakdown_by_model_id={},
            breakdown_by_market_id={},
            breakdown_by_selection={},
            source_attachment_set_fingerprint=source_attachment_set_fingerprint,
            evaluation_set_fingerprint=eval_set_fp,
        )

    correct_count = sum(1 for r in rows if r.is_correct)
    incorrect_count = total_count - correct_count
    accuracy = Decimal(correct_count) / Decimal(total_count)
    mean_prob = sum((r.model_probability for r in rows), Decimal("0")) / Decimal(total_count)
    brier_score = sum((r.brier_component for r in rows), Decimal("0")) / Decimal(total_count)

    # Compute breakdowns with deterministically sorted keys
    model_groups: dict[str, list[PredictionEvaluationRow]] = {}
    market_groups: dict[str, list[PredictionEvaluationRow]] = {}
    selection_groups: dict[str, list[PredictionEvaluationRow]] = {}

    for r in rows:
        model_groups.setdefault(r.model_id, []).append(r)
        market_groups.setdefault(r.market_id, []).append(r)
        selection_groups.setdefault(r.selection, []).append(r)

    breakdown_by_model_id = {
        key: _compute_group_breakdown(model_groups[key])
        for key in sorted(model_groups)
    }
    breakdown_by_market_id = {
        key: _compute_group_breakdown(market_groups[key])
        for key in sorted(market_groups)
    }
    breakdown_by_selection = {
        key: _compute_group_breakdown(selection_groups[key])
        for key in sorted(selection_groups)
    }

    return PredictionEvaluationScorecard(
        evaluation_row_count=total_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=accuracy,
        mean_selected_side_probability=mean_prob,
        brier_score=brier_score,
        breakdown_by_model_id=breakdown_by_model_id,
        breakdown_by_market_id=breakdown_by_market_id,
        breakdown_by_selection=breakdown_by_selection,
        source_attachment_set_fingerprint=source_attachment_set_fingerprint,
        evaluation_set_fingerprint=eval_set_fp,
    )
