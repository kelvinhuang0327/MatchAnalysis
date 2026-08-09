"""Replay a bounded P20A fold through the existing P19A inference path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Tuple, Union

from ...baseball.domain.canonical_utc import format_canonical_utc
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.moneyline_walk_forward_fold import (
    MoneylineWalkForwardFold,
    ReconstructedWalkForwardModel,
)
from ...core.identity import MatchIdentity
from .generate_moneyline_predictions import (
    MoneylineInferenceResult,
    generate_moneyline_predictions,
)
from .moneyline_inference_artifacts import render_predictions_jsonl
from .moneyline_walk_forward_artifacts import build_moneyline_model_artifact
from .reconstruct_moneyline_walk_forward_model import (
    load_moneyline_walk_forward_fold,
    reconstruct_moneyline_walk_forward_model,
)


P20A_REPLAY_SCHEMA_VERSION = "p20a.moneyline_walk_forward_replay.v1"


@dataclass(frozen=True)
class MoneylineParityRow:
    game_id: str
    expected_home_probability: Decimal
    reproduced_home_probability: Decimal
    absolute_difference: Decimal
    passed: bool

    def to_projection(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "expected_home_probability": str(self.expected_home_probability),
            "reproduced_home_probability": str(
                self.reproduced_home_probability
            ),
            "absolute_difference": str(self.absolute_difference),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class MoneylineWalkForwardReplayResult:
    fold: MoneylineWalkForwardFold
    model: ReconstructedWalkForwardModel
    model_artifact: MoneylineModelArtifact
    inference: MoneylineInferenceResult
    parity_rows: Tuple[MoneylineParityRow, ...]
    tolerance: Decimal = Decimal("0.000001")

    @property
    def max_absolute_difference(self) -> Decimal:
        return max(
            (row.absolute_difference for row in self.parity_rows),
            default=Decimal("0"),
        )

    @property
    def parity_passed(self) -> bool:
        return bool(self.parity_rows) and all(
            row.passed for row in self.parity_rows
        )

    def to_projection(self) -> dict[str, Any]:
        return {
            "schema_version": P20A_REPLAY_SCHEMA_VERSION,
            "fold_id": self.fold.fold_id,
            "fold_fingerprint": self.fold.fingerprint(),
            "reconstructed_model": {
                **self.model.to_projection(),
                "fingerprint": self.model.fingerprint(),
            },
            "model_artifact": self.model_artifact.to_projection(),
            "model_artifact_fingerprint": self.model_artifact.fingerprint(),
            "candidate_set_fingerprint": self.inference.candidate_set_fingerprint,
            "candidate_count": len(self.inference.candidates),
            "parity_rows": [row.to_projection() for row in self.parity_rows],
            "max_absolute_difference": str(self.max_absolute_difference),
            "tolerance": str(self.tolerance),
            "parity_passed": self.parity_passed,
            "claims": {
                "historical": True,
                "paper_only": True,
                "diagnostic": True,
                "bounded": True,
                "production": False,
                "profitability": False,
                "betting_performance": False,
                "model_promotion": False,
            },
        }


def _snapshot_for_row(
    fold: MoneylineWalkForwardFold,
    row,
) -> MoneylineFeatureSnapshot:
    as_of = datetime.fromisoformat(row.date + "T00:00:00+00:00")
    scheduled_start = datetime.fromisoformat(
        row.scheduled_start_utc.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    feature_values = {
        "recent_win_rate_delta": row.feature_values[0],
        "starter_era_delta": row.feature_values[1],
    }
    provenance = tuple(
        MoneylineFeatureProvenance(
            field_name=field_name,
            source_id=(
                f"{fold.legacy_source_commit}:{fold.legacy_source_paths[4]}:"
                f"{row.game_id}:{field_name}"
            ),
            source_kind="legacy_p13_committed_feature_matrix",
            observed_as_of_utc=as_of,
            source_fingerprint=sha256(
                (
                    f"{fold.legacy_source_commit}:{fold.legacy_source_paths[4]}:"
                    f"{row.game_id}:{field_name}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        for field_name in ("recent_win_rate_delta", "starter_era_delta")
    )
    return MoneylineFeatureSnapshot.from_record(
        feature_values,
        identity=MatchIdentity(
            sport="baseball",
            league="MLB",
            season=as_of.year,
            canonical_game_id=row.game_id,
            home_participant=row.home_team,
            away_participant=row.away_team,
            game_discriminator=None,
        ),
        provider_namespace="MLB_STATS_API",
        provider_game_id=row.game_id,
        game_number=1,
        source_schedule_observation_id=row.source_schedule_observation_id,
        as_of_utc=as_of,
        scheduled_start_utc=scheduled_start,
        feature_provenance=provenance,
    )


def replay_historical_moneyline_predictions(
    fold: MoneylineWalkForwardFold,
    model: ReconstructedWalkForwardModel,
    model_artifact: MoneylineModelArtifact | None = None,
) -> MoneylineWalkForwardReplayResult:
    """Run selected rows through P19A and compare every bounded HOME output."""

    if model_artifact is None:
        model_artifact = build_moneyline_model_artifact(fold, model)
    snapshots = tuple(_snapshot_for_row(fold, row) for row in fold.prediction_rows)
    first_as_of = datetime.fromisoformat(
        fold.prediction_rows[0].date + "T00:00:00+00:00"
    )
    generated = first_as_of + timedelta(minutes=1)
    received = generated + timedelta(seconds=1)
    ingested = received + timedelta(seconds=1)
    inference = generate_moneyline_predictions(
        snapshots,
        model_artifact,
        prediction_generated_at_utc=generated,
        response_received_at_utc=received,
        ingested_at_utc=ingested,
    )
    home_candidates = inference.candidates[::2]
    if len(home_candidates) != len(fold.prediction_rows):
        raise ValueError("P19A did not emit one HOME candidate per replay row")
    parity_rows = tuple(
        MoneylineParityRow(
            game_id=row.game_id,
            expected_home_probability=Decimal(expected),
            reproduced_home_probability=candidate.model_probability,
            absolute_difference=abs(
                candidate.model_probability - Decimal(expected)
            ),
            passed=abs(candidate.model_probability - Decimal(expected))
            <= Decimal("0.000001"),
        )
        for row, expected, candidate in zip(
            fold.prediction_rows,
            fold.expected_home_probabilities,
            home_candidates,
        )
    )
    return MoneylineWalkForwardReplayResult(
        fold=fold,
        model=model,
        model_artifact=model_artifact,
        inference=inference,
        parity_rows=parity_rows,
    )


def replay_historical_moneyline_predictions_from_path(
    fixture_path: Union[str, Path],
) -> MoneylineWalkForwardReplayResult:
    fold = load_moneyline_walk_forward_fold(fixture_path)
    model = reconstruct_moneyline_walk_forward_model(fold)
    return replay_historical_moneyline_predictions(fold, model)


def write_moneyline_walk_forward_replay_artifacts(
    output_dir: Union[str, Path],
    result: MoneylineWalkForwardReplayResult,
) -> None:
    """Write deterministic bounded replay artifacts without runtime timestamps."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    projection = result.to_projection()
    (directory / "reconstruction.json").write_text(
        json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "predictions.jsonl").write_text(
        render_predictions_jsonl(result.inference),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": P20A_REPLAY_SCHEMA_VERSION,
                "fold_id": result.fold.fold_id,
                "fold_fingerprint": result.fold.fingerprint(),
                "model_artifact_fingerprint": result.model_artifact.fingerprint(),
                "candidate_set_fingerprint": result.inference.candidate_set_fingerprint,
                "training_row_count": result.fold.training_row_count,
                "replay_row_count": result.fold.prediction_row_count,
                "candidate_count": len(result.inference.candidates),
                "max_absolute_difference": str(result.max_absolute_difference),
                "parity_passed": result.parity_passed,
                "claims": projection["claims"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# P20A P13 Walk-Forward Reconstruction",
        "",
        "This is a bounded historical, paper-only, diagnostic replay.",
        "",
        f"- Fold: `{result.fold.fold_id}`",
        f"- Fold fingerprint: `{result.fold.fingerprint()}`",
        f"- Training rows: `{result.fold.training_row_count}`",
        f"- Replay rows: `{result.fold.prediction_row_count}`",
        f"- Model artifact fingerprint: `{result.model_artifact.fingerprint()}`",
        f"- Maximum absolute parity difference: `{result.max_absolute_difference}`",
        f"- Every bounded row passed: `{result.parity_passed}`",
        "",
        "## Boundaries",
        "",
        "- Market: `moneyline` only.",
        "- Historical and paper-only; no production, profitability, or betting-performance claim.",
        "- No provider, network, database, settlement, odds, or P16–P18 operation.",
        "",
        "## Legacy provenance",
        "",
        f"- Repository: `{result.fold.legacy_source_commit}`",
        f"- Tree: `{result.fold.legacy_source_tree}`",
    ]
    lines.extend(f"- Path: `{path}`" for path in result.fold.legacy_source_paths)
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
