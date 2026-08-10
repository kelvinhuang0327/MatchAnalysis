"""Generate the P24C promoted-default paper Moneyline shadow batch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...baseball.domain.canonical_utc import format_canonical_utc
from ...baseball.domain.future_evaluation_fold import FutureFeatureRow
from ...baseball.domain.moneyline_feature_snapshot import (
    MoneylineFeatureProvenance,
    MoneylineFeatureSnapshot,
)
from ...baseball.domain.moneyline_model_artifact import MoneylineModelArtifact
from ...baseball.domain.prediction_admission import ProspectivePredictionCandidate
from ...core.identity import MatchIdentity
from .acquire_future_moneyline_history import (
    acquire_official_future_fold,
    load_normalized_rows,
)
from .generate_moneyline_predictions import generate_moneyline_predictions
from .materialize_future_moneyline_fold import (
    FutureFeatureEligibility,
    classify_future_feature_eligibility,
    materialize_future_moneyline_fold,
)
from .paper_moneyline_batch_artifacts import (
    P22B_ARTIFACT_FINGERPRINT,
    P22B_ARTIFACT_RELATIVE_PATH,
    P22B_MODEL_ID,
    P24C_BATCH_SCHEMA_VERSION,
    P24C_SOURCE_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    default_paper_moneyline_model_artifact_path,
    load_default_paper_moneyline_model_artifact,
    load_model_artifact_with_fingerprint,
    render_jsonl,
    sha256_bytes,
)


P24C_FOLD_ID = "wf_007"
P24C_WINDOW_DAYS = 7
P24C_FEATURE_SEMANTICS_VERSION = "p13.moneyline_features.v1"
P24C_RAW_MANIFEST_ROOT = Path(
    "data/fixtures/p24c_promoted_moneyline_shadow_batch/raw"
)
P24C_STOP_P24B_BASELINE = "STOP_MATCHANALYSIS_P24C_P24B_BASELINE_UNRESOLVED"
P24C_STOP_WINDOW_UNAVAILABLE = "STOP_MATCHANALYSIS_P24C_SEVEN_DAY_WINDOW_UNAVAILABLE"
P24C_STOP_DEFAULT_DRIFT = "STOP_MATCHANALYSIS_P24C_DEFAULT_MODEL_DRIFT"
P24C_STOP_OVERRIDE_DIVERGENCE = (
    "STOP_MATCHANALYSIS_P24C_DEFAULT_OVERRIDE_DIVERGENCE"
)
P24C_STOP_FEATURE_INPUT = "STOP_MATCHANALYSIS_P24C_FEATURE_INPUT_UNRESOLVED"
P24C_STOP_MANDATORY_VERIFICATION = (
    "STOP_MATCHANALYSIS_P24C_MANDATORY_VERIFICATION_FAILED"
)
P24C_PROBABILITY_TOLERANCE = "0.000001"


@dataclass(frozen=True, slots=True)
class P24CWindow:
    fold_id: str
    start_date: str
    end_date: str


@dataclass(frozen=True, slots=True)
class PaperMoneylineBatchResult:
    predictions: tuple[dict[str, Any], ...]
    feature_unavailable: tuple[dict[str, Any], ...]
    source_manifest: dict[str, Any]
    summary: dict[str, Any]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(UTC)
    if normalized.microsecond:
        return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


def _target_rows(
    schedule_rows: Sequence[Mapping[str, Any]],
    window: P24CWindow,
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        sorted(
            (
                row
                for row in schedule_rows
                if window.start_date <= str(row["official_date"]) <= window.end_date
            ),
            key=lambda row: (
                str(row["scheduled_start_utc"]),
                int(row["game_number"]),
                int(row["game_pk"]),
            ),
        )
    )
    if not rows or any(not bool(row.get("final")) for row in rows):
        raise RuntimeError(P24C_STOP_WINDOW_UNAVAILABLE)
    if len({str(row["provider_game_id"]) for row in rows}) != len(rows):
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    return rows


def _canonicalize_game_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Make source game identities unique without discarding duplicate source rows."""

    canonical: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        source_id = str(row.get("source_provider_game_id", row["provider_game_id"]))
        scheduled_start = str(row["scheduled_start_utc"])
        row["source_provider_game_id"] = source_id
        row["provider_game_id"] = f"{source_id}@{scheduled_start}"
        canonical.append(row)
    return tuple(canonical)


def resolve_p24c_window(repository_root: str | Path) -> P24CWindow:
    """Resolve the exact seven dates following committed wf_006."""

    root = Path(repository_root)
    summary_path = root / "data/fixtures/p23b_future_folds/wf_006/summary.json"
    history_path = root / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
    if not summary_path.is_file() or not history_path.is_file():
        raise RuntimeError(P24C_STOP_P24B_BASELINE)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("fold_id") != "wf_006":
        raise RuntimeError(P24C_STOP_P24B_BASELINE)
    try:
        last_date = date.fromisoformat(str(summary["validation_end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(P24C_STOP_P24B_BASELINE) from exc
    start = last_date + timedelta(days=1)
    end = start + timedelta(days=P24C_WINDOW_DAYS - 1)
    window = P24CWindow(
        fold_id=P24C_FOLD_ID,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    _target_rows(_canonicalize_game_rows(load_normalized_rows(history_path)), window)
    expected_dates = {
        (start + timedelta(days=index)).isoformat()
        for index in range(P24C_WINDOW_DAYS)
    }
    actual_dates = {
        str(row["official_date"])
        for row in load_normalized_rows(history_path)
        if window.start_date <= str(row["official_date"]) <= window.end_date
    }
    if actual_dates != expected_dates:
        raise RuntimeError(P24C_STOP_WINDOW_UNAVAILABLE)
    return window


def _manifest_path_for_p24c(path: str, fold_id: str) -> str:
    old_prefix = f"data/fixtures/p23b_future_folds/{fold_id}/raw"
    if old_prefix in path:
        return path.replace(old_prefix, P24C_RAW_MANIFEST_ROOT.as_posix())
    return path


def build_source_manifest(
    *,
    source_records: Sequence[Mapping[str, Any]],
    normalized_hashes: Mapping[str, str],
    window: P24CWindow,
    acquired_at_utc: str,
) -> dict[str, Any]:
    records = []
    for source_record in source_records:
        record = dict(source_record)
        record["path"] = _manifest_path_for_p24c(str(record["path"]), window.fold_id)
        records.append(record)
    acquired_at = _format_utc(_parse_utc(acquired_at_utc))
    return {
        "schema_version": P24C_SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_authority": "MLB_STATS_API",
        "source_domains": ["mlb.com"],
        "historical_date_scope": {
            "start": window.start_date,
            "end": window.end_date,
        },
        "acquisition_timestamp_utc": acquired_at,
        "records": sorted(records, key=lambda row: (row["path"], row["url"])),
        "normalized_hashes": dict(sorted(normalized_hashes.items())),
    }


def _verify_source_manifest(
    *,
    repository_root: Path,
    raw_root: Path,
    normalized_root: Path,
    source_manifest: Mapping[str, Any],
    window: P24CWindow,
) -> None:
    if source_manifest.get("source_authority") != "MLB_STATS_API":
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    scope = source_manifest.get("historical_date_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("start"), scope.get("end")
    ) != (window.start_date, window.end_date):
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    normalized_hashes = source_manifest.get("normalized_hashes")
    if not isinstance(normalized_hashes, Mapping) or set(normalized_hashes) != {
        "schedule.jsonl",
        "target_boxscores.jsonl",
        "pitcher_game_logs.jsonl",
    }:
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    for name, expected in normalized_hashes.items():
        path = normalized_root / str(name)
        if not path.is_file() or sha256_bytes(path.read_bytes()) != str(expected):
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)

    records = source_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)
        record_path = Path(str(record.get("path", "")))
        if not record_path.parts:
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)
        try:
            relative = record_path.relative_to(P24C_RAW_MANIFEST_ROOT)
        except ValueError:
            actual_path = repository_root / record_path
        else:
            actual_path = raw_root / relative
        if (
            not actual_path.is_file()
            or sha256_bytes(actual_path.read_bytes()) != str(record.get("sha256"))
        ):
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)
        if "statsapi.mlb.com" not in str(record.get("url", "")):
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)


def _outcome_blind_schedule_rows(
    schedule_rows: Sequence[Mapping[str, Any]],
    window: P24CWindow,
) -> tuple[dict[str, Any], ...]:
    """Remove target-window outcomes before the P13 feature pass."""

    rows: list[dict[str, Any]] = []
    for source_row in schedule_rows:
        row = dict(source_row)
        if window.start_date <= str(row["official_date"]) <= window.end_date:
            row["home_score"] = 0
            row["away_score"] = 0
            row["final"] = True
            row["status"] = "Final"
        rows.append(row)
    return tuple(rows)


def _feature_fold(
    *,
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    source_manifest_fingerprint: str,
    window: P24CWindow,
) -> tuple[Any, FutureFeatureEligibility]:
    eligibility = classify_future_feature_eligibility(
        schedule_rows=tuple(schedule_rows),
        target_boxscore_rows=tuple(target_boxscore_rows),
        pitcher_game_log_rows=tuple(pitcher_game_log_rows),
        fold_id=window.fold_id,
        validation_start=window.start_date,
        validation_end=window.end_date,
    )
    if len(eligibility.raw_game_ids) != (
        len(eligibility.evaluable_game_ids)
        + len(eligibility.feature_unavailable_rows)
    ):
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    fold = materialize_future_moneyline_fold(
        schedule_rows=_outcome_blind_schedule_rows(schedule_rows, window),
        target_boxscore_rows=tuple(target_boxscore_rows),
        pitcher_game_log_rows=tuple(pitcher_game_log_rows),
        source_manifest_fingerprint=source_manifest_fingerprint,
        fold_id=window.fold_id,
        validation_start=window.start_date,
        validation_end=window.end_date,
        evaluable_game_ids=frozenset(eligibility.evaluable_game_ids),
        raw_game_ids=eligibility.raw_game_ids,
        feature_unavailable_rows=eligibility.feature_unavailable_rows,
    )
    return fold, eligibility


def _schedule_observation_id(row: FutureFeatureRow) -> str:
    projection = row.projection(include_fingerprint=False)
    projection.pop("features", None)
    projection.pop("feature_fingerprint", None)
    return sha256_bytes(canonical_json_bytes(projection))


def _snapshot_for_feature_row(
    row: FutureFeatureRow,
    *,
    batch_id: str,
) -> MoneylineFeatureSnapshot:
    as_of = _parse_utc(row.feature_as_of_utc)
    scheduled = _parse_utc(row.scheduled_start_utc)
    schedule_observation_id = _schedule_observation_id(row)
    feature_projection = row.projection(include_fingerprint=False)["features"]
    provenance = tuple(
        MoneylineFeatureProvenance(
            field_name=field_name,
            source_id=f"{schedule_observation_id}:{batch_id}:{field_name}",
            source_kind="MLB_STATS_API_P24C_PIT_FEATURE",
            observed_as_of_utc=as_of,
            source_fingerprint=sha256_bytes(
                canonical_json_bytes(
                    {
                        "field_name": field_name,
                        "game_id": row.provider_game_id,
                        "schedule_observation_id": schedule_observation_id,
                        "value": feature_projection[field_name],
                    }
                )
            ),
        )
        for field_name in ("recent_win_rate_delta", "starter_era_delta")
    )
    return MoneylineFeatureSnapshot.from_record(
        feature_projection,
        identity=MatchIdentity(
            sport="baseball",
            league="MLB",
            season=int(row.official_date[:4]),
            canonical_game_id=f"MLB:{row.official_date}:{row.game_pk}:{row.game_number}",
            home_participant=row.home_team,
            away_participant=row.away_team,
        ),
        provider_namespace="MLB_STATS_API",
        provider_game_id=row.provider_game_id,
        game_number=row.game_number,
        source_schedule_observation_id=schedule_observation_id,
        as_of_utc=as_of,
        scheduled_start_utc=scheduled,
        feature_provenance=provenance,
    )


def _inference_times(
    snapshot: MoneylineFeatureSnapshot,
) -> tuple[datetime, datetime, datetime]:
    """Use a deterministic timestamp at the snapshot's own PIT boundary."""

    return snapshot.as_of_utc, snapshot.as_of_utc, snapshot.as_of_utc


def _inference_pairs(
    snapshots: Sequence[MoneylineFeatureSnapshot],
    artifact: MoneylineModelArtifact,
) -> tuple[tuple[MoneylineFeatureSnapshot, ProspectivePredictionCandidate, ProspectivePredictionCandidate], ...]:
    pairs = []
    for snapshot in snapshots:
        generated, received, ingested = _inference_times(snapshot)
        result = generate_moneyline_predictions(
            (snapshot,),
            artifact,
            prediction_generated_at_utc=generated,
            response_received_at_utc=received,
            ingested_at_utc=ingested,
        )
        by_selection = {candidate.selection: candidate for candidate in result.candidates}
        if set(by_selection) != {"HOME", "AWAY"}:
            raise RuntimeError(P24C_STOP_FEATURE_INPUT)
        pairs.append((snapshot, by_selection["HOME"], by_selection["AWAY"]))
    return tuple(pairs)


def _prediction_rows(
    pairs: Sequence[
        tuple[
            MoneylineFeatureSnapshot,
            ProspectivePredictionCandidate,
            ProspectivePredictionCandidate,
        ]
    ],
    *,
    batch_id: str,
    window: P24CWindow,
    model_id: str,
    model_fingerprint: str,
    inference_model_fingerprint: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for snapshot, home, away in pairs:
        rows.append(
            {
                "schema_version": P24C_BATCH_SCHEMA_VERSION,
                "prediction_id": home.prediction_observation_id,
                "source_prediction_id": home.source_prediction_id,
                "batch_id": batch_id,
                "fold_id": window.fold_id,
                "source_fold_window_identity": (
                    f"MLB_STATS_API:{window.fold_id}:{window.start_date}:{window.end_date}"
                ),
                "window_start_date": window.start_date,
                "window_end_date": window.end_date,
                "game_id": snapshot.provider_game_id,
                "source_provider_game_id": snapshot.provider_game_id.split("@", 1)[0],
                "scheduled_start": format_canonical_utc(snapshot.scheduled_start_utc),
                "home_team": snapshot.identity.home_participant,
                "away_team": snapshot.identity.away_participant,
                "feature_snapshot_id": snapshot.fingerprint(),
                "feature_snapshot_fingerprint": snapshot.fingerprint(),
                "model_id": model_id,
                "model_fingerprint": model_fingerprint,
                "inference_model_fingerprint": inference_model_fingerprint,
                "home_win_probability": str(home.model_probability),
                "away_win_probability": str(away.model_probability),
                "predicted_side": (
                    "HOME" if home.model_probability >= Decimal("0.5") else "AWAY"
                ),
                "inference_mode": "PAPER_DEFAULT",
                "generated_from_historical_shadow": True,
            }
        )
    return tuple(rows)


def _unavailable_rows(
    eligibility: FutureFeatureEligibility,
    *,
    batch_id: str,
    window: P24CWindow,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for source_row in eligibility.feature_unavailable_rows:
        affected = [dict(item) for item in source_row["affected_starters"]]
        rows.append(
            {
                "schema_version": P24C_BATCH_SCHEMA_VERSION,
                "batch_id": batch_id,
                "fold_id": window.fold_id,
                "window_start_date": window.start_date,
                "window_end_date": window.end_date,
                "game_id": str(source_row["game_id"]),
                "source_provider_game_id": str(
                    source_row.get("source_provider_game_id", source_row["game_id"])
                ),
                "scheduled_start": str(source_row["scheduled_start"]),
                "eligibility": "FEATURE_UNAVAILABLE",
                "status": "FEATURE_UNAVAILABLE",
                "reason": str(source_row["reason"]),
                "affected_feature": str(source_row["feature_name"]),
                "affected_starter_ids": [int(item["starter_id"]) for item in affected],
                "affected_starters": affected,
                "prior_qualifying_history": [
                    {
                        "starter_id": int(item["starter_id"]),
                        "count": int(item["qualifying_prior_start_count"]),
                        "required": int(item["required_prior_start_count"]),
                    }
                    for item in affected
                ],
                "required_history_count": max(
                    int(item["required_prior_start_count"]) for item in affected
                ),
                "generated_from_historical_shadow": True,
            }
        )
    return tuple(
        sorted(rows, key=lambda row: (row["scheduled_start"], row["game_id"]))
    )


def _raw_membership(target_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "game_id": str(row["provider_game_id"]),
            "source_provider_game_id": str(
                row.get("source_provider_game_id", row["provider_game_id"])
            ),
            "game_number": int(row["game_number"]),
            "scheduled_start": str(row["scheduled_start_utc"]),
        }
        for row in target_rows
    )


def _batch_id(
    *,
    window: P24CWindow,
    raw_membership: Sequence[Mapping[str, Any]],
    model_id: str,
    model_fingerprint: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": P24C_BATCH_SCHEMA_VERSION,
                "window_start_date": window.start_date,
                "window_end_date": window.end_date,
                "raw_game_membership": list(raw_membership),
                "source_authority": "MLB_STATS_API",
                "feature_semantics_version": P24C_FEATURE_SEMANTICS_VERSION,
                "promoted_default_model_id": model_id,
                "promoted_default_model_fingerprint": model_fingerprint,
            }
        )
    )


def _feature_rows_fingerprint(
    snapshots: Sequence[MoneylineFeatureSnapshot],
) -> str:
    return sha256_bytes(
        b"".join(snapshot.canonical_bytes() for snapshot in snapshots)
    )


def _verify_default_and_explicit_equivalence(
    *,
    snapshots: Sequence[MoneylineFeatureSnapshot],
    default_artifact: MoneylineModelArtifact,
    default_fingerprint: str,
    explicit_artifact: MoneylineModelArtifact,
    explicit_fingerprint: str,
) -> bool:
    if (
        explicit_artifact.model_id != P22B_MODEL_ID
        or explicit_fingerprint != P22B_ARTIFACT_FINGERPRINT
    ):
        raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE)
    default_pairs = _inference_pairs(snapshots, default_artifact)
    explicit_pairs = _inference_pairs(snapshots, explicit_artifact)
    if len(default_pairs) != len(explicit_pairs):
        raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE)
    tolerance = Decimal(P24C_PROBABILITY_TOLERANCE)
    for default, explicit in zip(default_pairs, explicit_pairs, strict=True):
        _, default_home, default_away = default
        _, explicit_home, explicit_away = explicit
        if (
            default_home.model_id != explicit_home.model_id
            or default_away.model_id != explicit_away.model_id
            or abs(default_home.model_probability - explicit_home.model_probability)
            > tolerance
            or abs(default_away.model_probability - explicit_away.model_probability)
            > tolerance
        ):
            raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE)
    return default_artifact.fingerprint() == explicit_artifact.fingerprint()


def _verify_outcome_isolation(
    *,
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    source_manifest_fingerprint: str,
    window: P24CWindow,
    batch_id: str,
    snapshots: Sequence[MoneylineFeatureSnapshot],
    predictions: Sequence[Mapping[str, Any]],
    unavailable: Sequence[Mapping[str, Any]],
    default_artifact: MoneylineModelArtifact,
    default_fingerprint: str,
) -> bool:
    mutated_schedule = []
    for source_row in schedule_rows:
        row = dict(source_row)
        if window.start_date <= str(row["official_date"]) <= window.end_date:
            row["home_score"] = 999
            row["away_score"] = 0
        mutated_schedule.append(row)
    mutated_fold, mutated_eligibility = _feature_fold(
        schedule_rows=tuple(mutated_schedule),
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest_fingerprint=source_manifest_fingerprint,
        window=window,
    )
    mutated_snapshots = tuple(
        _snapshot_for_feature_row(row, batch_id=batch_id)
        for row in mutated_fold.feature_rows
    )
    if tuple(snapshot.to_projection() for snapshot in snapshots) != tuple(
        snapshot.to_projection() for snapshot in mutated_snapshots
    ):
        return False
    mutated_unavailable = _unavailable_rows(
        mutated_eligibility,
        batch_id=batch_id,
        window=window,
    )
    if tuple(unavailable) != mutated_unavailable:
        return False
    mutated_pairs = _inference_pairs(mutated_snapshots, default_artifact)
    mutated_predictions = _prediction_rows(
        mutated_pairs,
        batch_id=batch_id,
        window=window,
        model_id=default_artifact.model_id,
        model_fingerprint=default_fingerprint,
        inference_model_fingerprint=default_artifact.fingerprint(),
    )
    return tuple(predictions) == mutated_predictions


def generate_paper_moneyline_batch(
    *,
    repository_root: str | Path,
    schedule_rows: Sequence[Mapping[str, Any]],
    target_boxscore_rows: Sequence[Mapping[str, Any]],
    pitcher_game_log_rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    offline_replay_verified: bool,
    explicit_model_artifact_path: str | Path | None = None,
    incumbent_model_artifact_path: str | Path | None = None,
) -> PaperMoneylineBatchResult:
    """Build one outcome-blind P24C batch and its deterministic ledgers."""

    repository_root = Path(repository_root)
    schedule_rows = _canonicalize_game_rows(schedule_rows)
    target_boxscore_rows = _canonicalize_game_rows(target_boxscore_rows)
    window = resolve_p24c_window(repository_root)
    target_rows = _target_rows(schedule_rows, window)
    default_artifact_path = default_paper_moneyline_model_artifact_path(repository_root)
    try:
        default_artifact, default_fingerprint = load_default_paper_moneyline_model_artifact(
            repository_root
        )
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(P24C_STOP_DEFAULT_DRIFT) from exc
    if (
        default_artifact.model_id != P22B_MODEL_ID
        or default_fingerprint != P22B_ARTIFACT_FINGERPRINT
        or default_artifact_path != repository_root / P22B_ARTIFACT_RELATIVE_PATH
    ):
        raise RuntimeError(P24C_STOP_DEFAULT_DRIFT)

    explicit_path = Path(explicit_model_artifact_path or default_artifact_path)
    try:
        explicit_artifact, explicit_fingerprint = load_model_artifact_with_fingerprint(
            explicit_path
        )
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE) from exc

    source_manifest_fingerprint = sha256_bytes(canonical_json_bytes(source_manifest))
    fold, eligibility = _feature_fold(
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest_fingerprint=source_manifest_fingerprint,
        window=window,
    )
    snapshots_without_batch = tuple(
        _snapshot_for_feature_row(row, batch_id="pending")
        for row in fold.feature_rows
    )
    raw_membership = _raw_membership(target_rows)
    batch_id = _batch_id(
        window=window,
        raw_membership=raw_membership,
        model_id=default_artifact.model_id,
        model_fingerprint=default_fingerprint,
    )
    snapshots = tuple(
        _snapshot_for_feature_row(row, batch_id=batch_id) for row in fold.feature_rows
    )
    if snapshots_without_batch and snapshots_without_batch[0].fingerprint() == snapshots[0].fingerprint():
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    default_pairs = _inference_pairs(snapshots, default_artifact)
    predictions = _prediction_rows(
        default_pairs,
        batch_id=batch_id,
        window=window,
        model_id=default_artifact.model_id,
        model_fingerprint=default_fingerprint,
        inference_model_fingerprint=default_artifact.fingerprint(),
    )
    equivalence = _verify_default_and_explicit_equivalence(
        snapshots=snapshots,
        default_artifact=default_artifact,
        default_fingerprint=default_fingerprint,
        explicit_artifact=explicit_artifact,
        explicit_fingerprint=explicit_fingerprint,
    )

    incumbent_path = Path(
        incumbent_model_artifact_path
        or repository_root / "data/fixtures/p19a_moneyline_inference/model_artifact.json"
    )
    try:
        incumbent, _ = load_model_artifact_with_fingerprint(incumbent_path)
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE) from exc
    incumbent_pairs = _inference_pairs((snapshots[0],), incumbent)
    incumbent_override_verified = (
        incumbent.model_id == "p13_walk_forward_logistic_v1_fixture"
        and bool(incumbent_pairs)
        and incumbent_pairs[0][1].model_id == incumbent.model_id
        and incumbent.model_id != default_artifact.model_id
    )
    if not incumbent_override_verified:
        raise RuntimeError(P24C_STOP_OVERRIDE_DIVERGENCE)

    unavailable = _unavailable_rows(
        eligibility,
        batch_id=batch_id,
        window=window,
    )
    outcome_isolation = _verify_outcome_isolation(
        schedule_rows=schedule_rows,
        target_boxscore_rows=target_boxscore_rows,
        pitcher_game_log_rows=pitcher_game_log_rows,
        source_manifest_fingerprint=source_manifest_fingerprint,
        window=window,
        batch_id=batch_id,
        snapshots=snapshots,
        predictions=predictions,
        unavailable=unavailable,
        default_artifact=default_artifact,
        default_fingerprint=default_fingerprint,
    )
    if not outcome_isolation:
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)

    reason_counts = dict(
        sorted(Counter(str(row["reason"]) for row in unavailable).items())
    )
    prediction_fingerprint = sha256_bytes(render_jsonl(predictions))
    unavailable_fingerprint = sha256_bytes(render_jsonl(unavailable))
    raw_membership_fingerprint = sha256_bytes(render_jsonl(raw_membership))
    summary = {
        "schema_version": P24C_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "fold_id": window.fold_id,
        "window_start_date": window.start_date,
        "window_end_date": window.end_date,
        "raw_game_count": len(raw_membership),
        "evaluable_game_count": len(predictions),
        "feature_unavailable_count": len(unavailable),
        "feature_unavailable_reason_counts": reason_counts,
        "evaluation_coverage": str(
            Decimal(len(predictions)) / Decimal(len(raw_membership))
        ),
        "raw_game_membership_fingerprint": raw_membership_fingerprint,
        "feature_snapshot_set_fingerprint": _feature_rows_fingerprint(snapshots),
        "prediction_set_fingerprint": prediction_fingerprint,
        "feature_unavailable_set_fingerprint": unavailable_fingerprint,
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "promoted_default_model_id": default_artifact.model_id,
        "promoted_default_model_fingerprint": default_fingerprint,
        "promoted_default_inference_model_fingerprint": default_artifact.fingerprint(),
        "default_explicit_equivalence_verified": equivalence,
        "explicit_incumbent_override_verified": incumbent_override_verified,
        "result_mutation_isolation_verified": outcome_isolation,
        "offline_replay_verified": offline_replay_verified,
        "historical_shadow": True,
        "paper_only": True,
        "model_promoted": True,
        "promotion_scope": "paper_only",
        "challenger_retrained": False,
        "production_ready": False,
        "deployment_performed": False,
        "real_betting_recommendation": False,
        "profitability_claim": False,
        "p20b_historical_runtime_compliance": "REMAINS_REFUTED",
        "claims": {
            "historical_shadow": True,
            "paper_only": True,
            "model_promoted": True,
            "promotion_scope": "paper_only",
            "challenger_retrained": False,
            "production_ready": False,
            "deployment_performed": False,
            "real_betting_recommendation": False,
            "profitability_claim": False,
        },
    }
    return PaperMoneylineBatchResult(
        predictions=predictions,
        feature_unavailable=unavailable,
        source_manifest=dict(source_manifest),
        summary=summary,
    )


def load_p24c_source_inputs(
    *,
    repository_root: str | Path,
    raw_root: str | Path,
    normalized_root: str | Path,
    source_manifest_path: str | Path,
    window: P24CWindow,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    root = Path(repository_root)
    raw_root = Path(raw_root)
    normalized_root = Path(normalized_root)
    source_manifest = json.loads(
        Path(source_manifest_path).read_text(encoding="utf-8")
    )
    if not isinstance(source_manifest, dict):
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    _verify_source_manifest(
        repository_root=root,
        raw_root=raw_root,
        normalized_root=normalized_root,
        source_manifest=source_manifest,
        window=window,
    )
    schedule_rows = _canonicalize_game_rows(load_normalized_rows(
        root / "data/fixtures/p23f2_official_2026_history/normalized/schedule.jsonl"
    ))
    target_schedule_rows = _canonicalize_game_rows(
        load_normalized_rows(normalized_root / "schedule.jsonl")
    )
    expected_target_ids = {str(row["provider_game_id"]) for row in _target_rows(schedule_rows, window)}
    if {str(row["provider_game_id"]) for row in target_schedule_rows} != expected_target_ids:
        raise RuntimeError(P24C_STOP_FEATURE_INPUT)
    boxscore_rows = _canonicalize_game_rows(
        load_normalized_rows(normalized_root / "target_boxscores.jsonl")
    )
    pitcher_rows = load_normalized_rows(normalized_root / "pitcher_game_logs.jsonl")
    return schedule_rows, boxscore_rows, pitcher_rows, source_manifest


def acquire_p24c_source_inputs(
    *,
    repository_root: str | Path,
    raw_root: str | Path,
    normalized_root: str | Path,
    window: P24CWindow,
    acquired_at_utc: str,
) -> dict[str, Any]:
    acquisition = acquire_official_future_fold(
        repository_root=repository_root,
        fold_id=window.fold_id,
        validation_start=window.start_date,
        validation_end=window.end_date,
        raw_root=raw_root,
        normalized_root=normalized_root,
        acquired_at_utc=_parse_utc(acquired_at_utc),
    )
    source_manifest = build_source_manifest(
        source_records=tuple(
            {
                "path": record.path,
                "url": record.url,
                "scope": record.scope,
                "acquired_at_utc": record.acquired_at_utc,
                "sha256": record.sha256,
            }
            for record in acquisition.source_records
        ),
        normalized_hashes=acquisition.normalized_hashes,
        window=window,
        acquired_at_utc=acquired_at_utc,
    )
    _verify_source_manifest(
        repository_root=Path(repository_root),
        raw_root=Path(raw_root),
        normalized_root=Path(normalized_root),
        source_manifest=source_manifest,
        window=window,
    )
    return {
        "schedule_rows": _canonicalize_game_rows(acquisition.schedule_rows),
        "target_boxscore_rows": _canonicalize_game_rows(
            acquisition.target_boxscore_rows
        ),
        "pitcher_game_log_rows": acquisition.pitcher_game_log_rows,
        "source_manifest": source_manifest,
    }


__all__ = (
    "P24C_FOLD_ID",
    "P24C_STOP_DEFAULT_DRIFT",
    "P24C_STOP_FEATURE_INPUT",
    "P24C_STOP_OVERRIDE_DIVERGENCE",
    "P24C_STOP_P24B_BASELINE",
    "P24C_STOP_WINDOW_UNAVAILABLE",
    "P24CWindow",
    "PaperMoneylineBatchResult",
    "acquire_p24c_source_inputs",
    "build_source_manifest",
    "generate_paper_moneyline_batch",
    "load_p24c_source_inputs",
    "resolve_p24c_window",
)
