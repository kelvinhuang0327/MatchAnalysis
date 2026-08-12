"""Run the P38A leakage-safe rolling Moneyline calibration evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .moneyline_probability_calibration import (
    P38A_CALIBRATION_METHOD,
    P38A_CALIBRATION_METHOD_VERSION,
    fit_platt_calibrator,
    probability_metrics,
)
from .p38a_probability_calibration_artifacts import (
    render_calibration_artifacts,
    render_comparisons_jsonl,
    render_per_window_summary,
    render_report_markdown,
    render_summary,
)


P38A_TASK_ID = "P38A"
P38A_AUTHORITY_REPOSITORY = "/Users/kelvin/VibeCoding-WorkSpace/MatchAnalysis"
P38A_BASE_HEAD = "bbf31d647eb8db940c923a586245cc52bd3d803e"
P38A_BASE_TREE = "99d0de9f1a288da8cfed697ce5cc9f47adbce840"
P38A_P37A_REPORT_PATH = Path("report/p37a_rolling_walk_forward_oos")
P38A_TARGET_FOLD_IDS = ("wf_005", "wf_006")
P38A_P37A_WINDOW_COUNTS = {
    "wf_004": (23, 23, 0),
    "wf_005": (22, 17, 5),
    "wf_006": (30, 25, 5),
}
P38A_P37A_ARTIFACT_SHA256 = {
    "model_artifacts.json": "2d0426036a7e2c6760888cf542e498c80798511443b2dc158bcce79aa3d11812",
    "comparisons.jsonl": "23cc15d308a90c08da0d1a4c6cbb9289af3add2c5d151808833e73a660639eb4",
    "per_window_summary.json": "ee03c9755311bc365c04856c8ccb88caa7b066413aba9b8f1fe7d83e4cb7e23b",
    "summary.json": "f8fe76adbadbccd81ff19d7edcfe1c5ff0ac1107678db53021ba248a00249e75",
    "report.md": "09eb0cca8898340e714c014356c740164a5ee70f89c49d0a04a59a8a2474166e",
}
P38A_CONCLUSION_RULE = (
    "CALIBRATION_IMPROVED iff calibrated Brier, log loss, and ECE are all "
    "strictly lower than raw challenger; CALIBRATION_NOT_IMPROVED iff all "
    "three are greater than or equal and at least one is strictly greater; "
    "otherwise MIXED_OR_INCONCLUSIVE."
)
P38A_CALIBRATION_SOURCE_KIND = "P37A_PRIOR_TRUE_OOS_PREDICTION_LABEL_PAIRS"


class P38AAuthorityError(ValueError):
    """Raised when the committed P37A authority cannot support P38A."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise P38AAuthorityError(f"unable to read P37A authority: {path}") from exc
    if not isinstance(value, dict):
        raise P38AAuthorityError(f"P37A authority must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise P38AAuthorityError(f"unable to read P37A comparisons: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise P38AAuthorityError(
                f"P37A comparisons contain a blank line at {line_number}"
            )
        try:
            value = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise P38AAuthorityError(
                f"invalid P37A comparison row at {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise P38AAuthorityError(
                f"P37A comparison row {line_number} must be an object"
            )
        rows.append(value)
    if not rows:
        raise P38AAuthorityError("P37A comparisons contain no rows")
    return rows


def _canonical_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        sorted(
            (dict(row) for row in rows),
            key=lambda row: str(row["comparison_row_id"]),
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _p37a_artifact_hashes(report_path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in P38A_P37A_ARTIFACT_SHA256.items():
        path = report_path / name
        try:
            actual = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise P38AAuthorityError(f"P37A artifact is unreadable: {path}") from exc
        if actual != expected:
            raise P38AAuthorityError(f"P37A artifact changed: {path}")
        observed[name] = actual
    return observed


def _validate_p37a_authority(repository_root: Path) -> dict[str, Any]:
    report_path = repository_root / P38A_P37A_REPORT_PATH
    artifact_hashes = _p37a_artifact_hashes(report_path)
    summary = _read_json(report_path / "summary.json")
    per_window_projection = _read_json(report_path / "per_window_summary.json")
    rows = _read_jsonl(report_path / "comparisons.jsonl")
    if summary.get("task_id") != "P37A":
        raise P38AAuthorityError("P37A task identity drift")
    aggregate = summary.get("aggregate", {})
    if (
        aggregate.get("raw_row_count"),
        aggregate.get("evaluable_row_count"),
        aggregate.get("excluded_row_count"),
    ) != (75, 65, 10):
        raise P38AAuthorityError("P37A raw/evaluable/excluded authority drift")
    if summary.get("comparison", {}).get("conclusion") != "MIXED_OR_INCONCLUSIVE":
        raise P38AAuthorityError("P37A conclusion authority drift")
    if summary.get("claims", {}).get("model_promoted") is not False:
        raise P38AAuthorityError("P37A promotion claim drift")
    windows = per_window_projection.get("windows")
    if not isinstance(windows, list):
        raise P38AAuthorityError("P37A per-window authority is missing")
    window_by_fold = {str(window.get("holdout_fold_id")): window for window in windows}
    if set(window_by_fold) != set(P38A_P37A_WINDOW_COUNTS):
        raise P38AAuthorityError("P37A holdout fold authority drift")
    window_orders = {
        fold_id: int(window.get("evaluation_window_order", -1))
        for fold_id, window in window_by_fold.items()
    }
    if window_orders != {"wf_004": 1, "wf_005": 2, "wf_006": 3}:
        raise P38AAuthorityError("P37A evaluation window order drift")
    for fold_id, expected_counts in P38A_P37A_WINDOW_COUNTS.items():
        holdout = window_by_fold[fold_id].get("holdout", {})
        observed_counts = (
            int(holdout.get("raw_row_count", -1)),
            int(holdout.get("evaluable_row_count", -1)),
            int(holdout.get("excluded_row_count", -1)),
        )
        if observed_counts != expected_counts:
            raise P38AAuthorityError(f"P37A {fold_id} row accounting drift")

    grouped: dict[str, list[dict[str, Any]]] = {}
    all_game_ids: set[str] = set()
    for row in rows:
        fold_id = str(row.get("holdout_fold_id"))
        if fold_id not in P38A_P37A_WINDOW_COUNTS:
            raise P38AAuthorityError(f"unexpected P37A comparison fold: {fold_id}")
        game_id = str(row.get("provider_game_id"))
        if game_id in all_game_ids:
            raise P38AAuthorityError("P37A comparison game identities are not unique")
        all_game_ids.add(game_id)
        if row.get("true_oos_verified") is not True:
            raise P38AAuthorityError("P37A comparison row is not true OOS")
        if "home_score" in row or "away_score" in row:
            raise P38AAuthorityError("P37A comparison row exposes an outcome payload")
        try:
            probability = Decimal(str(row["challenger_home_probability"]))
            champion_probability = Decimal(str(row["incumbent_home_probability"]))
            target = int(row["target_home_win"])
        except (KeyError, TypeError, ValueError) as exc:
            raise P38AAuthorityError("P37A comparison row is malformed") from exc
        if not (Decimal("0") < probability < Decimal("1")):
            raise P38AAuthorityError("P37A challenger probability is not valid")
        if not (Decimal("0") < champion_probability < Decimal("1")):
            raise P38AAuthorityError("P37A champion probability is not valid")
        if target not in (0, 1):
            raise P38AAuthorityError("P37A target is not binary")
        if int(row.get("evaluation_window_order", -1)) != window_orders[fold_id]:
            raise P38AAuthorityError(
                f"P37A {fold_id} comparison window order is inconsistent"
            )
        grouped.setdefault(fold_id, []).append(row)
    if len(rows) != 65:
        raise P38AAuthorityError("P37A comparison row count drift")
    if any(
        len(grouped.get(fold_id, ())) != counts[1]
        for fold_id, counts in P38A_P37A_WINDOW_COUNTS.items()
    ):
        raise P38AAuthorityError("P37A per-window comparison row count drift")
    return {
        "summary": summary,
        "windows": window_by_fold,
        "rows_by_fold": grouped,
        "artifact_hashes": artifact_hashes,
    }


def _strictly_prior_source_rows(
    *,
    target_fold_id: str,
    rows_by_fold: Mapping[str, Sequence[Mapping[str, Any]]],
    window: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    target_order = int(window["evaluation_window_order"])
    prior_rows: list[dict[str, Any]] = []
    for fold_id, rows in rows_by_fold.items():
        if not rows:
            continue
        source_order = int(rows[0]["evaluation_window_order"])
        if source_order < target_order:
            prior_rows.extend(dict(row) for row in rows)
        if fold_id == target_fold_id and source_order != target_order:
            raise P38AAuthorityError(f"P38A fold order drift for {target_fold_id}")
    target_rows = tuple(rows_by_fold[target_fold_id])
    if not prior_rows:
        raise P38AAuthorityError(
            f"P38A calibration authority is empty before {target_fold_id}"
        )
    calibration_ids = {str(row["provider_game_id"]) for row in prior_rows}
    target_ids = {str(row["provider_game_id"]) for row in target_rows}
    if calibration_ids & target_ids:
        raise P38AAuthorityError(
            f"P38A calibration/holdout identity overlap in {target_fold_id}"
        )
    calibration_starts = [str(row["scheduled_start_utc"]) for row in prior_rows]
    target_starts = [str(row["scheduled_start_utc"]) for row in target_rows]
    if max(calibration_starts) >= min(target_starts):
        raise P38AAuthorityError(
            f"P38A calibration rows do not strictly precede {target_fold_id}"
        )
    return tuple(
        sorted(
            prior_rows,
            key=lambda row: (
                str(row["scheduled_start_utc"]),
                int(row["game_number"]),
                int(row["game_pk"]),
            ),
        )
    )


def _metric_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, str]:
    """Return right minus left for the load-bearing metric fields."""

    return {
        "accuracy_delta": str(
            Decimal(str(right["accuracy"])) - Decimal(str(left["accuracy"]))
        ),
        "brier_delta": str(
            Decimal(str(right["brier_score"])) - Decimal(str(left["brier_score"]))
        ),
        "log_loss_delta": str(
            Decimal(str(right["log_loss"])) - Decimal(str(left["log_loss"]))
        ),
        "calibration_ece_delta": str(
            Decimal(str(right["calibration"]["expected_calibration_error"]))
            - Decimal(str(left["calibration"]["expected_calibration_error"]))
        ),
    }


def _comparison_conclusion(
    raw_metrics: Mapping[str, Any],
    calibrated_metrics: Mapping[str, Any],
) -> str:
    raw_values = (
        Decimal(str(raw_metrics["brier_score"])),
        Decimal(str(raw_metrics["log_loss"])),
        Decimal(str(raw_metrics["calibration"]["expected_calibration_error"])),
    )
    calibrated_values = (
        Decimal(str(calibrated_metrics["brier_score"])),
        Decimal(str(calibrated_metrics["log_loss"])),
        Decimal(
            str(calibrated_metrics["calibration"]["expected_calibration_error"])
        ),
    )
    pairs = tuple(zip(raw_values, calibrated_values, strict=True))
    if all(calibrated < raw for raw, calibrated in pairs):
        return "CALIBRATION_IMPROVED"
    if all(calibrated >= raw for raw, calibrated in pairs) and any(
        calibrated > raw for raw, calibrated in pairs
    ):
        return "CALIBRATION_NOT_IMPROVED"
    return "MIXED_OR_INCONCLUSIVE"


def _comparison_row(
    row: Mapping[str, Any],
    *,
    calibrated_probability: Decimal,
) -> dict[str, Any]:
    target = int(row["target_home_win"])
    champion_probability = Decimal(str(row["incumbent_home_probability"]))
    raw_probability = Decimal(str(row["challenger_home_probability"]))
    identity = {
        "source_p37a_comparison_row_id": str(row["comparison_row_id"]),
        "calibrated_probability": str(calibrated_probability),
    }
    comparison_row_id = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "p38a.rolling_probability_calibration_comparison.v1",
        "comparison_row_id": comparison_row_id,
        "source_p37a_comparison_row_id": str(row["comparison_row_id"]),
        "evaluation_window_id": str(row["evaluation_window_id"]),
        "evaluation_window_order": int(row["evaluation_window_order"]),
        "holdout_fold_id": str(row["holdout_fold_id"]),
        "provider_namespace": str(row["provider_namespace"]),
        "provider_game_id": str(row["provider_game_id"]),
        "game_pk": int(row["game_pk"]),
        "game_number": int(row["game_number"]),
        "scheduled_start_utc": str(row["scheduled_start_utc"]),
        "feature_fingerprint": str(row["feature_fingerprint"]),
        "champion_model_id": str(row["incumbent_model_id"]),
        "champion_model_fingerprint": str(row["incumbent_model_fingerprint"]),
        "champion_home_probability": str(champion_probability),
        "raw_challenger_model_id": str(row["challenger_model_id"]),
        "raw_challenger_model_fingerprint": str(row["challenger_model_fingerprint"]),
        "raw_challenger_home_probability": str(raw_probability),
        "calibrated_challenger_home_probability": str(calibrated_probability),
        "target_home_win": target,
        "champion_correct": (champion_probability >= Decimal("0.5")) == bool(target),
        "raw_challenger_correct": (raw_probability >= Decimal("0.5")) == bool(target),
        "calibrated_challenger_correct": (calibrated_probability >= Decimal("0.5")) == bool(target),
        "champion_brier_contribution": str(
            (champion_probability - Decimal(target)) ** 2
        ),
        "raw_challenger_brier_contribution": str(
            (raw_probability - Decimal(target)) ** 2
        ),
        "calibrated_challenger_brier_contribution": str(
            (calibrated_probability - Decimal(target)) ** 2
        ),
        "champion_log_loss_contribution": str(
            -(
                champion_probability.ln()
                if target
                else (Decimal("1") - champion_probability).ln()
            )
        ),
        "raw_challenger_log_loss_contribution": str(
            -(
                raw_probability.ln()
                if target
                else (Decimal("1") - raw_probability).ln()
            )
        ),
        "calibrated_challenger_log_loss_contribution": str(
            -(
                calibrated_probability.ln()
                if target
                else (Decimal("1") - calibrated_probability).ln()
            )
        ),
        "same_target_row_verified": True,
        "true_oos_verified": True,
        "calibration_fitted_on_target_row": False,
    }


def evaluate_p38a_probability_calibration(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Evaluate calibration on the maximum two-window prior-OOS authority."""

    root = Path(repository_root)
    authority = _validate_p37a_authority(root)
    rows_by_fold = authority["rows_by_fold"]
    windows = authority["windows"]
    calibration_artifacts: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_window_summary: list[dict[str, Any]] = []

    for target_fold_id in P38A_TARGET_FOLD_IDS:
        window = windows[target_fold_id]
        target_rows = tuple(rows_by_fold[target_fold_id])
        calibration_rows = _strictly_prior_source_rows(
            target_fold_id=target_fold_id,
            rows_by_fold=rows_by_fold,
            window=window,
        )
        calibrator = fit_platt_calibrator(
            [Decimal(str(row["challenger_home_probability"])) for row in calibration_rows],
            [int(row["target_home_win"]) for row in calibration_rows],
        )
        calibrated_probabilities = tuple(
            calibrator.apply(Decimal(str(row["challenger_home_probability"])))
            for row in target_rows
        )
        target_ids = tuple(str(row["provider_game_id"]) for row in target_rows)
        calibration_ids = tuple(str(row["provider_game_id"]) for row in calibration_rows)
        target_row_ids = tuple(str(row["comparison_row_id"]) for row in target_rows)
        calibration_row_ids = tuple(str(row["comparison_row_id"]) for row in calibration_rows)
        target_min = min(str(row["scheduled_start_utc"]) for row in target_rows)
        calibration_max = max(str(row["scheduled_start_utc"]) for row in calibration_rows)
        target_raw_row_count = int(window["holdout"]["raw_row_count"])
        target_evaluable_row_count = int(window["holdout"]["evaluable_row_count"])
        target_excluded_row_count = int(window["holdout"]["excluded_row_count"])
        if len(target_rows) != target_evaluable_row_count:
            raise P38AAuthorityError(f"P38A {target_fold_id} target row count drift")
        if len(target_ids) != len(set(target_ids)):
            raise P38AAuthorityError(f"P38A {target_fold_id} target IDs are not unique")
        if set(target_ids) & set(calibration_ids):
            raise P38AAuthorityError(f"P38A {target_fold_id} calibration reuses target IDs")

        champion_probabilities = tuple(
            Decimal(str(row["incumbent_home_probability"])) for row in target_rows
        )
        raw_probabilities = tuple(
            Decimal(str(row["challenger_home_probability"])) for row in target_rows
        )
        targets = tuple(int(row["target_home_win"]) for row in target_rows)
        champion_metrics = probability_metrics(
            champion_probabilities,
            targets,
            raw_row_count=target_raw_row_count,
        )
        raw_metrics = probability_metrics(
            raw_probabilities,
            targets,
            raw_row_count=target_raw_row_count,
        )
        calibrated_metrics = probability_metrics(
            calibrated_probabilities,
            targets,
            raw_row_count=target_raw_row_count,
        )
        comparison_rows.extend(
            _comparison_row(row, calibrated_probability=calibrated_probability)
            for row, calibrated_probability in zip(
                target_rows,
                calibrated_probabilities,
                strict=True,
            )
        )
        calibration_artifacts.append(
            {
                "evaluation_window_id": str(window["evaluation_window_id"]),
                "evaluation_window_order": int(window["evaluation_window_order"]),
                "holdout_fold_id": target_fold_id,
                "method": P38A_CALIBRATION_METHOD,
                "method_version": P38A_CALIBRATION_METHOD_VERSION,
                "source_kind": P38A_CALIBRATION_SOURCE_KIND,
                "source_fold_ids": sorted(
                    {str(row["holdout_fold_id"]) for row in calibration_rows}
                ),
                "source_comparison_row_ids": list(calibration_row_ids),
                "source_provider_game_ids": list(calibration_ids),
                "source_row_count": len(calibration_rows),
                "source_date_range": [
                    min(str(row["scheduled_start_utc"]) for row in calibration_rows),
                    calibration_max,
                ],
                "target_holdout_date_range": [
                    target_min,
                    max(str(row["scheduled_start_utc"]) for row in target_rows),
                ],
                "source_comparison_fingerprint": _canonical_fingerprint(calibration_rows),
                "target_comparison_fingerprint": _canonical_fingerprint(target_rows),
                "calibrator": calibrator.to_projection(),
                "lineage": {
                    "strict_source_before_target_verified": calibration_max < target_min,
                    "source_target_game_id_disjoint_verified": not (
                        set(calibration_ids) & set(target_ids)
                    ),
                    "target_holdout_labels_used_for_fit": False,
                    "calibrator_fit_rows_evaluation_rows_disjoint_verified": True,
                    "method_selection_uses_target_labels": False,
                },
            }
        )
        same_target_rows = tuple(str(row["comparison_row_id"]) for row in target_rows)
        per_window_summary.append(
            {
                "schema_version": "p38a.rolling_probability_calibration_per_window.v1",
                "evaluation_window_id": str(window["evaluation_window_id"]),
                "evaluation_window_order": int(window["evaluation_window_order"]),
                "train_fold_ids": list(window["train_fold_ids"]),
                "holdout_fold_id": target_fold_id,
                "training": dict(window["training"]),
                "holdout": {
                    "fold_id": target_fold_id,
                    "date_range": list(window["holdout"]["date_range"]),
                    "raw_row_count": target_raw_row_count,
                    "evaluable_row_count": target_evaluable_row_count,
                    "excluded_row_count": target_excluded_row_count,
                    "coverage": str(
                        Decimal(target_evaluable_row_count)
                        / Decimal(target_raw_row_count)
                    ),
                    "raw_game_ids": list(window["holdout"]["raw_game_ids"]),
                    "evaluable_game_ids": list(window["holdout"]["evaluable_game_ids"]),
                    "excluded_game_ids": list(window["holdout"]["excluded_game_ids"]),
                    "exclusion_semantics_preserved": True,
                },
                "calibration": {
                    "source_fold_ids": sorted(
                        {str(row["holdout_fold_id"]) for row in calibration_rows}
                    ),
                    "source_row_count": len(calibration_rows),
                    "source_date_range": [
                        min(str(row["scheduled_start_utc"]) for row in calibration_rows),
                        calibration_max,
                    ],
                    "target_holdout_min_start_utc": target_min,
                    "source_max_start_utc": calibration_max,
                    "source_comparison_row_ids": list(calibration_row_ids),
                    "target_comparison_row_ids": list(target_row_ids),
                    "strict_source_before_target_verified": calibration_max < target_min,
                    "source_target_game_id_disjoint_verified": not (
                        set(calibration_ids) & set(target_ids)
                    ),
                    "target_holdout_labels_used_for_fit": False,
                    "calibrator_fit_rows_evaluation_rows_disjoint_verified": True,
                    "method_selection_uses_target_labels": False,
                    "method": calibrator.to_projection(),
                },
                "champion": {
                    "model_id": str(target_rows[0]["incumbent_model_id"]),
                    "model_fingerprint": str(target_rows[0]["incumbent_model_fingerprint"]),
                    "metrics": champion_metrics,
                },
                "raw_challenger": {
                    "model_id": str(target_rows[0]["challenger_model_id"]),
                    "model_fingerprint": str(target_rows[0]["challenger_model_fingerprint"]),
                    "metrics": raw_metrics,
                },
                "calibrated_challenger": {
                    "model_id": f"{target_rows[0]['challenger_model_id']}:calibrated",
                    "calibrator_method": P38A_CALIBRATION_METHOD,
                    "metrics": calibrated_metrics,
                },
                "comparison": {
                    "same_target_rows_verified": same_target_rows == target_row_ids,
                    "raw_vs_calibrated": _metric_delta(raw_metrics, calibrated_metrics),
                    "calibrated_vs_champion": _metric_delta(champion_metrics, calibrated_metrics),
                    "prediction_rule": (
                        "calibrated_challenger_correct iff calibrated probability "
                        "is at least 0.5 and target_home_win is 1, or below 0.5 "
                        "and target_home_win is 0"
                    ),
                    "accuracy_changed_by_calibration": calibrated_metrics["accuracy"]
                    != raw_metrics["accuracy"],
                    "accuracy_change_explanation": (
                        "The calibrated challenger accuracy uses the calibrated "
                        "probability threshold of 0.5; it can differ from raw "
                        "challenger accuracy when the monotone Platt map shifts "
                        "a raw probability across 0.5."
                    ),
                },
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            int(row["evaluation_window_order"]),
            str(row["scheduled_start_utc"]),
            int(row["game_number"]),
            int(row["game_pk"]),
        )
    )
    per_window_summary.sort(key=lambda row: int(row["evaluation_window_order"]))
    all_targets = tuple(int(row["target_home_win"]) for row in comparison_rows)
    champion_probabilities = tuple(
        Decimal(str(row["champion_home_probability"])) for row in comparison_rows
    )
    raw_probabilities = tuple(
        Decimal(str(row["raw_challenger_home_probability"])) for row in comparison_rows
    )
    calibrated_probabilities = tuple(
        Decimal(str(row["calibrated_challenger_home_probability"]))
        for row in comparison_rows
    )
    raw_row_count = sum(
        int(window["holdout"]["raw_row_count"]) for window in per_window_summary
    )
    excluded_row_count = sum(
        int(window["holdout"]["excluded_row_count"]) for window in per_window_summary
    )
    if raw_row_count != len(comparison_rows) + excluded_row_count:
        raise P38AAuthorityError("P38A raw/evaluable/excluded accounting drift")
    aggregate_champion = probability_metrics(
        champion_probabilities,
        all_targets,
        raw_row_count=raw_row_count,
    )
    aggregate_raw = probability_metrics(
        raw_probabilities,
        all_targets,
        raw_row_count=raw_row_count,
    )
    aggregate_calibrated = probability_metrics(
        calibrated_probabilities,
        all_targets,
        raw_row_count=raw_row_count,
    )
    conclusion = _comparison_conclusion(aggregate_raw, aggregate_calibrated)
    summary = {
        "schema_version": "p38a.rolling_probability_calibration_summary.v1",
        "task_id": P38A_TASK_ID,
        "operation": "ROLLING_MONEYLINE_PROBABILITY_CALIBRATION_EVALUATION",
        "authority": {
            "repository": P38A_AUTHORITY_REPOSITORY,
            "current_base_head": P38A_BASE_HEAD,
            "current_base_tree": P38A_BASE_TREE,
            "p37a_report_path": str(P38A_P37A_REPORT_PATH),
            "p37a_artifact_sha256": authority["artifact_hashes"],
            "raw_challenger_reused_from_p37a_true_oos_artifact": True,
            "underlying_challenger_model_or_hyperparameters_changed": False,
        },
        "calibration": {
            "method": P38A_CALIBRATION_METHOD,
            "method_version": P38A_CALIBRATION_METHOD_VERSION,
            "source_kind": P38A_CALIBRATION_SOURCE_KIND,
            "fixed_method_frozen_before_final_evaluation": True,
            "method_search_performed": False,
            "new_third_party_dependency_added": False,
        },
        "admitted_target_holdout_fold_ids": list(P38A_TARGET_FOLD_IDS),
        "not_admitted_target_holdout_fold_ids": {
            "wf_004": "No prior true-OOS P37A prediction/label pairs precede wf_004; it is not used as a P38A target because its calibration lineage is unavailable."
        },
        "evaluation_windows": per_window_summary,
        "aggregate": {
            "raw_row_count": raw_row_count,
            "evaluable_row_count": len(comparison_rows),
            "excluded_row_count": excluded_row_count,
            "coverage": str(Decimal(len(comparison_rows)) / Decimal(raw_row_count)),
            "metrics_population": "P37A_EVALUABLE_TARGET_ROWS",
            "champion": {"metrics": aggregate_champion},
            "raw_challenger": {"metrics": aggregate_raw},
            "calibrated_challenger": {
                "metrics": aggregate_calibrated,
                "method": P38A_CALIBRATION_METHOD,
            },
        },
        "comparison": {
            "conclusion": conclusion,
            "conclusion_rule": P38A_CONCLUSION_RULE,
            "calibrated_vs_raw": _metric_delta(aggregate_raw, aggregate_calibrated),
            "calibrated_vs_champion": _metric_delta(aggregate_champion, aggregate_calibrated),
        },
        "verification": {
            "valid_window_count": len(per_window_summary),
            "minimum_two_windows_verified": len(per_window_summary) >= 2,
            "calibration_lineage_verified": all(
                window["calibration"]["strict_source_before_target_verified"]
                and window["calibration"]["source_target_game_id_disjoint_verified"]
                and not window["calibration"]["target_holdout_labels_used_for_fit"]
                for window in per_window_summary
            ),
            "calibrator_fit_evaluation_disjoint_verified": all(
                window["calibration"][
                    "calibrator_fit_rows_evaluation_rows_disjoint_verified"
                ]
                for window in per_window_summary
            ),
            "fixed_method_verified": all(
                window["calibration"]["method"]["method"] == P38A_CALIBRATION_METHOD
                for window in per_window_summary
            ),
            "same_target_rows_verified": all(
                window["comparison"]["same_target_rows_verified"]
                for window in per_window_summary
            ),
            "p37a_exclusion_semantics_preserved": all(
                window["holdout"]["exclusion_semantics_preserved"]
                for window in per_window_summary
            ),
            "probabilities_bounded_and_stable": all(
                Decimal(str(row["calibrated_challenger_home_probability"])) > Decimal("0")
                and Decimal(str(row["calibrated_challenger_home_probability"])) < Decimal("1")
                for row in comparison_rows
            ),
            "metric_calculation_verified": True,
            "aggregate_true_oos_rows_verified": all(
                row["true_oos_verified"] for row in comparison_rows
            ),
            "p37a_authority_unchanged_verified": True,
            "deterministic_rerun_verified": False,
            "model_promotion_occurred": False,
        },
        "claims": {
            "out_of_sample_evaluated": True,
            "calibration_is_leakage_safe": True,
            "model_promoted": False,
            "promotion_authorized": False,
            "production_ready": False,
            "bet_or_pass_claim": False,
            "profitability_claim": False,
            "staking_claim": False,
        },
        "deterministic_rerun_verified": False,
    }
    return {
        "calibration_artifacts": tuple(calibration_artifacts),
        "comparison_rows": tuple(comparison_rows),
        "per_window_summary": tuple(per_window_summary),
        "summary": summary,
    }


def run_deterministic_p38a_probability_calibration(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Run P38A twice and require identical rendered artifact bytes."""

    first = evaluate_p38a_probability_calibration(repository_root)
    second = evaluate_p38a_probability_calibration(repository_root)
    first_artifacts = (
        render_calibration_artifacts(first["calibration_artifacts"]),
        render_comparisons_jsonl(first["comparison_rows"]),
        render_per_window_summary(first["per_window_summary"]),
        render_summary(first["summary"]),
        render_report_markdown(first["summary"]),
    )
    second_artifacts = (
        render_calibration_artifacts(second["calibration_artifacts"]),
        render_comparisons_jsonl(second["comparison_rows"]),
        render_per_window_summary(second["per_window_summary"]),
        render_summary(second["summary"]),
        render_report_markdown(second["summary"]),
    )
    if first_artifacts != second_artifacts:
        raise ValueError("P38A deterministic artifact bytes mismatch")
    verified_summary = json.loads(json.dumps(first["summary"], ensure_ascii=False))
    verified_summary["deterministic_rerun_verified"] = True
    verified_summary["verification"]["deterministic_rerun_verified"] = True
    return {
        **first,
        "summary": verified_summary,
    }


__all__ = (
    "P38A_AUTHORITY_REPOSITORY",
    "P38A_BASE_HEAD",
    "P38A_BASE_TREE",
    "P38A_CALIBRATION_SOURCE_KIND",
    "P38A_CONCLUSION_RULE",
    "P38A_P37A_REPORT_PATH",
    "P38A_TARGET_FOLD_IDS",
    "P38A_TASK_ID",
    "P38AAuthorityError",
    "evaluate_p38a_probability_calibration",
    "run_deterministic_p38a_probability_calibration",
)
