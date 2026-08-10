"""CLI for deterministic paper-only Moneyline inference."""

import argparse
from pathlib import Path
import sys

from ...application.use_cases.generate_moneyline_predictions import (
    generate_moneyline_predictions,
)
from ...application.use_cases.moneyline_inference_artifacts import (
    load_moneyline_feature_snapshots,
    load_moneyline_model_artifact,
    write_moneyline_inference_artifacts,
)


DEFAULT_PAPER_MONEYLINE_MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "report"
    / "p22b_moneyline_challenger"
    / "model_artifact.json"
)


def main(argv: list[str] | None = None) -> int:
    """Run local Moneyline inference without provider, database, or runtime state."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic pregame Moneyline predictions."
    )
    parser.add_argument("--feature-snapshots", required=True, type=Path)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_PAPER_MONEYLINE_MODEL_ARTIFACT_PATH,
        help=(
            "Paper-only model artifact path; defaults to the frozen P22B "
            "challenger artifact."
        ),
    )
    parser.add_argument("--prediction-generated-at-utc", required=True)
    parser.add_argument("--response-received-at-utc", required=True)
    parser.add_argument("--ingested-at-utc", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    for path, label in (
        (args.feature_snapshots, "Feature snapshots"),
        (args.model_artifact, "Model artifact"),
    ):
        if not path.is_file():
            print(f"ERROR: {label} file does not exist: {path}", file=sys.stderr)
            return 1

    try:
        snapshots = load_moneyline_feature_snapshots(args.feature_snapshots)
        model_artifact = load_moneyline_model_artifact(args.model_artifact)
        result = generate_moneyline_predictions(
            snapshots,
            model_artifact,
            prediction_generated_at_utc=args.prediction_generated_at_utc,
            response_received_at_utc=args.response_received_at_utc,
            ingested_at_utc=args.ingested_at_utc,
        )
        write_moneyline_inference_artifacts(
            args.output_dir,
            result,
            model_artifact,
        )
    except (OSError, TypeError, ValueError, UnicodeDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Generated {len(result.candidates)} deterministic Moneyline candidates. "
        f"Fingerprint: {result.candidate_set_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
