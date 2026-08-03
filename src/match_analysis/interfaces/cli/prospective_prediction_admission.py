"""CLI interface for prospective prediction admission workflow."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys

from ...baseball.domain.canonical_utc import parse_canonical_utc
from ...baseball.domain.match_identity_authority import (
    MatchIdentityAuthorityEntry,
    build_match_identity_authority_catalog,
)
from ...baseball.domain.participant_identity_resolution import (
    ProviderParticipantIdentityMapping,
)
from ...baseball.domain.prediction_admission import (
    ProspectivePredictionCandidate,
)
from ...infrastructure.mlb_schedule import ExplicitMlbSchedulePayloadSource
from ...application.use_cases.prospective_prediction_admission_artifacts import (
    write_prospective_prediction_admission_artifacts,
)
from ...application.use_cases.run_prospective_prediction_admission_workflow import (
    run_prospective_prediction_admission_workflow,
)


def load_prediction_requests(path: Path) -> tuple[ProspectivePredictionCandidate, ...]:
    """Parse a jsonl file of prospective prediction candidates."""
    candidates = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        candidate = ProspectivePredictionCandidate(
            prediction_observation_id=str(obj["prediction_observation_id"]),
            source_prediction_id=str(obj["source_prediction_id"]),
            model_id=str(obj["model_id"]),
            market_id=str(obj["market_id"]),
            selection=str(obj["selection"]),
            model_probability=Decimal(str(obj["model_probability"])),
            line_value=Decimal(str(obj["line_value"])),
            push_policy=str(obj["push_policy"]),
            provider_namespace=str(obj["provider_namespace"]),
            provider_game_id=str(obj["provider_game_id"]),
            game_number=int(obj["game_number"]),
            source_schedule_observation_id=str(obj["source_schedule_observation_id"]),
            prediction_generated_at_utc=str(obj["prediction_generated_at_utc"]),
            response_received_at_utc=str(obj["response_received_at_utc"]),
            ingested_at_utc=str(obj["ingested_at_utc"]),
        )
        candidates.append(candidate)
    return tuple(candidates)


def load_raw_schedule_sources(path: Path) -> tuple[ExplicitMlbSchedulePayloadSource, ...]:
    """Parse a jsonl file of raw schedule payload sources."""
    sources = []
    text = path.read_text(encoding="utf-8")
    metadata_keys = {
        "response_received_at_utc",
        "ingested_at_utc",
        "endpoint_id",
        "parser_version",
        "schema_version",
        "supersedes_observation_id",
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "payload" in obj and isinstance(obj["payload"], dict):
            raw_bytes = json.dumps(obj["payload"], separators=(",", ":")).encode("utf-8")
            meta = obj
        else:
            raw_dict = {k: v for k, v in obj.items() if k not in metadata_keys}
            raw_bytes = json.dumps(raw_dict, separators=(",", ":")).encode("utf-8")
            meta = obj

        resp_dt = parse_canonical_utc(meta["response_received_at_utc"])
        ing_dt = parse_canonical_utc(meta["ingested_at_utc"])
        endpoint_id = meta.get("endpoint_id", "mlb_schedule_v1")
        parser_version = meta.get("parser_version", "matchanalysis_mlb_game_payload_parser_v1")
        schema_version = meta.get("schema_version", "mlb_schedule_api_game_payload_v1")
        supersedes_id = meta.get("supersedes_observation_id")

        source = ExplicitMlbSchedulePayloadSource(
            raw_payload_bytes=raw_bytes,
            response_received_at_utc=resp_dt,
            ingested_at_utc=ing_dt,
            endpoint_id=endpoint_id,
            parser_version=parser_version,
            schema_version=schema_version,
            supersedes_observation_id=supersedes_id,
        )
        sources.append(source)
    return tuple(sources)


def load_participant_mapping_catalog(path: Path) -> tuple[ProviderParticipantIdentityMapping, ...]:
    """Parse a json file of provider participant identity mappings."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("participant mapping catalog root must be a JSON array")
    mappings = [
        ProviderParticipantIdentityMapping(
            provider_namespace=str(item["provider_namespace"]),
            provider_participant_id=str(item["provider_participant_id"]),
            canonical_participant_id=str(item["canonical_participant_id"]),
            mapping_version=str(item["mapping_version"]),
        )
        for item in data
    ]
    return tuple(mappings)


def load_authority_catalog(path: Path):
    """Parse a json file of match identity authority catalog entries."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("authority catalog root must be a JSON array")
    entries = [
        MatchIdentityAuthorityEntry(
            provider_namespace=str(item["provider_namespace"]),
            provider_game_id=str(item["provider_game_id"]),
            game_number=int(item["game_number"]),
            league=str(item["league"]),
            season=int(item["season"]),
            canonical_game_id=str(item["canonical_game_id"]),
            game_discriminator=item.get("game_discriminator"),
            authority_version=str(item["authority_version"]),
        )
        for item in data
    ]
    return build_match_identity_authority_catalog(tuple(entries))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for prospective prediction admission workflow."""
    parser = argparse.ArgumentParser(
        description="Run prospective prediction admission workflow over real schedule pipeline."
    )
    parser.add_argument("--prediction-requests", required=True, type=Path)
    parser.add_argument("--raw-schedule-payloads", required=True, type=Path)
    parser.add_argument("--participant-mapping-catalog", required=True, type=Path)
    parser.add_argument("--authority-catalog", required=True, type=Path)
    parser.add_argument("--schedule-as-of-utc", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args(argv)

    # Calculate input file SHA-256 hashes
    input_hashes = {
        "prediction_requests_sha256": sha256(args.prediction_requests.read_bytes()).hexdigest(),
        "raw_schedule_payloads_sha256": sha256(args.raw_schedule_payloads.read_bytes()).hexdigest(),
        "participant_mapping_catalog_sha256": sha256(args.participant_mapping_catalog.read_bytes()).hexdigest(),
        "authority_catalog_sha256": sha256(args.authority_catalog.read_bytes()).hexdigest(),
    }

    requests = load_prediction_requests(args.prediction_requests)
    sources = load_raw_schedule_sources(args.raw_schedule_payloads)
    mappings = load_participant_mapping_catalog(args.participant_mapping_catalog)
    authority_catalog = load_authority_catalog(args.authority_catalog)
    as_of_dt = parse_canonical_utc(args.schedule_as_of_utc)

    workflow_result = run_prospective_prediction_admission_workflow(
        requests=requests,
        raw_schedule_sources=sources,
        participant_mappings=mappings,
        authority_catalog=authority_catalog,
        schedule_as_of_utc=as_of_dt,
    )

    write_prospective_prediction_admission_artifacts(
        args.output_dir,
        workflow_result,
        input_hashes,
    )

    print(
        f"Workflow completed: {workflow_result.admitted_count} admitted, "
        f"{workflow_result.rejected_count} rejected. "
        f"Fingerprint: {workflow_result.result_set_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
