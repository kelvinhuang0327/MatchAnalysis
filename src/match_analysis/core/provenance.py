"""Structural provenance for generated artifacts."""

from dataclasses import dataclass
import re


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_explicit(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be explicit and non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Evidence needed to identify an artifact and its inputs."""

    schema_version: str
    source_repository: str
    source_commit: str
    producer_id: str
    producer_version: str
    input_fingerprint: str
    content_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "schema_version",
            "source_repository",
            "source_commit",
            "producer_id",
            "producer_version",
            "input_fingerprint",
            "content_fingerprint",
        ):
            _require_explicit(getattr(self, field_name), field_name)

        for field_name in ("input_fingerprint", "content_fingerprint"):
            if _SHA256_PATTERN.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(
                    f"{field_name} must be a lowercase 64-character SHA-256"
                )
