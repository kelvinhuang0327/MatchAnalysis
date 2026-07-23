"""Read-only adapters for pinned Betting-pool artifacts."""

from .p83e_jsonl import (
    P83eJsonlSnapshotSource,
    P83eSnapshotValidationError,
)

__all__ = [
    "P83eJsonlSnapshotSource",
    "P83eSnapshotValidationError",
]
