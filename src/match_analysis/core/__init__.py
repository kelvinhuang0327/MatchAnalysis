"""Stable contracts shared by MatchAnalysis domains."""

from .identity import MatchIdentity
from .provenance import ArtifactProvenance
from .time import UtcTimestamp

__all__ = ["ArtifactProvenance", "MatchIdentity", "UtcTimestamp"]
