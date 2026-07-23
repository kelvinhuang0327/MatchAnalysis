"""Baseball game entity."""

from dataclasses import dataclass

from ...core.identity import MatchIdentity
from ...core.time import UtcTimestamp


@dataclass(frozen=True, slots=True)
class BaseballGame:
    """A scheduled baseball match without outcomes or provider behavior."""

    identity: MatchIdentity
    scheduled_start: UtcTimestamp

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MatchIdentity):
            raise TypeError("identity must be a MatchIdentity")
        if not isinstance(self.scheduled_start, UtcTimestamp):
            raise TypeError("scheduled_start must be a UtcTimestamp")
        if self.identity.sport != "baseball":
            raise ValueError("baseball games require sport == 'baseball'")
