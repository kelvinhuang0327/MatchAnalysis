"""UTC timestamp value object."""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class UtcTimestamp:
    """A timezone-aware datetime normalized to UTC."""

    value: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.value, datetime):
            raise TypeError("value must be a datetime")
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise ValueError("value must be timezone-aware")
        object.__setattr__(self, "value", self.value.astimezone(timezone.utc))

    def to_iso8601(self) -> str:
        """Serialize deterministically with an explicit UTC suffix."""

        return self.value.isoformat().replace("+00:00", "Z")

    def __str__(self) -> str:
        return self.to_iso8601()
