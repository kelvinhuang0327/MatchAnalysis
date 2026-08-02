"""Canonical UTC timestamp parsing for prospective prediction admission evidence."""

from datetime import datetime, timezone


def parse_canonical_utc(value: str) -> datetime:
    """Parse a strict, timezone-aware ISO-8601 string and normalize to UTC.

    Only an explicit, non-empty ISO-8601 string carrying its own UTC offset
    (``Z`` or an explicit ``+HH:MM``/``-HH:MM``) is accepted. Nothing is ever
    inferred: no silent trimming, no naive-to-UTC assumption, no current time.
    """

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not value:
        raise ValueError("value must be a non-empty ISO-8601 string")
    if value != value.strip():
        raise ValueError("value must not contain surrounding whitespace")

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("value must be a valid ISO-8601 timestamp") from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("value must be timezone-aware")

    return parsed.astimezone(timezone.utc)


def format_canonical_utc(value: datetime) -> str:
    """Serialize a timezone-aware datetime as deterministic canonical Z-form."""

    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("value must be timezone-aware")

    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")
