"""Read-only observations consumed by information-edge capture."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Protocol


class StartingPitcherSignal(Protocol):
    """Exact official schedule identity and the received pitcher state."""

    @property
    def provider_namespace(self) -> str: ...

    @property
    def provider_game_id(self) -> str: ...

    @property
    def official_date(self) -> str: ...

    @property
    def game_type(self) -> str: ...

    @property
    def game_number(self) -> int: ...

    @property
    def scheduled_start_utc(self) -> str: ...

    @property
    def home_team(self) -> Mapping[str, Any]: ...

    @property
    def away_team(self) -> Mapping[str, Any]: ...

    @property
    def signal_observed_at_utc(self) -> str: ...

    def schedule_identity(self) -> dict[str, Any]: ...

    def probable_pitcher_state(self) -> dict[str, str | None]: ...


class MoneylineSourceObservation(Protocol):
    """Normalized quote fields required for exact game and timestamp joins."""

    @property
    def row(self) -> Mapping[str, Any]: ...

    @property
    def market_code(self) -> str: ...

    @property
    def game_time(self) -> str: ...

    @property
    def fetched_at(self) -> str: ...

    @property
    def source_row_fingerprint(self) -> str: ...

    @property
    def home_decimal_odds(self) -> Decimal: ...

    @property
    def away_decimal_odds(self) -> Decimal: ...

