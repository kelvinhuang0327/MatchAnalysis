"""Canonical match identity."""

from dataclasses import dataclass


def _require_clean_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True, slots=True)
class MatchIdentity:
    """Explicit identity that is never inferred from a date and participants."""

    sport: str
    league: str
    season: int
    canonical_game_id: str
    home_participant: str
    away_participant: str
    game_discriminator: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "sport",
            "league",
            "canonical_game_id",
            "home_participant",
            "away_participant",
        ):
            _require_clean_non_empty(getattr(self, field_name), field_name)

        if isinstance(self.season, bool) or not isinstance(self.season, int):
            raise TypeError("season must be an integer")
        if self.season <= 0:
            raise ValueError("season must be positive")

        if self.home_participant == self.away_participant:
            raise ValueError("home and away participants must differ")

        if self.game_discriminator is not None:
            _require_clean_non_empty(
                self.game_discriminator,
                "game_discriminator",
            )
