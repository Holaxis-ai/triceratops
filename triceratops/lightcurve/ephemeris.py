"""Ephemeris and resolved-target types for light-curve preparation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ephemeris:
    """Transit ephemeris parameters."""

    period_days: float
    t0_btjd: float
    duration_hours: float | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    """A TIC target resolved from user input."""

    target_ref: str
    tic_id: int
    ephemeris: Ephemeris | None = None
    source: str = "unknown"
