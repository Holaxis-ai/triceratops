"""Light-curve preparation result type."""
from __future__ import annotations

from dataclasses import dataclass, field

from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.ephemeris import Ephemeris


@dataclass(frozen=True)
class LightCurvePreparationResult:
    """Output of prepare_from_raw(): a compute-ready LightCurve plus metadata."""

    light_curve: LightCurve
    ephemeris: Ephemeris
    sectors_used: tuple[int, ...]
    cadence_used: str
    warnings: list[str] = field(default_factory=list)
