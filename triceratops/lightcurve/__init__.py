"""Core light-curve domain types, config, and errors."""

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, EphemerisResolver, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult

# Errors
from triceratops.lightcurve.errors import (
    DownloadTimeoutError,
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurveError,
    LightCurveNotFoundError,
    LightCurvePreparationError,
    SectorNotAvailableError,
)

__all__ = [
    # Types
    "Ephemeris",
    "ResolvedTarget",
    "EphemerisResolver",
    "LightCurveConfig",
    "LightCurvePreparationResult",
    # Errors
    "LightCurveError",
    "LightCurveNotFoundError",
    "SectorNotAvailableError",
    "EphemerisRequiredError",
    "DownloadTimeoutError",
    "LightCurveEmptyError",
    "LightCurvePreparationError",
]
