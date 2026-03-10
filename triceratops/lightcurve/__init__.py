"""Light-curve preparation and ephemeris types."""

# Core conversion
from triceratops.lightcurve.convert import convert_folded_to_domain

# Types
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, EphemerisResolver, ResolvedTarget
from triceratops.lightcurve.result import LightCurvePreparationResult

# Convenience orchestration
from triceratops.lightcurve.orchestration import (
    prepare_lightcurve_from_file,
    prepare_lightcurve_from_tic,
)

# Sources
from triceratops.lightcurve.sources.file import FileSource
from triceratops.lightcurve.sources.lightkurve import LightkurveSource

# ExoFOP
from triceratops.lightcurve.exofop.toi_resolution import ExoFopEphemerisResolver

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
    # Core conversion
    "convert_folded_to_domain",
    # Types
    "Ephemeris",
    "ResolvedTarget",
    "EphemerisResolver",
    "LightCurveConfig",
    "LightCurvePreparationResult",
    # Convenience orchestration
    "prepare_lightcurve_from_tic",
    "prepare_lightcurve_from_file",
    # Sources
    "LightkurveSource",
    "FileSource",
    # ExoFOP
    "ExoFopEphemerisResolver",
    # Errors
    "LightCurveError",
    "LightCurveNotFoundError",
    "SectorNotAvailableError",
    "EphemerisRequiredError",
    "DownloadTimeoutError",
    "LightCurveEmptyError",
    "LightCurvePreparationError",
]
