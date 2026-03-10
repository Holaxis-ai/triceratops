"""Light-curve preparation and ephemeris types."""

# Core pipeline
from triceratops.lightcurve.convert import convert_folded_to_domain
from triceratops.lightcurve.fold import fold_and_clip
from triceratops.lightcurve.prep import prepare_from_raw

# Types
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, EphemerisResolver, ResolvedTarget
from triceratops.lightcurve.raw import RawLightCurveData, RawLightCurveSource
from triceratops.lightcurve.result import LightCurvePreparationResult

# Convenience orchestration
from triceratops.lightcurve.orchestration import (
    prepare_lightcurve_from_file,
    prepare_lightcurve_from_tic,
)

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
    # Core pipeline
    "prepare_from_raw",
    "fold_and_clip",
    "convert_folded_to_domain",
    # Types
    "Ephemeris",
    "ResolvedTarget",
    "EphemerisResolver",
    "RawLightCurveData",
    "RawLightCurveSource",
    "LightCurveConfig",
    "LightCurvePreparationResult",
    # Convenience orchestration
    "prepare_lightcurve_from_tic",
    "prepare_lightcurve_from_file",
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
