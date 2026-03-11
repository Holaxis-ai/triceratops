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
from triceratops.lightcurve.exofop.toi_resolution import (
    ExoFopEphemerisResolver,
    LookupStatus,
    ToiResolutionResult,
    resolve_toi_to_tic_ephemeris_depth,
)

# Public lightkurve-backed prep helpers used by higher-level orchestration layers
from triceratops.lightcurve.sources.lightkurve import (
    fold_lightcurve,
    process_lightcurve_collection,
    resolve_cadence_label,
    trim_folded_lightcurve,
)

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
    "LookupStatus",
    "ToiResolutionResult",
    "resolve_toi_to_tic_ephemeris_depth",
    # Public prep helpers
    "process_lightcurve_collection",
    "fold_lightcurve",
    "trim_folded_lightcurve",
    "resolve_cadence_label",
    # Errors
    "LightCurveError",
    "LightCurveNotFoundError",
    "SectorNotAvailableError",
    "EphemerisRequiredError",
    "DownloadTimeoutError",
    "LightCurveEmptyError",
    "LightCurvePreparationError",
]
