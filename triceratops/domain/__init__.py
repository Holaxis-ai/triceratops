"""Domain model: pure dataclasses, no I/O, no external dependencies."""
from .entities import (
    ExternalLightCurve,
    LightCurve,
    Star,
    StellarField,
)
from .result import ScenarioResult, ValidationResult
from .scenario_id import ScenarioID
from .value_objects import (
    ContrastCurve,
    LimbDarkeningCoeffs,
    OrbitalParameters,
    PeriodSpec,
    StellarParameters,
)

from triceratops.lightcurve.ephemeris import Ephemeris, EphemerisResolver, ResolvedTarget

__all__: list[str] = [
    "ContrastCurve",
    "Ephemeris",
    "EphemerisResolver",
    "ExternalLightCurve",
    "LightCurve",
    "LimbDarkeningCoeffs",
    "OrbitalParameters",
    "PeriodSpec",
    "ResolvedTarget",
    "ScenarioID",
    "ScenarioResult",
    "Star",
    "StellarField",
    "StellarParameters",
    "ValidationResult",
]
