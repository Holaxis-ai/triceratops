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
    StellarParameters,
)

__all__: list[str] = [
    "ContrastCurve",
    "ExternalLightCurve",
    "LightCurve",
    "LimbDarkeningCoeffs",
    "OrbitalParameters",
    "ScenarioID",
    "ScenarioResult",
    "Star",
    "StellarField",
    "StellarParameters",
    "ValidationResult",
]
