"""TRICERATOPS+ rewrite: astrophysical false-positive probability calculator."""
from __future__ import annotations

__version__ = "0.2.0.dev0"

from triceratops.config.config import Config, MissionConfig
from triceratops.domain.scenario_id import ScenarioID
from triceratops.validation import (
    ApertureConfig,
    FppRunConfig,
    FppRunResult,
    ValidationEngine,
    ValidationWorkspace,
    run_tess_fpp,
)

__all__ = [
    "__version__",
    "ValidationWorkspace",
    "ValidationEngine",
    "ApertureConfig",
    "FppRunConfig",
    "FppRunResult",
    "run_tess_fpp",
    "Config",
    "MissionConfig",
    "ScenarioID",
]
