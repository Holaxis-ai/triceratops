"""TRICERATOPS+ rewrite: astrophysical false-positive probability calculator."""
from __future__ import annotations

__version__ = "0.2.0.dev0"

from triceratops.config.config import Config, MissionConfig
from triceratops.domain.scenario_id import ScenarioID
from triceratops.plotting import plot_field, plot_fits, plot_fits_joint, plot_fits_palomar
from triceratops.validation import (
    PreparedValidationInputs,
    PreparedValidationMetadata,
    ValidationEngine,
    ValidationPreparer,
    ValidationWorkspace,
    probs_dataframe,
)

__all__ = [
    "__version__",
    "ValidationWorkspace",
    "ValidationEngine",
    "ValidationPreparer",
    "PreparedValidationInputs",
    "PreparedValidationMetadata",
    "Config",
    "MissionConfig",
    "ScenarioID",
    "plot_field",
    "plot_fits",
    "plot_fits_palomar",
    "plot_fits_joint",
    "probs_dataframe",
]
