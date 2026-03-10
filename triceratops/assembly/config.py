"""Assembly configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import AssemblyConfigError

if TYPE_CHECKING:
    from triceratops.lightcurve.config import LightCurveConfig

_VALID_MISSIONS = ("TESS", "Kepler", "K2")


@dataclass(frozen=True)
class AssemblyConfig:
    """Configuration for the data-assembly pipeline."""

    include_light_curve: bool = True
    include_trilegal: bool = True
    include_contrast_curve: bool = True
    include_molusc: bool = True
    include_external_lcs: bool = True

    lc_config: LightCurveConfig | None = None
    catalog_search_radius_px: int = 10
    mission: str = "TESS"
    contrast_curve_band: str = "TESS"
    trilegal_cache_path: str | None = None
    require_light_curve: bool = True
    require_stellar_params: bool = True

    def __post_init__(self) -> None:
        if self.catalog_search_radius_px < 1:
            raise AssemblyConfigError(
                f"catalog_search_radius_px must be >= 1, got {self.catalog_search_radius_px}"
            )
        if self.mission not in _VALID_MISSIONS:
            raise AssemblyConfigError(
                f"Unknown mission {self.mission!r}; expected one of {_VALID_MISSIONS}"
            )
