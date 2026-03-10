"""Raw light-curve data and source protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from triceratops.lightcurve.config import LightCurveConfig

_VALID_CADENCES = ("20sec", "2min", "10min", "30min")


@dataclass(frozen=True)
class RawLightCurveData:
    """Validated, normalised raw photometry — not yet phase-folded.

    All RawLightCurveSource implementations must return flux normalised such
    that the out-of-transit continuum has median ~ 1.0 per sector.
    """

    time_btjd: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    sectors: tuple[int, ...]
    cadence: str
    exptime_seconds: float
    target_id: int | None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # --- shape invariants ---
        if self.time_btjd.ndim != 1:
            raise ValueError("time_btjd must be 1-D")
        if self.flux.ndim != 1 or self.flux_err.ndim != 1:
            raise ValueError("flux and flux_err must be 1-D")
        if not (len(self.time_btjd) == len(self.flux) == len(self.flux_err)):
            raise ValueError("time_btjd, flux, flux_err must have equal length")
        if len(self.time_btjd) == 0:
            raise ValueError("RawLightCurveData arrays must be non-empty")

        # --- value invariants ---
        if not np.all(np.isfinite(self.time_btjd)):
            raise ValueError("time_btjd contains non-finite values")
        if not np.all(np.isfinite(self.flux)):
            raise ValueError("flux contains non-finite values")
        if not np.all(np.isfinite(self.flux_err)):
            raise ValueError("flux_err contains non-finite values")
        if not np.all(np.diff(self.time_btjd) > 0):
            raise ValueError("time_btjd must be strictly monotonically increasing")

        # --- metadata invariants ---
        if not self.sectors:
            raise ValueError("sectors must be non-empty")
        if self.cadence not in _VALID_CADENCES:
            raise ValueError(f"unrecognised cadence: {self.cadence!r}")
        if self.exptime_seconds <= 0:
            raise ValueError("exptime_seconds must be positive")


@runtime_checkable
class RawLightCurveSource(Protocol):
    """Protocol for acquiring raw photometry."""

    def fetch_raw(self, config: LightCurveConfig) -> RawLightCurveData: ...
