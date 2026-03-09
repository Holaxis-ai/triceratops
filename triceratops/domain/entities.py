"""Domain entities: mutable or composite data types representing domain concepts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .value_objects import LimbDarkeningCoeffs, StellarParameters


@dataclass
class Star:
    """One catalogued star in the photometric field."""

    tic_id: int
    ra_deg: float
    dec_deg: float
    tmag: float
    jmag: float
    hmag: float
    kmag: float
    bmag: float
    vmag: float
    gmag: float | None = None
    rmag: float | None = None
    imag: float | None = None
    zmag: float | None = None
    stellar_params: StellarParameters | None = None
    separation_arcsec: float = 0.0
    position_angle_deg: float = 0.0
    # Computed by flux_contributions module (P1-022), not set at construction:
    flux_ratio: float | None = None
    transit_depth_required: float | None = None

    def mag_for_band(self, band: str) -> float | None:
        """Return the magnitude for the given filter band, or None if unavailable.

        Args:
            band: One of "TESS", "J", "H", "K", "B", "V", "g", "r", "i", "z".
        """
        _map = {
            "TESS": self.tmag, "J": self.jmag, "H": self.hmag, "K": self.kmag,
            "B": self.bmag, "V": self.vmag, "g": self.gmag,
            "r": self.rmag, "i": self.imag, "z": self.zmag,
        }
        return _map.get(band)


@dataclass
class StellarField:
    """All stars in the photometric search aperture, target at index 0."""

    target_id: int
    mission: str
    search_radius_pixels: int
    stars: list[Star]          # stars[0] is always the target

    @property
    def target(self) -> Star:
        return self.stars[0]

    @property
    def neighbors(self) -> list[Star]:
        return self.stars[1:]

    def stars_with_flux_data(self) -> list[Star]:
        """Return stars that have a non-None, positive transit_depth_required."""
        return [
            s for s in self.stars
            if s.transit_depth_required is not None and s.transit_depth_required > 0
        ]


@dataclass
class LightCurve:
    """A phase-folded, normalised photometric time series ready for model fitting."""

    time_days: np.ndarray        # days from transit midpoint; t=0 at centre
    flux: np.ndarray             # normalised flux; 1.0 = out of transit
    flux_err: float              # scalar per-point uncertainty (sigma)
    cadence_days: float = 0.00139   # exposure time; 0.00139 ~ 2-min TESS cadence
    supersampling_rate: int = 20    # pytransit integration supersampling

    @property
    def sigma(self) -> float:
        return self.flux_err

    def with_renorm(self, flux_ratio: float) -> LightCurve:
        """Return a new LightCurve renormalised to a single star's contribution.

        This is the vectorised equivalent of renorm_flux() from funcs.py:225-238.
        flux_ratio is the fraction of aperture flux from the host star (0 < fr <= 1).
        """
        renormed = (self.flux - (1.0 - flux_ratio)) / flux_ratio
        renormed_err = self.flux_err / flux_ratio
        return LightCurve(
            time_days=self.time_days,
            flux=renormed,
            flux_err=renormed_err,
            cadence_days=self.cadence_days,
            supersampling_rate=self.supersampling_rate,
        )


@dataclass
class ExternalLightCurve:
    """A ground-based follow-up observation in a specific photometric band."""

    light_curve: LightCurve
    band: str                                   # "J", "H", "K", "g", "r", "i", "z"
    ldc: LimbDarkeningCoeffs | None = None  # resolved by BaseScenario._resolve_external_lc_ldcs
