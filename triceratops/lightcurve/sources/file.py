"""FileSource — load a pre-folded light curve from disk."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from triceratops.lightcurve.convert import convert_folded_to_domain
from triceratops.lightcurve.errors import LightCurveEmptyError, LightCurvePreparationError

if TYPE_CHECKING:
    from triceratops.domain.entities import LightCurve
    from triceratops.lightcurve.config import LightCurveConfig


class FileSource:
    """Load a pre-folded, already-prepared light curve from disk.

    The file must be already phase-folded with transit at phase=0.
    Supports FITS (via lightkurve) and plain text (phase_days, flux, flux_err columns).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, config: LightCurveConfig | None = None) -> LightCurve:
        """Load and convert to domain LightCurve.

        FITS files are read via lightkurve and must have a .phase attribute.
        Plain-text files must have columns: phase_days, flux[, flux_err].
        """
        from triceratops.lightcurve.config import LightCurveConfig as _Config

        config = config or _Config()

        if not self.path.exists():
            from triceratops.lightcurve.errors import LightCurveNotFoundError
            raise LightCurveNotFoundError(f"File not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix in (".fits", ".fit"):
            return self._load_fits(config)
        return self._load_text(config)

    def _load_fits(self, config: LightCurveConfig) -> LightCurve:
        import lightkurve as lk

        lk_lc = lk.io.read(str(self.path))
        if not hasattr(lk_lc, "phase"):
            raise LightCurvePreparationError(
                f"{self.path}: FITS file does not have a .phase attribute. "
                "File must be a pre-folded lightkurve FoldedLightCurve."
            )
        return convert_folded_to_domain(lk_lc, config=config)

    def _load_text(self, config: LightCurveConfig) -> LightCurve:
        """Load plain-text file with columns: phase_days, flux[, flux_err]."""
        from triceratops.domain.entities import LightCurve

        try:
            data = np.loadtxt(self.path, comments="#", delimiter=None)
        except Exception as exc:
            raise LightCurvePreparationError(
                f"{self.path}: failed to parse as plain text: {exc}"
            ) from exc

        if data.ndim != 2 or data.shape[1] < 2:
            raise LightCurvePreparationError(
                f"{self.path}: expected at least 2 columns (phase_days, flux)"
            )

        phase_days = data[:, 0].astype(np.float64)
        flux = data[:, 1].astype(np.float64)
        flux_err_arr = (
            data[:, 2].astype(np.float64)
            if data.shape[1] > 2
            else np.full_like(flux, float(np.std(flux)) or 1e-4)
        )

        finite = np.isfinite(phase_days) & np.isfinite(flux) & np.isfinite(flux_err_arr)
        phase_days, flux, flux_err_arr = phase_days[finite], flux[finite], flux_err_arr[finite]

        if len(phase_days) == 0:
            raise LightCurveEmptyError(f"{self.path}: no finite cadences after NaN sweep")

        flux_err_scalar = float(np.mean(flux_err_arr))
        if not (np.isfinite(flux_err_scalar) and flux_err_scalar > 0):
            raise LightCurvePreparationError(
                f"{self.path}: flux_err collapsed to non-positive scalar"
            )

        cadence_days = config.cadence_days_override or (
            {"20sec": 20, "2min": 120, "10min": 600, "30min": 1800}.get(config.cadence, 120)
            / 86400.0
        )

        return LightCurve(
            time_days=phase_days,
            flux=flux,
            flux_err=flux_err_scalar,
            cadence_days=cadence_days,
            supersampling_rate=config.supersampling_rate,
        )
