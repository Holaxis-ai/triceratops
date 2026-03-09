"""External (ground-based) light curve file loading.

The file format is a whitespace-delimited text file with columns::

    time(days)  flux  flux_err

(or two columns: time, flux, with flux_err computed from scatter).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from triceratops.domain.entities import ExternalLightCurve, LightCurve


def load_external_lc(
    path: Path,
    band: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an external light curve file.

    Args:
        path: Path to the light curve file.
        band: Filter band label for the ExternalLightCurve.

    Returns:
        (time, flux, flux_err): Three arrays of equal length.
        flux_err is taken from column 3 if present, otherwise np.std(flux).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file has fewer than 2 columns.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"External LC file not found: {path}")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"LC file must have at least 2 columns; got {data.shape[1]} in {path}")
    time = data[:, 0]
    flux = data[:, 1]
    flux_err = data[:, 2] if data.shape[1] >= 3 else np.full(len(flux), np.std(flux))
    return time, flux, flux_err


def load_external_lc_as_object(
    path: Path,
    band: str,
) -> ExternalLightCurve:
    """Load an external light curve and wrap it in an ExternalLightCurve object.

    LDC fields are left as None; they are resolved later by BaseScenario.
    """
    time, flux, flux_err = load_external_lc(path, band)
    lc = LightCurve(
        time_days=time,
        flux=flux,
        flux_err=float(np.mean(flux_err)),
        cadence_days=float(np.min(np.diff(time))) if len(time) > 1 else 0.00139,
    )
    return ExternalLightCurve(light_curve=lc, band=band, ldc=None)
