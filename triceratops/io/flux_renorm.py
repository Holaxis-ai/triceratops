"""Flux renormalisation helpers.

Replaces funcs.renorm_flux() (funcs.py:225-238).
"""
from __future__ import annotations

import numpy as np


def renorm_flux(
    flux: np.ndarray,
    flux_err: np.ndarray | float,
    star_flux_ratio: float,
) -> tuple[np.ndarray, np.ndarray | float]:
    """Renormalise flux to a single star's contribution.

    When a light curve contains flux from multiple stars in the aperture,
    this function extracts the flux contribution of one star.

    Equivalent to LightCurve.with_renorm() but operates on raw arrays.
    Source: funcs.py:225-238

    flux_renormed = (flux - (1 - flux_ratio)) / flux_ratio
    flux_err_renormed = flux_err / flux_ratio

    Args:
        flux: Normalised flux array.
        flux_err: Per-point uncertainty (array or scalar).
        star_flux_ratio: Fraction of total aperture flux from the star of interest.

    Returns:
        (flux_renormed, flux_err_renormed): Renormalised arrays.

    Raises:
        ValueError: If star_flux_ratio <= 0 or > 1.
    """
    if star_flux_ratio <= 0 or star_flux_ratio > 1:
        raise ValueError(f"star_flux_ratio must be in (0, 1], got {star_flux_ratio}")
    flux_renormed = (flux - (1.0 - star_flux_ratio)) / star_flux_ratio
    flux_err_renormed = flux_err / star_flux_ratio
    return flux_renormed, flux_err_renormed


class FluxRenormalizer:
    """Stateless helper class for flux renormalisation.

    Wraps renorm_flux() as a class for injection into scenario kernels
    that prefer object-based interfaces.
    """

    def renormalize(
        self,
        flux: np.ndarray,
        flux_err: np.ndarray | float,
        star_flux_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray | float]:
        """Delegate to renorm_flux()."""
        return renorm_flux(flux, flux_err, star_flux_ratio)
