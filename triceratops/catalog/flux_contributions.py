"""PSF flux ratio and transit depth computation.

Replaces the dblquad(Gauss2D) logic in triceratops.calc_depths().
Pure functions -- no network I/O, no file I/O.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad

from triceratops.domain.entities import StellarField


def gauss2d(
    y: float,
    x: float,
    x0: float,
    y0: float,
    sigma: float,
    amplitude: float = 1.0,
) -> float:
    """2D circular Gaussian PSF evaluated at (x, y).

    Source: funcs.Gauss2D and triceratops.py calc_depths().

    Args:
        y: Y-coordinate (first arg for dblquad compatibility).
        x: X-coordinate.
        x0, y0: Centre of the PSF.
        sigma: Standard deviation of the Gaussian in pixels.
        amplitude: Total integrated flux (area under PSF).

    Returns:
        PSF value at (x, y).
    """
    exponent = ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    return amplitude / (2 * np.pi * sigma ** 2) * np.exp(-exponent)


def compute_flux_ratios(
    field: StellarField,
    pixel_coords_per_sector: list[np.ndarray],
    aperture_pixels_per_sector: list[np.ndarray],
    sigma_psf_px: float = 0.75,
) -> list[float]:
    """Compute the fraction of aperture flux contributed by each star.

    Ports the PSF integration loop from triceratops.calc_depths() (lines 542-581).

    Args:
        field: StellarField with all stars.
        pixel_coords_per_sector: List of arrays, each shape (N_stars, 2),
            giving (col, row) pixel positions for each star in each sector.
        aperture_pixels_per_sector: List of arrays, each shape (N_pixels, 2),
            giving (col, row) of each aperture pixel per sector.
        sigma_psf_px: PSF sigma in pixels (original default: 0.75).

    Returns:
        List of flux ratios (one per star). Sum ~= 1.0.
    """
    n_sectors = len(pixel_coords_per_sector)
    n_stars = len(field.stars)

    # Relative brightness of each star normalised to brightest
    tmags = np.array([s.tmag for s in field.stars])
    min_tmag = np.min(tmags)
    brightness = 10.0 ** ((min_tmag - tmags) / 2.5)

    flux_ratio_per_sector = np.zeros((n_sectors, n_stars))

    for k in range(n_sectors):
        rel_flux = np.zeros(n_stars)
        for i in range(n_stars):
            mu_x = pixel_coords_per_sector[k][i, 0]
            mu_y = pixel_coords_per_sector[k][i, 1]
            a = brightness[i]
            total = 0.0
            for j in range(len(aperture_pixels_per_sector[k])):
                px = aperture_pixels_per_sector[k][j]
                total += dblquad(
                    gauss2d,
                    px[1] - 0.5,
                    px[1] + 0.5,
                    px[0] - 0.5,
                    px[0] + 0.5,
                    args=(mu_x, mu_y, sigma_psf_px, a),
                )[0]
            rel_flux[i] = total
        total_flux = np.sum(rel_flux)
        if total_flux > 0:
            flux_ratio_per_sector[k, :] = rel_flux / total_flux

    # Average across sectors
    avg_ratios = np.mean(flux_ratio_per_sector, axis=0)
    return avg_ratios.tolist()


def compute_transit_depths(
    flux_ratios: list[float],
    observed_transit_depth: float,
) -> list[float]:
    """Compute the intrinsic transit depth required if each star is the host.

    Ports the depth scaling from triceratops.calc_depths() (lines 584-588).
    The formula: tdepth = 1 - (flux_ratio - observed_depth) / flux_ratio
                        = observed_depth / flux_ratio

    Args:
        flux_ratios: Output of compute_flux_ratios().
        observed_transit_depth: The measured transit depth (fractional).

    Returns:
        List of intrinsic depths, one per star. Zero-flux stars get inf.
        Depths > 1.0 are set to 0.0 (unphysical).
    """
    depths: list[float] = []
    for fr in flux_ratios:
        if fr > 0:
            d = observed_transit_depth / fr
            depths.append(0.0 if d > 1.0 else d)
        else:
            depths.append(float("inf"))
    return depths
