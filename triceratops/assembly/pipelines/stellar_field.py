"""Stellar-field assembly: catalog query + flux-ratio computation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from triceratops.assembly.errors import CatalogAcquisitionError

if TYPE_CHECKING:
    from triceratops.assembly.config import AssemblyConfig
    from triceratops.catalog.protocols import StarCatalogProvider
    from triceratops.domain.entities import StellarField
    from triceratops.lightcurve.ephemeris import ResolvedTarget


def assemble_stellar_field(
    catalog_provider: StarCatalogProvider,
    target: ResolvedTarget,
    config: AssemblyConfig,
    transit_depth: float | None,
    pixel_coords_per_sector: list[np.ndarray] | None,
    aperture_pixels_per_sector: list[np.ndarray] | None,
    sigma_psf_px: float,
) -> tuple[StellarField, list[str]]:
    """Query the catalog and compute flux ratios.

    Returns:
        (stellar_field, warnings) tuple.

    Raises:
        CatalogAcquisitionError: If the catalog query fails.
    """
    warnings: list[str] = []

    try:
        stellar_field = catalog_provider.query_nearby_stars(
            tic_id=target.tic_id,
            search_radius_px=config.catalog_search_radius_px,
            mission=config.mission,
        )
    except Exception as exc:
        raise CatalogAcquisitionError(
            f"Catalog query failed for TIC {target.tic_id}: {exc}"
        ) from exc

    stellar_field.validate()

    # Flux ratios and transit depths (if depth data provided)
    if (
        transit_depth is not None
        and pixel_coords_per_sector is not None
        and aperture_pixels_per_sector is not None
    ):
        from triceratops.catalog.flux_contributions import (
            compute_flux_ratios,
            compute_transit_depths,
        )

        flux_ratios = compute_flux_ratios(
            stellar_field,
            pixel_coords_per_sector,
            aperture_pixels_per_sector,
            sigma_psf_px,
        )
        transit_depths = compute_transit_depths(flux_ratios, transit_depth)
        for star, fr, td in zip(stellar_field.stars, flux_ratios, transit_depths):
            star.flux_ratio = fr
            star.transit_depth_required = td
    else:
        if transit_depth is not None:
            warnings.append(
                "transit_depth provided but pixel_coords_per_sector or "
                "aperture_pixels_per_sector missing — flux ratios not computed."
            )

    return stellar_field, warnings
