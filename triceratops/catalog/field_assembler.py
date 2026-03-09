"""StellarFieldAssembler: orchestrates catalog query + flux computation."""

from __future__ import annotations

import numpy as np

from triceratops.catalog.flux_contributions import compute_flux_ratios, compute_transit_depths
from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
from triceratops.domain.entities import StellarField


class StellarFieldAssembler:
    """Assembles a complete StellarField including flux ratios and transit depths.

    Accepts injectable providers and computes derived flux quantities.
    """

    def __init__(
        self,
        catalog_provider: StarCatalogProvider | None = None,
        aperture_provider: ApertureProvider | None = None,
    ) -> None:
        if catalog_provider is None:
            from triceratops.catalog.mast_provider import MASTCatalogProvider
            catalog_provider = MASTCatalogProvider()
        if aperture_provider is None:
            from triceratops.catalog.mast_provider import TesscutApertureProvider
            aperture_provider = TesscutApertureProvider()
        self._catalog: StarCatalogProvider = catalog_provider
        self._aperture: ApertureProvider = aperture_provider

    def assemble(
        self,
        tic_id: int,
        sectors: np.ndarray,
        mission: str,
        search_radius_px: int,
        transit_depth: float,
        pixel_coords_per_sector: list[np.ndarray],
        aperture_pixels_per_sector: list[np.ndarray],
        sigma_psf_px: float = 0.75,
    ) -> StellarField:
        """Query catalog, compute flux ratios, return complete StellarField.

        Args:
            tic_id: Target TIC ID.
            sectors: Array of sector/quarter/campaign numbers.
            mission: "TESS", "Kepler", or "K2".
            search_radius_px: Search radius in pixels.
            transit_depth: Observed transit depth (fractional).
            pixel_coords_per_sector: Per-sector star pixel positions.
            aperture_pixels_per_sector: Per-sector aperture pixel positions.
            sigma_psf_px: PSF sigma in pixels (default 0.75).

        Returns:
            StellarField with flux_ratio and transit_depth_required populated.
        """
        field = self._catalog.query_nearby_stars(tic_id, search_radius_px, mission)

        flux_ratios = compute_flux_ratios(
            field, pixel_coords_per_sector, aperture_pixels_per_sector, sigma_psf_px,
        )
        depths = compute_transit_depths(flux_ratios, transit_depth)

        for i, star in enumerate(field.stars):
            star.flux_ratio = flux_ratios[i]
            star.transit_depth_required = depths[i]

        return field
