"""LegacyPreparerAdapter: backward-compatible 18-argument prepare() shim.

Migration shim only. Target removal: after all callers migrate to the
assembly-layer API (DataAssemblyOrchestrator + ValidationPreparer.prepare()).

This adapter preserves the original prepare() signature that accepts raw
file paths and provider references, converting them to the assembly-layer
protocol objects before delegating to the orchestrator and preparer.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from triceratops.catalog.protocols import ApertureProvider, StarCatalogProvider
from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.value_objects import ContrastCurve
from triceratops.population.protocols import PopulationSynthesisProvider
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry
from triceratops.validation.job import PreparedValidationInputs
from triceratops.validation.preparer import ValidationPreparer

if TYPE_CHECKING:
    from triceratops.domain.scenario_id import ScenarioID


class LegacyPreparerAdapter:
    """Backward-compatible adapter preserving the old 18-argument prepare().

    Wraps DataAssemblyOrchestrator + ValidationPreparer to provide the same
    external interface as the original ValidationPreparer.prepare().

    Migration shim only. Target removal: after all callers migrate.
    """

    def __init__(
        self,
        catalog_provider: StarCatalogProvider,
        population_provider: PopulationSynthesisProvider | None = None,
        aperture_provider: ApertureProvider | None = None,
        registry: ScenarioRegistry | None = None,
    ) -> None:
        self._catalog = catalog_provider
        self._population = population_provider
        self._aperture = aperture_provider
        self._registry = registry if registry is not None else DEFAULT_REGISTRY
        self._preparer = ValidationPreparer(registry=self._registry)

    def prepare(
        self,
        target_id: int,
        sectors: np.ndarray,
        light_curve: LightCurve,
        config: Config,
        period_days: float | list[float] | tuple[float, float],
        mission: str = "TESS",
        search_radius: int = 10,
        transit_depth: float | None = None,
        pixel_coords_per_sector: list[np.ndarray] | None = None,
        aperture_pixels_per_sector: list[np.ndarray] | None = None,
        sigma_psf_px: float = 0.75,
        trilegal_cache_path: str | None = None,
        contrast_curve_file: str | None = None,
        contrast_curve_band: str = "TESS",
        external_lc_files: list[str] | None = None,
        filt_lcs: list[str] | None = None,
        scenario_ids: list | None = None,
        molusc_file: str | None = None,
    ) -> PreparedValidationInputs:
        """Legacy 18-argument prepare() — delegates to orchestrator + preparer.

        See ValidationPreparer (original) for full argument documentation.
        """
        import dataclasses

        from triceratops.assembly.config import AssemblyConfig
        from triceratops.assembly.orchestrator import DataAssemblyOrchestrator
        from triceratops.lightcurve.ephemeris import ResolvedTarget

        # ---- 0. Mission gate (fast-fail before any IO) ----
        if mission != "TESS":
            from triceratops.validation.errors import UnsupportedComputeModeError
            raise UnsupportedComputeModeError(
                f"prepare() only supports mission='TESS'. Got {mission!r}. "
                "Kepler/K2 support is experimental and not available for "
                "prepared compute jobs."
            )

        # ---- 1. Validate scenario_ids against the registry (fast-fail) ----
        if scenario_ids is not None:
            unknown = [sid for sid in scenario_ids if self._registry.get_or_none(sid) is None]
            if unknown:
                raise ValueError(
                    f"scenario_ids contains IDs not registered in the registry: {unknown}. "
                    "Remove them or register the corresponding scenario before calling prepare()."
                )

        # ---- 2. Validate external LC pairing (fast-fail) ----
        _have_files = bool(external_lc_files)
        _have_filts = bool(filt_lcs)
        if _have_files or _have_filts:
            if not (_have_files and _have_filts):
                raise ValueError(
                    "external_lc_files and filt_lcs must both be provided together; "
                    f"got external_lc_files={external_lc_files!r} and filt_lcs={filt_lcs!r}."
                )
            if len(external_lc_files) != len(filt_lcs):  # type: ignore[arg-type]
                raise ValueError(
                    f"external_lc_files and filt_lcs must have the same length, "
                    f"got {len(external_lc_files)} files and {len(filt_lcs)} filters."  # type: ignore[arg-type]
                )

        # ---- 3. Build assembly-layer objects from legacy args ----
        resolved_target = ResolvedTarget(
            target_ref=str(target_id),
            tic_id=target_id,
            ephemeris=None,
            source="legacy",
        )

        assembly_config = AssemblyConfig(
            mission=mission,
            include_light_curve=False,
            catalog_search_radius_px=search_radius,
            trilegal_cache_path=trilegal_cache_path,
            contrast_curve_band=contrast_curve_band,
            include_contrast_curve=contrast_curve_file is not None,
            include_molusc=molusc_file is not None,
            include_external_lcs=_have_files,
            include_trilegal=self._population is not None,
        )

        # Build per-call source adapters from path strings
        contrast_source = (
            _path_to_contrast_source(contrast_curve_file)
            if contrast_curve_file is not None else None
        )
        molusc_source = (
            _path_to_molusc_source(molusc_file)
            if molusc_file is not None else None
        )
        external_lc_source = (
            _paths_to_external_lc_source(external_lc_files, filt_lcs)  # type: ignore[arg-type]
            if _have_files else None
        )

        # ---- 4. Assemble via orchestrator ----
        orchestrator = DataAssemblyOrchestrator(
            catalog_provider=self._catalog,
            population_provider=self._population,
            aperture_provider=self._aperture,
            contrast_source=contrast_source,
            molusc_source=molusc_source,
            external_lc_source=external_lc_source,
            registry=self._registry,
        )

        assembled = orchestrator.assemble(
            target=resolved_target,
            config=assembly_config,
            scenario_ids=scenario_ids,
            transit_depth=transit_depth,
            pixel_coords_per_sector=pixel_coords_per_sector,
            aperture_pixels_per_sector=aperture_pixels_per_sector,
            sigma_psf_px=sigma_psf_px,
        )

        # ---- 5. Inject pre-existing light curve ----
        assembled = dataclasses.replace(assembled, light_curve=light_curve)

        # ---- 6. Delegate to preparer ----
        return self._preparer.prepare(
            assembled, config, period_days, scenario_ids=scenario_ids,
        )


# ---------------------------------------------------------------------------
# Module-level source adapters: convert file-path arguments into protocol objects
# ---------------------------------------------------------------------------


def _path_to_contrast_source(path: str) -> object:
    """Wrap a file path into a ContrastCurveSource-compatible adapter."""
    from triceratops.io.contrast_curves import load_contrast_curve

    class _ContrastAdapter:
        def load(self, band: str) -> ContrastCurve:
            return load_contrast_curve(Path(path), band=band)

    return _ContrastAdapter()


def _path_to_molusc_source(path: str) -> object:
    """Wrap a file path into a MoluscSource-compatible adapter."""
    from triceratops.domain.molusc import MoluscData
    from triceratops.io.molusc import load_molusc_file

    class _MoluscAdapter:
        def load(self) -> MoluscData:
            return load_molusc_file(Path(path))

    return _MoluscAdapter()


def _paths_to_external_lc_source(
    paths: list[str], bands: list[str],
) -> object:
    """Wrap file paths + band labels into an ExternalLcSource-compatible adapter."""
    from triceratops.io.external_lc import load_external_lc_as_object

    class _ExternalLcAdapter:
        def load(self) -> list[ExternalLightCurve]:
            return [
                load_external_lc_as_object(Path(p), b)
                for p, b in zip(paths, bands)
            ]

    return _ExternalLcAdapter()
