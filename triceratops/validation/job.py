"""PreparedValidationInputs and PreparedValidationMetadata — the two-phase compute boundary.

This module defines the job payload that separates the *preparation* phase (provider-backed IO,
catalog queries, population synthesis) from the *compute* phase (pure scenario orchestration).

Two-phase design
----------------
Phase 1 — Prepare (ValidationPreparer / ValidationWorkspace):
    - Query star catalog; materialise StellarField with flux ratios and transit depths.
    - Fetch TRILEGAL background population (if needed).
    - Load contrast curve and external light curves from disk.
    - Emit one fully-populated PreparedValidationInputs.

Phase 2 — Compute (ValidationEngine.compute_prepared):
    - Accept PreparedValidationInputs.
    - Execute scenarios with NO provider access, NO network calls, NO filesystem assumptions.
    - Return ValidationResult.

Remote compute boundary guarantee
----------------------------------
Any function that receives only a ``PreparedValidationInputs`` and calls
``ValidationEngine.compute_prepared()`` must not instantiate providers, query MAST,
query TRILEGAL, or depend on the current working directory.

See: working_docs/iteration/priority-3_pure-compute-boundary.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from collections.abc import Sequence

if TYPE_CHECKING:
    from triceratops.config.config import Config
    from triceratops.domain.entities import ExternalLightCurve, LightCurve, StellarField
    from triceratops.domain.scenario_id import ScenarioID
    from triceratops.domain.value_objects import ContrastCurve
    from triceratops.population.protocols import TRILEGALResult


@dataclass
class PreparedValidationInputs:
    """Fully-materialised inputs for a provider-free validation compute.

    All fields required for ``ValidationEngine.compute_prepared()`` are present
    on this object.  No network access, no filesystem IO, and no provider
    instantiation should happen during or after construction.

    Fields
    ------
    target_id:
        TIC (or KIC/EPIC) identifier for the validation target.
    stellar_field:
        Assembled field including target and neighbours, with flux_ratio and
        transit_depth_required already computed on each Star.
    light_curve:
        Phase-folded, normalised photometric time series.
    config:
        Runtime configuration (n_mc_samples, lnz_const, …).
    period_days:
        Orbital period in days.  May be a scalar or [min, max] range.
    trilegal_population:
        Materialised TRILEGAL background population.  Must be provided for
        scenarios that require it (BTP, BEB, BEBx2P).  Pass None if those
        scenarios are not being run.
    external_lcs:
        Ground-based follow-up light curves.  None if not available.
    contrast_curve:
        AO/speckle contrast curve.  None if not available.
    molusc_file:
        Local filesystem path to a MOLUSC output file.
        NOTE: This is a bare path string — not yet materialised content.
        The compute boundary is not fully clean for this field until Phase 4,
        when remote execution requires the content to be embedded rather than
        referenced by path.  Deferred to Phase 4.
    scenario_ids:
        The scenario subset that was prepared for.  ``None`` means the full
        default registry.  ``compute_prepared()`` passes this directly to
        ``compute(scenario_ids=...)``, keeping the prepare/compute contract
        consistent: the engine runs exactly the scenarios that were prepared.
    """

    target_id: int
    stellar_field: StellarField
    light_curve: LightCurve
    config: Config
    period_days: float | list[float]
    trilegal_population: TRILEGALResult | None = None
    external_lcs: list[ExternalLightCurve] | None = None
    contrast_curve: ContrastCurve | None = None
    molusc_file: str | None = None  # local path — not yet materialised; deferred to Phase 4
    scenario_ids: Sequence[ScenarioID] | None = None  # None → run full default registry

    def validate(self) -> None:
        """Preflight validation: assert all scientific preconditions are met.

        Called by ``ValidationEngine.compute_prepared()`` before any
        scenario work begins.  May also be called explicitly to validate
        a directly-constructed or deserialized payload.

        Raises:
            ValueError: If structural field invariants are broken
                (delegated to ``StellarField.validate()``).
            PreparedInputIncompleteError: If a required field is missing
                (no stellar_params, missing TRILEGAL population).
            ValidationInputError: If a scientific input value is invalid
                (empty or inconsistent LightCurve, non-positive period).
        """
        import math

        from triceratops.validation.errors import (
            PreparedInputIncompleteError,
            ValidationInputError,
        )

        # 1. Field structure (target at index 0, no duplicates, etc.)
        self.stellar_field.validate()

        # 2. Target must have stellar parameters
        if self.stellar_field.target.stellar_params is None:
            raise PreparedInputIncompleteError(
                f"Target star (TIC {self.target_id}) has no stellar_params. "
                "Stellar parameters are required for all scenario computations. "
                "Check the catalog query result or set stellar_params manually."
            )

        # 3. LightCurve must be non-empty and shape-consistent
        n_time = len(self.light_curve.time_days)
        n_flux = len(self.light_curve.flux)
        if n_time == 0:
            raise ValidationInputError("light_curve.time_days is empty.")
        if n_flux == 0:
            raise ValidationInputError("light_curve.flux is empty.")
        if n_time != n_flux:
            raise ValidationInputError(
                f"light_curve shape mismatch: time_days has {n_time} points "
                f"but flux has {n_flux} points."
            )

        # 4. period_days must be positive and finite
        period = self.period_days
        if isinstance(period, list):
            if len(period) != 2:
                raise ValidationInputError(
                    f"period_days as a range must have exactly 2 elements [min, max], "
                    f"got {period!r}."
                )
            lo, hi = float(period[0]), float(period[1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValidationInputError(
                    f"period_days range contains non-finite values: {period!r}."
                )
            if lo <= 0 or hi <= 0:
                raise ValidationInputError(
                    f"period_days range must be positive, got {period!r}."
                )
            if lo >= hi:
                raise ValidationInputError(
                    f"period_days range must satisfy min < max, got {period!r}."
                )
        else:
            p = float(period)
            if not math.isfinite(p):
                raise ValidationInputError(
                    f"period_days must be finite, got {period!r}."
                )
            if p <= 0:
                raise ValidationInputError(
                    f"period_days must be positive, got {period!r}."
                )

        # 5. TRILEGAL population must be present for explicitly-requested TRILEGAL scenarios.
        #
        # Only checked when scenario_ids is explicitly set.  When scenario_ids=None
        # the engine uses its own (possibly custom) registry, which validate() cannot
        # access; the workspace/preparer already materialises the population before
        # calling compute_prepared(), so the check there would be redundant and would
        # incorrectly fire for tests that use a custom registry without TRILEGAL scenarios.
        if self.scenario_ids is not None and self.trilegal_population is None:
            from triceratops.domain.scenario_id import ScenarioID
            from triceratops.scenarios.registry import DEFAULT_REGISTRY
            trilegal_ids = set(ScenarioID.trilegal_scenarios())
            active_scenarios = [
                s for sid in self.scenario_ids
                if (s := DEFAULT_REGISTRY.get_or_none(sid)) is not None
            ]
            missing = [s.scenario_id for s in active_scenarios
                       if s.scenario_id in trilegal_ids]
            if missing:
                raise PreparedInputIncompleteError(
                    f"trilegal_population is required for scenarios {missing} but was not provided. "
                    "Pass a population_provider to ValidationPreparer or ValidationWorkspace, "
                    "or exclude TRILEGAL-dependent scenarios via scenario_ids."
                )


@dataclass
class PreparedValidationMetadata:
    """Optional provenance information attached to a PreparedValidationInputs.

    All fields are optional.  Consumers must not require any of these fields
    for correctness — they are for auditing, caching, and debugging only.

    Fields
    ------
    prep_timestamp:
        UTC datetime when the inputs were prepared.
    source:
        Human-readable label for where the inputs came from (e.g. "MAST/local").
    trilegal_cache_origin:
        Path or URL of the TRILEGAL cache file that was used, if any.
    warnings:
        List of non-fatal messages raised during preparation (e.g. missing
        magnitudes, fallback values used).
    """

    prep_timestamp: datetime | None = None
    source: str | None = None
    trilegal_cache_origin: str | None = None
    warnings: list[str] = field(default_factory=list)
