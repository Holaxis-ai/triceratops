"""ValidationPreparer: pure-validation boundary between assembly and compute.

This module validates pre-assembled domain inputs and maps them into
PreparedValidationInputs for provider-free compute via
ValidationEngine.compute_prepared().

Two-phase design
----------------
Assembly (DataAssemblyOrchestrator):
    - Query star catalog via StarCatalogProvider.
    - Compute flux ratios and transit depths.
    - Fetch TRILEGAL background population via PopulationSynthesisProvider.
    - Load contrast curve from disk.
    - Load external light curves from disk.
    - Emit a fully-populated AssembledInputs.

Prepare (this module):
    - Validate preconditions (mission, stellar params, light curve, period).
    - Map AssembledInputs fields to PreparedValidationInputs.

Compute (ValidationEngine.compute_prepared):
    - Accept PreparedValidationInputs.
    - Execute scenarios with no provider access.
    - Return ValidationResult.

Remote compute boundary guarantee
----------------------------------
Code that runs in a remote worker (e.g. Modal) should only ever call
``ValidationEngine.compute_prepared()``.  It must never instantiate a
``ValidationPreparer`` or any provider.

See: working_docs/iteration/priority-3_pure-compute-boundary.md
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from triceratops.config.config import Config
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import PeriodSpec
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry
from triceratops.validation.job import PreparedValidationInputs

if TYPE_CHECKING:
    from triceratops.assembly.inputs import AssembledInputs


class ValidationPreparer:
    """Validates pre-assembled inputs and produces PreparedValidationInputs.

    Pure validation — no I/O, no provider calls. Accepts AssembledInputs
    from the assembly layer and produces PreparedValidationInputs ready
    for provider-free compute via ValidationEngine.compute_prepared().
    """

    def __init__(
        self,
        registry: ScenarioRegistry | None = None,
    ) -> None:
        """Construct with an optional scenario registry.

        Args:
            registry: Scenario registry used to validate scenario_ids.
                Must be the same registry passed to the ValidationEngine
                that will call compute_prepared().  Defaults to
                DEFAULT_REGISTRY.
        """
        self._registry = registry if registry is not None else DEFAULT_REGISTRY

    def prepare(
        self,
        assembled: AssembledInputs,
        config: Config,
        period_days: PeriodSpec,
        *,
        scenario_ids: list[ScenarioID] | None = None,
    ) -> PreparedValidationInputs:
        """Validate pre-assembled domain inputs and produce compute-ready inputs.

        Pure validation — no I/O, no provider calls. Validates scientific
        and structural preconditions, then maps fields from AssembledInputs
        into PreparedValidationInputs.

        Args:
            assembled: Domain inputs from the assembly layer.
            config: Runtime compute configuration.
            period_days: Orbital period (scalar or [min, max] range).
            scenario_ids: Optional scenario filter. All must be registered.

        Returns:
            PreparedValidationInputs ready for ValidationEngine.compute_prepared().

        Raises:
            UnsupportedComputeModeError: If stellar field mission is not TESS.
            PreparedInputIncompleteError: If stellar params are missing.
            ValidationInputError: If light curve or period is invalid.
            ValueError: If scenario_ids contains unregistered IDs, or if
                stellar field structural invariants are violated.
        """
        from triceratops.validation.errors import (
            PreparedInputIncompleteError,
            UnsupportedComputeModeError,
            ValidationInputError,
        )

        # 1. Mission gate
        if assembled.stellar_field.mission != "TESS":
            raise UnsupportedComputeModeError(
                f"prepare() only supports mission='TESS'. "
                f"Stellar field has mission={assembled.stellar_field.mission!r}."
            )

        # 2. Structural validation
        assembled.stellar_field.validate()

        # 3. Stellar params presence
        if assembled.stellar_field.target.stellar_params is None:
            raise PreparedInputIncompleteError(
                "Target star has no stellar_params. Catalog enrichment "
                "must populate stellar_params before prepare."
            )

        # 4. Light curve validation
        if assembled.light_curve is None:
            raise ValidationInputError(
                "assembled.light_curve is None. A light curve is required "
                "for compute."
            )
        if len(assembled.light_curve.time_days) == 0:
            raise ValidationInputError(
                "assembled.light_curve is empty (zero data points)."
            )
        if len(assembled.light_curve.time_days) != len(assembled.light_curve.flux):
            raise ValidationInputError(
                f"Light curve shape mismatch: time_days has "
                f"{len(assembled.light_curve.time_days)} points but flux has "
                f"{len(assembled.light_curve.flux)} points."
            )

        # 5. Period validation
        if isinstance(period_days, (list, tuple)):
            if len(period_days) != 2:
                raise ValidationInputError(
                    f"period_days range must have exactly 2 elements, "
                    f"got {len(period_days)}."
                )
            p_min, p_max = period_days
            if not (np.isfinite(p_min) and np.isfinite(p_max)):
                raise ValidationInputError(
                    f"period_days range contains non-finite values: {period_days}."
                )
            if p_min <= 0 or p_max <= 0:
                raise ValidationInputError(
                    f"period_days must be positive, got range {period_days}."
                )
            if p_min >= p_max:
                raise ValidationInputError(
                    f"period_days range must satisfy min < max, "
                    f"got [{p_min}, {p_max}]."
                )
        else:
            if not np.isfinite(period_days):
                raise ValidationInputError(
                    f"period_days must be finite, got {period_days}."
                )
            if period_days <= 0:
                raise ValidationInputError(
                    f"period_days must be positive, got {period_days}."
                )

        # 6. Scenario registry check
        if scenario_ids is not None:
            unknown = [
                sid for sid in scenario_ids
                if self._registry.get_or_none(sid) is None
            ]
            if unknown:
                raise ValueError(
                    f"scenario_ids contains IDs not registered in the registry: "
                    f"{unknown}. Remove them or register the corresponding "
                    "scenario before calling prepare()."
                )

        # 7. Construct PreparedValidationInputs
        return PreparedValidationInputs(
            target_id=assembled.resolved_target.tic_id,
            stellar_field=assembled.stellar_field,
            light_curve=assembled.light_curve,
            config=config,
            period_days=period_days,
            trilegal_population=assembled.trilegal_population,
            external_lcs=assembled.external_lcs,
            contrast_curve=assembled.contrast_curve,
            molusc_data=assembled.molusc_data,
            scenario_ids=scenario_ids,
        )
