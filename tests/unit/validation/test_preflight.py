"""Tests for PreparedValidationInputs.validate() and the compute boundary errors.

Covers priority-5: fail loudly on invalid inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.validation.errors import (
    PreparedInputIncompleteError,
    ValidationInputError,
)
from triceratops.validation.job import PreparedValidationInputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sp() -> StellarParameters:
    return StellarParameters(
        mass_msun=1.0, radius_rsun=1.0, teff_k=5500.0,
        logg=4.4, metallicity_dex=0.0, parallax_mas=10.0,
    )


def _star(tic_id: int = 12345, with_params: bool = True) -> Star:
    return Star(
        tic_id=tic_id, ra_deg=10.0, dec_deg=5.0,
        tmag=11.0, jmag=10.5, hmag=10.3, kmag=10.2,
        bmag=11.5, vmag=11.2,
        stellar_params=_sp() if with_params else None,
        flux_ratio=1.0,
        transit_depth_required=0.01,
    )


def _field(with_params: bool = True) -> StellarField:
    return StellarField(
        target_id=12345, mission="TESS", search_radius_pixels=10,
        stars=[_star(with_params=with_params)],
    )


def _lc(n: int = 50) -> LightCurve:
    t = np.linspace(-0.1, 0.1, n)
    flux = np.ones(n)
    flux[20:30] = 0.999
    return LightCurve(time_days=t, flux=flux, flux_err=0.001)


def _cfg() -> Config:
    return Config(n_mc_samples=100, n_best_samples=10)


def _valid_payload(**overrides) -> PreparedValidationInputs:
    """Return a minimal valid payload; keyword args override defaults."""
    defaults = dict(
        target_id=12345,
        stellar_field=_field(),
        light_curve=_lc(),
        config=_cfg(),
        period_days=5.0,
        scenario_ids=[],   # empty = no scenarios, no TRILEGAL check
    )
    defaults.update(overrides)
    return PreparedValidationInputs(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPreflightValidPassesForValidPayload:
    def test_minimal_valid_payload(self) -> None:
        _valid_payload().validate()

    def test_period_as_range(self) -> None:
        _valid_payload(period_days=[1.0, 10.0]).validate()

    def test_scenario_ids_none_skips_trilegal_check(self) -> None:
        """scenario_ids=None does not trigger TRILEGAL check even with no population."""
        _valid_payload(scenario_ids=None).validate()


# ---------------------------------------------------------------------------
# Phase 1a: missing stellar_params → PreparedInputIncompleteError
# ---------------------------------------------------------------------------


class TestMissingStellarParams:
    def test_raises_prepared_input_incomplete_error(self) -> None:
        payload = _valid_payload(stellar_field=_field(with_params=False))
        with pytest.raises(PreparedInputIncompleteError, match="stellar_params"):
            payload.validate()

    def test_error_message_names_tic_id(self) -> None:
        payload = _valid_payload(stellar_field=_field(with_params=False))
        with pytest.raises(PreparedInputIncompleteError) as exc_info:
            payload.validate()
        assert "12345" in str(exc_info.value)

    def test_compute_raises_instead_of_returning_fpp_1(self) -> None:
        """Silent FPP=1.0 regression guard: compute() must raise, not return."""
        from triceratops.validation.engine import ValidationEngine

        engine = ValidationEngine()
        field = _field(with_params=False)
        with pytest.raises(PreparedInputIncompleteError, match="stellar_params"):
            engine.compute(
                light_curve=_lc(),
                stellar_field=field,
                period_days=5.0,
                config=_cfg(),
            )

    def test_compute_prepared_raises_instead_of_returning_fpp_1(self) -> None:
        """compute_prepared() must also raise, not return FPP=1.0."""
        from triceratops.validation.engine import ValidationEngine

        engine = ValidationEngine()
        payload = _valid_payload(stellar_field=_field(with_params=False))
        with pytest.raises(PreparedInputIncompleteError, match="stellar_params"):
            engine.compute_prepared(payload)


# ---------------------------------------------------------------------------
# Phase 1b: malformed LightCurve → ValidationInputError
# ---------------------------------------------------------------------------


class TestMalformedLightCurve:
    def test_empty_time_raises(self) -> None:
        bad_lc = LightCurve(
            time_days=np.array([]), flux=np.array([]), flux_err=0.001,
        )
        with pytest.raises(ValidationInputError, match="empty"):
            _valid_payload(light_curve=bad_lc).validate()

    def test_shape_mismatch_raises(self) -> None:
        bad_lc = LightCurve(
            time_days=np.linspace(-0.1, 0.1, 50),
            flux=np.ones(30),   # wrong length
            flux_err=0.001,
        )
        with pytest.raises(ValidationInputError, match="shape mismatch"):
            _valid_payload(light_curve=bad_lc).validate()

    def test_shape_mismatch_message_includes_counts(self) -> None:
        bad_lc = LightCurve(
            time_days=np.linspace(-0.1, 0.1, 50),
            flux=np.ones(30),
            flux_err=0.001,
        )
        with pytest.raises(ValidationInputError) as exc_info:
            _valid_payload(light_curve=bad_lc).validate()
        msg = str(exc_info.value)
        assert "50" in msg and "30" in msg


# ---------------------------------------------------------------------------
# Phase 1c: invalid period_days → ValidationInputError
# ---------------------------------------------------------------------------


class TestInvalidPeriodDays:
    def test_zero_period_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="positive"):
            _valid_payload(period_days=0.0).validate()

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="positive"):
            _valid_payload(period_days=-1.0).validate()

    def test_nan_period_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="finite"):
            _valid_payload(period_days=float("nan")).validate()

    def test_inf_period_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="finite"):
            _valid_payload(period_days=float("inf")).validate()

    def test_range_min_ge_max_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="min < max"):
            _valid_payload(period_days=[5.0, 5.0]).validate()

    def test_range_negative_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="positive"):
            _valid_payload(period_days=[-1.0, 5.0]).validate()

    def test_range_wrong_length_raises(self) -> None:
        with pytest.raises(ValidationInputError, match="2 elements"):
            _valid_payload(period_days=[1.0, 5.0, 10.0]).validate()

    def test_valid_range_passes(self) -> None:
        _valid_payload(period_days=[1.0, 10.0]).validate()


# ---------------------------------------------------------------------------
# Phase 2: missing TRILEGAL for explicit TRILEGAL scenario_ids
# ---------------------------------------------------------------------------


class TestMissingTrilegalPopulation:
    def test_trilegal_scenario_without_population_raises(self) -> None:
        """Explicit TRILEGAL scenario_ids with no population raises."""
        registered_trilegal = [
            sid for sid in ScenarioID.trilegal_scenarios()
            if sid in (ScenarioID.BTP, ScenarioID.BEB)
        ]
        assert registered_trilegal, "Need registered TRILEGAL scenarios for this test"

        payload = _valid_payload(
            scenario_ids=registered_trilegal,
            trilegal_population=None,
        )
        with pytest.raises(PreparedInputIncompleteError, match="trilegal_population"):
            payload.validate()

    def test_error_message_names_missing_scenarios(self) -> None:
        payload = _valid_payload(
            scenario_ids=[ScenarioID.BTP],
            trilegal_population=None,
        )
        with pytest.raises(PreparedInputIncompleteError) as exc_info:
            payload.validate()
        assert "BTP" in str(exc_info.value)

    def test_non_trilegal_scenario_without_population_passes(self) -> None:
        """Non-TRILEGAL scenario_ids should not trigger the population check."""
        _valid_payload(
            scenario_ids=[ScenarioID.TP],
            trilegal_population=None,
        ).validate()

    def test_trilegal_scenario_with_population_passes(self) -> None:
        """Providing a population satisfies the check."""
        from unittest.mock import MagicMock
        pop = MagicMock()
        _valid_payload(
            scenario_ids=[ScenarioID.BTP],
            trilegal_population=pop,
        ).validate()

    def test_scenario_ids_none_does_not_check_trilegal(self) -> None:
        """When scenario_ids=None the TRILEGAL check is deferred to the engine."""
        _valid_payload(scenario_ids=None, trilegal_population=None).validate()
