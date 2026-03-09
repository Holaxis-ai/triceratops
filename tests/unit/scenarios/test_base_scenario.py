"""Tests for triceratops.scenarios.base -- BaseScenario ABC."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.base import BaseScenario, Scenario

# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing
# ---------------------------------------------------------------------------

class _MinimalTPScenario(BaseScenario):
    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.TP

    @property
    def is_eb(self) -> bool:
        return False

    def _get_host_ldc(self, _sp, mission, _P_orb, _kwargs):
        return FixedLDCCatalog().get_coefficients(mission, 0, 5000, 4.5)

    def _sample_priors(self, n, sp, P_orb, config, **kw):
        return {
            "incs": np.full(n, 89.0),
            "eccs": np.zeros(n),
            "argps": np.zeros(n),
            "rps": np.full(n, 2.0),
        }

    def _compute_orbital_geometry(self, samples, P_orb, sp, config, **kw):
        n = len(P_orb)
        return {
            "a": np.full(n, 1e12),
            "Ptra": np.full(n, 0.05),
            "b": np.zeros(n),
            "coll": np.zeros(n, dtype=bool),
        }

    def _evaluate_lnL(self, lc, lnsigma, samples, geom, ldc, ext, config):
        n = config.n_mc_samples
        return np.full(n, -1.0), None

    def _pack_result(self, samples, geom, ldc, lnZ, idx, sp, ext, twin=False):
        n = len(idx)
        return ScenarioResult(
            scenario_id=ScenarioID.TP,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.ones(n),
            host_radius_rsun=np.ones(n),
            host_u1=np.full(n, 0.4),
            host_u2=np.full(n, 0.2),
            period_days=np.full(n, 5.0),
            inclination_deg=np.full(n, 89.0),
            impact_parameter=np.zeros(n),
            eccentricity=np.zeros(n),
            arg_periastron_deg=np.zeros(n),
            planet_radius_rearth=np.full(n, 2.0),
            eb_mass_msun=np.zeros(n),
            eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n),
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )


class _MinimalEBScenario(BaseScenario):
    @property
    def scenario_id(self) -> ScenarioID:
        return ScenarioID.EB

    @property
    def is_eb(self) -> bool:
        return True

    def _get_host_ldc(self, _sp, mission, _P_orb, _kwargs):
        return FixedLDCCatalog().get_coefficients(mission, 0, 5000, 4.5)

    def _sample_priors(self, n, sp, P_orb, config, **kw):
        return {
            "incs": np.full(n, 89.0),
            "eccs": np.zeros(n),
            "argps": np.zeros(n),
            "qs": np.full(n, 0.5),
        }

    def _compute_orbital_geometry(self, samples, P_orb, sp, config, **kw):
        n = len(P_orb)
        return {
            "a": np.full(n, 1e12),
            "Ptra": np.full(n, 0.05),
            "b": np.zeros(n),
            "coll": np.zeros(n, dtype=bool),
            "a_twin": np.full(n, 1e12),
            "Ptra_twin": np.full(n, 0.05),
            "b_twin": np.zeros(n),
            "coll_twin": np.zeros(n, dtype=bool),
        }

    def _evaluate_lnL(self, lc, lnsigma, samples, geom, ldc, ext, config):
        n = config.n_mc_samples
        return np.full(n, -1.0), np.full(n, -2.0)

    def _pack_result(self, samples, geom, ldc, lnZ, idx, sp, ext, twin=False):
        n = len(idx)
        sid = ScenarioID.EBX2P if twin else ScenarioID.EB
        return ScenarioResult(
            scenario_id=sid,
            host_star_tic_id=0,
            ln_evidence=lnZ,
            host_mass_msun=np.ones(n),
            host_radius_rsun=np.ones(n),
            host_u1=np.full(n, 0.4),
            host_u2=np.full(n, 0.2),
            period_days=np.full(n, 5.0),
            inclination_deg=np.full(n, 89.0),
            impact_parameter=np.zeros(n),
            eccentricity=np.zeros(n),
            arg_periastron_deg=np.zeros(n),
            planet_radius_rearth=np.zeros(n),
            eb_mass_msun=np.full(n, 0.5),
            eb_radius_rsun=np.full(n, 0.5),
            flux_ratio_eb_tess=np.full(n, 0.1),
            companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n),
            flux_ratio_companion_tess=np.zeros(n),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def stellar_params():
    return StellarParameters(
        mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
        logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
    )


@pytest.fixture()
def light_curve():
    return LightCurve(
        time_days=np.linspace(-0.1, 0.1, 50),
        flux=np.ones(50),
        flux_err=0.001,
    )


@pytest.fixture()
def config():
    return Config(n_mc_samples=50, n_best_samples=10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseScenarioABC:
    def test_abstract_class_not_instantiable(self) -> None:
        with pytest.raises(TypeError):
            BaseScenario(FixedLDCCatalog())  # type: ignore[abstract]

    def test_minimal_tp_compute_returns_scenario_result(
        self, light_curve, stellar_params, config
    ) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        result = scenario.compute(light_curve, stellar_params, 5.0, config)
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == ScenarioID.TP
        assert np.isfinite(result.ln_evidence)

    def test_tp_returns_single_result_not_tuple(
        self, light_curve, stellar_params, config
    ) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        result = scenario.compute(light_curve, stellar_params, 5.0, config)
        assert not isinstance(result, tuple)

    def test_eb_returns_tuple(
        self, light_curve, stellar_params, config
    ) -> None:
        scenario = _MinimalEBScenario(FixedLDCCatalog())
        result = scenario.compute(light_curve, stellar_params, 5.0, config)
        assert isinstance(result, tuple)
        assert len(result) == 2
        r, r_twin = result
        assert r.scenario_id == ScenarioID.EB
        assert r_twin.scenario_id == ScenarioID.EBX2P

    def test_is_eb_false_means_returns_twin_false(self) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        assert not scenario.is_eb
        assert not scenario.returns_twin

    def test_is_eb_true_means_returns_twin_true(self) -> None:
        scenario = _MinimalEBScenario(FixedLDCCatalog())
        assert scenario.is_eb
        assert scenario.returns_twin

    def test_phase1_period_scalar_becomes_array(
        self, light_curve, stellar_params, config
    ) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        # compute() calls resolve_period(5.0, N) internally; we verify it
        # doesn't crash and produces a valid result
        result = scenario.compute(light_curve, stellar_params, 5.0, config)
        assert isinstance(result, ScenarioResult)

    def test_resolve_external_lc_ldcs_empty_input(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        resolved = scenario._resolve_external_lc_ldcs([], stellar_params)
        assert resolved == []

    def test_resolve_external_lc_ldcs_populates_ldc(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog(u1=0.35, u2=0.15))
        ext_lcs = [
            ExternalLightCurve(
                light_curve=LightCurve(
                    time_days=np.linspace(-0.1, 0.1, 20),
                    flux=np.ones(20),
                    flux_err=0.002,
                ),
                band="J",
            ),
            ExternalLightCurve(
                light_curve=LightCurve(
                    time_days=np.linspace(-0.1, 0.1, 20),
                    flux=np.ones(20),
                    flux_err=0.002,
                ),
                band="K",
            ),
        ]
        resolved = scenario._resolve_external_lc_ldcs(ext_lcs, stellar_params)
        assert len(resolved) == 2
        for ext in resolved:
            assert ext.ldc is not None
            assert ext.ldc.u1 == pytest.approx(0.35)
            assert ext.ldc.u2 == pytest.approx(0.15)

    def test_scenario_protocol_satisfied(self) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        assert isinstance(scenario, Scenario)

    def test_n_best_samples_respected(
        self, light_curve, stellar_params
    ) -> None:
        cfg = Config(n_mc_samples=50, n_best_samples=5)
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        result = scenario.compute(light_curve, stellar_params, 5.0, cfg)
        assert isinstance(result, ScenarioResult)
        assert len(result.host_mass_msun) == 5
