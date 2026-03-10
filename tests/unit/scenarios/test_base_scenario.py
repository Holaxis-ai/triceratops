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


# ---------------------------------------------------------------------------
# _stellar_logg_from_mass_radius
# ---------------------------------------------------------------------------

class TestStellarLoggFromMassRadius:
    """Tests for BaseScenario._stellar_logg_from_mass_radius()."""

    _logg_fn = staticmethod(BaseScenario._stellar_logg_from_mass_radius)

    def _make_params(self, mass_msun: float, radius_rsun: float) -> StellarParameters:
        return StellarParameters(
            mass_msun=mass_msun, radius_rsun=radius_rsun,
            teff_k=5778.0, logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        )

    def test_solar_logg_approx_4_44(self) -> None:
        """M=1 Msun, R=1 Rsun should yield logg ≈ 4.44 (accepted solar value)."""
        from triceratops.config.config import CONST
        import math
        params = self._make_params(1.0, 1.0)
        result = self._logg_fn(params)
        expected = math.log10(CONST.G * CONST.Msun / CONST.Rsun ** 2)
        assert result == pytest.approx(expected, abs=1e-4)
        # And should be in the physically sensible neighbourhood of 4.44
        assert 4.40 <= result <= 4.48

    def test_small_radius_gives_larger_logg(self) -> None:
        """Smaller radius → higher surface gravity → larger logg."""
        params_small = self._make_params(1.0, 0.1)
        params_large = self._make_params(1.0, 1.0)
        assert self._logg_fn(params_small) > self._logg_fn(params_large)

    def test_large_radius_gives_smaller_logg(self) -> None:
        """Larger radius → lower surface gravity → smaller logg."""
        params_ref = self._make_params(1.0, 1.0)
        params_giant = self._make_params(1.0, 10.0)
        assert self._logg_fn(params_giant) < self._logg_fn(params_ref)

    def test_logg_formula_matches_manual_calculation(self) -> None:
        """Verify against log10(G * M * Msun / (R * Rsun)^2) using CONST values."""
        from triceratops.config.config import CONST
        import math
        mass, radius = 1.5, 2.0
        params = self._make_params(mass, radius)
        expected = math.log10(
            CONST.G * (mass * CONST.Msun) / (radius * CONST.Rsun) ** 2
        )
        result = self._logg_fn(params)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_high_mass_increases_logg(self) -> None:
        """Higher mass at same radius → higher logg."""
        params_lo = self._make_params(0.5, 1.0)
        params_hi = self._make_params(2.0, 1.0)
        assert self._logg_fn(params_hi) > self._logg_fn(params_lo)

    def test_returns_float(self) -> None:
        params = self._make_params(1.0, 1.0)
        result = self._logg_fn(params)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _resolve_external_lc_ldcs edge cases
# ---------------------------------------------------------------------------

class TestResolveExternalLcLdcs:
    """Tests for BaseScenario._resolve_external_lc_ldcs()."""

    @pytest.fixture()
    def stellar_params(self):
        return StellarParameters(
            mass_msun=1.0, radius_rsun=1.0, teff_k=5778.0,
            logg=4.44, metallicity_dex=0.0, parallax_mas=10.0,
        )

    def _make_ext_lc(self, band: str = "J") -> ExternalLightCurve:
        lc = LightCurve(
            time_days=np.linspace(-0.1, 0.1, 20),
            flux=np.ones(20),
            flux_err=0.002,
        )
        return ExternalLightCurve(light_curve=lc, band=band)

    def test_empty_list_returns_empty_list(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        result = scenario._resolve_external_lc_ldcs([], stellar_params)
        assert result == []

    def test_single_lc_returns_list_of_length_one(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog(u1=0.3, u2=0.1))
        ext_lcs = [self._make_ext_lc("J")]
        result = scenario._resolve_external_lc_ldcs(ext_lcs, stellar_params)
        assert len(result) == 1

    def test_single_lc_has_populated_ldc(self, stellar_params) -> None:
        u1, u2 = 0.35, 0.15
        scenario = _MinimalTPScenario(FixedLDCCatalog(u1=u1, u2=u2))
        ext_lcs = [self._make_ext_lc("J")]
        result = scenario._resolve_external_lc_ldcs(ext_lcs, stellar_params)
        assert result[0].ldc is not None
        assert result[0].ldc.u1 == pytest.approx(u1)
        assert result[0].ldc.u2 == pytest.approx(u2)

    def test_band_is_preserved(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        ext_lcs = [self._make_ext_lc("K")]
        result = scenario._resolve_external_lc_ldcs(ext_lcs, stellar_params)
        assert result[0].band == "K"

    def test_multiple_lcs_all_get_ldc(self, stellar_params) -> None:
        scenario = _MinimalTPScenario(FixedLDCCatalog(u1=0.4, u2=0.2))
        ext_lcs = [self._make_ext_lc("J"), self._make_ext_lc("H"), self._make_ext_lc("K")]
        result = scenario._resolve_external_lc_ldcs(ext_lcs, stellar_params)
        assert len(result) == 3
        for ext in result:
            assert ext.ldc is not None

    def test_light_curve_data_preserved(self, stellar_params) -> None:
        """Resolved ExternalLightCurve retains the original flux data."""
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        ext = self._make_ext_lc("J")
        result = scenario._resolve_external_lc_ldcs([ext], stellar_params)
        np.testing.assert_array_equal(
            result[0].light_curve.flux, ext.light_curve.flux
        )

    def test_ldc_band_matches_filter(self, stellar_params) -> None:
        """The LDC stored on the resolved LC carries the correct band string."""
        scenario = _MinimalTPScenario(FixedLDCCatalog())
        ext = self._make_ext_lc("H")
        result = scenario._resolve_external_lc_ldcs([ext], stellar_params)
        # FixedLDCCatalog.get_coefficients sets band=filter_name
        assert result[0].ldc is not None
        assert result[0].ldc.band == "H"
