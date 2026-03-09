"""Tests for TTPScenario and TEBScenario."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import CONST, Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import LimbDarkeningCoeffs, StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.target_scenarios import TEBScenario, TTPScenario

_LNL_MOD = "triceratops.scenarios.target_scenarios"


# ---------------------------------------------------------------------------
# Mock lnL functions (pytransit not usable under numpy 2.x in this env)
# ---------------------------------------------------------------------------

def _mock_lnL_planet_p(*, time, flux, sigma, rps, periods, incs, as_, rss,
                        u1s, u2s, eccs, argps, companion_flux_ratios, mask,
                        companion_is_host=False, exptime=0.00139, nsamples=20,
                        force_serial=False):
    n = len(rps)
    result = np.full(n, np.inf)
    result[mask] = 1.0
    return result


def _mock_lnL_eb_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
                    periods, incs, as_, u1s, u2s, eccs, argps,
                    companion_flux_ratios, mask, companion_is_host=False,
                    exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 1.5
    return result


def _mock_lnL_eb_twin_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
                         periods, incs, as_, u1s, u2s, eccs, argps,
                         companion_flux_ratios, mask, companion_is_host=False,
                         exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 2.0
    return result


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
def transit_lc():
    time = np.linspace(-0.1, 0.1, 100)
    flux = np.ones(100)
    flux[40:60] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


@pytest.fixture()
def small_config():
    return Config(n_mc_samples=200, n_best_samples=10)


# ---------------------------------------------------------------------------
# TTP Tests -- identity (no pytransit needed)
# ---------------------------------------------------------------------------

class TestTTPIdentity:
    def test_ttp_scenario_id(self) -> None:
        s = TTPScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.TP

    def test_ttp_is_not_eb(self) -> None:
        s = TTPScenario(FixedLDCCatalog())
        assert not s.is_eb
        assert not s.returns_twin


class _RecordingCatalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def get_coefficients(
        self, band: str, metallicity: float, teff: float, logg: float,
    ) -> LimbDarkeningCoeffs:
        self.calls.append((band, logg))
        return LimbDarkeningCoeffs(u1=0.4, u2=0.2, band=band)


def _derived_logg(stellar_params: StellarParameters) -> float:
    return float(np.log10(
        CONST.G * (stellar_params.mass_msun * CONST.Msun)
        / (stellar_params.radius_rsun * CONST.Rsun) ** 2
    ))


class TestTargetLDCParity:
    @pytest.mark.parametrize("scenario_cls", [TTPScenario, TEBScenario])
    def test_target_ldc_uses_mass_radius_logg(
        self, scenario_cls, stellar_params,
    ) -> None:
        catalog = _RecordingCatalog()
        adjusted = StellarParameters(
            mass_msun=stellar_params.mass_msun,
            radius_rsun=stellar_params.radius_rsun,
            teff_k=stellar_params.teff_k,
            logg=3.1,
            metallicity_dex=stellar_params.metallicity_dex,
            parallax_mas=stellar_params.parallax_mas,
        )
        scenario = scenario_cls(catalog)

        scenario._get_host_ldc(adjusted, "TESS", np.full(5, 5.0), {})

        assert catalog.calls[0][1] == pytest.approx(_derived_logg(adjusted))

    def test_external_lc_ldc_uses_mass_radius_logg(self, stellar_params) -> None:
        catalog = _RecordingCatalog()
        adjusted = StellarParameters(
            mass_msun=stellar_params.mass_msun,
            radius_rsun=stellar_params.radius_rsun,
            teff_k=stellar_params.teff_k,
            logg=3.1,
            metallicity_dex=stellar_params.metallicity_dex,
            parallax_mas=stellar_params.parallax_mas,
        )
        scenario = TTPScenario(catalog)
        ext_lc = ExternalLightCurve(
            light_curve=LightCurve(
                time_days=np.array([-0.01, 0.01]),
                flux=np.array([1.0, 0.999]),
                flux_err=0.001,
            ),
            band="i",
        )

        scenario._resolve_external_lc_ldcs([ext_lc], adjusted)

        assert catalog.calls[0] == ("i", pytest.approx(_derived_logg(adjusted)))


class TestTTPSamplePriors:
    def test_sample_priors_returns_expected_keys(self, stellar_params) -> None:
        s = TTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        for key in ("rps", "incs", "eccs", "argps", "P_orb"):
            assert key in samples
        assert samples["rps"].shape == (50,)

    def test_sample_priors_planet_radii_positive(self, stellar_params) -> None:
        s = TTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=100, n_best_samples=10)
        P_orb = np.full(100, 5.0)
        samples = s._sample_priors(100, stellar_params, P_orb, cfg)
        assert np.all(samples["rps"] > 0)


class TestTTPGeometry:
    def test_compute_orbital_geometry_keys(self, stellar_params) -> None:
        s = TTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        geom = s._compute_orbital_geometry(samples, P_orb, stellar_params, cfg)
        for key in ("a", "Ptra", "b", "coll"):
            assert key in geom
        assert geom["a"].shape == (50,)


# ---------------------------------------------------------------------------
# TTP Tests -- full compute with mocked lnL (no pytransit needed)
# ---------------------------------------------------------------------------

class TestTTPCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert not isinstance(result, tuple)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_compute_result_has_finite_lnz(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert result.ln_evidence > -np.inf

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_compute_result_planet_radius_nonzero(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert np.any(result.planet_radius_rearth > 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_compute_result_eb_fields_are_zero(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        np.testing.assert_array_equal(result.eb_mass_msun, 0)
        np.testing.assert_array_equal(result.eb_radius_rsun, 0)
        np.testing.assert_array_equal(result.flux_ratio_eb_tess, 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_result_arrays_have_correct_length(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        n = small_config.n_best_samples
        assert len(result.host_mass_msun) == n
        assert len(result.period_days) == n
        assert len(result.planet_radius_rearth) == n

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_ttp_scenario_id_in_result(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TTPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == ScenarioID.TP


# ---------------------------------------------------------------------------
# TEB Tests -- identity (no pytransit needed)
# ---------------------------------------------------------------------------

class TestTEBIdentity:
    def test_teb_scenario_id(self) -> None:
        s = TEBScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.EB

    def test_teb_is_eb(self) -> None:
        s = TEBScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin


class TestTEBSamplePriors:
    def test_sample_priors_returns_expected_keys(self, stellar_params) -> None:
        s = TEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        for key in ("qs", "masses", "radii", "fluxratios"):
            assert key in samples
        assert samples["qs"].shape == (50,)

    def test_sample_priors_masses_positive(self, stellar_params) -> None:
        s = TEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=100, n_best_samples=10)
        P_orb = np.full(100, 5.0)
        samples = s._sample_priors(100, stellar_params, P_orb, cfg)
        assert np.all(samples["masses"] > 0)


class TestTEBGeometry:
    def test_compute_orbital_geometry_has_twin_keys(self, stellar_params) -> None:
        s = TEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        geom = s._compute_orbital_geometry(samples, P_orb, stellar_params, cfg)
        for key in ("a_twin", "Ptra_twin", "b_twin", "coll_twin"):
            assert key in geom


# ---------------------------------------------------------------------------
# TEB Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestTEBCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_compute_primary_has_eb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        primary, _twin = result
        assert primary.scenario_id == ScenarioID.EB

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_compute_twin_has_ebx2p_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        _primary, twin = result
        assert twin.scenario_id == ScenarioID.EBX2P

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_compute_twin_planet_radius_is_zero(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        _primary, twin = result
        np.testing.assert_array_equal(twin.planet_radius_rearth, 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_primary_has_eb_mass_nonzero(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        primary, _twin = result
        assert np.any(primary.eb_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_teb_result_arrays_have_correct_length(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = TEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        primary, twin = result
        n = small_config.n_best_samples
        assert len(primary.host_mass_msun) == n
        assert len(twin.host_mass_msun) == n
