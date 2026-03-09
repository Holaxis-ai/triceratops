"""Tests for STPScenario and SEBScenario."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import ExternalLightCurve, LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.companion_scenarios import SEBScenario, STPScenario

_LNL_MOD = "triceratops.scenarios.companion_scenarios"


# ---------------------------------------------------------------------------
# Mock lnL functions
# ---------------------------------------------------------------------------

def _mock_lnL_planet_p(*, time, flux, sigma, rps, periods, incs, as_, rss,  # noqa: ARG001
                        u1s, u2s, eccs, argps, companion_flux_ratios, mask,
                        companion_is_host=False, exptime=0.00139, nsamples=20,
                        force_serial=False):
    n = len(rps)
    result = np.full(n, np.inf)
    result[mask] = 1.0
    return result


def _mock_lnL_eb_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,  # noqa: ARG001
                    periods, incs, as_, u1s, u2s, eccs, argps,
                    companion_flux_ratios, mask, companion_is_host=False,
                    exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 1.5
    return result


def _mock_lnL_eb_twin_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,  # noqa: ARG001
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
def external_lc():
    time = np.linspace(-0.05, 0.05, 40)
    flux = np.ones(40)
    return ExternalLightCurve(
        light_curve=LightCurve(time_days=time, flux=flux, flux_err=0.002),
        band="J",
        ldc=None,
    )


@pytest.fixture()
def small_config():
    return Config(n_mc_samples=200, n_best_samples=10)


# ---------------------------------------------------------------------------
# STP Tests -- identity
# ---------------------------------------------------------------------------

class TestSTPIdentity:
    def test_stp_scenario_id(self) -> None:
        s = STPScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.STP

    def test_stp_is_not_eb(self) -> None:
        s = STPScenario(FixedLDCCatalog())
        assert not s.is_eb
        assert not s.returns_twin


class TestSTPSamplePriors:
    def test_sample_priors_returns_expected_keys(self, stellar_params) -> None:
        s = STPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        for key in ("rps", "incs", "eccs", "argps", "qs_comp", "masses_comp",
                     "radii_comp", "teffs_comp", "loggs_comp", "fluxratios_comp"):
            assert key in samples
        assert samples["rps"].shape == (50,)

    def test_sample_priors_has_companion_properties(self, stellar_params) -> None:
        s = STPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        assert np.all(samples["masses_comp"] > 0)
        assert np.all(samples["radii_comp"] > 0)


class TestSTPGeometry:
    def test_compute_orbital_geometry_keys(self, stellar_params) -> None:
        s = STPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        geom = s._compute_orbital_geometry(samples, P_orb, stellar_params, cfg)
        for key in ("a", "ptra", "b", "coll"):
            assert key in geom
        assert geom["a"].shape == (50,)

    def test_stp_uses_companion_mass_for_semi_major_axis(self, stellar_params) -> None:
        """Semi-major axis should vary per sample since companion mass varies."""
        s = STPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        geom = s._compute_orbital_geometry(samples, P_orb, stellar_params, cfg)
        # If companion masses differ, semi-major axes should differ
        if not np.all(samples["masses_comp"] == samples["masses_comp"][0]):
            assert not np.all(geom["a"] == geom["a"][0])


# ---------------------------------------------------------------------------
# STP Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestSTPCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, ScenarioResult)
        assert not isinstance(result, tuple)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_compute_result_has_finite_lnz(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert result.ln_evidence > -np.inf

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_scenario_id_in_result(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert result.scenario_id == ScenarioID.STP

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_result_host_is_companion(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        """STP host = companion star, comp = target star."""
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        # Companion mass in result should be target star mass
        np.testing.assert_array_equal(
            result.companion_mass_msun,
            np.full(small_config.n_best_samples, stellar_params.mass_msun),
        )

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_result_eb_fields_are_zero(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        np.testing.assert_array_equal(result.eb_mass_msun, 0)
        np.testing.assert_array_equal(result.eb_radius_rsun, 0)
        np.testing.assert_array_equal(result.flux_ratio_eb_tess, 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_result_arrays_have_correct_length(
        self, _mock, transit_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        n = small_config.n_best_samples
        assert len(result.host_mass_msun) == n
        assert len(result.period_days) == n
        assert len(result.planet_radius_rearth) == n

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_stp_external_lcs_are_accumulated(
        self, mock_lnl, transit_lc, external_lc, stellar_params, small_config,
    ) -> None:
        s = STPScenario(FixedLDCCatalog())
        s.compute(
            transit_lc,
            stellar_params,
            5.0,
            small_config,
            external_lcs=[external_lc],
        )
        assert mock_lnl.call_count == 2


# ---------------------------------------------------------------------------
# SEB Tests -- identity
# ---------------------------------------------------------------------------

class TestSEBIdentity:
    def test_seb_scenario_id(self) -> None:
        s = SEBScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.SEB

    def test_seb_is_eb(self) -> None:
        s = SEBScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin


class TestSEBSamplePriors:
    def test_sample_priors_returns_expected_keys(self, stellar_params) -> None:
        s = SEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        for key in ("qs", "masses_eb", "radii_eb", "fluxratios_eb",
                     "qs_comp", "masses_comp", "radii_comp", "teffs_comp",
                     "loggs_comp", "fluxratios_comp"):
            assert key in samples
        assert samples["qs"].shape == (50,)

    def test_seb_eb_masses_use_companion_mass(self, stellar_params) -> None:
        """EB masses = qs * masses_comp (not qs * M_s)."""
        s = SEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        np.testing.assert_allclose(
            samples["masses_eb"],
            samples["qs"] * samples["masses_comp"],
        )


class TestSEBGeometry:
    def test_compute_orbital_geometry_has_twin_keys(self, stellar_params) -> None:
        s = SEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(50, stellar_params, P_orb, cfg)
        geom = s._compute_orbital_geometry(samples, P_orb, stellar_params, cfg)
        for key in ("a", "a_twin", "ptra", "ptra_twin", "b", "b_twin",
                     "coll", "coll_twin"):
            assert key in geom


# ---------------------------------------------------------------------------
# SEB Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestSEBCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_compute_primary_has_seb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        primary, _twin = result
        assert primary.scenario_id == ScenarioID.SEB

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_compute_twin_has_sebx2p_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        _primary, twin = result
        assert twin.scenario_id == ScenarioID.SEBX2P

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_compute_twin_planet_radius_is_zero(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        _primary, twin = result
        np.testing.assert_array_equal(twin.planet_radius_rearth, 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_result_host_is_companion(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        """SEB host = companion star, comp = target star."""
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        primary, _twin = result
        np.testing.assert_array_equal(
            primary.companion_mass_msun,
            np.full(small_config.n_best_samples, stellar_params.mass_msun),
        )

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_primary_has_eb_mass_nonzero(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        primary, _twin = result
        assert np.any(primary.eb_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_result_arrays_have_correct_length(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        result = s.compute(transit_lc, stellar_params, 5.0, small_config)
        primary, twin = result
        n = small_config.n_best_samples
        assert len(primary.host_mass_msun) == n
        assert len(twin.host_mass_msun) == n

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_seb_external_lcs_are_accumulated(
        self, mock_lnl_eb, mock_lnl_eb_twin, transit_lc, external_lc,
        stellar_params, small_config,
    ) -> None:
        s = SEBScenario(FixedLDCCatalog())
        s.compute(
            transit_lc,
            stellar_params,
            5.0,
            small_config,
            external_lcs=[external_lc],
        )
        assert mock_lnl_eb.call_count == 2
        assert mock_lnl_eb_twin.call_count == 2
