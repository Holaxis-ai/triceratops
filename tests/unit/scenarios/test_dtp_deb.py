"""Tests for DTPScenario and DEBScenario."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve
from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.population.protocols import TRILEGALResult
from triceratops.priors.sampling import (
    sample_arg_periastron,
    sample_eccentricity,
    sample_inclination,
    sample_mass_ratio,
)
from triceratops.scenarios.background_scenarios import (
    DEBScenario,
    DTPScenario,
    _compute_delta_mags_map,
    _compute_fluxratios_comp,
    _resolve_sdss_target_mags,
    _sample_population_indices,
)
from triceratops.stellar.relations import StellarRelations

_LNL_MOD = "triceratops.scenarios.background_scenarios"


# ---------------------------------------------------------------------------
# Mock lnL functions (pytransit not usable under numpy 2.x)
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


@pytest.fixture()
def mock_population():
    """Small TRILEGAL population for testing."""
    n = 50
    rng = np.random.default_rng(42)
    return TRILEGALResult(
        tmags=rng.uniform(10.0, 16.0, n),
        masses=rng.uniform(0.3, 1.5, n),
        loggs=rng.uniform(3.5, 5.0, n),
        teffs=rng.uniform(3500, 7000, n),
        metallicities=rng.uniform(-1.0, 0.5, n),
        jmags=rng.uniform(9.0, 15.0, n),
        hmags=rng.uniform(8.5, 14.5, n),
        kmags=rng.uniform(8.0, 14.0, n),
        gmags=rng.uniform(10.5, 16.5, n),
        rmags=rng.uniform(10.0, 16.0, n),
        imags=rng.uniform(9.5, 15.5, n),
        zmags=rng.uniform(9.0, 15.0, n),
    )


@pytest.fixture()
def host_mags():
    return {
        "tmag": 10.0, "jmag": 9.5, "hmag": 9.0, "kmag": 8.8,
        "gmag": 10.5, "rmag": 10.0, "imag": 9.5, "zmag": 9.0,
    }


# ---------------------------------------------------------------------------
# _compute_delta_mags_map tests -- BUG-04
# ---------------------------------------------------------------------------

class TestDeltaMagsMap:
    def test_k_band_bug04_fixed(self, mock_population) -> None:
        """BUG-04: delta_Kmags must use K-band, not H-band."""
        result = _compute_delta_mags_map(
            target_tmag=10.0, target_jmag=9.5,
            target_hmag=9.0, target_kmag=8.8,
            population=mock_population,
        )
        expected_k = 8.8 - mock_population.kmags
        expected_h = 9.0 - mock_population.hmags
        np.testing.assert_array_equal(result["delta_Kmags"], expected_k)
        # Confirm K != H (they would be equal under the original bug)
        assert not np.array_equal(result["delta_Kmags"], expected_h)

    def test_correct_subtraction(self, mock_population) -> None:
        """Spot-check: delta_TESSmags = target - population."""
        result = _compute_delta_mags_map(
            target_tmag=10.0, target_jmag=9.5,
            target_hmag=9.0, target_kmag=8.8,
            population=mock_population,
        )
        expected = 10.0 - mock_population.tmags
        np.testing.assert_array_equal(result["delta_TESSmags"], expected)

    def test_all_bands_present(self, mock_population) -> None:
        result = _compute_delta_mags_map(
            target_tmag=10.0, target_jmag=9.5,
            target_hmag=9.0, target_kmag=8.8,
            population=mock_population,
        )
        for key in ("delta_TESSmags", "delta_Jmags", "delta_Hmags", "delta_Kmags"):
            assert key in result
            assert result[key].shape == (mock_population.n_stars,)


# ---------------------------------------------------------------------------
# DTP Tests -- identity
# ---------------------------------------------------------------------------

class TestDTPIdentity:
    def test_dtp_scenario_id(self) -> None:
        s = DTPScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.DTP

    def test_dtp_is_not_eb(self) -> None:
        s = DTPScenario(FixedLDCCatalog())
        assert not s.is_eb
        assert not s.returns_twin


class TestDTPRaisesWithoutTrilegal:
    def test_dtp_raises_without_trilegal(self, stellar_params, small_config) -> None:
        s = DTPScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="trilegal_population"):
            s._sample_priors(10, stellar_params, P_orb, small_config)


class TestDTPSamplePriors:
    def test_sample_priors_returns_expected_keys(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        for key in ("rps", "incs", "eccs", "argps", "idxs",
                     "fluxratios_comp", "lnprior_companion"):
            assert key in samples
        assert samples["rps"].shape == (50,)


class TestBackgroundLegacyParity:
    def test_resolve_sdss_target_mags_estimates_missing_griz_for_sdss_followup(
        self,
    ) -> None:
        host_mags = {
            "bmag": 11.5,
            "vmag": 11.0,
            "jmag": 9.5,
            "gmag": np.nan,
            "rmag": np.nan,
            "imag": np.nan,
            "zmag": np.nan,
        }
        expected = StellarRelations().estimate_sdss_magnitudes(11.5, 11.0, 9.5)

        gmag, rmag, imag, zmag = _resolve_sdss_target_mags(
            host_mags, ("i",), None,
        )

        assert gmag == pytest.approx(expected["g"])
        assert rmag == pytest.approx(expected["r"])
        assert imag == pytest.approx(expected["i"])
        assert zmag == pytest.approx(expected["z"])

    @pytest.mark.parametrize("scenario_cls", [DTPScenario, DEBScenario])
    def test_d_scenarios_keep_original_population_draw_range(
        self, scenario_cls, stellar_params, host_mags,
    ) -> None:
        population = TRILEGALResult(
            tmags=np.array([12.0, 13.0]),
            masses=np.array([0.8, 0.9]),
            loggs=np.array([4.4, 4.5]),
            teffs=np.array([5200.0, 5400.0]),
            metallicities=np.array([0.0, 0.1]),
            jmags=np.array([11.0, 12.0]),
            hmags=np.array([10.5, 11.5]),
            kmags=np.array([10.2, 11.2]),
            gmags=np.array([12.4, 13.4]),
            rmags=np.array([11.8, 12.8]),
            imags=np.array([11.5, 12.5]),
            zmags=np.array([11.3, 12.3]),
        )
        scenario = scenario_cls(FixedLDCCatalog())
        cfg = Config(n_mc_samples=32, n_best_samples=10)
        P_orb = np.full(32, 5.0)

        np.random.seed(7)
        samples = scenario._sample_priors(
            32, stellar_params, P_orb, cfg,
            trilegal_population=population,
            host_magnitudes=host_mags,
        )

        np.testing.assert_array_equal(samples["idxs"], np.zeros(32, dtype=int))

    def test_deb_preserves_original_rng_draw_order(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        scenario = DEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=24, n_best_samples=10)
        P_orb = np.full(24, 5.0)

        np.random.seed(123)
        expected_incs = sample_inclination(np.random.rand(24))
        expected_qs = sample_mass_ratio(np.random.rand(24), stellar_params.mass_msun)
        expected_eccs = sample_eccentricity(
            np.random.rand(24), planet=False, period=float(np.mean(P_orb)),
        )
        expected_argps = sample_arg_periastron(np.random.rand(24))
        expected_idxs = _sample_population_indices(
            mock_population.n_stars, 24, legacy_exclude_last=True,
        )

        np.random.seed(123)
        samples = scenario._sample_priors(
            24, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )

        np.testing.assert_allclose(samples["incs"], expected_incs)
        np.testing.assert_allclose(samples["qs"], expected_qs)
        np.testing.assert_allclose(samples["eccs"], expected_eccs)
        np.testing.assert_allclose(samples["argps"], expected_argps)
        np.testing.assert_array_equal(samples["idxs"], expected_idxs)


# ---------------------------------------------------------------------------
# DTP Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestDTPCompute:
    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_compute_returns_single_result(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert not isinstance(result, tuple)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_compute_result_has_finite_lnz(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert result.ln_evidence > -np.inf

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_result_has_companion_mass(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert np.any(result.companion_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_result_has_companion_flux_ratio(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert np.any(result.flux_ratio_companion_tess > 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_result_eb_fields_are_zero(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        np.testing.assert_array_equal(result.eb_mass_msun, 0)
        np.testing.assert_array_equal(result.eb_radius_rsun, 0)

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_scenario_id_in_result(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        assert result.scenario_id == ScenarioID.DTP

    @patch(f"{_LNL_MOD}.lnL_planet_p", side_effect=_mock_lnL_planet_p)
    def test_dtp_result_arrays_have_correct_length(
        self, _mock, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DTPScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, ScenarioResult)
        n = small_config.n_best_samples
        assert len(result.host_mass_msun) == n
        assert len(result.companion_mass_msun) == n


# ---------------------------------------------------------------------------
# DEB Tests -- identity
# ---------------------------------------------------------------------------

class TestDEBIdentity:
    def test_deb_scenario_id(self) -> None:
        s = DEBScenario(FixedLDCCatalog())
        assert s.scenario_id == ScenarioID.DEB

    def test_deb_is_eb(self) -> None:
        s = DEBScenario(FixedLDCCatalog())
        assert s.is_eb
        assert s.returns_twin


class TestDEBRaisesWithoutTrilegal:
    def test_deb_raises_without_trilegal(self, stellar_params, small_config) -> None:
        s = DEBScenario(FixedLDCCatalog())
        P_orb = np.full(10, 5.0)
        with pytest.raises(ValueError, match="trilegal_population"):
            s._sample_priors(10, stellar_params, P_orb, small_config)


class TestDEBSamplePriors:
    def test_sample_priors_returns_expected_keys(
        self, stellar_params, mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        cfg = Config(n_mc_samples=50, n_best_samples=10)
        P_orb = np.full(50, 5.0)
        samples = s._sample_priors(
            50, stellar_params, P_orb, cfg,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        for key in ("qs", "masses", "radii", "fluxratios",
                     "idxs", "fluxratios_comp", "lnprior_companion"):
            assert key in samples
        assert samples["qs"].shape == (50,)


# ---------------------------------------------------------------------------
# DEB Tests -- full compute with mocked lnL
# ---------------------------------------------------------------------------

class TestDEBCompute:
    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_compute_returns_tuple(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_compute_primary_has_deb_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        assert primary.scenario_id == ScenarioID.DEB

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_compute_twin_has_debx2p_id(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        _primary, twin = result
        assert twin.scenario_id == ScenarioID.DEBX2P

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_primary_has_eb_mass(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        assert np.any(primary.eb_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_primary_has_companion_mass(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, _twin = result
        assert np.any(primary.companion_mass_msun > 0)

    @patch(f"{_LNL_MOD}.lnL_eb_twin_p", side_effect=_mock_lnL_eb_twin_p)
    @patch(f"{_LNL_MOD}.lnL_eb_p", side_effect=_mock_lnL_eb_p)
    def test_deb_result_arrays_have_correct_length(
        self, _m1, _m2, transit_lc, stellar_params, small_config,
        mock_population, host_mags,
    ) -> None:
        s = DEBScenario(FixedLDCCatalog())
        result = s.compute(
            transit_lc, stellar_params, 5.0, small_config,
            trilegal_population=mock_population,
            host_magnitudes=host_mags,
        )
        assert isinstance(result, tuple)
        primary, twin = result
        n = small_config.n_best_samples
        assert len(primary.host_mass_msun) == n
        assert len(twin.host_mass_msun) == n


# ---------------------------------------------------------------------------
# _compute_fluxratios_comp test
# ---------------------------------------------------------------------------

class TestFluxRatiosComp:
    def test_fluxratios_bounded(self) -> None:
        delta = np.array([-2.0, 0.0, 2.0, 5.0])
        fr = _compute_fluxratios_comp(delta)
        assert np.all(fr > 0)
        assert np.all(fr < 1)

    def test_positive_delta_gives_high_ratio(self) -> None:
        # Positive delta => background fainter => target dominates => high ratio
        fr = _compute_fluxratios_comp(np.array([5.0]))
        assert fr[0] > 0.9
