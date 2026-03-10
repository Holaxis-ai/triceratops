"""Unit tests for the _eval_eb_lnL module-level helper in nearby_scenarios.

Tests cover:
  - Standard branch gets standard collision mask (q<0.95 uses geometry["coll"])
  - Twin branch gets twin collision mask (q>=0.95 uses geometry["coll_twin"])
  - extra_mask=None passes through without error
  - extra_mask with some False values excludes those draws
  - Output shape matches input sample count
  - lnL=-inf for draws excluded by both masks (no transit geometry)
  - Returned tuple is (lnL, lnL_twin) both of length N
  - extra_mask all-False yields all -inf outputs
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from triceratops.domain.entities import LightCurve
from triceratops.scenarios.nearby_scenarios import _eval_eb_lnL

_MOD = "triceratops.scenarios.nearby_scenarios"

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_light_curve(n_pts: int = 50) -> LightCurve:
    time = np.linspace(-0.05, 0.05, n_pts)
    flux = np.ones(n_pts)
    flux[20:30] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


def _make_samples(N: int, *, q_values=None) -> dict:
    if q_values is None:
        q_values = np.full(N, 0.5)  # all standard branch by default
    return {
        "qs": np.asarray(q_values, dtype=float),
        "incs": np.full(N, 89.0),
        "eccs": np.zeros(N),
        "argps": np.zeros(N),
        "P_orb": np.full(N, 5.0),
        "radii": np.full(N, 0.5),
        "fluxratios": np.full(N, 0.3),
    }


def _make_geometry(N: int, *, coll_all: bool = False, coll_twin_all: bool = False):
    """Geometry dict for both standard and twin branches."""
    ptra = np.full(N, 0.1)   # small: arccos(0.1) > 84 deg, so inc=89 passes
    coll = np.full(N, coll_all, dtype=bool)
    ptra_twin = np.full(N, 0.08)
    coll_twin = np.full(N, coll_twin_all, dtype=bool)
    a = np.full(N, 0.05)
    a_twin = np.full(N, 0.063)
    b = np.zeros(N)
    b_twin = np.zeros(N)
    return {
        "Ptra": ptra, "coll": coll,
        "Ptra_twin": ptra_twin, "coll_twin": coll_twin,
        "a": a, "a_twin": a_twin,
        "b": b, "b_twin": b_twin,
    }


def _mock_eb_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
               periods, incs, as_, u1s, u2s, eccs, argps,
               companion_flux_ratios, mask, companion_is_host=False,
               exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 1.5
    return result


def _mock_eb_twin_p(*, time, flux, sigma, rss, rcomps, eb_flux_ratios,
                    periods, incs, as_, u1s, u2s, eccs, argps,
                    companion_flux_ratios, mask, companion_is_host=False,
                    exptime=0.00139, nsamples=20, force_serial=False):
    n = len(rss)
    result = np.full(n, np.inf)
    result[mask] = 2.0
    return result


# ---------------------------------------------------------------------------
# Test: output shape matches input sample count
# ---------------------------------------------------------------------------

class TestOutputShape:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_output_shape_matches_N(self, _m1, _m2) -> None:
        """Both lnL and lnL_twin must have shape (N,)."""
        N = 100
        lc = _make_light_curve()
        samples = _make_samples(N)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert lnL.shape == (N,), f"Expected ({N},), got {lnL.shape}"
        assert lnL_twin.shape == (N,), f"Expected ({N},), got {lnL_twin.shape}"

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_different_N_shape(self, _m1, _m2) -> None:
        """Shape test with a different N."""
        N = 37
        lc = _make_light_curve()
        samples = _make_samples(N)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert lnL.shape == (N,)
        assert lnL_twin.shape == (N,)


# ---------------------------------------------------------------------------
# Test: standard branch gets standard collision mask
# ---------------------------------------------------------------------------

class TestStandardBranchMask:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_standard_branch_uses_coll_not_coll_twin(self, _m1, _m2) -> None:
        """q<0.95 draws with coll=False (coll_twin=True) must get finite lnL."""
        N = 50
        lc = _make_light_curve()
        samples = _make_samples(N, q_values=np.full(N, 0.5))  # all standard
        geometry = _make_geometry(N, coll_all=False, coll_twin_all=True)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        # Standard branch should have finite lnL (coll=False passes)
        assert np.any(np.isfinite(lnL)), "Standard branch should produce finite lnL"
        # Twin branch: all coll_twin=True, so all excluded → all -inf
        assert np.all(lnL_twin == -np.inf), "Twin should be -inf when coll_twin all True"

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_standard_branch_excluded_by_coll(self, _m1, _m2) -> None:
        """q<0.95 draws with coll=True must be excluded (lnL=-inf)."""
        N = 50
        lc = _make_light_curve()
        samples = _make_samples(N, q_values=np.full(N, 0.5))  # all standard
        geometry = _make_geometry(N, coll_all=True, coll_twin_all=False)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert np.all(lnL == -np.inf), "Colliding standard-branch draws must be -inf"


# ---------------------------------------------------------------------------
# Test: twin branch gets twin collision mask
# ---------------------------------------------------------------------------

class TestTwinBranchMask:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_twin_branch_uses_coll_twin(self, _m1, _m2) -> None:
        """q>=0.95 draws with coll_twin=False must get finite lnL_twin."""
        N = 50
        lc = _make_light_curve()
        samples = _make_samples(N, q_values=np.full(N, 0.99))  # all twin
        geometry = _make_geometry(N, coll_all=True, coll_twin_all=False)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert np.all(lnL == -np.inf), "Standard branch should be -inf for q>=0.95"
        assert np.any(np.isfinite(lnL_twin)), "Twin branch should have finite lnL_twin"

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_twin_branch_excluded_by_coll_twin(self, _m1, _m2) -> None:
        """q>=0.95 draws with coll_twin=True must be excluded (lnL_twin=-inf)."""
        N = 50
        lc = _make_light_curve()
        samples = _make_samples(N, q_values=np.full(N, 0.99))  # all twin
        geometry = _make_geometry(N, coll_all=False, coll_twin_all=True)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert np.all(lnL_twin == -np.inf), "Colliding twin-branch draws must be -inf"


# ---------------------------------------------------------------------------
# Test: extra_mask=None passes through
# ---------------------------------------------------------------------------

class TestExtraMaskNone:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_extra_mask_none_does_not_raise(self, _m1, _m2) -> None:
        """extra_mask=None (default) must not raise any error."""
        N = 30
        lc = _make_light_curve()
        samples = _make_samples(N)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
            extra_mask=None,
        )
        assert lnL.shape == (N,)
        assert lnL_twin.shape == (N,)

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_extra_mask_none_same_as_omitted(self, _m1, _m2) -> None:
        """Passing extra_mask=None must give same result as not passing it."""
        N = 40
        lc = _make_light_curve()
        samples = _make_samples(N)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL_none, lnL_twin_none = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True, extra_mask=None,
        )
        lnL_omit, lnL_twin_omit = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        np.testing.assert_array_equal(lnL_none, lnL_omit)
        np.testing.assert_array_equal(lnL_twin_none, lnL_twin_omit)


# ---------------------------------------------------------------------------
# Test: extra_mask with some False values excludes those draws
# ---------------------------------------------------------------------------

class TestExtraMaskFiltering:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_extra_mask_false_excludes_draws(self, _m1, _m2) -> None:
        """Draws with extra_mask=False must be excluded from both branches."""
        N = 20
        lc = _make_light_curve()
        # Mix of q values
        qs = np.concatenate([np.full(10, 0.5), np.full(10, 0.99)])
        samples = _make_samples(N, q_values=qs)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        # Block the first 5 and last 5
        extra_mask = np.ones(N, dtype=bool)
        extra_mask[:5] = False
        extra_mask[-5:] = False

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True, extra_mask=extra_mask,
        )
        # First 5 draws (q<0.95) blocked by extra_mask → lnL=-inf
        assert np.all(lnL[:5] == -np.inf), "Draws blocked by extra_mask must be -inf"
        # Last 5 draws (q>=0.95) blocked by extra_mask → lnL_twin=-inf
        assert np.all(lnL_twin[-5:] == -np.inf), "Draws blocked by extra_mask must be -inf"

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_extra_mask_all_false_yields_all_neg_inf(self, _m1, _m2) -> None:
        """All-False extra_mask must yield all -inf in both branches."""
        N = 25
        lc = _make_light_curve()
        qs = np.concatenate([np.full(12, 0.5), np.full(13, 0.99)])
        samples = _make_samples(N, q_values=qs)
        geometry = _make_geometry(N)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)
        extra_mask = np.zeros(N, dtype=bool)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True, extra_mask=extra_mask,
        )
        assert np.all(lnL == -np.inf), "All-False extra_mask → all -inf in standard branch"
        assert np.all(lnL_twin == -np.inf), "All-False extra_mask → all -inf in twin branch"


# ---------------------------------------------------------------------------
# Test: lnL=-inf for draws excluded by both masks (no transit geometry)
# ---------------------------------------------------------------------------

class TestExcludedDraws:
    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_non_transiting_geometry_yields_neg_inf(self, _m1, _m2) -> None:
        """Draws that fail transit geometry (ptra>1 + inc<90) must be -inf."""
        N = 30
        lc = _make_light_curve()
        qs = np.concatenate([np.full(15, 0.5), np.full(15, 0.99)])
        samples = _make_samples(N, q_values=qs)
        # ptra > 1 means inc_min = 90, but we use inc=89 → excluded
        geometry = _make_geometry(N)
        geometry["Ptra"] = np.full(N, 1.5)      # standard branch excluded
        geometry["Ptra_twin"] = np.full(N, 1.5)  # twin branch excluded
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert np.all(lnL == -np.inf), "Non-transiting standard branch must be -inf"
        assert np.all(lnL_twin == -np.inf), "Non-transiting twin branch must be -inf"

    @patch(f"{_MOD}.lnL_eb_twin_p", side_effect=_mock_eb_twin_p)
    @patch(f"{_MOD}.lnL_eb_p", side_effect=_mock_eb_p)
    def test_both_collision_flags_yields_neg_inf(self, _m1, _m2) -> None:
        """Draws that collide in both branches must yield -inf."""
        N = 20
        lc = _make_light_curve()
        qs = np.concatenate([np.full(10, 0.5), np.full(10, 0.99)])
        samples = _make_samples(N, q_values=qs)
        geometry = _make_geometry(N, coll_all=True, coll_twin_all=True)
        rss = np.full(N, 1.0)
        u1s = np.full(N, 0.4)
        u2s = np.full(N, 0.2)

        lnL, lnL_twin = _eval_eb_lnL(
            lc, lnsigma=0.0, samples=samples, geometry=geometry,
            rss=rss, u1s=u1s, u2s=u2s,
            force_serial=True,
        )
        assert np.all(lnL == -np.inf)
        assert np.all(lnL_twin == -np.inf)
