"""Parity tests: verify new deterministic helpers match the original code exactly.

These tests import functions from both the original triceratops repo and the
new refactored code, run them with identical inputs (fixed seed), and assert
exact numerical equality.

Tolerance specification (from BRIEFING.md):
  - Deterministic helpers: exact match (no randomness involved)
  - Prior sampling (with same uniform inputs): exact match
  - Transit model: skipped (pytransit incompatible with numpy 2.x)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make original repo importable under a namespaced alias
_ORIGINAL_REPO = Path("/Users/collier/projects/reference_repos/TRICERATOPS-plus")

# ---------------------------------------------------------------------------
# Import helpers -- original code uses astropy constants which may differ
# from our hardcoded CONST. We test the sampling functions which are
# purely mathematical (no physical constants involved).
# ---------------------------------------------------------------------------


def _import_original_priors():
    """Import original priors module by temporarily hijacking sys.modules.

    The original priors.py uses relative imports (from .funcs import ...) so
    we need to set up the package properly. We save/restore sys.modules and
    sys.path to avoid polluting the test environment.
    """
    import importlib

    original_path = str(_ORIGINAL_REPO)
    priors_path = _ORIGINAL_REPO / "triceratops" / "priors.py"
    if not priors_path.exists():
        pytest.skip(f"Original priors.py not found at {priors_path}")

    # Save modules that we'll temporarily overwrite
    saved_modules = {}
    for key in list(sys.modules):
        if key == "triceratops" or key.startswith("triceratops."):
            saved_modules[key] = sys.modules.pop(key)

    sys.path.insert(0, original_path)
    try:
        mod = importlib.import_module("triceratops.priors")
        return mod
    except ImportError as exc:
        pytest.skip(f"Could not import original priors: {exc}")
    finally:
        # Remove the original path
        if original_path in sys.path:
            sys.path.remove(original_path)
        # Clean up any original modules we loaded
        for key in list(sys.modules):
            if key == "triceratops" or key.startswith("triceratops."):
                del sys.modules[key]
        # Restore our modules
        sys.modules.update(saved_modules)


# ---------------------------------------------------------------------------
# Prior sampling parity tests
# ---------------------------------------------------------------------------


class TestSamplePlanetRadiusParity:
    """sample_planet_radius (new) vs sample_rp (original): exact match."""

    @pytest.mark.parametrize("host_mass", [0.3, 0.45, 0.5, 1.0, 1.5])
    def test_non_flat_prior(self, host_mass: float) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_planet_radius

        np.random.seed(42)
        u = np.random.rand(1000)

        expected = orig.sample_rp(u.copy(), host_mass, False)
        actual = sample_planet_radius(u.copy(), host_mass, flat=False)

        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"sample_planet_radius mismatch for M_s={host_mass}",
        )

    def test_flat_prior(self) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_planet_radius

        np.random.seed(99)
        u = np.random.rand(500)

        expected = orig.sample_rp(u.copy(), 1.0, True)
        actual = sample_planet_radius(u.copy(), 1.0, flat=True)

        np.testing.assert_array_equal(actual, expected)


class TestSampleInclinationParity:
    """sample_inclination (new) vs sample_inc (original): exact match."""

    def test_default_bounds(self) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_inclination

        np.random.seed(42)
        u = np.random.rand(1000)

        expected = orig.sample_inc(u.copy())
        actual = sample_inclination(u.copy())

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("lower,upper", [(0, 45), (30, 90), (10, 80)])
    def test_custom_bounds(self, lower: float, upper: float) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_inclination

        np.random.seed(7)
        u = np.random.rand(500)

        expected = orig.sample_inc(u.copy(), lower, upper)
        actual = sample_inclination(u.copy(), lower, upper)

        np.testing.assert_array_equal(actual, expected)


class TestSampleArgPeriastronParity:
    """sample_arg_periastron (new) vs sample_w (original): exact match."""

    def test_exact_match(self) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_arg_periastron

        np.random.seed(42)
        u = np.random.rand(1000)

        expected = orig.sample_w(u.copy())
        actual = sample_arg_periastron(u.copy())

        np.testing.assert_array_equal(actual, expected)


class TestSampleMassRatioParity:
    """sample_mass_ratio (new) vs sample_q (original): exact match."""

    @pytest.mark.parametrize("M_s", [0.05, 0.2, 0.5, 1.0, 2.0])
    def test_parity(self, M_s: float) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_mass_ratio

        np.random.seed(42)
        u = np.random.rand(1000)

        expected = orig.sample_q(u.copy(), M_s)
        actual = sample_mass_ratio(u.copy(), M_s)

        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"sample_mass_ratio mismatch for M_s={M_s}",
        )


class TestSampleCompanionMassRatioParity:
    """sample_companion_mass_ratio (new) vs sample_q_companion (original)."""

    @pytest.mark.parametrize("M_s", [0.05, 0.2, 0.5, 1.0, 2.0])
    def test_parity(self, M_s: float) -> None:
        orig = _import_original_priors()
        from triceratops.priors.sampling import sample_companion_mass_ratio

        np.random.seed(42)
        u = np.random.rand(1000)

        expected = orig.sample_q_companion(u.copy(), M_s)
        actual = sample_companion_mass_ratio(u.copy(), M_s)

        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"sample_companion_mass_ratio mismatch for M_s={M_s}",
        )


# ---------------------------------------------------------------------------
# Geometry parity tests
# ---------------------------------------------------------------------------


class TestSemiMajorAxisParity:
    """semi_major_axis uses Kepler's 3rd law -- compare against direct calc."""

    def test_known_value(self) -> None:
        """Earth-Sun system: P=365.25 days, M=1 Msun => a ~ 1 AU."""
        from triceratops.likelihoods.geometry import semi_major_axis

        a = semi_major_axis(np.array([365.25]), 1.0)
        au_cm = 1.496e13  # 1 AU in cm
        np.testing.assert_allclose(a[0], au_cm, rtol=1e-3)

    def test_vectorized(self) -> None:
        """Multiple periods produce consistent scaling (a ~ P^(2/3))."""
        from triceratops.likelihoods.geometry import semi_major_axis

        periods = np.array([1.0, 8.0])
        a = semi_major_axis(periods, 1.0)
        # a2/a1 = (P2/P1)^(2/3) = 8^(2/3) = 4
        np.testing.assert_allclose(a[1] / a[0], 4.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Transit model parity (skipped if pytransit unavailable)
# ---------------------------------------------------------------------------

_pytransit_ok = True
try:
    from pytransit import QuadraticModel  # noqa: F401
except ImportError:
    _pytransit_ok = False


@pytest.mark.skipif(not _pytransit_ok, reason="pytransit incompatible with numpy 2.x")
class TestTransitModelParity:
    """simulate_planet_transit (new) vs simulate_TP_transit (original)."""

    def test_planet_transit_exact(self) -> None:
        original_path = str(_ORIGINAL_REPO)
        if original_path not in sys.path:
            sys.path.insert(0, original_path)
        try:
            from triceratops.likelihoods import simulate_TP_transit  # type: ignore[attr-defined]
        except ImportError:
            pytest.skip("Original likelihoods not importable")
        finally:
            if original_path in sys.path:
                sys.path.remove(original_path)

        from triceratops.likelihoods.transit_model import simulate_planet_transit

        time = np.linspace(-0.05, 0.05, 100)
        kwargs = {
            "rp": 2.0,
            "period": 5.0,
            "inc": 89.0,
            "a": 1e12,
            "rs": 1.0,
            "u1": 0.4,
            "u2": 0.2,
            "ecc": 0.0,
            "argp": 90.0,
        }

        orig_flux = simulate_TP_transit(
            time, kwargs["rp"], kwargs["period"], kwargs["inc"],
            kwargs["a"], kwargs["rs"], kwargs["u1"], kwargs["u2"],
            kwargs["ecc"], kwargs["argp"],
        )
        new_flux = simulate_planet_transit(
            time, rp=2.0, period=5.0, inc=89.0, a=1e12, rs=1.0,
            u1=0.4, u2=0.2, ecc=0.0, argp=90.0,
        )

        np.testing.assert_array_almost_equal(new_flux, orig_flux, decimal=12)

    def test_eb_transit_exact(self) -> None:
        original_path = str(_ORIGINAL_REPO)
        if original_path not in sys.path:
            sys.path.insert(0, original_path)
        try:
            from triceratops.likelihoods import simulate_EB_transit  # type: ignore[attr-defined]
        except ImportError:
            pytest.skip("Original likelihoods not importable")
        finally:
            if original_path in sys.path:
                sys.path.remove(original_path)

        from triceratops.likelihoods.transit_model import simulate_eb_transit

        time = np.linspace(-0.05, 0.05, 100)

        orig_flux, orig_sec = simulate_EB_transit(
            time, 0.5, 0.3, 5.0, 89.0, 1e12, 1.0, 0.4, 0.2, 0.0, 90.0,
        )
        new_flux, new_sec = simulate_eb_transit(
            time, rs=1.0, rcomp=0.5, eb_flux_ratio=0.3, period=5.0,
            inc=89.0, a=1e12, u1=0.4, u2=0.2, ecc=0.0, argp=90.0,
        )

        np.testing.assert_array_almost_equal(new_flux, orig_flux, decimal=12)
        np.testing.assert_almost_equal(new_sec, orig_sec, decimal=12)
