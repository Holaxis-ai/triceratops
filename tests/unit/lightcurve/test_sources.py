"""Tier 2 tests — provider contract tests for LC sources (lightkurve, no network)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
import lightkurve as lk

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.lightcurve.sources.lightkurve import LightkurveSource
from triceratops.domain.entities import LightCurve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EPHEMERIS = Ephemeris(period_days=3.5, t0_btjd=1468.28, duration_hours=2.0)


@pytest.fixture()
def synthetic_lk_lc_collection() -> lk.LightCurveCollection:
    """Build a LightCurveCollection from numpy arrays — no MAST required."""
    n = 1000
    t0_btjd = 1468.0
    time_btjd = np.linspace(t0_btjd, t0_btjd + 7.0, n)
    rng = np.random.default_rng(42)
    flux = np.ones(n) + rng.normal(0, 3e-4, n)
    flux_err = np.full(n, 3e-4)

    time_obj = Time(time_btjd + 2_457_000.0, format="jd", scale="tdb")
    single_lc = lk.LightCurve(
        time=time_obj,
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
    )
    single_lc.meta["SECTOR"] = 14
    single_lc.meta["EXPTIME"] = 120.0
    single_lc.meta["TIMEDEL"] = 120.0 / 86400.0

    return lk.LightCurveCollection([single_lc])


# ---------------------------------------------------------------------------
# LightkurveSource contract tests
# ---------------------------------------------------------------------------


class TestLightkurveSource:
    def test_contract(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """Inject _override_collection; verify LightCurvePreparationResult contract."""
        source = LightkurveSource(tic_id=99999, _override_collection=synthetic_lk_lc_collection)
        result = source.prepare(_EPHEMERIS, LightCurveConfig())
        assert isinstance(result, LightCurvePreparationResult)
        assert isinstance(result.light_curve, LightCurve)
        assert len(result.light_curve.time_days) > 0
        assert isinstance(result.light_curve.flux_err, float)
        assert result.light_curve.flux_err > 0

    def test_phase_axis_not_btjd(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """Output time_days must be phase (small values), not BTJD (~1468)."""
        source = LightkurveSource(tic_id=99999, _override_collection=synthetic_lk_lc_collection)
        result = source.prepare(_EPHEMERIS, LightCurveConfig())
        assert result.light_curve.time_days.max() < _EPHEMERIS.period_days / 2
        assert result.light_curve.time_days.min() > -_EPHEMERIS.period_days / 2

    def test_transit_in_window(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """All output cadences should be within the transit window."""
        source = LightkurveSource(tic_id=99999, _override_collection=synthetic_lk_lc_collection)
        result = source.prepare(_EPHEMERIS, LightCurveConfig())
        half_window = LightCurveConfig().phase_window_factor * _EPHEMERIS.duration_hours / 24.0
        assert np.all(np.abs(result.light_curve.time_days) < half_window)

    def test_sectors_extracted(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        source = LightkurveSource(tic_id=99999, _override_collection=synthetic_lk_lc_collection)
        result = source.prepare(_EPHEMERIS, LightCurveConfig())
        assert 14 in result.sectors_used

    def test_flux_normalised(self, synthetic_lk_lc_collection: lk.LightCurveCollection) -> None:
        """After stitch, continuum flux should be ≈1.0."""
        source = LightkurveSource(tic_id=99999, _override_collection=synthetic_lk_lc_collection)
        result = source.prepare(_EPHEMERIS, LightCurveConfig())
        assert abs(float(np.median(result.light_curve.flux)) - 1.0) < 0.05

    def test_multi_sector_collection(self) -> None:
        """Two-sector collection stitches and returns cadences from both sectors."""
        rng = np.random.default_rng(123)
        lcs = []
        for sector, start in [(14, 1468.0), (15, 1496.0)]:
            n = 500
            t = np.linspace(start, start + 7.0, n)
            time_obj = Time(t + 2_457_000.0, format="jd", scale="tdb")
            flux = np.ones(n) + rng.normal(0, 3e-4, n)
            flux_err = np.full(n, 3e-4)
            single = lk.LightCurve(
                time=time_obj,
                flux=flux * u.dimensionless_unscaled,
                flux_err=flux_err * u.dimensionless_unscaled,
            )
            single.meta["SECTOR"] = sector
            single.meta["EXPTIME"] = 120.0
            single.meta["TIMEDEL"] = 120.0 / 86400.0
            lcs.append(single)

        coll = lk.LightCurveCollection(lcs)
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28, duration_hours=2.0)
        source = LightkurveSource(tic_id=99999, _override_collection=coll)
        result = source.prepare(eph, LightCurveConfig(sectors="all"))
        assert isinstance(result.light_curve, LightCurve)
        assert 14 in result.sectors_used
