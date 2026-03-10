"""Tier 1 tests for prep.py — prepare_from_raw() full pipeline.

Uses numpy arrays only. No lightkurve imports, no network calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.lightcurve.errors import (
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurvePreparationError,
)
from triceratops.lightcurve.prep import prepare_from_raw
from triceratops.lightcurve.raw import RawLightCurveData
from triceratops.lightcurve.result import LightCurvePreparationResult


# ── helpers ──────────────────────────────────────────────────────────


def _make_raw(n=5000, **overrides) -> RawLightCurveData:
    """Build a valid RawLightCurveData with sensible defaults."""
    defaults = dict(
        time_btjd=np.linspace(1468.0, 1475.0, n),
        flux=np.ones(n),
        flux_err=np.full(n, 3e-4),
        sectors=(14,),
        cadence="2min",
        exptime_seconds=120.0,
        target_id=None,
    )
    defaults.update(overrides)
    return RawLightCurveData(**defaults)


def _default_ephemeris(**overrides) -> Ephemeris:
    defaults = dict(period_days=3.5, t0_btjd=1468.28, duration_hours=2.0)
    defaults.update(overrides)
    return Ephemeris(**defaults)


# ═══════════════════════════════════════════════════════════════════
# Happy path
# ═══════════════════════════════════════════════════════════════════


class TestPrepareFromRawHappyPath:
    def test_centres_at_zero(self):
        """Spec test: output time_days should be within the transit window."""
        raw = _make_raw()
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        # With duration_hours=2.0 and phase_window_factor=5.0,
        # half_window = 5.0 * (2.0/24.0) = 0.4167 days
        max_half_window = 5.0 * (2.0 / 24.0)
        assert np.abs(result.light_curve.time_days).max() <= max_half_window + 1e-10

    def test_returns_preparation_result(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        assert isinstance(result, LightCurvePreparationResult)
        assert result.ephemeris is eph
        assert result.sectors_used == (14,)
        assert result.cadence_used == "2min"

    def test_scalar_flux_err(self):
        """Spec test: flux_err on the output LightCurve must be a scalar float."""
        raw = _make_raw()
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        assert isinstance(result.light_curve.flux_err, float)
        assert result.light_curve.flux_err > 0

    def test_cadence_days_from_exptime(self):
        """cadence_days should come from exptime_seconds via cadence string,
        not from time spacing (spec gotcha #4)."""
        raw = _make_raw(cadence="2min", exptime_seconds=120.0)
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        assert result.light_curve.cadence_days == pytest.approx(120 / 86400)

    def test_cadence_days_override(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(cadence_days_override=0.01)
        result = prepare_from_raw(raw, eph, config=cfg)
        assert result.light_curve.cadence_days == pytest.approx(0.01)

    def test_output_flux_is_float64(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        assert result.light_curve.flux.dtype == np.float64
        assert result.light_curve.time_days.dtype == np.float64

    def test_default_config_when_none(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph, config=None)
        assert isinstance(result, LightCurvePreparationResult)

    def test_warnings_propagated(self):
        raw = _make_raw(warnings=("dropped column: centroid_col",))
        eph = _default_ephemeris()
        result = prepare_from_raw(raw, eph)
        assert "dropped column: centroid_col" in result.warnings

    def test_no_duration_uses_quarter_period_window(self):
        """When duration_hours is None, half_window = period * 0.25."""
        raw = _make_raw()
        eph = _default_ephemeris(duration_hours=None)
        result = prepare_from_raw(raw, eph)
        # half_window = 3.5 * 0.25 = 0.875 days
        assert np.abs(result.light_curve.time_days).max() <= 0.875 + 1e-10


# ═══════════════════════════════════════════════════════════════════
# Ephemeris validation
# ═══════════════════════════════════════════════════════════════════


class TestPrepareFromRawEphemerisValidation:
    def test_full_jd_epoch_rejected(self):
        """Spec test: full JD epoch should be rejected with BTJD message."""
        raw = _make_raw(n=100)
        eph = Ephemeris(period_days=3.5, t0_btjd=2_459_000.0)
        with pytest.raises(EphemerisRequiredError, match="BTJD"):
            prepare_from_raw(raw, eph)

    def test_nan_period_rejected(self):
        raw = _make_raw(n=100)
        eph = Ephemeris(period_days=float("nan"), t0_btjd=1468.28)
        with pytest.raises(EphemerisRequiredError, match="period_days"):
            prepare_from_raw(raw, eph)

    def test_zero_period_rejected(self):
        raw = _make_raw(n=100)
        eph = Ephemeris(period_days=0.0, t0_btjd=1468.28)
        with pytest.raises(EphemerisRequiredError, match="period_days"):
            prepare_from_raw(raw, eph)

    def test_negative_period_rejected(self):
        raw = _make_raw(n=100)
        eph = Ephemeris(period_days=-1.0, t0_btjd=1468.28)
        with pytest.raises(EphemerisRequiredError, match="period_days"):
            prepare_from_raw(raw, eph)

    def test_inf_t0_rejected(self):
        raw = _make_raw(n=100)
        eph = Ephemeris(period_days=3.5, t0_btjd=float("inf"))
        with pytest.raises(EphemerisRequiredError):
            prepare_from_raw(raw, eph)


# ═══════════════════════════════════════════════════════════════════
# Sigma clipping behaviour
# ═══════════════════════════════════════════════════════════════════


class TestPrepareFromRawSigmaClip:
    def test_upper_sigma_clip_preserves_transit_dip(self):
        """Spec test: a strong negative outlier (transit dip) must NOT be clipped.

        We verify via fold._upper_sigma_mask directly, since after
        prepare_from_raw the dip may be trimmed by the transit window.
        The key invariant is that sigma clipping is upper-only.
        """
        from triceratops.lightcurve.fold import _upper_sigma_mask

        rng = np.random.default_rng(42)
        n = 5000
        flux = np.ones(n) + rng.normal(0, 1e-4, n)
        # Insert a deep negative outlier (transit dip)
        flux[2500] = 0.90  # 1000-sigma below
        # Also insert a positive flare
        flux[1000] = 1.10  # 1000-sigma above

        mask = _upper_sigma_mask(flux, sigma=5.0, iters=5)

        # Transit dip must be preserved
        assert mask[2500] is np.True_
        # Flare must be clipped
        assert mask[1000] is np.False_

    def test_sigma_clip_disabled(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(sigma_clip=None)
        result = prepare_from_raw(raw, eph, config=cfg)
        assert isinstance(result, LightCurvePreparationResult)


# ═══════════════════════════════════════════════════════════════════
# Empty pipeline
# ═══════════════════════════════════════════════════════════════════


class TestPrepareFromRawEmpty:
    def test_raises_empty_error_when_no_cadences_survive_window_trim(self):
        """If the transit window is narrow and t0 is outside data range,
        no cadences survive and LightCurveEmptyError is raised."""
        # Data far from t0 so no points fall within the transit window
        t = np.linspace(2000.0, 2007.0, 500)
        raw = _make_raw(
            n=500,
            time_btjd=t,
            flux=np.ones(500),
            flux_err=np.full(500, 3e-4),
        )
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28, duration_hours=0.01)
        cfg = LightCurveConfig(phase_window_factor=1.0)
        with pytest.raises(LightCurveEmptyError, match="No cadences"):
            prepare_from_raw(raw, eph, config=cfg)


# ═══════════════════════════════════════════════════════════════════
# Config variations
# ═══════════════════════════════════════════════════════════════════


class TestPrepareFromRawConfigVariations:
    def test_with_time_domain_binning(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(bin_minutes=10.0)
        result = prepare_from_raw(raw, eph, config=cfg)
        assert isinstance(result, LightCurvePreparationResult)
        assert len(result.light_curve.time_days) > 0

    def test_with_phase_binning(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(phase_bin_count=100)
        result = prepare_from_raw(raw, eph, config=cfg)
        assert len(result.light_curve.time_days) <= 100

    def test_with_detrend_flatten(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(detrend_method="flatten")
        result = prepare_from_raw(raw, eph, config=cfg)
        assert isinstance(result, LightCurvePreparationResult)

    def test_supersampling_rate_propagated(self):
        raw = _make_raw()
        eph = _default_ephemeris()
        cfg = LightCurveConfig(supersampling_rate=10)
        result = prepare_from_raw(raw, eph, config=cfg)
        assert result.light_curve.supersampling_rate == 10

    def test_different_cadence_strings(self):
        for cadence, exptime in [
            ("20sec", 20), ("2min", 120), ("10min", 600), ("30min", 1800),
        ]:
            raw = _make_raw(cadence=cadence, exptime_seconds=float(exptime))
            eph = _default_ephemeris()
            result = prepare_from_raw(raw, eph)
            assert result.light_curve.cadence_days == pytest.approx(exptime / 86400)
            assert result.cadence_used == cadence
