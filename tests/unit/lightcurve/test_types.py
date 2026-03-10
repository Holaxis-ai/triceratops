"""Tier 1 tests for lightcurve pure types: Ephemeris, ResolvedTarget,
RawLightCurveData, LightCurveConfig, LightCurvePreparationResult,
and the error hierarchy.

No lightkurve imports, no network calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import (
    Ephemeris,
    EphemerisResolver,
    ResolvedTarget,
)
from triceratops.lightcurve.errors import (
    DownloadTimeoutError,
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurveError,
    LightCurveNotFoundError,
    LightCurvePreparationError,
    SectorNotAvailableError,
)
from triceratops.lightcurve.raw import RawLightCurveData
from triceratops.lightcurve.result import LightCurvePreparationResult


# ── helpers ──────────────────────────────────────────────────────────

def _make_raw(**overrides) -> RawLightCurveData:
    """Build a valid RawLightCurveData, overriding any field."""
    defaults = dict(
        time_btjd=np.linspace(1468.0, 1475.0, 500),
        flux=np.ones(500),
        flux_err=np.full(500, 3e-4),
        sectors=(14,),
        cadence="2min",
        exptime_seconds=120.0,
        target_id=None,
    )
    defaults.update(overrides)
    return RawLightCurveData(**defaults)


# ═══════════════════════════════════════════════════════════════════
# Ephemeris
# ═══════════════════════════════════════════════════════════════════


class TestEphemeris:
    def test_basic_construction(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        assert eph.period_days == 3.5
        assert eph.t0_btjd == 1468.28
        assert eph.duration_hours is None
        assert eph.source == "manual"
        assert eph.warnings == ()

    def test_frozen(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        with pytest.raises(AttributeError):
            eph.period_days = 1.0  # type: ignore[misc]

    def test_with_all_fields(self):
        eph = Ephemeris(
            period_days=3.5,
            t0_btjd=1468.28,
            duration_hours=2.1,
            source="exofop",
            warnings=("low S/N",),
        )
        assert eph.duration_hours == 2.1
        assert eph.source == "exofop"
        assert eph.warnings == ("low S/N",)


# ═══════════════════════════════════════════════════════════════════
# ResolvedTarget
# ═══════════════════════════════════════════════════════════════════


class TestResolvedTarget:
    def test_basic_construction(self):
        rt = ResolvedTarget(target_ref="395.01", tic_id=395171208, source="exofop")
        assert rt.target_ref == "395.01"
        assert rt.tic_id == 395171208
        assert rt.ephemeris is None
        assert rt.source == "exofop"

    def test_with_ephemeris(self):
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        rt = ResolvedTarget(
            target_ref="395.01", tic_id=395171208, ephemeris=eph, source="exofop"
        )
        assert rt.ephemeris is eph

    def test_frozen(self):
        rt = ResolvedTarget(target_ref="395.01", tic_id=395171208, source="exofop")
        with pytest.raises(AttributeError):
            rt.tic_id = 0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# RawLightCurveData validation
# ═══════════════════════════════════════════════════════════════════


class TestRawLightCurveData:
    def test_valid_construction(self):
        raw = _make_raw()
        assert raw.cadence == "2min"
        assert raw.sectors == (14,)
        assert len(raw.time_btjd) == 500

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            _make_raw(flux=np.ones(99))

    def test_rejects_empty_arrays(self):
        with pytest.raises(ValueError, match="non-empty"):
            _make_raw(
                time_btjd=np.array([]),
                flux=np.array([]),
                flux_err=np.array([]),
            )

    def test_rejects_non_monotonic_time(self):
        t = np.linspace(1468, 1475, 100)
        t[50] = t[49]  # duplicate — not strictly monotonic
        with pytest.raises(ValueError, match="monotonically"):
            _make_raw(
                time_btjd=t,
                flux=np.ones(100),
                flux_err=np.full(100, 3e-4),
            )

    def test_rejects_nan_in_flux(self):
        flux = np.ones(500)
        flux[42] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _make_raw(flux=flux)

    def test_rejects_inf_in_time(self):
        t = np.linspace(1468.0, 1475.0, 500)
        t[0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _make_raw(time_btjd=t)

    def test_rejects_nan_in_flux_err(self):
        ferr = np.full(500, 3e-4)
        ferr[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _make_raw(flux_err=ferr)

    def test_rejects_2d_time(self):
        with pytest.raises(ValueError, match="1-D"):
            _make_raw(time_btjd=np.ones((10, 2)))

    def test_rejects_empty_sectors(self):
        with pytest.raises(ValueError, match="non-empty"):
            _make_raw(sectors=())

    def test_rejects_unknown_cadence(self):
        with pytest.raises(ValueError, match="unrecognised cadence"):
            _make_raw(cadence="5min")

    def test_rejects_non_positive_exptime(self):
        with pytest.raises(ValueError, match="positive"):
            _make_raw(exptime_seconds=0)

    def test_frozen(self):
        raw = _make_raw()
        with pytest.raises(AttributeError):
            raw.cadence = "10min"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# LightCurveConfig validation
# ═══════════════════════════════════════════════════════════════════


class TestLightCurveConfig:
    def test_default_config_is_valid(self):
        cfg = LightCurveConfig()
        assert cfg.cadence == "auto"
        assert cfg.sigma_clip == 5.0
        assert cfg.detrend_method == "none"
        assert cfg.supersampling_rate == 20

    def test_even_window_length_rejected(self):
        with pytest.raises(ValueError, match="odd integer"):
            LightCurveConfig(flatten_window_length=400)

    def test_window_length_too_small_rejected(self):
        with pytest.raises(ValueError, match="odd integer"):
            LightCurveConfig(flatten_window_length=1)

    def test_polyorder_out_of_range(self):
        with pytest.raises(ValueError, match="between 1 and 5"):
            LightCurveConfig(flatten_polyorder=0)
        with pytest.raises(ValueError, match="between 1 and 5"):
            LightCurveConfig(flatten_polyorder=6)

    def test_sigma_clip_iters_must_be_positive(self):
        with pytest.raises(ValueError, match="sigma_clip_iters"):
            LightCurveConfig(sigma_clip_iters=0)

    def test_phase_window_factor_must_be_ge_one(self):
        with pytest.raises(ValueError, match="phase_window_factor"):
            LightCurveConfig(phase_window_factor=0.5)

    def test_supersampling_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="supersampling_rate"):
            LightCurveConfig(supersampling_rate=0)

    def test_negative_sigma_clip_rejected(self):
        with pytest.raises(ValueError, match="sigma_clip"):
            LightCurveConfig(sigma_clip=-1.0)

    def test_none_sigma_clip_is_allowed(self):
        cfg = LightCurveConfig(sigma_clip=None)
        assert cfg.sigma_clip is None

    def test_negative_bin_minutes_rejected(self):
        with pytest.raises(ValueError, match="bin_minutes"):
            LightCurveConfig(bin_minutes=-5.0)

    def test_negative_cadence_days_override_rejected(self):
        with pytest.raises(ValueError, match="cadence_days_override"):
            LightCurveConfig(cadence_days_override=-0.001)

    def test_frozen(self):
        cfg = LightCurveConfig()
        with pytest.raises(AttributeError):
            cfg.cadence = "2min"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# LightCurvePreparationResult
# ═══════════════════════════════════════════════════════════════════


class TestLightCurvePreparationResult:
    def test_construction(self):
        from triceratops.domain.entities import LightCurve

        lc = LightCurve(
            time_days=np.linspace(-0.05, 0.05, 100),
            flux=np.ones(100),
            flux_err=3e-4,
            cadence_days=120 / 86400,
        )
        eph = Ephemeris(period_days=3.5, t0_btjd=1468.28)
        result = LightCurvePreparationResult(
            light_curve=lc,
            ephemeris=eph,
            sectors_used=(14,),
            cadence_used="2min",
        )
        assert result.light_curve is lc
        assert result.ephemeris is eph
        assert result.sectors_used == (14,)
        assert result.cadence_used == "2min"
        assert result.warnings == []


# ═══════════════════════════════════════════════════════════════════
# Error hierarchy
# ═══════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    def test_all_errors_inherit_from_lightcurve_error(self):
        assert issubclass(LightCurveNotFoundError, LightCurveError)
        assert issubclass(SectorNotAvailableError, LightCurveError)
        assert issubclass(EphemerisRequiredError, LightCurveError)
        assert issubclass(DownloadTimeoutError, LightCurveError)
        assert issubclass(LightCurveEmptyError, LightCurveError)
        assert issubclass(LightCurvePreparationError, LightCurveError)

    def test_download_timeout_retryable(self):
        err = DownloadTimeoutError("failed", retryable=True)
        assert err.retryable is True

    def test_download_timeout_not_retryable(self):
        err = DownloadTimeoutError("permanent", retryable=False)
        assert err.retryable is False

    def test_lightcurve_error_is_exception(self):
        assert issubclass(LightCurveError, Exception)


# ═══════════════════════════════════════════════════════════════════
# EphemerisResolver protocol
# ═══════════════════════════════════════════════════════════════════


class TestEphemerisResolverProtocol:
    def test_custom_resolver_satisfies_protocol(self):
        class StubResolver:
            def resolve(self, target: str) -> ResolvedTarget:
                return ResolvedTarget(
                    target_ref=target, tic_id=12345, source="stub"
                )

        resolver = StubResolver()
        assert isinstance(resolver, EphemerisResolver)
