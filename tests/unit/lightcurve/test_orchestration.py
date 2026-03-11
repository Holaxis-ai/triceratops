"""Tests for light-curve orchestration wrappers."""
from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
import lightkurve as lk

from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.orchestration import prepare_lightcurve_from_tic
from triceratops.lightcurve.result import LightCurvePreparationResult


def _make_target() -> ResolvedTarget:
    return ResolvedTarget(
        target_ref="TIC 99999",
        tic_id=99999,
        ephemeris=Ephemeris(period_days=3.5, t0_btjd=1468.28, duration_hours=2.0),
        source="test",
    )


def _make_folded_lc(n: int = 1000) -> lk.FoldedLightCurve:
    time_btjd = np.linspace(1468.0, 1475.0, n)
    rng = np.random.default_rng(42)
    flux = np.ones(n) + rng.normal(0, 3e-4, n)
    flux_err = np.full(n, 3e-4)
    lc = lk.LightCurve(
        time=Time(time_btjd + 2_457_000.0, format="jd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=flux_err * u.dimensionless_unscaled,
    )
    return lc.fold(
        period=3.5,
        epoch_time=Time(1468.28, format="btjd", scale="tdb"),
    )


class StubLightkurveSource:
    def __init__(self, tic_id: int) -> None:
        self.tic_id = tic_id
        self.prepare_calls = 0
        self.prepare_folded_calls = 0

    def prepare(
        self, ephemeris: Ephemeris, config: LightCurveConfig,
    ) -> LightCurvePreparationResult:
        self.prepare_calls += 1
        return LightCurvePreparationResult(
            light_curve=LightCurve(
                time_days=np.linspace(-0.1, 0.1, 20),
                flux=np.ones(20),
                flux_err=0.001,
            ),
            ephemeris=ephemeris,
            sectors_used=(14,),
            cadence_used="2min",
            warnings=[],
        )

    def prepare_folded(
        self, ephemeris: Ephemeris, config: LightCurveConfig,
    ) -> tuple[lk.FoldedLightCurve, tuple[int, ...], str]:
        self.prepare_folded_calls += 1
        folded = _make_folded_lc()
        folded.meta["EXPTIME"] = 120.0
        return folded, (14,), "2min"


def test_prepare_lightcurve_from_tic_without_bin_count_delegates(monkeypatch) -> None:
    source = StubLightkurveSource(tic_id=99999)
    monkeypatch.setattr(
        "triceratops.lightcurve.orchestration.LightkurveSource",
        lambda tic_id: source,
    )

    result = prepare_lightcurve_from_tic(_make_target(), LightCurveConfig())

    assert source.prepare_calls == 1
    assert source.prepare_folded_calls == 0
    assert len(result.light_curve.time_days) == 20


def test_prepare_lightcurve_from_tic_bins_at_orchestration_boundary(monkeypatch) -> None:
    source = StubLightkurveSource(tic_id=99999)
    monkeypatch.setattr(
        "triceratops.lightcurve.orchestration.LightkurveSource",
        lambda tic_id: source,
    )

    result = prepare_lightcurve_from_tic(_make_target(), LightCurveConfig(), bin_count=80)

    assert source.prepare_calls == 0
    assert source.prepare_folded_calls == 1
    assert len(result.light_curve.time_days) == 80
    assert result.light_curve.flux_err < 3e-4


def test_prepare_lightcurve_from_tic_rejects_small_bin_count() -> None:
    with pytest.raises(ValueError, match="bin_count"):
        prepare_lightcurve_from_tic(_make_target(), LightCurveConfig(), bin_count=1)
