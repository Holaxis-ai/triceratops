"""Tests for triceratops.domain.entities."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.entities import LightCurve, Star, StellarField


def _make_star(tic_id: int = 12345678, **kwargs) -> Star:
    defaults = dict(
        tic_id=tic_id, ra_deg=83.82, dec_deg=-5.39,
        tmag=10.5, jmag=9.8, hmag=9.5, kmag=9.4,
        bmag=11.2, vmag=10.8,
    )
    defaults.update(kwargs)
    return Star(**defaults)  # type: ignore[arg-type]


def _make_field() -> StellarField:
    target = _make_star(tic_id=100)
    neighbor = _make_star(tic_id=101, separation_arcsec=2.0)
    return StellarField(
        target_id=100, mission="TESS", search_radius_pixels=10,
        stars=[target, neighbor],
    )


class TestStellarField:
    def test_stellar_field_target_is_first_star(self) -> None:
        field = _make_field()
        assert field.target.tic_id == 100

    def test_stellar_field_neighbors(self) -> None:
        field = _make_field()
        assert len(field.neighbors) == 1
        assert field.neighbors[0].tic_id == 101


class TestLightCurve:
    @pytest.fixture()
    def lc(self) -> LightCurve:
        time = np.linspace(-0.2, 0.2, 100)
        flux = np.ones(100) * 0.99
        return LightCurve(time_days=time, flux=flux, flux_err=0.001)

    def test_light_curve_with_renorm_flux_ratio_1(self, lc: LightCurve) -> None:
        renormed = lc.with_renorm(1.0)
        np.testing.assert_array_almost_equal(renormed.flux, lc.flux)
        assert renormed.flux_err == pytest.approx(lc.flux_err)

    def test_light_curve_with_renorm_scales_flux_and_err(self, lc: LightCurve) -> None:
        renormed = lc.with_renorm(0.5)
        # flux_renormed = (0.99 - 0.5) / 0.5 = 0.98
        expected_flux = (0.99 - 0.5) / 0.5
        np.testing.assert_array_almost_equal(renormed.flux, expected_flux)
        assert renormed.flux_err == pytest.approx(0.001 / 0.5)


class TestStar:
    def test_star_mag_for_band_all_bands(self) -> None:
        star = _make_star(gmag=12.0, rmag=11.5, imag=11.0, zmag=10.5)
        expected = {
            "TESS": 10.5, "J": 9.8, "H": 9.5, "K": 9.4,
            "B": 11.2, "V": 10.8, "g": 12.0, "r": 11.5,
            "i": 11.0, "z": 10.5,
        }
        for band, mag in expected.items():
            assert star.mag_for_band(band) == pytest.approx(mag), f"Failed for band {band}"

    def test_star_mag_for_band_unknown(self) -> None:
        star = _make_star()
        assert star.mag_for_band("U") is None
