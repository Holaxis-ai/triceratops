"""Tests for triceratops.limb_darkening.catalog."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triceratops.domain.value_objects import LimbDarkeningCoeffs
from triceratops.limb_darkening.catalog import (
    SUPPORTED_FILTERS,
    FixedLDCCatalog,
    LimbDarkeningCatalog,
)

# Path to original triceratops data for integration tests
_ORIGINAL_DATA_DIR = Path(__file__).parents[3] / "triceratops"
_REF_DATA_DIR = Path("/Users/collier/projects/reference_repos/TRICERATOPS-plus/triceratops/data")


class TestFixedLDCCatalog:
    def test_fixed_ldc_returns_specified_values(self) -> None:
        ldc = FixedLDCCatalog(u1=0.3, u2=0.1)
        result = ldc.get_coefficients("TESS", 0.0, 5778.0, 4.44)
        assert result.u1 == pytest.approx(0.3)
        assert result.u2 == pytest.approx(0.1)
        assert result.band == "TESS"

    def test_fixed_ldc_bulk_shape(self) -> None:
        ldc = FixedLDCCatalog(u1=0.4, u2=0.2)
        teffs = np.array([5000.0, 6000.0, 7000.0])
        loggs = np.array([4.0, 4.5, 5.0])
        mets = np.array([0.0, 0.0, 0.0])
        u1s, u2s = ldc.get_coefficients_bulk("J", teffs, loggs, mets)
        assert u1s.shape == (3,)
        assert u2s.shape == (3,)
        np.testing.assert_array_almost_equal(u1s, 0.4)
        np.testing.assert_array_almost_equal(u2s, 0.2)


class TestLimbDarkeningCatalog:
    def test_unsupported_filter_raises(self) -> None:
        ldc = LimbDarkeningCatalog()
        with pytest.raises(ValueError, match="Unsupported filter"):
            ldc.get_coefficients("foo", 0.0, 5778.0, 4.44)

    def test_lazy_loading_no_file_on_construction(self) -> None:
        ldc = LimbDarkeningCatalog()
        assert ldc._cache == {}

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_get_coefficients_returns_ldc_type(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        result = ldc.get_coefficients("TESS", 0.0, 5778.0, 4.44)
        assert isinstance(result, LimbDarkeningCoeffs)
        assert isinstance(result.u1, float)
        assert isinstance(result.u2, float)
        assert result.band == "TESS"

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_get_coefficients_caches_after_first_call(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        assert "TESS" not in ldc._cache
        ldc.get_coefficients("TESS", 0.0, 5778.0, 4.44)
        assert "TESS" in ldc._cache

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_nn_lookup_matches_original_algorithm(self) -> None:
        """Verify NN lookup matches the original global-array result for a known point."""
        import pandas as pd

        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)

        # Use the original algorithm directly for comparison
        df = pd.read_csv(_REF_DATA_DIR / "ldc_tess.csv")
        df.columns = [c.strip() for c in df.columns]
        zs = np.array(df["Z"], dtype=float)
        teffs = np.array(df["Teff"], dtype=float)
        loggs = np.array(df["logg"], dtype=float)
        u1s_orig = np.array(df["aLSM"], dtype=float)
        u2s_orig = np.array(df["bLSM"], dtype=float)

        test_z, test_teff, test_logg = 0.0, 5778.0, 4.44
        this_z = zs[np.argmin(np.abs(zs - test_z))]
        this_teff = teffs[np.argmin(np.abs(teffs - test_teff))]
        this_logg = loggs[np.argmin(np.abs(loggs - test_logg))]
        mask = (zs == this_z) & (teffs == this_teff) & (loggs == this_logg)
        expected_u1 = float(u1s_orig[mask][0])
        expected_u2 = float(u2s_orig[mask][0])

        result = ldc.get_coefficients("TESS", test_z, test_teff, test_logg)
        assert result.u1 == pytest.approx(expected_u1)
        assert result.u2 == pytest.approx(expected_u2)

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_bulk_lookup_single_star_matches_scalar(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        scalar = ldc.get_coefficients("TESS", 0.0, 5778.0, 4.44)
        u1s, u2s = ldc.get_coefficients_bulk(
            "TESS",
            np.array([5778.0]),
            np.array([4.44]),
            np.array([0.0]),
        )
        assert u1s[0] == pytest.approx(scalar.u1)
        assert u2s[0] == pytest.approx(scalar.u2)

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_edge_metallicity_below_grid_minimum(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        # Very low metallicity should snap to the grid minimum
        result = ldc.get_coefficients("TESS", -99.0, 5778.0, 4.44)
        assert isinstance(result, LimbDarkeningCoeffs)

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_edge_teff_above_grid_maximum(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        result = ldc.get_coefficients("TESS", 0.0, 99999.0, 4.44)
        assert isinstance(result, LimbDarkeningCoeffs)

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_kepler_filter_uses_different_columns(self) -> None:
        """Kepler CSV uses 'a'/'b' columns instead of 'aLSM'/'bLSM'."""
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        result = ldc.get_coefficients("Kepler", 0.0, 5778.0, 4.44)
        assert isinstance(result, LimbDarkeningCoeffs)
        assert result.band == "Kepler"

    @pytest.mark.skipif(
        not _REF_DATA_DIR.exists(),
        reason="Reference triceratops data not available"
    )
    def test_all_supported_filters_loadable(self) -> None:
        ldc = LimbDarkeningCatalog(data_dir=_REF_DATA_DIR)
        for f in SUPPORTED_FILTERS:
            result = ldc.get_coefficients(f, 0.0, 5778.0, 4.44)
            assert isinstance(result, LimbDarkeningCoeffs), f"Failed for filter {f}"
