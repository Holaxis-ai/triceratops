"""Limb-darkening coefficient catalog: lazy-loading service for 10 photometric filters.

Replaces the 30+ module-level global arrays in marginal_likelihoods.py:20-104.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files as _resource_files
from pathlib import Path

import numpy as np
import pandas as pd

from triceratops.domain.value_objects import LimbDarkeningCoeffs

# Supported filter names -- correspond to the 10 CSV files in the original package
SUPPORTED_FILTERS: frozenset[str] = frozenset({
    "TESS", "Kepler", "J", "H", "K", "g", "r", "i", "z", "V"
})


@dataclass(frozen=True)
class _FilterFileInfo:
    """Internal mapping from filter name to CSV file and column names."""

    filename: str
    u1_col: str
    u2_col: str


# Maps filter name -> CSV filename and column names for u1/u2
# Based on marginal_likelihoods.py:20-104
_FILTER_FILE_MAP: dict[str, _FilterFileInfo] = {
    "TESS":   _FilterFileInfo("ldc_tess.csv",   "aLSM", "bLSM"),
    "Kepler": _FilterFileInfo("ldc_kepler.csv",  "a",    "b"),
    "V":      _FilterFileInfo("ldc_V.csv",       "aLSM", "bLSM"),
    "J":      _FilterFileInfo("ldc_wirc.csv",    "aLSM", "bLSM"),
    "H":      _FilterFileInfo("ldc_H.csv",       "aLSM", "bLSM"),
    "K":      _FilterFileInfo("ldc_K.csv",       "aLSM", "bLSM"),
    "g":      _FilterFileInfo("ldc_sdss_g.csv",  "aLSM", "bLSM"),
    "r":      _FilterFileInfo("ldc_sdss_r.csv",  "aLSM", "bLSM"),
    "i":      _FilterFileInfo("ldc_sdss_i.csv",  "aLSM", "bLSM"),
    "z":      _FilterFileInfo("ldc_sdss_z.csv",  "aLSM", "bLSM"),
}


class LimbDarkeningCatalog:
    """Lazy-loading quadratic LDC table for 10 photometric filters.

    Replaces the 30+ global arrays in marginal_likelihoods.py:20-104.

    The 9-line nearest-neighbour lookup block repeated in the original code
    is centralised in get_coefficients().

    Thread-safe: each instance has its own cache; no module-level state.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """
        Args:
            data_dir: Path to directory containing ldc_*.csv files.
                      Defaults to the triceratops package data directory.
        """
        self._data_dir = data_dir
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]] = {}

    def _get_data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        return Path(str(_resource_files("triceratops").joinpath("data")))

    def _load_filter(
        self, filter_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and cache a single filter's LDC data as arrays.

        Returns:
            (Zs, Teffs, loggs, u1s, u2s) arrays.
        """
        if filter_name not in self._cache:
            info = _FILTER_FILE_MAP[filter_name]
            path = self._get_data_dir() / info.filename
            df = pd.read_csv(path)
            # Strip whitespace from column names for robustness
            df.columns = [c.strip() for c in df.columns]
            zs = np.array(df["Z"], dtype=float)
            teffs = np.array(df["Teff"], dtype=float)
            loggs = np.array(df["logg"], dtype=float)
            u1s = np.array(df[info.u1_col], dtype=float)
            u2s = np.array(df[info.u2_col], dtype=float)
            self._cache[filter_name] = (zs, teffs, loggs, u1s, u2s)
        return self._cache[filter_name]

    def get_coefficients(
        self,
        filter_name: str,
        metallicity: float,
        teff: float,
        logg: float,
    ) -> LimbDarkeningCoeffs:
        """Return the nearest-neighbour LDC (u1, u2) for the given stellar parameters.

        Implements the nearest-neighbour lookup from marginal_likelihoods.py:167-175.

        The lookup finds the nearest grid point independently for each of Z, Teff, logg,
        then applies a 3-way mask. This matches the original algorithm exactly.

        Args:
            filter_name: One of the supported filter names.
            metallicity: [M/H] in dex.
            teff: Effective temperature in Kelvin.
            logg: log10(g / cm s^-2).

        Returns:
            LimbDarkeningCoeffs with scalar u1, u2 floats.

        Raises:
            ValueError: If filter_name is not supported.
            KeyError: If no grid point matches the nearest-neighbour criteria.
        """
        if filter_name not in SUPPORTED_FILTERS:
            raise ValueError(
                f"Unsupported filter {filter_name!r}. "
                f"Supported: {sorted(SUPPORTED_FILTERS)}"
            )
        zs, teffs, loggs_arr, u1s, u2s = self._load_filter(filter_name)

        # Nearest-neighbour lookup -- original algorithm from lines 167-175
        this_z = zs[np.argmin(np.abs(zs - metallicity))]
        this_teff = teffs[np.argmin(np.abs(teffs - teff))]
        this_logg = loggs_arr[np.argmin(np.abs(loggs_arr - logg))]

        mask = (zs == this_z) & (teffs == this_teff) & (loggs_arr == this_logg)
        u1 = float(u1s[mask][0])
        u2 = float(u2s[mask][0])

        return LimbDarkeningCoeffs(u1=u1, u2=u2, band=filter_name)

    def get_coefficients_bulk(
        self,
        filter_name: str,
        teffs: np.ndarray,
        loggs: np.ndarray,
        metallicities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-star LDC lookup for N stars (used in B and N_unknown scenarios).

        Fully vectorised: no Python loop over samples.  Nearest-neighbour on
        each of Z, Teff, logg independently (matching get_coefficients logic),
        then a single fancy-index into the pre-loaded table.

        Args:
            filter_name: Filter for lookup.
            teffs: Array of Teff values, shape (N,).
            loggs: Array of logg values, shape (N,).
            metallicities: Array of [M/H] values, shape (N,).

        Returns:
            (u1s, u2s): two arrays of shape (N,).
        """
        zs, teff_grid, logg_grid, u1s_table, u2s_table = self._load_filter(filter_name)

        # Build a 3-D index table once per filter (cached separately from raw data).
        # Axes: (z_idx, teff_idx, logg_idx) → flat row index into the table.
        if not hasattr(self, "_bulk_cache"):
            self._bulk_cache: dict[str, tuple] = {}
        if filter_name not in self._bulk_cache:
            z_unique = np.unique(zs)
            t_unique = np.unique(teff_grid)
            g_unique = np.unique(logg_grid)
            nz, nt, ng = len(z_unique), len(t_unique), len(g_unique)
            u1_3d = np.zeros((nz, nt, ng))
            u2_3d = np.zeros((nz, nt, ng))
            zi = np.searchsorted(z_unique, zs)
            ti = np.searchsorted(t_unique, teff_grid)
            gi = np.searchsorted(g_unique, logg_grid)
            u1_3d[zi, ti, gi] = u1s_table
            u2_3d[zi, ti, gi] = u2s_table
            self._bulk_cache[filter_name] = (z_unique, t_unique, g_unique, u1_3d, u2_3d)

        z_unique, t_unique, g_unique, u1_3d, u2_3d = self._bulk_cache[filter_name]

        # Snap each input to nearest grid point using searchsorted (vectorised)
        zi = np.clip(
            np.searchsorted(z_unique, metallicities, side="left"),
            0, len(z_unique) - 1,
        )
        # Adjust for nearest (not just left-insertion) neighbour
        lo = np.clip(zi - 1, 0, len(z_unique) - 1)
        zi = np.where(
            np.abs(z_unique[lo] - metallicities) < np.abs(z_unique[zi] - metallicities),
            lo, zi,
        )

        ti = np.clip(np.searchsorted(t_unique, teffs, side="left"), 0, len(t_unique) - 1)
        lo = np.clip(ti - 1, 0, len(t_unique) - 1)
        ti = np.where(np.abs(t_unique[lo] - teffs) < np.abs(t_unique[ti] - teffs), lo, ti)

        gi = np.clip(np.searchsorted(g_unique, loggs, side="left"), 0, len(g_unique) - 1)
        lo = np.clip(gi - 1, 0, len(g_unique) - 1)
        gi = np.where(np.abs(g_unique[lo] - loggs) < np.abs(g_unique[gi] - loggs), lo, gi)

        return u1_3d[zi, ti, gi], u2_3d[zi, ti, gi]


class FixedLDCCatalog:
    """Test stub that always returns the same (u1, u2) regardless of input.

    Use this in unit tests to avoid any file I/O and to make LDC values
    explicit and predictable.
    """

    def __init__(self, u1: float = 0.4, u2: float = 0.2) -> None:
        self._u1 = u1
        self._u2 = u2

    def get_coefficients(
        self,
        filter_name: str,
        metallicity: float,
        teff: float,
        logg: float,
    ) -> LimbDarkeningCoeffs:
        return LimbDarkeningCoeffs(u1=self._u1, u2=self._u2, band=filter_name)

    def get_coefficients_bulk(
        self,
        filter_name: str,
        teffs: np.ndarray,
        loggs: np.ndarray,
        metallicities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(teffs)
        return np.full(n, self._u1), np.full(n, self._u2)
