"""CSV-backed provider implementations for integration tests.

These load pre-captured fixture data from CSV files instead of querying
external services, allowing deterministic integration tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from triceratops.catalog.protocols import StarCatalogProvider
from triceratops.domain.entities import Star, StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.population.protocols import PopulationSynthesisProvider, TRILEGALResult
from triceratops.population.trilegal_parser import parse_trilegal_csv


def _safe_float(val: object, default: float) -> float:
    """Convert to float, returning *default* for NaN / None."""
    if val is None:
        return default
    try:
        f = float(val)  # type: ignore[arg-type]
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(val: object) -> float | None:
    """Convert to float, returning None for NaN / None."""
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class CsvStarCatalogProvider:
    """Implements StarCatalogProvider by loading stars from a CSV fixture.

    The CSV is expected to have columns matching the MAST TIC output:
    ID, Bmag, Vmag, Tmag, Jmag, Hmag, Kmag, gmag, rmag, imag, zmag,
    ra, dec, mass, rad, Teff, d, plx, sep (arcsec), PA (E of N),
    fluxratio, tdepth.
    """

    def __init__(self, stars_csv: Path) -> None:
        self._csv_path = stars_csv

    def query_nearby_stars(
        self,
        tic_id: int,
        search_radius_px: int,
        mission: str,
    ) -> StellarField:
        df = pd.read_csv(self._csv_path)
        # Drop pandas truncation artifact rows (ID == '...')
        df = df[df["ID"].astype(str) != "..."].reset_index(drop=True)

        star_list: list[Star] = []
        for _, row in df.iterrows():
            row_dict: dict[str, object] = row.to_dict()
            sp = StellarParameters.from_tic_row(row_dict)
            star_list.append(Star(
                tic_id=int(row_dict["ID"]),  # type: ignore[arg-type]
                ra_deg=float(row_dict["ra"]),  # type: ignore[arg-type]
                dec_deg=float(row_dict["dec"]),  # type: ignore[arg-type]
                tmag=_safe_float(row.get("Tmag"), 99.0),
                jmag=_safe_float(row.get("Jmag"), 99.0),
                hmag=_safe_float(row.get("Hmag"), 99.0),
                kmag=_safe_float(row.get("Kmag"), 99.0),
                bmag=_safe_float(row.get("Bmag"), 99.0),
                vmag=_safe_float(row.get("Vmag"), 99.0),
                gmag=_safe_float_or_none(row.get("gmag")),
                rmag=_safe_float_or_none(row.get("rmag")),
                imag=_safe_float_or_none(row.get("imag")),
                zmag=_safe_float_or_none(row.get("zmag")),
                stellar_params=sp,
                separation_arcsec=_safe_float(row.get("sep (arcsec)"), 0.0),
                position_angle_deg=_safe_float(row.get("PA (E of N)"), 0.0),
                flux_ratio=_safe_float_or_none(row.get("fluxratio")),
                transit_depth_required=_safe_float_or_none(row.get("tdepth")),
            ))

        return StellarField(
            target_id=tic_id,
            mission=mission,
            search_radius_pixels=search_radius_px,
            stars=star_list,
        )


class CsvTRILEGALProvider:
    """Implements PopulationSynthesisProvider by loading a TRILEGAL CSV fixture.

    Delegates parsing to the existing ``parse_trilegal_csv()`` function.
    """

    def __init__(self, trilegal_csv: Path) -> None:
        self._csv_path = trilegal_csv
        self._result: TRILEGALResult | None = None

    def query(
        self,
        ra_deg: float = 0.0,
        dec_deg: float = 0.0,
        target_tmag: float = 10.0,
        cache_path: Path | None = None,
    ) -> TRILEGALResult:
        if self._result is None:
            self._result = parse_trilegal_csv(self._csv_path)
        return self._result
