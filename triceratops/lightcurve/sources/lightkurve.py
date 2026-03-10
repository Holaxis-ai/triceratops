"""LightkurveRawSource — acquire raw photometry from MAST via lightkurve."""
from __future__ import annotations

import logging
import os
import time as time_mod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.errors import (
    DownloadTimeoutError,
    LightCurveNotFoundError,
    SectorNotAvailableError,
)
from triceratops.lightcurve.raw import RawLightCurveData

if TYPE_CHECKING:
    import lightkurve as lk

log = logging.getLogger(__name__)

# Cadence → (author, exptime) mapping for MAST search
_CADENCE_MAP: dict[str, tuple[str, float | None]] = {
    "20sec": ("SPOC", 20.0),
    "2min": ("SPOC", 120.0),
    "10min": ("QLP", 600.0),
    "30min": ("SPOC", 1800.0),
}

_MAX_RETRIES = 3


def _cache_dir() -> Path:
    base = os.environ.get("TRICERATOPS_CACHE_DIR", str(Path.home() / ".triceratops" / "cache"))
    cache = Path(base) / "lightkurve"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _quality_bitmask(quality_mask: str) -> str:
    return {"none": "none", "default": "default", "hard": "hard"}[quality_mask]


class LightkurveRawSource:
    """Raw source backed by MAST via lightkurve.

    Does NOT sigma-clip, detrend, fold, or trim.
    Those are prepare_from_raw()'s job.
    """

    def __init__(
        self,
        tic_id: int,
        _override_collection: Any = None,
    ) -> None:
        self.tic_id = tic_id
        self._override_collection = _override_collection

    def fetch_raw(self, config: LightCurveConfig) -> RawLightCurveData:
        import lightkurve as lk

        if self._override_collection is not None:
            lc_coll = self._override_collection
        else:
            lc_coll = self._search_and_download(config, lk)

        # Step 4: Stitch (normalises each sector by its median)
        # WARNING: stitch() already normalises. Do NOT call lc.normalize() again.
        stitched = lc_coll.stitch()

        # Step 5: Remove NaNs
        stitched = stitched.remove_nans()

        if len(stitched.time) == 0:
            raise LightCurveNotFoundError(
                f"TIC {self.tic_id}: all cadences removed by quality mask / NaN removal"
            )

        # Step 6: Extract sectors used
        # lightkurve stitch() may or may not preserve sector info in meta;
        # extract from the collection directly as a fallback
        sectors_used = self._extract_sectors(lc_coll, stitched)

        # Detect dropped columns and collect warnings
        warn_list: list[str] = []
        if "flux_err" not in stitched.colnames:
            raise LightCurveNotFoundError(
                f"TIC {self.tic_id}: flux_err column missing after stitch"
            )

        # Step 7: Return RawLightCurveData
        cadence_used = self._resolve_cadence(stitched, config.cadence)
        exptime = float(stitched.meta.get("TIMEDEL", stitched.meta.get("EXPTIME", 120.0)))
        # TIMEDEL is in days for TESS, convert to seconds if it looks like days
        if exptime < 1.0:
            exptime = exptime * 86400.0

        return RawLightCurveData(
            time_btjd=stitched.time.btjd.astype(np.float64),
            flux=stitched.flux.value.astype(np.float64),
            flux_err=stitched.flux_err.value.astype(np.float64),
            sectors=sectors_used,
            cadence=cadence_used,
            exptime_seconds=exptime,
            target_id=self.tic_id,
            warnings=tuple(warn_list),
        )

    def _search_and_download(
        self, config: LightCurveConfig, lk: Any
    ) -> Any:
        # Step 1: Search MAST
        search_kwargs: dict[str, Any] = {
            "target": f"TIC {self.tic_id}",
            "mission": "TESS",
        }
        if config.cadence != "auto" and config.cadence in _CADENCE_MAP:
            author, exptime = _CADENCE_MAP[config.cadence]
            search_kwargs["author"] = author
            search_kwargs["exptime"] = exptime
        if isinstance(config.sectors, tuple):
            search_kwargs["sector"] = list(config.sectors)

        search = lk.search_lightcurve(**search_kwargs)
        if len(search) == 0:
            raise LightCurveNotFoundError(
                f"No TESS light curves found for TIC {self.tic_id}"
            )

        # Step 2: Sector selection
        search_filtered = self._select_sectors(search, config.sectors)

        # Step 3: Download with retry-with-backoff
        return self._download_with_retry(
            search_filtered,
            quality_bitmask=_quality_bitmask(config.quality_mask),
            flux_column=config.flux_type,
        )

    @staticmethod
    def _select_sectors(search: Any, sectors: Any) -> Any:
        if isinstance(sectors, tuple):
            # Already filtered in search
            return search
        if sectors == "auto":
            # Pick the single longest-baseline sector (last available)
            return search[-1:]
        # "all" — use everything
        return search

    @staticmethod
    def _download_with_retry(
        search: Any,
        quality_bitmask: str,
        flux_column: str,
    ) -> Any:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return search.download_all(
                    quality_bitmask=quality_bitmask,
                    flux_column=flux_column,
                    download_dir=str(_cache_dir()),
                )
            except Exception as exc:
                last_err = exc
                exc_str = str(exc).lower()
                retryable = (
                    "timeout" in exc_str
                    or "429" in exc_str
                    or "500" in exc_str
                    or "502" in exc_str
                    or "503" in exc_str
                    or "connection" in exc_str
                )
                if not retryable:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning(
                    "MAST download attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time_mod.sleep(wait)
        raise DownloadTimeoutError(
            f"MAST download failed after {_MAX_RETRIES} attempts: {last_err}",
            retryable=True,
        )

    @staticmethod
    def _extract_sectors(lc_coll: Any, stitched: Any) -> tuple[int, ...]:
        sectors: set[int] = set()
        # Try to get from individual LCs in the collection
        for individual_lc in lc_coll:
            sector = individual_lc.meta.get("SECTOR")
            if sector is not None:
                sectors.add(int(sector))
        # Fallback: stitched meta
        if not sectors:
            meta_sectors = stitched.meta.get("SECTOR")
            if meta_sectors is not None:
                if isinstance(meta_sectors, (list, tuple)):
                    sectors.update(int(s) for s in meta_sectors)
                else:
                    sectors.add(int(meta_sectors))
        if not sectors:
            sectors.add(0)  # unknown sector placeholder
        return tuple(sorted(sectors))

    @staticmethod
    def _resolve_cadence(stitched: Any, config_cadence: str) -> str:
        if config_cadence != "auto":
            return config_cadence
        # Auto-detect from EXPTIME/TIMEDEL
        exptime = stitched.meta.get("TIMEDEL", stitched.meta.get("EXPTIME"))
        if exptime is not None:
            exptime_sec = float(exptime)
            if exptime_sec < 1.0:
                exptime_sec *= 86400.0
            if exptime_sec < 60:
                return "20sec"
            if exptime_sec < 300:
                return "2min"
            if exptime_sec < 900:
                return "10min"
            return "30min"
        return "2min"  # default fallback
