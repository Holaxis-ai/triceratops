"""Pure preparation pipeline — architectural center of lightcurve prep.

prepare_from_raw() is a pure function: no network, no filesystem,
no provider dependencies. All acquisition and ephemeris resolution
happens before this call.
"""
from __future__ import annotations

import numpy as np

from triceratops.domain.entities import LightCurve
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.lightcurve.errors import (
    EphemerisRequiredError,
    LightCurveEmptyError,
    LightCurvePreparationError,
)
from triceratops.lightcurve.fold import (
    _bin_phase,
    _bin_timeseries,
    _cadence_days,
    _savitzky_golay_flatten,
    _upper_sigma_mask,
    fold_and_clip,
)
from triceratops.lightcurve.raw import RawLightCurveData
from triceratops.lightcurve.result import LightCurvePreparationResult


def prepare_from_raw(
    raw: RawLightCurveData,
    ephemeris: Ephemeris,
    config: LightCurveConfig | None = None,
) -> LightCurvePreparationResult:
    """Transform raw photometry + ephemeris into a compute-ready LightCurve.

    Pure function — no network, no filesystem, no provider dependencies.
    This is the architectural center of the lightcurve preparation feature.

    All acquisition (MAST, file, array) happens before this call.
    All ephemeris resolution (ExoFOP, manual, future BLS) happens before this call.
    """
    config = config or LightCurveConfig()

    time = raw.time_btjd.copy()
    flux = raw.flux.copy()
    ferr = raw.flux_err.copy()

    # Step 1: Sigma clip (upper-only — never clip transit dips)
    if config.sigma_clip is not None:
        valid = _upper_sigma_mask(flux, config.sigma_clip, config.sigma_clip_iters)
        time = time[valid]
        flux = flux[valid]
        ferr = ferr[valid]

    # Step 2: Optional time-domain binning (before fold)
    if config.bin_minutes is not None:
        time, flux, ferr = _bin_timeseries(time, flux, ferr, config.bin_minutes)

    # Step 3: Optional detrending
    if config.detrend_method == "flatten":
        flux = _savitzky_golay_flatten(
            time,
            flux,
            window_length=config.flatten_window_length,
            polyorder=config.flatten_polyorder,
            ephemeris=ephemeris,
        )

    # Step 4: Phase fold — validate ephemeris first
    if not (np.isfinite(ephemeris.period_days) and ephemeris.period_days > 0):
        raise EphemerisRequiredError("period_days must be finite and positive")
    if not (np.isfinite(ephemeris.t0_btjd) and ephemeris.t0_btjd < 10_000):
        raise EphemerisRequiredError(
            "t0_btjd must be in BTJD (BJD - 2,457,000); "
            f"got {ephemeris.t0_btjd} which looks like full JD"
        )

    phase_days = fold_and_clip(time, ephemeris.period_days, ephemeris.t0_btjd)

    # Step 5: Trim to transit window
    duration_days = (
        (ephemeris.duration_hours / 24.0) if ephemeris.duration_hours else None
    )
    half_window = (
        config.phase_window_factor * duration_days
        if duration_days is not None
        else ephemeris.period_days * 0.25
    )
    mask = (phase_days > -half_window) & (phase_days < half_window)
    phase_days = phase_days[mask]
    flux = flux[mask]
    ferr = ferr[mask]

    # Step 6: Optional phase binning
    if config.phase_bin_count is not None:
        phase_days, flux, ferr = _bin_phase(
            phase_days, flux, ferr, config.phase_bin_count
        )

    # Step 7: Final NaN sweep
    finite = np.isfinite(phase_days) & np.isfinite(flux) & np.isfinite(ferr)
    phase_days = phase_days[finite]
    flux = flux[finite]
    ferr = ferr[finite]
    if len(phase_days) == 0:
        raise LightCurveEmptyError("No cadences survived processing pipeline")

    # Step 8: Scalar error collapse
    flux_err_scalar = float(np.mean(ferr))
    if not (np.isfinite(flux_err_scalar) and flux_err_scalar > 0):
        raise LightCurvePreparationError(
            "flux_err collapsed to non-positive scalar"
        )

    # Step 9: cadence_days
    cadence_days_val = config.cadence_days_override or _cadence_days(
        raw.cadence, raw.exptime_seconds
    )

    # Step 10: Construct domain type
    lc = LightCurve(
        time_days=phase_days.astype(np.float64),
        flux=flux.astype(np.float64),
        flux_err=flux_err_scalar,
        cadence_days=cadence_days_val,
        supersampling_rate=config.supersampling_rate,
    )

    return LightCurvePreparationResult(
        light_curve=lc,
        ephemeris=ephemeris,
        sectors_used=raw.sectors,
        cadence_used=raw.cadence,
        warnings=list(raw.warnings),
    )
