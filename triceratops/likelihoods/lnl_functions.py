"""Log-likelihood functions for planet and EB scenarios.

Serial (scalar) variants: accept individual parameter values.
Vectorised (_p) variants: accept arrays and a boolean mask.

All functions return the NEGATIVE log-likelihood (chi-squared / 2).
Source: triceratops/likelihoods.py (lines 152-575).
"""
from __future__ import annotations

import numpy as np

from .transit_model import (
    simulate_eb_transit,
    simulate_eb_transit_p,
    simulate_planet_transit,
    simulate_planet_transit_p,
)


def lnL_planet(
    flux: np.ndarray,
    sigma: float,
    model_flux: np.ndarray,
) -> float:
    """Gaussian negative log-likelihood for a planet transit model.

    Returns 0.5 * sum((flux - model)^2 / sigma^2).
    Source: likelihoods.py:lnL_TP().
    """
    return 0.5 * float(np.sum((flux - model_flux) ** 2 / sigma**2))


def lnL_eb(
    flux: np.ndarray,
    sigma: float,
    model_flux: np.ndarray,
    secondary_depth: float,
) -> float:
    """Gaussian negative log-likelihood for an EB model (q < 0.95).

    Returns inf if secondary eclipse depth >= 1.5 * sigma (detectable secondary).
    Source: likelihoods.py:lnL_EB().
    """
    if secondary_depth >= 1.5 * sigma:
        return np.inf
    return 0.5 * float(np.sum((flux - model_flux) ** 2 / sigma**2))


def lnL_eb_twin(
    flux: np.ndarray,
    sigma: float,
    model_flux: np.ndarray,
) -> float:
    """Gaussian negative log-likelihood for a twin EB model (q >= 0.95, half-period).

    Source: likelihoods.py:lnL_EB_twin().
    """
    return 0.5 * float(np.sum((flux - model_flux) ** 2 / sigma**2))


# ---------------------------------------------------------------------------
# Vectorised variants (_p suffix = parallel/vectorised)
# ---------------------------------------------------------------------------

def lnL_planet_p(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    rps: np.ndarray,
    periods: np.ndarray,
    incs: np.ndarray,
    as_: np.ndarray,
    rss: np.ndarray,
    u1s: np.ndarray,
    u2s: np.ndarray,
    eccs: np.ndarray,
    argps: np.ndarray,
    companion_flux_ratios: np.ndarray,
    mask: np.ndarray,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    force_serial: bool = False,
) -> np.ndarray:
    """Vectorised planet transit log-likelihood for N MC samples.

    Evaluates only where mask[i] is True; returns inf for masked-out samples.

    Args:
        force_serial: If True, evaluate samples one-by-one using the scalar
            transit model rather than the vectorised batch variant. Useful
            for debugging or when the vectorised path is unavailable.

    Returns:
        lnL array of shape (N,).
    """
    n = len(rps)
    lnL = np.full(n, np.inf)
    if not np.any(mask):
        return lnL

    # Subset to masked entries
    idx = np.where(mask)[0]
    if force_serial:
        for i in idx:
            period_i = float(periods[i]) if isinstance(periods, np.ndarray) else float(periods)
            model = simulate_planet_transit(
                time,
                float(rps[i]),
                period_i,
                float(incs[i]),
                float(as_[i]),
                float(rss[i]),
                float(u1s[i]),
                float(u2s[i]),
                float(eccs[i]),
                float(argps[i]),
                float(companion_flux_ratios[i]),
                companion_is_host=companion_is_host,
                exptime=exptime,
                nsamples=nsamples,
            )
            lnL[i] = lnL_planet(flux, sigma, model)
        return lnL

    model = simulate_planet_transit_p(
        time,
        rps[idx],
        periods[idx] if isinstance(periods, np.ndarray) else periods,
        incs[idx],
        as_[idx],
        rss[idx],
        u1s[idx],
        u2s[idx],
        eccs[idx],
        argps[idx],
        companion_flux_ratios[idx],
        companion_is_host=companion_is_host,
        exptime=exptime,
        nsamples=nsamples,
    )
    lnL[idx] = 0.5 * np.sum((flux - model) ** 2 / sigma**2, axis=1)
    return lnL


def lnL_eb_p(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    rss: np.ndarray,
    rcomps: np.ndarray,
    eb_flux_ratios: np.ndarray,
    periods: np.ndarray,
    incs: np.ndarray,
    as_: np.ndarray,
    u1s: np.ndarray,
    u2s: np.ndarray,
    eccs: np.ndarray,
    argps: np.ndarray,
    companion_flux_ratios: np.ndarray,
    mask: np.ndarray,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    force_serial: bool = False,
) -> np.ndarray:
    """Vectorised EB log-likelihood for q < 0.95 samples.

    Returns inf for masked-out samples and for samples with detectable secondary.

    Args:
        force_serial: If True, evaluate samples one-by-one using the scalar
            transit model rather than the vectorised batch variant.
    """
    n = len(rss)
    lnL = np.full(n, np.inf)
    if not np.any(mask):
        return lnL

    idx = np.where(mask)[0]
    if force_serial:
        for i in idx:
            period_i = float(periods[i]) if isinstance(periods, np.ndarray) else float(periods)
            model, secdepth = simulate_eb_transit(
                time,
                float(rss[i]),
                float(rcomps[i]),
                float(eb_flux_ratios[i]),
                period_i,
                float(incs[i]),
                float(as_[i]),
                float(u1s[i]),
                float(u2s[i]),
                float(eccs[i]),
                float(argps[i]),
                float(companion_flux_ratios[i]),
                companion_is_host=companion_is_host,
                exptime=exptime,
                nsamples=nsamples,
            )
            lnL[i] = lnL_eb(flux, sigma, model, float(secdepth))
        return lnL

    model, secdepth = simulate_eb_transit_p(
        time,
        rcomps[idx],
        eb_flux_ratios[idx],
        periods[idx] if isinstance(periods, np.ndarray) else periods,
        incs[idx],
        as_[idx],
        rss[idx],
        u1s[idx],
        u2s[idx],
        eccs[idx],
        argps[idx],
        companion_flux_ratios[idx],
        companion_is_host=companion_is_host,
        exptime=exptime,
        nsamples=nsamples,
    )
    # secdepth shape (n_masked, 1)
    sec_ok = (secdepth[:, 0] < 1.5 * sigma)
    lnL_vals = 0.5 * np.sum((flux - model) ** 2 / sigma**2, axis=1)
    lnL_vals[~sec_ok] = np.inf
    lnL[idx] = lnL_vals
    return lnL


def lnL_eb_twin_p(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float,
    rss: np.ndarray,
    rcomps: np.ndarray,
    eb_flux_ratios: np.ndarray,
    periods: np.ndarray,
    incs: np.ndarray,
    as_: np.ndarray,
    u1s: np.ndarray,
    u2s: np.ndarray,
    eccs: np.ndarray,
    argps: np.ndarray,
    companion_flux_ratios: np.ndarray,
    mask: np.ndarray,
    companion_is_host: bool = False,
    exptime: float = 0.00139,
    nsamples: int = 20,
    force_serial: bool = False,
) -> np.ndarray:
    """Vectorised EB log-likelihood for q >= 0.95 (twin) samples.

    No secondary depth check for twins.

    Args:
        force_serial: If True, evaluate samples one-by-one using the scalar
            transit model rather than the vectorised batch variant.
    """
    n = len(rss)
    lnL = np.full(n, np.inf)
    if not np.any(mask):
        return lnL

    idx = np.where(mask)[0]
    if force_serial:
        for i in idx:
            period_i = float(periods[i]) if isinstance(periods, np.ndarray) else float(periods)
            model, _secdepth = simulate_eb_transit(
                time,
                float(rss[i]),
                float(rcomps[i]),
                float(eb_flux_ratios[i]),
                period_i,
                float(incs[i]),
                float(as_[i]),
                float(u1s[i]),
                float(u2s[i]),
                float(eccs[i]),
                float(argps[i]),
                float(companion_flux_ratios[i]),
                companion_is_host=companion_is_host,
                exptime=exptime,
                nsamples=nsamples,
            )
            lnL[i] = lnL_eb_twin(flux, sigma, model)
        return lnL

    model, _secdepth = simulate_eb_transit_p(
        time,
        rcomps[idx],
        eb_flux_ratios[idx],
        periods[idx] if isinstance(periods, np.ndarray) else periods,
        incs[idx],
        as_[idx],
        rss[idx],
        u1s[idx],
        u2s[idx],
        eccs[idx],
        argps[idx],
        companion_flux_ratios[idx],
        companion_is_host=companion_is_host,
        exptime=exptime,
        nsamples=nsamples,
    )
    lnL[idx] = 0.5 * np.sum((flux - model) ** 2 / sigma**2, axis=1)
    return lnL
