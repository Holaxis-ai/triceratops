"""Orbital geometry helper functions.

All functions are pure (no I/O, no state). They encapsulate the repeated
orbital-mechanics calculations from the 9-phase scenario skeleton.

Source: marginal_likelihoods.py Phases 2 and 6.
"""
from __future__ import annotations

import numpy as np

from triceratops.config.config import CONST


def semi_major_axis(
    period_days: np.ndarray,
    mass_msun: float | np.ndarray,
) -> np.ndarray:
    """Compute semi-major axis in cm using Kepler's third law.

    a = ((G * M * Msun) / (4 * pi^2) * (P * 86400)^2)^(1/3)

    Args:
        period_days: Orbital period(s) in days, shape (N,).
        mass_msun: Total system mass in Solar masses. Scalar or shape (N,).

    Returns:
        Semi-major axis in cm, shape (N,).
    """
    G = CONST.G
    Msun = CONST.Msun
    pi = np.pi
    a = ((G * mass_msun * Msun) / (4 * pi**2) * (period_days * 86400) ** 2) ** (1.0 / 3.0)
    return a


def transit_probability(
    a_cm: np.ndarray,
    r_host_rsun: float | np.ndarray,
    r_transiting_rsun: float | np.ndarray,
    ecc: np.ndarray,
    argp_deg: np.ndarray,
) -> np.ndarray:
    """Compute geometric transit probability P_tra.

    P_tra = (R_host + R_transiting) / a * e_corr
    where e_corr = (1 + ecc * sin(argp)) / (1 - ecc^2)
    """
    Rsun = CONST.Rsun
    argp_rad = argp_deg * np.pi / 180.0
    e_corr = (1.0 + ecc * np.sin(argp_rad)) / (1.0 - ecc**2)
    r_sum_cm = (r_host_rsun + r_transiting_rsun) * Rsun
    return r_sum_cm / a_cm * e_corr


def impact_parameter(
    a_cm: np.ndarray,
    inc_deg: np.ndarray,
    r_host_rsun: float | np.ndarray,
    ecc: np.ndarray,
    argp_deg: np.ndarray,
) -> np.ndarray:
    """Compute transit impact parameter b.

    b = a*(1-e^2) * cos(inc) / ((1 + e*sin(argp)) * R_host)

    Returns:
        Impact parameter (dimensionless), shape (N,).
    """
    Rsun = CONST.Rsun
    argp_rad = argp_deg * np.pi / 180.0
    inc_rad = inc_deg * np.pi / 180.0
    r = a_cm * (1.0 - ecc**2) / (1.0 + ecc * np.sin(argp_rad))
    b = r * np.cos(inc_rad) / (r_host_rsun * Rsun)
    return b


def collision_check(
    a_cm: np.ndarray,
    r_host_rsun: float | np.ndarray,
    r_transiting_rsun: float | np.ndarray,
    ecc: np.ndarray,
) -> np.ndarray:
    """Return boolean array: True where orbit is physically impossible (collision).

    coll = (R_host + R_transiting) > a * (1 - ecc)
    """
    Rsun = CONST.Rsun
    r_sum_cm = (r_host_rsun + r_transiting_rsun) * Rsun
    return r_sum_cm > a_cm * (1.0 - ecc)
