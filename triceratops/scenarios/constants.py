"""Shared physical and computational constants for all scenario classes.

Centralises magic numbers that were previously scattered across scenario
modules, making thresholds easy to find and update.
"""
import math

# Log(2π) — used in every Gaussian log-likelihood computation
LN2PI: float = math.log(2 * math.pi)

# Eclipsing binary mass-ratio threshold for the half-period twin alias.
# Draws with q >= EB_Q_TWIN_THRESHOLD use 2×P_orb (twin-EB branch).
EB_Q_TWIN_THRESHOLD: float = 0.95

# Main-sequence stellar filter applied to TRILEGAL background populations.
MAIN_SEQUENCE_LOGG_MIN: float = 3.5      # log10(cm/s²)
MAIN_SEQUENCE_TEFF_MAX: float = 10_000.0  # K

# Default separation used for the background companion prior when no
# contrast curve is supplied (the TRICERATOPS+ legacy value).
COMPANION_DEFAULT_SEP_ARCSEC: float = 2.2

# Arcsec-to-degree conversion factor (1/3600), used in background priors.
ARCSEC_TO_DEG: float = 1.0 / 3600.0
