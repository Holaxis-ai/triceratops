"""Contrast curve file I/O.

Replaces funcs.file_to_contrast_curve() and funcs.separation_at_contrast().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from triceratops.domain.value_objects import ContrastCurve


def load_contrast_curve(path: Path, band: str = "unknown") -> ContrastCurve:
    """Load a contrast curve from a two-column whitespace-delimited file.

    The file format is::

        <separation_arcsec>  <delta_mag>
        ...

    Separations must be in ascending order in the file.

    Replaces funcs.file_to_contrast_curve() (funcs.py:200-225).

    Args:
        path: Path to the contrast curve file.
        band: Filter band label for the ContrastCurve (e.g. "J", "K").

    Returns:
        ContrastCurve with sorted separations_arcsec and delta_mags.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file has fewer than 2 rows or wrong column count.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Contrast curve file not found: {path}")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"Contrast curve file must have at least 2 columns; "
            f"got {data.shape[1]} in {path}"
        )
    separations = data[:, 0]
    delta_mags = data[:, 1]
    order = np.argsort(separations)
    return ContrastCurve(
        separations_arcsec=separations[order],
        delta_mags=delta_mags[order],
        band=band,
    )


def separation_at_contrast(
    curve: ContrastCurve,
    delta_mag: float,
) -> float:
    """Return the angular separation at which a given contrast delta_mag is achieved.

    Replaces funcs.separation_at_contrast() (funcs.py:225-240).

    Args:
        curve: Loaded ContrastCurve.
        delta_mag: The contrast level to query.

    Returns:
        Separation in arcsec where delta_mag is reached.
        Returns 0.0 if delta_mag is below the minimum contrast in the curve.
        Returns the outermost separation if delta_mag exceeds the maximum.
    """
    if delta_mag < curve.delta_mags[0]:
        return 0.0
    if delta_mag >= curve.delta_mags[-1]:
        return float(curve.separations_arcsec[-1])
    return float(np.interp(delta_mag, curve.delta_mags, curve.separations_arcsec))
