"""Convenience orchestration wrappers for light-curve preparation."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import ResolvedTarget
from triceratops.lightcurve.errors import EphemerisRequiredError
from triceratops.lightcurve.result import LightCurvePreparationResult
from triceratops.lightcurve.sources.file import FileSource
from triceratops.lightcurve.sources.lightkurve import LightkurveSource

if TYPE_CHECKING:
    from triceratops.domain.entities import LightCurve


def prepare_lightcurve_from_tic(
    target: ResolvedTarget,
    config: LightCurveConfig | None = None,
) -> LightCurvePreparationResult:
    """Download from MAST and prepare a light curve for a resolved target.

    Requires target.ephemeris to be present; raises EphemerisRequiredError if None.
    Uses lightkurve for all photometry processing (fold, sigma-clip, flatten).
    """
    if target.ephemeris is None:
        raise EphemerisRequiredError(
            f"ResolvedTarget for TIC {target.tic_id} has no ephemeris. "
            "Resolve ephemeris first (e.g. via ExoFopEphemerisResolver) or "
            "provide one manually."
        )
    config = config or LightCurveConfig()
    return LightkurveSource(target.tic_id).prepare(target.ephemeris, config)


def prepare_lightcurve_from_file(
    path: str | Path,
    config: LightCurveConfig | None = None,
) -> LightCurve:
    """Load a pre-folded light curve from a file on disk.

    The file must already be phase-folded with transit at phase=0.
    Supports FITS (lightkurve FoldedLightCurve) and plain text (phase_days, flux, flux_err).
    Returns a domain LightCurve ready for ValidationWorkspace.compute_probs().
    """
    config = config or LightCurveConfig()
    return FileSource(path).load(config)
