"""Convenience orchestration wrappers for light-curve preparation."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.convert import convert_folded_to_domain
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
    *,
    bin_count: int | None = None,
) -> LightCurvePreparationResult:
    """Download from MAST and prepare a light curve for a resolved target.

    Requires target.ephemeris to be present; raises EphemerisRequiredError if None.
    Uses lightkurve for all photometry processing (fold, sigma-clip, flatten).
    Optional binning is applied here, at the orchestration boundary, using
    lightkurve's own binning implementation before domain conversion.
    """
    if target.ephemeris is None:
        raise EphemerisRequiredError(
            f"ResolvedTarget for TIC {target.tic_id} has no ephemeris. "
            "Resolve ephemeris first (e.g. via ExoFopEphemerisResolver) or "
            "provide one manually."
        )
    config = config or LightCurveConfig()
    source = LightkurveSource(target.tic_id)
    if bin_count is None:
        return source.prepare(target.ephemeris, config)
    if bin_count < 2:
        raise ValueError(f"bin_count must be >= 2 or None, got {bin_count}")

    lc_folded, sectors, cadence_used = source.prepare_folded(target.ephemeris, config)
    lc_binned = lc_folded.bin(bins=bin_count)
    lc_domain = convert_folded_to_domain(lc_binned, cadence=cadence_used, config=config)
    return LightCurvePreparationResult(
        light_curve=lc_domain,
        ephemeris=target.ephemeris,
        sectors_used=sectors,
        cadence_used=cadence_used,
        warnings=[],
    )


def prepare_lightcurve_from_file(
    path: str | Path,
    config: LightCurveConfig | None = None,
) -> LightCurve:
    """Load a pre-folded light curve from a file on disk.

    The file must already be phase-folded with transit at phase=0.
    Supports FITS (lightkurve FoldedLightCurve) and plain text (phase_days, flux, flux_err).
    If you want binned input here, bin it upstream before loading the file.
    Returns a domain LightCurve ready for ValidationWorkspace.compute_probs().
    """
    config = config or LightCurveConfig()
    return FileSource(path).load(config)
