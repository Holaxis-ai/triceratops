"""Light-curve assembly: adapter for the LC sub-pipeline.

This is a placeholder — the full LC sub-pipeline (fetch_raw -> prepare_from_raw)
will be implemented when the lightcurve package is fleshed out.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from triceratops.assembly.errors import AssemblyLightCurveError

if TYPE_CHECKING:
    from triceratops.assembly.protocols import ArtifactStore, RawLightCurveSource
    from triceratops.domain.entities import LightCurve
    from triceratops.lightcurve.config import LightCurveConfig
    from triceratops.lightcurve.ephemeris import ResolvedTarget


def assemble_light_curve(
    lc_source: RawLightCurveSource,
    artifact_store: ArtifactStore | None,
    target: ResolvedTarget,
    lc_config: LightCurveConfig | None,
    require: bool,
) -> tuple[LightCurve | None, str, list[str], list[str]]:
    """Fetch and prepare a light curve via the LC sub-pipeline.

    Returns:
        (light_curve, source_label, warnings, artifact_ids) tuple.
        light_curve is None iff require=False and prep failed.

    Raises:
        AssemblyLightCurveError: If require=True and the LC pipeline fails.
        NotImplementedError: Always, until the LC sub-pipeline is implemented.
    """
    raise NotImplementedError(
        "Light-curve sub-pipeline is not yet implemented. "
        "Set include_light_curve=False in AssemblyConfig or provide a "
        "pre-assembled LightCurve via AssembledInputs."
    )
