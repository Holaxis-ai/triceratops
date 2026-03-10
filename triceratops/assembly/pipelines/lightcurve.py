"""Light-curve assembly: adapter for the LC sub-pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

from triceratops.assembly.errors import AssemblyLightCurveError

if TYPE_CHECKING:
    from triceratops.assembly.protocols import ArtifactStore, LightCurveSource
    from triceratops.domain.entities import LightCurve
    from triceratops.lightcurve.config import LightCurveConfig
    from triceratops.lightcurve.ephemeris import Ephemeris


def assemble_light_curve(
    lc_source: LightCurveSource,
    artifact_store: ArtifactStore | None,
    ephemeris: Ephemeris,
    lc_config: LightCurveConfig | None,
    require: bool,
) -> tuple[LightCurve | None, str, list[str], list[str]]:
    """Prepare a light curve via the LC source.

    Returns:
        (light_curve, source_label, warnings, artifact_ids) tuple.
        light_curve is None iff require=False and prep failed.

    Raises:
        AssemblyLightCurveError: If require=True and the LC pipeline fails.
        NotImplementedError: Until the orchestrator wires this up.
    """
    raise NotImplementedError(
        "Light-curve sub-pipeline is not yet wired into the orchestrator. "
        "Pass a pre-assembled LightCurve via AssembledInputs, or use "
        "prepare_lightcurve_from_tic() / prepare_lightcurve_from_file() directly."
    )
