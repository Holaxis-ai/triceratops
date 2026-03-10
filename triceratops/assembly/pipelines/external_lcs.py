"""External light-curve assembly: load from an ExternalLcSource."""
from __future__ import annotations

from typing import TYPE_CHECKING

from triceratops.assembly.errors import AcquisitionError

if TYPE_CHECKING:
    from triceratops.assembly.protocols import ExternalLcSource
    from triceratops.domain.entities import ExternalLightCurve


def assemble_external_lcs(
    source: ExternalLcSource,
) -> tuple[list[ExternalLightCurve], list[str]]:
    """Load external (ground-based) light curves.

    Returns:
        (external_lcs, warnings) tuple.

    Raises:
        AcquisitionError: If loading fails.
    """
    warnings: list[str] = []
    try:
        lcs = source.load()
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to load external light curves: {exc}"
        ) from exc
    return lcs, warnings
