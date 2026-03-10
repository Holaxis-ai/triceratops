"""Contrast-curve assembly: load from a ContrastCurveSource."""
from __future__ import annotations

from typing import TYPE_CHECKING

from triceratops.assembly.errors import AcquisitionError

if TYPE_CHECKING:
    from triceratops.assembly.protocols import ContrastCurveSource
    from triceratops.domain.value_objects import ContrastCurve


def assemble_contrast_curve(
    source: ContrastCurveSource,
    band: str,
) -> tuple[ContrastCurve, list[str]]:
    """Load a contrast curve from the given source.

    Returns:
        (contrast_curve, warnings) tuple.

    Raises:
        AcquisitionError: If loading fails.
    """
    warnings: list[str] = []
    try:
        curve = source.load(band)
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to load contrast curve for band {band!r}: {exc}"
        ) from exc
    return curve, warnings
