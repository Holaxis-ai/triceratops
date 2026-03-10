"""MOLUSC data assembly: load from a MoluscSource."""
from __future__ import annotations

from typing import TYPE_CHECKING

from triceratops.assembly.errors import AcquisitionError

if TYPE_CHECKING:
    from triceratops.assembly.protocols import MoluscSource
    from triceratops.domain.molusc import MoluscData


def assemble_molusc(
    source: MoluscSource,
) -> tuple[MoluscData, list[str]]:
    """Load MOLUSC companion population data.

    Returns:
        (molusc_data, warnings) tuple.

    Raises:
        AcquisitionError: If loading fails.
    """
    warnings: list[str] = []
    try:
        data = source.load()
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to load MOLUSC data: {exc}"
        ) from exc
    return data, warnings
