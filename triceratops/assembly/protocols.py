"""Protocol definitions for assembly-layer dependency injection."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from triceratops.domain.entities import ExternalLightCurve, LightCurve
    from triceratops.domain.molusc import MoluscData
    from triceratops.domain.value_objects import ContrastCurve
    from triceratops.lightcurve.config import LightCurveConfig


@runtime_checkable
class RawLightCurveSource(Protocol):
    """Protocol for fetching raw light-curve data."""

    def fetch_raw(self, config: LightCurveConfig) -> object:
        """Fetch raw light-curve data for preparation."""
        ...


@runtime_checkable
class ContrastCurveSource(Protocol):
    """Protocol for loading a contrast curve."""

    def load(self, band: str) -> ContrastCurve:
        """Load a contrast curve for the given photometric band."""
        ...


@runtime_checkable
class MoluscSource(Protocol):
    """Protocol for loading MOLUSC companion population data."""

    def load(self) -> MoluscData:
        """Load MOLUSC data."""
        ...


@runtime_checkable
class ExternalLcSource(Protocol):
    """Protocol for loading external (ground-based) light curves."""

    def load(self) -> list[ExternalLightCurve]:
        """Load all external light curves."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Protocol for storing assembly artifacts (e.g. raw/prepared LCs)."""

    def put_raw_lc(self, data: object) -> str:
        """Store raw light-curve data, return artifact ID."""
        ...

    def put_prepared_lc(self, lc: LightCurve) -> str:
        """Store a prepared light curve, return artifact ID."""
        ...
