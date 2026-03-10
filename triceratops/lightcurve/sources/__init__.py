"""Light-curve acquisition sources."""
from triceratops.lightcurve.sources.file import FileSource
from triceratops.lightcurve.sources.lightkurve import LightkurveSource

__all__ = ["FileSource", "LightkurveSource"]
