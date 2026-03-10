"""Assembly error hierarchy."""


class AssemblyError(Exception):
    """Base error for all assembly operations."""


class AcquisitionError(AssemblyError):
    """Failed to acquire a data source during assembly."""


class TRILEGALAcquisitionError(AcquisitionError):
    """Failed to acquire TRILEGAL population data."""


class CatalogAcquisitionError(AcquisitionError):
    """Failed to acquire catalog data."""


class ArtifactLoadError(AcquisitionError):
    """Failed to load a stored artifact."""


class AssemblyDataError(AssemblyError):
    """Data-level error during assembly."""


class AssemblyLightCurveError(AssemblyDataError):
    """Light-curve-specific data error during assembly."""


class AssemblyConfigError(AssemblyError):
    """Invalid assembly configuration."""
