class AtlasNavError(RuntimeError):
    """Base error for public AtlasNav commands."""


class ConfigurationError(AtlasNavError):
    """Raised when a frozen experiment configuration is inconsistent."""


class ArtifactError(AtlasNavError):
    """Raised when a released artifact fails its contract."""

