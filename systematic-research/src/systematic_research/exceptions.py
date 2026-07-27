"""Domain exceptions with actionable error messages."""


class ResearchError(Exception):
    """Base exception for the package."""


class DataValidationError(ResearchError):
    """Raised when market data violates its declared schema."""


class LeakageError(ResearchError):
    """Raised when information is used before it was historically available."""


class ConstraintError(ResearchError):
    """Raised when portfolio constraints cannot be satisfied."""


class ConfigurationError(ResearchError):
    """Raised when an experiment configuration is invalid."""
