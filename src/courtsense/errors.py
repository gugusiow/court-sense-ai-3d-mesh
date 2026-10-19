"""Errors raised by the CourtSense prototype."""


class CourtSenseError(Exception):
    """Base class for expected prototype errors."""


class ValidationError(CourtSenseError):
    """Raised when required input metadata or array constraints are invalid."""
