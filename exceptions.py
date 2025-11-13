"""
Custom exceptions for the Recursive Capability Protocol and AdaptiveGenieNetwork.

These exceptions provide clear, specific error messages to help with debugging
and error handling throughout the system.
"""


class SpinException(Exception):
    """Base exception for all Spin-related errors"""
    pass


class ValidationError(SpinException):
    """Raised when input validation fails"""
    pass


class InvalidDepthError(ValidationError):
    """Raised when an invalid recursive depth is provided"""
    pass


class InvalidBoundsError(ValidationError):
    """Raised when problem bounds are invalid"""
    pass


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid"""
    pass


class EmptyInputError(ValidationError):
    """Raised when required input is empty or None"""
    pass


class CapabilityGenerationError(SpinException):
    """Raised when capability generation fails"""
    pass


class NegotiationError(SpinException):
    """Raised when agent negotiation fails"""
    pass


class ConvergenceError(SpinException):
    """Raised when convergence detection fails"""
    pass


class VisualizationError(SpinException):
    """Raised when visualization operations fail"""
    pass


class StateExportError(SpinException):
    """Raised when state export/import fails"""
    pass


class DimensionMismatchError(ValidationError):
    """Raised when dimensions don't match expected values"""
    pass


class NumericError(SpinException):
    """Raised when numeric computations fail (overflow, underflow, etc.)"""
    pass
