"""
Tests for custom exception classes.
"""
import pytest
from exceptions import (
    SpinException, ValidationError, InvalidDepthError,
    InvalidBoundsError, InvalidParameterError, EmptyInputError,
    CapabilityGenerationError, NegotiationError, ConvergenceError,
    VisualizationError, StateExportError, DimensionMismatchError, NumericError
)


class TestExceptionHierarchy:
    """Test exception class hierarchy"""

    def test_spin_exception_is_base_exception(self):
        """Test that SpinException is the base exception"""
        assert issubclass(SpinException, Exception)

    def test_validation_error_inherits_from_spin_exception(self):
        """Test ValidationError inherits from SpinException"""
        assert issubclass(ValidationError, SpinException)

    def test_invalid_depth_error_inherits_from_validation_error(self):
        """Test InvalidDepthError inherits from ValidationError"""
        assert issubclass(InvalidDepthError, ValidationError)

    def test_all_exceptions_can_be_raised(self):
        """Test that all exception types can be raised"""
        exceptions = [
            SpinException, ValidationError, InvalidDepthError,
            InvalidBoundsError, InvalidParameterError, EmptyInputError,
            CapabilityGenerationError, NegotiationError, ConvergenceError,
            VisualizationError, StateExportError, DimensionMismatchError, NumericError
        ]

        for exc_class in exceptions:
            with pytest.raises(exc_class):
                raise exc_class("Test error message")

    def test_exception_messages(self):
        """Test that exception messages are preserved"""
        message = "Test error message"
        try:
            raise ValidationError(message)
        except ValidationError as e:
            assert str(e) == message
