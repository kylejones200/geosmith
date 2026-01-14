"""Standardized error messages for GeoSmith.

Provides consistent error message formatting across the codebase
for better user experience and easier debugging.
"""

from typing import Any, Optional


class GeoSmithError(Exception):
    """Base exception for GeoSmith errors."""

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Initialize GeoSmith error.

        Args:
            message: Primary error message.
            suggestion: Optional suggestion for fixing the error.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}

    def __str__(self) -> str:
        """Format error message with suggestion if available."""
        if self.suggestion:
            return f"{self.message}\n\nSuggestion: {self.suggestion}"
        return self.message


class DataValidationError(GeoSmithError):
    """Error raised when data validation fails."""

    pass


class ParameterError(GeoSmithError):
    """Error raised when parameters are invalid."""

    pass


class DependencyError(GeoSmithError):
    """Error raised when required dependencies are missing."""

    pass


def format_validation_error(
    message: str,
    expected: Optional[str] = None,
    received: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> str:
    """Format a standardized validation error message.

    Args:
        message: Primary error message.
        expected: What was expected (optional).
        received: What was received (optional).
        suggestion: How to fix the error (optional).

    Returns:
        Formatted error message string.
    """
    parts = [message]
    if expected and received:
        parts.append(f"Expected: {expected}, Received: {received}")
    elif expected:
        parts.append(f"Expected: {expected}")
    elif received:
        parts.append(f"Received: {received}")
    if suggestion:
        parts.append(f"Suggestion: {suggestion}")
    return "\n".join(parts)


def format_parameter_error(
    parameter_name: str,
    value: Any,
    valid_values: Optional[list[str]] = None,
    constraint: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> str:
    """Format a standardized parameter error message.

    Args:
        parameter_name: Name of the invalid parameter.
        value: Invalid value that was provided.
        valid_values: List of valid values (optional).
        constraint: Constraint that was violated (optional).
        suggestion: How to fix the error (optional).

    Returns:
        Formatted error message string.
    """
    parts = [f"Invalid value for parameter '{parameter_name}': {value}"]
    if valid_values:
        parts.append(f"Valid values: {', '.join(map(str, valid_values))}")
    if constraint:
        parts.append(f"Constraint: {constraint}")
    if suggestion:
        parts.append(f"Suggestion: {suggestion}")
    return "\n".join(parts)


def format_dependency_error(
    dependency_name: str,
    install_command: Optional[str] = None,
    optional_group: Optional[str] = None,
) -> str:
    """Format a standardized dependency error message.

    Args:
        dependency_name: Name of the missing dependency.
        install_command: Command to install the dependency (optional).
        optional_group: Optional dependency group name (optional).

    Returns:
        Formatted error message string.
    """
    parts = [f"Missing required dependency: {dependency_name}"]
    if optional_group:
        parts.append(f"Install with: pip install geosmith[{optional_group}]")
    elif install_command:
        parts.append(f"Install with: {install_command}")
    else:
        parts.append(f"Install with: pip install {dependency_name}")
    return "\n".join(parts)


def raise_validation_error(
    message: str,
    expected: Optional[str] = None,
    received: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> None:
    """Raise a standardized validation error.

    Args:
        message: Primary error message.
        expected: What was expected (optional).
        received: What was received (optional).
        suggestion: How to fix the error (optional).

    Raises:
        DataValidationError: Always raises this exception.
    """
    error_msg = format_validation_error(message, expected, received, suggestion)
    raise DataValidationError(error_msg, suggestion=suggestion)


def raise_parameter_error(
    parameter_name: str,
    value: Any,
    valid_values: Optional[list[str]] = None,
    constraint: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> None:
    """Raise a standardized parameter error.

    Args:
        parameter_name: Name of the invalid parameter.
        value: Invalid value that was provided.
        valid_values: List of valid values (optional).
        constraint: Constraint that was violated (optional).
        suggestion: How to fix the error (optional).

    Raises:
        ParameterError: Always raises this exception.
    """
    error_msg = format_parameter_error(
        parameter_name, value, valid_values, constraint, suggestion
    )
    raise ParameterError(error_msg, suggestion=suggestion)


def raise_dependency_error(
    dependency_name: str,
    install_command: Optional[str] = None,
    optional_group: Optional[str] = None,
) -> None:
    """Raise a standardized dependency error.

    Args:
        dependency_name: Name of the missing dependency.
        install_command: Command to install the dependency (optional).
        optional_group: Optional dependency group name (optional).

    Raises:
        DependencyError: Always raises this exception.
    """
    error_msg = format_dependency_error(dependency_name, install_command, optional_group)
    raise DependencyError(error_msg)

