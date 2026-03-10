"""Typed exception hierarchy for TRICERATOPS+ validation errors.

These exceptions allow callers — and remote/batch infrastructure — to
distinguish between different failure modes without inspecting message
strings:

    ValidationError                 (base)
    ├── PreparedInputIncompleteError  required field missing or None
    ├── ValidationInputError          invalid scientific parameter value
    ├── UnsupportedComputeModeError   requested mode is not supported
    └── PreparationError              failure during provider/IO phase

Use ``PreparedInputIncompleteError`` for structurally missing data
(no stellar_params, no TRILEGAL population when required).

Use ``ValidationInputError`` for logically invalid values (negative
period, empty light curve, shape mismatch).

Use ``PreparationError`` to wrap provider/IO failures in the prepare
phase so callers can distinguish retryable prep failures from
non-retryable input errors.
"""
from __future__ import annotations


class ValidationError(Exception):
    """Base class for all TRICERATOPS+ validation errors."""


class PreparedInputIncompleteError(ValidationError):
    """A required field is missing or None in PreparedValidationInputs.

    Examples: no ``stellar_params`` on the target star; no
    ``trilegal_population`` when a TRILEGAL-dependent scenario is
    being run.
    """


class ValidationInputError(ValidationError):
    """A scientific input value is invalid or internally inconsistent.

    Examples: non-positive ``period_days``; empty or shape-mismatched
    ``LightCurve`` arrays.
    """


class UnsupportedComputeModeError(ValidationError):
    """The requested compute mode or configuration is not supported."""


class PreparationError(ValidationError):
    """A failure occurred during the provider-backed preparation phase.

    Wrap provider/IO exceptions in this so callers can distinguish
    retryable prep failures (network, quota) from non-retryable input
    errors (bad stellar_params, wrong period sign).
    """
