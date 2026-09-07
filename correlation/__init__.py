"""Fixed-window ZNCC for WRTI Step 4."""

from .zncc import (
    CorrelationError,
    CorrelationResult,
    PreprocessHook,
    compute_fixed_window_zncc,
    compute_zncc,
    iter_zncc,
)
from .preprocess import BrutalPickerEarlyMute, PreprocessError, brutal_picker

__all__ = [
    "CorrelationError",
    "CorrelationResult",
    "PreprocessHook",
    "compute_fixed_window_zncc",
    "compute_zncc",
    "iter_zncc",
    "BrutalPickerEarlyMute",
    "PreprocessError",
    "brutal_picker",
]
