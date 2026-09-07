"""Fixed selected-window construction for WRTI Step 3."""

from .builder import (
    WindowError,
    WindowOverlapWarning,
    WindowResult,
    build_window_result,
    build_windows,
)

__all__ = [
    "WindowError",
    "WindowOverlapWarning",
    "WindowResult",
    "build_window_result",
    "build_windows",
]
