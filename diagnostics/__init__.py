"""Diagnostic helpers for WRTI Step 6."""

from .warp import WarpDiagnosticError, warp_synthetic_by_shift
from .plots import (
    plot_correlation_image,
    plot_reflectors_velocity,
    plot_selected_window_overlay,
    plot_theoretical_reflection_curve,
    plot_tracked_ridge,
    plot_warped_synthetic_comparison,
    save_vfsa_diagnostic,
    save_vfsa_diagnostics,
)

__all__ = [
    "WarpDiagnosticError",
    "plot_correlation_image",
    "plot_reflectors_velocity",
    "plot_selected_window_overlay",
    "plot_theoretical_reflection_curve",
    "plot_tracked_ridge",
    "plot_warped_synthetic_comparison",
    "save_vfsa_diagnostic",
    "save_vfsa_diagnostics",
    "warp_synthetic_by_shift",
]
