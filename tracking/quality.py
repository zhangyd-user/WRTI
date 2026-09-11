"""Quality masks derived from a completed WRTI tracking result.

These helpers deliberately sit outside the DP implementation.  They define
how downstream workflow, misfit, and diagnostics classify an already tracked
receiver row.
"""

from __future__ import annotations

import numpy as np


def path_failure_mask(tracking) -> np.ndarray:
    """Return only rows for which no usable tracked lag is available.

    A low correlation value and a boundary flag are not failures when the DP
    returned a valid path.  The explicit ``path_index`` term keeps the
    downstream definition tied to the integer DP state rather than only to a
    derived floating-point shift.
    """

    success = np.asarray(tracking.success_mask, dtype=bool)
    path_index = np.asarray(tracking.path_index, dtype=int)
    shift_time = np.asarray(tracking.shift_time, dtype=float)
    if success.shape != path_index.shape or success.shape != shift_time.shape:
        raise AssertionError(
            "TrackingResult success_mask, path_index, and shift_time must have "
            "identical shapes."
        )
    failure = (~success) | (path_index < 0) | (~np.isfinite(shift_time))
    if np.any(success & failure):
        raise AssertionError(
            "TrackingResult invariant violated: a successful row has no valid "
            "path index or finite shift."
        )
    return failure


def low_correlation_qc_mask(
    tracking,
    min_correlation: float | None,
) -> np.ndarray:
    """Return low-ZNCC QC rows without changing path validity."""

    correlation = np.asarray(tracking.tracked_correlation, dtype=float)
    if min_correlation is None:
        return np.zeros(correlation.shape, dtype=bool)
    if not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be in [-1, 1] or None.")
    return (
        ~path_failure_mask(tracking)
        & np.isfinite(correlation)
        & (correlation < float(min_correlation))
    )


def boundary_qc_mask(tracking) -> np.ndarray:
    """Return valid tracked rows carrying a search-boundary QC flag."""

    boundary = np.asarray(tracking.boundary_flag, dtype=bool)
    return (~path_failure_mask(tracking)) & boundary
