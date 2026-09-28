"""Finite lag completion for receivers not covered by the DP path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..correlation import CorrelationResult
from .tracker import TrackingResult, _refine_peak
from .quality import path_failure_mask


@dataclass(frozen=True)
class ShiftCompletionResult:
    """Final shifts plus diagnostics for one reflector/shot pair."""

    final_shift: np.ndarray
    dp_failure_mask: np.ndarray
    fallback_local_mask: np.ndarray
    fallback_global_mask: np.ndarray
    fallback_argmax_mask: np.ndarray


def complete_tracked_shift(
    tracking: TrackingResult,
    correlation: CorrelationResult,
    *,
    safe_lag_range_time: float,
    local_radius: int = 2,
) -> ShiftCompletionResult:
    """Complete DP gaps with local/global medians or a safe ZNCC argmax.

    The DP output is never changed.  ``tracking.shift_time`` is copied as the
    initial result; only receiver rows classified as DP failures are filled.
    A local/global fill uses finite DP shifts from neighboring receivers or
    from the same reflector/shot and does not require the failed receiver to
    have its own finite waveform ZNCC.  The receiver-level validity and ZNCC
    checks apply only to the final all-failure argmax fallback, which is
    additionally restricted to the configured safe lag range.  This keeps
    genuinely invalid windows/correlations without any usable path support
    as NaN so the objective can handle them separately.
    """

    if not np.isfinite(safe_lag_range_time) or safe_lag_range_time < 0:
        raise ValueError("safe_lag_range_time must be finite and non-negative.")
    if (
        isinstance(local_radius, bool)
        or int(local_radius) != local_radius
        or local_radius < 0
    ):
        raise ValueError("local_radius must be a non-negative integer.")

    tracked_shift = np.asarray(tracking.shift_time, dtype=float)
    dp_failure = path_failure_mask(tracking)
    lags_time = np.asarray(correlation.lags_time, dtype=float)
    lags_samples = np.asarray(correlation.lags_samples, dtype=int)
    if lags_time.ndim != 1 or lags_samples.shape != lags_time.shape:
        raise ValueError("Correlation lag axes are inconsistent.")

    # ``waveform_correlation`` is the existing waveform ZNCC before the
    # optional envelope/fine ownership gate.  Older result objects fall back
    # to the public gated matrix, preserving compatibility with callers that
    # construct CorrelationResult directly.
    raw_correlation = getattr(correlation, "waveform_correlation", None)
    if raw_correlation is None:
        raw_correlation = correlation.correlation
    raw_correlation = np.asarray(raw_correlation, dtype=float)
    if raw_correlation.ndim != 2 or raw_correlation.shape[1] != lags_time.size:
        raise ValueError("Correlation matrix and lag axes are inconsistent.")
    if raw_correlation.shape[0] != tracked_shift.size:
        raise ValueError("Tracking and correlation receiver counts differ.")

    nreceiver = tracked_shift.size
    final_shift = np.array(tracked_shift, dtype=float, copy=True)
    fallback_local = np.zeros(nreceiver, dtype=bool)
    fallback_global = np.zeros(nreceiver, dtype=bool)
    fallback_argmax = np.zeros(nreceiver, dtype=bool)

    safe_lag = np.abs(lags_time) <= float(safe_lag_range_time)
    if hasattr(correlation, "ownership_lag_min"):
        ownership_min = np.asarray(correlation.ownership_lag_min, dtype=float)
        ownership_max = np.asarray(correlation.ownership_lag_max, dtype=float)
        if ownership_min.shape == (nreceiver,) and ownership_max.shape == (nreceiver,):
            safe_lag_by_receiver = (
                safe_lag[None, :]
                & np.isfinite(ownership_min[:, None])
                & np.isfinite(ownership_max[:, None])
                & (lags_time[None, :] > ownership_min[:, None])
                & (lags_time[None, :] < ownership_max[:, None])
            )
        else:
            raise ValueError("Correlation ownership bounds have an invalid shape.")
    else:
        safe_lag_by_receiver = np.broadcast_to(safe_lag, raw_correlation.shape)

    finite_safe = np.isfinite(raw_correlation) & safe_lag_by_receiver
    receiver_has_zncc = np.asarray(correlation.valid, dtype=bool) & np.any(
        np.isfinite(raw_correlation), axis=1
    )
    if receiver_has_zncc.shape != (nreceiver,):
        raise ValueError("Correlation valid mask has an invalid shape.")

    local_success = (~dp_failure) & np.isfinite(tracked_shift)
    global_values = tracked_shift[local_success]
    global_median = (
        float(np.median(global_values)) if global_values.size else None
    )

    for receiver in np.flatnonzero(dp_failure):
        receiver = int(receiver)
        left = max(0, receiver - int(local_radius))
        right = min(nreceiver, receiver + int(local_radius) + 1)
        local_indices = np.flatnonzero(local_success[left:right]) + left
        if local_indices.size:
            final_shift[receiver] = float(np.median(tracked_shift[local_indices]))
            fallback_local[receiver] = True
        elif global_median is not None:
            final_shift[receiver] = global_median
            fallback_global[receiver] = True
        else:
            # A per-receiver ZNCC peak is needed only when no finite tracked
            # path exists to borrow.  In particular, a zero-energy candidate
            # trace may have no own ZNCC while still being recoverable from
            # neighboring or reflector/shot-level tracked shifts above.
            if not receiver_has_zncc[receiver]:
                continue
            valid_indices = np.flatnonzero(finite_safe[receiver])
            if valid_indices.size == 0:
                continue
            best = int(valid_indices[np.argmax(raw_correlation[receiver, valid_indices])])
            # Peak refinement must not read a neighboring lag that was
            # rejected by the safe-range or reflector-ownership gate.  A
            # masked neighbor makes ``_refine_peak`` retain the safe integer
            # lag instead of interpolating outside the permitted interval.
            safe_row = np.where(
                finite_safe[receiver], raw_correlation[receiver], np.nan
            )
            final_shift[receiver] = _refine_peak(
                safe_row, best, lags_samples
            ) * float(correlation.dt)
            fallback_argmax[receiver] = True

    return ShiftCompletionResult(
        final_shift=final_shift,
        dp_failure_mask=dp_failure,
        fallback_local_mask=fallback_local,
        fallback_global_mask=fallback_global,
        fallback_argmax_mask=fallback_argmax,
    )
