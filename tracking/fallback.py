"""Finite lag completion for receivers not covered by a trusted DP path."""

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
    failure_mask: np.ndarray | None = None,
    receiver_mask: np.ndarray | None = None,
    event_collapse: bool = False,
    shot_global_shift: float | None = None,
    argmax_trust_correlation: float | None = None,
) -> ShiftCompletionResult:
    """Complete post-DP failures without changing the DP result itself.

    Normal/point failures keep the legacy completion order:

        trusted local median -> same-Rk global median -> safe raw-ZNCC argmax.

    ``failure_mask`` is expected to contain the rewritten post-DP failure
    classification (legacy hard failures plus low-quality point failures).
    Only receivers not present in ``failure_mask`` are allowed to donate lags
    to local/global medians.

    If ``event_collapse`` is true, the whole requested receiver scope is
    replaced by one shot-level global lag supplied by the caller.  That lag is
    the median of the global lags from all non-collapsed reflectors in the same
    shot.  If no such lag exists, zero is used, i.e. no local correction is
    added to the reflector's Eikonal theoretical time.

    As before, fallback changes only ``final_shift``.  Failed receivers remain
    marked in ``dp_failure_mask`` so the existing ``path_failures`` diagnostic
    counts every value repaired by fallback.

    ``argmax_trust_correlation`` is intentionally optional.  ``None`` keeps
    the legacy final argmax behavior (used by reference/Tobs construction).
    Candidate dual-center production passes the post-DP trust threshold, so a
    point rejected for low |ZNCC| cannot be reintroduced by the last fallback.
    """

    if not np.isfinite(safe_lag_range_time) or safe_lag_range_time < 0:
        raise ValueError("safe_lag_range_time must be finite and non-negative.")
    if (
        isinstance(local_radius, bool)
        or int(local_radius) != local_radius
        or local_radius < 0
    ):
        raise ValueError("local_radius must be a non-negative integer.")
    if not isinstance(event_collapse, (bool, np.bool_)):
        raise ValueError("event_collapse must be boolean.")
    if argmax_trust_correlation is not None and (
        not np.isfinite(argmax_trust_correlation)
        or not 0.0 <= float(argmax_trust_correlation) <= 1.0
    ):
        raise ValueError(
            "argmax_trust_correlation must be finite and in [0, 1], or None."
        )

    tracked_shift = np.asarray(tracking.shift_time, dtype=float)
    nreceiver = tracked_shift.size

    if receiver_mask is None:
        scope = np.ones(nreceiver, dtype=bool)
    else:
        scope = np.asarray(receiver_mask, dtype=bool)
        if scope.shape != (nreceiver,):
            raise ValueError(
                f"receiver_mask must have shape {(nreceiver,)}; got {scope.shape}."
            )

    if failure_mask is None:
        dp_failure = path_failure_mask(tracking) & scope
    else:
        dp_failure = np.asarray(failure_mask, dtype=bool)
        if dp_failure.shape != (nreceiver,):
            raise ValueError(
                f"failure_mask must have shape {(nreceiver,)}; got {dp_failure.shape}."
            )
        dp_failure = dp_failure & scope

    if event_collapse:
        # A collapsed event has no trustworthy same-Rk path.  Ensure the whole
        # objective scope is reported through the existing path-failure mask.
        dp_failure = np.array(scope, dtype=bool, copy=True)

    lags_time = np.asarray(correlation.lags_time, dtype=float)
    lags_samples = np.asarray(correlation.lags_samples, dtype=int)
    if lags_time.ndim != 1 or lags_samples.shape != lags_time.shape:
        raise ValueError("Correlation lag axes are inconsistent.")

    # Raw waveform ZNCC before optional guide/fine gates.  Older result objects
    # fall back to the public correlation matrix for compatibility.
    raw_correlation = getattr(correlation, "waveform_correlation", None)
    if raw_correlation is None:
        raw_correlation = correlation.correlation
    raw_correlation = np.asarray(raw_correlation, dtype=float)
    if raw_correlation.ndim != 2 or raw_correlation.shape[1] != lags_time.size:
        raise ValueError("Correlation matrix and lag axes are inconsistent.")
    if raw_correlation.shape[0] != nreceiver:
        raise ValueError("Tracking and correlation receiver counts differ.")

    final_shift = np.array(tracked_shift, dtype=float, copy=True)
    fallback_local = np.zeros(nreceiver, dtype=bool)
    fallback_global = np.zeros(nreceiver, dtype=bool)
    fallback_argmax = np.zeros(nreceiver, dtype=bool)

    # Whole-event collapse is completed directly from the shot-level global
    # lag.  It must not borrow from this reflector's own unreliable DP path.
    if event_collapse:
        replacement = (
            float(shot_global_shift)
            if shot_global_shift is not None and np.isfinite(shot_global_shift)
            else 0.0
        )
        final_shift[scope] = replacement
        fallback_global[scope] = True
        return ShiftCompletionResult(
            final_shift=final_shift,
            dp_failure_mask=dp_failure,
            fallback_local_mask=fallback_local,
            fallback_global_mask=fallback_global,
            fallback_argmax_mask=fallback_argmax,
        )

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

    # Only trusted, non-failed DP measurements may donate a local/global lag.
    local_success = scope & (~dp_failure) & np.isfinite(tracked_shift)
    global_values = tracked_shift[local_success]
    global_median = (
        float(np.median(global_values)) if global_values.size else None
    )

    for receiver in np.flatnonzero(dp_failure & scope):
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
            # Preserve the legacy final fallback for isolated point failures.
            # It is deliberately not used for whole-event collapse above.
            if not receiver_has_zncc[receiver]:
                continue
            valid_indices = np.flatnonzero(finite_safe[receiver])
            if valid_indices.size == 0:
                continue

            if argmax_trust_correlation is None:
                # Exact legacy behavior for reference/Tobs construction.
                best = int(
                    valid_indices[
                        np.argmax(raw_correlation[receiver, valid_indices])
                    ]
                )
                refine_row = np.where(
                    finite_safe[receiver], raw_correlation[receiver], np.nan
                )
            else:
                # Candidate production: a low-quality point rejected by the
                # post-DP trust test must not be put back by the final argmax.
                values = raw_correlation[receiver, valid_indices]
                trusted_indices = valid_indices[
                    np.abs(values) >= float(argmax_trust_correlation)
                ]
                if trusted_indices.size == 0:
                    continue

                trusted_values = raw_correlation[receiver, trusted_indices]
                best = int(
                    trusted_indices[np.argmax(np.abs(trusted_values))]
                )

                # _refine_peak refines a maximum.  If the selected physical
                # peak is negative-polarity, flip the row only for refinement
                # so the strongest negative peak is refined as a positive max.
                peak_sign = (
                    -1.0 if raw_correlation[receiver, best] < 0.0 else 1.0
                )
                trusted_safe = (
                    finite_safe[receiver]
                    & (np.abs(raw_correlation[receiver])
                       >= float(argmax_trust_correlation))
                )
                refine_row = np.where(
                    trusted_safe,
                    peak_sign * raw_correlation[receiver],
                    np.nan,
                )

            final_shift[receiver] = _refine_peak(
                refine_row, best, lags_samples
            ) * float(correlation.dt)
            fallback_argmax[receiver] = True

    return ShiftCompletionResult(
        final_shift=final_shift,
        dp_failure_mask=dp_failure,
        fallback_local_mask=fallback_local,
        fallback_global_mask=fallback_global,
        fallback_argmax_mask=fallback_argmax,
    )
