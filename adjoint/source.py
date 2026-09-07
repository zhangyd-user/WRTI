"""Wang et al. selected-window WRTI adjoint source (Eq. 4)."""

from __future__ import annotations

import numpy as np


def _time_derivatives(data: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return centered first and second derivatives with zero end samples."""

    first = np.zeros_like(data, dtype=float)
    second = np.zeros_like(data, dtype=float)
    if data.shape[-1] >= 3:
        first[..., 1:-1] = (data[..., 2:] - data[..., :-2]) / (2.0 * dt)
        second[..., 1:-1] = (
            data[..., 2:] - 2.0 * data[..., 1:-1] + data[..., :-2]
        ) / (dt * dt)
    return first, second


def build_wrti_adjoint_source(
    observed_data,
    synthetic_data,
    evaluation_result,
    reference_state,
    dt,
    *,
    denominator_regularization=0.10,
    reflector_weights=None,
    return_diagnostics=False,
):
    """Build Eq. (4) on frozen reflector windows and sum overlaps.

    The observed derivatives are sampled at ``t + shift_time``. The raw
    reciprocal is Tikhonov-stabilized to avoid cancellation amplification.
    """

    observed = np.asarray(observed_data, dtype=float)
    synthetic = np.asarray(synthetic_data, dtype=float)
    if observed.ndim != 3 or observed.shape != synthetic.shape:
        raise ValueError(
            "observed_data and synthetic_data must have the same [ns,nr,nt] shape."
        )
    if not np.isfinite(observed).all() or not np.isfinite(synthetic).all():
        raise ValueError("WRTI adjoint inputs must contain only finite samples.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    denominator_regularization = float(denominator_regularization)
    if (
        not np.isfinite(denominator_regularization)
        or denominator_regularization <= 0.0
    ):
        raise ValueError("denominator_regularization must be finite and positive.")

    windows = reference_state.windows
    shifts = np.asarray(evaluation_result.shift_time, dtype=float)
    fixed = np.asarray(evaluation_result.fixed_mask, dtype=bool)
    success = np.asarray(evaluation_result.tracking_success, dtype=bool)
    path_failure = np.asarray(evaluation_result.path_failure_mask, dtype=bool)
    window_valid = np.asarray(windows.valid, dtype=bool)
    left = np.asarray(windows.left_sample, dtype=int)
    right = np.asarray(windows.right_sample, dtype=int)
    expected = (fixed.shape[0], observed.shape[0], observed.shape[1])
    for name, value in {
        "shift_time": shifts,
        "fixed_mask": fixed,
        "tracking_success": success,
        "path_failure_mask": path_failure,
        "window valid": window_valid,
        "window left_sample": left,
        "window right_sample": right,
    }.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")

    if reflector_weights is None:
        weights = np.ones(expected[0], dtype=float)
    else:
        weights = np.asarray(reflector_weights, dtype=float)
        if (
            weights.shape != (expected[0],)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ValueError(
                "reflector_weights must be a finite non-negative sequence with one value per reflector."
            )

    first, second = _time_derivatives(observed, float(dt))
    source = np.zeros_like(observed, dtype=float)
    sample_axis = np.arange(observed.shape[-1], dtype=float)
    # A finite fallback shift remains part of the frozen objective.
    eligible = fixed & success & np.isfinite(shifts) & window_valid
    active_by_reflector_shot = np.zeros((expected[0], observed.shape[0]), dtype=int)
    denominator_failures = 0
    out_of_bounds = 0
    regularized_count = 0
    min_cancellation_ratio = np.inf
    per_reflector = []

    for reflector in range(expected[0]):
        built = 0
        reflector_denominator_failures = 0
        reflector_out_of_bounds = 0
        for shot, receiver in np.argwhere(eligible[reflector]):
            lo = int(left[reflector, shot, receiver])
            hi = int(right[reflector, shot, receiver])
            if lo < 0 or hi < lo or hi >= observed.shape[-1]:
                reflector_out_of_bounds += 1
                continue

            indices = np.arange(lo, hi + 1)
            shifted = indices.astype(float) + shifts[reflector, shot, receiver] / dt
            if shifted[0] < 0.0 or shifted[-1] > sample_axis[-1]:
                reflector_out_of_bounds += 1
                continue

            observed_first = np.interp(shifted, sample_axis, first[shot, receiver])
            observed_second = np.interp(shifted, sample_axis, second[shot, receiver])
            products = observed_second * synthetic[shot, receiver, indices]
            denominator = float(np.sum(products) * dt)
            scale = float(np.sum(np.abs(products)) * dt)
            if not np.isfinite(denominator) or not np.isfinite(scale) or scale <= 0.0:
                reflector_denominator_failures += 1
                continue

            cancellation_ratio = abs(denominator) / scale
            min_cancellation_ratio = min(min_cancellation_ratio, cancellation_ratio)
            if cancellation_ratio < denominator_regularization:
                regularized_count += 1
            epsilon = denominator_regularization * scale
            inverse_denominator = denominator / (denominator * denominator + epsilon * epsilon)
            contribution = (
                -weights[reflector]
                * shifts[reflector, shot, receiver]
                * observed_first
                * inverse_denominator
            )
            source[shot, receiver, indices] += contribution
            built += 1
            active_by_reflector_shot[reflector, shot] += 1

        denominator_failures += reflector_denominator_failures
        out_of_bounds += reflector_out_of_bounds
        per_reflector.append(
            {
                "eligible": int(np.count_nonzero(eligible[reflector])),
                "active": built,
                "fallback_used": int(
                    np.count_nonzero(eligible[reflector] & path_failure[reflector])
                ),
                "denominator_failures": reflector_denominator_failures,
                "out_of_bounds": reflector_out_of_bounds,
            }
        )

    diagnostics = {
        "eligible_reflector_traces": int(np.count_nonzero(eligible)),
        "active_reflector_traces": int(sum(item["active"] for item in per_reflector)),
        "fallbacks_used": int(np.count_nonzero(eligible & path_failure)),
        "path_failures_skipped": int(np.count_nonzero(fixed & path_failure & ~eligible)),
        "denominator_failures": denominator_failures,
        "out_of_bounds": out_of_bounds,
        "regularized_denominators": regularized_count,
        "min_cancellation_ratio": (
            float(min_cancellation_ratio)
            if np.isfinite(min_cancellation_ratio)
            else float("nan")
        ),
        "denominator_regularization": denominator_regularization,
        "nonfinite_output": int(np.count_nonzero(~np.isfinite(source))),
        "rms": float(np.sqrt(np.mean(source * source))),
        "abs_max": float(np.max(np.abs(source))),
        "per_reflector": per_reflector,
        "shot_31_active": (
            active_by_reflector_shot[:, 30].tolist() if observed.shape[0] > 30 else []
        ),
    }
    if diagnostics["nonfinite_output"]:
        raise FloatingPointError("WRTI adjoint source contains NaN or Inf.")
    if return_diagnostics:
        return source, diagnostics
    return source
