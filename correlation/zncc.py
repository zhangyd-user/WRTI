"""Fixed-window zero-mean normalized cross-correlation for WRTI Step 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve, hilbert

from ..window import WindowResult


class CorrelationError(ValueError):
    """Raised when fixed-window ZNCC inputs are inconsistent."""


PreprocessHook = Callable[
    [np.ndarray, np.ndarray], Optional[tuple[np.ndarray, np.ndarray]]
]


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CorrelationResult:
    """ZNCC for one reflector and one shot.

    ``correlation`` has shape ``[nreceiver, nlag]``.  ``valid`` has shape
    ``[nreceiver]`` and records geometric window validity plus a non-zero
    synthetic zero-mean energy.  Individual lag values whose observed
    energy is zero remain ``NaN`` in ``correlation``.  ``lag_valid`` is the
    explicit per-lag state mask after reflector ownership and optional
    envelope/fine restrictions; rejected states are also stored as ``NaN``.

    ``coarse_lag_*`` records the ownership-constrained, seed-directed envelope
    ridge used to define the waveform fine search.  It is diagnostic metadata
    only: the final WRTI lag still comes from the waveform correlation matrix.

    ``waveform_correlation`` is the raw waveform ZNCC before the optional
    envelope/fine lag gate.  It remains the source for final measurement,
    QC, and fallback.  ``tracking_correlation`` is an optional AGC/stacked
    guide matrix used only to identify the DP ridge.

    ``window_energy_obs`` is the weighted zero-mean observed energy at zero
    lag.  The denominator used for the correlation itself is recomputed for
    every lag because the observed window shifts.

    ``ownership_center_time`` is an optional complete center field used only
    for reflector ownership bounds.  The target window still comes from
    ``window_result``.
    """

    lags_samples: np.ndarray
    lags_time: np.ndarray
    correlation: np.ndarray
    valid: np.ndarray
    window_energy_obs: np.ndarray
    window_energy_syn: np.ndarray
    reflector: int
    shot: int
    max_lag_samples: int
    dt: float
    # Receiver-level ``valid`` remains the geometric/energy flag.  The new
    # reflector and fine-search restrictions are lag dependent.
    lag_valid: np.ndarray | None = None
    coarse_lag_samples: np.ndarray | None = None
    coarse_lag_time: np.ndarray | None = None
    coarse_correlation_peak: np.ndarray | None = None
    ownership_lag_min: np.ndarray | None = None
    ownership_lag_max: np.ndarray | None = None
    waveform_correlation: np.ndarray | None = None
    tracking_correlation: np.ndarray | None = None
    coarse_search_center_time: np.ndarray | None = None
    coarse_tracking_fallback_mask: np.ndarray | None = None
    coarse_seed_receiver: int = -1

    def __post_init__(self) -> None:
        lags_samples = np.asarray(self.lags_samples, dtype=int)
        lags_time = np.asarray(self.lags_time, dtype=float)
        correlation = np.asarray(self.correlation, dtype=float)
        waveform_correlation = (
            correlation
            if self.waveform_correlation is None
            else np.asarray(self.waveform_correlation, dtype=float)
        )
        tracking_correlation = (
            None
            if self.tracking_correlation is None
            else np.asarray(self.tracking_correlation, dtype=float)
        )
        valid = np.asarray(self.valid, dtype=bool)
        energy_obs = np.asarray(self.window_energy_obs, dtype=float)
        energy_syn = np.asarray(self.window_energy_syn, dtype=float)

        if lags_samples.ndim != 1 or lags_time.shape != lags_samples.shape:
            raise CorrelationError("lags_samples and lags_time must be 1-D arrays with equal length.")
        if correlation.ndim != 2 or correlation.shape[1] != lags_samples.size:
            raise CorrelationError("correlation must have shape [nreceiver, nlag].")
        if waveform_correlation.shape != correlation.shape:
            raise CorrelationError(
                "waveform_correlation must have the same shape as correlation."
            )
        if tracking_correlation is not None and tracking_correlation.shape != correlation.shape:
            raise CorrelationError(
                "tracking_correlation must have the same shape as correlation."
            )
        n_receiver = correlation.shape[0]
        for name, value in {
            "valid": valid,
            "window_energy_obs": energy_obs,
            "window_energy_syn": energy_syn,
        }.items():
            if value.shape != (n_receiver,):
                raise CorrelationError(f"{name} must have shape [nreceiver].")
        if self.lag_valid is None:
            lag_valid = np.isfinite(correlation)
        else:
            lag_valid = np.asarray(self.lag_valid, dtype=bool)
            if lag_valid.shape != correlation.shape:
                raise CorrelationError("lag_valid must have shape [nreceiver, nlag].")
            lag_valid = lag_valid & np.isfinite(correlation)

        def _receiver_array(value, name, default):
            if value is None:
                return np.full(n_receiver, default, dtype=float)
            result = np.asarray(value, dtype=float)
            if result.shape != (n_receiver,):
                raise CorrelationError(f"{name} must have shape [nreceiver].")
            if not (np.isfinite(result) | np.isnan(result)).all():
                raise CorrelationError(f"{name} must contain finite values or NaN.")
            return result

        coarse_samples = _receiver_array(
            self.coarse_lag_samples, "coarse_lag_samples", np.nan
        )
        coarse_time = _receiver_array(self.coarse_lag_time, "coarse_lag_time", np.nan)
        if self.coarse_lag_time is None and self.coarse_lag_samples is not None:
            coarse_time = coarse_samples * float(self.dt)
        ownership_min = _receiver_array(
            self.ownership_lag_min,
            "ownership_lag_min",
            -float(self.max_lag_samples) * float(self.dt),
        )
        ownership_max = _receiver_array(
            self.ownership_lag_max,
            "ownership_lag_max",
            float(self.max_lag_samples) * float(self.dt),
        )
        coarse_search_center = _receiver_array(
            self.coarse_search_center_time,
            "coarse_search_center_time",
            np.nan,
        )
        if self.coarse_tracking_fallback_mask is None:
            coarse_fallback_mask = np.zeros(n_receiver, dtype=bool)
        else:
            coarse_fallback_mask = np.asarray(
                self.coarse_tracking_fallback_mask, dtype=bool
            )
            if coarse_fallback_mask.shape != (n_receiver,):
                raise CorrelationError(
                    "coarse_tracking_fallback_mask must have shape [nreceiver]."
                )
        if not np.isfinite(lags_time).all():
            raise CorrelationError("lags_time must be finite.")
        if (
            isinstance(self.reflector, bool)
            or isinstance(self.shot, bool)
            or int(self.reflector) != self.reflector
            or int(self.shot) != self.shot
        ):
            raise CorrelationError("reflector and shot must be integer indices.")
        if (
            isinstance(self.max_lag_samples, bool)
            or int(self.max_lag_samples) != self.max_lag_samples
            or self.max_lag_samples < 0
        ):
            raise CorrelationError("max_lag_samples must be a non-negative integer.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise CorrelationError("dt must be finite and positive.")
        if (
            isinstance(self.coarse_seed_receiver, bool)
            or int(self.coarse_seed_receiver) != self.coarse_seed_receiver
            or not -1 <= int(self.coarse_seed_receiver) < n_receiver
        ):
            raise CorrelationError(
                "coarse_seed_receiver must be -1 or a receiver index."
            )
        object.__setattr__(self, "lags_samples", _readonly(lags_samples, int))
        object.__setattr__(self, "lags_time", _readonly(lags_time, float))
        object.__setattr__(self, "correlation", _readonly(correlation, float))
        object.__setattr__(
            self,
            "waveform_correlation",
            _readonly(waveform_correlation, float),
        )
        object.__setattr__(
            self,
            "tracking_correlation",
            None
            if tracking_correlation is None
            else _readonly(tracking_correlation, float),
        )
        object.__setattr__(self, "valid", _readonly(valid, bool))
        object.__setattr__(self, "lag_valid", _readonly(lag_valid, bool))
        object.__setattr__(self, "window_energy_obs", _readonly(energy_obs, float))
        object.__setattr__(self, "window_energy_syn", _readonly(energy_syn, float))
        object.__setattr__(self, "reflector", int(self.reflector))
        object.__setattr__(self, "shot", int(self.shot))
        object.__setattr__(self, "max_lag_samples", int(self.max_lag_samples))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "coarse_lag_samples", _readonly(coarse_samples, float))
        object.__setattr__(self, "coarse_lag_time", _readonly(coarse_time, float))
        object.__setattr__(
            self,
            "coarse_correlation_peak",
            _readonly(
                _receiver_array(
                    self.coarse_correlation_peak,
                    "coarse_correlation_peak",
                    np.nan,
                ),
                float,
            ),
        )
        object.__setattr__(self, "ownership_lag_min", _readonly(ownership_min, float))
        object.__setattr__(self, "ownership_lag_max", _readonly(ownership_max, float))
        object.__setattr__(
            self,
            "coarse_search_center_time",
            _readonly(coarse_search_center, float),
        )
        object.__setattr__(
            self,
            "coarse_tracking_fallback_mask",
            _readonly(coarse_fallback_mask, bool),
        )
        object.__setattr__(
            self, "coarse_seed_receiver", int(self.coarse_seed_receiver)
        )

    @property
    def coarse_seed_lag_time(self) -> float:
        if self.coarse_seed_receiver < 0:
            return float("nan")
        return float(self.coarse_lag_time[self.coarse_seed_receiver])

    @property
    def coarse_tracking_fallback_count(self) -> int:
        return int(np.count_nonzero(self.coarse_tracking_fallback_mask))


def _validate_data(
    observed: np.ndarray,
    synthetic: np.ndarray,
    window_result: WindowResult,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    syn = np.asarray(synthetic, dtype=float)
    if obs.ndim != 3 or syn.ndim != 3:
        raise CorrelationError("observed and synthetic must have shape [ns, nr, nt].")
    if obs.shape != syn.shape:
        raise CorrelationError("observed and synthetic must have equal shapes.")
    nref, ns, nr = window_result.shape
    expected = (ns, nr, window_result.nt)
    if obs.shape != expected:
        raise CorrelationError(
            f"observed and synthetic must have shape {expected}; got {obs.shape}."
        )
    if not np.isfinite(obs).all() or not np.isfinite(syn).all():
        raise CorrelationError("observed and synthetic must contain only finite values.")
    return obs, syn


def _prepare_data(
    observed: np.ndarray,
    synthetic: np.ndarray,
    window_result: WindowResult,
    preprocess_hook: PreprocessHook | None,
) -> tuple[np.ndarray, np.ndarray]:
    obs, syn = _validate_data(observed, synthetic, window_result)
    if preprocess_hook is None:
        return obs, syn

    processed = preprocess_hook(obs, syn)
    if processed is None:
        processed_obs, processed_syn = obs, syn
    elif isinstance(processed, (tuple, list)) and len(processed) == 2:
        processed_obs, processed_syn = processed
    else:
        raise CorrelationError(
            "preprocess_hook must return (observed, synthetic) or None."
        )
    return _validate_data(processed_obs, processed_syn, window_result)


def _normalise_max_lag(
    max_lag_time: float,
    dt: float,
    window_result: WindowResult,
) -> int:
    if not np.isfinite(max_lag_time) or max_lag_time < 0:
        raise CorrelationError("max_lag_time must be finite and non-negative.")
    max_lag_samples = int(round(float(max_lag_time) / float(dt)))
    if window_result.max_lag_samples != max_lag_samples:
        raise CorrelationError(
            "WindowResult.max_lag_samples does not match round(max_lag_time / dt); "
            "rebuild the fixed WindowResult with the same lag configuration."
        )
    return max_lag_samples


def _ownership_lag_gate(
    center_time: np.ndarray,
    reflector: int,
    max_lag_time: float,
    lags_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reflector-identity gate from neighboring fixed centers.

    The gate is deliberately independent of ``half_window_time``.  The
    returned bounds are the intersection with the global lag interval and
    are expressed in seconds for diagnostic use.
    """

    centers = np.asarray(center_time, dtype=float)
    if centers.ndim != 2:
        raise CorrelationError("center_time must have shape [nreflector, nreceiver].")
    nref, nreceiver = centers.shape
    if not 0 <= reflector < nref:
        raise IndexError("reflector index is out of range.")
    target = centers[reflector]
    lower = np.full(nreceiver, -float(max_lag_time), dtype=float)
    upper = np.full(nreceiver, float(max_lag_time), dtype=float)
    finite = np.isfinite(target)

    if reflector > 0:
        previous = centers[reflector - 1]
        finite &= np.isfinite(previous)
        lower = np.maximum(lower, 0.5 * (previous + target) - target)
    if reflector + 1 < nref:
        following = centers[reflector + 1]
        finite &= np.isfinite(following)
        upper = np.minimum(upper, 0.5 * (target + following) - target)

    lags = np.asarray(lags_time, dtype=float)
    allowed = (
        finite[:, None]
        & (lags[None, :] > lower[:, None])
        & (lags[None, :] < upper[:, None])
    )
    lower[~finite] = np.nan
    upper[~finite] = np.nan
    return allowed, lower, upper


def _weighted_zncc_lags(
    observed_trace: np.ndarray,
    target_trace: np.ndarray,
    local_weights: np.ndarray,
    left: int,
    lags_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vectorized weighted zero-mean correlation for a fixed target window."""

    weights = np.asarray(local_weights, dtype=float)
    target = np.asarray(target_trace, dtype=float)
    observed = np.asarray(observed_trace, dtype=float)
    weight_sum = float(np.sum(weights))
    target_mean = float(np.sum(weights * target) / weight_sum)
    target_centered = target - target_mean
    target_energy = float(np.sum(weights * target_centered * target_centered))
    if not np.isfinite(target_energy) or target_energy <= 0:
        return np.full(lags_samples.size, np.nan, dtype=float), np.full(
            lags_samples.size, np.nan, dtype=float
        ), target_energy

    weighted_target = weights * target_centered
    numerator_all = fftconvolve(observed, weighted_target[::-1], mode="valid")
    observed_sum_all = fftconvolve(observed, weights[::-1], mode="valid")
    observed_square_sum_all = fftconvolve(
        observed * observed, weights[::-1], mode="valid"
    )
    starts = int(left) + np.asarray(lags_samples, dtype=int)
    numerator = numerator_all[starts]
    observed_sum = observed_sum_all[starts]
    observed_square_sum = observed_square_sum_all[starts]
    observed_energy = observed_square_sum - observed_sum * observed_sum / weight_sum
    observed_energy = np.maximum(observed_energy, 0.0)
    denominator = np.sqrt(target_energy * observed_energy)
    good = np.isfinite(denominator) & (denominator > 0)
    result = np.full(lags_samples.size, np.nan, dtype=float)
    result[good] = np.clip(numerator[good] / denominator[good], -1.0, 1.0)
    return result, observed_energy, target_energy


def _tracking_local_rms_normalize(
    traces: np.ndarray,
    *,
    fraction: float = 0.25,
    floor_ratio: float = 0.20,
) -> np.ndarray:
    """Return a non-mutating, centered local-RMS tracking copy."""

    values = np.asarray(traces, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise CorrelationError("tracking traces must be a finite [nreceiver, ntime] array.")
    if not 0 < fraction <= 1 or not np.isfinite(fraction):
        raise CorrelationError("tracking_agc_fraction must be in (0, 1].")
    if not np.isfinite(floor_ratio) or floor_ratio <= 0:
        raise CorrelationError("tracking_agc_floor_ratio must be positive.")
    length = max(int(round(fraction * values.shape[1])), 3)
    if length % 2 == 0:
        length += 1
    half = length // 2
    padded = np.pad(values * values, ((0, 0), (half, half)), mode="edge")
    cumulative = np.pad(
        np.cumsum(padded, axis=1), ((0, 0), (1, 0)), mode="constant"
    )
    local_rms = np.sqrt((cumulative[:, length:] - cumulative[:, :-length]) / length)
    global_rms = np.sqrt(np.mean(values * values, axis=1, keepdims=True))
    denominator = np.maximum(
        local_rms,
        np.maximum(floor_ratio * global_rms, np.finfo(float).eps),
    )
    return values / denominator


def _tracking_receiver_stack(
    traces: np.ndarray,
    receiver_valid: np.ndarray,
) -> np.ndarray:
    """Return a normalized 3-receiver stack without crossing invalid gaps."""

    values = np.asarray(traces, dtype=float)
    valid = np.asarray(receiver_valid, dtype=bool)
    if values.ndim != 2 or valid.shape != (values.shape[0],):
        raise CorrelationError(
            "tracking receiver stack needs [nreceiver, ntime] traces and [nreceiver] validity."
        )
    output = np.array(values, copy=True)
    numerator = 0.5 * values * valid[:, None]
    weights = 0.5 * valid.astype(float)
    numerator[1:] += 0.25 * values[:-1] * valid[:-1, None]
    weights[1:] += 0.25 * valid[:-1]
    numerator[:-1] += 0.25 * values[1:] * valid[1:, None]
    weights[:-1] += 0.25 * valid[1:]
    output[valid] = numerator[valid] / weights[valid, None]
    return output


def _tracking_common_buffer_valid(
    window_result: WindowResult,
    reflector: int,
    shot: int,
    receiver_valid: np.ndarray,
    max_lag_samples: int,
    nt: int,
) -> np.ndarray:
    """Return receivers that safely share one center-aligned tracking buffer.

    Step 4 receiver validity only guarantees that each receiver's *own* target
    window supports the full +/- lag search.  AGC/receiver stacking uses one
    common center-aligned buffer whose extents are the maxima over all selected
    receivers, so a receiver near a trace boundary can still become unsafe.

    Starting from the waveform-valid receivers, iteratively remove rows that
    cannot contain the current common extents.  The fixed point guarantees that
    ``_tracking_local_buffers`` cannot cross trace bounds.  Removed rows are not
    discarded from the measurement; callers keep raw waveform ZNCC as the
    tracking-guide fallback for those receivers.
    """

    valid = np.asarray(receiver_valid, dtype=bool)
    if valid.ndim != 1:
        raise CorrelationError("receiver_valid must be a one-dimensional array.")
    if not isinstance(nt, (int, np.integer)) or int(nt) <= 0:
        raise CorrelationError("tracking trace length nt must be a positive integer.")
    if (
        isinstance(max_lag_samples, bool)
        or int(max_lag_samples) != max_lag_samples
        or max_lag_samples < 0
    ):
        raise CorrelationError("max_lag_samples must be a non-negative integer.")

    center = np.asarray(window_result.center_sample[reflector, shot], dtype=int)
    left = np.asarray(window_result.left_sample[reflector, shot], dtype=int)
    right = np.asarray(window_result.right_sample[reflector, shot], dtype=int)
    if center.shape != valid.shape or left.shape != valid.shape or right.shape != valid.shape:
        raise CorrelationError(
            "tracking window center/left/right arrays must match receiver_valid."
        )

    tracking_valid = np.array(valid, copy=True)
    while np.any(tracking_valid):
        left_extent = (
            int(np.max(center[tracking_valid] - left[tracking_valid]))
            + int(max_lag_samples)
        )
        right_extent = (
            int(np.max(right[tracking_valid] - center[tracking_valid]))
            + int(max_lag_samples)
        )
        fits_common_buffer = (
            (center - left_extent >= 0)
            & (center + right_extent < int(nt))
        )
        updated = tracking_valid & fits_common_buffer
        if np.array_equal(updated, tracking_valid):
            break
        tracking_valid = updated

    return tracking_valid


def _tracking_local_buffers(
    observed_shot: np.ndarray,
    synthetic_shot: np.ndarray,
    window_result: WindowResult,
    reflector: int,
    shot: int,
    receiver_valid: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center-aligned local buffers and original target offsets.

    Every valid receiver uses its own reflector center as local time zero.  A
    common buffer therefore aligns moveout before stacking while retaining the
    original ``left:right`` target and its complete +/- lag search support.
    """

    observed = np.asarray(observed_shot, dtype=float)
    synthetic = np.asarray(synthetic_shot, dtype=float)
    valid = np.asarray(receiver_valid, dtype=bool)
    if observed.ndim != 2 or synthetic.shape != observed.shape:
        raise CorrelationError("tracking buffers need equal [nreceiver, ntime] gathers.")
    if valid.shape != (observed.shape[0],):
        raise CorrelationError("receiver_valid must have shape [nreceiver].")
    if not np.any(valid):
        raise CorrelationError("tracking buffers require at least one valid receiver.")

    center = np.asarray(window_result.center_sample[reflector, shot], dtype=int)
    left = np.asarray(window_result.left_sample[reflector, shot], dtype=int)
    right = np.asarray(window_result.right_sample[reflector, shot], dtype=int)
    left_extent = int(np.max(center[valid] - left[valid])) + max_lag_samples
    right_extent = int(np.max(right[valid] - center[valid])) + max_lag_samples
    offsets = np.arange(-left_extent, right_extent + 1, dtype=int)
    # Invalid rows are never used for tracking, but their indices must still
    # be safe for vectorized gather extraction.
    safe_center = np.where(valid, center, left_extent)
    indices = safe_center[:, None] + offsets
    if (
        np.any(indices[valid] < 0)
        or np.any(indices[valid] >= observed.shape[1])
    ):
        raise CorrelationError("tracking local buffer exceeds trace bounds.")
    indices = np.clip(indices, 0, observed.shape[1] - 1)
    receiver = np.arange(observed.shape[0], dtype=int)[:, None]
    target_left = left - safe_center + left_extent
    return (
        observed[receiver, indices],
        synthetic[receiver, indices],
        target_left,
    )


def _seeded_envelope_path(
    correlation: np.ndarray,
    allowed: np.ndarray,
    lags_samples: np.ndarray,
    receiver_x: np.ndarray,
    source_x: float,
    seed_lag_range_samples: int,
    epsilon_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Track an envelope ridge from the near-zero-offset receiver outward.

    The left and right receiver directions are tracked independently.  After
    the first adjacent receiver, each direction uses the mean of its two most
    recent valid coarse lags as the next search center.  The envelope is only
    used to choose a local maximum; it never supplies a hard correlation
    threshold.
    """

    values = np.asarray(correlation, dtype=float)
    allowed_mask = np.asarray(allowed, dtype=bool)
    lags = np.asarray(lags_samples, dtype=int)
    if values.ndim != 2 or allowed_mask.shape != values.shape:
        raise CorrelationError(
            "Envelope correlation and allowed mask must have shape "
            "[nreceiver, nlag]."
        )
    state_valid = allowed_mask & np.isfinite(values)
    if lags.ndim != 1 or lags.size != values.shape[1]:
        raise CorrelationError("lags_samples must have length nlag.")
    nreceiver = values.shape[0]
    receiver_axis = np.asarray(receiver_x, dtype=float)
    if receiver_axis.shape != (nreceiver,) or not np.isfinite(receiver_axis).all():
        raise CorrelationError("receiver_x must be finite with shape [nreceiver].")
    if not np.isfinite(source_x):
        raise CorrelationError("source_x must be finite.")
    for value, name in (
        (seed_lag_range_samples, "seed_lag_range_samples"),
        (epsilon_samples, "epsilon_samples"),
    ):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise CorrelationError(f"{name} must be a non-negative integer.")

    path = np.full(nreceiver, -1, dtype=int)
    search_center = np.full(nreceiver, np.nan, dtype=float)
    fallback_mask = np.zeros(nreceiver, dtype=bool)

    seed_receiver = int(np.argmin(np.abs(receiver_axis - float(source_x))))
    seed_allowed = state_valid[seed_receiver] & (
        np.abs(lags) <= int(seed_lag_range_samples)
    )
    if not np.any(seed_allowed):
        return path, search_center, fallback_mask, seed_receiver

    seed_candidates = np.flatnonzero(seed_allowed)
    seed_index = int(seed_candidates[np.argmax(values[seed_receiver, seed_candidates])])
    path[seed_receiver] = seed_index
    search_center[seed_receiver] = float(lags[seed_index])
    seed_lag = float(lags[seed_index])

    def track_direction(indices: range) -> None:
        history = [seed_lag]
        for receiver in indices:
            if len(history) >= 2:
                center = 0.5 * (history[-1] + history[-2])
            else:
                center = seed_lag
            search_center[receiver] = center
            local = state_valid[receiver] & (
                np.abs(lags.astype(float) - center) <= int(epsilon_samples)
            )
            candidates = np.flatnonzero(local)
            if candidates.size:
                chosen = int(candidates[np.argmax(values[receiver, candidates])])
            else:
                fallback_mask[receiver] = True
                # Use the ownership mask directly so tracking can continue
                # even when this row's envelope values are all NaN.
                candidates = np.flatnonzero(allowed_mask[receiver])
                if not candidates.size:
                    continue
                distances = np.abs(lags[candidates].astype(float) - center)
                chosen = int(candidates[np.argmin(distances)])
            path[receiver] = chosen
            history.append(float(lags[chosen]))

    track_direction(range(seed_receiver + 1, nreceiver))
    track_direction(range(seed_receiver - 1, -1, -1))

    tracked = (path >= 0) & ~fallback_mask
    if np.any(
        np.abs(lags[path[tracked]].astype(float) - search_center[tracked])
        > int(epsilon_samples)
    ):
        raise AssertionError("Envelope coarse tracking escaped its local search.")
    return path, search_center, fallback_mask, seed_receiver


def _compute_for_one_shot(
    observed: np.ndarray,
    synthetic: np.ndarray,
    window_result: WindowResult,
    reflector: int,
    shot: int,
    max_lag_samples: int,
    dt: float,
    *,
    use_envelope_coarse: bool = True,
    envelope_fine_half_width_time: float = 0.04,
    envelope_tracking_epsilon_time: float = 0.040,
    seed_lag_range_time: float | None = None,
    receiver_x: np.ndarray | None = None,
    source_x: float | None = None,
    observed_envelope: np.ndarray | None = None,
    synthetic_envelope: np.ndarray | None = None,
    ownership_center_time: np.ndarray | None = None,
    fixed_side: str = "synthetic",
    tracking_enhancement_enabled: bool = True,
    tracking_agc_fraction: float = 0.25,
    tracking_agc_floor_ratio: float = 0.20,
    tracking_receiver_stack: bool = True,
) -> CorrelationResult:
    nref, ns, nr = window_result.shape
    if not (0 <= reflector < nref and 0 <= shot < ns):
        raise IndexError("reflector or shot index is out of range.")

    lags_samples = np.arange(
        -max_lag_samples, max_lag_samples + 1, dtype=int
    )
    lags_time = lags_samples.astype(float) * float(dt)
    nlag = lags_samples.size
    correlation = np.full((nr, nlag), np.nan, dtype=float)
    valid = np.zeros(nr, dtype=bool)
    lag_valid = np.zeros((nr, nlag), dtype=bool)
    energy_obs = np.full(nr, np.nan, dtype=float)
    energy_syn = np.full(nr, np.nan, dtype=float)
    coarse_lag_samples = np.full(nr, np.nan, dtype=float)
    coarse_lag_time = np.full(nr, np.nan, dtype=float)
    coarse_peak = np.full(nr, np.nan, dtype=float)
    coarse_search_center_time = np.full(nr, np.nan, dtype=float)
    coarse_tracking_fallback_mask = np.zeros(nr, dtype=bool)
    coarse_seed_receiver = -1
    envelope_correlation = np.full((nr, nlag), np.nan, dtype=float)
    waveform_correlation = np.full((nr, nlag), np.nan, dtype=float)

    if not isinstance(use_envelope_coarse, (bool, np.bool_)):
        raise CorrelationError("use_envelope_coarse must be boolean.")
    if fixed_side not in {"synthetic", "observed"}:
        raise CorrelationError("fixed_side must be 'synthetic' or 'observed'.")
    if not isinstance(tracking_enhancement_enabled, (bool, np.bool_)):
        raise CorrelationError("tracking_enhancement_enabled must be boolean.")
    if not isinstance(tracking_receiver_stack, (bool, np.bool_)):
        raise CorrelationError("tracking_receiver_stack must be boolean.")
    if (
        not np.isfinite(envelope_fine_half_width_time)
        or envelope_fine_half_width_time < 0
    ):
        raise CorrelationError(
            "envelope_fine_half_width_time must be finite and non-negative."
        )
    if use_envelope_coarse:
        if observed_envelope is not None:
            observed_envelope = np.asarray(observed_envelope, dtype=float)
            if observed_envelope.shape != (nr, window_result.nt):
                raise CorrelationError(
                    "observed_envelope must have shape [nreceiver, nt]."
                )
        if synthetic_envelope is not None:
            synthetic_envelope = np.asarray(synthetic_envelope, dtype=float)
            if synthetic_envelope.shape != (nr, window_result.nt):
                raise CorrelationError(
                    "synthetic_envelope must have shape [nreceiver, nt]."
                )
    if ownership_center_time is None:
        center_times = window_result.center_time[:, shot, :]
    else:
        center_times = np.asarray(ownership_center_time, dtype=float)
        if center_times.shape != (nref, nr):
            raise CorrelationError(
                "ownership_center_time must have shape [nreflector, nreceiver] "
                "for the selected shot."
            )
    ownership_allowed, ownership_min, ownership_max = _ownership_lag_gate(
        center_times,
        reflector,
        max_lag_samples * float(dt),
        lags_time if fixed_side == "synthetic" else -lags_time,
    )
    if fixed_side == "observed":
        # Candidate synthetic time is center - lag, so convert the
        # observed-center ownership bounds back to the public lag axis.
        ownership_min, ownership_max = -ownership_max, -ownership_min
    fine_half_width_samples = int(
        np.ceil(float(envelope_fine_half_width_time) / float(dt))
    )
    if (
        not np.isfinite(envelope_tracking_epsilon_time)
        or envelope_tracking_epsilon_time <= 0
    ):
        raise CorrelationError(
            "envelope_tracking_epsilon_time must be finite and positive."
        )
    envelope_tracking_epsilon_samples = int(
        round(float(envelope_tracking_epsilon_time) / float(dt))
    )
    seed_lag_range = (
        float(max_lag_samples) * float(dt)
        if seed_lag_range_time is None
        else float(seed_lag_range_time)
    )
    if not np.isfinite(seed_lag_range) or seed_lag_range < 0:
        raise CorrelationError(
            "seed_lag_range_time must be finite and non-negative."
        )
    if receiver_x is None:
        receiver_axis = np.arange(nr, dtype=float)
    else:
        receiver_axis = np.asarray(receiver_x, dtype=float)
    if receiver_axis.shape != (nr,) or not np.isfinite(receiver_axis).all():
        raise CorrelationError("receiver_x must be finite with shape [nreceiver].")
    source_value = 0.0 if source_x is None else float(source_x)
    if not np.isfinite(source_value):
        raise CorrelationError("source_x must be finite.")

    for receiver in range(nr):
        if not bool(window_result.valid[reflector, shot, receiver]):
            continue

        left = int(window_result.left_sample[reflector, shot, receiver])
        right = int(window_result.right_sample[reflector, shot, receiver])
        length = right - left + 1
        if (
            length <= 0
            or left - max_lag_samples < 0
            or right + max_lag_samples >= window_result.nt
        ):
            continue

        local_weights = window_result.weights_for_trace(
            reflector, shot, receiver
        )[left : right + 1]
        weight_sum = float(np.sum(local_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue

        fixed_trace = (
            synthetic[shot, receiver] if fixed_side == "synthetic" else observed[shot, receiver]
        )
        shifted_trace = (
            observed[shot, receiver] if fixed_side == "synthetic" else synthetic[shot, receiver]
        )
        fixed_target = fixed_trace[left : right + 1]
        fixed_mean = float(
            np.sum(local_weights * fixed_target) / weight_sum
        )
        fixed_centered = fixed_target - fixed_mean
        fixed_energy = float(
            np.sum(local_weights * fixed_centered * fixed_centered)
        )
        if not np.isfinite(fixed_energy) or fixed_energy <= 0:
            if fixed_side == "synthetic":
                energy_syn[receiver] = max(fixed_energy, 0.0)
            else:
                energy_obs[receiver] = max(fixed_energy, 0.0)
            continue

        if fixed_side == "synthetic":
            energy_syn[receiver] = fixed_energy
        else:
            energy_obs[receiver] = fixed_energy
        valid[receiver] = True

        # Build the analytic-signal envelope correlation matrix.  Selection
        # is deferred until every receiver row is available, because the
        # coarse lag is a seed-directed ridge rather than a tracewise argmax.
        # Hilbert is evaluated once per full trace, never once per lag.
        if use_envelope_coarse:
            fixed_envelope = (
                synthetic_envelope[receiver]
                if fixed_side == "synthetic" and synthetic_envelope is not None
                else observed_envelope[receiver]
                if fixed_side == "observed" and observed_envelope is not None
                else np.abs(hilbert(fixed_trace))
            )
            shifted_envelope = (
                observed_envelope[receiver]
                if fixed_side == "synthetic" and observed_envelope is not None
                else synthetic_envelope[receiver]
                if fixed_side == "observed" and synthetic_envelope is not None
                else np.abs(hilbert(shifted_trace))
            )
            envelope_corr, _, _ = _weighted_zncc_lags(
                shifted_envelope,
                fixed_envelope[left : right + 1],
                local_weights,
                left,
                lags_samples if fixed_side == "synthetic" else -lags_samples,
            )
            envelope_correlation[receiver] = envelope_corr

        # shift_time is always T_obs - T_syn.  With a fixed synthetic window
        # positive L shifts observed later; with a fixed observed window it
        # shifts synthetic earlier by -L.
        waveform_corr, shifted_energy, _ = _weighted_zncc_lags(
            shifted_trace,
            fixed_target,
            local_weights,
            left,
            lags_samples if fixed_side == "synthetic" else -lags_samples,
        )
        waveform_correlation[receiver] = waveform_corr
        if fixed_side == "synthetic":
            energy_obs[receiver] = float(shifted_energy[max_lag_samples])
        else:
            energy_syn[receiver] = float(shifted_energy[max_lag_samples])

    tracking_correlation = None
    if tracking_enhancement_enabled:
        # Raw waveform ZNCC is the safe fallback.  AGC/receiver-stack tracking
        # overwrites only receivers that can share one common local buffer.
        # This keeps boundary receivers usable without clipping/padding traces
        # and prevents one unsafe row from aborting the whole shot.
        tracking_correlation = np.array(waveform_correlation, copy=True)
        if np.any(valid):
            tracking_valid = _tracking_common_buffer_valid(
                window_result,
                reflector,
                shot,
                valid,
                max_lag_samples,
                observed.shape[-1],
            )
            if np.any(tracking_valid):
                observed_local, synthetic_local, target_left = _tracking_local_buffers(
                    observed[shot],
                    synthetic[shot],
                    window_result,
                    reflector,
                    shot,
                    tracking_valid,
                    max_lag_samples,
                )
                observed_tracking = _tracking_local_rms_normalize(
                    observed_local,
                    fraction=tracking_agc_fraction,
                    floor_ratio=tracking_agc_floor_ratio,
                )
                synthetic_tracking = _tracking_local_rms_normalize(
                    synthetic_local,
                    fraction=tracking_agc_fraction,
                    floor_ratio=tracking_agc_floor_ratio,
                )
                if tracking_receiver_stack:
                    observed_tracking = _tracking_receiver_stack(
                        observed_tracking, tracking_valid
                    )
                    synthetic_tracking = _tracking_receiver_stack(
                        synthetic_tracking, tracking_valid
                    )
                for receiver in np.flatnonzero(tracking_valid):
                    left = int(window_result.left_sample[reflector, shot, receiver])
                    right = int(window_result.right_sample[reflector, shot, receiver])
                    local_weights = window_result.weights_for_trace(
                        reflector, shot, receiver
                    )[left : right + 1]
                    fixed_tracking = (
                        synthetic_tracking[receiver]
                        if fixed_side == "synthetic"
                        else observed_tracking[receiver]
                    )
                    shifted_tracking = (
                        observed_tracking[receiver]
                        if fixed_side == "synthetic"
                        else synthetic_tracking[receiver]
                    )
                    tracking_correlation[receiver], _, _ = _weighted_zncc_lags(
                        shifted_tracking,
                        fixed_tracking[
                            target_left[receiver] : target_left[receiver] + right - left + 1
                        ],
                        local_weights,
                        target_left[receiver],
                        lags_samples if fixed_side == "synthetic" else -lags_samples,
                    )

    if use_envelope_coarse:
        (
            coarse_path,
            coarse_search_center_samples,
            coarse_tracking_fallback_mask,
            coarse_seed_receiver,
        ) = _seeded_envelope_path(
            envelope_correlation,
            ownership_allowed & valid[:, None],
            lags_samples,
            receiver_axis,
            source_value,
            int(round(seed_lag_range / float(dt))),
            envelope_tracking_epsilon_samples,
        )
        coarse_search_center_time = (
            coarse_search_center_samples * float(dt)
        )
        coarse_success = coarse_path >= 0
        coarse_lag_samples[coarse_success] = lags_samples[coarse_path[coarse_success]]
        coarse_lag_time[coarse_success] = (
            coarse_lag_samples[coarse_success] * float(dt)
        )
        receivers = np.flatnonzero(coarse_success)
        coarse_peak[receivers] = envelope_correlation[
            receivers, coarse_path[receivers]
        ]
        fine_allowed = ownership_allowed & coarse_success[:, None] & (
            np.abs(lags_samples[None, :] - coarse_lag_samples[:, None])
            <= fine_half_width_samples
        )
    else:
        fine_allowed = ownership_allowed & valid[:, None]

    lag_valid = fine_allowed & np.isfinite(waveform_correlation)
    correlation[lag_valid] = waveform_correlation[lag_valid]

    return CorrelationResult(
        lags_samples=lags_samples,
        lags_time=lags_time,
        correlation=correlation,
        waveform_correlation=waveform_correlation,
        tracking_correlation=tracking_correlation,
        valid=valid,
        window_energy_obs=energy_obs,
        window_energy_syn=energy_syn,
        reflector=reflector,
        shot=shot,
        max_lag_samples=max_lag_samples,
        dt=dt,
        lag_valid=lag_valid,
        coarse_lag_samples=coarse_lag_samples,
        coarse_lag_time=coarse_lag_time,
        coarse_correlation_peak=coarse_peak,
        ownership_lag_min=ownership_min,
        ownership_lag_max=ownership_max,
        coarse_search_center_time=coarse_search_center_time,
        coarse_tracking_fallback_mask=coarse_tracking_fallback_mask,
        coarse_seed_receiver=coarse_seed_receiver,
    )


def compute_zncc(
    observed: np.ndarray,
    synthetic: np.ndarray,
    window_result: WindowResult,
    reflector: int,
    shot: int,
    *,
    max_lag_time: float,
    dt: float,
    preprocess_hook: PreprocessHook | None = None,
    use_envelope_coarse: bool = True,
    envelope_fine_half_width_time: float = 0.04,
    envelope_tracking_epsilon_time: float = 0.040,
    seed_lag_range_time: float | None = None,
    receiver_x: np.ndarray | None = None,
    source_x: float | None = None,
    ownership_center_time: np.ndarray | None = None,
    fixed_side: str = "synthetic",
    tracking_enhancement_enabled: bool = True,
    tracking_agc_fraction: float = 0.25,
    tracking_agc_floor_ratio: float = 0.20,
    tracking_receiver_stack: bool = True,
) -> CorrelationResult:
    """Compute fixed-window ZNCC for one reflector and one shot.

    ``fixed_side='synthetic'`` retains the reference-stage behavior: the
    observed window shifts by ``+L``.  ``fixed_side='observed'`` shifts the
    synthetic window by ``-L``.  Both return ``shift_time = T_obs - T_syn``.
    When supplied, ``ownership_center_time`` must have the same
    ``[nreflector, nshot, nreceiver]`` shape as ``window_result`` and is used
    only to build neighboring-reflector ownership bounds.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise CorrelationError("dt must be finite and positive.")
    obs, syn = _prepare_data(
        observed, synthetic, window_result, preprocess_hook
    )
    ownership_centers = None
    if ownership_center_time is not None:
        ownership_centers = np.asarray(ownership_center_time, dtype=float)
        if ownership_centers.shape != window_result.center_time.shape:
            raise CorrelationError(
                "ownership_center_time must have the same shape as window centers."
            )
    max_lag_samples = _normalise_max_lag(max_lag_time, dt, window_result)
    observed_envelope = (
        np.abs(hilbert(obs[shot], axis=-1)) if use_envelope_coarse else None
    )
    synthetic_envelope = (
        np.abs(hilbert(syn[shot], axis=-1)) if use_envelope_coarse else None
    )
    return _compute_for_one_shot(
        obs,
        syn,
        window_result,
        reflector,
        shot,
        max_lag_samples,
        dt,
        use_envelope_coarse=use_envelope_coarse,
        envelope_fine_half_width_time=envelope_fine_half_width_time,
        envelope_tracking_epsilon_time=envelope_tracking_epsilon_time,
        seed_lag_range_time=seed_lag_range_time,
        receiver_x=receiver_x,
        source_x=source_x,
        observed_envelope=observed_envelope,
        synthetic_envelope=synthetic_envelope,
        ownership_center_time=(
            None if ownership_centers is None else ownership_centers[:, shot, :]
        ),
        fixed_side=fixed_side,
        tracking_enhancement_enabled=tracking_enhancement_enabled,
        tracking_agc_fraction=tracking_agc_fraction,
        tracking_agc_floor_ratio=tracking_agc_floor_ratio,
        tracking_receiver_stack=tracking_receiver_stack,
    )


def iter_zncc(
    observed: np.ndarray,
    synthetic: np.ndarray,
    window_result: WindowResult,
    *,
    max_lag_time: float,
    dt: float,
    preprocess_hook: PreprocessHook | None = None,
    reflectors: Sequence[int] | None = None,
    shots: Sequence[int] | None = None,
    use_envelope_coarse: bool = True,
    envelope_fine_half_width_time: float = 0.04,
    envelope_tracking_epsilon_time: float = 0.040,
    seed_lag_range_time: float | None = None,
    receiver_x: np.ndarray | None = None,
    source_x: float | None = None,
    ownership_center_time: np.ndarray | None = None,
    fixed_side: str = "synthetic",
    tracking_enhancement_enabled: bool = True,
    tracking_agc_fraction: float = 0.25,
    tracking_agc_floor_ratio: float = 0.20,
    tracking_receiver_stack: bool = True,
) -> Iterator[tuple[int, int, CorrelationResult]]:
    """Yield one ``CorrelationResult`` at a time for bounded memory use."""

    if not np.isfinite(dt) or dt <= 0:
        raise CorrelationError("dt must be finite and positive.")
    obs, syn = _prepare_data(
        observed, synthetic, window_result, preprocess_hook
    )
    ownership_centers = None
    if ownership_center_time is not None:
        ownership_centers = np.asarray(ownership_center_time, dtype=float)
        if ownership_centers.shape != window_result.center_time.shape:
            raise CorrelationError(
                "ownership_center_time must have the same shape as window centers."
            )
    max_lag_samples = _normalise_max_lag(max_lag_time, dt, window_result)
    nref, ns, _ = window_result.shape
    reflector_indices = range(nref) if reflectors is None else reflectors
    shot_indices = range(ns) if shots is None else shots

    # Process one shot at a time so its observed/synthetic envelopes are
    # computed once and reused for every reflector.  Only one shot's envelope
    # pair is resident, avoiding a full-survey envelope allocation.
    for shot in shot_indices:
        shot_observed_envelope = (
            np.abs(hilbert(obs[int(shot)], axis=-1))
            if use_envelope_coarse
            else None
        )
        shot_synthetic_envelope = (
            np.abs(hilbert(syn[int(shot)], axis=-1))
            if use_envelope_coarse
            else None
        )
        for reflector in reflector_indices:
            yield reflector, shot, _compute_for_one_shot(
                obs,
                syn,
                window_result,
                int(reflector),
                int(shot),
                max_lag_samples,
                dt,
                use_envelope_coarse=use_envelope_coarse,
                envelope_fine_half_width_time=envelope_fine_half_width_time,
                envelope_tracking_epsilon_time=envelope_tracking_epsilon_time,
                seed_lag_range_time=seed_lag_range_time,
                receiver_x=receiver_x,
                source_x=source_x,
                observed_envelope=shot_observed_envelope,
                synthetic_envelope=shot_synthetic_envelope,
                ownership_center_time=(
                    None
                    if ownership_centers is None
                    else ownership_centers[:, int(shot), :]
                ),
                fixed_side=fixed_side,
                tracking_enhancement_enabled=tracking_enhancement_enabled,
                tracking_agc_fraction=tracking_agc_fraction,
                tracking_agc_floor_ratio=tracking_agc_floor_ratio,
                tracking_receiver_stack=tracking_receiver_stack,
            )


def compute_fixed_window_zncc(*args, **kwargs) -> CorrelationResult:
    """Alias for :func:`compute_zncc`."""

    return compute_zncc(*args, **kwargs)
