"""Observed reflection-center bootstrap from Eikonal moveout and observed data.

This module implements the one-time ASCRT-style construction of
``Tobs[nref, nshot, nreceiver]``.  It deliberately has no dependency on a
reference synthetic gather.

For each reflector/shot pair it

1. chooses the zero/nearest-offset receiver as the control receiver,
2. removes only the *relative* Eikonal moveout from the whole observed gather,
3. uses the supplied same-x time only as a theoretical reflector anchor,
4. discovers a pilot-validated positive observed peak and fixed anchor segment,
5. extends that anchor with the sparse DP and restores the removed moveout.

The supplied absolute time is a search prior, not a forced observed pick.
All Eikonal times are used only through their relative moveout.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks, hilbert

from ..window import WindowResult, build_window_result


class TobsBootstrapError(ValueError):
    """Raised when ASCRT-style observed-center bootstrap inputs are invalid."""


def build_bootstrap_fixed_mask(
    tobs_fixed: np.ndarray,
    *,
    dt: float,
    t0: float,
    nt: int,
    half_window_time,
    window_type: str,
    tukey_alpha: float,
) -> tuple[np.ndarray, WindowResult]:
    """Build an independent receiver-level mask for final bootstrap centers."""

    tobs = np.asarray(tobs_fixed, dtype=float)
    target_windows = build_window_result(
        tobs,
        dt=dt,
        t0=t0,
        nt=nt,
        half_window_time=half_window_time,
        window_type=window_type,
        tukey_alpha=tukey_alpha,
        max_lag_samples=0,
    )
    return np.isfinite(tobs) & target_windows.valid, target_windows


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FlatEventTrackingResult:
    """One reflector/shot event tracked in the flattened observed gather."""

    pick_time: np.ndarray
    pick_sample: np.ndarray
    guide_pick_time: np.ndarray
    guide_pick_sample: np.ndarray
    legacy_pick_time: np.ndarray
    success_mask: np.ndarray
    trace_valid: np.ndarray
    score: np.ndarray
    guide_score: np.ndarray
    boundary_flag: np.ndarray
    seed_receiver: int
    seed_time: float
    seed_sample: int
    epsilon_samples: int

    def __post_init__(self) -> None:
        pick_time = np.asarray(self.pick_time, dtype=float)
        pick_sample = np.asarray(self.pick_sample, dtype=int)
        guide_pick_time = np.asarray(self.guide_pick_time, dtype=float)
        guide_pick_sample = np.asarray(self.guide_pick_sample, dtype=int)
        legacy_pick_time = np.asarray(self.legacy_pick_time, dtype=float)
        success = np.asarray(self.success_mask, dtype=bool)
        trace_valid = np.asarray(self.trace_valid, dtype=bool)
        score = np.asarray(self.score, dtype=float)
        guide_score = np.asarray(self.guide_score, dtype=float)
        boundary = np.asarray(self.boundary_flag, dtype=bool)
        if pick_time.ndim != 1:
            raise TobsBootstrapError("flat tracking arrays must be one-dimensional.")
        shape = pick_time.shape
        for name, value in {
            "pick_sample": pick_sample,
            "guide_pick_time": guide_pick_time,
            "guide_pick_sample": guide_pick_sample,
            "legacy_pick_time": legacy_pick_time,
            "success_mask": success,
            "trace_valid": trace_valid,
            "score": score,
            "guide_score": guide_score,
            "boundary_flag": boundary,
        }.items():
            if value.shape != shape:
                raise TobsBootstrapError(f"{name} must have shape {shape}.")
        if not 0 <= int(self.seed_receiver) < shape[0]:
            raise TobsBootstrapError("seed_receiver is outside the receiver axis.")
        if int(self.epsilon_samples) < 0:
            raise TobsBootstrapError("epsilon_samples must be non-negative.")
        object.__setattr__(self, "pick_time", _readonly(pick_time, float))
        object.__setattr__(self, "pick_sample", _readonly(pick_sample, int))
        object.__setattr__(self, "guide_pick_time", _readonly(guide_pick_time, float))
        object.__setattr__(self, "guide_pick_sample", _readonly(guide_pick_sample, int))
        object.__setattr__(self, "legacy_pick_time", _readonly(legacy_pick_time, float))
        object.__setattr__(self, "success_mask", _readonly(success, bool))
        object.__setattr__(self, "trace_valid", _readonly(trace_valid, bool))
        object.__setattr__(self, "score", _readonly(score, float))
        object.__setattr__(self, "guide_score", _readonly(guide_score, float))
        object.__setattr__(self, "boundary_flag", _readonly(boundary, bool))
        object.__setattr__(self, "seed_receiver", int(self.seed_receiver))
        object.__setattr__(self, "seed_time", float(self.seed_time))
        object.__setattr__(self, "seed_sample", int(self.seed_sample))
        object.__setattr__(self, "epsilon_samples", int(self.epsilon_samples))


@dataclass(frozen=True)
class ObservedCenterBootstrapResult:
    """One-time frozen observed centers and bootstrap diagnostics."""

    observed_center: np.ndarray
    tracking_success: np.ndarray
    trace_valid: np.ndarray
    tracking_score: np.ndarray
    guide_pick_time: np.ndarray
    guide_pick_sample: np.ndarray
    guide_score: np.ndarray
    legacy_pick_time: np.ndarray
    neighbor_correlation: np.ndarray
    residual_slope: np.ndarray
    prediction_error: np.ndarray
    candidate_count: np.ndarray
    candidate_count_before_ownership: np.ndarray
    predicted_center: np.ndarray
    ownership_upper_time: np.ndarray
    ownership_lower_time: np.ndarray
    ownership_upper_flat: np.ndarray
    ownership_lower_flat: np.ndarray
    ownership_order_violation: np.ndarray
    selected_amplitude: np.ndarray
    selected_envelope: np.ndarray
    selected_polarity: np.ndarray
    stop_reason_left: np.ndarray
    stop_reason_right: np.ndarray
    rescue_used_mask: np.ndarray
    skipped_valid_receiver_mask: np.ndarray
    ownership_escape_used_mask: np.ndarray
    stop_receiver_left: np.ndarray
    stop_receiver_right: np.ndarray
    boundary_flag: np.ndarray
    control_receiver: np.ndarray
    control_time: np.ndarray
    tracking_seed_time: np.ndarray
    seed_snap_delta_time: np.ndarray
    seed_snap_success: np.ndarray
    seed_snap_amplitude: np.ndarray
    seed_valid: np.ndarray
    seed_search_mode: np.ndarray
    seed_stack_evidence: np.ndarray
    seed_positive_support_fraction: np.ndarray
    seed_hypothesis_count: np.ndarray
    seed_selected_hypothesis: np.ndarray
    pilot_coverage: np.ndarray
    pilot_median_zncc: np.ndarray
    pilot_median_abs_prediction_error: np.ndarray
    pilot_median_abs_slope_ms_per_100m: np.ndarray
    pilot_phase_switch_count: np.ndarray
    pilot_left_support: np.ndarray
    pilot_right_support: np.ndarray
    anchor_left_count: np.ndarray
    anchor_right_count: np.ndarray
    anchor_receiver_mask: np.ndarray
    pilot_pick_time: np.ndarray
    tracking_usable_receiver: np.ndarray
    counts: Mapping[str, object]
    quality_available_receiver: np.ndarray | None = None
    unrecoverable_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.observed_center, dtype=float)
        success = np.asarray(self.tracking_success, dtype=bool)
        trace_valid = np.asarray(self.trace_valid, dtype=bool)
        score = np.asarray(self.tracking_score, dtype=float)
        guide_pick_time = np.asarray(self.guide_pick_time, dtype=float)
        guide_pick_sample = np.asarray(self.guide_pick_sample, dtype=int)
        guide_score = np.asarray(self.guide_score, dtype=float)
        legacy_pick_time = np.asarray(self.legacy_pick_time, dtype=float)
        neighbor_correlation = np.asarray(self.neighbor_correlation, dtype=float)
        residual_slope = np.asarray(self.residual_slope, dtype=float)
        prediction_error = np.asarray(self.prediction_error, dtype=float)
        candidate_count = np.asarray(self.candidate_count, dtype=int)
        candidate_count_before = np.asarray(self.candidate_count_before_ownership, dtype=int)
        selected_amplitude = np.asarray(self.selected_amplitude, dtype=float)
        selected_envelope = np.asarray(self.selected_envelope, dtype=float)
        selected_polarity = np.asarray(self.selected_polarity, dtype=int)
        stop_reason_left = np.asarray(self.stop_reason_left, dtype=str)
        stop_reason_right = np.asarray(self.stop_reason_right, dtype=str)
        boundary = np.asarray(self.boundary_flag, dtype=bool)
        control_receiver = np.asarray(self.control_receiver, dtype=int)
        control_time = np.asarray(self.control_time, dtype=float)
        tracking_seed_time = np.asarray(self.tracking_seed_time, dtype=float)
        seed_snap_delta_time = np.asarray(self.seed_snap_delta_time, dtype=float)
        seed_snap_success = np.asarray(self.seed_snap_success, dtype=bool)
        seed_snap_amplitude = np.asarray(self.seed_snap_amplitude, dtype=float)
        if center.ndim != 3:
            raise TobsBootstrapError(
                "observed_center must have shape [nref, nshot, nreceiver]."
            )
        shape = center.shape
        for name, value in {
            "tracking_success": success,
            "trace_valid": trace_valid,
            "tracking_score": score,
            "guide_pick_time": guide_pick_time,
            "guide_pick_sample": guide_pick_sample,
            "guide_score": guide_score,
            "legacy_pick_time": legacy_pick_time,
            "neighbor_correlation": neighbor_correlation,
            "residual_slope": residual_slope,
            "prediction_error": prediction_error,
            "candidate_count": candidate_count,
            "candidate_count_before_ownership": candidate_count_before,
            "predicted_center": np.asarray(self.predicted_center),
            "ownership_upper_time": np.asarray(self.ownership_upper_time),
            "ownership_lower_time": np.asarray(self.ownership_lower_time),
            "ownership_upper_flat": np.asarray(self.ownership_upper_flat),
            "ownership_lower_flat": np.asarray(self.ownership_lower_flat),
            "ownership_order_violation": np.asarray(self.ownership_order_violation),
            "selected_amplitude": selected_amplitude,
            "selected_envelope": selected_envelope,
            "selected_polarity": selected_polarity,
            "boundary_flag": boundary,
            "rescue_used_mask": np.asarray(self.rescue_used_mask),
            "skipped_valid_receiver_mask": np.asarray(self.skipped_valid_receiver_mask),
            "ownership_escape_used_mask": np.asarray(self.ownership_escape_used_mask),
            "tracking_usable_receiver": np.asarray(self.tracking_usable_receiver),
        }.items():
            if value.shape != shape:
                raise TobsBootstrapError(f"{name} must have shape {shape}.")
        if control_receiver.shape != (shape[1],):
            raise TobsBootstrapError("control_receiver must have shape [nshot].")
        if control_time.shape != (shape[0], shape[1]):
            raise TobsBootstrapError("control_time must have shape [nref, nshot].")
        for name, value in {"tracking_seed_time": tracking_seed_time, "seed_snap_delta_time": seed_snap_delta_time,
                            "seed_snap_success": seed_snap_success, "seed_snap_amplitude": seed_snap_amplitude}.items():
            if value.shape != control_time.shape:
                raise TobsBootstrapError(f"{name} must have shape {control_time.shape}.")
        object.__setattr__(self, "observed_center", _readonly(center, float))
        object.__setattr__(self, "tracking_success", _readonly(success, bool))
        object.__setattr__(self, "trace_valid", _readonly(trace_valid, bool))
        object.__setattr__(self, "tracking_score", _readonly(score, float))
        object.__setattr__(self, "guide_pick_time", _readonly(guide_pick_time, float))
        object.__setattr__(self, "guide_pick_sample", _readonly(guide_pick_sample, int))
        object.__setattr__(self, "guide_score", _readonly(guide_score, float))
        object.__setattr__(self, "legacy_pick_time", _readonly(legacy_pick_time, float))
        object.__setattr__(self, "neighbor_correlation", _readonly(neighbor_correlation, float))
        object.__setattr__(self, "residual_slope", _readonly(residual_slope, float))
        object.__setattr__(self, "prediction_error", _readonly(prediction_error, float))
        object.__setattr__(self, "candidate_count", _readonly(candidate_count, int))
        object.__setattr__(self, "candidate_count_before_ownership", _readonly(candidate_count_before, int))
        for name in ("predicted_center", "ownership_upper_time", "ownership_lower_time", "ownership_upper_flat", "ownership_lower_flat"):
            object.__setattr__(self, name, _readonly(np.asarray(getattr(self, name)), float))
        object.__setattr__(self, "ownership_order_violation", _readonly(np.asarray(self.ownership_order_violation), bool))
        object.__setattr__(self, "selected_amplitude", _readonly(selected_amplitude, float))
        object.__setattr__(self, "selected_envelope", _readonly(selected_envelope, float))
        object.__setattr__(self, "selected_polarity", _readonly(selected_polarity, int))
        object.__setattr__(self, "stop_reason_left", _readonly(stop_reason_left, str))
        object.__setattr__(self, "stop_reason_right", _readonly(stop_reason_right, str))
        object.__setattr__(self, "rescue_used_mask", _readonly(np.asarray(self.rescue_used_mask), bool))
        object.__setattr__(self, "skipped_valid_receiver_mask", _readonly(np.asarray(self.skipped_valid_receiver_mask), bool))
        object.__setattr__(self, "ownership_escape_used_mask", _readonly(np.asarray(self.ownership_escape_used_mask), bool))
        for name in ("stop_receiver_left", "stop_receiver_right"):
            value = np.asarray(getattr(self, name), dtype=int)
            if value.shape != control_time.shape:
                raise TobsBootstrapError(f"{name} must have shape {control_time.shape}.")
            object.__setattr__(self, name, _readonly(value, int))
        object.__setattr__(self, "boundary_flag", _readonly(boundary, bool))
        object.__setattr__(self, "control_receiver", _readonly(control_receiver, int))
        object.__setattr__(self, "control_time", _readonly(control_time, float))
        object.__setattr__(self, "tracking_seed_time", _readonly(tracking_seed_time, float))
        object.__setattr__(self, "seed_snap_delta_time", _readonly(seed_snap_delta_time, float))
        object.__setattr__(self, "seed_snap_success", _readonly(seed_snap_success, bool))
        object.__setattr__(self, "seed_snap_amplitude", _readonly(seed_snap_amplitude, float))
        for name, dtype in (("seed_valid", bool), ("seed_search_mode", str),
                            ("seed_stack_evidence", float), ("seed_hypothesis_count", int),
                            ("seed_positive_support_fraction", float),
                            ("seed_selected_hypothesis", int), ("pilot_coverage", float),
                            ("pilot_median_zncc", float), ("pilot_median_abs_prediction_error", float),
                            ("pilot_median_abs_slope_ms_per_100m", float),
                            ("pilot_phase_switch_count", int),
                            ("anchor_left_count", int), ("anchor_right_count", int),
                            ("pilot_left_support", int), ("pilot_right_support", int)):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != control_time.shape:
                raise TobsBootstrapError(f"{name} must have shape {control_time.shape}.")
            object.__setattr__(self, name, _readonly(value, dtype))
        for name, dtype in (("anchor_receiver_mask", bool), ("pilot_pick_time", float)):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape:
                raise TobsBootstrapError(f"{name} must have shape {shape}.")
            object.__setattr__(self, name, _readonly(value, dtype))
        object.__setattr__(self, "tracking_usable_receiver", _readonly(
            np.asarray(self.tracking_usable_receiver), bool
        ))
        quality_available = (
            np.asarray(self.tracking_usable_receiver, dtype=bool)
            if self.quality_available_receiver is None
            else np.asarray(self.quality_available_receiver, dtype=bool)
        )
        if quality_available.shape != shape:
            raise TobsBootstrapError("quality_available_receiver must have shape " + str(shape) + ".")
        object.__setattr__(self, "quality_available_receiver", _readonly(quality_available, bool))
        unrecoverable = (
            np.zeros(control_time.shape, dtype=bool)
            if self.unrecoverable_mask is None
            else np.asarray(self.unrecoverable_mask, dtype=bool)
        )
        if unrecoverable.shape != control_time.shape:
            raise TobsBootstrapError("unrecoverable_mask must match control_time.")
        object.__setattr__(self, "unrecoverable_mask", _readonly(unrecoverable, bool))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))


def nearest_offset_receiver(
    source_coordinate: np.ndarray,
    receiver_coordinates: np.ndarray,
) -> int:
    """Return the receiver with the smallest source-receiver Euclidean offset."""

    source = np.asarray(source_coordinate, dtype=float)
    receivers = np.asarray(receiver_coordinates, dtype=float)
    if source.shape != (2,) or receivers.ndim != 2 or receivers.shape[1] != 2:
        raise TobsBootstrapError(
            "source_coordinate must be [2] and receiver_coordinates must be [nr, 2]."
        )
    if not np.isfinite(source).all() or not np.isfinite(receivers).all():
        raise TobsBootstrapError("source/receiver coordinates must be finite.")
    distance2 = np.sum((receivers - source[None, :]) ** 2, axis=1)
    return int(np.argmin(distance2))


def flatten_observed_gather(
    observed_shot: np.ndarray,
    theoretical_traveltime: np.ndarray,
    *,
    seed_receiver: int,
    dt: float,
    t0: float = 0.0,
    return_support: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove Eikonal relative moveout from one complete observed shot gather.

    The returned ``moveout`` is

    ``Tref(receiver) - Tref(seed_receiver)``.

    No absolute time search window and no maximum lag are used here.
    ``valid_receiver`` only describes rows whose theoretical time and observed
    trace are usable for tracking.
    """

    observed = np.asarray(observed_shot, dtype=float)
    tref = np.asarray(theoretical_traveltime, dtype=float)
    if observed.ndim != 2:
        raise TobsBootstrapError("observed_shot must have shape [nreceiver, ntime].")
    nr, nt = observed.shape
    if tref.shape != (nr,):
        raise TobsBootstrapError("theoretical_traveltime must have shape [nreceiver].")
    if not 0 <= int(seed_receiver) < nr:
        raise TobsBootstrapError("seed_receiver is outside the receiver axis.")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise TobsBootstrapError("dt must be positive and t0 must be finite.")
    if not np.isfinite(tref[int(seed_receiver)]):
        raise TobsBootstrapError("the control-receiver theoretical time is not finite.")

    trace_finite = np.isfinite(observed).all(axis=1)
    trace_energy = np.sqrt(
        np.mean(np.where(np.isfinite(observed), observed, 0.0) ** 2, axis=1)
    )
    energy_floor = max(np.finfo(float).eps, 1e-12 * float(np.max(trace_energy, initial=0.0)))
    valid_receiver = trace_finite & (trace_energy > energy_floor) & np.isfinite(tref)

    moveout = tref - float(tref[int(seed_receiver)])
    time = float(t0) + np.arange(nt, dtype=float) * float(dt)
    flat = np.zeros_like(observed, dtype=float)
    for receiver in np.flatnonzero(valid_receiver):
        query_time = time + moveout[receiver]
        flat[receiver] = np.interp(
            query_time,
            time,
            observed[receiver],
            left=0.0,
            right=0.0,
        )
    if return_support:
        support = (
            (time[None, :] + moveout[:, None] >= time[0])
            & (time[None, :] + moveout[:, None] <= time[-1])
            & valid_receiver[:, None]
        )
        return flat, moveout, valid_receiver, support
    return flat, moveout, valid_receiver


def _normalised_envelope(flat_gather: np.ndarray, valid_receiver: np.ndarray) -> np.ndarray:
    """Return per-trace RMS-normalized analytic envelopes for DP evidence."""

    flat = np.asarray(flat_gather, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    if flat.ndim != 2 or valid.shape != (flat.shape[0],):
        raise TobsBootstrapError("flat_gather/valid_receiver shapes are inconsistent.")
    envelope = np.zeros_like(flat, dtype=float)
    if not np.any(valid):
        return envelope
    envelope[valid] = np.abs(hilbert(flat[valid], axis=-1))
    scale = np.sqrt(np.mean(envelope * envelope, axis=1))
    good = valid & np.isfinite(scale) & (scale > np.finfo(float).eps)
    envelope[good] /= scale[good, None]
    envelope[~good] = 0.0
    return envelope


def _normalised_absolute_waveform(
    flat_gather: np.ndarray, valid_receiver: np.ndarray
) -> np.ndarray:
    """Return per-trace RMS-normalized absolute waveform evidence."""

    evidence = np.abs(np.asarray(flat_gather, dtype=float))
    valid = np.asarray(valid_receiver, dtype=bool)
    scale = np.sqrt(np.mean(evidence * evidence, axis=1))
    good = valid & np.isfinite(scale) & (scale > np.finfo(float).eps)
    evidence[good] /= scale[good, None]
    evidence[~good] = 0.0
    return evidence


def _track_direction(
    evidence: np.ndarray,
    valid_receiver: np.ndarray,
    receiver_order: np.ndarray,
    *,
    seed_sample: int,
    epsilon_samples: int,
    slope_penalty: float,
    state_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track one receiver direction from a fixed seed with full-time DP.

    The DP has no absolute-time corridor.  Only the adjacent-receiver slope
    constraint ``|j_i-j_(i-1)| <= epsilon_samples`` is imposed. Among
    full-coverage paths, evidence minus ``slope_penalty * |j-k|`` is maximized.
    """

    order = np.asarray(receiver_order, dtype=int)
    ntime = evidence.shape[1]
    pick = np.full(evidence.shape[0], -1, dtype=int)
    score_at_pick = np.full(evidence.shape[0], np.nan, dtype=float)
    if order.size == 0:
        return pick, score_at_pick
    if not bool(valid_receiver[order[0]]):
        return pick, score_at_pick
    allowed = (
        np.ones(evidence.shape, dtype=bool)
        if state_valid is None
        else np.asarray(state_valid, dtype=bool)
    )
    if allowed.shape != evidence.shape or not allowed[order[0], seed_sample]:
        return pick, score_at_pick

    # A direction stops at the first unusable receiver.  It never jumps across
    # muted/missing rows and then restarts on the far side.
    usable_order = [int(order[0])]
    for receiver in order[1:]:
        receiver = int(receiver)
        if not bool(valid_receiver[receiver]):
            break
        usable_order.append(receiver)
    order = np.asarray(usable_order, dtype=int)

    scores = np.full((order.size, ntime), -np.inf, dtype=float)
    scores[0, seed_sample] = evidence[order[0], seed_sample]
    sample = np.arange(ntime, dtype=float)
    size = int(epsilon_samples) + 1
    for local in range(1, order.size):
        previous = scores[local - 1]
        best_from_left = maximum_filter1d(
            previous + float(slope_penalty) * sample,
            size=size, origin=(size - 1) // 2, mode="constant", cval=-np.inf,
        ) - float(slope_penalty) * sample
        best_from_right = maximum_filter1d(
            previous - float(slope_penalty) * sample,
            size=size, origin=-(size // 2), mode="constant", cval=-np.inf,
        ) + float(slope_penalty) * sample
        next_scores = np.maximum(best_from_left, best_from_right)
        receiver = int(order[local])
        usable = allowed[receiver] & np.isfinite(next_scores)
        scores[local] = np.where(usable, evidence[receiver] + next_scores, -np.inf)

    state = int(np.argmax(scores[-1]))
    if not np.isfinite(scores[-1, state]):
        return pick, score_at_pick

    for local in range(order.size - 1, -1, -1):
        receiver = int(order[local])
        pick[receiver] = state
        score_at_pick[receiver] = evidence[receiver, state]
        if local == 0:
            break
        lo = max(0, state - int(epsilon_samples))
        hi = min(ntime, state + int(epsilon_samples) + 1)
        previous = scores[local - 1, lo:hi]
        candidate = previous - float(slope_penalty) * np.abs(
            np.arange(lo, hi) - state
        )
        if not np.isfinite(candidate).any():
            break
        state = lo + int(np.argmax(candidate))

    return pick, score_at_pick


def track_flattened_event(
    flat_gather: np.ndarray,
    *,
    valid_receiver: np.ndarray,
    seed_receiver: int,
    seed_time: float,
    dt: float,
    t0: float = 0.0,
    epsilon_time: float = 0.020,
    slope_penalty: float = 0.10,
    refine_radius_samples: int = 3,
    boundary_margin_samples: int = 0,
) -> FlatEventTrackingResult:
    """Track a flattened observed event from an Eikonal control point."""

    flat = np.asarray(flat_gather, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    if flat.ndim != 2 or valid.shape != (flat.shape[0],):
        raise TobsBootstrapError("flat_gather must be [nr, nt] and valid_receiver [nr].")
    nr, nt = flat.shape
    if not 0 <= int(seed_receiver) < nr:
        raise TobsBootstrapError("seed_receiver is outside the receiver axis.")
    if not np.isfinite(seed_time) or not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise TobsBootstrapError("seed_time/t0 must be finite and dt must be positive.")
    if not np.isfinite(epsilon_time) or epsilon_time < 0:
        raise TobsBootstrapError("epsilon_time must be finite and non-negative.")
    if not np.isfinite(slope_penalty) or slope_penalty < 0:
        raise TobsBootstrapError("slope_penalty must be finite and non-negative.")
    if (
        isinstance(refine_radius_samples, bool)
        or int(refine_radius_samples) != refine_radius_samples
        or refine_radius_samples < 0
    ):
        raise TobsBootstrapError("refine_radius_samples must be non-negative.")
    if (
        isinstance(boundary_margin_samples, bool)
        or int(boundary_margin_samples) != boundary_margin_samples
        or boundary_margin_samples < 0
    ):
        raise TobsBootstrapError("boundary_margin_samples must be non-negative.")

    epsilon_samples = int(round(float(epsilon_time) / float(dt)))
    seed_float = (float(seed_time) - float(t0)) / float(dt)
    seed_sample = int(np.rint(seed_float))
    if not 0 <= seed_sample < nt or not bool(valid[int(seed_receiver)]):
        empty_time = np.full(nr, np.nan, dtype=float)
        return FlatEventTrackingResult(
            pick_time=empty_time,
            pick_sample=np.full(nr, -1, dtype=int),
            guide_pick_time=empty_time,
            guide_pick_sample=np.full(nr, -1, dtype=int),
            legacy_pick_time=empty_time,
            success_mask=np.zeros(nr, dtype=bool),
            trace_valid=valid,
            score=np.full(nr, np.nan, dtype=float),
            guide_score=np.full(nr, np.nan, dtype=float),
            boundary_flag=np.zeros(nr, dtype=bool),
            seed_receiver=int(seed_receiver),
            seed_time=float(seed_time),
            seed_sample=seed_sample,
            epsilon_samples=epsilon_samples,
        )

    guide_evidence = _normalised_envelope(flat, valid)
    left_order = np.arange(int(seed_receiver), -1, -1, dtype=int)
    right_order = np.arange(int(seed_receiver), nr, dtype=int)
    legacy_left, _ = _track_direction(
        guide_evidence, valid, left_order, seed_sample=seed_sample,
        epsilon_samples=epsilon_samples, slope_penalty=0.0,
    )
    legacy_right, _ = _track_direction(
        guide_evidence, valid, right_order, seed_sample=seed_sample,
        epsilon_samples=epsilon_samples, slope_penalty=0.0,
    )
    legacy_pick = np.where(legacy_right >= 0, legacy_right, legacy_left)
    legacy_pick_time = np.where(
        legacy_pick >= 0, float(t0) + legacy_pick * float(dt), np.nan
    )
    if legacy_pick[int(seed_receiver)] >= 0:
        legacy_pick_time[int(seed_receiver)] = float(seed_time)
    left_pick, left_score = _track_direction(
        guide_evidence,
        valid,
        left_order,
        seed_sample=seed_sample,
        epsilon_samples=epsilon_samples,
        slope_penalty=slope_penalty,
    )
    right_pick, right_score = _track_direction(
        guide_evidence,
        valid,
        right_order,
        seed_sample=seed_sample,
        epsilon_samples=epsilon_samples,
        slope_penalty=slope_penalty,
    )

    guide_pick = np.where(right_pick >= 0, right_pick, left_pick)
    guide_score = np.where(np.isfinite(right_score), right_score, left_score)
    corridor = np.zeros((nr, nt), dtype=bool)
    radius = int(refine_radius_samples)
    for receiver in np.flatnonzero(guide_pick >= 0):
        lo = max(0, int(guide_pick[receiver]) - radius)
        hi = min(nt, int(guide_pick[receiver]) + radius + 1)
        corridor[receiver, lo:hi] = True
    fine_evidence = _normalised_absolute_waveform(flat, valid)
    left_pick, left_score = _track_direction(
        fine_evidence, valid, left_order, seed_sample=seed_sample,
        epsilon_samples=epsilon_samples, slope_penalty=slope_penalty,
        state_valid=corridor,
    )
    right_pick, right_score = _track_direction(
        fine_evidence, valid, right_order, seed_sample=seed_sample,
        epsilon_samples=epsilon_samples, slope_penalty=slope_penalty,
        state_valid=corridor,
    )
    pick = np.where(right_pick >= 0, right_pick, left_pick)
    score = np.where(np.isfinite(right_score), right_score, left_score)
    # Both directions share the same forced seed.  Keep that seed exactly at
    # the theoretical control time rather than quantizing it to a sample.
    success = pick >= 0
    pick_time = np.full(nr, np.nan, dtype=float)
    pick_sample_float = pick.astype(float)
    for receiver in np.flatnonzero(success):
        sample = int(pick[receiver])
        if 0 < sample < nt - 1:
            left, center, right = fine_evidence[receiver, sample - 1 : sample + 2]
            denominator = left - 2.0 * center + right
            if np.isfinite(denominator) and denominator != 0.0:
                offset = 0.5 * (left - right) / denominator
                pick_sample_float[receiver] += float(np.clip(offset, -0.5, 0.5))
    pick_time[success] = float(t0) + pick_sample_float[success] * float(dt)
    if success[int(seed_receiver)]:
        pick_time[int(seed_receiver)] = float(seed_time)
    guide_pick_time = np.where(
        guide_pick >= 0, float(t0) + guide_pick * float(dt), np.nan
    )
    if guide_pick[int(seed_receiver)] >= 0:
        guide_pick_time[int(seed_receiver)] = float(seed_time)

    margin = int(boundary_margin_samples)
    boundary = success & ((pick <= margin) | (pick >= nt - 1 - margin))
    return FlatEventTrackingResult(
        pick_time=pick_time,
        pick_sample=pick,
        guide_pick_time=guide_pick_time,
        guide_pick_sample=guide_pick,
        legacy_pick_time=legacy_pick_time,
        success_mask=success,
        trace_valid=valid,
        score=score,
        guide_score=guide_score,
        boundary_flag=boundary,
        seed_receiver=int(seed_receiver),
        seed_time=float(seed_time),
        seed_sample=seed_sample,
        epsilon_samples=epsilon_samples,
    )


def snap_control_to_observed_peak(trace, *, control_time, dt, t0=0.0,
                                  state_valid_row=None, half_width_time=0.030):
    """Snap a theoretical control to the strongest nearby valid phase lobe."""
    values = np.asarray(trace, dtype=float)
    peaks, _ = find_peaks(values)
    peaks = peaks[values[peaks] > 0.0]
    time = float(t0) + peaks * float(dt)
    keep = np.abs(time - float(control_time)) <= float(half_width_time)
    if state_valid_row is not None:
        valid = np.asarray(state_valid_row, dtype=bool)
        if valid.shape != values.shape:
            raise TobsBootstrapError("state_valid_row must match the seed trace.")
        keep &= valid[peaks]
    peaks = peaks[keep]
    if peaks.size == 0:
        sample = int(np.rint((float(control_time) - float(t0)) / float(dt)))
        amplitude = values[sample] if 0 <= sample < values.size else np.nan
        return float(control_time), sample, False, 0.0, float(amplitude)
    sample = int(peaks[np.argmax(values[peaks])])
    snapped = float(t0) + sample * float(dt)
    return snapped, sample, True, snapped - float(control_time), float(values[sample])


def _ownership_corridors(predicted, control, overlap, edge_fraction):
    """Return original-time ownership bounds without changing reflector IDs."""
    predicted = np.asarray(predicted, dtype=float)
    nref, nr = predicted.shape
    upper = np.full_like(predicted, np.nan)
    lower = np.full_like(predicted, np.nan)
    violation = np.zeros_like(predicted, dtype=bool)
    for receiver in range(nr):
        column = predicted[:, receiver]
        bad = ~np.isfinite(column)
        if nref > 1:
            bad |= np.r_[np.diff(column) <= 0, False] | np.r_[False, np.diff(column) <= 0]
        violation[:, receiver] = bad
        for reflector in range(nref):
            if bad[reflector]:
                gaps = []
                if reflector:
                    gaps.append(control[reflector] - control[reflector - 1])
                if reflector + 1 < nref:
                    gaps.append(control[reflector + 1] - control[reflector])
                positive = [gap for gap in gaps if np.isfinite(gap) and gap > 0]
                half = 0.5 * min(positive) if positive else np.inf
                upper[reflector, receiver] = column[reflector] - half
                lower[reflector, receiver] = column[reflector] + half
                continue
            if reflector == 0:
                gap = column[1] - column[0] if nref > 1 else np.inf
                upper[reflector, receiver] = column[0] - edge_fraction * gap
            else:
                gap = column[reflector] - column[reflector - 1]
                upper[reflector, receiver] = column[reflector] - (0.5 + overlap) * gap
            if reflector == nref - 1:
                gap = column[-1] - column[-2] if nref > 1 else np.inf
                lower[reflector, receiver] = column[-1] + edge_fraction * gap
            else:
                gap = column[reflector + 1] - column[reflector]
                lower[reflector, receiver] = column[reflector] + (0.5 + overlap) * gap
    return upper, lower, violation


def _ownership_escape_bounds(predicted, upper, lower, fraction, maximum):
    """Expand base bounds using only adjacent-reflector theoretical gaps."""
    predicted = np.asarray(predicted, dtype=float)
    nref = predicted.shape[0]
    escape_upper = np.array(upper, copy=True)
    escape_lower = np.array(lower, copy=True)
    if nref < 2:
        return escape_upper, escape_lower
    for reflector in range(nref):
        upper_gap = np.abs(predicted[reflector] - predicted[max(0, reflector - 1)])
        lower_gap = np.abs(predicted[min(nref - 1, reflector + 1)] - predicted[reflector])
        if reflector == 0:
            upper_gap = lower_gap
        if reflector == nref - 1:
            lower_gap = upper_gap
        escape_upper[reflector] -= np.minimum(float(fraction) * upper_gap, float(maximum))
        escape_lower[reflector] += np.minimum(float(fraction) * lower_gap, float(maximum))
    return escape_upper, escape_lower


def build_observed_centers_from_eikonal(
    observed_data: np.ndarray,
    theoretical_traveltime: np.ndarray,
    source_coordinates: np.ndarray,
    receiver_coordinates: np.ndarray,
    *,
    dt: float,
    t0: float = 0.0,
    control_time: np.ndarray,
    epsilon_time: float = 0.020,
    slope_penalty: float = 0.10,
    refine_radius_samples: int = 3,
    tracking_method: str = "sparse_event_dp",
    sparse_tracking_options: Mapping[str, object] | None = None,
    ownership_enabled: bool = True,
    ownership_overlap_fraction: float = 0.10,
    ownership_edge_fraction: float = 0.60,
    seed_snap_enabled: bool = True,
    seed_snap_half_width_time: float = 0.030,
    compute_legacy_dense_diagnostic: bool = False,
    legacy_dense_diagnostic_shots: tuple[int, ...] = (),
    seed_search_options: Mapping[str, object] | None = None,
    boundary_margin_samples: int = 0,
    quality_audit_config=None,
) -> ObservedCenterBootstrapResult:
    """Construct frozen ``Tobs`` directly from observed data and Eikonal times.

    ``theoretical_traveltime`` must be the Eikonal reflection time calculated
    with the *same migration velocity and reflector pair* used to create the
    initial migrated reflector. ``control_time`` is an already shifted
    ``[nref, nshot]`` theoretical search anchor; the observed positive peak is
    discovered from the flattened data and may differ from it. No reference
    synthetic waveform is accepted by this API.
    """

    observed = np.asarray(observed_data, dtype=float)
    tref = np.asarray(theoretical_traveltime, dtype=float)
    sources = np.asarray(source_coordinates, dtype=float)
    receivers = np.asarray(receiver_coordinates, dtype=float)
    if observed.ndim != 3:
        raise TobsBootstrapError("observed_data must have shape [nshot, nreceiver, ntime].")
    if tref.ndim != 3:
        raise TobsBootstrapError(
            "theoretical_traveltime must have shape [nref, nshot, nreceiver]."
        )
    nref, nshot, nreceiver = tref.shape
    if observed.shape[:2] != (nshot, nreceiver):
        raise TobsBootstrapError("observed_data geometry does not match theoretical_traveltime.")
    if sources.shape != (nshot, 2):
        raise TobsBootstrapError("source_coordinates must have shape [nshot, 2].")
    if receivers.shape != (nshot, nreceiver, 2):
        raise TobsBootstrapError(
            "receiver_coordinates must have shape [nshot, nreceiver, 2]."
        )
    if not np.isfinite(observed).all():
        raise TobsBootstrapError("observed_data must contain only finite samples.")
    if not np.isfinite(sources).all() or not np.isfinite(receivers).all():
        raise TobsBootstrapError("source/receiver coordinates must be finite.")
    supplied_control_time = np.asarray(control_time, dtype=float)
    if supplied_control_time.shape != (nref, nshot):
        raise TobsBootstrapError("control_time must have shape [nref, nshot].")

    center = np.full(tref.shape, np.nan, dtype=float)
    tracking_success = np.zeros(tref.shape, dtype=bool)
    trace_valid = np.zeros(tref.shape, dtype=bool)
    tracking_score = np.full(tref.shape, np.nan, dtype=float)
    guide_pick_time = np.full(tref.shape, np.nan, dtype=float)
    guide_pick_sample = np.full(tref.shape, -1, dtype=int)
    guide_score = np.full(tref.shape, np.nan, dtype=float)
    legacy_pick_time = np.full(tref.shape, np.nan, dtype=float)
    neighbor_correlation = np.full(tref.shape, np.nan)
    residual_slope = np.full(tref.shape, np.nan)
    prediction_error = np.full(tref.shape, np.nan)
    candidate_count = np.zeros(tref.shape, dtype=int)
    candidate_count_before = np.zeros(tref.shape, dtype=int)
    predicted_center = np.full(tref.shape, np.nan)
    ownership_upper_time = np.full(tref.shape, np.nan)
    ownership_lower_time = np.full(tref.shape, np.nan)
    ownership_upper_flat = np.full(tref.shape, np.nan)
    ownership_lower_flat = np.full(tref.shape, np.nan)
    ownership_order_violation = np.zeros(tref.shape, dtype=bool)
    selected_amplitude = np.full(tref.shape, np.nan)
    selected_envelope = np.full(tref.shape, np.nan)
    selected_polarity = np.zeros(tref.shape, dtype=int)
    stop_reason_left = np.full((nref, nshot), "NOT_RUN", dtype="<U128")
    stop_reason_right = np.full((nref, nshot), "NOT_RUN", dtype="<U128")
    sparse_totals = {"transitions": 0, "correlations": 0, "active_max": 0, "runtime": 0.0,
                     "rescue_attempts": 0, "rescue_successes": 0, "rescue_candidates": 0,
                     "rescue_transitions": 0,
                     "valid_gaps": 0, "escape_attempts": 0,
                     "escape_successes": 0, "escape_candidates": 0}
    rescue_used_mask = np.zeros(tref.shape, dtype=bool)
    skipped_valid_receiver_mask = np.zeros(tref.shape, dtype=bool)
    ownership_escape_used_mask = np.zeros(tref.shape, dtype=bool)
    stop_receiver_left = np.full((nref, nshot), -1, dtype=int)
    stop_receiver_right = np.full((nref, nshot), -1, dtype=int)
    boundary_flag = np.zeros(tref.shape, dtype=bool)
    control_receiver = np.empty(nshot, dtype=int)
    used_control_time = np.full((nref, nshot), np.nan, dtype=float)
    tracking_seed_time = np.full((nref, nshot), np.nan, dtype=float)
    seed_snap_delta_time = np.full((nref, nshot), np.nan, dtype=float)
    seed_snap_success = np.zeros((nref, nshot), dtype=bool)
    seed_snap_amplitude = np.full((nref, nshot), np.nan, dtype=float)
    seed_valid = np.zeros((nref, nshot), dtype=bool)
    seed_search_mode = np.full((nref, nshot), "FAILED", dtype="<U20")
    seed_stack_evidence = np.full((nref, nshot), np.nan)
    seed_positive_support_fraction = np.full((nref, nshot), np.nan)
    seed_hypothesis_count = np.zeros((nref, nshot), dtype=int)
    seed_selected_hypothesis = np.full((nref, nshot), -1, dtype=int)
    pilot_coverage = np.full((nref, nshot), np.nan)
    pilot_median_zncc = np.full((nref, nshot), np.nan)
    pilot_median_abs_prediction_error = np.full((nref, nshot), np.nan)
    pilot_median_abs_slope_ms_per_100m = np.full((nref, nshot), np.nan)
    pilot_phase_switch_count = np.zeros((nref, nshot), dtype=int)
    pilot_left_support = np.zeros((nref, nshot), dtype=int)
    pilot_right_support = np.zeros((nref, nshot), dtype=int)
    anchor_left_count = np.zeros((nref, nshot), dtype=int)
    anchor_right_count = np.zeros((nref, nshot), dtype=int)
    anchor_receiver_mask = np.zeros((nref, nshot, nreceiver), dtype=bool)
    pilot_pick_time = np.full((nref, nshot, nreceiver), np.nan)
    tracking_usable_receiver = np.zeros(tref.shape, dtype=bool)
    quality_available_receiver = np.zeros(tref.shape, dtype=bool)
    unrecoverable_mask = np.zeros((nref, nshot), dtype=bool)
    seed_repair_attempt_count = 0
    seed_repair_success_count = 0
    tracking_break_repair_entered = np.zeros((nref, nshot), dtype=bool)
    tracking_break_trusted_segments = np.full((nref, nshot), "", dtype="<U256")
    tracking_break_seed_receiver_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_seed_attempts = np.full((nref, nshot), "", dtype="<U128")
    tracking_break_full_candidate_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_normal_candidate_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_winner_seed_receiver = np.full((nref, nshot), -1, dtype=int)
    tracking_break_winner_seed_time = np.full((nref, nshot), np.nan)
    tracking_break_trusted_match_fraction = np.full((nref, nshot), np.nan)
    tracking_break_winner_rank = np.full((nref, nshot), "", dtype="<U256")
    tracking_break_repair_success = np.zeros((nref, nshot), dtype=bool)
    tracking_break_original_trusted_segments = np.full((nref, nshot), "", dtype="<U256")
    tracking_break_boundary_seed_receivers = np.full((nref, nshot), "", dtype="<U128")
    tracking_break_boundary_seed_times = np.full((nref, nshot), "", dtype="<U256")
    tracking_break_boundary_seed_energy = np.full((nref, nshot), "", dtype="<U256")
    tracking_break_boundary_candidate_diagnostics = np.full((nref, nshot), "", dtype="<U4096")
    tracking_break_trusted_coverage = np.full((nref, nshot), np.nan)
    tracking_break_selected_segment_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_candidate_segment_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_boundary_seed_candidate_count = np.zeros((nref, nshot), dtype=int)
    tracking_break_boundary_seed_pilots_accepted = np.zeros((nref, nshot), dtype=int)
    tracking_break_merged_raw_audit = np.full((nref, nshot), "", dtype="<U32")
    seed_search_runtime = 0.0
    pilot_transition_count = 0
    seed_local_high_confidence_count = 0
    dense_diagnostic_runtime = 0.0
    dense_diagnostic_calls = 0
    # Dense tracking is intentionally retired from the production bootstrap.
    # Legacy diagnostic arguments remain in the public signature only for config compatibility.

    for shot in range(nshot):
        shot_started = perf_counter()
        i0 = nearest_offset_receiver(sources[shot], receivers[shot])
        control_receiver[shot] = i0
        moveouts = tref[:, shot] - tref[:, shot, i0, None]
        predicted_center[:, shot] = supplied_control_time[:, shot, None] + moveouts
        upper, lower, violation = _ownership_corridors(
            predicted_center[:, shot], supplied_control_time[:, shot],
            float(ownership_overlap_fraction), float(ownership_edge_fraction),
        )
        ownership_upper_time[:, shot] = upper
        ownership_lower_time[:, shot] = lower
        ownership_order_violation[:, shot] = violation
        ownership_upper_flat[:, shot] = upper - moveouts
        ownership_lower_flat[:, shot] = lower - moveouts
        for reflector in range(nref):
            row = tref[reflector, shot]
            if not np.isfinite(row[i0]):
                continue
            seed_time = float(supplied_control_time[reflector, shot])
            if not np.isfinite(seed_time):
                continue
            used_control_time[reflector, shot] = seed_time
            flat, moveout, receiver_valid, flat_support = flatten_observed_gather(
                observed[shot],
                row,
                seed_receiver=i0,
                dt=dt,
                t0=t0,
                return_support=True,
            )
            dense_tracking = None
            sparse = None
            if tracking_method == "sparse_event_dp":
                from ..tracking.flat_event import track_flattened_event_sparse

                tracker_cfg = dict(sparse_tracking_options or {})
                seed_cfg = dict(seed_search_options or {})
                ownership_escape_cfg = dict(tracker_cfg.pop("ownership_escape", {}) or {})
                # Accept continuation_rescue from either config block, but pass
                # it explicitly only once.  Pilot validation disables rescue
                # internally; this configuration is for the production full
                # continuation after a seed/anchor has been accepted.
                continuation_rescue_cfg = seed_cfg.get(
                    "continuation_rescue",
                    tracker_cfg.pop("continuation_rescue", None),
                )

                state_valid = None
                escape_state_valid = None
                if ownership_enabled:
                    time = float(t0) + np.arange(observed.shape[2]) * float(dt)
                    state_valid = (
                        (time[None, :] >= ownership_upper_flat[reflector, shot, :, None])
                        & (time[None, :] <= ownership_lower_flat[reflector, shot, :, None])
                    )
                    if ownership_escape_cfg.get("enabled", False):
                        escape_upper, escape_lower = _ownership_escape_bounds(
                            predicted_center[:, shot], ownership_upper_time[:, shot],
                            ownership_lower_time[:, shot],
                            ownership_escape_cfg.get("extra_gap_fraction", 0.20),
                            ownership_escape_cfg.get("max_extra_time", 0.120),
                        )
                        escape_upper_flat = escape_upper[reflector] - moveouts[reflector]
                        escape_lower_flat = escape_lower[reflector] - moveouts[reflector]
                        escape_state_valid = (
                            (time[None, :] >= escape_upper_flat[:, None])
                            & (time[None, :] <= escape_lower_flat[:, None])
                        )
                    control_sample = int(np.rint((seed_time - float(t0)) / float(dt)))
                    if 0 <= control_sample < state_valid.shape[1] and not state_valid[i0, control_sample]:
                        print(
                            f"[warning] ownership excludes theoretical control: "
                            f"R{reflector + 1}, shot {shot + 1}; control remains a search prior",
                            flush=True,
                        )
                tracking_usable_receiver[reflector, shot] = receiver_valid & np.any(
                    flat_support & (state_valid if state_valid is not None else True), axis=1
                )
                quality_available_receiver[reflector, shot] = receiver_valid & np.any(
                    flat_support
                    & (state_valid if state_valid is not None else True)
                    & (flat != 0.0),
                    axis=1,
                )
                control_sample = int(np.rint((seed_time - t0) / dt))
                control_amplitude = (
                    float(flat[i0, control_sample])
                    if 0 <= control_sample < flat.shape[1]
                    else np.nan
                )
                snapped = (seed_time, control_sample, False, 0.0, control_amplitude)
                search_result = None
                seed_repair_exhausted_to_break = False
                if seed_search_options is not None:
                    from ..tracking.seed_search import discover_tracking_seed
                    if state_valid is None:
                        state_valid = np.ones_like(flat, dtype=bool)
                    search_result = discover_tracking_seed(
                        flat, receiver_x=receivers[shot, :, 0], valid_receiver=receiver_valid,
                        seed_receiver=i0, control_time=seed_time, state_valid=state_valid,
                        dt=dt, t0=t0, tracker_options=tracker_cfg,
                        **seed_cfg,
                    )
                    seed_search_runtime += search_result.runtime_seconds
                    pilot_transition_count += search_result.pilot_transition_count
                    if search_result.mode == "LOCAL" and bool(search_result.high_confidence):
                        seed_local_high_confidence_count += 1
                    seed_valid[reflector, shot] = search_result.valid
                    seed_search_mode[reflector, shot] = search_result.mode
                    seed_stack_evidence[reflector, shot] = search_result.stack_evidence
                    seed_positive_support_fraction[reflector, shot] = search_result.positive_support_fraction
                    seed_hypothesis_count[reflector, shot] = search_result.hypothesis_count
                    seed_selected_hypothesis[reflector, shot] = search_result.selected_hypothesis
                    pilot_coverage[reflector, shot] = search_result.pilot_coverage
                    pilot_median_zncc[reflector, shot] = search_result.pilot_median_zncc
                    pilot_median_abs_prediction_error[reflector, shot] = search_result.pilot_median_abs_prediction_error
                    pilot_median_abs_slope_ms_per_100m[reflector, shot] = search_result.pilot_median_abs_slope_ms_per_100m
                    pilot_phase_switch_count[reflector, shot] = search_result.phase_switch_count
                    pilot_left_support[reflector, shot] = search_result.pilot_left_support
                    pilot_right_support[reflector, shot] = search_result.pilot_right_support
                    anchor_left_count[reflector, shot] = search_result.anchor_left_count
                    anchor_right_count[reflector, shot] = search_result.anchor_right_count
                    anchor_receiver_mask[reflector, shot] = search_result.anchor_receiver_mask
                    pilot_pick_time[reflector, shot] = search_result.pilot_pick_time
                    if not search_result.valid:
                        if quality_audit_config is None:
                            trace_valid[reflector, shot] = receiver_valid
                            continue
                        failed_seed_times = [seed_time]
                        last_failed_seed = None
                        last_failed_sparse = None
                        for _ in range(3):
                            seed_repair_attempt_count += 1
                            repaired_seed = discover_tracking_seed(
                                flat,
                                receiver_x=receivers[shot, :, 0],
                                valid_receiver=receiver_valid,
                                seed_receiver=i0,
                                control_time=seed_time,
                                state_valid=state_valid,
                                dt=dt,
                                t0=t0,
                                tracker_options=tracker_cfg,
                                repair_exhaustive=True,
                                excluded_seed_times=failed_seed_times,
                                **seed_cfg,
                            )
                            seed_search_runtime += repaired_seed.runtime_seconds
                            pilot_transition_count += repaired_seed.pilot_transition_count
                            if not repaired_seed.valid:
                                break
                            repaired_sparse = track_flattened_event_sparse(
                                flat,
                                receiver_x=receivers[shot, :, 0],
                                valid_receiver=receiver_valid,
                                seed_receiver=i0,
                                seed_time=repaired_seed.seed_time,
                                state_valid=state_valid,
                                escape_state_valid=escape_state_valid,
                                anchor_receiver_mask=repaired_seed.anchor_receiver_mask,
                                anchor_pick_sample=repaired_seed.anchor_pick_sample,
                                anchor_pick_time=repaired_seed.anchor_pick_time,
                                dt=dt,
                                t0=t0,
                                continuation_rescue=continuation_rescue_cfg,
                                ownership_escape=ownership_escape_cfg,
                                **tracker_cfg,
                            )
                            repaired_success = repaired_sparse.success_mask & np.isfinite(moveout)
                            from ..tracking.quality_control import TrackingStatus, audit_rkshot
                            repaired_audit = audit_rkshot(
                                success_mask=repaired_success,
                                valid_receiver=receiver_valid,
                                neighbor_similarity=repaired_sparse.neighbor_correlation,
                                prediction_error=repaired_sparse.prediction_error,
                                seed_receiver=i0,
                                anchor_receiver_mask=repaired_seed.anchor_receiver_mask,
                                tracking_usable_receiver=tracking_usable_receiver[reflector, shot],
                                quality_available_receiver=quality_available_receiver[reflector, shot],
                                config=quality_audit_config,
                                reflector=reflector + 1,
                            )
                            last_failed_seed = repaired_seed
                            last_failed_sparse = repaired_sparse
                            if repaired_audit.status == TrackingStatus.NORMAL:
                                search_result = repaired_seed
                                sparse = repaired_sparse
                                seed_repair_success_count += 1
                                break
                            failed_seed_times.append(float(repaired_seed.seed_time))
                        if sparse is None:
                            if last_failed_sparse is None:
                                unrecoverable_mask[reflector, shot] = True
                                trace_valid[reflector, shot] = receiver_valid
                                continue
                            search_result = last_failed_seed
                            sparse = last_failed_sparse
                            seed_repair_exhausted_to_break = True
                        seed_valid[reflector, shot] = True
                        seed_search_mode[reflector, shot] = search_result.mode
                        seed_stack_evidence[reflector, shot] = search_result.stack_evidence
                        seed_positive_support_fraction[reflector, shot] = search_result.positive_support_fraction
                        seed_hypothesis_count[reflector, shot] = search_result.hypothesis_count
                        seed_selected_hypothesis[reflector, shot] = search_result.selected_hypothesis
                        pilot_coverage[reflector, shot] = search_result.pilot_coverage
                        pilot_median_zncc[reflector, shot] = search_result.pilot_median_zncc
                        pilot_median_abs_prediction_error[reflector, shot] = search_result.pilot_median_abs_prediction_error
                        pilot_median_abs_slope_ms_per_100m[reflector, shot] = search_result.pilot_median_abs_slope_ms_per_100m
                        pilot_phase_switch_count[reflector, shot] = search_result.phase_switch_count
                        pilot_left_support[reflector, shot] = search_result.pilot_left_support
                        pilot_right_support[reflector, shot] = search_result.pilot_right_support
                        anchor_left_count[reflector, shot] = search_result.anchor_left_count
                        anchor_right_count[reflector, shot] = search_result.anchor_right_count
                        anchor_receiver_mask[reflector, shot] = search_result.anchor_receiver_mask
                        pilot_pick_time[reflector, shot] = search_result.pilot_pick_time
                    snapped = (search_result.seed_time, search_result.seed_sample, True,
                               search_result.seed_time - seed_time, search_result.positive_amplitude)
                elif seed_snap_enabled:
                    snapped = snap_control_to_observed_peak(
                        flat[i0], control_time=seed_time, dt=dt, t0=t0,
                        state_valid_row=None if state_valid is None else state_valid[i0],
                        half_width_time=seed_snap_half_width_time,
                    )
                    seed_valid[reflector, shot] = bool(snapped[2])
                    seed_search_mode[reflector, shot] = "LOCAL" if snapped[2] else "FAILED"
                    if not snapped[2]:
                        trace_valid[reflector, shot] = receiver_valid
                        continue
                else:
                    seed_valid[reflector, shot] = True
                    seed_search_mode[reflector, shot] = "LOCAL"
                tracked_seed, _, snapped_ok, snapped_delta, snapped_amplitude = snapped
                tracking_seed_time[reflector, shot] = tracked_seed
                seed_snap_success[reflector, shot] = snapped_ok
                seed_snap_delta_time[reflector, shot] = snapped_delta
                seed_snap_amplitude[reflector, shot] = snapped_amplitude
                if sparse is None:
                    sparse = track_flattened_event_sparse(
                        flat,
                        receiver_x=receivers[shot, :, 0],
                        valid_receiver=receiver_valid,
                        seed_receiver=i0,
                        seed_time=tracked_seed,
                        state_valid=state_valid,
                        escape_state_valid=escape_state_valid,
                        anchor_receiver_mask=(search_result.anchor_receiver_mask if search_result is not None else None),
                        anchor_pick_sample=(search_result.anchor_pick_sample if search_result is not None else None),
                        anchor_pick_time=(search_result.anchor_pick_time if search_result is not None else None),
                        dt=dt,
                        t0=t0,
                        continuation_rescue=continuation_rescue_cfg,
                        ownership_escape=ownership_escape_cfg,
                        **tracker_cfg,
                    )
                if quality_audit_config is not None and search_result is not None:
                    from ..tracking.quality_control import TrackingStatus, audit_rkshot

                    initial_success = sparse.success_mask & np.isfinite(moveout)
                    initial_audit = audit_rkshot(
                        success_mask=initial_success,
                        valid_receiver=receiver_valid,
                        neighbor_similarity=sparse.neighbor_correlation,
                        prediction_error=sparse.prediction_error,
                        seed_receiver=i0,
                        anchor_receiver_mask=search_result.anchor_receiver_mask,
                        tracking_usable_receiver=tracking_usable_receiver[reflector, shot],
                        quality_available_receiver=quality_available_receiver[reflector, shot],
                        config=quality_audit_config,
                        reflector=reflector + 1,
                    )
                    cascade_seed_to_break = seed_repair_exhausted_to_break
                    break_control_time = (
                        float(search_result.seed_time)
                        if seed_repair_exhausted_to_break else seed_time
                    )
                    if (
                        initial_audit.status == TrackingStatus.SEED_ERROR
                        and not cascade_seed_to_break
                    ):
                        failed_seed_times = [float(search_result.seed_time)]
                        repaired = False
                        last_failed_seed = None
                        last_failed_sparse = None
                        last_failed_audit = None
                        for _ in range(3):
                            seed_repair_attempt_count += 1
                            repaired_seed = discover_tracking_seed(
                                flat,
                                receiver_x=receivers[shot, :, 0],
                                valid_receiver=receiver_valid,
                                seed_receiver=i0,
                                control_time=seed_time,
                                state_valid=state_valid,
                                dt=dt,
                                t0=t0,
                                tracker_options=tracker_cfg,
                                repair_exhaustive=True,
                                excluded_seed_times=failed_seed_times,
                                **seed_cfg,
                            )
                            seed_search_runtime += repaired_seed.runtime_seconds
                            pilot_transition_count += repaired_seed.pilot_transition_count
                            if not repaired_seed.valid:
                                break
                            repaired_sparse = track_flattened_event_sparse(
                                flat,
                                receiver_x=receivers[shot, :, 0],
                                valid_receiver=receiver_valid,
                                seed_receiver=i0,
                                seed_time=repaired_seed.seed_time,
                                state_valid=state_valid,
                                escape_state_valid=escape_state_valid,
                                anchor_receiver_mask=repaired_seed.anchor_receiver_mask,
                                anchor_pick_sample=repaired_seed.anchor_pick_sample,
                                anchor_pick_time=repaired_seed.anchor_pick_time,
                                dt=dt,
                                t0=t0,
                                continuation_rescue=continuation_rescue_cfg,
                                ownership_escape=ownership_escape_cfg,
                                **tracker_cfg,
                            )
                            repaired_success = repaired_sparse.success_mask & np.isfinite(moveout)
                            repaired_audit = audit_rkshot(
                                success_mask=repaired_success,
                                valid_receiver=receiver_valid,
                                neighbor_similarity=repaired_sparse.neighbor_correlation,
                                prediction_error=repaired_sparse.prediction_error,
                                seed_receiver=i0,
                                anchor_receiver_mask=repaired_seed.anchor_receiver_mask,
                                tracking_usable_receiver=tracking_usable_receiver[reflector, shot],
                                quality_available_receiver=quality_available_receiver[reflector, shot],
                                config=quality_audit_config,
                                reflector=reflector + 1,
                            )
                            last_failed_seed = repaired_seed
                            last_failed_sparse = repaired_sparse
                            last_failed_audit = repaired_audit
                            if repaired_audit.status == TrackingStatus.NORMAL:
                                search_result = repaired_seed
                                sparse = repaired_sparse
                                tracked_seed = float(repaired_seed.seed_time)
                                tracking_seed_time[reflector, shot] = tracked_seed
                                seed_search_mode[reflector, shot] = repaired_seed.mode
                                seed_stack_evidence[reflector, shot] = repaired_seed.stack_evidence
                                seed_positive_support_fraction[reflector, shot] = repaired_seed.positive_support_fraction
                                seed_hypothesis_count[reflector, shot] = repaired_seed.hypothesis_count
                                seed_selected_hypothesis[reflector, shot] = repaired_seed.selected_hypothesis
                                pilot_coverage[reflector, shot] = repaired_seed.pilot_coverage
                                pilot_median_zncc[reflector, shot] = repaired_seed.pilot_median_zncc
                                pilot_median_abs_prediction_error[reflector, shot] = repaired_seed.pilot_median_abs_prediction_error
                                pilot_median_abs_slope_ms_per_100m[reflector, shot] = repaired_seed.pilot_median_abs_slope_ms_per_100m
                                pilot_phase_switch_count[reflector, shot] = repaired_seed.phase_switch_count
                                pilot_left_support[reflector, shot] = repaired_seed.pilot_left_support
                                pilot_right_support[reflector, shot] = repaired_seed.pilot_right_support
                                anchor_left_count[reflector, shot] = repaired_seed.anchor_left_count
                                anchor_right_count[reflector, shot] = repaired_seed.anchor_right_count
                                anchor_receiver_mask[reflector, shot] = repaired_seed.anchor_receiver_mask
                                pilot_pick_time[reflector, shot] = repaired_seed.pilot_pick_time
                                seed_repair_success_count += 1
                                repaired = True
                                break
                            failed_seed_times.append(float(repaired_seed.seed_time))
                        if not repaired:
                            if last_failed_sparse is None:
                                unrecoverable_mask[reflector, shot] = True
                            else:
                                cascade_seed_to_break = True
                                search_result = last_failed_seed
                                sparse = last_failed_sparse
                                break_control_time = float(last_failed_seed.seed_time)
                    if (
                        initial_audit.status == TrackingStatus.TRACKING_BREAK
                        or cascade_seed_to_break
                    ):
                        from ..tracking.tracking_break_repair import repair_tracking_break

                        if cascade_seed_to_break:
                            print(
                                f"[SEED-REPAIR] R{reflector + 1} shot {shot + 1} "
                                f"exhausted; enter TRACKING_BREAK repair once",
                                flush=True,
                            )
                            print(f"  source_seed_time = {break_control_time:.6f}", flush=True)
                            print(f"  source_audit = {initial_audit.status.value}", flush=True)
                        tracking_break_repair_entered[reflector, shot] = True
                        upper_neighbor_state_valid = None
                        lower_neighbor_state_valid = None
                        if ownership_enabled and reflector > 0:
                            upper_neighbor_state_valid = (
                                (time[None, :] >= ownership_upper_time[reflector - 1, shot, :, None] - moveout[:, None])
                                & (time[None, :] <= ownership_lower_time[reflector - 1, shot, :, None] - moveout[:, None])
                            )
                        if ownership_enabled and reflector + 1 < nref:
                            lower_neighbor_state_valid = (
                                (time[None, :] >= ownership_upper_time[reflector + 1, shot, :, None] - moveout[:, None])
                                & (time[None, :] <= ownership_lower_time[reflector + 1, shot, :, None] - moveout[:, None])
                            )
                        repair = repair_tracking_break(
                            flat=flat,
                            receiver_x=receivers[shot, :, 0],
                            valid_receiver=receiver_valid,
                            control_receiver=i0,
                            control_time=break_control_time,
                            state_valid=state_valid,
                            escape_state_valid=escape_state_valid,
                            tracking_usable_receiver=tracking_usable_receiver[reflector, shot],
                            quality_available_receiver=quality_available_receiver[reflector, shot],
                            original_tracking=sparse,
                            tracker_options=tracker_cfg,
                            seed_search_options=seed_cfg,
                            continuation_rescue=continuation_rescue_cfg,
                            ownership_escape=ownership_escape_cfg,
                            audit_config=quality_audit_config,
                            reflector=reflector + 1,
                            shot=shot + 1,
                            dt=dt,
                            t0=t0,
                            upper_neighbor_state_valid=upper_neighbor_state_valid,
                            lower_neighbor_state_valid=lower_neighbor_state_valid,
                        )
                        tracking_break_original_trusted_segments[reflector, shot] = ";".join(
                            f"{item.start_receiver}:{item.end_receiver}"
                            for item in repair.original_trusted_segments
                        )
                        tracking_break_trusted_segments[reflector, shot] = ";".join(
                            f"{item.start_receiver}:{item.end_receiver}:{item.source_type}:"
                            f"coverage={item.coverage:.8g}:energy={item.event_strength:.8g}"
                            for item in repair.selected_segments
                        )
                        tracking_break_boundary_seed_receivers[reflector, shot] = ",".join(
                            str(value) for value in repair.boundary_seed_receivers
                        )
                        tracking_break_boundary_seed_times[reflector, shot] = ",".join(
                            f"{value:.8g}" for value in repair.boundary_seed_times
                        )
                        tracking_break_boundary_seed_energy[reflector, shot] = ",".join(
                            f"{value:.8g}" for value in repair.boundary_seed_energy
                        )
                        tracking_break_boundary_candidate_diagnostics[reflector, shot] = ";".join(
                            repair.boundary_candidate_diagnostics
                        )
                        tracking_break_seed_receiver_count[reflector, shot] = len(repair.boundary_seed_receivers)
                        tracking_break_seed_attempts[reflector, shot] = str(len(repair.boundary_seed_times))
                        tracking_break_full_candidate_count[reflector, shot] = repair.full_candidate_count
                        tracking_break_trusted_match_fraction[reflector, shot] = repair.trusted_union_coverage
                        tracking_break_trusted_coverage[reflector, shot] = repair.trusted_union_coverage
                        tracking_break_selected_segment_count[reflector, shot] = len(repair.selected_segments)
                        tracking_break_candidate_segment_count[reflector, shot] = repair.candidate_trusted_segment_count
                        tracking_break_boundary_seed_candidate_count[reflector, shot] = sum(
                            not item.startswith("full:")
                            for item in repair.boundary_candidate_diagnostics
                        )
                        tracking_break_boundary_seed_pilots_accepted[reflector, shot] = (
                            repair.boundary_seed_pilots_accepted
                        )
                        if repair.audit is not None:
                            tracking_break_merged_raw_audit[reflector, shot] = repair.audit.status.value
                        tracking_break_repair_success[reflector, shot] = repair.success
                        if repair.success:
                            sparse = repair.tracking
                            if cascade_seed_to_break:
                                tracked_seed = break_control_time
                                tracking_seed_time[reflector, shot] = tracked_seed
                                seed_search_mode[reflector, shot] = search_result.mode
                        else:
                            unrecoverable_mask[reflector, shot] = True
                sparse_totals["transitions"] += sparse.transition_count
                sparse_totals["correlations"] += sparse.correlation_count
                sparse_totals["active_max"] = max(sparse_totals["active_max"], sparse.active_state_max)
                sparse_totals["runtime"] += sparse.runtime_seconds
                sparse_totals["rescue_attempts"] += sparse.rescue_attempt_count
                sparse_totals["rescue_successes"] += sparse.rescue_success_count
                sparse_totals["rescue_candidates"] += sparse.rescue_candidate_count
                sparse_totals["rescue_transitions"] += sparse.rescue_transition_count
                sparse_totals["valid_gaps"] += int(np.count_nonzero(sparse.skipped_valid_receiver_mask))
                sparse_totals["escape_attempts"] += sparse.ownership_escape_attempt_count
                sparse_totals["escape_successes"] += sparse.ownership_escape_success_count
                sparse_totals["escape_candidates"] += sparse.ownership_escape_candidate_count
                rescue_used_mask[reflector, shot] = sparse.rescue_used_mask
                skipped_valid_receiver_mask[reflector, shot] = sparse.skipped_valid_receiver_mask
                ownership_escape_used_mask[reflector, shot] = sparse.ownership_escape_used_mask
                stop_receiver_left[reflector, shot] = sparse.stop_receiver_left
                stop_receiver_right[reflector, shot] = sparse.stop_receiver_right
                stop_reason_left[reflector, shot] = sparse.stop_reason_left
                stop_reason_right[reflector, shot] = sparse.stop_reason_right
                neighbor_correlation[reflector, shot] = sparse.neighbor_correlation
                residual_slope[reflector, shot] = sparse.residual_slope
                prediction_error[reflector, shot] = sparse.prediction_error
                candidate_count[reflector, shot] = sparse.candidate_count
                candidate_count_before[reflector, shot] = sparse.candidate_count_before_ownership
                selected_amplitude[reflector, shot] = sparse.selected_amplitude
                selected_envelope[reflector, shot] = sparse.selected_envelope
                selected_polarity[reflector, shot] = sparse.selected_polarity
            else:
                raise TobsBootstrapError(
                    "Only tracking_method='sparse_event_dp' is supported by the "
                    "current Tobs bootstrap."
                )
            tracking = sparse
            success = tracking.success_mask & np.isfinite(moveout)
            center[reflector, shot, success] = (
                tracking.pick_time[success] + moveout[success]
            )
            tracking_success[reflector, shot] = success
            trace_valid[reflector, shot] = receiver_valid
            tracking_score[reflector, shot] = sparse.neighbor_correlation
            guide_pick_time[reflector, shot] = np.where(
                sparse.candidate_sample >= 0,
                float(t0) + sparse.candidate_sample * float(dt),
                np.nan,
            )
            guide_pick_sample[reflector, shot] = sparse.candidate_sample
            guide_score[reflector, shot] = sparse.neighbor_correlation
            margin = int(boundary_margin_samples)
            boundary_flag[reflector, shot] = success & (
                (sparse.pick_sample <= margin)
                | (sparse.pick_sample >= observed.shape[2] - 1 - margin)
            )
        if tracking_method == "sparse_event_dp":
            before = candidate_count_before[:, shot]
            after = candidate_count[:, shot]
            print(
                f"[Tobs bootstrap] shot {shot + 1}/{nshot} complete | "
                f"candidates median {np.median(before):.1f}->{np.median(after):.1f} | "
                f"transitions={sparse_totals['transitions']} | "
                f"elapsed={perf_counter() - shot_started:.1f}s",
                flush=True,
            )

    counts = {
        "total_slots": int(center.size),
        "tracked_slots": int(np.count_nonzero(tracking_success)),
        "untracked_slots": int(center.size - np.count_nonzero(tracking_success)),
        "valid_trace_slots": int(np.count_nonzero(trace_valid)),
        "control_points": int(np.count_nonzero(np.isfinite(used_control_time))),
        "tracked_by_reflector": tuple(
            int(np.count_nonzero(tracking_success[reflector]))
            for reflector in range(nref)
        ),
        "sparse_transition_count": int(sparse_totals["transitions"]),
        "sparse_correlation_count": int(sparse_totals["correlations"]),
        "sparse_active_state_max": int(sparse_totals["active_max"]),
        "sparse_runtime_seconds": float(sparse_totals["runtime"]),
        "ownership_order_violation_count": int(np.count_nonzero(ownership_order_violation)),
        "dense_diagnostic_runtime_seconds": float(dense_diagnostic_runtime),
        "dense_diagnostic_call_count": int(dense_diagnostic_calls),
        "seed_search_runtime_seconds": float(seed_search_runtime),
        "pilot_transition_count": int(pilot_transition_count),
        "full_transition_count": int(sparse_totals["transitions"]),
        "rescue_attempt_count": int(sparse_totals["rescue_attempts"]),
        "rescue_success_count": int(sparse_totals["rescue_successes"]),
        "rescue_candidate_count": int(sparse_totals["rescue_candidates"]),
        "normal_transition_count": int(sparse_totals["transitions"] - sparse_totals["rescue_transitions"]),
        "skipped_valid_receiver_count": int(sparse_totals["valid_gaps"]),
        "ownership_escape_attempt_count": int(sparse_totals["escape_attempts"]),
        "ownership_escape_success_count": int(sparse_totals["escape_successes"]),
        "ownership_escape_candidate_count": int(sparse_totals["escape_candidates"]),
        "final_continuation_failure_count": int(np.count_nonzero(
            seed_valid & (
                ~np.isin(stop_reason_left, ("END_OF_GATHER", "END_OF_VALID_GATHER"))
                | ~np.isin(stop_reason_right, ("END_OF_GATHER", "END_OF_VALID_GATHER"))
            )
        )),
        "seed_local_success_count": int(np.count_nonzero(seed_search_mode == "LOCAL")),
        "seed_local_high_confidence_count": int(seed_local_high_confidence_count),
        "seed_adaptive_success_count": int(np.count_nonzero(seed_search_mode == "ADAPTIVE")),
        "seed_ownership_wide_success_count": int(np.count_nonzero(seed_search_mode == "OWNERSHIP_WIDE")),
        "seed_failed_count": int(np.count_nonzero(seed_search_mode == "FAILED")),
        "seed_repair_attempt_count": int(seed_repair_attempt_count),
        "seed_repair_success_count": int(seed_repair_success_count),
        "tracking_break_repair_entered": tracking_break_repair_entered,
        "tracking_break_trusted_segments": tracking_break_trusted_segments,
        "tracking_break_seed_receiver_count": tracking_break_seed_receiver_count,
        "tracking_break_seed_attempts": tracking_break_seed_attempts,
        "tracking_break_full_candidate_count": tracking_break_full_candidate_count,
        "tracking_break_normal_candidate_count": tracking_break_normal_candidate_count,
        "tracking_break_winner_seed_receiver": tracking_break_winner_seed_receiver,
        "tracking_break_winner_seed_time": tracking_break_winner_seed_time,
        "tracking_break_trusted_match_fraction": tracking_break_trusted_match_fraction,
        "tracking_break_winner_rank": tracking_break_winner_rank,
        "tracking_break_repair_success": tracking_break_repair_success,
        "tracking_break_original_trusted_segments": tracking_break_original_trusted_segments,
        "tracking_break_boundary_seed_receivers": tracking_break_boundary_seed_receivers,
        "tracking_break_boundary_seed_times": tracking_break_boundary_seed_times,
        "tracking_break_boundary_seed_energy": tracking_break_boundary_seed_energy,
        "tracking_break_boundary_candidate_diagnostics": tracking_break_boundary_candidate_diagnostics,
        "tracking_break_trusted_coverage": tracking_break_trusted_coverage,
        "tracking_break_selected_segment_count": tracking_break_selected_segment_count,
        "tracking_break_candidate_segment_count": tracking_break_candidate_segment_count,
        "tracking_break_boundary_seed_candidate_count": tracking_break_boundary_seed_candidate_count,
        "tracking_break_boundary_seed_pilots_accepted": tracking_break_boundary_seed_pilots_accepted,
        "tracking_break_merged_raw_audit": tracking_break_merged_raw_audit,
        "median_anchor_length": float(np.median(
            (anchor_left_count + anchor_right_count + 1)[seed_valid]
        )) if np.any(seed_valid) else 0.0,
    }
    return ObservedCenterBootstrapResult(
        observed_center=center,
        tracking_success=tracking_success,
        trace_valid=trace_valid,
        tracking_score=tracking_score,
        guide_pick_time=guide_pick_time,
        guide_pick_sample=guide_pick_sample,
        guide_score=guide_score,
        legacy_pick_time=legacy_pick_time,
        neighbor_correlation=neighbor_correlation,
        residual_slope=residual_slope,
        prediction_error=prediction_error,
        candidate_count=candidate_count,
        candidate_count_before_ownership=candidate_count_before,
        predicted_center=predicted_center,
        ownership_upper_time=ownership_upper_time,
        ownership_lower_time=ownership_lower_time,
        ownership_upper_flat=ownership_upper_flat,
        ownership_lower_flat=ownership_lower_flat,
        ownership_order_violation=ownership_order_violation,
        selected_amplitude=selected_amplitude,
        selected_envelope=selected_envelope,
        selected_polarity=selected_polarity,
        stop_reason_left=stop_reason_left,
        stop_reason_right=stop_reason_right,
        rescue_used_mask=rescue_used_mask,
        skipped_valid_receiver_mask=skipped_valid_receiver_mask,
        ownership_escape_used_mask=ownership_escape_used_mask,
        stop_receiver_left=stop_receiver_left,
        stop_receiver_right=stop_receiver_right,
        boundary_flag=boundary_flag,
        control_receiver=control_receiver,
        control_time=used_control_time,
        tracking_seed_time=tracking_seed_time,
        seed_snap_delta_time=seed_snap_delta_time,
        seed_snap_success=seed_snap_success,
        seed_snap_amplitude=seed_snap_amplitude,
        seed_valid=seed_valid,
        seed_search_mode=seed_search_mode,
        seed_stack_evidence=seed_stack_evidence,
        seed_positive_support_fraction=seed_positive_support_fraction,
        seed_hypothesis_count=seed_hypothesis_count,
        seed_selected_hypothesis=seed_selected_hypothesis,
        pilot_coverage=pilot_coverage,
        pilot_median_zncc=pilot_median_zncc,
        pilot_median_abs_prediction_error=pilot_median_abs_prediction_error,
        pilot_median_abs_slope_ms_per_100m=pilot_median_abs_slope_ms_per_100m,
        pilot_phase_switch_count=pilot_phase_switch_count,
        pilot_left_support=pilot_left_support,
        pilot_right_support=pilot_right_support,
        anchor_left_count=anchor_left_count,
        anchor_right_count=anchor_right_count,
        anchor_receiver_mask=anchor_receiver_mask,
        pilot_pick_time=pilot_pick_time,
        tracking_usable_receiver=tracking_usable_receiver,
        quality_available_receiver=quality_available_receiver,
        unrecoverable_mask=unrecoverable_mask,
        counts=counts,
    )


__all__ = [
    "FlatEventTrackingResult",
    "ObservedCenterBootstrapResult",
    "TobsBootstrapError",
    "build_bootstrap_fixed_mask",
    "build_observed_centers_from_eikonal",
    "flatten_observed_gather",
    "nearest_offset_receiver",
    "snap_control_to_observed_peak",
    "track_flattened_event",
]
