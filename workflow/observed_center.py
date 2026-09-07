"""Observed-center recovery for edge gaps in the frozen reference stage."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import hilbert


DIRECT_TRACKING = 1
QUADRATIC_PLUS_ENVELOPE = 2
QUADRATIC_ONLY = 3


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservedCenterRecoveryResult:
    """Frozen observed centers and diagnostics from one reference bootstrap."""

    observed_center: np.ndarray
    observed_center_source: np.ndarray
    counts: Mapping[str, object]

    def __post_init__(self) -> None:
        center = np.asarray(self.observed_center, dtype=float)
        source = np.asarray(self.observed_center_source, dtype=int)
        if center.ndim != 3 or source.shape != center.shape:
            raise ValueError(
                "observed_center and observed_center_source must have shape [nref, ns, nr]."
            )
        if np.any((source < 0) | (source > QUADRATIC_ONLY)):
            raise ValueError("observed_center_source contains an unknown source code.")
        object.__setattr__(self, "observed_center", _readonly(center, float))
        object.__setattr__(self, "observed_center_source", _readonly(source, int))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def recovered_mask(self) -> np.ndarray:
        return _readonly(
            (self.observed_center_source == QUADRATIC_PLUS_ENVELOPE)
            | (self.observed_center_source == QUADRATIC_ONLY),
            bool,
        )

    @property
    def observed_center_final(self) -> np.ndarray:
        return self.observed_center


def select_observed_center_anchors(
    fixed_mask: np.ndarray,
    tracking,
    *,
    correlation_threshold: float = 0.70,
) -> np.ndarray:
    """Select high-confidence anchors without changing tracking or QC state."""

    if not np.isfinite(correlation_threshold):
        raise ValueError("correlation_threshold must be finite.")
    fixed = np.asarray(fixed_mask, dtype=bool)
    success = np.asarray(tracking.success_mask, dtype=bool)
    shift = np.asarray(tracking.shift_time, dtype=float)
    correlation = np.asarray(tracking.tracked_correlation, dtype=float)
    boundary = np.asarray(tracking.boundary_flag, dtype=bool)
    if fixed.ndim != 1 or any(
        value.shape != fixed.shape for value in (success, shift, correlation, boundary)
    ):
        raise ValueError(
            "fixed_mask and tracking arrays must have equal one-dimensional shapes."
        )
    return (
        fixed
        & success
        & np.isfinite(shift)
        & np.isfinite(correlation)
        & (correlation >= float(correlation_threshold))
        & ~boundary
    )


def weighted_quadratic_fit(
    offset: np.ndarray,
    observed_center: np.ndarray,
    correlation: np.ndarray,
) -> np.ndarray:
    """Fit ``a0 + a1*offset + a2*offset**2`` with correlation-squared weights."""

    h = np.asarray(offset, dtype=float)
    center = np.asarray(observed_center, dtype=float)
    corr = np.asarray(correlation, dtype=float)
    if h.ndim != 1 or center.shape != h.shape or corr.shape != h.shape:
        raise ValueError(
            "offset, observed_center, and correlation must be equal 1-D arrays."
        )
    valid = np.isfinite(h) & np.isfinite(center) & np.isfinite(corr)
    if np.count_nonzero(valid) < 3:
        raise ValueError("at least three finite samples are required for a quadratic fit.")
    h = h[valid]
    center = center[valid]
    weights = corr[valid] ** 2
    if not np.any(weights > 0.0):
        raise ValueError("quadratic fit requires at least one positive weight.")
    design = np.column_stack((np.ones(h.size), h, h * h))
    root_weight = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * root_weight[:, None], center * root_weight, rcond=None
    )
    if rank < 3 or not np.isfinite(coefficients).all():
        raise ValueError("quadratic fit is rank deficient or non-finite.")
    return coefficients


def _validate_snap_args(dt: float, t0: float, half_width_time: float) -> None:
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("dt must be finite and positive and t0 must be finite.")
    if not np.isfinite(half_width_time) or half_width_time < 0:
        raise ValueError("half_width_time must be finite and non-negative.")


def _snap_envelope(
    predicted_center: float,
    envelope: np.ndarray,
    *,
    dt: float,
    t0: float,
    half_width_time: float,
) -> tuple[float, bool]:
    left = predicted_center - half_width_time
    right = predicted_center + half_width_time
    last_time = t0 + (envelope.size - 1) * float(dt)
    if envelope.size == 0 or left < t0 or right > last_time:
        return float(predicted_center), False
    times = t0 + np.arange(envelope.size, dtype=float) * float(dt)
    selected = (
        np.isfinite(envelope)
        & (times >= left)
        & (times <= right)
    )
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        return float(predicted_center), False
    return float(times[indices[np.argmax(envelope[indices])]]), True


def snap_observed_center(
    predicted_center: float,
    observed_trace: np.ndarray,
    *,
    dt: float,
    t0: float = 0.0,
    half_width_time: float = 0.040,
) -> tuple[float, bool]:
    """Snap a prediction to the largest finite analytic-envelope sample."""

    if not np.isfinite(predicted_center):
        return float(predicted_center), False
    _validate_snap_args(dt, t0, half_width_time)
    trace = np.asarray(observed_trace, dtype=float)
    if trace.ndim != 1 or not np.isfinite(trace).all():
        return float(predicted_center), False
    try:
        envelope = np.abs(hilbert(trace))
    except Exception:
        return float(predicted_center), False
    return _snap_envelope(
        predicted_center,
        envelope,
        dt=dt,
        t0=t0,
        half_width_time=half_width_time,
    )


def recover_edge_observed_centers(
    theoretical_center: np.ndarray,
    fixed_mask: np.ndarray,
    reference_tracking: Sequence[Sequence],
    source_coordinates: np.ndarray,
    receiver_coordinates: np.ndarray,
    observed_data: np.ndarray,
    *,
    dt: float,
    t0: float = 0.0,
    anchor_corr_threshold: float = 0.70,
    max_fit_anchors: int = 30,
    min_anchors: int = 10,
    envelope_snap_half_width_time: float = 0.040,
) -> ObservedCenterRecoveryResult:
    """Recover only fixed-mask unresolved receiver runs at either edge."""

    theoretical = np.asarray(theoretical_center, dtype=float)
    fixed = np.asarray(fixed_mask, dtype=bool)
    observed = np.asarray(observed_data, dtype=float)
    sources = np.asarray(source_coordinates, dtype=float)
    receivers = np.asarray(receiver_coordinates, dtype=float)
    if theoretical.ndim != 3 or fixed.shape != theoretical.shape:
        raise ValueError("theoretical_center and fixed_mask must have shape [nref, ns, nr].")
    nref, nshot, nreceiver = theoretical.shape
    if observed.ndim != 3 or observed.shape[:2] != (nshot, nreceiver):
        raise ValueError("observed_data must have shape [ns, nr, nt].")
    if sources.shape != (nshot, 2) or receivers.shape != (nshot, nreceiver, 2):
        raise ValueError("source and receiver coordinates have an invalid shape.")
    if (
        isinstance(max_fit_anchors, bool)
        or int(max_fit_anchors) != max_fit_anchors
        or max_fit_anchors < 1
        or isinstance(min_anchors, bool)
        or int(min_anchors) != min_anchors
        or min_anchors < 3
    ):
        raise ValueError("anchor limits must be positive integers.")
    _validate_snap_args(dt, t0, envelope_snap_half_width_time)

    centers = np.full(theoretical.shape, np.nan, dtype=float)
    source = np.zeros(theoretical.shape, dtype=int)
    anchor_masks: list[list[np.ndarray]] = []
    for reflector in range(nref):
        row: list[np.ndarray] = []
        for shot in range(nshot):
            tracking = reference_tracking[reflector][shot]
            fixed_row = fixed[reflector, shot]
            shift = np.asarray(tracking.shift_time, dtype=float)
            success = np.asarray(tracking.success_mask, dtype=bool)
            if shift.shape != (nreceiver,) or success.shape != (nreceiver,):
                raise ValueError("tracking arrays must match the receiver count.")
            direct = (
                fixed_row
                & success
                & np.isfinite(shift)
                & np.isfinite(theoretical[reflector, shot])
            )
            centers[reflector, shot, direct] = (
                theoretical[reflector, shot, direct] + shift[direct]
            )
            source[reflector, shot, direct] = DIRECT_TRACKING
            row.append(
                select_observed_center_anchors(
                    fixed_row,
                    tracking,
                    correlation_threshold=anchor_corr_threshold,
                )
            )
        anchor_masks.append(row)

    counts = {
        "direct_anchors": 0,
        "recovered_edge_receivers": 0,
        "quadratic_plus_envelope": 0,
        "quadratic_only_fallback": 0,
        "unresolved_internal_gaps": 0,
        "recovery_skipped_insufficient_anchors": 0,
        "recovered_by_reflector": [0 for _ in range(nref)],
    }
    envelope_cache: dict[tuple[int, int], np.ndarray | None] = {}

    def get_envelope(shot: int, receiver: int) -> np.ndarray | None:
        key = (shot, receiver)
        if key not in envelope_cache:
            trace = observed[shot, receiver]
            if trace.ndim != 1 or not np.isfinite(trace).all():
                envelope_cache[key] = None
            else:
                try:
                    envelope_cache[key] = np.abs(hilbert(trace))
                except Exception:
                    envelope_cache[key] = None
        return envelope_cache[key]

    for reflector in range(nref):
        for shot in range(nshot):
            anchors = np.flatnonzero(anchor_masks[reflector][shot])
            counts["direct_anchors"] += int(anchors.size)
            unresolved = fixed[reflector, shot] & ~np.isfinite(centers[reflector, shot])
            if anchors.size:
                first = int(anchors[0])
                last = int(anchors[-1])
                counts["unresolved_internal_gaps"] += int(
                    np.count_nonzero(
                        unresolved
                        & (np.arange(nreceiver) > first)
                        & (np.arange(nreceiver) < last)
                    )
                )
            if anchors.size < min_anchors:
                counts["recovery_skipped_insufficient_anchors"] += 1
                continue

            index = np.arange(nreceiver)
            left_missing = np.flatnonzero(unresolved & (index < anchors[0]))
            right_missing = np.flatnonzero(unresolved & (index > anchors[-1]))
            for missing, edge in ((left_missing, "left"), (right_missing, "right")):
                if missing.size == 0:
                    continue
                fit_indices = (
                    anchors[anchors >= anchors[0]]
                    if edge == "left"
                    else anchors[anchors <= anchors[-1]]
                )
                fit_indices = (
                    fit_indices[:max_fit_anchors]
                    if edge == "left"
                    else fit_indices[-max_fit_anchors:]
                )
                shift = np.asarray(reference_tracking[reflector][shot].shift_time, dtype=float)
                correlation = np.asarray(
                    reference_tracking[reflector][shot].tracked_correlation,
                    dtype=float,
                )
                fit_centers = theoretical[reflector, shot, fit_indices] + shift[fit_indices]
                try:
                    coefficients = weighted_quadratic_fit(
                        receivers[shot, fit_indices, 0] - sources[shot, 0],
                        fit_centers,
                        correlation[fit_indices],
                    )
                except ValueError:
                    continue
                for receiver in missing:
                    receiver = int(receiver)
                    h = receivers[shot, receiver, 0] - sources[shot, 0]
                    predicted = float(np.array([1.0, h, h * h]) @ coefficients)
                    if not np.isfinite(predicted):
                        continue
                    envelope = get_envelope(shot, receiver)
                    snapped = False
                    final = predicted
                    if envelope is not None:
                        final, snapped = _snap_envelope(
                            predicted,
                            envelope,
                            dt=dt,
                            t0=t0,
                            half_width_time=envelope_snap_half_width_time,
                        )
                    centers[reflector, shot, receiver] = final
                    source[reflector, shot, receiver] = (
                        QUADRATIC_PLUS_ENVELOPE if snapped else QUADRATIC_ONLY
                    )
                    counts["recovered_edge_receivers"] += 1
                    counts[
                        "quadratic_plus_envelope" if snapped else "quadratic_only_fallback"
                    ] += 1
                    counts["recovered_by_reflector"][reflector] += 1

    counts["recovered_by_reflector"] = tuple(counts["recovered_by_reflector"])
    return ObservedCenterRecoveryResult(centers, source, MappingProxyType(counts))


__all__ = [
    "DIRECT_TRACKING",
    "QUADRATIC_ONLY",
    "QUADRATIC_PLUS_ENVELOPE",
    "ObservedCenterRecoveryResult",
    "recover_edge_observed_centers",
    "select_observed_center_anchors",
    "snap_observed_center",
    "weighted_quadratic_fit",
]
