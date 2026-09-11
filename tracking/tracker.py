"""Seed-centered bidirectional global ZNCC ridge tracking for WRTI Step 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class TrackingError(ValueError):
    """Raised when tracking inputs are inconsistent or invalid."""


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrackingResult:
    """Tracked reflection traveltime residual for one receiver axis.

    ``path_index`` is the selected integer lag column, or ``-1`` where the
    receiver row has no finite/valid path state.  ``shift_samples`` applies
    the optional local three-point parabolic refinement to that integer path.
    """

    path_index: np.ndarray
    shift_samples: np.ndarray
    shift_time: np.ndarray
    tracked_correlation: np.ndarray
    seed_receiver: int
    seed_lag: float
    success_mask: np.ndarray
    boundary_flag: np.ndarray
    dt: float
    epsilon_samples: int
    boundary_margin_samples: int

    def __post_init__(self) -> None:
        path_index = np.asarray(self.path_index, dtype=int)
        shift_samples = np.asarray(self.shift_samples, dtype=float)
        shift_time = np.asarray(self.shift_time, dtype=float)
        tracked_correlation = np.asarray(self.tracked_correlation, dtype=float)
        success_mask = np.asarray(self.success_mask, dtype=bool)
        boundary_flag = np.asarray(self.boundary_flag, dtype=bool)
        if path_index.ndim != 1:
            raise TrackingError("path_index must be a one-dimensional array.")
        shape = path_index.shape
        for name, value in {
            "shift_samples": shift_samples,
            "shift_time": shift_time,
            "tracked_correlation": tracked_correlation,
            "success_mask": success_mask,
            "boundary_flag": boundary_flag,
        }.items():
            if value.shape != shape:
                raise TrackingError(f"{name} must have shape {shape}.")
        if (
            isinstance(self.seed_receiver, bool)
            or int(self.seed_receiver) != self.seed_receiver
            or self.seed_receiver < 0
        ):
            raise TrackingError("seed_receiver must be a non-negative integer.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise TrackingError("dt must be finite and positive.")
        if (
            isinstance(self.epsilon_samples, bool)
            or int(self.epsilon_samples) != self.epsilon_samples
            or self.epsilon_samples < 0
        ):
            raise TrackingError("epsilon_samples must be a non-negative integer.")
        if (
            isinstance(self.boundary_margin_samples, bool)
            or int(self.boundary_margin_samples) != self.boundary_margin_samples
            or self.boundary_margin_samples < 0
        ):
            raise TrackingError(
                "boundary_margin_samples must be a non-negative integer."
            )
        if not np.isfinite(self.seed_lag) and not np.isnan(self.seed_lag):
            raise TrackingError("seed_lag must be finite or NaN when seeding fails.")

        object.__setattr__(self, "path_index", _readonly(path_index, int))
        object.__setattr__(self, "shift_samples", _readonly(shift_samples, float))
        object.__setattr__(self, "shift_time", _readonly(shift_time, float))
        object.__setattr__(
            self,
            "tracked_correlation",
            _readonly(tracked_correlation, float),
        )
        object.__setattr__(self, "success_mask", _readonly(success_mask, bool))
        object.__setattr__(self, "boundary_flag", _readonly(boundary_flag, bool))
        object.__setattr__(self, "seed_receiver", int(self.seed_receiver))
        object.__setattr__(self, "seed_lag", float(self.seed_lag))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "epsilon_samples", int(self.epsilon_samples))
        object.__setattr__(
            self,
            "boundary_margin_samples",
            int(self.boundary_margin_samples),
        )


def _validate_inputs(
    correlation: np.ndarray,
    lags_samples: np.ndarray,
    receiver_x: np.ndarray,
    source_x: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(correlation, dtype=float)
    lags = np.asarray(lags_samples, dtype=float)
    receivers = np.asarray(receiver_x, dtype=float)
    if values.ndim != 2:
        raise TrackingError("correlation must have shape [nreceiver, nlag].")
    if lags.ndim != 1 or lags.size != values.shape[1]:
        raise TrackingError("lags_samples must have length nlag.")
    if receivers.ndim != 1 or receivers.size != values.shape[0]:
        raise TrackingError("receiver_x must have length nreceiver.")
    if not np.isfinite(lags).all() or not np.isfinite(receivers).all():
        raise TrackingError("lags_samples and receiver_x must be finite.")
    if not np.allclose(lags, np.rint(lags)):
        raise TrackingError("lags_samples must contain integer sample lags.")
    lags = np.rint(lags).astype(int)
    if lags.size == 0 or not np.all(np.diff(lags) == 1):
        raise TrackingError("lags_samples must be contiguous and increasing by one.")
    if not np.isfinite(source_x):
        raise TrackingError("source_x must be finite.")
    if not np.isfinite(dt) or dt <= 0:
        raise TrackingError("dt must be finite and positive.")
    return values, lags, receivers


def _normalise_valid(
    correlation: np.ndarray,
    valid: np.ndarray | None,
) -> np.ndarray:
    finite = np.isfinite(correlation)
    if valid is None:
        return finite
    validity = np.asarray(valid, dtype=bool)
    if validity.shape == (correlation.shape[0],):
        validity = np.broadcast_to(validity[:, None], correlation.shape)
    elif validity.shape != correlation.shape:
        raise TrackingError(
            "valid must have shape [nreceiver] or [nreceiver, nlag]."
        )
    return finite & validity


def _normalise_nonnegative_time(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0:
        raise TrackingError(f"{name} must be finite and non-negative.")
    return float(value)


def _refine_peak(row: np.ndarray, peak_index: int, lags_samples: np.ndarray) -> float:
    """Return integer or locally refined lag in sample units."""

    integer_lag = float(lags_samples[peak_index])
    if peak_index == 0 or peak_index == row.size - 1:
        return integer_lag
    left = float(row[peak_index - 1])
    center = float(row[peak_index])
    right = float(row[peak_index + 1])
    if not np.isfinite(left) or not np.isfinite(center) or not np.isfinite(right):
        return integer_lag
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator == 0.0:
        return integer_lag
    delta = (left - right) / (2.0 * denominator)
    # A three-point correction is accepted only when it remains in the
    # local sample cell around the selected integer peak.
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return integer_lag
    return integer_lag + float(delta)


def _normalise_boundary_margin(
    dt: float,
    boundary_margin_samples: int,
    boundary_margin_time: float | None,
) -> int:
    if boundary_margin_time is not None:
        if boundary_margin_samples != 0:
            raise TrackingError(
                "Specify boundary_margin_time or boundary_margin_samples, not both."
            )
        value = _normalise_nonnegative_time(
            boundary_margin_time, "boundary_margin_time"
        )
        boundary_margin_samples = int(round(value / dt))
    if (
        isinstance(boundary_margin_samples, bool)
        or int(boundary_margin_samples) != boundary_margin_samples
        or boundary_margin_samples < 0
    ):
        raise TrackingError(
            "boundary_margin_samples must be a non-negative integer."
        )
    return int(boundary_margin_samples)


def _valid_receiver_segments(row_valid: np.ndarray) -> list[np.ndarray]:
    """Return contiguous receiver runs containing at least one valid lag."""

    has_state = np.any(row_valid, axis=1)
    segments: list[np.ndarray] = []
    start = None
    for receiver, present in enumerate(has_state):
        if present and start is None:
            start = receiver
        elif not present and start is not None:
            segments.append(np.arange(start, receiver, dtype=int))
            start = None
    if start is not None:
        segments.append(np.arange(start, has_state.size, dtype=int))
    return segments


def _directional_dp_from_seed(
    values: np.ndarray,
    row_valid: np.ndarray,
    receiver_order: np.ndarray,
    *,
    epsilon_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return best outward path metrics for every seed lag state.

    ``receiver_order`` starts at the seed and then moves monotonically away from
    it in one receiver direction.  The dynamic program is evaluated from the
    outermost receiver back toward the seed, so all seed lag states are solved
    simultaneously without enumerating seed states.

    For each state, path length is the primary objective and accumulated ZNCC
    is the secondary objective.  Therefore a path continues whenever the next
    receiver is reachable; a shorter path cannot win merely because extending
    through low/negative ZNCC would reduce its raw accumulated score.
    """

    order = np.asarray(receiver_order, dtype=int)
    if order.ndim != 1 or order.size == 0:
        raise TrackingError("receiver_order must be a non-empty 1-D array.")

    nrow = order.size
    nlag = values.shape[1]

    # DP scoring uses absolute ZNCC to allow polarity reversal.
    # Final reported correlation remains the original signed ZNCC.
    score_values = np.abs(values)
    # next_state[k, j] is the selected lag state at receiver_order[k + 1]
    # when receiver_order[k] uses state j.  -1 means that the path stops here.
    next_state = np.full((max(nrow - 1, 0), nlag), -1, dtype=int)

    outer_receiver = int(order[-1])
    outer_valid = row_valid[outer_receiver]
    coverage = np.zeros(nlag, dtype=int)
    score = np.full(nlag, -np.inf, dtype=float)
    coverage[outer_valid] = 1
    score[outer_valid] = score_values[outer_receiver, outer_valid]
    for local_row in range(nrow - 2, -1, -1):
        receiver = int(order[local_row])
        current_valid = row_valid[receiver]

        best_next_coverage = np.zeros(nlag, dtype=int)
        best_next_score = np.full(nlag, -np.inf, dtype=float)
        best_next_state = np.full(nlag, -1, dtype=int)

        # For each allowed lag offset, compare the already-solved outward path
        # at the neighboring receiver.  Coverage is compared first, then ZNCC.
        for offset in range(-epsilon_samples, epsilon_samples + 1):
            if offset < 0:
                current_index = np.arange(-offset, nlag, dtype=int)
                outward_index = current_index + offset
            elif offset > 0:
                current_index = np.arange(0, nlag - offset, dtype=int)
                outward_index = current_index + offset
            else:
                current_index = np.arange(nlag, dtype=int)
                outward_index = current_index

            candidate_coverage = coverage[outward_index]
            candidate_score = score[outward_index]
            usable = candidate_coverage > 0
            if not np.any(usable):
                continue

            target = current_index[usable]
            candidate_coverage = candidate_coverage[usable]
            candidate_score = candidate_score[usable]
            current_best_coverage = best_next_coverage[target]
            current_best_score = best_next_score[target]
            improve = (
                (candidate_coverage > current_best_coverage)
                | (
                    (candidate_coverage == current_best_coverage)
                    & (candidate_score > current_best_score)
                )
            )
            if np.any(improve):
                chosen = target[improve]
                best_next_coverage[chosen] = candidate_coverage[improve]
                best_next_score[chosen] = candidate_score[improve]
                best_next_state[chosen] = outward_index[usable][improve]

        current_coverage = np.zeros(nlag, dtype=int)
        current_score = np.full(nlag, -np.inf, dtype=float)

        # A valid current state always supports a length-one path.  If any next
        # state is reachable, coverage-first scoring forces continuation.
        can_continue = current_valid & (best_next_state >= 0)
        stop_here = current_valid & ~can_continue
        current_coverage[stop_here] = 1
        current_score[stop_here] = score_values[receiver, stop_here]
        current_coverage[can_continue] = 1 + best_next_coverage[can_continue]
        current_score[can_continue] = (
            score_values[receiver, can_continue]
            + best_next_score[can_continue]
        )
        next_state[local_row, can_continue] = best_next_state[can_continue]

        coverage = current_coverage
        score = current_score

    return coverage, score, next_state


def _trace_direction(
    receiver_order: np.ndarray,
    seed_state: int,
    next_state: np.ndarray,
) -> np.ndarray:
    """Reconstruct one seed-to-outward directional path."""

    order = np.asarray(receiver_order, dtype=int)
    path = np.full(order.size, -1, dtype=int)
    current = int(seed_state)
    for local_row in range(order.size):
        if current < 0:
            break
        path[local_row] = current
        if local_row == order.size - 1:
            break
        current = int(next_state[local_row, current])
    return path


def _global_dp_segment(
    values: np.ndarray,
    row_valid: np.ndarray,
    lags: np.ndarray,
    segment: np.ndarray,
    *,
    seed_receiver: int,
    epsilon_samples: int,
) -> np.ndarray:
    """Return a seed-centered, bidirectional integer-lag path.

    Both receiver directions are solved simultaneously for every valid seed lag
    without seed-state enumeration.  The final shared seed state first maximizes
    total receiver coverage and then accumulated waveform ZNCC, with the seed
    contribution counted once.  A direction stops at its first unreachable
    receiver and never restarts beyond it.
    """

    del lags  # Lag values are not needed once epsilon is expressed in samples.
    path = np.full(segment.size, -1, dtype=int)
    seed_location = np.flatnonzero(segment == seed_receiver)
    if seed_location.size == 0:
        return path
    seed_local = int(seed_location[0])
    seed_valid = np.asarray(row_valid[seed_receiver], dtype=bool)
    if not np.any(seed_valid):
        return path

    left_order = segment[seed_local::-1]
    right_order = segment[seed_local:]
    left_coverage, left_score, left_next = _directional_dp_from_seed(
        values,
        row_valid,
        left_order,
        epsilon_samples=epsilon_samples,
    )
    right_coverage, right_score, right_next = _directional_dp_from_seed(
        values,
        row_valid,
        right_order,
        epsilon_samples=epsilon_samples,
    )

    seed_value = values[seed_receiver]
    total_coverage = left_coverage + right_coverage - 1
    total_score = left_score + right_score - seed_value
    candidates = seed_valid & (total_coverage > 0) & np.isfinite(total_score)
    if not np.any(candidates):
        return path

    candidate_states = np.flatnonzero(candidates)
    best_coverage = int(np.max(total_coverage[candidate_states]))
    longest_states = candidate_states[
        total_coverage[candidate_states] == best_coverage
    ]
    best_seed_state = int(
        longest_states[np.argmax(total_score[longest_states])]
    )

    left_path = _trace_direction(left_order, best_seed_state, left_next)
    right_path = _trace_direction(right_order, best_seed_state, right_next)

    for receiver, state in zip(left_order, left_path):
        if state >= 0:
            path[int(receiver - segment[0])] = state
    for receiver, state in zip(right_order[1:], right_path[1:]):
        if state >= 0:
            path[int(receiver - segment[0])] = state
    return path

def _assert_path_continuity(
    path_index: np.ndarray,
    lags: np.ndarray,
    epsilon_samples: int,
) -> None:
    """Raise if adjacent successful waveform-DP states break continuity."""

    path = np.asarray(path_index, dtype=int)
    adjacent = (path[:-1] >= 0) & (path[1:] >= 0)
    if not np.any(adjacent):
        return
    left = lags[path[:-1][adjacent]]
    right = lags[path[1:][adjacent]]
    jumps = np.abs(right - left)
    if np.any(jumps > int(epsilon_samples)):
        location = int(np.flatnonzero(adjacent)[np.argmax(jumps)])
        raise AssertionError(
            "Waveform DP violated its hard receiver-to-receiver continuity "
            f"bound at receivers {location}->{location + 1}: "
            f"jump={int(np.max(jumps))} samples, epsilon={epsilon_samples}."
        )


def track_zncc(
    correlation: np.ndarray,
    lags_samples: np.ndarray,
    receiver_x: np.ndarray,
    source_x: float,
    seed_lag_range_time: float,
    epsilon_time: float,
    dt: float,
    *,
    measurement_correlation: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    lag_valid: np.ndarray | None = None,
    min_correlation: float | None = None,
    boundary_margin_samples: int = 0,
    boundary_margin_time: float | None = None,
    raw_refine_radius_samples: int = 1,
) -> TrackingResult:
    """Track the best seed-centered bidirectional receiver-lag ZNCC ridge.

    ``correlation`` identifies a guide ridge.  ``measurement_correlation``
    defaults to it for compatibility, otherwise a second raw DP is constrained
    to one small corridor around the guide before reporting the final lag.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise TrackingError("dt must be finite and positive.")
    values, lags, receivers = _validate_inputs(
        correlation, lags_samples, receiver_x, source_x, dt
    )
    measurement = values if measurement_correlation is None else np.asarray(
        measurement_correlation, dtype=float
    )
    if measurement.shape != values.shape:
        raise TrackingError("measurement_correlation must match correlation shape.")
    row_valid = _normalise_valid(values, valid)
    # Step 4 can invalidate individual lag states without invalidating the
    # receiver row.  Intersect before segmentation; the DP recurrence is
    # otherwise unchanged.
    if lag_valid is not None:
        row_valid &= _normalise_valid(values, lag_valid)
    _normalise_nonnegative_time(
        seed_lag_range_time, "seed_lag_range_time"
    )
    epsilon = _normalise_nonnegative_time(epsilon_time, "epsilon_time")
    epsilon_samples = int(round(epsilon / float(dt)))
    margin_samples = _normalise_boundary_margin(
        float(dt), boundary_margin_samples, boundary_margin_time
    )
    if (
        isinstance(raw_refine_radius_samples, bool)
        or int(raw_refine_radius_samples) != raw_refine_radius_samples
        or raw_refine_radius_samples < 0
    ):
        raise TrackingError("raw_refine_radius_samples must be a non-negative integer.")
    raw_refine_radius_samples = int(raw_refine_radius_samples)
    if min_correlation is not None:
        if not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0:
            raise TrackingError("min_correlation must be in [-1, 1] or None.")
        min_correlation = float(min_correlation)

    n_receiver = values.shape[0]
    seed_receiver = int(np.argmin(np.abs(receivers - float(source_x))))
    guide_path_index = np.full(n_receiver, -1, dtype=int)
    shift_samples = np.full(n_receiver, np.nan, dtype=float)
    shift_time = np.full(n_receiver, np.nan, dtype=float)
    tracked_correlation = np.full(n_receiver, np.nan, dtype=float)
    success_mask = np.zeros(n_receiver, dtype=bool)
    boundary_flag = np.zeros(n_receiver, dtype=bool)
    seed_lag = float("nan")

    max_abs_lag = float(np.max(np.abs(lags)))
    boundary_limit = max_abs_lag - float(margin_samples)
    for segment in _valid_receiver_segments(row_valid):
        if not np.any(segment == seed_receiver):
            continue
        segment_path = _global_dp_segment(
            values,
            row_valid,
            lags,
            segment,
            seed_receiver=seed_receiver,
            epsilon_samples=epsilon_samples,
        )
        selected = segment_path >= 0
        guide_path_index[segment[selected]] = segment_path[selected]

    raw_valid = _normalise_valid(measurement, valid)
    if lag_valid is not None:
        raw_valid &= _normalise_valid(measurement, lag_valid)
    raw_corridor_valid = np.zeros_like(raw_valid, dtype=bool)
    guide_success = np.flatnonzero(guide_path_index >= 0)
    if guide_success.size:
        offsets = np.arange(
            -raw_refine_radius_samples,
            raw_refine_radius_samples + 1,
            dtype=int,
        )
        columns = guide_path_index[guide_success, None] + offsets
        inside = (columns >= 0) & (columns < lags.size)
        rows = np.broadcast_to(guide_success[:, None], columns.shape)
        raw_corridor_valid[rows[inside], columns[inside]] = True
    raw_corridor_valid &= raw_valid

    path_index = np.full(n_receiver, -1, dtype=int)
    for segment in _valid_receiver_segments(raw_corridor_valid):
        if not np.any(segment == seed_receiver):
            continue
        segment_path = _global_dp_segment(
            measurement,
            raw_corridor_valid,
            lags,
            segment,
            seed_receiver=seed_receiver,
            epsilon_samples=epsilon_samples,
        )
        selected = segment_path >= 0
        path_index[segment[selected]] = segment_path[selected]

    success_mask = path_index >= 0
    _assert_path_continuity(path_index, lags, epsilon_samples)
    for receiver in np.flatnonzero(success_mask):
        index = int(path_index[receiver])
        safe_measurement = np.where(
            raw_corridor_valid[receiver], measurement[receiver], np.nan
        )
        shift_samples[receiver] = _refine_peak(safe_measurement, index, lags)
        shift_time[receiver] = shift_samples[receiver] * float(dt)
        tracked_correlation[receiver] = measurement[receiver, index]
        boundary_flag[receiver] = abs(float(lags[index])) >= boundary_limit

    if success_mask[seed_receiver]:
        seed_lag = float(lags[path_index[seed_receiver]])

    return TrackingResult(
        path_index=path_index,
        shift_samples=shift_samples,
        shift_time=shift_time,
        tracked_correlation=tracked_correlation,
        seed_receiver=seed_receiver,
        seed_lag=seed_lag,
        success_mask=success_mask,
        boundary_flag=boundary_flag,
        dt=dt,
        epsilon_samples=epsilon_samples,
        boundary_margin_samples=margin_samples,
    )


def track_correlation_result(
    correlation_result,
    receiver_x: np.ndarray,
    source_x: float,
    seed_lag_range_time: float,
    epsilon_time: float,
    *,
    valid: np.ndarray | None = None,
    lag_valid: np.ndarray | None = None,
    min_correlation: float | None = None,
    boundary_margin_samples: int = 0,
    boundary_margin_time: float | None = None,
    raw_refine_radius_samples: int = 1,
) -> TrackingResult:
    """Track a Step 4 ``CorrelationResult`` without copying its matrix."""

    result_valid = correlation_result.valid if valid is None else valid
    result_lag_valid = (
        getattr(correlation_result, "lag_valid", None)
        if lag_valid is None
        else lag_valid
    )
    tracking_values = getattr(correlation_result, "tracking_correlation", None)
    if tracking_values is None:
        tracking_values = correlation_result.correlation
    raw_values = getattr(correlation_result, "waveform_correlation", None)
    if raw_values is None:
        raw_values = correlation_result.correlation
    return track_zncc(
        tracking_values,
        correlation_result.lags_samples,
        receiver_x,
        source_x,
        seed_lag_range_time,
        epsilon_time,
        correlation_result.dt,
        measurement_correlation=raw_values,
        valid=result_valid,
        lag_valid=result_lag_valid,
        min_correlation=min_correlation,
        boundary_margin_samples=boundary_margin_samples,
        boundary_margin_time=boundary_margin_time,
        raw_refine_radius_samples=raw_refine_radius_samples,
    )
