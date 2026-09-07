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
    dp_success_mask: np.ndarray | None = None
    bridge_success_mask: np.ndarray | None = None
    candidate_count: np.ndarray | None = None
    coarse_deviation_samples: np.ndarray | None = None
    tracked_strength: np.ndarray | None = None
    tracked_polarity: np.ndarray | None = None
    component_id: np.ndarray | None = None
    provenance: np.ndarray | None = None
    predicted_lag: np.ndarray | None = None
    ambiguous_mask: np.ndarray | None = None
    gap_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        path_index = np.asarray(self.path_index, dtype=int)
        shift_samples = np.asarray(self.shift_samples, dtype=float)
        shift_time = np.asarray(self.shift_time, dtype=float)
        tracked_correlation = np.asarray(self.tracked_correlation, dtype=float)
        success_mask = np.asarray(self.success_mask, dtype=bool)
        boundary_flag = np.asarray(self.boundary_flag, dtype=bool)
        bridge_success = (
            np.zeros(success_mask.shape, dtype=bool)
            if self.bridge_success_mask is None
            else np.asarray(self.bridge_success_mask, dtype=bool)
        )
        if bridge_success.shape != success_mask.shape:
            raise TrackingError(
                "bridge_success_mask must have the same shape as success_mask."
            )
        dp_success = (
            success_mask & ~bridge_success
            if self.dp_success_mask is None
            else np.asarray(self.dp_success_mask, dtype=bool)
        )
        candidate_count = (
            np.zeros(success_mask.shape, dtype=int)
            if self.candidate_count is None
            else np.asarray(self.candidate_count, dtype=int)
        )
        coarse_deviation = (
            np.full(success_mask.shape, np.nan, dtype=float)
            if self.coarse_deviation_samples is None
            else np.asarray(self.coarse_deviation_samples, dtype=float)
        )
        strength = (
            tracked_correlation
            if self.tracked_strength is None
            else np.asarray(self.tracked_strength, dtype=float)
        )
        polarity = (
            np.zeros(success_mask.shape, dtype=int)
            if self.tracked_polarity is None
            else np.asarray(self.tracked_polarity, dtype=int)
        )
        component_id = (
            np.full(success_mask.shape, -1, dtype=int)
            if self.component_id is None
            else np.asarray(self.component_id, dtype=int)
        )
        provenance = (
            np.where(success_mask, "PRIMARY_DP", "NO_MEASUREMENT")
            if self.provenance is None
            else np.asarray(self.provenance, dtype=str)
        )
        predicted_lag = (
            np.full(success_mask.shape, np.nan, dtype=float)
            if self.predicted_lag is None else np.asarray(self.predicted_lag, dtype=float)
        )
        ambiguous = (
            np.zeros(success_mask.shape, dtype=bool)
            if self.ambiguous_mask is None else np.asarray(self.ambiguous_mask, dtype=bool)
        )
        gap_count = (
            np.zeros(success_mask.shape, dtype=int)
            if self.gap_count is None else np.asarray(self.gap_count, dtype=int)
        )
        if path_index.ndim != 1:
            raise TrackingError("path_index must be a one-dimensional array.")
        shape = path_index.shape
        for name, value in {
            "shift_samples": shift_samples,
            "shift_time": shift_time,
            "tracked_correlation": tracked_correlation,
            "success_mask": success_mask,
            "boundary_flag": boundary_flag,
            "dp_success_mask": dp_success,
            "bridge_success_mask": bridge_success,
            "candidate_count": candidate_count,
            "coarse_deviation_samples": coarse_deviation,
            "tracked_strength": strength,
            "tracked_polarity": polarity,
            "component_id": component_id,
            "provenance": provenance,
            "predicted_lag": predicted_lag,
            "ambiguous_mask": ambiguous,
            "gap_count": gap_count,
        }.items():
            if value.shape != shape:
                raise TrackingError(f"{name} must have shape {shape}.")
        if np.any(dp_success & bridge_success):
            raise TrackingError("DP and bridge provenance masks must not overlap.")
        if not np.array_equal(success_mask, dp_success | bridge_success):
            raise TrackingError(
                "success_mask must equal dp_success_mask | bridge_success_mask."
            )
        if np.any(candidate_count < 0):
            raise TrackingError("candidate_count must be non-negative.")
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
        object.__setattr__(self, "dp_success_mask", _readonly(dp_success, bool))
        object.__setattr__(self, "bridge_success_mask", _readonly(bridge_success, bool))
        object.__setattr__(self, "candidate_count", _readonly(candidate_count, int))
        object.__setattr__(
            self,
            "coarse_deviation_samples",
            _readonly(coarse_deviation, float),
        )
        object.__setattr__(self, "tracked_strength", _readonly(strength, float))
        object.__setattr__(self, "tracked_polarity", _readonly(polarity, int))
        object.__setattr__(self, "component_id", _readonly(component_id, int))
        object.__setattr__(self, "provenance", _readonly(provenance, str))
        object.__setattr__(self, "predicted_lag", _readonly(predicted_lag, float))
        object.__setattr__(self, "ambiguous_mask", _readonly(ambiguous, bool))
        object.__setattr__(self, "gap_count", _readonly(gap_count, int))
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
    lags: np.ndarray,
    coarse_lag_samples: np.ndarray,
    *,
    epsilon_samples: int,
    smooth_weight: float = 0.0,
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
    # next_state[k, j] is the selected lag state at receiver_order[k + 1]
    # when receiver_order[k] uses state j.  -1 means that the path stops here.
    next_state = np.full((max(nrow - 1, 0), nlag), -1, dtype=int)

    outer_receiver = int(order[-1])
    outer_valid = row_valid[outer_receiver]
    coverage = np.zeros(nlag, dtype=int)
    score = np.full(nlag, -np.inf, dtype=float)
    coverage[outer_valid] = 1
    score[outer_valid] = values[outer_receiver, outer_valid]

    for local_row in range(nrow - 2, -1, -1):
        receiver = int(order[local_row])
        current_valid = row_valid[receiver]

        best_next_coverage = np.zeros(nlag, dtype=int)
        best_next_score = np.full(nlag, -np.inf, dtype=float)
        best_next_state = np.full(nlag, -1, dtype=int)

        # Main tracking rows contain only Top-K candidates, so compare sparse
        # state pairs directly.  This keeps the recurrence near O(nr * K^2).
        outward_states = np.flatnonzero(coverage > 0)
        outward_receiver = int(order[local_row + 1])
        for current_state in np.flatnonzero(current_valid):
            if np.isfinite(coarse_lag_samples[receiver]) and np.isfinite(coarse_lag_samples[outward_receiver]):
                distance = np.abs(
                    (lags[outward_states] - coarse_lag_samples[outward_receiver])
                    - (lags[current_state] - coarse_lag_samples[receiver])
                )
            else:
                distance = np.abs(lags[outward_states] - lags[current_state])
            legal = outward_states[distance <= int(epsilon_samples)]
            if legal.size == 0:
                continue
            legal_coverage = coverage[legal]
            longest = legal[legal_coverage == np.max(legal_coverage)]
            if np.isfinite(coarse_lag_samples[receiver]) and np.isfinite(coarse_lag_samples[outward_receiver]):
                transition_distance = np.abs((lags[longest] - coarse_lag_samples[outward_receiver]) - (lags[current_state] - coarse_lag_samples[receiver]))
            else:
                transition_distance = np.abs(lags[longest] - lags[current_state])
            legal_score = score[longest] - float(smooth_weight) * transition_distance
            best = int(longest[np.argmax(legal_score)])
            best_next_coverage[current_state] = coverage[best]
            best_next_score[current_state] = (
                score[best] - float(smooth_weight) * float(transition_distance[np.flatnonzero(longest == best)[0]])
            )
            best_next_state[current_state] = best

        current_coverage = np.zeros(nlag, dtype=int)
        current_score = np.full(nlag, -np.inf, dtype=float)

        # A valid current state always supports a length-one path.  If any next
        # state is reachable, coverage-first scoring forces continuation.
        can_continue = current_valid & (best_next_state >= 0)
        stop_here = current_valid & ~can_continue
        current_coverage[stop_here] = 1
        current_score[stop_here] = values[receiver, stop_here]
        current_coverage[can_continue] = 1 + best_next_coverage[can_continue]
        current_score[can_continue] = (
            values[receiver, can_continue] + best_next_score[can_continue]
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
    coarse_lag_samples: np.ndarray,
    epsilon_samples: int,
    smooth_weight: float = 0.0,
) -> np.ndarray:
    """Return a seed-centered, bidirectional integer-lag path.

    Both receiver directions are solved simultaneously for every valid seed lag
    without seed-state enumeration.  The final shared seed state first maximizes
    total receiver coverage and then accumulated waveform ZNCC, with the seed
    contribution counted once.  A direction stops at its first unreachable
    receiver and never restarts beyond it.
    """

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
        lags,
        coarse_lag_samples,
        epsilon_samples=epsilon_samples,
        smooth_weight=smooth_weight,
    )
    right_coverage, right_score, right_next = _directional_dp_from_seed(
        values,
        row_valid,
        right_order,
        lags,
        coarse_lag_samples,
        epsilon_samples=epsilon_samples,
        smooth_weight=smooth_weight,
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
    coarse_lag_samples: np.ndarray | None = None,
    component_id: np.ndarray | None = None,
) -> None:
    """Raise if adjacent successful waveform-DP states break continuity."""

    path = np.asarray(path_index, dtype=int)
    adjacent = (path[:-1] >= 0) & (path[1:] >= 0)
    if component_id is not None:
        component = np.asarray(component_id, dtype=int)
        if component.shape != path.shape:
            raise ValueError("component_id must match path_index shape.")
        adjacent &= (
            (component[:-1] >= 0)
            & (component[:-1] == component[1:])
        )
    if not np.any(adjacent):
        return
    left = lags[path[:-1][adjacent]]
    right = lags[path[1:][adjacent]]
    if coarse_lag_samples is not None:
        coarse = np.asarray(coarse_lag_samples, dtype=float)
        rows = np.flatnonzero(adjacent)
        use_residual = np.isfinite(coarse[rows]) & np.isfinite(coarse[rows + 1])
        jumps = np.abs(right - left).astype(float)
        jumps[use_residual] = np.abs(
            (right[use_residual] - coarse[rows[use_residual] + 1])
            - (left[use_residual] - coarse[rows[use_residual]])
        )
    else:
        jumps = np.abs(right - left)
    if np.any(jumps > int(epsilon_samples)):
        location = int(np.flatnonzero(adjacent)[np.argmax(jumps)])
        raise AssertionError(
            "Waveform DP violated its hard receiver-to-receiver continuity "
            f"bound at receivers {location}->{location + 1}: "
            f"jump={int(np.max(jumps))} samples, epsilon={epsilon_samples}."
        )


def _top_k_peak_mask(
    values: np.ndarray,
    base_valid: np.ndarray,
    lags: np.ndarray,
    *,
    top_k: int,
    min_separation_samples: int,
    coarse_lag_samples: np.ndarray,
) -> np.ndarray:
    """Compress each raw ZNCC row to a few meaningful candidate states."""

    candidates = np.zeros(base_valid.shape, dtype=bool)
    for receiver in range(values.shape[0]):
        valid_indices = np.flatnonzero(base_valid[receiver])
        if valid_indices.size == 0:
            continue
        row = values[receiver]
        left = np.full(row.shape, -np.inf, dtype=float)
        right = np.full(row.shape, -np.inf, dtype=float)
        left[1:] = np.where(base_valid[receiver, :-1], row[:-1], -np.inf)
        right[:-1] = np.where(base_valid[receiver, 1:], row[1:], -np.inf)
        left[0] = row[0]
        right[-1] = row[-1]
        peaks = valid_indices[
            (row[valid_indices] >= left[valid_indices])
            & (row[valid_indices] >= right[valid_indices])
            & (
                (row[valid_indices] > left[valid_indices])
                | (row[valid_indices] > right[valid_indices])
            )
        ]
        ranked = list(peaks[np.argsort(-row[peaks], kind="stable")])
        ranked.extend(
            int(index)
            for index in valid_indices[np.argsort(-row[valid_indices], kind="stable")]
            if int(index) not in ranked
        )

        coarse = float(coarse_lag_samples[receiver])
        if np.isfinite(coarse):
            pool = peaks if peaks.size else valid_indices
            prior_index = int(pool[np.argmin(np.abs(lags[pool] - coarse))])
            ranked = [prior_index] + [index for index in ranked if index != prior_index]

        selected: list[int] = []
        for index in ranked:
            index = int(index)
            if all(
                abs(int(lags[index]) - int(lags[other]))
                >= int(min_separation_samples)
                for other in selected
            ):
                selected.append(index)
            if len(selected) == int(top_k):
                break
        candidates[receiver, selected] = True
    return candidates


def _candidate_segments(
    state_valid: np.ndarray,
    lags: np.ndarray,
    max_jump_samples: int,
    coarse_lag_samples: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Split sparse candidate rows where no legal transition exists."""

    segments: list[np.ndarray] = []
    start: int | None = None
    previous_states = np.empty(0, dtype=int)
    for receiver in range(state_valid.shape[0]):
        states = np.flatnonzero(state_valid[receiver])
        if (
            coarse_lag_samples is not None
            and receiver > 0
            and np.isfinite(coarse_lag_samples[receiver])
            and np.isfinite(coarse_lag_samples[receiver - 1])
        ):
            current_lags = lags[states] - coarse_lag_samples[receiver]
            previous_lags = (
                lags[previous_states] - coarse_lag_samples[receiver - 1]
            )
        else:
            current_lags = lags[states]
            previous_lags = lags[previous_states]
        connected = bool(
            states.size
            and previous_states.size
            and np.any(
                np.abs(current_lags[:, None] - previous_lags[None, :])
                <= int(max_jump_samples)
            )
        )
        if states.size and (start is None or connected):
            if start is None:
                start = receiver
        else:
            if start is not None:
                segments.append(np.arange(start, receiver, dtype=int))
            start = receiver if states.size else None
        previous_states = states
    if start is not None:
        segments.append(np.arange(start, state_valid.shape[0], dtype=int))
    return segments


def _fixed_end_path(
    values: np.ndarray,
    state_valid: np.ndarray,
    lags: np.ndarray,
    rows: np.ndarray,
    start_state: int,
    end_state: int,
    *,
    coarse_lag_samples: np.ndarray | None,
    max_jump_samples: int,
    smooth_weight: float,
) -> np.ndarray:
    """Return a full raw-ZNCC path between two fixed anchor states."""

    path = np.full(rows.size, -1, dtype=int)
    score = np.full(values.shape[1], -np.inf, dtype=float)
    score[int(start_state)] = values[int(rows[0]), int(start_state)]
    previous = np.full((rows.size, values.shape[1]), -1, dtype=int)
    for local_row in range(1, rows.size):
        receiver = int(rows[local_row])
        current_score = np.full(values.shape[1], -np.inf, dtype=float)
        current_states = np.flatnonzero(state_valid[local_row])
        if local_row == rows.size - 1:
            current_states = current_states[current_states == int(end_state)]
        previous_states = np.flatnonzero(np.isfinite(score))
        for state in current_states:
            previous_receiver = int(rows[local_row - 1])
            if (
                coarse_lag_samples is not None
                and np.isfinite(coarse_lag_samples[receiver])
                and np.isfinite(coarse_lag_samples[previous_receiver])
            ):
                jumps = np.abs(
                    (lags[previous_states] - coarse_lag_samples[previous_receiver])
                    - (lags[state] - coarse_lag_samples[receiver])
                )
            else:
                jumps = np.abs(lags[previous_states] - lags[state])
            legal = previous_states[jumps <= int(max_jump_samples)]
            if legal.size == 0:
                continue
            transition_jump = jumps[np.isin(previous_states, legal)]
            transition = score[legal] - float(smooth_weight) * transition_jump
            best = int(legal[np.argmax(transition)])
            current_score[state] = values[receiver, state] + transition[
                np.argmax(transition)
            ]
            previous[local_row, state] = best
        score = current_score
    if not np.isfinite(score[int(end_state)]):
        return path
    state = int(end_state)
    for local_row in range(rows.size - 1, -1, -1):
        path[local_row] = state
        if local_row:
            state = int(previous[local_row, state])
    return path


def _bridge_between(
    values: np.ndarray,
    base_valid: np.ndarray,
    lags: np.ndarray,
    receiver_x: np.ndarray,
    coarse_lag_samples: np.ndarray | None,
    left_receiver: int,
    left_state: int,
    right_receiver: int,
    right_state: int,
    *,
    half_width_samples: int,
    max_width_samples: int,
    max_jump_samples: int,
    max_receivers: int,
    max_distance: float | None,
    smooth_weight: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    gap = int(right_receiver) - int(left_receiver) - 1
    if gap < 1 or gap > int(max_receivers):
        return None
    if max_distance is not None and (
        abs(receiver_x[right_receiver] - receiver_x[left_receiver])
        > float(max_distance)
    ):
        return None
    rows = np.arange(left_receiver, right_receiver + 1, dtype=int)
    centers = np.interp(
        receiver_x[rows],
        [receiver_x[left_receiver], receiver_x[right_receiver]],
        [lags[left_state], lags[right_state]],
    )
    width = min(
        int(max_width_samples),
        int(half_width_samples) + max(gap - 1, 0),
    )
    corridor = base_valid[rows] & (
        np.abs(lags[None, :] - centers[:, None]) <= width
    )
    corridor[0] = False
    corridor[0, left_state] = True
    corridor[-1] = False
    corridor[-1, right_state] = True
    local_path = _fixed_end_path(
        values,
        corridor,
        lags,
        rows,
        left_state,
        right_state,
        coarse_lag_samples=coarse_lag_samples,
        max_jump_samples=max_jump_samples,
        smooth_weight=smooth_weight,
    )
    if np.any(local_path < 0):
        return None
    return rows, local_path


def _bridge_one_side(
    values: np.ndarray,
    base_valid: np.ndarray,
    lags: np.ndarray,
    receiver_x: np.ndarray,
    coarse_lag_samples: np.ndarray | None,
    path_index: np.ndarray,
    direction: int,
    *,
    half_width_samples: int,
    max_width_samples: int,
    max_jump_samples: int,
    max_receivers: int,
    max_distance: float | None,
    smooth_weight: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    success = np.flatnonzero(path_index >= 0)
    if success.size == 0 or max_receivers == 0:
        return None
    anchor = int(success[-1] if direction > 0 else success[0])
    available = (
        values.shape[0] - anchor - 1 if direction > 0 else anchor
    )
    count = min(int(max_receivers), available)
    if count == 0:
        return None
    order = anchor + int(direction) * np.arange(count + 1, dtype=int)
    if max_distance is not None:
        distance = np.abs(receiver_x[order] - receiver_x[anchor])
        order = order[distance <= float(max_distance)]
    if order.size < 2:
        return None

    inward = success[-3:] if direction > 0 else success[:3][::-1]
    history_lags = lags[path_index[inward]].astype(float)
    if (
        coarse_lag_samples is not None
        and np.isfinite(coarse_lag_samples[inward]).all()
    ):
        history_lags -= coarse_lag_samples[inward]
    slope = float(np.median(np.diff(history_lags))) if history_lags.size > 1 else 0.0
    corridor = np.zeros_like(base_valid, dtype=bool)
    corridor[anchor, path_index[anchor]] = True
    for step, receiver in enumerate(order[1:], start=1):
        width = min(int(max_width_samples), int(half_width_samples) + step - 1)
        predicted = float(lags[path_index[anchor]]) + slope * step
        if (
            coarse_lag_samples is not None
            and np.isfinite(coarse_lag_samples[anchor])
            and np.isfinite(coarse_lag_samples[receiver])
        ):
            predicted += (
                coarse_lag_samples[receiver] - coarse_lag_samples[anchor]
            )
        corridor[receiver] = base_valid[receiver] & (
            np.abs(lags.astype(float) - predicted) <= width
        )

        coverage, _, next_state = _directional_dp_from_seed(
            values,
            corridor,
            order,
            lags,
            (
                np.full(values.shape[0], np.nan, dtype=float)
                if coarse_lag_samples is None
                else coarse_lag_samples
            ),
            epsilon_samples=max_jump_samples,
            smooth_weight=smooth_weight,
    )
    anchor_state = int(path_index[anchor])
    if coverage[anchor_state] <= 1:
        return None
    local_path = _trace_direction(order, anchor_state, next_state)
    recovered = local_path >= 0
    return order[recovered][1:], local_path[recovered][1:]


def _dense_direction(
    score_values: np.ndarray,
    state_valid: np.ndarray,
    lags: np.ndarray,
    receiver_x: np.ndarray,
    coarse: np.ndarray,
    seed_receiver: int,
    seed_state: int,
    direction: int,
    *,
    max_residual_jump_samples: int,
    max_skip_rows: int,
    smooth_weight: float,
    gap_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anchored dense Viterbi for one receiver direction."""
    nr, nlag = score_values.shape
    value = np.full((nr, nlag), -np.inf)
    coverage = np.zeros((nr, nlag), dtype=int)
    previous_row = np.full((nr, nlag), -1, dtype=int)
    previous_state = np.full((nr, nlag), -1, dtype=int)
    value[seed_receiver, seed_state] = score_values[seed_receiver, seed_state]
    coverage[seed_receiver, seed_state] = 1
    spacings = np.abs(np.diff(receiver_x))
    nominal_dx = float(np.median(spacings[spacings > 0])) if np.any(spacings > 0) else 1.0
    rows = range(seed_receiver + direction, nr if direction > 0 else -1, direction)
    for row in rows:
        for state in np.flatnonzero(state_valid[row]):
            best_coverage = 0
            best_value = -np.inf
            best_row = -1
            best_state = -1
            for distance in range(1, max_skip_rows + 2):
                prior = row - direction * distance
                if not 0 <= prior < nr:
                    continue
                if np.isfinite(coarse[row]) and np.isfinite(coarse[prior]):
                    target = float(lags[state] - coarse[row] + coarse[prior])
                else:
                    target = float(lags[state])
                physical_scale = max(abs(receiver_x[row] - receiver_x[prior]) / nominal_dx, 1.0)
                width = int(np.ceil(max_residual_jump_samples * physical_scale))
                lower = int(np.searchsorted(lags, target - width, side="left"))
                upper = int(np.searchsorted(lags, target + width, side="right"))
                candidates = np.flatnonzero(np.isfinite(value[prior, lower:upper])) + lower
                if candidates.size == 0:
                    continue
                if np.isfinite(coarse[row]) and np.isfinite(coarse[prior]):
                    jump = np.abs((lags[state] - coarse[row]) - (lags[candidates] - coarse[prior]))
                else:
                    jump = np.abs(lags[state] - lags[candidates])
                trial_coverage = coverage[prior, candidates] + 1
                trial_value = value[prior, candidates] - smooth_weight * jump - gap_penalty * (distance - 1)
                longest = np.flatnonzero(trial_coverage == np.max(trial_coverage))
                candidate = int(longest[np.argmax(trial_value[longest])])
                if trial_coverage[candidate] > best_coverage or (
                    trial_coverage[candidate] == best_coverage and trial_value[candidate] > best_value
                ):
                    best_coverage = int(trial_coverage[candidate])
                    best_value = float(trial_value[candidate])
                    best_row = prior
                    best_state = int(candidates[candidate])
            if best_state >= 0:
                coverage[row, state] = best_coverage
                value[row, state] = score_values[row, state] + best_value
                previous_row[row, state] = best_row
                previous_state[row, state] = best_state
    available = np.flatnonzero(coverage.max(axis=1) > 0)
    terminal = seed_receiver if available.size == 1 else int(available[-1] if direction > 0 else available[0])
    states = np.flatnonzero(coverage[terminal] > 0)
    if states.size == 0:
        return np.full(nr, -1, dtype=int), value, previous_row, previous_state
    longest = states[coverage[terminal, states] == np.max(coverage[terminal, states])]
    state = int(longest[np.argmax(value[terminal, longest])])
    path = np.full(nr, -1, dtype=int)
    row = terminal
    while row >= 0:
        path[row] = state
        if row == seed_receiver:
            break
        next_row, next_state = previous_row[row, state], previous_state[row, state]
        if next_row < 0 or next_state < 0:
            break
        row, state = int(next_row), int(next_state)
    return path, value, previous_row, previous_state


def _dense_track_zncc(
    correlation: np.ndarray, lags_samples: np.ndarray, receiver_x: np.ndarray, source_x: float,
    seed_lag_range_time: float, epsilon_time: float, dt: float, *, measurement_correlation=None,
    valid=None, lag_valid=None, min_correlation=None, boundary_margin_samples=0,
    boundary_margin_time=None, coarse_lag_samples=None, coarse_soft_width_time=0.04,
    coarse_soft_weight=0.05, coarse_soft_penalty_cap=0.25, smooth_weight=0.02,
    max_residual_jump_time=None, max_skip_rows=3, gap_penalty=0.05,
    correlation_mode="auto_polarity", restart_enabled=True, restart_confirm_rows=4,
    restart_min_mean_correlation=0.55, restart_max_coarse_deviation_time=0.10,
) -> TrackingResult:
    values, lags, receivers = _validate_inputs(correlation, lags_samples, receiver_x, source_x, dt)
    raw = values if measurement_correlation is None else np.asarray(measurement_correlation, dtype=float)
    if raw.shape != values.shape:
        raise TrackingError("measurement_correlation must match correlation shape.")
    epsilon = int(round(_normalise_nonnegative_time(epsilon_time, "epsilon_time") / dt))
    residual_jump = epsilon if max_residual_jump_time is None else int(round(_normalise_nonnegative_time(max_residual_jump_time, "max_residual_jump_time") / dt))
    if isinstance(max_skip_rows, bool) or int(max_skip_rows) != max_skip_rows or max_skip_rows < 0:
        raise TrackingError("max_skip_rows must be a non-negative integer.")
    margin = _normalise_boundary_margin(dt, boundary_margin_samples, boundary_margin_time)
    base_valid = _normalise_valid(raw, valid)
    if lag_valid is not None:
        base_valid &= _normalise_valid(raw, lag_valid)
    coarse = np.full(raw.shape[0], np.nan) if coarse_lag_samples is None else np.asarray(coarse_lag_samples, dtype=float)
    if coarse.shape != (raw.shape[0],):
        raise TrackingError("coarse_lag_samples must have shape [nreceiver].")
    if correlation_mode not in {"positive", "negative", "absolute", "auto_polarity"}:
        raise TrackingError("correlation_mode must be positive, negative, absolute, or auto_polarity.")
    penalty = np.zeros_like(raw)
    width = max(float(coarse_soft_width_time) / dt, 1.0)
    for row in np.flatnonzero(np.isfinite(coarse)):
        penalty[row] = np.minimum(
            coarse_soft_weight * ((lags - coarse[row]) / width) ** 2,
            coarse_soft_penalty_cap,
        )
    usable_receivers = np.flatnonzero(np.any(base_valid, axis=1))
    seed = (
        int(usable_receivers[np.argmin(np.abs(receivers[usable_receivers] - source_x))])
        if usable_receivers.size else int(np.argmin(np.abs(receivers - source_x)))
    )
    seed_states = np.flatnonzero(base_valid[seed])
    center = coarse[seed] if np.isfinite(coarse[seed]) else 0.0
    range_samples = int(round(seed_lag_range_time / dt))
    polarity = 1.0
    component_id = np.full(raw.shape[0], -1, dtype=int)
    row_polarity = np.zeros(raw.shape[0], dtype=int)
    provenance = np.full(raw.shape[0], "NO_MEASUREMENT", dtype="<U20")
    path = np.full(raw.shape[0], -1, dtype=int)
    dp_success = np.zeros(raw.shape[0], dtype=bool)
    if seed_states.size:
        seed_score = raw[seed, seed_states] - penalty[seed, seed_states]
        if correlation_mode in {"absolute", "auto_polarity"}:
            # Equal-magnitude polarities are physically ambiguous; prefer the
            # conventional positive branch deterministically.
            seed_score = np.abs(raw[seed, seed_states]) + 1e-6 * raw[seed, seed_states] - penalty[seed, seed_states]
        elif correlation_mode == "negative":
            seed_score = -raw[seed, seed_states] - penalty[seed, seed_states]
        # A seed is an anchor, not a one-row peak pick.  One direct neighbour
        # on each side suppresses an isolated periodic peak without imposing a
        # Top-K candidate mask on the actual DP state grid.
        range_scale = max(range_samples, 1)
        seed_score -= 0.05 * np.abs(lags[seed_states] - center) / range_scale
        for neighbour in (seed - 1, seed + 1):
            if not 0 <= neighbour < raw.shape[0]:
                continue
            neighbour_states = np.flatnonzero(base_valid[neighbour])
            if neighbour_states.size == 0:
                continue
            neighbour_score = raw[neighbour, neighbour_states] - penalty[neighbour, neighbour_states]
            if correlation_mode in {"absolute", "auto_polarity"}:
                neighbour_score = np.abs(raw[neighbour, neighbour_states]) - penalty[neighbour, neighbour_states]
            elif correlation_mode == "negative":
                neighbour_score = -raw[neighbour, neighbour_states] - penalty[neighbour, neighbour_states]
            for index, state in enumerate(seed_states):
                if np.isfinite(coarse[seed]) and np.isfinite(coarse[neighbour]):
                    jump = np.abs((lags[neighbour_states] - coarse[neighbour]) - (lags[state] - coarse[seed]))
                else:
                    jump = np.abs(lags[neighbour_states] - lags[state])
                compatible = neighbour_score[jump <= residual_jump]
                if compatible.size:
                    seed_score[index] += float(np.max(compatible))
        seed_state = int(seed_states[np.argmax(seed_score)])
        polarity = 1.0 if raw[seed, seed_state] >= 0 else -1.0
        if correlation_mode == "positive":
            polarity = 1.0
        elif correlation_mode == "negative":
            polarity = -1.0
        if correlation_mode == "absolute":
            score = np.abs(raw) - penalty
        else:
            score = polarity * raw - penalty
        left, _, left_row, _ = _dense_direction(score, base_valid, lags, receivers, coarse, seed, seed_state, -1, max_residual_jump_samples=residual_jump, max_skip_rows=int(max_skip_rows), smooth_weight=float(smooth_weight), gap_penalty=float(gap_penalty))
        right, _, right_row, _ = _dense_direction(score, base_valid, lags, receivers, coarse, seed, seed_state, 1, max_residual_jump_samples=residual_jump, max_skip_rows=int(max_skip_rows), smooth_weight=float(smooth_weight), gap_penalty=float(gap_penalty))
        path = np.where(left >= 0, left, right)
        dp_success = path >= 0
        component_id[dp_success] = 0
        row_polarity[dp_success] = int(polarity)
        provenance[dp_success] = "PRIMARY_DP"
        for row in np.flatnonzero(dp_success):
            state = path[row]
            parent = left_row[row, state] if left[row] >= 0 else right_row[row, state]
            if parent >= 0 and abs(row - parent) > 1:
                provenance[row] = "SKIP_CONNECTED_DP"
    if restart_enabled:
        unresolved = np.any(base_valid, axis=1) & ~dp_success
        for restart_id, region in enumerate(_valid_receiver_segments(unresolved[:, None]), start=1):
            if region.size < int(restart_confirm_rows):
                continue
            support = np.max(np.abs(raw[region[: int(restart_confirm_rows)]]), axis=1)
            if float(np.mean(support)) < float(restart_min_mean_correlation):
                continue
            anchor = int(region[0])
            states = np.flatnonzero(base_valid[anchor])
            if np.isfinite(coarse[anchor]):
                maximum = float(restart_max_coarse_deviation_time) / dt
                states = states[np.abs(lags[states] - coarse[anchor]) <= maximum]
            if states.size == 0:
                continue
            restart_polarity = polarity
            if correlation_mode == "auto_polarity":
                state = int(states[np.argmax(np.abs(raw[anchor, states]))])
                restart_polarity = 1.0 if raw[anchor, state] >= 0 else -1.0
                local_score = restart_polarity * raw - penalty
            else:
                local_score = score
                state = int(states[np.argmax(local_score[anchor, states])])
            region_valid = np.zeros_like(base_valid)
            region_valid[region] = base_valid[region]
            left, _, _, _ = _dense_direction(local_score, region_valid, lags, receivers, coarse, anchor, state, -1, max_residual_jump_samples=residual_jump, max_skip_rows=int(max_skip_rows), smooth_weight=float(smooth_weight), gap_penalty=float(gap_penalty))
            right, _, _, _ = _dense_direction(local_score, region_valid, lags, receivers, coarse, anchor, state, 1, max_residual_jump_samples=residual_jump, max_skip_rows=int(max_skip_rows), smooth_weight=float(smooth_weight), gap_penalty=float(gap_penalty))
            restarted = np.where(left >= 0, left, right)
            if np.count_nonzero(restarted[region[: int(restart_confirm_rows)]] >= 0) < int(restart_confirm_rows):
                continue
            fill = (path < 0) & (restarted >= 0)
            path[fill] = restarted[fill]
            dp_success[fill] = True
            component_id[fill] = restart_id
            row_polarity[fill] = int(restart_polarity)
            provenance[fill] = "RESTARTED_DP"
    shift_samples = np.full(raw.shape[0], np.nan)
    tracked = np.full(raw.shape[0], np.nan)
    strength = np.full(raw.shape[0], np.nan)
    boundary = np.zeros(raw.shape[0], dtype=bool)
    for row in np.flatnonzero(dp_success):
        state = int(path[row]); safe = np.where(base_valid[row], raw[row], np.nan)
        shift_samples[row] = _refine_peak(safe, state, lags)
        tracked[row] = raw[row, state]
        strength[row] = abs(raw[row, state]) if correlation_mode == "absolute" else row_polarity[row] * raw[row, state]
        boundary[row] = abs(lags[state]) >= np.max(np.abs(lags)) - margin
    deviation = np.full(raw.shape[0], np.nan)
    comparable = dp_success & np.isfinite(coarse)
    deviation[comparable] = lags[path[comparable]] - coarse[comparable]
    return TrackingResult(path, shift_samples, shift_samples * dt, tracked, seed, float(lags[path[seed]]) if dp_success[seed] else float("nan"), dp_success, boundary, dt, residual_jump, margin, dp_success_mask=dp_success, bridge_success_mask=np.zeros_like(dp_success), candidate_count=np.count_nonzero(base_valid, axis=1), coarse_deviation_samples=deviation, tracked_strength=strength, tracked_polarity=row_polarity, component_id=component_id, provenance=provenance)


def _local_peak(raw, valid, lags, predicted, width, polarity, weight, minimum, margin):
    """Return the best unambiguous local extremum, or ``-1``."""
    states = np.flatnonzero(valid & (np.abs(lags - predicted) <= width))
    if states.size == 0:
        return -1, np.nan, True
    strength = polarity * raw[states]
    peaks = states[(states == states[0]) | (states == states[-1])]
    middle = states[1:-1]
    if middle.size:
        peaks = np.r_[peaks, middle[(strength[1:-1] >= strength[:-2]) & (strength[1:-1] >= strength[2:])]]
    values = polarity * raw[peaks] - weight * ((lags[peaks] - predicted) / max(width, 1.0)) ** 2
    order = np.argsort(values)[::-1]
    best = int(peaks[order[0]])
    best_strength = float(polarity * raw[best])
    ambiguous = order.size > 1 and values[order[0]] - values[order[1]] < margin
    return (best if best_strength >= minimum and not ambiguous else -1), best_strength, bool(ambiguous)


def _predict(history_lag, history_coarse):
    """Median-slope predictor in residual coordinates when coarse is usable."""
    last_lag = history_lag[-1]
    if np.isfinite(history_coarse[-1]):
        residual = np.asarray(history_lag) - np.asarray(history_coarse)
        slope = np.median(np.diff(residual)) if len(residual) > 1 else 0.0
        return float(history_coarse[-1] + residual[-1] + slope)
    slope = np.median(np.diff(history_lag)) if len(history_lag) > 1 else 0.0
    return float(last_lag + slope)


def _track_local_direction(raw, valid, lags, coarse, seed, seed_state, direction, *,
                           history_size, base_width, max_width, expand, minimum,
                           margin, weight, max_gap, confirm_rows, restart_minimum):
    """One directional predictor-corrector pass; no global state graph."""
    nr = raw.shape[0]
    path = np.full(nr, -1, dtype=int)
    predicted = np.full(nr, np.nan)
    ambiguous = np.zeros(nr, dtype=bool)
    gaps = np.zeros(nr, dtype=int)
    segment = np.full(nr, -1, dtype=int)
    polarity = np.zeros(nr, dtype=int)
    path[seed] = seed_state
    segment[seed] = 0
    polarity[seed] = 1 if raw[seed, seed_state] >= 0 else -1
    active_polarity = int(polarity[seed])
    history_lag, history_coarse = [float(lags[seed_state])], [float(coarse[seed])]
    gap = 0
    segment_id = 0
    for row in range(seed + direction, nr if direction > 0 else -1, direction):
        pred = _predict(history_lag, history_coarse)
        if np.isfinite(coarse[row]) and np.isfinite(history_coarse[-1]):
            pred += coarse[row] - history_coarse[-1]
        width = min(max_width, base_width + gap * expand)
        predicted[row] = pred
        state, _, is_ambiguous = _local_peak(
            raw[row], valid[row], lags, pred, width, active_polarity,
            weight, minimum, margin,
        )
        ambiguous[row] = is_ambiguous
        if state >= 0:
            path[row] = state; segment[row] = segment_id; polarity[row] = active_polarity
            history_lag.append(float(lags[state])); history_coarse.append(float(coarse[row]))
            history_lag, history_coarse = history_lag[-history_size:], history_coarse[-history_size:]
            gap = 0; continue
        gap += 1; gaps[row] = gap
        if gap < max_gap:
            continue
        # Long-gap relock: require a small future run around coarse, not one peak.
        candidate = np.flatnonzero(valid[row] & np.isfinite(raw[row]))
        in_prediction = np.abs(lags[candidate] - pred) <= max_width
        in_coarse = (
            np.abs(lags[candidate] - coarse[row]) <= max_width
            if np.isfinite(coarse[row]) else np.zeros(candidate.size, dtype=bool)
        )
        candidate = candidate[in_prediction | in_coarse]
        if candidate.size:
            state = int(candidate[np.argmax(np.abs(raw[row, candidate]))])
            sign = 1 if raw[row, state] >= 0 else -1
            supported = 1
            check_lag = float(lags[state])
            for future in range(row + direction, nr if direction > 0 else -1, direction):
                next_state, strength, _ = _local_peak(raw[future], valid[future], lags, check_lag, max_width, sign, weight, restart_minimum, margin)
                if next_state < 0: break
                supported += 1; check_lag = float(lags[next_state])
                if supported >= confirm_rows: break
            if supported >= confirm_rows and abs(raw[row, state]) >= restart_minimum:
                segment_id += 1; path[row] = state; segment[row] = segment_id; polarity[row] = sign
                active_polarity = sign
                history_lag, history_coarse = [float(lags[state])], [float(coarse[row])]; gap = 0
    return path, predicted, ambiguous, gaps, segment, polarity


def _local_track_zncc(correlation, lags_samples, receiver_x, source_x, seed_lag_range_time, epsilon_time, dt, *,
                      measurement_correlation=None, valid=None, lag_valid=None, boundary_margin_samples=0,
                      boundary_margin_time=None, min_correlation=None, correlation_mode="auto_polarity",
                      coarse_lag_samples=None,
                      local_search_half_width_time=0.040, max_search_half_width_time=0.080,
                      gap_expand_time=0.010, residual_history=4, min_peak_margin=0.03,
                      prediction_weight=0.10, max_gap_rows=4, relock_confirm_rows=3,
                      restart_min_correlation=0.60, **_):
    values, lags, receivers = _validate_inputs(correlation, lags_samples, receiver_x, source_x, dt)
    raw = values if measurement_correlation is None else np.asarray(measurement_correlation, dtype=float)
    base_valid = _normalise_valid(raw, valid)
    if lag_valid is not None: base_valid &= _normalise_valid(raw, lag_valid)
    coarse = np.full(raw.shape[0], np.nan) if coarse_lag_samples is None else np.asarray(coarse_lag_samples, dtype=float)
    if coarse.shape != (raw.shape[0],): raise TrackingError("coarse_lag_samples must have shape [nreceiver].")
    usable_receivers = np.flatnonzero(np.any(base_valid, axis=1))
    seed = (
        int(usable_receivers[np.argmin(np.abs(receivers[usable_receivers] - source_x))])
        if usable_receivers.size else int(np.argmin(np.abs(receivers - source_x)))
    )
    seed_states = np.flatnonzero(base_valid[seed])
    center = coarse[seed] if np.isfinite(coarse[seed]) else 0.0
    if seed_states.size:
        nearby = seed_states[np.abs(lags[seed_states] - center) <= int(round(seed_lag_range_time / dt))]
        if nearby.size: seed_states = nearby
    path = np.full(raw.shape[0], -1, dtype=int)
    predicted = np.full(raw.shape[0], np.nan); ambiguous = np.zeros(raw.shape[0], bool); gaps = np.zeros(raw.shape[0], int)
    segment = np.full(raw.shape[0], -1, int); polarity = np.zeros(raw.shape[0], int)
    if seed_states.size:
        strength = np.abs(raw[seed, seed_states]) if correlation_mode in {"absolute", "auto_polarity"} else raw[seed, seed_states] * (-1 if correlation_mode == "negative" else 1)
        seed_state = int(seed_states[np.argmax(strength)])
        options = dict(history_size=max(1, int(residual_history)), base_width=max(1, int(round(local_search_half_width_time / dt))), max_width=max(1, int(round(max_search_half_width_time / dt))), expand=max(0, int(round(gap_expand_time / dt))), minimum=0.55 if min_correlation is None else float(min_correlation), margin=float(min_peak_margin), weight=float(prediction_weight), max_gap=int(max_gap_rows), confirm_rows=int(relock_confirm_rows), restart_minimum=float(restart_min_correlation))
        left = _track_local_direction(raw, base_valid, lags, coarse, seed, seed_state, -1, **options)
        right = _track_local_direction(raw, base_valid, lags, coarse, seed, seed_state, 1, **options)
        for result in (left, right):
            mask = result[0] >= 0; path[mask] = result[0][mask]; predicted[mask | ~np.isfinite(predicted)] = result[1][mask | ~np.isfinite(predicted)]; ambiguous |= result[2]; gaps = np.maximum(gaps, result[3]); segment[mask] = result[4][mask]; polarity[mask] = result[5][mask]
    success = path >= 0; shifts = np.full(raw.shape[0], np.nan); tracked = np.full(raw.shape[0], np.nan); strength = np.full(raw.shape[0], np.nan)
    for row in np.flatnonzero(success):
        shifts[row] = _refine_peak(np.where(base_valid[row], raw[row], np.nan), int(path[row]), lags)
        tracked[row] = raw[row, path[row]]; strength[row] = abs(tracked[row]) if correlation_mode == "absolute" else polarity[row] * tracked[row]
    margin_samples = _normalise_boundary_margin(dt, boundary_margin_samples, boundary_margin_time)
    boundary = success & (np.abs(lags[path.clip(0)]) >= np.max(np.abs(lags)) - margin_samples)
    provenance = np.where(success, "PRIMARY_LOCAL", "NO_MEASUREMENT").astype("<U20")
    provenance[success & (segment > 0)] = "RELOCKED_LOCAL"
    return TrackingResult(path, shifts, shifts * dt, tracked, seed, float(lags[path[seed]]) if success[seed] else np.nan, success, boundary, dt, int(round(epsilon_time / dt)), margin_samples, dp_success_mask=success, bridge_success_mask=np.zeros_like(success), candidate_count=np.count_nonzero(base_valid, axis=1), tracked_strength=strength, tracked_polarity=polarity, component_id=segment, provenance=provenance, predicted_lag=predicted, ambiguous_mask=ambiguous, gap_count=gaps)


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
    coarse_lag_samples: np.ndarray | None = None,
    top_k_peaks: int = 4,
    peak_min_separation_samples: int = 2,
    coarse_soft_width_time: float = 0.05,
    coarse_soft_weight: float = 0.05,
    coarse_soft_penalty_cap: float = 0.25,
    smooth_weight: float = 0.02,
    bridge_enabled: bool = True,
    bridge_half_width_time: float = 0.05,
    bridge_max_width_time: float = 0.08,
    bridge_max_receivers: int = 8,
    bridge_max_distance: float | None = None,
    max_residual_jump_time: float | None = None,
    max_skip_rows: int = 3,
    gap_penalty: float = 0.05,
    correlation_mode: str = "auto_polarity",
    restart_enabled: bool = True,
    restart_confirm_rows: int = 3,
    restart_min_mean_correlation: float = 0.55,
    restart_max_coarse_deviation_time: float = 0.10,
    local_search_half_width_time: float = 0.040,
    max_search_half_width_time: float = 0.080,
    gap_expand_time: float = 0.010,
    residual_history: int = 4,
    min_peak_margin: float = 0.03,
    prediction_weight: float = 0.10,
    max_gap_rows: int = 4,
    relock_confirm_rows: int = 3,
    restart_min_correlation: float = 0.60,
    tracker_mode: str = "sparse_global",
    _auto_polarity_trial: bool = False,
) -> TrackingResult:
    """Track sparse raw-waveform peaks with a soft coarse prior and gap bridge."""
    if (
        tracker_mode == "sparse_global"
        and correlation_mode == "auto_polarity"
        and not _auto_polarity_trial
    ):
        options = dict(
            measurement_correlation=measurement_correlation,
            valid=valid,
            lag_valid=lag_valid,
            min_correlation=min_correlation,
            boundary_margin_samples=boundary_margin_samples,
            boundary_margin_time=boundary_margin_time,
            raw_refine_radius_samples=raw_refine_radius_samples,
            coarse_lag_samples=coarse_lag_samples,
            top_k_peaks=top_k_peaks,
            peak_min_separation_samples=peak_min_separation_samples,
            coarse_soft_width_time=coarse_soft_width_time,
            coarse_soft_weight=coarse_soft_weight,
            coarse_soft_penalty_cap=coarse_soft_penalty_cap,
            smooth_weight=smooth_weight,
            bridge_enabled=bridge_enabled,
            bridge_half_width_time=bridge_half_width_time,
            bridge_max_width_time=bridge_max_width_time,
            bridge_max_receivers=bridge_max_receivers,
            bridge_max_distance=bridge_max_distance,
            max_residual_jump_time=max_residual_jump_time,
            max_skip_rows=max_skip_rows,
            gap_penalty=gap_penalty,
            restart_enabled=restart_enabled,
            restart_confirm_rows=restart_confirm_rows,
            restart_min_mean_correlation=restart_min_mean_correlation,
            restart_max_coarse_deviation_time=restart_max_coarse_deviation_time,
            local_search_half_width_time=local_search_half_width_time,
            max_search_half_width_time=max_search_half_width_time,
            gap_expand_time=gap_expand_time,
            residual_history=residual_history,
            min_peak_margin=min_peak_margin,
            prediction_weight=prediction_weight,
            max_gap_rows=max_gap_rows,
            relock_confirm_rows=relock_confirm_rows,
            restart_min_correlation=restart_min_correlation,
            tracker_mode="sparse_global",
            _auto_polarity_trial=True,
        )
        positive = track_zncc(
            correlation, lags_samples, receiver_x, source_x,
            seed_lag_range_time, epsilon_time, dt,
            correlation_mode="positive", **options,
        )
        negative = track_zncc(
            correlation, lags_samples, receiver_x, source_x,
            seed_lag_range_time, epsilon_time, dt,
            correlation_mode="negative", **options,
        )
        positive_key = (
            int(np.count_nonzero(positive.success_mask)),
            float(np.nansum(positive.tracked_strength)),
        )
        negative_key = (
            int(np.count_nonzero(negative.success_mask)),
            float(np.nansum(negative.tracked_strength)),
        )
        return positive if positive_key >= negative_key else negative
    if tracker_mode == "local":
        return _local_track_zncc(
            correlation, lags_samples, receiver_x, source_x, seed_lag_range_time,
            epsilon_time, dt, measurement_correlation=measurement_correlation,
            valid=valid, lag_valid=lag_valid, boundary_margin_samples=boundary_margin_samples,
            boundary_margin_time=boundary_margin_time, coarse_lag_samples=coarse_lag_samples,
            min_correlation=min_correlation, correlation_mode=correlation_mode,
            local_search_half_width_time=local_search_half_width_time,
            max_search_half_width_time=max_search_half_width_time,
            gap_expand_time=gap_expand_time, residual_history=residual_history,
            min_peak_margin=min_peak_margin, prediction_weight=prediction_weight,
            max_gap_rows=max_gap_rows, relock_confirm_rows=relock_confirm_rows,
            restart_min_correlation=restart_min_correlation,
        )
    if tracker_mode != "sparse_global":
        raise TrackingError("tracker_mode must be 'sparse_global' or 'local'.")
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
    # Compatibility-only after removing the guide-locked raw corridor.  Peak
    # refinement remains the existing three-point sub-sample interpolation.
    if min_correlation is not None:
        if not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0:
            raise TrackingError("min_correlation must be in [-1, 1] or None.")
        min_correlation = float(min_correlation)

    for value, name in (
        (top_k_peaks, "top_k_peaks"),
        (peak_min_separation_samples, "peak_min_separation_samples"),
        (bridge_max_receivers, "bridge_max_receivers"),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise TrackingError(f"{name} must be a non-negative integer.")
    if int(top_k_peaks) < 1:
        raise TrackingError("top_k_peaks must be positive.")
    for value, name in (
        (coarse_soft_width_time, "coarse_soft_width_time"),
        (coarse_soft_weight, "coarse_soft_weight"),
        (coarse_soft_penalty_cap, "coarse_soft_penalty_cap"),
        (smooth_weight, "smooth_weight"),
        (bridge_half_width_time, "bridge_half_width_time"),
        (bridge_max_width_time, "bridge_max_width_time"),
    ):
        _normalise_nonnegative_time(value, name)
    if coarse_soft_width_time == 0:
        raise TrackingError("coarse_soft_width_time must be positive.")
    if bridge_max_width_time < bridge_half_width_time:
        raise TrackingError(
            "bridge_max_width_time must be at least bridge_half_width_time."
        )
    if not isinstance(bridge_enabled, (bool, np.bool_)):
        raise TrackingError("bridge_enabled must be boolean.")
    if bridge_max_distance is not None:
        _normalise_nonnegative_time(bridge_max_distance, "bridge_max_distance")

    n_receiver = values.shape[0]
    seed_receiver = int(np.argmin(np.abs(receivers - float(source_x))))
    coarse = (
        np.full(n_receiver, np.nan, dtype=float)
        if coarse_lag_samples is None
        else np.asarray(coarse_lag_samples, dtype=float)
    )
    if coarse.shape != (n_receiver,) or not (np.isfinite(coarse) | np.isnan(coarse)).all():
        raise TrackingError(
            "coarse_lag_samples must contain finite values or NaN with shape [nreceiver]."
        )

    # Missing coarse data must fall back to absolute-lag continuity.  It must
    # never be fabricated from a local high guide peak, which would turn an
    # isolated periodic peak into an artificial residual-continuous branch.
    soft_prior = np.array(coarse, copy=True)

    # Ownership/finite validity is the only hard lag-state gate.  The enhanced
    # guide no longer deletes raw waveform evidence.
    base_valid = _normalise_valid(measurement, valid)
    if lag_valid is not None:
        base_valid &= _normalise_valid(measurement, lag_valid)
    raw_candidate_values = (
        np.abs(measurement) + 1.0e-8 * measurement
        if correlation_mode in {"absolute", "auto_polarity"}
        else (-measurement if correlation_mode == "negative" else measurement)
    )
    candidate_valid = _top_k_peak_mask(
        raw_candidate_values,
        base_valid,
        lags,
        top_k=int(top_k_peaks),
        min_separation_samples=int(peak_min_separation_samples),
        coarse_lag_samples=soft_prior,
    )
    # Enhanced/stacked correlation may reveal a ridge peak, but every accepted
    # state remains ownership-valid finite waveform ZNCC and is scored on raw.
    guide_candidate_values = (
        np.abs(values) + 1.0e-8 * values
        if correlation_mode in {"absolute", "auto_polarity"}
        else (-values if correlation_mode == "negative" else values)
    )
    guide_candidates = _top_k_peak_mask(
        guide_candidate_values,
        base_valid & np.isfinite(values),
        lags,
        top_k=int(top_k_peaks),
        min_separation_samples=int(peak_min_separation_samples),
        coarse_lag_samples=soft_prior,
    )
    candidate_valid |= guide_candidates & base_valid
    candidate_count = np.count_nonzero(candidate_valid, axis=1)

    usable_receivers = np.flatnonzero(np.any(candidate_valid, axis=1))
    if usable_receivers.size:
        seed_receiver = int(
            usable_receivers[np.argmin(np.abs(receivers[usable_receivers] - float(source_x)))]
        )

    if correlation_mode not in {"positive", "negative", "absolute", "auto_polarity"}:
        raise TrackingError("correlation_mode must be positive, negative, absolute, or auto_polarity.")
    seed_states = np.flatnonzero(candidate_valid[seed_receiver])
    polarity = 1
    if correlation_mode == "negative":
        polarity = -1
    elif correlation_mode == "auto_polarity" and seed_states.size:
        seed_values = measurement[seed_receiver, seed_states]
        seed_index = int(np.argmax(np.abs(seed_values) + 1.0e-8 * seed_values))
        polarity = 1 if seed_values[seed_index] >= 0 else -1
    score_values = (
        np.abs(measurement)
        if correlation_mode == "absolute"
        else float(polarity) * np.asarray(measurement, dtype=float)
    )
    width_samples = max(float(coarse_soft_width_time) / float(dt), 1.0)
    for receiver in np.flatnonzero(np.isfinite(soft_prior)):
        rho = (lags.astype(float) - soft_prior[receiver]) / width_samples
        penalty = np.minimum(
            float(coarse_soft_weight) * rho * rho,
            float(coarse_soft_penalty_cap),
        )
        score_values[receiver] -= penalty

    component_paths: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in _candidate_segments(
        candidate_valid, lags, epsilon_samples, soft_prior
    ):
        local_seed = (
            seed_receiver
            if np.any(segment == seed_receiver)
            else int(segment[np.argmin(np.abs(receivers[segment] - float(source_x)))])
        )
        local_path = _global_dp_segment(
            score_values,
            candidate_valid,
            lags,
            segment,
            seed_receiver=local_seed,
            coarse_lag_samples=soft_prior,
            epsilon_samples=epsilon_samples,
            smooth_weight=float(smooth_weight),
        )
        component_paths.append((segment, local_path))

    path_index = np.full(n_receiver, -1, dtype=int)
    dp_success = np.zeros(n_receiver, dtype=bool)
    bridge_success = np.zeros(n_receiver, dtype=bool)
    component_id = np.full(n_receiver, -1, dtype=int)
    main_component = next(
        (
            index
            for index, (segment, local_path) in enumerate(component_paths)
            if np.any(segment == seed_receiver)
            and local_path[int(seed_receiver - segment[0])] >= 0
        ),
        None,
    )
    if main_component is not None:
        segment, local_path = component_paths[main_component]
        selected = local_path >= 0
        path_index[segment[selected]] = local_path[selected]
        dp_success[segment[selected]] = True
        component_id[segment[selected]] = 0

    half_width_samples = max(
        int(round(float(bridge_half_width_time) / float(dt))), 1
    )
    max_width_samples = max(
        int(round(float(bridge_max_width_time) / float(dt))),
        half_width_samples,
    )
    if bool(bridge_enabled) and main_component is not None:
        # Connect independently tracked raw-ZNCC components to the seed
        # component.  Interpolation defines only the reopened search corridor;
        # every recovered lag still comes from raw ZNCC DP.
        for direction in (1, -1):
            component_index = main_component
            next_index = component_index + direction
            while 0 <= next_index < len(component_paths):
                current_rows = np.flatnonzero(dp_success | bridge_success)
                next_segment, next_path = component_paths[next_index]
                next_selected = next_path >= 0
                if not np.any(next_selected):
                    next_index += direction
                    continue
                if direction > 0:
                    left_receiver = int(current_rows[-1])
                    right_receiver = int(next_segment[next_selected][0])
                    left_state = int(path_index[left_receiver])
                    right_state = int(next_path[next_selected][0])
                else:
                    left_receiver = int(next_segment[next_selected][-1])
                    right_receiver = int(current_rows[0])
                    left_state = int(next_path[next_selected][-1])
                    right_state = int(path_index[right_receiver])
                bridged = _bridge_between(
                    score_values,
                    base_valid,
                    lags,
                    receivers,
                    soft_prior,
                    left_receiver,
                    left_state,
                    right_receiver,
                    right_state,
                    half_width_samples=half_width_samples,
                    max_width_samples=max_width_samples,
                    max_jump_samples=epsilon_samples,
                    max_receivers=int(bridge_max_receivers),
                    max_distance=bridge_max_distance,
                    smooth_weight=float(smooth_weight),
                )
                if bridged is None:
                    gap = right_receiver - left_receiver - 1
                    if gap > int(bridge_max_receivers):
                        break
                    next_index += direction
                    continue
                bridge_rows, bridge_path = bridged
                internal = bridge_rows[1:-1]
                connected_component = (
                    component_id[left_receiver]
                    if component_id[left_receiver] >= 0
                    else component_id[right_receiver]
                )
                path_index[internal] = bridge_path[1:-1]
                bridge_success[internal] = True
                component_id[internal] = connected_component
                remote_rows = next_segment[next_selected]
                path_index[remote_rows] = next_path[next_selected]
                dp_success[remote_rows] = True
                component_id[remote_rows] = connected_component
                component_index = next_index
                next_index = component_index + direction

        for direction in (1, -1):
            extended = _bridge_one_side(
                score_values,
                base_valid,
                lags,
                receivers,
                soft_prior,
                path_index,
                direction,
                half_width_samples=half_width_samples,
                max_width_samples=max_width_samples,
                max_jump_samples=epsilon_samples,
                max_receivers=int(bridge_max_receivers),
                max_distance=bridge_max_distance,
                smooth_weight=float(smooth_weight),
            )
            if extended is not None:
                bridge_rows, bridge_path = extended
                still_missing = path_index[bridge_rows] < 0
                path_index[bridge_rows[still_missing]] = bridge_path[still_missing]
                bridge_success[bridge_rows[still_missing]] = True
                component_id[bridge_rows[still_missing]] = component_id[
                    bridge_rows[0]
                ]

    # A gap longer than the bridge limit must not erase a later, independently
    # supported raw ridge.  This is a new DP component, not interpolation or
    # fallback: require the configured number of consecutive DP states first.
    if bool(restart_enabled):
        for index, (segment, local_path) in enumerate(component_paths):
            if index == main_component:
                continue
            selected = local_path >= 0
            rows = segment[selected]
            missing = rows[path_index[rows] < 0]
            if missing.size < int(restart_confirm_rows):
                continue
            path_index[missing] = local_path[missing - segment[0]]
            dp_success[missing] = True
            component_id[missing] = index + 1

    shift_samples = np.full(n_receiver, np.nan, dtype=float)
    shift_time = np.full(n_receiver, np.nan, dtype=float)
    tracked_correlation = np.full(n_receiver, np.nan, dtype=float)
    boundary_flag = np.zeros(n_receiver, dtype=bool)
    seed_lag = float("nan")
    max_abs_lag = float(np.max(np.abs(lags)))
    boundary_limit = max_abs_lag - float(margin_samples)
    success_mask = dp_success | bridge_success
    _assert_path_continuity(
        path_index,
        lags,
        epsilon_samples,
        soft_prior,
        component_id,
    )
    for receiver in np.flatnonzero(success_mask):
        index = int(path_index[receiver])
        safe_measurement = np.where(
            base_valid[receiver], measurement[receiver], np.nan
        )
        shift_samples[receiver] = _refine_peak(safe_measurement, index, lags)
        shift_time[receiver] = shift_samples[receiver] * float(dt)
        tracked_correlation[receiver] = measurement[receiver, index]
        boundary_flag[receiver] = abs(float(lags[index])) >= boundary_limit

    if success_mask[seed_receiver]:
        seed_lag = float(lags[path_index[seed_receiver]])

    coarse_deviation = np.full(n_receiver, np.nan, dtype=float)
    comparable = success_mask & np.isfinite(coarse)
    coarse_deviation[comparable] = (
        lags[path_index[comparable]].astype(float) - coarse[comparable]
    )

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
        dp_success_mask=dp_success,
        bridge_success_mask=bridge_success,
        candidate_count=candidate_count,
        coarse_deviation_samples=coarse_deviation,
        tracked_strength=np.where(success_mask, np.abs(tracked_correlation) if correlation_mode == "absolute" else polarity * tracked_correlation, np.nan),
        tracked_polarity=np.where(success_mask, polarity, 0),
        component_id=component_id,
        provenance=np.where(
            bridge_success,
            "BRIDGE_DP",
            np.where(dp_success, "PRIMARY_DP", "NO_MEASUREMENT"),
        ),
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
    top_k_peaks: int = 4,
    peak_min_separation_samples: int = 2,
    coarse_soft_width_time: float = 0.05,
    coarse_soft_weight: float = 0.05,
    coarse_soft_penalty_cap: float = 0.25,
    smooth_weight: float = 0.02,
    bridge_enabled: bool = True,
    bridge_half_width_time: float = 0.05,
    bridge_max_width_time: float = 0.08,
    bridge_max_receivers: int = 8,
    bridge_max_distance: float | None = None,
    max_residual_jump_time: float | None = None,
    max_skip_rows: int = 3,
    gap_penalty: float = 0.05,
    correlation_mode: str = "auto_polarity",
    restart_enabled: bool = True,
    restart_confirm_rows: int = 3,
    restart_min_mean_correlation: float = 0.55,
    restart_max_coarse_deviation_time: float = 0.10,
    local_search_half_width_time: float = 0.040,
    max_search_half_width_time: float = 0.080,
    gap_expand_time: float = 0.010,
    residual_history: int = 4,
    min_peak_margin: float = 0.03,
    prediction_weight: float = 0.10,
    max_gap_rows: int = 4,
    relock_confirm_rows: int = 3,
    restart_min_correlation: float = 0.60,
    tracker_mode: str = "sparse_global",
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
        coarse_lag_samples=getattr(correlation_result, "coarse_lag_samples", None),
        top_k_peaks=top_k_peaks,
        peak_min_separation_samples=peak_min_separation_samples,
        coarse_soft_width_time=coarse_soft_width_time,
        coarse_soft_weight=coarse_soft_weight,
        coarse_soft_penalty_cap=coarse_soft_penalty_cap,
        smooth_weight=smooth_weight,
        bridge_enabled=bridge_enabled,
        bridge_half_width_time=bridge_half_width_time,
        bridge_max_width_time=bridge_max_width_time,
        bridge_max_receivers=bridge_max_receivers,
        bridge_max_distance=bridge_max_distance,
        max_residual_jump_time=max_residual_jump_time,
        max_skip_rows=max_skip_rows,
        gap_penalty=gap_penalty,
        correlation_mode=correlation_mode,
        restart_enabled=restart_enabled,
        restart_confirm_rows=restart_confirm_rows,
        restart_min_mean_correlation=restart_min_mean_correlation,
        restart_max_coarse_deviation_time=restart_max_coarse_deviation_time,
        local_search_half_width_time=local_search_half_width_time,
        max_search_half_width_time=max_search_half_width_time,
        gap_expand_time=gap_expand_time,
        residual_history=residual_history,
        min_peak_margin=min_peak_margin,
        prediction_weight=prediction_weight,
        max_gap_rows=max_gap_rows,
        relock_confirm_rows=relock_confirm_rows,
        restart_min_correlation=restart_min_correlation,
        tracker_mode=tracker_mode,
    )
