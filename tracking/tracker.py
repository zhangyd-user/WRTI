"""ZNCC ridge tracking for WRTI Step 5.

Reference/legacy calls keep the original single-seed connected behavior.
Candidate dual-center calls may pass ``receiver_mask`` (BootstrapFixedMask):
every contiguous True segment gets its own seed and the DP may skip up to
``max_consecutive_failures`` bad receivers while keeping the same lag jump
constraint against the last successful pick.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class TrackingError(ValueError):
    pass


def _readonly(value: np.ndarray, dtype) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class TrackingResult:
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
        path = np.asarray(self.path_index, dtype=int)
        if path.ndim != 1:
            raise TrackingError("path_index must be one-dimensional.")
        shape = path.shape
        arrays = {
            "shift_samples": np.asarray(self.shift_samples, dtype=float),
            "shift_time": np.asarray(self.shift_time, dtype=float),
            "tracked_correlation": np.asarray(self.tracked_correlation, dtype=float),
            "success_mask": np.asarray(self.success_mask, dtype=bool),
            "boundary_flag": np.asarray(self.boundary_flag, dtype=bool),
        }
        for name, arr in arrays.items():
            if arr.shape != shape:
                raise TrackingError(f"{name} must have shape {shape}.")
        if int(self.seed_receiver) != self.seed_receiver or self.seed_receiver < 0:
            raise TrackingError("seed_receiver must be a non-negative integer.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise TrackingError("dt must be finite and positive.")
        if int(self.epsilon_samples) != self.epsilon_samples or self.epsilon_samples < 0:
            raise TrackingError("epsilon_samples must be a non-negative integer.")
        if (
            int(self.boundary_margin_samples) != self.boundary_margin_samples
            or self.boundary_margin_samples < 0
        ):
            raise TrackingError("boundary_margin_samples must be non-negative.")

        object.__setattr__(self, "path_index", _readonly(path, int))
        for name, arr in arrays.items():
            object.__setattr__(self, name, _readonly(arr, arr.dtype))
        object.__setattr__(self, "seed_receiver", int(self.seed_receiver))
        object.__setattr__(self, "seed_lag", float(self.seed_lag))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "epsilon_samples", int(self.epsilon_samples))
        object.__setattr__(
            self, "boundary_margin_samples", int(self.boundary_margin_samples)
        )


def _validate_inputs(correlation, lags_samples, receiver_x, source_x, dt):
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
        raise TrackingError("lags_samples must be contiguous and increase by one.")
    if not np.isfinite(source_x):
        raise TrackingError("source_x must be finite.")
    if not np.isfinite(dt) or dt <= 0:
        raise TrackingError("dt must be finite and positive.")
    return values, lags, receivers


def _normalise_valid(values: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    finite = np.isfinite(values)
    if valid is None:
        return finite
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == (values.shape[0],):
        mask = np.broadcast_to(mask[:, None], values.shape)
    elif mask.shape != values.shape:
        raise TrackingError("valid must have shape [nreceiver] or [nreceiver,nlag].")
    return finite & mask


def _refine_peak(row: np.ndarray, peak_index: int, lags_samples: np.ndarray) -> float:
    integer_lag = float(lags_samples[peak_index])
    if peak_index == 0 or peak_index == row.size - 1:
        return integer_lag
    left, center, right = map(float, row[peak_index - 1 : peak_index + 2])
    if not np.isfinite([left, center, right]).all():
        return integer_lag
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator == 0.0:
        return integer_lag
    delta = (left - right) / (2.0 * denominator)
    return integer_lag if (not np.isfinite(delta) or abs(delta) > 0.5) else integer_lag + delta


def _segments_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise TrackingError("receiver mask must be one-dimensional.")
    out, start = [], None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            out.append(np.arange(start, i, dtype=int))
            start = None
    if start is not None:
        out.append(np.arange(start, mask.size, dtype=int))
    return out


def _valid_receiver_segments(row_valid: np.ndarray) -> list[np.ndarray]:
    """Legacy helper retained for compatibility."""
    return _segments_from_mask(np.any(row_valid, axis=1))


def _choose_segment_seed(
    row_valid: np.ndarray,
    segment: np.ndarray,
    receivers: np.ndarray,
    source_x: float,
    seed_allowed: np.ndarray,
) -> int | None:
    candidates = [
        int(r) for r in segment if np.any(row_valid[int(r)] & seed_allowed)
    ]
    if not candidates:
        return None
    c = np.asarray(candidates, dtype=int)
    return int(c[np.argmin(np.abs(receivers[c] - float(source_x)))])


def _directional_dp_from_seed(
    values: np.ndarray,
    row_valid: np.ndarray,
    receiver_order: np.ndarray,
    *,
    epsilon_samples: int,
    max_consecutive_failures: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Global directional DP with optional bounded receiver skipping.

    A jump of ``k`` receiver positions skips ``k-1`` failed receivers.  The
    allowed lag jump remains ``epsilon_samples`` regardless of k.
    """
    order = np.asarray(receiver_order, dtype=int)
    if order.ndim != 1 or order.size == 0:
        raise TrackingError("receiver_order must be a non-empty 1-D array.")
    if (
        isinstance(max_consecutive_failures, bool)
        or int(max_consecutive_failures) != max_consecutive_failures
        or max_consecutive_failures < 0
    ):
        raise TrackingError("max_consecutive_failures must be non-negative integer.")

    nrow, nlag = order.size, values.shape[1]
    max_jump = int(max_consecutive_failures) + 1
    coverage = np.zeros((nrow, nlag), dtype=int)
    score = np.full((nrow, nlag), -np.inf, dtype=float)
    next_local = np.full((nrow, nlag), -1, dtype=int)
    next_state = np.full((nrow, nlag), -1, dtype=int)
    abs_values = np.abs(values)

    for k in range(nrow - 1, -1, -1):
        receiver = int(order[k])
        current_valid = row_valid[receiver]
        coverage[k, current_valid] = 1
        score[k, current_valid] = abs_values[receiver, current_valid]
        if not np.any(current_valid):
            continue

        for target_k in range(k + 1, min(nrow, k + max_jump + 1)):
            target_receiver = int(order[target_k])
            target_cov = coverage[target_k]
            target_score = score[target_k]

            best_cov = np.zeros(nlag, dtype=int)
            best_score = np.full(nlag, -np.inf, dtype=float)
            best_state = np.full(nlag, -1, dtype=int)

            for offset in range(-epsilon_samples, epsilon_samples + 1):
                if offset < 0:
                    src = np.arange(-offset, nlag, dtype=int)
                elif offset > 0:
                    src = np.arange(0, nlag - offset, dtype=int)
                else:
                    src = np.arange(nlag, dtype=int)
                dst = src + offset
                same_polarity = (
                    values[receiver, src] * values[target_receiver, dst]
                ) > 0.0
                usable = (target_cov[dst] > 0) & same_polarity
                if not np.any(usable):
                    continue
                s = src[usable]
                d = dst[usable]
                cand_cov = target_cov[d]
                cand_score = target_score[d]
                improve = (cand_cov > best_cov[s]) | (
                    (cand_cov == best_cov[s]) & (cand_score > best_score[s])
                )
                if np.any(improve):
                    chosen = s[improve]
                    best_cov[chosen] = cand_cov[improve]
                    best_score[chosen] = cand_score[improve]
                    best_state[chosen] = d[improve]

            candidate = current_valid & (best_state >= 0)
            states = np.flatnonzero(candidate)
            if states.size == 0:
                continue
            cand_cov = 1 + best_cov[states]
            cand_score = abs_values[receiver, states] + best_score[states]
            improve = (cand_cov > coverage[k, states]) | (
                (cand_cov == coverage[k, states])
                & (cand_score > score[k, states])
            )
            if np.any(improve):
                chosen = states[improve]
                coverage[k, chosen] = cand_cov[improve]
                score[k, chosen] = cand_score[improve]
                next_local[k, chosen] = target_k
                next_state[k, chosen] = best_state[chosen]

    return coverage[0], score[0], next_local, next_state


def _trace_direction(order, seed_state, next_local, next_state):
    path = np.full(len(order), -1, dtype=int)
    k, state = 0, int(seed_state)
    while 0 <= k < len(order) and state >= 0:
        path[k] = state
        nk = int(next_local[k, state])
        if nk < 0:
            break
        state = int(next_state[k, state])
        k = nk
    return path


def _global_dp_segment(
    values: np.ndarray,
    row_valid: np.ndarray,
    lags: np.ndarray,
    segment: np.ndarray,
    *,
    seed_receiver: int,
    epsilon_samples: int,
    max_consecutive_failures: int = 0,
    seed_allowed: np.ndarray | None = None,
) -> np.ndarray:
    del lags
    path = np.full(segment.size, -1, dtype=int)
    loc = np.flatnonzero(segment == seed_receiver)
    if loc.size == 0:
        return path
    seed_local = int(loc[0])
    seed_valid = np.asarray(row_valid[seed_receiver], dtype=bool)
    if seed_allowed is not None:
        seed_valid &= np.asarray(seed_allowed, dtype=bool)
    if not np.any(seed_valid):
        return path

    left_order = segment[seed_local::-1]
    right_order = segment[seed_local:]
    lc, ls, lnl, lns = _directional_dp_from_seed(
        values,
        row_valid,
        left_order,
        epsilon_samples=epsilon_samples,
        max_consecutive_failures=max_consecutive_failures,
    )
    rc, rs, rnl, rns = _directional_dp_from_seed(
        values,
        row_valid,
        right_order,
        epsilon_samples=epsilon_samples,
        max_consecutive_failures=max_consecutive_failures,
    )

    total_cov = lc + rc - 1
    total_score = ls + rs - np.abs(values[seed_receiver])
    candidates = seed_valid & (total_cov > 0) & np.isfinite(total_score)
    if not np.any(candidates):
        return path
    states = np.flatnonzero(candidates)
    best_cov = np.max(total_cov[states])
    states = states[total_cov[states] == best_cov]
    best_score = np.max(total_score[states])
    states = states[np.isclose(total_score[states], best_score)]
    if states.size > 1:
        signed_scores = []
        for state in states:
            left = _trace_direction(left_order, int(state), lnl, lns)
            right = _trace_direction(right_order, int(state), rnl, rns)
            score = sum(
                values[int(receiver), int(path_state)]
                for receiver, path_state in zip(left_order, left)
                if path_state >= 0
            )
            score += sum(
                values[int(receiver), int(path_state)]
                for receiver, path_state in zip(right_order[1:], right[1:])
                if path_state >= 0
            )
            signed_scores.append(float(score))
        seed_state = int(states[int(np.argmax(signed_scores))])
    else:
        seed_state = int(states[0])

    left_path = _trace_direction(left_order, seed_state, lnl, lns)
    right_path = _trace_direction(right_order, seed_state, rnl, rns)
    base = int(segment[0])
    for receiver, state in zip(left_order, left_path):
        if state >= 0:
            path[int(receiver) - base] = state
    for receiver, state in zip(right_order[1:], right_path[1:]):
        if state >= 0:
            path[int(receiver) - base] = state
    return path


def _assert_path_continuity(path_index, lags, epsilon_samples):
    path = np.asarray(path_index, dtype=int)
    adjacent = (path[:-1] >= 0) & (path[1:] >= 0)
    if not np.any(adjacent):
        return
    jumps = np.abs(lags[path[1:][adjacent]] - lags[path[:-1][adjacent]])
    if np.any(jumps > int(epsilon_samples)):
        raise AssertionError("Waveform DP violated its hard receiver-to-receiver continuity bound.")


def _assert_gap_continuity(path_index, lags, epsilon_samples, receiver_mask, max_failures):
    path = np.asarray(path_index, dtype=int)
    for segment in _segments_from_mask(receiver_mask):
        success = segment[path[segment] >= 0]
        for a, b in zip(success[:-1], success[1:]):
            if int(b - a - 1) > int(max_failures):
                raise AssertionError("Gap-aware DP crossed too many failed receivers.")
            if abs(int(lags[path[b]]) - int(lags[path[a]])) > int(epsilon_samples):
                raise AssertionError("Gap-aware DP violated lag continuity across a failure gap.")


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
    restrict_seed_lag_range: bool = False,
    receiver_mask: np.ndarray | None = None,
    max_consecutive_failures: int = 3,
    hard_min_correlation: bool = False,
) -> TrackingResult:
    values, lags, receivers = _validate_inputs(
        correlation, lags_samples, receiver_x, source_x, dt
    )
    measurement = values if measurement_correlation is None else np.asarray(
        measurement_correlation, dtype=float
    )
    if measurement.shape != values.shape:
        raise TrackingError("measurement_correlation must match correlation shape.")

    if not np.isfinite(seed_lag_range_time) or seed_lag_range_time < 0:
        raise TrackingError("seed_lag_range_time must be finite and non-negative.")
    if not np.isfinite(epsilon_time) or epsilon_time < 0:
        raise TrackingError("epsilon_time must be finite and non-negative.")
    epsilon_samples = int(round(float(epsilon_time) / float(dt)))
    if boundary_margin_time is not None:
        if boundary_margin_samples != 0:
            raise TrackingError("Specify boundary_margin_time or boundary_margin_samples, not both.")
        boundary_margin_samples = int(round(float(boundary_margin_time) / float(dt)))
    if int(boundary_margin_samples) != boundary_margin_samples or boundary_margin_samples < 0:
        raise TrackingError("boundary_margin_samples must be non-negative integer.")
    if int(raw_refine_radius_samples) != raw_refine_radius_samples or raw_refine_radius_samples < 0:
        raise TrackingError("raw_refine_radius_samples must be non-negative integer.")
    if int(max_consecutive_failures) != max_consecutive_failures or max_consecutive_failures < 0:
        raise TrackingError("max_consecutive_failures must be non-negative integer.")
    if min_correlation is not None and (
        not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0
    ):
        raise TrackingError("min_correlation must be in [-1,1] or None.")

    nreceiver = values.shape[0]
    domain = None if receiver_mask is None else np.asarray(receiver_mask, dtype=bool)
    if domain is not None and domain.shape != (nreceiver,):
        raise TrackingError("receiver_mask must have shape [nreceiver].")

    legacy_seed = int(np.argmin(np.abs(receivers - float(source_x))))
    seed_allowed = np.ones(lags.size, dtype=bool)
    if restrict_seed_lag_range:
        seed_allowed = (
            np.abs(lags.astype(float) * float(dt))
            <= float(seed_lag_range_time) + 1.0e-12
        )

    row_valid = _normalise_valid(values, valid)
    if lag_valid is not None:
        row_valid &= _normalise_valid(values, lag_valid)

    quality_allowed = np.ones_like(row_valid, dtype=bool)
    if hard_min_correlation and min_correlation is not None:
        quality_allowed = np.isfinite(measurement) & (
            np.abs(measurement) >= abs(float(min_correlation))
        )

    guide_valid = np.array(row_valid, copy=True)
    guide_path = np.full(nreceiver, -1, dtype=int)

    if domain is None:
        if restrict_seed_lag_range:
            guide_valid[legacy_seed] &= seed_allowed
        for segment in _valid_receiver_segments(guide_valid):
            if legacy_seed not in segment:
                continue
            p = _global_dp_segment(
                values,
                guide_valid,
                lags,
                segment,
                seed_receiver=legacy_seed,
                epsilon_samples=epsilon_samples,
            )
            ok = p >= 0
            guide_path[segment[ok]] = p[ok]
    else:
        guide_valid &= domain[:, None]
        guide_valid &= quality_allowed
        for segment in _segments_from_mask(domain):
            seed = _choose_segment_seed(
                guide_valid, segment, receivers, source_x, seed_allowed
            )
            if seed is None:
                continue
            p = _global_dp_segment(
                values,
                guide_valid,
                lags,
                segment,
                seed_receiver=seed,
                epsilon_samples=epsilon_samples,
                max_consecutive_failures=int(max_consecutive_failures),
                seed_allowed=seed_allowed,
            )
            ok = p >= 0
            guide_path[segment[ok]] = p[ok]

    raw_valid = _normalise_valid(measurement, valid)
    if lag_valid is not None:
        raw_valid &= _normalise_valid(measurement, lag_valid)
    if domain is not None:
        raw_valid &= domain[:, None]
        raw_valid &= quality_allowed

    raw_corridor = np.zeros_like(raw_valid, dtype=bool)
    guide_success = np.flatnonzero(guide_path >= 0)
    if guide_success.size:
        offsets = np.arange(-int(raw_refine_radius_samples), int(raw_refine_radius_samples) + 1)
        columns = guide_path[guide_success, None] + offsets
        inside = (columns >= 0) & (columns < lags.size)
        rows = np.broadcast_to(guide_success[:, None], columns.shape)
        raw_corridor[rows[inside], columns[inside]] = True
    raw_corridor &= raw_valid

    path = np.full(nreceiver, -1, dtype=int)
    segment_seeds: list[int] = []
    if domain is None:
        if restrict_seed_lag_range:
            raw_corridor[legacy_seed] &= seed_allowed
        for segment in _valid_receiver_segments(raw_corridor):
            if legacy_seed not in segment:
                continue
            p = _global_dp_segment(
                measurement,
                raw_corridor,
                lags,
                segment,
                seed_receiver=legacy_seed,
                epsilon_samples=epsilon_samples,
            )
            ok = p >= 0
            path[segment[ok]] = p[ok]
        seed_receiver = legacy_seed
        _assert_path_continuity(path, lags, epsilon_samples)
    else:
        for segment in _segments_from_mask(domain):
            seed = _choose_segment_seed(
                raw_corridor, segment, receivers, source_x, seed_allowed
            )
            if seed is None:
                continue
            segment_seeds.append(seed)
            p = _global_dp_segment(
                measurement,
                raw_corridor,
                lags,
                segment,
                seed_receiver=seed,
                epsilon_samples=epsilon_samples,
                max_consecutive_failures=int(max_consecutive_failures),
                seed_allowed=seed_allowed,
            )
            ok = p >= 0
            path[segment[ok]] = p[ok]
        if segment_seeds:
            s = np.asarray(segment_seeds, dtype=int)
            seed_receiver = int(s[np.argmin(np.abs(receivers[s] - float(source_x)))])
        else:
            seed_receiver = legacy_seed
        _assert_gap_continuity(
            path, lags, epsilon_samples, domain, int(max_consecutive_failures)
        )

    success = path >= 0
    shift_samples = np.full(nreceiver, np.nan)
    shift_time = np.full(nreceiver, np.nan)
    tracked_correlation = np.full(nreceiver, np.nan)
    boundary_flag = np.zeros(nreceiver, dtype=bool)
    boundary_limit = float(np.max(np.abs(lags))) - float(boundary_margin_samples)

    for receiver in np.flatnonzero(success):
        index = int(path[receiver])
        safe_row = np.where(raw_corridor[receiver], measurement[receiver], np.nan)
        shift_samples[receiver] = _refine_peak(safe_row, index, lags)
        shift_time[receiver] = shift_samples[receiver] * float(dt)
        tracked_correlation[receiver] = measurement[receiver, index]
        boundary_flag[receiver] = abs(float(lags[index])) >= boundary_limit

    seed_lag = float("nan")
    if success[seed_receiver]:
        seed_lag = float(lags[path[seed_receiver]])

    return TrackingResult(
        path_index=path,
        shift_samples=shift_samples,
        shift_time=shift_time,
        tracked_correlation=tracked_correlation,
        seed_receiver=seed_receiver,
        seed_lag=seed_lag,
        success_mask=success,
        boundary_flag=boundary_flag,
        dt=float(dt),
        epsilon_samples=epsilon_samples,
        boundary_margin_samples=int(boundary_margin_samples),
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
    restrict_seed_lag_range: bool = False,
    receiver_mask: np.ndarray | None = None,
    max_consecutive_failures: int = 3,
    hard_min_correlation: bool = False,
) -> TrackingResult:
    result_valid = correlation_result.valid if valid is None else valid
    result_lag_valid = (
        getattr(correlation_result, "lag_valid", None) if lag_valid is None else lag_valid
    )
    guide = getattr(correlation_result, "tracking_correlation", None)
    if guide is None:
        guide = correlation_result.correlation
    raw = getattr(correlation_result, "waveform_correlation", None)
    if raw is None:
        raw = correlation_result.correlation
    return track_zncc(
        guide,
        correlation_result.lags_samples,
        receiver_x,
        source_x,
        seed_lag_range_time,
        epsilon_time,
        correlation_result.dt,
        measurement_correlation=raw,
        valid=result_valid,
        lag_valid=result_lag_valid,
        min_correlation=min_correlation,
        boundary_margin_samples=boundary_margin_samples,
        boundary_margin_time=boundary_margin_time,
        raw_refine_radius_samples=raw_refine_radius_samples,
        restrict_seed_lag_range=restrict_seed_lag_range,
        receiver_mask=receiver_mask,
        max_consecutive_failures=max_consecutive_failures,
        hard_min_correlation=hard_min_correlation,
    )
