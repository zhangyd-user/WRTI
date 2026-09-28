"""Sparse event tracking with 15-receiver slow-guide main-axis protection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks, hilbert


@dataclass(frozen=True)
class FlatEventCandidate:
    sample: int
    time: float
    amplitude: float
    envelope: float
    polarity: int
    prominence: float
    waveform: np.ndarray


@dataclass(frozen=True)
class SparseFlatTrackingResult:
    pick_sample: np.ndarray
    pick_time: np.ndarray
    candidate_sample: np.ndarray
    success_mask: np.ndarray
    trace_valid: np.ndarray
    neighbor_correlation: np.ndarray
    residual_slope: np.ndarray
    prediction_error: np.ndarray
    candidate_count: np.ndarray
    candidate_count_before_ownership: np.ndarray
    selected_amplitude: np.ndarray
    selected_envelope: np.ndarray
    selected_polarity: np.ndarray
    accumulated_coherence: np.ndarray
    seed_receiver: int
    seed_time: float
    stop_reason_left: str
    stop_reason_right: str
    active_state_max: int
    transition_count: int
    correlation_count: int
    rescue_attempt_count: int
    rescue_success_count: int
    rescue_candidate_count: int
    rescue_transition_count: int
    rescue_used_mask: np.ndarray
    ownership_escape_used_mask: np.ndarray
    ownership_escape_attempt_count: int
    ownership_escape_success_count: int
    ownership_escape_candidate_count: int
    skipped_valid_receiver_mask: np.ndarray
    selected_transition_kind: np.ndarray
    stop_receiver_left: int
    stop_receiver_right: int
    runtime_seconds: float


class TransitionKind(IntEnum):
    SEED = 0
    ANCHOR = 1
    NORMAL = 2
    RESCUE = 3
    GAP = 4


@dataclass
class _PathNode:
    parent_id: int
    receiver: int
    candidate_index: int
    transition_kind: int
    rho: float
    prediction_error: float
    score_delta: float
    cumulative_score: float
    cumulative_prediction_error: float


@dataclass
class _SparseDPState:
    score: float
    prediction_error: float
    envelope: float
    node_id: int
    older_success_node_id: int
    previous_success_node_id: int
    valid_gap_count: int = 0


def _append_node(
    arena,
    *,
    parent_id,
    receiver,
    candidate_index,
    kind,
    rho,
    prediction_error,
    score_delta,
    score,
    cumulative_prediction_error,
):
    arena.append(
        _PathNode(
            parent_id,
            receiver,
            candidate_index,
            int(kind),
            rho,
            prediction_error,
            score_delta,
            score,
            cumulative_prediction_error,
        )
    )
    return len(arena) - 1


def _node_lineage(arena, node_id):
    result = []
    while node_id >= 0:
        result.append(node_id)
        node_id = arena[node_id].parent_id
    return list(reversed(result))


def _better(candidate: _SparseDPState, current: _SparseDPState | None) -> bool:
    if current is None:
        return True
    tolerance = 1e-12
    if candidate.score > current.score + tolerance:
        return True
    if abs(candidate.score - current.score) <= tolerance:
        if candidate.prediction_error < current.prediction_error - tolerance:
            return True
        if abs(candidate.prediction_error - current.prediction_error) <= tolerance:
            return candidate.envelope > current.envelope
    return False


def _normalise_traces(flat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(flat, dtype=float)
    scale = np.sqrt(np.mean(flat * flat, axis=1))
    good = valid & np.isfinite(scale) & (scale > np.finfo(float).eps)
    result[good] = flat[good] / scale[good, None]
    return result


def _candidate_waveform(trace: np.ndarray, sample: int, half: int) -> np.ndarray | None:
    if sample - half < 0 or sample + half >= trace.size:
        return None
    waveform = np.array(trace[sample - half : sample + half + 1], copy=True)
    waveform -= np.mean(waveform)
    norm = float(np.linalg.norm(waveform))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return waveform / norm


def _build_candidates(
    traces: np.ndarray,
    valid: np.ndarray,
    *,
    dt: float,
    t0: float,
    min_distance_samples: int,
    min_prominence: float,
    max_candidates: int,
    half_window_samples: int,
    seed_receiver: int,
    seed_sample: int,
    seed_time: float,
    state_valid: np.ndarray | None,
) -> tuple[list[list[FlatEventCandidate]], np.ndarray]:
    envelope = np.abs(hilbert(traces, axis=-1))
    candidates: list[list[FlatEventCandidate]] = []
    before = np.zeros(traces.shape[0], dtype=int)

    for receiver, trace in enumerate(traces):
        if not valid[receiver]:
            candidates.append([])
            continue

        peaks, properties = find_peaks(
            trace,
            distance=max(1, int(min_distance_samples)),
            prominence=float(min_prominence),
        )
        if peaks.size:
            positive = trace[peaks] > 0.0
            peaks = peaks[positive]
            if "prominences" in properties:
                properties["prominences"] = np.asarray(properties["prominences"])[positive]

        if receiver == seed_receiver:
            peaks = np.asarray([seed_sample], dtype=int)
            prominences = np.asarray([np.inf])
        else:
            prominences = np.asarray(properties.get("prominences", np.zeros(peaks.size)))
            if int(max_candidates) > 0 and peaks.size > int(max_candidates):
                keep = np.argsort(prominences)[-int(max_candidates) :]
                peaks, prominences = peaks[keep], prominences[keep]

        before[receiver] = peaks.size
        if state_valid is not None and peaks.size:
            keep = state_valid[receiver, peaks]
            peaks, prominences = peaks[keep], prominences[keep]

        row = []
        for sample, prominence in sorted(zip(peaks, prominences)):
            waveform = _candidate_waveform(trace, int(sample), half_window_samples)
            if waveform is None:
                continue
            amplitude = float(trace[sample])
            candidate_time = (
                float(seed_time)
                if receiver == seed_receiver and int(sample) == int(seed_sample)
                else float(t0) + int(sample) * float(dt)
            )
            row.append(
                FlatEventCandidate(
                    sample=int(sample),
                    time=candidate_time,
                    amplitude=amplitude,
                    envelope=float(envelope[receiver, sample]),
                    polarity=int(np.sign(amplitude)),
                    prominence=float(prominence),
                    waveform=waveform,
                )
            )
        candidates.append(row)

    return candidates, before, envelope


def _stop_reason(rejected: dict[str, int], has_candidates: bool, *, hard_zncc_gate: bool) -> str:
    if not has_candidates:
        return "NO_CANDIDATE"
    if rejected.get("SLOPE_REJECTED", 0) and not rejected.get("PREDICTION_HARD_REJECTED", 0):
        return f"SLOPE_REJECTED:n={rejected['SLOPE_REJECTED']}"
    if rejected.get("PREDICTION_HARD_REJECTED", 0):
        return (
            f"NO_FEASIBLE_TRANSITION:slope={rejected.get('SLOPE_REJECTED', 0)}:"
            f"prediction_hard={rejected['PREDICTION_HARD_REJECTED']}"
        )
    if hard_zncc_gate and rejected.get("CORRELATION_REJECTED", 0):
        return "CORRELATION_REJECTED"
    return "NO_FEASIBLE_TRANSITION"


def _positive_local_maxima(trace: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return positive local maxima in [lo, hi], without distance/prominence pruning."""
    n = int(trace.size)
    lo = max(1, int(lo))
    hi = min(n - 2, int(hi))
    if hi < lo:
        return np.empty(0, dtype=int)
    samples = np.arange(lo, hi + 1, dtype=int)
    values = trace[samples]
    keep = (
        np.isfinite(values)
        & (values > 0.0)
        & (values >= trace[samples - 1])
        & (values > trace[samples + 1])
    )
    return samples[keep]


def _track_direction(
    candidates: list[list[FlatEventCandidate]],
    receiver_x: np.ndarray,
    order: np.ndarray,
    *,
    candidate_envelope: np.ndarray,
    max_slope: float,
    max_prediction_error: float,
    min_zncc: float,
    hard_zncc_gate: bool,
    prediction_soft_scale: float,
    prediction_penalty_weight: float,
    max_active_states: int,
    phase_switch_zncc: float,
    phase_switch_prediction_error: float,
    traces: np.ndarray,
    state_valid: np.ndarray | None,
    dt: float,
    t0: float,
    half_window_samples: int,
    rescue_enabled: bool,
    rescue_half_width: float,
    rescue_max_parent_states: int,
    rescue_min_prominence: float,
    max_valid_receiver_gap: int,
    valid_receiver: np.ndarray,
    anchor_mask: np.ndarray,
    escape_state_valid: np.ndarray | None,
    escape_enabled: bool,
    escape_min_zncc: float,
    escape_max_prediction_error: float,
    slow_guide_enabled: bool,
    slow_guide_memory_receivers: int,
    slow_guide_candidate_half_width: float,
    slow_guide_max_added_candidates: int,
    slow_guide_score_weight: float,
    slow_guide_soft_scale: float,
    slow_guide_update_max_error: float,
    slow_guide_update_min_zncc: float,
    slow_guide_update_max_prediction_error: float,
    slow_guide_update_consensus_time: float,
    slow_guide_update_confirm_receivers: int,
    slow_guide_protected_fast_weight_scale: float,
) -> tuple[dict[int, int], str, int, int, int, dict[str, object]]:
    """Track one side with local DP and one shared long-memory guide."""
    diagnostic = {
        "attempts": 0,
        "successes": 0,
        "candidates": 0,
        "rescue_transitions": 0,
        "rescue_mask": np.zeros(valid_receiver.size, dtype=bool),
        "gap_mask": np.zeros(valid_receiver.size, dtype=bool),
        "transition_kind": np.full(valid_receiver.size, -1, dtype=np.int8),
        "stop_receiver": -1,
        "escape_attempts": 0,
        "escape_successes": 0,
        "escape_candidates": 0,
        "guide_candidates": 0,
        "guide_updates": 0,
        "guide_freezes": 0,
        "guide_protected_transitions": 0,
    }
    if order.size <= 1:
        return {int(order[0]): 0}, "END_OF_GATHER", 0, 0, 0, diagnostic

    seed = int(order[0])
    available_order = [seed]
    for receiver in order[1:]:
        receiver = int(receiver)
        if not valid_receiver[receiver]:
            break
        available_order.append(receiver)
    natural_end_reason = (
        "END_OF_GATHER"
        if len(available_order) == int(order.size)
        else "END_OF_VALID_GATHER"
    )

    arena: list[_PathNode] = []
    seed_node = _append_node(
        arena,
        parent_id=-1,
        receiver=seed,
        candidate_index=0,
        kind=TransitionKind.SEED,
        rho=np.nan,
        prediction_error=np.nan,
        score_delta=0.0,
        score=0.0,
        cumulative_prediction_error=0.0,
    )
    states: dict[tuple, _SparseDPState] = {}
    transitions = correlations = 0
    active_max = 0

    seed_candidate = candidates[seed][0]
    guide_receivers = [seed]
    guide_times = [float(seed_candidate.time)]
    guide_pending = []

    def _state_sort_key(state: _SparseDPState):
        return (state.score, -state.prediction_error, state.envelope, -state.node_id)

    def _best_state(values):
        values = list(values)
        if not values:
            return None
        return max(values, key=_state_sort_key)

    def _prune(source_states, limit):
        if len(source_states) <= limit:
            return source_states
        ranked = sorted(
            source_states.items(), key=lambda item: _state_sort_key(item[1]), reverse=True
        )
        return dict(ranked[:limit])

    def _append_guide_candidates(target_receiver, predicted_times):
        """Add weak positive peaks near the slow-axis prediction before normal DP.

        This intentionally ignores the global candidate ``distance`` and
        ``prominence`` settings.  It adds only a few peaks close to the main
        axis, so a strong diffraction peak cannot delete a weaker target peak
        before the DP gets a chance to compare them.
        """
        if not slow_guide_enabled or anchor_mask[int(target_receiver)]:
            return 0
        predicted_times = np.asarray(predicted_times, dtype=float)
        predicted_times = predicted_times[np.isfinite(predicted_times)]
        if not predicted_times.size:
            return 0
        # Deduplicate nearly identical guide predictions from many parent states.
        predicted_samples = np.unique(
            np.rint((predicted_times - float(t0)) / float(dt)).astype(int)
        )
        half = max(1, int(round(float(slow_guide_candidate_half_width) / float(dt))))
        trace = traces[int(target_receiver)]
        allowed = escape_state_valid if escape_enabled else state_valid
        existing = {int(item.sample) for item in candidates[int(target_receiver)]}
        ranked_samples = {}
        for center in predicted_samples:
            peaks = _positive_local_maxima(trace, int(center) - half, int(center) + half)
            for sample in peaks:
                sample = int(sample)
                if sample in existing:
                    continue
                if allowed is not None and not bool(allowed[int(target_receiver), sample]):
                    continue
                distance = int(np.min(np.abs(predicted_samples - sample)))
                # Primary rank is closeness to the long-memory axis.  Amplitude
                # is only a tie break; otherwise the strong diffraction would
                # defeat the purpose of preserving a weak target peak.
                rank = (distance, -float(trace[sample]))
                old = ranked_samples.get(sample)
                if old is None or rank < old:
                    ranked_samples[sample] = rank
        if not ranked_samples:
            return 0
        keep = [
            sample
            for sample, _ in sorted(ranked_samples.items(), key=lambda item: item[1])[
                : max(1, int(slow_guide_max_added_candidates))
            ]
        ]
        added = 0
        for sample in sorted(keep):
            waveform = _candidate_waveform(trace, sample, half_window_samples)
            if waveform is None:
                continue
            candidates[int(target_receiver)].append(
                FlatEventCandidate(
                    sample=sample,
                    time=float(t0) + sample * float(dt),
                    amplitude=float(trace[sample]),
                    envelope=float(candidate_envelope[int(target_receiver), sample]),
                    polarity=1,
                    prominence=0.0,
                    waveform=waveform,
                )
            )
            added += 1
        if added:
            # Safe here: this target receiver has not yet created any path node,
            # so sorting cannot invalidate an existing candidate index.
            candidates[int(target_receiver)].sort(key=lambda item: item.time)
            diagnostic["guide_candidates"] += added
        return added

    def _shared_guide_prediction(target_receiver):
        if not slow_guide_enabled or not guide_receivers:
            return np.nan
        receivers = np.asarray(
            guide_receivers[-slow_guide_memory_receivers:], dtype=int
        )
        times = np.asarray(guide_times[-slow_guide_memory_receivers:], dtype=float)
        good = np.isfinite(times) & np.isfinite(receiver_x[receivers])
        receivers = receivers[good]
        times = times[good]
        if times.size == 0:
            return np.nan
        if times.size == 1:
            return float(times[-1])
        xx = receiver_x[receivers].astype(float)
        x_target = float(receiver_x[int(target_receiver)])
        x0 = float(np.mean(xx))
        scale = float(np.max(np.abs(xx - x0)))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            return float(np.median(times))
        try:
            coefficient = np.polyfit((xx - x0) / scale, times, deg=1)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return float(times[-1])
        return float(np.polyval(coefficient, (x_target - x0) / scale))

    def _guide_penalty(target_receiver, current_time):
        predicted = _shared_guide_prediction(target_receiver)
        if not np.isfinite(predicted):
            return 0.0, predicted
        scale = max(float(slow_guide_soft_scale), np.finfo(float).eps)
        error = float(current_time) - float(predicted)
        penalty = float(slow_guide_score_weight) * (error / scale) ** 2
        return float(penalty), float(predicted)

    def _state_current_time(state):
        node = arena[state.previous_success_node_id]
        if node.candidate_index < 0:
            return np.nan
        return float(candidates[int(node.receiver)][int(node.candidate_index)].time)

    def _freeze_guide():
        guide_pending.clear()
        diagnostic["guide_freezes"] += 1

    def _maybe_update_shared_guide(current_receiver, current_states):
        if not slow_guide_enabled or not current_states:
            return
        ranked = sorted(
            current_states.values(), key=_state_sort_key, reverse=True
        )[:4]
        best = ranked[0]
        best_node = arena[best.previous_success_node_id]
        best_time = _state_current_time(best)
        if best_node.transition_kind != TransitionKind.NORMAL:
            _freeze_guide()
            return
        predicted_guide = _shared_guide_prediction(current_receiver)
        node_quality_ok = bool(
            np.isfinite(best_node.rho)
            and best_node.rho >= slow_guide_update_min_zncc
            and np.isfinite(best_node.prediction_error)
            and abs(best_node.prediction_error)
            <= slow_guide_update_max_prediction_error
        )
        guide_error_ok = bool(
            np.isfinite(predicted_guide)
            and np.isfinite(best_time)
            and abs(best_time - predicted_guide) <= slow_guide_update_max_error
        )
        ranked_times = np.asarray(
            [_state_current_time(state) for state in ranked], dtype=float
        )
        ranked_times = ranked_times[np.isfinite(ranked_times)]
        consensus_ok = bool(
            ranked_times.size >= 1
            and (
                ranked_times.size == 1
                or np.max(np.abs(ranked_times - best_time))
                <= slow_guide_update_consensus_time
            )
        )
        center = int(round((predicted_guide - t0) / dt)) if np.isfinite(predicted_guide) else -1
        half = max(1, int(round(slow_guide_candidate_half_width / dt)))
        local_peaks = (
            _positive_local_maxima(
                traces[int(current_receiver)], center - half, center + half
            )
            if center >= 0
            else np.empty(0, dtype=int)
        )
        local_ambiguous = local_peaks.size >= 2
        if not (node_quality_ok and guide_error_ok and consensus_ok and not local_ambiguous):
            _freeze_guide()
            return
        guide_pending.append((int(current_receiver), float(best_time)))
        if len(guide_pending) < slow_guide_update_confirm_receivers:
            return
        for receiver_value, time_value in guide_pending:
            guide_receivers.append(receiver_value)
            guide_times.append(time_value)
        while len(guide_receivers) > slow_guide_memory_receivers:
            guide_receivers.pop(0)
            guide_times.pop(0)
        diagnostic["guide_updates"] += len(guide_pending)
        guide_pending.clear()

    def _child_state(
        state,
        *,
        target_receiver,
        current,
        current_index,
        transition_kind,
        rho,
        prediction_error,
        score_delta,
        score,
        cumulative_prediction,
    ):
        node_id = _append_node(
            arena,
            parent_id=state.node_id,
            receiver=int(target_receiver),
            candidate_index=int(current_index),
            kind=transition_kind,
            rho=float(rho),
            prediction_error=float(prediction_error),
            score_delta=float(score_delta),
            score=float(score),
            cumulative_prediction_error=float(cumulative_prediction),
        )
        return _SparseDPState(
            float(score),
            float(cumulative_prediction),
            state.envelope + current.envelope,
            node_id,
            state.previous_success_node_id,
            node_id,
            0,
        )

    def _advance_at(
        source_states,
        target_receiver,
        transition_kind=TransitionKind.NORMAL,
        *,
        allowed_samples=None,
        ownership_mode="base",
    ):
        nonlocal transitions, correlations
        next_states: dict[tuple, _SparseDPState] = {}
        rejected = {
            "SLOPE_REJECTED": 0,
            "PREDICTION_HARD_REJECTED": 0,
            "CORRELATION_REJECTED": 0,
        }
        row = candidates[int(target_receiver)]
        if not row:
            return next_states, rejected
        current_times = np.asarray([item.time for item in row])
        shared_guide_time = _shared_guide_prediction(target_receiver)

        def _is_guide_protected(candidate_time):
            return bool(
                slow_guide_enabled
                and transition_kind == TransitionKind.NORMAL
                and ownership_mode == "base"
                and np.isfinite(shared_guide_time)
                and abs(float(candidate_time) - float(shared_guide_time))
                <= float(slow_guide_candidate_half_width)
            )

        allowed_samples_set = (
            None if allowed_samples is None else set(int(v) for v in allowed_samples)
        )
        for state in source_states.values():
            older_node = arena[state.older_success_node_id]
            previous_node = arena[state.previous_success_node_id]
            older_receiver = int(older_node.receiver)
            previous_receiver = int(previous_node.receiver)
            older_index = int(older_node.candidate_index)
            previous_index = int(previous_node.candidate_index)
            dx_previous = float(receiver_x[previous_receiver] - receiver_x[older_receiver])
            dx_current = float(receiver_x[int(target_receiver)] - receiver_x[previous_receiver])
            if dx_previous == 0.0 or dx_current == 0.0:
                continue
            older = candidates[older_receiver][older_index]
            previous = candidates[previous_receiver][previous_index]
            lower = previous.time - max_slope * abs(dx_current)
            upper = previous.time + max_slope * abs(dx_current)
            if allowed_samples_set is None:
                normal_lo = int(np.searchsorted(current_times, lower, side="left"))
                normal_hi = int(np.searchsorted(current_times, upper, side="right"))
                normal_indices = set(range(normal_lo, normal_hi))
                guide_indices = set()
                if (
                    transition_kind == TransitionKind.NORMAL
                    and ownership_mode == "base"
                    and slow_guide_enabled
                    and np.isfinite(shared_guide_time)
                ):
                    guide_lower = shared_guide_time - slow_guide_candidate_half_width
                    guide_upper = shared_guide_time + slow_guide_candidate_half_width
                    guide_lo = int(
                        np.searchsorted(current_times, guide_lower, side="left")
                    )
                    guide_hi = int(
                        np.searchsorted(current_times, guide_upper, side="right")
                    )
                    guide_indices.update(range(guide_lo, guide_hi))
                candidate_indices = sorted(normal_indices | guide_indices)
                if not candidate_indices:
                    rejected["SLOPE_REJECTED"] += 1
                    continue
            else:
                candidate_indices = [
                    idx
                    for idx, item in enumerate(row)
                    if int(item.sample) in allowed_samples_set
                    and lower <= item.time <= upper
                ]
                if not candidate_indices:
                    rejected["SLOPE_REJECTED"] += 1
                    continue
            previous_slope = (previous.time - older.time) / dx_previous
            predicted = previous.time + previous_slope * dx_current
            for current_index in candidate_indices:
                current = row[current_index]
                in_base = state_valid is None or bool(
                    state_valid[int(target_receiver), current.sample]
                )
                if ownership_mode == "base" and not in_base:
                    continue
                if ownership_mode == "escape" and in_base:
                    continue
                prediction_error = abs(current.time - predicted)
                guide_protected = _is_guide_protected(current.time)
                if prediction_error > max_prediction_error and not guide_protected:
                    rejected["PREDICTION_HARD_REJECTED"] += 1
                    continue
                rho = float(np.dot(previous.waveform, current.waveform))
                correlations += 1
                if not in_base and (
                    rho < escape_min_zncc
                    or prediction_error > escape_max_prediction_error
                ):
                    continue
                if (
                    rho < phase_switch_zncc
                    and prediction_error > phase_switch_prediction_error
                ):
                    rejected["CORRELATION_REJECTED"] += 1
                    continue
                if hard_zncc_gate and rho < min_zncc:
                    rejected["CORRELATION_REJECTED"] += 1
                    continue
                transitions += 1
                if guide_protected and prediction_error > max_prediction_error:
                    diagnostic["guide_protected_transitions"] += 1
                guide_penalty, _ = _guide_penalty(target_receiver, current.time)
                effective_fast_weight = prediction_penalty_weight * (
                    slow_guide_protected_fast_weight_scale
                    if guide_protected
                    else 1.0
                )
                fast_prediction_penalty = effective_fast_weight * (
                    prediction_error / prediction_soft_scale
                ) ** 2
                score_delta = (
                    rho
                    - fast_prediction_penalty
                    - guide_penalty
                )
                score = state.score + score_delta
                cumulative_prediction = state.prediction_error + prediction_error
                candidate_state = _child_state(
                    state,
                    target_receiver=int(target_receiver),
                    current=current,
                    current_index=current_index,
                    transition_kind=transition_kind,
                    rho=rho,
                    prediction_error=prediction_error,
                    score_delta=score_delta,
                    score=score,
                    cumulative_prediction=cumulative_prediction,
                )
                key = (previous_receiver, previous_index, current_index)
                if _better(candidate_state, next_states.get(key)):
                    next_states[key] = candidate_state
        return next_states, rejected

    def _rescue_samples(source_states, target_receiver):
        ranked_parents = sorted(
            source_states.values(), key=_state_sort_key, reverse=True
        )[:rescue_max_parent_states]
        half = max(1, int(round(rescue_half_width / dt)))
        all_positive_peaks, _ = find_peaks(
            traces[int(target_receiver)], prominence=rescue_min_prominence
        )
        if all_positive_peaks.size:
            all_positive_peaks = all_positive_peaks[
                traces[int(target_receiver), all_positive_peaks] > 0.0
            ]
        result = set()
        for state in ranked_parents:
            older_node = arena[state.older_success_node_id]
            previous_node = arena[state.previous_success_node_id]
            older_receiver = int(older_node.receiver)
            previous_receiver = int(previous_node.receiver)
            older = candidates[older_receiver][older_node.candidate_index]
            previous = candidates[previous_receiver][previous_node.candidate_index]
            dx = float(receiver_x[previous_receiver] - receiver_x[older_receiver])
            if dx == 0.0:
                continue
            slope_value = (previous.time - older.time) / dx
            predicted = previous.time + slope_value * (
                receiver_x[int(target_receiver)] - receiver_x[previous_receiver]
            )
            center = int(round((predicted - t0) / dt))
            lo_sample = max(0, center - half)
            hi_sample = min(traces.shape[1], center + half + 1)
            inside = all_positive_peaks[
                (all_positive_peaks >= lo_sample)
                & (all_positive_peaks < hi_sample)
            ]
            for sample in inside:
                sample = int(sample)
                allowed = escape_state_valid if escape_enabled else state_valid
                if allowed is not None and not allowed[int(target_receiver), sample]:
                    continue
                result.add(sample)
        return result

    def _append_rescue_candidates(target_receiver, rescue_samples):
        existing = {int(item.sample) for item in candidates[int(target_receiver)]}
        new_samples = sorted(set(int(v) for v in rescue_samples) - existing)
        if not new_samples:
            return 0
        added = 0
        for sample in new_samples:
            if traces[int(target_receiver), sample] <= 0.0:
                continue
            allowed = escape_state_valid if escape_enabled else state_valid
            if allowed is not None and not allowed[int(target_receiver), sample]:
                continue
            waveform = _candidate_waveform(
                traces[int(target_receiver)], sample, half_window_samples
            )
            if waveform is None:
                continue
            candidates[int(target_receiver)].append(
                FlatEventCandidate(
                    sample,
                    float(t0 + sample * dt),
                    float(traces[int(target_receiver), sample]),
                    float(candidate_envelope[int(target_receiver), sample]),
                    1,
                    0.0,
                    waveform,
                )
            )
            added += 1
        diagnostic["candidates"] += added
        return added

    def _gap_states(source_states, target_receiver):
        result = {}
        for state in source_states.values():
            if state.valid_gap_count >= max_valid_receiver_gap:
                continue
            node_id = _append_node(
                arena,
                parent_id=state.node_id,
                receiver=int(target_receiver),
                candidate_index=-1,
                kind=TransitionKind.GAP,
                rho=np.nan,
                prediction_error=np.nan,
                score_delta=0.0,
                score=state.score,
                cumulative_prediction_error=state.prediction_error,
            )
            gap_state = _SparseDPState(
                state.score,
                state.prediction_error,
                state.envelope,
                node_id,
                state.older_success_node_id,
                state.previous_success_node_id,
                state.valid_gap_count + 1,
            )
            result[state.node_id] = gap_state
        return result

    # First receiver. Add weak peaks close to the shared seed-axis prediction
    # before normal candidate selection.
    first = int(order[1])
    dx = float(receiver_x[first] - receiver_x[seed])
    if dx == 0.0:
        return {seed: 0}, "ZERO_RECEIVER_SPACING", 0, 0, 0, diagnostic
    if slow_guide_enabled and not anchor_mask[first]:
        _append_guide_candidates(first, [seed_candidate.time])
    first_indices = [
        index
        for index, candidate in enumerate(candidates[first])
        if state_valid is None or state_valid[first, candidate.sample]
    ]
    if not first_indices:
        diagnostic["stop_receiver"] = first
        return {seed: 0}, "NO_POSITIVE_CANDIDATE", 0, 0, 0, diagnostic

    rejected = {
        "SLOPE_REJECTED": 0,
        "PREDICTION_HARD_REJECTED": 0,
        "CORRELATION_REJECTED": 0,
    }
    seed_state = _SparseDPState(
        score=0.0,
        prediction_error=0.0,
        envelope=seed_candidate.envelope,
        node_id=seed_node,
        older_success_node_id=seed_node,
        previous_success_node_id=seed_node,
        valid_gap_count=0,
    )
    for current_index in first_indices:
        current = candidates[first][current_index]
        slope = (current.time - seed_candidate.time) / dx
        guide_prediction = _shared_guide_prediction(first)
        guide_protected = bool(
            slow_guide_enabled
            and np.isfinite(guide_prediction)
            and abs(current.time - guide_prediction)
            <= slow_guide_candidate_half_width
        )
        if not anchor_mask[first] and abs(slope) > max_slope and not guide_protected:
            rejected["SLOPE_REJECTED"] += 1
            continue
        rho = float(np.dot(seed_candidate.waveform, current.waveform))
        correlations += 1
        guide_penalty = 0.0
        if not anchor_mask[first]:
            guide_penalty, _ = _guide_penalty(first, current.time)
        score_delta = rho - guide_penalty
        transitions += 1
        candidate_state = _child_state(
            seed_state,
            target_receiver=first,
            current=current,
            current_index=current_index,
            transition_kind=(
                TransitionKind.ANCHOR if anchor_mask[first] else TransitionKind.NORMAL
            ),
            rho=rho,
            prediction_error=0.0,
            score_delta=score_delta,
            score=score_delta,
            cumulative_prediction=0.0,
        )
        states[(0, current_index)] = candidate_state

    if not states:
        return (
            {seed: 0},
            _stop_reason(
                rejected,
                bool(candidates[first]),
                hard_zncc_gate=hard_zncc_gate,
            ),
            transitions,
            correlations,
            0,
            diagnostic,
        )
    active_max = len(states)
    if anchor_mask[first]:
        guide_receivers.append(first)
        guide_times.append(float(candidates[first][0].time))
        guide_pending.clear()
    else:
        _maybe_update_shared_guide(first, states)

    stop_reason = natural_end_reason
    local = 2
    while local < order.size:
        current_receiver = int(order[local])
        if not valid_receiver[current_receiver]:
            remaining = order[local + 1 :]
            if remaining.size == 0 or not np.any(valid_receiver[remaining]):
                stop_reason = "END_OF_VALID_GATHER"
                diagnostic["stop_receiver"] = -1
            else:
                stop_reason = "INVALID_RECEIVER_GAP"
                diagnostic["stop_receiver"] = current_receiver
            break

        # Fixed pilot anchor remains immutable and is used as trusted guide
        # training data.  The guide never changes which anchor sample is used.
        if anchor_mask[current_receiver]:
            next_states = {}
            if not candidates[current_receiver]:
                diagnostic["stop_receiver"] = current_receiver
                stop_reason = "INVALID_ANCHOR_CANDIDATE"
                break
            current_index = 0
            current = candidates[current_receiver][current_index]
            for state in states.values():
                older_node = arena[state.older_success_node_id]
                previous_node = arena[state.previous_success_node_id]
                older_receiver = int(older_node.receiver)
                previous_receiver = int(previous_node.receiver)
                older = candidates[older_receiver][older_node.candidate_index]
                previous = candidates[previous_receiver][previous_node.candidate_index]
                dx_previous = float(
                    receiver_x[previous_receiver] - receiver_x[older_receiver]
                )
                dx_current = float(
                    receiver_x[current_receiver] - receiver_x[previous_receiver]
                )
                if dx_previous == 0.0 or dx_current == 0.0:
                    continue
                previous_slope = (previous.time - older.time) / dx_previous
                predicted = previous.time + previous_slope * dx_current
                prediction_error = abs(current.time - predicted)
                rho = float(np.dot(previous.waveform, current.waveform))
                correlations += 1
                transitions += 1
                score_delta = rho - prediction_penalty_weight * (
                    prediction_error / prediction_soft_scale
                ) ** 2
                score = state.score + score_delta
                cumulative_prediction = state.prediction_error + prediction_error
                forced = _child_state(
                    state,
                    target_receiver=current_receiver,
                    current=current,
                    current_index=current_index,
                    transition_kind=TransitionKind.ANCHOR,
                    rho=rho,
                    prediction_error=prediction_error,
                    score_delta=score_delta,
                    score=score,
                    cumulative_prediction=cumulative_prediction,
                )
                key = (
                    previous_receiver,
                    previous_node.candidate_index,
                    current_index,
                )
                if _better(forced, next_states.get(key)):
                    next_states[key] = forced
            if not next_states:
                diagnostic["stop_receiver"] = current_receiver
                stop_reason = "INVALID_ANCHOR_TRANSITION"
                break
            states = _prune(next_states, max_active_states)
            guide_receivers.append(current_receiver)
            guide_times.append(float(current.time))
            while len(guide_receivers) > slow_guide_memory_receivers:
                guide_receivers.pop(0)
                guide_times.pop(0)
            guide_pending.clear()
            active_max = max(active_max, len(states))
            local += 1
            continue

        # Main-axis protection runs before ordinary DP.  Therefore a weak peak
        # around the 15-receiver guide remains available even when the initial
        # find_peaks(distance=...) retained only a stronger nearby diffraction.
        if slow_guide_enabled:
            guide_prediction = _shared_guide_prediction(current_receiver)
            if np.isfinite(guide_prediction):
                _append_guide_candidates(current_receiver, [guide_prediction])

        next_states, rejected = _advance_at(
            states, current_receiver, TransitionKind.NORMAL
        )

        used_non_normal = False
        if not next_states and escape_enabled:
            diagnostic["escape_attempts"] += 1
            diagnostic["escape_candidates"] += sum(
                escape_state_valid[current_receiver, item.sample]
                and not state_valid[current_receiver, item.sample]
                for item in candidates[current_receiver]
            )
            next_states, rejected = _advance_at(
                states,
                current_receiver,
                TransitionKind.NORMAL,
                ownership_mode="escape",
            )
            if next_states:
                diagnostic["escape_successes"] += 1
                used_non_normal = True

        rescue_attempted = False
        rescue_added = 0
        if not next_states and rescue_enabled:
            rescue_attempted = True
            diagnostic["attempts"] += 1
            samples = _rescue_samples(states, current_receiver)
            rescue_added = _append_rescue_candidates(current_receiver, samples)
            before_transitions = transitions
            next_states, rejected = _advance_at(
                states,
                current_receiver,
                TransitionKind.RESCUE,
                allowed_samples=samples,
                ownership_mode="any",
            )
            diagnostic["rescue_transitions"] += transitions - before_transitions
            if next_states:
                diagnostic["successes"] += 1
                used_non_normal = True

        if not next_states:
            gap = (
                _gap_states(states, current_receiver)
                if max_valid_receiver_gap > 0
                else {}
            )
            if gap:
                states = gap
                _freeze_guide()
                active_max = max(active_max, len(states))
                local += 1
                continue
            if rescue_attempted:
                stop_reason = (
                    "RESCUE_NO_POSITIVE_PEAK"
                    if rescue_added == 0
                    else "RESCUE_TRANSITION_FAILED"
                )
            elif max_valid_receiver_gap > 0:
                stop_reason = "VALID_GAP_LIMIT_REACHED"
            else:
                stop_reason = _stop_reason(
                    rejected,
                    bool(candidates[current_receiver]),
                    hard_zncc_gate=hard_zncc_gate,
                )
            diagnostic["stop_receiver"] = current_receiver
            break

        states = _prune(next_states, max_active_states)
        if used_non_normal:
            _freeze_guide()
        else:
            _maybe_update_shared_guide(current_receiver, states)
        active_max = max(active_max, len(states))
        local += 1

    best = _best_state(states.values())
    selected = {seed: 0}
    if best is not None:
        lineage = [arena[node_id] for node_id in _node_lineage(arena, best.node_id)]
        for node in lineage:
            if node.candidate_index >= 0:
                selected[int(node.receiver)] = int(node.candidate_index)
        diagnostic["rescue_mask"][:] = False
        diagnostic["gap_mask"][:] = False
        diagnostic["transition_kind"][:] = -1
        for node in lineage:
            diagnostic["transition_kind"][node.receiver] = node.transition_kind
            if node.transition_kind == TransitionKind.RESCUE:
                diagnostic["rescue_mask"][node.receiver] = True
            elif node.transition_kind == TransitionKind.GAP:
                diagnostic["gap_mask"][node.receiver] = True

    return selected, stop_reason, transitions, correlations, active_max, diagnostic


def track_flattened_event_sparse(
    flat_gather,
    *,
    receiver_x,
    valid_receiver,
    seed_receiver,
    seed_time,
    state_valid=None,
    escape_state_valid=None,
    dt,
    t0=0.0,
    candidate_min_distance_time=0.015,
    candidate_min_prominence=0.05,
    max_candidates_per_trace=0,
    coherence_half_window_time=0.040,
    min_neighbor_zncc=0.50,
    hard_neighbor_zncc_gate=False,
    max_residual_slope_ms_per_100m=150.0,
    prediction_soft_scale_time=0.010,
    prediction_penalty_weight=0.20,
    max_prediction_error_time=0.050,
    max_active_states=2000,
    phase_switch_zncc=0.0,
    phase_switch_prediction_time=0.015,
    anchor_receiver_mask=None,
    anchor_pick_sample=None,
    anchor_pick_time=None,
    continuation_rescue=None,
    ownership_escape=None,
    slow_guide=None,
) -> SparseFlatTrackingResult:
    """Continue one event with local second-order DP plus an optional shared
    long-memory main-axis guide.

    The ordinary two-successful-pick predictor remains the local motion model.
    When the shared guide is enabled:

    * weak positive peaks near the 15-receiver main-axis prediction are inserted
      before DP, bypassing global distance/prominence peak pruning;
    * all DP states in one tracking direction share the same guide, so a competing
      path cannot train its own reflector identity model;
    * candidates close to the shared guide may survive the local two-point hard
      prediction corridor, while the local prediction still contributes a reduced
      soft penalty;
    * the guide is updated only from unambiguous, high-quality NORMAL continuation
      that remains consistent for several receivers. It freezes through ambiguous
      multi-peak regions, rescue, gaps, and ownership escape.

    This makes the local two-point predictor a short-scale motion reference and
    the 15-receiver guide the long-scale reflector-identity reference.
    """
    started = perf_counter()
    flat = np.asarray(flat_gather, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    nr, nt = flat.shape
    seed = int(seed_receiver)
    seed_sample = int(np.rint((float(seed_time) - float(t0)) / float(dt)))
    if x.shape != (nr,) or valid.shape != (nr,) or not 0 <= seed < nr:
        raise ValueError("sparse flat-event geometry is inconsistent")
    if not 0 <= seed_sample < nt or not valid[seed]:
        raise ValueError("sparse flat-event seed is outside valid data")

    ownership = None if state_valid is None else np.asarray(state_valid, dtype=bool)
    if ownership is not None:
        if ownership.shape != flat.shape:
            raise ValueError("state_valid must have the same shape as flat_gather")
        ownership = ownership.copy()
        if not ownership[seed, seed_sample]:
            raise ValueError("selected observed seed lies outside state_valid/ownership")

    escape = dict(ownership_escape or {})
    escape_enabled = bool(escape.get("enabled", False))
    escape_ownership = (
        None
        if escape_state_valid is None
        else np.asarray(escape_state_valid, dtype=bool)
    )
    if escape_enabled:
        if (
            ownership is None
            or escape_ownership is None
            or escape_ownership.shape != flat.shape
        ):
            raise ValueError("ownership escape requires base and escape state_valid masks")
        if np.any(ownership & ~escape_ownership):
            raise ValueError("escape_state_valid must contain base state_valid")

    guide = dict(slow_guide or {})
    default_slow_enabled = bool(
        anchor_receiver_mask is not None
        and np.any(np.asarray(anchor_receiver_mask, dtype=bool))
    )
    slow_enabled = bool(guide.get("enabled", default_slow_enabled))
    slow_memory = int(guide.get("memory_receivers", 15))
    if slow_memory < 2:
        raise ValueError("slow_guide.memory_receivers must be at least 2")
    slow_candidate_half = float(guide.get("candidate_half_width_time", 0.012))
    # Retained as a no-op read for configuration compatibility with the former
    # per-state guide implementation.
    int(guide.get("max_parent_states", 4))
    slow_added_candidates = int(guide.get("max_added_candidates", 4))
    # Increased from the earlier proposed 0.40: at full confidence a 10 ms
    # guide departure now costs 0.65 score units, enough to compete with a high
    # local ZNCC without imposing a hard cutoff.
    slow_weight = float(guide.get("score_weight", 0.65))
    slow_scale = float(guide.get("soft_scale_time", 0.010))
    slow_update_error = float(guide.get("update_max_error_time", 0.010))
    slow_update_zncc = float(guide.get("update_min_neighbor_zncc", 0.50))
    slow_update_prediction = float(
        guide.get("update_max_prediction_error_time", 0.015)
    )
    slow_update_consensus = float(guide.get("update_consensus_time", 0.005))
    slow_update_confirm = int(guide.get("update_confirm_receivers", 3))
    slow_protected_fast_weight = float(
        guide.get("protected_fast_prediction_weight_scale", 0.25)
    )
    if slow_candidate_half < 0.0 or slow_scale <= 0.0:
        raise ValueError("slow-guide time scales must be positive")
    if slow_added_candidates < 1 or slow_update_confirm < 1:
        raise ValueError("slow-guide candidate/confirmation counts must be positive")
    if slow_weight < 0.0:
        raise ValueError("slow_guide.score_weight must be non-negative")
    if (
        slow_update_error < 0.0
        or slow_update_prediction < 0.0
        or slow_update_consensus < 0.0
    ):
        raise ValueError("slow-guide update error limits must be non-negative")
    if not 0.0 <= slow_protected_fast_weight <= 1.0:
        raise ValueError(
            "slow_guide.protected_fast_prediction_weight_scale must be in [0, 1]"
        )
    if not -1.0 <= slow_update_zncc <= 1.0:
        raise ValueError("slow_guide.update_min_neighbor_zncc must be in [-1, 1]")

    candidate_ownership = escape_ownership if escape_enabled else ownership
    traces = _normalise_traces(flat, valid)
    half_window_samples = max(
        1, int(round(float(coherence_half_window_time) / float(dt)))
    )
    candidates, candidate_count_before, candidate_envelope = _build_candidates(
        traces,
        valid,
        dt=dt,
        t0=t0,
        min_distance_samples=max(
            1, int(round(float(candidate_min_distance_time) / float(dt)))
        ),
        min_prominence=candidate_min_prominence,
        max_candidates=int(max_candidates_per_trace),
        half_window_samples=half_window_samples,
        seed_receiver=seed,
        seed_sample=seed_sample,
        seed_time=float(seed_time),
        state_valid=candidate_ownership,
    )
    if not candidates[seed]:
        raise ValueError("same-x seed waveform window is outside the record")

    anchor_mask = (
        np.zeros(nr, dtype=bool)
        if anchor_receiver_mask is None
        else np.asarray(anchor_receiver_mask, dtype=bool)
    )
    anchor_samples = np.full(nr, -1, dtype=int)
    anchor_times = np.full(nr, np.nan)
    if anchor_mask.shape != (nr,):
        raise ValueError("anchor_receiver_mask must have shape [nreceiver]")
    if np.any(anchor_mask):
        anchor_samples = np.asarray(anchor_pick_sample, dtype=int)
        anchor_times = np.asarray(anchor_pick_time, dtype=float)
        if anchor_samples.shape != (nr,) or anchor_times.shape != (nr,):
            raise ValueError("anchor picks must have shape [nreceiver]")
        if not anchor_mask[seed]:
            raise ValueError("anchor segment must include seed receiver")
        anchor_indices = np.flatnonzero(anchor_mask)
        if anchor_indices.size:
            expected = np.arange(anchor_indices[0], anchor_indices[-1] + 1)
            if not np.array_equal(anchor_indices, expected):
                raise ValueError("anchor segment must be contiguous")
        for receiver in np.flatnonzero(anchor_mask):
            sample = int(anchor_samples[receiver])
            if not valid[receiver]:
                raise ValueError("anchor contains an invalid receiver")
            if not 0 <= sample < nt:
                raise ValueError("anchor contains an invalid sample")
            if ownership is not None and not ownership[receiver, sample]:
                raise ValueError("anchor contains a sample outside ownership")
            waveform = _candidate_waveform(
                traces[receiver], sample, half_window_samples
            )
            if waveform is None or traces[receiver, sample] <= 0.0:
                raise ValueError("anchor contains an invalid positive peak")
            candidates[receiver] = [
                FlatEventCandidate(
                    sample=sample,
                    time=float(anchor_times[receiver]),
                    amplitude=float(traces[receiver, sample]),
                    envelope=float(candidate_envelope[receiver, sample]),
                    polarity=1,
                    prominence=np.inf,
                    waveform=waveform,
                )
            ]

    max_slope = float(max_residual_slope_ms_per_100m) * 1e-5
    rescue = dict(continuation_rescue or {})
    direction_options = dict(
        candidate_envelope=candidate_envelope,
        traces=traces,
        state_valid=ownership,
        dt=float(dt),
        t0=float(t0),
        half_window_samples=half_window_samples,
        rescue_enabled=bool(rescue.get("enabled", False)),
        rescue_half_width=float(rescue.get("half_width_time", 0.020)),
        rescue_max_parent_states=int(rescue.get("max_parent_states", 8)),
        rescue_min_prominence=float(rescue.get("min_prominence", 0.0)),
        max_valid_receiver_gap=int(rescue.get("max_valid_receiver_gap", 0)),
        valid_receiver=valid,
        anchor_mask=anchor_mask,
        escape_state_valid=escape_ownership,
        escape_enabled=escape_enabled,
        escape_min_zncc=float(escape.get("min_neighbor_similarity", 0.85)),
        escape_max_prediction_error=float(
            escape.get("max_prediction_error_time", 0.015)
        ),
        slow_guide_enabled=slow_enabled,
        slow_guide_memory_receivers=slow_memory,
        slow_guide_candidate_half_width=slow_candidate_half,
        slow_guide_max_added_candidates=slow_added_candidates,
        slow_guide_score_weight=slow_weight,
        slow_guide_soft_scale=slow_scale,
        slow_guide_update_max_error=slow_update_error,
        slow_guide_update_min_zncc=slow_update_zncc,
        slow_guide_update_max_prediction_error=slow_update_prediction,
        slow_guide_update_consensus_time=slow_update_consensus,
        slow_guide_update_confirm_receivers=slow_update_confirm,
        slow_guide_protected_fast_weight_scale=slow_protected_fast_weight,
    )

    left = _track_direction(
        candidates,
        x,
        np.arange(seed, -1, -1),
        max_slope=max_slope,
        max_prediction_error=max_prediction_error_time,
        min_zncc=min_neighbor_zncc,
        hard_zncc_gate=bool(hard_neighbor_zncc_gate),
        prediction_soft_scale=float(prediction_soft_scale_time),
        prediction_penalty_weight=float(prediction_penalty_weight),
        max_active_states=int(max_active_states),
        phase_switch_zncc=float(phase_switch_zncc),
        phase_switch_prediction_error=float(phase_switch_prediction_time),
        **direction_options,
    )
    right = _track_direction(
        candidates,
        x,
        np.arange(seed, nr),
        max_slope=max_slope,
        max_prediction_error=max_prediction_error_time,
        min_zncc=min_neighbor_zncc,
        hard_zncc_gate=bool(hard_neighbor_zncc_gate),
        prediction_soft_scale=float(prediction_soft_scale_time),
        prediction_penalty_weight=float(prediction_penalty_weight),
        max_active_states=int(max_active_states),
        phase_switch_zncc=float(phase_switch_zncc),
        phase_switch_prediction_error=float(phase_switch_prediction_time),
        **direction_options,
    )

    selected = dict(left[0])
    selected.update(right[0])
    ownership_escape_used = np.zeros(nr, dtype=bool)
    pick_sample = np.full(nr, -1, dtype=int)
    pick_time = np.full(nr, np.nan)
    candidate_sample = np.full(nr, -1, dtype=int)
    amplitude = np.full(nr, np.nan)
    selected_envelope = np.full(nr, np.nan)
    polarity = np.zeros(nr, dtype=int)
    for receiver, candidate_index in selected.items():
        candidate = candidates[receiver][candidate_index]
        sample = candidate.sample
        ownership_escape_used[receiver] = bool(
            escape_enabled and not ownership[receiver, sample]
        )
        pick_sample[receiver] = candidate_sample[receiver] = sample
        offset = 0.0
        if (
            not anchor_mask[receiver]
            and receiver != seed
            and 0 < sample < nt - 1
        ):
            values = traces[receiver, sample - 1 : sample + 2]
            denominator = values[0] - 2.0 * values[1] + values[2]
            if np.isfinite(denominator) and denominator != 0.0:
                offset = float(
                    np.clip(
                        0.5 * (values[0] - values[2]) / denominator,
                        -0.5,
                        0.5,
                    )
                )
        pick_time[receiver] = (
            anchor_times[receiver]
            if anchor_mask[receiver]
            else float(t0) + (sample + offset) * float(dt)
        )
        amplitude[receiver] = candidate.amplitude
        selected_envelope[receiver] = candidate.envelope
        polarity[receiver] = candidate.polarity
    pick_time[seed] = float(seed_time)

    neighbor = np.full(nr, np.nan)
    slope = np.full(nr, np.nan)
    prediction = np.full(nr, np.nan)
    accumulated = np.full(nr, np.nan)
    for receiver_order in (np.arange(seed, -1, -1), np.arange(seed, nr)):
        total = 0.0
        available = [int(receiver) for receiver in receiver_order if receiver in selected]
        for local, receiver in enumerate(available):
            accumulated[receiver] = total
            if local == 0:
                continue
            previous = available[local - 1]
            a = candidates[previous][selected[previous]]
            b = candidates[receiver][selected[receiver]]
            neighbor[receiver] = float(np.dot(a.waveform, b.waveform))
            slope[receiver] = (b.time - a.time) / (
                x[receiver] - x[previous]
            )
            total += neighbor[receiver]
            accumulated[receiver] = total
            if local >= 2:
                older = available[local - 2]
                c = candidates[older][selected[older]]
                previous_slope = (a.time - c.time) / (
                    x[previous] - x[older]
                )
                prediction[receiver] = b.time - (
                    a.time
                    + previous_slope * (x[receiver] - x[previous])
                )

    return SparseFlatTrackingResult(
        pick_sample=pick_sample,
        pick_time=pick_time,
        candidate_sample=candidate_sample,
        success_mask=pick_sample >= 0,
        trace_valid=valid,
        neighbor_correlation=neighbor,
        residual_slope=slope,
        prediction_error=prediction,
        candidate_count=np.asarray([len(row) for row in candidates], dtype=int),
        candidate_count_before_ownership=candidate_count_before,
        selected_amplitude=amplitude,
        selected_envelope=selected_envelope,
        selected_polarity=polarity,
        accumulated_coherence=accumulated,
        seed_receiver=seed,
        seed_time=float(seed_time),
        stop_reason_left=left[1],
        stop_reason_right=right[1],
        active_state_max=max(left[4], right[4]),
        transition_count=left[2] + right[2],
        correlation_count=left[3] + right[3],
        rescue_attempt_count=left[5]["attempts"] + right[5]["attempts"],
        rescue_success_count=left[5]["successes"] + right[5]["successes"],
        rescue_candidate_count=left[5]["candidates"] + right[5]["candidates"],
        rescue_transition_count=(
            left[5]["rescue_transitions"] + right[5]["rescue_transitions"]
        ),
        rescue_used_mask=left[5]["rescue_mask"] | right[5]["rescue_mask"],
        ownership_escape_used_mask=ownership_escape_used,
        ownership_escape_attempt_count=(
            left[5]["escape_attempts"] + right[5]["escape_attempts"]
        ),
        ownership_escape_success_count=(
            left[5]["escape_successes"] + right[5]["escape_successes"]
        ),
        ownership_escape_candidate_count=(
            left[5]["escape_candidates"] + right[5]["escape_candidates"]
        ),
        skipped_valid_receiver_mask=(
            left[5]["gap_mask"] | right[5]["gap_mask"]
        ),
        selected_transition_kind=np.where(
            right[5]["transition_kind"] >= 0,
            right[5]["transition_kind"],
            left[5]["transition_kind"],
        ),
        stop_receiver_left=int(left[5]["stop_receiver"]),
        stop_receiver_right=int(right[5]["stop_receiver"]),
        runtime_seconds=perf_counter() - started,
    )


__all__ = [
    "FlatEventCandidate",
    "SparseFlatTrackingResult",
    "TransitionKind",
    "track_flattened_event_sparse",
]
