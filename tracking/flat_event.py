"""Sparse, seed-fixed tracking of one event in a flattened observed gather."""

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



def _append_node(arena, *, parent_id, receiver, candidate_index, kind, rho,
                 prediction_error, score_delta, score, cumulative_prediction_error):
    arena.append(_PathNode(parent_id, receiver, candidate_index, int(kind), rho,
                           prediction_error, score_delta, score,
                           cumulative_prediction_error))
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
                keep = np.argsort(prominences)[-int(max_candidates):]
                peaks, prominences = peaks[keep], prominences[keep]
        before[receiver] = peaks.size
        if state_valid is not None:
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
    return candidates, before


def _stop_reason(rejected: dict[str, int], has_candidates: bool, *, hard_zncc_gate: bool) -> str:
    if not has_candidates:
        return "NO_CANDIDATE"
    # Report the stage that actually removed all feasible transitions.  Correlation
    # is a soft score by default and therefore cannot stop a path unless the
    # optional hard gate is explicitly enabled.
    if rejected.get("SLOPE_REJECTED", 0) and not rejected.get("PREDICTION_HARD_REJECTED", 0):
        return f"SLOPE_REJECTED:n={rejected['SLOPE_REJECTED']}"
    if rejected.get("PREDICTION_HARD_REJECTED", 0):
        return (f"NO_FEASIBLE_TRANSITION:slope={rejected.get('SLOPE_REJECTED', 0)}:"
                f"prediction_hard={rejected['PREDICTION_HARD_REJECTED']}")
    if hard_zncc_gate and rejected.get("CORRELATION_REJECTED", 0):
        return "CORRELATION_REJECTED"
    return "NO_FEASIBLE_TRANSITION"


def _track_direction(
    candidates: list[list[FlatEventCandidate]],
    receiver_x: np.ndarray,
    order: np.ndarray,
    *,
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
) -> tuple[dict[int, int], str, int, int, int, dict[str, object]]:
    """Track one side of the seed with sparse second-order DP."""
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
    }
    if order.size <= 1:
        return {int(order[0]): 0}, "END_OF_GATHER", 0, 0, 0, diagnostic

    seed = int(order[0])
    order_position = {int(receiver): int(i) for i, receiver in enumerate(order)}
    available_order = [seed]
    for receiver in order[1:]:
        receiver = int(receiver)
        if not valid_receiver[receiver]:
            break
        available_order.append(receiver)
    direction_end_receiver = int(available_order[-1])
    natural_end_reason = (
        "END_OF_GATHER" if len(available_order) == int(order.size)
        else "END_OF_VALID_GATHER"
    )

    arena: list[_PathNode] = []
    seed_node = _append_node(
        arena, parent_id=-1, receiver=seed, candidate_index=0,
        kind=TransitionKind.SEED, rho=np.nan, prediction_error=np.nan,
        score_delta=0.0, score=0.0, cumulative_prediction_error=0.0,
    )
    states: dict[tuple, _SparseDPState] = {}
    transitions = correlations = 0
    active_max = 0

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
        row = candidates[target_receiver]
        if not row:
            return next_states, rejected
        current_times = np.asarray([item.time for item in row])
        allowed_samples_set = None if allowed_samples is None else set(int(v) for v in allowed_samples)
        for state in source_states.values():
            older_node = arena[state.older_success_node_id]
            previous_node = arena[state.previous_success_node_id]
            older_receiver = int(older_node.receiver)
            previous_receiver = int(previous_node.receiver)
            older_index = int(older_node.candidate_index)
            previous_index = int(previous_node.candidate_index)
            dx_previous = float(receiver_x[previous_receiver] - receiver_x[older_receiver])
            dx_current = float(receiver_x[target_receiver] - receiver_x[previous_receiver])
            if dx_previous == 0.0 or dx_current == 0.0:
                continue
            older = candidates[older_receiver][older_index]
            previous = candidates[previous_receiver][previous_index]
            lower = previous.time - max_slope * abs(dx_current)
            upper = previous.time + max_slope * abs(dx_current)
            if allowed_samples_set is None:
                # Normal candidate rows are time-sorted, so use binary bounds.
                lo = int(np.searchsorted(current_times, lower, side="left"))
                hi = int(np.searchsorted(current_times, upper, side="right"))
                candidate_indices = range(lo, hi)
                if lo == hi:
                    rejected["SLOPE_REJECTED"] += 1
                    continue
            else:
                # Rescue candidates may be appended after normal states for this
                # receiver have already been created.  Never re-sort that row,
                # because doing so would invalidate candidate indices stored in
                # those normal path nodes. Scan only the requested rescue samples.
                candidate_indices = [
                    idx for idx, item in enumerate(row)
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
                    state_valid[target_receiver, current.sample]
                )
                if ownership_mode == "base" and not in_base:
                    continue
                if ownership_mode == "escape" and in_base:
                    continue
                prediction_error = abs(current.time - predicted)
                if prediction_error > max_prediction_error:
                    rejected["PREDICTION_HARD_REJECTED"] += 1
                    continue
                rho = float(np.dot(previous.waveform, current.waveform))
                correlations += 1
                if not in_base and (
                    rho < escape_min_zncc
                    or prediction_error > escape_max_prediction_error
                ):
                    continue
                if rho < phase_switch_zncc and prediction_error > phase_switch_prediction_error:
                    rejected["CORRELATION_REJECTED"] += 1
                    continue
                if hard_zncc_gate and rho < min_zncc:
                    rejected["CORRELATION_REJECTED"] += 1
                    continue
                transitions += 1
                score_delta = rho - prediction_penalty_weight * (
                    prediction_error / prediction_soft_scale
                ) ** 2
                score = state.score + score_delta
                cumulative_prediction = state.prediction_error + prediction_error
                node_id = _append_node(
                    arena,
                    parent_id=state.node_id,
                    receiver=target_receiver,
                    candidate_index=current_index,
                    kind=transition_kind,
                    rho=rho,
                    prediction_error=prediction_error,
                    score_delta=score_delta,
                    score=score,
                    cumulative_prediction_error=cumulative_prediction,
                )
                candidate_state = _SparseDPState(
                    score,
                    cumulative_prediction,
                    state.envelope + current.envelope,
                    node_id,
                    state.previous_success_node_id,
                    node_id,
                    0,
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
            traces[target_receiver], prominence=rescue_min_prominence
        )
        if all_positive_peaks.size:
            all_positive_peaks = all_positive_peaks[
                traces[target_receiver, all_positive_peaks] > 0.0
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
                receiver_x[target_receiver] - receiver_x[previous_receiver]
            )
            center = int(round((predicted - t0) / dt))
            lo_sample = max(0, center - half)
            hi_sample = min(traces.shape[1], center + half + 1)
            inside = all_positive_peaks[
                (all_positive_peaks >= lo_sample) & (all_positive_peaks < hi_sample)
            ]
            for sample in inside:
                sample = int(sample)
                allowed = escape_state_valid if escape_enabled else state_valid
                if allowed is not None and not allowed[target_receiver, sample]:
                    continue
                result.add(sample)
        return result

    def _append_rescue_candidates(target_receiver, rescue_samples):
        existing = {int(item.sample) for item in candidates[target_receiver]}
        new_samples = sorted(set(int(v) for v in rescue_samples) - existing)
        if not new_samples:
            return 0
        rescue_envelope = np.abs(hilbert(traces[target_receiver]))
        added = 0
        for sample in new_samples:
            if traces[target_receiver, sample] <= 0.0:
                continue
            allowed = escape_state_valid if escape_enabled else state_valid
            if allowed is not None and not allowed[target_receiver, sample]:
                continue
            waveform = _candidate_waveform(
                traces[target_receiver], sample, half_window_samples
            )
            if waveform is None:
                continue
            candidates[target_receiver].append(
                FlatEventCandidate(
                    sample,
                    float(t0 + sample * dt),
                    float(traces[target_receiver, sample]),
                    float(rescue_envelope[sample]),
                    1,
                    0.0,
                    waveform,
                )
            )
            added += 1
        # Do not sort after appending: existing path nodes store candidate indices.
        # Rescue advance scans the appended candidates by sample.
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
                receiver=target_receiver,
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

    # ------------------------------------------------------------------
    # First receiver: use the seed as an observed phase control.  The first
    # transition is scored by signed ZNCC but never hard-gated by ZNCC.
    # ------------------------------------------------------------------
    first = int(order[1])
    dx = float(receiver_x[first] - receiver_x[seed])
    if dx == 0.0:
        return {seed: 0}, "ZERO_RECEIVER_SPACING", 0, 0, 0, diagnostic
    first_indices = [
        index for index, candidate in enumerate(candidates[first])
        if state_valid is None or state_valid[first, candidate.sample]
    ]
    if not first_indices:
        diagnostic["stop_receiver"] = first
        return {seed: 0}, "NO_POSITIVE_CANDIDATE", 0, 0, 0, diagnostic

    seed_candidate = candidates[seed][0]
    rejected = {
        "SLOPE_REJECTED": 0,
        "PREDICTION_HARD_REJECTED": 0,
        "CORRELATION_REJECTED": 0,
    }
    for current_index in first_indices:
        current = candidates[first][current_index]
        slope = (current.time - seed_candidate.time) / dx
        if not anchor_mask[first] and abs(slope) > max_slope:
            rejected["SLOPE_REJECTED"] += 1
            continue
        rho = float(np.dot(seed_candidate.waveform, current.waveform))
        correlations += 1
        node_id = _append_node(
            arena,
            parent_id=seed_node,
            receiver=first,
            candidate_index=current_index,
            kind=TransitionKind.ANCHOR if anchor_mask[first] else TransitionKind.NORMAL,
            rho=rho,
            prediction_error=0.0,
            score_delta=rho,
            score=rho,
            cumulative_prediction_error=0.0,
        )
        states[(0, current_index)] = _SparseDPState(
            rho,
            0.0,
            seed_candidate.envelope + current.envelope,
            node_id,
            seed_node,
            node_id,
            0,
        )
        transitions += 1

    if not states:
        return (
            {seed: 0},
            _stop_reason(rejected, bool(candidates[first]), hard_zncc_gate=hard_zncc_gate),
            transitions,
            correlations,
            0,
            diagnostic,
        )
    active_max = len(states)

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

        # --------------------------------------------------------------
        # Fixed pilot anchor: immutable during continuation.
        # --------------------------------------------------------------
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
                dx_previous = float(receiver_x[previous_receiver] - receiver_x[older_receiver])
                dx_current = float(receiver_x[current_receiver] - receiver_x[previous_receiver])
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
                node_id = _append_node(
                    arena,
                    parent_id=state.node_id,
                    receiver=current_receiver,
                    candidate_index=current_index,
                    kind=TransitionKind.ANCHOR,
                    rho=rho,
                    prediction_error=prediction_error,
                    score_delta=score_delta,
                    score=score,
                    cumulative_prediction_error=cumulative_prediction,
                )
                forced = _SparseDPState(
                    score,
                    cumulative_prediction,
                    state.envelope + current.envelope,
                    node_id,
                    state.previous_success_node_id,
                    node_id,
                    0,
                )
                key = (previous_receiver, previous_node.candidate_index, current_index)
                if _better(forced, next_states.get(key)):
                    next_states[key] = forced
            if not next_states:
                diagnostic["stop_receiver"] = current_receiver
                stop_reason = "INVALID_ANCHOR_TRANSITION"
                break
            states = _prune(next_states, max_active_states)
            active_max = max(active_max, len(states))
            local += 1
            continue

        # --------------------------------------------------------------
        # Normal single-path DP.
        # --------------------------------------------------------------
        next_states, rejected = _advance_at(
            states, current_receiver, TransitionKind.NORMAL
        )

        if not next_states and escape_enabled:
            diagnostic["escape_attempts"] += 1
            diagnostic["escape_candidates"] += sum(
                escape_state_valid[current_receiver, item.sample]
                and not state_valid[current_receiver, item.sample]
                for item in candidates[current_receiver]
            )
            next_states, rejected = _advance_at(
                states, current_receiver, TransitionKind.NORMAL,
                ownership_mode="escape",
            )
            if next_states:
                diagnostic["escape_successes"] += 1

        # --------------------------------------------------------------
        # Single-path rescue / one-valid-receiver gap.
        # --------------------------------------------------------------
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

        if not next_states:
            gap = _gap_states(states, current_receiver) if max_valid_receiver_gap > 0 else {}
            if gap:
                states = gap
                active_max = max(active_max, len(states))
                local += 1
                continue
            if rescue_attempted:
                stop_reason = (
                    "RESCUE_NO_POSITIVE_PEAK" if rescue_added == 0
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
) -> SparseFlatTrackingResult:
    """Continue the seed event using sparse candidates and second-order DP.

    ``seed_time`` is a hard geometric control.  The first adjacent receiver is
    initialized from slope-reachable candidates without waveform gating; signed
    neighbor ZNCC becomes a soft accumulated score from the next receiver on.
    ``min_neighbor_zncc`` is only used when ``hard_neighbor_zncc_gate=True``.
    A non-positive ``max_candidates_per_trace`` disables the global prominence
    cap so weak target reflectors cannot be discarded before DP.
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
    escape_ownership = None if escape_state_valid is None else np.asarray(escape_state_valid, dtype=bool)
    if escape_enabled:
        if ownership is None or escape_ownership is None or escape_ownership.shape != flat.shape:
            raise ValueError("ownership escape requires base and escape state_valid masks")
        if np.any(ownership & ~escape_ownership):
            raise ValueError("escape_state_valid must contain base state_valid")
    candidate_ownership = escape_ownership if escape_enabled else ownership
    traces = _normalise_traces(flat, valid)
    candidates, candidate_count_before = _build_candidates(
        traces, valid, dt=dt, t0=t0,
        min_distance_samples=max(1, int(round(candidate_min_distance_time / dt))),
        min_prominence=candidate_min_prominence,
        max_candidates=int(max_candidates_per_trace),
        half_window_samples=max(1, int(round(coherence_half_window_time / dt))),
        seed_receiver=seed, seed_sample=seed_sample, seed_time=float(seed_time),
        state_valid=candidate_ownership,
    )
    if not candidates[seed]:
        raise ValueError("same-x seed waveform window is outside the record")
    anchor_mask = np.zeros(nr, dtype=bool) if anchor_receiver_mask is None else np.asarray(anchor_receiver_mask, dtype=bool)
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
        # The anchor must be a single contiguous segment through the seed on
        # each side.  Holes would make the meaning of a fixed pilot segment
        # ambiguous and can silently reintroduce free DP inside the anchor.
        anchor_indices = np.flatnonzero(anchor_mask)
        if anchor_indices.size:
            expected = np.arange(anchor_indices[0], anchor_indices[-1] + 1)
            if not np.array_equal(anchor_indices, expected):
                raise ValueError("anchor segment must be contiguous")
        for receiver in np.flatnonzero(anchor_mask):
            sample = int(anchor_samples[receiver])
            if not valid[receiver]:
                raise ValueError("anchor contains an invalid receiver")
            if ownership is not None and not ownership[receiver, sample]:
                raise ValueError("anchor contains a sample outside ownership")
            waveform = _candidate_waveform(traces[receiver], sample, max(1, int(round(coherence_half_window_time / dt))))
            if waveform is None or not 0 <= sample < nt or traces[receiver, sample] <= 0.0:
                raise ValueError("anchor contains an invalid positive peak")
            candidates[receiver] = [FlatEventCandidate(
                sample=sample, time=float(anchor_times[receiver]),
                amplitude=float(traces[receiver, sample]),
                envelope=float(np.abs(hilbert(traces[receiver]))[sample]),
                polarity=1, prominence=np.inf, waveform=waveform,
            )]
    max_slope = float(max_residual_slope_ms_per_100m) * 1e-5
    rescue = dict(continuation_rescue or {})
    direction_options = dict(
        traces=traces, state_valid=ownership, dt=float(dt), t0=float(t0),
        half_window_samples=max(1, int(round(coherence_half_window_time / dt))),
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
        escape_max_prediction_error=float(escape.get("max_prediction_error_time", 0.015)),
    )
    left = _track_direction(
        candidates, x, np.arange(seed, -1, -1), max_slope=max_slope,
        max_prediction_error=max_prediction_error_time, min_zncc=min_neighbor_zncc,
        hard_zncc_gate=bool(hard_neighbor_zncc_gate),
        prediction_soft_scale=float(prediction_soft_scale_time),
        prediction_penalty_weight=float(prediction_penalty_weight),
        max_active_states=int(max_active_states),
        phase_switch_zncc=float(phase_switch_zncc),
        phase_switch_prediction_error=float(phase_switch_prediction_time),
        **direction_options,
    )
    right = _track_direction(
        candidates, x, np.arange(seed, nr), max_slope=max_slope,
        max_prediction_error=max_prediction_error_time, min_zncc=min_neighbor_zncc,
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
        if not anchor_mask[receiver] and receiver != seed and 0 < sample < nt - 1:
            values = traces[receiver, sample - 1 : sample + 2]
            denominator = values[0] - 2.0 * values[1] + values[2]
            if np.isfinite(denominator) and denominator != 0.0:
                offset = float(np.clip(0.5 * (values[0] - values[2]) / denominator, -0.5, 0.5))
        pick_time[receiver] = (anchor_times[receiver] if anchor_mask[receiver]
                               else float(t0) + (sample + offset) * float(dt))
        amplitude[receiver] = candidate.amplitude
        selected_envelope[receiver] = candidate.envelope
        polarity[receiver] = candidate.polarity
    pick_time[seed] = float(seed_time)

    neighbor = np.full(nr, np.nan)
    slope = np.full(nr, np.nan)
    prediction = np.full(nr, np.nan)
    accumulated = np.full(nr, np.nan)
    for order in (np.arange(seed, -1, -1), np.arange(seed, nr)):
        total = 0.0
        available = [int(receiver) for receiver in order if receiver in selected]
        for local, receiver in enumerate(available):
            accumulated[receiver] = total
            if local == 0:
                continue
            previous = available[local - 1]
            a = candidates[previous][selected[previous]]
            b = candidates[receiver][selected[receiver]]
            neighbor[receiver] = float(np.dot(a.waveform, b.waveform))
            slope[receiver] = (b.time - a.time) / (x[receiver] - x[previous])
            total += neighbor[receiver]
            accumulated[receiver] = total
            if local >= 2:
                older = available[local - 2]
                c = candidates[older][selected[older]]
                previous_slope = (a.time - c.time) / (x[previous] - x[older])
                prediction[receiver] = b.time - (a.time + previous_slope * (x[receiver] - x[previous]))

    return SparseFlatTrackingResult(
        pick_sample=pick_sample, pick_time=pick_time, candidate_sample=candidate_sample,
        success_mask=pick_sample >= 0, trace_valid=valid,
        neighbor_correlation=neighbor, residual_slope=slope,
        prediction_error=prediction,
        candidate_count=np.asarray([len(row) for row in candidates], dtype=int),
        candidate_count_before_ownership=candidate_count_before,
        selected_amplitude=amplitude, selected_envelope=selected_envelope,
        selected_polarity=polarity, accumulated_coherence=accumulated,
        seed_receiver=seed, seed_time=float(seed_time),
        stop_reason_left=left[1], stop_reason_right=right[1],
        active_state_max=max(left[4], right[4]),
        transition_count=left[2] + right[2], correlation_count=left[3] + right[3],
        rescue_attempt_count=left[5]["attempts"] + right[5]["attempts"],
        rescue_success_count=left[5]["successes"] + right[5]["successes"],
        rescue_candidate_count=left[5]["candidates"] + right[5]["candidates"],
        rescue_transition_count=left[5]["rescue_transitions"] + right[5]["rescue_transitions"],
        rescue_used_mask=left[5]["rescue_mask"] | right[5]["rescue_mask"],
        ownership_escape_used_mask=ownership_escape_used,
        ownership_escape_attempt_count=left[5]["escape_attempts"] + right[5]["escape_attempts"],
        ownership_escape_success_count=left[5]["escape_successes"] + right[5]["escape_successes"],
        ownership_escape_candidate_count=left[5]["escape_candidates"] + right[5]["escape_candidates"],
        skipped_valid_receiver_mask=left[5]["gap_mask"] | right[5]["gap_mask"],
        selected_transition_kind=np.where(
            right[5]["transition_kind"] >= 0,
            right[5]["transition_kind"], left[5]["transition_kind"],
        ),
        stop_receiver_left=int(left[5]["stop_receiver"]),
        stop_receiver_right=int(right[5]["stop_receiver"]),
        runtime_seconds=perf_counter() - started,
    )


__all__ = ["FlatEventCandidate", "SparseFlatTrackingResult", "TransitionKind",
           "track_flattened_event_sparse"]
