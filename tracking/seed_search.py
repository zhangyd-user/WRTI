"""Positive-peak seed discovery with pilot-validated anchor segments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.signal import find_peaks, hilbert

from .flat_event import track_flattened_event_sparse


@dataclass(frozen=True)
class SeedPilotResult:
    passed: bool
    coverage: float
    median_zncc: float
    median_abs_prediction_error: float
    median_abs_slope_ms_per_100m: float
    p90_abs_slope_ms_per_100m: float
    phase_switch_count: int
    pick_sample: np.ndarray
    pick_time: np.ndarray
    success_mask: np.ndarray
    neighbor_zncc: np.ndarray
    residual_slope: np.ndarray
    prediction_error: np.ndarray
    tested_mask: np.ndarray
    left_support: int
    right_support: int
    transition_count: int


@dataclass(frozen=True)
class SeedSearchResult:
    seed_time: float
    seed_sample: int
    valid: bool
    mode: str
    positive_amplitude: float
    stack_evidence: float
    positive_support_fraction: float
    hypothesis_count: int
    selected_hypothesis: int
    pilot_coverage: float
    pilot_median_zncc: float
    pilot_median_abs_prediction_error: float
    pilot_median_abs_slope_ms_per_100m: float
    phase_switch_count: int
    pilot_left_support: int
    pilot_right_support: int
    anchor_left_count: int
    anchor_right_count: int
    anchor_receiver_mask: np.ndarray
    anchor_pick_sample: np.ndarray
    anchor_pick_time: np.ndarray
    pilot_pick_time: np.ndarray
    local_hypothesis_count: int
    adaptive_hypothesis_count: int
    pilot_transition_count: int
    runtime_seconds: float
    high_confidence: bool


@dataclass(frozen=True)
class BoundarySeedCandidate:
    seed_time: float
    seed_sample: int
    seed_amplitude: float
    pilot_coverage: float
    pilot_median_zncc: float
    pilot_median_abs_prediction_error: float
    pilot_receiver_start: int
    pilot_receiver_end: int
    pilot_success_count: int
    pilot_receiver_count: int
    anchor_length: int
    short_segment_energy: float
    stack_evidence: float
    positive_support_fraction: float
    long_coverage: float
    long_median_zncc: float
    long_median_abs_prediction_error: float
    long_median_abs_slope_ms_per_100m: float
    long_p90_abs_slope_ms_per_100m: float
    long_phase_switch_count: int
    current_ownership_peak_count: int
    exclusive_ownership_peak_count: int
    overlap_fallback: bool
    rank: tuple
    rejection_reason: str
    selected: bool
    seed: SeedSearchResult | None
    # Repair-only event-first seed selection needs the complete short path,
    # not just the final anchor/long-pilot verdict.  A hypothesis whose own
    # anchor later fails may still be useful evidence that a coherent short
    # event exists in the gather.
    short_pilot: SeedPilotResult | None = None


def _contiguous_valid_mask(valid: np.ndarray, seed: int) -> np.ndarray:
    """Return only the contiguous valid receiver segment containing ``seed``."""
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros_like(valid)
    if not 0 <= int(seed) < valid.size or not bool(valid[int(seed)]):
        return result
    left = int(seed)
    right = int(seed)
    while left > 0 and bool(valid[left - 1]):
        left -= 1
    while right + 1 < valid.size and bool(valid[right + 1]):
        right += 1
    result[left : right + 1] = True
    return result


def _centered_indices(valid, seed, count):
    """Closest receiver indices, restricted to one contiguous valid segment."""
    valid = np.asarray(valid, dtype=bool)
    segment = _contiguous_valid_mask(valid, int(seed))
    indices = np.flatnonzero(segment)
    if indices.size == 0:
        return indices
    if int(count) <= 0 or indices.size <= int(count):
        return indices
    order = np.argsort(np.abs(indices - int(seed)), kind="stable")
    return np.sort(indices[order[: int(count)]])


def _normalised(flat, valid):
    scale = np.sqrt(np.mean(flat * flat, axis=1))
    result = np.zeros_like(flat, dtype=float)
    good = valid & np.isfinite(scale) & (scale > np.finfo(float).eps)
    result[good] = flat[good] / scale[good, None]
    return result


def _evidence(traces, envelope, receivers, sample, half):
    """Robust near-seed evidence for one positive seed-trace peak.

    The envelope is precomputed once.  Each nearby receiver is allowed to
    contribute its strongest envelope sample inside a small time tolerance,
    so a modest residual moveout does not smear the evidence.
    """
    lo = max(0, int(sample) - int(half))
    hi = min(traces.shape[1], int(sample) + int(half) + 1)
    if hi <= lo or receivers.size == 0:
        return 0.0, 0.0

    local_envelope = envelope[:, lo:hi]
    coherent = float(np.median(np.max(local_envelope, axis=1)))

    support = []
    for trace in traces[receivers]:
        peaks, _ = find_peaks(trace[lo:hi])
        support.append(bool(peaks.size))
    return coherent, float(np.mean(support)) if support else 0.0


def _hypotheses(
    traces,
    envelope,
    seed,
    receivers,
    time,
    search_valid,
    *,
    max_hypotheses,
    evidence_half_samples,
):
    """Generate hypotheses directly from every positive seed-trace peak."""
    peaks, _ = find_peaks(traces[seed])
    if peaks.size:
        peaks = peaks[search_valid[peaks] & (traces[seed, peaks] > 0.0)]

    items = []
    for sample in peaks:
        coherent, support = _evidence(
            traces,
            envelope,
            receivers,
            int(sample),
            evidence_half_samples,
        )
        items.append(
            (
                float(time[sample]),
                int(sample),
                float(traces[seed, sample]),
                coherent,
                support,
            )
        )

    # Multi-trace evidence is a ranking aid only; it never determines whether
    # a positive seed-trace peak is allowed to enter pilot validation.
    items.sort(key=lambda item: (item[3], item[4], item[2]), reverse=True)
    if int(max_hypotheses) > 0:
        items = items[: int(max_hypotheses)]
    return items


def build_interlayer_repair_state_valid(
    state_valid, upper_neighbor_state_valid=None, lower_neighbor_state_valid=None,
    *, guard_samples=0,
):
    """Return the repair-only safe band between adjacent reflector ownerships.

    Boundary repair must not be forced to remain inside the failed reflector's
    original ownership corridor.  Instead, the upper/lower neighboring
    reflectors define hard physical bounds.  A small guard is removed next to
    each neighbor so repair seeds/tracks cannot sit on or immediately beside a
    neighboring reflector.  If both neighbors are absent, fall back to the
    original ownership mask.
    """
    current = np.asarray(state_valid, dtype=bool)
    if current.ndim != 2:
        raise ValueError("state_valid must have shape [nreceiver, ntime]")
    upper = None if upper_neighbor_state_valid is None else np.asarray(upper_neighbor_state_valid, dtype=bool)
    lower = None if lower_neighbor_state_valid is None else np.asarray(lower_neighbor_state_valid, dtype=bool)
    for name, value in (("upper_neighbor_state_valid", upper), ("lower_neighbor_state_valid", lower)):
        if value is not None and value.shape != current.shape:
            raise ValueError(f"{name} must match state_valid")

    if upper is None and lower is None:
        return current.copy()

    guard = max(0, int(guard_samples))
    nreceiver, ntime = current.shape
    safe = np.zeros_like(current)
    for receiver in range(nreceiver):
        start = 0
        stop = ntime
        if upper is not None:
            indices = np.flatnonzero(upper[receiver])
            if indices.size:
                start = int(indices[-1]) + 1 + guard
        if lower is not None:
            indices = np.flatnonzero(lower[receiver])
            if indices.size:
                stop = int(indices[0]) - guard
        start = max(0, min(start, ntime))
        stop = max(0, min(stop, ntime))
        if start < stop:
            safe[receiver, start:stop] = True

    # Explicitly remove the neighboring ownerships and their guard zones even
    # if a pathological/non-contiguous neighbor mask was supplied.
    for neighbor in (upper, lower):
        if neighbor is None:
            continue
        forbidden = neighbor.copy()
        for step in range(1, guard + 1):
            forbidden[:, step:] |= neighbor[:, :-step]
            forbidden[:, :-step] |= neighbor[:, step:]
        safe &= ~forbidden
    return safe


def _pilot(
    flat,
    x,
    valid,
    ownership,
    seed,
    hypothesis,
    options,
    subset,
    min_side_support,
    min_coverage,
    min_median_zncc,
    phase_zncc,
    phase_prediction,
):
    """Run one pilot without jumping across muted/invalid receiver gaps."""
    subset = np.asarray(subset, dtype=int)
    if subset.size == 0 or not np.any(subset == int(seed)):
        return None

    local_seed = int(np.flatnonzero(subset == int(seed))[0])
    tracked = track_flattened_event_sparse(
        flat[subset],
        receiver_x=x[subset],
        valid_receiver=valid[subset],
        seed_receiver=local_seed,
        seed_time=hypothesis[0],
        state_valid=ownership[subset],
        **options,
    )

    success_local = tracked.success_mask.copy()
    success_local[local_seed] = False

    left_local = np.flatnonzero(subset < int(seed))
    right_local = np.flatnonzero(subset > int(seed))
    left_success = int(np.count_nonzero(success_local[left_local]))
    right_success = int(np.count_nonzero(success_local[right_local]))

    min_side = max(1, int(min_side_support))
    internal = left_local.size >= min_side and right_local.size >= min_side
    if internal:
        required = np.r_[left_local, right_local]
        side_ok = left_success >= min_side and right_success >= min_side
    elif left_local.size >= min_side:
        required = left_local
        side_ok = left_success >= min_side
    elif right_local.size >= min_side:
        required = right_local
        side_ok = right_success >= min_side
    else:
        required = np.r_[left_local, right_local]
        side_ok = False

    if required.size:
        required_success = success_local[required]
        coverage = float(np.count_nonzero(required_success) / required.size)
        used_local = required[required_success]
    else:
        coverage = 0.0
        used_local = np.empty(0, dtype=int)

    rho = tracked.neighbor_correlation[used_local]
    rho = rho[np.isfinite(rho)]
    prediction = np.abs(tracked.prediction_error[used_local])
    prediction = prediction[np.isfinite(prediction)]
    slopes = np.abs(tracked.residual_slope[used_local]) * 1e5
    slopes = slopes[np.isfinite(slopes)]

    phase_local = (
        tracked.success_mask
        & np.isfinite(tracked.neighbor_correlation)
        & np.isfinite(tracked.prediction_error)
        & (tracked.neighbor_correlation < float(phase_zncc))
        & (np.abs(tracked.prediction_error) > float(phase_prediction))
    )
    phase_count = int(np.count_nonzero(phase_local))

    sample = np.full(valid.size, -1, dtype=int)
    sample[subset] = tracked.pick_sample
    pick_time = np.full(valid.size, np.nan)
    pick_time[subset] = tracked.pick_time
    mask = np.zeros(valid.size, dtype=bool)
    mask[subset] = tracked.success_mask
    tested = np.zeros(valid.size, dtype=bool)
    tested[subset] = True
    neighbor = np.full(valid.size, np.nan)
    neighbor[subset] = tracked.neighbor_correlation
    slope = np.full(valid.size, np.nan)
    slope[subset] = tracked.residual_slope
    pred = np.full(valid.size, np.nan)
    pred[subset] = tracked.prediction_error

    median_rho = float(np.median(rho)) if rho.size else -np.inf
    median_prediction = float(np.median(prediction)) if prediction.size else np.inf
    median_slope = float(np.median(slopes)) if slopes.size else np.inf
    p90_slope = float(np.percentile(slopes, 90.0)) if slopes.size else np.inf
    passed = (
        bool(side_ok)
        and coverage >= float(min_coverage)
        and median_rho >= float(min_median_zncc)
    )

    return SeedPilotResult(
        passed,
        coverage,
        median_rho,
        median_prediction,
        median_slope,
        p90_slope,
        phase_count,
        sample,
        pick_time,
        mask,
        neighbor,
        slope,
        pred,
        tested,
        left_success,
        right_success,
        tracked.transition_count,
    )


def _anchor(
    pilot,
    traces,
    seed,
    min_per_side,
    min_rho,
    max_prediction,
    max_slope,
):
    """Extract the contiguous, locally trustworthy part of a winning pilot."""
    mask = np.zeros(traces.shape[0], dtype=bool)
    if not pilot.success_mask[int(seed)]:
        return mask, 0, 0, False
    mask[int(seed)] = True

    counts = []
    for step in (-1, 1):
        count = 0
        receiver = int(seed) + step
        while 0 <= receiver < traces.shape[0] and pilot.tested_mask[receiver]:
            sample = int(pilot.pick_sample[receiver])
            if (
                not pilot.success_mask[receiver]
                or sample < 0
                or traces[receiver, sample] <= 0.0
            ):
                break

            rho = pilot.neighbor_zncc[receiver]
            error = pilot.prediction_error[receiver]
            slope = abs(pilot.residual_slope[receiver]) * 1e5
            if not np.isfinite(rho) or rho < float(min_rho):
                break
            if not np.isfinite(slope) or slope > float(max_slope):
                break
            if count > 0 and (
                not np.isfinite(error) or abs(error) > float(max_prediction)
            ):
                break

            mask[receiver] = True
            count += 1
            receiver += step
        counts.append(count)

    min_side = max(1, int(min_per_side))
    left_available = int(np.count_nonzero(pilot.tested_mask[: int(seed)]))
    right_available = int(np.count_nonzero(pilot.tested_mask[int(seed) + 1 :]))
    internal = left_available >= min_side and right_available >= min_side

    if internal:
        ok = counts[0] >= min_side and counts[1] >= min_side
    elif left_available >= min_side:
        ok = counts[0] >= min_side
    elif right_available >= min_side:
        ok = counts[1] >= min_side
    else:
        ok = False

    if not ok:
        mask[:] = False
    return mask, counts[0], counts[1], bool(ok)


def _empty_result(valid, started, local_count=0, adaptive_count=0, transitions=0):
    empty_bool = np.zeros(valid.size, dtype=bool)
    empty_int = np.full(valid.size, -1, dtype=int)
    empty_float = np.full(valid.size, np.nan)
    return SeedSearchResult(
        np.nan,
        -1,
        False,
        "FAILED",
        np.nan,
        np.nan,
        np.nan,
        int(local_count + adaptive_count),
        -1,
        0.0,
        np.nan,
        np.nan,
        np.nan,
        0,
        0,
        0,
        0,
        0,
        empty_bool,
        empty_int,
        empty_float,
        empty_float,
        int(local_count),
        int(adaptive_count),
        int(transitions),
        perf_counter() - started,
        False,
    )


def discover_boundary_seed_candidates(
    flat_gather, *, receiver_x, valid_receiver, seed_receiver, inward_direction,
    state_valid, dt, t0=0.0, pilot_short_receiver_count=25,
    pilot_min_coverage=0.80, pilot_min_median_zncc=0.70,
    pilot_min_side_support_receivers=3, phase_switch_zncc=0.0,
    phase_switch_prediction_time=0.015, evidence_half_width_time=0.020,
    pilot_long_aperture_m=800.0, local_receiver_count=21,
    max_ownership_hypotheses=12,
    anchor=None, tracker_options=None, reference_time=None,
    upper_neighbor_state_valid=None, lower_neighbor_state_valid=None, **_unused,
):
    """Evaluate boundary positive-peak hypotheses for repair.

    The complete short-pilot path is retained in every returned row so the
    repair caller can identify a coherent event first and choose a seed on
    that event second.  Normal ``discover_tracking_seed`` behavior is not
    changed by this repair-only data exposure.
    """

    flat = np.asarray(flat_gather, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    ownership = np.asarray(state_valid, dtype=bool)
    seed = int(seed_receiver)
    direction = int(inward_direction)
    if direction not in (-1, 1):
        raise ValueError("inward_direction must be -1 or 1")
    if flat.ndim != 2 or x.shape != (flat.shape[0],):
        raise ValueError("flat_gather/receiver_x shapes are inconsistent")
    if valid.shape != (flat.shape[0],) or ownership.shape != flat.shape:
        raise ValueError("valid_receiver/state_valid shapes are inconsistent")
    if not 0 <= seed < flat.shape[0] or not valid[seed]:
        return ()

    segment = _contiguous_valid_mask(valid, seed)
    inward = np.arange(seed, flat.shape[0] if direction > 0 else -1, direction)
    inward = inward[segment[inward]]
    short_subset = np.sort(inward[: int(pilot_short_receiver_count)])
    evidence_subset = np.sort(inward[: int(local_receiver_count)])
    long_subset = np.sort(
        inward[np.abs(x[inward] - x[seed]) <= float(pilot_long_aperture_m)]
    )
    if seed not in long_subset:
        long_subset = np.sort(np.r_[long_subset, seed])
    traces = _normalised(flat, valid)
    envelope = np.abs(hilbert(traces[evidence_subset], axis=-1))
    time = float(t0) + np.arange(flat.shape[1]) * float(dt)
    evidence_half = max(1, int(round(float(evidence_half_width_time) / float(dt))))

    # Repair-only search domain: do not force the seed back into the failed
    # reflector's original ownership.  The neighboring reflectors are the
    # physical bounds.  Reuse the existing evidence half-width as the guard so
    # there is no new tuning parameter.
    repair_ownership = build_interlayer_repair_state_valid(
        ownership,
        upper_neighbor_state_valid,
        lower_neighbor_state_valid,
        guard_samples=evidence_half,
    )
    safe_hypotheses = _hypotheses(
        traces, envelope, seed, evidence_subset, time, repair_ownership[seed],
        max_hypotheses=0, evidence_half_samples=evidence_half,
    )
    # Keep the old current-ownership count only as a diagnostic.  It no longer
    # controls which peaks are allowed to enter repair validation.
    ownership_hypotheses = _hypotheses(
        traces, envelope, seed, evidence_subset, time, ownership[seed],
        max_hypotheses=0, evidence_half_samples=evidence_half,
    )
    all_hypotheses = safe_hypotheses
    hypotheses = list(all_hypotheses[: int(max_ownership_hypotheses)])
    strongest = sorted(all_hypotheses, key=lambda item: item[2], reverse=True)[:2]
    selected_samples = {int(item[1]) for item in hypotheses}
    hypotheses.extend(item for item in strongest if int(item[1]) not in selected_samples)

    options = dict(tracker_options or {})
    options.update(
        dt=float(dt), t0=float(t0), phase_switch_zncc=phase_switch_zncc,
        phase_switch_prediction_time=phase_switch_prediction_time,
        hard_neighbor_zncc_gate=False,
        continuation_rescue={"enabled": False, "max_valid_receiver_gap": 0},
    )
    anchor_options = dict(anchor or {})
    anchor_min = int(anchor_options.get("min_receivers_per_side", 3))
    anchor_min_rho = float(anchor_options.get("min_neighbor_zncc", 0.50))
    anchor_max_prediction = float(
        anchor_options.get("max_prediction_error_time", phase_switch_prediction_time)
    )
    rows = []
    for index, hypothesis in enumerate(hypotheses):
        short = _pilot(
            flat, x, valid, repair_ownership, seed, hypothesis, options, short_subset,
            pilot_min_side_support_receivers, pilot_min_coverage,
            pilot_min_median_zncc, phase_switch_zncc,
            phase_switch_prediction_time,
        )
        accepted = short is not None and short.passed and short.phase_switch_count == 0
        mask = np.zeros(valid.size, dtype=bool)
        left = right = 0
        if accepted:
            mask, left, right, accepted = _anchor(
                short, traces, seed, anchor_min, anchor_min_rho,
                anchor_max_prediction,
                float(options.get("max_residual_slope_ms_per_100m", 150.0)),
            )
        long = None
        if accepted:
            long = _pilot(
                flat, x, valid, repair_ownership, seed, hypothesis, options, long_subset,
                pilot_min_side_support_receivers, pilot_min_coverage,
                pilot_min_median_zncc, phase_switch_zncc,
                phase_switch_prediction_time,
            )
            accepted = long is not None
        used = np.flatnonzero(short.success_mask & short.tested_mask) if short is not None else np.empty(0, int)
        amplitudes = traces[used, short.pick_sample[used]] if used.size else np.empty(0)
        amplitudes = amplitudes[np.isfinite(amplitudes) & (amplitudes > 0.0)]
        energy = float(np.median(amplitudes)) if amplitudes.size else 0.0
        result = None
        if accepted:
            anchor_sample = np.where(mask, short.pick_sample, -1)
            anchor_time = np.where(mask, short.pick_time, np.nan)
            result = SeedSearchResult(
                hypothesis[0], hypothesis[1], True, "BOUNDARY_REPAIR",
                hypothesis[2], hypothesis[3], hypothesis[4], len(hypotheses), index,
                long.coverage, long.median_zncc,
                long.median_abs_prediction_error,
                long.median_abs_slope_ms_per_100m, long.phase_switch_count,
                long.left_support, long.right_support, left, right, mask,
                anchor_sample, anchor_time, long.pick_time, 0, len(hypotheses),
                short.transition_count + long.transition_count, 0.0, False,
            )
        rank = (
            long.coverage if long is not None else 0.0,
            -(long.phase_switch_count if long is not None else np.inf),
            -(long.p90_abs_slope_ms_per_100m if long is not None else np.inf),
            long.median_zncc if long is not None else -np.inf,
            -(long.median_abs_slope_ms_per_100m if long is not None else np.inf),
            -(long.median_abs_prediction_error if long is not None else np.inf),
            hypothesis[3], hypothesis[4], int(left + right + 1),
            -(abs(float(hypothesis[0]) - float(reference_time)) if reference_time is not None else 0.0),
            hypothesis[2],
        )
        rows.append((rank, hypothesis, short, long, energy, result))

    accepted_rows = sorted((row for row in rows if row[5] is not None), key=lambda row: row[0], reverse=True)
    selected_ids = {id(row) for row in accepted_rows[:3]}
    output = []
    for row in rows:
        rank, hypothesis, short, long, energy, result = row
        if short is None:
            rejection = "PILOT_FAILED"
        elif short.phase_switch_count:
            rejection = "PHASE_SWITCH"
        elif short.coverage < float(pilot_min_coverage):
            rejection = "PILOT_LOW_COVERAGE"
        elif short.median_zncc < float(pilot_min_median_zncc):
            rejection = "PILOT_LOW_ZNCC"
        elif not short.passed:
            rejection = "PILOT_SUPPORT_FAILED"
        elif result is None:
            rejection = "ANCHOR_FAILED" if long is None else "LONG_PILOT_FAILED"
        else:
            rejection = ""
        output.append(BoundarySeedCandidate(
            float(hypothesis[0]), int(hypothesis[1]), float(hypothesis[2]),
            float(short.coverage) if short is not None else 0.0,
            float(short.median_zncc) if short is not None else np.nan,
            float(short.median_abs_prediction_error) if short is not None else np.nan,
            int(short_subset[0]), int(short_subset[-1]),
            int(np.count_nonzero(short.success_mask[short_subset])) if short is not None else 0,
            int(short_subset.size),
            int(np.count_nonzero(result.anchor_receiver_mask)) if result is not None else 0,
            float(energy), float(hypothesis[3]), float(hypothesis[4]),
            float(long.coverage) if long is not None else 0.0,
            float(long.median_zncc) if long is not None else np.nan,
            float(long.median_abs_prediction_error) if long is not None else np.nan,
            float(long.median_abs_slope_ms_per_100m) if long is not None else np.nan,
            float(long.p90_abs_slope_ms_per_100m) if long is not None else np.nan,
            int(long.phase_switch_count) if long is not None else 0,
            len(ownership_hypotheses), len(safe_hypotheses),
            False, tuple(rank), rejection, id(row) in selected_ids, result,
            short,
        ))
    return tuple(output)


def discover_tracking_seed(
    flat_gather,
    *,
    receiver_x,
    valid_receiver,
    seed_receiver,
    control_time,
    state_valid,
    dt,
    t0=0.0,
    local_half_width_time=0.030,
    adaptive_enabled=True,
    adaptive_max_half_width_time=0.150,
    adaptive_ownership_fraction=0.40,
    local_receiver_count=21,
    max_positive_hypotheses=8,
    evidence_half_width_time=0.020,
    pilot_short_receiver_count=25,
    pilot_long_aperture_m=800.0,
    pilot_min_coverage=0.80,
    pilot_min_median_zncc=0.70,
    pilot_min_side_support_receivers=3,
    phase_switch_zncc=0.0,
    phase_switch_prediction_time=0.015,
    local_high_conf_max_median_abs_slope_ms_per_100m=40.0,
    local_high_conf_min_positive_support_fraction=0.60,
    local_high_conf_min_evidence_ratio=0.70,
    ownership_wide_enabled=True,
    max_ownership_hypotheses=12,
    repair_exhaustive=False,
    excluded_seed_times=None,
    continuation_rescue=None,
    anchor=None,
    tracker_options=None,
    **_legacy,
):
    """Return the best positive seed and a fixed pilot-validated anchor.

    LOCAL is accepted early only when it is not merely trackable, but also
    strongly supported by near-seed observed data.  Otherwise ADAPTIVE is
    evaluated and competes with the LOCAL candidates.
    """
    started = perf_counter()
    flat = np.asarray(flat_gather, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    ownership = np.asarray(state_valid, dtype=bool)
    seed = int(seed_receiver)

    if flat.ndim != 2 or x.shape != (flat.shape[0],):
        raise ValueError("flat_gather/receiver_x shapes are inconsistent")
    if valid.shape != (flat.shape[0],) or ownership.shape != flat.shape:
        raise ValueError("valid_receiver/state_valid shapes are inconsistent")
    if not 0 <= seed < flat.shape[0] or not valid[seed]:
        return _empty_result(valid, started)

    time = float(t0) + np.arange(flat.shape[1]) * float(dt)
    traces = _normalised(flat, valid)

    segment = _contiguous_valid_mask(valid, seed)
    evidence_receivers = _centered_indices(segment, seed, local_receiver_count)
    short_subset = _centered_indices(segment, seed, pilot_short_receiver_count)
    long_subset = np.flatnonzero(
        segment & (np.abs(x - x[seed]) <= float(pilot_long_aperture_m))
    )
    if seed not in long_subset:
        long_subset = np.sort(np.r_[long_subset, seed])

    if evidence_receivers.size:
        evidence_envelope = np.abs(hilbert(traces[evidence_receivers], axis=-1))
    else:
        evidence_envelope = np.empty((0, flat.shape[1]), dtype=float)

    # Seed validation must be stricter than the production continuation.
    # In particular, pilot paths are not allowed to rescue a missed peak or
    # jump over a valid receiver.  Otherwise an incorrect seed can look
    # artificially robust simply because the continuation machinery repaired
    # it during validation.
    options = dict(tracker_options or {})
    options.update(
        dt=float(dt),
        t0=float(t0),
        phase_switch_zncc=phase_switch_zncc,
        phase_switch_prediction_time=phase_switch_prediction_time,
        hard_neighbor_zncc_gate=False,
        continuation_rescue={"enabled": False, "max_valid_receiver_gap": 0},
    )
    evidence_half = max(1, int(round(float(evidence_half_width_time) / float(dt))))
    anchor_options = dict(anchor or {})
    anchor_min = int(anchor_options.get("min_receivers_per_side", 3))
    anchor_min_rho = float(anchor_options.get("min_neighbor_zncc", 0.50))
    anchor_max_prediction = float(
        anchor_options.get("max_prediction_error_time", phase_switch_prediction_time)
    )

    evaluations = []
    evaluated_samples = set()
    transitions = 0

    def hypotheses_for(half_width, max_hypotheses=None):
        search_valid = ownership[seed].copy()
        if excluded_seed_times:
            for failed_time in excluded_seed_times:
                search_valid &= (
                    np.abs(time - float(failed_time))
                    > float(evidence_half_width_time)
                )
        if half_width is not None:
            search_valid &= np.abs(time - float(control_time)) <= float(half_width)
        return _hypotheses(
            traces,
            evidence_envelope,
            seed,
            evidence_receivers,
            time,
            search_valid,
            max_hypotheses=(max_positive_hypotheses if max_hypotheses is None else max_hypotheses),
            evidence_half_samples=evidence_half,
        )

    def evaluate_hypotheses(hypotheses, mode):
        nonlocal transitions
        for index, hypothesis in enumerate(hypotheses):
            sample = int(hypothesis[1])
            if sample in evaluated_samples:
                continue
            evaluated_samples.add(sample)

            short = _pilot(
                flat,
                x,
                valid,
                ownership,
                seed,
                hypothesis,
                options,
                short_subset,
                pilot_min_side_support_receivers,
                pilot_min_coverage,
                pilot_min_median_zncc,
                phase_switch_zncc,
                phase_switch_prediction_time,
            )
            if short is None:
                continue
            transitions += short.transition_count
            if not short.passed:
                continue

            mask, left, right, anchor_ok = _anchor(
                short, traces, seed, anchor_min, anchor_min_rho,
                anchor_max_prediction,
                float(options.get("max_residual_slope_ms_per_100m", 150.0)),
            )
            if not anchor_ok or short.phase_switch_count:
                continue

            long = _pilot(
                flat,
                x,
                valid,
                ownership,
                seed,
                hypothesis,
                options,
                long_subset,
                pilot_min_side_support_receivers,
                pilot_min_coverage,
                pilot_min_median_zncc,
                phase_switch_zncc,
                phase_switch_prediction_time,
            )
            if long is None:
                continue
            transitions += long.transition_count

            accepted = True
            anchor_length = int(left + right + 1)
            rank = (
                # Once a short, contiguous anchor exists, event identity is
                # better judged by how naturally the branch continues over a
                # wider aperture.  Anchor length therefore acts only as a
                # later tie-break instead of dominating the ranking.
                long.coverage,
                -long.phase_switch_count,
                -long.p90_abs_slope_ms_per_100m,
                long.median_zncc,
                -long.median_abs_slope_ms_per_100m,
                -long.median_abs_prediction_error,
                hypothesis[3],
                hypothesis[4],
                anchor_length,
                -abs(hypothesis[0] - float(control_time)),
                hypothesis[2],
            )
            evaluations.append(
                (
                    accepted,
                    rank,
                    mode,
                    index,
                    hypothesis,
                    long,
                    mask,
                    left,
                    right,
                    short,
                )
            )

    local = hypotheses_for(float(local_half_width_time))
    evaluate_hypotheses(local, "LOCAL")
    local_ok = [
        item for item in evaluations if item[0] and item[2] == "LOCAL"
    ]

    ownership_samples = np.flatnonzero(ownership[seed])
    if ownership_samples.size:
        ownership_width = float(
            (ownership_samples[-1] - ownership_samples[0] + 1) * float(dt)
        )
    else:
        ownership_width = 0.0
    adaptive_half = min(
        float(adaptive_max_half_width_time),
        float(adaptive_ownership_fraction) * ownership_width,
    )
    adaptive_probe = (
        hypotheses_for(adaptive_half)
        if adaptive_enabled and adaptive_half > float(local_half_width_time)
        else []
    )

    # Always perform the *cheap* positive-peak/evidence scan over the complete
    # ownership corridor.  Keep the original evidence-ranked shortlist, but
    # additionally rescue up to two strongest positive seed-trace peaks.
    #
    # This is intentionally only a recall safety net:
    # - the original top evidence hypotheses are unchanged;
    # - no existing candidate is removed;
    # - at most two extra candidates are allowed to enter pilot validation.
    ownership_all = (
        hypotheses_for(None, 0)
        if ownership_wide_enabled
        else []
    )

    if ownership_wide_enabled:
        if repair_exhaustive:
            ownership_probe = ownership_all
        else:
            ownership_probe = ownership_all[: int(max_ownership_hypotheses)]

            # Rescue up to two strongest positive peaks on the seed trace.
            # hypothesis[2] is the positive seed-trace amplitude.
            amplitude_rescue = sorted(
                ownership_all,
                key=lambda item: item[2],
                reverse=True,
            )[:2]

            existing_samples = {int(item[1]) for item in ownership_probe}
            for item in amplitude_rescue:
                sample = int(item[1])
                if sample not in existing_samples:
                    ownership_probe.append(item)
                    existing_samples.add(sample)
    else:
        ownership_probe = []

    best_probe_evidence = max(
        (item[3] for item in ownership_probe),
        default=max(
            (item[3] for item in adaptive_probe),
            default=max((item[3] for item in local), default=0.0),
        ),
    )

    high_confidence_items = []
    for item in local_ok:
        hypothesis = item[4]
        pilot = item[5]
        evidence_ratio = (
            hypothesis[3] / best_probe_evidence
            if best_probe_evidence > np.finfo(float).eps
            else 1.0
        )
        if (
            pilot.passed
            and pilot.phase_switch_count == 0
            and pilot.median_abs_slope_ms_per_100m
            <= float(local_high_conf_max_median_abs_slope_ms_per_100m)
            and hypothesis[4]
            >= float(local_high_conf_min_positive_support_fraction)
            and evidence_ratio >= float(local_high_conf_min_evidence_ratio)
        ):
            high_confidence_items.append(item)

    # ``high_confidence`` is now a diagnostic property, not an early-exit
    # switch.  A smooth but wrong event near the theoretical control can satisfy
    # every local criterion.  Because the complete ownership scan is cheap
    # compared with the one-time inversion workflow, all shortlisted positive
    # peaks are allowed to compete before the winner is frozen.
    high_confidence = bool(high_confidence_items)
    adaptive = []
    if adaptive_enabled and adaptive_half > float(local_half_width_time):
        adaptive = adaptive_probe
        evaluate_hypotheses(adaptive, "ADAPTIVE")

    ownership_wide = []
    # If LOCAL was not sufficiently convincing, complete the search over the
    # entire ownership corridor.  Evaluated samples are de-duplicated, so this
    # only pilots previously unseen positive peaks.  This is deliberately more
    # conservative than trusting a merely "credible" ADAPTIVE branch: a wrong
    # but smooth event can look excellent inside ±150 ms while the true Rk peak
    # lies farther away inside the same reflector ownership.
    if ownership_wide_enabled:
        ownership_wide = ownership_probe
        evaluate_hypotheses(ownership_wide, "OWNERSHIP_WIDE")

    accepted = [item for item in evaluations if item[0]]

    if not accepted:
        return _empty_result(
            valid,
            started,
            local_count=len(local),
            adaptive_count=len(adaptive) + len(ownership_wide),
            transitions=transitions,
        )


    # First choose the winner exactly as before.
    selected_item = max(
        accepted,
        key=lambda item: item[1],
    )

    # ------------------------------------------------------------------
    # Weak-axis safety rescue
    #
    # Do NOT change the normal ranking.  Only override the normal winner
    # when it is dramatically weaker than another already pilot-validated
    # event, while that stronger event has almost the same long-pilot
    # coverage and no long-pilot phase switch.
    # ------------------------------------------------------------------
    def tracked_event_strength(item):
        long_pilot = item[5]

        used = (
            long_pilot.success_mask
            & (long_pilot.pick_sample >= 0)
        )
        receivers_used = np.flatnonzero(used)

        if receivers_used.size == 0:
            return 0.0

        samples_used = long_pilot.pick_sample[receivers_used]
        amplitudes = traces[receivers_used, samples_used]

        amplitudes = amplitudes[
            np.isfinite(amplitudes)
            & (amplitudes > 0.0)
        ]

        if amplitudes.size == 0:
            return 0.0

        return float(np.median(amplitudes))

    selected_strength = tracked_event_strength(selected_item)

    strongest_item = max(
        accepted,
        key=tracked_event_strength,
    )
    strongest_strength = tracked_event_strength(strongest_item)

    selected_long = selected_item[5]
    strongest_long = strongest_item[5]

    if (
        strongest_item is not selected_item
        and strongest_strength > np.finfo(float).eps

        # Normal winner must be dramatically weaker:
        # less than 45% of the strongest accepted event.
        and selected_strength < 0.45 * strongest_strength

        # The stronger event must track almost as completely:
        # allow at most 5 percentage points less coverage.
        and strongest_long.coverage >= selected_long.coverage - 0.05

        # Never rescue a branch with a long-pilot phase switch.
        and strongest_long.phase_switch_count == 0
    ):
        selected_item = strongest_item

    _, _, mode, index, hypothesis, pilot, mask, left, right, anchor_pilot = selected_item

    # The mask and frozen picks must come from the same short pilot.  The long
    # pilot is ranking evidence only and may choose a different local sample.
    anchor_sample = np.where(mask, anchor_pilot.pick_sample, -1)
    anchor_time = np.where(mask, anchor_pilot.pick_time, np.nan)
    total_hypotheses = len(local) + len(adaptive) + len(ownership_wide)

    return SeedSearchResult(
        hypothesis[0],
        hypothesis[1],
        True,
        mode,
        hypothesis[2],
        hypothesis[3],
        hypothesis[4],
        total_hypotheses,
        index,
        pilot.coverage,
        pilot.median_zncc,
        pilot.median_abs_prediction_error,
        pilot.median_abs_slope_ms_per_100m,
        pilot.phase_switch_count,
        pilot.left_support,
        pilot.right_support,
        left,
        right,
        mask,
        anchor_sample,
        anchor_time,
        pilot.pick_time,
        len(local),
        len(adaptive),
        transitions,
        perf_counter() - started,
        bool(mode == "LOCAL" and any(selected_item is item for item in high_confidence_items)),
    )


__all__ = [
    "BoundarySeedCandidate", "SeedPilotResult", "SeedSearchResult",
    "build_interlayer_repair_state_valid",
    "discover_boundary_seed_candidates", "discover_tracking_seed",
]
