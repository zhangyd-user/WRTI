"""Positive-peak seed discovery with staged movable-receiver validation.

This module keeps the repository's existing sparse pilot/anchor logic, but makes
normal seed discovery cheaper without giving up the movable-receiver search:

1. candidate peaks over LOCAL / ADAPTIVE / OWNERSHIP are scanned cheaply first;
2. LOCAL is validated before wider time modes;
3. only a sparse first batch of nearby seed receivers is piloted initially;
4. accepted short pilots that describe the same event are clustered;
5. only a small number of distinct events receive the expensive long pilot;
6. the receiver aperture is expanded only if the first batch fails;
7. ADAPTIVE and OWNERSHIP are entered only when the previous tier fails.

The only production behavior added beyond current GitHub ``main`` is seed
search itself: movable seed receiver, apex-like diffraction rejection, and the
staged/clustered pilot schedule.  Full-event DP scoring is unchanged.
"""

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
    left_signed_slope_ms_per_100m: float = np.nan
    right_signed_slope_ms_per_100m: float = np.nan
    macro_curvature_time: float = np.nan


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
    # New diagnostics are appended with defaults to preserve compatibility with
    # the existing positional constructors in repair code/tests.
    seed_receiver: int = -1
    diffraction_risk: bool = False
    macro_curvature_time: float = np.nan
    left_signed_slope_ms_per_100m: float = np.nan
    right_signed_slope_ms_per_100m: float = np.nan


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
    """Robust near-seed evidence for one positive seed-trace peak."""
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

    items.sort(key=lambda item: (item[3], item[4], item[2]), reverse=True)
    if int(max_hypotheses) > 0:
        items = items[: int(max_hypotheses)]
    return items


def build_interlayer_repair_state_valid(
    state_valid,
    upper_neighbor_state_valid=None,
    lower_neighbor_state_valid=None,
    *,
    guard_samples=0,
):
    """Return the repair-only safe band between adjacent reflector ownerships."""
    current = np.asarray(state_valid, dtype=bool)
    if current.ndim != 2:
        raise ValueError("state_valid must have shape [nreceiver, ntime]")
    upper = (
        None
        if upper_neighbor_state_valid is None
        else np.asarray(upper_neighbor_state_valid, dtype=bool)
    )
    lower = (
        None
        if lower_neighbor_state_valid is None
        else np.asarray(lower_neighbor_state_valid, dtype=bool)
    )
    for name, value in (
        ("upper_neighbor_state_valid", upper),
        ("lower_neighbor_state_valid", lower),
    ):
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

    for neighbor in (upper, lower):
        if neighbor is None:
            continue
        forbidden = neighbor.copy()
        for step in range(1, guard + 1):
            forbidden[:, step:] |= neighbor[:, :-step]
            forbidden[:, :-step] |= neighbor[:, step:]
        safe &= ~forbidden
    return safe


def _geometry_metrics(pick_time, success_mask, x, seed, mask=None):
    """Return signed side slopes and a scale-independent macro curvature time.

    ``macro_curvature_time`` is the magnitude of the quadratic term at the
    edge of the evaluated receiver aperture.  It is therefore measured in
    seconds and can be compared directly with a time threshold.
    """
    pick = np.asarray(pick_time, dtype=float)
    success = np.asarray(success_mask, dtype=bool) & np.isfinite(pick)
    if mask is not None:
        success &= np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(success)
    if indices.size < 3:
        return np.nan, np.nan, np.nan, 0, 0

    def side_slope(side_indices):
        side_indices = np.asarray(side_indices, dtype=int)
        if side_indices.size < 2:
            return np.nan
        dx = np.diff(x[side_indices])
        dt = np.diff(pick[side_indices])
        usable = np.isfinite(dx) & np.isfinite(dt) & (dx != 0.0)
        if not np.any(usable):
            return np.nan
        return float(np.median(dt[usable] / dx[usable]) * 1e5)

    left_indices = indices[indices <= int(seed)]
    right_indices = indices[indices >= int(seed)]
    left = side_slope(left_indices)
    right = side_slope(right_indices)

    curvature = np.nan
    if indices.size >= 5:
        xc = float(x[int(seed)])
        dx = np.asarray(x[indices] - xc, dtype=float)
        scale = float(np.max(np.abs(dx)))
        if np.isfinite(scale) and scale > 0.0:
            u = dx / scale
            try:
                coefficient = np.polyfit(u, pick[indices], 2)
                curvature = float(abs(coefficient[0]))
            except (ValueError, np.linalg.LinAlgError):
                curvature = np.nan

    return (
        left,
        right,
        curvature,
        int(np.count_nonzero(indices < int(seed))),
        int(np.count_nonzero(indices > int(seed))),
    )


def _diffraction_risk(
    pilot,
    x,
    seed,
    anchor_mask,
    *,
    max_macro_curvature_time,
    min_opposite_slope_ms_per_100m,
    min_points_per_side,
):
    """Reject only a genuine apex-like anchor, not generic curvature.

    A real reflector may retain noticeable curvature after imperfect Eikonal
    flattening.  Curvature by itself is therefore diagnostic only.  A short
    production anchor is rejected only when both sides are sufficiently
    populated, their signed slopes are significant and opposite, and the
    quadratic departure is also non-trivial.  This avoids the v1 failure mode
    that could reject a legitimate smooth curved reflector merely because its
    fitted quadratic coefficient was large.
    """
    left, right, curvature, left_count, right_count = _geometry_metrics(
        pilot.pick_time,
        pilot.success_mask,
        x,
        seed,
        mask=anchor_mask,
    )
    enough_sides = (
        left_count >= int(min_points_per_side)
        and right_count >= int(min_points_per_side)
    )
    opposite = bool(
        enough_sides
        and np.isfinite(left)
        and np.isfinite(right)
        and left * right < 0.0
        and min(abs(left), abs(right))
        >= float(min_opposite_slope_ms_per_100m)
    )
    apex_like = bool(
        opposite
        and np.isfinite(curvature)
        and curvature >= 0.5 * float(max_macro_curvature_time)
    )
    return apex_like, left, right, curvature



def _quantize(value, step, *, default=np.inf):
    """Stable rank value that is insensitive to floating round-off noise."""
    if not np.isfinite(value):
        return float(default)
    step = float(step)
    return float(np.rint(float(value) / step) * step) if step > 0.0 else float(value)


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
    try:
        tracked = track_flattened_event_sparse(
            flat[subset],
            receiver_x=x[subset],
            valid_receiver=valid[subset],
            seed_receiver=local_seed,
            seed_time=hypothesis[0],
            state_valid=ownership[subset],
            **options,
        )
    except ValueError as error:
        if str(error) != "same-x seed waveform window is outside the record":
            raise
        return None

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
    left_signed, right_signed, curvature, _, _ = _geometry_metrics(
        pick_time, mask, x, seed
    )

    return SeedPilotResult(
        passed=passed,
        coverage=coverage,
        median_zncc=median_rho,
        median_abs_prediction_error=median_prediction,
        median_abs_slope_ms_per_100m=median_slope,
        p90_abs_slope_ms_per_100m=p90_slope,
        phase_switch_count=phase_count,
        pick_sample=sample,
        pick_time=pick_time,
        success_mask=mask,
        neighbor_zncc=neighbor,
        residual_slope=slope,
        prediction_error=pred,
        tested_mask=tested,
        left_support=left_success,
        right_support=right_success,
        transition_count=tracked.transition_count,
        left_signed_slope_ms_per_100m=left_signed,
        right_signed_slope_ms_per_100m=right_signed,
        macro_curvature_time=curvature,
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
        seed_time=np.nan,
        seed_sample=-1,
        valid=False,
        mode="FAILED",
        positive_amplitude=np.nan,
        stack_evidence=np.nan,
        positive_support_fraction=np.nan,
        hypothesis_count=int(local_count + adaptive_count),
        selected_hypothesis=-1,
        pilot_coverage=0.0,
        pilot_median_zncc=np.nan,
        pilot_median_abs_prediction_error=np.nan,
        pilot_median_abs_slope_ms_per_100m=np.nan,
        phase_switch_count=0,
        pilot_left_support=0,
        pilot_right_support=0,
        anchor_left_count=0,
        anchor_right_count=0,
        anchor_receiver_mask=empty_bool,
        anchor_pick_sample=empty_int,
        anchor_pick_time=empty_float,
        pilot_pick_time=empty_float,
        local_hypothesis_count=int(local_count),
        adaptive_hypothesis_count=int(adaptive_count),
        pilot_transition_count=int(transitions),
        runtime_seconds=perf_counter() - started,
        high_confidence=False,
    )


def discover_boundary_seed_candidates(
    flat_gather,
    *,
    receiver_x,
    valid_receiver,
    seed_receiver,
    inward_direction,
    state_valid,
    dt,
    t0=0.0,
    pilot_short_receiver_count=25,
    pilot_min_coverage=0.80,
    pilot_min_median_zncc=0.70,
    pilot_min_side_support_receivers=3,
    phase_switch_zncc=0.0,
    phase_switch_prediction_time=0.015,
    evidence_half_width_time=0.020,
    pilot_long_aperture_m=800.0,
    local_receiver_count=21,
    max_ownership_hypotheses=12,
    anchor=None,
    tracker_options=None,
    reference_time=None,
    upper_neighbor_state_valid=None,
    lower_neighbor_state_valid=None,
    **_unused,
):
    """Evaluate boundary positive-peak hypotheses for repair.

    This repair-only search intentionally keeps its existing fixed boundary
    receiver semantics; movable receiver selection is for the primary bootstrap
    seed.  All current repair diagnostics are preserved.
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
    inward = np.arange(
        seed, flat.shape[0] if direction > 0 else -1, direction
    )
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
    evidence_half = max(
        1, int(round(float(evidence_half_width_time) / float(dt)))
    )

    repair_ownership = build_interlayer_repair_state_valid(
        ownership,
        upper_neighbor_state_valid,
        lower_neighbor_state_valid,
        guard_samples=evidence_half,
    )
    safe_hypotheses = _hypotheses(
        traces,
        envelope,
        seed,
        evidence_subset,
        time,
        repair_ownership[seed],
        max_hypotheses=0,
        evidence_half_samples=evidence_half,
    )
    ownership_hypotheses = _hypotheses(
        traces,
        envelope,
        seed,
        evidence_subset,
        time,
        ownership[seed],
        max_hypotheses=0,
        evidence_half_samples=evidence_half,
    )
    all_hypotheses = safe_hypotheses
    hypotheses = list(all_hypotheses[: int(max_ownership_hypotheses)])
    strongest = sorted(all_hypotheses, key=lambda item: item[2], reverse=True)[:2]
    selected_samples = {int(item[1]) for item in hypotheses}
    hypotheses.extend(
        item for item in strongest if int(item[1]) not in selected_samples
    )

    options = dict(tracker_options or {})
    options.update(
        dt=float(dt),
        t0=float(t0),
        phase_switch_zncc=phase_switch_zncc,
        phase_switch_prediction_time=phase_switch_prediction_time,
        hard_neighbor_zncc_gate=False,
        continuation_rescue={"enabled": False, "max_valid_receiver_gap": 0},
        # Seed/anchor pilots must remain identical to fastseedsearch.
        # Slow-guide protection is a production-continuation feature only.
        slow_guide={"enabled": False},
    )
    anchor_options = dict(anchor or {})
    anchor_min = int(anchor_options.get("min_receivers_per_side", 3))
    anchor_min_rho = float(anchor_options.get("min_neighbor_zncc", 0.50))
    anchor_max_prediction = float(
        anchor_options.get(
            "max_prediction_error_time", phase_switch_prediction_time
        )
    )

    rows = []
    for index, hypothesis in enumerate(hypotheses):
        short = _pilot(
            flat,
            x,
            valid,
            repair_ownership,
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
        accepted = (
            short is not None
            and short.passed
            and short.phase_switch_count == 0
        )
        mask = np.zeros(valid.size, dtype=bool)
        left = right = 0
        if accepted:
            mask, left, right, accepted = _anchor(
                short,
                traces,
                seed,
                anchor_min,
                anchor_min_rho,
                anchor_max_prediction,
                float(
                    options.get(
                        "max_residual_slope_ms_per_100m", 150.0
                    )
                ),
            )

        long = None
        if accepted:
            long = _pilot(
                flat,
                x,
                valid,
                repair_ownership,
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
            accepted = long is not None

        used = (
            np.flatnonzero(short.success_mask & short.tested_mask)
            if short is not None
            else np.empty(0, int)
        )
        amplitudes = (
            traces[used, short.pick_sample[used]]
            if used.size
            else np.empty(0)
        )
        amplitudes = amplitudes[
            np.isfinite(amplitudes) & (amplitudes > 0.0)
        ]
        energy = float(np.median(amplitudes)) if amplitudes.size else 0.0

        result = None
        if accepted:
            anchor_sample = np.where(mask, short.pick_sample, -1)
            anchor_time = np.where(mask, short.pick_time, np.nan)
            result = SeedSearchResult(
                seed_time=hypothesis[0],
                seed_sample=hypothesis[1],
                valid=True,
                mode="BOUNDARY_REPAIR",
                positive_amplitude=hypothesis[2],
                stack_evidence=hypothesis[3],
                positive_support_fraction=hypothesis[4],
                hypothesis_count=len(hypotheses),
                selected_hypothesis=index,
                pilot_coverage=long.coverage,
                pilot_median_zncc=long.median_zncc,
                pilot_median_abs_prediction_error=long.median_abs_prediction_error,
                pilot_median_abs_slope_ms_per_100m=long.median_abs_slope_ms_per_100m,
                phase_switch_count=long.phase_switch_count,
                pilot_left_support=long.left_support,
                pilot_right_support=long.right_support,
                anchor_left_count=left,
                anchor_right_count=right,
                anchor_receiver_mask=mask,
                anchor_pick_sample=anchor_sample,
                anchor_pick_time=anchor_time,
                pilot_pick_time=long.pick_time,
                local_hypothesis_count=0,
                adaptive_hypothesis_count=len(hypotheses),
                pilot_transition_count=short.transition_count + long.transition_count,
                runtime_seconds=0.0,
                high_confidence=False,
                seed_receiver=seed,
            )

        rank = (
            long.coverage if long is not None else 0.0,
            -(long.phase_switch_count if long is not None else np.inf),
            -(long.p90_abs_slope_ms_per_100m if long is not None else np.inf),
            long.median_zncc if long is not None else -np.inf,
            -(
                long.median_abs_slope_ms_per_100m
                if long is not None
                else np.inf
            ),
            -(
                long.median_abs_prediction_error
                if long is not None
                else np.inf
            ),
            hypothesis[3],
            hypothesis[4],
            int(left + right + 1),
            -(
                abs(float(hypothesis[0]) - float(reference_time))
                if reference_time is not None
                else 0.0
            ),
            hypothesis[2],
        )
        rows.append((rank, hypothesis, short, long, energy, result))

    accepted_rows = sorted(
        (row for row in rows if row[5] is not None),
        key=lambda row: row[0],
        reverse=True,
    )
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
        output.append(
            BoundarySeedCandidate(
                float(hypothesis[0]),
                int(hypothesis[1]),
                float(hypothesis[2]),
                float(short.coverage) if short is not None else 0.0,
                float(short.median_zncc) if short is not None else np.nan,
                float(short.median_abs_prediction_error)
                if short is not None
                else np.nan,
                int(short_subset[0]),
                int(short_subset[-1]),
                int(np.count_nonzero(short.success_mask[short_subset]))
                if short is not None
                else 0,
                int(short_subset.size),
                int(np.count_nonzero(result.anchor_receiver_mask))
                if result is not None
                else 0,
                float(energy),
                float(hypothesis[3]),
                float(hypothesis[4]),
                float(long.coverage) if long is not None else 0.0,
                float(long.median_zncc) if long is not None else np.nan,
                float(long.median_abs_prediction_error)
                if long is not None
                else np.nan,
                float(long.median_abs_slope_ms_per_100m)
                if long is not None
                else np.nan,
                float(long.p90_abs_slope_ms_per_100m)
                if long is not None
                else np.nan,
                int(long.phase_switch_count) if long is not None else 0,
                len(ownership_hypotheses),
                len(safe_hypotheses),
                False,
                tuple(rank),
                rejection,
                id(row) in selected_ids,
                result,
                short,
            )
        )
    return tuple(output)


def _seed_receiver_candidates(
    valid_segment,
    base_seed,
    *,
    enabled,
    half_width,
    stride,
    exhaustive,
):
    """Receiver candidates ordered from same-x outward on both sides."""
    segment = np.asarray(valid_segment, dtype=bool)
    base = int(base_seed)
    if not enabled:
        return np.asarray([base], dtype=int)
    half = max(0, int(half_width))
    step = 1 if exhaustive else max(1, int(stride))
    result = [base]
    for distance in range(step, half + 1, step):
        for receiver in (base - distance, base + distance):
            if 0 <= receiver < segment.size and segment[receiver]:
                result.append(int(receiver))
    if half > 0:
        for receiver in (base - half, base + half):
            if (
                0 <= receiver < segment.size
                and segment[receiver]
                and receiver not in result
            ):
                result.append(int(receiver))
    return np.asarray(result, dtype=int)


def _seed_receiver_batches(
    valid_segment,
    base_seed,
    *,
    enabled,
    search_half_width,
    search_stride,
    primary_half_width,
    primary_stride,
    exhaustive,
):
    """Return a cheap first receiver batch and an on-demand expansion batch."""
    all_receivers = _seed_receiver_candidates(
        valid_segment,
        base_seed,
        enabled=enabled,
        half_width=search_half_width,
        stride=search_stride,
        exhaustive=exhaustive,
    )
    if all_receivers.size <= 1:
        return all_receivers, np.empty(0, dtype=int)

    primary = _seed_receiver_candidates(
        valid_segment,
        base_seed,
        enabled=enabled,
        half_width=min(int(primary_half_width), int(search_half_width)),
        stride=primary_stride,
        exhaustive=False,
    )
    allowed = set(int(value) for value in all_receivers)
    primary = np.asarray(
        [int(value) for value in primary if int(value) in allowed], dtype=int
    )
    primary_set = set(int(value) for value in primary)
    expansion = np.asarray(
        [int(value) for value in all_receivers if int(value) not in primary_set],
        dtype=int,
    )
    return primary, expansion


@dataclass(frozen=True)
class _ShortSeedEvaluation:
    mode: str
    hypothesis_index: int
    seed_receiver: int
    hypothesis: tuple
    pilot: SeedPilotResult
    anchor_mask: np.ndarray
    anchor_left: int
    anchor_right: int
    diffraction_risk: bool
    left_signed_slope_ms_per_100m: float
    right_signed_slope_ms_per_100m: float
    macro_curvature_time: float


@dataclass(frozen=True)
class _LongSeedEvaluation:
    short: _ShortSeedEvaluation
    pilot: SeedPilotResult
    rank: tuple


def _production_seed_from_short(item: _ShortSeedEvaluation):
    indices = np.flatnonzero(item.anchor_mask)
    if indices.size == 0:
        return -1, -1, np.nan
    receiver = int(indices[len(indices) // 2])
    sample = int(item.pilot.pick_sample[receiver])
    time = float(item.pilot.pick_time[receiver])
    return receiver, sample, time


def _short_seed_rank(item, *, base_seed, control_time, slope_quantum, time_quantum):
    hypothesis = item.hypothesis
    pilot = item.pilot
    return (
        -_quantize(abs(hypothesis[0] - float(control_time)), time_quantum),
        pilot.coverage,
        pilot.median_zncc,
        -_quantize(pilot.p90_abs_slope_ms_per_100m, slope_quantum),
        -pilot.median_abs_prediction_error,
        hypothesis[3],
        hypothesis[4],
        int(item.anchor_left + item.anchor_right + 1),
        -abs(int(item.seed_receiver) - int(base_seed)),
        hypothesis[2],
    )


def _same_short_event(first, second, *, minimum_overlap, tolerance_time):
    first_mask = (
        np.asarray(first.pilot.success_mask, dtype=bool)
        & np.isfinite(first.pilot.pick_time)
    )
    second_mask = (
        np.asarray(second.pilot.success_mask, dtype=bool)
        & np.isfinite(second.pilot.pick_time)
    )
    overlap = first_mask & second_mask
    if int(np.count_nonzero(overlap)) < int(minimum_overlap):
        return False
    delta = np.abs(first.pilot.pick_time[overlap] - second.pilot.pick_time[overlap])
    if not delta.size:
        return False
    return bool(
        float(np.median(delta)) <= float(tolerance_time)
        and float(np.percentile(delta, 90.0)) <= 2.0 * float(tolerance_time)
    )


def _cluster_short_events(
    items,
    *,
    base_seed,
    control_time,
    minimum_overlap,
    tolerance_time,
    slope_quantum,
    time_quantum,
):
    """Greedily merge duplicate short pilots that describe the same event."""
    ranked = sorted(
        items,
        key=lambda item: _short_seed_rank(
            item,
            base_seed=base_seed,
            control_time=control_time,
            slope_quantum=slope_quantum,
            time_quantum=time_quantum,
        ),
        reverse=True,
    )
    clusters = []
    for item in ranked:
        placed = False
        for cluster in clusters:
            if _same_short_event(
                item,
                cluster[0],
                minimum_overlap=minimum_overlap,
                tolerance_time=tolerance_time,
            ):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters


def _apply_failed_seed_exclusions(
    items,
    excluded_seed_times,
    *,
    tolerance_time,
    base_seed,
    control_time,
    slope_quantum,
    time_quantum,
):
    """Suppress failed moved seeds without deleting that time on every receiver."""
    remaining = list(items)
    for failed in list(excluded_seed_times or ()):
        explicit_receiver = None
        if isinstance(failed, (tuple, list, np.ndarray)) and len(failed) >= 2:
            explicit_receiver = int(failed[0])
            failed_time = float(failed[1])
        else:
            failed_time = float(failed)

        matches = []
        for item in remaining:
            production_receiver, _, production_time = _production_seed_from_short(item)
            if not np.isfinite(production_time):
                continue
            if abs(production_time - failed_time) > float(tolerance_time):
                continue
            if explicit_receiver is not None and production_receiver != explicit_receiver:
                continue
            matches.append(item)
        if not matches:
            continue

        if explicit_receiver is not None:
            doomed = matches
        else:
            doomed = [max(
                matches,
                key=lambda item: _short_seed_rank(
                    item,
                    base_seed=base_seed,
                    control_time=control_time,
                    slope_quantum=slope_quantum,
                    time_quantum=time_quantum,
                ),
            )]
        doomed_ids = {id(item) for item in doomed}
        remaining = [item for item in remaining if id(item) not in doomed_ids]
    return remaining


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
    # Movable receiver search.  The first batch is intentionally sparse;
    # receivers omitted here are added only if that batch fails.
    movable_seed_enabled=True,
    seed_receiver_search_half_width=12,
    seed_receiver_search_stride=2,
    seed_receiver_primary_half_width=8,
    seed_receiver_primary_stride=4,
    # Duplicate short pilots are merged before the long pilot.
    seed_event_cluster_tolerance_time=0.008,
    seed_event_cluster_min_overlap_receivers=5,
    max_long_pilot_events_per_stage=2,
    # Reject only a genuine short-aperture apex-like seed anchor.
    diffraction_reject_enabled=True,
    diffraction_max_macro_curvature_time=0.020,
    diffraction_min_opposite_slope_ms_per_100m=15.0,
    diffraction_min_points_per_side=3,
    # Quantised ranking avoids tiny floating-point differences moving a
    # perfectly equivalent event away from same-x.
    seed_rank_slope_quantum_ms_per_100m=1.0,
    seed_rank_time_quantum=0.001,
    **_legacy,
):
    """Find a movable production seed with staged, on-demand pilot validation.

    Peak discovery over all receiver/time tiers remains broad and cheap.  Sparse
    DP pilots are the expensive part, so they are scheduled hierarchically:
    LOCAL before ADAPTIVE before OWNERSHIP_WIDE; sparse receiver batch before
    expansion; short pilots before event clustering; long pilots only for a few
    distinct short events.  The first successful tier returns immediately.
    """
    started = perf_counter()
    flat = np.asarray(flat_gather, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    ownership = np.asarray(state_valid, dtype=bool)
    base_seed = int(seed_receiver)

    if flat.ndim != 2 or x.shape != (flat.shape[0],):
        raise ValueError("flat_gather/receiver_x shapes are inconsistent")
    if valid.shape != (flat.shape[0],) or ownership.shape != flat.shape:
        raise ValueError("valid_receiver/state_valid shapes are inconsistent")
    if not 0 <= base_seed < flat.shape[0] or not valid[base_seed]:
        return _empty_result(valid, started)

    time = float(t0) + np.arange(flat.shape[1]) * float(dt)
    traces = _normalised(flat, valid)
    # Hilbert transform is paid once for the whole gather, not once per movable
    # seed receiver.
    full_envelope = np.abs(hilbert(traces, axis=-1))
    base_segment = _contiguous_valid_mask(valid, base_seed)
    primary_receivers, expansion_receivers = _seed_receiver_batches(
        base_segment,
        base_seed,
        enabled=bool(movable_seed_enabled),
        search_half_width=int(seed_receiver_search_half_width),
        search_stride=int(seed_receiver_search_stride),
        primary_half_width=int(seed_receiver_primary_half_width),
        primary_stride=int(seed_receiver_primary_stride),
        exhaustive=bool(repair_exhaustive),
    )
    all_receivers = np.r_[primary_receivers, expansion_receivers].astype(int)

    options = dict(tracker_options or {})
    options.update(
        dt=float(dt),
        t0=float(t0),
        phase_switch_zncc=phase_switch_zncc,
        phase_switch_prediction_time=phase_switch_prediction_time,
        hard_neighbor_zncc_gate=False,
        continuation_rescue={"enabled": False, "max_valid_receiver_gap": 0},
        # Seed/anchor pilots must remain identical to fastseedsearch.
        # Slow-guide protection is a production-continuation feature only.
        slow_guide={"enabled": False},
    )
    evidence_half = max(
        1, int(round(float(evidence_half_width_time) / float(dt)))
    )
    anchor_options = dict(anchor or {})
    anchor_min = int(anchor_options.get("min_receivers_per_side", 3))
    anchor_min_rho = float(anchor_options.get("min_neighbor_zncc", 0.50))
    anchor_max_prediction = float(
        anchor_options.get(
            "max_prediction_error_time", phase_switch_prediction_time
        )
    )
    max_anchor_slope = float(
        options.get("max_residual_slope_ms_per_100m", 150.0)
    )
    slope_quantum = float(seed_rank_slope_quantum_ms_per_100m)
    time_quantum = float(seed_rank_time_quantum)

    # --------------------------------------------------------------
    # Cheap scan: build positive-peak lists for every movable receiver and
    # every time-search tier.  No sparse DP is run here.
    # --------------------------------------------------------------
    contexts = {}
    local_count = adaptive_count = ownership_count = 0
    best_probe_evidence = 0.0
    for seed in all_receivers:
        seed = int(seed)
        segment = _contiguous_valid_mask(valid, seed)
        evidence_receivers = _centered_indices(
            segment, seed, local_receiver_count
        )
        short_subset = _centered_indices(
            segment, seed, pilot_short_receiver_count
        )
        long_subset = np.flatnonzero(
            segment & (np.abs(x - x[seed]) <= float(pilot_long_aperture_m))
        )
        if seed not in long_subset:
            long_subset = np.sort(np.r_[long_subset, seed])
        evidence_envelope = (
            full_envelope[evidence_receivers]
            if evidence_receivers.size
            else np.empty((0, flat.shape[1]), dtype=float)
        )

        def hypotheses_for(half_width, max_hypotheses=None):
            search_valid = ownership[seed].copy()
            if half_width is not None:
                search_valid &= (
                    np.abs(time - float(control_time)) <= float(half_width)
                )
            return _hypotheses(
                traces,
                evidence_envelope,
                seed,
                evidence_receivers,
                time,
                search_valid,
                max_hypotheses=(
                    max_positive_hypotheses
                    if max_hypotheses is None
                    else max_hypotheses
                ),
                evidence_half_samples=evidence_half,
            )

        local = hypotheses_for(float(local_half_width_time))
        ownership_samples = np.flatnonzero(ownership[seed])
        ownership_width = (
            float(
                (ownership_samples[-1] - ownership_samples[0] + 1)
                * float(dt)
            )
            if ownership_samples.size
            else 0.0
        )
        adaptive_half = min(
            float(adaptive_max_half_width_time),
            float(adaptive_ownership_fraction) * ownership_width,
        )
        adaptive = (
            hypotheses_for(adaptive_half)
            if adaptive_enabled
            and adaptive_half > float(local_half_width_time)
            else []
        )
        ownership_all = (
            hypotheses_for(None, 0) if ownership_wide_enabled else []
        )
        if ownership_wide_enabled:
            if repair_exhaustive:
                ownership_probe = list(ownership_all)
            else:
                ownership_probe = list(
                    ownership_all[: int(max_ownership_hypotheses)]
                )
                strongest = sorted(
                    ownership_all, key=lambda item: item[2], reverse=True
                )[:2]
                existing_samples = {int(item[1]) for item in ownership_probe}
                for item in strongest:
                    sample = int(item[1])
                    if sample not in existing_samples:
                        ownership_probe.append(item)
                        existing_samples.add(sample)
        else:
            ownership_probe = []

        local_count += len(local)
        adaptive_count += len(adaptive)
        ownership_count += len(ownership_probe)
        for group in (local, adaptive, ownership_probe):
            best_probe_evidence = max(
                best_probe_evidence,
                max((item[3] for item in group), default=0.0),
            )
        contexts[seed] = {
            "short_subset": short_subset,
            "long_subset": long_subset,
            "LOCAL": local,
            "ADAPTIVE": adaptive,
            "OWNERSHIP_WIDE": ownership_probe,
        }

    total_hypotheses = local_count + adaptive_count + ownership_count
    transitions = 0
    evaluated_pairs = set()
    long_tried_pairs = set()

    def run_short(mode, receivers, collected):
        nonlocal transitions
        for seed in receivers:
            seed = int(seed)
            context = contexts.get(seed)
            if context is None:
                continue
            for index, hypothesis in enumerate(context[mode]):
                pair = (seed, int(hypothesis[1]))
                if pair in evaluated_pairs:
                    continue
                evaluated_pairs.add(pair)
                short = _pilot(
                    flat,
                    x,
                    valid,
                    ownership,
                    seed,
                    hypothesis,
                    options,
                    context["short_subset"],
                    pilot_min_side_support_receivers,
                    pilot_min_coverage,
                    pilot_min_median_zncc,
                    phase_switch_zncc,
                    phase_switch_prediction_time,
                )
                if short is None:
                    continue
                transitions += short.transition_count
                if not short.passed or short.phase_switch_count:
                    continue
                mask, left, right, anchor_ok = _anchor(
                    short,
                    traces,
                    seed,
                    anchor_min,
                    anchor_min_rho,
                    anchor_max_prediction,
                    max_anchor_slope,
                )
                if not anchor_ok:
                    continue
                risk, left_signed, right_signed, curvature = _diffraction_risk(
                    short,
                    x,
                    seed,
                    mask,
                    max_macro_curvature_time=float(
                        diffraction_max_macro_curvature_time
                    ),
                    min_opposite_slope_ms_per_100m=float(
                        diffraction_min_opposite_slope_ms_per_100m
                    ),
                    min_points_per_side=int(diffraction_min_points_per_side),
                )
                if bool(diffraction_reject_enabled) and risk:
                    continue
                collected.append(
                    _ShortSeedEvaluation(
                        mode=mode,
                        hypothesis_index=index,
                        seed_receiver=seed,
                        hypothesis=hypothesis,
                        pilot=short,
                        anchor_mask=mask,
                        anchor_left=left,
                        anchor_right=right,
                        diffraction_risk=risk,
                        left_signed_slope_ms_per_100m=left_signed,
                        right_signed_slope_ms_per_100m=right_signed,
                        macro_curvature_time=curvature,
                    )
                )
        return collected

    def validate_distinct_events(mode_items):
        nonlocal transitions
        candidates = _apply_failed_seed_exclusions(
            mode_items,
            excluded_seed_times,
            tolerance_time=max(float(dt), 0.5 * float(evidence_half_width_time)),
            base_seed=base_seed,
            control_time=control_time,
            slope_quantum=slope_quantum,
            time_quantum=time_quantum,
        )
        if not candidates:
            return []
        clusters = _cluster_short_events(
            candidates,
            base_seed=base_seed,
            control_time=control_time,
            minimum_overlap=max(1, int(seed_event_cluster_min_overlap_receivers)),
            tolerance_time=float(seed_event_cluster_tolerance_time),
            slope_quantum=slope_quantum,
            time_quantum=time_quantum,
        )
        limit = int(max_long_pilot_events_per_stage)
        if bool(repair_exhaustive) or limit <= 0:
            limit = len(clusters)
        accepted = []
        attempted_clusters = 0
        for cluster in clusters:
            members = sorted(
                cluster,
                key=lambda item: _short_seed_rank(
                    item,
                    base_seed=base_seed,
                    control_time=control_time,
                    slope_quantum=slope_quantum,
                    time_quantum=time_quantum,
                ),
                reverse=True,
            )
            member = next(
                (
                    item for item in members
                    if (item.seed_receiver, int(item.hypothesis[1]))
                    not in long_tried_pairs
                ),
                None,
            )
            if member is None:
                continue
            if attempted_clusters >= limit:
                break
            attempted_clusters += 1
            pair = (member.seed_receiver, int(member.hypothesis[1]))
            long_tried_pairs.add(pair)
            context = contexts[member.seed_receiver]
            long = _pilot(
                flat,
                x,
                valid,
                ownership,
                member.seed_receiver,
                member.hypothesis,
                options,
                context["long_subset"],
                pilot_min_side_support_receivers,
                pilot_min_coverage,
                pilot_min_median_zncc,
                phase_switch_zncc,
                phase_switch_prediction_time,
            )
            if long is None:
                continue
            transitions += long.transition_count
            rank = (
                -_quantize(
                    abs(member.hypothesis[0] - float(control_time)),
                    time_quantum,
                ),
                long.coverage,
                -long.phase_switch_count,
                long.median_zncc,
                -_quantize(
                    long.p90_abs_slope_ms_per_100m, slope_quantum
                ),
                -long.median_abs_prediction_error,
                member.hypothesis[3],
                member.hypothesis[4],
                int(member.anchor_left + member.anchor_right + 1),
                -abs(member.seed_receiver - base_seed),
                member.hypothesis[2],
            )
            accepted.append(_LongSeedEvaluation(member, long, rank))
        return accepted

    def finish(accepted):
        selected = max(accepted, key=lambda item: item.rank)
        short_item = selected.short
        pilot = selected.pilot
        production_seed, production_sample, production_time = (
            _production_seed_from_short(short_item)
        )
        if production_seed < 0 or production_sample < 0:
            return None
        anchor_sample = np.where(
            short_item.anchor_mask, short_item.pilot.pick_sample, -1
        )
        anchor_time = np.where(
            short_item.anchor_mask, short_item.pilot.pick_time, np.nan
        )
        anchor_indices = np.flatnonzero(short_item.anchor_mask)
        amplitude = float(traces[production_seed, production_sample])
        evidence_ratio = (
            short_item.hypothesis[3] / best_probe_evidence
            if best_probe_evidence > np.finfo(float).eps
            else 1.0
        )
        high_confidence = bool(
            short_item.mode == "LOCAL"
            and pilot.passed
            and pilot.phase_switch_count == 0
            and pilot.median_abs_slope_ms_per_100m
            <= float(local_high_conf_max_median_abs_slope_ms_per_100m)
            and short_item.hypothesis[4]
            >= float(local_high_conf_min_positive_support_fraction)
            and evidence_ratio >= float(local_high_conf_min_evidence_ratio)
        )
        return SeedSearchResult(
            seed_time=production_time,
            seed_sample=production_sample,
            valid=True,
            mode=short_item.mode,
            positive_amplitude=amplitude,
            stack_evidence=short_item.hypothesis[3],
            positive_support_fraction=short_item.hypothesis[4],
            hypothesis_count=total_hypotheses,
            selected_hypothesis=short_item.hypothesis_index,
            pilot_coverage=pilot.coverage,
            pilot_median_zncc=pilot.median_zncc,
            pilot_median_abs_prediction_error=pilot.median_abs_prediction_error,
            pilot_median_abs_slope_ms_per_100m=pilot.median_abs_slope_ms_per_100m,
            phase_switch_count=pilot.phase_switch_count,
            pilot_left_support=pilot.left_support,
            pilot_right_support=pilot.right_support,
            anchor_left_count=int(np.count_nonzero(anchor_indices < production_seed)),
            anchor_right_count=int(np.count_nonzero(anchor_indices > production_seed)),
            anchor_receiver_mask=short_item.anchor_mask,
            anchor_pick_sample=anchor_sample,
            anchor_pick_time=anchor_time,
            pilot_pick_time=pilot.pick_time,
            local_hypothesis_count=local_count,
            adaptive_hypothesis_count=adaptive_count + ownership_count,
            pilot_transition_count=transitions,
            runtime_seconds=perf_counter() - started,
            high_confidence=high_confidence,
            seed_receiver=production_seed,
            diffraction_risk=bool(short_item.diffraction_risk),
            macro_curvature_time=float(short_item.macro_curvature_time),
            left_signed_slope_ms_per_100m=float(
                short_item.left_signed_slope_ms_per_100m
            ),
            right_signed_slope_ms_per_100m=float(
                short_item.right_signed_slope_ms_per_100m
            ),
        )

    # --------------------------------------------------------------
    # Hierarchical DP work.  A successful tier stops immediately.
    # --------------------------------------------------------------
    mode_enabled = {
        "LOCAL": True,
        "ADAPTIVE": bool(adaptive_enabled),
        "OWNERSHIP_WIDE": bool(ownership_wide_enabled),
    }
    for mode in ("LOCAL", "ADAPTIVE", "OWNERSHIP_WIDE"):
        if not mode_enabled[mode]:
            continue
        collected = []
        run_short(mode, primary_receivers, collected)
        accepted = validate_distinct_events(collected)
        if accepted:
            result = finish(accepted)
            if result is not None:
                return result

        if expansion_receivers.size:
            run_short(mode, expansion_receivers, collected)
            accepted = validate_distinct_events(collected)
            if accepted:
                result = finish(accepted)
                if result is not None:
                    return result

    return _empty_result(
        valid,
        started,
        local_count=local_count,
        adaptive_count=adaptive_count + ownership_count,
        transitions=transitions,
    )


__all__ = [
    "BoundarySeedCandidate",
    "SeedPilotResult",
    "SeedSearchResult",
    "build_interlayer_repair_state_valid",
    "discover_boundary_seed_candidates",
    "discover_tracking_seed",
]
