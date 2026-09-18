"""Two-segment trusted-evidence repair for TRACKING_BREAK results."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import numpy as np

from .flat_event import track_flattened_event_sparse
from .quality_control import QualityAuditConfig, audit_rkshot
from .seed_search import discover_boundary_seed_candidates


@dataclass(frozen=True)
class TrustedSegment:
    start_receiver: int
    end_receiver: int
    source_type: str
    source_seed_receiver: int
    source_seed_time: float
    trusted_mask: np.ndarray
    tracking: object
    quality_receiver_count: int
    success_count: int
    coverage: float
    median_zncc: float
    median_abs_prediction_error: float
    event_strength: float
    max_low_similarity_count: int
    max_severe_bad_count: int
    similarity_good_count: int = 0
    similarity_valid_count: int = 0
    similarity_support_fraction: float = 0.0
    max_prediction_bad_count: int = 0
    max_identity_outside_count: int = 0
    identity_valid: bool = True
    raw_start_receiver: int = -1
    raw_end_receiver: int = -1


@dataclass(frozen=True)
class TrustedQuality:
    quality_receivers: np.ndarray
    successful_receivers: np.ndarray
    coverage: float
    median_zncc: float
    median_abs_prediction_error: float
    similarity_good_count: int
    similarity_valid_count: int
    similarity_support_fraction: float
    low_similarity_count: int
    severe_bad_count: int
    prediction_bad_count: int
    identity_outside_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrackingBreakRepairResult:
    tracking: object | None
    audit: object | None
    original_trusted_segments: tuple[TrustedSegment, ...]
    selected_segments: tuple[TrustedSegment, ...]
    trusted_mask: np.ndarray
    trusted_union_coverage: float
    boundary_seed_receivers: tuple[int, ...]
    boundary_seed_times: tuple[float, ...]
    boundary_seed_energy: tuple[float, ...]
    boundary_candidate_diagnostics: tuple[str, ...]
    full_candidate_count: int
    candidate_trusted_segment_count: int
    boundary_seed_pilots_accepted: int

    @property
    def success(self):
        return self.tracking is not None


def _bad_window(values, window, minimum):
    if values.size < int(window):
        return False
    counts = np.convolve(values.astype(int), np.ones(int(window), int), mode="valid")
    return bool(np.any(counts >= int(minimum)))


def _max_window_count(values, window):
    if not values.size:
        return 0
    if values.size < int(window):
        return int(np.count_nonzero(values))
    return int(np.max(np.convolve(values.astype(int), np.ones(int(window), int), mode="valid")))


def _runs(mask):
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, splits) if part.size]


def _event_strength(flat, tracking, mask):
    rms = np.sqrt(np.mean(flat * flat, axis=1))
    good = mask & (tracking.pick_sample >= 0) & np.isfinite(rms) & (rms > np.finfo(float).eps)
    receivers = np.flatnonzero(good)
    values = flat[receivers, tracking.pick_sample[receivers]] / rms[receivers]
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(values)) if values.size else 0.0


def evaluate_trusted_quality(
    tracking, domain, start, end, config, reference_time=None,
    identity_half_width=None, require_similarity_support=True,
    neighbor_intrusion_mask=None,
):
    interval = np.arange(int(start), int(end) + 1)
    quality = interval[domain[interval]]
    successful = quality[tracking.success_mask[quality]]
    coverage = float(successful.size / quality.size) if quality.size else 0.0
    rho = tracking.neighbor_correlation[successful]
    error = tracking.prediction_error[successful]
    low = np.isfinite(rho) & (rho < float(config.low_similarity_threshold))
    finite_rho = np.isfinite(rho)
    similarity_good_count = int(np.count_nonzero(finite_rho & ~low))
    similarity_valid_count = int(np.count_nonzero(finite_rho))
    similarity_support = (
        float(similarity_good_count / similarity_valid_count)
        if similarity_valid_count else 0.0
    )
    severe = low & np.isfinite(error) & (np.abs(error) > float(config.severe_bad_prediction_error_time))
    prediction_bad = np.isfinite(error) & (np.abs(error) > float(config.severe_bad_prediction_error_time))
    identity_outside = np.zeros(successful.size, bool)
    if neighbor_intrusion_mask is not None:
        identity_outside = np.asarray(neighbor_intrusion_mask, bool)[successful]
    elif reference_time is not None and identity_half_width is not None:
        identity_outside = (
            np.isfinite(tracking.pick_time[successful])
            & (np.abs(tracking.pick_time[successful] - float(reference_time)) > float(identity_half_width))
        )
    window = int(config.quality_window_receivers)
    low_count = _max_window_count(low, window)
    severe_count = _max_window_count(severe, window)
    prediction_count = _max_window_count(prediction_bad, window)
    identity_count = _max_window_count(identity_outside, window)
    reasons = []
    if quality.size < window:
        reasons.append("TOO_SHORT_FOR_QUALITY_WINDOW")
    if coverage < float(config.normal_min_side_coverage):
        reasons.append("LOW_COVERAGE")
    if require_similarity_support and similarity_support < float(config.normal_min_side_coverage):
        reasons.append("LOW_SIMILARITY_SUPPORT")
    if low_count >= int(config.low_similarity_count_to_fail):
        reasons.append("PERSISTENT_LOW_SIMILARITY")
    if severe_count >= int(config.severe_bad_count_to_fail):
        reasons.append("PERSISTENT_POOR_CONTINUITY")
    # Large prediction errors by themselves are diagnostic only for trusted
    # repair segments.  A correct curved/diffracted event can produce several
    # large second-order prediction residuals while still keeping strong
    # waveform continuity.  Keep ``prediction_count`` for diagnostics, but do
    # not reject/cut a segment unless the existing joint continuity test
    # (low similarity + large prediction error) or the reflector-identity
    # corridor also fails.
    if identity_count >= int(config.severe_bad_count_to_fail):
        reasons.append(
            "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION"
            if neighbor_intrusion_mask is not None else "PERSISTENT_IDENTITY_DRIFT"
        )
    finite_rho = rho[finite_rho]
    finite_error = np.abs(error[np.isfinite(error)])
    return TrustedQuality(
        quality, successful, coverage,
        float(np.median(finite_rho)) if finite_rho.size else -1.0,
        float(np.median(finite_error)) if finite_error.size else np.inf,
        similarity_good_count, similarity_valid_count, similarity_support,
        low_count, severe_count, prediction_count, identity_count, tuple(reasons),
    )


def extract_trusted_segments(
    *, flat, tracking, tracking_usable_receiver, quality_available_receiver,
    config, source_type, source_seed_receiver=-1, source_seed_time=np.nan,
    allowed_receiver_mask=None, reference_time=None, identity_half_width=None,
    require_similarity_support=True, neighbor_intrusion_mask=None,
):
    """Extract maximal intervals satisfying the current NORMAL quality rules."""
    usable = np.asarray(tracking_usable_receiver, bool)
    domain = usable & np.asarray(quality_available_receiver, bool)
    if allowed_receiver_mask is not None:
        domain &= np.asarray(allowed_receiver_mask, bool)
    minimum = int(config.quality_window_receivers)
    candidates = []
    successful_domain = domain & np.asarray(tracking.success_mask, bool)
    for run in _runs(domain):
        valid_starts = np.flatnonzero(successful_domain[run])
        for offset in valid_starts:
            start, last, last_quality = int(run[offset]), None, None
            for end in run[offset + minimum - 1:]:
                if not successful_domain[int(end)]:
                    continue
                quality = evaluate_trusted_quality(
                    tracking, domain, start, int(end), config, reference_time,
                    identity_half_width, False,
                    neighbor_intrusion_mask=neighbor_intrusion_mask,
                )
                if quality.reasons:
                    if last is not None:
                        break
                    continue
                last, last_quality = int(end), quality
            if last is not None:
                candidates.append((start, last, last_quality))
    candidates.sort(key=lambda item: item[1] - item[0], reverse=True)
    chosen, occupied = [], np.zeros(domain.size, bool)
    for start, end, quality in candidates:
        interval = np.arange(start, end + 1)
        if np.any(occupied[interval]):
            continue
        occupied[interval] = True
        if (
            require_similarity_support
            and quality.similarity_support_fraction < float(config.normal_min_side_coverage)
        ):
            continue
        quality_receivers = quality.quality_receivers
        successful = quality.successful_receivers
        mask = np.zeros(domain.size, bool)
        mask[successful] = True
        chosen.append(TrustedSegment(
            start, end, source_type, int(source_seed_receiver), float(source_seed_time),
            mask, tracking, int(quality_receivers.size), int(successful.size), quality.coverage,
            quality.median_zncc, quality.median_abs_prediction_error,
            _event_strength(np.asarray(flat), tracking, mask),
            quality.low_similarity_count, quality.severe_bad_count,
            quality.similarity_good_count, quality.similarity_valid_count,
            quality.similarity_support_fraction,
            quality.prediction_bad_count, quality.identity_outside_count,
            not quality.reasons, start, end,
        ))
    return tuple(sorted(chosen, key=lambda item: item.start_receiver))


def extract_seed_connected_trusted_prefix(
    *, flat, tracking, tracking_usable_receiver, quality_available_receiver,
    config, source_type, seed_receiver, direction, source_seed_time,
    allowed_receiver_mask, reference_time=None, identity_half_width=None,
    neighbor_intrusion_mask=None, receiver_x=None, macro_window=None,
    macro_residual_limit_time=None,
):
    """Keep the longest trusted prefix connected to a boundary repair seed.

    The boundary seed has already passed the short pilot.  Therefore an early
    aggregate failure such as temporarily low coverage or temporarily low
    similarity support must not discard the whole full-track candidate before
    the prefix has had a chance to grow.  Those aggregate quantities can
    recover as more successful receivers are added.

    Persistent local failures are different: once an 8-receiver-style bad
    window for low similarity, poor continuity, or reflector-identity drift
    appears anywhere inside a seed-connected prefix, extending the prefix
    cannot make that bad window disappear.  The same applies to the one added
    large-scale smoothness condition: once a coarse 16-receiver-style window
    develops a large broken-line residual, the seed-connected trusted prefix is
    cut before that bend instead of allowing the repaired path to carry the
    wrong axis across the gather.

    Prediction-error instability alone remains diagnostic only; it is not a
    hard cut condition.
    """
    domain = (
        np.asarray(tracking_usable_receiver, bool)
        & np.asarray(quality_available_receiver, bool)
        & np.asarray(allowed_receiver_mask, bool)
    )
    order = np.arange(
        int(seed_receiver), domain.size if int(direction) > 0 else -1, int(direction)
    )
    order = order[domain[order]]
    successful = order[np.asarray(tracking.success_mask, bool)[order]]
    minimum = int(config.quality_window_receivers)
    if successful.size < minimum or not tracking.success_mask[int(seed_receiver)]:
        return None, None, None

    # ``LOW_COVERAGE`` and ``LOW_SIMILARITY_SUPPORT`` are aggregate prefix
    # statistics.  They may fail for the first few endpoints and become valid
    # later, so keep scanning.  The persistent window failures below are
    # monotonic for an expanding seed-connected prefix and therefore define a
    # real cut point.
    hard_reasons = {
        "PERSISTENT_LOW_SIMILARITY",
        "PERSISTENT_POOR_CONTINUITY",
        "PERSISTENT_IDENTITY_DRIFT",
        "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION",
        "MACRO_SMOOTHNESS_FAILED",
    }

    last_end = None
    last_quality = None
    tail_rejection = None
    tail_rejected_end = None

    for end in successful[minimum - 1:]:
        start, stop = sorted((int(seed_receiver), int(end)))
        quality = evaluate_trusted_quality(
            tracking, domain, start, stop, config, reference_time, identity_half_width,
            neighbor_intrusion_mask=neighbor_intrusion_mask,
        )
        if (
            receiver_x is not None
            and macro_window is not None
            and macro_residual_limit_time is not None
        ):
            macro_mask = np.zeros(domain.size, dtype=bool)
            macro_mask[quality.successful_receivers] = True
            macro_passed, macro_evaluable, _, _, _ = _macro_smoothness_mask(
                receiver_x, tracking, macro_mask, macro_window, macro_residual_limit_time
            )
            if macro_evaluable and not macro_passed:
                quality = replace(
                    quality,
                    reasons=tuple(quality.reasons) + ("MACRO_SMOOTHNESS_FAILED",),
                )
        reasons = set(quality.reasons)

        if reasons & hard_reasons:
            tail_rejection = quality
            tail_rejected_end = int(end)
            break

        if quality.reasons:
            # Coverage/support can recover after the first few receivers or
            # after a short isolated miss.  Do not throw away the whole track.
            if tail_rejection is None:
                tail_rejection = quality
                tail_rejected_end = int(end)
            continue

        # This is a fully trusted seed-connected prefix endpoint.  Keep the
        # furthest one reached so far.
        last_end = int(end)
        last_quality = quality
        tail_rejection = None
        tail_rejected_end = None

    if last_end is None or last_quality is None:
        return None, tail_rejection, tail_rejected_end

    start, stop = sorted((int(seed_receiver), int(last_end)))
    mask = np.zeros(domain.size, bool)
    mask[last_quality.successful_receivers] = True
    segment = TrustedSegment(
        start, stop, source_type, int(seed_receiver), float(source_seed_time),
        mask, tracking, int(last_quality.quality_receivers.size),
        int(last_quality.successful_receivers.size), last_quality.coverage,
        last_quality.median_zncc, last_quality.median_abs_prediction_error,
        _event_strength(np.asarray(flat), tracking, mask),
        last_quality.low_similarity_count, last_quality.severe_bad_count,
        last_quality.similarity_good_count, last_quality.similarity_valid_count,
        last_quality.similarity_support_fraction,
        last_quality.prediction_bad_count, last_quality.identity_outside_count,
        True, start, stop,
    )
    return segment, tail_rejection, tail_rejected_end


def _boundary_receivers(segments, domain):
    indices = np.flatnonzero(domain)
    if not indices.size:
        return ()
    if not segments:
        return tuple(dict.fromkeys((int(indices[0]), int(indices[-1]))))
    trusted = np.zeros(domain.size, bool)
    for segment in segments:
        trusted |= segment.trusted_mask
    covered = np.flatnonzero(trusted)
    result = []
    left, right = indices[indices < covered[0]], indices[indices > covered[-1]]
    if left.size:
        result.append(int(left[0]))
    if right.size:
        result.append(int(right[-1]))
    return tuple(result)


def _compatible(first, second):
    overlap = first.trusted_mask & second.trusted_mask
    if not np.any(overlap):
        return True
    difference = np.abs(first.tracking.pick_time[overlap] - second.tracking.pick_time[overlap])
    return bool(np.all(np.isfinite(difference) & (difference <= 0.010)))


def _combination_rank(segments, domain):
    union, strengths, correlations, errors = np.zeros(domain.size, bool), [], [], []
    for segment in segments:
        union |= segment.trusted_mask
        strengths.append(segment.event_strength)
        correlations.extend(segment.tracking.neighbor_correlation[segment.trusted_mask])
        errors.extend(np.abs(segment.tracking.prediction_error[segment.trusted_mask]))
    count, denominator = int(np.count_nonzero(union & domain)), int(np.count_nonzero(domain))
    coverage = float(count / denominator) if denominator else 0.0
    rho, error = np.asarray(correlations, float), np.asarray(errors, float)
    rho, error = rho[np.isfinite(rho)], error[np.isfinite(error)]
    return (
        coverage, count, float(np.median(strengths)),
        float(np.median(rho)) if rho.size else -1.0,
        -(float(np.median(error)) if error.size else np.inf),
    ), union


def _best_segments(pool, domain):
    choices = [(segment,) for segment in pool]
    choices.extend(pair for pair in combinations(pool, 2) if _compatible(*pair))
    if not choices:
        return (), np.zeros(domain.size, bool), 0.0
    selected = max(choices, key=lambda choice: _combination_rank(choice, domain)[0])
    rank, union = _combination_rank(selected, domain)
    return tuple(selected), union, float(rank[0])


def _macro_smoothness_mask(receiver_x, tracking, trusted_mask, macro_window, residual_limit_time):
    """Large-scale smoothness check for any trusted path mask.

    The check is intentionally coarse: in every consecutive ``macro_window``
    successful trusted picks, fit one quadratic travel-time curve t(x) and use
    the 90th-percentile absolute residual.  A few small local corners are
    tolerated, but a large broken-line jump / layer switch creates a window
    whose residual exceeds the existing prediction-error time scale.
    """
    x = np.asarray(receiver_x, dtype=float)
    picks = np.asarray(tracking.pick_time, dtype=float)
    trusted = (
        np.asarray(trusted_mask, dtype=bool)
        & np.asarray(tracking.success_mask, dtype=bool)
        & np.isfinite(picks)
        & np.isfinite(x)
    )
    receivers = np.flatnonzero(trusted)
    window = max(5, int(macro_window))
    if receivers.size < window:
        return True, False, np.nan, np.nan, 0

    window_errors_ms = []
    for first in range(0, receivers.size - window + 1):
        indices = receivers[first:first + window]
        xx = x[indices]
        tt = picks[indices]
        x0 = float(np.mean(xx))
        scale = float(np.max(np.abs(xx - x0)))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            continue
        u = (xx - x0) / scale
        try:
            coeff = np.polyfit(u, tt, deg=2)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
        fitted = np.polyval(coeff, u)
        residual_ms = np.abs(tt - fitted) * 1.0e3
        residual_ms = residual_ms[np.isfinite(residual_ms)]
        if residual_ms.size:
            window_errors_ms.append(float(np.percentile(residual_ms, 90.0)))

    if not window_errors_ms:
        return True, False, np.nan, np.nan, 0

    errors = np.asarray(window_errors_ms, dtype=float)
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))
    limit_ms = float(residual_limit_time) * 1.0e3
    return bool(max_error <= limit_ms), True, median_error, max_error, int(errors.size)


def _macro_smoothness(receiver_x, tracking, segment, macro_window, residual_limit_time):
    return _macro_smoothness_mask(
        receiver_x, tracking, segment.trusted_mask, macro_window, residual_limit_time
    )


def _composite_tracking(original, segments, mask):
    floats = ("pick_time", "neighbor_correlation", "residual_slope", "prediction_error",
              "selected_amplitude", "selected_envelope", "accumulated_coherence")
    ints = ("pick_sample", "candidate_sample", "selected_polarity", "selected_transition_kind")
    counts = ("candidate_count", "candidate_count_before_ownership")
    bools = ("rescue_used_mask", "ownership_escape_used_mask", "skipped_valid_receiver_mask")
    arrays = {name: np.full_like(getattr(original, name), np.nan, dtype=float) for name in floats}
    arrays.update({name: np.full_like(getattr(original, name), -1, dtype=int) for name in ints})
    arrays.update({name: np.zeros_like(getattr(original, name), dtype=int) for name in counts})
    arrays.update({name: np.zeros_like(getattr(original, name), dtype=bool) for name in bools})
    filled = np.zeros(mask.size, bool)
    order = sorted(segments, key=lambda item: (
        item.event_strength, item.coverage, item.median_zncc, -item.median_abs_prediction_error
    ), reverse=True)
    for segment in order:
        take = segment.trusted_mask & ~filled
        for name in arrays:
            arrays[name][take] = getattr(segment.tracking, name)[take]
        filled |= take
    return replace(
        original, **arrays, success_mask=mask.copy(),
        stop_reason_left="TRUSTED_SEGMENT_COMPOSITE",
        stop_reason_right="TRUSTED_SEGMENT_COMPOSITE",
        stop_receiver_left=-1, stop_receiver_right=-1,
    )


def repair_tracking_break(
    *, flat, receiver_x, valid_receiver, control_receiver, control_time,
    state_valid, escape_state_valid, tracking_usable_receiver,
    quality_available_receiver, original_tracking, tracker_options,
    seed_search_options, continuation_rescue, ownership_escape,
    audit_config, reflector, shot, dt, t0,
    upper_neighbor_state_valid=None, lower_neighbor_state_valid=None,
):
    config = audit_config or QualityAuditConfig()
    flat = np.asarray(flat, float)
    state_valid = np.asarray(state_valid, bool)
    escape_state_valid = np.asarray(escape_state_valid, bool)
    usable = np.asarray(tracking_usable_receiver, bool) & np.asarray(valid_receiver, bool)
    available = np.asarray(quality_available_receiver, bool) & usable
    prefix = f"[TB-REPAIR] R{int(reflector)} shot {int(shot)}"
    quality_count = int(np.count_nonzero(available))
    original_success = int(np.count_nonzero(original_tracking.success_mask & available))
    print(f"{prefix} ENTER", flush=True)
    print(f"  quality receivers = {quality_count}", flush=True)
    print(f"  original successful receivers = {original_success}", flush=True)
    print(f"  original coverage = {original_success / quality_count if quality_count else 0.0:.3f}", flush=True)
    original_quality_segments = extract_trusted_segments(
        flat=flat, tracking=original_tracking, tracking_usable_receiver=usable,
        quality_available_receiver=available, config=config, source_type="ORIGINAL",
    )

    # One added quality condition is used consistently for both ORIGINAL
    # trusted segments and boundary-repair trusted prefixes: large-scale
    # smoothness from coarse quadratic-fit residuals. Existing local quality
    # rules stay unchanged.
    qwindow = int(config.quality_window_receivers)
    macro_window = max(5, 2 * qwindow)
    # One extra trusted-path condition: on a 2*quality-window scale,
    # the tracked event must remain close to a smooth quadratic trend.  Reuse
    # the existing severe prediction-error time scale rather than adding a new
    # independent tuning knob.
    macro_residual_limit_time = 2.0 * float(config.severe_bad_prediction_error_time)
    original_segments = []
    original_macro = {}
    for segment in original_quality_segments:
        passed, evaluable, median_error_ms, max_error_ms, count = _macro_smoothness(
            receiver_x, original_tracking, segment, macro_window, macro_residual_limit_time
        )
        original_macro[(segment.start_receiver, segment.end_receiver)] = (
            evaluable, median_error_ms, max_error_ms, count
        )
        if passed:
            original_segments.append(segment)
            continue
        print(f"{prefix} TRUSTED-SEGMENT source=ORIGINAL", flush=True)
        print(f"  receivers = {segment.start_receiver}:{segment.end_receiver}", flush=True)
        print(f"  macro_window_receivers = {macro_window}", flush=True)
        print(f"  macro_windows = {count}", flush=True)
        print(f"  macro_fit_median_p90_residual_ms = {median_error_ms:.3f}", flush=True)
        print(f"  macro_fit_max_p90_residual_ms = {max_error_ms:.3f}", flush=True)
        print(f"  macro_fit_limit_ms = {macro_residual_limit_time * 1e3:.3f}", flush=True)
        print("  saved = False", flush=True)
        print("  reason = MACRO_SMOOTHNESS_FAILED", flush=True)
    original_segments = tuple(original_segments)

    original_before_similarity_support = tuple(
        extract_trusted_segments(
            flat=flat, tracking=original_tracking, tracking_usable_receiver=usable,
            quality_available_receiver=available, config=config, source_type="ORIGINAL",
            require_similarity_support=False,
        )
    )
    saved_ranges = {(item.start_receiver, item.end_receiver) for item in original_segments}
    for segment in original_before_similarity_support:
        if (
            segment.similarity_support_fraction >= float(config.normal_min_side_coverage)
            or (segment.start_receiver, segment.end_receiver) in saved_ranges
        ):
            continue
        print(f"{prefix} TRUSTED-SEGMENT source=ORIGINAL", flush=True)
        print(f"  receivers = {segment.start_receiver}:{segment.end_receiver}", flush=True)
        print(f"  similarity_good = {segment.similarity_good_count}/{segment.similarity_valid_count}", flush=True)
        print(f"  similarity_support = {segment.similarity_support_fraction:.3f}", flush=True)
        print(f"  required = {config.normal_min_side_coverage:.3f}", flush=True)
        print("  saved = False", flush=True)
        print("  reason = LOW_SIMILARITY_SUPPORT", flush=True)
    if not original_segments:
        print(f"{prefix} ORIGINAL-TRUSTED none", flush=True)
        print("  action = discard original tracking as trusted evidence", flush=True)
    for index, segment in enumerate(original_segments):
        print(f"{prefix} TRUSTED-SEGMENT source=ORIGINAL id={index}", flush=True)
        print(f"  receivers = {segment.start_receiver}:{segment.end_receiver}", flush=True)
        print(f"  span = {segment.end_receiver - segment.start_receiver + 1}", flush=True)
        print(f"  quality receivers = {segment.quality_receiver_count}", flush=True)
        print(f"  successful quality receivers = {segment.success_count}", flush=True)
        print(f"  coverage = {segment.coverage:.3f}", flush=True)
        print(f"  median_zncc = {segment.median_zncc:.3f}", flush=True)
        print(f"  similarity_good = {segment.similarity_good_count}/{segment.similarity_valid_count}", flush=True)
        print(f"  similarity_support = {segment.similarity_support_fraction:.3f}", flush=True)
        print(f"  required = {config.normal_min_side_coverage:.3f}", flush=True)
        print("  persistent_low_similarity = False", flush=True)
        print("  persistent_poor_continuity = False", flush=True)
        evaluable, median_error_ms, max_error_ms, count = original_macro.get(
            (segment.start_receiver, segment.end_receiver),
            (False, np.nan, np.nan, 0),
        )
        print(f"  macro_window_receivers = {macro_window}", flush=True)
        print(f"  macro_smoothness_evaluable = {evaluable}", flush=True)
        print(f"  macro_windows = {count}", flush=True)
        print(f"  macro_fit_median_p90_residual_ms = {median_error_ms:.3f}", flush=True)
        print(f"  macro_fit_max_p90_residual_ms = {max_error_ms:.3f}", flush=True)
        print(f"  macro_fit_limit_ms = {macro_residual_limit_time * 1e3:.3f}", flush=True)
        print("  saved = True", flush=True)

    preserved, preserved_mask, coverage = _best_segments(original_segments, available)
    boundary_receivers = () if coverage >= 0.80 else _boundary_receivers(preserved, available)
    tracker_cfg = dict(tracker_options or {})
    for key in ("continuation_rescue", "ownership_escape", "dt", "t0", "seed_receiver",
                "seed_time", "state_valid", "escape_state_valid", "anchor_receiver_mask",
                "anchor_pick_sample", "anchor_pick_time"):
        tracker_cfg.pop(key, None)
    seed_cfg = dict(seed_search_options or {})
    for key in ("tracker_options", "seed_receiver", "state_valid", "dt", "t0"):
        seed_cfg.pop(key, None)
    upper_neighbor = None if upper_neighbor_state_valid is None else np.asarray(upper_neighbor_state_valid, bool)
    lower_neighbor = None if lower_neighbor_state_valid is None else np.asarray(lower_neighbor_state_valid, bool)
    repair_pool, seed_times, seed_energy, diagnostics = [], [], [], []
    full_count = accepted_pilots = 0
    used_boundaries = []

    if coverage < 0.80 and np.any(available):
        aperture_left, aperture_right = np.flatnonzero(available)[[0, -1]]
        for requested in boundary_receivers:
            direction = 1 if requested == aperture_left else -1
            side = "LEFT" if direction > 0 else "RIGHT"
            if preserved:
                nearest = min(preserved, key=lambda item: item.start_receiver) if direction > 0 else max(
                    preserved, key=lambda item: item.end_receiver
                )
                edge = np.flatnonzero(nearest.trusted_mask)
                edge = edge[:qwindow] if direction > 0 else edge[-qwindow:]
                reference_time = float(np.median(original_tracking.pick_time[edge]))
                prior_source = "ORIGINAL_TRUSTED_EDGE"
                trusted_text = f"{nearest.start_receiver}:{nearest.end_receiver}"
                reference_text = f"{edge[0]}:{edge[-1]}"
            else:
                reference_time = float(control_time)
                prior_source, trusted_text, reference_text = "THEORETICAL_CONTROL", "NONE", "NONE"
            print(f"{prefix} TARGET-PRIOR side={side}", flush=True)
            print(f"  source = {prior_source}", flush=True)
            print(f"  trusted_segment = {trusted_text}", flush=True)
            print(f"  reference_receivers = {reference_text}", flush=True)
            print(f"  reference_time = {reference_time:.6f}", flush=True)

            candidates = ()
            receiver = int(requested)
            available_order = np.flatnonzero(available)
            inward = available_order[available_order >= receiver] if direction > 0 else available_order[available_order <= receiver][::-1]
            for trial_receiver in inward[:qwindow + 1]:
                trial = discover_boundary_seed_candidates(
                    flat, receiver_x=receiver_x, valid_receiver=valid_receiver,
                    seed_receiver=int(trial_receiver), inward_direction=direction, state_valid=state_valid,
                    dt=dt, t0=t0, tracker_options=tracker_cfg, reference_time=reference_time,
                    upper_neighbor_state_valid=upper_neighbor,
                    lower_neighbor_state_valid=lower_neighbor, **seed_cfg,
                )
                current_peaks = trial[0].current_ownership_peak_count if trial else 0
                exclusive_peaks = trial[0].exclusive_ownership_peak_count if trial else 0
                print(f"{prefix} SEED-SEARCH side={side}", flush=True)
                print(f"  receiver = {int(trial_receiver)}", flush=True)
                print(f"  current_Rk_ownership_peaks = {current_peaks}", flush=True)
                print(f"  exclusive_ownership_peaks = {exclusive_peaks}", flush=True)
                print(f"  overlap_fallback = {bool(trial and trial[0].overlap_fallback)}", flush=True)
                for candidate_index, candidate in enumerate(trial, 1):
                    delta = abs(candidate.seed_time - reference_time) * 1e3
                    if candidate.seed is None:
                        print(f"{prefix} SEED-PILOT side={side} candidate={candidate_index}", flush=True)
                        print(f"  seed_time={candidate.seed_time:.6f} rejected={candidate.rejection_reason} "
                              f"delta_from_reference_ms={delta:.1f}", flush=True)
                if any(item.selected and item.seed is not None for item in trial):
                    receiver, candidates = int(trial_receiver), trial
                    break
            if not candidates:
                print(f"{prefix} BOUNDARY side={side} failed", flush=True)
                print("  reason = NO_VALID_BOUNDARY_SEED", flush=True)
                continue
            used_boundaries.append(receiver)
            accepted_pilots += sum(candidate.seed is not None for candidate in candidates)
            for candidate_index, candidate in enumerate(candidates, 1):
                if candidate.seed is None:
                    continue
                print(f"{prefix} SEED-PILOT side={side} candidate={candidate_index}", flush=True)
                print(f"  seed_receiver = {receiver}", flush=True)
                print(f"  seed_time = {candidate.seed_time:.6f}", flush=True)
                print(f"  delta_from_reference_ms = {abs(candidate.seed_time-reference_time)*1e3:.1f}", flush=True)
                print(f"  seed_sample = {candidate.seed_sample}", flush=True)
                print(f"  seed_amplitude = {candidate.seed_amplitude:.3f}", flush=True)
                print(f"  inward_direction = {direction:+d}", flush=True)
                print(f"  pilot receivers = {candidate.pilot_receiver_start}:{candidate.pilot_receiver_end}", flush=True)
                print(f"  pilot_success = {candidate.pilot_success_count}/{candidate.pilot_receiver_count}", flush=True)
                print(f"  pilot_coverage = {candidate.pilot_coverage:.3f}", flush=True)
                print(f"  pilot_median_zncc = {candidate.pilot_median_zncc:.3f}", flush=True)
                print(f"  pilot_median_abs_prediction_error_ms = {candidate.pilot_median_abs_prediction_error*1e3:.1f}", flush=True)
                print(f"  anchor_length = {candidate.anchor_length}", flush=True)
                print(f"  short_segment_energy = {candidate.short_segment_energy:.3f}", flush=True)
                print(f"  evidence = {candidate.stack_evidence:.3f}", flush=True)
                print(f"  positive_support = {candidate.positive_support_fraction:.3f}", flush=True)
                print(f"  long_coverage = {candidate.long_coverage:.3f}", flush=True)
                print(f"  long_ZNCC = {candidate.long_median_zncc:.3f}", flush=True)
                print(f"  long_prediction_error_ms = {candidate.long_median_abs_prediction_error*1e3:.1f}", flush=True)
                print(f"  long_p90_slope = {candidate.long_p90_abs_slope_ms_per_100m:.3f}", flush=True)
                print(f"  phase_switch_count = {candidate.long_phase_switch_count}", flush=True)
                print("  accepted = True", flush=True)
            selected_candidates = sorted(
                (candidate for candidate in candidates if candidate.selected and candidate.seed is not None),
                key=lambda item: item.rank, reverse=True,
            )
            print(f"{prefix} TOP-SEEDS side={side}", flush=True)
            for rank_index, candidate in enumerate(selected_candidates, 1):
                print(f"  rank {rank_index}: seed_time={candidate.seed_time:.6f}", flush=True)
                print(f"    long_coverage = {candidate.long_coverage:.3f}", flush=True)
                print(f"    phase_switch = {candidate.long_phase_switch_count}", flush=True)
                print(f"    long_p90_slope = {candidate.long_p90_abs_slope_ms_per_100m:.3f}", flush=True)
                print(f"    long_ZNCC = {candidate.long_median_zncc:.3f}", flush=True)
                print(f"    long_prediction_error_ms = {candidate.long_median_abs_prediction_error*1e3:.1f}", flush=True)
                print(f"    evidence = {candidate.stack_evidence:.3f}", flush=True)
                print(f"    support = {candidate.positive_support_fraction:.3f}", flush=True)
                print(f"    anchor_length = {candidate.anchor_length}", flush=True)
                print(f"    delta_from_reference_ms = {abs(candidate.seed_time-reference_time)*1e3:.1f}", flush=True)
                print(f"    amplitude = {candidate.seed_amplitude:.3f}", flush=True)
            for seed_rank, candidate in enumerate(selected_candidates, 1):
                seed = candidate.seed
                seed_times.append(float(seed.seed_time))
                seed_energy.append(float(candidate.short_segment_energy))
                tracking = track_flattened_event_sparse(
                    flat, receiver_x=receiver_x, valid_receiver=valid_receiver,
                    seed_receiver=receiver, seed_time=seed.seed_time, state_valid=state_valid,
                    escape_state_valid=escape_state_valid, anchor_receiver_mask=seed.anchor_receiver_mask,
                    anchor_pick_sample=seed.anchor_pick_sample, anchor_pick_time=seed.anchor_pick_time,
                    dt=dt, t0=t0, continuation_rescue=continuation_rescue,
                    ownership_escape=ownership_escape, **tracker_cfg,
                )
                full_count += 1
                raw_audit = audit_rkshot(
                    success_mask=tracking.success_mask, valid_receiver=valid_receiver,
                    neighbor_similarity=tracking.neighbor_correlation,
                    prediction_error=tracking.prediction_error, seed_receiver=receiver,
                    anchor_receiver_mask=seed.anchor_receiver_mask,
                    tracking_usable_receiver=usable, quality_available_receiver=available,
                    config=config, reflector=reflector,
                )
                tracked = np.flatnonzero(tracking.success_mask & available)
                neighbor_intrusion = np.zeros(available.size, bool)
                picked = np.flatnonzero(tracking.success_mask & (tracking.pick_sample >= 0))
                if upper_neighbor is not None:
                    neighbor_intrusion[picked] |= upper_neighbor[picked, tracking.pick_sample[picked]]
                if lower_neighbor is not None:
                    neighbor_intrusion[picked] |= lower_neighbor[picked, tracking.pick_sample[picked]]
                persistent_intrusion = _bad_window(
                    neighbor_intrusion[tracked], qwindow, config.severe_bad_count_to_fail
                )
                print(f"{prefix} FULL-TRACK side={side} seed_rank={seed_rank}", flush=True)
                print(f"  reference_time = {reference_time:.6f}", flush=True)
                print(f"  seed_time = {seed.seed_time:.6f}", flush=True)
                print(f"  successful receivers = {tracked.size}", flush=True)
                print(f"  neighbor_ownership_intrusions = {np.count_nonzero(neighbor_intrusion[tracked])}", flush=True)
                print(f"  persistent_neighbor_ownership_intrusion = {persistent_intrusion}", flush=True)
                print(f"  raw_audit = {raw_audit.status.value}", flush=True)
                allowed = np.zeros(available.size, bool)
                allowed[receiver:] = direction > 0
                if direction < 0:
                    allowed[:receiver + 1] = True
                source_type = f"{side}_SEED_{seed_rank}"
                raw, prefix_rejection, rejected_end = extract_seed_connected_trusted_prefix(
                    flat=flat, tracking=tracking, tracking_usable_receiver=usable,
                    quality_available_receiver=available, config=config, source_type=source_type,
                    seed_receiver=receiver, direction=direction, source_seed_time=seed.seed_time,
                    allowed_receiver_mask=allowed,
                    neighbor_intrusion_mask=neighbor_intrusion,
                    receiver_x=receiver_x, macro_window=macro_window,
                    macro_residual_limit_time=macro_residual_limit_time,
                )
                raw_segments = (() if raw is None else (raw,))
                if raw is None:
                    print(f"{prefix} TRUSTED-FROM-SEED side={side} seed_rank={seed_rank}", flush=True)
                    print(f"  raw successful range = {(f'{tracked[0]}:{tracked[-1]}' if tracked.size else 'NONE')}", flush=True)
                    print("  trusted range = NONE", flush=True)
                    print("  saved = False", flush=True)
                    reasons = prefix_rejection.reasons if prefix_rejection is not None else ("TOO_SHORT_FOR_QUALITY_WINDOW",)
                    print(f"  reason = {'+'.join(reasons)}_AT_SEED", flush=True)
                for raw in raw_segments:
                    print(f"{prefix} TRUSTED-FROM-SEED side={side} seed_rank={seed_rank}", flush=True)
                    print(f"  raw successful range = {(f'{tracked[0]}:{tracked[-1]}' if tracked.size else 'NONE')}", flush=True)
                    print(f"  trusted range = {raw.start_receiver}:{raw.end_receiver}", flush=True)
                    print(f"  trusted span = {raw.end_receiver-raw.start_receiver+1}", flush=True)
                    print(f"  trusted quality receivers = {raw.quality_receiver_count}", flush=True)
                    print(f"  trusted successful receivers = {raw.success_count}", flush=True)
                    print(f"  trusted coverage = {raw.coverage:.3f}", flush=True)
                    print(f"  similarity_good = {raw.similarity_good_count}/{raw.similarity_valid_count}", flush=True)
                    print(f"  similarity_support = {raw.similarity_support_fraction:.3f}", flush=True)
                    print(f"  required = {config.normal_min_side_coverage:.3f}", flush=True)
                    print(f"  prediction_bad = {raw.max_prediction_bad_count}/{qwindow}", flush=True)
                    print(f"  neighbor_intrusion = {raw.max_identity_outside_count}/{qwindow}", flush=True)
                    macro_passed, macro_evaluable, macro_median_ms, macro_max_ms, macro_count = _macro_smoothness(
                        receiver_x, tracking, raw, macro_window, macro_residual_limit_time
                    )
                    print(f"  macro_smoothness_evaluable = {macro_evaluable}", flush=True)
                    print(f"  macro_windows = {macro_count}", flush=True)
                    print(f"  macro_fit_max_p90_residual_ms = {macro_max_ms:.3f}", flush=True)
                    print(f"  macro_fit_limit_ms = {macro_residual_limit_time * 1e3:.3f}", flush=True)
                    print(f"  macro_smoothness_passed = {macro_passed}", flush=True)
                    print("  accepted = True", flush=True)
                    print(f"{prefix} TRIM side={side} seed_rank={seed_rank}", flush=True)
                    if prefix_rejection is not None:
                        start, end = sorted((receiver, rejected_end))
                        print(f"  first_bad_window = {start}:{end}", flush=True)
                        print(f"  cut_after_receiver = {raw.end_receiver if direction > 0 else raw.start_receiver}", flush=True)
                        print(f"  window_coverage = {prefix_rejection.coverage:.3f}", flush=True)
                        print(f"  low_similarity = {prefix_rejection.low_similarity_count}/{qwindow}", flush=True)
                        print(f"  severe_bad = {prefix_rejection.severe_bad_count}/{qwindow}", flush=True)
                        print(f"  prediction_bad = {prefix_rejection.prediction_bad_count}/{qwindow}", flush=True)
                        print(f"  neighbor_intrusion = {prefix_rejection.identity_outside_count}/{qwindow}", flush=True)
                        if "MACRO_SMOOTHNESS_FAILED" in prefix_rejection.reasons:
                            rejected_mask = np.zeros(available.size, dtype=bool)
                            rejected_mask[prefix_rejection.successful_receivers] = True
                            _, macro_evaluable, macro_median_ms, macro_max_ms, macro_count = _macro_smoothness_mask(
                                receiver_x, tracking, rejected_mask, macro_window, macro_residual_limit_time
                            )
                            print(f"  macro_smoothness_evaluable = {macro_evaluable}", flush=True)
                            print(f"  macro_windows = {macro_count}", flush=True)
                            print(f"  macro_fit_max_p90_residual_ms = {macro_max_ms:.3f}", flush=True)
                            print(f"  macro_fit_limit_ms = {macro_residual_limit_time * 1e3:.3f}", flush=True)
                        print(f"  reason = {'+'.join(prefix_rejection.reasons)}", flush=True)
                    else:
                        print("  none", flush=True)
                        print("  reason = REACHED_QUALITY_EDGE", flush=True)
                blocked = np.zeros(available.size, bool)
                for original in preserved:
                    blocked[original.start_receiver:original.end_receiver + 1] = True
                for raw in raw_segments:
                    unique_allowed = allowed & ~blocked
                    unique_allowed[:raw.start_receiver] = False
                    unique_allowed[raw.end_receiver + 1:] = False
                    unique = extract_trusted_segments(
                        flat=flat, tracking=tracking, tracking_usable_receiver=usable,
                        quality_available_receiver=available, config=config, source_type=source_type,
                        source_seed_receiver=receiver, source_seed_time=seed.seed_time,
                        allowed_receiver_mask=unique_allowed,
                        neighbor_intrusion_mask=neighbor_intrusion,
                    )
                    for item in unique:
                        gain = int(np.count_nonzero(item.trusted_mask & available & ~preserved_mask))
                        if gain < qwindow:
                            print(f"{prefix} TRUSTED-FROM-SEED side={side} seed_rank={seed_rank} saved=False", flush=True)
                            print(f"  reason = INSUFFICIENT_NEW_COVERAGE ({gain} < {qwindow})", flush=True)
                            continue
                        repair_pool.append(replace(
                            item,
                            raw_start_receiver=raw.start_receiver, raw_end_receiver=raw.end_receiver,
                        ))
                diagnostics.append(
                    f"full:r={receiver},t={seed.seed_time:.8g},tracked={tracked.size},"
                    f"neighbor_intrusion_valid={int(not persistent_intrusion)},reference={reference_time:.8g}"
                )

    if preserved:
        selected = tuple(preserved)
        if len(selected) < 2 and repair_pool:
            addition = max(repair_pool, key=lambda item: _combination_rank((item,), available & ~preserved_mask)[0])
            selected += (addition,)
    else:
        selected, _, _ = _best_segments(repair_pool, available)
    _, union = _combination_rank(selected, available) if selected else ((0,), np.zeros(available.size, bool))
    coverage = float(np.count_nonzero(union & available) / quality_count) if quality_count else 0.0
    pool = list(preserved) + repair_pool
    print(f"{prefix} SEGMENT-POOL", flush=True)
    for index, segment in enumerate(pool):
        print(f"  id={index} source={segment.source_type}", flush=True)
        if segment.source_type == "ORIGINAL":
            print(f"    receivers={segment.start_receiver}:{segment.end_receiver}", flush=True)
            print(f"    trusted_receivers={segment.success_count}", flush=True)
            print("    fixed=True", flush=True)
        else:
            print(f"    raw_receivers={segment.raw_start_receiver}:{segment.raw_end_receiver}", flush=True)
            print(f"    after_original_overlap_trim={segment.start_receiver}:{segment.end_receiver}", flush=True)
            print(f"    new_trusted_receivers={np.count_nonzero(segment.trusted_mask & available & ~preserved_mask)}", flush=True)
            print(f"    neighbor_ownership_valid={segment.identity_valid}", flush=True)
    selected_ids = [next(index for index, item in enumerate(pool) if item is segment) for segment in selected]
    print(f"{prefix} SELECTED", flush=True)
    for position in range(2):
        if position < len(selected):
            segment = selected[position]
            print(
                f"  segment {position + 1} = id={selected_ids[position]} "
                f"receivers={segment.start_receiver}:{segment.end_receiver} source={segment.source_type}",
                flush=True,
            )
        else:
            print(f"  segment {position + 1} = NONE", flush=True)
    first_mask = selected[0].trusted_mask & available if selected else np.zeros(available.size, bool)
    second_mask = selected[1].trusted_mask & available & ~first_mask if len(selected) > 1 else np.zeros(available.size, bool)
    print(f"  quality_domain_receivers = {np.count_nonzero(available)}", flush=True)
    print(f"  segment_1_unique = {np.count_nonzero(first_mask)}", flush=True)
    print(f"  segment_2_unique = {np.count_nonzero(second_mask)}", flush=True)
    print(f"  union_trusted_receivers = {np.count_nonzero(union & available)}", flush=True)
    print(f"  trusted_union_coverage = {coverage:.3f}", flush=True)
    print("  required_coverage = 0.800", flush=True)
    neighbor_ownership_valid = all(segment.identity_valid for segment in selected)
    if coverage < 0.80 or len(selected) > 2 or not neighbor_ownership_valid:
        print(f"{prefix} RESULT = FAILURE", flush=True)
        print(f"  selected_segments = {len(selected)}", flush=True)
        print(f"  best_trusted_union_coverage = {coverage:.3f}", flush=True)
        print("  required = 0.800", flush=True)
        reason = "NO_VALID_TRUSTED_SEGMENT" if not pool else (
            "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION"
            if not neighbor_ownership_valid else "INSUFFICIENT_TRUSTED_COVERAGE"
        )
        print(f"  reason = {reason}", flush=True)
        return TrackingBreakRepairResult(
            None, None, original_segments, selected, union, coverage, boundary_receivers,
            tuple(seed_times), tuple(seed_energy), tuple(diagnostics), full_count,
            len(repair_pool), accepted_pilots,
        )
    composite = _composite_tracking(original_tracking, selected, union)
    diagnostic_audit = audit_rkshot(
        success_mask=composite.success_mask, valid_receiver=valid_receiver,
        neighbor_similarity=composite.neighbor_correlation,
        prediction_error=composite.prediction_error, seed_receiver=control_receiver,
        tracking_usable_receiver=usable, quality_available_receiver=available,
        config=config, reflector=reflector,
    )
    print(f"{prefix} RESULT = SUCCESS", flush=True)
    print(f"  selected_segments = {len(selected)}", flush=True)
    print(f"  trusted_union_coverage = {coverage:.3f}", flush=True)
    print("  required = 0.800", flush=True)
    return TrackingBreakRepairResult(
        composite, diagnostic_audit, original_segments, selected, union, coverage,
        tuple(used_boundaries), tuple(seed_times), tuple(seed_energy), tuple(diagnostics),
        full_count, len(repair_pool), accepted_pilots,
    )


__all__ = ["TrackingBreakRepairResult", "TrustedSegment", "extract_trusted_segments", "repair_tracking_break"]
