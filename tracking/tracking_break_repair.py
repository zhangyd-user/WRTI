"""Two-segment trusted-evidence repair for TRACKING_BREAK results."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import numpy as np

from .flat_event import track_flattened_event_sparse
from .quality_control import QualityAuditConfig, audit_rkshot
from .seed_search import (
    build_interlayer_repair_state_valid,
    discover_boundary_seed_candidates,
)
from .repair_identity_core import (
    bank_from_trusted_segment,
    evaluate_anchor_backed_repair_identity,
    evaluate_dual_repair_identity_without_anchor,
    evaluate_single_repair_without_anchor,
    select_representative_waveform_bank,
)


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


@dataclass(frozen=True)
class BoundaryShortEvent:
    """One repair-side short event reconstructed from short-pilot paths.

    The event is the object being judged.  Individual positive peaks merely
    propose paths; only after a coherent event has been formed do we construct
    a production seed and anchor directly from the accepted event itself.
    """

    members: tuple
    origin_receivers: tuple[int, ...]
    tested_mask: np.ndarray
    success_mask: np.ndarray
    pick_sample: np.ndarray
    pick_time: np.ndarray
    neighbor_zncc: np.ndarray
    residual_slope: np.ndarray
    prediction_error: np.ndarray
    tested_count: int
    success_count: int
    coverage: float
    median_zncc: float
    median_abs_prediction_error: float
    median_abs_slope_ms_per_100m: float
    p90_abs_slope_ms_per_100m: float
    phase_switch_count: int
    median_member_deviation_time: float
    p90_member_deviation_time: float
    representative_start_receiver: int
    representative_end_receiver: int
    representative_center_receiver: int
    representative_rank: tuple
    passed: bool
    rank: tuple


@dataclass(frozen=True)
class EventProductionSeed:
    """A production seed/anchor constructed from an accepted short event.

    This deliberately breaks the old dependency on a single hypothesis having
    already passed its own anchor+long-pilot validation.  Once the 2-D short
    event is accepted, its clean representative window *is* the repair anchor.
    """

    seed_time: float
    seed_sample: int
    anchor_receiver_mask: np.ndarray
    anchor_pick_sample: np.ndarray
    anchor_pick_time: np.ndarray


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



def extract_seed_connected_trusted_segment(
    *, flat, tracking, tracking_usable_receiver, quality_available_receiver,
    config, source_type, seed_receiver, source_seed_time, allowed_receiver_mask,
    reference_time=None, identity_half_width=None, neighbor_intrusion_mask=None,
    receiver_x=None, macro_window=None, macro_residual_limit_time=None,
    anchor_receiver_mask=None,
):
    """Grow a trusted interval outward from the confirmed event core.

    Event-first repair may place the production seed well inside the aperture.
    The trusted interval therefore has to grow on both sides of the seed.  The
    previous implementation searched every seed-containing interval and ranked
    them by length first; a long but visibly degraded tail could therefore win
    merely because it added receivers.  Here the accepted short-event anchor is
    treated as the trusted core and each side is expanded independently.

    Local persistent failures stop only the side on which they appear.  Aggregate
    coverage/similarity support is judged on a wider local window before it is
    allowed to stop growth, so one isolated miss does not truncate an otherwise
    coherent reflector.  After both sides are grown, the combined interval must
    still satisfy the unchanged trusted-quality and macro-smoothness rules.
    """
    domain = (
        np.asarray(tracking_usable_receiver, bool)
        & np.asarray(quality_available_receiver, bool)
        & np.asarray(allowed_receiver_mask, bool)
    )
    seed = int(seed_receiver)
    success = np.asarray(tracking.success_mask, bool)
    if not 0 <= seed < domain.size or not domain[seed] or not success[seed]:
        return None, None

    containing_run = None
    for run in _runs(domain):
        if run.size and int(run[0]) <= seed <= int(run[-1]):
            containing_run = run
            break
    if containing_run is None:
        return None, None

    successful_run = containing_run[success[containing_run]]
    minimum = int(config.quality_window_receivers)
    if successful_run.size < minimum:
        return None, None

    # The event-first production anchor is the clean representative short-event
    # window.  Use it as the expansion core instead of letting a later exhaustive
    # interval search redefine the core by receiver count alone.
    if anchor_receiver_mask is not None:
        anchor = np.asarray(anchor_receiver_mask, bool)
        if anchor.shape != domain.shape:
            raise ValueError("anchor_receiver_mask must have shape [nreceiver]")
        core = containing_run[anchor[containing_run] & success[containing_run]]
    else:
        core = np.empty(0, dtype=int)
    if not core.size:
        core = np.asarray([seed], dtype=int)
    core_start = int(np.min(core))
    core_end = int(np.max(core))

    hard_reasons = {
        "PERSISTENT_LOW_SIMILARITY",
        "PERSISTENT_POOR_CONTINUITY",
        "PERSISTENT_IDENTITY_DRIFT",
        "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION",
        "MACRO_SMOOTHNESS_FAILED",
    }
    aggregate_reasons = {"LOW_COVERAGE", "LOW_SIMILARITY_SUPPORT"}
    local_width = max(
        minimum,
        int(macro_window) if macro_window is not None else 2 * minimum,
    )

    def checked_quality(start, end):
        quality = evaluate_trusted_quality(
            tracking, domain, int(start), int(end), config,
            reference_time, identity_half_width,
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
                receiver_x, tracking, macro_mask, macro_window,
                macro_residual_limit_time,
            )
            if macro_evaluable and not macro_passed:
                quality = replace(
                    quality,
                    reasons=tuple(quality.reasons) + ("MACRO_SMOOTHNESS_FAILED",),
                )
        return quality

    def edge_quality(start, end, side):
        receivers = successful_run[
            (successful_run >= int(start)) & (successful_run <= int(end))
        ]
        if receivers.size < minimum:
            return None, receivers.size
        if side == "LEFT":
            local = receivers[:local_width]
        else:
            local = receivers[-local_width:]
        return checked_quality(int(local[0]), int(local[-1])), int(local.size)

    def should_stop(quality, local_count):
        if quality is None:
            return False
        reasons = set(quality.reasons)
        if reasons & hard_reasons:
            return True
        # LOW_COVERAGE / LOW_SIMILARITY_SUPPORT are aggregate statistics.  Let a
        # short edge extension accumulate evidence first, then require the same
        # NORMAL-quality support on the existing 2*quality-window scale.
        if local_count >= local_width and reasons & aggregate_reasons:
            return True
        return False

    left_bound = core_start
    right_bound = core_end
    first_rejection = None

    left_candidates = successful_run[successful_run < core_start][::-1]
    for candidate in left_candidates:
        quality, local_count = edge_quality(int(candidate), right_bound, "LEFT")
        if should_stop(quality, local_count):
            first_rejection = quality
            break
        left_bound = int(candidate)

    right_candidates = successful_run[successful_run > core_end]
    for candidate in right_candidates:
        quality, local_count = edge_quality(left_bound, int(candidate), "RIGHT")
        if should_stop(quality, local_count):
            if first_rejection is None:
                first_rejection = quality
            break
        right_bound = int(candidate)

    # The independently grown interval still has to pass the unchanged global
    # trusted rules.  If aggregate statistics fail only after the two sides are
    # combined, fall back to the largest fully valid interval *inside the local
    # side cut points*.  This fallback can no longer re-admit a tail that already
    # failed the local expansion gate.
    quality = checked_quality(left_bound, right_bound)
    if quality.reasons or quality.successful_receivers.size < minimum:
        bounded = successful_run[
            (successful_run >= left_bound) & (successful_run <= right_bound)
        ]
        starts = bounded[bounded <= core_start]
        ends = bounded[bounded >= core_end]
        best = None
        best_failed = None
        for start in starts:
            for end in ends:
                if int(end) < int(start):
                    continue
                candidate_quality = checked_quality(int(start), int(end))
                if candidate_quality.reasons:
                    failure_rank = (
                        int(candidate_quality.successful_receivers.size),
                        float(candidate_quality.coverage),
                    )
                    if best_failed is None or failure_rank > best_failed[0]:
                        best_failed = (failure_rank, candidate_quality)
                    continue
                if candidate_quality.successful_receivers.size < minimum:
                    continue
                rank = (
                    int(candidate_quality.successful_receivers.size),
                    float(candidate_quality.similarity_support_fraction),
                    -int(candidate_quality.severe_bad_count),
                    -int(candidate_quality.low_similarity_count),
                    float(candidate_quality.median_zncc),
                    -float(candidate_quality.median_abs_prediction_error),
                )
                if best is None or rank > best[0]:
                    best = (rank, int(start), int(end), candidate_quality)
        if best is None:
            failure = first_rejection
            if failure is None and best_failed is not None:
                failure = best_failed[1]
            if failure is None:
                failure = quality
            return None, failure
        _, left_bound, right_bound, quality = best

    mask = np.zeros(domain.size, bool)
    mask[quality.successful_receivers] = True
    segment = TrustedSegment(
        int(left_bound), int(right_bound), source_type, seed, float(source_seed_time),
        mask, tracking, int(quality.quality_receivers.size),
        int(quality.successful_receivers.size), quality.coverage,
        quality.median_zncc, quality.median_abs_prediction_error,
        _event_strength(np.asarray(flat), tracking, mask),
        quality.low_similarity_count, quality.severe_bad_count,
        quality.similarity_good_count, quality.similarity_valid_count,
        quality.similarity_support_fraction,
        quality.prediction_bad_count, quality.identity_outside_count,
        True, int(left_bound), int(right_bound),
    )
    return segment, first_rejection


def _same_receiver_overlap_identity(
    flat, repair_segment, original_segments, *, dt, config, tracker_options,
):
    """Compare REPAIR and ORIGINAL on the same receivers in a clean overlap window."""
    width = int(config.quality_window_receivers)
    threshold = float(config.low_similarity_threshold)
    low_fail = int(config.low_similarity_count_to_fail)
    half_time = float(dict(tracker_options or {}).get("coherence_half_window_time", 0.040))
    half = max(1, int(round(half_time / float(dt))))

    best = None
    for original in original_segments:
        overlap = (
            np.asarray(repair_segment.trusted_mask, bool)
            & np.asarray(original.trusted_mask, bool)
            & np.asarray(repair_segment.tracking.success_mask, bool)
            & np.asarray(original.tracking.success_mask, bool)
        )
        for run in _runs(overlap):
            if run.size < width:
                continue
            for offset in range(run.size - width + 1):
                receivers = run[offset:offset + width]
                rho = []
                valid = True
                for receiver in receivers:
                    a = _normalised_event_waveform(
                        flat, int(receiver),
                        int(repair_segment.tracking.pick_sample[int(receiver)]), half,
                    )
                    b = _normalised_event_waveform(
                        flat, int(receiver),
                        int(original.tracking.pick_sample[int(receiver)]), half,
                    )
                    if a is None or b is None:
                        valid = False
                        break
                    rho.append(float(np.dot(a, b)))
                if not valid:
                    continue
                rho = np.asarray(rho, dtype=float)
                low_count = int(np.count_nonzero(rho < threshold))
                median = float(np.median(rho))
                p25 = float(np.percentile(rho, 25.0))
                rank = (-low_count, p25, median)
                item = {
                    "passed": bool(median >= threshold and low_count < low_fail),
                    "receivers": receivers.copy(),
                    "values": rho,
                    "median": median,
                    "p25": p25,
                    "low_count": low_count,
                    "rank": rank,
                }
                if best is None or rank > best["rank"]:
                    best = item
    if best is None:
        return {
            "passed": False, "receivers": np.empty(0, dtype=int),
            "values": np.empty(0, dtype=float), "median": -np.inf,
            "p25": -np.inf, "low_count": width,
            "rank": (-width, -np.inf, -np.inf),
            "reason": "NO_QUALITY_WINDOW_OVERLAP",
        }
    best["reason"] = "" if best["passed"] else "OVERLAP_WAVEFORM_ZNCC_FAILED"
    return best


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


def _final_choice_rank(segments, domain, *, required_coverage=0.80):
    """Use coverage as a success threshold; above it, prefer repair quality."""
    base_rank, union = _combination_rank(segments, domain)
    coverage = float(base_rank[0])
    count = int(base_rank[1])

    repairs = [
        segment for segment in segments
        if _segment_side(segment) != "ORIGINAL"
    ]
    judged = repairs if repairs else list(segments)
    if judged:
        similarity_support = min(
            float(segment.similarity_support_fraction) for segment in judged
        )
        severe_bad = max(int(segment.max_severe_bad_count) for segment in judged)
        low_similarity = max(int(segment.max_low_similarity_count) for segment in judged)
        median_zncc = min(float(segment.median_zncc) for segment in judged)
        median_prediction_error = max(
            float(segment.median_abs_prediction_error) for segment in judged
        )
        prediction_bad = max(int(segment.max_prediction_bad_count) for segment in judged)
    else:
        similarity_support = -np.inf
        severe_bad = np.iinfo(np.int32).max
        low_similarity = np.iinfo(np.int32).max
        median_zncc = -np.inf
        median_prediction_error = np.inf
        prediction_bad = np.iinfo(np.int32).max

    if coverage >= float(required_coverage):
        # Every choice here already has enough receivers.  Additional coverage is
        # only a late tie-break; the cleaner identified repair path wins first.
        rank = (
            1,
            similarity_support,
            -severe_bad,
            -low_similarity,
            median_zncc,
            -median_prediction_error,
            -prediction_bad,
            coverage,
            count,
        )
    else:
        # If no choice reaches the required coverage, retain the widest failed
        # combination for diagnostics rather than pretending quality can rescue
        # insufficient evidence.
        rank = (
            0,
            coverage,
            count,
            similarity_support,
            -severe_bad,
            -low_similarity,
            median_zncc,
            -median_prediction_error,
            -prediction_bad,
        )
    return rank, union


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

    # Identity has already been established before this merge.  If two repair
    # segments overlap, prefer the upper (earlier-time) pick because the known
    # diffraction failure mode drags a competing branch downward.  ORIGINAL,
    # when present, remains the fixed source of truth in its own trusted mask.
    for receiver in np.flatnonzero(mask):
        owners = [
            segment for segment in segments
            if bool(np.asarray(segment.trusted_mask, bool)[int(receiver)])
        ]
        if not owners:
            continue
        original_owners = [item for item in owners if str(item.source_type) == "ORIGINAL"]
        if original_owners:
            chosen = max(
                original_owners,
                key=lambda item: (
                    item.coverage, item.median_zncc, -item.median_abs_prediction_error
                ),
            )
        elif len(owners) == 1:
            chosen = owners[0]
        else:
            finite = [
                item for item in owners
                if np.isfinite(item.tracking.pick_time[int(receiver)])
            ]
            chosen = min(
                finite if finite else owners,
                key=lambda item: (
                    item.tracking.pick_time[int(receiver)]
                    if np.isfinite(item.tracking.pick_time[int(receiver)]) else np.inf,
                    -item.median_zncc,
                ),
            )
        for name in arrays:
            arrays[name][int(receiver)] = getattr(chosen.tracking, name)[int(receiver)]

    return replace(
        original, **arrays, success_mask=mask.copy(),
        stop_reason_left="TRUSTED_SEGMENT_COMPOSITE",
        stop_reason_right="TRUSTED_SEGMENT_COMPOSITE",
        stop_receiver_left=-1, stop_receiver_right=-1,
    )


def _boundary_short_pilot_eligible(candidate, minimum_success_receivers):
    """Keep usable partial short paths for event reconstruction.

    A path no longer has to pass the old per-seed short-pilot coverage gate.
    That gate was the remaining seed-first bottleneck: a real event fragmented
    by diffraction could be observed from several origins yet every individual
    25-receiver pilot could fail coverage before the event was assembled.
    Event quality is judged later on the consensus path.
    """
    short = getattr(candidate, "short_pilot", None)
    if short is None or int(short.phase_switch_count) != 0:
        return False
    success = (
        np.asarray(short.success_mask, dtype=bool)
        & np.asarray(short.tested_mask, dtype=bool)
    )
    return bool(np.count_nonzero(success) >= int(minimum_success_receivers))


def _boundary_short_candidate_rank(entry):
    """Rank only the short path; long-pilot evidence must not define the event."""
    _, candidate = entry
    short = candidate.short_pilot
    return (
        float(short.coverage),
        -int(short.phase_switch_count),
        -float(short.p90_abs_slope_ms_per_100m),
        float(short.median_zncc),
        -float(short.median_abs_slope_ms_per_100m),
        -float(short.median_abs_prediction_error),
        float(candidate.stack_evidence),
        float(candidate.positive_support_fraction),
        float(candidate.seed_amplitude),
    )


def _normalised_event_waveform(flat, receiver, sample, half_window_samples):
    trace = np.asarray(flat[int(receiver)], dtype=float)
    sample = int(sample)
    half = int(half_window_samples)
    if sample - half < 0 or sample + half >= trace.size:
        return None
    waveform = np.array(trace[sample - half:sample + half + 1], copy=True)
    waveform -= np.mean(waveform)
    norm = float(np.linalg.norm(waveform))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return waveform / norm


def _short_event_consensus(members, receiver_count):
    """Build one real-sample consensus path from member short-pilot paths."""
    tested = np.zeros(int(receiver_count), dtype=bool)
    observations = [[] for _ in range(int(receiver_count))]
    for _, candidate in members:
        short = candidate.short_pilot
        if short is None:
            continue
        tested |= np.asarray(short.tested_mask, dtype=bool)
        success = (
            np.asarray(short.success_mask, dtype=bool)
            & np.asarray(short.tested_mask, dtype=bool)
            & (np.asarray(short.pick_sample, dtype=int) >= 0)
            & np.isfinite(np.asarray(short.pick_time, dtype=float))
        )
        for receiver in np.flatnonzero(success):
            rho = float(short.neighbor_zncc[receiver])
            observations[int(receiver)].append(
                (
                    float(short.pick_time[receiver]),
                    int(short.pick_sample[receiver]),
                    rho if np.isfinite(rho) else -np.inf,
                )
            )

    pick_sample = np.full(int(receiver_count), -1, dtype=int)
    pick_time = np.full(int(receiver_count), np.nan, dtype=float)
    success = np.zeros(int(receiver_count), dtype=bool)
    deviations = []
    for receiver, values in enumerate(observations):
        if not values:
            continue
        times = np.asarray([item[0] for item in values], dtype=float)
        center = float(np.median(times))
        # Keep an actually observed positive-peak sample, not an interpolated
        # median time that may not coincide with any seismic peak.
        chosen = min(values, key=lambda item: (abs(item[0] - center), -item[2]))
        pick_time[receiver] = float(chosen[0])
        pick_sample[receiver] = int(chosen[1])
        success[receiver] = True
        deviations.extend(np.abs(times - center).tolist())
    return tested, success, pick_sample, pick_time, np.asarray(deviations, dtype=float)


def _short_event_path_metrics(
    flat, receiver_x, success_mask, pick_sample, pick_time,
    *, half_window_samples, phase_switch_zncc, phase_switch_prediction_time,
):
    """Evaluate waveform continuity and second-order geometry of a short event."""
    success = np.asarray(success_mask, dtype=bool)
    sample = np.asarray(pick_sample, dtype=int)
    time = np.asarray(pick_time, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    neighbor = np.full(success.size, np.nan, dtype=float)
    slope = np.full(success.size, np.nan, dtype=float)
    prediction = np.full(success.size, np.nan, dtype=float)

    for run in _runs(success):
        if run.size < 2:
            continue
        waveforms = {}
        for receiver in run:
            waveforms[int(receiver)] = _normalised_event_waveform(
                flat, int(receiver), int(sample[receiver]), half_window_samples
            )
        for local in range(1, run.size):
            previous = int(run[local - 1])
            current = int(run[local])
            previous_waveform = waveforms.get(previous)
            current_waveform = waveforms.get(current)
            if previous_waveform is not None and current_waveform is not None:
                neighbor[current] = float(np.dot(previous_waveform, current_waveform))
            dx = float(x[current] - x[previous])
            if dx == 0.0:
                continue
            slope[current] = float((time[current] - time[previous]) / dx)
            if local >= 2:
                older = int(run[local - 2])
                previous_dx = float(x[previous] - x[older])
                if previous_dx == 0.0:
                    continue
                previous_slope = float((time[previous] - time[older]) / previous_dx)
                prediction[current] = float(
                    time[current] - (time[previous] + previous_slope * dx)
                )

    finite_rho = neighbor[np.isfinite(neighbor)]
    finite_prediction = np.abs(prediction[np.isfinite(prediction)])
    finite_slope = np.abs(slope[np.isfinite(slope)]) * 1.0e5
    phase = (
        np.isfinite(neighbor)
        & np.isfinite(prediction)
        & (neighbor < float(phase_switch_zncc))
        & (np.abs(prediction) > float(phase_switch_prediction_time))
    )
    return (
        neighbor,
        slope,
        prediction,
        float(np.median(finite_rho)) if finite_rho.size else -np.inf,
        float(np.median(finite_prediction)) if finite_prediction.size else np.inf,
        float(np.median(finite_slope)) if finite_slope.size else np.inf,
        float(np.percentile(finite_slope, 90.0)) if finite_slope.size else np.inf,
        int(np.count_nonzero(phase)),
    )


def _best_short_event_window(success_mask, neighbor_zncc, prediction_error, window, low_threshold):
    """Find the cleanest contiguous part of the event, independent of seed origin."""
    success = np.asarray(success_mask, dtype=bool)
    neighbor = np.asarray(neighbor_zncc, dtype=float)
    prediction = np.asarray(prediction_error, dtype=float)
    width = max(1, int(window))
    best = None
    for run in _runs(success):
        if run.size < width:
            continue
        for first in range(0, run.size - width + 1):
            indices = run[first:first + width]
            rho = neighbor[indices[1:]] if indices.size > 1 else np.empty(0)
            finite_rho = rho[np.isfinite(rho)]
            nonfinite = int(rho.size - finite_rho.size)
            low_count = nonfinite + int(np.count_nonzero(finite_rho < float(low_threshold)))
            if finite_rho.size:
                lower_rho = float(np.percentile(finite_rho, 10.0))
                median_rho = float(np.median(finite_rho))
            else:
                lower_rho = median_rho = -np.inf
            error = np.abs(prediction[indices])
            error = error[np.isfinite(error)]
            median_error = float(np.median(error)) if error.size else np.inf
            rank = (-low_count, lower_rho, median_rho, -median_error)
            item = (
                rank,
                int(indices[0]),
                int(indices[-1]),
                int(indices[len(indices) // 2]),
            )
            if best is None or item[0] > best[0]:
                best = item
    if best is None:
        return -1, -1, -1, (-np.inf, -np.inf, -np.inf, -np.inf)
    rank, start, end, center = best
    return start, end, center, tuple(rank)


def _build_boundary_short_event(
    members, *, flat, receiver_x, quality_window_receivers,
    low_similarity_threshold, pilot_min_coverage, pilot_min_median_zncc,
    phase_switch_zncc, phase_switch_prediction_time, coherence_half_window_time, dt,
):
    tested, success, pick_sample, pick_time, deviations = _short_event_consensus(
        members, np.asarray(flat).shape[0]
    )
    half_window_samples = max(1, int(round(float(coherence_half_window_time) / float(dt))))
    (
        neighbor, slope, prediction, median_rho, median_prediction,
        median_slope, p90_slope, phase_count,
    ) = _short_event_path_metrics(
        flat, receiver_x, success, pick_sample, pick_time,
        half_window_samples=half_window_samples,
        phase_switch_zncc=phase_switch_zncc,
        phase_switch_prediction_time=phase_switch_prediction_time,
    )
    tested_count = int(np.count_nonzero(tested))
    success_count = int(np.count_nonzero(success & tested))
    coverage = float(success_count / tested_count) if tested_count else 0.0
    median_deviation = float(np.median(deviations)) if deviations.size else 0.0
    p90_deviation = float(np.percentile(deviations, 90.0)) if deviations.size else 0.0
    rep_start, rep_end, rep_center, rep_rank = _best_short_event_window(
        success & tested, neighbor, prediction,
        quality_window_receivers, low_similarity_threshold,
    )
    origins = tuple(sorted({int(receiver) for receiver, _ in members}))
    passed = bool(
        rep_start >= 0
        and success_count >= int(quality_window_receivers)
        and coverage >= float(pilot_min_coverage)
        and median_rho >= float(pilot_min_median_zncc)
        and int(phase_count) == 0
    )
    # Event ranking uses only event/path evidence.  Number of origins is a late
    # tie-break, never a hard identity condition and never the leading score.
    rank = (
        coverage,
        -int(phase_count),
        -float(p90_slope),
        float(median_rho),
        -float(median_slope),
        -float(median_prediction),
        tuple(rep_rank),
        -float(p90_deviation),
        len(origins),
        len(members),
    )
    return BoundaryShortEvent(
        tuple(members), origins, tested, success, pick_sample, pick_time,
        neighbor, slope, prediction, tested_count, success_count, coverage,
        median_rho, median_prediction, median_slope, p90_slope, phase_count,
        median_deviation, p90_deviation, rep_start, rep_end, rep_center,
        tuple(rep_rank), passed, tuple(rank),
    )


def _short_candidate_matches_event(entry, event, *, median_tolerance_time, p90_tolerance_time, minimum_overlap):
    candidate = entry[1]
    short = candidate.short_pilot
    if short is None:
        return None
    overlap = (
        np.asarray(short.success_mask, dtype=bool)
        & np.asarray(event.success_mask, dtype=bool)
        & np.isfinite(np.asarray(short.pick_time, dtype=float))
        & np.isfinite(np.asarray(event.pick_time, dtype=float))
    )
    receivers = np.flatnonzero(overlap)
    if receivers.size < int(minimum_overlap):
        return None
    difference = np.abs(
        np.asarray(short.pick_time, dtype=float)[receivers]
        - np.asarray(event.pick_time, dtype=float)[receivers]
    )
    difference = difference[np.isfinite(difference)]
    if difference.size < int(minimum_overlap):
        return None
    median_difference = float(np.median(difference))
    p90_difference = float(np.percentile(difference, 90.0))
    if (
        median_difference > float(median_tolerance_time)
        or p90_difference > float(p90_tolerance_time)
    ):
        return None
    return int(difference.size), median_difference, p90_difference


def _cluster_boundary_short_events(
    entries, *, flat, receiver_x, quality_window_receivers, low_similarity_threshold,
    pilot_min_coverage, pilot_min_median_zncc, phase_switch_zncc,
    phase_switch_prediction_time, coherence_half_window_time, dt,
):
    """Turn short-pilot hypotheses into coherent short-event candidates.

    This deliberately does *not* require two or more seed origins.  The event
    itself is judged by its two-dimensional path quality.  Multiple origins
    strengthen a cluster only as a late ranking tie-break.
    """
    eligible = [
        entry for entry in entries
        if _boundary_short_pilot_eligible(entry[1], quality_window_receivers)
    ]
    eligible.sort(key=_boundary_short_candidate_rank, reverse=True)
    if not eligible:
        return ()

    minimum_overlap = int(quality_window_receivers)
    median_tolerance = float(phase_switch_prediction_time)
    p90_tolerance = 2.0 * float(phase_switch_prediction_time)
    groups = []
    for entry in eligible:
        best_group = None
        best_score = None
        for group_index, members in enumerate(groups):
            event = _build_boundary_short_event(
                members,
                flat=flat, receiver_x=receiver_x,
                quality_window_receivers=quality_window_receivers,
                low_similarity_threshold=low_similarity_threshold,
                pilot_min_coverage=pilot_min_coverage,
                pilot_min_median_zncc=pilot_min_median_zncc,
                phase_switch_zncc=phase_switch_zncc,
                phase_switch_prediction_time=phase_switch_prediction_time,
                coherence_half_window_time=coherence_half_window_time,
                dt=dt,
            )
            match = _short_candidate_matches_event(
                entry, event,
                median_tolerance_time=median_tolerance,
                p90_tolerance_time=p90_tolerance,
                minimum_overlap=minimum_overlap,
            )
            if match is None:
                continue
            overlap_count, median_difference, p90_difference = match
            score = (overlap_count, -p90_difference, -median_difference)
            if best_score is None or score > best_score:
                best_score = score
                best_group = group_index
        if best_group is None:
            groups.append([entry])
        else:
            groups[best_group].append(entry)

    events = []
    for members in groups:
        event = _build_boundary_short_event(
            members,
            flat=flat, receiver_x=receiver_x,
            quality_window_receivers=quality_window_receivers,
            low_similarity_threshold=low_similarity_threshold,
            pilot_min_coverage=pilot_min_coverage,
            pilot_min_median_zncc=pilot_min_median_zncc,
            phase_switch_zncc=phase_switch_zncc,
            phase_switch_prediction_time=phase_switch_prediction_time,
            coherence_half_window_time=coherence_half_window_time,
            dt=dt,
        )
        if event.passed:
            events.append(event)
    events.sort(key=lambda item: item.rank, reverse=True)
    return tuple(events)


def _select_seed_from_short_event(event):
    """Construct a real production seed directly from the accepted event.

    The representative window contains observed positive-peak samples chosen by
    the consensus builder.  It therefore provides both a real seed sample and
    a contiguous frozen anchor without asking any one original hypothesis to
    have survived the legacy anchor/long-pilot gate.
    """
    start = int(event.representative_start_receiver)
    end = int(event.representative_end_receiver)
    center = int(event.representative_center_receiver)
    success = np.asarray(event.success_mask, dtype=bool)
    samples = np.asarray(event.pick_sample, dtype=int)
    times = np.asarray(event.pick_time, dtype=float)
    if start < 0 or end < start:
        return None
    anchor_mask = np.zeros(success.size, dtype=bool)
    anchor_mask[start:end + 1] = success[start:end + 1]
    receivers = np.flatnonzero(anchor_mask & (samples >= 0) & np.isfinite(times))
    if not receivers.size:
        return None
    # The best short-event window is contiguous by construction.  Choose the
    # observed peak nearest its center; never manufacture an interpolated peak.
    receiver = int(receivers[np.argmin(np.abs(receivers - center))])
    anchor_sample = np.where(anchor_mask, samples, -1)
    anchor_time = np.where(anchor_mask, times, np.nan)
    seed = EventProductionSeed(
        seed_time=float(times[receiver]),
        seed_sample=int(samples[receiver]),
        anchor_receiver_mask=anchor_mask,
        anchor_pick_sample=anchor_sample,
        anchor_pick_time=anchor_time,
    )

    # Keep one member only for diagnostics/evidence logging.  It no longer
    # controls the production seed or anchor.
    members = list(event.members)
    representative = min(
        members,
        key=lambda entry: (
            abs(int(entry[0]) - receiver),
            -float(getattr(entry[1], "pilot_median_zncc", -np.inf)),
        ),
    ) if members else None
    candidate = None if representative is None else representative[1]
    return receiver, candidate, seed



def _status_name(status):
    return str(getattr(status, "value", status))


def _original_anchor_mask(tracking):
    """Recover the frozen ORIGINAL seed/anchor receivers from tracker diagnostics."""
    success = np.asarray(tracking.success_mask, dtype=bool)
    kinds = np.asarray(
        getattr(tracking, "selected_transition_kind", np.full(success.size, -1)),
        dtype=int,
    )
    if kinds.shape != success.shape:
        kinds = np.full(success.size, -1, dtype=int)
    # Sparse tracker TransitionKind.SEED == 0 and ANCHOR == 1.
    mask = success & ((kinds == 0) | (kinds == 1))
    seed = int(getattr(tracking, "seed_receiver", -1))
    if 0 <= seed < mask.size and success[seed]:
        mask[seed] = True
    return mask


def _anchor_bank_from_tracking(flat, tracking, *, dt, config, tracker_options):
    mask = _original_anchor_mask(tracking)
    return select_representative_waveform_bank(
        flat,
        pick_sample=tracking.pick_sample,
        eligible_mask=mask,
        dt=dt,
        prediction_error=tracking.prediction_error,
        window_receivers=int(config.quality_window_receivers),
        low_similarity_threshold=float(config.low_similarity_threshold),
        low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
        coherence_half_window_time=float(
            dict(tracker_options or {}).get("coherence_half_window_time", 0.040)
        ),
        allow_short=True,
        label="ANCHOR",
    )


def _best_bank_from_segments(flat, segments, *, dt, config, tracker_options, label):
    choices = []
    for index, segment in enumerate(segments):
        bank = bank_from_trusted_segment(
            flat,
            segment,
            dt=dt,
            config=config,
            tracker_options=tracker_options,
            label=f"{label}_{index}",
        )
        if bank is not None:
            choices.append((bank.rank, int(segment.success_count), bank, segment))
    if not choices:
        return None, None
    _, _, bank, segment = max(choices, key=lambda item: (item[0], item[1]))
    bank = replace(bank, label=str(label))
    return bank, segment


def _segment_inside_state_valid(segment, state_valid):
    valid = np.asarray(state_valid, dtype=bool)
    tracking = segment.tracking
    used = np.flatnonzero(
        np.asarray(segment.trusted_mask, bool)
        & np.asarray(tracking.success_mask, bool)
        & (np.asarray(tracking.pick_sample, int) >= 0)
    )
    if not used.size:
        return False
    samples = np.asarray(tracking.pick_sample, int)[used]
    return bool(np.all(valid[used, samples]))


def _required_repair_sides(original_audit):
    required = []
    for name in ("left", "right"):
        side = getattr(original_audit, name, None)
        if side is None or not bool(getattr(side, "present", False)):
            continue
        if _status_name(getattr(side, "status", "")) == "TRACKING_BREAK":
            required.append(name.upper())
    return tuple(required)


def _segment_side(segment):
    source = str(segment.source_type).upper()
    if source.startswith("LEFT_"):
        return "LEFT"
    if source.startswith("RIGHT_"):
        return "RIGHT"
    if source == "ORIGINAL":
        return "ORIGINAL"
    return ""


def _print_waveform_bank(prefix, bank, *, source):
    print(f"{prefix} WAVEFORM-BANK source={source}", flush=True)
    if bank is None:
        print("  available = False", flush=True)
        return
    print("  available = True", flush=True)
    print(f"  receivers = {int(bank.receivers[0])}:{int(bank.receivers[-1])}", flush=True)
    print(f"  waveform_count = {bank.size}", flush=True)
    print(f"  median_adjacent_ZNCC = {bank.median_adjacent_zncc:.3f}", flush=True)
    print(f"  p25_adjacent_ZNCC = {bank.lower_quartile_adjacent_zncc:.3f}", flush=True)
    print(f"  low_adjacent_count = {bank.low_adjacent_count}", flush=True)
    print(f"  representative = {bank.representative}", flush=True)


def _print_identity_comparison(prefix, result, *, name):
    print(f"{prefix} IDENTITY-CHECK {name}", flush=True)
    print(f"  {result.label_a}_median_best_ZNCC = {result.median_best_for_a:.3f}", flush=True)
    print(f"  {result.label_b}_median_best_ZNCC = {result.median_best_for_b:.3f}", flush=True)
    print(f"  {result.label_a}_low_count = {result.low_count_a}", flush=True)
    print(f"  {result.label_b}_low_count = {result.low_count_b}", flush=True)
    print(f"  template_ZNCC = {result.template_zncc:.3f}", flush=True)
    print(f"  threshold = {result.threshold:.3f}", flush=True)
    print(f"  passed = {result.passed}", flush=True)
    if not result.passed:
        print(f"  reason = {result.reason}", flush=True)

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

    tracker_cfg = dict(tracker_options or {})
    for key in ("continuation_rescue", "ownership_escape", "dt", "t0", "seed_receiver",
                "seed_time", "state_valid", "escape_state_valid", "anchor_receiver_mask",
                "anchor_pick_sample", "anchor_pick_time"):
        tracker_cfg.pop(key, None)
    seed_cfg = dict(seed_search_options or {})
    for key in ("tracker_options", "seed_receiver", "state_valid", "dt", "t0"):
        seed_cfg.pop(key, None)

    # Rebuild the ORIGINAL anchor identity directly from the finished sparse
    # track.  Transition kinds 0/1 are the frozen seed/anchor path.
    original_anchor_mask = _original_anchor_mask(original_tracking)
    original_seed_receiver = int(getattr(original_tracking, "seed_receiver", control_receiver))
    original_audit = audit_rkshot(
        success_mask=original_tracking.success_mask, valid_receiver=valid_receiver,
        neighbor_similarity=original_tracking.neighbor_correlation,
        prediction_error=original_tracking.prediction_error,
        seed_receiver=original_seed_receiver, anchor_receiver_mask=original_anchor_mask,
        tracking_usable_receiver=usable, quality_available_receiver=available,
        config=config, reflector=reflector,
    )
    anchor_bank = _anchor_bank_from_tracking(
        flat, original_tracking, dt=dt, config=config, tracker_options=tracker_cfg
    )
    original_bank, original_bank_segment = _best_bank_from_segments(
        flat, original_segments, dt=dt, config=config, tracker_options=tracker_cfg,
        label="ORIGINAL",
    )
    anchor_trusted = bool(
        _status_name(original_audit.status) != "SEED_ERROR"
        and anchor_bank is not None and anchor_bank.representative
        and original_bank is not None and original_bank.representative
    )

    if anchor_trusted:
        preserved, preserved_mask, coverage = _best_segments(original_segments, available)
        required_sides = _required_repair_sides(original_audit)
    else:
        # The original seed/anchor cannot certify reflector identity.  Do not
        # preserve ORIGINAL as final evidence; independently repair both ends.
        preserved = ()
        preserved_mask = np.zeros(available.size, dtype=bool)
        coverage = 0.0
        required_sides = ("LEFT", "RIGHT") if np.any(available) else ()

    available_indices = np.flatnonzero(available)
    if available_indices.size:
        aperture_left, aperture_right = map(int, available_indices[[0, -1]])
        side_receiver = {"LEFT": aperture_left, "RIGHT": aperture_right}
        boundary_receivers = tuple(
            side_receiver[side] for side in required_sides if side in side_receiver
        )
        if anchor_trusted and not boundary_receivers:
            # A TRACKING_BREAK is never allowed to return SUCCESS merely because
            # preserved coverage already exceeds 0.80.  If side status is
            # unexpectedly non-specific, fall back to the uncovered edge(s).
            boundary_receivers = _boundary_receivers(preserved, available)
            inferred = []
            for receiver in boundary_receivers:
                if int(receiver) == aperture_left:
                    inferred.append("LEFT")
                elif int(receiver) == aperture_right:
                    inferred.append("RIGHT")
            required_sides = tuple(inferred)
    else:
        aperture_left = aperture_right = -1
        boundary_receivers = ()

    print(f"{prefix} IDENTITY-REFERENCE", flush=True)
    print(f"  original_audit = {_status_name(original_audit.status)}", flush=True)
    print(f"  original_seed_receiver = {original_seed_receiver}", flush=True)
    print(f"  anchor_receivers = {int(np.count_nonzero(original_anchor_mask))}", flush=True)
    print(f"  anchor_trusted = {anchor_trusted}", flush=True)
    print(f"  required_repair_sides = {required_sides}", flush=True)
    _print_waveform_bank(prefix, anchor_bank, source="ANCHOR")
    _print_waveform_bank(prefix, original_bank, source="ORIGINAL")
    if not anchor_trusted:
        print(f"{prefix} ORIGINAL-IDENTITY discarded", flush=True)
        print("  reason = ANCHOR_OR_ORIGINAL_REFERENCE_NOT_REPRESENTATIVE", flush=True)

    upper_neighbor = None if upper_neighbor_state_valid is None else np.asarray(upper_neighbor_state_valid, bool)
    lower_neighbor = None if lower_neighbor_state_valid is None else np.asarray(lower_neighbor_state_valid, bool)
    repair_guard_samples = max(
        1, int(round(float(seed_cfg.get("evidence_half_width_time", 0.020)) / float(dt)))
    )
    repair_state_valid = build_interlayer_repair_state_valid(
        state_valid, upper_neighbor, lower_neighbor, guard_samples=repair_guard_samples
    )
    repair_pool, seed_times, seed_energy, diagnostics = [], [], [], []
    repair_metadata = {}
    full_count = accepted_pilots = 0
    used_boundaries = []

    # Once repair is entered, every required side is searched regardless of the
    # current preserved coverage.  Coverage is a final quantity gate only.
    if boundary_receivers and np.any(available):
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

            # Boundary repair is event-first, not seed-first.  Scan a complete
            # boundary strip and retain every *short-pilot* path that passes.
            # These paths are clustered into coherent two-dimensional events.
            # Only after an event itself passes do we construct the production
            # seed and frozen anchor directly from that event's clean window.
            candidate_entries = []
            receiver = int(requested)
            available_order = np.flatnonzero(available)
            inward = (
                available_order[available_order >= receiver]
                if direction > 0
                else available_order[available_order <= receiver][::-1]
            )
            search_receiver_count = max(
                qwindow + 1, int(seed_cfg.get("pilot_short_receiver_count", 25))
            )
            trial_receivers = inward[:search_receiver_count]
            for trial_receiver in trial_receivers:
                trial = discover_boundary_seed_candidates(
                    flat, receiver_x=receiver_x, valid_receiver=valid_receiver,
                    seed_receiver=int(trial_receiver), inward_direction=direction, state_valid=state_valid,
                    dt=dt, t0=t0, tracker_options=tracker_cfg, reference_time=reference_time,
                    upper_neighbor_state_valid=upper_neighbor,
                    lower_neighbor_state_valid=lower_neighbor, **seed_cfg,
                )
                current_peaks = trial[0].current_ownership_peak_count if trial else 0
                safe_peaks = trial[0].exclusive_ownership_peak_count if trial else 0
                short_passed_here = [
                    candidate for candidate in trial
                    if _boundary_short_pilot_eligible(candidate, qwindow)
                ]
                viable_seed_here = [candidate for candidate in short_passed_here if candidate.seed is not None]
                candidate_entries.extend(
                    (int(trial_receiver), candidate) for candidate in short_passed_here
                )
                print(f"{prefix} SEED-SEARCH side={side}", flush=True)
                print(f"  receiver = {int(trial_receiver)}", flush=True)
                print(f"  current_Rk_ownership_peaks = {current_peaks}", flush=True)
                print(f"  safe_interlayer_peaks = {safe_peaks}", flush=True)
                print(f"  short_pilots_passed_here = {len(short_passed_here)}", flush=True)
                print(f"  anchor_long_validated_here = {len(viable_seed_here)}", flush=True)
                for candidate_index, candidate in enumerate(trial, 1):
                    delta = abs(candidate.seed_time - reference_time) * 1e3
                    if candidate.seed is None:
                        print(f"{prefix} SEED-PILOT side={side} candidate={candidate_index}", flush=True)
                        print(f"  seed_receiver={int(trial_receiver)} seed_time={candidate.seed_time:.6f} "
                              f"rejected={candidate.rejection_reason} "
                              f"delta_from_reference_ms={delta:.1f}", flush=True)

            accepted_pilots += len(candidate_entries)
            phase_switch_zncc = float(seed_cfg.get("phase_switch_zncc", 0.0))
            phase_switch_prediction_time = float(
                seed_cfg.get("phase_switch_prediction_time", config.severe_bad_prediction_error_time)
            )
            pilot_min_coverage = float(seed_cfg.get("pilot_min_coverage", 0.80))
            pilot_min_median_zncc = float(seed_cfg.get("pilot_min_median_zncc", 0.70))
            coherence_half_window_time = float(tracker_cfg.get("coherence_half_window_time", 0.040))
            events = _cluster_boundary_short_events(
                candidate_entries,
                flat=flat,
                receiver_x=receiver_x,
                quality_window_receivers=qwindow,
                low_similarity_threshold=float(config.low_similarity_threshold),
                pilot_min_coverage=pilot_min_coverage,
                pilot_min_median_zncc=pilot_min_median_zncc,
                phase_switch_zncc=phase_switch_zncc,
                phase_switch_prediction_time=phase_switch_prediction_time,
                coherence_half_window_time=coherence_half_window_time,
                dt=dt,
            )
            print(f"{prefix} SHORT-EVENTS side={side}", flush=True)
            print(f"  scanned_receivers = {len(trial_receivers)}", flush=True)
            print(f"  short_pilots_passed = {len(candidate_entries)}", flush=True)
            print(f"  accepted_event_count = {len(events)}", flush=True)
            print(f"  event_min_overlap_receivers = {qwindow}", flush=True)
            print(f"  event_median_match_tolerance_ms = {phase_switch_prediction_time*1e3:.1f}", flush=True)
            print(f"  event_p90_match_tolerance_ms = {2.0*phase_switch_prediction_time*1e3:.1f}", flush=True)
            print("  minimum_seed_origin_support = NONE", flush=True)
            for event_index, event in enumerate(events, 1):
                viable_count = sum(member[1].seed is not None for member in event.members)
                print(f"  event {event_index}: origins={event.origin_receivers} "
                      f"members={len(event.members)} viable_seed_members={viable_count}", flush=True)
                print(f"    event_success = {event.success_count}/{event.tested_count}", flush=True)
                print(f"    event_coverage = {event.coverage:.3f}", flush=True)
                print(f"    event_median_ZNCC = {event.median_zncc:.3f}", flush=True)
                print(f"    event_prediction_error_ms = {event.median_abs_prediction_error*1e3:.1f}", flush=True)
                print(f"    event_p90_slope_ms_per_100m = {event.p90_abs_slope_ms_per_100m:.3f}", flush=True)
                print(f"    event_phase_switch_count = {event.phase_switch_count}", flush=True)
                print(f"    member_median_deviation_ms = {event.median_member_deviation_time*1e3:.1f}", flush=True)
                print(f"    member_p90_deviation_ms = {event.p90_member_deviation_time*1e3:.1f}", flush=True)
                print(f"    representative_receivers = "
                      f"{event.representative_start_receiver}:{event.representative_end_receiver}", flush=True)

            if not events:
                print(f"{prefix} BOUNDARY side={side} failed", flush=True)
                print("  reason = NO_VALID_SHORT_EVENT", flush=True)
                continue

            selected_entries = []
            selected_events = []
            for event in events:
                selected = _select_seed_from_short_event(event)
                if selected is None:
                    continue
                selected_entries.append(selected)
                selected_events.append(event)
                if len(selected_entries) >= 3:
                    break
            if not selected_entries:
                print(f"{prefix} BOUNDARY side={side} failed", flush=True)
                print("  reason = SHORT_EVENT_HAS_NO_PRODUCTION_SEED", flush=True)
                continue

            used_boundaries.extend(int(candidate_receiver) for candidate_receiver, _, _ in selected_entries)
            print(f"{prefix} TOP-SEEDS side={side}", flush=True)
            for rank_index, (candidate_receiver, candidate, event_seed) in enumerate(selected_entries, 1):
                event = selected_events[rank_index - 1]
                print(f"  rank {rank_index}: seed_receiver={candidate_receiver} "
                      f"seed_time={event_seed.seed_time:.6f}", flush=True)
                print(f"    event_origins = {event.origin_receivers}", flush=True)
                print(f"    event_members = {len(event.members)}", flush=True)
                print(f"    event_representative_receivers = "
                      f"{event.representative_start_receiver}:{event.representative_end_receiver}", flush=True)
                print(f"    event_coverage = {event.coverage:.3f}", flush=True)
                print(f"    event_ZNCC = {event.median_zncc:.3f}", flush=True)
                print(f"    event_prediction_error_ms = {event.median_abs_prediction_error*1e3:.1f}", flush=True)
                print(f"    event_p90_slope = {event.p90_abs_slope_ms_per_100m:.3f}", flush=True)
                if candidate is not None:
                    print(f"    member_long_coverage = {candidate.long_coverage:.3f}", flush=True)
                    print(f"    member_phase_switch = {candidate.long_phase_switch_count}", flush=True)
                    print(f"    member_long_p90_slope = {candidate.long_p90_abs_slope_ms_per_100m:.3f}", flush=True)
                    print(f"    member_long_ZNCC = {candidate.long_median_zncc:.3f}", flush=True)
                    print(f"    member_long_prediction_error_ms = {candidate.long_median_abs_prediction_error*1e3:.1f}", flush=True)
                    print(f"    evidence = {candidate.stack_evidence:.3f}", flush=True)
                    print(f"    support = {candidate.positive_support_fraction:.3f}", flush=True)
                    print(f"    old_anchor_length = {candidate.anchor_length}", flush=True)
                    print(f"    amplitude = {candidate.seed_amplitude:.3f}", flush=True)
                print(f"    event_anchor_length = {int(np.count_nonzero(event_seed.anchor_receiver_mask))}", flush=True)
                print(f"    delta_from_reference_ms = {abs(event_seed.seed_time-reference_time)*1e3:.1f}", flush=True)

            for seed_rank, (receiver, candidate, seed) in enumerate(selected_entries, 1):
                seed_times.append(float(seed.seed_time))
                seed_energy.append(float(candidate.short_segment_energy) if candidate is not None else 0.0)
                tracking = track_flattened_event_sparse(
                    flat, receiver_x=receiver_x, valid_receiver=valid_receiver,
                    seed_receiver=receiver, seed_time=seed.seed_time, state_valid=repair_state_valid,
                    escape_state_valid=repair_state_valid, anchor_receiver_mask=seed.anchor_receiver_mask,
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
                print(f"  stop_reason_left = {getattr(tracking, 'stop_reason_left', '')}", flush=True)
                print(f"  stop_receiver_left = {getattr(tracking, 'stop_receiver_left', -1)}", flush=True)
                print(f"  stop_reason_right = {getattr(tracking, 'stop_reason_right', '')}", flush=True)
                print(f"  stop_receiver_right = {getattr(tracking, 'stop_receiver_right', -1)}", flush=True)
                print(f"  neighbor_ownership_intrusions = {np.count_nonzero(neighbor_intrusion[tracked])}", flush=True)
                print(f"  persistent_neighbor_ownership_intrusion = {persistent_intrusion}", flush=True)
                print(f"  raw_audit = {raw_audit.status.value}", flush=True)
                # Event-first seeds are often interior.  Judge the trusted
                # repair interval on both sides of the seed instead of silently
                # discarding the aperture-edge side as the old boundary-prefix
                # logic did.
                allowed = np.asarray(available, dtype=bool).copy()
                source_type = f"{side}_SEED_{seed_rank}"
                raw, prefix_rejection = extract_seed_connected_trusted_segment(
                    flat=flat, tracking=tracking, tracking_usable_receiver=usable,
                    quality_available_receiver=available, config=config, source_type=source_type,
                    seed_receiver=receiver, source_seed_time=seed.seed_time,
                    allowed_receiver_mask=allowed,
                    neighbor_intrusion_mask=neighbor_intrusion,
                    receiver_x=receiver_x, macro_window=macro_window,
                    macro_residual_limit_time=macro_residual_limit_time,
                    anchor_receiver_mask=seed.anchor_receiver_mask,
                )
                rejected_end = None
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
                        print("  first_bad_window = SYMMETRIC_SEARCH_DIAGNOSTIC", flush=True)
                        print(f"  kept_seed_connected_interval = {raw.start_receiver}:{raw.end_receiver}", flush=True)
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
                overlap_identity_by_raw = {}
                for raw in raw_segments:
                    if anchor_trusted:
                        overlap_identity = _same_receiver_overlap_identity(
                            flat, raw, preserved, dt=dt, config=config,
                            tracker_options=tracker_cfg,
                        )
                        overlap_identity_by_raw[id(raw)] = overlap_identity
                        receivers_text = (
                            f"{int(overlap_identity['receivers'][0])}:{int(overlap_identity['receivers'][-1])}"
                            if overlap_identity['receivers'].size else "NONE"
                        )
                        print(f"{prefix} ORIGINAL-OVERLAP-IDENTITY source={source_type}", flush=True)
                        print(f"  receivers = {receivers_text}", flush=True)
                        print(f"  median_same_receiver_ZNCC = {overlap_identity['median']:.3f}", flush=True)
                        print(f"  p25_same_receiver_ZNCC = {overlap_identity['p25']:.3f}", flush=True)
                        print(f"  low_count = {overlap_identity['low_count']}/{qwindow}", flush=True)
                        print(f"  threshold = {config.low_similarity_threshold:.3f}", flush=True)
                        print(f"  passed = {overlap_identity['passed']}", flush=True)
                        if not overlap_identity['passed']:
                            print(f"  reason = {overlap_identity['reason']}", flush=True)

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
                        saved_item = replace(
                            item,
                            raw_start_receiver=raw.start_receiver, raw_end_receiver=raw.end_receiver,
                        )
                        repair_pool.append(saved_item)
                        repair_metadata[id(saved_item)] = {
                            "raw_audit_status": _status_name(raw_audit.status),
                            "trusted_quality_passed": True,
                            "macro_smoothness_passed": bool(macro_passed),
                            "neighbor_ownership_passed": bool(saved_item.identity_valid),
                            "interlayer_safe_band_passed": _segment_inside_state_valid(
                                saved_item, repair_state_valid
                            ),
                            "side": side,
                            "seed_rank": int(seed_rank),
                            "original_overlap_identity_passed": bool(
                                overlap_identity_by_raw.get(id(raw), {}).get(
                                    "passed", not anchor_trusted
                                )
                            ),
                            "original_overlap_identity_reason": str(
                                overlap_identity_by_raw.get(id(raw), {}).get("reason", "")
                            ),
                        }
                diagnostics.append(
                    f"full:r={receiver},t={seed.seed_time:.8g},tracked={tracked.size},"
                    f"neighbor_intrusion_valid={int(not persistent_intrusion)},reference={reference_time:.8g}"
                )

    # ------------------------------------------------------------------
    # Unified final identity gate.
    # Coverage is only a quantity condition; it never certifies Rk identity.
    # ------------------------------------------------------------------
    repair_banks = {}
    for segment in repair_pool:
        bank = bank_from_trusted_segment(
            flat,
            segment,
            dt=dt,
            config=config,
            tracker_options=tracker_cfg,
            label=str(segment.source_type),
        )
        repair_banks[id(segment)] = bank
        _print_waveform_bank(prefix, bank, source=segment.source_type)

    identity_mode = "ANCHOR_BACKED" if anchor_trusted else "NO_TRUSTED_ANCHOR"
    identity_passed = False
    identity_failure_reason = "WAVEFORM_IDENTITY_FAILED"
    selected = ()

    if anchor_trusted:
        valid_repairs = []
        for segment in repair_pool:
            bank = repair_banks.get(id(segment))
            metadata = repair_metadata.get(id(segment), {})
            local_ok = bool(
                metadata.get("trusted_quality_passed", False)
                and metadata.get("macro_smoothness_passed", False)
                and metadata.get("neighbor_ownership_passed", False)
                and metadata.get("interlayer_safe_band_passed", False)
                and metadata.get("original_overlap_identity_passed", False)
            )
            if bank is None or not bank.representative:
                print(f"{prefix} IDENTITY-GATE source={segment.source_type}", flush=True)
                print(f"  local_quality_passed = {local_ok}", flush=True)
                print("  waveform_identity_passed = False", flush=True)
                print("  passed = False", flush=True)
                print("  reason = NO_REPRESENTATIVE_REPAIR_WAVEFORM_BANK", flush=True)
                continue

            gate = evaluate_anchor_backed_repair_identity(
                anchor_bank=anchor_bank,
                original_bank=original_bank,
                repair_bank=bank,
                similarity_threshold=float(config.low_similarity_threshold),
                low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
            )
            for comparison in gate.comparisons:
                name = (
                    f"source={segment.source_type} "
                    f"{comparison.label_a}_vs_{comparison.label_b}"
                )
                _print_identity_comparison(prefix, comparison, name=name)
            passed = bool(local_ok and gate.passed)
            print(f"{prefix} IDENTITY-GATE source={segment.source_type}", flush=True)
            print(f"  local_quality_passed = {local_ok}", flush=True)
            print(f"  waveform_identity_passed = {gate.passed}", flush=True)
            print(f"  original_overlap_identity_passed = {metadata.get('original_overlap_identity_passed', False)}", flush=True)
            if not metadata.get("original_overlap_identity_passed", False):
                print(f"  overlap_reason = {metadata.get('original_overlap_identity_reason', 'NO_QUALITY_WINDOW_OVERLAP')}", flush=True)
            print(f"  passed = {passed}", flush=True)
            if passed:
                valid_repairs.append(segment)

        # Keep at most two final segments.  A valid anchor-backed choice must
        # contain ORIGINAL plus independently identified repair evidence for
        # every side that caused TRACKING_BREAK.  ORIGINAL alone is forbidden.
        choices = []
        candidate_pool = list(preserved) + valid_repairs
        for size in (1, 2):
            for choice in combinations(candidate_pool, size):
                has_original = any(_segment_side(item) == "ORIGINAL" for item in choice)
                repair_sides = {
                    _segment_side(item)
                    for item in choice
                    if _segment_side(item) in ("LEFT", "RIGHT")
                }
                if not has_original or not repair_sides:
                    continue
                if any(side not in repair_sides for side in required_sides):
                    continue
                choices.append(choice)
        if choices:
            selected = max(
                choices,
                key=lambda choice: _final_choice_rank(
                    choice, available, required_coverage=0.80
                )[0],
            )
            identity_passed = True
            identity_failure_reason = ""
        else:
            identity_failure_reason = (
                "NO_ANCHOR_BACKED_REPAIR_IDENTITY"
                if not valid_repairs else "MISSING_REQUIRED_REPAIR_SIDE"
            )

    else:
        # No trustworthy original identity root.  ORIGINAL cannot participate
        # in final evidence.  First try LEFT+RIGHT repair cross-identification.
        pair_choices = []
        for first, second in combinations(repair_pool, 2):
            if {_segment_side(first), _segment_side(second)} != {"LEFT", "RIGHT"}:
                continue
            bank_first = repair_banks.get(id(first))
            bank_second = repair_banks.get(id(second))
            if bank_first is None or bank_second is None:
                continue
            meta_first = repair_metadata.get(id(first), {})
            meta_second = repair_metadata.get(id(second), {})
            local_first = bool(
                meta_first.get("trusted_quality_passed", False)
                and meta_first.get("macro_smoothness_passed", False)
                and meta_first.get("neighbor_ownership_passed", False)
                and meta_first.get("interlayer_safe_band_passed", False)
            )
            local_second = bool(
                meta_second.get("trusted_quality_passed", False)
                and meta_second.get("macro_smoothness_passed", False)
                and meta_second.get("neighbor_ownership_passed", False)
                and meta_second.get("interlayer_safe_band_passed", False)
            )
            gate = evaluate_dual_repair_identity_without_anchor(
                left_bank=(bank_first if _segment_side(first) == "LEFT" else bank_second),
                right_bank=(bank_second if _segment_side(second) == "RIGHT" else bank_first),
                similarity_threshold=float(config.low_similarity_threshold),
                low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
            )
            comparison = gate.comparisons[0]
            _print_identity_comparison(
                prefix,
                comparison,
                name=f"{first.source_type}_vs_{second.source_type}",
            )
            passed = bool(local_first and local_second and gate.passed)
            print(f"{prefix} DUAL-REPAIR-GATE", flush=True)
            print(f"  first = {first.source_type}", flush=True)
            print(f"  second = {second.source_type}", flush=True)
            print(f"  first_local_quality = {local_first}", flush=True)
            print(f"  second_local_quality = {local_second}", flush=True)
            print(f"  waveform_identity_passed = {gate.passed}", flush=True)
            print(f"  passed = {passed}", flush=True)
            if passed:
                pair_choices.append((first, second))

        if pair_choices:
            selected = max(
                pair_choices,
                key=lambda choice: _final_choice_rank(
                    choice, available, required_coverage=0.80
                )[0],
            )
            identity_passed = True
            identity_failure_reason = ""
        else:
            # With only one usable repair identity chain, there is no external
            # waveform reference.  It may succeed only if the full retracking
            # audit is NORMAL and all local/safe-band checks are already valid.
            single_choices = []
            for segment in repair_pool:
                metadata = repair_metadata.get(id(segment), {})
                segment_coverage = (
                    float(np.count_nonzero(segment.trusted_mask & available) / quality_count)
                    if quality_count else 0.0
                )
                gate = evaluate_single_repair_without_anchor(
                    raw_audit_status=metadata.get("raw_audit_status", ""),
                    trusted_quality_passed=metadata.get("trusted_quality_passed", False),
                    macro_smoothness_passed=metadata.get("macro_smoothness_passed", False),
                    neighbor_ownership_passed=metadata.get("neighbor_ownership_passed", False),
                    interlayer_safe_band_passed=metadata.get("interlayer_safe_band_passed", False),
                    trusted_union_coverage=segment_coverage,
                    required_coverage=0.80,
                )
                print(f"{prefix} SINGLE-REPAIR-GATE source={segment.source_type}", flush=True)
                print(f"  raw_audit = {metadata.get('raw_audit_status', 'UNKNOWN')}", flush=True)
                print(f"  trusted_coverage = {segment_coverage:.3f}", flush=True)
                print(f"  passed = {gate.passed}", flush=True)
                if gate.reasons:
                    print(f"  reason = {'+'.join(gate.reasons)}", flush=True)
                if gate.passed:
                    single_choices.append((segment,))
            if single_choices:
                selected = max(
                    single_choices,
                    key=lambda choice: _final_choice_rank(
                    choice, available, required_coverage=0.80
                )[0],
                )
                identity_passed = True
                identity_failure_reason = ""
                identity_mode = "SINGLE_REPAIR_SELF_PROVING"
            else:
                identity_failure_reason = "NO_CROSS_REPAIR_IDENTITY_OR_NORMAL_SINGLE_REPAIR"

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
            metadata = repair_metadata.get(id(segment), {})
            bank = repair_banks.get(id(segment))
            print(f"    raw_receivers={segment.raw_start_receiver}:{segment.raw_end_receiver}", flush=True)
            print(f"    after_original_overlap_trim={segment.start_receiver}:{segment.end_receiver}", flush=True)
            print(f"    new_trusted_receivers={np.count_nonzero(segment.trusted_mask & available & ~preserved_mask)}", flush=True)
            print(f"    neighbor_ownership_valid={segment.identity_valid}", flush=True)
            print(f"    raw_audit={metadata.get('raw_audit_status', 'UNKNOWN')}", flush=True)
            print(f"    interlayer_safe_band_valid={metadata.get('interlayer_safe_band_passed', False)}", flush=True)
            print(f"    representative_waveform_bank={bool(bank is not None and bank.representative)}", flush=True)

    selected_ids = [next(index for index, item in enumerate(pool) if item is segment) for segment in selected]
    print(f"{prefix} SELECTED", flush=True)
    print(f"  identity_mode = {identity_mode}", flush=True)
    print(f"  identity_passed = {identity_passed}", flush=True)
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
    success_ready = bool(
        identity_passed
        and coverage >= 0.80
        and 0 < len(selected) <= 2
        and neighbor_ownership_valid
    )
    if not success_ready:
        print(f"{prefix} RESULT = FAILURE", flush=True)
        print(f"  selected_segments = {len(selected)}", flush=True)
        print(f"  best_trusted_union_coverage = {coverage:.3f}", flush=True)
        print("  required = 0.800", flush=True)
        if not identity_passed:
            reason = identity_failure_reason
        elif not neighbor_ownership_valid:
            reason = "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION"
        elif len(selected) > 2:
            reason = "TOO_MANY_SELECTED_SEGMENTS"
        elif coverage < 0.80:
            reason = "INSUFFICIENT_TRUSTED_COVERAGE"
        else:
            reason = "NO_VALID_TRUSTED_SEGMENT"
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
