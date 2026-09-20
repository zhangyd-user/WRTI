"""Core helpers for TRACKING_BREAK repair identity and short-event grouping.

This module is intentionally standalone.  It does not modify the production
repair/search flow; it only implements the reusable decisions discussed for:

1. selecting a small, representative high-ZNCC waveform bank from one axis;
2. comparing two waveform banks with signed, zero-lag ZNCC;
3. deciding anchor-backed / dual-repair / single-repair identity cases;
4. clustering short-pilot paths into short EVENT hypotheses before choosing a seed.

The functions are written against plain NumPy arrays so they can later be
wired to the current TrustedSegment / SparseFlatTrackingResult objects with
minimal plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Waveform identity
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RepresentativeWaveformBank:
    """Small high-quality local waveform family representing one tracked axis."""

    label: str
    receivers: np.ndarray
    samples: np.ndarray
    waveforms: np.ndarray
    adjacent_zncc: np.ndarray
    median_adjacent_zncc: float
    lower_quartile_adjacent_zncc: float
    low_adjacent_count: int
    median_abs_prediction_error: float
    max_abs_prediction_error: float
    representative: bool
    template: np.ndarray
    rank: tuple

    @property
    def size(self) -> int:
        return int(self.receivers.size)


@dataclass(frozen=True)
class WaveformIdentityResult:
    """Symmetric bank-to-bank identity comparison."""

    passed: bool
    label_a: str
    label_b: str
    correlation_matrix: np.ndarray
    best_for_a: np.ndarray
    best_for_b: np.ndarray
    median_best_for_a: float
    median_best_for_b: float
    low_count_a: int
    low_count_b: int
    template_zncc: float
    threshold: float
    low_count_to_fail: int
    reason: str


@dataclass(frozen=True)
class IdentityGateResult:
    passed: bool
    mode: str
    reasons: tuple[str, ...]
    comparisons: tuple[WaveformIdentityResult, ...]


@dataclass(frozen=True)
class SingleRepairGateResult:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShortEventSeedSelection:
    source_id: int
    seed_receiver: int
    seed_time: float
    seed_sample: int
    median_bank_zncc: float
    inside_representative_window: bool
    rank: tuple


def _status_name(status) -> str:
    return str(getattr(status, "value", status))


def _contiguous_runs(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, cuts) if part.size]


def _unit_waveform(trace: np.ndarray, sample: int, half_samples: int) -> np.ndarray | None:
    """Match the production candidate-waveform definition: demean + L2 norm."""

    trace = np.asarray(trace, dtype=float)
    sample = int(sample)
    half_samples = int(half_samples)
    if sample - half_samples < 0 or sample + half_samples >= trace.size:
        return None
    waveform = np.array(
        trace[sample - half_samples : sample + half_samples + 1],
        dtype=float,
        copy=True,
    )
    if not np.all(np.isfinite(waveform)):
        return None
    waveform -= float(np.mean(waveform))
    norm = float(np.linalg.norm(waveform))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return waveform / norm


def _normalise_template(waveforms: np.ndarray) -> np.ndarray:
    template = np.median(np.asarray(waveforms, dtype=float), axis=0)
    template -= float(np.mean(template))
    norm = float(np.linalg.norm(template))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return np.full(template.shape, np.nan, dtype=float)
    return template / norm


def select_representative_waveform_bank(
    flat_gather: np.ndarray,
    *,
    pick_sample: np.ndarray,
    eligible_mask: np.ndarray,
    dt: float,
    prediction_error: np.ndarray | None = None,
    window_receivers: int = 8,
    low_similarity_threshold: float = 0.70,
    low_similarity_count_to_fail: int = 4,
    coherence_half_window_time: float = 0.040,
    allow_short: bool = False,
    label: str = "AXIS",
) -> RepresentativeWaveformBank | None:
    """Choose the most representative *local* part of one already-trusted axis.

    The search is over contiguous receiver windows, not the whole segment and
    not a hard-coded near-seed / far-edge region.  Each candidate window is
    scored using ZNCC of its own adjacent picked waveforms; prediction error is
    only a late tie-break.

    For normal ORIGINAL/REPAIR trusted segments, ``window_receivers`` should be
    the existing quality-window length (currently 8).  ``allow_short=True`` is
    intended only for an already validated anchor that contains fewer than 8
    receivers; in that case the whole contiguous anchor run is allowed.
    """

    flat = np.asarray(flat_gather, dtype=float)
    samples = np.asarray(pick_sample, dtype=int)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if flat.ndim != 2:
        raise ValueError("flat_gather must have shape [receiver, time]")
    if samples.shape != (flat.shape[0],) or eligible.shape != (flat.shape[0],):
        raise ValueError("pick_sample/eligible_mask must match receiver axis")
    if prediction_error is not None:
        prediction = np.asarray(prediction_error, dtype=float)
        if prediction.shape != samples.shape:
            raise ValueError("prediction_error must match receiver axis")
    else:
        prediction = None

    usable = eligible & (samples >= 0)
    runs = _contiguous_runs(usable)
    if not runs:
        return None

    width = max(2, int(window_receivers))
    half_samples = max(1, int(round(float(coherence_half_window_time) / float(dt))))
    candidates: list[RepresentativeWaveformBank] = []

    for run in runs:
        receiver_windows: list[np.ndarray] = []
        if run.size >= width:
            receiver_windows.extend(run[i : i + width] for i in range(run.size - width + 1))
        elif allow_short and run.size >= 2:
            receiver_windows.append(run)

        for receivers in receiver_windows:
            local_samples = samples[receivers]
            waveforms = []
            valid_window = True
            for receiver, sample in zip(receivers, local_samples):
                waveform = _unit_waveform(flat[int(receiver)], int(sample), half_samples)
                if waveform is None:
                    valid_window = False
                    break
                waveforms.append(waveform)
            if not valid_window:
                continue

            bank = np.stack(waveforms, axis=0)
            adjacent = np.einsum("ij,ij->i", bank[:-1], bank[1:])
            finite_adjacent = adjacent[np.isfinite(adjacent)]
            if finite_adjacent.size != adjacent.size or finite_adjacent.size == 0:
                continue

            low_count = int(np.count_nonzero(finite_adjacent < float(low_similarity_threshold)))
            median_rho = float(np.median(finite_adjacent))
            lower_quartile = float(np.percentile(finite_adjacent, 25.0))

            if prediction is None:
                median_prediction = 0.0
                max_prediction = 0.0
            else:
                local_prediction = np.abs(prediction[receivers])
                local_prediction = local_prediction[np.isfinite(local_prediction)]
                median_prediction = (
                    float(np.median(local_prediction)) if local_prediction.size else np.inf
                )
                max_prediction = (
                    float(np.max(local_prediction)) if local_prediction.size else np.inf
                )

            representative = bool(
                median_rho >= float(low_similarity_threshold)
                and low_count < int(low_similarity_count_to_fail)
            )

            # Do not use amplitude here.  We want the most internally coherent
            # part of the axis, not the strongest part of the gather.
            rank = (
                int(representative),
                -low_count,
                lower_quartile,
                median_rho,
                -median_prediction,
                -max_prediction,
            )
            candidates.append(
                RepresentativeWaveformBank(
                    label=str(label),
                    receivers=np.array(receivers, dtype=int, copy=True),
                    samples=np.array(local_samples, dtype=int, copy=True),
                    waveforms=bank,
                    adjacent_zncc=np.array(adjacent, dtype=float, copy=True),
                    median_adjacent_zncc=median_rho,
                    lower_quartile_adjacent_zncc=lower_quartile,
                    low_adjacent_count=low_count,
                    median_abs_prediction_error=median_prediction,
                    max_abs_prediction_error=max_prediction,
                    representative=representative,
                    template=_normalise_template(bank),
                    rank=tuple(rank),
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item.rank)


def compare_waveform_banks(
    bank_a: RepresentativeWaveformBank,
    bank_b: RepresentativeWaveformBank,
    *,
    similarity_threshold: float = 0.70,
    low_similarity_count_to_fail: int = 4,
) -> WaveformIdentityResult:
    """Compare two representative waveform families with ordered signed ZNCC.

    The old implementation let every waveform in A choose the maximum ZNCC
    against *any* waveform in B.  For band-limited seismic data that mostly
    tested whether the two banks contained the same source wavelet and could
    give a wrong reflector spuriously high scores.

    Here receiver order is preserved.  Each row is matched to the corresponding
    normalized position in the other bank.  With the normal eight-receiver
    banks this is simply A[i] vs B[i].  Unequal bank lengths are mapped by
    normalized receiver rank, so there is no all-to-all cherry-picking.
    """

    a = np.asarray(bank_a.waveforms, dtype=float)
    b = np.asarray(bank_b.waveforms, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("waveform banks must be 2-D and use the same waveform length")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("waveform banks must be non-empty")

    matrix = a @ b.T

    def aligned(source_count, target_count):
        if source_count <= 1 or target_count <= 1:
            return np.zeros(source_count, dtype=int)
        position = np.linspace(0.0, 1.0, source_count)
        return np.rint(position * float(target_count - 1)).astype(int)

    a_to_b = aligned(a.shape[0], b.shape[0])
    b_to_a = aligned(b.shape[0], a.shape[0])
    best_a = matrix[np.arange(a.shape[0]), a_to_b]
    best_b = matrix[b_to_a, np.arange(b.shape[0])]
    median_a = float(np.median(best_a))
    median_b = float(np.median(best_b))
    low_a = int(np.count_nonzero(best_a < float(similarity_threshold)))
    low_b = int(np.count_nonzero(best_b < float(similarity_threshold)))

    if np.all(np.isfinite(bank_a.template)) and np.all(np.isfinite(bank_b.template)):
        template_zncc = float(np.dot(bank_a.template, bank_b.template))
    else:
        template_zncc = np.nan

    passed = bool(
        bank_a.representative
        and bank_b.representative
        and median_a >= float(similarity_threshold)
        and median_b >= float(similarity_threshold)
        and low_a < int(low_similarity_count_to_fail)
        and low_b < int(low_similarity_count_to_fail)
    )

    reasons = []
    if not bank_a.representative:
        reasons.append(f"{bank_a.label}_NOT_REPRESENTATIVE")
    if not bank_b.representative:
        reasons.append(f"{bank_b.label}_NOT_REPRESENTATIVE")
    if median_a < float(similarity_threshold):
        reasons.append(f"{bank_a.label}_MEDIAN_MATCH_LOW")
    if median_b < float(similarity_threshold):
        reasons.append(f"{bank_b.label}_MEDIAN_MATCH_LOW")
    if low_a >= int(low_similarity_count_to_fail):
        reasons.append(f"{bank_a.label}_PERSISTENT_LOW_MATCH")
    if low_b >= int(low_similarity_count_to_fail):
        reasons.append(f"{bank_b.label}_PERSISTENT_LOW_MATCH")

    return WaveformIdentityResult(
        passed=passed,
        label_a=bank_a.label,
        label_b=bank_b.label,
        correlation_matrix=matrix,
        best_for_a=best_a,
        best_for_b=best_b,
        median_best_for_a=median_a,
        median_best_for_b=median_b,
        low_count_a=low_a,
        low_count_b=low_b,
        template_zncc=template_zncc,
        threshold=float(similarity_threshold),
        low_count_to_fail=int(low_similarity_count_to_fail),
        reason="" if passed else "+".join(reasons),
    )


def evaluate_anchor_backed_repair_identity(
    *,
    anchor_bank: RepresentativeWaveformBank,
    original_bank: RepresentativeWaveformBank,
    repair_bank: RepresentativeWaveformBank,
    similarity_threshold: float = 0.70,
    low_similarity_count_to_fail: int = 4,
) -> IdentityGateResult:
    """ANCHOR-trusted case: REPAIR must agree with the trusted ANCHOR family.

    ORIGINAL identity is now checked separately on the *same receivers* in the
    actual REPAIR/ORIGINAL overlap.  That is stronger than comparing two
    unrelated representative windows and prevents a generic seismic wavelet
    match from certifying the wrong reflector.  ``original_bank`` is retained
    in the signature for compatibility with the caller.
    """

    repair_anchor = compare_waveform_banks(
        repair_bank,
        anchor_bank,
        similarity_threshold=similarity_threshold,
        low_similarity_count_to_fail=low_similarity_count_to_fail,
    )
    comparisons = (repair_anchor,)
    reasons = (() if repair_anchor.passed else (repair_anchor.reason,))
    return IdentityGateResult(
        passed=repair_anchor.passed,
        mode="ANCHOR_BACKED_REPAIR",
        reasons=reasons,
        comparisons=comparisons,
    )


def evaluate_dual_repair_identity_without_anchor(
    *,
    left_bank: RepresentativeWaveformBank,
    right_bank: RepresentativeWaveformBank,
    similarity_threshold: float = 0.70,
    low_similarity_count_to_fail: int = 4,
) -> IdentityGateResult:
    """ANCHOR-untrusted case: LEFT and RIGHT repair axes cross-identify."""

    comparison = compare_waveform_banks(
        left_bank,
        right_bank,
        similarity_threshold=similarity_threshold,
        low_similarity_count_to_fail=low_similarity_count_to_fail,
    )
    return IdentityGateResult(
        passed=comparison.passed,
        mode="DUAL_REPAIR_CROSS_IDENTITY",
        reasons=(() if comparison.passed else (comparison.reason,)),
        comparisons=(comparison,),
    )


def evaluate_single_repair_without_anchor(
    *,
    raw_audit_status: str,
    trusted_quality_passed: bool,
    macro_smoothness_passed: bool,
    neighbor_ownership_passed: bool,
    interlayer_safe_band_passed: bool,
    trusted_union_coverage: float,
    required_coverage: float = 0.80,
) -> SingleRepairGateResult:
    """ANCHOR-untrusted and only one repair: require the track to self-prove NORMAL."""

    reasons = []
    if _status_name(raw_audit_status) != "NORMAL":
        reasons.append("RAW_AUDIT_NOT_NORMAL")
    if not bool(trusted_quality_passed):
        reasons.append("TRUSTED_QUALITY_FAILED")
    if not bool(macro_smoothness_passed):
        reasons.append("MACRO_SMOOTHNESS_FAILED")
    if not bool(neighbor_ownership_passed):
        reasons.append("NEIGHBOR_OWNERSHIP_FAILED")
    if not bool(interlayer_safe_band_passed):
        reasons.append("INTERLAYER_SAFE_BAND_FAILED")
    if float(trusted_union_coverage) < float(required_coverage):
        reasons.append("INSUFFICIENT_TRUSTED_COVERAGE")
    return SingleRepairGateResult(not reasons, tuple(reasons))


def anchor_identity_is_trustworthy(
    *,
    original_audit_status: str,
    anchor_bank: RepresentativeWaveformBank | None,
) -> bool:
    """Minimal branch decision: SEED_ERROR or bad anchor bank => no anchor identity."""

    return bool(
        _status_name(original_audit_status) != "SEED_ERROR"
        and anchor_bank is not None
        and anchor_bank.representative
    )


def required_repair_sides(original_audit) -> tuple[str, ...]:
    """Return every side that the existing audit already classifies as broken.

    This deliberately ignores current total coverage.  If a shot entered
    TRACKING_BREAK because RIGHT is broken, RIGHT remains a required repair even
    when the preserved ORIGINAL coverage happens to exceed 0.80.
    """

    required = []
    for name in ("left", "right"):
        side = getattr(original_audit, name, None)
        if side is None or not bool(getattr(side, "present", False)):
            continue
        status = getattr(side, "status", "")
        status_name = getattr(status, "value", status)
        if str(status_name) == "TRACKING_BREAK":
            required.append(name.upper())
    return tuple(required)


def bank_from_trusted_segment(
    flat_gather: np.ndarray,
    segment,
    *,
    dt: float,
    config,
    tracker_options: dict | None = None,
    label: str | None = None,
) -> RepresentativeWaveformBank | None:
    """Adapter for the current TrustedSegment structure."""

    tracking = segment.tracking
    options = dict(tracker_options or {})
    return select_representative_waveform_bank(
        flat_gather,
        pick_sample=tracking.pick_sample,
        eligible_mask=segment.trusted_mask,
        dt=dt,
        prediction_error=tracking.prediction_error,
        window_receivers=int(config.quality_window_receivers),
        low_similarity_threshold=float(config.low_similarity_threshold),
        low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
        coherence_half_window_time=float(options.get("coherence_half_window_time", 0.040)),
        allow_short=False,
        label=(str(label) if label is not None else str(segment.source_type)),
    )


def bank_from_anchor(
    flat_gather: np.ndarray,
    seed_result,
    *,
    dt: float,
    config,
    tracker_options: dict | None = None,
    prediction_error: np.ndarray | None = None,
    label: str = "ANCHOR",
) -> RepresentativeWaveformBank | None:
    """Adapter for the current SeedSearchResult fixed anchor arrays."""

    options = dict(tracker_options or {})
    return select_representative_waveform_bank(
        flat_gather,
        pick_sample=seed_result.anchor_pick_sample,
        eligible_mask=seed_result.anchor_receiver_mask,
        dt=dt,
        prediction_error=prediction_error,
        window_receivers=int(config.quality_window_receivers),
        low_similarity_threshold=float(config.low_similarity_threshold),
        low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
        coherence_half_window_time=float(options.get("coherence_half_window_time", 0.040)),
        allow_short=True,
        label=str(label),
    )


# -----------------------------------------------------------------------------
# Short EVENT first, seed second
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortPilotPath:
    """One pilot path emitted by one positive-peak hypothesis."""

    source_id: int
    origin_receiver: int
    seed_time: float
    seed_sample: int
    pick_time: np.ndarray
    success_mask: np.ndarray
    tested_mask: np.ndarray
    rank_hint: tuple = ()


@dataclass(frozen=True)
class ShortPathMatch:
    matched: bool
    overlap_count: int
    median_abs_time_difference: float
    p90_abs_time_difference: float


@dataclass(frozen=True)
class ShortEventCluster:
    member_ids: tuple[int, ...]
    consensus_pick_time: np.ndarray
    support_count: np.ndarray
    receiver_mask: np.ndarray
    tested_mask: np.ndarray
    coverage: float


def compare_short_pilot_paths(
    first: ShortPilotPath,
    second: ShortPilotPath,
    *,
    quality_window_receivers: int = 8,
    severe_bad_prediction_error_time: float = 0.015,
) -> ShortPathMatch:
    """Decide whether two pilot paths describe the same short 2-D event."""

    a = np.asarray(first.pick_time, dtype=float)
    b = np.asarray(second.pick_time, dtype=float)
    sa = np.asarray(first.success_mask, dtype=bool)
    sb = np.asarray(second.success_mask, dtype=bool)
    if a.shape != b.shape or sa.shape != a.shape or sb.shape != a.shape:
        raise ValueError("short pilot paths must share one receiver axis")

    common = sa & sb & np.isfinite(a) & np.isfinite(b)
    count = int(np.count_nonzero(common))
    minimum = int(quality_window_receivers)
    if count < minimum:
        return ShortPathMatch(False, count, np.inf, np.inf)

    difference = np.abs(a[common] - b[common])
    median = float(np.median(difference))
    p90 = float(np.percentile(difference, 90.0))
    limit = float(severe_bad_prediction_error_time)
    return ShortPathMatch(
        matched=bool(median <= limit and p90 <= 2.0 * limit),
        overlap_count=count,
        median_abs_time_difference=median,
        p90_abs_time_difference=p90,
    )


def _consensus_for_paths(paths: Sequence[ShortPilotPath]) -> ShortEventCluster:
    if not paths:
        raise ValueError("paths must be non-empty")
    nreceiver = np.asarray(paths[0].pick_time).size
    values = np.full((len(paths), nreceiver), np.nan, dtype=float)
    tested_union = np.zeros(nreceiver, dtype=bool)
    for row, path in enumerate(paths):
        pick = np.asarray(path.pick_time, dtype=float)
        success = np.asarray(path.success_mask, dtype=bool)
        if pick.shape != (nreceiver,) or success.shape != (nreceiver,):
            raise ValueError("all short pilot paths must share one receiver axis")
        tested = np.asarray(path.tested_mask, dtype=bool)
        if tested.shape != (nreceiver,):
            raise ValueError("all tested masks must share one receiver axis")
        tested_union |= tested
        valid = success & np.isfinite(pick)
        values[row, valid] = pick[valid]

    consensus = np.full(nreceiver, np.nan, dtype=float)
    support = np.sum(np.isfinite(values), axis=0).astype(int)
    for receiver in np.flatnonzero(support > 0):
        consensus[receiver] = float(np.median(values[:, receiver][np.isfinite(values[:, receiver])]))
    mask = np.isfinite(consensus)
    denominator = int(np.count_nonzero(tested_union))
    coverage = float(np.count_nonzero(mask & tested_union) / denominator) if denominator else 0.0
    return ShortEventCluster(
        member_ids=tuple(int(path.source_id) for path in paths),
        consensus_pick_time=consensus,
        support_count=support,
        receiver_mask=mask,
        tested_mask=tested_union,
        coverage=coverage,
    )


def _path_matches_consensus(
    path: ShortPilotPath,
    consensus: ShortEventCluster,
    *,
    quality_window_receivers: int,
    severe_bad_prediction_error_time: float,
) -> ShortPathMatch:
    pseudo = ShortPilotPath(
        source_id=-1,
        origin_receiver=-1,
        seed_time=np.nan,
        seed_sample=-1,
        pick_time=consensus.consensus_pick_time,
        success_mask=consensus.receiver_mask,
        tested_mask=consensus.receiver_mask,
    )
    return compare_short_pilot_paths(
        path,
        pseudo,
        quality_window_receivers=quality_window_receivers,
        severe_bad_prediction_error_time=severe_bad_prediction_error_time,
    )


def cluster_short_pilot_paths(
    paths: Sequence[ShortPilotPath],
    *,
    quality_window_receivers: int = 8,
    severe_bad_prediction_error_time: float = 0.015,
) -> tuple[ShortEventCluster, ...]:
    """Cluster pilots into short EVENT hypotheses rather than counting seed origins.

    Greedy clustering is followed by consensus pruning.  A path is admitted to
    a cluster only when it matches the current receiver-time consensus over at
    least one existing quality-window scale.  This avoids the old logic where
    merely two nearby seed receivers could certify each other.
    """

    remaining = list(paths)
    # Prefer already-strong pilot hypotheses only to reduce order sensitivity;
    # the cluster identity itself is still path based.
    remaining.sort(key=lambda item: tuple(item.rank_hint), reverse=True)
    clusters: list[list[ShortPilotPath]] = []

    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        changed = True
        while changed:
            changed = False
            consensus = _consensus_for_paths(members)
            best_index = None
            best_metric = None
            for index, candidate in enumerate(remaining):
                match = _path_matches_consensus(
                    candidate,
                    consensus,
                    quality_window_receivers=quality_window_receivers,
                    severe_bad_prediction_error_time=severe_bad_prediction_error_time,
                )
                if not match.matched:
                    continue
                metric = (
                    -match.median_abs_time_difference,
                    -match.p90_abs_time_difference,
                    match.overlap_count,
                )
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    best_index = index
            if best_index is not None:
                members.append(remaining.pop(best_index))
                changed = True

        # Final consensus and pruning: every retained member must still match
        # the cluster after all additions.
        while len(members) > 1:
            consensus = _consensus_for_paths(members)
            matches = [
                _path_matches_consensus(
                    member,
                    consensus,
                    quality_window_receivers=quality_window_receivers,
                    severe_bad_prediction_error_time=severe_bad_prediction_error_time,
                )
                for member in members
            ]
            bad = [i for i, match in enumerate(matches) if not match.matched]
            if not bad:
                break
            # Drop the worst mismatch, not the whole event.
            worst = max(
                bad,
                key=lambda i: (
                    matches[i].median_abs_time_difference,
                    matches[i].p90_abs_time_difference,
                    -matches[i].overlap_count,
                ),
            )
            remaining.append(members.pop(worst))

        clusters.append(members)

    result = [_consensus_for_paths(members) for members in clusters if members]
    result.sort(
        key=lambda item: (
            int(np.count_nonzero(item.receiver_mask)),
            int(np.max(item.support_count)) if item.support_count.size else 0,
        ),
        reverse=True,
    )
    return tuple(result)


def consensus_pick_samples(
    cluster: ShortEventCluster,
    *,
    dt: float,
    t0: float = 0.0,
) -> np.ndarray:
    """Convert a short-event consensus path to sample indices for bank selection."""

    pick_time = np.asarray(cluster.consensus_pick_time, dtype=float)
    samples = np.full(pick_time.shape, -1, dtype=int)
    finite = np.isfinite(pick_time)
    samples[finite] = np.rint((pick_time[finite] - float(t0)) / float(dt)).astype(int)
    return samples


def bank_from_short_event_cluster(
    flat_gather: np.ndarray,
    cluster: ShortEventCluster,
    *,
    dt: float,
    t0: float = 0.0,
    config,
    tracker_options: dict | None = None,
    label: str = "SHORT_EVENT",
) -> RepresentativeWaveformBank | None:
    """Select the best local waveform bank from a consensus short event."""

    samples = consensus_pick_samples(cluster, dt=dt, t0=t0)
    options = dict(tracker_options or {})
    return select_representative_waveform_bank(
        flat_gather,
        pick_sample=samples,
        eligible_mask=cluster.receiver_mask,
        dt=dt,
        prediction_error=None,
        window_receivers=int(config.quality_window_receivers),
        low_similarity_threshold=float(config.low_similarity_threshold),
        low_similarity_count_to_fail=int(config.low_similarity_count_to_fail),
        coherence_half_window_time=float(options.get("coherence_half_window_time", 0.040)),
        allow_short=False,
        label=str(label),
    )


def select_seed_from_short_event(
    flat_gather: np.ndarray,
    *,
    cluster: ShortEventCluster,
    member_paths: Sequence[ShortPilotPath],
    event_bank: RepresentativeWaveformBank,
    dt: float,
    coherence_half_window_time: float = 0.040,
) -> ShortEventSeedSelection | None:
    """Choose a real positive-peak seed only *after* the short event is accepted.

    The seed must be one of the original pilot hypotheses; the consensus path
    itself is never snapped into a synthetic seed.  Preference is given to an
    origin inside the representative event window, then to waveform agreement
    with the event bank, then to the candidate's existing short-event rank.
    """

    members = set(int(value) for value in cluster.member_ids)
    bank_receivers = set(int(value) for value in event_bank.receivers)
    center = float(np.median(event_bank.receivers))
    half_samples = max(1, int(round(float(coherence_half_window_time) / float(dt))))
    best = None
    best_rank = None
    for path in member_paths:
        if int(path.source_id) not in members:
            continue
        receiver = int(path.origin_receiver)
        sample = int(path.seed_sample)
        if not (0 <= receiver < np.asarray(flat_gather).shape[0]):
            continue
        waveform = _unit_waveform(np.asarray(flat_gather)[receiver], sample, half_samples)
        if waveform is None:
            continue
        correlations = event_bank.waveforms @ waveform
        median_bank = float(np.median(correlations))
        inside = receiver in bank_receivers
        rank = (
            int(inside),
            median_bank,
            -abs(float(receiver) - center),
            *tuple(path.rank_hint),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = ShortEventSeedSelection(
                source_id=int(path.source_id),
                seed_receiver=receiver,
                seed_time=float(path.seed_time),
                seed_sample=sample,
                median_bank_zncc=median_bank,
                inside_representative_window=bool(inside),
                rank=tuple(rank),
            )
    return best


__all__ = [
    "RepresentativeWaveformBank",
    "WaveformIdentityResult",
    "IdentityGateResult",
    "SingleRepairGateResult",
    "ShortEventSeedSelection",
    "ShortPilotPath",
    "ShortPathMatch",
    "ShortEventCluster",
    "select_representative_waveform_bank",
    "compare_waveform_banks",
    "evaluate_anchor_backed_repair_identity",
    "evaluate_dual_repair_identity_without_anchor",
    "evaluate_single_repair_without_anchor",
    "anchor_identity_is_trustworthy",
    "required_repair_sides",
    "bank_from_trusted_segment",
    "bank_from_anchor",
    "compare_short_pilot_paths",
    "cluster_short_pilot_paths",
    "consensus_pick_samples",
    "bank_from_short_event_cluster",
    "select_seed_from_short_event",
]
