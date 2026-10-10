"""Quality-first exact second-order Top-K DP for flattened reflection tracking.

This module is solver-only.  It is intentionally independent of
``SparseFlatTrackingResult`` so the production bootstrap can keep its existing
public API and switch between the legacy beam tracker and this exact global
solver with one option.

Revision 2 addresses four failure modes found during review of the first patch:

* the candidate pool is broad but bounded instead of containing every positive
  local maximum;
* anchor-template similarity tolerates a small sample lag;
* Top-K consensus uses path-length-normalized scores and weighted clustering,
  not a cluster forced around the best path;
* a common anchor-aware quality score is provided so a legacy repair result is
  a challenger to a global result rather than an unconditional replacement.

The exactness claim applies to the chosen candidate graph: for every
second-order state ``(older_candidate, previous_candidate)`` the solver keeps
its exact K best histories.  There is no global beam prune across different
states.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, MutableSequence, Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline


# Keep values aligned with tracking.flat_event.TransitionKind without importing
# the module here (which would create a circular import).
TRANSITION_SEED = 0
TRANSITION_ANCHOR = 1
TRANSITION_NORMAL = 2
TRANSITION_RESCUE = 3
TRANSITION_GAP = 4


@dataclass(frozen=True)
class GlobalDPPath:
    """One complete left+anchor+right path hypothesis."""

    selected_items: tuple[tuple[int, int], ...]
    score: float
    template_score: float
    neighbor_score: float
    curvature_penalty: float
    gap_penalty: float
    escape_penalty: float
    guide_penalty: float
    selected_count: int

    def as_dict(self) -> dict[int, int]:
        return dict(self.selected_items)


@dataclass(frozen=True)
class GlobalDPSolution:
    """Global solver result translated to the legacy continuation vocabulary."""

    selected: Mapping[int, int]
    left_selected: Mapping[int, int]
    right_selected: Mapping[int, int]
    top_paths: tuple[GlobalDPPath, ...]

    best_score: float
    second_score: float
    score_margin: float

    template_correlation: np.ndarray
    consensus_fraction: np.ndarray
    consensus_spread: np.ndarray
    guide_time: np.ndarray

    ownership_escape_used_mask: np.ndarray
    skipped_valid_receiver_mask: np.ndarray
    selected_transition_kind: np.ndarray

    stop_reason_left: str
    stop_reason_right: str
    stop_receiver_left: int
    stop_receiver_right: int

    transition_count_left: int
    transition_count_right: int
    correlation_count_left: int
    correlation_count_right: int
    active_state_max: int
    escape_candidate_count: int
    escape_transition_count: int
    candidate_count_after_global_pool: np.ndarray

    runtime_seconds: float

    @property
    def left_diagnostic(self) -> dict[str, object]:
        nr = self.template_correlation.size
        transition = np.full(nr, -1, dtype=np.int8)
        gap = np.zeros(nr, dtype=bool)
        indices = np.asarray(list(self.left_selected), dtype=int)
        if indices.size:
            transition[indices] = self.selected_transition_kind[indices]
            hi = int(np.max(indices))
            gap[: hi + 1] = self.skipped_valid_receiver_mask[: hi + 1]
        return {
            "attempts": 0,
            "successes": 0,
            "candidates": 0,
            "rescue_transitions": 0,
            "rescue_mask": np.zeros(nr, dtype=bool),
            "gap_mask": gap,
            "transition_kind": transition,
            "stop_receiver": int(self.stop_receiver_left),
            "escape_attempts": int(self.escape_transition_count),
            "escape_successes": int(
                np.count_nonzero(
                    self.ownership_escape_used_mask
                    & np.isin(np.arange(nr), indices)
                )
            ),
            "escape_candidates": int(self.escape_candidate_count),
            "guide_candidates": 0,
            "guide_updates": 0,
            "guide_freezes": 0,
            "guide_protected_transitions": 0,
        }

    @property
    def right_diagnostic(self) -> dict[str, object]:
        nr = self.template_correlation.size
        transition = np.full(nr, -1, dtype=np.int8)
        gap = np.zeros(nr, dtype=bool)
        indices = np.asarray(list(self.right_selected), dtype=int)
        if indices.size:
            transition[indices] = self.selected_transition_kind[indices]
            lo = int(np.min(indices))
            gap[lo:] = self.skipped_valid_receiver_mask[lo:]
        return {
            "attempts": 0,
            "successes": 0,
            "candidates": 0,
            "rescue_transitions": 0,
            "rescue_mask": np.zeros(nr, dtype=bool),
            "gap_mask": gap,
            "transition_kind": transition,
            "stop_receiver": int(self.stop_receiver_right),
            "escape_attempts": int(self.escape_transition_count),
            "escape_successes": int(
                np.count_nonzero(
                    self.ownership_escape_used_mask
                    & np.isin(np.arange(nr), indices)
                )
            ),
            "escape_candidates": int(self.escape_candidate_count),
            "guide_candidates": 0,
            "guide_updates": 0,
            "guide_freezes": 0,
            "guide_protected_transitions": 0,
        }


@dataclass(frozen=True)
class TrackingChallengeScore:
    """Common score used to compare a global incumbent and a repair challenger."""

    total: float
    coverage: float
    template_similarity: float
    neighbor_similarity: float
    curvature_penalty: float
    gap_fraction: float


@dataclass(frozen=True)
class TrackingChallengeDecision:
    use_challenger: bool
    incumbent: TrackingChallengeScore
    challenger: TrackingChallengeScore
    improvement: float


@dataclass(frozen=True)
class _Node:
    node_id: int
    receiver: int
    candidate_index: int
    time: float
    sample: int
    waveform: np.ndarray
    template_corr: float
    envelope_reward: float
    is_escape: bool
    is_anchor: bool


@dataclass(frozen=True)
class _PairFeature:
    slope: float
    neighbor_corr: float
    gap_penalty: float
    increment: float
    escape_penalty: float
    guide_penalty: float


@dataclass(frozen=True)
class _Entry:
    score: float
    prev_key: tuple[int, int] | None
    prev_rank: int
    template_score: float
    neighbor_score: float
    curvature_penalty: float
    gap_penalty: float
    escape_penalty: float
    guide_penalty: float
    selected_count: int


@dataclass(frozen=True)
class _DirectionResult:
    paths: tuple[GlobalDPPath, ...]
    transition_count: int
    correlation_count: int
    active_state_max: int
    escape_transition_count: int


DEFAULT_GLOBAL_DP: dict[str, object] = {
    "enabled": False,
    "pass1_top_k": 5,
    "pass2_top_k": 2,
    # Backward-compatible fallback for callers that still inspect top_k.
    "top_k": 5,
    # Bounded multi-source candidate pool.  These are quotas, not hard
    # partitions; their union is capped by candidate_max_per_trace.
    "candidate_max_per_trace": 16,
    "candidate_keep_regular": 6,
    "candidate_keep_template": 8,
    "candidate_keep_axis": 6,
    "candidate_keep_amplitude": 2,
    "candidate_axis_half_width_time": 0.080,
    "pass2_max_candidates_per_trace": 8,
    "pass2_candidate_half_width_time": 0.035,
    "guide_added_candidates_per_trace": 4,
    "guide_candidate_half_width_time": 0.025,
    # Reflector identity.
    "template_top_m": 3,
    "template_lag_samples": 2,
    "template_weight": 1.00,
    # Local coherence is deliberately weaker than anchor identity.
    "neighbor_weight": 0.45,
    "amplitude_weight": 0.00,
    "coverage_reward": 0.12,
    # Second-order trajectory regularization.
    "curvature_weight": 0.40,
    "curvature_scale_ms_per_100m": 25.0,
    "absolute_max_slope_ms_per_100m": 300.0,
    "hard_min_neighbor_corr": -1.0,
    # Short gaps and ownership escape compete from the beginning.
    "max_gap_receivers": 3,
    "gap_trigger_template_corr": 0.45,
    "valid_gap_penalty": 0.80,
    "invalid_gap_penalty": 0.20,
    "tail_miss_penalty": 0.04,
    "escape_penalty": 0.60,
    # Pass-2 full-gather guide.
    "second_pass_enabled": True,
    "guide_weight": 0.75,
    "guide_scale_time": 0.015,
    # Consensus uses normalized complete-path scores and weighted clustering.
    "consensus_radius_time": 0.008,
    "consensus_min_fraction": 0.60,
    "consensus_max_spread_time": 0.020,
    "path_weight_temperature": 0.05,
    # Robust spline guide.
    "guide_smoothing_time": 0.010,
    "guide_huber_delta_time": 0.020,
    "guide_fit_iterations": 4,
    "anchor_guide_weight": 8.0,
    # Global-vs-repair challenger selection.
    "repair_challenge_enabled": True,
    "repair_challenge_min_improvement": 0.05,
    "repair_coverage_weight": 0.50,
    "repair_gap_weight": 0.35,
}


def normalise_global_dp_options(options: Mapping[str, object] | None) -> dict[str, object]:
    """Return validated solver options with quality-first defaults."""

    provided = dict(options or {})
    cfg = dict(DEFAULT_GLOBAL_DP)
    cfg.update(provided)
    if "top_k" in provided:
        if "pass1_top_k" not in provided:
            cfg["pass1_top_k"] = provided["top_k"]
        if "pass2_top_k" not in provided:
            cfg["pass2_top_k"] = provided["top_k"]

    positive_ints = (
        "top_k",
        "pass1_top_k",
        "pass2_top_k",
        "candidate_max_per_trace",
        "candidate_keep_regular",
        "candidate_keep_template",
        "candidate_keep_axis",
        "candidate_keep_amplitude",
        "pass2_max_candidates_per_trace",
        "guide_added_candidates_per_trace",
        "template_top_m",
        "guide_fit_iterations",
    )
    nonnegative_ints = ("max_gap_receivers", "template_lag_samples")
    for name in positive_ints:
        value = cfg[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"global_dp.{name} must be a positive integer")
        cfg[name] = int(value)
    for name in nonnegative_ints:
        value = cfg[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"global_dp.{name} must be a non-negative integer")
        cfg[name] = int(value)

    nonnegative = (
        "candidate_axis_half_width_time",
        "pass2_candidate_half_width_time",
        "guide_candidate_half_width_time",
        "gap_trigger_template_corr",
        "template_weight",
        "neighbor_weight",
        "amplitude_weight",
        "coverage_reward",
        "curvature_weight",
        "curvature_scale_ms_per_100m",
        "absolute_max_slope_ms_per_100m",
        "valid_gap_penalty",
        "invalid_gap_penalty",
        "tail_miss_penalty",
        "escape_penalty",
        "guide_weight",
        "guide_scale_time",
        "consensus_radius_time",
        "consensus_max_spread_time",
        "guide_smoothing_time",
        "guide_huber_delta_time",
        "anchor_guide_weight",
        "repair_challenge_min_improvement",
        "repair_coverage_weight",
        "repair_gap_weight",
    )
    for name in nonnegative:
        value = float(cfg[name])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"global_dp.{name} must be finite and non-negative")
        cfg[name] = value

    for name in (
        "curvature_scale_ms_per_100m",
        "absolute_max_slope_ms_per_100m",
        "guide_scale_time",
        "path_weight_temperature",
    ):
        value = float(cfg[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"global_dp.{name} must be finite and positive")
        cfg[name] = value

    cfg["hard_min_neighbor_corr"] = float(cfg["hard_min_neighbor_corr"])
    if not -1.0 <= cfg["hard_min_neighbor_corr"] <= 1.0:
        raise ValueError("global_dp.hard_min_neighbor_corr must be in [-1, 1]")
    cfg["consensus_min_fraction"] = float(cfg["consensus_min_fraction"])
    if not 0.0 <= cfg["consensus_min_fraction"] <= 1.0:
        raise ValueError("global_dp.consensus_min_fraction must be in [0, 1]")
    if not 0.0 <= float(cfg["gap_trigger_template_corr"]) <= 1.0:
        raise ValueError("global_dp.gap_trigger_template_corr must be in [0, 1]")
    if int(cfg["pass2_max_candidates_per_trace"]) > int(
        cfg["candidate_max_per_trace"]
    ):
        raise ValueError(
            "global_dp.pass2_max_candidates_per_trace must not exceed "
            "candidate_max_per_trace"
        )

    for name in ("enabled", "second_pass_enabled", "repair_challenge_enabled"):
        if not isinstance(cfg[name], (bool, np.bool_)):
            raise ValueError(f"global_dp.{name} must be boolean")
        cfg[name] = bool(cfg[name])
    return cfg


def _huber(value: float) -> float:
    value = abs(float(value))
    return 0.5 * value * value if value <= 1.0 else value - 0.5


def _normalise_waveform(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return values / norm


def _candidate_waveform(trace: np.ndarray, sample: int, half: int) -> np.ndarray | None:
    sample = int(sample)
    half = int(half)
    if sample - half < 0 or sample + half >= trace.size:
        return None
    return _normalise_waveform(trace[sample - half : sample + half + 1])


def _positive_local_maxima(trace: np.ndarray) -> np.ndarray:
    if trace.size < 3:
        return np.empty(0, dtype=int)
    samples = np.arange(1, trace.size - 1, dtype=int)
    values = trace[samples]
    keep = (
        np.isfinite(values)
        & (values > 0.0)
        & (values >= trace[samples - 1])
        & (values > trace[samples + 1])
    )
    return samples[keep]


def _lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Maximum normalized correlation over a small integer lag range."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    best = -np.inf
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag >= 0:
            aa = a[lag:n]
            bb = b[: n - lag]
        else:
            aa = a[: n + lag]
            bb = b[-lag:n]
        if aa.size < 3:
            continue
        aa_n = _normalise_waveform(aa)
        bb_n = _normalise_waveform(bb)
        if aa_n is None or bb_n is None:
            continue
        value = float(np.dot(aa_n, bb_n))
        if value > best:
            best = value
    return float(best) if np.isfinite(best) else np.nan


def _template_bank(
    candidates: Sequence[Sequence[Any]],
    anchor_mask: np.ndarray,
    seed_receiver: int,
) -> tuple[np.ndarray, ...]:
    bank: list[np.ndarray] = []
    for receiver in np.flatnonzero(anchor_mask):
        row = candidates[int(receiver)]
        if row:
            bank.append(np.asarray(row[0].waveform, dtype=float))
    if not bank:
        row = candidates[int(seed_receiver)]
        if not row:
            raise ValueError("global DP seed has no candidate waveform")
        bank.append(np.asarray(row[0].waveform, dtype=float))
    return tuple(bank)


def _template_similarity(
    waveform: np.ndarray,
    bank: Sequence[np.ndarray],
    top_m: int,
    max_lag: int,
) -> float:
    values = np.asarray(
        [_lagged_corr(waveform, template, max_lag) for template in bank], dtype=float
    )
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0
    count = min(int(top_m), int(values.size))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def _make_candidate_like(
    prototype: Any,
    *,
    sample: int,
    time: float,
    amplitude: float,
    envelope: float,
    waveform: np.ndarray,
) -> Any:
    kwargs: dict[str, object] = {
        "sample": int(sample),
        "time": float(time),
        "amplitude": float(amplitude),
        "envelope": float(envelope),
        "waveform": np.asarray(waveform, dtype=float),
    }
    if hasattr(prototype, "polarity"):
        kwargs["polarity"] = 1
    if hasattr(prototype, "prominence"):
        kwargs["prominence"] = 0.0
    return type(prototype)(**kwargs)


def _anchor_axis_prediction(
    candidates: Sequence[Sequence[Any]],
    receiver_x: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    nr = receiver_x.size
    result = np.full(nr, np.nan, dtype=float)
    indices = np.flatnonzero(anchor_mask)
    if not indices.size:
        return result
    times = np.asarray(
        [float(candidates[int(receiver)][0].time) for receiver in indices], dtype=float
    )
    xx = np.asarray(receiver_x[indices], dtype=float)
    if indices.size == 1 or np.ptp(xx) <= np.finfo(float).eps:
        result[:] = float(np.median(times))
        return result
    x0 = float(np.mean(xx))
    scale = float(np.max(np.abs(xx - x0)))
    try:
        coef = np.polyfit((xx - x0) / scale, times, deg=1)
        result[:] = np.polyval(coef, (receiver_x - x0) / scale)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        result[:] = float(np.median(times))
    return result


def prepare_global_candidate_pool(
    candidates: MutableSequence[MutableSequence[Any]],
    *,
    traces: np.ndarray,
    candidate_envelope: np.ndarray,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    seed_receiver: int,
    anchor_mask: np.ndarray,
    state_valid: np.ndarray | None,
    escape_state_valid: np.ndarray | None,
    dt: float,
    t0: float,
    half_window_samples: int,
    config: Mapping[str, object],
    guide_time: np.ndarray | None = None,
    guide_only: bool = False,
) -> np.ndarray:
    """Mutate candidate rows into a broad but bounded quality-first pool.

    Pass 1 uses a union of existing regular peaks, template-similar weak peaks,
    peaks near the immutable anchor-axis extrapolation, and a small amplitude
    quota.  Pass 2 only appends a few weak peaks near the full-gather guide so
    pass-1 candidate indices remain valid if pass 2 has to fall back.
    """

    cfg = normalise_global_dp_options(config)
    traces = np.asarray(traces, dtype=float)
    envelope = np.asarray(candidate_envelope, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool)
    nr, nt = traces.shape
    if len(candidates) != nr or envelope.shape != traces.shape:
        raise ValueError("global candidate-pool arrays are inconsistent")

    prototype = None
    for row in candidates:
        if row:
            prototype = row[0]
            break
    if prototype is None:
        raise ValueError("global candidate pool has no prototype candidate")

    bank = _template_bank(candidates, anchor, int(seed_receiver))
    axis = _anchor_axis_prediction(candidates, x, anchor)
    allowed = escape_state_valid if escape_state_valid is not None else state_valid
    top_m = int(cfg["template_top_m"])
    lag = int(cfg["template_lag_samples"])

    max_count = int(cfg["candidate_max_per_trace"])
    regular_quota = int(cfg["candidate_keep_regular"])
    template_quota = int(cfg["candidate_keep_template"])
    axis_quota = int(cfg["candidate_keep_axis"])
    amplitude_quota = int(cfg["candidate_keep_amplitude"])
    axis_half = float(cfg["candidate_axis_half_width_time"])
    guide_quota = int(cfg["guide_added_candidates_per_trace"])
    guide_half = float(cfg["guide_candidate_half_width_time"])

    for receiver in range(nr):
        if not valid[receiver] or anchor[receiver] or receiver == int(seed_receiver):
            continue
        trace = traces[receiver]
        peaks = _positive_local_maxima(trace)
        if allowed is not None and peaks.size:
            peaks = peaks[np.asarray(allowed[receiver, peaks], dtype=bool)]
        if not peaks.size:
            continue

        existing_by_sample = {
            int(candidate.sample): candidate for candidate in candidates[receiver]
        }
        waveforms: dict[int, np.ndarray] = {}
        template_score: dict[int, float] = {}
        for sample in peaks:
            sample = int(sample)
            existing = existing_by_sample.get(sample)
            waveform = (
                np.asarray(existing.waveform, dtype=float)
                if existing is not None
                else _candidate_waveform(trace, sample, half_window_samples)
            )
            if waveform is None:
                continue
            waveforms[sample] = waveform
            template_score[sample] = _template_similarity(
                waveform, bank, top_m, lag
            )
        if not waveforms:
            continue

        available_samples = np.asarray(sorted(waveforms), dtype=int)
        if guide_only:
            if guide_time is None or not np.isfinite(guide_time[receiver]):
                continue
            order = sorted(
                available_samples,
                key=lambda sample: (
                    abs(float(t0 + sample * dt) - float(guide_time[receiver])),
                    -template_score[int(sample)],
                    -float(envelope[receiver, int(sample)]),
                ),
            )
            added = 0
            for sample in order:
                sample = int(sample)
                if sample in existing_by_sample:
                    continue
                error = abs(float(t0 + sample * dt) - float(guide_time[receiver]))
                if error > guide_half:
                    continue
                candidates[receiver].append(
                    _make_candidate_like(
                        prototype,
                        sample=sample,
                        time=float(t0 + sample * dt),
                        amplitude=float(trace[sample]),
                        envelope=float(envelope[receiver, sample]),
                        waveform=waveforms[sample],
                    )
                )
                existing_by_sample[sample] = candidates[receiver][-1]
                added += 1
                if added >= guide_quota:
                    break
            continue

        # Rank each source independently, then add in a fixed round-robin order
        # so no single feature can monopolize the bounded pool.
        existing_samples = list(existing_by_sample)
        regular_rank = sorted(
            existing_samples,
            key=lambda sample: (
                -float(getattr(existing_by_sample[sample], "prominence", 0.0)),
                -float(getattr(existing_by_sample[sample], "envelope", 0.0)),
            ),
        )[:regular_quota]
        template_rank = sorted(
            available_samples,
            key=lambda sample: (
                -template_score[int(sample)],
                -float(envelope[receiver, int(sample)]),
            ),
        )[:template_quota]
        if np.isfinite(axis[receiver]):
            axis_rank_all = sorted(
                available_samples,
                key=lambda sample: (
                    abs(float(t0 + int(sample) * dt) - float(axis[receiver])),
                    -template_score[int(sample)],
                ),
            )
            axis_rank = [
                int(sample)
                for sample in axis_rank_all
                if abs(float(t0 + int(sample) * dt) - float(axis[receiver]))
                <= axis_half
            ][:axis_quota]
        else:
            axis_rank = []
        amplitude_rank = sorted(
            available_samples,
            key=lambda sample: -float(envelope[receiver, int(sample)]),
        )[:amplitude_quota]

        sources = [regular_rank, template_rank, axis_rank, amplitude_rank]
        selected: list[int] = []
        seen: set[int] = set()
        depth = 0
        while len(selected) < max_count:
            added_any = False
            for source in sources:
                if depth >= len(source):
                    continue
                sample = int(source[depth])
                if sample not in seen and sample in waveforms:
                    selected.append(sample)
                    seen.add(sample)
                    added_any = True
                    if len(selected) >= max_count:
                        break
            if not added_any and all(depth >= len(source) for source in sources):
                break
            depth += 1

        # If quotas overlap heavily, fill the remainder by a conservative
        # composite score: template identity first, then anchor-axis proximity.
        if len(selected) < max_count:
            remainder = [int(sample) for sample in available_samples if int(sample) not in seen]
            remainder.sort(
                key=lambda sample: (
                    -template_score[sample],
                    abs(float(t0 + sample * dt) - float(axis[receiver]))
                    if np.isfinite(axis[receiver])
                    else 0.0,
                    -float(envelope[receiver, sample]),
                )
            )
            selected.extend(remainder[: max_count - len(selected)])

        new_row: list[Any] = []
        for sample in selected:
            candidate = existing_by_sample.get(sample)
            if candidate is None:
                candidate = _make_candidate_like(
                    prototype,
                    sample=sample,
                    time=float(t0 + sample * dt),
                    amplitude=float(trace[sample]),
                    envelope=float(envelope[receiver, sample]),
                    waveform=waveforms[sample],
                )
            new_row.append(candidate)
        # Sorting by time is not required by the global solver, but deterministic
        # ordering makes diagnostics and selected candidate indices reproducible.
        new_row.sort(key=lambda item: float(item.time))
        candidates[receiver][:] = new_row

    return np.asarray([len(row) for row in candidates], dtype=int)


def _build_pass2_allowed_candidate_indices(
    candidates: Sequence[Sequence[Any]],
    pass1_paths: Sequence[GlobalDPPath],
    guide_time: np.ndarray,
    anchor_mask: np.ndarray,
    seed_receiver: int,
    config: Mapping[str, object],
) -> list[set[int] | None]:
    """Keep pass-1 hypotheses and a small set of guide-near candidates."""

    cap = int(config["pass2_max_candidates_per_trace"])
    half = float(config["pass2_candidate_half_width_time"])
    used_by_pass1: list[set[int]] = [set() for _ in candidates]
    for path in pass1_paths:
        for receiver, candidate_index in path.selected_items:
            used_by_pass1[int(receiver)].add(int(candidate_index))

    result: list[set[int] | None] = []
    for receiver, row in enumerate(candidates):
        if not row:
            result.append(set())
            continue
        if anchor_mask[receiver] or receiver == int(seed_receiver):
            result.append(set(range(len(row))))
            continue

        selected = set(used_by_pass1[receiver])
        guide = float(guide_time[receiver])
        if np.isfinite(guide):
            ranked = sorted(
                range(len(row)),
                key=lambda index: (
                    abs(float(row[index].time) - guide),
                    -float(getattr(row[index], "envelope", 0.0)),
                ),
            )
            for index in ranked:
                if len(selected) >= cap:
                    break
                if abs(float(row[index].time) - guide) <= half:
                    selected.add(int(index))
        result.append(selected)
    return result


def _build_nodes(
    candidates: Sequence[Sequence[Any]],
    *,
    anchor_mask: np.ndarray,
    seed_receiver: int,
    state_valid: np.ndarray | None,
    escape_state_valid: np.ndarray | None,
    config: Mapping[str, object],
    allowed_candidate_indices: Sequence[set[int] | None] | None = None,
) -> tuple[list[_Node], list[list[int]], int]:
    bank = _template_bank(candidates, anchor_mask, seed_receiver)
    nodes: list[_Node] = []
    by_receiver: list[list[int]] = []
    escape_candidates = 0
    top_m = int(config["template_top_m"])
    lag = int(config["template_lag_samples"])
    amplitude_weight = float(config["amplitude_weight"])

    for receiver, row in enumerate(candidates):
        local: list[int] = []
        allowed_here = (
            None
            if allowed_candidate_indices is None
            else allowed_candidate_indices[receiver]
        )
        for candidate_index, candidate in enumerate(row):
            if allowed_here is not None and candidate_index not in allowed_here:
                continue
            sample = int(candidate.sample)
            in_base = True if state_valid is None else bool(state_valid[receiver, sample])
            in_escape = in_base if escape_state_valid is None else bool(
                escape_state_valid[receiver, sample]
            )
            if not in_escape:
                continue
            is_escape = bool(not in_base)
            escape_candidates += int(is_escape)
            waveform = np.asarray(candidate.waveform, dtype=float)
            template_corr = _template_similarity(waveform, bank, top_m, lag)
            envelope_reward = amplitude_weight * float(
                np.tanh(max(0.0, float(candidate.envelope)))
            )
            node_id = len(nodes)
            nodes.append(
                _Node(
                    node_id=node_id,
                    receiver=int(receiver),
                    candidate_index=int(candidate_index),
                    time=float(candidate.time),
                    sample=sample,
                    waveform=waveform,
                    template_corr=template_corr,
                    envelope_reward=envelope_reward,
                    is_escape=is_escape,
                    is_anchor=bool(anchor_mask[receiver]),
                )
            )
            local.append(node_id)
        by_receiver.append(local)
    return nodes, by_receiver, escape_candidates


def _fixed_prefix(
    by_receiver: Sequence[Sequence[int]],
    anchor_mask: np.ndarray,
    seed_receiver: int,
    direction: int,
) -> list[int]:
    if direction > 0:
        receivers = np.flatnonzero(
            anchor_mask & (np.arange(anchor_mask.size) >= seed_receiver)
        )
        receivers = receivers[np.argsort(receivers)]
    else:
        receivers = np.flatnonzero(
            anchor_mask & (np.arange(anchor_mask.size) <= seed_receiver)
        )
        receivers = receivers[np.argsort(-receivers)]
    if not receivers.size:
        receivers = np.asarray([seed_receiver])
    prefix: list[int] = []
    for receiver in receivers:
        row = by_receiver[int(receiver)]
        if not row:
            raise ValueError("global DP anchor receiver has no candidate")
        prefix.append(int(row[0]))
    return prefix


def _gap_penalty(
    valid_receiver: np.ndarray,
    source_receiver: int,
    target_receiver: int,
    config: Mapping[str, object],
) -> float:
    step = 1 if target_receiver > source_receiver else -1
    skipped = np.arange(source_receiver + step, target_receiver, step, dtype=int)
    if not skipped.size:
        return 0.0
    valid_count = int(np.count_nonzero(valid_receiver[skipped]))
    invalid_count = int(skipped.size - valid_count)
    return (
        valid_count * float(config["valid_gap_penalty"])
        + invalid_count * float(config["invalid_gap_penalty"])
    )


def _pair_feature(
    source: _Node,
    target: _Node,
    *,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    guide_time: np.ndarray | None,
    config: Mapping[str, object],
) -> _PairFeature | None:
    dx = float(receiver_x[target.receiver] - receiver_x[source.receiver])
    if dx == 0.0:
        return None
    slope = (target.time - source.time) / dx
    absolute_slope = float(config["absolute_max_slope_ms_per_100m"]) * 1e-5
    if abs(slope) > absolute_slope:
        return None
    rho = float(np.dot(source.waveform, target.waveform))
    if rho < float(config["hard_min_neighbor_corr"]):
        return None
    gap_penalty = _gap_penalty(
        valid_receiver, source.receiver, target.receiver, config
    )
    escape_penalty = float(config["escape_penalty"]) if target.is_escape else 0.0
    guide_penalty = 0.0
    if guide_time is not None and np.isfinite(guide_time[target.receiver]):
        guide_penalty = float(config["guide_weight"]) * _huber(
            (target.time - float(guide_time[target.receiver]))
            / float(config["guide_scale_time"])
        )
    increment = (
        float(config["coverage_reward"])
        + float(config["template_weight"]) * target.template_corr
        + target.envelope_reward
        + float(config["neighbor_weight"]) * rho
        - gap_penalty
        - escape_penalty
        - guide_penalty
    )
    return _PairFeature(
        slope=float(slope),
        neighbor_corr=float(rho),
        gap_penalty=float(gap_penalty),
        increment=float(increment),
        escape_penalty=float(escape_penalty),
        guide_penalty=float(guide_penalty),
    )


def _backtrack(
    final_key: tuple[int, int],
    final_rank: int,
    entries: Mapping[tuple[int, int], list[_Entry]],
    fixed_prefix: Sequence[int],
    nodes: Sequence[_Node],
    *,
    terminal_score: float,
) -> GlobalDPPath:
    key = final_key
    rank = int(final_rank)
    recovered: list[int] = []
    final_entry = entries[key][rank]
    while True:
        older, previous = key
        recovered.append(int(previous))
        entry = entries[key][rank]
        if entry.prev_key is None:
            if older >= 0:
                recovered.append(int(older))
            break
        key = entry.prev_key
        rank = int(entry.prev_rank)
    recovered.reverse()

    ordered_nodes: list[int] = []
    seen: set[int] = set()
    for node_id in list(fixed_prefix) + recovered:
        if node_id not in seen:
            ordered_nodes.append(int(node_id))
            seen.add(int(node_id))
    selected = tuple(
        (nodes[node_id].receiver, nodes[node_id].candidate_index)
        for node_id in ordered_nodes
    )
    return GlobalDPPath(
        selected_items=selected,
        score=float(terminal_score),
        template_score=float(final_entry.template_score),
        neighbor_score=float(final_entry.neighbor_score),
        curvature_penalty=float(final_entry.curvature_penalty),
        gap_penalty=float(final_entry.gap_penalty),
        escape_penalty=float(final_entry.escape_penalty),
        guide_penalty=float(final_entry.guide_penalty),
        selected_count=int(final_entry.selected_count + max(0, len(fixed_prefix) - 2)),
    )


def _materialize_top_k_bucket(
    proposals: Sequence[
        tuple[tuple[int, int], _PairFeature, float, np.ndarray]
    ],
    *,
    target: _Node,
    all_entries: Mapping[tuple[int, int], list[_Entry]],
    top_k: int,
    template_weight: float,
    neighbor_weight: float,
) -> list[_Entry]:
    """Select exact per-state Top-K scores before creating Python objects."""

    if not proposals:
        return []
    score_parts = [proposal[3] for proposal in proposals]
    sizes = np.asarray([part.size for part in score_parts], dtype=int)
    if not np.any(sizes):
        return []
    flat_scores = np.concatenate(score_parts)
    keep = min(int(top_k), int(flat_scores.size))
    if flat_scores.size > keep:
        selected = np.argpartition(flat_scores, -keep)[-keep:]
        selected = selected[np.argsort(flat_scores[selected])[::-1]]
    else:
        selected = np.argsort(flat_scores)[::-1]
    offsets = np.concatenate(([0], np.cumsum(sizes)))

    bucket: list[_Entry] = []
    for flat_index in selected:
        proposal_index = int(
            np.searchsorted(offsets[1:], int(flat_index), side="right")
        )
        local_rank = int(flat_index - offsets[proposal_index])
        predecessor_key, feature, curvature_penalty, _ = proposals[proposal_index]
        predecessor = all_entries[predecessor_key][local_rank]
        bucket.append(
            _Entry(
                score=float(flat_scores[flat_index]),
                prev_key=predecessor_key,
                prev_rank=local_rank,
                template_score=float(
                    predecessor.template_score
                    + template_weight * target.template_corr
                ),
                neighbor_score=float(
                    predecessor.neighbor_score
                    + neighbor_weight * feature.neighbor_corr
                ),
                curvature_penalty=float(
                    predecessor.curvature_penalty + curvature_penalty
                ),
                gap_penalty=float(predecessor.gap_penalty + feature.gap_penalty),
                escape_penalty=float(
                    predecessor.escape_penalty + feature.escape_penalty
                ),
                guide_penalty=float(
                    predecessor.guide_penalty + feature.guide_penalty
                ),
                selected_count=int(predecessor.selected_count + 1),
            )
        )
    return bucket


def _solve_direction(
    nodes: Sequence[_Node],
    by_receiver: Sequence[Sequence[int]],
    *,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    anchor_mask: np.ndarray,
    seed_receiver: int,
    direction: int,
    guide_time: np.ndarray | None,
    config: Mapping[str, object],
) -> _DirectionResult:
    top_k = int(config["top_k"])
    max_gap = int(config["max_gap_receivers"])
    curvature_weight = float(config["curvature_weight"])
    curvature_scale = float(config["curvature_scale_ms_per_100m"])
    tail_miss_penalty = float(config["tail_miss_penalty"])
    template_weight = float(config["template_weight"])
    neighbor_weight = float(config["neighbor_weight"])
    gap_trigger_template = float(config["gap_trigger_template_corr"])

    prefix = _fixed_prefix(by_receiver, anchor_mask, seed_receiver, direction)
    boundary_receiver = nodes[prefix[-1]].receiver
    initial_key = (
        (int(prefix[-2]), int(prefix[-1]))
        if len(prefix) >= 2
        else (-1, int(prefix[-1]))
    )
    all_entries: dict[tuple[int, int], list[_Entry]] = {
        initial_key: [
            _Entry(
                score=0.0,
                prev_key=None,
                prev_rank=-1,
                template_score=0.0,
                neighbor_score=0.0,
                curvature_penalty=0.0,
                gap_penalty=0.0,
                escape_penalty=0.0,
                guide_penalty=0.0,
                selected_count=1,
            )
        ]
    }
    states_by_last_receiver: dict[int, set[tuple[int, int]]] = {
        boundary_receiver: {initial_key}
    }
    endpoint_candidates: list[tuple[float, tuple[int, int], int]] = []
    transition_count = 0
    correlation_count = 0
    escape_transition_count = 0
    active_state_max = 1

    best_template_by_receiver = np.full(len(by_receiver), -np.inf, dtype=float)
    for receiver, node_ids in enumerate(by_receiver):
        if node_ids:
            best_template_by_receiver[receiver] = max(
                nodes[node_id].template_corr for node_id in node_ids
            )
    weak_receiver = (
        ~np.asarray(valid_receiver, dtype=bool)
        | (best_template_by_receiver < gap_trigger_template)
    )

    def gap_allowed(source_receiver: int, target_receiver: int) -> bool:
        if abs(int(target_receiver) - int(source_receiver)) <= 1:
            return True
        step = 1 if target_receiver > source_receiver else -1
        skipped = np.arange(
            source_receiver + step, target_receiver, step, dtype=int
        )
        return bool(not skipped.size or np.all(weak_receiver[skipped]))

    def remaining_valid(last_receiver: int) -> int:
        tail = (
            valid_receiver[last_receiver + 1 :]
            if direction > 0
            else valid_receiver[:last_receiver]
        )
        return int(np.count_nonzero(tail))

    def record_endpoints(last_receiver: int, keys: Sequence[tuple[int, int]]) -> None:
        tail_cost = tail_miss_penalty * remaining_valid(last_receiver)
        for key in keys:
            for rank, entry in enumerate(all_entries[key]):
                endpoint_candidates.append((entry.score - tail_cost, key, rank))

    record_endpoints(boundary_receiver, [initial_key])
    target_receivers = (
        range(boundary_receiver + 1, len(by_receiver))
        if direction > 0
        else range(boundary_receiver - 1, -1, -1)
    )

    for target_receiver in target_receivers:
        target_nodes = by_receiver[target_receiver]
        if not target_nodes:
            continue
        proposals_by_new_key: dict[
            tuple[int, int],
            list[tuple[tuple[int, int], _PairFeature, float, np.ndarray]],
        ] = {}
        pair_cache: dict[tuple[int, int], _PairFeature | None] = {}

        for hop in range(1, max_gap + 2):
            source_receiver = target_receiver - direction * hop
            if source_receiver < 0 or source_receiver >= len(by_receiver):
                continue
            state_keys = states_by_last_receiver.get(int(source_receiver))
            if not state_keys:
                continue
            if hop > 1 and not gap_allowed(source_receiver, target_receiver):
                continue
            for key in state_keys:
                older_id, previous_id = key
                previous = nodes[previous_id]
                predecessor_entries = all_entries[key]
                predecessor_scores = np.asarray(
                    [entry.score for entry in predecessor_entries], dtype=float
                )
                if not predecessor_scores.size:
                    continue
                for target_id in target_nodes:
                    target_id = int(target_id)
                    cache_key = (int(previous_id), target_id)
                    if cache_key not in pair_cache:
                        pair_cache[cache_key] = _pair_feature(
                            previous,
                            nodes[target_id],
                            receiver_x=receiver_x,
                            valid_receiver=valid_receiver,
                            guide_time=guide_time,
                            config=config,
                        )
                        correlation_count += 1
                    feature = pair_cache[cache_key]
                    if feature is None:
                        continue
                    target = nodes[target_id]
                    curvature_penalty = 0.0
                    if older_id >= 0:
                        older = nodes[older_id]
                        dx_old = float(
                            receiver_x[previous.receiver] - receiver_x[older.receiver]
                        )
                        if dx_old == 0.0:
                            continue
                        previous_slope = (previous.time - older.time) / dx_old
                        slope_change_ms100m = (
                            feature.slope - previous_slope
                        ) * 1e5
                        curvature_penalty = curvature_weight * _huber(
                            slope_change_ms100m / curvature_scale
                        )
                    proposed_scores = (
                        predecessor_scores + feature.increment - curvature_penalty
                    )
                    new_key = (int(previous_id), target_id)
                    proposals_by_new_key.setdefault(new_key, []).append(
                        (key, feature, float(curvature_penalty), proposed_scores)
                    )
                    transition_count += int(predecessor_scores.size)
                    if target.is_escape:
                        escape_transition_count += int(predecessor_scores.size)

        if not proposals_by_new_key:
            continue
        new_buckets: dict[tuple[int, int], list[_Entry]] = {}
        for new_key, proposals in proposals_by_new_key.items():
            bucket = _materialize_top_k_bucket(
                proposals,
                target=nodes[new_key[1]],
                all_entries=all_entries,
                top_k=top_k,
                template_weight=template_weight,
                neighbor_weight=neighbor_weight,
            )
            if bucket:
                new_buckets[new_key] = bucket
        if not new_buckets:
            continue
        keys = set(new_buckets)
        for key, bucket in new_buckets.items():
            all_entries[key] = bucket
        states_by_last_receiver[target_receiver] = keys
        active_state_max = max(
            active_state_max,
            sum(len(all_entries[key]) for key in keys),
        )
        record_endpoints(target_receiver, list(keys))

    endpoint_candidates.sort(key=lambda item: item[0], reverse=True)
    paths: list[GlobalDPPath] = []
    seen_paths: set[tuple[tuple[int, int], ...]] = set()
    for terminal_score, key, rank in endpoint_candidates:
        path = _backtrack(
            key,
            rank,
            all_entries,
            prefix,
            nodes,
            terminal_score=float(terminal_score),
        )
        if path.selected_items in seen_paths:
            continue
        seen_paths.add(path.selected_items)
        paths.append(path)
        if len(paths) >= top_k:
            break

    if not paths:
        fixed_selected = tuple(
            (nodes[node_id].receiver, nodes[node_id].candidate_index)
            for node_id in prefix
        )
        paths = [
            GlobalDPPath(
                selected_items=fixed_selected,
                score=-tail_miss_penalty * remaining_valid(boundary_receiver),
                template_score=0.0,
                neighbor_score=0.0,
                curvature_penalty=0.0,
                gap_penalty=0.0,
                escape_penalty=0.0,
                guide_penalty=0.0,
                selected_count=len(fixed_selected),
            )
        ]

    return _DirectionResult(
        paths=tuple(paths),
        transition_count=int(transition_count),
        correlation_count=int(correlation_count),
        active_state_max=int(active_state_max),
        escape_transition_count=int(escape_transition_count),
    )


def _combine_paths(
    left: _DirectionResult,
    right: _DirectionResult,
    top_k: int,
) -> tuple[GlobalDPPath, ...]:
    combined: list[GlobalDPPath] = []
    for left_path in left.paths:
        left_selected = left_path.as_dict()
        for right_path in right.paths:
            selected = dict(left_selected)
            selected.update(right_path.as_dict())
            combined.append(
                GlobalDPPath(
                    selected_items=tuple(sorted(selected.items())),
                    score=float(left_path.score + right_path.score),
                    template_score=float(
                        left_path.template_score + right_path.template_score
                    ),
                    neighbor_score=float(
                        left_path.neighbor_score + right_path.neighbor_score
                    ),
                    curvature_penalty=float(
                        left_path.curvature_penalty + right_path.curvature_penalty
                    ),
                    gap_penalty=float(left_path.gap_penalty + right_path.gap_penalty),
                    escape_penalty=float(
                        left_path.escape_penalty + right_path.escape_penalty
                    ),
                    guide_penalty=float(
                        left_path.guide_penalty + right_path.guide_penalty
                    ),
                    selected_count=len(selected),
                )
            )
    combined.sort(
        key=lambda path: (
            path.score,
            path.template_score,
            path.neighbor_score,
            -path.curvature_penalty,
            path.selected_count,
        ),
        reverse=True,
    )
    unique: list[GlobalDPPath] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for path in combined:
        if path.selected_items in seen:
            continue
        seen.add(path.selected_items)
        unique.append(path)
        if len(unique) >= int(top_k):
            break
    return tuple(unique)


def _path_weights(paths: Sequence[GlobalDPPath], temperature: float) -> np.ndarray:
    """Softmax on score per selected receiver, not raw accumulated score."""

    normalized = np.asarray(
        [path.score / max(1, path.selected_count) for path in paths], dtype=float
    )
    if not normalized.size:
        return normalized
    delta = (normalized - float(np.max(normalized))) / max(float(temperature), 1e-12)
    weights = np.exp(np.clip(delta, -50.0, 0.0))
    total = float(np.sum(weights))
    return (
        weights / total
        if total > 0.0
        else np.full(normalized.shape, 1.0 / normalized.size)
    )


def _consensus(
    paths: Sequence[GlobalDPPath],
    candidates: Sequence[Sequence[Any]],
    *,
    nr: int,
    radius: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the maximum-weight time cluster independently at each receiver."""

    fraction = np.zeros(nr, dtype=float)
    spread = np.full(nr, np.nan, dtype=float)
    center = np.full(nr, np.nan, dtype=float)
    if not paths:
        return fraction, spread, center
    weights = _path_weights(paths, temperature)
    path_maps = [path.as_dict() for path in paths]
    total_weight = float(np.sum(weights))

    for receiver in range(nr):
        times: list[float] = []
        local_weights: list[float] = []
        for weight, mapping in zip(weights, path_maps):
            if receiver not in mapping:
                continue
            times.append(float(candidates[receiver][mapping[receiver]].time))
            local_weights.append(float(weight))
        if not times:
            continue
        times_a = np.asarray(times, dtype=float)
        weights_a = np.asarray(local_weights, dtype=float)

        best_mask = None
        best_key = None
        for trial in times_a:
            mask = np.abs(times_a - trial) <= float(radius)
            mass = float(np.sum(weights_a[mask]))
            if not np.any(mask):
                continue
            w = weights_a[mask]
            t = times_a[mask]
            wsum = float(np.sum(w))
            mean = float(np.sum(w * t) / wsum)
            variance = float(np.sum(w * (t - mean) ** 2) / wsum)
            key = (mass, -variance, -mean)
            if best_key is None or key > best_key:
                best_key = key
                best_mask = mask
        if best_mask is None:
            continue
        w = weights_a[best_mask]
        t = times_a[best_mask]
        mass = float(np.sum(w))
        wsum = max(mass, np.finfo(float).eps)
        mean = float(np.sum(w * t) / wsum)
        center[receiver] = mean
        spread[receiver] = float(np.sqrt(np.sum(w * (t - mean) ** 2) / wsum))
        fraction[receiver] = mass / max(total_weight, np.finfo(float).eps)
    return fraction, spread, center


def _fit_global_guide(
    receiver_x: np.ndarray,
    consensus_center: np.ndarray,
    consensus_fraction: np.ndarray,
    consensus_spread: np.ndarray,
    anchor_mask: np.ndarray,
    best_selected: Mapping[int, int],
    candidates: Sequence[Sequence[Any]],
    config: Mapping[str, object],
) -> np.ndarray:
    nr = receiver_x.size
    y = np.array(consensus_center, copy=True)
    good = (
        np.isfinite(y)
        & (consensus_fraction >= float(config["consensus_min_fraction"]))
        & np.isfinite(consensus_spread)
        & (consensus_spread <= float(config["consensus_max_spread_time"]))
    )
    for receiver in np.flatnonzero(anchor_mask):
        if int(receiver) in best_selected:
            y[receiver] = float(
                candidates[int(receiver)][best_selected[int(receiver)]].time
            )
            good[receiver] = True

    indices = np.flatnonzero(good)
    guide = np.full(nr, np.nan, dtype=float)
    if indices.size < 2:
        return guide

    x = np.asarray(receiver_x[indices], dtype=float)
    values = np.asarray(y[indices], dtype=float)
    order = np.argsort(x)
    x = x[order]
    values = values[order]
    source_indices = indices[order]
    base_weights = np.clip(consensus_fraction[source_indices], 0.05, 1.0)
    base_weights = np.where(
        anchor_mask[source_indices],
        base_weights * float(config["anchor_guide_weight"]),
        base_weights,
    )
    weights = np.array(base_weights, copy=True)
    smoothing_time = float(config["guide_smoothing_time"])
    delta = float(config["guide_huber_delta_time"])
    spline = None

    for _ in range(int(config["guide_fit_iterations"])):
        degree = min(3, x.size - 1)
        smoothing = float(x.size) * smoothing_time * smoothing_time
        try:
            spline = UnivariateSpline(
                x,
                values,
                w=np.sqrt(np.maximum(weights, 1e-8)),
                k=degree,
                s=smoothing,
            )
        except Exception:
            spline = None
            break
        residual = values - spline(x)
        robust_scale = 1.4826 * float(
            np.median(np.abs(residual - np.median(residual)))
        )
        robust_scale = max(robust_scale, delta / 3.0, 1e-6)
        threshold = max(delta, 1.5 * robust_scale)
        robust = np.ones_like(residual)
        outside = np.abs(residual) > threshold
        robust[outside] = threshold / np.abs(residual[outside])
        weights = base_weights * robust

    inside = (receiver_x >= x[0]) & (receiver_x <= x[-1])
    if spline is None:
        guide[inside] = np.interp(receiver_x[inside], x, values)
    else:
        guide[inside] = spline(receiver_x[inside])
    return guide


def _side_stop(
    selected: Mapping[int, int],
    valid_receiver: np.ndarray,
    seed_receiver: int,
    direction: int,
) -> tuple[str, int]:
    side_receivers = [
        receiver
        for receiver in selected
        if direction * (receiver - seed_receiver) >= 0
    ]
    if not side_receivers:
        return "GLOBAL_NO_PATH", int(seed_receiver)
    outer = max(side_receivers) if direction > 0 else min(side_receivers)
    remaining = (
        np.flatnonzero(valid_receiver[outer + 1 :]) + outer + 1
        if direction > 0
        else np.flatnonzero(valid_receiver[:outer])
    )
    if remaining.size == 0:
        return "END_OF_VALID_GATHER", -1
    stop = int(remaining[0] if direction > 0 else remaining[-1])
    return "GLOBAL_EVIDENCE_STOP", stop


def _selected_masks(
    best: Mapping[int, int],
    nodes: Sequence[_Node],
    valid_receiver: np.ndarray,
    anchor_mask: np.ndarray,
    seed_receiver: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nr = valid_receiver.size
    node_lookup = {(node.receiver, node.candidate_index): node for node in nodes}
    template = np.full(nr, np.nan, dtype=float)
    escape = np.zeros(nr, dtype=bool)
    transition = np.full(nr, -1, dtype=np.int8)
    for receiver, candidate_index in best.items():
        node = node_lookup[(int(receiver), int(candidate_index))]
        template[int(receiver)] = node.template_corr
        escape[int(receiver)] = node.is_escape
        transition[int(receiver)] = (
            TRANSITION_SEED
            if int(receiver) == int(seed_receiver)
            else TRANSITION_ANCHOR
            if anchor_mask[int(receiver)]
            else TRANSITION_NORMAL
        )
    skipped = np.zeros(nr, dtype=bool)
    selected_receivers = sorted(best)
    for first, second in zip(selected_receivers[:-1], selected_receivers[1:]):
        if second - first <= 1:
            continue
        middle = np.arange(first + 1, second, dtype=int)
        skipped[middle] = valid_receiver[middle]
        transition[middle[valid_receiver[middle]]] = TRANSITION_GAP
    return template, escape, skipped, transition


def _run_pass(
    nodes: Sequence[_Node],
    by_receiver: Sequence[Sequence[int]],
    *,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    anchor_mask: np.ndarray,
    seed_receiver: int,
    guide_time: np.ndarray | None,
    config: Mapping[str, object],
    top_k: int,
) -> tuple[_DirectionResult, _DirectionResult, tuple[GlobalDPPath, ...]]:
    pass_config = dict(config)
    pass_config["top_k"] = int(top_k)
    left = _solve_direction(
        nodes,
        by_receiver,
        receiver_x=receiver_x,
        valid_receiver=valid_receiver,
        anchor_mask=anchor_mask,
        seed_receiver=seed_receiver,
        direction=-1,
        guide_time=guide_time,
        config=pass_config,
    )
    right = _solve_direction(
        nodes,
        by_receiver,
        receiver_x=receiver_x,
        valid_receiver=valid_receiver,
        anchor_mask=anchor_mask,
        seed_receiver=seed_receiver,
        direction=1,
        guide_time=guide_time,
        config=pass_config,
    )
    return left, right, _combine_paths(left, right, int(top_k))


def solve_global_topk_two_pass(
    candidates: MutableSequence[MutableSequence[Any]],
    *,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    seed_receiver: int,
    anchor_mask: np.ndarray,
    state_valid: np.ndarray | None,
    escape_state_valid: np.ndarray | None,
    config: Mapping[str, object] | None = None,
    traces: np.ndarray | None = None,
    candidate_envelope: np.ndarray | None = None,
    dt: float | None = None,
    t0: float = 0.0,
    half_window_samples: int | None = None,
) -> GlobalDPSolution:
    """Solve one flattened reflector with exact second-order Top-K DP."""

    started = perf_counter()
    profile: dict[str, float] = {}
    cfg = normalise_global_dp_options(config)
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    anchor = np.asarray(anchor_mask, dtype=bool)
    nr = x.size
    if valid.shape != (nr,) or anchor.shape != (nr,):
        raise ValueError("global DP geometry masks must match receiver_x")
    if len(candidates) != nr:
        raise ValueError("global DP candidates must contain one row per receiver")
    if not 0 <= int(seed_receiver) < nr:
        raise ValueError("global DP seed receiver is outside the gather")

    base = None if state_valid is None else np.asarray(state_valid, dtype=bool)
    escape = base if escape_state_valid is None else np.asarray(
        escape_state_valid, dtype=bool
    )
    if base is not None and base.shape[0] != nr:
        raise ValueError("global DP state_valid receiver dimension is inconsistent")
    if escape is not None and (base is None or escape.shape != base.shape):
        raise ValueError("global DP escape_state_valid must match state_valid")

    can_augment = (
        traces is not None
        and candidate_envelope is not None
        and dt is not None
        and half_window_samples is not None
    )
    stage_started = perf_counter()
    if can_augment:
        prepare_global_candidate_pool(
            candidates,
            traces=np.asarray(traces, dtype=float),
            candidate_envelope=np.asarray(candidate_envelope, dtype=float),
            receiver_x=x,
            valid_receiver=valid,
            seed_receiver=int(seed_receiver),
            anchor_mask=anchor,
            state_valid=base,
            escape_state_valid=escape,
            dt=float(dt),
            t0=float(t0),
            half_window_samples=int(half_window_samples),
            config=cfg,
            guide_time=None,
            guide_only=False,
        )
    profile["candidate_pool_pass1"] = perf_counter() - stage_started
    pass1_candidate_counts = np.asarray([len(row) for row in candidates], dtype=int)

    stage_started = perf_counter()
    nodes1, by_receiver1, escape_candidate_count1 = _build_nodes(
        candidates,
        anchor_mask=anchor,
        seed_receiver=int(seed_receiver),
        state_valid=base,
        escape_state_valid=escape,
        config=cfg,
    )
    profile["build_nodes_pass1"] = perf_counter() - stage_started
    stage_started = perf_counter()
    pass1_left, pass1_right, pass1_paths = _run_pass(
        nodes1,
        by_receiver1,
        receiver_x=x,
        valid_receiver=valid,
        anchor_mask=anchor,
        seed_receiver=int(seed_receiver),
        guide_time=None,
        config=cfg,
        top_k=int(cfg["pass1_top_k"]),
    )
    profile["dp_pass1"] = perf_counter() - stage_started
    if not pass1_paths:
        raise ValueError("global DP did not produce any complete path")

    stage_started = perf_counter()
    fraction1, spread1, center1 = _consensus(
        pass1_paths,
        candidates,
        nr=nr,
        radius=float(cfg["consensus_radius_time"]),
        temperature=float(cfg["path_weight_temperature"]),
    )
    guide = _fit_global_guide(
        x,
        center1,
        fraction1,
        spread1,
        anchor,
        pass1_paths[0].as_dict(),
        candidates,
        cfg,
    )
    profile["consensus_and_guide"] = perf_counter() - stage_started

    left, right, paths = pass1_left, pass1_right, pass1_paths
    nodes_final, by_receiver_final = nodes1, by_receiver1
    escape_candidate_count = escape_candidate_count1
    if bool(cfg["second_pass_enabled"]) and np.count_nonzero(np.isfinite(guide)) >= 2:
        stage_started = perf_counter()
        if can_augment:
            # Only append guide-near weak peaks. Existing indices are preserved,
            # so pass-1 paths remain valid for a safe fallback.
            prepare_global_candidate_pool(
                candidates,
                traces=np.asarray(traces, dtype=float),
                candidate_envelope=np.asarray(candidate_envelope, dtype=float),
                receiver_x=x,
                valid_receiver=valid,
                seed_receiver=int(seed_receiver),
                anchor_mask=anchor,
                state_valid=base,
                escape_state_valid=escape,
                dt=float(dt),
                t0=float(t0),
                half_window_samples=int(half_window_samples),
                config=cfg,
                guide_time=guide,
                guide_only=True,
            )
        profile["candidate_pool_pass2"] = perf_counter() - stage_started
        pass2_allowed = _build_pass2_allowed_candidate_indices(
            candidates,
            pass1_paths,
            guide,
            anchor,
            int(seed_receiver),
            cfg,
        )
        stage_started = perf_counter()
        nodes2, by_receiver2, escape_candidate_count2 = _build_nodes(
            candidates,
            anchor_mask=anchor,
            seed_receiver=int(seed_receiver),
            state_valid=base,
            escape_state_valid=escape,
            config=cfg,
            allowed_candidate_indices=pass2_allowed,
        )
        profile["build_nodes_pass2"] = perf_counter() - stage_started
        stage_started = perf_counter()
        pass2_left, pass2_right, pass2_paths = _run_pass(
            nodes2,
            by_receiver2,
            receiver_x=x,
            valid_receiver=valid,
            anchor_mask=anchor,
            seed_receiver=int(seed_receiver),
            guide_time=guide,
            config=cfg,
            top_k=int(cfg["pass2_top_k"]),
        )
        profile["dp_pass2"] = perf_counter() - stage_started
        if pass2_paths:
            left, right, paths = pass2_left, pass2_right, pass2_paths
            nodes_final, by_receiver_final = nodes2, by_receiver2
            escape_candidate_count = escape_candidate_count2
        else:
            guide = np.full(nr, np.nan, dtype=float)

    best = paths[0].as_dict()
    fraction, spread, _ = _consensus(
        paths,
        candidates,
        nr=nr,
        radius=float(cfg["consensus_radius_time"]),
        temperature=float(cfg["path_weight_temperature"]),
    )
    template, escape_used, skipped, transition_kind = _selected_masks(
        best,
        nodes_final,
        valid,
        anchor,
        int(seed_receiver),
    )
    left_selected = {
        receiver: index for receiver, index in best.items() if receiver <= seed_receiver
    }
    right_selected = {
        receiver: index for receiver, index in best.items() if receiver >= seed_receiver
    }
    stop_left, stop_receiver_left = _side_stop(best, valid, int(seed_receiver), -1)
    stop_right, stop_receiver_right = _side_stop(best, valid, int(seed_receiver), 1)

    best_score = float(paths[0].score)
    second_score = float(paths[1].score) if len(paths) > 1 else np.nan
    margin = best_score - second_score if np.isfinite(second_score) else np.nan
    return GlobalDPSolution(
        selected=dict(best),
        left_selected=left_selected,
        right_selected=right_selected,
        top_paths=tuple(paths),
        best_score=best_score,
        second_score=second_score,
        score_margin=float(margin),
        template_correlation=template,
        consensus_fraction=fraction,
        consensus_spread=spread,
        guide_time=np.asarray(guide, dtype=float),
        ownership_escape_used_mask=escape_used,
        skipped_valid_receiver_mask=skipped,
        selected_transition_kind=transition_kind,
        stop_reason_left=stop_left,
        stop_reason_right=stop_right,
        stop_receiver_left=int(stop_receiver_left),
        stop_receiver_right=int(stop_receiver_right),
        transition_count_left=int(left.transition_count),
        transition_count_right=int(right.transition_count),
        correlation_count_left=int(left.correlation_count),
        correlation_count_right=int(right.correlation_count),
        active_state_max=int(max(left.active_state_max, right.active_state_max)),
        escape_candidate_count=int(escape_candidate_count),
        escape_transition_count=int(
            left.escape_transition_count + right.escape_transition_count
        ),
        candidate_count_after_global_pool=np.asarray(
            [len(row) for row in candidates], dtype=int
        ),
        runtime_seconds=float(perf_counter() - started),
    )


def _normalise_traces_for_challenge(flat_gather: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat_gather, dtype=float)
    result = np.zeros_like(flat)
    scale = np.sqrt(np.mean(flat * flat, axis=1))
    good = np.isfinite(scale) & (scale > np.finfo(float).eps)
    result[good] = flat[good] / scale[good, None]
    return result


def score_tracking_hypothesis(
    *,
    flat_gather: np.ndarray,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    pick_sample: np.ndarray,
    success_mask: np.ndarray,
    anchor_receiver_mask: np.ndarray,
    anchor_pick_sample: np.ndarray,
    coherence_half_window_time: float,
    dt: float,
    config: Mapping[str, object] | None = None,
) -> TrackingChallengeScore:
    """Score any completed tracking result against the same immutable anchor."""

    cfg = normalise_global_dp_options(config)
    traces = _normalise_traces_for_challenge(np.asarray(flat_gather, dtype=float))
    x = np.asarray(receiver_x, dtype=float)
    valid = np.asarray(valid_receiver, dtype=bool)
    pick = np.asarray(pick_sample, dtype=int)
    success = np.asarray(success_mask, dtype=bool) & valid
    anchor = np.asarray(anchor_receiver_mask, dtype=bool)
    anchor_samples = np.asarray(anchor_pick_sample, dtype=int)
    nr, nt = traces.shape
    if any(arr.shape != (nr,) for arr in (x, valid, pick, success, anchor, anchor_samples)):
        raise ValueError("tracking challenge arrays must match the gather receiver axis")
    half = max(1, int(round(float(coherence_half_window_time) / float(dt))))

    bank: list[np.ndarray] = []
    for receiver in np.flatnonzero(anchor):
        sample = int(anchor_samples[receiver])
        if not 0 <= sample < nt:
            continue
        waveform = _candidate_waveform(traces[receiver], sample, half)
        if waveform is not None:
            bank.append(waveform)
    if not bank:
        return TrackingChallengeScore(-np.inf, 0.0, -1.0, -1.0, np.inf, 1.0)

    waveforms: dict[int, np.ndarray] = {}
    template_values: list[float] = []
    for receiver in np.flatnonzero(success):
        sample = int(pick[receiver])
        if not 0 <= sample < nt:
            continue
        waveform = _candidate_waveform(traces[receiver], sample, half)
        if waveform is None:
            continue
        waveforms[int(receiver)] = waveform
        template_values.append(
            _template_similarity(
                waveform,
                bank,
                int(cfg["template_top_m"]),
                int(cfg["template_lag_samples"]),
            )
        )

    usable_count = int(np.count_nonzero(valid))
    coverage = float(np.count_nonzero(success) / usable_count) if usable_count else 0.0
    template_similarity = (
        float(np.median(template_values)) if template_values else -1.0
    )

    neighbor_values: list[float] = []
    curvature_values: list[float] = []
    selected = sorted(waveforms)
    times = {
        receiver: float(pick[receiver]) * float(dt) for receiver in selected
    }
    for first, second in zip(selected[:-1], selected[1:]):
        neighbor_values.append(float(np.dot(waveforms[first], waveforms[second])))
    for a, b, c in zip(selected[:-2], selected[1:-1], selected[2:]):
        dx1 = float(x[b] - x[a])
        dx2 = float(x[c] - x[b])
        if dx1 == 0.0 or dx2 == 0.0:
            continue
        s1 = (times[b] - times[a]) / dx1
        s2 = (times[c] - times[b]) / dx2
        curvature_values.append(
            _huber(
                ((s2 - s1) * 1e5)
                / float(cfg["curvature_scale_ms_per_100m"])
            )
        )
    neighbor_similarity = (
        float(np.median(neighbor_values)) if neighbor_values else -1.0
    )
    curvature_penalty = (
        float(np.median(curvature_values)) if curvature_values else 0.0
    )
    gap_fraction = 1.0 - coverage
    total = (
        float(cfg["template_weight"]) * template_similarity
        + float(cfg["neighbor_weight"]) * neighbor_similarity
        + float(cfg["repair_coverage_weight"]) * coverage
        - float(cfg["curvature_weight"]) * curvature_penalty
        - float(cfg["repair_gap_weight"]) * gap_fraction
    )
    return TrackingChallengeScore(
        total=float(total),
        coverage=coverage,
        template_similarity=template_similarity,
        neighbor_similarity=neighbor_similarity,
        curvature_penalty=curvature_penalty,
        gap_fraction=gap_fraction,
    )


def compare_tracking_hypotheses(
    *,
    flat_gather: np.ndarray,
    receiver_x: np.ndarray,
    valid_receiver: np.ndarray,
    incumbent_pick_sample: np.ndarray,
    incumbent_success_mask: np.ndarray,
    challenger_pick_sample: np.ndarray,
    challenger_success_mask: np.ndarray,
    anchor_receiver_mask: np.ndarray,
    anchor_pick_sample: np.ndarray,
    coherence_half_window_time: float,
    dt: float,
    config: Mapping[str, object] | None = None,
) -> TrackingChallengeDecision:
    """Choose a repair challenger only if it materially beats the global result."""

    cfg = normalise_global_dp_options(config)
    incumbent = score_tracking_hypothesis(
        flat_gather=flat_gather,
        receiver_x=receiver_x,
        valid_receiver=valid_receiver,
        pick_sample=incumbent_pick_sample,
        success_mask=incumbent_success_mask,
        anchor_receiver_mask=anchor_receiver_mask,
        anchor_pick_sample=anchor_pick_sample,
        coherence_half_window_time=coherence_half_window_time,
        dt=dt,
        config=cfg,
    )
    challenger = score_tracking_hypothesis(
        flat_gather=flat_gather,
        receiver_x=receiver_x,
        valid_receiver=valid_receiver,
        pick_sample=challenger_pick_sample,
        success_mask=challenger_success_mask,
        anchor_receiver_mask=anchor_receiver_mask,
        anchor_pick_sample=anchor_pick_sample,
        coherence_half_window_time=coherence_half_window_time,
        dt=dt,
        config=cfg,
    )
    improvement = challenger.total - incumbent.total
    use = bool(
        cfg["repair_challenge_enabled"]
        and np.isfinite(challenger.total)
        and improvement >= float(cfg["repair_challenge_min_improvement"])
    )
    return TrackingChallengeDecision(
        use_challenger=use,
        incumbent=incumbent,
        challenger=challenger,
        improvement=float(improvement),
    )


__all__ = [
    "DEFAULT_GLOBAL_DP",
    "GlobalDPPath",
    "GlobalDPSolution",
    "TrackingChallengeDecision",
    "TrackingChallengeScore",
    "compare_tracking_hypotheses",
    "normalise_global_dp_options",
    "prepare_global_candidate_pool",
    "score_tracking_hypothesis",
    "solve_global_topk_two_pass",
]
