"""Opt-in runtime evidence for ZNCC tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..correlation import CorrelationResult
from .fallback import ShiftCompletionResult
from .tracker import TrackingResult, _top_k_peak_mask


REASONS = ("SUCCESS", "NO_BASE_STATE", "NO_REACHABLE_TRANSITION", "SEED_INVALID", "BACKTRACK_FAILURE", "OTHER")


def save_tracking_snapshot(
    directory: str | Path,
    *,
    shot: int,
    reflector: int,
    receiver_x: np.ndarray,
    correlation: CorrelationResult,
    tracking: TrackingResult,
    completion: ShiftCompletionResult,
    top_k_peaks: int,
    peak_min_separation_samples: int,
    epsilon_samples: int,
) -> Path:
    """Save the exact arrays used by the current sparse-DP reachability test."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(correlation.waveform_correlation, dtype=float)
    lags = np.asarray(correlation.lags_samples, dtype=int)
    base_valid = np.isfinite(raw) & np.asarray(correlation.valid, bool)[:, None] & np.asarray(correlation.lag_valid, bool)
    coarse = np.asarray(correlation.coarse_lag_samples, dtype=float)
    lag_time = lags.astype(float) * float(correlation.dt)
    ownership = (
        lag_time[None, :] >= np.asarray(correlation.ownership_lag_min)[:, None]
    ) & (
        lag_time[None, :] <= np.asarray(correlation.ownership_lag_max)[:, None]
    )
    candidates = _top_k_peak_mask(raw, base_valid, lags, top_k=top_k_peaks, min_separation_samples=peak_min_separation_samples, coarse_lag_samples=coarse)
    nreceiver = raw.shape[0]
    reachable = np.zeros_like(candidates)
    previous = np.full(nreceiver, -1, dtype=int)
    min_abs = np.full(nreceiver, np.nan)
    min_residual = np.full(nreceiver, np.nan)
    coarse_delta = np.full(nreceiver, np.nan)
    seed = int(tracking.seed_receiver)
    seed_state = int(tracking.path_index[seed]) if tracking.success_mask[seed] else -1
    if seed_state >= 0:
        reachable[seed, seed_state] = True
    for direction in (-1, 1):
        prior = seed
        for current in range(seed + direction, nreceiver if direction > 0 else -1, direction):
            previous[current] = prior
            prev_states = np.flatnonzero(reachable[prior])
            now_states = np.flatnonzero(candidates[current])
            if prev_states.size and now_states.size:
                delta = np.abs(lags[now_states, None] - lags[prev_states][None, :])
                min_abs[current] = float(np.min(delta))
                if np.isfinite(coarse[current]) and np.isfinite(coarse[prior]):
                    residual = (lags[now_states, None] - coarse[current]) - (lags[prev_states][None, :] - coarse[prior])
                    min_residual[current] = float(np.min(np.abs(residual)))
                    coarse_delta[current] = coarse[current] - coarse[prior]
                reachable[current, now_states] = np.any(delta <= epsilon_samples, axis=1)
            prior = current
    reason = np.full(nreceiver, "OTHER", dtype="U32")
    reason[tracking.success_mask] = "SUCCESS"
    reason[~np.any(base_valid, axis=1)] = "NO_BASE_STATE"
    reason[~tracking.success_mask & np.any(base_valid, axis=1) & ~np.any(reachable, axis=1)] = "NO_REACHABLE_TRANSITION"
    if seed_state < 0:
        reason[seed] = "SEED_INVALID"
    score = np.array(raw, copy=True)
    finite_coarse = np.isfinite(coarse)
    for row in np.flatnonzero(finite_coarse):
        score[row] -= 0.05 * np.minimum(((lags - coarse[row]) / 40.0) ** 2, 5.0)
    peaks = np.flatnonzero(base_valid[seed])
    ranked = peaks[np.argsort(-raw[seed, peaks], kind="stable")[:5]]
    seed_peak_lag = np.full(5, np.nan)
    seed_peak_zncc = np.full(5, np.nan)
    seed_peak_distance = np.full(5, np.nan)
    seed_peak_lag[:ranked.size] = lags[ranked]
    seed_peak_zncc[:ranked.size] = raw[seed, ranked]
    if np.isfinite(coarse[seed]):
        seed_peak_distance[:ranked.size] = np.abs(lags[ranked] - coarse[seed])
    stem = output / f"tracking_eval_shot_{shot + 1:03d}_R{reflector + 1}"
    np.savez_compressed(
        stem.with_suffix(".npz"), receiver_x=receiver_x, lags=lags, raw_waveform_zncc=raw,
        ownership_allowed=ownership, base_valid=base_valid, coarse_lag=coarse,
        coarse_valid=finite_coarse, dp_measurement_score=score, dp_path_index=tracking.path_index,
        dp_shift=tracking.shift_time, dp_success=tracking.dp_success_mask,
        measurement_success=tracking.success_mask, fallback_success=completion.fallback_success_mask,
        seed_receiver=seed, seed_lag=tracking.seed_lag, seed_state_index=seed_state,
        valid_state_count=np.count_nonzero(base_valid, axis=1), reachable_state_count=np.count_nonzero(reachable, axis=1),
        failure_reason=reason, previous_receiver=previous, min_absolute_lag_jump=min_abs,
        min_residual_lag_jump=min_residual, coarse_delta=coarse_delta,
        max_jump_samples=epsilon_samples, receiver_spacing=np.r_[np.nan, np.abs(np.diff(receiver_x))],
        candidate_mask=candidates, seed_top5_lag=seed_peak_lag, seed_top5_zncc=seed_peak_zncc,
        seed_top5_distance_to_coarse=seed_peak_distance,
        tracked_signed_zncc=tracking.tracked_correlation,
        tracked_strength=tracking.tracked_strength,
        tracked_polarity=tracking.tracked_polarity,
        component_id=tracking.component_id,
        provenance=tracking.provenance,
        restart_receiver=np.flatnonzero(np.asarray(tracking.provenance) == "RESTARTED_DP"),
        predicted_lag=tracking.predicted_lag,
        ambiguous_mask=tracking.ambiguous_mask,
        gap_count=tracking.gap_count,
    )
    _save_figure(stem.with_suffix(".png"), receiver_x, lags, raw, ownership, coarse, tracking, reason, min_abs, min_residual, epsilon_samples, base_valid, reachable)
    return stem.with_suffix(".npz")


def _save_figure(path, receiver_x, lags, raw, ownership, coarse, tracking, reason, min_abs, min_residual, epsilon, base_valid, reachable):
    import matplotlib.pyplot as plt
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    image = np.where(ownership, raw, np.nan)
    top.pcolormesh(receiver_x / 1000.0, lags, image.T, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    top.plot(receiver_x / 1000.0, coarse, "c--", label="coarse")
    top.plot(receiver_x[tracking.success_mask] / 1000.0, tracking.shift_samples[tracking.success_mask], "k-", label="DP")
    top.plot(receiver_x[tracking.seed_receiver] / 1000.0, tracking.seed_lag, "yo", label="seed")
    failures = np.flatnonzero(reason == "NO_REACHABLE_TRANSITION")
    for row in failures[:2]:
        top.axvline(receiver_x[row] / 1000.0, color="m", ls=":")
        top.text(receiver_x[row] / 1000.0, float(np.nanmax(lags)), f"abs={min_abs[row]:.0f}, res={min_residual[row]:.0f}, max={epsilon}", color="m", fontsize=8)
    top.set_ylabel("Lag (samples)"); top.legend(loc="best")
    bottom.plot(receiver_x / 1000.0, np.count_nonzero(base_valid, axis=1), label="valid")
    bottom.plot(receiver_x / 1000.0, np.count_nonzero(reachable, axis=1), label="reachable")
    bottom.set(xlabel="Receiver x (km)", ylabel="State count"); bottom.legend()
    fig.savefig(path, dpi=160); plt.close(fig)
