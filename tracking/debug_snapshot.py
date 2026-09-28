"""Opt-in runtime evidence for fixed-polarity ZNCC tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..correlation import CorrelationResult
from .fallback import ShiftCompletionResult
from .tracker import TrackingResult, _top_k_peak_mask


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
) -> Path:
    """Save the raw correlation, candidates, fixed polarity, and ridge."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(correlation.waveform_correlation, dtype=float)
    lags = np.asarray(correlation.lags_samples, dtype=int)
    valid = np.asarray(correlation.valid, dtype=bool)
    lag_valid = np.asarray(correlation.lag_valid, dtype=bool)
    base_valid = np.isfinite(raw) & valid[:, None] & lag_valid
    candidates = _top_k_peak_mask(
        raw,
        base_valid,
        lags,
        top_k=top_k_peaks,
        min_separation_samples=peak_min_separation_samples,
    )
    stem = output / f"tracking_eval_shot_{shot + 1:03d}_R{reflector + 1}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        receiver_x=np.asarray(receiver_x, dtype=float),
        lags=lags,
        raw_waveform_zncc=raw,
        valid=valid,
        lag_valid=lag_valid,
        ownership_lag_min=np.asarray(correlation.ownership_lag_min, dtype=float),
        ownership_lag_max=np.asarray(correlation.ownership_lag_max, dtype=float),
        candidate_mask=candidates,
        path_index=tracking.path_index,
        shift_samples=tracking.shift_samples,
        shift_time=tracking.shift_time,
        success_mask=tracking.success_mask,
        dp_success_mask=tracking.dp_success_mask,
        bridge_success_mask=tracking.bridge_success_mask,
        bridge_geometric_shift_samples=tracking.bridge_geometric_shift_samples,
        seed_receiver=tracking.seed_receiver,
        seed_lag=tracking.seed_lag,
        tracked_correlation=tracking.tracked_correlation,
        tracked_strength=tracking.tracked_strength,
        tracked_polarity=tracking.tracked_polarity,
        component_id=tracking.component_id,
        provenance=tracking.provenance,
        ambiguous_mask=tracking.ambiguous_mask,
        dominant_polarity=tracking.dominant_polarity,
        polarity_confidence=tracking.polarity_confidence,
        positive_support=tracking.positive_support,
        negative_support=tracking.negative_support,
        vote_count=tracking.vote_count,
        final_shift=completion.final_shift,
        measurement_success=completion.measurement_success_mask,
        fallback_success=completion.fallback_success_mask,
        final_available=completion.final_available_mask,
    )
    _save_figure(
        stem.with_suffix(".png"),
        np.asarray(receiver_x, dtype=float),
        lags,
        raw,
        lag_valid,
        candidates,
        tracking,
    )
    return stem.with_suffix(".npz")


def _save_figure(path, receiver_x, lags, raw, lag_valid, candidates, tracking):
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    image = np.where(lag_valid, raw, np.nan)
    top.pcolormesh(
        receiver_x / 1000.0,
        lags,
        image.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    for polarity, color, label in ((1, "tab:red", "+"), (-1, "tab:blue", "-")):
        mask = candidates & (np.sign(raw) == polarity)
        rows, columns = np.nonzero(mask)
        if rows.size:
            top.scatter(
                receiver_x[rows] / 1000.0,
                lags[columns],
                s=10,
                color=color,
                label=f"candidate {label}",
            )
    measured = np.asarray(tracking.dp_success_mask, dtype=bool)
    bridge = np.asarray(tracking.bridge_success_mask, dtype=bool)
    top.plot(
        receiver_x / 1000.0,
        np.where(measured, tracking.shift_samples, np.nan),
        "k-",
        lw=1.4,
        label="measured ridge",
    )
    if bridge.any():
        top.plot(
            receiver_x / 1000.0,
            np.where(
                bridge,
                tracking.bridge_geometric_shift_samples,
                np.nan,
            ),
            color="0.45",
            ls=":",
            lw=1.2,
            label="cubic geometric bridge",
        )
        top.plot(
            receiver_x / 1000.0,
            np.where(bridge, tracking.shift_samples, np.nan),
            "m--",
            lw=1.2,
            label="cubic bridge",
        )
    top.plot(
        receiver_x[tracking.seed_receiver] / 1000.0,
        tracking.seed_lag,
        "yo",
        label="seed",
    )
    top.set_ylabel("Lag (samples)")
    top.set_title(
        "fixed polarity={} confidence={:.3f} votes={}".format(
            tracking.dominant_polarity,
            tracking.polarity_confidence,
            tracking.vote_count,
        )
    )
    top.legend(loc="best", fontsize=8)

    bottom.plot(
        receiver_x / 1000.0,
        np.count_nonzero(lag_valid, axis=1),
        label="valid lag cells",
    )
    bottom.plot(
        receiver_x / 1000.0,
        np.count_nonzero(candidates, axis=1),
        label="candidates",
    )
    bottom.plot(
        receiver_x / 1000.0,
        measured.astype(int),
        label="tracked rows",
    )
    if bridge.any():
        bottom.plot(
            receiver_x / 1000.0,
            bridge.astype(int),
            label="bridge rows",
        )
    bottom.set(xlabel="Receiver x (km)", ylabel="Count")
    bottom.legend(loc="best", fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)
