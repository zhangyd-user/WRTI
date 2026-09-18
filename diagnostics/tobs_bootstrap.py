"""Diagnostic plot for observed-center bootstrap comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..workflow.tobs_bootstrap import flatten_observed_gather


def _display_limit(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, 99.0))
    return limit if limit > 0.0 else 1.0


def save_tobs_bootstrap_diagnostic(
    output_path,
    *,
    observed_shot,
    theoretical_traveltime,
    old_observed_center,
    new_observed_center,
    receiver_x,
    control_receiver,
    control_time,
    eikonal_control_time=None,
    dt,
    t0=0.0,
    title=None,
    rescue_used_mask=None,
    ownership_escape_used_mask=None,
    skipped_valid_receiver_mask=None,
    stop_receivers=(),
    dpi=180,
):
    """Save raw and Eikonal-flattened gathers with tracked Tobs overlays."""

    observed = np.asarray(observed_shot, dtype=float)
    tref = np.asarray(theoretical_traveltime, dtype=float)
    new_tobs = np.asarray(new_observed_center, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    if observed.ndim != 2:
        raise ValueError("observed_shot must have shape [nreceiver, ntime].")
    nr, nt = observed.shape
    if any(value.shape != (nr,) for value in (tref, new_tobs, x)):
        raise ValueError("traveltime, centers, and receiver_x must match nreceiver.")
    if old_observed_center is not None:
        old_tobs = np.asarray(old_observed_center, dtype=float)
        if old_tobs.shape != (nr,):
            raise ValueError("old_observed_center must match nreceiver when provided.")
    i0 = int(control_receiver)
    if not 0 <= i0 < nr:
        raise ValueError("control_receiver is outside the receiver axis.")
    old_control = float(tref[i0]) if eikonal_control_time is None else float(eikonal_control_time)

    flat, moveout, _ = flatten_observed_gather(
        observed,
        tref,
        seed_receiver=i0,
        dt=float(dt),
        t0=float(t0),
    )
    flat_pick = new_tobs - moveout
    time_end = float(t0) + (nt - 1) * float(dt)
    extent = [float(x[0]), float(x[-1]), time_end, float(t0)]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, gather, label in zip(
        axes,
        (observed, flat),
        ("Original observed gather", "Eikonal-flattened observed gather"),
    ):
        limit = _display_limit(gather)
        ax.imshow(
            gather.T,
            cmap="gray",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_xlabel("Receiver x")
        ax.grid(False)

    axes[0].plot(x, tref, color="tab:blue", linewidth=1.5, label="Eikonal Tref")
    axes[0].plot(x, new_tobs, color="tab:red", linewidth=1.5, label="Bootstrap Tobs")
    axes[0].scatter(x[i0], control_time, s=45, facecolor="white", edgecolor="black", zorder=5,
                    label="Same-x control / DP seed")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].axhline(old_control, color="gold", linestyle="--", linewidth=1.2,
                    label="Eikonal global-min control")
    axes[1].axhline(float(control_time), color="tab:blue", linewidth=1.4,
                    label="Same-x control / DP seed")
    axes[1].plot(x, flat_pick, color="tab:red", linewidth=1.5, label="Tracked flat path")
    if rescue_used_mask is not None:
        rescue = np.asarray(rescue_used_mask, dtype=bool)
        axes[1].scatter(x[rescue], flat_pick[rescue], marker="^", s=45,
                        color="tab:green", zorder=6, label="Predictive rescue")
    if ownership_escape_used_mask is not None:
        escaped = np.asarray(ownership_escape_used_mask, dtype=bool)
        axes[1].scatter(x[escaped], flat_pick[escaped], marker="s", s=24,
                        facecolor="none", edgecolor="tab:purple", zorder=6,
                        label="Ownership escape")
    if skipped_valid_receiver_mask is not None:
        gaps = np.asarray(skipped_valid_receiver_mask, dtype=bool)
        axes[1].scatter(x[gaps], np.full(np.count_nonzero(gaps), float(control_time)),
                        marker="x", s=45, color="tab:orange", zorder=6,
                        label="Skipped valid receiver")
    for stop in stop_receivers:
        if 0 <= int(stop) < nr:
            axes[1].axvline(x[int(stop)], color="tab:purple", linestyle=":", linewidth=1.2,
                            label="Stop receiver")
    axes[1].scatter(x[i0], control_time, s=45, facecolor="white", edgecolor="black", zorder=5,
                    label="Same-x control")
    axes[1].legend(loc="best", fontsize=8)

    if title:
        figure.suptitle(str(title))
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return path


def save_same_x_control_test(
    output_path,
    *,
    observed_shot,
    theoretical_traveltime,
    new_observed_center,
    receiver_x,
    control_receiver,
    eikonal_control_time,
    same_x_control_time,
    tracking_seed_time=None,
    guide_pick_time,
    legacy_pick_time,
    dt,
    t0=0.0,
    dpi=180,
):
    """Save the four-reflector same-x seed experiment as a 4-by-2 figure."""

    observed = np.asarray(observed_shot, dtype=float)
    tref = np.asarray(theoretical_traveltime, dtype=float)
    tobs = np.asarray(new_observed_center, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    old_control = np.asarray(eikonal_control_time, dtype=float)
    new_control = np.asarray(same_x_control_time, dtype=float)
    snapped_control = new_control if tracking_seed_time is None else np.asarray(tracking_seed_time, dtype=float)
    guide_time = np.asarray(guide_pick_time, dtype=float)
    legacy_time = np.asarray(legacy_pick_time, dtype=float)
    nref, nreceiver = tref.shape
    if observed.ndim != 2 or observed.shape[0] != nreceiver or tobs.shape != tref.shape:
        raise ValueError("same-x diagnostic gather and time arrays have inconsistent shapes")
    if (old_control.shape != (nref,) or new_control.shape != (nref,) or snapped_control.shape != (nref,)
            or guide_time.shape != tref.shape or legacy_time.shape != tref.shape):
        raise ValueError("control-time arrays must contain one value per reflector")

    i0 = int(control_receiver)
    time_end = float(t0) + (observed.shape[1] - 1) * float(dt)
    extent = [float(x[0]), float(x[-1]), time_end, float(t0)]
    figure, axes = plt.subplots(nref, 2, figsize=(14, 3.2 * nref), sharex=True)
    axes = np.atleast_2d(axes)
    for reflector in range(nref):
        flat, moveout, _ = flatten_observed_gather(
            observed, tref[reflector], seed_receiver=i0, dt=dt, t0=t0
        )
        flat_pick = tobs[reflector] - moveout
        for ax, gather in zip(axes[reflector], (observed, flat)):
            limit = _display_limit(gather)
            ax.imshow(gather.T, cmap="gray", vmin=-limit, vmax=limit,
                      aspect="auto", extent=extent, interpolation="nearest")
            ax.grid(False)
        left, right = axes[reflector]
        left.plot(x, tref[reflector], color="#0072B2", linewidth=1.4,
                  label="Eikonal Tref + shift")
        left.plot(x, tobs[reflector], color="#D55E00", linewidth=1.3,
                  label="Bootstrap Tobs_new")
        left.scatter(x[i0], old_control[reflector], marker="x", s=45, color="#E69F00",
                     label="Old Eikonal control", zorder=5)
        left.scatter(x[i0], new_control[reflector], s=50, facecolor="white",
                     edgecolor="black", label="Same-x DP seed", zorder=5)
        left.scatter(x[i0], snapped_control[reflector], s=38, color="#F0E442",
                     edgecolor="black", label="Observed snapped seed", zorder=6)
        right.axhline(old_control[reflector], color="#E69F00", linestyle="--",
                      linewidth=1.3, label="Old Eikonal control")
        right.axhline(new_control[reflector], color="#0072B2", linewidth=1.5,
                      label="Theoretical same-x control")
        right.scatter(x[i0], snapped_control[reflector], s=38, color="#F0E442",
                      edgecolor="black", label="Observed snapped seed", zorder=6)
        right.plot(x, legacy_time[reflector], color="0.55", linestyle=":",
                   linewidth=1.1, label="Old envelope DP path")
        right.plot(x, guide_time[reflector], color="#009E73", linestyle="--",
                   linewidth=1.3, label="Envelope guide path")
        right.plot(x, flat_pick, color="#D55E00", linewidth=1.3,
                   label="Refined final path")
        center = float(new_control[reflector])
        left.set_ylim(center + 0.25, center - 0.25)
        right.set_ylim(center + 0.25, center - 0.25)
        left.set_ylabel(f"R{reflector + 1}\nTime (s)")
        if reflector == 0:
            left.set_title("Original observed gather")
            right.set_title("Eikonal relative-moveout flattened gather")
            left.legend(loc="upper right", fontsize=7)
            right.legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("Receiver x (m)")
    axes[-1, 1].set_xlabel("Receiver x (m)")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return path


def save_sparse_event_dp_qc(
    output_path,
    *,
    receiver_x,
    neighbor_zncc,
    residual_slope,
    prediction_error,
    candidate_count,
    candidate_count_before=None,
    ownership_upper=None,
    ownership_lower=None,
    dpi=180,
):
    """Save per-reflector sparse-DP transition and candidate diagnostics."""

    x = np.asarray(receiver_x, dtype=float)
    before = np.asarray(candidate_count if candidate_count_before is None else candidate_count_before, dtype=float)
    after = np.asarray(candidate_count, dtype=float)
    width = (np.asarray(ownership_lower) - np.asarray(ownership_upper)) if ownership_upper is not None else np.full_like(after, np.nan)
    values = [
        np.asarray(neighbor_zncc, dtype=float),
        1e5 * np.asarray(residual_slope, dtype=float),
        1e3 * np.asarray(prediction_error, dtype=float),
        after,
        1e3 * width,
    ]
    labels = ["Neighbor ZNCC", "Slope (ms/100 m)", "Prediction error (ms)", "Candidates", "Corridor width (ms)"]
    nref = values[0].shape[0]
    figure, axes = plt.subplots(nref, 5, figsize=(19, 2.8 * nref), sharex=True)
    axes = np.atleast_2d(axes)
    for reflector in range(nref):
        for column, (data, label) in enumerate(zip(values, labels)):
            axes[reflector, column].plot(x, data[reflector], color="#0072B2", linewidth=1.0)
            if column == 3:
                axes[reflector, column].plot(x, before[reflector], color="0.55", linewidth=0.9, label="before")
                axes[reflector, column].legend(fontsize=7)
            axes[reflector, column].axhline(0.0, color="0.75", linewidth=0.7)
            if reflector == 0:
                axes[reflector, column].set_title(label)
            if column == 0:
                axes[reflector, column].set_ylabel(f"R{reflector + 1}")
    for ax in axes[-1]:
        ax.set_xlabel("Receiver x (m)")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return path


def save_ownership_sparse_dp(output_path, *, observed_shot, theoretical_traveltime,
                             final_tobs, receiver_x, control_receiver, control_time,
                             tracking_seed_time,
                             ownership_upper, ownership_lower, guide_pick_time,
                             dt, t0=0.0, dpi=180):
    """Show original/flattened ownership corridors and the selected sparse path."""
    observed = np.asarray(observed_shot, dtype=float)
    tref = np.asarray(theoretical_traveltime, dtype=float)
    final = np.asarray(final_tobs, dtype=float)
    upper = np.asarray(ownership_upper, dtype=float)
    lower = np.asarray(ownership_lower, dtype=float)
    guide = np.asarray(guide_pick_time, dtype=float)
    snapped = np.asarray(tracking_seed_time, dtype=float)
    x = np.asarray(receiver_x, dtype=float)
    i0 = int(control_receiver)
    nref = tref.shape[0]
    extent = [x[0], x[-1], t0 + (observed.shape[1] - 1) * dt, t0]
    figure, axes = plt.subplots(nref, 2, figsize=(14, 3.2 * nref), sharex=True)
    axes = np.atleast_2d(axes)
    for reflector in range(nref):
        flat, moveout, _ = flatten_observed_gather(observed, tref[reflector], seed_receiver=i0, dt=dt, t0=t0)
        for ax, gather in zip(axes[reflector], (observed, flat)):
            limit = _display_limit(gather)
            ax.imshow(gather.T, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", extent=extent)
        left, right = axes[reflector]
        left.plot(x, control_time[reflector] + moveout, "#0072B2", lw=1.1, label="Pk")
        left.plot(x, upper[reflector], "--", color="#CC79A7", lw=1.0, label="ownership")
        left.plot(x, lower[reflector], "--", color="#CC79A7", lw=1.0)
        left.plot(x, final[reflector], "#D55E00", lw=1.3, label="final Tobs")
        right.plot(x, upper[reflector] - moveout, "--", color="#CC79A7", lw=1.0)
        right.plot(x, lower[reflector] - moveout, "--", color="#CC79A7", lw=1.0)
        right.scatter(x, guide[reflector], s=4, color="#009E73", label="selected candidates")
        right.plot(x, final[reflector] - moveout, "#D55E00", lw=1.2, label="final path")
        left.scatter(x[i0], control_time[reflector], s=35, facecolor="white", edgecolor="black", zorder=5)
        right.scatter(x[i0], control_time[reflector], s=35, facecolor="white", edgecolor="black", zorder=5)
        left.scatter(x[i0], snapped[reflector], s=32, color="#F0E442", edgecolor="black", zorder=6)
        right.scatter(x[i0], snapped[reflector], s=32, color="#F0E442", edgecolor="black", zorder=6)
        left.set_ylabel(f"R{reflector + 1}\nTime (s)")
        if reflector == 0:
            left.set_title("Original gather and ownership")
            right.set_title("Flattened gather and sparse DP")
            left.legend(fontsize=7)
            right.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Receiver x (m)")
    axes[-1, 1].set_xlabel("Receiver x (m)")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = ["save_ownership_sparse_dp", "save_same_x_control_test", "save_sparse_event_dp_qc", "save_tobs_bootstrap_diagnostic"]
