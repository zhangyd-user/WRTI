"""Matplotlib diagnostics that consume existing WRTI result objects.

These functions create or update axes and never call ``plt.show()``.  The
caller decides whether to display or save the returned figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from collections.abc import Mapping

from ..correlation import CorrelationResult
from ..reflector import Reflector
from ..tracking import TrackingResult
from ..tracking import boundary_qc_mask, low_correlation_qc_mask, path_failure_mask
from ..window import WindowResult
from ..workflow.observed_center import DIRECT_TRACKING


def _candidate_failure_mask(tracking, boundary_policy: str) -> np.ndarray:
    """Return only true path failures; boundary policy is compatibility-only."""

    del boundary_policy
    return path_failure_mask(tracking)


def _axis(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


def plot_reflectors_velocity(velocity, reflectors, *, x_axis=None, z_axis=None, ax=None):
    ax = _axis(ax)
    values = np.asarray(velocity, dtype=float)
    if values.ndim != 2:
        raise ValueError("velocity must have shape [nx, nz].")
    if x_axis is None:
        x_axis = np.arange(values.shape[0], dtype=float)
    if z_axis is None:
        z_axis = np.arange(values.shape[1], dtype=float)
    image = ax.imshow(
        values.T,
        aspect="auto",
        extent=[x_axis[0], x_axis[-1], z_axis[-1], z_axis[0]],
    )
    for reflector in reflectors:
        if not isinstance(reflector, Reflector):
            raise ValueError("reflectors must contain Reflector objects.")
        ax.plot(reflector.x, reflector.z, linewidth=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return ax


def plot_theoretical_reflection_curve(
    shot_gather,
    reflection_traveltime,
    reflector: int,
    shot: int,
    dt: float,
    *,
    t0: float = 0.0,
    receiver_x=None,
    ax=None,
):
    ax = _axis(ax)
    gather = np.asarray(shot_gather, dtype=float)
    if gather.ndim != 2:
        raise ValueError("shot_gather must have shape [nr, nt].")
    trace_count, nt = gather.shape
    times = t0 + np.arange(nt, dtype=float) * dt
    x = np.arange(trace_count, dtype=float) if receiver_x is None else np.asarray(receiver_x)
    ax.imshow(
        gather.T,
        aspect="auto",
        extent=[x[0], x[-1], times[-1], times[0]],
    )
    curve = np.asarray(reflection_traveltime, dtype=float)[reflector, shot]
    ax.plot(x, curve, linewidth=1.5)
    ax.set_xlabel("receiver x")
    ax.set_ylabel("time (s)")
    return ax


def plot_selected_window_overlay(
    observed_trace,
    synthetic_trace,
    windows: WindowResult,
    reflector: int,
    shot: int,
    receiver: int,
    *,
    ax=None,
):
    ax = _axis(ax)
    observed = np.asarray(observed_trace, dtype=float)
    synthetic = np.asarray(synthetic_trace, dtype=float)
    if observed.ndim != 1 or synthetic.shape != observed.shape:
        raise ValueError("observed_trace and synthetic_trace must be equal 1-D arrays.")
    if observed.size != windows.nt:
        raise ValueError("trace length must equal windows.nt.")
    time = windows.t0 + np.arange(windows.nt, dtype=float) * windows.dt
    ax.plot(time, observed, label="observed")
    ax.plot(time, synthetic, label="synthetic")
    left = int(windows.left_sample[reflector, shot, receiver])
    right = int(windows.right_sample[reflector, shot, receiver])
    if bool(windows.valid[reflector, shot, receiver]):
        ax.axvspan(time[left], time[right], alpha=0.2)
    ax.axvline(windows.center_time[reflector, shot, receiver], linestyle="--")
    ax.set_xlabel("time (s)")
    ax.legend()
    return ax


def plot_correlation_image(correlation: CorrelationResult, *, receiver_x=None, ax=None):
    ax = _axis(ax)
    values = np.ma.masked_where(
        ~correlation.lag_valid,
        correlation.correlation,
    )
    receiver = (
        np.arange(values.shape[0], dtype=float)
        if receiver_x is None
        else np.asarray(receiver_x)
    )
    ax.imshow(
        values.T,
        aspect="auto",
        origin="lower",
        extent=[
            receiver[0],
            receiver[-1],
            correlation.lags_time[0],
            correlation.lags_time[-1],
        ],
    )
    ax.set_xlabel("receiver")
    ax.set_ylabel("lag (s)")
    return ax


def plot_tracked_ridge(
    correlation: CorrelationResult,
    tracking: TrackingResult,
    *,
    receiver_x=None,
    ax=None,
):
    ax = plot_correlation_image(correlation, receiver_x=receiver_x, ax=ax)
    receiver = (
        np.arange(tracking.shift_time.size, dtype=float)
        if receiver_x is None
        else np.asarray(receiver_x)
    )
    ax.plot(
        receiver,
        correlation.ownership_lag_min,
        color="0.25",
        ls="--",
        lw=0.8,
        label="ownership bounds",
    )
    ax.plot(
        receiver,
        correlation.ownership_lag_max,
        color="0.25",
        ls="--",
        lw=0.8,
    )
    ax.plot(
        receiver,
        correlation.coarse_lag_time,
        color="#D55E00",
        ls=":",
        lw=1.0,
        label="envelope coarse",
    )
    ax.plot(
        receiver,
        tracking.shift_time,
        color="black",
        linewidth=1.5,
        label="waveform DP",
    )
    ax.legend(fontsize=7, frameon=True)
    return ax


def plot_warped_synthetic_comparison(
    observed_trace,
    synthetic_trace,
    warped_synthetic_trace,
    dt: float,
    *,
    t0: float = 0.0,
    ax=None,
):
    ax = _axis(ax)
    observed = np.asarray(observed_trace, dtype=float)
    synthetic = np.asarray(synthetic_trace, dtype=float)
    warped = np.asarray(warped_synthetic_trace, dtype=float)
    if observed.ndim != 1 or synthetic.shape != observed.shape or warped.shape != observed.shape:
        raise ValueError("comparison traces must be equal 1-D arrays.")
    time = t0 + np.arange(observed.size, dtype=float) * dt
    ax.plot(time, observed, label="observed")
    ax.plot(time, synthetic, label="synthetic")
    ax.plot(time, warped, label="warped synthetic")
    ax.set_xlabel("time (s)")
    ax.legend()
    return ax


def save_vfsa_diagnostic(
    output_path,
    *,
    reference_state,
    observed_data,
    candidate_data,
    correlations: Mapping,
    trackings: Mapping,
    evaluation: int,
    misfit: float,
    failure_count: int,
    path_failure_mask_array=None,
    low_correlation_qc_mask_array=None,
    boundary_qc_mask_array=None,
    shot: int | None = None,
    dpi: int = 180,
):
    """Save one representative WRTI diagnostic for a VFSA evaluation.

    The WRTI evaluator retains the configured representative
    ``(reflector, shot)`` correlation/tracking results.  This function uses
    those results and the already processed gathers, so plotting does not
    rerun preprocessing, correlation, tracking, or Eikonal calculations.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    observed = np.asarray(observed_data, dtype=float)
    candidate = np.asarray(candidate_data, dtype=float)
    if observed.shape != candidate.shape or observed.ndim != 3:
        raise ValueError("observed_data and candidate_data must have shape [ns, nr, nt].")

    nref, nshot, nreceiver = reference_state.shape
    supplied_masks = (
        path_failure_mask_array,
        low_correlation_qc_mask_array,
        boundary_qc_mask_array,
    )
    if any(value is not None for value in supplied_masks):
        if not all(value is not None for value in supplied_masks):
            raise ValueError(
                "All three failure/QC masks must be supplied together."
            )
        supplied_masks = tuple(
            np.asarray(value, dtype=bool) for value in supplied_masks
        )
        if any(value.shape != reference_state.shape for value in supplied_masks):
            raise ValueError(
                "Failure/QC masks must match reference_state.shape."
            )
    available = {}
    for key, correlation in correlations.items():
        reflector, shot_index = (int(key[0]), int(key[1]))
        if (reflector, shot_index) in trackings:
            available[(reflector, shot_index)] = (correlation, trackings[key])
    if not available:
        raise ValueError("No representative WRTI correlation/tracking result was retained.")

    available_shots = sorted({key[1] for key in available})
    if shot is None:
        shot = min(available_shots, key=lambda value: abs(value - (nshot - 1) / 2.0))
    shot = int(shot)
    if shot not in available_shots:
        shot = available_shots[0]

    receiver_x = np.asarray(reference_state.receiver_coordinates[shot, :, 0]) / 1000.0
    time = reference_state.windows.t0 + np.arange(reference_state.windows.nt) * reference_state.windows.dt
    if reference_state.observed_windows is None:
        raise ValueError("reference_state has no observed_windows.")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    recovered_all = np.asarray(reference_state.recovered_mask, dtype=bool)
    center_source = np.asarray(reference_state.observed_center_source, dtype=int)

    def draw_gather(ax, gather, title, *, role):
        clip = float(np.percentile(np.abs(gather), 99.5))
        clip = max(clip, np.finfo(float).eps)
        ax.imshow(
            gather.T,
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="nearest",
            extent=[receiver_x[0], receiver_x[-1], time[-1], time[0]],
        )
        for reflector in range(nref):
            color = colors[reflector % len(colors)]
            reference_center = reference_state.windows.center_time[reflector, shot]
            valid = reference_state.observed_windows.valid[reflector, shot]
            center = reference_state.observed_windows.center_time[reflector, shot]
            good = valid & np.isfinite(center)
            recovered = recovered_all[reflector, shot]
            direct = center_source[reflector, shot] == DIRECT_TRACKING
            reference_good = np.isfinite(reference_center)
            if role == "observed":
                half = reference_state.observed_windows.half_window_time[reflector]
                ax.fill_between(
                    receiver_x,
                    center - half,
                    center + half,
                    where=good,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
                ax.plot(
                    receiver_x[reference_good],
                    reference_center[reference_good],
                    ":",
                    color=color,
                    lw=0.8,
                )
                ax.plot(receiver_x[good], center[good], "-", color=color, lw=1.1)
                recovered_good = good & recovered
                ax.scatter(
                    receiver_x[recovered_good],
                    center[recovered_good],
                    marker="^",
                    s=14,
                    color=color,
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=6,
                )
            item = available.get((reflector, shot))
            if item is None:
                continue
            _, tracking = item
            raw_shift = np.asarray(tracking.shift_time, dtype=float)
            success = tracking.success_mask & np.isfinite(raw_shift) & good
            candidate_failure = _candidate_failure_mask(
                tracking,
                reference_state.config.misfit.candidate_boundary_policy,
            )
            failed = reference_state.fixed_mask[reflector, shot] & candidate_failure

            if role == "observed":
                ax.plot(receiver_x[good], center[good], color=color, lw=1.1)
                ax.scatter(
                    receiver_x[good & direct][::5],
                    center[good & direct][::5],
                    s=7,
                    color=color,
                    edgecolor="white",
                    linewidth=0.2,
                    zorder=5,
                )
            else:
                # shift_time = T_obs - T_syn, so candidate synthetic events
                # lie at fixed_observed_center - tracked_shift.
                plotted = np.where(success, center - raw_shift, np.nan)
                ax.plot(receiver_x, plotted, color=color, lw=1.2)

            if role == "candidate":
                ax.scatter(
                    receiver_x[failed],
                    center[failed],
                    marker="x",
                    s=10,
                    color=color,
                    linewidth=0.7,
                    zorder=6,
                )
        ax.set_xlim(receiver_x[0], receiver_x[-1])
        ax.set_ylim(time[-1], max(0.25, time[0]))
        ax.set_xlabel("Receiver x (km)")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)

    figure = plt.figure(figsize=(15.2, 11.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.45, 1.2, 1.0))
    observed_ax = figure.add_subplot(grid[0, :2])
    candidate_ax = figure.add_subplot(grid[0, 2:])
    draw_gather(
        observed_ax,
        observed[shot],
        "Observed processed gather: fixed observed center",
        role="observed",
    )
    draw_gather(
        candidate_ax,
        candidate[shot],
        "Candidate processed gather: observed center - tracked lag",
        role="candidate",
    )

    correlation_axes = [figure.add_subplot(grid[1, index]) for index in range(4)]
    image = None
    for reflector, ax in enumerate(correlation_axes):
        item = available.get((reflector, shot))
        if item is None:
            ax.set_axis_off()
            continue
        correlation, tracking = item
        # CorrelationResult.correlation is [nreceiver, nlag].  Put receiver
        # position on the horizontal axis and lag on the vertical axis.
        # Keep the complete global +/- max-lag axis, but hide states rejected
        # by the reflector-ownership and envelope/fine gates.
        values = np.ma.masked_where(
            ~correlation.lag_valid,
            np.ma.masked_invalid(correlation.correlation),
        )
        image = ax.imshow(
            values.T,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                receiver_x[0],
                receiver_x[-1],
                correlation.lags_time[0] * 1000.0,
                correlation.lags_time[-1] * 1000.0,
            ],
        )
        success = tracking.success_mask & np.isfinite(tracking.shift_time)
        own_min = np.asarray(correlation.ownership_lag_min, dtype=float)
        own_max = np.asarray(correlation.ownership_lag_max, dtype=float)
        coarse = np.asarray(correlation.coarse_lag_time, dtype=float)
        ax.plot(
            receiver_x,
            own_min * 1000.0,
            color="white",
            ls="--",
            lw=0.8,
            alpha=0.95,
            label="ownership bounds" if reflector == 0 else None,
        )
        ax.plot(
            receiver_x,
            own_max * 1000.0,
            color="white",
            ls="--",
            lw=0.8,
            alpha=0.95,
        )
        ax.plot(
            receiver_x,
            coarse * 1000.0,
            color="#F0E442",
            ls=":",
            lw=1.0,
            label="envelope coarse" if reflector == 0 else None,
        )
        ax.plot(
            receiver_x,
            np.where(success, tracking.shift_time * 1000.0, np.nan),
            color="black",
            lw=1.2,
            label="waveform DP" if reflector == 0 else None,
        )
        boundary = success & tracking.boundary_flag
        ax.scatter(
            receiver_x[boundary],
            tracking.shift_time[boundary] * 1000.0,
            marker="x",
            s=12,
            color="#F0E442",
            linewidth=0.7,
        )
        ax.axhline(0.0, color="white", lw=0.5, alpha=0.7)
        adjacent = np.isfinite(coarse[:-1]) & np.isfinite(coarse[1:])
        max_adjacent_jump = (
            float(np.max(np.abs(np.diff(coarse)[adjacent])) * 1000.0)
            if np.any(adjacent)
            else float("nan")
        )
        seed_receiver = int(correlation.coarse_seed_receiver)
        seed_text = (
            f"x={receiver_x[seed_receiver]:.3g} km, "
            f"lag={correlation.coarse_seed_lag_time * 1000.0:.1f} ms"
            if 0 <= seed_receiver < receiver_x.size
            else "none"
        )
        ax.set_title(
            f"R{reflector + 1} ZNCC / tracking\n"
            f"coarse seed {seed_text}; "
            f"eps={reference_state.config.envelope_tracking_epsilon_time * 1000.0:.0f} ms; "
            f"fallback={correlation.coarse_tracking_fallback_count}; "
            f"max Δ={max_adjacent_jump:.1f} ms",
            fontsize=8,
        )
        ax.set_xlabel("Receiver x (km)")
        if reflector == 0:
            ax.set_ylabel("Lag (ms)")
            ax.legend(loc="upper right", fontsize=6, frameon=True)
    if image is not None:
        figure.colorbar(image, ax=correlation_axes, shrink=0.82, pad=0.012, label="ZNCC")

    residual_ax = figure.add_subplot(grid[2, :2])
    quality_ax = figure.add_subplot(grid[2, 2:])
    for reflector in range(nref):
        item = available.get((reflector, shot))
        if item is None:
            continue
        _, tracking = item
        color = colors[reflector % len(colors)]
        success = tracking.success_mask & np.isfinite(tracking.shift_time)
        candidate_failure = _candidate_failure_mask(
            tracking,
            reference_state.config.misfit.candidate_boundary_policy,
        )
        included = reference_state.fixed_mask[reflector, shot] & ~candidate_failure
        residual_ax.plot(
            receiver_x,
            np.where(success, tracking.shift_time * 1000.0, np.nan),
            color=color,
            lw=1.1,
            label=f"R{reflector + 1}",
        )
        residual_ax.scatter(
            receiver_x[included][::5],
            tracking.shift_time[included][::5] * 1000.0,
            color=color,
            s=7,
        )
        quality = success & np.isfinite(tracking.tracked_correlation)
        quality_ax.plot(
            receiver_x,
            np.where(quality, tracking.tracked_correlation, np.nan),
            color=color,
            lw=1.1,
            label=f"R{reflector + 1}",
        )
    residual_ax.axhline(0.0, color="0.5", lw=0.7)
    residual_ax.set_xlabel("Receiver x (km)")
    residual_ax.set_ylabel("Tracked shift (ms)")
    residual_ax.set_title("Traveltime residual")
    residual_ax.legend(ncol=4, fontsize=7, frameon=False)
    qc_threshold = reference_state.config.misfit.min_correlation
    if qc_threshold is not None:
        quality_ax.axhline(
            qc_threshold,
            color="0.25",
            ls="--",
            lw=0.8,
            label="QC threshold",
        )
    quality_ax.set_ylim(-1.05, 1.05)
    quality_ax.set_xlabel("Receiver x (km)")
    quality_ax.set_ylabel("Tracked ZNCC")
    quality_ax.set_title("Tracking quality")
    quality_ax.legend(ncol=5, fontsize=7, frameon=False)

    fixed_count = int(np.count_nonzero(reference_state.fixed_mask[:, shot]))
    shot_recovered_count = int(np.count_nonzero(recovered_all[:, shot]))
    shot_failure_count = 0
    shot_low_corr_count = 0
    shot_boundary_count = 0
    reflector_breakdown = []
    for reflector in range(nref):
        item = available.get((reflector, shot))
        if item is None:
            continue
        _, tracking = item
        fixed = reference_state.fixed_mask[reflector, shot]
        if path_failure_mask_array is None:
            path_mask = path_failure_mask(tracking)
            low_mask = low_correlation_qc_mask(
                tracking,
                reference_state.config.misfit.min_correlation,
            )
            boundary_mask = boundary_qc_mask(tracking)
        else:
            path_mask = supplied_masks[0][reflector, shot]
            low_mask = supplied_masks[1][reflector, shot]
            boundary_mask = supplied_masks[2][reflector, shot]
        shot_failure_count += int(np.count_nonzero(fixed & path_mask))
        shot_low_corr_count += int(
            np.count_nonzero(fixed & low_mask)
        )
        shot_boundary_count += int(
            np.count_nonzero(fixed & boundary_mask)
        )
        reflector_breakdown.append(
            f"R{reflector + 1}: fixed={int(np.count_nonzero(fixed))}, "
            f"path={int(np.count_nonzero(fixed & path_mask))}, "
            f"low={int(np.count_nonzero(fixed & low_mask))}, "
            f"boundary={int(np.count_nonzero(fixed & boundary_mask))}, "
            f"recovered={int(np.count_nonzero(recovered_all[reflector, shot]))}"
        )
    figure.suptitle(
        f"WRTI evaluation {int(evaluation):03d} | shot {shot + 1} | "
        f"sum_misfit={float(misfit):.6e} | "
         f"shot path failures={shot_failure_count}/{fixed_count} | "
         f"shot recovered centers={shot_recovered_count} | "
         f"low-corr QC={shot_low_corr_count} | "
        f"boundary QC={shot_boundary_count} | "
        f"global path failures={int(failure_count)}\n"
        + " | ".join(reflector_breakdown),
        fontsize=11,
        fontweight="bold",
    )
    output_path = str(output_path)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_vfsa_diagnostics(
    output_dir,
    *,
    reference_state,
    observed_data,
    candidate_data,
    correlations: Mapping,
    trackings: Mapping,
    evaluation: int,
    misfit: float,
    failure_count: int,
    path_failure_mask_array=None,
    low_correlation_qc_mask_array=None,
    boundary_qc_mask_array=None,
    shots=None,
    dpi: int = 180,
):
    """Save PNG diagnostics for five evenly spaced shots.

    The first and last acquisition shots are always included.  No PDF or SVG
    files are produced by this VFSA-loop helper.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shots is None:
        shot_count = int(reference_state.shape[1])
        shots = np.rint(np.linspace(0, shot_count - 1, min(5, shot_count))).astype(int)
    shots = tuple(dict.fromkeys(int(shot) for shot in shots))
    paths = []
    for shot in shots:
        path = output_dir / (
            f"wrtidianostic_eval_{int(evaluation):03d}_shot_{shot + 1:03d}.png"
        )
        paths.append(
            save_vfsa_diagnostic(
                path,
                reference_state=reference_state,
                observed_data=observed_data,
                candidate_data=candidate_data,
                correlations=correlations,
                trackings=trackings,
                evaluation=evaluation,
                misfit=misfit,
                failure_count=failure_count,
                path_failure_mask_array=path_failure_mask_array,
                low_correlation_qc_mask_array=low_correlation_qc_mask_array,
                boundary_qc_mask_array=boundary_qc_mask_array,
                shot=shot,
                dpi=dpi,
            )
        )
    return tuple(paths)
