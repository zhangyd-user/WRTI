"""Outer-stage construction of the frozen WRTI reference state."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from ..correlation import CorrelationResult, PreprocessHook
from ..misfit import FixedMaskResult, build_fixed_mask_from_config
from ..reflector import (
    GridMappingResult,
    Reflector,
    map_reflectors_to_grid,
    read_reflectors,
)
from ..tracking import TrackingResult
from ..traveltime import ReflectionTraveltimeResult, compute_reflection_traveltimes
from ..window import WindowOverlapWarning, WindowResult, build_window_result
from ..diagnostics.plots import save_reference_center_diagnostics

from .config import WRTIConfig
from .geometry import (
    GeometryError,
    normalize_receiver_coordinates,
    normalize_source_coordinates,
)
from .parallel import (
    ShotTrackingTask,
    run_shot_tasks,
    slice_windows_for_shot,
)
from .observed_center import recover_edge_observed_centers
from .state import WRTIReferenceState


class WorkflowBuildError(RuntimeError):
    """Raised when an outer reference state cannot be constructed."""


class OuterStageWindowBuilder:
    """Build Steps 1–6 once after migration/reflector update.

    The builder has no forward-model or VFSA dependency.  ``traveltime_table``
    is an optional Step 2 backend injection point, normally PyLops on Linux.
    """

    def __init__(
        self,
        config: WRTIConfig,
        *,
        traveltime_table=None,
        preprocess_hook: PreprocessHook | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.traveltime_table = traveltime_table
        self.preprocess_hook = preprocess_hook
        self.logger = logger or logging.getLogger("WRTI.workflow")

    def _observed_data_for_recovery(
        self,
        observed_data: np.ndarray,
        reference_synthetic_data: np.ndarray,
    ) -> np.ndarray:
        """Return the same observed representation used by reference tracking."""

        observed = np.asarray(observed_data, dtype=float)
        if self.preprocess_hook is None:
            return observed
        synthetic = np.asarray(reference_synthetic_data, dtype=float)
        if observed.ndim != 3 or synthetic.shape != observed.shape:
            raise WorkflowBuildError(
                "observed and reference synthetic data must have equal [ns, nr, nt] shapes."
            )
        processed_observed = np.empty_like(observed)
        for shot in range(observed.shape[0]):
            observed_shot = observed[shot : shot + 1]
            processed = self.preprocess_hook(
                observed_shot, synthetic[shot : shot + 1]
            )
            if processed is None:
                result = observed_shot
            elif isinstance(processed, (tuple, list)) and len(processed) == 2:
                result = np.asarray(processed[0], dtype=float)
            else:
                raise WorkflowBuildError(
                    "preprocess_hook must return (observed, synthetic) or None."
                )
            if result.shape != observed_shot.shape:
                raise WorkflowBuildError(
                    "preprocess_hook returned observed data with an invalid shape."
                )
            processed_observed[shot] = result[0]
        if not np.isfinite(processed_observed).all():
            raise WorkflowBuildError(
                "preprocess_hook returned non-finite observed data."
            )
        return processed_observed

    def _load_reflectors(
        self,
        reflector_source: str | Path | Iterable[Reflector],
    ) -> tuple[Reflector, ...]:
        x_axis = self.config.grid.x_axis
        z_axis = self.config.grid.z_axis
        bounds = (float(x_axis[0]), float(x_axis[-1])), (
            float(z_axis[0]),
            float(z_axis[-1]),
        )
        if isinstance(reflector_source, (str, Path)):
            reflectors = read_reflectors(
                reflector_source,
                x_bounds=bounds[0],
                z_bounds=bounds[1],
            )
        else:
            reflectors = tuple(reflector_source)
            if not reflectors or not all(isinstance(item, Reflector) for item in reflectors):
                raise WorkflowBuildError(
                    "reflector_source must be a path or an iterable of Reflector objects."
                )
        return reflectors

    def _build_reference_tracking(
        self,
        observed_data: np.ndarray,
        reference_synthetic_data: np.ndarray,
        windows: WindowResult,
        source_coordinates: np.ndarray,
        receiver_coordinates: np.ndarray,
    ) -> tuple[
        FixedMaskResult,
        tuple[tuple[TrackingResult, ...], ...],
        dict[tuple[int, int], CorrelationResult],
        np.ndarray,
    ]:
        nref, ns, nr = windows.shape
        tracked_shift = np.full((nref, ns, nr), np.nan, dtype=float)
        tracked_correlation = np.full((nref, ns, nr), np.nan, dtype=float)
        energy_obs = np.full((nref, ns, nr), np.nan, dtype=float)
        energy_syn = np.full((nref, ns, nr), np.nan, dtype=float)
        boundary_flag = np.zeros((nref, ns, nr), dtype=bool)
        tracking_success = np.zeros((nref, ns, nr), dtype=bool)
        trace_valid = np.zeros((nref, ns, nr), dtype=bool)
        dp_success = np.zeros((nref, ns, nr), dtype=bool)
        bridge_success = np.zeros((nref, ns, nr), dtype=bool)
        fallback_success = np.zeros((nref, ns, nr), dtype=bool)
        candidate_count = np.zeros((nref, ns, nr), dtype=int)
        reference_tracking: list[list[TrackingResult | None]] = [
            [None for _ in range(ns)] for _ in range(nref)
        ]
        saved_correlations: dict[tuple[int, int], CorrelationResult] = {}
        selected = set(self.config.save_correlation_for)
        selected.update(
            (reflector, shot)
            for shot in self.config.debug_tracking_shots
            if 0 <= shot < ns
            for reflector in range(nref)
        )
        tasks = tuple(
            ShotTrackingTask(
                shot_index=shot,
                observed_shot=observed_data[shot],
                synthetic_shot=reference_synthetic_data[shot],
                windows=slice_windows_for_shot(windows, shot),
                source_coordinates=source_coordinates[shot],
                receiver_coordinates=receiver_coordinates[shot],
                max_lag_time=self.config.max_lag_time,
                seed_lag_range_time=self.config.seed_lag_range_time,
                epsilon_time=self.config.epsilon_time,
                dt=windows.dt,
                use_envelope_coarse=self.config.use_envelope_coarse,
                envelope_fine_half_width_time=self.config.envelope_fine_half_width_time,
                envelope_tracking_epsilon_time=self.config.envelope_tracking_epsilon_time,
                tracking_min_correlation=self.config.tracking_min_correlation,
                boundary_margin_samples=self.config.boundary_margin_samples,
                selected_reflectors=tuple(
                    reflector
                    for reflector in range(nref)
                    if (reflector, shot) in selected
                ),
                preprocess_hook=self.preprocess_hook,
                fixed_side="synthetic",
                tracking_enhancement_enabled=self.config.tracking_enhancement_enabled,
                tracking_agc_fraction=self.config.tracking_agc_fraction,
                tracking_agc_floor_ratio=self.config.tracking_agc_floor_ratio,
                tracking_receiver_stack=self.config.tracking_receiver_stack,
                tracking_raw_refine_radius_samples=self.config.tracking_raw_refine_radius_samples,
                ownership_guard_time=self.config.ownership_guard_time,
                tracking_top_k_peaks=self.config.tracking_top_k_peaks,
                tracking_peak_min_separation_samples=self.config.tracking_peak_min_separation_samples,
                tracking_coarse_soft_width_time=self.config.tracking_coarse_soft_width_time,
                tracking_coarse_soft_weight=self.config.tracking_coarse_soft_weight,
                tracking_coarse_soft_penalty_cap=self.config.tracking_coarse_soft_penalty_cap,
                tracking_smooth_weight=self.config.tracking_smooth_weight,
                tracking_max_residual_jump_time=self.config.tracking_max_residual_jump_time,
                tracking_max_skip_rows=self.config.tracking_max_skip_rows,
                tracking_gap_penalty=self.config.tracking_gap_penalty,
                tracking_correlation_mode=self.config.tracking_correlation_mode,
                tracking_tracker_mode=self.config.tracking_tracker_mode,
                # Bootstrap freezes the first observed centers.  Only the
                # seed-connected DP and verified raw-data bridge may enter it.
                tracking_restart_enabled=False,
                tracking_restart_confirm_rows=self.config.tracking_restart_confirm_rows,
                tracking_restart_min_mean_correlation=self.config.tracking_restart_min_mean_correlation,
                tracking_restart_max_coarse_deviation_time=self.config.tracking_restart_max_coarse_deviation_time,
                local_search_half_width_time=self.config.local_search_half_width_time,
                max_search_half_width_time=self.config.max_search_half_width_time,
                gap_expand_time=self.config.gap_expand_time,
                residual_history=self.config.residual_history,
                min_peak_margin=self.config.min_peak_margin,
                prediction_weight=self.config.prediction_weight,
                max_gap_rows=self.config.max_gap_rows,
                relock_confirm_rows=self.config.relock_confirm_rows,
                restart_min_correlation=self.config.restart_min_correlation,
                bridge_enabled=self.config.bridge_enabled,
                bridge_half_width_time=self.config.bridge_half_width_time,
                bridge_max_width_time=self.config.bridge_max_width_time,
                bridge_max_receivers=self.config.bridge_max_receivers,
                bridge_max_distance=self.config.bridge_max_distance,
                debug_tracking_shots=self.config.debug_tracking_shots,
                debug_tracking_reflectors=self.config.debug_tracking_reflectors,
                save_tracking_snapshot=self.config.save_tracking_snapshot,
                diagnostics_output_dir=self.config.diagnostics_output_dir,
            )
            for shot in range(ns)
        )
        for shot_result in run_shot_tasks(tasks, self.config.wrti_workers):
            shot = shot_result.shot_index
            for reflector, tracking in enumerate(shot_result.tracking):
                reference_tracking[reflector][shot] = tracking
            trace_valid[:, shot] = shot_result.trace_valid
            tracked_shift[:, shot] = shot_result.shift_time
            tracked_correlation[:, shot] = shot_result.tracked_correlation
            energy_obs[:, shot] = shot_result.energy_obs
            energy_syn[:, shot] = shot_result.energy_syn
            boundary_flag[:, shot] = shot_result.boundary_flag
            tracking_success[:, shot] = (
                shot_result.final_available
                if self.config.misfit.fallback_use_in_misfit
                else shot_result.tracking_success
            )
            dp_success[:, shot] = shot_result.dp_success
            bridge_success[:, shot] = shot_result.bridge_success
            fallback_success[:, shot] = shot_result.fallback_success
            candidate_count[:, shot] = shot_result.candidate_count
            saved_correlations.update(shot_result.correlations)

        if any(item is None for row in reference_tracking for item in row):
            raise WorkflowBuildError("Reference tracking did not produce all reflector/shot results.")
        frozen_tracking = tuple(
            tuple(item for item in row if item is not None)
            for row in reference_tracking
        )
        fixed_quality = build_fixed_mask_from_config(
            trace_valid,
            windows.valid,
            tracking_success,
            tracked_correlation,
            energy_obs,
            energy_syn,
            boundary_flag,
            config=self.config.misfit,
        )
        stage_summary = []
        for reflector in range(nref):
            trace_ok = fixed_quality.trace_valid[reflector]
            window_ok = fixed_quality.window_valid[reflector]
            tracking_ok = fixed_quality.tracking_success[reflector]
            energy_ok = (
                np.isfinite(fixed_quality.window_energy_obs[reflector])
                & (fixed_quality.window_energy_obs[reflector] > 0.0)
            )
            after_trace = trace_ok
            after_window = after_trace & window_ok
            after_tracking = after_window & tracking_ok
            after_energy = after_tracking & energy_ok
            stage_summary.append(
                f"R{reflector + 1}: trace={int(np.count_nonzero(after_trace))} "
                f"window={int(np.count_nonzero(after_window))} "
                f"tracking={int(np.count_nonzero(after_tracking))} "
                f"dp={int(np.count_nonzero(dp_success[reflector]))} "
                f"bridge={int(np.count_nonzero(bridge_success[reflector]))} "
                f"fallback={int(np.count_nonzero(fallback_success[reflector]))} "
                f"candidates_mean={float(np.mean(candidate_count[reflector])):.2f} "
                f"energy={int(np.count_nonzero(after_energy))} "
                f"fixed={int(np.count_nonzero(fixed_quality.fixed_mask[reflector]))}"
            )
        self.logger.info("Reference fixed-mask stages: %s", " | ".join(stage_summary))
        return fixed_quality, frozen_tracking, saved_correlations, tracked_shift

    def _build_observed_windows(
        self,
        reference_windows: WindowResult,
        fixed_quality: FixedMaskResult,
        reference_tracking: tuple[tuple[TrackingResult, ...], ...],
        observed_data: np.ndarray,
        source_coordinates: np.ndarray,
        receiver_coordinates: np.ndarray,
    ) -> tuple[WindowResult, np.ndarray, dict[str, object]]:
        """Freeze observed-event windows after reference tracking.

        Direct centers use the raw reference DP shift.  Edge recovery is
        applied once here, before the observed windows become immutable.
        """

        recovery = recover_edge_observed_centers(
            reference_windows.center_time,
            fixed_quality.trace_valid,
            fixed_quality.window_valid,
            fixed_quality.window_energy_obs,
            reference_tracking,
            source_coordinates,
            receiver_coordinates,
            observed_data,
            dt=reference_windows.dt,
            t0=reference_windows.t0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WindowOverlapWarning)
            observed_windows = build_window_result(
                recovery.observed_center,
                dt=reference_windows.dt,
                t0=reference_windows.t0,
                nt=reference_windows.nt,
                half_window_time=reference_windows.half_window_time,
                window_type=reference_windows.window_type,
                tukey_alpha=reference_windows.tukey_alpha,
                max_lag_samples=reference_windows.max_lag_samples,
            )
        return observed_windows, recovery.observed_center_source, dict(recovery.counts)

    def build(
        self,
        current_velocity: np.ndarray,
        reflector_source: str | Path | Iterable[Reflector],
        source_coordinates: np.ndarray,
        receiver_coordinates: np.ndarray,
        *,
        dt: float,
        t0: float,
        nt: int,
        observed_data: np.ndarray | None = None,
        reference_synthetic_data: np.ndarray | None = None,
    ) -> WRTIReferenceState:
        """Run Steps 1–6 for one outer migration stage."""

        velocity = np.asarray(current_velocity, dtype=float)
        expected_velocity_shape = (self.config.grid.nx, self.config.grid.nz)
        if velocity.shape != expected_velocity_shape:
            raise WorkflowBuildError(
                f"current_velocity must have shape {expected_velocity_shape}; got {velocity.shape}."
            )
        if not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise WorkflowBuildError("current_velocity must be finite and strictly positive.")
        if not np.isfinite(dt) or dt <= 0:
            raise WorkflowBuildError("dt must be finite and positive.")

        try:
            sources = normalize_source_coordinates(source_coordinates)
            receivers = normalize_receiver_coordinates(
                receiver_coordinates, sources.shape[0]
            )
        except GeometryError as exc:
            raise WorkflowBuildError(str(exc)) from exc
        reflectors = self._load_reflectors(reflector_source)
        mappings = map_reflectors_to_grid(
            reflectors,
            self.config.grid.x0,
            self.config.grid.z0,
            self.config.grid.dx,
            self.config.grid.dz,
            self.config.grid.nx,
            self.config.grid.nz,
            self.config.grid.tolerance,
            strict=True,
        )

        traveltime_result: ReflectionTraveltimeResult = compute_reflection_traveltimes(
            velocity,
            self.config.grid.x_axis,
            self.config.grid.z_axis,
            sources,
            receivers,
            mappings,
            traveltime_table=self.traveltime_table,
            workers=self.config.eikonal_workers,
        )
        # Eikonal output is a geometric propagation time.  A causal source
        # wavelet may define its diagnostic event (for example, its peak) at
        # a non-zero time.  The configured shift converts geometric time to
        # the record-time reference used only for fixed-window construction.
        window_center_time = (
            traveltime_result.traveltime + self.config.center_time_shift
        )
        # The overlap count is retained in the frozen state and reported by
        # SWIT misfit initialization.  Avoid duplicating it as a Python
        # warning in long VFSA logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WindowOverlapWarning)
            windows = build_window_result(
                window_center_time,
                dt=dt,
                t0=t0,
                nt=nt,
                half_window_time=self.config.half_window_time,
                window_type=self.config.window_type,
                tukey_alpha=self.config.tukey_alpha,
                max_lag_samples=int(round(self.config.max_lag_time / float(dt))),
            )

        if (observed_data is None) != (reference_synthetic_data is None):
            raise WorkflowBuildError(
                "observed_data and reference_synthetic_data must be supplied together."
            )
        fixed_quality = None
        reference_tracking = None
        reference_shift_time = None
        observed_windows = None
        observed_center_source = None
        recovery_counts: dict[str, object] | None = None
        saved_correlations: dict[tuple[int, int], CorrelationResult] = {}
        if observed_data is not None and reference_synthetic_data is not None:
            (
                fixed_quality,
                reference_tracking,
                saved_correlations,
                reference_shift_time,
            ) = self._build_reference_tracking(
                observed_data,
                reference_synthetic_data,
                windows,
                sources,
                receivers,
            )
            observed_for_recovery = self._observed_data_for_recovery(
                observed_data,
                reference_synthetic_data,
            )
            # Persist raw ZNCC/tracking before center recovery: a recovery
            # exception must not erase the evidence needed to diagnose it.
            save_reference_center_diagnostics(
                self.config.diagnostics_output_dir or "wrti_diagnostics",
                correlations=saved_correlations,
                trackings=reference_tracking,
                receiver_coordinates=receivers,
                observed_center=windows.center_time,
                observed_center_source=np.zeros_like(windows.center_time, dtype=int),
                theoretical_center=windows.center_time,
                shots=self.config.debug_tracking_shots,
                stage="input",
            )
            observed_windows, observed_center_source, recovery_counts = self._build_observed_windows(
                windows,
                fixed_quality,
                reference_tracking,
                observed_for_recovery,
                sources,
                receivers,
            )
            if self.config.diagnostics_output_dir:
                save_reference_center_diagnostics(
                    self.config.diagnostics_output_dir,
                    correlations=saved_correlations,
                    trackings=reference_tracking,
                    receiver_coordinates=receivers,
                    observed_center=observed_windows.center_time,
                    observed_center_source=observed_center_source,
                    theoretical_center=windows.center_time,
                    shots=self.config.debug_tracking_shots,
                    stage="result",
                )

        state = WRTIReferenceState(
            reflectors=reflectors,
            reflector_grid_indices=tuple(mapping.indices for mapping in mappings),
            reflection_traveltime=traveltime_result.traveltime,
            reflection_point_index=traveltime_result.reflection_point_index,
            traveltime_valid=traveltime_result.valid,
            windows=windows,
            fixed_mask=None if fixed_quality is None else fixed_quality.fixed_mask,
            config=self.config,
            source_coordinates=sources,
            receiver_coordinates=receivers,
            reference_qc=fixed_quality,
            reference_tracking=reference_tracking,
            reference_shift_time=reference_shift_time,
            observed_windows=observed_windows,
            observed_center_source=observed_center_source,
            reference_correlations=saved_correlations,
        )
        if recovery_counts is not None:
            self.logger.info(
                "Observed-center recovery: direct anchors=%d recovered edge receivers=%d "
                "quadratic+envelope=%d quadratic-only fallback=%d "
                "unresolved internal gaps=%d skipped insufficient anchors=%d",
                recovery_counts["direct_anchors"],
                recovery_counts["recovered_edge_receivers"],
                recovery_counts["quadratic_plus_envelope"],
                recovery_counts["quadratic_only_fallback"],
                recovery_counts["unresolved_internal_gaps"],
                recovery_counts["recovery_skipped_insufficient_anchors"],
            )
            self.logger.info(
                "Observed-center recovery by reflector: %s",
                " ".join(
                    f"R{index + 1} recovered={count}"
                    for index, count in enumerate(recovery_counts["recovered_by_reflector"])
                ),
            )
        self.logger.info(
            "WRTI reference state built: n_reflectors=%d n_fixed=%d "
            "window_invalid=%d observed_window_invalid=%d overlap_warnings=%d",
            len(state.reflectors),
            state.n_fixed,
            state.window_invalid_count,
            state.observed_window_invalid_count,
            state.overlap_warning_count,
        )
        return state
