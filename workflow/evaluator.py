"""Frozen-state objective evaluator for VFSA proposals."""

from __future__ import annotations

import logging
from types import MappingProxyType

import numpy as np

from ..correlation import CorrelationResult, PreprocessHook
from ..misfit import MisfitError, MisfitResult, compute_vfsa_misfit
from ..tracking import TrackingResult

from .parallel import ShotTrackingTask, run_shot_tasks, slice_windows_for_shot
from .state import WRTIReferenceState


class EvaluatorError(RuntimeError):
    """Raised when a candidate cannot be evaluated against the frozen state."""


class WRTIObjectiveEvaluator:
    """Evaluate candidate synthetic data using only the frozen outer state."""

    def __init__(
        self,
        reference_state: WRTIReferenceState,
        observed_data: np.ndarray,
        *,
        preprocess_hook: PreprocessHook | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if reference_state.fixed_mask is None or reference_state.reference_qc is None:
            raise EvaluatorError(
                "WRTIReferenceState has no fixed_mask; build it with observed_data "
                "and reference_synthetic_data before creating the evaluator."
            )
        if reference_state.observed_windows is None:
            raise EvaluatorError(
                "WRTIReferenceState has no observed_windows; rebuild the reference "
                "state with observed_data and reference_synthetic_data."
            )
        observed = np.asarray(observed_data, dtype=float)
        expected = (
            reference_state.shape[1],
            reference_state.shape[2],
            reference_state.windows.nt,
        )
        if observed.shape != expected:
            raise EvaluatorError(
                f"observed_data must have shape {expected}; got {observed.shape}."
            )
        if not np.isfinite(observed).all():
            raise EvaluatorError("observed_data must contain only finite values.")
        self._reference_state = reference_state
        self._observed_data = np.array(observed, dtype=float, copy=True)
        self._observed_data.setflags(write=False)
        self.preprocess_hook = preprocess_hook
        self.logger = logger or logging.getLogger("WRTI.workflow")
        self.last_correlation_results = MappingProxyType({})
        self.last_tracking_results = MappingProxyType({})

    @property
    def reference_state(self) -> WRTIReferenceState:
        return self._reference_state

    @property
    def observed_data(self) -> np.ndarray:
        return self._observed_data

    def evaluate(self, candidate_synthetic: np.ndarray) -> tuple[float, MisfitResult]:
        """Run only fixed-window ZNCC, tracking, and fixed-mask misfit."""

        candidate = np.asarray(candidate_synthetic, dtype=float)
        if candidate.shape != self._observed_data.shape:
            raise EvaluatorError(
                f"candidate_synthetic must have shape {self._observed_data.shape}; "
                f"got {candidate.shape}."
            )
        if not np.isfinite(candidate).all():
            raise EvaluatorError("candidate_synthetic must contain only finite values.")

        state = self._reference_state
        nref, ns, nr = state.shape
        shift_time = np.full((nref, ns, nr), np.nan, dtype=float)
        tracked_correlation = np.full((nref, ns, nr), np.nan, dtype=float)
        energy_obs = np.full((nref, ns, nr), np.nan, dtype=float)
        energy_syn = np.full((nref, ns, nr), np.nan, dtype=float)
        boundary_flag = np.zeros((nref, ns, nr), dtype=bool)
        tracking_success = np.zeros((nref, ns, nr), dtype=bool)
        path_failure_mask = np.zeros((nref, ns, nr), dtype=bool)
        fallback_local_mask = np.zeros((nref, ns, nr), dtype=bool)
        fallback_global_mask = np.zeros((nref, ns, nr), dtype=bool)
        fallback_argmax_mask = np.zeros((nref, ns, nr), dtype=bool)
        # The candidate window uses observed centers.  Ownership needs a
        # complete neighboring-center field, so fill only missing observed
        # centers from the original theoretical/reference windows.
        ownership_center_time = np.array(
            state.observed_windows.center_time,
            dtype=float,
            copy=True,
        )
        missing_ownership_centers = (
            ~state.observed_windows.valid
            | ~np.isfinite(ownership_center_time)
        )
        ownership_center_time[missing_ownership_centers] = state.windows.center_time[
            missing_ownership_centers
        ]
        # Retain five acquisition shots, evenly distributed from the first
        # to the last shot, for one PNG diagnostic per VFSA evaluation.
        diagnostic_shots = np.rint(
            np.linspace(0, ns - 1, min(5, ns))
        ).astype(int)
        selected = set(state.config.save_correlation_for)
        selected.update(
            (reflector, int(shot))
            for reflector in range(nref)
            for shot in diagnostic_shots
        )
        saved_correlations: dict[tuple[int, int], CorrelationResult] = {}
        saved_tracking: dict[tuple[int, int], TrackingResult] = {}
        dp_success_count = 0
        bridge_success_count = 0
        fallback_success_count = 0
        candidate_state_count = 0
        candidate_receiver_count = 0
        coarse_deviations: list[np.ndarray] = []
        tasks = tuple(
            ShotTrackingTask(
                shot_index=shot,
                observed_shot=self._observed_data[shot],
                synthetic_shot=candidate[shot],
                windows=slice_windows_for_shot(state.observed_windows, shot),
                ownership_center_time=ownership_center_time[:, shot : shot + 1, :],
                source_coordinates=state.source_coordinates[shot],
                receiver_coordinates=state.receiver_coordinates[shot],
                max_lag_time=state.config.max_lag_time,
                seed_lag_range_time=state.config.seed_lag_range_time,
                epsilon_time=state.config.epsilon_time,
                dt=state.windows.dt,
                use_envelope_coarse=state.config.use_envelope_coarse,
                envelope_fine_half_width_time=state.config.envelope_fine_half_width_time,
                envelope_tracking_epsilon_time=state.config.envelope_tracking_epsilon_time,
                tracking_min_correlation=state.config.tracking_min_correlation,
                boundary_margin_samples=state.config.boundary_margin_samples,
                selected_reflectors=tuple(
                    reflector
                    for reflector in range(nref)
                    if (reflector, shot) in selected
                ),
                preprocess_hook=self.preprocess_hook,
                fixed_side="observed",
                tracking_enhancement_enabled=state.config.tracking_enhancement_enabled,
                tracking_agc_fraction=state.config.tracking_agc_fraction,
                tracking_agc_floor_ratio=state.config.tracking_agc_floor_ratio,
                tracking_receiver_stack=state.config.tracking_receiver_stack,
                tracking_raw_refine_radius_samples=state.config.tracking_raw_refine_radius_samples,
                ownership_guard_time=state.config.ownership_guard_time,
                tracking_top_k_peaks=state.config.tracking_top_k_peaks,
                tracking_peak_min_separation_samples=state.config.tracking_peak_min_separation_samples,
                tracking_coarse_soft_width_time=state.config.tracking_coarse_soft_width_time,
                tracking_coarse_soft_weight=state.config.tracking_coarse_soft_weight,
                tracking_coarse_soft_penalty_cap=state.config.tracking_coarse_soft_penalty_cap,
                tracking_smooth_weight=state.config.tracking_smooth_weight,
                tracking_max_residual_jump_time=state.config.tracking_max_residual_jump_time,
                tracking_max_skip_rows=state.config.tracking_max_skip_rows,
                tracking_gap_penalty=state.config.tracking_gap_penalty,
                tracking_correlation_mode=state.config.tracking_correlation_mode,
                tracking_restart_enabled=state.config.tracking_restart_enabled,
                tracking_restart_confirm_rows=state.config.tracking_restart_confirm_rows,
                tracking_restart_min_mean_correlation=state.config.tracking_restart_min_mean_correlation,
                tracking_restart_max_coarse_deviation_time=state.config.tracking_restart_max_coarse_deviation_time,
                local_search_half_width_time=state.config.local_search_half_width_time,
                max_search_half_width_time=state.config.max_search_half_width_time,
                gap_expand_time=state.config.gap_expand_time,
                residual_history=state.config.residual_history,
                min_peak_margin=state.config.min_peak_margin,
                prediction_weight=state.config.prediction_weight,
                max_gap_rows=state.config.max_gap_rows,
                relock_confirm_rows=state.config.relock_confirm_rows,
                restart_min_correlation=state.config.restart_min_correlation,
                bridge_enabled=state.config.bridge_enabled,
                bridge_half_width_time=state.config.bridge_half_width_time,
                bridge_max_width_time=state.config.bridge_max_width_time,
                bridge_max_receivers=state.config.bridge_max_receivers,
                bridge_max_distance=state.config.bridge_max_distance,
                debug_tracking_shots=state.config.debug_tracking_shots,
                debug_tracking_reflectors=state.config.debug_tracking_reflectors,
                save_tracking_snapshot=state.config.save_tracking_snapshot,
                diagnostics_output_dir=state.config.diagnostics_output_dir,
            )
            for shot in range(ns)
        )
        for shot_result in run_shot_tasks(tasks, state.config.wrti_workers):
            shot = shot_result.shot_index
            shift_time[:, shot] = shot_result.shift_time
            tracked_correlation[:, shot] = shot_result.tracked_correlation
            energy_obs[:, shot] = shot_result.energy_obs
            energy_syn[:, shot] = shot_result.energy_syn
            boundary_flag[:, shot] = shot_result.boundary_flag
            tracking_success[:, shot] = (
                shot_result.final_available
                if state.config.misfit.fallback_use_in_misfit
                else shot_result.tracking_success
            )
            path_failure_mask[:, shot] = shot_result.path_failure_mask
            fallback_local_mask[:, shot] = shot_result.fallback_local_mask
            fallback_global_mask[:, shot] = shot_result.fallback_global_mask
            fallback_argmax_mask[:, shot] = shot_result.fallback_argmax_mask
            dp_success_count += int(np.count_nonzero(shot_result.dp_success))
            bridge_success_count += int(np.count_nonzero(shot_result.bridge_success))
            fallback_success_count += int(np.count_nonzero(shot_result.fallback_success))
            candidate_state_count += int(np.sum(shot_result.candidate_count))
            candidate_receiver_count += int(np.count_nonzero(shot_result.candidate_count))
            finite_deviation = shot_result.coarse_deviation_samples[
                np.isfinite(shot_result.coarse_deviation_samples)
            ]
            if finite_deviation.size:
                coarse_deviations.append(np.abs(finite_deviation))
            saved_correlations.update(shot_result.correlations)
            for reflector, tracking in enumerate(shot_result.tracking):
                if (reflector, shot) in selected:
                    saved_tracking[(reflector, shot)] = tracking

        try:
            result = compute_vfsa_misfit(
                state.fixed_mask,
                shift_time=shift_time,
                tracked_correlation=tracked_correlation,
                window_energy_obs=energy_obs,
                window_energy_syn=energy_syn,
                boundary_flag=boundary_flag,
                tracking_success=tracking_success,
                path_failure_mask=path_failure_mask,
                fallback_local_mask=fallback_local_mask,
                fallback_global_mask=fallback_global_mask,
                fallback_argmax_mask=fallback_argmax_mask,
                config=state.config.misfit,
            )
        except MisfitError as exc:
            raise EvaluatorError(str(exc)) from exc

        reference_fixed_count = state.n_fixed
        current_fixed_count = result.n_fixed
        if current_fixed_count != reference_fixed_count:
            raise EvaluatorError(
                "WRTI fixed-mask count changed during candidate evaluation: "
                f"reference={reference_fixed_count}, current={current_fixed_count}."
            )

        self.last_correlation_results = MappingProxyType(saved_correlations)
        self.last_tracking_results = MappingProxyType(saved_tracking)
        successful_correlation = tracked_correlation[tracking_success]
        mean_correlation = (
            float(np.nanmean(successful_correlation))
            if successful_correlation.size and np.isfinite(successful_correlation).any()
            else float("nan")
        )
        sum_misfit = result.misfit_sum
        mean_misfit = result.mean_misfit
        all_deviations = (
            np.concatenate(coarse_deviations)
            if coarse_deviations
            else np.empty(0, dtype=float)
        )
        self.logger.info(
            "WRTI objective evaluation: sum_misfit=%.9g mean_misfit=%.9g "
            "fixed=%d dp_success=%d bridge_success=%d fallback_success=%d "
            "path_failures=%d data_invalid=%d candidates_mean=%.3g "
            "coarse_deviation_median=%.3g coarse_deviation_max=%.3g "
            "fallback_local=%d fallback_global=%d fallback_argmax=%d "
            "low_corr_qc=%d boundary_qc=%d "
            "mean_tracked_correlation=%.6g",
            sum_misfit,
            mean_misfit,
            current_fixed_count,
            dp_success_count,
            bridge_success_count,
            fallback_success_count,
            result.path_failure_count,
            result.data_invalid_count,
            candidate_state_count / max(candidate_receiver_count, 1),
            float(np.median(all_deviations)) if all_deviations.size else float("nan"),
            float(np.max(all_deviations)) if all_deviations.size else float("nan"),
            result.fallback_local_count,
            result.fallback_global_count,
            result.fallback_argmax_count,
            result.low_correlation_qc_count,
            result.boundary_qc_count,
            mean_correlation,
        )
        return sum_misfit, result
