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
            tracking_success[:, shot] = shot_result.tracking_success
            path_failure_mask[:, shot] = shot_result.path_failure_mask
            fallback_local_mask[:, shot] = shot_result.fallback_local_mask
            fallback_global_mask[:, shot] = shot_result.fallback_global_mask
            fallback_argmax_mask[:, shot] = shot_result.fallback_argmax_mask
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
        self.logger.info(
            "WRTI objective evaluation: sum_misfit=%.9g mean_misfit=%.9g "
            "fixed=%d path_failures=%d data_invalid=%d "
            "fallback_local=%d fallback_global=%d fallback_argmax=%d "
            "low_corr_qc=%d boundary_qc=%d "
            "mean_tracked_correlation=%.6g",
            sum_misfit,
            mean_misfit,
            current_fixed_count,
            result.path_failure_count,
            result.data_invalid_count,
            result.fallback_local_count,
            result.fallback_global_count,
            result.fallback_argmax_count,
            result.low_correlation_qc_count,
            result.boundary_qc_count,
            mean_correlation,
        )
        return sum_misfit, result
