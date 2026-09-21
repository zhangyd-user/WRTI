"""Frozen-state objective evaluator for VFSA proposals."""

from __future__ import annotations

from dataclasses import replace
import logging
from types import MappingProxyType
import warnings

import numpy as np

from ..correlation import CorrelationResult, PreprocessHook
from ..misfit import MisfitError, MisfitResult, compute_vfsa_misfit
from ..tracking import TrackingResult
from ..window import WindowOverlapWarning, WindowResult, build_window_result

from .parallel import ShotTrackingTask, run_shot_tasks, slice_windows_for_shot
from .state import WRTIReferenceState


class EvaluatorError(RuntimeError):
    """Raised when a candidate cannot be evaluated against the frozen state."""


def _paired_dual_center_windows(
    observed_windows: WindowResult,
    tsyn_theory: np.ndarray,
    max_lag_time: float,
) -> tuple[WindowResult, WindowResult]:
    """Build equal-length observed/synthetic windows around independent centers."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WindowOverlapWarning)
        observed = build_window_result(
            observed_windows.center_time,
            observed_windows.dt,
            observed_windows.t0,
            observed_windows.nt,
            observed_windows.half_window_time,
            window_type=observed_windows.window_type,
            tukey_alpha=observed_windows.tukey_alpha,
            max_lag_time=max_lag_time,
        )
        synthetic = build_window_result(
            tsyn_theory,
            observed_windows.dt,
            observed_windows.t0,
            observed_windows.nt,
            observed_windows.half_window_time,
            window_type=observed_windows.window_type,
            tukey_alpha=observed_windows.tukey_alpha,
            max_lag_time=max_lag_time,
        )
    left_offset = observed.left_sample - observed.center_sample
    right_offset = observed.right_sample - observed.center_sample
    left = synthetic.center_sample + left_offset
    right = synthetic.center_sample + right_offset
    valid = (
        observed.valid
        & np.isfinite(tsyn_theory)
        & (left >= synthetic.max_lag_samples)
        & (right + synthetic.max_lag_samples < synthetic.nt)
    )
    return observed, replace(
        synthetic,
        left_sample=left,
        right_sample=right,
        valid=valid,
    )


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
        if reference_state.evaluation_fixed_mask is None:
            raise EvaluatorError(
                "WRTIReferenceState has no fixed mask for candidate evaluation."
            )
        if reference_state.evaluation_observed_windows is None:
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
        self._evaluation_count = 0

    @property
    def reference_state(self) -> WRTIReferenceState:
        return self._reference_state

    @property
    def observed_data(self) -> np.ndarray:
        return self._observed_data

    def evaluate(
        self,
        candidate_synthetic: np.ndarray,
        candidate_eikonal_traveltime: np.ndarray | None = None,
    ) -> tuple[float, MisfitResult]:
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
        dual_center = state.config.dual_center_enabled
        eikonal_traveltime = None
        tsyn_theory = None
        synthetic_windows = None
        observed_windows = state.evaluation_observed_windows
        evaluation_fixed_mask = state.evaluation_fixed_mask
        max_lag_time = state.config.max_lag_time
        ownership_center_time = None
        if dual_center:
            if (
                not state.config.dual_center_use_candidate_eikonal
                or candidate_eikonal_traveltime is None
            ):
                raise EvaluatorError(
                    "dual-center evaluation requires candidate Eikonal traveltimes."
                )
            eikonal_traveltime = np.asarray(
                candidate_eikonal_traveltime, dtype=float
            )
            if eikonal_traveltime.shape != state.shape:
                raise EvaluatorError(
                    "candidate_eikonal_traveltime must match reference_state.shape."
                )
            tsyn_theory = eikonal_traveltime + state.config.center_time_shift
            max_lag_time = state.config.dual_center_local_max_shift_time
            observed_windows, synthetic_windows = _paired_dual_center_windows(
                observed_windows,
                tsyn_theory,
                max_lag_time,
            )
            ownership_center_time = tsyn_theory
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
        if ownership_center_time is None:
            ownership_center_time = np.array(
                observed_windows.center_time,
                dtype=float,
                copy=True,
            )
            missing_ownership_centers = (
                ~observed_windows.valid
                | ~np.isfinite(ownership_center_time)
            )
            ownership_center_time[missing_ownership_centers] = state.windows.center_time[
                missing_ownership_centers
            ]
        # Retain five acquisition shots, evenly distributed from the first
        # to the last shot, for one PNG diagnostic per VFSA evaluation.
        diagnostic_shots = (
            np.arange(ns, dtype=int)
            if self._evaluation_count < 3
            else np.rint(np.linspace(0, ns - 1, min(5, ns))).astype(int)
        )
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
                windows=slice_windows_for_shot(observed_windows, shot),
                synthetic_windows=(
                    None
                    if synthetic_windows is None
                    else slice_windows_for_shot(synthetic_windows, shot)
                ),
                receiver_mask=(
                    evaluation_fixed_mask[:, shot, :] if dual_center else None
                ),
                max_consecutive_dp_failures=3,
                ownership_center_time=ownership_center_time[:, shot : shot + 1, :],
                source_coordinates=state.source_coordinates[shot],
                receiver_coordinates=state.receiver_coordinates[shot],
                max_lag_time=max_lag_time,
                seed_lag_range_time=min(
                    state.config.seed_lag_range_time, max_lag_time
                ),
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

        local_shift_time = None
        coarse_shift_time = None
        tsyn_picked = None
        tobs = None
        if dual_center:
            local_shift_time = np.array(shift_time, copy=True)
            tobs = np.asarray(
                state.evaluation_observed_windows.center_time, dtype=float
            )
            coarse_shift_time = tobs - tsyn_theory
            tsyn_picked = tsyn_theory + local_shift_time
            shift_time = tobs - tsyn_picked

        try:
            result = compute_vfsa_misfit(
                evaluation_fixed_mask,
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
                tobs=tobs,
                eikonal_traveltime=eikonal_traveltime,
                tsyn_theory=tsyn_theory,
                coarse_shift_time=coarse_shift_time,
                local_shift_time=local_shift_time,
                tsyn_picked=tsyn_picked,
            )
        except MisfitError as exc:
            raise EvaluatorError(str(exc)) from exc

        reference_fixed_count = state.evaluation_n_fixed
        current_fixed_count = result.n_fixed
        if current_fixed_count != reference_fixed_count:
            raise EvaluatorError(
                "WRTI fixed-mask count changed during candidate evaluation: "
                f"reference={reference_fixed_count}, current={current_fixed_count}."
            )

        self.last_correlation_results = MappingProxyType(saved_correlations)
        self.last_tracking_results = MappingProxyType(saved_tracking)
        self._evaluation_count += 1
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
