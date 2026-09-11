"""Small, shot-level process helpers for WRTI correlation and tracking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import multiprocessing as mp
from typing import Iterable

import numpy as np

from ..correlation import CorrelationResult, PreprocessHook, iter_zncc
from ..tracking import TrackingResult, complete_tracked_shift, track_correlation_result
from ..window import WindowResult


class WRTIParallelError(RuntimeError):
    """Raised when a shot-level WRTI worker cannot complete."""


@dataclass(frozen=True)
class ShotTrackingTask:
    """All data needed by one independent shot worker.

    The gather is shot-local and the window state is sliced to one shot.  No
    reflector task receives a separate copy of the full survey gathers.
    """

    shot_index: int
    observed_shot: np.ndarray
    synthetic_shot: np.ndarray
    windows: WindowResult
    source_coordinates: np.ndarray
    receiver_coordinates: np.ndarray
    max_lag_time: float
    seed_lag_range_time: float
    epsilon_time: float
    dt: float
    use_envelope_coarse: bool
    envelope_fine_half_width_time: float
    envelope_tracking_epsilon_time: float
    tracking_min_correlation: float | None
    boundary_margin_samples: int
    selected_reflectors: tuple[int, ...]
    ownership_center_time: np.ndarray | None = None
    preprocess_hook: PreprocessHook | None = None
    fixed_side: str = "synthetic"
    tracking_enhancement_enabled: bool = True
    tracking_agc_fraction: float = 0.25
    tracking_agc_floor_ratio: float = 0.20
    tracking_receiver_stack: bool = True
    tracking_raw_refine_radius_samples: int = 1


@dataclass(frozen=True)
class ShotTrackingResult:
    """Compact per-shot result merged by the parent in shot order."""

    shot_index: int
    tracking: tuple[TrackingResult, ...]
    correlations: dict[tuple[int, int], CorrelationResult]
    trace_valid: np.ndarray
    shift_time: np.ndarray
    path_failure_mask: np.ndarray
    fallback_local_mask: np.ndarray
    fallback_global_mask: np.ndarray
    fallback_argmax_mask: np.ndarray
    tracked_correlation: np.ndarray
    energy_obs: np.ndarray
    energy_syn: np.ndarray
    boundary_flag: np.ndarray
    tracking_success: np.ndarray


def slice_windows_for_shot(windows: WindowResult, shot: int) -> WindowResult:
    """Return a one-shot WindowResult without changing window construction."""

    nshot = windows.shape[1]
    if not 0 <= int(shot) < nshot:
        raise IndexError("shot index is out of range.")
    selection = slice(int(shot), int(shot) + 1)
    return replace(
        windows,
        reference_traveltime=windows.reference_traveltime[:, selection, :],
        center_time=windows.center_time[:, selection, :],
        center_float=windows.center_float[:, selection, :],
        center_sample=windows.center_sample[:, selection, :],
        left_sample=windows.left_sample[:, selection, :],
        right_sample=windows.right_sample[:, selection, :],
        valid=windows.valid[:, selection, :],
        overlap=windows.overlap[:, selection, :],
        overlap_delta_time=windows.overlap_delta_time[:, selection, :],
    )


def run_shot_zncc_tracking(task: ShotTrackingTask) -> ShotTrackingResult:
    """Compute all reflector ZNCC and DP results for one shot."""

    shot = int(task.shot_index)
    observed = np.asarray(task.observed_shot, dtype=float)[None, ...]
    synthetic = np.asarray(task.synthetic_shot, dtype=float)[None, ...]
    nref, _, nr = task.windows.shape
    if observed.shape != synthetic.shape or observed.shape != (1, nr, task.windows.nt):
        raise WRTIParallelError(
            f"Shot {shot} gather shape is inconsistent with its windows."
        )
    receiver_x = np.asarray(task.receiver_coordinates[:, 0], dtype=float)
    source_x = float(task.source_coordinates[0])
    selected = set(int(value) for value in task.selected_reflectors)

    tracking_results: list[TrackingResult] = []
    saved_correlations: dict[tuple[int, int], CorrelationResult] = {}
    trace_valid = np.zeros((nref, nr), dtype=bool)
    shift_time = np.full((nref, nr), np.nan, dtype=float)
    tracked_correlation = np.full((nref, nr), np.nan, dtype=float)
    energy_obs = np.full((nref, nr), np.nan, dtype=float)
    energy_syn = np.full((nref, nr), np.nan, dtype=float)
    boundary_flag = np.zeros((nref, nr), dtype=bool)
    tracking_success = np.zeros((nref, nr), dtype=bool)
    path_failure_mask_array = np.zeros((nref, nr), dtype=bool)
    fallback_local_mask = np.zeros((nref, nr), dtype=bool)
    fallback_global_mask = np.zeros((nref, nr), dtype=bool)
    fallback_argmax_mask = np.zeros((nref, nr), dtype=bool)

    for reflector, _, correlation in iter_zncc(
        observed,
        synthetic,
        task.windows,
        max_lag_time=task.max_lag_time,
        dt=task.dt,
        preprocess_hook=task.preprocess_hook,
        reflectors=range(nref),
        shots=(0,),
        use_envelope_coarse=task.use_envelope_coarse,
        envelope_fine_half_width_time=task.envelope_fine_half_width_time,
        envelope_tracking_epsilon_time=task.envelope_tracking_epsilon_time,
        seed_lag_range_time=task.seed_lag_range_time,
        receiver_x=receiver_x,
        source_x=source_x,
        ownership_center_time=task.ownership_center_time,
        fixed_side=task.fixed_side,
        tracking_enhancement_enabled=task.tracking_enhancement_enabled,
        tracking_agc_fraction=task.tracking_agc_fraction,
        tracking_agc_floor_ratio=task.tracking_agc_floor_ratio,
        tracking_receiver_stack=task.tracking_receiver_stack,
    ):
        tracking = track_correlation_result(
            correlation,
            receiver_x,
            source_x,
            task.seed_lag_range_time,
            task.epsilon_time,
            valid=correlation.valid,
            lag_valid=correlation.lag_valid,
            min_correlation=task.tracking_min_correlation,
            boundary_margin_samples=task.boundary_margin_samples,
            raw_refine_radius_samples=task.tracking_raw_refine_radius_samples,
        )
        completion = complete_tracked_shift(
            tracking,
            correlation,
            # The existing seed lag range is the configured central/safe lag
            # range.  It is deliberately narrower than the full ZNCC range.
            safe_lag_range_time=task.seed_lag_range_time,
        )
        tracking_results.append(tracking)
        trace_valid[reflector] = correlation.valid
        shift_time[reflector] = completion.final_shift
        path_failure_mask_array[reflector] = completion.dp_failure_mask
        fallback_local_mask[reflector] = completion.fallback_local_mask
        fallback_global_mask[reflector] = completion.fallback_global_mask
        fallback_argmax_mask[reflector] = completion.fallback_argmax_mask
        tracked_correlation[reflector] = tracking.tracked_correlation
        energy_obs[reflector] = correlation.window_energy_obs
        energy_syn[reflector] = correlation.window_energy_syn
        boundary_flag[reflector] = tracking.boundary_flag
        # This array now means finite final data available to the objective;
        # DP quality remains available separately as path_failure_mask_array.
        tracking_success[reflector] = np.isfinite(completion.final_shift)
        if reflector in selected:
            # The one-shot window uses local shot index 0; restore the global
            # shot index only for diagnostic results that leave the worker.
            saved_correlations[(reflector, shot)] = replace(
                correlation,
                shot=shot,
            )

    if len(tracking_results) != nref:
        raise WRTIParallelError(
            f"Shot {shot} returned {len(tracking_results)} reflector results; "
            f"expected {nref}."
        )
    return ShotTrackingResult(
        shot_index=shot,
        tracking=tuple(tracking_results),
        correlations=saved_correlations,
        trace_valid=trace_valid,
        shift_time=shift_time,
        path_failure_mask=path_failure_mask_array,
        fallback_local_mask=fallback_local_mask,
        fallback_global_mask=fallback_global_mask,
        fallback_argmax_mask=fallback_argmax_mask,
        tracked_correlation=tracked_correlation,
        energy_obs=energy_obs,
        energy_syn=energy_syn,
        boundary_flag=boundary_flag,
        tracking_success=tracking_success,
    )


def run_shot_tasks(
    tasks: Iterable[ShotTrackingTask],
    workers: int,
) -> tuple[ShotTrackingResult, ...]:
    """Run shot tasks serially or with one non-nested process pool."""

    task_list = tuple(tasks)
    if not task_list:
        return ()
    if int(workers) <= 1 or len(task_list) == 1:
        results = [run_shot_zncc_tracking(task) for task in task_list]
    else:
        worker_count = min(int(workers), len(task_list))
        results_by_shot: dict[int, ShotTrackingResult] = {}
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(run_shot_zncc_tracking, task): task.shot_index
                    for task in task_list
                }
                for future in as_completed(futures):
                    shot = int(futures[future])
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise WRTIParallelError(
                            f"WRTI shot worker failed for shot {shot}: {exc}"
                        ) from exc
                    results_by_shot[result.shot_index] = result
        except WRTIParallelError:
            raise
        except Exception as exc:
            raise WRTIParallelError(
                "Unable to start or complete the WRTI shot process pool."
            ) from exc
        if set(results_by_shot) != {task.shot_index for task in task_list}:
            raise WRTIParallelError("WRTI shot process pool returned incomplete results.")
        results = [results_by_shot[task.shot_index] for task in task_list]
    return tuple(sorted(results, key=lambda item: item.shot_index))
