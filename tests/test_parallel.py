from __future__ import annotations

import os
import unittest

import numpy as np

from WRTI.window import build_window_result
from WRTI.workflow.parallel import (
    ShotTrackingTask,
    run_shot_tasks,
    slice_windows_for_shot,
)


class ShotParallelConsistencyTests(unittest.TestCase):
    @staticmethod
    def _pulse(nt: int, center: int) -> np.ndarray:
        samples = np.arange(nt, dtype=float)
        return np.exp(-0.5 * ((samples - center) / 2.0) ** 2)

    @staticmethod
    def _delayed(trace: np.ndarray, shift: int) -> np.ndarray:
        result = np.zeros_like(trace)
        result[shift:] = trace[:-shift]
        return result

    def _tasks(self):
        ns, nr, nt = 3, 2, 128
        centers = np.full((1, ns, nr), 40.0)
        windows = build_window_result(
            centers,
            dt=1.0,
            t0=0.0,
            nt=nt,
            half_window_time=12.0,
            max_lag_samples=8,
        )
        synthetic = np.stack(
            [np.tile(self._pulse(nt, 40), (nr, 1)) for _ in range(ns)],
            axis=0,
        )
        observed = np.stack(
            [np.tile(self._delayed(self._pulse(nt, 40), 2), (nr, 1)) for _ in range(ns)],
            axis=0,
        )
        source = np.zeros((ns, 2), dtype=float)
        source[:, 0] = 1.0
        receivers = np.zeros((ns, nr, 2), dtype=float)
        receivers[:, :, 0] = np.arange(nr, dtype=float)
        return tuple(
            ShotTrackingTask(
                shot_index=shot,
                observed_shot=observed[shot],
                synthetic_shot=synthetic[shot],
                windows=slice_windows_for_shot(windows, shot),
                source_coordinates=source[shot],
                receiver_coordinates=receivers[shot],
                max_lag_time=8.0,
                seed_lag_range_time=8.0,
                epsilon_time=2.0,
                dt=1.0,
                use_envelope_coarse=True,
                envelope_fine_half_width_time=8.0,
                envelope_tracking_epsilon_time=2.0,
                tracking_min_correlation=None,
                boundary_margin_samples=0,
                selected_reflectors=(0,),
            )
            for shot in range(ns)
        )

    def test_serial_shot_task_results_are_in_stable_order(self) -> None:
        results = run_shot_tasks(self._tasks(), workers=1)
        self.assertEqual([item.shot_index for item in results], [0, 1, 2])
        for item in results:
            np.testing.assert_allclose(item.shift_time, 2.0, atol=0.05)
            self.assertTrue(item.tracking_success.all())
            self.assertEqual(
                item.correlations[(0, item.shot_index)].coarse_seed_receiver,
                1,
            )

    @unittest.skipIf(
        os.name == "nt",
        "The managed Windows test environment blocks ProcessPool named pipes.",
    )
    def test_process_shot_results_match_serial_results(self) -> None:
        serial = run_shot_tasks(self._tasks(), workers=1)
        parallel = run_shot_tasks(self._tasks(), workers=2)
        self.assertEqual(
            [item.shot_index for item in parallel],
            [item.shot_index for item in serial],
        )
        for expected, actual in zip(serial, parallel):
            np.testing.assert_array_equal(actual.tracking[0].path_index, expected.tracking[0].path_index)
            np.testing.assert_allclose(actual.shift_time, expected.shift_time)
            np.testing.assert_allclose(
                actual.tracked_correlation,
                expected.tracked_correlation,
                equal_nan=True,
            )
            np.testing.assert_array_equal(actual.tracking_success, expected.tracking_success)
            self.assertEqual(actual.correlations.keys(), expected.correlations.keys())
            for key in expected.correlations:
                np.testing.assert_allclose(
                    actual.correlations[key].correlation,
                    expected.correlations[key].correlation,
                    equal_nan=True,
                )


if __name__ == "__main__":
    unittest.main()
