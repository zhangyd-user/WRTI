from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from WRTI.misfit import FixedMaskResult, build_fixed_mask
from WRTI.diagnostics.plots import save_vfsa_diagnostic
from WRTI.reflector import Reflector
from WRTI.window import build_window_result
from WRTI.workflow import (
    OuterStageWindowBuilder,
    WRTIConfig,
    WRTIObjectiveEvaluator,
    WRTIReferenceState,
    adfwi_to_wrti,
)


class WorkflowSmokeTests(unittest.TestCase):
    @staticmethod
    def _pulse(nt: int, center: int) -> np.ndarray:
        samples = np.arange(nt, dtype=float)
        return np.exp(-0.5 * ((samples - center) / 1.5) ** 2)

    @staticmethod
    def _delay(trace: np.ndarray, shift: int) -> np.ndarray:
        result = np.zeros_like(trace)
        result[shift:] = trace[:-shift]
        return result

    def _state_and_data(self):
        dt = 1.0
        nt = 64
        nreceiver = 3
        tref = np.full((1, 1, nreceiver), 20.0)
        windows = build_window_result(
            tref,
            dt=dt,
            t0=0.0,
            nt=nt,
            half_window_time=5.0,
            max_lag_samples=4,
        )
        observed_windows = build_window_result(
            tref + 2.0,
            dt=dt,
            t0=0.0,
            nt=nt,
            half_window_time=5.0,
            max_lag_samples=4,
        )
        base = self._pulse(nt, 20)
        synthetic_reference = np.tile(base, (1, nreceiver, 1))
        observed = np.tile(self._delay(base, 2), (1, nreceiver, 1))
        quality = build_fixed_mask(
            np.ones((1, 1, nreceiver), dtype=bool),
            windows.valid,
            np.ones((1, 1, nreceiver), dtype=bool),
            np.ones((1, 1, nreceiver), dtype=float),
            np.ones((1, 1, nreceiver), dtype=float),
            np.ones((1, 1, nreceiver), dtype=float),
            np.zeros((1, 1, nreceiver), dtype=bool),
        )
        config = WRTIConfig.from_mapping(
            {
                "grid": {
                    "x0": 0.0,
                    "z0": 0.0,
                    "dx": 1.0,
                    "dz": 1.0,
                    "nx": 8,
                    "nz": 8,
                },
                "window": {
                    "type": "rectangular",
                    "half_window_time": 5.0,
                    "tukey_alpha": 0.5,
                },
                "correlation": {
                    "method": "zncc",
                    "max_lag_time": 4.0,
                    "seed_lag_range_time": 4.0,
                },
                "tracking": {"epsilon_time": 1.0},
                "parallel": {"eikonal_workers": 1, "wrti_workers": 1},
                "qc": {
                    "min_correlation": None,
                    "use_energy_threshold": False,
                    "boundary_margin": 0,
                    "failure_penalty_time": 10.0,
                },
                "misfit": {"reflector_weights": [1.0]},
            }
        )
        state = WRTIReferenceState(
            reflectors=(Reflector(0, np.array([1.0, 2.0]), np.array([3.0, 3.0])),),
            reflector_grid_indices=(np.array([[3, 1], [3, 2]]),),
            reflection_traveltime=tref,
            reflection_point_index=np.zeros_like(tref, dtype=int),
            traveltime_valid=np.ones_like(tref, dtype=bool),
            windows=windows,
            fixed_mask=quality.fixed_mask,
            config=config,
            source_coordinates=np.array([[10.0, 0.0]]),
            receiver_coordinates=np.array(
                [[[8.0, 0.0], [10.0, 0.0], [12.0, 0.0]]]
            ),
            reference_qc=quality,
            reference_shift_time=np.full_like(tref, 2.0),
            observed_windows=observed_windows,
        )
        return state, observed, synthetic_reference

    def test_evaluator_uses_frozen_windows_and_fixed_mask(self) -> None:
        state, observed, synthetic = self._state_and_data()
        evaluator = WRTIObjectiveEvaluator(state, observed)
        misfit, result = evaluator.evaluate(synthetic)

        np.testing.assert_allclose(result.shift_time, 2.0, atol=0.05)
        self.assertAlmostEqual(misfit, 6.0, delta=0.3)
        self.assertAlmostEqual(result.misfit_sum, 6.0, delta=0.3)
        self.assertAlmostEqual(result.mean_misfit, 2.0, delta=0.1)
        self.assertEqual(result.failure_count, 0)
        np.testing.assert_array_equal(result.fixed_mask, state.fixed_mask)
        self.assertTrue(state.windows.center_sample.flags.writeable is False)
        np.testing.assert_allclose(state.observed_windows.center_time, 22.0)
        np.testing.assert_array_equal(state.observed_windows.left_sample, 17)
        np.testing.assert_array_equal(state.observed_windows.right_sample, 27)

    def test_diagnostic_plot_is_written_after_evaluation(self) -> None:
        state, observed, synthetic = self._state_and_data()
        evaluator = WRTIObjectiveEvaluator(state, observed)
        misfit, result = evaluator.evaluate(synthetic)

        with TemporaryDirectory() as directory:
            output_path = save_vfsa_diagnostic(
                Path(directory) / "wrti-diagnostic.png",
                reference_state=state,
                observed_data=observed,
                candidate_data=synthetic,
                correlations=evaluator.last_correlation_results,
                trackings=evaluator.last_tracking_results,
                evaluation=8,
                misfit=misfit,
                failure_count=result.failure_count,
                path_failure_mask_array=result.path_failure_mask,
                low_correlation_qc_mask_array=result.low_correlation_qc_mask,
                boundary_qc_mask_array=result.boundary_qc_mask,
            )
            self.assertTrue(Path(output_path).is_file())

    @unittest.skipIf(
        os.name == "nt",
        "The managed Windows test environment blocks ProcessPool named pipes.",
    )
    def test_parallel_evaluator_matches_serial_evaluator(self) -> None:
        state, observed, synthetic = self._state_and_data()
        serial_misfit, serial = WRTIObjectiveEvaluator(state, observed).evaluate(
            synthetic
        )
        parallel_config = replace(state.config, wrti_workers=2)
        parallel_state = replace(state, config=parallel_config)
        parallel_misfit, parallel = WRTIObjectiveEvaluator(
            parallel_state, observed
        ).evaluate(synthetic)

        self.assertAlmostEqual(parallel_misfit, serial_misfit, places=12)
        np.testing.assert_array_equal(parallel.shift_time, serial.shift_time)
        np.testing.assert_array_equal(
            parallel.path_failure_mask, serial.path_failure_mask
        )
        self.assertEqual(
            parallel.path_failure_count,
            serial.path_failure_count,
        )
        self.assertEqual(
            parallel.low_correlation_qc_count,
            serial.low_correlation_qc_count,
        )

    def test_adfwi_shot_time_receiver_is_transposed(self) -> None:
        adfwi_output = {"p": np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)}
        converted = adfwi_to_wrti(adfwi_output)
        self.assertEqual(converted.shape, (2, 3, 4))
        np.testing.assert_array_equal(converted[1, 2], adfwi_output["p"][1, :, 2])

    def test_outer_builder_builds_reference_state_with_injected_traveltime_backend(self) -> None:
        config = WRTIConfig.from_mapping(
            {
                "grid": {"x0": 0.0, "z0": 0.0, "dx": 1.0, "dz": 1.0, "nx": 8, "nz": 8},
                "window": {
                    "type": "rectangular",
                    "half_window_time": 5.0,
                    "center_time_shift": 2.0,
                    "tukey_alpha": 0.5,
                },
                "correlation": {"method": "zncc", "max_lag_time": 4.0, "seed_lag_range_time": 4.0},
                "tracking": {"epsilon_time": 1.0},
                "parallel": {"eikonal_workers": 1, "wrti_workers": 1},
                "qc": {"boundary_margin": 0, "failure_penalty_time": 10.0},
                "misfit": {"reflector_weights": [1.0]},
            }
        )

        def fake_traveltime_table(z_axis, x_axis, sources, receivers, velocity, mode):
            self.assertEqual(mode, "eikonal")
            return (
                np.full((sources.shape[0], z_axis.size, x_axis.size), 10.0),
                np.full((receivers.shape[0], z_axis.size, x_axis.size), 10.0),
            )

        base = self._pulse(64, 22)
        observed = np.tile(self._delay(base, 2), (1, 3, 1))
        state = OuterStageWindowBuilder(
            config,
            traveltime_table=fake_traveltime_table,
        ).build(
            np.full((8, 8), 2.0),
            [Reflector(0, np.array([1.0, 2.0]), np.array([2.0, 2.0]))],
            np.array([[10.0, 0.0]]),
            np.array([[[8.0, 0.0], [10.0, 0.0], [12.0, 0.0]]]),
            dt=1.0,
            t0=0.0,
            nt=64,
            observed_data=observed,
            reference_synthetic_data=np.tile(base, (1, 3, 1)),
        )

        self.assertEqual(state.shape, (1, 1, 3))
        self.assertEqual(state.n_fixed, 3)
        self.assertIsNotNone(state.reference_qc)
        np.testing.assert_allclose(state.reflection_traveltime, 20.0)
        np.testing.assert_allclose(state.windows.center_time, 22.0)
        np.testing.assert_allclose(state.reference_shift_time, 2.0, atol=0.05)
        np.testing.assert_allclose(state.observed_windows.center_time, 24.0, atol=0.05)
        np.testing.assert_array_equal(
            state.observed_windows.right_sample - state.observed_windows.left_sample,
            state.windows.right_sample - state.windows.left_sample,
        )

    def test_recovery_preprocessing_matches_per_shot_tracking_calls(self) -> None:
        state, _, _ = self._state_and_data()
        call_shapes = []

        def preprocess(observed, synthetic):
            call_shapes.append(observed.shape)
            return observed + observed.shape[0], synthetic

        builder = OuterStageWindowBuilder(state.config, preprocess_hook=preprocess)
        observed = np.zeros((2, 3, 4), dtype=float)
        processed = builder._observed_data_for_recovery(observed, observed)

        self.assertEqual(call_shapes, [(1, 3, 4), (1, 3, 4)])
        np.testing.assert_array_equal(processed, np.ones_like(observed))


if __name__ == "__main__":
    unittest.main()
