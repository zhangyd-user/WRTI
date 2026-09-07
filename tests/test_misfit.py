from __future__ import annotations

import unittest

import numpy as np

from WRTI.misfit import (
    MisfitConfig,
    MisfitError,
    build_fixed_mask,
    build_fixed_mask_from_config,
    compute_vfsa_misfit,
)


class MisfitStep6Tests(unittest.TestCase):
    shape = (2, 1, 5)

    def _reference_quality(self):
        trace_valid = np.ones(self.shape, dtype=bool)
        window_valid = np.ones(self.shape, dtype=bool)
        tracking_success = np.ones(self.shape, dtype=bool)
        tracked_correlation = np.ones(self.shape, dtype=float)
        window_energy_obs = np.ones(self.shape, dtype=float)
        window_energy_syn = np.ones(self.shape, dtype=float)
        boundary_flag = np.zeros(self.shape, dtype=bool)
        return build_fixed_mask(
            trace_valid,
            window_valid,
            tracking_success,
            tracked_correlation,
            window_energy_obs,
            window_energy_syn,
            boundary_flag,
        )

    def _candidate(self, shift_time, tracking_success=None, boundary_flag=None):
        if tracking_success is None:
            tracking_success = np.ones(self.shape, dtype=bool)
        if boundary_flag is None:
            boundary_flag = np.zeros(self.shape, dtype=bool)
        return dict(
            shift_time=np.asarray(shift_time, dtype=float),
            tracked_correlation=np.ones(self.shape, dtype=float),
            window_energy_obs=np.ones(self.shape, dtype=float),
            window_energy_syn=np.ones(self.shape, dtype=float),
            boundary_flag=boundary_flag,
            tracking_success=tracking_success,
        )

    def test_candidate_failures_keep_fixed_count_and_receive_penalty(self) -> None:
        reference = self._reference_quality()
        config = MisfitConfig(failure_penalty_time=10.0)

        candidate_a = compute_vfsa_misfit(
            reference,
            config=config,
            **self._candidate(np.full(self.shape, 0.1)),
        )
        success_b = np.ones(self.shape, dtype=bool)
        success_b.reshape(-1)[:5] = False
        candidate_b = compute_vfsa_misfit(
            reference,
            config=config,
            **self._candidate(np.full(self.shape, 0.1), success_b),
        )

        self.assertEqual(int(np.count_nonzero(candidate_a.fixed_mask)), 10)
        self.assertEqual(int(np.count_nonzero(candidate_b.fixed_mask)), 10)
        self.assertEqual(candidate_b.failure_count, 5)
        self.assertGreater(candidate_b.misfit_mean, candidate_a.misfit_mean)
        np.testing.assert_allclose(candidate_b.shift_time.reshape(-1)[:5], 10.0)
        self.assertEqual(np.count_nonzero(candidate_b.path_failure_mask), 5)

    def test_zero_failure_penalty_is_valid_and_keeps_failure_count(self) -> None:
        reference = self._reference_quality()
        success = np.ones(self.shape, dtype=bool)
        success.reshape(-1)[:4] = False
        result = compute_vfsa_misfit(
            reference,
            config=MisfitConfig(failure_penalty_time=0.0),
            **self._candidate(np.full(self.shape, 0.1), success),
        )

        self.assertEqual(result.path_failure_count, 4)
        np.testing.assert_allclose(result.shift_time.reshape(-1)[:4], 0.0)
        self.assertAlmostEqual(result.misfit_sum, 6.0 * 0.1**2 / 2.0)
        self.assertAlmostEqual(result.misfit_mean, 6.0 * 0.1**2 / 20.0)

    def test_nonfinite_shift_normalises_success_to_path_failure(self) -> None:
        reference = self._reference_quality()
        shift = np.full(self.shape, 0.1)
        shift.reshape(-1)[0] = np.nan
        result = compute_vfsa_misfit(
            reference,
            config=MisfitConfig(failure_penalty_time=2.0),
            **self._candidate(shift),
        )

        self.assertEqual(result.path_failure_count, 1)
        self.assertFalse(result.tracking_success.reshape(-1)[0])
        self.assertTrue(result.path_failure_mask.reshape(-1)[0])

    def test_reflector_weights_are_applied_without_data_count_normalisation(self) -> None:
        reference = self._reference_quality()
        shift = np.zeros(self.shape, dtype=float)
        shift[0, :, :] = 1.0
        shift[1, :, :] = 2.0
        result = compute_vfsa_misfit(
            reference,
            config=MisfitConfig(
                failure_penalty_time=10.0,
                reflector_weights=[1.0, 3.0],
            ),
            **self._candidate(shift),
        )

        # Sum objective: [5*1*1^2 + 5*3*2^2] / 2 = 32.5.
        self.assertAlmostEqual(result.misfit_sum, 32.5)
        self.assertAlmostEqual(result.misfit_mean, 3.25)

    def test_sum_objective_uses_all_fixed_residuals_without_normalisation(self) -> None:
        shape = (1, 1, 3)
        valid = np.ones(shape, dtype=bool)
        reference = build_fixed_mask(
            valid,
            valid,
            valid,
            np.ones(shape, dtype=float),
            np.ones(shape, dtype=float),
            np.ones(shape, dtype=float),
            np.zeros(shape, dtype=bool),
        )
        shift = np.array([[[1.0, 2.0, 3.0]]])
        result = compute_vfsa_misfit(
            reference,
            config=MisfitConfig(failure_penalty_time=10.0),
            shift_time=shift,
            tracked_correlation=np.ones(shape, dtype=float),
            window_energy_obs=np.ones(shape, dtype=float),
            window_energy_syn=np.ones(shape, dtype=float),
            boundary_flag=np.zeros(shape, dtype=bool),
            tracking_success=valid,
        )

        self.assertAlmostEqual(result.misfit_sum, 7.0)
        self.assertAlmostEqual(result.mean_misfit, 7.0 / 3.0)

    def test_zero_observed_energy_is_excluded_from_fixed_mask(self) -> None:
        reference_quality = self._reference_quality()
        zero_observed_energy = np.zeros(self.shape, dtype=float)
        reference = build_fixed_mask(
            reference_quality.trace_valid,
            reference_quality.window_valid,
            reference_quality.tracking_success,
            reference_quality.tracked_correlation,
            zero_observed_energy,
            np.ones(self.shape, dtype=float),
            reference_quality.boundary_flag,
        )
        self.assertEqual(reference.n_fixed, 0)

        positive_observed_energy = np.ones(self.shape, dtype=float)
        reference = build_fixed_mask(
            reference_quality.trace_valid,
            reference_quality.window_valid,
            reference_quality.tracking_success,
            reference_quality.tracked_correlation,
            positive_observed_energy,
            np.zeros(self.shape, dtype=float),
            reference_quality.boundary_flag,
        )
        self.assertEqual(reference.n_fixed, 10)

        thresholded = build_fixed_mask(
            reference_quality.trace_valid,
            reference_quality.window_valid,
            reference_quality.tracking_success,
            reference_quality.tracked_correlation,
            positive_observed_energy,
            np.zeros(self.shape, dtype=float),
            reference_quality.boundary_flag,
            use_energy_threshold=True,
            min_window_energy=1.0,
        )
        self.assertEqual(thresholded.n_fixed, 0)

    def test_low_correlation_valid_path_is_qc_not_failure_or_mask_removal(self) -> None:
        reference = self._reference_quality()
        config = MisfitConfig(failure_penalty_time=10.0, min_correlation=0.55)
        candidate = self._candidate(np.full(self.shape, 0.1))
        candidate["tracked_correlation"][:] = 0.4

        result = compute_vfsa_misfit(reference, config=config, **candidate)

        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.path_failure_count, 0)
        self.assertEqual(result.low_correlation_qc_count, 10)
        self.assertEqual(result.boundary_qc_count, 0)
        np.testing.assert_allclose(result.shift_time, 0.1)

        reference_from_config = build_fixed_mask_from_config(
            reference.trace_valid,
            reference.window_valid,
            reference.tracking_success,
            np.full(self.shape, 0.4),
            reference.window_energy_obs,
            reference.window_energy_syn,
            reference.boundary_flag,
            config=config,
        )
        self.assertEqual(reference_from_config.n_fixed, 10)

        direct = build_fixed_mask(
            reference.trace_valid,
            reference.window_valid,
            reference.tracking_success,
            np.full(self.shape, 0.4),
            reference.window_energy_obs,
            reference.window_energy_syn,
            reference.boundary_flag,
            min_correlation=0.55,
        )
        self.assertEqual(direct.n_fixed, 10)

    def test_boundary_valid_path_is_qc_not_failure_even_with_penalty_policy(self) -> None:
        reference = self._reference_quality()
        boundary = np.ones(self.shape, dtype=bool)
        result = compute_vfsa_misfit(
            reference,
            config=MisfitConfig(failure_penalty_time=10.0),
            **self._candidate(np.full(self.shape, 0.1), boundary_flag=boundary),
        )

        self.assertEqual(result.failure_count, 0)
        self.assertEqual(result.boundary_qc_count, 10)
        np.testing.assert_allclose(result.shift_time, 0.1)

    def test_zero_fixed_mask_raises_instead_of_returning_zero(self) -> None:
        reference = self._reference_quality()
        empty = np.zeros(self.shape, dtype=bool)
        with self.assertRaises(MisfitError):
            compute_vfsa_misfit(
                empty,
                config=MisfitConfig(failure_penalty_time=10.0),
                **self._candidate(np.zeros(self.shape)),
            )

    def test_mapping_requires_penalty_and_preserves_yaml_style_sections(self) -> None:
        config = MisfitConfig.from_mapping(
            {
                "qc": {
                    "failure_penalty_time": 3.0,
                    "candidate_boundary_policy": "use_shift",
                },
                "misfit": {"reflector_weights": [1.0, 2.0]},
            }
        )
        self.assertEqual(config.failure_penalty_time, 3.0)
        self.assertEqual(config.candidate_boundary_policy, "use_shift")
        self.assertEqual(config.reflector_weights, (1.0, 2.0))
        with self.assertRaises(MisfitError):
            MisfitConfig.from_mapping({"misfit": {"reflector_weights": [1.0]}})

        zero_penalty = MisfitConfig.from_mapping(
            {"qc": {"failure_penalty_time": 0.0}}
        )
        self.assertEqual(zero_penalty.failure_penalty_time, 0.0)


if __name__ == "__main__":
    unittest.main()
