from __future__ import annotations

import unittest

import numpy as np

from WRTI.correlation import CorrelationResult
from WRTI.misfit import MisfitConfig, build_fixed_mask, compute_vfsa_misfit
from WRTI.tracking import TrackingResult, complete_tracked_shift, track_zncc


class ShiftFallbackTests(unittest.TestCase):
    lags = np.arange(-5, 6, dtype=int)

    def _correlation(self, values=None, valid=None):
        nreceiver = 7
        if values is None:
            values = np.ones((nreceiver, self.lags.size), dtype=float)
        if valid is None:
            valid = np.ones(nreceiver, dtype=bool)
        return CorrelationResult(
            lags_samples=self.lags,
            lags_time=self.lags.astype(float),
            correlation=np.asarray(values, dtype=float),
            waveform_correlation=np.asarray(values, dtype=float),
            valid=np.asarray(valid, dtype=bool),
            window_energy_obs=np.ones(nreceiver),
            window_energy_syn=np.ones(nreceiver),
            reflector=0,
            shot=0,
            max_lag_samples=5,
            dt=1.0,
        )

    def _tracking(self, success, shifts):
        success = np.asarray(success, dtype=bool)
        shifts = np.asarray(shifts, dtype=float)
        path_index = np.where(success, self.lags.size // 2, -1)
        return TrackingResult(
            path_index=path_index,
            shift_samples=shifts,
            shift_time=shifts,
            tracked_correlation=np.where(success, 0.8, np.nan),
            seed_receiver=3,
            seed_lag=0.0,
            success_mask=success,
            boundary_flag=np.zeros(success.size, dtype=bool),
            dt=1.0,
            epsilon_samples=1,
            boundary_margin_samples=0,
        )

    def test_all_dp_success_is_unchanged(self) -> None:
        shifts = np.arange(7, dtype=float) / 10.0
        result = complete_tracked_shift(
            self._tracking(np.ones(7, dtype=bool), shifts),
            self._correlation(),
            safe_lag_range_time=2.0,
        )
        np.testing.assert_array_equal(result.final_shift, shifts)
        self.assertFalse(result.dp_failure_mask.any())
        self.assertFalse(result.fallback_local_mask.any())

    def test_failed_receiver_uses_local_median(self) -> None:
        success = np.ones(7, dtype=bool)
        success[3] = False
        shifts = np.array([0.0, 0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
        result = complete_tracked_shift(
            self._tracking(success, shifts),
            self._correlation(),
            safe_lag_range_time=2.0,
            local_radius=2,
        )
        self.assertAlmostEqual(result.final_shift[3], 0.3)
        self.assertTrue(result.fallback_local_mask[3])

    def test_edge_failure_uses_available_local_successes(self) -> None:
        success = np.ones(7, dtype=bool)
        success[0] = False
        shifts = np.array([np.nan, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        result = complete_tracked_shift(
            self._tracking(success, shifts),
            self._correlation(),
            safe_lag_range_time=2.0,
            local_radius=2,
        )
        self.assertAlmostEqual(result.final_shift[0], 0.3)
        self.assertTrue(result.fallback_local_mask[0])

    def test_failed_receiver_without_own_zncc_uses_local_median(self) -> None:
        success = np.ones(7, dtype=bool)
        success[3] = False
        shifts = np.array([0.0, 0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
        values = np.ones((7, self.lags.size), dtype=float)
        values[3, :] = np.nan
        valid = np.ones(7, dtype=bool)
        valid[3] = False
        result = complete_tracked_shift(
            self._tracking(success, shifts),
            self._correlation(values, valid),
            safe_lag_range_time=2.0,
            local_radius=2,
        )
        self.assertAlmostEqual(result.final_shift[3], 0.3)
        self.assertTrue(result.fallback_local_mask[3])

    def test_failure_without_local_success_uses_global_median(self) -> None:
        success = np.array([True, False, False, False, False, False, True])
        shifts = np.array([0.2, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0])
        result = complete_tracked_shift(
            self._tracking(success, shifts),
            self._correlation(),
            safe_lag_range_time=2.0,
            local_radius=1,
        )
        self.assertAlmostEqual(result.final_shift[3], 0.6)
        self.assertTrue(result.fallback_global_mask[3])
        self.assertFalse(result.fallback_local_mask[3])
        self.assertFalse(result.fallback_argmax_mask[3])

    def test_all_dp_failures_use_safe_zncc_argmax(self) -> None:
        values = np.full((7, self.lags.size), 0.1)
        values[:, self.lags == 0] = 0.8
        values[:, self.lags == 3] = 1.0
        result = complete_tracked_shift(
            self._tracking(np.zeros(7, dtype=bool), np.full(7, np.nan)),
            self._correlation(values),
            safe_lag_range_time=2.0,
        )
        np.testing.assert_allclose(result.final_shift, 0.0)
        self.assertTrue(result.fallback_argmax_mask.all())

    def test_safe_argmax_refinement_does_not_cross_safe_lag_boundary(self) -> None:
        values = np.full((7, self.lags.size), 0.0)
        values[:, self.lags == 2] = 1.0
        values[:, self.lags == 3] = 0.9
        result = complete_tracked_shift(
            self._tracking(np.zeros(7, dtype=bool), np.full(7, np.nan)),
            self._correlation(values),
            safe_lag_range_time=2.0,
        )
        np.testing.assert_allclose(result.final_shift, 2.0)
        self.assertTrue(result.fallback_argmax_mask.all())

    def test_no_finite_zncc_remains_invalid(self) -> None:
        values = np.ones((7, self.lags.size), dtype=float)
        values[2, :] = np.nan
        valid = np.ones(7, dtype=bool)
        result = complete_tracked_shift(
            self._tracking(np.zeros(7, dtype=bool), np.full(7, np.nan)),
            self._correlation(values, valid),
            safe_lag_range_time=2.0,
        )
        self.assertTrue(np.isnan(result.final_shift[2]))
        self.assertFalse(result.fallback_local_mask[2])
        self.assertFalse(result.fallback_global_mask[2])
        self.assertFalse(result.fallback_argmax_mask[2])

    def test_dp_failure_with_finite_fallback_enters_objective_without_penalty(self) -> None:
        shape = (1, 1, 1)
        valid = np.ones(shape, dtype=bool)
        reference = build_fixed_mask(
            valid,
            valid,
            valid,
            np.ones(shape),
            np.ones(shape),
            np.ones(shape),
            np.zeros(shape, dtype=bool),
        )
        result = compute_vfsa_misfit(
            reference,
            shift_time=np.array([[[0.25]]]),
            tracked_correlation=np.ones(shape),
            window_energy_obs=np.ones(shape),
            window_energy_syn=np.ones(shape),
            boundary_flag=np.zeros(shape, dtype=bool),
            tracking_success=valid,
            path_failure_mask=np.ones(shape, dtype=bool),
            fallback_local_mask=np.ones(shape, dtype=bool),
            config=MisfitConfig(failure_penalty_time=10.0),
        )
        self.assertEqual(result.path_failure_count, 1)
        self.assertEqual(result.data_invalid_count, 0)
        self.assertEqual(result.fallback_local_count, 1)
        self.assertAlmostEqual(result.misfit_sum, 0.5 * 0.25**2)
        self.assertEqual(result.shift_time[0, 0, 0], 0.25)

    def test_track_complete_objective_chain_separates_dp_failure_from_invalid(self) -> None:
        nreceiver = 7
        ridge = int(np.flatnonzero(self.lags == 0)[0])
        waveform = np.full((nreceiver, self.lags.size), 0.1)
        waveform[:, ridge] = 0.9
        gated = np.array(waveform, copy=True)
        gated[3, :] = np.nan
        correlation = CorrelationResult(
            lags_samples=self.lags,
            lags_time=self.lags.astype(float),
            correlation=gated,
            waveform_correlation=waveform,
            valid=np.ones(nreceiver, dtype=bool),
            window_energy_obs=np.ones(nreceiver),
            window_energy_syn=np.ones(nreceiver),
            reflector=0,
            shot=0,
            max_lag_samples=5,
            dt=1.0,
        )
        tracking = track_zncc(
            correlation.correlation,
            correlation.lags_samples,
            receiver_x=np.arange(nreceiver, dtype=float),
            source_x=2.0,
            seed_lag_range_time=2.0,
            epsilon_time=1.0,
            dt=1.0,
            valid=correlation.valid,
            lag_valid=correlation.lag_valid,
        )
        completion = complete_tracked_shift(
            tracking,
            correlation,
            safe_lag_range_time=2.0,
        )
        self.assertTrue(completion.dp_failure_mask[3])
        self.assertTrue(completion.fallback_local_mask[3])
        self.assertTrue(np.isfinite(completion.final_shift[3]))

        shape = (1, 1, nreceiver)
        fixed = build_fixed_mask(
            np.ones(shape, dtype=bool),
            np.ones(shape, dtype=bool),
            np.ones(shape, dtype=bool),
            np.ones(shape),
            np.ones(shape),
            np.ones(shape),
            np.zeros(shape, dtype=bool),
        )
        result = compute_vfsa_misfit(
            fixed,
            shift_time=completion.final_shift[None, None, :],
            tracked_correlation=tracking.tracked_correlation[None, None, :],
            window_energy_obs=np.ones(shape),
            window_energy_syn=np.ones(shape),
            boundary_flag=np.zeros(shape, dtype=bool),
            tracking_success=np.isfinite(completion.final_shift)[None, None, :],
            path_failure_mask=completion.dp_failure_mask[None, None, :],
            fallback_local_mask=completion.fallback_local_mask[None, None, :],
            fallback_global_mask=completion.fallback_global_mask[None, None, :],
            fallback_argmax_mask=completion.fallback_argmax_mask[None, None, :],
            config=MisfitConfig(failure_penalty_time=10.0),
        )
        self.assertEqual(result.path_failure_count, 4)
        self.assertEqual(result.fallback_local_count, 2)
        self.assertEqual(result.fallback_global_count, 2)
        self.assertEqual(result.data_invalid_count, 0)


if __name__ == "__main__":
    unittest.main()
