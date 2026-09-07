from __future__ import annotations

import unittest

import numpy as np

from WRTI.correlation import CorrelationResult
from WRTI.tracking import track_correlation_result, track_zncc
from WRTI.tracking.tracker import _assert_path_continuity
from WRTI.tracking import boundary_qc_mask, low_correlation_qc_mask, path_failure_mask


class TrackingStep5Tests(unittest.TestCase):
    lags = np.arange(-5, 6, dtype=int)
    receiver_x = np.arange(9, dtype=float) * 10.0
    ridge = np.array([-2, -1, -1, 0, 0, 1, 1, 2, 2], dtype=int)

    def _ridge_correlation(self) -> np.ndarray:
        correlation = np.full((self.receiver_x.size, self.lags.size), 0.05)
        for receiver, lag in enumerate(self.ridge):
            index = int(np.flatnonzero(self.lags == lag)[0])
            correlation[receiver, index] = 1.0
            if index > 0:
                correlation[receiver, index - 1] = 0.7
            if index + 1 < self.lags.size:
                correlation[receiver, index + 1] = 0.7
        return correlation

    def _track(self, correlation: np.ndarray, **kwargs):
        return track_zncc(
            correlation,
            self.lags,
            self.receiver_x,
            source_x=40.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
            **kwargs,
        )

    def test_global_dp_recovers_a_smooth_strong_ridge(self) -> None:
        correlation = self._ridge_correlation()
        # Isolated high samples must not be selected when they cannot be
        # connected to the globally optimal continuous ridge.
        correlation[3, np.flatnonzero(self.lags == 5)[0]] = 2.0
        correlation[6, np.flatnonzero(self.lags == -5)[0]] = 2.0

        result = self._track(correlation)

        np.testing.assert_array_equal(
            result.path_index,
            np.array([3, 4, 4, 5, 5, 6, 6, 7, 7]),
        )
        np.testing.assert_array_equal(result.shift_samples, self.ridge.astype(float))
        self.assertEqual(result.seed_receiver, 4)
        self.assertEqual(result.seed_lag, 0.0)
        self.assertTrue(result.success_mask.all())
        self.assertFalse(result.boundary_flag.any())

    def test_low_correlation_row_does_not_stop_a_valid_path(self) -> None:
        correlation = self._ridge_correlation()
        ridge_index = int(np.flatnonzero(self.lags == 1)[0])
        correlation[5, :] = 0.1
        correlation[5, ridge_index] = 0.4

        result = self._track(correlation, min_correlation=0.55)

        self.assertTrue(result.success_mask.all())
        self.assertEqual(result.path_index[5], ridge_index)
        self.assertEqual(result.tracked_correlation[5], 0.4)

        np.testing.assert_array_equal(path_failure_mask(result), np.zeros(9, dtype=bool))
        self.assertTrue(low_correlation_qc_mask(result, 0.55)[5])
        self.assertFalse(boundary_qc_mask(result).any())

    def test_plateau_without_strict_local_maximum_is_trackable(self) -> None:
        correlation = np.full((5, self.lags.size), 0.05)
        lag_zero = int(np.flatnonzero(self.lags == 0)[0])
        correlation[:, lag_zero : lag_zero + 2] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=2.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        self.assertTrue(result.success_mask.all())
        self.assertTrue(np.isin(result.path_index, [lag_zero, lag_zero + 1]).all())
        np.testing.assert_allclose(result.tracked_correlation, 1.0)

    def test_global_score_beats_a_greedy_local_branch(self) -> None:
        receiver_x = np.arange(5, dtype=float) * 10.0
        correlation = np.full((receiver_x.size, self.lags.size), -0.2)
        branch_a = [0, 0, 0, 1, 1]
        branch_b = [1, 1, 1, 2, 2]
        for receiver, lag in enumerate(branch_a):
            correlation[receiver, np.flatnonzero(self.lags == lag)[0]] = 0.85
        for receiver, lag in enumerate(branch_b):
            correlation[receiver, np.flatnonzero(self.lags == lag)[0]] = 0.95
        # At the seed receiver, the local branch A peak is stronger.  A
        # global cumulative objective must nevertheless select branch B.
        correlation[2, np.flatnonzero(self.lags == 0)[0]] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=receiver_x,
            source_x=20.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        self.assertTrue(result.success_mask.all())
        self.assertLessEqual(
            np.max(np.abs(np.diff(self.lags[result.path_index]))), 1
        )

    def test_wrong_isolated_seed_peak_is_not_preselected(self) -> None:
        receiver_x = np.arange(5, dtype=float) * 10.0
        correlation = np.full((receiver_x.size, self.lags.size), -0.2)
        correct = int(np.flatnonzero(self.lags == 0)[0])
        wrong = int(np.flatnonzero(self.lags == 2)[0])
        correlation[:, correct] = 0.9
        correlation[2, wrong] = 1.5

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=receiver_x,
            source_x=20.0,
            seed_lag_range_time=3.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.path_index, [correct] * 5)
        self.assertEqual(result.seed_lag, 0.0)

    def test_invalid_middle_row_stops_the_seed_centered_direction(self) -> None:
        receiver_x = np.arange(5, dtype=float) * 10.0
        correlation = np.full((receiver_x.size, self.lags.size), -0.2)
        ridge_index = int(np.flatnonzero(self.lags == 0)[0])
        correlation[:, ridge_index] = 0.9
        correlation[2, :] = np.nan

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=receiver_x,
            source_x=10.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(
            result.success_mask,
            [True, True, False, False, False],
        )
        self.assertEqual(result.path_index[2], -1)
        self.assertTrue(np.isnan(result.shift_time[2]))
        self.assertTrue(np.isfinite(result.shift_time[[0, 1]]).all())
        self.assertTrue(np.isnan(result.shift_time[[3, 4]]).all())

    def test_boundary_is_qc_flag_not_a_hard_stop(self) -> None:
        receiver_x = np.arange(4, dtype=float) * 10.0
        correlation = np.full((receiver_x.size, self.lags.size), -0.2)
        edge = int(np.flatnonzero(self.lags == 5)[0])
        correlation[:, edge] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=receiver_x,
            source_x=10.0,
            seed_lag_range_time=5.0,
            epsilon_time=1.0,
            dt=1.0,
            boundary_margin_samples=1,
        )

        self.assertTrue(result.success_mask.all())
        self.assertTrue(result.boundary_flag.all())
        self.assertEqual(result.seed_lag, 5.0)

    def test_three_point_parabolic_refinement_is_applied_after_integer_path(self) -> None:
        correlation = np.full((3, self.lags.size), 0.1)
        center = int(np.flatnonzero(self.lags == 0)[0])
        correlation[:, center - 1] = 0.8
        correlation[:, center] = 1.0
        correlation[:, center + 1] = 0.6

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.array([0.0, 10.0, 20.0]),
            source_x=10.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.path_index, [5, 5, 5])
        np.testing.assert_allclose(result.shift_samples, -1.0 / 6.0)
        np.testing.assert_allclose(result.shift_time, -1.0 / 6.0)

    def test_per_lag_validity_is_respected_by_global_dp(self) -> None:
        correlation = np.full((self.receiver_x.size, self.lags.size), 0.1)
        allowed = np.zeros_like(correlation, dtype=bool)
        desired = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        for receiver, lag in enumerate(desired):
            index = int(np.flatnonzero(self.lags == lag)[0])
            allowed[receiver, index] = True
            correlation[receiver, index] = 1.0
            # A stronger state outside the Step 4 gate must not enter DP.
            wrong = int(np.flatnonzero(self.lags == 5)[0])
            correlation[receiver, wrong] = 2.0

        result = self._track(correlation, lag_valid=allowed)

        np.testing.assert_array_equal(
            result.path_index,
            [int(np.flatnonzero(self.lags == lag)[0]) for lag in desired],
        )
        self.assertTrue(result.success_mask.all())

    def test_disconnected_finite_states_do_not_restart_waveform_dp(self) -> None:
        correlation = np.full((4, self.lags.size), np.nan, dtype=float)
        allowed = np.zeros_like(correlation, dtype=bool)
        positive = int(np.flatnonzero(self.lags == 4)[0])
        negative = int(np.flatnonzero(self.lags == -4)[0])
        correlation[:2, positive] = 0.9
        correlation[2:, negative] = 0.9
        allowed[:2, positive] = True
        allowed[2:, negative] = True

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=0.0,
            seed_lag_range_time=5.0,
            epsilon_time=1.0,
            dt=1.0,
            lag_valid=allowed,
        )

        np.testing.assert_array_equal(result.success_mask, [True, True, False, False])
        np.testing.assert_array_equal(result.path_index[:2], [positive, positive])

    def test_waveform_continuity_assertion_rejects_impossible_jump(self) -> None:
        with self.assertRaisesRegex(AssertionError, "hard receiver-to-receiver"):
            _assert_path_continuity(
                np.array([5, 6, 10], dtype=int),
                self.lags,
                epsilon_samples=1,
            )

    def test_seed_lag_range_does_not_limit_waveform_path(self) -> None:
        lags = np.arange(-200, 201, dtype=int)
        correlation = np.full((5, lags.size), -1.0)
        ridge = int(np.flatnonzero(lags == 150)[0])
        correlation[:, ridge] = 1.0

        result = track_zncc(
            correlation,
            lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=2.0,
            seed_lag_range_time=0.1,
            epsilon_time=0.001,
            dt=0.001,
        )

        self.assertEqual(result.seed_lag, 150.0)
        self.assertTrue(result.success_mask.all())

    def test_seed_tracks_right_with_hard_continuity(self) -> None:
        correlation = np.full((4, self.lags.size), np.nan)
        desired = [0, 1, 2, 3]
        for receiver, lag in enumerate(desired):
            correlation[receiver, self.lags.tolist().index(lag)] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=0.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.shift_samples, desired)

    def test_seed_tracks_left_with_hard_continuity(self) -> None:
        correlation = np.full((4, self.lags.size), np.nan)
        desired = [0, 1, 2, 3]
        for receiver, lag in enumerate(desired):
            correlation[receiver, self.lags.tolist().index(lag)] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=3.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.shift_samples, desired)

    def test_seed_tracks_both_sides_as_one_path(self) -> None:
        correlation = np.full((5, self.lags.size), np.nan)
        desired = [-2, -1, 0, 1, 2]
        for receiver, lag in enumerate(desired):
            correlation[receiver, self.lags.tolist().index(lag)] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=2.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        self.assertTrue(result.success_mask.all())
        np.testing.assert_array_equal(result.shift_samples, desired)

    def test_left_and_right_share_one_seed_state(self) -> None:
        correlation = np.full((5, self.lags.size), np.nan)
        left = self.lags.tolist().index(-1)
        right = self.lags.tolist().index(1)
        correlation[0:2, left] = 1.0
        correlation[2, [left, right]] = 0.5
        correlation[3:5, right] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=2.0,
            seed_lag_range_time=0.0,
            epsilon_time=2.0,
            dt=1.0,
        )

        self.assertTrue(result.success_mask.all())
        np.testing.assert_array_equal(result.path_index, [left, left, left, right, right])
        self.assertEqual(result.seed_lag, -1.0)

    def test_direction_does_not_restart_after_unreachable_receiver(self) -> None:
        correlation = np.full((5, self.lags.size), np.nan)
        zero = self.lags.tolist().index(0)
        correlation[[0, 1, 3, 4], zero] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=0.0,
            seed_lag_range_time=0.0,
            epsilon_time=0.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.success_mask, [True, True, False, False, False])

    def test_invalid_seed_does_not_start_from_another_receiver(self) -> None:
        correlation = np.full((5, self.lags.size), np.nan)
        zero = self.lags.tolist().index(0)
        correlation[[0, 1, 3, 4], zero] = 1.0

        result = track_zncc(
            correlation,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=2.0,
            seed_lag_range_time=0.0,
            epsilon_time=0.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.path_index, [zero, zero, -1, -1, -1])
        self.assertTrue(np.isfinite(result.shift_time[:2]).all())
        self.assertFalse(result.success_mask[2:].any())

    def test_raw_measurement_dp_refines_the_guide_path(self) -> None:
        guide = np.full((3, self.lags.size), 0.1)
        raw = np.full((3, self.lags.size), 0.1)
        guide_index = int(np.flatnonzero(self.lags == 1)[0])
        raw_index = int(np.flatnonzero(self.lags == 0)[0])
        guide[:, guide_index] = 1.0
        raw[:, guide_index] = 0.8
        raw[:, raw_index] = 1.0

        result = track_zncc(
            guide,
            self.lags,
            receiver_x=np.arange(3, dtype=float),
            source_x=1.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
            measurement_correlation=raw,
            raw_refine_radius_samples=1,
        )

        np.testing.assert_array_equal(result.path_index, [raw_index] * 3)
        np.testing.assert_allclose(result.tracked_correlation, 1.0)

    def test_raw_refinement_and_quality_use_measurement_correlation(self) -> None:
        guide = np.full((3, self.lags.size), 0.1)
        raw = np.full((3, self.lags.size), 0.1)
        center = int(np.flatnonzero(self.lags == 0)[0])
        guide[:, center - 1 : center + 2] = [0.6, 1.0, 0.8]
        raw[:, center - 1 : center + 2] = [0.8, 1.0, 0.6]

        result = track_zncc(
            guide,
            self.lags,
            receiver_x=np.arange(3, dtype=float),
            source_x=1.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
            measurement_correlation=raw,
        )

        np.testing.assert_array_equal(result.path_index, [center] * 3)
        np.testing.assert_allclose(result.shift_samples, -1.0 / 6.0)
        np.testing.assert_allclose(result.tracked_correlation, 1.0)

    def test_missing_guide_does_not_delete_raw_receiver_coverage(self) -> None:
        guide = np.full((5, self.lags.size), np.nan)
        raw = np.full((5, self.lags.size), 0.1)
        center = int(np.flatnonzero(self.lags == 0)[0])
        guide[[0, 1], center] = 1.0

        result = track_zncc(
            guide,
            self.lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=0.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
            dt=1.0,
            measurement_correlation=raw,
        )

        self.assertTrue(result.success_mask.all())

    def test_correlation_result_wrapper_uses_tracking_for_guide_and_waveform_for_measurement(self) -> None:
        guide = np.full((1, self.lags.size), 0.1)
        raw = np.full((1, self.lags.size), 0.1)
        guide_index = int(np.flatnonzero(self.lags == 1)[0])
        raw_index = int(np.flatnonzero(self.lags == 0)[0])
        guide[0, guide_index] = 1.0
        raw[0, raw_index] = 1.0
        result = CorrelationResult(
            lags_samples=self.lags,
            lags_time=self.lags.astype(float),
            correlation=guide,
            tracking_correlation=guide,
            waveform_correlation=raw,
            valid=np.array([True]),
            window_energy_obs=np.array([1.0]),
            window_energy_syn=np.array([1.0]),
            reflector=0,
            shot=0,
            max_lag_samples=5,
            dt=1.0,
        )

        tracking = track_correlation_result(
            result,
            receiver_x=np.array([0.0]),
            source_x=0.0,
            seed_lag_range_time=0.0,
            epsilon_time=1.0,
        )

        self.assertEqual(tracking.path_index[0], raw_index)
        self.assertEqual(tracking.tracked_correlation[0], 1.0)

    def test_isolated_stronger_peaks_do_not_beat_a_continuous_raw_ridge(self) -> None:
        lags = np.arange(-6, 7, dtype=int)
        raw = np.full((7, lags.size), -0.4)
        true = int(np.flatnonzero(lags == 0)[0])
        false = int(np.flatnonzero(lags == 5)[0])
        raw[:, true] = 0.8
        raw[[2, 5], false] = 0.95

        result = track_zncc(
            raw,
            lags,
            np.arange(7, dtype=float),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
        )

        np.testing.assert_array_equal(result.path_index, np.full(7, true))

    def test_strong_raw_ridge_can_leave_wrong_coarse_prior(self) -> None:
        lags = np.arange(-6, 7, dtype=int)
        raw = np.full((6, lags.size), -0.4)
        true = int(np.flatnonzero(lags == 0)[0])
        wrong = int(np.flatnonzero(lags == 4)[0])
        raw[:, true] = 0.9
        raw[:, wrong] = 0.2

        result = track_zncc(
            raw,
            lags,
            np.arange(6, dtype=float),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
            coarse_lag_samples=np.array([0.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        )

        np.testing.assert_array_equal(result.path_index, np.full(6, true))
        np.testing.assert_array_equal(
            result.coarse_deviation_samples,
            [0.0, -4.0, -4.0, -4.0, -4.0, -4.0],
        )

    def test_short_continuous_false_branch_loses_to_long_true_ridge(self) -> None:
        lags = np.arange(-4, 5, dtype=int)
        raw = np.full((8, lags.size), -0.4)
        true = int(np.flatnonzero(lags == 0)[0])
        false = int(np.flatnonzero(lags == 2)[0])
        raw[:, true] = 0.8
        raw[3:5, false] = 0.95

        result = track_zncc(
            raw,
            lags,
            np.arange(8, dtype=float),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=2.0,
            dt=1.0,
            coarse_lag_samples=np.zeros(8),
            smooth_weight=0.1,
        )

        np.testing.assert_array_equal(result.path_index, np.full(8, true))

    def test_relaxed_ownership_remains_a_hard_tracker_constraint(self) -> None:
        raw = np.full((self.receiver_x.size, self.lags.size), 0.1)
        owned = int(np.flatnonzero(self.lags == 0)[0])
        forbidden = int(np.flatnonzero(self.lags == 4)[0])
        raw[:, owned] = 0.8
        raw[:, forbidden] = 1.0
        ownership = np.zeros_like(raw, dtype=bool)
        ownership[:, owned] = True

        result = self._track(raw, lag_valid=ownership)

        np.testing.assert_array_equal(
            result.path_index,
            np.full(self.receiver_x.size, owned),
        )

    def test_two_sided_bridge_reopens_raw_zncc_not_interpolated_shift(self) -> None:
        lags = np.arange(-6, 7, dtype=int)
        raw = np.full((6, lags.size), -0.5)
        true_lags = np.array([0, 0, 1, 1, 2, 2])
        for receiver, lag in enumerate(true_lags):
            raw[receiver, np.flatnonzero(lags == lag)[0]] = 0.8
        false = int(np.flatnonzero(lags == 5)[0])
        raw[2:4, false] = 0.95

        result = track_zncc(
            raw,
            lags,
            np.arange(6, dtype=float) * 500.0,
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
            top_k_peaks=1,
            bridge_half_width_time=2.0,
            bridge_max_width_time=3.0,
            bridge_max_receivers=4,
        )

        np.testing.assert_array_equal(result.shift_samples, true_lags)
        self.assertTrue(result.bridge_success_mask.any())

    def test_one_sided_bridge_uses_recent_raw_ridge_slope(self) -> None:
        lags = np.arange(-6, 7, dtype=int)
        raw = np.full((5, lags.size), -0.5)
        true_lags = np.arange(5)
        for receiver, lag in enumerate(true_lags):
            raw[receiver, np.flatnonzero(lags == lag)[0]] = 0.8
        false = int(np.flatnonzero(lags == -5)[0])
        raw[2:, false] = 0.95

        result = track_zncc(
            raw,
            lags,
            np.arange(5, dtype=float),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
            top_k_peaks=1,
            bridge_half_width_time=1.0,
            bridge_max_width_time=2.0,
            bridge_max_receivers=3,
        )

        np.testing.assert_array_equal(result.shift_samples, true_lags)
        self.assertTrue(result.bridge_success_mask.any())

    def test_gap_longer_than_bridge_limit_remains_measurement_failure(self) -> None:
        lags = np.arange(-3, 4, dtype=int)
        raw = np.full((8, lags.size), np.nan)
        zero = int(np.flatnonzero(lags == 0)[0])
        raw[:2, zero] = 0.8
        raw[6:, zero] = 0.8

        result = track_zncc(
            raw,
            lags,
            np.arange(8, dtype=float),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=1.0,
            dt=1.0,
            bridge_max_receivers=2,
        )

        np.testing.assert_array_equal(
            result.success_mask,
            [True, True, False, False, False, False, False, False],
        )

    def test_dp_bridge_provenance_and_candidate_counts_are_explicit(self) -> None:
        correlation = self._ridge_correlation()
        result = self._track(correlation)

        self.assertTrue(result.dp_success_mask.all())
        self.assertFalse(result.bridge_success_mask.any())
        np.testing.assert_array_equal(
            result.success_mask,
            result.dp_success_mask | result.bridge_success_mask,
        )
        self.assertTrue(np.all((result.candidate_count >= 1) & (result.candidate_count <= 8)))

    def test_confirmed_long_gap_restarts_as_raw_measurement(self) -> None:
        lags = np.arange(-3, 4)
        raw = np.full((8, lags.size), np.nan)
        raw[:2, lags.tolist().index(0)] = 0.9
        raw[5:, lags.tolist().index(1)] = 0.9
        result = track_zncc(raw, lags, np.arange(8.0), 0.0, 1.0, 1.0, 1.0, max_skip_rows=1, restart_confirm_rows=3)
        np.testing.assert_array_equal(result.success_mask, [True, True, False, False, False, True, True, True])
        self.assertTrue(np.all(result.provenance[5:] == "PRIMARY_DP"))

    def test_bootstrap_can_disable_independent_restart(self) -> None:
        lags = np.arange(-3, 4)
        raw = np.full((8, lags.size), np.nan)
        raw[:2, lags.tolist().index(0)] = 0.9
        raw[5:, lags.tolist().index(1)] = 0.9

        result = track_zncc(
            raw, lags, np.arange(8.0), 0.0, 1.0, 1.0, 1.0,
            restart_enabled=False,
        )

        np.testing.assert_array_equal(
            result.success_mask,
            [True, True, False, False, False, False, False, False],
        )

    def test_auto_polarity_tracks_negative_ridge_with_positive_strength(self) -> None:
        lags = np.arange(-3, 4)
        raw = np.full((5, lags.size), 0.1)
        raw[:, lags.tolist().index(1)] = -0.9
        result = track_zncc(raw, lags, np.arange(5.0), 0.0, 2.0, 1.0, 1.0)
        self.assertTrue(result.success_mask.all())
        self.assertTrue(np.all(result.tracked_polarity == -1))
        self.assertTrue(np.allclose(result.tracked_strength, 0.9))

    def test_auto_polarity_keeps_one_global_sign(self) -> None:
        lags = np.arange(-3, 4)
        raw = np.full((7, lags.size), 0.1)
        raw[:, lags.tolist().index(0)] = 0.8
        # A stronger opposite-polarity lobe occupies only part of the gather.
        raw[2:5, lags.tolist().index(2)] = -0.99

        result = track_zncc(raw, lags, np.arange(7.0), 3.0, 2.0, 1.0, 1.0)

        self.assertTrue(result.success_mask.all())
        self.assertTrue(np.all(result.tracked_polarity == 1))
        np.testing.assert_array_equal(result.path_index, [lags.tolist().index(0)] * 7)

    def test_independent_restart_components_are_not_forced_continuous(self) -> None:
        lags = np.arange(0, 401, dtype=int)
        raw = np.full((4, lags.size), np.nan)
        raw[0, 0] = 0.9
        raw[1:, 350] = 0.9

        result = track_zncc(
            raw,
            lags,
            np.arange(4.0),
            source_x=0.0,
            seed_lag_range_time=1.0,
            epsilon_time=15.0,
            dt=1.0,
            restart_confirm_rows=3,
        )

        self.assertTrue(result.success_mask.all())
        self.assertNotEqual(result.component_id[0], result.component_id[1])

    def test_wrong_middle_coarse_is_only_a_soft_guide(self) -> None:
        lags = np.arange(-150, 151, dtype=int)
        raw = np.full((9, lags.size), -0.2)
        raw[:, 150] = 0.9  # lag = 0
        coarse = np.array([0, 0, 0, 100, 100, 100, 0, 0, 0], dtype=float)

        result = track_zncc(
            raw, lags, np.arange(9.0), 4.0, 20.0, 20.0, 1.0,
            coarse_lag_samples=coarse,
        )

        np.testing.assert_array_equal(result.path_index, np.full(9, 150))

    def test_fallback_coarse_does_not_bias_the_raw_path(self) -> None:
        lags = np.arange(-20, 21, dtype=int)
        raw = np.full((6, lags.size), -0.2)
        raw[:, 20] = 0.9

        result = track_zncc(
            raw, lags, np.arange(6.0), 2.0, 3.0, 3.0, 1.0,
            coarse_lag_samples=np.full(6, 15.0),
            coarse_fallback_mask=np.ones(6, dtype=bool),
        )

        np.testing.assert_array_equal(result.path_index, np.full(6, 20))

    def test_seed_range_rejects_a_stronger_far_seed_peak(self) -> None:
        lags = np.arange(-20, 21, dtype=int)
        raw = np.full((7, lags.size), -0.3)
        raw[:, 20] = 0.8
        raw[:, 35] = 0.95

        result = track_zncc(
            raw, lags, np.arange(7.0), 3.0, 2.0, 2.0, 1.0,
            coarse_lag_samples=np.zeros(7),
        )

        np.testing.assert_array_equal(result.path_index, np.full(7, 20))


if __name__ == "__main__":
    unittest.main()
