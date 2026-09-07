from __future__ import annotations

import unittest

import numpy as np

from WRTI.tracking import TrackingResult
from WRTI.workflow import (
    QUADRATIC_ONLY,
    QUADRATIC_PLUS_ENVELOPE,
    recover_edge_observed_centers,
    select_observed_center_anchors,
    snap_observed_center,
    weighted_quadratic_fit,
)


class ObservedCenterRecoveryTests(unittest.TestCase):
    @staticmethod
    def _tracking(success, shifts, correlations=None, boundary=None):
        success = np.asarray(success, dtype=bool)
        shifts = np.asarray(shifts, dtype=float)
        nreceiver = success.size
        if correlations is None:
            correlations = np.where(success, 0.9, np.nan)
        if boundary is None:
            boundary = np.zeros(nreceiver, dtype=bool)
        return TrackingResult(
            path_index=np.where(success, 0, -1),
            shift_samples=shifts,
            shift_time=shifts,
            tracked_correlation=correlations,
            seed_receiver=0,
            seed_lag=0.0,
            success_mask=success,
            boundary_flag=boundary,
            dt=0.001,
            epsilon_samples=1,
            boundary_margin_samples=0,
        )

    @staticmethod
    def _data(nreceiver, nt=2000):
        return np.full((1, nreceiver, nt), np.nan, dtype=float)

    def _edge_case(self, success):
        nreceiver = len(success)
        x = np.arange(nreceiver, dtype=float)
        target = 1.0 + 0.05 * x + 0.002 * x * x
        tracking = self._tracking(success, target, np.where(success, 0.9, np.nan))
        result = recover_edge_observed_centers(
            np.zeros((1, 1, nreceiver)),
            np.ones((1, 1, nreceiver), dtype=bool),
            np.ones((1, 1, nreceiver), dtype=bool),
            np.ones((1, 1, nreceiver), dtype=float),
            ((tracking,),),
            np.array([[0.0, 0.0]]),
            x.reshape(1, nreceiver, 1).repeat(2, axis=2),
            self._data(nreceiver),
            dt=0.001,
        )
        return target, result

    def test_anchor_selection_uses_bootstrap_threshold_only(self):
        tracking = self._tracking([True, True], [0.1, 0.2], [0.8, 0.6])
        anchors = select_observed_center_anchors(
            np.ones(2, dtype=bool), tracking, correlation_threshold=0.70
        )
        np.testing.assert_array_equal(anchors, [True, False])
        np.testing.assert_array_equal(tracking.success_mask, [True, True])

    def test_negative_raw_correlation_uses_positive_tracking_strength(self):
        tracking = self._tracking([True], [0.1], [-0.9])
        tracking = tracking.__class__(
            **{**tracking.__dict__, "tracked_strength": np.array([0.9])}
        )
        anchors = select_observed_center_anchors(
            np.ones(1, dtype=bool), tracking, correlation_threshold=0.7
        )
        np.testing.assert_array_equal(anchors, [True])

    def test_right_edge_quadratic_recovery(self):
        target, result = self._edge_case([True] * 30 + [False] * 10)
        np.testing.assert_allclose(result.observed_center[0, 0, 30:], target[30:])
        np.testing.assert_array_equal(
            result.observed_center_source[0, 0, 30:], QUADRATIC_ONLY
        )

    def test_left_edge_quadratic_recovery(self):
        target, result = self._edge_case([False] * 10 + [True] * 30)
        np.testing.assert_allclose(result.observed_center[0, 0, :10], target[:10])
        np.testing.assert_array_equal(
            result.observed_center_source[0, 0, :10], QUADRATIC_ONLY
        )

    def test_recovery_snaps_to_observed_envelope(self):
        nreceiver = 11
        x = np.arange(nreceiver, dtype=float)
        target = 1.5 + 0.001 * x
        success = np.array([True] * 10 + [False])
        tracking = self._tracking(success, target, np.where(success, 0.9, np.nan))
        observed = np.zeros((1, nreceiver, 2000), dtype=float)
        peak = int(round((target[-1] + 0.025) / 0.001))
        observed[0, -1] = np.exp(-0.5 * ((np.arange(2000) - peak) / 3.0) ** 2)
        result = recover_edge_observed_centers(
            np.zeros((1, 1, nreceiver)),
            np.ones((1, 1, nreceiver), dtype=bool),
            np.ones((1, 1, nreceiver), dtype=bool),
            np.ones((1, 1, nreceiver), dtype=float),
            ((tracking,),),
            np.array([[0.0, 0.0]]),
            x.reshape(1, nreceiver, 1).repeat(2, axis=2),
            observed,
            dt=0.001,
        )
        self.assertAlmostEqual(result.observed_center[0, 0, -1], target[-1] + 0.025, places=3)
        self.assertEqual(result.observed_center_source[0, 0, -1], QUADRATIC_PLUS_ENVELOPE)

    def test_internal_gap_is_untouched(self):
        success = [True] * 15 + [False, False] + [True] * 23
        _, result = self._edge_case(success)
        self.assertTrue(np.isnan(result.observed_center[0, 0, 15:17]).all())
        self.assertEqual(result.counts["unresolved_internal_gaps"], 2)

    def test_insufficient_anchor_count_skips_recovery(self):
        _, result = self._edge_case([True] * 9 + [False] * 31)
        self.assertTrue(np.isnan(result.observed_center[0, 0, 9:]).all())
        self.assertEqual(result.counts["recovery_skipped_insufficient_anchors"], 1)

    def test_weighted_fit_downweights_low_correlation_outlier(self):
        h = np.arange(30, dtype=float)
        target = 1.0 + 0.05 * h + 0.002 * h * h
        h = np.append(h, 30.0)
        values = np.append(target, 20.0)
        corr = np.append(np.ones(30), 0.1)
        weighted = weighted_quadratic_fit(h, values, corr)
        unweighted = weighted_quadratic_fit(h, values, np.ones_like(corr))
        prediction = np.array([1.0, 35.0, 35.0**2]) @ weighted
        unweighted_prediction = np.array([1.0, 35.0, 35.0**2]) @ unweighted
        expected = 1.0 + 0.05 * 35.0 + 0.002 * 35.0**2
        self.assertLess(abs(prediction - expected), abs(unweighted_prediction - expected))

    def test_envelope_snap(self):
        dt = 0.001
        trace = np.exp(-0.5 * ((np.arange(2000) - 1525) / 3.0) ** 2)
        center, snapped = snap_observed_center(1.500, trace, dt=dt)
        self.assertTrue(snapped)
        self.assertAlmostEqual(center, 1.525, places=3)

    def test_envelope_peak_outside_range_is_not_used(self):
        dt = 0.001
        trace = np.exp(-0.5 * ((np.arange(2000) - 1570) / 2.0) ** 2)
        center, snapped = snap_observed_center(1.500, trace, dt=dt)
        self.assertTrue(snapped)
        self.assertLessEqual(center, 1.540)
        self.assertNotAlmostEqual(center, 1.570, places=3)

    def test_unavailable_envelope_falls_back_to_prediction(self):
        center, snapped = snap_observed_center(
            1.500, np.full(2000, np.nan), dt=0.001
        )
        self.assertFalse(snapped)
        self.assertEqual(center, 1.500)

    def test_snap_interval_outside_record_falls_back_to_prediction(self):
        trace = np.exp(-0.5 * ((np.arange(100) - 30) / 2.0) ** 2)
        center, snapped = snap_observed_center(0.020, trace, dt=0.001)
        self.assertFalse(snapped)
        self.assertEqual(center, 0.020)

    def test_frozen_recovery_result_is_candidate_independent(self):
        _, result = self._edge_case([True] * 30 + [False] * 10)
        frozen_center = result.observed_center.copy()
        frozen_source = result.observed_center_source.copy()
        for candidate in (np.zeros(40), np.ones(40)):
            del candidate
            np.testing.assert_array_equal(result.observed_center, frozen_center)
            np.testing.assert_array_equal(result.observed_center_source, frozen_source)
        self.assertEqual(result.observed_center_source[0, 0, 30], QUADRATIC_ONLY)


if __name__ == "__main__":
    unittest.main()
