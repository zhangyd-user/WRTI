from __future__ import annotations

import unittest

import numpy as np

from WRTI.correlation import CorrelationError, compute_zncc
from WRTI.correlation.zncc import (
    _seeded_envelope_path,
    _tracking_local_rms_normalize,
    _tracking_local_buffers,
    _tracking_receiver_stack,
)
from WRTI.tracking import track_correlation_result
from WRTI.window import build_window_result


class ZNCCStep4Tests(unittest.TestCase):
    dt = 1.0
    nt = 256
    event_sample = 120
    max_lag_samples = 32

    @classmethod
    def _ricker(cls, center: int, sigma: float = 5.0) -> np.ndarray:
        samples = np.arange(cls.nt, dtype=float)
        scaled = (samples - center) / sigma
        return (1.0 - 2.0 * scaled * scaled) * np.exp(-scaled * scaled)

    @classmethod
    def _shift(cls, trace: np.ndarray, shift: int) -> np.ndarray:
        shifted = np.zeros_like(trace)
        if shift >= 0:
            shifted[shift:] = trace[: cls.nt - shift]
        else:
            shifted[:shift] = trace[-shift:]
        return shifted

    @classmethod
    def _inputs(cls, observed_trace: np.ndarray, synthetic_trace: np.ndarray):
        observed = observed_trace[None, None, :]
        synthetic = synthetic_trace[None, None, :]
        window = build_window_result(
            np.array([[[float(cls.event_sample)]]]),
            dt=cls.dt,
            t0=0.0,
            nt=cls.nt,
            half_window_time=20.0,
            max_lag_samples=cls.max_lag_samples,
        )
        return observed, synthetic, window

    def _compute(self, observed_trace: np.ndarray, synthetic_trace: np.ndarray):
        observed, synthetic, window = self._inputs(
            observed_trace, synthetic_trace
        )
        return compute_zncc(
            observed,
            synthetic,
            window,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples) * self.dt,
            dt=self.dt,
            # These legacy unit tests use dt=1 as an abstract sample unit;
            # use a wide fine basin so they continue to test the original
            # ZNCC normalization independently of the physical 40 ms default.
            envelope_fine_half_width_time=40.0,
        )

    @staticmethod
    def _peak(result):
        return int(result.lags_samples[np.nanargmax(result.correlation[0])])

    def test_identical_traces_peak_at_zero_and_one(self) -> None:
        synthetic = self._ricker(self.event_sample)
        result = self._compute(synthetic, synthetic)
        self.assertEqual(self._peak(result), 0)
        self.assertAlmostEqual(result.correlation[0, self.max_lag_samples], 1.0)

    def test_observed_delayed_by_twenty_samples_peak_is_positive_twenty(self) -> None:
        synthetic = self._ricker(self.event_sample)
        observed = self._shift(synthetic, 20)
        result = self._compute(observed, synthetic)
        self.assertEqual(self._peak(result), 20)

    def test_observed_advanced_by_thirteen_samples_peak_is_negative_thirteen(self) -> None:
        synthetic = self._ricker(self.event_sample)
        observed = self._shift(synthetic, -13)
        result = self._compute(observed, synthetic)
        self.assertEqual(self._peak(result), -13)

    def test_observed_centered_candidate_keeps_positive_lag_sign(self) -> None:
        synthetic = self._ricker(self.event_sample)
        observed = self._shift(synthetic, 10)
        observed_data = observed[None, None, :]
        synthetic_data = synthetic[None, None, :]
        windows = build_window_result(
            np.array([[[float(self.event_sample + 10)]]]),
            dt=self.dt,
            t0=0.0,
            nt=self.nt,
            half_window_time=20.0,
            max_lag_samples=self.max_lag_samples,
        )
        result = compute_zncc(
            observed_data,
            synthetic_data,
            windows,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples),
            dt=self.dt,
            envelope_fine_half_width_time=40.0,
            fixed_side="observed",
        )
        self.assertEqual(self._peak(result), 10)

    def test_observed_centered_candidate_keeps_negative_lag_sign(self) -> None:
        observed = self._ricker(self.event_sample)
        synthetic = self._shift(observed, 10)
        observed_data = observed[None, None, :]
        synthetic_data = synthetic[None, None, :]
        windows = build_window_result(
            np.array([[[float(self.event_sample)]]]),
            dt=self.dt,
            t0=0.0,
            nt=self.nt,
            half_window_time=20.0,
            max_lag_samples=self.max_lag_samples,
        )
        result = compute_zncc(
            observed_data,
            synthetic_data,
            windows,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples),
            dt=self.dt,
            envelope_fine_half_width_time=40.0,
            fixed_side="observed",
        )
        self.assertEqual(self._peak(result), -10)

    def test_amplitude_scaling_does_not_change_peak_or_zncc(self) -> None:
        synthetic = self._ricker(self.event_sample)
        observed = 4.0 * self._shift(synthetic, 20)
        result = self._compute(observed, synthetic)
        peak_index = int(np.nanargmax(result.correlation[0]))
        self.assertEqual(int(result.lags_samples[peak_index]), 20)
        self.assertAlmostEqual(result.correlation[0, peak_index], 1.0)

    def test_different_dc_offsets_do_not_change_peak(self) -> None:
        synthetic = self._ricker(self.event_sample) - 2.5
        observed = self._shift(self._ricker(self.event_sample), 20) + 4.0
        result = self._compute(observed, synthetic)
        self.assertEqual(self._peak(result), 20)

    def test_reflector_ownership_gate_uses_neighbor_midpoints(self) -> None:
        nt = 512
        centers = np.array([[[100.0]], [[200.0]], [[300.0]]])
        samples = np.arange(nt, dtype=float)
        trace = sum(
            np.exp(-0.5 * ((samples - center) / 4.0) ** 2)
            for center in centers[:, 0, 0]
        )
        observed = trace[None, None, :]
        synthetic = trace[None, None, :]
        windows = build_window_result(
            centers,
            dt=1.0,
            t0=0.0,
            nt=nt,
            half_window_time=15.0,
            max_lag_samples=100,
        )
        result = compute_zncc(
            observed,
            synthetic,
            windows,
            reflector=1,
            shot=0,
            max_lag_time=100.0,
            dt=1.0,
            use_envelope_coarse=False,
        )

        self.assertEqual(result.ownership_lag_min[0], -50.0)
        self.assertEqual(result.ownership_lag_max[0], 50.0)
        allowed_lags = result.lags_samples[result.lag_valid[0]]
        self.assertTrue(np.all(allowed_lags > -50))
        self.assertTrue(np.all(allowed_lags < 50))
        self.assertFalse(result.lag_valid[0, np.flatnonzero(result.lags_samples == -50)[0]])
        self.assertFalse(result.lag_valid[0, np.flatnonzero(result.lags_samples == 50)[0]])

    def test_observed_centered_ownership_gate_uses_observed_centers(self) -> None:
        nt = 768
        centers = np.array([[[100.0]], [[250.0]], [[550.0]]])
        samples = np.arange(nt, dtype=float)
        trace = sum(
            np.exp(-0.5 * ((samples - center) / 4.0) ** 2)
            for center in centers[:, 0, 0]
        )
        windows = build_window_result(
            centers,
            dt=1.0,
            t0=0.0,
            nt=nt,
            half_window_time=15.0,
            max_lag_samples=200,
        )
        result = compute_zncc(
            trace[None, None, :],
            trace[None, None, :],
            windows,
            reflector=1,
            shot=0,
            max_lag_time=200.0,
            dt=1.0,
            use_envelope_coarse=False,
            fixed_side="observed",
        )

        # The observed-domain event bounds are [-75, +150] samples.  Since
        # candidate time is observed_center - lag, the public lag bounds are
        # the negated interval [-150, +75].
        self.assertEqual(result.ownership_lag_min[0], -150.0)
        self.assertEqual(result.ownership_lag_max[0], 75.0)

    def test_envelope_coarse_and_waveform_fine_fields_are_readonly(self) -> None:
        samples = np.arange(self.nt, dtype=float)
        center = self.event_sample
        envelope = np.exp(-0.5 * ((samples - center) / 25.0) ** 2)
        waveform = envelope * np.cos(2.0 * np.pi * 0.01 * samples)
        observed = self._shift(waveform, 12)
        observed_data, synthetic_data, window = self._inputs(observed, waveform)
        result = compute_zncc(
            observed_data,
            synthetic_data,
            window,
            reflector=0,
            shot=0,
            max_lag_time=32.0,
            dt=1.0,
            envelope_fine_half_width_time=40.0,
        )
        self.assertEqual(result.lag_valid.shape, result.correlation.shape)
        self.assertEqual(result.coarse_lag_samples.shape, (1,))
        self.assertEqual(result.coarse_lag_time.shape, (1,))
        self.assertEqual(result.coarse_correlation_peak.shape, (1,))
        self.assertFalse(result.lag_valid.flags.writeable)
        self.assertFalse(result.coarse_lag_samples.flags.writeable)
        self.assertTrue(np.isfinite(result.coarse_lag_time[0]))

    def test_envelope_basin_then_waveform_fine_recovers_oscillatory_lag(self) -> None:
        dt = 0.001
        nt = 1200
        center = 0.5
        time = np.arange(nt, dtype=float) * dt
        synthetic_trace = np.exp(-0.5 * ((time - center) / 0.06) ** 2) * np.cos(
            2.0 * np.pi * 10.0 * (time - center)
        )
        observed_trace = np.zeros_like(synthetic_trace)
        observed_trace[15:] = synthetic_trace[:-15]
        windows = build_window_result(
            np.array([[[center]]]),
            dt=dt,
            t0=0.0,
            nt=nt,
            half_window_time=0.15,
            max_lag_samples=200,
        )
        result = compute_zncc(
            observed_trace[None, None, :],
            synthetic_trace[None, None, :],
            windows,
            reflector=0,
            shot=0,
            max_lag_time=0.2,
            dt=dt,
        )

        self.assertAlmostEqual(result.coarse_lag_time[0], 0.015, places=6)
        peak = self._peak(result)
        self.assertEqual(peak, 15)
        self.assertEqual(int(np.count_nonzero(result.lag_valid[0])), 81)
        tracking = track_correlation_result(
            result,
            receiver_x=np.array([0.0]),
            source_x=0.0,
            seed_lag_range_time=0.2,
            epsilon_time=0.02,
        )
        self.assertTrue(tracking.success_mask[0])
        self.assertAlmostEqual(tracking.shift_time[0], 0.015, places=6)

    def test_envelope_coarse_starts_at_near_zero_seed(self) -> None:
        lags = np.arange(-5, 6, dtype=int)
        correlation = np.full((7, lags.size), 0.05, dtype=float)
        zero = int(np.flatnonzero(lags == 0)[0])
        outlier = int(np.flatnonzero(lags == 5)[0])
        correlation[:, zero] = 0.9
        correlation[3, outlier] = 1.0

        path, centers, fallback, seed_receiver = _seeded_envelope_path(
            correlation,
            np.ones_like(correlation, dtype=bool),
            lags,
            receiver_x=np.arange(7, dtype=float),
            source_x=3.0,
            seed_lag_range_samples=1,
            epsilon_samples=1,
        )

        # The strong +5 peak is outside the seed range and must not become the
        # starting branch.  Both directions then remain on the zero-lag ridge.
        np.testing.assert_array_equal(path, np.full(7, zero, dtype=int))
        self.assertEqual(seed_receiver, 3)
        self.assertEqual(fallback.sum(), 0)
        np.testing.assert_array_equal(centers, np.zeros(7))

    def test_envelope_coarse_rejects_broadcastable_allowed_mask(self) -> None:
        with self.assertRaises(CorrelationError):
            _seeded_envelope_path(
                np.ones((2, 3), dtype=float),
                np.ones(3, dtype=bool),
                np.arange(3, dtype=int),
                receiver_x=np.arange(2, dtype=float),
                source_x=0.0,
                seed_lag_range_samples=1,
                epsilon_samples=1,
            )

    def test_envelope_coarse_rejects_single_remote_wrong_strong_peak(self) -> None:
        lags = np.arange(-2, 9, dtype=int)
        expected_lags = np.array([0, 1, 2, 3, 4, 5])
        correlation = np.full((6, lags.size), 0.01, dtype=float)
        for receiver, lag in enumerate(expected_lags):
            correlation[receiver, np.flatnonzero(lags == lag)[0]] = 0.8
        correlation[3, np.flatnonzero(lags == 8)[0]] = 1.0

        path, _, fallback, _ = _seeded_envelope_path(
            correlation,
            np.ones_like(correlation, dtype=bool),
            lags,
            receiver_x=np.arange(6, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=0,
            epsilon_samples=2,
        )

        np.testing.assert_array_equal(lags[path], expected_lags)
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_keeps_ownership_as_hard_gate(self) -> None:
        lags = np.arange(-5, 6, dtype=int)
        correlation = np.zeros((4, lags.size), dtype=float)
        forbidden = int(np.flatnonzero(lags == 4)[0])
        owned = int(np.flatnonzero(lags == 1)[0])
        correlation[:, forbidden] = 2.0
        correlation[:, owned] = 0.8
        ownership = np.zeros_like(correlation, dtype=bool)
        ownership[:, owned] = True

        path, _, fallback, _ = _seeded_envelope_path(
            correlation,
            ownership,
            lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=5,
            epsilon_samples=1,
        )

        np.testing.assert_array_equal(path, np.full(4, owned, dtype=int))
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_searches_only_ownership_intersection(self) -> None:
        lags = np.arange(-2, 4, dtype=int)
        correlation = np.full((3, lags.size), 0.01, dtype=float)
        ownership = np.zeros_like(correlation, dtype=bool)
        for receiver, lag in enumerate([0, 1, 1]):
            index = int(np.flatnonzero(lags == lag)[0])
            ownership[receiver, index] = True
            correlation[receiver, index] = 0.8
        correlation[1, np.flatnonzero(lags == 2)[0]] = 1.0

        path, _, fallback, _ = _seeded_envelope_path(
            correlation,
            ownership,
            lags,
            receiver_x=np.arange(3, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=0,
            epsilon_samples=2,
        )

        np.testing.assert_array_equal(lags[path], [0, 1, 1])
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_tracks_smooth_ridge_from_seed_both_directions(self) -> None:
        lags = np.arange(-5, 6, dtype=int)
        expected_lags = np.array([-3, -2, -1, 0, 1, 2, 3])
        correlation = np.full((7, lags.size), 0.01, dtype=float)
        for receiver, lag in enumerate(expected_lags):
            correlation[receiver, np.flatnonzero(lags == lag)[0]] = 0.9

        path, centers, fallback, seed_receiver = _seeded_envelope_path(
            correlation,
            np.ones_like(correlation, dtype=bool),
            lags,
            receiver_x=np.arange(7, dtype=float),
            source_x=3.0,
            seed_lag_range_samples=0,
            epsilon_samples=2,
        )

        np.testing.assert_array_equal(lags[path], expected_lags)
        np.testing.assert_allclose(centers, [-1.5, -0.5, 0, 0, 0, 0.5, 1.5])
        self.assertEqual(seed_receiver, 3)
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_uses_average_of_two_previous_lags(self) -> None:
        lags = np.arange(-2, 6, dtype=int)
        correlation = np.full((4, lags.size), 0.01, dtype=float)
        for receiver, lag in enumerate([0, 0, 1]):
            correlation[receiver, np.flatnonzero(lags == lag)[0]] = 0.8
        correlation[3, np.flatnonzero(lags == 1)[0]] = 0.8
        correlation[3, np.flatnonzero(lags == 2)[0]] = 1.0

        path, centers, fallback, _ = _seeded_envelope_path(
            correlation,
            np.ones_like(correlation, dtype=bool),
            lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=0,
            epsilon_samples=1,
        )

        # At row 3 the center is (0 + 1) / 2 = 0.5, so lag 1 is selected;
        # using only the immediately preceding lag would select the stronger 2.
        np.testing.assert_array_equal(lags[path], [0, 0, 1, 1])
        self.assertAlmostEqual(centers[3], 0.5)
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_low_correlation_does_not_stop_tracking(self) -> None:
        lags = np.arange(-3, 4, dtype=int)
        correlation = np.full((5, lags.size), -0.5, dtype=float)
        for receiver, lag in enumerate([0, 1, 2, 3, 3]):
            correlation[receiver, np.flatnonzero(lags == lag)[0]] = 0.01

        path, _, fallback, _ = _seeded_envelope_path(
            correlation,
            np.ones_like(correlation, dtype=bool),
            lags,
            receiver_x=np.arange(5, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=0,
            epsilon_samples=2,
        )

        np.testing.assert_array_equal(lags[path], [0, 1, 2, 3, 3])
        self.assertEqual(fallback.sum(), 0)

    def test_envelope_coarse_fallback_continues_after_empty_local_window(self) -> None:
        lags = np.arange(-1, 7, dtype=int)
        correlation = np.full((4, lags.size), np.nan, dtype=float)
        allowed = np.zeros_like(correlation, dtype=bool)
        for receiver, lag in enumerate([0, 5, 3, 4]):
            index = int(np.flatnonzero(lags == lag)[0])
            allowed[receiver, index] = True
            correlation[receiver, index] = 0.5

        path, centers, fallback, _ = _seeded_envelope_path(
            correlation,
            allowed,
            lags,
            receiver_x=np.arange(4, dtype=float),
            source_x=0.0,
            seed_lag_range_samples=0,
            epsilon_samples=1,
        )

        np.testing.assert_array_equal(lags[path], [0, 5, 3, 4])
        self.assertTrue(fallback[1])
        self.assertEqual(int(fallback.sum()), 1)
        self.assertAlmostEqual(centers[2], 2.5)

    def test_tracking_preprocessing_preserves_inputs_and_time_samples(self) -> None:
        samples = np.arange(81, dtype=float)
        traces = np.vstack(
            [
                np.exp(-0.5 * ((samples - 40.0) / 3.0) ** 2),
                2.0 * np.exp(-0.5 * ((samples - 40.0) / 3.0) ** 2),
            ]
        )
        original = traces.copy()

        normalized = _tracking_local_rms_normalize(traces)
        stacked = _tracking_receiver_stack(normalized, np.array([True, True]))

        np.testing.assert_array_equal(traces, original)
        np.testing.assert_array_equal(np.argmax(normalized, axis=1), [40, 40])
        np.testing.assert_array_equal(np.argmax(stacked, axis=1), [40, 40])

    def test_tracking_stack_keeps_invalid_receiver_invalid_and_unbridged(self) -> None:
        traces = np.array([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]])
        valid = np.array([True, False, True])

        stacked = _tracking_receiver_stack(traces, valid)

        np.testing.assert_array_equal(stacked[1], traces[1])
        np.testing.assert_array_equal(stacked[0], traces[0])
        np.testing.assert_array_equal(stacked[2], traces[2])

    def test_tracking_local_buffers_align_moveout_before_stacking(self) -> None:
        nt = 256
        centers = np.array([100, 112])
        samples = np.arange(nt, dtype=float)
        observed = np.vstack(
            [np.exp(-0.5 * ((samples - center) / 2.0) ** 2) for center in centers]
        )
        windows = build_window_result(
            centers[None, None, :].astype(float),
            dt=1.0,
            t0=0.0,
            nt=nt,
            half_window_time=12.0,
            max_lag_samples=8,
        )

        local_obs, _, target_left = _tracking_local_buffers(
            observed,
            observed,
            windows,
            reflector=0,
            shot=0,
            receiver_valid=np.array([True, True]),
            max_lag_samples=8,
        )
        stacked = _tracking_receiver_stack(local_obs, np.array([True, True]))

        np.testing.assert_array_equal(np.argmax(local_obs, axis=1), np.argmax(stacked, axis=1))
        reflector_offset = (
            windows.center_sample[0, 0] - windows.left_sample[0, 0]
        )
        np.testing.assert_array_equal(
            np.argmax(stacked, axis=1), target_left + reflector_offset
        )

    def test_tracking_agc_uses_reflector_local_buffer_not_record_length(self) -> None:
        nt = 1024
        center = 512
        samples = np.arange(nt, dtype=float)
        trace = np.exp(-0.5 * ((samples - center) / 3.0) ** 2)
        windows = build_window_result(
            np.array([[[float(center)]]]),
            dt=1.0,
            t0=0.0,
            nt=nt,
            half_window_time=20.0,
            max_lag_samples=16,
        )

        local, _, _ = _tracking_local_buffers(
            trace[None, :],
            trace[None, :],
            windows,
            reflector=0,
            shot=0,
            receiver_valid=np.array([True]),
            max_lag_samples=16,
        )
        normalized = _tracking_local_rms_normalize(local)

        self.assertEqual(local.shape[1], 73)
        self.assertLess(local.shape[1], nt)
        self.assertEqual(np.argmax(normalized[0]), np.argmax(local[0]))

    def test_tracking_enhancement_can_be_disabled_without_changing_raw_zncc(self) -> None:
        synthetic = self._ricker(self.event_sample)
        observed = self._shift(synthetic, 8)
        observed_data, synthetic_data, window = self._inputs(observed, synthetic)
        enhanced = compute_zncc(
            observed_data,
            synthetic_data,
            window,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples),
            dt=self.dt,
            envelope_fine_half_width_time=40.0,
        )
        disabled = compute_zncc(
            observed_data,
            synthetic_data,
            window,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples),
            dt=self.dt,
            envelope_fine_half_width_time=40.0,
            tracking_enhancement_enabled=False,
        )

        self.assertIsNotNone(enhanced.tracking_correlation)
        self.assertIsNone(disabled.tracking_correlation)
        np.testing.assert_array_equal(enhanced.waveform_correlation, disabled.waveform_correlation)
        np.testing.assert_array_equal(enhanced.correlation, disabled.correlation)

    def test_tracking_enhancement_strengthens_a_coherent_weak_event(self) -> None:
        centers = [116, 118, 120, 122, 124]
        synthetic = np.zeros((1, 5, self.nt), dtype=float)
        observed = np.zeros_like(synthetic)
        interference_offsets = [-25, 25, -15, 30, -30]
        for receiver, (center, interference_offset) in enumerate(
            zip(centers, interference_offsets)
        ):
            synthetic[0, receiver] = self._ricker(center)
            observed[0, receiver] = (
                0.6 * self._ricker(center + 5)
                + self._ricker(center + interference_offset)
            )
        windows = build_window_result(
            np.array(centers, dtype=float)[None, None, :],
            dt=self.dt,
            t0=0.0,
            nt=self.nt,
            half_window_time=20.0,
            max_lag_samples=self.max_lag_samples,
        )

        result = compute_zncc(
            observed,
            synthetic,
            windows,
            reflector=0,
            shot=0,
            max_lag_time=float(self.max_lag_samples),
            dt=self.dt,
            use_envelope_coarse=False,
        )
        raw_peak = result.lags_samples[np.nanargmax(result.waveform_correlation, axis=1)]
        guide_peak = result.lags_samples[np.nanargmax(result.tracking_correlation, axis=1)]

        self.assertFalse(np.all(raw_peak[1:4] == 5))
        np.testing.assert_array_equal(guide_peak[1:4], [5, 5, 5])


if __name__ == "__main__":
    unittest.main()
