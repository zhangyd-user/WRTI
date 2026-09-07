from __future__ import annotations

import unittest
import warnings

import numpy as np

from WRTI.window import WindowOverlapWarning, build_window_result


class WindowStep3Tests(unittest.TestCase):
    def test_center_fields_and_scalar_or_per_reflector_half_window(self) -> None:
        tref = np.array(
            [
                [[1.27, 1.37]],
                [[2.44, 2.54]],
            ]
        )
        result = build_window_result(
            tref,
            dt=0.1,
            t0=0.0,
            nt=100,
            half_window_time=[0.2, 0.3],
        )

        np.testing.assert_allclose(result.center_time, tref)
        np.testing.assert_allclose(result.center_float[:, 0, 0], [12.7, 24.4])
        np.testing.assert_array_equal(result.center_sample[:, 0, 0], [13, 24])
        np.testing.assert_array_equal(result.left_sample[:, 0, 0], [11, 22])
        np.testing.assert_array_equal(result.right_sample[:, 0, 0], [14, 27])
        self.assertTrue(result.valid.all())

        scalar = build_window_result(
            tref[:1], dt=0.1, t0=0.0, nt=100, half_window_time=0.2
        )
        np.testing.assert_allclose(scalar.half_window_time, [0.2])

    def test_strict_window_and_lag_boundaries(self) -> None:
        tref = np.array([[[0.15, 0.55, 9.85]]])
        result = build_window_result(
            tref,
            dt=0.1,
            t0=0.0,
            nt=100,
            half_window_time=0.2,
            max_lag_samples=1,
        )

        # First trace is valid: [0, 3] plus one lag sample stays in bounds.
        # Middle trace is valid.  Last trace exceeds the right boundary.
        np.testing.assert_array_equal(result.valid, [[[False, True, False]]])
        self.assertEqual(result.left_sample[0, 0, 0], 0)
        self.assertEqual(result.right_sample[0, 0, 0], 3)

    def test_rectangular_weights_and_logical_mask_are_on_demand(self) -> None:
        result = build_window_result(
            np.array([[[0.5]]]),
            dt=0.1,
            t0=0.0,
            nt=20,
            half_window_time=0.2,
            window_type="rectangular",
        )

        mask = result.mask_for_trace(0, 0, 0)
        weights = result.weights_for_trace(0, 0, 0)
        np.testing.assert_array_equal(np.flatnonzero(mask), [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(weights[mask], np.ones(5))
        np.testing.assert_array_equal(weights[~mask], np.zeros(15))
        self.assertFalse(hasattr(result, "mask"))

    def test_tukey_weights_have_requested_length_and_taper(self) -> None:
        result = build_window_result(
            np.array([[[0.5]]]),
            dt=0.1,
            t0=0.0,
            nt=20,
            half_window_time=0.2,
            window_type="tukey",
            tukey_alpha=0.5,
        )

        mask = result.mask_for_trace(0, 0, 0)
        local = result.weights_for_trace(0, 0, 0)[mask]
        self.assertEqual(local.size, 5)
        self.assertAlmostEqual(local[0], 0.0)
        self.assertAlmostEqual(local[-1], 0.0)
        self.assertAlmostEqual(local[2], 1.0)
        self.assertTrue(np.all(local >= 0.0))
        self.assertTrue(np.all(local <= 1.0))

    def test_adjacent_window_overlap_warns_without_modification(self) -> None:
        tref = np.array([[[1.0]], [[1.2]]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = build_window_result(
                tref,
                dt=0.1,
                t0=0.0,
                nt=50,
                half_window_time=[0.2, 0.2],
            )

        self.assertTrue(any(item.category is WindowOverlapWarning for item in caught))
        self.assertTrue(result.overlap[0, 0, 0])
        self.assertAlmostEqual(result.overlap_delta_time[0, 0, 0], 0.2)
        np.testing.assert_allclose(result.reference_traveltime[:, 0, 0], [1.0, 1.2])

    def test_window_state_is_immutable(self) -> None:
        result = build_window_result(
            np.array([[[0.5]]]), dt=0.1, t0=0.0, nt=20, half_window_time=0.2
        )
        with self.assertRaises(ValueError):
            result.center_sample[0, 0, 0] = 0


if __name__ == "__main__":
    unittest.main()
