from __future__ import annotations

import unittest

import numpy as np

from WRTI.diagnostics import warp_synthetic_by_shift


class WarpDiagnosticStep6Tests(unittest.TestCase):
    def test_positive_residual_delays_selected_synthetic_gather(self) -> None:
        synthetic = np.zeros((1, 2, 16), dtype=float)
        synthetic[0, 0, 5] = 1.0
        synthetic[0, 1, 5] = 2.0
        shifts = np.array([[[2.0, -1.0]]])

        warped = warp_synthetic_by_shift(
            synthetic,
            shifts,
            reflector=0,
            shot=0,
            dt=1.0,
        )

        self.assertEqual(int(np.argmax(warped[0])), 7)
        self.assertEqual(int(np.argmax(warped[1])), 4)
        self.assertEqual(warped.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
