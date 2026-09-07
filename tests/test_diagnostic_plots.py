from __future__ import annotations

import unittest

import numpy as np

from WRTI.diagnostics.plots import _candidate_failure_mask


class _TrackingStub:
    def __init__(self):
        self.success_mask = np.array([True, False, False, True])
        self.path_index = np.array([0, -1, -1, 2])
        self.shift_time = np.array([0.01, np.nan, np.nan, 0.02])
        self.boundary_flag = np.array([False, False, False, True])


class DiagnosticFailureMaskTests(unittest.TestCase):
    def test_penalty_policy_matches_objective_failure_definition(self) -> None:
        result = _candidate_failure_mask(_TrackingStub(), "penalty")
        np.testing.assert_array_equal(result, [False, True, True, False])

    def test_use_shift_policy_does_not_reject_finite_boundary_pick(self) -> None:
        result = _candidate_failure_mask(_TrackingStub(), "use_shift")
        np.testing.assert_array_equal(result, [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
