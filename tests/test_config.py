from __future__ import annotations

import unittest

from WRTI.workflow import WRTIConfig, WorkflowConfigError


def _mapping(parallel=None):
    result = {
        "grid": {"x0": 0.0, "z0": 0.0, "dx": 1.0, "dz": 1.0, "nx": 4, "nz": 4},
        "window": {"type": "rectangular", "half_window_time": 2.0},
        "correlation": {
            "method": "zncc",
            "max_lag_time": 1.0,
            "seed_lag_range_time": 1.0,
        },
        "tracking": {"epsilon_time": 1.0},
        "qc": {"failure_penalty_time": 1.0},
        "misfit": {"reflector_weights": [1.0]},
    }
    if parallel is not None:
        result["parallel"] = parallel
    return result


class ParallelConfigTests(unittest.TestCase):
    def test_unified_parallel_defaults_are_sixteen(self) -> None:
        config = WRTIConfig.from_mapping(_mapping())
        self.assertEqual(config.eikonal_workers, 16)
        self.assertEqual(config.wrti_workers, 16)

    def test_unified_parallel_values_are_read(self) -> None:
        config = WRTIConfig.from_mapping(
            _mapping({"eikonal_workers": 8, "wrti_workers": 4})
        )
        self.assertEqual(config.eikonal_workers, 8)
        self.assertEqual(config.wrti_workers, 4)

    def test_worker_counts_must_be_positive_integers(self) -> None:
        with self.assertRaises(WorkflowConfigError):
            WRTIConfig.from_mapping(_mapping({"eikonal_workers": 0, "wrti_workers": 4}))
        with self.assertRaises(WorkflowConfigError):
            WRTIConfig.from_mapping(_mapping({"eikonal_workers": 4, "wrti_workers": 0}))

    def test_tracking_enhancement_settings_are_read_and_validated(self) -> None:
        mapping = _mapping()
        mapping["tracking"].update(
            {
                "enhancement_enabled": False,
                "agc_fraction": 0.5,
                "agc_floor_ratio": 0.1,
                "receiver_stack": False,
                "raw_refine_radius_samples": 2,
            }
        )
        config = WRTIConfig.from_mapping(mapping)
        self.assertFalse(config.tracking_enhancement_enabled)
        self.assertEqual(config.tracking_agc_fraction, 0.5)
        self.assertEqual(config.tracking_agc_floor_ratio, 0.1)
        self.assertFalse(config.tracking_receiver_stack)
        self.assertEqual(config.tracking_raw_refine_radius_samples, 2)

        mapping["tracking"]["agc_fraction"] = 0.0
        with self.assertRaises(WorkflowConfigError):
            WRTIConfig.from_mapping(mapping)

    def test_envelope_tracking_epsilon_is_positive_and_configurable(self) -> None:
        config = WRTIConfig.from_mapping(_mapping())
        self.assertEqual(config.envelope_tracking_epsilon_time, 0.040)

        mapping = _mapping()
        mapping["correlation"]["envelope_tracking_epsilon_time"] = 0.02
        config = WRTIConfig.from_mapping(mapping)
        self.assertEqual(config.envelope_tracking_epsilon_time, 0.02)

        mapping["correlation"]["envelope_tracking_epsilon_time"] = 0.0
        with self.assertRaises(WorkflowConfigError):
            WRTIConfig.from_mapping(mapping)

    def test_sparse_dp_bridge_defaults_and_overrides(self) -> None:
        config = WRTIConfig.from_mapping(_mapping())
        self.assertEqual(config.ownership_guard_time, 0.02)
        self.assertEqual(config.tracking_top_k_peaks, 4)
        self.assertTrue(config.bridge_enabled)
        self.assertEqual(config.bridge_max_receivers, 8)
        self.assertFalse(config.misfit.fallback_use_in_misfit)

        mapping = _mapping()
        mapping["correlation"]["ownership_guard_time"] = 0.03
        mapping["tracking"].update(
            {
                "top_k_peaks": 3,
                "smooth_weight": 0.1,
                "bridge_max_receivers": 5,
            }
        )
        mapping["misfit"]["fallback_use_in_misfit"] = True
        config = WRTIConfig.from_mapping(mapping)
        self.assertEqual(config.ownership_guard_time, 0.03)
        self.assertEqual(config.tracking_top_k_peaks, 3)
        self.assertEqual(config.tracking_smooth_weight, 0.1)
        self.assertEqual(config.bridge_max_receivers, 5)
        self.assertTrue(config.misfit.fallback_use_in_misfit)

    def test_tracking_min_correlation_is_wired_to_qc(self) -> None:
        mapping = _mapping()
        mapping["tracking"]["min_correlation"] = 0.55
        config = WRTIConfig.from_mapping(mapping)
        self.assertEqual(config.tracking_min_correlation, 0.55)
        self.assertEqual(config.misfit.min_correlation, 0.55)

    def test_debug_tracking_selection_uses_one_based_shots_and_r_names(self) -> None:
        mapping = _mapping()
        mapping["diagnostics"] = {
            "debug_tracking_shots": [31, 45],
            "debug_tracking_reflectors": ["R2", "R4"],
            "save_tracking_snapshot": True,
        }
        config = WRTIConfig.from_mapping(mapping)
        self.assertEqual(config.debug_tracking_shots, (30, 44))
        self.assertEqual(config.debug_tracking_reflectors, (1, 3))
        self.assertTrue(config.save_tracking_snapshot)


if __name__ == "__main__":
    unittest.main()
