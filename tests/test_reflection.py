from __future__ import annotations

import unittest

import numpy as np

from WRTI.traveltime import (
    build_source_receiver_fields,
    compute_reflection_traveltimes,
)


def analytic_table_backend(z_axis, x_axis, sources, receivers, velocity, mode="eikonal"):
    """Test double with the exact homogeneous point-source solution."""

    del mode
    v = float(velocity[0, 0])
    grid_x = x_axis[:, None]
    grid_z = z_axis[None, :]

    def fields(points):
        batches = []
        for x_source, z_source in points:
            field = np.sqrt((grid_x - x_source) ** 2 + (grid_z - z_source) ** 2) / v
            batches.append(field.reshape(-1))
        return np.stack(batches, axis=1)

    return fields(sources), fields(receivers)


class ReflectionTraveltimeStep2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.arange(0.0, 401.0, 10.0)
        self.z = np.arange(0.0, 301.0, 10.0)
        self.velocity = 2000.0 * np.ones((self.x.size, self.z.size))
        self.interface = np.column_stack(
            (
                np.full(self.x.size, 20, dtype=int),
                np.arange(self.x.size, dtype=int),
            )
        )

    def test_point_source_field_matches_homogeneous_analytic_solution(self) -> None:
        sources = np.array([[0.0], [0.0]])
        receivers = np.array([[400.0], [0.0]])
        source_fields, receiver_fields, mapping = build_source_receiver_fields(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            traveltime_table=analytic_table_backend,
        )

        expected = np.sqrt(
            (self.x[:, None] - 0.0) ** 2 + (self.z[None, :] - 0.0) ** 2
        ).T / 2000.0
        np.testing.assert_allclose(source_fields[0], expected)
        np.testing.assert_allclose(
            receiver_fields[0],
            np.sqrt(
                (self.x[:, None] - 400.0) ** 2
                + (self.z[None, :] - 0.0) ** 2
            ).T
            / 2000.0,
        )
        np.testing.assert_array_equal(mapping, [[0]])

    def test_reflection_traveltime_matches_horizontal_interface_solution(self) -> None:
        sources = np.array([[0.0], [0.0]])
        receivers = np.array([[400.0], [0.0]])
        result = compute_reflection_traveltimes(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            [self.interface],
            traveltime_table=analytic_table_backend,
        )

        expected = np.sqrt(400.0**2 + (2.0 * 200.0) ** 2) / 2000.0
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertTrue(result.valid[0, 0, 0])
        self.assertAlmostEqual(result.traveltime[0, 0, 0], expected, places=12)
        self.assertEqual(result.reflection_point_index[0, 0, 0], 20)

    def test_reflection_traveltime_is_reciprocal(self) -> None:
        forward = compute_reflection_traveltimes(
            self.velocity,
            self.x,
            self.z,
            np.array([[0.0], [0.0]]),
            np.array([[400.0], [0.0]]),
            [self.interface],
            traveltime_table=analytic_table_backend,
        )
        reverse = compute_reflection_traveltimes(
            self.velocity,
            self.x,
            self.z,
            np.array([[400.0], [0.0]]),
            np.array([[0.0], [0.0]]),
            [self.interface],
            traveltime_table=analytic_table_backend,
        )
        np.testing.assert_allclose(forward.traveltime, reverse.traveltime)
        np.testing.assert_array_equal(
            forward.reflection_point_index, reverse.reflection_point_index
        )

    def test_per_shot_receiver_geometry_uses_unique_coordinate_mapping(self) -> None:
        sources = np.array([[0.0, 100.0], [0.0, 0.0]])
        receivers = np.array(
            [
                [[0.0, 100.0], [0.0, 0.0]],
                [[100.0, 200.0], [0.0, 0.0]],
            ]
        )
        source_fields, receiver_fields, mapping = build_source_receiver_fields(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            traveltime_table=analytic_table_backend,
        )
        self.assertEqual(source_fields.shape[0], 2)
        self.assertEqual(receiver_fields.shape[0], 3)
        np.testing.assert_array_equal(mapping, [[0, 1], [1, 2]])

    def test_parallel_field_batches_match_serial_field_order(self) -> None:
        sources = np.array(
            [[0.0, 100.0, 200.0, 300.0], [0.0, 0.0, 0.0, 0.0]]
        )
        receivers = np.array(
            [[0.0, 100.0, 200.0, 300.0, 400.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        serial = build_source_receiver_fields(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            traveltime_table=analytic_table_backend,
            workers=1,
        )
        parallel = build_source_receiver_fields(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            traveltime_table=analytic_table_backend,
            workers=3,
        )
        np.testing.assert_allclose(parallel[0], serial[0])
        np.testing.assert_allclose(parallel[1], serial[1])
        np.testing.assert_array_equal(parallel[2], serial[2])

    def test_parallel_reflection_traveltime_matches_serial_result(self) -> None:
        sources = np.array(
            [[0.0, 100.0, 200.0, 300.0], [0.0, 0.0, 0.0, 0.0]]
        )
        receivers = np.array(
            [[0.0, 100.0, 200.0, 300.0, 400.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        serial = compute_reflection_traveltimes(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            [self.interface],
            traveltime_table=analytic_table_backend,
            workers=1,
        )
        parallel = compute_reflection_traveltimes(
            self.velocity,
            self.x,
            self.z,
            sources,
            receivers,
            [self.interface],
            traveltime_table=analytic_table_backend,
            workers=3,
        )
        np.testing.assert_allclose(parallel.traveltime, serial.traveltime)
        np.testing.assert_array_equal(
            parallel.reflection_point_index,
            serial.reflection_point_index,
        )
        np.testing.assert_array_equal(parallel.valid, serial.valid)


try:
    import pylops  # noqa: F401

    _PYLOPS_AVAILABLE = True
except ImportError:
    _PYLOPS_AVAILABLE = False


@unittest.skipUnless(_PYLOPS_AVAILABLE, "PyLops is not installed in this runtime")
class PyLopsIntegrationTests(unittest.TestCase):
    def test_pylops_uniform_source_field(self) -> None:
        x = np.arange(0.0, 401.0, 10.0)
        z = np.arange(0.0, 301.0, 10.0)
        velocity = 2000.0 * np.ones((x.size, z.size))
        source_fields, _, _ = build_source_receiver_fields(
            velocity,
            x,
            z,
            np.array([[0.0], [0.0]]),
            np.array([[400.0], [0.0]]),
        )
        expected = np.sqrt(x[:, None] ** 2 + z[None, :] ** 2).T / 2000.0
        np.testing.assert_allclose(source_fields[0], expected, atol=0.01)

    def test_pylops_horizontal_reflection_and_reciprocity(self) -> None:
        x = np.arange(0.0, 401.0, 10.0)
        z = np.arange(0.0, 301.0, 10.0)
        velocity = 2000.0 * np.ones((x.size, z.size))
        interface = np.column_stack(
            (np.full(x.size, 20, dtype=int), np.arange(x.size, dtype=int))
        )
        forward = compute_reflection_traveltimes(
            velocity,
            x,
            z,
            np.array([[0.0], [0.0]]),
            np.array([[400.0], [0.0]]),
            [interface],
        )
        reverse = compute_reflection_traveltimes(
            velocity,
            x,
            z,
            np.array([[400.0], [0.0]]),
            np.array([[0.0], [0.0]]),
            [interface],
        )
        expected = np.sqrt(400.0**2 + 400.0**2) / 2000.0
        self.assertTrue(forward.valid[0, 0, 0])
        self.assertAlmostEqual(forward.traveltime[0, 0, 0], expected, delta=0.01)
        np.testing.assert_allclose(forward.traveltime, reverse.traveltime, atol=0.01)


if __name__ == "__main__":
    unittest.main()
