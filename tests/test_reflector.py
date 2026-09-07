from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import warnings

import numpy as np

from WRTI.reflector import (
    GridMappingError,
    ReflectorInputError,
    map_reflector_to_grid,
    read_reflectors,
    reflector_grid_indices,
)


class ReflectorStep1Tests(unittest.TestCase):
    def _write(self, text: str, *, name: str = "layers.txt") -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def _actual_layer_file_from_context(self) -> Path:
        context_path = Path(__file__).parents[1] / "WRTI_codex_context.json"
        context = json.loads(context_path.read_text(encoding="utf-8"))
        path = self._write(
            context["embedded_user_file"]["content"],
            name="layer_parameters_unet_001.txt",
        )
        return path

    def test_reads_actual_layer_parameters_file(self) -> None:
        path = self._actual_layer_file_from_context()
        reflectors = read_reflectors(path)

        self.assertEqual(len(reflectors), 4)
        self.assertEqual([reflector.n_points for reflector in reflectors], [651] * 4)
        for reflector in reflectors:
            self.assertTrue(np.isfinite(reflector.x).all())
            self.assertTrue(np.isfinite(reflector.z).all())
            self.assertTrue(np.all(np.diff(reflector.x) > 0))

    def test_coordinate_range_check_is_explicit(self) -> None:
        path = self._write("layers = (((0.0, 1.0), (1.0, 1.1)),)\n")
        with self.assertRaises(ReflectorInputError):
            read_reflectors(path, x_bounds=(0.0, 0.5))
        with self.assertRaises(ReflectorInputError):
            read_reflectors(path, z_bounds=(0.0, 1.05))

    def test_reader_rejects_executable_content(self) -> None:
        path = self._write("layers = __import__('os').system('echo unsafe')\n")
        with self.assertRaises(ReflectorInputError):
            read_reflectors(path)

    def test_mapping_returns_ix_iz_valid_and_iz_ix_indices(self) -> None:
        path = self._write(
            "layers = (((10.0, 20.0), (12.0, 22.0), (14.0, 24.0)),)\n"
        )
        reflector = read_reflectors(path)[0]

        mapping = map_reflector_to_grid(
            reflector,
            x0=10.0,
            z0=20.0,
            dx=2.0,
            dz=2.0,
            nx=3,
            nz=3,
            tolerance=0.0,
        )

        np.testing.assert_array_equal(mapping.ix, [0, 1, 2])
        np.testing.assert_array_equal(mapping.iz, [0, 1, 2])
        np.testing.assert_array_equal(mapping.valid, [True, True, True])
        np.testing.assert_array_equal(mapping.indices, [[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            reflector_grid_indices([mapping])[0], [[0, 0], [1, 1], [2, 2]]
        )

    def test_mapping_rejects_outside_grid_or_tolerance(self) -> None:
        path = self._write("layers = (((10.0, 20.4),),)\n")
        reflector = read_reflectors(path)[0]
        with self.assertRaises(GridMappingError):
            map_reflector_to_grid(
                reflector, 10.0, 20.0, 2.0, 2.0, 1, 1, tolerance=0.1
            )

    def test_mapping_can_return_invalid_points_with_warning(self) -> None:
        path = self._write("layers = (((8.0, 20.0), (10.0, 20.4)),)\n")
        reflector = read_reflectors(path)[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mapping = map_reflector_to_grid(
                reflector,
                10.0,
                20.0,
                2.0,
                2.0,
                2,
                2,
                tolerance=0.1,
                strict=False,
            )

        self.assertTrue(caught)
        np.testing.assert_array_equal(mapping.valid, [False, False])

    def test_duplicate_indices_can_be_removed_in_original_order(self) -> None:
        path = self._write(
            "layers = (((10.0, 20.0), (10.0, 20.0), (12.0, 22.0)),)\n"
        )
        # Duplicate x coordinates are rejected by the ordered reflector
        # contract, so exercise the result-level operation with a valid
        # reflector whose points round to the same grid node.
        path = self._write(
            "layers = (((10.0, 20.0), (10.1, 20.1), (12.0, 22.0)),)\n"
        )
        reflector = read_reflectors(path)[0]
        mapping = map_reflector_to_grid(
            reflector, 10.0, 20.0, 2.0, 2.0, 2, 2, tolerance=0.2
        )
        np.testing.assert_array_equal(
            reflector_grid_indices([mapping], deduplicate=True)[0], [[0, 0], [1, 1]]
        )


if __name__ == "__main__":
    unittest.main()
