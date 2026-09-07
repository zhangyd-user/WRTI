"""Data contracts for WRTI Step 1 reflector input and grid mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ReflectorInputError(ValueError):
    """Raised when a reflector file or pick violates the input contract."""


class GridMappingError(ValueError):
    """Raised when a reflector cannot be safely mapped to a model grid."""


@dataclass(frozen=True)
class Reflector:
    """One ordered reflector in physical ``(x, z)`` coordinates."""

    id: int
    x: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        z = np.asarray(self.z, dtype=float)
        if x.ndim != 1 or z.ndim != 1 or x.size == 0 or x.shape != z.shape:
            raise ReflectorInputError(
                "Reflector x and z must be non-empty one-dimensional arrays "
                "with identical shapes."
            )
        if not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ReflectorInputError("Reflector coordinates must be finite.")
        if x.size > 1 and not np.all(np.diff(x) > 0):
            raise ReflectorInputError(
                "Reflector x coordinates must be strictly increasing; "
                "input order is not changed."
            )

        x = np.array(x, dtype=float, copy=True)
        z = np.array(z, dtype=float, copy=True)
        x.setflags(write=False)
        z.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "z", z)

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    @property
    def coordinates(self) -> np.ndarray:
        """Return ordered ``[n_points, 2]`` coordinates as ``(x, z)``."""

        return np.column_stack((self.x, self.z))


@dataclass(frozen=True)
class GridMappingResult:
    """Result of snapping one reflector to a regular ``(x, z)`` grid."""

    ix: np.ndarray
    iz: np.ndarray
    valid: np.ndarray
    snap_error_x: np.ndarray
    snap_error_z: np.ndarray

    def __post_init__(self) -> None:
        ix = np.asarray(self.ix, dtype=int)
        iz = np.asarray(self.iz, dtype=int)
        valid = np.asarray(self.valid, dtype=bool)
        error_x = np.asarray(self.snap_error_x, dtype=float)
        error_z = np.asarray(self.snap_error_z, dtype=float)
        shapes = {value.shape for value in (ix, iz, valid, error_x, error_z)}
        if len(shapes) != 1 or ix.ndim != 1:
            raise GridMappingError(
                "ix, iz, valid, and snap errors must be one-dimensional arrays "
                "with identical shapes."
            )
        if not np.isfinite(error_x).all() or not np.isfinite(error_z).all():
            raise GridMappingError("Grid snap errors must be finite.")

        for value in (ix, iz, valid, error_x, error_z):
            value.setflags(write=False)
        object.__setattr__(self, "ix", ix)
        object.__setattr__(self, "iz", iz)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "snap_error_x", error_x)
        object.__setattr__(self, "snap_error_z", error_z)

    @property
    def n_points(self) -> int:
        return int(self.ix.size)

    @property
    def indices(self) -> np.ndarray:
        """Return grid indices with columns ``(iz, ix)``."""

        return np.column_stack((self.iz, self.ix))

    def __iter__(self):
        """Allow ``ix, iz, valid = result`` while retaining named fields."""

        yield self.ix
        yield self.iz
        yield self.valid

    def deduplicated(self) -> "GridMappingResult":
        """Keep the first occurrence of each grid index in layer order."""

        if self.n_points < 2:
            return self
        pairs = self.indices
        keep = np.zeros(self.n_points, dtype=bool)
        seen: set[tuple[int, int]] = set()
        for point_id, pair in enumerate(pairs):
            key = (int(pair[0]), int(pair[1]))
            if key not in seen:
                seen.add(key)
                keep[point_id] = True
        return GridMappingResult(
            ix=self.ix[keep],
            iz=self.iz[keep],
            valid=self.valid[keep],
            snap_error_x=self.snap_error_x[keep],
            snap_error_z=self.snap_error_z[keep],
        )
