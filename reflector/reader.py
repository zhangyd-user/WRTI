"""Safe reflector loading, validation, and regular-grid mapping."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np

from .interface import GridMappingError, GridMappingResult, Reflector, ReflectorInputError


def _literal_layers(text: str, *, source_name: str) -> object:
    """Parse exactly one ``layers = <literal>`` assignment safely."""

    try:
        tree = ast.parse(text, filename=source_name, mode="exec")
    except SyntaxError as exc:
        raise ReflectorInputError(f"Invalid reflector file syntax: {source_name}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ReflectorInputError(
            "Reflector file must contain exactly one assignment: layers = (...)"
        )

    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise ReflectorInputError("Only a simple assignment to 'layers' is allowed.")
    if assignment.targets[0].id != "layers":
        raise ReflectorInputError("Reflector file assignment must be named 'layers'.")

    try:
        return ast.literal_eval(assignment.value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ReflectorInputError(
            "The layers value must be composed only of Python literals."
        ) from exc


def _validate_bounds(bounds: tuple[float, float] | None, name: str) -> None:
    if bounds is None:
        return
    if len(bounds) != 2:
        raise ReflectorInputError(f"{name} must be a (min, max) pair.")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ReflectorInputError(f"{name} must contain finite values with min <= max.")


def _validate_coordinate_ranges(
    reflectors: Iterable[Reflector],
    *,
    x_bounds: tuple[float, float] | None,
    z_bounds: tuple[float, float] | None,
) -> None:
    _validate_bounds(x_bounds, "x_bounds")
    _validate_bounds(z_bounds, "z_bounds")
    for reflector in reflectors:
        if x_bounds is not None and (
            reflector.x.min() < x_bounds[0] or reflector.x.max() > x_bounds[1]
        ):
            raise ReflectorInputError(
                f"Reflector {reflector.id} x coordinates exceed x_bounds={x_bounds}."
            )
        if z_bounds is not None and (
            reflector.z.min() < z_bounds[0] or reflector.z.max() > z_bounds[1]
        ):
            raise ReflectorInputError(
                f"Reflector {reflector.id} z coordinates exceed z_bounds={z_bounds}."
            )


def _as_reflector(layer: object, reflector_id: int) -> Reflector:
    if not isinstance(layer, (tuple, list)) or len(layer) == 0:
        raise ReflectorInputError(
            f"Reflector {reflector_id} must be a non-empty sequence of (x, z) points."
        )

    points: list[tuple[float, float]] = []
    for point_id, point in enumerate(layer):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ReflectorInputError(
                f"Reflector {reflector_id} point {point_id} is not an (x, z) pair."
            )
        try:
            x, z = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ReflectorInputError(
                f"Reflector {reflector_id} point {point_id} is not numeric."
            ) from exc
        points.append((x, z))

    coordinates = np.asarray(points, dtype=float)
    return Reflector(
        id=reflector_id,
        x=coordinates[:, 0],
        z=coordinates[:, 1],
    )


def read_reflectors(
    path: str | Path,
    *,
    x_bounds: tuple[float, float] | None = None,
    z_bounds: tuple[float, float] | None = None,
) -> tuple[Reflector, ...]:
    """Read and validate ordered reflectors from ``layers = (...)`` text."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReflectorInputError(f"Cannot read reflector file: {source}") from exc

    layers = _literal_layers(text, source_name=str(source))
    if not isinstance(layers, (tuple, list)) or len(layers) == 0:
        raise ReflectorInputError("'layers' must be a non-empty sequence.")

    reflectors = tuple(
        _as_reflector(layer, reflector_id)
        for reflector_id, layer in enumerate(layers)
    )
    _validate_coordinate_ranges(reflectors, x_bounds=x_bounds, z_bounds=z_bounds)
    return reflectors


def _validate_grid_parameters(
    x0: float,
    z0: float,
    dx: float,
    dz: float,
    nx: int,
    nz: int,
    tolerance: float,
) -> None:
    values = np.asarray([x0, z0, dx, dz, tolerance], dtype=float)
    if not np.isfinite(values).all():
        raise GridMappingError("x0, z0, dx, dz, and tolerance must be finite.")
    if dx <= 0 or dz <= 0:
        raise GridMappingError("dx and dz must be positive.")
    if tolerance < 0:
        raise GridMappingError("tolerance must be non-negative.")
    if isinstance(nx, bool) or isinstance(nz, bool) or int(nx) != nx or int(nz) != nz:
        raise GridMappingError("nx and nz must be positive integers.")
    if nx <= 0 or nz <= 0:
        raise GridMappingError("nx and nz must be positive integers.")


def map_reflector_to_grid(
    reflector: Reflector,
    x0: float,
    z0: float,
    dx: float,
    dz: float,
    nx: int,
    nz: int,
    tolerance: float = 1e-6,
    *,
    strict: bool = True,
) -> GridMappingResult:
    """Map one reflector to a regular grid without interpolation.

    The returned arrays have one element per input point.  ``valid`` requires
    both coordinate snap errors to be within ``tolerance`` and both rounded
    indices to lie inside the model grid.  With ``strict=True`` any invalid
    point raises ``GridMappingError``; with ``strict=False`` it emits a
    warning and returns ``valid=False`` for that point.
    """

    if not isinstance(reflector, Reflector):
        raise GridMappingError("reflector must be a Reflector instance.")
    _validate_grid_parameters(x0, z0, dx, dz, nx, nz, tolerance)

    ix_float = (reflector.x - x0) / dx
    iz_float = (reflector.z - z0) / dz
    ix = np.rint(ix_float).astype(int)
    iz = np.rint(iz_float).astype(int)

    snapped_x = x0 + ix * dx
    snapped_z = z0 + iz * dz
    error_x = reflector.x - snapped_x
    error_z = reflector.z - snapped_z
    valid = (
        (np.abs(error_x) <= tolerance)
        & (np.abs(error_z) <= tolerance)
        & (ix >= 0)
        & (ix < nx)
        & (iz >= 0)
        & (iz < nz)
    )

    if not np.all(valid):
        bad_ids = np.flatnonzero(~valid).tolist()
        message = (
            f"Reflector {reflector.id} has invalid grid mapping at point(s) "
            f"{bad_ids}; tolerance={tolerance}."
        )
        if strict:
            raise GridMappingError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return GridMappingResult(
        ix=ix,
        iz=iz,
        valid=valid,
        snap_error_x=error_x,
        snap_error_z=error_z,
    )


def map_reflectors_to_grid(
    reflectors: Iterable[Reflector],
    x0: float,
    z0: float,
    dx: float,
    dz: float,
    nx: int,
    nz: int,
    tolerance: float = 1e-6,
    *,
    strict: bool = True,
) -> tuple[GridMappingResult, ...]:
    """Apply :func:`map_reflector_to_grid` to each reflector independently."""

    return tuple(
        map_reflector_to_grid(
            reflector,
            x0,
            z0,
            dx,
            dz,
            nx,
            nz,
            tolerance,
            strict=strict,
        )
        for reflector in reflectors
    )


def reflector_grid_indices(
    mappings: Iterable[GridMappingResult],
    *,
    deduplicate: bool = False,
) -> tuple[np.ndarray, ...]:
    """Return ``[n_points, 2]`` arrays with columns ``(iz, ix)``."""

    results = []
    for mapping in mappings:
        if not isinstance(mapping, GridMappingResult):
            raise GridMappingError("mappings must contain GridMappingResult objects.")
        result = mapping.deduplicated() if deduplicate else mapping
        results.append(result.indices)
    return tuple(results)
