"""PyLops-backed Ns+Nr reflection traveltimes for WRTI Step 2.

This module deliberately does not implement an Eikonal solver.  It wraps the
same PyLops traveltime-table call used by the existing Volve notebooks and
organizes its source/receiver fields for reflection traveltime evaluation.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Callable, Iterable, Sequence

import numpy as np

from ..reflector.interface import GridMappingResult


class ReflectionTraveltimeError(ValueError):
    """Raised when Step 2 inputs or PyLops outputs are inconsistent."""


TraveltimeTableBackend = Callable[..., Sequence[np.ndarray]]


@dataclass(frozen=True)
class ReflectionTraveltimeResult:
    """The frozen theoretical reflection traveltimes for one outer stage."""

    traveltime: np.ndarray
    reflection_point_index: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        traveltime = np.asarray(self.traveltime, dtype=float)
        point_index = np.asarray(self.reflection_point_index, dtype=int)
        valid = np.asarray(self.valid, dtype=bool)
        if traveltime.ndim != 3:
            raise ReflectionTraveltimeError(
                "traveltime must have shape [nref, ns, nr]."
            )
        if point_index.shape != traveltime.shape or valid.shape != traveltime.shape:
            raise ReflectionTraveltimeError(
                "traveltime, reflection_point_index, and valid must have identical shapes."
            )
        if np.any(valid & ~np.isfinite(traveltime)):
            raise ReflectionTraveltimeError(
                "Every valid reflection traveltime must be finite and expressed in seconds."
            )

        traveltime = np.array(traveltime, copy=True)
        point_index = np.array(point_index, copy=True)
        valid = np.array(valid, copy=True)
        for value in (traveltime, point_index, valid):
            value.setflags(write=False)
        object.__setattr__(self, "traveltime", traveltime)
        object.__setattr__(self, "reflection_point_index", point_index)
        object.__setattr__(self, "valid", valid)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.traveltime.shape)


@dataclass(frozen=True)
class _ReceiverGeometry:
    unique_coordinates: np.ndarray  # [n_unique_receiver, 2], columns x,z
    trace_to_unique: np.ndarray  # [ns, nr]


def _validate_model(
    velocity: np.ndarray,
    x_axis: Sequence[float] | np.ndarray,
    z_axis: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.asarray(velocity, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    z_axis = np.asarray(z_axis, dtype=float)
    if velocity.ndim != 2:
        raise ReflectionTraveltimeError("velocity must have shape [nx, nz].")
    nx, nz = velocity.shape
    if x_axis.shape != (nx,) or z_axis.shape != (nz,):
        raise ReflectionTraveltimeError(
            "x_axis/z_axis lengths must match velocity.shape == [nx, nz]."
        )
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ReflectionTraveltimeError("velocity must be finite and strictly positive.")
    if not np.isfinite(x_axis).all() or not np.isfinite(z_axis).all():
        raise ReflectionTraveltimeError("x_axis and z_axis must be finite.")
    if (nx > 1 and not np.all(np.diff(x_axis) > 0)) or (
        nz > 1 and not np.all(np.diff(z_axis) > 0)
    ):
        raise ReflectionTraveltimeError("x_axis and z_axis must be strictly increasing.")
    return velocity, x_axis, z_axis


def _normalise_points(
    coordinates: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return point coordinates as ``[n, 2]`` in ``(x, z)`` order."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2:
        raise ReflectionTraveltimeError(f"{name} must be a two-dimensional coordinate array.")
    if values.shape[0] == 2:
        points = values.T
    elif values.shape[1] == 2:
        points = values
    else:
        raise ReflectionTraveltimeError(
            f"{name} must have shape [2, n] or [n, 2]."
        )
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ReflectionTraveltimeError(f"{name} must contain finite non-empty points.")
    return np.array(points, dtype=float, copy=True)


def _receiver_geometry(
    coordinates: Sequence[float] | np.ndarray,
    n_shot: int,
) -> _ReceiverGeometry:
    """Normalize shared or per-shot receiver coordinates.

    Supported forms are shared ``[2, nr]``/``[nr, 2]`` and per-shot
    ``[ns, 2, nr]``/``[ns, nr, 2]``.  Per-shot tables are deduplicated by
    exact physical coordinate while preserving first-occurrence order.
    """

    values = np.asarray(coordinates, dtype=float)
    if values.ndim == 2:
        points = _normalise_points(values, "receiver_coordinates")
        mapping = np.tile(np.arange(points.shape[0], dtype=int), (n_shot, 1))
        return _ReceiverGeometry(points, mapping)

    if values.ndim != 3 or values.shape[0] != n_shot:
        raise ReflectionTraveltimeError(
            "Per-shot receiver_coordinates must have shape [ns, 2, nr] or [ns, nr, 2]."
        )
    if values.shape[1] == 2:
        per_shot = values.transpose(0, 2, 1)
    elif values.shape[2] == 2:
        per_shot = values
    else:
        raise ReflectionTraveltimeError(
            "Per-shot receiver_coordinates must have shape [ns, 2, nr] or [ns, nr, 2]."
        )
    if per_shot.shape[1] == 0 or not np.isfinite(per_shot).all():
        raise ReflectionTraveltimeError(
            "receiver_coordinates must contain finite non-empty points."
        )

    unique: list[tuple[float, float]] = []
    lookup: dict[tuple[float, float], int] = {}
    mapping = np.empty(per_shot.shape[:2], dtype=int)
    for ishot in range(n_shot):
        for irec in range(per_shot.shape[1]):
            point = tuple(float(value) for value in per_shot[ishot, irec])
            if point not in lookup:
                lookup[point] = len(unique)
                unique.append(point)
            mapping[ishot, irec] = lookup[point]
    return _ReceiverGeometry(np.asarray(unique, dtype=float), mapping)


def _pylops_traveltime_table(
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    source_coordinates: np.ndarray,
    receiver_coordinates: np.ndarray,
    velocity: np.ndarray,
    mode: str = "eikonal",
) -> Sequence[np.ndarray]:
    """Call the PyLops backend used by the Volve migration notebooks."""

    try:
        from pylops.waveeqprocessing.kirchhoff import Kirchhoff
    except ImportError as exc:
        raise ReflectionTraveltimeError(
            "PyLops is required for Step 2. Install pylops in the Linux runtime "
            "before running the Eikonal calculation."
        ) from exc

    return Kirchhoff._traveltime_table(
        z_axis,
        x_axis,
        source_coordinates.T,
        receiver_coordinates.T,
        velocity,
        mode=mode,
    )


def _normalise_field_batch(
    raw_field: np.ndarray,
    n_point: int,
    nx: int,
    nz: int,
    name: str,
) -> np.ndarray:
    """Normalize PyLops fields to ``[n_point, nz, nx]``."""

    values = np.asarray(raw_field, dtype=float)
    n_grid = nx * nz
    if values.ndim == 1 and n_point == 1 and values.size == n_grid:
        values = values.reshape(1, n_grid)
    if values.ndim == 2 and values.shape == (n_grid, n_point):
        values = values.T.reshape(n_point, nx, nz).transpose(0, 2, 1)
    elif values.ndim == 2 and values.shape == (n_point, n_grid):
        values = values.reshape(n_point, nx, nz).transpose(0, 2, 1)
    elif values.ndim == 3 and values.shape == (n_point, nz, nx):
        values = values
    elif values.ndim == 3 and values.shape == (n_point, nx, nz):
        values = values.transpose(0, 2, 1)
    else:
        raise ReflectionTraveltimeError(
            f"{name} has unsupported shape {values.shape}; expected a PyLops "
            f"flattened field with {n_grid} grid samples."
        )
    return np.asarray(values, dtype=float)


def _compute_eikonal_field_batch(
    velocity: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    source_indices: np.ndarray,
    receiver_indices: np.ndarray,
    backend: TraveltimeTableBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute and normalize one independent source/receiver field batch."""

    raw = backend(
        z_axis,
        x_axis,
        source_points,
        receiver_points,
        velocity,
        mode="eikonal",
    )
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        raise ReflectionTraveltimeError(
            "The PyLops traveltime table must return source and receiver fields "
            "as its first two results."
        )
    source_fields = _normalise_field_batch(
        raw[0],
        source_points.shape[0],
        velocity.shape[0],
        velocity.shape[1],
        "source field",
    )
    receiver_fields = _normalise_field_batch(
        raw[1],
        receiver_points.shape[0],
        velocity.shape[0],
        velocity.shape[1],
        "receiver field",
    )
    return (
        source_indices,
        source_fields,
        receiver_indices,
        receiver_fields,
        os.getpid(),
    )


def _compute_eikonal_process_task(task):
    """Process-pool entry point for one Eikonal point batch."""

    backend, arguments = task
    return _compute_eikonal_field_batch(
        *arguments,
        backend=backend,
    )


def build_source_receiver_fields(
    velocity: np.ndarray,
    x_axis: Sequence[float] | np.ndarray,
    z_axis: Sequence[float] | np.ndarray,
    source_coordinates: Sequence[float] | np.ndarray,
    receiver_coordinates: Sequence[float] | np.ndarray,
    *,
    traveltime_table: TraveltimeTableBackend | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all source and unique-receiver Eikonal fields.

    Returns ``source_fields[nsource, nz, nx]``,
    ``receiver_fields[nunique_receiver, nz, nx]``, and
    ``trace_to_unique_receiver[shot, receiver]``.  The backend computes the
    fields for all source and receiver locations; WRTI then reuses them for
    every reflector and shot-receiver trace.  When ``workers > 1``, the source
    and receiver locations are divided into the same number of independent
    batches and each batch invokes one Eikonal backend call concurrently.
    The returned fields are assembled in the original point order.
    """

    velocity, x_axis, z_axis = _validate_model(velocity, x_axis, z_axis)
    sources = _normalise_points(source_coordinates, "source_coordinates")
    geometry = _receiver_geometry(receiver_coordinates, sources.shape[0])
    backend = traveltime_table or _pylops_traveltime_table
    try:
        worker_count = int(workers)
    except (TypeError, ValueError) as exc:
        raise ReflectionTraveltimeError("workers must be a positive integer.") from exc
    if isinstance(workers, bool) or worker_count != workers or worker_count < 1:
        raise ReflectionTraveltimeError("workers must be a positive integer.")

    n_source = sources.shape[0]
    n_receiver = geometry.unique_coordinates.shape[0]
    if worker_count == 1:
        source_indices = np.arange(n_source, dtype=int)
        receiver_indices = np.arange(n_receiver, dtype=int)
        source_indices, source_fields, receiver_indices, receiver_fields, _ = (
            _compute_eikonal_field_batch(
                velocity,
                x_axis,
                z_axis,
                sources,
                geometry.unique_coordinates,
                source_indices,
                receiver_indices,
                backend,
            )
        )
    else:
        # Each task must receive a non-empty source and receiver block because
        # the PyLops table backend expects both point sets.  The number of
        # tasks is therefore bounded by the smaller point population.
        task_count = min(worker_count, n_source, n_receiver)
        source_chunks = np.array_split(np.arange(n_source, dtype=int), task_count)
        receiver_chunks = np.array_split(
            np.arange(n_receiver, dtype=int), task_count
        )
        tasks = [
            (
                velocity,
                x_axis,
                z_axis,
                sources[source_chunk],
                geometry.unique_coordinates[receiver_chunk],
                source_chunk,
                receiver_chunk,
            )
            for source_chunk, receiver_chunk in zip(
                source_chunks, receiver_chunks
            )
        ]
        # Use the same process path for production and injected backends so
        # tests exercise the concurrency mechanism that runs on the cluster.
        # The built-in backend is module-level and pickleable.  An injected
        # backend used with workers>1 must satisfy the same requirement.
        results = []
        try:
            # ``spawn`` avoids inheriting OpenMP, PyLops, or solver state from
            # the long-lived SWIT process.  It is slower to start than Linux
            # ``fork`` but is safer after repeated MPI forward-model launches.
            with ProcessPoolExecutor(
                max_workers=task_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_batches = {
                    executor.submit(
                        _compute_eikonal_process_task,
                        (backend, task),
                    ): batch_index
                    for batch_index, task in enumerate(tasks)
                }
                for future in as_completed(future_batches):
                    batch_index = future_batches[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        raise ReflectionTraveltimeError(
                            "Parallel Eikonal batch "
                            f"{batch_index + 1}/{task_count} failed: {exc}"
                        ) from exc
        except ReflectionTraveltimeError:
            raise
        except Exception as exc:
            raise ReflectionTraveltimeError(
                "Unable to start or complete the Eikonal process pool. "
                "For an injected backend, use a module-level pickleable "
                "callable, or set workers=1."
            ) from exc

        source_fields = np.empty(
            (n_source, velocity.shape[1], velocity.shape[0]), dtype=float
        )
        receiver_fields = np.empty(
            (
                n_receiver,
                velocity.shape[1],
                velocity.shape[0],
            ),
            dtype=float,
        )
        for (
            source_indices,
            source_batch,
            receiver_indices,
            receiver_batch,
            worker_pid,
        ) in results:
            source_fields[source_indices] = source_batch
            receiver_fields[receiver_indices] = receiver_batch
    return source_fields, receiver_fields, geometry.trace_to_unique


def build_source_fields(
    velocity: np.ndarray,
    x_axis: Sequence[float] | np.ndarray,
    z_axis: Sequence[float] | np.ndarray,
    source_coordinates: Sequence[float] | np.ndarray,
    receiver_coordinates: Sequence[float] | np.ndarray,
    *,
    traveltime_table: TraveltimeTableBackend | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Return source fields; the supplied receiver table is used by PyLops."""

    return build_source_receiver_fields(
        velocity,
        x_axis,
        z_axis,
        source_coordinates,
        receiver_coordinates,
        traveltime_table=traveltime_table,
        workers=workers,
    )[0]


def build_receiver_fields(
    velocity: np.ndarray,
    x_axis: Sequence[float] | np.ndarray,
    z_axis: Sequence[float] | np.ndarray,
    source_coordinates: Sequence[float] | np.ndarray,
    receiver_coordinates: Sequence[float] | np.ndarray,
    *,
    traveltime_table: TraveltimeTableBackend | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Return unique-receiver fields; source geometry is passed to PyLops."""

    return build_source_receiver_fields(
        velocity,
        x_axis,
        z_axis,
        source_coordinates,
        receiver_coordinates,
        traveltime_table=traveltime_table,
        workers=workers,
    )[1]


def _normalise_reflector_indices(
    reflector_indices: Iterable[GridMappingResult | Sequence[Sequence[int]]],
    nx: int,
    nz: int,
) -> list[np.ndarray]:
    normalized: list[np.ndarray] = []
    for reflector_id, item in enumerate(reflector_indices):
        if isinstance(item, GridMappingResult):
            if not np.all(item.valid):
                raise ReflectionTraveltimeError(
                    f"Reflector {reflector_id} contains invalid Step 1 grid points."
                )
            indices = item.indices
        else:
            indices = np.asarray(item)
        if indices.ndim != 2 or indices.shape[1] != 2 or indices.shape[0] == 0:
            raise ReflectionTraveltimeError(
                f"Reflector {reflector_id} indices must have shape [n_points, 2]."
            )
        if not np.isfinite(indices).all() or not np.all(indices == np.rint(indices)):
            raise ReflectionTraveltimeError(
                f"Reflector {reflector_id} indices must be finite integers."
            )
        indices = np.asarray(indices, dtype=int)
        iz, ix = indices[:, 0], indices[:, 1]
        if np.any(iz < 0) or np.any(iz >= nz) or np.any(ix < 0) or np.any(ix >= nx):
            raise ReflectionTraveltimeError(
                f"Reflector {reflector_id} contains indices outside [iz, ix] grid bounds."
            )
        normalized.append(indices)
    if not normalized:
        raise ReflectionTraveltimeError("At least one reflector is required.")
    return normalized


def compute_reflection_traveltimes(
    velocity: np.ndarray,
    x_axis: Sequence[float] | np.ndarray,
    z_axis: Sequence[float] | np.ndarray,
    source_coordinates: Sequence[float] | np.ndarray,
    receiver_coordinates: Sequence[float] | np.ndarray,
    reflector_indices: Iterable[GridMappingResult | Sequence[Sequence[int]]],
    *,
    traveltime_table: TraveltimeTableBackend | None = None,
    workers: int = 1,
) -> ReflectionTraveltimeResult:
    """Compute ``T_ref[nref, ns, nr] = min_k(Ts + Tr)`` in seconds.

    ``source_coordinates`` and ``receiver_coordinates`` are passed to PyLops
    as physical coordinates.  No sub-grid source algorithm or additional
    snapping is introduced here; the existing PyLops behavior is reused.
    Reflector indices come from Step 1 and use ``(iz, ix)`` order.
    """

    velocity, x_axis, z_axis = _validate_model(velocity, x_axis, z_axis)
    sources = _normalise_points(source_coordinates, "source_coordinates")
    mappings = _normalise_reflector_indices(
        reflector_indices, velocity.shape[0], velocity.shape[1]
    )
    source_fields, receiver_fields, trace_to_unique = build_source_receiver_fields(
        velocity,
        x_axis,
        z_axis,
        sources,
        receiver_coordinates,
        traveltime_table=traveltime_table,
        workers=workers,
    )
    n_shot, n_receiver = trace_to_unique.shape
    n_reflector = len(mappings)
    traveltime = np.full((n_reflector, n_shot, n_receiver), np.nan, dtype=float)
    point_index = np.full((n_reflector, n_shot, n_receiver), -1, dtype=int)
    valid = np.zeros((n_reflector, n_shot, n_receiver), dtype=bool)

    for reflector_id, indices in enumerate(mappings):
        iz, ix = indices[:, 0], indices[:, 1]
        source_samples = source_fields[:, iz, ix]  # [ns, n_points]
        receiver_samples = receiver_fields[trace_to_unique][:, :, iz, ix]
        # [ns, nr, n_points] = source-side plus receiver-side travel time.
        total = source_samples[:, None, :] + receiver_samples
        finite = np.isfinite(total)
        usable = finite.any(axis=-1)
        safe_total = np.where(finite, total, np.inf)
        best_point = np.argmin(safe_total, axis=-1)
        best_time = np.min(safe_total, axis=-1)
        best_time = np.where(usable, best_time, np.nan)
        best_point = np.where(usable, best_point, -1)

        traveltime[reflector_id] = best_time
        point_index[reflector_id] = best_point
        valid[reflector_id] = usable

    return ReflectionTraveltimeResult(
        traveltime=traveltime,
        reflection_point_index=point_index,
        valid=valid,
    )
