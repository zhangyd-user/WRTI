"""Geometry normalization shared by the outer builder and evaluator."""

from __future__ import annotations

import numpy as np


class GeometryError(ValueError):
    """Raised when source/receiver geometry cannot be normalized."""


def normalize_source_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2:
        raise GeometryError("source_coordinates must have shape [ns, 2] or [2, ns].")
    if values.shape[1] == 2:
        points = values
    elif values.shape[0] == 2:
        points = values.T
    else:
        raise GeometryError("source_coordinates must have shape [ns, 2] or [2, ns].")
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise GeometryError("source_coordinates must contain finite non-empty points.")
    return np.array(points, dtype=float, copy=True)


def normalize_receiver_coordinates(
    coordinates: np.ndarray,
    n_shot: int,
) -> np.ndarray:
    """Return receiver coordinates as ``[ns, nr, 2]`` in ``(x,z)`` order."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim == 2:
        if values.shape[1] == 2:
            shared = values
        elif values.shape[0] == 2:
            shared = values.T
        else:
            raise GeometryError(
                "receiver_coordinates must have shape [nr, 2], [2, nr], "
                "[ns, nr, 2], or [ns, 2, nr]."
            )
        points = np.broadcast_to(shared[None, :, :], (n_shot, shared.shape[0], 2))
    elif values.ndim == 3 and values.shape[0] == n_shot:
        if values.shape[2] == 2:
            points = values
        elif values.shape[1] == 2:
            points = values.transpose(0, 2, 1)
        else:
            raise GeometryError(
                "per-shot receiver_coordinates must have shape [ns, nr, 2] "
                "or [ns, 2, nr]."
            )
    else:
        raise GeometryError(
            "receiver_coordinates must have shape [nr, 2], [2, nr], "
            "[ns, nr, 2], or [ns, 2, nr]."
        )
    if points.shape[1] == 0 or not np.isfinite(points).all():
        raise GeometryError("receiver_coordinates must contain finite non-empty points.")
    return np.array(points, dtype=float, copy=True)
