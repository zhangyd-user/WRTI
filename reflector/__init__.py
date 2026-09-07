"""Reflector input and grid mapping utilities for WRTI Step 1."""

from .interface import (
    GridMappingError,
    GridMappingResult,
    Reflector,
    ReflectorInputError,
)
from .reader import (
    map_reflector_to_grid,
    map_reflectors_to_grid,
    read_reflectors,
    reflector_grid_indices,
)

__all__ = [
    "GridMappingError",
    "GridMappingResult",
    "Reflector",
    "ReflectorInputError",
    "map_reflector_to_grid",
    "map_reflectors_to_grid",
    "read_reflectors",
    "reflector_grid_indices",
]
