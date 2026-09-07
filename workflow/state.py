"""Immutable outer-stage reference state shared by VFSA proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..correlation import CorrelationResult
from ..misfit import FixedMaskResult
from ..reflector import GridMappingResult, Reflector
from ..tracking import TrackingResult
from ..window import WindowResult

from .config import WRTIConfig
from .observed_center import QUADRATIC_ONLY, QUADRATIC_PLUS_ENVELOPE


class ReferenceStateError(ValueError):
    """Raised when an outer-stage reference state is inconsistent."""


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WRTIReferenceState:
    """Frozen result of one migration-to-window outer-stage update."""

    reflectors: tuple[Reflector, ...]
    reflector_grid_indices: tuple[np.ndarray, ...]
    reflection_traveltime: np.ndarray
    reflection_point_index: np.ndarray
    traveltime_valid: np.ndarray
    windows: WindowResult
    fixed_mask: np.ndarray | None
    config: WRTIConfig
    source_coordinates: np.ndarray
    receiver_coordinates: np.ndarray
    reference_qc: FixedMaskResult | None = None
    reference_tracking: tuple[tuple[TrackingResult, ...], ...] | None = None
    reference_shift_time: np.ndarray | None = None
    observed_windows: WindowResult | None = None
    reference_correlations: Mapping[tuple[int, int], CorrelationResult] = field(
        default_factory=dict
    )
    observed_center_source: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.reflectors:
            raise ReferenceStateError("At least one reflector is required.")
        if len(self.reflector_grid_indices) != len(self.reflectors):
            raise ReferenceStateError(
                "reflector_grid_indices must have one entry per reflector."
            )
        traveltime = np.asarray(self.reflection_traveltime, dtype=float)
        point_index = np.asarray(self.reflection_point_index, dtype=int)
        traveltime_valid = np.asarray(self.traveltime_valid, dtype=bool)
        if traveltime.ndim != 3:
            raise ReferenceStateError(
                "reflection_traveltime must have shape [nref, ns, nr]."
            )
        if point_index.shape != traveltime.shape or traveltime_valid.shape != traveltime.shape:
            raise ReferenceStateError(
                "reflection traveltime diagnostic arrays must match its shape."
            )
        if self.windows.shape != traveltime.shape:
            raise ReferenceStateError("windows and reflection_traveltime shapes must match.")
        if self.observed_windows is not None and self.observed_windows.shape != traveltime.shape:
            raise ReferenceStateError(
                "observed_windows and reflection_traveltime shapes must match."
            )
        source = np.asarray(self.source_coordinates, dtype=float)
        receiver = np.asarray(self.receiver_coordinates, dtype=float)
        if source.shape != (traveltime.shape[1], 2):
            raise ReferenceStateError("source_coordinates must have shape [ns, 2].")
        if receiver.shape != (traveltime.shape[1], traveltime.shape[2], 2):
            raise ReferenceStateError("receiver_coordinates must have shape [ns, nr, 2].")
        if not np.isfinite(source).all() or not np.isfinite(receiver).all():
            raise ReferenceStateError("stored geometry must be finite.")

        stored_indices = []
        for indices in self.reflector_grid_indices:
            value = np.asarray(indices, dtype=int)
            if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
                raise ReferenceStateError(
                    "each reflector_grid_indices entry must have shape [n_points, 2]."
                )
            stored_indices.append(_readonly(value, int))

        if self.fixed_mask is not None:
            fixed = np.asarray(self.fixed_mask, dtype=bool)
            if fixed.shape != traveltime.shape:
                raise ReferenceStateError("fixed_mask must match [nref, ns, nr].")
        else:
            fixed = None
        if self.reference_qc is not None:
            if fixed is None or self.reference_qc.fixed_mask.shape != traveltime.shape:
                raise ReferenceStateError("reference_qc must match fixed_mask shape.")
        if self.reference_shift_time is not None:
            reference_shift = np.asarray(self.reference_shift_time, dtype=float)
            if reference_shift.shape != traveltime.shape:
                raise ReferenceStateError(
                    "reference_shift_time must match reflection_traveltime shape."
                )
        else:
            reference_shift = None
        if self.observed_center_source is None:
            observed_source = np.zeros(traveltime.shape, dtype=int)
        else:
            observed_source = np.asarray(self.observed_center_source, dtype=int)
            if observed_source.shape != traveltime.shape:
                raise ReferenceStateError(
                    "observed_center_source must match reflection_traveltime shape."
                )
            if np.any((observed_source < 0) | (observed_source > QUADRATIC_ONLY)):
                raise ReferenceStateError("observed_center_source contains an unknown source code.")

        object.__setattr__(self, "reflectors", tuple(self.reflectors))
        object.__setattr__(self, "reflector_grid_indices", tuple(stored_indices))
        object.__setattr__(self, "reflection_traveltime", _readonly(traveltime, float))
        object.__setattr__(self, "reflection_point_index", _readonly(point_index, int))
        object.__setattr__(self, "traveltime_valid", _readonly(traveltime_valid, bool))
        object.__setattr__(self, "fixed_mask", None if fixed is None else _readonly(fixed, bool))
        object.__setattr__(
            self,
            "reference_shift_time",
            None if reference_shift is None else _readonly(reference_shift, float),
        )
        object.__setattr__(self, "observed_center_source", _readonly(observed_source, int))
        object.__setattr__(self, "source_coordinates", _readonly(source, float))
        object.__setattr__(self, "receiver_coordinates", _readonly(receiver, float))
        object.__setattr__(
            self,
            "reference_correlations",
            MappingProxyType(dict(self.reference_correlations)),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.reflection_traveltime.shape)

    @property
    def n_fixed(self) -> int:
        return 0 if self.fixed_mask is None else int(np.count_nonzero(self.fixed_mask))

    @property
    def window_invalid_count(self) -> int:
        return int(np.count_nonzero(~self.windows.valid))

    @property
    def observed_window_invalid_count(self) -> int:
        return 0 if self.observed_windows is None else int(
            np.count_nonzero(~self.observed_windows.valid)
        )

    @property
    def recovered_mask(self) -> np.ndarray:
        """Observed centers obtained by edge recovery rather than tracking."""

        return _readonly(
            (self.observed_center_source == QUADRATIC_PLUS_ENVELOPE)
            | (self.observed_center_source == QUADRATIC_ONLY),
            bool,
        )

    @property
    def observed_window_available_mask(self) -> np.ndarray:
        """Fixed support whose frozen observed-centered window is valid."""

        if self.fixed_mask is None or self.observed_windows is None:
            return _readonly(np.zeros(self.shape, dtype=bool), bool)
        return _readonly(self.fixed_mask & self.observed_windows.valid, bool)

    @property
    def overlap_warning_count(self) -> int:
        return int(np.count_nonzero(self.windows.overlap))
