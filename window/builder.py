"""Frozen selected-time-window construction for WRTI Step 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from scipy.signal.windows import tukey


class WindowError(ValueError):
    """Raised when window inputs are inconsistent or invalid."""


class WindowOverlapWarning(UserWarning):
    """Warning emitted when adjacent reflector windows overlap."""


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class WindowResult:
    """Immutable window state reusable throughout one VFSA inner loop.

    No full ``[nref, ns, nr, nt]`` mask is stored.  ``mask_for_trace`` and
    ``weights_for_trace`` generate one trace's logical mask/weights on demand.
    """

    reference_traveltime: np.ndarray
    center_time: np.ndarray
    center_float: np.ndarray
    center_sample: np.ndarray
    left_sample: np.ndarray
    right_sample: np.ndarray
    valid: np.ndarray
    half_window_time: np.ndarray
    dt: float
    t0: float
    nt: int
    max_lag_samples: int
    window_type: str
    tukey_alpha: float
    overlap: np.ndarray
    overlap_delta_time: np.ndarray

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_traveltime, dtype=float)
        shape = reference.shape
        if reference.ndim != 3:
            raise WindowError("reference_traveltime must have shape [nref, ns, nr].")
        arrays = {
            "center_time": np.asarray(self.center_time, dtype=float),
            "center_float": np.asarray(self.center_float, dtype=float),
            "center_sample": np.asarray(self.center_sample, dtype=int),
            "left_sample": np.asarray(self.left_sample, dtype=int),
            "right_sample": np.asarray(self.right_sample, dtype=int),
            "valid": np.asarray(self.valid, dtype=bool),
        }
        for name, value in arrays.items():
            if value.shape != shape:
                raise WindowError(f"{name} must have shape {shape}.")

        half_window = np.asarray(self.half_window_time, dtype=float)
        if half_window.ndim != 1 or half_window.shape[0] != shape[0]:
            raise WindowError("half_window_time must have shape [nref].")
        if not np.isfinite(half_window).all() or np.any(half_window < 0):
            raise WindowError("half_window_time must be finite and non-negative.")

        overlap_shape = (max(shape[0] - 1, 0), shape[1], shape[2])
        overlap = np.asarray(self.overlap, dtype=bool)
        overlap_delta = np.asarray(self.overlap_delta_time, dtype=float)
        if overlap.shape != overlap_shape or overlap_delta.shape != overlap_shape:
            raise WindowError(
                "overlap and overlap_delta_time must have shape "
                f"{overlap_shape}."
            )

        if not np.isfinite(self.dt) or self.dt <= 0:
            raise WindowError("dt must be finite and positive.")
        if not np.isfinite(self.t0):
            raise WindowError("t0 must be finite.")
        if isinstance(self.nt, bool) or int(self.nt) != self.nt or self.nt <= 0:
            raise WindowError("nt must be a positive integer.")
        if isinstance(self.max_lag_samples, bool) or int(self.max_lag_samples) != self.max_lag_samples:
            raise WindowError("max_lag_samples must be a non-negative integer.")
        if self.max_lag_samples < 0:
            raise WindowError("max_lag_samples must be a non-negative integer.")
        if self.window_type not in {"rectangular", "tukey"}:
            raise WindowError("window_type must be 'rectangular' or 'tukey'.")
        if not np.isfinite(self.tukey_alpha) or not 0 <= self.tukey_alpha <= 1:
            raise WindowError("tukey_alpha must be in [0, 1].")

        object.__setattr__(self, "reference_traveltime", _readonly(reference, float))
        for name, value in arrays.items():
            object.__setattr__(self, name, _readonly(value, value.dtype))
        object.__setattr__(self, "half_window_time", _readonly(half_window, float))
        object.__setattr__(self, "overlap", _readonly(overlap, bool))
        object.__setattr__(self, "overlap_delta_time", _readonly(overlap_delta, float))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "t0", float(self.t0))
        object.__setattr__(self, "nt", int(self.nt))
        object.__setattr__(self, "max_lag_samples", int(self.max_lag_samples))
        object.__setattr__(self, "window_type", str(self.window_type))
        object.__setattr__(self, "tukey_alpha", float(self.tukey_alpha))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.reference_traveltime.shape)

    @property
    def has_overlap_warning(self) -> bool:
        return bool(np.any(self.overlap))

    def _trace_bounds(self, reflector: int, shot: int, receiver: int) -> tuple[int, int, bool]:
        nref, ns, nr = self.shape
        if not (0 <= reflector < nref and 0 <= shot < ns and 0 <= receiver < nr):
            raise IndexError("reflector, shot, or receiver index is out of range.")
        return (
            int(self.left_sample[reflector, shot, receiver]),
            int(self.right_sample[reflector, shot, receiver]),
            bool(self.valid[reflector, shot, receiver]),
        )

    def mask_for_trace(self, reflector: int, shot: int, receiver: int) -> np.ndarray:
        """Generate one trace's boolean window mask on demand."""

        left, right, valid = self._trace_bounds(reflector, shot, receiver)
        mask = np.zeros(self.nt, dtype=bool)
        if valid:
            mask[left : right + 1] = True
        return mask

    def weights_for_trace(self, reflector: int, shot: int, receiver: int) -> np.ndarray:
        """Generate one trace's rectangular or Tukey weights on demand."""

        left, right, valid = self._trace_bounds(reflector, shot, receiver)
        weights = np.zeros(self.nt, dtype=float)
        if not valid:
            return weights

        length = right - left + 1
        if self.window_type == "rectangular":
            local = np.ones(length, dtype=float)
        else:
            local = np.ones(1, dtype=float) if length == 1 else tukey(
                length, alpha=self.tukey_alpha, sym=True
            )
        weights[left : right + 1] = local
        return weights


def _normalise_half_window(
    half_window_time: float | Sequence[float] | np.ndarray,
    n_reflector: int,
) -> np.ndarray:
    values = np.asarray(half_window_time, dtype=float)
    if values.ndim == 0:
        values = np.full(n_reflector, float(values), dtype=float)
    elif values.ndim == 1 and values.shape == (n_reflector,):
        values = np.array(values, dtype=float, copy=True)
    else:
        raise WindowError(
            "half_window_time must be a scalar or a one-dimensional array "
            "with length nref."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise WindowError("half_window_time must be finite and non-negative.")
    return values


def _normalise_lag_samples(
    dt: float,
    max_lag_time: float | None,
    max_lag_samples: int | None,
) -> int:
    if max_lag_samples is not None:
        if max_lag_time is not None:
            raise WindowError("Specify max_lag_time or max_lag_samples, not both.")
        if (
            isinstance(max_lag_samples, bool)
            or int(max_lag_samples) != max_lag_samples
            or max_lag_samples < 0
        ):
            raise WindowError("max_lag_samples must be a non-negative integer.")
        return int(max_lag_samples)
    if max_lag_time is None:
        return 0
    if not np.isfinite(max_lag_time) or max_lag_time < 0:
        raise WindowError("max_lag_time must be finite and non-negative.")
    return int(np.ceil(max_lag_time / dt))


def _validate_inputs(
    reflection_traveltime: np.ndarray,
    dt: float,
    t0: float,
    nt: int,
    window_type: str,
    tukey_alpha: float,
) -> np.ndarray:
    values = np.asarray(reflection_traveltime, dtype=float)
    if values.ndim != 3:
        raise WindowError("reflection_traveltime must have shape [nref, ns, nr].")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise WindowError("reflection_traveltime dimensions must be non-empty.")
    if not np.isfinite(dt) or dt <= 0:
        raise WindowError("dt must be finite and positive.")
    if not np.isfinite(t0):
        raise WindowError("t0 must be finite.")
    if isinstance(nt, bool) or int(nt) != nt or nt <= 0:
        raise WindowError("nt must be a positive integer.")
    if window_type not in {"rectangular", "tukey"}:
        raise WindowError("window_type must be 'rectangular' or 'tukey'.")
    if not np.isfinite(tukey_alpha) or not 0 <= tukey_alpha <= 1:
        raise WindowError("tukey_alpha must be in [0, 1].")
    return values


def build_window_result(
    reflection_traveltime: np.ndarray,
    dt: float,
    t0: float,
    nt: int,
    half_window_time: float | Sequence[float] | np.ndarray,
    *,
    window_type: str = "rectangular",
    tukey_alpha: float = 0.5,
    max_lag_time: float | None = None,
    max_lag_samples: int | None = None,
) -> WindowResult:
    """Build immutable fixed windows from Step 2 reflection traveltimes.

    Discrete boundaries are selected from the physical window
    ``center_time ± half_window_time`` using ``ceil`` for the left boundary
    and ``floor`` for the right boundary.  Thus every retained sample lies
    inside the requested physical window.  The strict validity check includes
    the complete ``±max_lag`` search range without padding or partial overlap.
    """

    values = _validate_inputs(
        reflection_traveltime, dt, t0, nt, window_type, tukey_alpha
    )
    n_reflector, n_shot, n_receiver = values.shape
    half_window = _normalise_half_window(half_window_time, n_reflector)
    lag_samples = _normalise_lag_samples(dt, max_lag_time, max_lag_samples)

    finite = np.isfinite(values)
    center_float = np.full(values.shape, np.nan, dtype=float)
    center_float[finite] = (values[finite] - t0) / dt
    center_time = np.full(values.shape, np.nan, dtype=float)
    center_time[finite] = t0 + center_float[finite] * dt
    center_sample = np.full(values.shape, -1, dtype=int)
    center_sample[finite] = np.rint(center_float[finite]).astype(int)

    half_samples = half_window[:, None, None] / dt
    left_float = center_float - half_samples
    right_float = center_float + half_samples
    left_sample = np.full(values.shape, -1, dtype=int)
    right_sample = np.full(values.shape, -1, dtype=int)
    left_sample[finite] = np.ceil(left_float[finite]).astype(int)
    right_sample[finite] = np.floor(right_float[finite]).astype(int)

    valid = (
        finite
        & (left_sample <= right_sample)
        & ((left_sample - lag_samples) >= 0)
        & ((right_sample + lag_samples) < nt)
    )

    overlap_shape = (max(n_reflector - 1, 0), n_shot, n_receiver)
    overlap = np.zeros(overlap_shape, dtype=bool)
    overlap_delta = np.full(overlap_shape, np.nan, dtype=float)
    if n_reflector > 1:
        pair_finite = finite[1:] & finite[:-1]
        overlap_delta = np.abs(values[1:] - values[:-1])
        overlap[...]=pair_finite & (
            (half_window[1:, None, None] + half_window[:-1, None, None])
            >= overlap_delta
        )
        if np.any(overlap):
            count = int(np.count_nonzero(overlap))
            warnings.warn(
                f"{count} adjacent reflector trace window(s) overlap; "
                "window centers and half-widths were not modified.",
                WindowOverlapWarning,
                stacklevel=2,
            )

    return WindowResult(
        reference_traveltime=values,
        center_time=center_time,
        center_float=center_float,
        center_sample=center_sample,
        left_sample=left_sample,
        right_sample=right_sample,
        valid=valid,
        half_window_time=half_window,
        dt=dt,
        t0=t0,
        nt=nt,
        max_lag_samples=lag_samples,
        window_type=window_type,
        tukey_alpha=tukey_alpha,
        overlap=overlap,
        overlap_delta_time=overlap_delta,
    )


def build_windows(*args, **kwargs) -> WindowResult:
    """Alias for :func:`build_window_result`."""

    return build_window_result(*args, **kwargs)
