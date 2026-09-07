"""Reference fixed-mask construction for WRTI Step 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MisfitError(ValueError):
    """Raised when Step 6 mask or objective inputs are invalid."""


def _readonly(value: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FixedMaskResult:
    """Frozen reference QC arrays and the mask used by every VFSA proposal."""

    fixed_mask: np.ndarray
    trace_valid: np.ndarray
    window_valid: np.ndarray
    tracking_success: np.ndarray
    tracked_correlation: np.ndarray
    window_energy_obs: np.ndarray
    window_energy_syn: np.ndarray
    boundary_flag: np.ndarray

    def __post_init__(self) -> None:
        fixed_mask = np.asarray(self.fixed_mask, dtype=bool)
        if fixed_mask.ndim != 3:
            raise MisfitError("fixed_mask must have shape [nref, ns, nr].")
        shape = fixed_mask.shape
        converted = {
            "trace_valid": np.asarray(self.trace_valid, dtype=bool),
            "window_valid": np.asarray(self.window_valid, dtype=bool),
            "tracking_success": np.asarray(self.tracking_success, dtype=bool),
            "tracked_correlation": np.asarray(
                self.tracked_correlation, dtype=float
            ),
            "window_energy_obs": np.asarray(self.window_energy_obs, dtype=float),
            "window_energy_syn": np.asarray(self.window_energy_syn, dtype=float),
            "boundary_flag": np.asarray(self.boundary_flag, dtype=bool),
        }
        for name, value in converted.items():
            if value.shape != shape:
                raise MisfitError(f"{name} must have shape {shape}.")

        object.__setattr__(self, "fixed_mask", _readonly(fixed_mask, bool))
        for name, value in converted.items():
            object.__setattr__(self, name, _readonly(value, value.dtype))

    @property
    def n_fixed(self) -> int:
        return int(np.count_nonzero(self.fixed_mask))


def _validate_same_shape(*arrays: np.ndarray) -> tuple[int, int, int]:
    first = np.asarray(arrays[0])
    if first.ndim != 3:
        raise MisfitError("Step 6 quality arrays must have shape [nref, ns, nr].")
    for value in arrays[1:]:
        if np.asarray(value).shape != first.shape:
            raise MisfitError(
                "trace/window/tracking QC arrays must have identical shapes."
            )
    return tuple(int(item) for item in first.shape)


def build_fixed_mask(
    trace_valid: np.ndarray,
    window_valid: np.ndarray,
    tracking_success: np.ndarray,
    tracked_correlation: np.ndarray,
    window_energy_obs: np.ndarray,
    window_energy_syn: np.ndarray,
    boundary_flag: np.ndarray,
    *,
    min_correlation: float | None = None,
    exclude_search_boundary: bool = False,
    use_energy_threshold: bool = False,
    min_window_energy: float | None = None,
) -> FixedMaskResult:
    """Build the reference mask once, before the VFSA inner loop.

    Only explicitly configured optional filters affect the mask.  By default
    energies and boundary flags are saved but do not remove data items.
    """

    arrays = [
        trace_valid,
        window_valid,
        tracking_success,
        tracked_correlation,
        window_energy_obs,
        window_energy_syn,
        boundary_flag,
    ]
    shape = _validate_same_shape(*arrays)
    correlation = np.asarray(tracked_correlation, dtype=float)
    energy_obs = np.asarray(window_energy_obs, dtype=float)
    energy_syn = np.asarray(window_energy_syn, dtype=float)
    if min_correlation is not None:
        if not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0:
            raise MisfitError("min_correlation must be in [-1, 1].")
    if not isinstance(exclude_search_boundary, (bool, np.bool_)):
        raise MisfitError("exclude_search_boundary must be boolean.")
    if use_energy_threshold:
        if min_window_energy is None:
            raise MisfitError(
                "min_window_energy is required when use_energy_threshold=True."
            )
        if not np.isfinite(min_window_energy) or min_window_energy < 0:
            raise MisfitError(
                "min_window_energy must be finite and non-negative."
            )
    elif min_window_energy is not None and (
        not np.isfinite(min_window_energy) or min_window_energy < 0
    ):
        raise MisfitError("min_window_energy must be finite and non-negative.")

    mask = (
        np.asarray(trace_valid, dtype=bool)
        & np.asarray(window_valid, dtype=bool)
        & np.asarray(tracking_success, dtype=bool)
    )
    # A zero-energy observed window carries no usable input waveform.  This
    # is also how an offset-muted all-zero trace appears after fixed-window
    # ZNCC: ``window_energy_obs`` is zero and the correlation is undefined.
    # Exclude it from the frozen reference set regardless of the optional
    # candidate energy-threshold policy below.
    mask &= np.isfinite(energy_obs) & (energy_obs > 0.0)
    # ``min_correlation`` is retained as a validated compatibility argument,
    # but it is QC-only.  A finite path selected by Step 5 must remain in the
    # frozen mask and in the Step 6 objective.
    if exclude_search_boundary:
        mask &= ~np.asarray(boundary_flag, dtype=bool)
    if use_energy_threshold:
        mask &= (
            np.isfinite(energy_obs)
            & np.isfinite(energy_syn)
            & (energy_obs >= min_window_energy)
            & (energy_syn >= min_window_energy)
        )

    return FixedMaskResult(
        fixed_mask=mask,
        trace_valid=np.asarray(trace_valid, dtype=bool),
        window_valid=np.asarray(window_valid, dtype=bool),
        tracking_success=np.asarray(tracking_success, dtype=bool),
        tracked_correlation=correlation,
        window_energy_obs=energy_obs,
        window_energy_syn=energy_syn,
        boundary_flag=np.asarray(boundary_flag, dtype=bool),
    )
