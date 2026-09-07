"""Diagnostic-only synthetic trace warping for selected WRTI paths."""

from __future__ import annotations

import numpy as np


class WarpDiagnosticError(ValueError):
    """Raised when selected warped-synthetic diagnostic inputs are invalid."""


def warp_synthetic_by_shift(
    synthetic: np.ndarray,
    shift_time: np.ndarray,
    reflector: int,
    shot: int,
    dt: float,
) -> np.ndarray:
    """Return one ``[receiver, time]`` synthetic gather shifted by residual.

    Positive ``shift_time`` delays the synthetic event, matching
    ``Delta_tau = T_obs - T_syn``.  Linear interpolation and zero-valued
    diagnostic boundaries are used only for visualization; this function is
    never called by the objective.

    ``shift_time`` may be the complete ``[nref, ns, nr]`` array or the
    selected ``[nr]`` receiver vector.
    """

    data = np.asarray(synthetic, dtype=float)
    if data.ndim != 3:
        raise WarpDiagnosticError("synthetic must have shape [ns, nr, nt].")
    if not np.isfinite(data).all():
        raise WarpDiagnosticError("synthetic must contain only finite values.")
    if not np.isfinite(dt) or dt <= 0:
        raise WarpDiagnosticError("dt must be finite and positive.")
    if (
        isinstance(reflector, bool)
        or isinstance(shot, bool)
        or int(reflector) != reflector
        or int(shot) != shot
    ):
        raise WarpDiagnosticError("reflector and shot must be integer indices.")
    reflector = int(reflector)
    shot = int(shot)
    if not (0 <= reflector and 0 <= shot < data.shape[0]):
        raise WarpDiagnosticError("shot index is out of range.")

    shifts = np.asarray(shift_time, dtype=float)
    if shifts.ndim == 3:
        if not (0 <= reflector < shifts.shape[0] and 0 <= shot < shifts.shape[1]):
            raise WarpDiagnosticError("reflector or shot index is out of range.")
        shifts = shifts[reflector, shot]
    if shifts.ndim != 1 or shifts.shape[0] != data.shape[1]:
        raise WarpDiagnosticError("shift_time must have shape [nr] or [nref, ns, nr].")
    if not np.isfinite(shifts).all():
        raise WarpDiagnosticError(
            "selected shift_time contains invalid values; choose a successful path."
        )

    nt = data.shape[2]
    sample_axis = np.arange(nt, dtype=float)
    warped = np.empty((data.shape[1], nt), dtype=float)
    for receiver in range(data.shape[1]):
        shift_samples = float(shifts[receiver]) / float(dt)
        # output[q] = input[q - shift], so positive residual delays input.
        warped[receiver] = np.interp(
            sample_axis - shift_samples,
            sample_axis,
            data[shot, receiver],
            left=0.0,
            right=0.0,
        )
    return warped
