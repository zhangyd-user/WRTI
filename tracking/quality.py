"""Quality masks derived from a completed WRTI tracking result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# These thresholds are intentionally separate from the DP search threshold.
# The DP itself may use weaker states (for example |ZNCC| >= 0.3) to maintain
# continuity.  A tracked point is considered a trustworthy measurement only
# when its raw tracked waveform ZNCC reaches this stronger threshold.
DEFAULT_TRUST_CORRELATION = 0.5
DEFAULT_COLLAPSE_DIRECT_LOW_QUALITY_FRACTION = 0.45
DEFAULT_COLLAPSE_SECONDARY_LOW_QUALITY_FRACTION = 0.25
DEFAULT_COLLAPSE_ZIGZAG_FRACTION = 0.5


@dataclass(frozen=True)
class DPFailureResult:
    """Post-DP quality classification for one reflector/shot pair.

    ``hard_failure_mask`` preserves the legacy structural DP-failure meaning:
    no successful path index or no finite tracked shift.

    ``failure_mask`` is the mask used by fallback and by the existing
    ``path_failure`` diagnostics.  For a normal event it is the union of the
    legacy hard failures and receiver-level low-quality points.  For an event
    collapse it marks the whole requested receiver scope as failed.

    ``global_lag`` is the median tracked lag from trusted points of this
    reflector/shot.  It is used only when this reflector is not collapsed.
    """

    hard_failure_mask: np.ndarray
    failure_mask: np.ndarray
    event_collapse: bool
    trusted_fraction: float
    low_quality_fraction: float
    oscillation_index: float
    global_lag: float | None


def _hard_failure_mask(tracking) -> np.ndarray:
    """Return the original structural DP-failure mask."""

    success = np.asarray(tracking.success_mask, dtype=bool)
    path_index = np.asarray(tracking.path_index, dtype=int)
    shift_time = np.asarray(tracking.shift_time, dtype=float)
    if success.shape != path_index.shape or success.shape != shift_time.shape:
        raise AssertionError(
            "TrackingResult success_mask, path_index, and shift_time must have "
            "identical shapes."
        )

    failure = (~success) | (path_index < 0) | (~np.isfinite(shift_time))
    if np.any(success & failure):
        raise AssertionError(
            "TrackingResult invariant violated: a successful row has no valid "
            "path index or finite shift."
        )
    return failure


def _normalise_receiver_mask(receiver_mask, shape: tuple[int, ...]) -> np.ndarray:
    if receiver_mask is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(receiver_mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(
            f"receiver_mask must have shape {shape}; got {result.shape}."
        )
    return result


def _path_oscillation_index(
    shift_time: np.ndarray,
    path_mask: np.ndarray,
    *,
    epsilon_time: float,
) -> float:
    """Return the fraction of significant zigzags in consecutive triples."""

    if not np.isfinite(epsilon_time) or epsilon_time <= 0.0:
        raise ValueError("epsilon_time must be finite and positive.")

    shift_time = np.asarray(shift_time, dtype=float)
    path_mask = np.asarray(path_mask, dtype=bool)
    if shift_time.shape != path_mask.shape:
        raise ValueError("shift_time and path_mask must have identical shapes.")

    indices = np.flatnonzero(path_mask)
    if indices.size < 3:
        return 0.0

    threshold = 0.3 * float(epsilon_time)
    zigzag_count = 0
    triple_count = 0
    for index in range(1, indices.size - 1):
        i0, i1, i2 = (int(value) for value in indices[index - 1 : index + 2])
        if i1 != i0 + 1 or i2 != i1 + 1:
            continue

        tau0, tau1, tau2 = (float(shift_time[i]) for i in (i0, i1, i2))
        if not np.isfinite([tau0, tau1, tau2]).all():
            continue

        d0 = tau1 - tau0
        d1 = tau2 - tau1
        triple_count += 1
        if d0 * d1 < 0.0 and abs(d1 - d0) >= threshold:
            zigzag_count += 1

    return float(zigzag_count) / float(triple_count) if triple_count else 0.0


def evaluate_dp_failure(
    tracking,
    *,
    receiver_mask=None,
    apply_quality: bool = True,
    trust_correlation: float = DEFAULT_TRUST_CORRELATION,
    collapse_direct_low_quality_fraction: float = (
        DEFAULT_COLLAPSE_DIRECT_LOW_QUALITY_FRACTION
    ),
    collapse_secondary_low_quality_fraction: float = (
        DEFAULT_COLLAPSE_SECONDARY_LOW_QUALITY_FRACTION
    ),
    collapse_zigzag_fraction: float = DEFAULT_COLLAPSE_ZIGZAG_FRACTION,
    epsilon_time: float = 0.040,
) -> DPFailureResult:
    """Classify post-DP receiver failures and whole-event collapse.

    The existing structural DP failure is always preserved.  When
    ``apply_quality`` is true, a receiver is additionally classified as a
    point failure when the absolute tracked waveform ZNCC is below
    ``trust_correlation`` (or is non-finite).

    A whole reflector/shot collapses directly when its low-quality fraction
    reaches ``collapse_direct_low_quality_fraction``.  It also collapses when
    the lower ``collapse_secondary_low_quality_fraction`` is reached together
    with a significant local-zigzag fraction of at least
    ``collapse_zigzag_fraction``.  Zigzags are measured only over strictly
    consecutive receiver triples and must turn by at least 0.3 times
    ``epsilon_time``.

    For a collapsed event every receiver in ``receiver_mask`` is returned as
    a failure, so the existing ``path_failure`` diagnostics continue to count
    all points that were replaced by fallback.
    """

    if not isinstance(apply_quality, (bool, np.bool_)):
        raise ValueError("apply_quality must be boolean.")
    if not np.isfinite(trust_correlation) or not 0.0 <= trust_correlation <= 1.0:
        raise ValueError("trust_correlation must be finite and in [0, 1].")
    for name, value in (
        ("collapse_direct_low_quality_fraction", collapse_direct_low_quality_fraction),
        (
            "collapse_secondary_low_quality_fraction",
            collapse_secondary_low_quality_fraction,
        ),
        ("collapse_zigzag_fraction", collapse_zigzag_fraction),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    if (
        float(collapse_secondary_low_quality_fraction)
        > float(collapse_direct_low_quality_fraction)
    ):
        raise ValueError(
            "collapse_secondary_low_quality_fraction must not exceed "
            "collapse_direct_low_quality_fraction."
        )
    if not np.isfinite(epsilon_time) or float(epsilon_time) <= 0.0:
        raise ValueError("epsilon_time must be finite and positive.")

    success = np.asarray(tracking.success_mask, dtype=bool)
    path_index = np.asarray(tracking.path_index, dtype=int)
    shift_time = np.asarray(tracking.shift_time, dtype=float)

    if not (success.shape == path_index.shape == shift_time.shape):
        raise AssertionError(
            "TrackingResult success_mask, path_index, and shift_time must have "
            "identical shapes."
        )

    scope = _normalise_receiver_mask(receiver_mask, success.shape)
    hard_failure_full = _hard_failure_mask(tracking)
    hard_failure = scope & hard_failure_full

    # Reference/Tobs and any legacy caller can explicitly bypass the new
    # post-DP quality logic.  Do not even require tracked_correlation here, so
    # old lightweight TrackingResult stubs remain compatible.
    if not apply_quality:
        trusted = scope & (~hard_failure_full) & np.isfinite(shift_time)
        global_values = shift_time[trusted]
        global_lag = (
            float(np.median(global_values)) if global_values.size else None
        )
        n_scope = int(np.count_nonzero(scope))
        trusted_fraction = (
            float(np.count_nonzero(trusted)) / float(n_scope)
            if n_scope > 0
            else 0.0
        )
        return DPFailureResult(
            hard_failure_mask=hard_failure,
            failure_mask=hard_failure,
            event_collapse=False,
            trusted_fraction=trusted_fraction,
            low_quality_fraction=1.0 - trusted_fraction,
            oscillation_index=0.0,
            global_lag=global_lag,
        )

    correlation = np.asarray(tracking.tracked_correlation, dtype=float)
    if correlation.shape != success.shape:
        raise AssertionError(
            "TrackingResult tracked_correlation must have the same shape as "
            "success_mask when post-DP quality is enabled."
        )

    abs_correlation = np.abs(correlation)
    trusted = (
        scope
        & (~hard_failure_full)
        & np.isfinite(abs_correlation)
        & (abs_correlation >= float(trust_correlation))
    )

    n_scope = int(np.count_nonzero(scope))
    trusted_count = int(np.count_nonzero(trusted))
    trusted_fraction = (
        float(trusted_count) / float(n_scope) if n_scope > 0 else 0.0
    )
    low_quality_fraction = (
        float(n_scope - trusted_count) / float(n_scope)
        if n_scope > 0
        else 1.0
    )

    path_mask = scope & (~hard_failure_full) & np.isfinite(shift_time)
    oscillation_index = _path_oscillation_index(
        shift_time,
        path_mask,
        epsilon_time=float(epsilon_time),
    )

    direct_quality_collapse = (
        n_scope > 0
        and low_quality_fraction
        >= float(collapse_direct_low_quality_fraction)
    )
    secondary_path_collapse = (
        n_scope > 0
        and low_quality_fraction
        >= float(collapse_secondary_low_quality_fraction)
        and oscillation_index >= float(collapse_zigzag_fraction)
    )
    event_collapse = bool(direct_quality_collapse or secondary_path_collapse)

    low_quality_point = (
        scope
        & (~hard_failure_full)
        & (
            (~np.isfinite(abs_correlation))
            | (abs_correlation < float(trust_correlation))
        )
    )
    failure = hard_failure | low_quality_point
    if event_collapse:
        failure = np.array(scope, dtype=bool, copy=True)

    global_values = shift_time[trusted]
    global_lag = float(np.median(global_values)) if global_values.size else None

    return DPFailureResult(
        hard_failure_mask=hard_failure,
        failure_mask=failure,
        event_collapse=event_collapse,
        trusted_fraction=float(trusted_fraction),
        low_quality_fraction=float(low_quality_fraction),
        oscillation_index=float(oscillation_index),
        global_lag=global_lag,
    )


def dp_failure_mask(tracking) -> np.ndarray:
    """Return receiver rows for which DP produced no usable tracked lag.

    This public function intentionally preserves the legacy structural-failure
    semantics.  Post-DP correlation quality and whole-event collapse are
    evaluated only by :func:`evaluate_dp_failure` and must be requested
    explicitly by the candidate production workflow.
    """

    return _hard_failure_mask(tracking)


# Compatibility alias: keep exactly the same legacy meaning as before.
def path_failure_mask(tracking) -> np.ndarray:
    return dp_failure_mask(tracking)

def low_correlation_qc_mask(
    tracking,
    min_correlation: float | None,
) -> np.ndarray:
    """Legacy signed-correlation QC, based only on structural DP success."""

    correlation = np.asarray(tracking.tracked_correlation, dtype=float)
    if min_correlation is None:
        return np.zeros(correlation.shape, dtype=bool)
    if not np.isfinite(min_correlation) or not -1.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be in [-1, 1] or None.")
    return (
        ~dp_failure_mask(tracking)
        & np.isfinite(correlation)
        & (correlation < float(min_correlation))
    )


def boundary_qc_mask(tracking) -> np.ndarray:
    """Legacy boundary QC, based only on structural DP success."""

    boundary = np.asarray(tracking.boundary_flag, dtype=bool)
    return (~dp_failure_mask(tracking)) & boundary
