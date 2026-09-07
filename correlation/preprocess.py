"""Optional paired preprocessing hooks for fixed-window correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PreprocessError(ValueError):
    """Raised when correlation preprocessing inputs are invalid."""


def brutal_picker(data: np.ndarray, threshold_fraction: float = 0.001) -> np.ndarray:
    """Pick the first threshold crossing independently on every trace.

    This is the NumPy ``[shot, receiver, time]`` equivalent of SWIT's
    ``brutal_picker``.  A completely zero trace receives pick ``-1``.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise PreprocessError("data must be finite with shape [shot, receiver, time].")
    if not np.isfinite(threshold_fraction) or not 0 < threshold_fraction < 1:
        raise PreprocessError("threshold_fraction must be finite and in (0, 1).")

    peak = np.max(np.abs(values), axis=-1)
    threshold = threshold_fraction * peak
    crossing = np.abs(values) > threshold[..., None]
    picks = np.argmax(crossing, axis=-1).astype(int)
    picks[~np.any(crossing, axis=-1)] = -1
    return picks


def _early_mute_mask(
    picks: np.ndarray,
    *,
    nt: int,
    mute_window_samples: int,
    taper_samples: int,
) -> np.ndarray:
    """Build SWIT-compatible early-arrival sine-taper masks."""

    mask = np.ones((*picks.shape, nt), dtype=float)
    half_taper = taper_samples // 2
    taper = np.sin(np.linspace(0.0, np.pi, 2 * taper_samples))[:taper_samples]

    for shot, receiver in np.ndindex(picks.shape):
        first_break = int(picks[shot, receiver])
        if first_break < 0:
            mask[shot, receiver] = 0.0
            continue
        center = first_break + mute_window_samples
        left = center - half_taper
        right = left + taper_samples
        trace_mask = mask[shot, receiver]
        if 1 < left < right < nt:
            trace_mask[:left] = 0.0
            trace_mask[left:right] = taper
        elif left < 1 <= right:
            clipped_right = min(right, nt)
            trace_mask[:clipped_right] = taper[
                taper_samples - clipped_right : taper_samples
            ]
        elif left < nt < right:
            trace_mask[: max(left, 0)] = 0.0
            trace_mask[max(left, 0) : nt] = taper[: nt - max(left, 0)]
        elif left >= nt:
            trace_mask[:] = 0.0
    return mask


@dataclass(frozen=True)
class BrutalPickerEarlyMute:
    """Paired preprocess hook that removes early/direct arrivals.

    Observed and synthetic gathers are picked independently, following the
    existing SWIT behavior, while sharing the same threshold, delay, taper,
    and sampling interval.  Inputs are copied and never modified in place.
    """

    dt: float
    mute_window_time: float = 0.25
    threshold_fraction: float = 0.001
    taper_samples: int = 100

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise PreprocessError("dt must be finite and positive.")
        if not np.isfinite(self.mute_window_time) or self.mute_window_time < 0:
            raise PreprocessError("mute_window_time must be finite and non-negative.")
        if not np.isfinite(self.threshold_fraction) or not 0 < self.threshold_fraction < 1:
            raise PreprocessError("threshold_fraction must be finite and in (0, 1).")
        if (
            isinstance(self.taper_samples, bool)
            or int(self.taper_samples) != self.taper_samples
            or self.taper_samples <= 0
        ):
            raise PreprocessError("taper_samples must be a positive integer.")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "mute_window_time", float(self.mute_window_time))
        object.__setattr__(self, "threshold_fraction", float(self.threshold_fraction))
        object.__setattr__(self, "taper_samples", int(self.taper_samples))

    def apply(self, data: np.ndarray) -> np.ndarray:
        values = np.asarray(data, dtype=float)
        picks = brutal_picker(values, self.threshold_fraction)
        masks = _early_mute_mask(
            picks,
            nt=values.shape[-1],
            mute_window_samples=int(np.ceil(self.mute_window_time / self.dt)),
            taper_samples=self.taper_samples,
        )
        return np.array(values * masks, dtype=float, copy=True)

    def __call__(
        self,
        observed: np.ndarray,
        synthetic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed_values = np.asarray(observed, dtype=float)
        synthetic_values = np.asarray(synthetic, dtype=float)
        if observed_values.shape != synthetic_values.shape:
            raise PreprocessError("observed and synthetic must have equal shapes.")
        return self.apply(observed_values), self.apply(synthetic_values)


__all__ = ["BrutalPickerEarlyMute", "PreprocessError", "brutal_picker"]
