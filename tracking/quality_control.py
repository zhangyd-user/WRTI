"""Read-only quality audit for completed sparse flat-event tracks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TrackingStatus(str, Enum):
    NORMAL = "NORMAL"
    SEED_ERROR = "SEED_ERROR"
    TRACKING_BREAK = "TRACKING_BREAK"
    UNRECOVERABLE = "UNRECOVERABLE"


@dataclass(frozen=True)
class QualityAuditConfig:
    early_receiver_count: int = 20
    early_min_coverage: float = 0.80
    early_min_median_similarity: float = 0.70
    hard_break_min_remaining_usable_receivers: int = 5
    normal_min_side_coverage: float = 0.90
    quality_window_receivers: int = 8
    low_similarity_threshold: float = 0.70
    low_similarity_count_to_fail: int = 4
    severe_bad_prediction_error_time: float = 0.015
    severe_bad_count_to_fail: int = 3

    def __post_init__(self) -> None:
        for name in (
            "early_receiver_count", "hard_break_min_remaining_usable_receivers",
            "quality_window_receivers", "low_similarity_count_to_fail",
            "severe_bad_count_to_fail",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "early_min_coverage", "early_min_median_similarity",
            "normal_min_side_coverage", "low_similarity_threshold",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        value = float(self.severe_bad_prediction_error_time)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("severe_bad_prediction_error_time must be finite and non-negative")


@dataclass(frozen=True)
class SideAudit:
    status: TrackingStatus
    present: bool
    early_bad: bool
    break_receiver: int
    early_coverage: float
    early_median_similarity: float
    max_time_jump: float
    large_jump_count: int
    reached_edge: bool
    reachable_edge: int
    remaining_usable_after_stop: int
    side_coverage: float
    persistent_low_similarity: bool
    persistent_poor_continuity: bool
    reason: str
    usable_receiver_count: int
    successful_receiver_count: int


@dataclass(frozen=True)
class RkShotAudit:
    status: TrackingStatus
    left: SideAudit
    right: SideAudit


def _median(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float(default)


def _has_bad_window(values: np.ndarray, window: int, minimum: int) -> bool:
    if values.size < window:
        return False
    counts = np.convolve(values.astype(int), np.ones(window, dtype=int), mode="valid")
    return bool(np.any(counts >= minimum))


def _ignore_short_edge_run(values: np.ndarray, maximum: int) -> np.ndarray:
    """Do not turn a short terminal disturbance at the data edge into a break."""
    trailing = 0
    for value in values[::-1]:
        if not value:
            break
        trailing += 1
    if 0 < trailing <= maximum:
        values = values.copy()
        values[-trailing:] = False
    return values


def audit_tracking_side(
    *, success_mask, valid_receiver, neighbor_similarity, prediction_error,
    seed_receiver: int, direction: int, anchor_receiver_mask=None,
    tracking_usable_receiver=None, quality_available_receiver=None,
    config: QualityAuditConfig | None = None,
) -> SideAudit:
    """Audit one side after all continuation, rescue, and escape logic is done."""

    cfg = config or QualityAuditConfig()
    success = np.asarray(success_mask, dtype=bool)
    valid = np.asarray(valid_receiver, dtype=bool)
    similarity = np.asarray(neighbor_similarity, dtype=float)
    error = np.asarray(prediction_error, dtype=float)
    if success.ndim != 1 or any(value.shape != success.shape for value in (valid, similarity, error)):
        raise ValueError("audit arrays must be one-dimensional and have equal shapes")
    seed = int(seed_receiver)
    if not 0 <= seed < success.size or direction not in (-1, 1):
        raise ValueError("seed_receiver or direction is invalid")
    anchor = np.zeros(success.size, dtype=bool) if anchor_receiver_mask is None else np.asarray(anchor_receiver_mask, dtype=bool)
    if anchor.shape != success.shape:
        raise ValueError("anchor_receiver_mask must match the tracking arrays")
    usable = valid.copy() if tracking_usable_receiver is None else np.asarray(tracking_usable_receiver, dtype=bool)
    if usable.shape != success.shape:
        raise ValueError("tracking_usable_receiver must match the tracking arrays")
    usable = usable & valid
    available = usable.copy() if quality_available_receiver is None else np.asarray(quality_available_receiver, dtype=bool)
    if available.shape != success.shape:
        raise ValueError("quality_available_receiver must match the tracking arrays")
    available = available & usable

    order = np.arange(seed + direction, -1 if direction < 0 else success.size, direction)
    outward = order[~anchor[order]]
    unusable_positions = np.flatnonzero(~usable[outward])
    side = outward[: int(unusable_positions[0])] if unusable_positions.size else outward
    reachable_edge = int(side[-1]) if side.size else -1
    quality_side = side[available[side]]
    if not quality_side.size:
        return SideAudit(TrackingStatus.NORMAL, False, False, -1, 1.0, 1.0, 0.0, 0,
                         True, reachable_edge, 0, 1.0, False, False, "", 0, 0)

    early = quality_side[: cfg.early_receiver_count]
    early_coverage = float(np.count_nonzero(success[early]) / early.size)
    early_similarity = _median(similarity[early][success[early]], -np.inf)
    early_bad = early_coverage < cfg.early_min_coverage or early_similarity < cfg.early_min_median_similarity

    successful_positions = np.flatnonzero(success[side])
    last_success_position = int(successful_positions[-1]) if successful_positions.size else -1
    remaining = side[last_success_position + 1 :]
    remaining_usable = int(remaining.size)
    reached_edge = remaining_usable < cfg.hard_break_min_remaining_usable_receivers
    break_receiver = -1 if reached_edge else int(remaining[0])
    side_coverage = float(np.count_nonzero(success[quality_side]) / quality_side.size)

    successful = side[success[side]]
    low_similarity = np.isfinite(similarity[successful]) & (similarity[successful] < cfg.low_similarity_threshold)
    severe = low_similarity & np.isfinite(error[successful]) & (np.abs(error[successful]) > cfg.severe_bad_prediction_error_time)
    if reached_edge:
        maximum = cfg.hard_break_min_remaining_usable_receivers - 1
        low_similarity = _ignore_short_edge_run(low_similarity, maximum)
        severe = _ignore_short_edge_run(severe, maximum)
    persistent_low = _has_bad_window(low_similarity, cfg.quality_window_receivers, cfg.low_similarity_count_to_fail)
    persistent_poor = _has_bad_window(severe, cfg.quality_window_receivers, cfg.severe_bad_count_to_fail)

    reasons = []
    if early_bad:
        reasons.append("UNRELIABLE_START")
    if not reached_edge:
        reasons.append("INCOMPLETE")
    if side_coverage < cfg.normal_min_side_coverage:
        reasons.append("LOW_COVERAGE")
    if persistent_low:
        reasons.append("PERSISTENT_LOW_SIMILARITY")
    if persistent_poor:
        reasons.append("PERSISTENT_POOR_CONTINUITY")
    status = TrackingStatus.TRACKING_BREAK if reasons else TrackingStatus.NORMAL

    finite_jumps = np.abs(error[side][np.isfinite(error[side])])
    max_jump = float(np.max(finite_jumps)) if finite_jumps.size else 0.0
    large_jump_count = int(np.count_nonzero(finite_jumps > cfg.severe_bad_prediction_error_time))
    return SideAudit(
        status, True, early_bad, break_receiver, early_coverage, early_similarity,
        max_jump, large_jump_count, reached_edge, reachable_edge, remaining_usable,
        side_coverage, persistent_low, persistent_poor, "+".join(reasons),
        int(quality_side.size), int(np.count_nonzero(success[quality_side])),
    )


def audit_rkshot(
    *, success_mask, valid_receiver, neighbor_similarity, prediction_error,
    seed_receiver: int, anchor_receiver_mask=None, tracking_usable_receiver=None,
    quality_available_receiver=None, config: QualityAuditConfig | None = None,
    reflector: int | None = None,
) -> RkShotAudit:
    """Classify one reflector-shot in the required exclusive priority order."""

    cfg = config or QualityAuditConfig()
    arguments = dict(
        success_mask=success_mask, valid_receiver=valid_receiver,
        neighbor_similarity=neighbor_similarity, prediction_error=prediction_error,
        seed_receiver=seed_receiver, anchor_receiver_mask=anchor_receiver_mask,
        tracking_usable_receiver=tracking_usable_receiver,
        quality_available_receiver=quality_available_receiver, config=cfg,
    )
    left = audit_tracking_side(direction=-1, **arguments)
    right = audit_tracking_side(direction=1, **arguments)
    present = [side for side in (left, right) if side.present]
    if present and all(side.early_bad for side in present):
        status = TrackingStatus.SEED_ERROR
    else:
        usable_count = sum(side.usable_receiver_count for side in present)
        success_count = sum(side.successful_receiver_count for side in present)
        anchor = (
            np.zeros_like(np.asarray(success_mask, dtype=bool))
            if anchor_receiver_mask is None
            else np.asarray(anchor_receiver_mask, dtype=bool)
        )
        available = (
            np.asarray(
                valid_receiver if tracking_usable_receiver is None else tracking_usable_receiver,
                dtype=bool,
            )
            if quality_available_receiver is None
            else np.asarray(quality_available_receiver, dtype=bool)
        )
        available_anchor = anchor & available
        usable_count += int(np.count_nonzero(available_anchor))
        success_count += int(np.count_nonzero(
            np.asarray(success_mask, dtype=bool) & available_anchor
        ))
        no_persistent_collapse = not any(
            side.persistent_low_similarity or side.persistent_poor_continuity
            for side in present
        )
        coverage = success_count / usable_count if usable_count else 0.0
        threshold = 0.50 if int(reflector or 0) == 1 else 0.80
        if coverage >= threshold and no_persistent_collapse:
            status = TrackingStatus.NORMAL
        elif int(reflector or 0) == 1 or any(
            side.status == TrackingStatus.TRACKING_BREAK for side in present
        ):
            status = TrackingStatus.TRACKING_BREAK
        else:
            status = TrackingStatus.NORMAL
    return RkShotAudit(status, left, right)


__all__ = [
    "QualityAuditConfig", "RkShotAudit", "SideAudit", "TrackingStatus",
    "audit_rkshot", "audit_tracking_side",
]
