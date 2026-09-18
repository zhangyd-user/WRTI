from types import SimpleNamespace

import numpy as np

from WRTI.tracking.quality_control import QualityAuditConfig
from WRTI.tracking.tracking_break_repair import (
    _best_segments,
    _boundary_receivers,
    evaluate_trusted_quality,
    extract_seed_connected_trusted_prefix,
    extract_trusted_segments,
)


def _tracking(size, success=None, similarity=None, error=None):
    success = np.ones(size, dtype=bool) if success is None else success
    samples = np.arange(size, dtype=int) + 10
    return SimpleNamespace(
        success_mask=success,
        pick_sample=np.where(success, samples, -1),
        pick_time=np.where(success, samples * 0.001, np.nan),
        neighbor_correlation=(np.full(size, 0.9) if similarity is None else similarity),
        prediction_error=(np.zeros(size) if error is None else error),
        residual_slope=np.zeros(size),
    )


def test_trusted_segment_uses_audit_coverage_and_allows_isolated_miss():
    success = np.ones(20, dtype=bool)
    success[5] = False
    usable = np.ones(20, dtype=bool)
    segments = extract_trusted_segments(
        flat=np.ones((20, 100)), tracking=_tracking(20, success=success),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="ORIGINAL",
    )
    assert len(segments) == 1
    assert segments[0].coverage == 0.95


def test_persistent_low_similarity_is_excluded_from_trusted_segment():
    similarity = np.full(32, 0.9)
    similarity[12:20] = 0.5
    usable = np.ones(32, dtype=bool)
    segments = extract_trusted_segments(
        flat=np.ones((32, 100)), tracking=_tracking(32, similarity=similarity),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="ORIGINAL",
    )
    assert all(not np.all(segment.trusted_mask[12:20]) for segment in segments)


def test_distributed_low_similarity_fails_segment_support_fraction():
    similarity = np.full(35, 0.95)
    similarity[[3, 12, 14, 21, 27, 30, 34]] = 0.5
    usable = np.ones(35, dtype=bool)
    segments = extract_trusted_segments(
        flat=np.ones((35, 100)), tracking=_tracking(35, similarity=similarity),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="ORIGINAL",
    )
    assert not any(segment.start_receiver == 0 and segment.end_receiver == 34 for segment in segments)
    quality = evaluate_trusted_quality(
        _tracking(35, similarity=similarity), usable, 0, 34, QualityAuditConfig(),
    )
    assert quality.similarity_support_fraction == 0.8
    assert "LOW_SIMILARITY_SUPPORT" in quality.reasons


def test_boundary_receivers_are_aperture_edges_outside_trusted_segment():
    usable = np.ones(20, dtype=bool)
    segment = extract_trusted_segments(
        flat=np.ones((20, 100)), tracking=_tracking(20),
        tracking_usable_receiver=usable,
        quality_available_receiver=np.r_[np.zeros(5, bool), np.ones(10, bool), np.zeros(5, bool)],
        config=QualityAuditConfig(), source_type="ORIGINAL",
    )[0]
    assert _boundary_receivers((segment,), usable) == (0, 19)


def test_best_combination_uses_at_most_two_segments():
    usable = np.ones(20, dtype=bool)
    segments = []
    for start in (0, 7, 14):
        available = np.zeros(20, dtype=bool)
        available[start:min(start + 8, 20)] = True
        segments.extend(extract_trusted_segments(
            flat=np.ones((20, 100)), tracking=_tracking(20),
            tracking_usable_receiver=usable, quality_available_receiver=available,
            config=QualityAuditConfig(), source_type="ORIGINAL",
        ))
    selected, _, _ = _best_segments(segments, usable)
    assert len(selected) <= 2


def test_trusted_segment_endpoints_are_successful_receivers():
    success = np.ones(20, dtype=bool)
    success[17:] = False
    usable = np.ones(20, dtype=bool)
    segment = extract_trusted_segments(
        flat=np.ones((20, 100)), tracking=_tracking(20, success=success),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="ORIGINAL",
    )[0]
    assert segment.end_receiver == 16


def test_repair_quality_rejects_persistent_prediction_instability_without_low_zncc():
    error = np.zeros(16)
    error[4:7] = 0.020
    quality = evaluate_trusted_quality(
        _tracking(16, error=error), np.ones(16, bool), 0, 15,
        QualityAuditConfig(), reference_time=0.010, identity_half_width=1.0,
    )
    assert "PERSISTENT_PREDICTION_INSTABILITY" in quality.reasons
    assert "PERSISTENT_POOR_CONTINUITY" not in quality.reasons


def test_repair_quality_rejects_only_persistent_identity_drift():
    tracking = _tracking(16)
    tracking.pick_time[4:7] = 0.5
    quality = evaluate_trusted_quality(
        tracking, np.ones(16, bool), 0, 15, QualityAuditConfig(),
        reference_time=0.010, identity_half_width=0.2,
    )
    assert "PERSISTENT_IDENTITY_DRIFT" in quality.reasons


def test_boundary_repair_keeps_seed_connected_prefix_before_bad_tail():
    similarity = np.full(24, 0.95)
    similarity[12:] = 0.5
    usable = np.ones(24, bool)
    segment, rejection, rejected_end = extract_seed_connected_trusted_prefix(
        flat=np.ones((24, 100)), tracking=_tracking(24, similarity=similarity),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="LEFT_SEED_1", seed_receiver=0,
        direction=1, source_seed_time=0.010, allowed_receiver_mask=usable,
        reference_time=0.010, identity_half_width=1.0,
    )
    assert segment.end_receiver == 12
    assert rejected_end == 13
    assert "LOW_SIMILARITY_SUPPORT" in rejection.reasons


def test_boundary_repair_rejects_prefix_when_seed_window_is_bad():
    similarity = np.full(24, 0.95)
    similarity[:4] = 0.5
    usable = np.ones(24, bool)
    segment, rejection, _ = extract_seed_connected_trusted_prefix(
        flat=np.ones((24, 100)), tracking=_tracking(24, similarity=similarity),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="LEFT_SEED_1", seed_receiver=0,
        direction=1, source_seed_time=0.010, allowed_receiver_mask=usable,
        reference_time=0.010, identity_half_width=1.0,
    )
    assert segment is None
    assert "LOW_SIMILARITY_SUPPORT" in rejection.reasons


def test_boundary_repair_cuts_persistent_neighbor_ownership_intrusion():
    usable = np.ones(24, bool)
    intrusion = np.zeros(24, bool)
    intrusion[12:15] = True
    segment, rejection, _ = extract_seed_connected_trusted_prefix(
        flat=np.ones((24, 100)), tracking=_tracking(24),
        tracking_usable_receiver=usable, quality_available_receiver=usable,
        config=QualityAuditConfig(), source_type="LEFT_SEED_1", seed_receiver=0,
        direction=1, source_seed_time=0.010, allowed_receiver_mask=usable,
        neighbor_intrusion_mask=intrusion,
    )
    assert segment is not None and segment.end_receiver < 15
    assert "PERSISTENT_NEIGHBOR_OWNERSHIP_INTRUSION" in rejection.reasons
