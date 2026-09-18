import numpy as np

from WRTI.tracking.quality_control import TrackingStatus, audit_rkshot


def _audit(*, size=41, success=None, similarity=None, error=None, valid=None, usable=None, seed=20):
    valid_mask = np.ones(size, dtype=bool) if valid is None else valid
    return audit_rkshot(
        success_mask=np.ones(size, dtype=bool) if success is None else success,
        valid_receiver=valid_mask,
        neighbor_similarity=np.full(size, 0.90) if similarity is None else similarity,
        prediction_error=np.zeros(size) if error is None else error,
        tracking_usable_receiver=valid_mask if usable is None else usable,
        seed_receiver=seed,
    )


def test_clean_complete_track_is_normal():
    assert _audit().status == TrackingStatus.NORMAL


def test_single_large_prediction_spike_with_high_similarity_is_normal():
    error = np.zeros(41)
    error[12] = 0.030
    assert _audit(error=error).status == TrackingStatus.NORMAL


def test_single_low_similarity_with_small_error_is_normal():
    similarity = np.full(41, 0.90)
    similarity[12] = 0.45
    assert _audit(similarity=similarity).status == TrackingStatus.NORMAL


def test_two_receiver_gap_with_high_coverage_is_normal():
    success = np.ones(61, dtype=bool)
    success[[11, 12]] = False
    assert _audit(size=61, seed=30, success=success).status == TrackingStatus.NORMAL


def test_rescue_or_escape_usage_is_not_an_audit_input():
    # The audit sees only the final track quality after continuation is complete.
    assert _audit().status == TrackingStatus.NORMAL


def test_four_low_similarity_points_in_eight_successes_breaks():
    similarity = np.full(41, 0.90)
    similarity[[19, 18, 17, 16]] = 0.60
    result = _audit(similarity=similarity)
    assert result.status == TrackingStatus.TRACKING_BREAK
    assert result.left.persistent_low_similarity


def test_three_severe_points_in_eight_successes_breaks():
    similarity = np.full(41, 0.90)
    error = np.zeros(41)
    similarity[[19, 17, 15]] = 0.60
    error[[19, 17, 15]] = 0.016
    result = _audit(similarity=similarity, error=error)
    assert result.status == TrackingStatus.TRACKING_BREAK
    assert result.left.persistent_poor_continuity


def test_local_low_coverage_is_normal_when_overall_coverage_exceeds_eighty_percent():
    success = np.ones(41, dtype=bool)
    success[[2, 4, 6]] = False
    result = _audit(success=success)
    assert result.status == TrackingStatus.NORMAL
    assert "LOW_COVERAGE" in result.left.reason


def test_early_stop_is_normal_when_overall_coverage_exceeds_eighty_percent():
    success = np.ones(41, dtype=bool)
    success[:5] = False
    result = _audit(success=success)
    assert result.status == TrackingStatus.NORMAL
    assert not result.left.reached_edge
    assert result.left.remaining_usable_after_stop == 5


def test_last_four_bad_points_at_reachable_edge_are_normal():
    similarity = np.full(41, 0.90)
    error = np.zeros(41)
    similarity[:4] = 0.60
    error[:4] = 0.020
    assert _audit(similarity=similarity, error=error).status == TrackingStatus.NORMAL


def test_both_bad_early_sides_are_seed_error():
    success = np.ones(41, dtype=bool)
    success[1:20:2] = False
    success[21:40:2] = False
    assert _audit(success=success).status == TrackingStatus.SEED_ERROR


def test_one_bad_early_side_is_tracking_break():
    similarity = np.full(41, 0.90)
    similarity[1:20] = 0.40
    result = _audit(similarity=similarity)
    assert result.status == TrackingStatus.TRACKING_BREAK
    assert result.left.status == TrackingStatus.TRACKING_BREAK
    assert result.right.status == TrackingStatus.NORMAL


def test_side_without_usable_data_is_ignored():
    usable = np.ones(41, dtype=bool)
    usable[:20] = False
    assert _audit(usable=usable).status == TrackingStatus.NORMAL


def test_support_gap_hides_distant_valid_receivers_from_incomplete_check():
    usable = np.ones(41, dtype=bool)
    usable[5] = False
    success = np.ones(41, dtype=bool)
    success[:6] = False
    result = _audit(success=success, usable=usable)
    assert result.status == TrackingStatus.NORMAL
    assert result.left.reachable_edge == 6


def test_four_remaining_usable_receivers_count_as_reaching_edge():
    success = np.ones(101, dtype=bool)
    success[:4] = False
    result = _audit(size=101, seed=50, success=success)
    assert result.status == TrackingStatus.NORMAL
    assert result.left.reached_edge


def test_single_available_acquisition_side_can_be_seed_error():
    usable = np.ones(41, dtype=bool)
    usable[:20] = False
    success = usable.copy()
    success[22:40:2] = False
    assert _audit(success=success, usable=usable).status == TrackingStatus.SEED_ERROR


def test_r1_is_normal_with_half_overall_coverage_and_no_persistent_collapse():
    success = np.ones(41, dtype=bool)
    success[:10] = False
    result = audit_rkshot(
        success_mask=success,
        valid_receiver=np.ones(41, dtype=bool),
        neighbor_similarity=np.full(41, 0.90),
        prediction_error=np.zeros(41),
        seed_receiver=20,
        reflector=1,
    )
    assert result.left.side_coverage == 0.50
    assert not result.left.reached_edge
    assert result.status == TrackingStatus.NORMAL


def test_r1_half_coverage_does_not_override_persistent_quality_collapse():
    similarity = np.full(41, 0.90)
    similarity[[19, 18, 17, 16]] = 0.60
    result = audit_rkshot(
        success_mask=np.ones(41, dtype=bool),
        valid_receiver=np.ones(41, dtype=bool),
        neighbor_similarity=similarity,
        prediction_error=np.zeros(41),
        seed_receiver=20,
        reflector=1,
    )
    assert result.status == TrackingStatus.TRACKING_BREAK


def test_r1_relaxation_does_not_apply_to_other_reflectors():
    success = np.ones(41, dtype=bool)
    success[:10] = False
    result = audit_rkshot(
        success_mask=success,
        valid_receiver=np.ones(41, dtype=bool),
        neighbor_similarity=np.full(41, 0.90),
        prediction_error=np.zeros(41),
        seed_receiver=20,
        reflector=2,
    )
    assert result.status == TrackingStatus.TRACKING_BREAK


def test_eighty_percent_overall_coverage_without_persistent_collapse_is_normal():
    success = np.ones(101, dtype=bool)
    success[:20] = False
    result = _audit(size=101, seed=50, success=success)
    assert result.left.side_coverage == 0.60
    assert result.status == TrackingStatus.NORMAL


def test_eighty_percent_coverage_does_not_override_persistent_collapse():
    success = np.ones(101, dtype=bool)
    success[:20] = False
    similarity = np.full(101, 0.90)
    similarity[[49, 48, 47, 46]] = 0.60
    result = _audit(size=101, seed=50, success=success, similarity=similarity)
    assert result.status == TrackingStatus.TRACKING_BREAK


def test_coverage_denominator_excludes_receivers_without_rk_data():
    success = np.zeros(41, dtype=bool)
    success[20:] = True
    available = np.zeros(41, dtype=bool)
    available[20:] = True
    result = audit_rkshot(
        success_mask=success,
        valid_receiver=np.ones(41, dtype=bool),
        tracking_usable_receiver=np.ones(41, dtype=bool),
        quality_available_receiver=available,
        neighbor_similarity=np.full(41, 0.90),
        prediction_error=np.zeros(41),
        seed_receiver=20,
        reflector=1,
    )
    assert not result.left.present
    assert result.right.side_coverage == 1.0
    assert result.status == TrackingStatus.NORMAL


def test_r1_overall_coverage_includes_successful_available_anchor_receivers():
    success = np.zeros(51, dtype=bool)
    success[16:41] = True
    usable = np.ones(51, dtype=bool)
    usable[41:] = False
    anchor = np.zeros(51, dtype=bool)
    anchor[16:26] = True
    result = audit_rkshot(
        success_mask=success,
        valid_receiver=usable,
        tracking_usable_receiver=usable,
        quality_available_receiver=usable,
        anchor_receiver_mask=anchor,
        neighbor_similarity=np.full(51, 0.90),
        prediction_error=np.zeros(51),
        seed_receiver=25,
        reflector=1,
    )
    assert result.status == TrackingStatus.NORMAL
