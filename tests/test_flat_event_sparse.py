import numpy as np

from WRTI.tracking.flat_event import TransitionKind, track_flattened_event_sparse


def _ricker(time, center, frequency=20.0):
    value = np.pi * frequency * (time - center)
    return (1.0 - 2.0 * value**2) * np.exp(-(value**2))


def _track(gather, x, seed, seed_time, **overrides):
    options = {
        "candidate_min_distance_time": 0.004,
        "candidate_min_prominence": 0.02,
        "max_candidates_per_trace": 64,
        "coherence_half_window_time": 0.025,
        "min_neighbor_zncc": 0.30,
        "max_residual_slope_ms_per_100m": 100.0,
        "max_prediction_error_time": 0.006,
        "max_active_states": 2000,
    }
    options.update(overrides)
    return track_flattened_event_sparse(
        gather, receiver_x=x, valid_receiver=np.ones(len(x), dtype=bool),
        seed_receiver=seed, seed_time=seed_time, dt=0.001, **options,
    )


def test_sparse_tracker_keeps_seed_event_despite_stronger_reflector():
    time = np.arange(500) * 0.001
    x = np.arange(21) * 10.0
    seed = 10
    gather = np.stack([
        _ricker(time, 0.25) + 4.0 * _ricker(time, 0.34) for _ in x
    ])
    result = _track(gather, x, seed, 0.25)
    assert result.pick_time[seed] == 0.25
    np.testing.assert_allclose(result.pick_time, 0.25, atol=0.002)


def test_sparse_tracker_follows_smooth_curve_through_crossing():
    time = np.arange(600) * 0.001
    x = np.arange(31) * 10.0
    seed = 15
    target = 0.28 + 0.00008 * (x - x[seed]) + 2e-7 * (x - x[seed]) ** 2
    crossing = 22
    competitor = target[crossing] - 0.00010 * (x - x[crossing])
    gather = np.stack([
        _ricker(time, target[i], frequency=50.0)
        + 0.6 * _ricker(time, competitor[i], frequency=50.0)
        for i in range(x.size)
    ])
    result = _track(
        gather, x, seed, target[seed],
        max_residual_slope_ms_per_100m=150.0,
        max_prediction_error_time=0.008,
        coherence_half_window_time=0.010,
        min_neighbor_zncc=0.10,
    )
    np.testing.assert_allclose(result.pick_time, target, atol=0.003)


def test_sparse_tracker_rejects_abrupt_v_and_opposite_phase():
    time = np.arange(500) * 0.001
    x = np.arange(25) * 10.0
    seed = 12
    target = np.full(x.size, 0.25)
    crossing = 18
    v_shape = 0.25 + 0.003 * np.abs(np.arange(x.size) - crossing)
    gather = np.stack([
        _ricker(time, target[i], frequency=50.0)
        - 0.6 * _ricker(time, v_shape[i], frequency=18.0)
        for i in range(x.size)
    ])
    result = _track(
        gather, x, seed, 0.25, min_neighbor_zncc=0.20,
        max_prediction_error_time=0.008,
    )
    np.testing.assert_allclose(result.pick_time, target, atol=0.003)


def test_sparse_tracker_stops_at_gap_and_uses_physical_receiver_spacing():
    time = np.arange(400) * 0.001
    x = np.array([0.0, 8.0, 21.0, 35.0, 52.0, 70.0, 91.0])
    centers = 0.20 + 0.00005 * x
    gather = np.stack([np.exp(-((time - center) / 0.003) ** 2) for center in centers])
    valid = np.ones(x.size, dtype=bool)
    valid[1] = False
    result = track_flattened_event_sparse(
        gather, receiver_x=x, valid_receiver=valid, seed_receiver=3,
        seed_time=centers[3], dt=0.001,
        candidate_min_distance_time=0.012, candidate_min_prominence=0.02,
        coherence_half_window_time=0.025, min_neighbor_zncc=0.30,
        max_residual_slope_ms_per_100m=10.0,
        max_prediction_error_time=0.004,
    )
    assert not result.success_mask[0]
    assert not result.success_mask[1]
    assert result.success_mask[2:].all()
    assert result.pick_time[3] == centers[3]


def test_sparse_tracker_large_gather_keeps_sparse_state_counts():
    time = np.arange(4001) * 0.001
    x = np.arange(651) * 10.0
    gather = np.stack([_ricker(time, 1.0) for _ in x])
    result = _track(gather, x, 325, 1.0)
    assert result.candidate_count.max() <= 64
    assert result.active_state_max <= 2000
    assert result.pick_sample.shape == (651,)


def test_state_valid_filters_candidates_after_peak_detection():
    time = np.arange(500) * 0.001
    x = np.arange(11) * 10.0
    gather = np.stack([_ricker(time, 0.25) + 5 * _ricker(time, 0.34) for _ in x])
    ownership = np.broadcast_to((time > 0.22) & (time < 0.28), gather.shape).copy()
    result = _track(gather, x, 5, 0.25, state_valid=ownership)
    assert np.all(result.candidate_count <= result.candidate_count_before_ownership)
    np.testing.assert_allclose(result.pick_time, 0.25, atol=0.002)


def test_prediction_error_is_soft_below_emergency_bound():
    time = np.arange(500) * 0.001
    x = np.arange(9) * 10.0
    centers = np.full(x.size, 0.25)
    centers[6:] += 0.020
    gather = np.stack([_ricker(time, center) for center in centers])
    result = _track(
        gather, x, 4, 0.25, max_residual_slope_ms_per_100m=300.0,
        prediction_soft_scale_time=0.010, prediction_penalty_weight=0.20,
        max_prediction_error_time=0.050,
    )
    assert result.success_mask[6]


def test_prediction_error_over_emergency_bound_stops_direction():
    time = np.arange(600) * 0.001
    x = np.arange(9) * 10.0
    centers = np.full(x.size, 0.25)
    centers[6:] += 0.080
    gather = np.stack([np.exp(-((time - center) / 0.003) ** 2) for center in centers])
    result = _track(
        gather, x, 4, 0.25, max_residual_slope_ms_per_100m=1000.0,
        max_prediction_error_time=0.050,
    )
    assert not result.success_mask[6]


def test_fixed_anchor_is_not_reoptimized_by_full_tracking():
    time = np.arange(500) * 0.001
    x = np.arange(15) * 10.0
    seed = 7
    target = np.full(x.size, 0.250)
    competitor = np.full(x.size, 0.290)
    gather = np.stack([_ricker(time, target[i]) + 3.0 * _ricker(time, competitor[i])
                       for i in range(x.size)])
    anchor_mask = np.zeros(x.size, dtype=bool)
    anchor_mask[4:11] = True
    anchor_sample = np.full(x.size, -1, dtype=int)
    anchor_sample[anchor_mask] = 250
    anchor_time = np.full(x.size, np.nan)
    anchor_time[anchor_mask] = 0.250
    result = track_flattened_event_sparse(
        gather, receiver_x=x, valid_receiver=np.ones(x.size, dtype=bool),
        seed_receiver=seed, seed_time=0.250, dt=0.001,
        candidate_min_prominence=0.01, coherence_half_window_time=0.010,
        anchor_receiver_mask=anchor_mask, anchor_pick_sample=anchor_sample,
        anchor_pick_time=anchor_time,
    )
    np.testing.assert_array_equal(result.pick_sample[anchor_mask], anchor_sample[anchor_mask])
    np.testing.assert_allclose(result.pick_time[anchor_mask], anchor_time[anchor_mask])


def _anchored_rescue_case(gather, valid=None, **rescue):
    nr = gather.shape[0]
    seed = nr // 2
    mask = np.zeros(nr, dtype=bool); mask[seed - 2:seed + 3] = True
    sample = np.full(nr, -1, dtype=int); sample[mask] = 250
    pick_time = np.full(nr, np.nan); pick_time[mask] = 0.250
    options = {"candidate_min_prominence": 0.05, "coherence_half_window_time": 0.010}
    options.update(rescue)
    return track_flattened_event_sparse(
        gather, receiver_x=np.arange(nr) * 10.0,
        valid_receiver=np.ones(nr, dtype=bool) if valid is None else valid,
        seed_receiver=seed, seed_time=0.250, dt=0.001,
        anchor_receiver_mask=mask, anchor_pick_sample=sample,
        anchor_pick_time=pick_time, continuation_rescue={"enabled": True,
            "half_width_time": 0.020, "max_parent_states": 8,
            "min_prominence": 0.0, "max_valid_receiver_gap": 1}, **options,
    )


def test_predictive_rescue_recovers_positive_peak_missed_by_prominence():
    time = np.arange(500) * 0.001
    gather = np.stack([_ricker(time, 0.250) for _ in range(15)])
    result = _anchored_rescue_case(gather, candidate_min_prominence=1000.0)
    assert result.rescue_attempt_count > 0
    assert result.rescue_success_count > 0
    assert np.any(result.rescue_used_mask)
    assert np.all(result.selected_amplitude[result.success_mask] > 0.0)
    assert np.all(result.selected_transition_kind[result.rescue_used_mask] == TransitionKind.RESCUE)


def test_normal_transition_does_not_call_rescue():
    time = np.arange(500) * 0.001
    gather = np.stack([_ricker(time, 0.250) for _ in range(15)])
    result = _anchored_rescue_case(gather)
    assert result.rescue_attempt_count == 0


def test_one_unrecognisable_valid_receiver_is_nan_then_path_reconnects():
    time = np.arange(500) * 0.001
    gather = np.stack([_ricker(time, 0.250) for _ in range(15)])
    gather[10] = 0.0
    result = _anchored_rescue_case(gather)
    assert not result.success_mask[10] and np.isnan(result.pick_time[10])
    assert result.skipped_valid_receiver_mask[10]
    assert result.selected_transition_kind[10] == TransitionKind.GAP
    assert result.success_mask[11]


def test_two_unrecognisable_valid_receivers_stop_continuation():
    time = np.arange(500) * 0.001
    gather = np.stack([_ricker(time, 0.250) for _ in range(15)])
    gather[10:12] = 0.0
    result = _anchored_rescue_case(gather)
    assert result.skipped_valid_receiver_mask[10]
    assert not result.success_mask[12]


def test_invalid_receiver_is_not_treated_as_skippable_waveform_gap():
    time = np.arange(500) * 0.001
    gather = np.stack([_ricker(time, 0.250) for _ in range(15)])
    valid = np.ones(15, dtype=bool); valid[10] = False
    result = _anchored_rescue_case(gather, valid=valid)
    assert not result.skipped_valid_receiver_mask[10]
    assert not result.success_mask[11]


def _escape_track(centers, *, base_upper=0.252, similarity=0.85, rescue=False,
                  candidate_min_prominence=0.01, outside_frequency=None):
    time = np.arange(400) * 0.001
    nr = len(centers)
    seed = 5
    gather = np.stack([
        _ricker(time, center, frequency=(outside_frequency if outside_frequency and i >= 8 else 40.0))
        for i, center in enumerate(centers)
    ])
    base = np.broadcast_to(time <= base_upper, gather.shape).copy()
    expanded = np.broadcast_to(time <= 0.285, gather.shape).copy()
    anchor = np.zeros(nr, dtype=bool); anchor[3:8] = True
    samples = np.full(nr, -1, dtype=int); samples[anchor] = 250
    times = np.full(nr, np.nan); times[anchor] = 0.250
    return track_flattened_event_sparse(
        gather, receiver_x=np.arange(nr) * 10.0,
        valid_receiver=np.ones(nr, dtype=bool), seed_receiver=seed,
        seed_time=0.250, dt=0.001, state_valid=base,
        escape_state_valid=expanded,
        anchor_receiver_mask=anchor, anchor_pick_sample=samples,
        anchor_pick_time=times,
        candidate_min_prominence=candidate_min_prominence,
        coherence_half_window_time=0.010,
        max_residual_slope_ms_per_100m=300.0,
        max_prediction_error_time=0.050,
        continuation_rescue={"enabled": rescue, "half_width_time": 0.030,
                             "max_parent_states": 8, "min_prominence": 0.0,
                             "max_valid_receiver_gap": 0},
        ownership_escape={"enabled": True, "min_neighbor_similarity": similarity,
                          "max_prediction_error_time": 0.015},
    )


def test_base_candidate_does_not_call_ownership_escape():
    result = _escape_track(np.full(20, 0.250))
    assert result.success_mask.all()
    assert result.ownership_escape_attempt_count == 0
    assert not result.ownership_escape_used_mask.any()


def test_coherent_peak_just_outside_base_ownership_escapes():
    centers = np.full(20, 0.250); centers[8:] = 0.255
    result = _escape_track(centers)
    assert result.success_mask.all()
    assert result.ownership_escape_used_mask[8:].all()
    assert result.ownership_escape_success_count > 0


def test_ownership_escape_rejects_low_similarity_candidate():
    centers = np.full(20, 0.250); centers[8:] = 0.255
    result = _escape_track(centers, outside_frequency=120.0)
    assert not result.success_mask[8]


def test_ownership_escape_rejects_large_prediction_error():
    centers = np.full(20, 0.250); centers[8:] = 0.270
    result = _escape_track(centers)
    assert not result.success_mask[8]


def test_ownership_escape_has_no_consecutive_receiver_limit():
    centers = np.full(50, 0.250); centers[8:] = 0.255
    result = _escape_track(centers)
    assert result.success_mask.all()
    assert np.count_nonzero(result.ownership_escape_used_mask) == 42


def test_ownership_escape_returns_to_base_automatically():
    centers = np.full(30, 0.250); centers[8:20] = 0.255
    result = _escape_track(centers)
    assert result.success_mask.all()
    assert result.ownership_escape_used_mask[8:20].all()
    assert not result.ownership_escape_used_mask[20:].any()


def test_predictive_rescue_can_use_expanded_ownership():
    centers = np.full(20, 0.250); centers[8:] = 0.255
    result = _escape_track(centers, rescue=True, candidate_min_prominence=1000.0)
    assert result.success_mask.all()
    assert result.rescue_success_count > 0
    assert result.ownership_escape_used_mask[8:].all()


def test_escape_mask_is_ignored_when_escape_is_disabled():
    time = np.arange(400) * 0.001
    centers = np.full(12, 0.250); centers[7:] = 0.255
    gather = np.stack([_ricker(time, center, frequency=40.0) for center in centers])
    base = np.broadcast_to(time <= 0.252, gather.shape).copy()
    expanded = np.ones_like(base)
    result = track_flattened_event_sparse(
        gather, receiver_x=np.arange(12) * 10.0,
        valid_receiver=np.ones(12, dtype=bool), seed_receiver=5,
        seed_time=0.250, dt=0.001, state_valid=base,
        escape_state_valid=expanded, ownership_escape={"enabled": False},
        candidate_min_prominence=0.01, coherence_half_window_time=0.010,
    )
    assert not result.success_mask[7]
