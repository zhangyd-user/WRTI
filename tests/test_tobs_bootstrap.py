import numpy as np
import WRTI.workflow.tobs_bootstrap as bootstrap_module

from WRTI.workflow.tobs_bootstrap import (
    _ownership_corridors,
    build_bootstrap_fixed_mask,
    snap_control_to_observed_peak,
)


def _bootstrap_mask(tobs):
    return build_bootstrap_fixed_mask(
        np.asarray(tobs, dtype=float),
        dt=0.01,
        t0=0.0,
        nt=31,
        half_window_time=0.05,
        window_type="rectangular",
        tukey_alpha=0.5,
    )


def test_bootstrap_mask_removes_only_one_missing_receiver():
    tobs = np.full((3, 2, 81), 0.15)
    tobs[2, 1, 80] = np.nan
    fixed, _ = _bootstrap_mask(tobs)
    assert not fixed[2, 1, 80]
    assert fixed[2, 1, 79]
    assert fixed[1, 1, 80]


def test_bootstrap_mask_keeps_finite_prefix_of_unrecoverable_rkshot():
    tobs = np.full((1, 1, 10), 0.15)
    tobs[0, 0, 5:] = np.nan
    fixed, _ = _bootstrap_mask(tobs)
    np.testing.assert_array_equal(fixed[0, 0], [True] * 5 + [False] * 5)


def test_bootstrap_mask_keeps_finite_repaired_receiver():
    fixed, _ = _bootstrap_mask([[[0.15]]])
    assert fixed[0, 0, 0]


def test_bootstrap_mask_rejects_incomplete_target_window():
    fixed, _ = _bootstrap_mask([[[0.04]]])
    assert not fixed[0, 0, 0]


def test_bootstrap_mask_does_not_reserve_candidate_lag_margin():
    fixed, windows = _bootstrap_mask([[[0.06]]])
    assert windows.max_lag_samples == 0
    assert fixed[0, 0, 0]


def test_bootstrap_mask_does_not_modify_legacy_mask():
    legacy = np.array([[[False, True]]])
    before = legacy.copy()
    _bootstrap_mask([[[0.15, 0.15]]])
    np.testing.assert_array_equal(legacy, before)


def test_bootstrap_mask_is_independent_of_legacy_false_slot():
    legacy = np.array([[[False]]])
    bootstrap, _ = _bootstrap_mask([[[0.15]]])
    assert not legacy[0, 0, 0]
    assert bootstrap[0, 0, 0]


def test_ownership_corridors_keep_edge_ids_and_report_order_violation():
    predicted = np.array([[1.0, 1.0], [2.0, 2.2], [3.0, 2.1], [4.0, 4.0]])
    control = np.array([1.0, 2.0, 3.0, 4.0])
    upper, lower, violation = _ownership_corridors(predicted, control, 0.1, 0.6)
    assert upper[0, 0] == 0.4
    assert lower[-1, 0] == 4.6
    assert violation[1, 1] and violation[2, 1]
    np.testing.assert_allclose(predicted[:, 1], [1.0, 2.2, 2.1, 4.0])


def test_seed_snap_selects_positive_nearby_valid_local_peak():
    trace = np.zeros(101)
    trace[45], trace[52], trace[81] = 1.0, -2.0, 9.0
    valid = np.ones(101, dtype=bool)
    snapped, sample, success, delta, amplitude = snap_control_to_observed_peak(
        trace, control_time=0.050, dt=0.001, state_valid_row=valid,
        half_width_time=0.030,
    )
    assert (sample, success, amplitude) == (45, True, 1.0)
    assert np.isclose(snapped, 0.045) and np.isclose(delta, -0.005)


def test_seed_snap_rejects_stronger_peak_outside_ownership():
    trace = np.zeros(101)
    trace[48], trace[55] = 1.0, 5.0
    valid = np.ones(101, dtype=bool)
    valid[55] = False
    assert snap_control_to_observed_peak(
        trace, control_time=0.050, dt=0.001, state_valid_row=valid,
    )[1] == 48


def test_seed_snap_falls_back_to_control_without_peak():
    result = snap_control_to_observed_peak(
        np.zeros(101), control_time=0.050, dt=0.001, half_width_time=0.030,
    )
    assert result[:4] == (0.050, 50, False, 0.0)


def _small_sparse_bootstrap(**kwargs):
    dt = 0.001
    time = np.arange(300) * dt
    observed = np.stack([_pulse(time, 0.12) for _ in range(5)])[None]
    tref = np.full((1, 1, 5), 0.10)
    receivers = np.zeros((1, 5, 2))
    receivers[0, :, 0] = np.arange(5) * 10.0
    return build_observed_centers_from_eikonal(
        observed, tref, np.array([[20.0, 0.0]]), receivers,
        control_time=np.array([[0.10]]), dt=dt, tracking_method="sparse_event_dp",
        sparse_tracking_options={"candidate_min_prominence": 0.01,
                                 "coherence_half_window_time": 0.01,
                                 "max_prediction_error_time": 0.05},
        **kwargs,
    )


def test_sparse_without_diagnostic_never_calls_dense(monkeypatch):
    monkeypatch.setattr(bootstrap_module, "track_flattened_event",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dense called")))
    result = _small_sparse_bootstrap(compute_legacy_dense_diagnostic=False)
    assert result.counts["dense_diagnostic_call_count"] == 0
    assert np.isnan(result.legacy_pick_time).all()
    assert np.isclose(result.observed_center[0, 0, 2], result.tracking_seed_time[0, 0])


def test_sparse_never_runs_abandoned_dense_diagnostic(monkeypatch):
    original = bootstrap_module.track_flattened_event
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(bootstrap_module, "track_flattened_event", counted)
    result = _small_sparse_bootstrap(
        compute_legacy_dense_diagnostic=True, legacy_dense_diagnostic_shots=(1,)
    )
    assert len(calls) == result.counts["dense_diagnostic_call_count"] == 0
    assert np.isnan(result.legacy_pick_time).all()

from WRTI.workflow import (
    build_observed_centers_from_eikonal,
    compute_same_x_control_time,
    flatten_observed_gather,
    track_flattened_event,
)
from WRTI.diagnostics import save_same_x_control_test, save_tobs_bootstrap_diagnostic


def _pulse(time, center, sigma=0.008):
    return np.exp(-0.5 * ((time - center) / sigma) ** 2)


def test_bootstrap_recovers_observed_center_after_eikonal_flattening():
    dt = 0.002
    time = np.arange(700) * dt
    receiver_x = np.arange(9, dtype=float) * 20.0
    seed = 4
    moveout = 0.003 * (np.arange(9) - seed) ** 2
    tref = (0.60 + moveout)[None, None, :]
    expected = tref[0, 0] + 0.10
    observed = np.stack([_pulse(time, center) for center in expected])[None, :, :]
    sources = np.array([[receiver_x[seed], 0.0]])
    receivers = np.zeros((1, 9, 2), dtype=float)
    receivers[0, :, 0] = receiver_x

    result = build_observed_centers_from_eikonal(
        observed,
        tref,
        sources,
        receivers,
        control_time=np.array([[0.70]]),
        dt=dt,
        epsilon_time=0.004,
    )

    assert result.tracking_success.all()
    np.testing.assert_allclose(result.observed_center[0, 0], expected, atol=dt)


def test_same_x_time_for_constant_velocity_horizontal_reflector():
    time, depth = compute_same_x_control_time(
        np.full((3, 5), 2000.0),
        np.array([0.0, 10.0, 20.0]),
        np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
        np.array([[3, 0], [3, 1], [3, 2]]),
        np.array([10.0, 0.0]),
    )
    assert depth == 30.0
    assert time == 2.0 * 30.0 / 2000.0


def test_same_x_time_matches_trapezoid_for_depth_varying_velocity():
    z = np.array([0.0, 10.0, 20.0, 30.0])
    velocity = np.tile(np.array([1000.0, 1500.0, 2000.0, 2500.0]), (2, 1))
    time, _ = compute_same_x_control_time(
        velocity, np.array([0.0, 10.0]), z,
        np.array([[3, 0], [3, 1]]), np.array([0.0, 0.0])
    )
    expected = 2.0 * np.sum(0.5 * (1.0 / velocity[0, :-1] + 1.0 / velocity[0, 1:]) * np.diff(z))
    np.testing.assert_allclose(time, expected)


def test_same_x_time_starts_at_nonzero_source_depth():
    time, _ = compute_same_x_control_time(
        np.full((2, 5), 2000.0), np.array([0.0, 10.0]),
        np.arange(5, dtype=float) * 10.0,
        np.array([[4, 0], [4, 1]]), np.array([10.0, 10.0])
    )
    assert time == 2.0 * 30.0 / 2000.0


def test_bootstrap_uses_supplied_seed_but_keeps_eikonal_moveout():
    dt = 0.002
    time = np.arange(500) * dt
    tref_row = np.array([0.54, 0.51, 0.50, 0.51, 0.54])
    moveout = tref_row - tref_row[2]
    seed_time = 0.68
    expected = seed_time + moveout
    observed = np.stack([_pulse(time, center) for center in expected])[None]
    receivers = np.zeros((1, 5, 2))
    receivers[0, :, 0] = np.arange(5) * 10.0
    result = build_observed_centers_from_eikonal(
        observed, tref_row[None, None], np.array([[20.0, 0.0]]), receivers,
        control_time=np.array([[seed_time]]), dt=dt, epsilon_time=0.004,
    )
    assert result.control_time[0, 0] == seed_time
    np.testing.assert_allclose(result.observed_center[0, 0], expected, atol=dt)


def test_flattening_removes_only_relative_moveout():
    dt = 0.002
    time = np.arange(700) * dt
    moveout = np.array([0.04, 0.01, 0.0, 0.01, 0.04])
    tref = 0.60 + moveout
    observed = np.stack([_pulse(time, center) for center in tref])

    flat, recovered, valid = flatten_observed_gather(
        observed, tref, seed_receiver=2, dt=dt
    )

    assert valid.all()
    np.testing.assert_allclose(recovered, moveout)
    np.testing.assert_allclose(np.argmax(flat, axis=1) * dt, 0.60, atol=dt)


def test_flattening_reports_original_record_time_support():
    dt = 0.1
    observed = np.ones((3, 6))
    tref = np.array([0.2, 0.0, -0.2])
    _, moveout, valid, support = flatten_observed_gather(
        observed, tref, seed_receiver=1, dt=dt, return_support=True,
    )
    assert valid.all()
    np.testing.assert_allclose(moveout, [0.2, 0.0, -0.2])
    np.testing.assert_array_equal(support[0], [True, True, True, True, False, False])
    np.testing.assert_array_equal(support[1], np.ones(6, dtype=bool))
    np.testing.assert_array_equal(support[2], [False, False, True, True, True, True])


def test_flat_tracker_keeps_horizontal_event_and_fixed_seed():
    dt = 0.001
    time = np.arange(500) * dt
    seed = 5
    flat = np.stack([_pulse(time, 0.25, sigma=0.004) for _ in range(11)])
    result = track_flattened_event(
        flat, valid_receiver=np.ones(11, dtype=bool), seed_receiver=seed,
        seed_time=0.25, dt=dt, epsilon_time=0.006,
        slope_penalty=0.2, refine_radius_samples=3,
    )
    assert result.pick_time[seed] == 0.25
    assert result.guide_pick_time[seed] == 0.25
    np.testing.assert_allclose(result.pick_time, 0.25, atol=dt)


def test_flat_tracker_resists_stronger_diverging_event():
    dt = 0.001
    time = np.arange(500) * dt
    seed = 10
    traces = []
    for receiver in range(21):
        distance = abs(receiver - seed)
        target = _pulse(time, 0.25, sigma=0.004)
        competitor = 4.0 * _pulse(time, 0.25 + 0.003 * distance, sigma=0.004)
        traces.append(target if distance == 0 else target + competitor)
    result = track_flattened_event(
        np.stack(traces), valid_receiver=np.ones(21, dtype=bool),
        seed_receiver=seed, seed_time=0.25, dt=dt, epsilon_time=0.006,
        slope_penalty=2.0, refine_radius_samples=3,
    )
    np.testing.assert_allclose(result.pick_time, 0.25, atol=0.002)


def test_flat_tracker_tracks_residual_moveout_and_respects_guide_corridor():
    dt = 0.001
    time = np.arange(500) * dt
    seed = 5
    centers = 0.25 + 0.001 * (np.arange(11) - seed)
    flat = np.stack([_pulse(time, center, sigma=0.004) for center in centers])
    result = track_flattened_event(
        flat, valid_receiver=np.ones(11, dtype=bool), seed_receiver=seed,
        seed_time=centers[seed], dt=dt, epsilon_time=0.003,
        slope_penalty=0.1, refine_radius_samples=2,
    )
    assert np.max(np.abs(np.diff(result.guide_pick_sample))) <= 3
    assert np.max(np.abs(np.diff(result.pick_sample))) <= 3
    assert np.max(np.abs(result.pick_sample - result.guide_pick_sample)) <= 2
    refined_sample = result.pick_time / dt
    assert np.max(np.abs(refined_sample - result.pick_sample)) <= 0.5 + 1e-12
    np.testing.assert_allclose(result.pick_time, centers, atol=0.002)


def test_flat_tracker_stops_at_muted_gap_without_restarting():
    dt = 0.001
    time = np.arange(400) * dt
    flat = np.stack([_pulse(time, 0.2) for _ in range(9)])
    valid = np.ones(9, dtype=bool)
    valid[2] = False
    result = track_flattened_event(
        flat, valid_receiver=valid, seed_receiver=4, seed_time=0.2,
        dt=dt, epsilon_time=0.003, slope_penalty=0.1,
        refine_radius_samples=2,
    )
    assert result.success_mask[3:].all()
    assert not result.success_mask[:3].any()


def test_comparison_plot_is_written(tmp_path):
    dt = 0.002
    time = np.arange(400) * dt
    x = np.arange(5, dtype=float) * 20.0
    tref = np.array([0.54, 0.51, 0.50, 0.51, 0.54])
    tobs = tref + 0.10
    observed = np.stack([_pulse(time, center) for center in tobs])
    output = tmp_path / "comparison.png"

    result = save_tobs_bootstrap_diagnostic(
        output,
        observed_shot=observed,
        theoretical_traveltime=tref,
        old_observed_center=tobs,
        new_observed_center=tobs,
        receiver_x=x,
        control_receiver=2,
        control_time=0.60,
        dt=dt,
    )

    assert result == output
    assert output.stat().st_size > 0


def test_same_x_control_plot_is_written(tmp_path):
    dt = 0.002
    time = np.arange(400) * dt
    x = np.arange(5, dtype=float) * 20.0
    tref = np.tile(np.array([0.54, 0.51, 0.50, 0.51, 0.54]), (4, 1))
    observed = np.stack([_pulse(time, center) for center in tref[0] + 0.10])
    output = tmp_path / "samex.png"
    result = save_same_x_control_test(
        output,
        observed_shot=observed,
        theoretical_traveltime=tref + 0.10,
        new_observed_center=tref + 0.10,
        receiver_x=x,
        control_receiver=2,
        eikonal_control_time=np.full(4, 0.60),
        same_x_control_time=np.full(4, 0.62),
        guide_pick_time=np.full((4, 5), 0.62),
        legacy_pick_time=np.full((4, 5), 0.61),
        dt=dt,
    )
    assert result == output
    assert output.stat().st_size > 0
