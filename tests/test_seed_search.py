import numpy as np

from WRTI.tracking import seed_search
from WRTI.tracking.seed_search import (
    discover_boundary_seed_candidates,
    discover_tracking_seed,
)


def _event(time, center, width=0.004):
    return np.exp(-0.5 * ((time - center) / width) ** 2)


def _discover(flat, control, **overrides):
    nr, nt = flat.shape
    options = dict(
        receiver_x=np.arange(nr) * 10.0,
        valid_receiver=np.ones(nr, dtype=bool), seed_receiver=nr // 2,
        control_time=control, state_valid=np.ones((nr, nt), dtype=bool),
        dt=0.001, local_half_width_time=0.030,
        tracker_options={"candidate_min_prominence": 0.01,
                         "coherence_half_window_time": 0.010,
                         "max_prediction_error_time": 0.050},
    )
    options.update(overrides)
    return discover_tracking_seed(flat, **options)


def _pilot_result(times, **overrides):
    times = np.asarray(times, dtype=float)
    size = times.size
    values = dict(
        passed=True,
        coverage=1.0,
        median_zncc=0.9,
        median_abs_prediction_error=0.0,
        median_abs_slope_ms_per_100m=0.0,
        p90_abs_slope_ms_per_100m=0.0,
        phase_switch_count=0,
        pick_sample=np.rint(times * 1000).astype(int),
        pick_time=times,
        success_mask=np.ones(size, dtype=bool),
        neighbor_zncc=np.full(size, 0.9),
        residual_slope=np.zeros(size),
        prediction_error=np.zeros(size),
        tested_mask=np.ones(size, dtype=bool),
        left_support=size // 2,
        right_support=size // 2,
        transition_count=0,
    )
    values.update(overrides)
    return seed_search.SeedPilotResult(**values)


def _short_evaluation(pilot, *, mode="LOCAL", seed_time=2.4):
    size = pilot.pick_time.size
    return seed_search._ShortSeedEvaluation(
        mode=mode,
        hypothesis_index=0,
        seed_receiver=size // 2,
        hypothesis=(seed_time, 0, 1.0, 1.0, 1.0),
        pilot=pilot,
        anchor_mask=np.ones(size, dtype=bool),
        anchor_left=size // 2,
        anchor_right=size - size // 2 - 1,
        diffraction_risk=False,
        control_divergence_risk=False,
        left_signed_slope_ms_per_100m=0.0,
        right_signed_slope_ms_per_100m=0.0,
        macro_curvature_time=0.0,
    )


def test_control_divergence_detects_diffraction_flank_without_apex():
    pilot = _pilot_result([2.40, 2.41, 2.43, 2.46, 2.50, 2.56])
    assert seed_search._control_divergence_risk(pilot, 0, 2.40)


def test_seed_rank_control_class_beats_stronger_lower_diffraction():
    correct = _short_evaluation(_pilot_result(
        [2.420] * 7,
        median_zncc=0.85,
        median_signed_control_residual=0.020,
        robust_linear_slope_ms_per_100m=1.0,
        p90_abs_flatness_error=0.002,
    ))
    diffraction = _short_evaluation(_pilot_result(
        [2.455] * 7,
        median_zncc=0.99,
        median_signed_control_residual=0.055,
        robust_linear_slope_ms_per_100m=0.5,
        p90_abs_flatness_error=0.001,
    ), seed_time=2.455)
    rank_options = dict(
        base_seed=3,
        control_time=2.40,
        slope_quantum=1.0,
        time_quantum=0.001,
    )
    assert seed_search._control_class(correct.pilot) == 3
    assert seed_search._control_class(diffraction.pilot) == 1
    assert seed_search._short_seed_rank(
        correct, **rank_options
    ) > seed_search._short_seed_rank(
        diffraction, **rank_options
    )


def test_same_control_class_prefers_stable_event_over_exact_distance():
    closer = _short_evaluation(_pilot_result(
        [2.405] * 7,
        median_signed_control_residual=0.005,
        robust_linear_slope_ms_per_100m=8.0,
        p90_abs_flatness_error=0.002,
    ))
    stable = _short_evaluation(_pilot_result(
        [2.420] * 7,
        median_signed_control_residual=0.020,
        robust_linear_slope_ms_per_100m=1.0,
        p90_abs_flatness_error=0.002,
    ))
    options = dict(
        base_seed=3, control_time=2.4, slope_quantum=1.0, time_quantum=0.001
    )
    assert seed_search._control_class(closer.pilot) == 3
    assert seed_search._control_class(stable.pilot) == 3
    assert seed_search._short_seed_rank(
        stable, **options
    ) > seed_search._short_seed_rank(closer, **options)


def test_same_quality_and_control_class_prefers_earlier_event():
    earlier = _short_evaluation(_pilot_result(
        [2.385] * 7,
        median_signed_control_residual=-0.015,
        robust_linear_slope_ms_per_100m=1.0,
        p90_abs_flatness_error=0.002,
    ))
    later = _short_evaluation(_pilot_result(
        [2.415] * 7,
        median_signed_control_residual=0.015,
        robust_linear_slope_ms_per_100m=1.0,
        p90_abs_flatness_error=0.002,
    ))
    options = dict(
        base_seed=3, control_time=2.4, slope_quantum=1.0, time_quantum=0.001
    )
    assert seed_search._short_seed_rank(
        earlier, **options
    ) > seed_search._short_seed_rank(later, **options)


def test_production_seed_avoids_control_crossing_corner():
    pilot = _pilot_result([3.060, 3.061, 3.062, 3.090, 3.063, 3.064, 3.065])
    item = _short_evaluation(pilot)
    receiver, _, _ = seed_search._production_seed_from_short(
        item, control_time=3.090
    )
    assert receiver != 3


def test_local_search_snaps_to_positive_main_peak_not_stronger_negative_lobe():
    time = np.arange(400) * 0.001
    flat = np.stack([_event(time, 0.220) for _ in range(25)])
    flat[12] -= 3.0 * _event(time, 0.205, 0.002)
    result = _discover(flat, 0.200)
    assert result.valid and result.mode == "LOCAL"
    assert np.isclose(result.seed_time, 0.220)
    assert result.positive_amplitude > 0


def test_adaptive_runs_only_when_local_pilot_has_no_valid_hypothesis():
    time = np.arange(500) * 0.001
    flat = np.stack([_event(time, 0.300) for _ in range(25)])
    result = _discover(flat, 0.200)
    assert result.valid and result.mode == "ADAPTIVE"
    assert result.local_hypothesis_count == 0
    assert np.isclose(result.seed_time, 0.300)


def test_failed_pilots_return_invalid_seed_instead_of_control_fallback():
    time = np.arange(400) * 0.001
    flat = np.zeros((25, time.size))
    flat[12] = _event(time, 0.220)
    result = _discover(flat, 0.200)
    assert not result.valid and result.mode == "FAILED"
    assert np.isnan(result.seed_time)


def test_pilot_rejects_seed_waveform_outside_record(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("same-x seed waveform window is outside the record")

    monkeypatch.setattr(seed_search, "track_flattened_event_sparse", reject)
    flat = np.zeros((3, 20))
    result = seed_search._pilot(
        flat, np.arange(3.0), np.ones(3, bool), np.ones_like(flat, bool),
        1, (0.001,), {}, np.arange(3), 1, 0.8, 0.7, 0.0, 0.015,
    )
    assert result is None


def test_ownership_wide_finds_positive_event_beyond_adaptive_window():
    time = np.arange(600) * 0.001
    flat = np.stack([_event(time, 0.380) for _ in range(25)])
    result = _discover(flat, 0.200)
    assert result.valid
    assert result.mode == "OWNERSHIP_WIDE"
    assert np.isclose(result.seed_time, 0.380)


def test_repair_excludes_failed_seed_and_searches_full_ownership():
    time = np.arange(600) * 0.001
    flat = np.stack([
        _event(time, 0.220) + 0.8 * _event(time, 0.380)
        for _ in range(25)
    ])
    result = _discover(
        flat,
        0.220,
        repair_exhaustive=True,
        excluded_seed_times=[0.220],
    )
    assert result.valid
    assert result.mode == "OWNERSHIP_WIDE"
    assert np.isclose(result.seed_time, 0.380)


def test_reliable_short_anchor_survives_poor_extended_coverage():
    time = np.arange(500) * 0.001
    flat = np.zeros((101, time.size))
    flat[38:63] = np.stack([_event(time, 0.220) for _ in range(25)])
    result = _discover(flat, 0.200, pilot_long_aperture_m=800.0)
    assert result.valid
    assert result.anchor_left_count >= 3 and result.anchor_right_count >= 3


def test_boundary_seed_search_uses_full_ownership_not_reference_band():
    time = np.arange(600) * 0.001
    flat = np.stack([_event(time, 0.380) for _ in range(40)])
    candidates = discover_boundary_seed_candidates(
        flat, receiver_x=np.arange(40) * 20.0,
        valid_receiver=np.ones(40, bool), seed_receiver=0,
        inward_direction=1, state_valid=np.ones_like(flat, bool),
        dt=0.001, reference_time=0.100,
        tracker_options={"candidate_min_prominence": 0.01},
    )
    selected = [item for item in candidates if item.selected]
    assert selected and np.isclose(selected[0].seed_time, 0.380)


def test_boundary_seed_search_prefers_exclusive_ownership():
    time = np.arange(500) * 0.001
    flat = np.stack([
        _event(time, 0.220) + 2.0 * _event(time, 0.320)
        for _ in range(40)
    ])
    neighbor = np.zeros_like(flat, bool)
    neighbor[:, 300:341] = True
    candidates = discover_boundary_seed_candidates(
        flat, receiver_x=np.arange(40) * 20.0,
        valid_receiver=np.ones(40, bool), seed_receiver=0,
        inward_direction=1, state_valid=np.ones_like(flat, bool),
        lower_neighbor_state_valid=neighbor, dt=0.001,
        tracker_options={"candidate_min_prominence": 0.01},
    )
    selected = [item for item in candidates if item.selected]
    assert selected
    assert all(np.isclose(item.seed_time, 0.220) for item in selected)
    assert not selected[0].overlap_fallback


def test_boundary_seed_ranking_prefers_long_continuity_over_short_energy():
    time = np.arange(500) * 0.001
    flat = np.stack([
        _event(time, 0.220) + (3.0 * _event(time, 0.320) if receiver < 25 else 0.0)
        for receiver in range(60)
    ])
    candidates = discover_boundary_seed_candidates(
        flat, receiver_x=np.arange(60) * 20.0,
        valid_receiver=np.ones(60, bool), seed_receiver=0,
        inward_direction=1, state_valid=np.ones_like(flat, bool), dt=0.001,
        pilot_long_aperture_m=1000.0,
        tracker_options={"candidate_min_prominence": 0.01},
    )
    selected = sorted(
        (item for item in candidates if item.selected),
        key=lambda item: item.rank,
        reverse=True,
    )
    assert selected
    assert np.isclose(selected[0].seed_time, 0.220)
