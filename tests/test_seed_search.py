import numpy as np

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
