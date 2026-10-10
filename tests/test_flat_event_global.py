from dataclasses import dataclass

import numpy as np

from WRTI.tracking import flat_event_global as module

solve_global_topk_two_pass = module.solve_global_topk_two_pass
prepare_global_candidate_pool = module.prepare_global_candidate_pool
compare_tracking_hypotheses = module.compare_tracking_hypotheses


@dataclass(frozen=True)
class Candidate:
    sample: int
    time: float
    amplitude: float
    envelope: float
    waveform: np.ndarray
    prominence: float = 0.0


def _unit(values):
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    return values / np.linalg.norm(values)


def _candidate(time, waveform, dt=0.001, amplitude=1.0, prominence=1.0):
    return Candidate(
        sample=int(round(time / dt)),
        time=float(time),
        amplitude=float(amplitude),
        envelope=abs(float(amplitude)),
        waveform=np.asarray(waveform, dtype=float),
        prominence=float(prominence),
    )


def test_template_identity_beats_locally_coherent_diffraction():
    nr = 13
    x = np.arange(nr, dtype=float) * 25.0
    valid = np.ones(nr, dtype=bool)
    template = _unit([0.0, 1.0, 0.1, -0.8, -0.1])
    diffraction = _unit([0.0, 0.2, 1.0, 0.2, -0.6])

    candidates = []
    for receiver in range(nr):
        true = _candidate(1.000 + 0.0003 * receiver, template, amplitude=0.45)
        if 5 <= receiver <= 9:
            wrong = _candidate(
                1.010 + 0.004 * (receiver - 5), diffraction, amplitude=2.0
            )
            candidates.append([true, wrong])
        else:
            candidates.append([true])

    anchor = np.zeros(nr, dtype=bool)
    anchor[2:5] = True
    state_valid = np.ones((nr, 2000), dtype=bool)
    result = solve_global_topk_two_pass(
        candidates,
        receiver_x=x,
        valid_receiver=valid,
        seed_receiver=3,
        anchor_mask=anchor,
        state_valid=state_valid,
        escape_state_valid=state_valid,
        config={
            "enabled": True,
            "top_k": 8,
            "template_weight": 1.2,
            "neighbor_weight": 0.45,
            "curvature_weight": 0.25,
            "second_pass_enabled": True,
        },
    )
    for receiver in range(5, 10):
        assert result.selected[receiver] == 0


def test_global_dp_can_skip_one_bad_valid_receiver_and_rejoin_same_event():
    nr = 11
    x = np.arange(nr, dtype=float) * 20.0
    valid = np.ones(nr, dtype=bool)
    template = _unit([0.0, 1.0, 0.0, -1.0, 0.0])
    wrong = _unit([1.0, -0.2, -0.8, 0.2, 0.8])

    candidates = [[_candidate(1.0, template)] for _ in range(nr)]
    candidates[7] = [_candidate(1.040, wrong, amplitude=2.0)]
    anchor = np.zeros(nr, dtype=bool)
    anchor[3:6] = True
    state_valid = np.ones((nr, 2000), dtype=bool)

    result = solve_global_topk_two_pass(
        candidates,
        receiver_x=x,
        valid_receiver=valid,
        seed_receiver=4,
        anchor_mask=anchor,
        state_valid=state_valid,
        escape_state_valid=state_valid,
        config={
            "enabled": True,
            "top_k": 6,
            "max_gap_receivers": 2,
            "valid_gap_penalty": 0.35,
            "template_weight": 1.2,
            "neighbor_weight": 0.4,
            "second_pass_enabled": False,
        },
    )
    assert 7 not in result.selected
    assert 8 in result.selected
    assert result.skipped_valid_receiver_mask[7]


def test_escape_candidate_is_legal_but_penalized_against_base_candidate():
    nr = 9
    nt = 1500
    x = np.arange(nr, dtype=float) * 25.0
    valid = np.ones(nr, dtype=bool)
    template = _unit([0.0, 1.0, 0.2, -0.9, -0.1])

    candidates = [[_candidate(0.800, template)] for _ in range(nr)]
    candidates[7] = [_candidate(0.800, template), _candidate(0.806, template)]
    base = np.zeros((nr, nt), dtype=bool)
    escape = np.zeros((nr, nt), dtype=bool)
    base[:, 760:830] = True
    escape[:, 740:850] = True
    base[7, candidates[7][1].sample] = False

    anchor = np.zeros(nr, dtype=bool)
    anchor[2:5] = True
    result = solve_global_topk_two_pass(
        candidates,
        receiver_x=x,
        valid_receiver=valid,
        seed_receiver=3,
        anchor_mask=anchor,
        state_valid=base,
        escape_state_valid=escape,
        config={
            "enabled": True,
            "top_k": 5,
            "escape_penalty": 0.8,
            "second_pass_enabled": False,
        },
    )
    assert result.selected[7] == 0
    assert not result.ownership_escape_used_mask[7]


def test_template_similarity_tolerates_small_sample_lag():
    a = _unit([0, 0, 1, 2, 0, -2, -1, 0, 0])
    b = np.roll(a, 2)
    strict = module._lagged_corr(a, b, 0)
    tolerant = module._lagged_corr(a, b, 2)
    assert tolerant > 0.99
    assert tolerant > strict + 0.15


def test_consensus_uses_majority_cluster_not_best_path_center():
    true = _unit([0, 1, 0, -1, 0])
    candidates = [[_candidate(1.000, true), _candidate(1.050, true)]]
    paths = (
        module.GlobalDPPath(((0, 1),), 10.00, 0, 0, 0, 0, 0, 0, 10),
        module.GlobalDPPath(((0, 0),), 9.98, 0, 0, 0, 0, 0, 0, 10),
        module.GlobalDPPath(((0, 0),), 9.97, 0, 0, 0, 0, 0, 0, 10),
        module.GlobalDPPath(((0, 0),), 9.96, 0, 0, 0, 0, 0, 0, 10),
    )
    fraction, spread, center = module._consensus(
        paths, candidates, nr=1, radius=0.008, temperature=0.05
    )
    assert abs(center[0] - 1.000) < 1e-9
    assert fraction[0] > 0.70
    assert spread[0] < 1e-9


def _add_wavelet(trace, center, wavelet):
    half = len(wavelet) // 2
    trace[center - half : center + half + 1] += wavelet


def test_bounded_candidate_pool_retains_weak_template_peak():
    nr, nt = 7, 500
    dt = 0.001
    x = np.arange(nr, dtype=float) * 20.0
    valid = np.ones(nr, dtype=bool)
    traces = np.zeros((nr, nt), dtype=float)
    target_wavelet = np.asarray([-0.3, 0.2, 1.0, 0.2, -0.3])
    wrong_wavelet = np.asarray([-0.1, 0.5, 1.8, 0.5, -0.1])
    target_sample = 250
    for r in range(nr):
        _add_wavelet(traces[r], target_sample, 0.35 * target_wavelet)
        # Many stronger positive peaks to stress the cap.
        for j, sample in enumerate(range(100, 420, 12)):
            if abs(sample - target_sample) < 6:
                continue
            _add_wavelet(traces[r], sample, (0.8 + 0.02 * j) * wrong_wavelet)
    scale = np.sqrt(np.mean(traces * traces, axis=1))
    traces = traces / scale[:, None]
    envelope = np.abs(traces)
    half = 2
    target_wf = module._candidate_waveform(traces[3], target_sample, half)
    candidates = [[] for _ in range(nr)]
    # Mimic the legacy find_peaks failure we care about: outside the anchor,
    # only a stronger competing peak survived the original prominence/distance
    # pruning. The weak target exists in the trace but is absent from candidates.
    wrong_sample = 304
    wrong_wf = module._candidate_waveform(traces[0], wrong_sample, half)
    for r in range(nr):
        candidates[r] = [_candidate(wrong_sample * dt, wrong_wf, dt=dt, amplitude=1.5)]
    anchor = np.zeros(nr, dtype=bool)
    anchor[2:5] = True
    for r in np.flatnonzero(anchor):
        candidates[int(r)] = [_candidate(target_sample * dt, target_wf, dt=dt, amplitude=0.2)]
    state_valid = np.ones((nr, nt), dtype=bool)

    counts = prepare_global_candidate_pool(
        candidates,
        traces=traces,
        candidate_envelope=envelope,
        receiver_x=x,
        valid_receiver=valid,
        seed_receiver=3,
        anchor_mask=anchor,
        state_valid=state_valid,
        escape_state_valid=state_valid,
        dt=dt,
        t0=0.0,
        half_window_samples=half,
        config={
            "enabled": True,
            "candidate_max_per_trace": 8,
            "candidate_keep_regular": 2,
            "candidate_keep_template": 4,
            "candidate_keep_axis": 4,
            "candidate_keep_amplitude": 2,
        },
    )
    assert int(np.max(counts[~anchor])) <= 8
    for r in (0, 1, 5, 6):
        assert target_sample in {int(c.sample) for c in candidates[r]}


def test_repair_challenger_does_not_replace_better_identity_path():
    nr, nt = 9, 500
    dt = 0.001
    x = np.arange(nr, dtype=float) * 20.0
    valid = np.ones(nr, dtype=bool)
    flat = np.zeros((nr, nt), dtype=float)
    target = np.asarray([-0.3, 0.1, 1.0, 0.1, -0.3])
    wrong = np.asarray([0.4, -0.2, 1.0, -0.2, 0.4])
    target_sample = 250
    wrong_sample = 280
    for r in range(nr):
        _add_wavelet(flat[r], target_sample, target)
        _add_wavelet(flat[r], wrong_sample, wrong)

    anchor = np.zeros(nr, dtype=bool)
    anchor[3:6] = True
    anchor_samples = np.full(nr, -1, dtype=int)
    anchor_samples[anchor] = target_sample

    incumbent_pick = np.full(nr, target_sample, dtype=int)
    incumbent_success = np.ones(nr, dtype=bool)
    challenger_pick = np.full(nr, wrong_sample, dtype=int)
    challenger_pick[anchor] = target_sample
    challenger_success = np.ones(nr, dtype=bool)

    decision = compare_tracking_hypotheses(
        flat_gather=flat,
        receiver_x=x,
        valid_receiver=valid,
        incumbent_pick_sample=incumbent_pick,
        incumbent_success_mask=incumbent_success,
        challenger_pick_sample=challenger_pick,
        challenger_success_mask=challenger_success,
        anchor_receiver_mask=anchor,
        anchor_pick_sample=anchor_samples,
        coherence_half_window_time=0.002,
        dt=dt,
        config={"enabled": True, "repair_challenge_min_improvement": 0.02},
    )
    assert not decision.use_challenger
    assert decision.incumbent.template_similarity > decision.challenger.template_similarity
