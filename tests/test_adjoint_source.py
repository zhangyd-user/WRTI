from types import SimpleNamespace

import numpy as np

from WRTI.adjoint import build_wrti_adjoint_source


def _inputs(observed, synthetic, shifts, left, right, **overrides):
    shifts = np.asarray(shifts, dtype=float)
    shape = shifts.shape
    result = SimpleNamespace(
        shift_time=shifts,
        fixed_mask=overrides.get("fixed", np.ones(shape, dtype=bool)),
        tracking_success=overrides.get("success", np.ones(shape, dtype=bool)),
        path_failure_mask=overrides.get(
            "path_failure", np.zeros(shape, dtype=bool)
        ),
        tracked_correlation=overrides.get(
            "correlation", np.ones(shape, dtype=float)
        ),
    )
    windows = SimpleNamespace(
        left_sample=np.broadcast_to(np.asarray(left, dtype=int), shape),
        right_sample=np.broadcast_to(np.asarray(right, dtype=int), shape),
        valid=overrides.get("window_valid", np.ones(shape, dtype=bool)),
    )
    return np.asarray(observed)[None, None, :], np.asarray(synthetic)[None, None, :], result, SimpleNamespace(windows=windows)


def _reference(observed, synthetic, tau, left, right, dt, regularization=0.10):
    first = np.zeros_like(observed, dtype=float)
    second = np.zeros_like(observed, dtype=float)
    first[1:-1] = (observed[2:] - observed[:-2]) / (2.0 * dt)
    second[1:-1] = (
        observed[2:] - 2.0 * observed[1:-1] + observed[:-2]
    ) / dt**2
    indices = np.arange(left, right + 1)
    shifted = indices + tau / dt
    numerator = np.interp(shifted, np.arange(observed.size), first)
    shifted_second = np.interp(shifted, np.arange(observed.size), second)
    denominator = np.sum(shifted_second * synthetic[indices]) * dt
    scale = np.sum(np.abs(shifted_second * synthetic[indices])) * dt
    expected = np.zeros(observed.size)
    expected[indices] = -tau * numerator * denominator / (
        denominator**2 + (regularization * scale) ** 2
    )
    return expected, denominator


def test_eq4_uses_tikhonov_stabilized_reciprocal_and_negative_sign():
    dt = 0.002
    time = np.arange(40) * dt
    observed = np.sin(35.0 * time) + 0.15 * np.sin(71.0 * time)
    synthetic = np.cos(19.0 * time) + 0.3
    tau, left, right = 0.7 * dt, 5, 29
    args = _inputs(observed, synthetic, [[[tau]]], left, right)

    actual = build_wrti_adjoint_source(*args, dt)[0, 0]
    expected, denominator = _reference(
        observed, synthetic, tau, left, right, dt
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert denominator != 0.0
    assert not np.allclose(actual, -expected)


def test_uses_observed_derivatives_not_synthetic_derivatives():
    dt = 0.001
    time = np.arange(50) * dt
    observed = np.sin(83.0 * time) + 0.2 * np.cos(37.0 * time)
    synthetic = 1.0 + 0.02 * np.arange(time.size)
    tau, left, right = 0.4 * dt, 7, 38
    args = _inputs(observed, synthetic, [[[tau]]], left, right)

    actual = build_wrti_adjoint_source(*args, dt)[0, 0]
    expected, _ = _reference(observed, synthetic, tau, left, right, dt)

    np.testing.assert_allclose(actual, expected)
    synthetic_first = np.zeros_like(synthetic)
    synthetic_first[1:-1] = (synthetic[2:] - synthetic[:-2]) / (2.0 * dt)
    assert not np.allclose(actual[left : right + 1], synthetic_first[left : right + 1])


def test_subsample_shift_is_interpolated_not_rounded():
    dt = 0.001
    time = np.arange(60) * dt
    observed = np.sin(121.0 * time) + 0.1 * np.sin(49.0 * time)
    synthetic = np.cos(67.0 * time) + 0.4
    tau, left, right = 0.0134, 16, 38
    args = _inputs(observed, synthetic, [[[tau]]], left, right)

    actual = build_wrti_adjoint_source(*args, dt)[0, 0]
    expected, _ = _reference(observed, synthetic, tau, left, right, dt)
    rounded, _ = _reference(observed, synthetic, round(tau / dt) * dt, left, right, dt)

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, rounded)


def test_overlapping_reflector_windows_sum():
    dt = 0.001
    time = np.arange(70) * dt
    observed = np.sin(91.0 * time) + 0.3 * np.cos(33.0 * time)
    synthetic = np.cos(52.0 * time) + 0.5
    shifts = np.array([[[0.6 * dt]], [[1.3 * dt]]])
    left = np.array([[[10]], [[22]]])
    right = np.array([[[34]], [[48]]])
    combined_args = _inputs(observed, synthetic, shifts, left, right)
    combined = build_wrti_adjoint_source(*combined_args, dt)

    first = build_wrti_adjoint_source(
        *_inputs(observed, synthetic, shifts[:1], left[:1], right[:1]), dt
    )
    second = build_wrti_adjoint_source(
        *_inputs(observed, synthetic, shifts[1:], left[1:], right[1:]), dt
    )

    np.testing.assert_allclose(combined, first + second)
    assert np.any(first[..., 22:35] * second[..., 22:35] != 0.0)


def test_path_failure_and_nonfinite_shift_produce_zero():
    observed = np.sin(np.arange(30) * 0.2)
    synthetic = np.cos(np.arange(30) * 0.1) + 0.5
    args = _inputs(
        observed,
        synthetic,
        [[[np.nan]]],
        5,
        20,
        success=np.array([[[False]]]),
        path_failure=np.array([[[True]]]),
    )

    source, diagnostics = build_wrti_adjoint_source(
        *args, 0.001, return_diagnostics=True
    )

    assert np.array_equal(source, np.zeros_like(source))
    assert np.isfinite(source).all()
    assert diagnostics["path_failures_skipped"] == 1


def test_path_failure_with_finite_fallback_shift_is_used():
    dt = 0.001
    time = np.arange(40) * dt
    observed = np.sin(100.0 * time)
    synthetic = np.cos(61.0 * time) + 0.2
    args = _inputs(
        observed,
        synthetic,
        [[[0.5 * dt]]],
        5,
        30,
        success=np.array([[[True]]]),
        path_failure=np.array([[[True]]]),
    )

    source, diagnostics = build_wrti_adjoint_source(
        *args, dt, return_diagnostics=True
    )

    assert np.any(source)
    assert diagnostics["fallbacks_used"] == 1
    assert diagnostics["path_failures_skipped"] == 0


def test_low_correlation_valid_path_is_not_excluded():
    dt = 0.001
    time = np.arange(40) * dt
    observed = np.sin(100.0 * time)
    synthetic = np.cos(61.0 * time) + 0.2
    args = _inputs(
        observed,
        synthetic,
        [[[0.5 * dt]]],
        5,
        30,
        correlation=np.array([[[-0.95]]]),
    )

    source, diagnostics = build_wrti_adjoint_source(
        *args, dt, return_diagnostics=True
    )

    assert np.any(source)
    assert diagnostics["active_reflector_traces"] == 1


def test_zero_denominator_has_zero_contribution_without_skipping_the_trace():
    dt = 0.001
    observed = np.sin(np.arange(30) * 0.4)
    second = np.zeros_like(observed)
    second[1:-1] = observed[2:] - 2.0 * observed[1:-1] + observed[:-2]
    synthetic = np.zeros_like(observed)
    synthetic[10] = 1.0 / second[10]
    synthetic[11] = -1.0 / second[11]
    args = _inputs(observed, synthetic, [[[0.0]]], 5, 20)

    source, diagnostics = build_wrti_adjoint_source(
        *args, dt, return_diagnostics=True
    )

    assert not np.any(source)
    assert diagnostics["active_reflector_traces"] == 1
    assert diagnostics["denominator_failures"] == 0
    assert diagnostics["regularized_denominators"] == 1
    assert diagnostics["min_cancellation_ratio"] < 1e-12
    assert diagnostics["nonfinite_output"] == 0


def test_reflector_weights_match_the_weighted_objective():
    dt = 0.001
    time = np.arange(50) * dt
    observed = np.sin(83.0 * time) + 0.2 * np.cos(37.0 * time)
    synthetic = 1.0 + 0.02 * np.arange(time.size)
    shifts = np.array([[[0.4 * dt]], [[0.4 * dt]]])
    left = np.array([[[7]], [[7]]])
    right = np.array([[[38]], [[38]]])
    args = _inputs(observed, synthetic, shifts, left, right)

    weighted = build_wrti_adjoint_source(*args, dt, reflector_weights=[1.0, 3.0])
    unweighted = build_wrti_adjoint_source(*args, dt)

    np.testing.assert_allclose(weighted, 2.0 * unweighted)


def test_regularization_must_be_positive_and_finite():
    args = _inputs(np.arange(30.0), np.ones(30), [[[0.25e-3]]], 5, 20)

    for value in (0.0, -0.1, np.inf, np.nan):
        try:
            build_wrti_adjoint_source(*args, 0.001, denominator_regularization=value)
        except ValueError as error:
            assert "denominator_regularization" in str(error)
        else:
            raise AssertionError("invalid regularization was accepted")


def test_shifted_window_out_of_bounds_skips_whole_reflector_trace():
    dt = 0.001
    time = np.arange(30) * dt
    args = _inputs(
        np.sin(90.0 * time),
        np.cos(50.0 * time) + 0.2,
        [[[8.0 * dt]]],
        10,
        25,
    )

    source, diagnostics = build_wrti_adjoint_source(
        *args, dt, return_diagnostics=True
    )

    assert not np.any(source)
    assert diagnostics["out_of_bounds"] == 1
