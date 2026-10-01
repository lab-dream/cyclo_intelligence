import numpy as np
import pytest
from interpolation import blend, sample_joints, rates, motion_metrics


@pytest.mark.parametrize('method', ['cubic', 'quintic'])
def test_spline_boundary_conditions_and_no_overshoot(method):
    u = np.linspace(0, 1, 1001)
    s = blend(u, method)
    assert s[0] == 0 and s[-1] == 1
    assert np.all(np.diff(s) >= 0)
    assert np.all((s >= 0) & (s <= 1))
    np.testing.assert_allclose(blend([0, 1], method, 1), 0, atol=1e-14)
    if method == 'quintic':
        np.testing.assert_allclose(blend([0, 1], method, 2), 0, atol=1e-14)
    else:
        np.testing.assert_allclose(blend([0, 1], method, 2), [6, -6])


@pytest.mark.parametrize('method', ['none', 'cubic', 'quintic'])
def test_waypoints_are_exact_and_failed_ik_remains_a_hold(method):
    start = np.array([.3, -.4, .8]); goal = np.array([.38, -.47, .77])
    assert np.array_equal(sample_joints(start, goal, 0, method), start)
    assert np.array_equal(sample_joints(start, goal, 1, method), goal)
    for u in np.linspace(0, 1, 25):
        q = sample_joints(start, goal, u, method)
        assert np.all(q >= np.minimum(start, goal)) and np.all(q <= np.maximum(start, goal))
        assert np.array_equal(sample_joints(start, start, u, method), start)


def test_both_splines_reduce_sampled_jerk_relative_to_steps():
    endpoints = [.0, .05, .10, .02, .02, -.02]
    results = {}
    for mode in ['none', 'cubic', 'quintic']:
        path = [[endpoints[0]]]
        for a, b in zip(endpoints[:-1], endpoints[1:]):
            path.extend(sample_joints([a], [b], u, mode) for u in np.arange(1, 9) / 8)
        results[mode] = motion_metrics(path, ['arm_l_joint1'], 1/240)
    # C2 continuity does not imply lower finite-difference jerk than cubic at
    # every sample rate or waypoint sequence. Compare each against the step.
    for mode in ['cubic', 'quintic']:
        assert results[mode]['jerk']['rms'] < results['none']['jerk']['rms']


def test_invalid_rates_and_phases_rejected():
    assert rates(30, 240, 120) == (8, 2)
    with pytest.raises(ValueError): rates(30, 200, 120)
    with pytest.raises(ValueError): blend(-.1, 'cubic')
    with pytest.raises(ValueError): blend(np.nan, 'quintic')
    with pytest.raises(ValueError): blend(.5, 'none', 1)
