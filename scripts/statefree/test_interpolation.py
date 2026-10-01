import numpy as np
import pytest
from interpolation import blend, sample_joints, rates, control_bounds, motion_metrics


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
    assert rates(30, 100, 100) == (100/30, 1)
    with pytest.raises(ValueError): rates(30, 200, 120)
    with pytest.raises(ValueError): rates(30, 20, 20)
    with pytest.raises(ValueError): blend(-.1, 'cubic')
    with pytest.raises(ValueError): blend(np.nan, 'quintic')
    with pytest.raises(ValueError): blend(.5, 'none', 1)


def test_100hz_clock_has_no_cumulative_drift_or_early_observations():
    bounds=np.array([control_bounds(i,30,100) for i in range(300)])
    np.testing.assert_array_equal(bounds[:3],[[0,4],[4,7],[7,10]])
    np.testing.assert_array_equal(bounds[:-1,1],bounds[1:,0])
    assert bounds[-1,1] == 1000
    assert control_bounds(17999,30,100)[1] == 60000
    delay=bounds[:,0]/100-np.arange(300)/30
    assert delay.min() >= -1e-14 and delay.max() < .01
    ticks=np.concatenate([np.arange(start+1,end+1) for start,end in bounds])
    np.testing.assert_array_equal(ticks,np.arange(1,1001))
    np.testing.assert_allclose(np.diff(ticks/100),.01,rtol=0,atol=2e-15)


@pytest.mark.parametrize('method',['none','cubic','quintic'])
def test_100hz_noninteger_segments_reach_each_waypoint_on_a_sent_tick(method):
    goals=np.array([[.1,-.05],[.12,-.07],[.12,-.07],[0.,0.]])
    previous=np.zeros(2);path=[previous]
    for frame,goal in enumerate(goals):
        start,end=control_bounds(frame,30,100)
        for tick in range(start+1,end+1):
            q=sample_joints(previous,goal,(tick-start)/(end-start),method)
            assert np.all(q>=np.minimum(previous,goal)-1e-14)
            assert np.all(q<=np.maximum(previous,goal)+1e-14)
            path.append(q)
        assert np.array_equal(path[end],goal)
        previous=path[end]
    np.testing.assert_array_equal(np.asarray(path)[7:11],np.repeat(goals[1][None],4,axis=0))
