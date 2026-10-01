"""Rest-to-rest joint splines between accepted IK waypoints.

Policy targets remain at the dataset rate. Interpolation samples the already
accepted joint segment; it neither re-applies a relative action nor runs IK
against a succession of different intermediate targets.
"""
from fractions import Fraction
import numpy as np

METHODS = ('none', 'cubic', 'quintic')


def blend(u, method, derivative=0):
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('Segment phase must be finite and within [0, 1]')
    if method not in METHODS or derivative not in (0, 1, 2, 3):
        raise ValueError('Unknown interpolation method or derivative')
    if method == 'none':
        if derivative:
            raise ValueError('A step trajectory has no finite derivative at a waypoint')
        return (u == 1).astype(float)
    coefficients = {'cubic': [0, 0, 3, -2], 'quintic': [0, 0, 0, 10, -15, 6]}[method]
    polynomial = np.polynomial.Polynomial(coefficients).deriv(derivative)
    return polynomial(u)


def sample_joints(start, goal, u, method):
    start, goal = np.asarray(start), np.asarray(goal)
    if start.shape != goal.shape or not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError('Joint endpoints must be finite with matching shapes')
    alpha = float(blend(u, method))
    # Exact endpoint copies keep IK decisions and relative-action origins equal
    # across all comparison modes, including floating-point boundary cases.
    if u == 0:
        return start.copy()
    if u == 1:
        return goal.copy()
    return start + alpha * (goal - start)


def rates(policy_fps, control_fps, video_fps):
    if not np.isfinite([policy_fps,control_fps,video_fps]).all() or min(policy_fps,control_fps,video_fps) <= 0:
        raise ValueError('All rates must be finite and positive')
    if control_fps < policy_fps or control_fps % video_fps:
        raise ValueError('Control rate must cover policy rate and be divisible by video rate')
    return control_fps / policy_fps, int(control_fps // video_fps)


def control_bounds(frame, policy_fps, control_fps):
    """Causal policy boundaries on an independent, uniform control clock.

Ceiling to the next tick adds less than one tick of delay. For 30 -> 100 Hz,
durations are 4, 3, 3 ticks, repeating without accumulated drift. Each actual
sent endpoint becomes the next relative-action origin; no hidden state update
is made at a 33.333 ms timestamp between real control commands.
"""
    if frame < 0 or int(frame) != frame or policy_fps <= 0 or control_fps < policy_fps:
        raise ValueError('Invalid frame or control/policy rate')
    ratio = Fraction(str(control_fps)) / Fraction(str(policy_fps))
    def ceil_tick(index):
        value = index * ratio
        return (value.numerator + value.denominator - 1) // value.denominator
    return ceil_tick(frame), ceil_tick(frame+1)


def motion_metrics(joints, joint_names, dt):
    arms = [i for i, name in enumerate(joint_names) if name.startswith('arm_')]
    q = np.asarray(joints)[:, arms]
    result = {'sampling_hz':1 / dt, 'arm_joint_count':len(arms),
              'definition':'finite differences at the same control rate; includes segment boundaries'}
    for order, key, unit in ((1,'velocity','rad/s'), (2,'acceleration','rad/s^2'), (3,'jerk','rad/s^3')):
        values = np.diff(q, n=order, axis=0) / dt**order
        result[key] = {'max_abs':float(np.abs(values).max()),
                       'rms':float(np.sqrt(np.mean(values**2))), 'unit':unit}
    result['max_arm_increment_rad'] = float(np.abs(np.diff(q, axis=0)).max())
    return result
