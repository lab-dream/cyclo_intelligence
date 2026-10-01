"""Rest-to-rest joint splines between accepted IK waypoints.

Policy targets remain at the dataset rate. Interpolation samples the already
accepted joint segment; it neither re-applies a relative action nor runs IK
against a succession of different intermediate targets.
"""
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
    if min(policy_fps, control_fps, video_fps) <= 0:
        raise ValueError('All rates must be positive')
    ratio = control_fps / policy_fps
    if not np.isclose(ratio, round(ratio)) or control_fps % video_fps:
        raise ValueError('Control rate must be divisible by policy and video rates')
    return int(round(ratio)), int(control_fps // video_fps)


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
