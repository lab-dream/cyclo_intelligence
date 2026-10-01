"""Named-joint rank audit and real demonstration checks before ERJ training."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from core import Robot, checked_action_spec, read_json, transform, write_json


def rank_audit(robot, references, selected_joints):
    samples = {f'arm_{s}_joint{i}': [] for s in ('l', 'r') for i in range(1, 8)}
    source_frames = []
    for ref in references:
        names = ref['names'].tolist()
        for frame in np.linspace(0, len(ref['state'])-1, min(30, len(ref['state'])), dtype=int):
            for kind in ('state', 'joint_action'):
                robot.set_joints(ref[kind][frame], names)
                source_frames.append({'source_episode': int(ref['source_episode']), 'frame': int(frame), 'kind': kind})
                for side, pose in zip(('l', 'r'), robot.eef()):
                    dids = [robot.model.joint(f'arm_{side}_joint{i}').dofadr[0] for i in range(1, 8)]
                    jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
                    robot.mj.mj_jac(robot.model, robot.data, jp, jr, pose[:3, 3], robot.model.body(f'arm_{side}_link7').id)
                    jac = np.vstack((jp[:, dids], .3*jr[:, dids]))
                    for index in range(7):
                        samples[f'arm_{side}_joint{index+1}'].append(np.linalg.svd(np.delete(jac, index, axis=1), compute_uv=False))
    results = {}
    for name, spectra in samples.items():
        spectra = np.asarray(spectra); ranks = np.sum(spectra > 1e-6, axis=1)
        joint = robot.model.joint(name)
        results[name] = {'samples': len(spectra), 'rank6': int(np.sum(ranks == 6)), 'minimum_rank': int(ranks.min()),
                         'sigma_min_min': float(spectra[:, -1].min()), 'sigma_min_median': float(np.median(spectra[:, -1])),
                         'local_axis': joint.axis.tolist(), 'limits_rad': joint.range.tolist()}
    if any(results[name]['minimum_rank'] < 6 for name in selected_joints):
        raise ValueError('Selected ERJ joint loses EEF rank in sampled poses; inspect rank audit')
    return {'selected_joints': selected_joints, 'candidates': results, 'sample_frames': source_frames,
            'jacobian': '6x6 after removing selected column; rotation rows weighted by 0.3 as in DLS IK',
            'rank_absolute_tolerance': 1e-6,
            'selection_reason': 'Default first joint of each arm follows ERJ joint-first control and retains rank 6 in actual FFW samples; robot base and lift are not selected',
            'scope': 'Sampled local rank is not a guarantee of global reachability or absence of singularities'}


def validate(dataset, scene, output):
    dataset = Path(dataset)
    meta = read_json(dataset/'meta/action_representation.json'); spec = checked_action_spec(meta)
    if spec['action_mode'] != 'erj':
        raise ValueError('ERJ validation requires an ERJ dataset')
    references = [np.load(path) for path in sorted((dataset/'reference').glob('*.npz'))]
    robot = Robot(scene); audit = rank_audit(robot, references, spec['erj_joints'])
    checks = []; nearby_checks = []; labels = []; different_from_state = 0
    for episode, ref in enumerate(references):
        names = ref['names'].tolist(); selected = [names.index(n) for n in spec['erj_joints']]
        action = np.asarray(pq.read_table(dataset/f'data/chunk-{episode//1000:03d}/file-{episode%1000:03d}.parquet', columns=['action'])['action'].to_pylist())
        np.testing.assert_array_equal(action[:, 17:], ref['joint_action'][:, selected].astype(np.float32))
        labels.append(action[:, 17:])
        different_from_state += int(np.any(np.abs(action[:, 17:] - ref['state'][:, selected]) > 1e-5, axis=1).sum())
        arm_names = [n for n in names if n.startswith('arm_')]
        arm_ids = [robot.model.joint(n).qposadr[0] for n in arm_names]
        for frame in np.linspace(0, len(ref['state'])-1, min(40, len(ref['state'])), dtype=int):
            fixed = {name: float(ref['joint_action'][frame, names.index(name)]) for name in spec['erj_joints']}
            for nearby in (False, True):
                robot.set_joints(ref['joint_action'][frame] if nearby else ref['state'][frame], names)
                # Non-arm DOFs use the same command as the demonstration target.
                # These are IK fixtures, not predicted-policy replay.
                for name in names[:19]:
                    if not name.startswith('arm_'):
                        robot.data.qpos[robot.model.joint(name).qposadr[0]] = ref['joint_action'][frame, names.index(name)]
                if nearby:
                    for i, name in enumerate(arm_names):
                        joint = robot.model.joint(name); qid = joint.qposadr[0]
                        robot.data.qpos[qid] = np.clip(robot.data.qpos[qid]+(.015 if i % 2 else -.015), *joint.range)
                robot.mimic(); robot.mj.mj_forward(robot.model, robot.data)
                base = robot.data.body('base_link'); base_pose = transform(base.xpos, base.xmat.reshape(3, 3))
                target = base_pose @ ref['target_eef'][frame]
                old = robot.data.qpos.copy()
                ok, position, rotation = robot.ik(target, fixed_joints=fixed)
                selected_error = [abs(robot.data.qpos[robot.model.joint(n).qposadr[0]]-v) for n, v in fixed.items()]
                if ok:
                    assert max(selected_error) < 1e-12 and max(position) < .002 and max(rotation) < .03
                    assert np.abs(robot.data.qpos[arm_ids]-old[arm_ids]).max() <= .12+1e-12
                    for name in arm_names:
                        joint = robot.model.joint(name)
                        assert joint.range[0] <= robot.data.qpos[joint.qposadr[0]] <= joint.range[1]
                else:
                    np.testing.assert_array_equal(robot.data.qpos, old)
                (nearby_checks if nearby else checks).append({'source_episode': int(ref['source_episode']), 'frame': int(frame),
                    'ok': bool(ok), 'reason': robot.last_ik['reason'], 'selected_error_rad': selected_error,
                    'position_residual_m': position.tolist(), 'rotation_residual_rad': rotation.tolist()})
    stats = read_json(dataset/'meta/stats.json')['action']; labels = np.concatenate(labels)
    np.testing.assert_allclose(np.asarray(stats['min'])[17:], labels.min(0), atol=1e-7)
    np.testing.assert_allclose(np.asarray(stats['max'])[17:], labels.max(0), atol=1e-7)
    assert different_from_state > 0, 'Need actual command/state differences to verify label source'
    assert any(row['ok'] for row in checks), 'No successful demonstration constrained IK'
    assert all(row['ok'] for row in nearby_checks), 'Nearby ground-truth constrained IK regression'
    report = {'status': 'PASS', 'action_spec': spec, 'rank_audit': audit, 'same_row_action_labels_exact': True,
              'frames_with_labels_different_from_observation_state': different_from_state,
              'normalization_statistics_verified': True, 'demonstration_checks': checks,
              'demonstration_successes': sum(row['ok'] for row in checks), 'demonstration_samples': len(checks),
              'nearby_target_checks': nearby_checks, 'nearby_successes': sum(row['ok'] for row in nearby_checks),
              'description': 'Actual demonstration EEF and absolute joint targets; exact fixed equality during reduced-Jacobian solve; failures hold old qpos'}
    write_json(output, report)
    print(f'ERJ demonstration validation PASS: {report["demonstration_successes"]}/{len(checks)} measured-state starts, {len(nearby_checks)} nearby starts', flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('dataset', 'scene', 'output'): p.add_argument('--'+key, required=True)
    a = p.parse_args(); validate(a.dataset, a.scene, a.output)
