"""Shared SE(3), dataset conversion and FFW kinematics for the offline smoke test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'shared/shared/robot_configs/urdf/ffw_sg2_follower.urdf'
BASE_COMMIT = '240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b'
CAMERAS = ['observation.images.cam_wrist_left', 'observation.images.cam_wrist_right']
TCP = np.array([0., 0., -.215])
LAYOUT = [f'{side}.{term}' for side in ('left', 'right')
          for term in ('dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'gripper_rad')]
LAYOUT += ['head_joint1_rad', 'head_joint2_rad', 'lift_joint_m']


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def transform(xyz=(0., 0., 0.), rotation=None):
    t = np.eye(4)
    t[:3, 3] = xyz
    if rotation is not None:
        t[:3, :3] = rotation
    return t


def relative(ref, target):
    r = ref[:3, :3].T
    return np.r_[r @ (target[:3, 3] - ref[:3, 3]),
                 Rotation.from_matrix(r @ target[:3, :3]).as_rotvec()]


def compose(ref, delta):
    return ref @ transform(delta[:3], Rotation.from_rotvec(delta[3:6]).as_matrix())


class URDFFK:
    """Independent, named-joint FK; no alphabetical joint-vector mapping."""
    def __init__(self, path=URDF):
        self.joints = {}
        for j in ET.parse(path).getroot().findall('joint'):
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            axis = j.find('axis')
            self.joints[j.find('child').get('link')] = (
                j.find('parent').get('link'), j.get('name'), j.get('type'),
                transform(xyz, Rotation.from_euler('xyz', rpy).as_matrix()),
                np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0., 0.]))

    def pose(self, q, link):
        if link == 'base_link':
            return np.eye(4)
        parent, name, kind, origin, axis = self.joints[link]
        motion = np.eye(4)
        if kind in ('revolute', 'continuous'):
            motion[:3, :3] = Rotation.from_rotvec(axis * q.get(name, 0.)).as_matrix()
        elif kind == 'prismatic':
            motion[:3, 3] = axis * q.get(name, 0.)
        elif kind != 'fixed':
            raise ValueError(f'Unsupported URDF joint {name}: {kind}')
        return self.pose(q, parent) @ origin @ motion

    def eef(self, vector, names):
        q = dict(zip(names, vector, strict=True))
        return np.stack([self.pose(q, f'end_effector_{s}_link') for s in ('l', 'r')])


class Robot:
    def __init__(self, scene, align_urdf=True):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.corrections = []
        # The existing menagerie and Cyclo URDF differ in the lift attachment.
        # Apply only this documented static frame correction, never joint data.
        if align_urdf:
            fk = URDFFK()
            desired = fk.joints['arm_base_link'][3][:3, 3]
            body = self.model.body('arm_base_link')
            parent = self.model.body(int(body.parentid[0]))
            parent_offset = parent.pos.copy() if parent.name == 'lift_link' else np.zeros(3)
            wanted = desired - parent_offset
            if not np.allclose(body.pos, wanted, atol=1e-10):
                self.corrections.append({'body': 'arm_base_link', 'old_pos': body.pos.tolist(),
                                         'new_pos': wanted.tolist(), 'reason': 'Cyclo URDF lift_joint origin'})
                body.pos[:] = wanted
        mujoco.mj_forward(self.model, self.data)

    def set_joints(self, vector, names):
        for value, name in zip(vector, names, strict=True):
            if name in ('linear_x', 'linear_y', 'angular_z'):
                continue  # velocity channels, not configuration coordinates
            joint = self.model.joint(name)
            self.data.qpos[joint.qposadr[0]] = value
        self.mimic()
        self.mj.mj_forward(self.model, self.data)

    def mimic(self):
        for side in ('l', 'r'):
            g = self.data.qpos[self.model.joint(f'gripper_{side}_joint1').qposadr[0]]
            for index, sign in ((2, -1), (3, -1), (4, 1)):
                self.data.qpos[self.model.joint(f'gripper_{side}_joint{index}').qposadr[0]] = sign * g

    def eef(self, base_relative=False):
        poses = []
        for s in ('l', 'r'):
            body = self.data.body(f'arm_{s}_link7')
            rot = body.xmat.reshape(3, 3)
            poses.append(transform(body.xpos + rot @ TCP, rot))
        poses = np.stack(poses)
        if base_relative:
            base = self.data.body('base_link')
            inv = np.linalg.inv(transform(base.xpos, base.xmat.reshape(3, 3)))
            poses = inv @ poses
        return poses

    def ik(self, targets, max_change=.12, iterations=60):
        old = self.data.qpos.copy()
        arms = []
        for side in ('l', 'r'):
            joints = [self.model.joint(f'arm_{side}_joint{i}') for i in range(1, 8)]
            arms.append((np.array([j.qposadr[0] for j in joints]),
                         np.array([j.dofadr[0] for j in joints]),
                         np.array([j.range for j in joints]),
                         self.model.body(f'arm_{side}_link7').id))
        for _ in range(iterations):
            poses = self.eef()
            for s, (qids, dids, limits, bid) in enumerate(arms):
                e = np.r_[targets[s, :3, 3] - poses[s, :3, 3],
                          Rotation.from_matrix(targets[s, :3, :3] @ poses[s, :3, :3].T).as_rotvec()]
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                self.mj.mj_jac(self.model, self.data, jp, jr, poses[s, :3, 3], bid)
                jac = np.vstack([jp[:, dids], .3 * jr[:, dids]])
                e[3:] *= .3
                dq = jac.T @ np.linalg.solve(jac @ jac.T + .001**2 * np.eye(6), e)
                lo = np.maximum(limits[:, 0], old[qids] - max_change)
                hi = np.minimum(limits[:, 1], old[qids] + max_change)
                self.data.qpos[qids] = np.clip(self.data.qpos[qids] + np.clip(dq, -.05, .05), lo, hi)
            self.mj.mj_forward(self.model, self.data)
            poses = self.eef()
            poserr = np.linalg.norm(targets[:, :3, 3] - poses[:, :3, 3], axis=1)
            roterr = np.array([Rotation.from_matrix(t[:3, :3] @ p[:3, :3].T).magnitude()
                               for t, p in zip(targets, poses)])
            if poserr.max() < .002 and roterr.max() < .03:
                return True, poserr, roterr
        self.data.qpos[:] = old
        self.mj.mj_forward(self.model, self.data)
        return False, poserr, roterr


def scalar_stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {**{k: v.tolist() for k, v in {
        'min': x.min(0), 'max': x.max(0), 'mean': x.mean(0), 'std': x.std(0)}.items()},
        'count': [len(x)]}


def convert(source, destination, episodes, scene):
    """Write real LeRobot v3 metadata/parquet; link unchanged source videos."""
    import copy
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Conversion destination already exists: {destination}')
    info = read_json(source / 'meta/info.json')
    features = info['features']
    names = features['observation.state']['names']
    action_names = features['action']['names']
    expected = ([f'arm_l_joint{i}' for i in range(1, 8)] + ['gripper_l_joint1'] +
                [f'arm_r_joint{i}' for i in range(1, 8)] + ['gripper_r_joint1', 'head_joint1',
                'head_joint2', 'lift_joint', 'linear_x', 'linear_y', 'angular_z'])
    if names != expected or action_names != expected or info['robot_type'] != 'ffw_sg2_rev1':
        raise ValueError('Unverified joint layout or robot; inspect recording configuration before conversion')
    episode_rows = pq.read_table(source / 'meta/episodes').to_pylist()
    selected = list(range(len(episode_rows))) if episodes == 'all' else [int(x) for x in episodes.split(',')]
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(episode_rows) for i in selected):
        raise ValueError('Invalid or duplicate episode selection')
    fk, robot = URDFFK(), Robot(scene)
    new_features = {k: copy.deepcopy(v) for k, v in features.items()
                    if k != 'observation.state' and (not k.startswith('observation.images.') or k in CAMERAS)}
    new_features['action'].update(shape=[len(LAYOUT)], names=LAYOUT)
    new_rows, all_stats, source_files = [], [], {source/'meta/info.json', source/'meta/tasks.parquet'}
    source_files.update((source/'meta/episodes').rglob('*.parquet'))
    offset, worst_reconstruction, worst_fk = 0, 0., 0.
    checks = []
    for new_index, ep_index in enumerate(selected):
        ep = episode_rows[ep_index]
        data_path = source / info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
        source_files.add(data_path)
        table = pq.read_table(data_path, filters=[('episode_index', '=', ep_index)])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
        joint_action = np.asarray(table['action'].to_pylist(), dtype=np.float64)
        n = len(state)
        assert state.shape == joint_action.shape == (ep['length'], len(names))
        assert np.isfinite(state).all() and np.isfinite(joint_action).all()
        assert table['frame_index'].to_pylist() == list(range(n))
        reference = np.stack([fk.eef(q, names) for q in state])
        target = np.stack([fk.eef(q, action_names) for q in joint_action])
        actions = np.empty((n, len(LAYOUT)), dtype=np.float32)
        for i in range(n):
            for s, grip_index in enumerate((7, 15)):
                actions[i, s*7:s*7+6] = relative(reference[i, s], target[i, s])
                actions[i, s*7+6] = joint_action[i, grip_index]
                error = np.max(np.abs(compose(reference[i, s], actions[i, s*7:s*7+6]) - target[i, s]))
                worst_reconstruction = max(worst_reconstruction, float(error))
            actions[i, 14:] = joint_action[i, 16:19]
        for i in np.linspace(0, n-1, min(12, n), dtype=int):
            for q, poses in ((state[i], reference[i]), (joint_action[i], target[i])):
                robot.set_joints(q, names)
                worst_fk = max(worst_fk, float(np.max(np.abs(poses - robot.eef(base_relative=True)))))
        assert worst_reconstruction < 1e-6 and worst_fk < 1e-6
        out = table.drop(['observation.state', 'action']).replace_schema_metadata(None)
        out = out.set_column(out.schema.get_field_index('episode_index'), 'episode_index', pa.array([new_index]*n, type=pa.int64()))
        out = out.set_column(out.schema.get_field_index('index'), 'index', pa.array(range(offset, offset+n), type=pa.int64()))
        out = out.append_column('action', pa.FixedSizeListArray.from_arrays(pa.array(actions.ravel()), len(LAYOUT)))
        rel_data = Path(f'data/chunk-{new_index//1000:03d}/file-{new_index%1000:03d}.parquet')
        (destination/rel_data).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, destination/rel_data)
        (destination/'reference').mkdir(exist_ok=True)
        np.savez_compressed(destination/'reference'/f'episode_{new_index:06d}.npz',
                            state=state, joint_action=joint_action, reference_eef=reference, target_eef=target,
                            source_episode=ep_index, names=np.array(names))
        row = {k: v for k, v in ep.items() if not k.startswith('stats/') and not k.startswith('videos/')}
        row.update(episode_index=new_index, **{'data/chunk_index':new_index//1000, 'data/file_index':new_index%1000,
                                             'dataset_from_index':offset, 'dataset_to_index':offset+n,
                                             'meta/episodes/chunk_index':0, 'meta/episodes/file_index':0})
        stats = {k: scalar_stats(out[k].to_pylist()) for k in out.column_names}
        for camera in CAMERAS:
            prefix = f'videos/{camera}/'
            row.update({k: v for k, v in ep.items() if k.startswith(prefix)})
            video = info['video_path'].format(video_key=camera, chunk_index=ep[prefix+'chunk_index'], file_index=ep[prefix+'file_index'])
            src, dst = source/video, destination/video
            source_files.add(src)
            assert src.is_file(), src
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src)
            stats[camera] = {k.removeprefix(f'stats/{camera}/'): v for k, v in ep.items() if k.startswith(f'stats/{camera}/')}
        for key, values in stats.items():
            for stat, value in values.items():
                row[f'stats/{key}/{stat}'] = value
        new_rows.append(row)
        all_stats.append({k: {s: np.asarray(v) for s, v in vals.items()} for k, vals in stats.items()})
        checks.append({'source_episode':ep_index, 'frames':n, 'validated_before_next_episode':True})
        print(f'Converted and verified episode {ep_index}: {n} frames', flush=True)
        offset += n
    new_info = copy.deepcopy(info)
    new_info.update(features=new_features, total_episodes=len(selected), total_frames=offset,
                    total_videos=len(selected)*len(CAMERAS), splits={'train':f'0:{len(selected)}'}, repo_id='local/statefree-smoke')
    write_json(destination/'meta/info.json', new_info)
    combined = aggregate_stats(all_stats)
    write_json(destination/'meta/stats.json', {k: {s:v.tolist() for s,v in vals.items()} for k,vals in combined.items()})
    ep_path = destination/'meta/episodes/chunk-000/file-000.parquet'
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(new_rows), ep_path)
    shutil.copy2(source/'meta/tasks.parquet', destination/'meta/tasks.parquet')
    model_xmls = sorted(Path(scene).parent.glob('*.xml'))
    metadata = {
        'schema_version':1, 'representation':'per_step_body_se3_rotvec', 'action_layout':LAYOUT,
        'action_units':['m']*3+['rad']*4+['m']*3+['rad']*4+['rad','rad','m'],
        'formula':'Delta_t = inverse(FK(measured_state_t)) @ FK(command_t); target = current_sim_pose @ Delta_t',
        'rotation':'SO(3) rotation vector, radians; composed by matrix multiplication',
        'quaternion_order':None, 'fps':info['fps'], 'joint_names':names,
        'eef_frames':['end_effector_l_link','end_effector_r_link'], 'tcp_parent':['arm_l_link7','arm_r_link7'],
        'tcp_translation_m':TCP.tolist(), 'tcp_rotation_rpy_rad':[0,0,0], 'fk_frame':'base_link',
        'state_semantics':'/joint_states measured positions; last 3 columns /odom linear xy and angular z velocities',
        'action_semantics':'same-row recorded JointTrajectory.points[0].positions targets; last 3 columns /cmd_vel velocities',
        'alignment':'preserve source same-row 30 Hz alignment; current Cyclo converter causal previous-value resampling; original bag timestamps unavailable',
        'grippers':'absolute revolute joint target radians, unchanged; not metres or normalized opening fraction',
        'auxiliary':'head absolute radians and lift absolute metres output; lift included in reference and command FK',
        'base':'velocity columns preserved in reference NPZ; no base output, fixed base for this kinematic smoke',
        'policy_input_keys':CAMERAS, 'reference_storage':'reference/*.npz; never loaded by training dataset',
        'source_root':str(source), 'source_episodes':selected, 'source_total_episodes':info['total_episodes'],
        'source_total_frames':info['total_frames'], 'converted_frames':offset,
        'source_sha256':{str(p.relative_to(source)):sha256(p) for p in sorted(source_files)},
        'urdf_sha256':sha256(URDF), 'model_scene':str(Path(scene).resolve()),
        'model_base_revision':'d8344c0dbe7a00208d0301111523dde65efc174a (local practice copy; hashes authoritative)',
        'model_xml_sha256':{p.name:sha256(p) for p in model_xmls}, 'model_corrections':robot.corrections,
        'validation':{'max_se3_reconstruction_error':worst_reconstruction, 'max_urdf_mujoco_error':worst_fk, 'episodes':checks},
        'lerobot_base_commit':BASE_COMMIT,
    }
    write_json(destination/'meta/action_representation.json', metadata)
    return metadata
