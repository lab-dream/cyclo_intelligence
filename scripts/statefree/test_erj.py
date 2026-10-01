"""Action compatibility and hard-equality IK regression tests."""
import os
from pathlib import Path

import numpy as np
import pytest

from core import LAYOUT, Robot, action_spec, checked_action_spec


def test_erj_preserves_legacy_prefix_and_names_absolute_labels():
    legacy = action_spec()
    erj = action_spec('erj', ['arm_l_joint1', 'arm_r_joint1'])
    assert legacy['action_layout'] == LAYOUT and legacy['action_dim'] == 17
    assert erj['action_dim'] == 19 and erj['action_layout'][:17] == LAYOUT
    assert erj['action_layout'][17:] == ['arm_l_joint1_absolute_rad', 'arm_r_joint1_absolute_rad']
    assert erj['action_units'][17:] == ['rad', 'rad']
    assert checked_action_spec({'action_layout': LAYOUT}) == legacy


@pytest.mark.parametrize('mode,joints', [('eef',['arm_l_joint1','arm_r_joint1']),
    ('erj',['arm_r_joint1','arm_l_joint1']), ('erj',['lift_joint','arm_r_joint1']),
    ('erj',['arm_l_joint4']), ('erj',['arm_l_joint1','arm_l_joint1']), ('joint',[])])
def test_invalid_joint_spec_is_rejected(mode,joints):
    with pytest.raises(ValueError):action_spec(mode,joints)


@pytest.mark.parametrize('key,value',[('action_dim',17),('action_units',['rad']*19),('erj_offset',14)])
def test_mismatched_erj_metadata_is_rejected(key,value):
    metadata=action_spec('erj');metadata[key]=value
    with pytest.raises(ValueError,match='Inconsistent'):checked_action_spec(metadata)


@pytest.fixture
def robot():
    scene=Path(os.environ.get('STATEFREE_SCENE','/home/son/Downloads/AI_Worker_Practice/third_party/robotis_mujoco_menagerie/robotis_ffw/scene_ffw_sg2.xml'))
    if not scene.is_file():pytest.skip('Set STATEFREE_SCENE to run actual FFW-SG2 IK tests')
    r=Robot(scene)
    for side,values in [('l',[.3,.5,.4,-.9,.5,.2,-.2]),('r',[-.3,-.5,-.4,-.9,-.5,.2,.2])]:
        r.set_joints(values,[f'arm_{side}_joint{i}' for i in range(1,8)])
    return r


def test_predicted_joints_stay_fixed_during_every_jacobian_evaluation(robot,monkeypatch):
    target=robot.eef();fixed={'arm_l_joint1':.3,'arm_r_joint1':-.3}
    for side in ('l','r'):
        for i in range(1,8):robot.data.qpos[robot.model.joint(f'arm_{side}_joint{i}').qposadr[0]]+=.015
    robot.mj.mj_forward(robot.model,robot.data)
    jacobian=robot.mj.mj_jac;checks=[]
    def checked_jacobian(*args):
        values=[robot.data.qpos[robot.model.joint(name).qposadr[0]] for name in fixed]
        np.testing.assert_array_equal(values,list(fixed.values()));checks.append(values)
        return jacobian(*args)
    monkeypatch.setattr(robot.mj,'mj_jac',checked_jacobian)
    ok,pos,rot=robot.ik(target,fixed_joints=fixed)
    assert ok and checks and pos.max()<.002 and rot.max()<.03


@pytest.mark.parametrize('value',[5.,.5])
def test_invalid_fixed_joint_goal_holds_every_joint(robot,value):
    old=robot.data.qpos.copy();target=robot.eef()
    ok,_,_=robot.ik(target,fixed_joints={'arm_l_joint1':value})
    assert not ok and robot.last_ik['reason']=='fixed_target_outside_joint_or_step_limit'
    np.testing.assert_array_equal(robot.data.qpos,old)


def test_unreachable_erj_target_rolls_back_without_relaxing_constraint(robot):
    old=robot.data.qpos.copy();target=robot.eef();target[:,:3,3]+=10.
    ok,_,_=robot.ik(target,fixed_joints={'arm_l_joint1':.31,'arm_r_joint1':-.31})
    assert not ok and robot.last_ik['reason']=='iteration_limit'
    np.testing.assert_array_equal(robot.data.qpos,old)


def test_off_and_legacy_ik_are_identical(robot):
    initial=robot.data.qpos.copy();target=robot.eef();target[:,0,3]+=.003
    a=robot.ik(target);qa=robot.data.qpos.copy()
    robot.data.qpos[:]=initial;robot.mj.mj_forward(robot.model,robot.data)
    b=robot.ik(target,fixed_joints={})
    assert a[0]==b[0]
    np.testing.assert_array_equal(qa,robot.data.qpos)
    np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
