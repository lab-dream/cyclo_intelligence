"""Small independent numerical checks, in addition to policy.py's real-data tests."""
import numpy as np
from scipy.spatial.transform import Rotation
from core import URDFFK, compose, relative, transform


def test_body_transform_is_not_world_translation_or_euler_subtraction():
    ref=transform([1,2,3],Rotation.from_euler('xyz',[.7,-.3,1.2]).as_matrix())
    goal=transform([1.2,1.9,3.1],Rotation.from_euler('xyz',[-.4,.5,-2.9]).as_matrix())
    action=relative(ref,goal)
    np.testing.assert_allclose(compose(ref,action),goal,atol=1e-12)
    assert not np.allclose(action[:3],goal[:3,3]-ref[:3,3])


def test_fk_joint_names_and_left_right_are_independent():
    fk=URDFFK();names=['arm_r_joint3','lift_joint','arm_l_joint1']
    q=[.2,-.1,.3];a=fk.eef(q,names)
    b=fk.eef(q[::-1],names[::-1])
    np.testing.assert_allclose(a,b,atol=1e-12)
    moved=fk.eef([.2,-.1,.8],names)
    np.testing.assert_allclose(a[1],moved[1],atol=1e-12)
    assert not np.allclose(a[0],moved[0])


def test_manifest_reuse_rejects_changed_inputs_or_output(tmp_path):
    from pipeline import Pipeline
    from core import read_json
    p=object.__new__(Pipeline);p.path=tmp_path/'manifest.json';p.manifest={'stages':{}}
    output=tmp_path/'artifact';calls=[]
    def operation():
        calls.append(1);output.write_text('verified result');return {'status':'PASS'}
    p.stage('test',{'code':'a','input':'x'},operation,[output])
    p.stage('test',{'code':'a','input':'x'},operation,[output])
    assert len(calls)==1
    output.write_text('corrupted')
    p.stage('test',{'code':'a','input':'x'},operation,[output])
    p.stage('test',{'code':'b','input':'x'},operation,[output])
    p.stage('test',{'code':'b','input':'y'},operation,[output])
    assert len(calls)==4
    assert read_json(p.path)['stages']['test']['status']=='PASS'


def test_failed_stage_is_retried(tmp_path):
    import pytest
    from pipeline import Pipeline
    p=object.__new__(Pipeline);p.path=tmp_path/'manifest.json';p.manifest={'stages':{}}
    output=tmp_path/'artifact'
    def failure():raise RuntimeError('recoverable')
    with pytest.raises(RuntimeError):p.stage('test',{},failure,[output])
    assert p.manifest['stages']['test']['status']=='FAIL'
    p.stage('test',{},lambda: output.write_text('recovered'),[output])
    assert p.manifest['stages']['test']['status']=='PASS'
