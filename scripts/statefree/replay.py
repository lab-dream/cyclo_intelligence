"""Cyclo recorded-observation inference -> SE(3) -> DLS IK -> MuJoCo video."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import subprocess
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from core import CAMERAS, Robot, compose, read_json, sha256, transform, write_json
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_engine import LeRobotEngine


def limit_norm(v, maximum):
    return v * min(1., maximum/max(float(np.linalg.norm(v)), 1e-12))


def add_line(scene, a, b, color, radius=.002, arrow=False):
    if scene.ngeom >= scene.maxgeom:
        return
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b)
    scene.ngeom+=1


def decorations(scene, target, achieved, reference, trails):
    colors=((1.,.35,.12,1.),(.1,1.,.3,1.),(.2,.7,1.,1.))
    for points,color in zip(trails,colors):
        for i in range(max(1,len(points)-180),len(points),2):
            for side in range(2):
                add_line(scene,points[i-1][side,:3,3],points[i][side,:3,3],color)
    for poses,color in zip((target,achieved,reference),colors):
        for pose in poses:
            p=pose[:3,3]
            for axis,rgb in enumerate(((1,0,0,1),(0,1,0,1),(0,0,1,1))):
                add_line(scene,p,p+.055*pose[:3,axis],rgb,.0015,True)
            # A colored cross identifies whose frame the RGB axes belong to.
            for axis in range(3):
                d=np.eye(3)[axis]*.008
                add_line(scene,p-d,p+d,color,.003)


def render_frame(renderer, robot, camera, target, achieved, reference, trails, sample, frame, fps, ok, clips, latency):
    renderer.update_scene(robot.data,camera=camera)
    decorations(renderer.scene,target,achieved,reference,trails)
    rgb=renderer.render().copy()
    panel=np.full((720,320,3),22,dtype=np.uint8)
    for side,key in enumerate(CAMERAS):
        img=sample[key].cpu().numpy().transpose(1,2,0)
        if img.dtype!=np.uint8:
            img=np.rint(img*255).astype(np.uint8)
        panel[35+side*210:216+side*210]=cv2.resize(img,(320,181))
        cv2.putText(panel,('LEFT' if side==0 else 'RIGHT')+' wrist input',(8,25+side*210),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
    lines=[('ORANGE  policy EEF target',(255,90,35)),('GREEN   achieved EEF',(30,255,80)),
           ('BLUE    recorded reference',(50,180,255)),('RGB axes: X / Y / Z',(230,230,230)),
           (f'Frame {frame}  t={frame/fps:.2f}s',(235,235,235)),
           (f'IK {"OK" if ok else "HOLD"}   limited: {clips}',(235,235,235)),
           (f'Chunk inference: {latency*1000:.1f} ms',(235,235,235)),
           ('Recorded-observation replay',(240,205,90)),('Kinematic qpos, fixed base',(240,205,90))]
    for i,(text,color) in enumerate(lines):
        cv2.putText(panel,text,(8,460+i*27),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1,cv2.LINE_AA)
    cv2.putText(rgb,'FFW-SG2 | image-only Diffusion | predicted motion',(15,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
    return np.concatenate([rgb,panel],axis=1)


def replay(dataset,checkpoint,scene,output,frames=300,episode=0,gui=False):
    torch.set_num_threads(4);torch.manual_seed(42)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    engine=LeRobotEngine();metadata=engine.load_recorded_policy(checkpoint)
    ds=LeRobotDataset('local/statefree-smoke',root=dataset,video_backend='pyav')
    ref=np.load(Path(dataset)/'reference'/f'episode_{episode:06d}.npz')
    names=ref['names'].tolist(); robot=Robot(scene)
    assert metadata['urdf_sha256']==sha256(Path(__file__).resolve().parents[2]/'shared/shared/robot_configs/urdf/ffw_sg2_follower.urdf')
    for name,digest in metadata['model_xml_sha256'].items():
        assert sha256(Path(scene).parent/name)==digest,'Robot model changed since conversion'
    robot.set_joints(ref['state'][0],names)
    initial=robot.data.qpos.copy()
    initial_limit_violations=[]
    for name in names[:19]:
        joint=robot.model.joint(name);q=float(robot.data.qpos[joint.qposadr[0]])
        if joint.limited[0] and not joint.range[0] <= q <= joint.range[1]:
            initial_limit_violations.append({'joint':name,'value':q,'range':joint.range.tolist()})
    base=robot.data.body('base_link')
    base_pose=transform(base.xpos,base.xmat.reshape(3,3))
    start=int(ds.meta.episodes[episode]['dataset_from_index'])
    frames=min(frames,len(ref['state']))
    fps=metadata['fps']; camera=mujoco.MjvCamera()
    camera.lookat[:]=[.02,0,.92];camera.distance=2.65;camera.azimuth=215;camera.elevation=-16
    # The original scene framebuffer may be smaller than the saved video.
    robot.model.vis.global_.offwidth=960;robot.model.vis.global_.offheight=720
    renderer=mujoco.Renderer(robot.model,height=720,width=960)
    video=output/'replay.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    viewer=None
    if gui:
        from mujoco import viewer as viewer_module
        viewer=viewer_module.launch_passive(robot.model,robot.data)
        viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance;viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
    logs={k:[] for k in ('raw_action','normalized_action','applied_action','raw_target','target','achieved','reference','qpos','joint_result',
                             'limited','ik_ok','ik_position_residual','ik_rotation_residual','achieved_position_error','inference_seconds','chunk_index')}
    chunk=None;chunk_position=0;chunk_index=-1;latency=0.;inference_total=0.;ik_total=0.
    trails=[[],[],[]];wall_start=time.perf_counter()
    try:
        for frame in range(frames):
            sample=ds[start+frame]
            engine.observe_recorded(sample)
            if chunk is None or chunk_position==len(chunk['action']):
                before=time.perf_counter();chunk=engine.predict_recorded_chunk();torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency=time.perf_counter()-before;inference_total+=latency;chunk_index+=1;chunk_position=0
                assert chunk['representation']==metadata['representation']
                current_latency=latency
            else:current_latency=0.
            raw=chunk['action'][chunk_position].copy();normalized=chunk['normalized_action'][chunk_position].copy();chunk_position+=1
            assert np.isfinite(raw).all()
            applied=raw.copy();current=robot.eef();previous=robot.data.qpos.copy()
            raw_target=np.stack([compose(current[s],raw[s*7:s*7+6]) for s in range(2)])
            for s in range(2):
                applied[s*7:s*7+3]=limit_norm(raw[s*7:s*7+3],.02)
                applied[s*7+3:s*7+6]=limit_norm(raw[s*7+3:s*7+6],.08)
            # Preserve the original predictions; limit only the applied commands.
            aux=((6,'gripper_l_joint1',.05),(13,'gripper_r_joint1',.05),
                 (14,'head_joint1',.03),(15,'head_joint2',.03),(16,'lift_joint',.008))
            for index,name,step_limit in aux:
                j=robot.model.joint(name);old=previous[j.qposadr[0]]
                applied[index]=np.clip(np.clip(raw[index],old-step_limit,old+step_limit),j.range[0],j.range[1])
                robot.data.qpos[j.qposadr[0]]=applied[index]
            robot.mimic();mujoco.mj_forward(robot.model,robot.data)
            target=np.stack([compose(current[s],applied[s*7:s*7+6]) for s in range(2)])
            before=time.perf_counter();ok,poserr,roterr=robot.ik(target);ik_total+=time.perf_counter()-before
            if not ok:
                robot.data.qpos[:]=previous;mujoco.mj_forward(robot.model,robot.data)
            achieved=robot.eef();reference=base_pose @ ref['reference_eef'][frame]
            limited=np.abs(applied-raw)>1e-10
            joint_result=np.array([robot.data.qpos[robot.model.joint(name).qposadr[0]] for name in names[:19]])
            values=(raw,normalized,applied,raw_target,target,achieved,reference,robot.data.qpos.copy(),joint_result,
                    limited,ok,poserr,roterr,np.linalg.norm(target[:,:3,3]-achieved[:,:3,3],axis=1),current_latency,chunk_index)
            for key,value in zip(logs,values,strict=True):logs[key].append(value)
            for trail,poses in zip(trails,(target,achieved,reference)):trail.append(poses.copy())
            rgb=render_frame(renderer,robot,camera,target,achieved,reference,trails,sample,frame,fps,ok,int(limited.sum()),latency)
            encoder.stdin.write(rgb.tobytes())
            if frame in (0,frames//2,frames-1):cv2.imwrite(str(output/f'frame_{frame:04d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if viewer:
                with viewer.lock():
                    viewer.user_scn.ngeom=0;decorations(viewer.user_scn,target,achieved,reference,trails)
                viewer.sync();time.sleep(1/fps)
            if frame%60==0:print(f'Replay frame {frame}/{frames}; chunks={chunk_index+1}; IK ok={sum(logs["ik_ok"])}',flush=True)
    finally:
        encoder.stdin.close();encoder.wait();renderer.close()
        if viewer:viewer.close()
    assert encoder.returncode==0 and video.stat().st_size>10000
    arrays={k:np.asarray(v) for k,v in logs.items()};np.savez_compressed(output/'replay.npz',**arrays,initial_qpos=initial,joint_names=names[:19])
    motion=float(np.max(np.abs(arrays['qpos']-initial)))
    arm_ids=[robot.model.joint(f'arm_{s}_joint{i}').qposadr[0] for s in ('l','r') for i in range(1,8)]
    arm_motion=float(np.max(np.abs(arrays['qpos'][:,arm_ids]-initial[arm_ids])))
    assert arm_motion>1e-5, 'Markers and auxiliary joints alone do not verify arm motion'
    assert motion>1e-5 and chunk_index>=2,'Robot must actually move over multiple policy chunks'
    report={'status':'PASS','mode':'recorded-observation replay; MuJoCo kinematic qpos update; no closed-loop/task evaluation',
        'episode':episode,'source_episode':int(ref['source_episode']),'frames':frames,'fps':fps,'simulation_seconds':frames/fps,
        'wall_seconds':time.perf_counter()-wall_start,'inference_seconds':inference_total,'ik_seconds':ik_total,
        'chunks':chunk_index+1,'ik_successes':int(np.sum(arrays['ik_ok'])),'ik_failures':int(np.sum(~arrays['ik_ok'])),
        'limited_frames':int(np.any(arrays['limited'],axis=1).sum()),'limited_components':int(arrays['limited'].sum()),
        'max_arm_joint_motion_rad':arm_motion,'max_qpos_motion':motion,'max_achieved_position_error_m':float(arrays['achieved_position_error'].max()),
        'mean_chunk_latency_s':inference_total/(chunk_index+1),'video':str(video.resolve()),'video_sha256':sha256(video),
        'initial_limit_projection':'Original measured initialization is preserved; first accepted auxiliary command projects out-of-range joints into model limits, which can exceed the per-step cap for that initial correction',
        'initial_limit_violations':initial_limit_violations,'model_corrections':robot.corrections,
        'input_keys':metadata['policy_input_keys'],'checkpoint':str(Path(checkpoint).resolve()),
        'limits':{'eef_translation_norm_m':.02,'eef_rotation_norm_rad':.08,'arm_joint_step_rad':.12,
                  'gripper_step_rad':.05,'head_step_rad':.03,'lift_step_m':.008},
        'failure_behavior':'hold all preceding qpos; never replace with recorded joints'}
    write_json(output/'replay_report.json',report);print(report,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=300)
    p.add_argument('--episode',type=int,default=0);p.add_argument('--gui',action='store_true');a=p.parse_args()
    replay(a.dataset,a.checkpoint,a.scene,a.output,a.frames,a.episode,a.gui)
