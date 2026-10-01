"""Cyclo recorded-observation inference -> SE(3) -> DLS IK -> MuJoCo video."""
from __future__ import annotations
import argparse
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import subprocess
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from core import CAMERAS, Robot, compose, read_json, sha256, transform, write_json
from interpolation import METHODS, blend, sample_joints, rates, control_bounds, motion_metrics
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


def render_frame(renderer, robot, camera, target, achieved, reference, trails, sample, frame, fps, ok, clips, latency,
                 interpolation='none', control_time=None, control_fps=30):
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
           (f'Frame {frame}  t={(frame/fps if control_time is None else control_time):.3f}s',(235,235,235)),
           (f'IK {"OK" if ok else "HOLD"}   limited: {clips}',(235,235,235)),
           (f'Chunk inference: {latency*1000:.1f} ms',(235,235,235)),
           ('Recorded-observation replay',(240,205,90)),('Kinematic qpos, fixed base',(240,205,90))]
    for i,(text,color) in enumerate(lines):
        cv2.putText(panel,text,(8,460+i*27),cv2.FONT_HERSHEY_SIMPLEX,.47,color,1,cv2.LINE_AA)
    cv2.putText(rgb,f'{interpolation.upper()} | IK joint spline | control {control_fps} Hz',(15,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
    return np.concatenate([rgb,panel],axis=1)


def replay(dataset,checkpoint,scene,output,frames=300,episode=0,gui=False,
           interpolation='quintic',control_fps=100,video_fps=100,actions_from=None):
    torch.set_num_threads(4);torch.manual_seed(42)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if interpolation not in METHODS:raise ValueError(f'Unknown interpolation: {interpolation}')
    metadata=read_json(Path(checkpoint)/'action_representation.json')
    checkpoint_hash=sha256(Path(checkpoint)/'model.safetensors')
    action_metadata_hash=sha256(Path(dataset)/'meta/action_representation.json')
    cached=None;engine=None
    if actions_from:
        receipt=read_json(Path(actions_from).parent/'replay_report.json')
        if (receipt['checkpoint_model_sha256'] != checkpoint_hash or
            receipt['action_metadata_sha256'] != action_metadata_hash or receipt['episode'] != episode):
            raise ValueError('Cached actions belong to a different checkpoint, dataset or episode')
        cached=np.load(actions_from)
    else:
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
    if cached is not None and len(cached['raw_action']) < frames:raise ValueError('Not enough cached actions')
    fps=metadata['fps']; camera=mujoco.MjvCamera()
    ratio,render_stride=rates(fps,control_fps,video_fps)
    if interpolation!='none' and ratio < 2:raise ValueError('Spline interpolation requires at least two control ticks per policy frame')
    bounds=np.array([control_bounds(frame,fps,control_fps) for frame in range(frames)],dtype=np.int64)
    total_ticks=int(bounds[-1,1])
    if total_ticks % render_stride:raise ValueError('Video must contain a whole number of frames; use --video-fps equal to control rate')
    joint_names=names[:19]
    joint_ids=np.array([robot.model.joint(name).qposadr[0] for name in joint_names])
    camera.lookat[:]=[.02,0,.92];camera.distance=2.65;camera.azimuth=215;camera.elevation=-16
    # The original scene framebuffer may be smaller than the saved video.
    robot.model.vis.global_.offwidth=960;robot.model.vis.global_.offheight=720
    renderer=mujoco.Renderer(robot.model,height=720,width=960)
    video=output/'replay.mp4'
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r',str(video_fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    viewer=None;viewer_threads=[]
    if gui:
        from mujoco import viewer as viewer_module
        preceding_threads=set(threading.enumerate())
        viewer=viewer_module.launch_passive(robot.model,robot.data)
        viewer_threads=[thread for thread in threading.enumerate() if thread not in preceding_threads]
        viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance;viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
    logs={k:[] for k in ('raw_action','normalized_action','applied_action','raw_target','target','achieved','reference','qpos','joint_result',
                             'limited','ik_ok','ik_position_residual','ik_rotation_residual','achieved_position_error','inference_seconds','chunk_index',
                             'source_time','segment_start_time','segment_end_time')}
    chunk=None;chunk_position=0;chunk_index=-1;latency=0.;inference_total=0.;ik_total=0.
    trajectory={'time':[0.], 'joint_result':[initial[joint_ids].copy()], 'eef':[robot.eef()],
                'cartesian_chord_error_m':[np.zeros(2)],'control_tick':[0],'policy_frame':[-1],'phase':[0.]}
    rendered_frames=0
    trails=[[],[],[]];wall_start=time.perf_counter()
    try:
        for frame in range(frames):
            start_tick,end_tick=map(int,bounds[frame]);substeps=end_tick-start_tick
            sample=ds[start+frame]
            if cached is None:
                engine.observe_recorded(sample)
                if chunk is None or chunk_position==len(chunk['action']):
                    before=time.perf_counter();chunk=engine.predict_recorded_chunk();torch.cuda.synchronize() if torch.cuda.is_available() else None
                    latency=time.perf_counter()-before;inference_total+=latency;chunk_index+=1;chunk_position=0
                    assert chunk['representation']==metadata['representation']
                    current_latency=latency
                else:current_latency=0.
                raw=chunk['action'][chunk_position].copy();normalized=chunk['normalized_action'][chunk_position].copy();chunk_position+=1
            else:
                raw=cached['raw_action'][frame].copy();normalized=cached['normalized_action'][frame].copy()
                chunk_index=int(cached['chunk_index'][frame]);current_latency=0.
                if cached['inference_seconds'][frame] > 0:latency=float(cached['inference_seconds'][frame])
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
            goal=robot.data.qpos.copy()
            limited=np.abs(applied-raw)>1e-10
            joint_result=np.array([robot.data.qpos[robot.model.joint(name).qposadr[0]] for name in names[:19]])
            values=(raw,normalized,applied,raw_target,target,achieved,reference,robot.data.qpos.copy(),joint_result,
                    limited,ok,poserr,roterr,np.linalg.norm(target[:,:3,3]-achieved[:,:3,3],axis=1),current_latency,chunk_index,
                    frame/fps,start_tick/control_fps,end_tick/control_fps)
            for key,value in zip(logs,values,strict=True):logs[key].append(value)
            # The IK endpoint is solved once. Only its accepted joint motion is
            # time-parameterized, so all modes retain identical next-frame origins.
            for substep in range(1,substeps+1):
                u=substep/substeps
                robot.data.qpos[:]=previous
                robot.data.qpos[joint_ids]=sample_joints(previous[joint_ids],goal[joint_ids],u,interpolation)
                robot.mimic()
                if substep==substeps:robot.data.qpos[:]=goal
                mujoco.mj_forward(robot.model,robot.data)
                pose=robot.eef();tick=start_tick+substep;control_time=tick/control_fps
                alpha=float(blend(u,interpolation))
                chord=current[:,:3,3]+alpha*(achieved[:,:3,3]-current[:,:3,3])
                trajectory['time'].append(control_time)
                trajectory['joint_result'].append(robot.data.qpos[joint_ids].copy())
                trajectory['eef'].append(pose)
                trajectory['cartesian_chord_error_m'].append(np.linalg.norm(pose[:,:3,3]-chord,axis=1))
                trajectory['control_tick'].append(tick)
                trajectory['policy_frame'].append(frame)
                trajectory['phase'].append(u)
                if tick % render_stride==0:
                    for trail,poses in zip(trails,(target,pose,reference)):trail.append(poses.copy())
                    rgb=render_frame(renderer,robot,camera,target,pose,reference,trails,sample,frame,fps,ok,int(limited.sum()),latency,
                                     interpolation,control_time,control_fps)
                    encoder.stdin.write(rgb.tobytes());rendered_frames+=1
                    if frame in (0,frames//2,frames-1) and substep==substeps:
                        cv2.imwrite(str(output/f'frame_{frame:04d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    if viewer:
                        with viewer.lock():
                            viewer.user_scn.ngeom=0;decorations(viewer.user_scn,target,pose,reference,trails)
                        viewer.sync()
                if viewer:time.sleep(max(0,wall_start+control_time-time.perf_counter()))
            if frame%60==0:print(f'Replay frame {frame}/{frames}; chunks={chunk_index+1}; IK ok={sum(logs["ik_ok"])}',flush=True)
    finally:
        encoder.stdin.close();encoder.wait()
        if viewer:
            # Handle.close() requests exit but does not join the daemon render
            # thread. Let our viewer finish before freeing the other GL context
            # or letting GLFW's process-exit cleanup run.
            viewer.close()
            for thread in viewer_threads:thread.join(timeout=5)
            if any(thread.is_alive() for thread in viewer_threads):
                raise RuntimeError('MuJoCo viewer did not finish closing')
        renderer.close()
    assert encoder.returncode==0 and video.stat().st_size>10000
    arrays={k:np.asarray(v) for k,v in logs.items()};np.savez_compressed(output/'replay.npz',**arrays,initial_qpos=initial,joint_names=names[:19])
    path={k:np.asarray(v) for k,v in trajectory.items()}
    np.savez_compressed(output/'trajectory.npz',**path,joint_names=joint_names)
    np.savetxt(output/'commands.csv',np.column_stack((path['time'][1:],path['control_tick'][1:],
               path['policy_frame'][1:],path['joint_result'][1:])),delimiter=',',
               header=','.join(['time_s','control_tick','policy_frame']+joint_names),comments='',
               fmt=['%.9f','%d','%d']+['%.12f']*len(joint_names))
    assert np.array_equal(path['joint_result'][bounds[:,1]],arrays['joint_result'])
    assert np.array_equal(path['control_tick'],np.arange(total_ticks+1))
    motion=float(np.max(np.abs(arrays['qpos']-initial)))
    arm_ids=[robot.model.joint(f'arm_{s}_joint{i}').qposadr[0] for s in ('l','r') for i in range(1,8)]
    arm_motion=float(np.max(np.abs(arrays['qpos'][:,arm_ids]-initial[arm_ids])))
    failures=[]
    if arm_motion<=1e-5:failures.append('No verified arm motion; marker or auxiliary motion alone is insufficient')
    if motion<=1e-5 or chunk_index<2:failures.append('Robot must actually move over multiple policy chunks')
    report={'status':'FAIL' if failures else 'PASS','validation_failures':failures,
        'mode':'recorded-observation replay; MuJoCo kinematic qpos update; no closed-loop/task evaluation',
        'episode':episode,'source_episode':int(ref['source_episode']),'frames':frames,'fps':fps,
        'simulation_seconds':total_ticks/control_fps,'source_seconds':frames/fps,
        'interpolation':interpolation,'interpolation_space':'IK joint waypoints; cubic C1 / quintic C2, zero boundary velocity; quintic also zero boundary acceleration',
        'control_fps':control_fps,'control_samples':len(path['time']),'video_fps':video_fps,'video_frames':rendered_frames,
        'timing':{'schedule':'ceil source frame boundaries to uniform control ticks; no early observations or hidden endpoint jumps',
                  'control_interval_s':1/control_fps,'commands':total_ticks,'initial_state_samples':1,
                  'segment_tick_counts':{str(n):int(np.sum(np.diff(bounds,axis=1).ravel()==n)) for n in np.unique(np.diff(bounds,axis=1))},
                  'max_source_release_delay_s':float(np.max(arrays['segment_start_time']-arrays['source_time'])),
                  'max_control_period_error_s':float(np.max(np.abs(np.diff(path['time'])-1/control_fps))),
                  'last_source_boundary_delay_s':total_ticks/control_fps-frames/fps,
                  'execution':'offline time-indexed command stream; no hardware publisher or real-time deadline guarantee'},
        'base_velocity_nonzero_frames':int(np.any(np.abs(ref['joint_action'][:frames,19:])>1e-8,axis=1).sum()),
        'action_source':'live Cyclo inference' if cached is None else 'frozen actions from '+str(Path(actions_from).resolve()),
        'checkpoint_model_sha256':checkpoint_hash,'action_metadata_sha256':action_metadata_hash,
        'motion_metrics':motion_metrics(path['joint_result'],joint_names,1/control_fps),
        'max_eef_chord_deviation_m':float(path['cartesian_chord_error_m'].max()),
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
    if failures:raise RuntimeError('; '.join(failures))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=300)
    p.add_argument('--episode',type=int,default=0);p.add_argument('--gui',action='store_true')
    p.add_argument('--interpolation',choices=METHODS,default='quintic')
    p.add_argument('--control-fps',type=int,default=100);p.add_argument('--video-fps',type=int,default=100)
    p.add_argument('--actions-from',help='Replay frozen predictions from a verified replay.npz for a fair comparison')
    a=p.parse_args()
    replay(a.dataset,a.checkpoint,a.scene,a.output,a.frames,a.episode,a.gui,
           a.interpolation,a.control_fps,a.video_fps,a.actions_from)
