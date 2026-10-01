"""Run and retain a reproducible multi-episode interpolation comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from core import read_json, sha256, write_json
from interpolation import control_bounds


def validate_case(case, control_fps):
    endpoints={mode:np.load(case/mode/'replay.npz') for mode in ('none','cubic','quintic')}
    reports={mode:read_json(case/mode/'replay_report.json') for mode in endpoints}
    base=endpoints['none']; checks={}
    for mode,endpoint in endpoints.items():
        report=reports[mode];path=np.load(case/mode/'trajectory.npz')
        for key in ('raw_action','normalized_action','applied_action','target','qpos','ik_ok','achieved'):
            if not np.array_equal(base[key],endpoint[key]):raise AssertionError(f'{mode}: different {key}')
        bounds=np.array([control_bounds(i,report['fps'],control_fps) for i in range(report['frames'])])
        assert np.array_equal(path['control_tick'],np.arange(bounds[-1,1]+1))
        np.testing.assert_allclose(np.diff(path['time']),1/control_fps,rtol=0,atol=2e-14)
        assert np.array_equal(path['joint_result'][bounds[:,1]],endpoint['joint_result'])
        assert np.all(endpoint['segment_start_time']>=endpoint['source_time']-1e-14)
        for frame,(start,end) in enumerate(bounds):
            previous=path['joint_result'][start];goal=endpoint['joint_result'][frame]
            segment=path['joint_result'][start+1:end+1]
            assert np.isfinite(segment).all()
            assert np.all(segment>=np.minimum(previous,goal)-1e-12)
            assert np.all(segment<=np.maximum(previous,goal)+1e-12)
            assert np.all(path['policy_frame'][start+1:end+1]==frame)
            if not endpoint['ik_ok'][frame]:assert np.array_equal(segment,np.repeat(previous[None],end-start,axis=0))
        csv=np.loadtxt(case/mode/'commands.csv',delimiter=',',skiprows=1)
        assert len(csv)==bounds[-1,1]
        np.testing.assert_allclose(csv[:,0],path['time'][1:],atol=5e-10,rtol=0)
        np.testing.assert_allclose(csv[:,3:],path['joint_result'][1:],atol=6e-13,rtol=0)
        checks[mode]={'uniform_control_clock':True,'no_early_observation':True,'waypoints_on_sent_ticks':True,
                      'identical_predictions_and_ik':True,'no_joint_overshoot':True,'ik_failure_hold':True,
                      'csv_matches_commands':True}
    return {'status':'PASS' if all(r['status']=='PASS' for r in reports.values()) else 'FAIL',
            'source_episode':reports['none']['source_episode'],'episode':reports['none']['episode'],
            'checks':checks,'reports':reports}


def comparison_video(case, slow=False):
    command=['ffmpeg','-hide_banner','-loglevel','error','-y']
    modes=('none','cubic','quintic')
    for mode in modes:command+=['-i',str(case/mode/'replay.mp4')]
    filters=[]
    for i,(name,color) in enumerate(zip(('NONE - STEP','CUBIC - C1','QUINTIC - C2'),('142b36','087f73','4365b7'))):
        filters.append(f'[{i}:v]crop=960:720:0:0,scale=640:480,setpts={4 if slow else 1}*PTS,'
                       f'drawbox=x=0:y=0:w=iw:h=38:color=0x{color}:t=fill,'
                       f"drawtext=text='{name}':x=14:y=8:fontsize=22:fontcolor=white[v{i}]")
    filters.append('[v0][v1][v2]hstack=inputs=3[v]')
    fps=read_json(case/'none/replay_report.json')['video_fps']
    output=case/('comparison_slow.mp4' if slow else 'comparison_normal.mp4')
    command+=['-filter_complex',';'.join(filters),'-map','[v]','-r',str(fps/4 if slow else fps),'-fps_mode','cfr',
              '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True)
    return {'file':str(output.resolve()),'sha256':sha256(output),'speed':.25 if slow else 1.}


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','checkpoint','scene','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--source-episodes',default='0,1,56,170,368')
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--control-fps',type=int,default=100);p.add_argument('--video-fps',type=int,default=100)
    args=p.parse_args();output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    metadata=read_json(Path(args.dataset)/'meta/action_representation.json')
    selected=[int(x) for x in args.source_episodes.split(',')]
    if len(set(selected))!=len(selected):raise ValueError('Source episodes must be unique')
    for source in selected:metadata['source_episodes'].index(source)
    output.mkdir(parents=True,exist_ok=True)
    summary={'status':'RUNNING','source_episodes':selected,'control_fps':args.control_fps,
             'checkpoint_model_sha256':sha256(Path(args.checkpoint)/'model.safetensors'),
             'checkpoint':str(Path(args.checkpoint).resolve()),'cases':[]}
    write_json(output/'suite.json',summary)
    for source in selected:
        episode=metadata['source_episodes'].index(source)
        case=output/f'episode_{source:06d}';case.mkdir(exist_ok=True)
        print(f'SOURCE EPISODE {source}, converted index {episode}',flush=True)
        for mode in ('none','cubic','quintic'):
            destination=case/mode
            if destination.exists():raise FileExistsError(f'Use a fresh output directory: {destination}')
            command=[sys.executable,str(Path(__file__).with_name('replay.py')),'--dataset',args.dataset,
                     '--checkpoint',args.checkpoint,'--scene',args.scene,'--output',str(destination),
                     '--episode',str(episode),'--frames',str(args.frames),'--interpolation',mode,
                     '--control-fps',str(args.control_fps),'--video-fps',str(args.video_fps)]
            if mode!='none':command+=['--actions-from',str(case/'none/replay.npz')]
            with (case/(mode+'.log')).open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            report_path=destination/'replay_report.json'
            if completed.returncode and not report_path.exists():
                raise RuntimeError(f'{mode} failed before producing evidence; see {case/(mode+".log")}')
            status=read_json(report_path)['status']
            if completed.returncode and status!='FAIL':
                raise RuntimeError(f'{mode} exited {completed.returncode} despite {status} report; see {case/(mode+".log")}')
            print(f'  {mode}: {status}, exit {completed.returncode}',flush=True)
        result=validate_case(case,args.control_fps)
        result['videos']=[comparison_video(case),comparison_video(case,slow=True)]
        write_json(case/'comparison.json',result)
        summary['cases'].append(result);write_json(output/'suite.json',summary)
    summary['status']='PASS' if all(c['status']=='PASS' for c in summary['cases']) else 'FAIL'
    write_json(output/'suite.json',summary)
    print(f'{summary["status"]}: {len(summary["cases"])} episodes; '+str(output/'suite.json'),flush=True)
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
