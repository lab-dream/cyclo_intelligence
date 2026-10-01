"""Freeze one Cyclo prediction stream per episode, then replay ERJ ON and OFF."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from core import checked_action_spec, read_json, sha256, write_json
from interpolation import control_bounds


def validate_pair(case):
    arrays={mode:np.load(case/mode/'replay.npz') for mode in ('on','off')}
    reports={mode:read_json(case/mode/'replay_report.json') for mode in arrays}
    for key in ('raw_action','normalized_action','initial_qpos','source_time','chunk_index'):
        np.testing.assert_array_equal(arrays['on'][key],arrays['off'][key])
    for mode,log in arrays.items():
        report=reports[mode];path=np.load(case/mode/'trajectory.npz')
        spec=report['action_spec'];assert spec['action_mode']=='erj' and log['raw_action'].shape[1]==19
        bounds=np.array([control_bounds(i,report['fps'],report['control_fps']) for i in range(report['frames'])])
        np.testing.assert_array_equal(path['control_tick'],np.arange(bounds[-1,1]+1))
        np.testing.assert_allclose(np.diff(path['time']),1/report['control_fps'],atol=2e-14,rtol=0)
        np.testing.assert_array_equal(path['joint_result'][bounds[:,1]],log['joint_result'])
        previous=path['joint_result'][bounds[:,0]]
        failed=~log['ik_ok']
        np.testing.assert_array_equal(log['joint_result'][failed],previous[failed])
        for start,end in bounds[failed]:
            np.testing.assert_array_equal(path['joint_result'][start+1:end+1],np.repeat(path['joint_result'][start][None],end-start,axis=0))
        if mode=='on':
            selected=[log['joint_names'].tolist().index(n) for n in spec['erj_joints']]
            np.testing.assert_array_equal(log['joint_result'][log['ik_ok']][:,selected],log['raw_action'][log['ik_ok'],17:])
        assert np.isfinite(path['elbow_position']).all()
    return {'status':'PASS' if all(r['status']=='PASS' for r in reports.values()) else 'FAIL',
            'same_predictions':True,'same_initial_qpos':True,'hard_constraints_on_accepted_targets':True,
            'failed_segments_hold_previous_pose':True,'uniform_clock':True,'reports':reports,
            'note':'Per-step relative EEF actions are identical; absolute EEF targets can diverge as ON/OFF poses diverge. No task-success claim.'}


def video(case,slow=False):
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y']
    for mode in ('on','off'):cmd+=['-i',str(case/mode/'replay.mp4')]
    filters=[]
    for i,(label,color) in enumerate((('ERJ ON - FIXED JOINT IK','087f73'),('ERJ OFF - EEF IK','4365b7'))):
        filters.append(f'[{i}:v]scale=960:540,setpts={4 if slow else 1}*PTS,'
                       f'drawbox=x=0:y=0:w=iw:h=29:color=0x{color}:t=fill,'
                       f"drawtext=text='{label}':x=12:y=5:fontsize=19:fontcolor=white[v{i}]")
    filters.append('[v0][v1]hstack=inputs=2[v]')
    fps=read_json(case/'on/replay_report.json')['video_fps']
    dest=case/('comparison_slow.mp4' if slow else 'comparison_normal.mp4')
    cmd+=['-filter_complex',';'.join(filters),'-map','[v]','-r',str(fps/4 if slow else fps),'-fps_mode','cfr',
          '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    subprocess.run(cmd,check=True)
    return {'file':str(dest.resolve()),'sha256':sha256(dest),'speed':.25 if slow else 1.}


def main():
    p=argparse.ArgumentParser()
    for key in ('dataset','checkpoint','scene','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--episodes',default='0,1,2',help='Converted dataset indices')
    p.add_argument('--frames',type=int,default=300)
    p.add_argument('--control-fps',type=int,default=100);p.add_argument('--video-fps',type=int,default=100)
    a=p.parse_args();output=Path(a.output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh output directory')
    spec=checked_action_spec(read_json(Path(a.checkpoint)/'action_representation.json'))
    if spec['action_mode']!='erj':raise ValueError('ON/OFF comparison requires a separately trained ERJ checkpoint')
    selected=[int(x) for x in a.episodes.split(',')]
    if len(set(selected))!=len(selected):raise ValueError('Duplicate episodes')
    output.mkdir(parents=True,exist_ok=True);summary={'status':'RUNNING','action_spec':spec,'cases':[]}
    write_json(output/'suite.json',summary)
    for episode in selected:
        case=output/f'episode_{episode:06d}';case.mkdir()
        for mode in ('on','off'):
            cmd=[sys.executable,str(Path(__file__).with_name('replay.py')),'--dataset',a.dataset,
                 '--checkpoint',a.checkpoint,'--scene',a.scene,'--output',str(case/mode),
                 '--episode',str(episode),'--frames',str(a.frames),'--action-mode','erj','--elbow-control',mode,
                 '--control-fps',str(a.control_fps),'--video-fps',str(a.video_fps)]
            if mode=='off':cmd+=['--actions-from',str(case/'on/replay.npz')]
            with (case/(mode+'.log')).open('w') as log:process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            report=case/mode/'replay_report.json'
            if not report.exists() or (process.returncode and read_json(report)['status']!='FAIL'):
                raise RuntimeError(f'Episode {episode} {mode} crashed; see {case/(mode+".log")}')
            print(f'Episode {episode} {mode}: {read_json(report)["status"]}',flush=True)
        result=validate_pair(case);result['videos']=[video(case),video(case,True)]
        write_json(case/'comparison.json',result);summary['cases'].append(result);write_json(output/'suite.json',summary)
    summary['status']='PASS' if all(c['status']=='PASS' for c in summary['cases']) else 'FAIL'
    write_json(output/'suite.json',summary);print(summary['status'],flush=True)
    if summary['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
