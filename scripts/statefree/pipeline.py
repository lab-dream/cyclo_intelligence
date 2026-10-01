"""Resumable orchestration; all artifacts remain local or on the specified SSH host."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from core import BASE_COMMIT, CAMERAS, ROOT, URDF, convert, read_json, sha256, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def files_digest(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in str(p)}


def run(command, log=None):
    print('+ '+shlex.join(map(str,command)),flush=True)
    if log:
        with open(log,'w') as f:
            subprocess.run(list(map(str,command)),check=True,stdout=f,stderr=subprocess.STDOUT)
    else:subprocess.run(list(map(str,command)),check=True)


class Pipeline:
    def __init__(self,args):
        self.a=args;self.work=Path(args.work_dir).resolve();self.work.mkdir(parents=True,exist_ok=True)
        self.path=self.work/'manifest.json';self.manifest=read_json(self.path) if self.path.exists() else {'schema':1,'stages':{}}
        self.dataset=self.work/'relative_eef';self.checkpoint=self.work/'checkpoint'
        self.sub=ROOT/'cyclo_brain/policy/lerobot/lerobot'
        self.model_files={p.name:sha256(p) for p in sorted(Path(args.scene).parent.glob('*.xml'))}
        self.code={str(p.relative_to(ROOT)):sha256(p) for p in
            list((ROOT/'scripts/statefree').glob('*.py'))+list((ROOT/'cyclo_brain/policy/lerobot/lerobot_engine').glob('*.py'))+
            [ROOT/'scripts/statefree/lerobot-image-only.patch',URDF]}
        train_paths=['scripts/statefree/policy.py','scripts/statefree/core.py','scripts/statefree/lerobot-image-only.patch']
        self.train_code={p:self.code[p] for p in train_paths}
        self.inference_code={p:h for p,h in self.code.items() if '/lerobot_engine/' in p}
        self.verify_code={**self.train_code,**self.inference_code}
        self.replay_code={**self.verify_code,**{p:self.code[p] for p in
                          ('scripts/statefree/replay.py','scripts/statefree/interpolation.py')}}
        self.common={'lerobot':BASE_COMMIT,'model':self.model_files}

    def stage(self,name,inputs,operation,outputs):
        fingerprint=digest(inputs);old=self.manifest['stages'].get(name)
        if old and old.get('fingerprint')==fingerprint:
            if all(Path(p).is_file() and sha256(p)==h for p,h in old.get('outputs',{}).items()) and old.get('outputs'):
                print(f'REUSE {name}: inputs and output SHA-256 verified',flush=True);return old
        print(f'RUN {name}',flush=True)
        started=time.time()
        try:
            result=operation()
            paths=[]
            for p in outputs:
                p=Path(p)
                paths.extend(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in str(x)) if p.is_dir() else paths.append(p)
            receipt={'status':'PASS','fingerprint':fingerprint,'seconds':time.time()-started,
                     'outputs':{str(p):sha256(p) for p in paths},'result':result}
            self.manifest['stages'][name]=receipt;write_json(self.path,self.manifest);return receipt
        except Exception as exc:
            self.manifest['stages'][name]={'status':'FAIL','fingerprint':fingerprint,'error':str(exc)}
            write_json(self.path,self.manifest);raise

    def source_inputs(self):
        import pyarrow.parquet as pq
        source=Path(self.a.dataset_root).resolve();info=read_json(source/'meta/info.json')
        eps=pq.read_table(source/'meta/episodes').to_pylist()
        selected=list(range(len(eps))) if self.a.episodes=='all' else [int(i) for i in self.a.episodes.split(',')]
        paths={source/'meta/info.json',source/'meta/tasks.parquet'};paths.update((source/'meta/episodes').rglob('*.parquet'))
        for i in selected:
            ep=eps[i];paths.add(source/info['data_path'].format(chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index']))
            for cam in CAMERAS:
                paths.add(source/info['video_path'].format(video_key=cam,chunk_index=ep[f'videos/{cam}/chunk_index'],file_index=ep[f'videos/{cam}/file_index']))
        return {str(p):sha256(p) for p in sorted(paths)}

    def prepare(self):
        if self.dataset.exists():
            self.dataset.rename(self.work/f'relative_eef.previous-{time.time_ns()}')
        return convert(self.a.dataset_root,self.dataset,self.a.episodes,self.a.scene)['validation']

    def verify(self):
        run([sys.executable,ROOT/'scripts/statefree/policy.py','verify','--dataset',self.dataset,'--output',self.work/'verification'],self.work/'verify.log')
        return read_json(self.work/'verification/policy_tests.json')

    def ssh(self,command,log=None):
        run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',self.a.ssh_host,command],log)

    def train(self):
        remote=self.a.remote_dir.rstrip('/');code=remote+'/code';python=self.a.remote_python
        # Isolated source checkout: verify the actual pinned Git base before applying the patch.
        self.ssh('mkdir -p '+shlex.quote(code+'/scripts/statefree')+' '+shlex.quote(code+'/cyclo_brain/policy/lerobot'))
        sub=code+'/cyclo_brain/policy/lerobot/lerobot'
        self.ssh(f'if [ ! -d {shlex.quote(sub+"/.git")} ]; then '
                 f'git clone --no-checkout https://github.com/ROBOTIS-GIT/lerobot-cyclo.git {shlex.quote(sub+".clone")}; '
                 f'git -C {shlex.quote(sub+".clone")} checkout {BASE_COMMIT}; '
                 f'if [ -d {shlex.quote(sub)} ]; then mv {shlex.quote(sub)} {shlex.quote(sub+".previous-"+str(time.time_ns()))}; fi; '
                 f'mv {shlex.quote(sub+".clone")} {shlex.quote(sub)}; fi')
        run(['rsync','-az','--exclude=__pycache__',str(ROOT/'scripts/statefree')+'/',f'{self.a.ssh_host}:{code}/scripts/statefree/'])
        self.ssh(f'bash {shlex.quote(code+"/scripts/statefree/bootstrap.sh")}')
        # A hash-specific remote output prevents mixing a stale checkpoint with new inputs.
        train_key=digest({'code':self.train_code,'data':files_digest(self.dataset),'seconds':self.a.train_seconds})[:16]
        remote_run=remote+'/runs/'+train_key
        self.ssh('mkdir -p '+shlex.quote(remote_run))
        remote_dataset=remote_run+'/relative_eef'
        run(['rsync','-azL',str(self.dataset)+'/',f'{self.a.ssh_host}:{remote_dataset}/'])
        env=f'cd {shlex.quote(code)} && PYTHONPATH=cyclo_brain/policy/lerobot/lerobot/src:scripts/statefree HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES={shlex.quote(self.a.gpu)} '
        progress_path=self.work/'remote_attempt.json'
        progress=read_json(progress_path) if progress_path.exists() else {}
        if progress.get('train_key') != train_key:
            progress={'train_key':train_key,'attempt':str(time.time_ns())}
        attempt=progress['attempt']
        def launch(output,steps,phase):
            output=progress.get(phase,output)
            def status():
                script=('import json,pathlib\np=pathlib.Path('+repr(output)+')\ncmds=[]\n'
                        'for f in pathlib.Path("/proc").glob("[0-9]*/cmdline"):\n'
                        ' try: cmds.append(f.read_bytes().split(bytes([0])))\n'
                        ' except (FileNotFoundError,PermissionError,ProcessLookupError): pass\n'
                        'print(json.dumps({"complete":(p/"training_evidence.json").is_file(),"exists":p.exists(),'
                        '"running":any(b"scripts/statefree/policy.py" in c and str(p).encode() in c for c in cmds)}))')
                return json.loads(subprocess.check_output(['ssh',self.a.ssh_host,shlex.join([python,'-c',script])],text=True))
            state=status()
            while state['running'] and not state['complete']:
                print('Waiting for this run\'s existing remote trainer: '+output,flush=True)
                time.sleep(5);state=status()
            if not state['complete']:
                if state['exists']:
                    output += '.retry-'+str(time.time_ns())
                progress[phase]=output;write_json(progress_path,progress)
                command=[python,'scripts/statefree/policy.py','train','--dataset',remote_dataset,'--output',output,'--steps',str(steps)]
                self.ssh(env+shlex.join(command),self.work/(phase+'.log'))
            else:
                print('REUSE completed remote '+phase+': '+output,flush=True)
            run(['rsync','-az',f'{self.a.ssh_host}:{output}/training_evidence.json',str(self.work/(phase+'.json'))])
        launch(remote_run+'/calibration-'+attempt,100,'calibration')
        calibration=read_json(self.work/'calibration.json');metrics=calibration['metrics']
        # Exclude the first 20 optimizer steps (CUDA autotuning); retain measured startup separately.
        start=next(m for m in metrics if m['message'].startswith('step:20 '));last=metrics[-1]
        seconds_per_step=(last['elapsed']-start['elapsed'])/80
        startup=max(0.,start['elapsed']-20*seconds_per_step)
        steps=max(10,round((self.a.train_seconds-startup)/seconds_per_step))
        write_json(self.work/'step_estimate.json',{'target_seconds':self.a.train_seconds,'steps':steps,'seconds_per_step':seconds_per_step,'startup_seconds':startup})
        output=remote_run+'/train-'+attempt;launch(output,steps,'training')
        result=read_json(self.work/'training.json')
        if self.checkpoint.exists():self.checkpoint.rename(self.work/f'checkpoint.previous-{time.time_ns()}')
        run(['rsync','-az',f'{self.a.ssh_host}:{result["checkpoint"]}/',str(self.checkpoint)+'/'])
        # Compare checkpoint bytes on both machines, including processors and metadata.
        script='import hashlib,json,pathlib; p=pathlib.Path('+repr(result['checkpoint'])+'); print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob("*") if f.is_file()}))'
        data=subprocess.check_output(['ssh',self.a.ssh_host,shlex.join([python,'-c',script])],text=True)
        assert json.loads(data)==files_digest(self.checkpoint)
        write_json(self.work/'checkpoint_sha256.json',json.loads(data))
        expected=sha256(self.sub/'src/lerobot/policies/diffusion/modeling_diffusion.py')
        assert result['diffusion_code_sha256']==expected
        return result

    def replay(self):
        run([sys.executable,ROOT/'scripts/statefree/policy.py','reload','--dataset',self.dataset,'--output',self.checkpoint],self.work/'checkpoint_reload.log')
        command=[sys.executable,ROOT/'scripts/statefree/replay.py','--dataset',self.dataset,'--checkpoint',self.checkpoint,
                 '--scene',self.a.scene,'--output',self.work/'replay','--frames',self.a.frames,'--episode',self.a.episode,
                 '--interpolation',self.a.interpolation,'--control-fps',self.a.control_fps,'--video-fps',self.a.video_fps]
        if self.a.gui:command.append('--gui')
        run(command,self.work/'replay.log');return read_json(self.work/'replay/replay_report.json')

    def preflight(self):
        import torch
        import mujoco
        if not Path(self.a.dataset_root, 'meta/info.json').is_file():
            raise FileNotFoundError('Missing local dataset metadata')
        if not Path(self.a.scene).is_file():
            raise FileNotFoundError('Missing FFW-SG2 MJCF; specify --scene')
        if not shutil.which('ffmpeg'):
            raise FileNotFoundError('ffmpeg is required for the replay video')
        report={'python':sys.executable,'torch':torch.__version__,'mujoco':mujoco.__version__,
                'cuda':torch.cuda.is_available(),'free_bytes':shutil.disk_usage(self.work).free}
        if not self.a.replay_only:
            script='import torch,json; assert torch.cuda.is_available(); print(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"devices":torch.cuda.device_count()}))'
            command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',self.a.ssh_host,
                     shlex.join([self.a.remote_python,'-c',script])]
            report['server']=json.loads(subprocess.check_output(command,text=True))
        write_json(self.work/'environment_check.json',report)

    def execute(self):
        self.preflight()
        source=self.source_inputs()
        self.stage('convert',{'source':source,'episodes':self.a.episodes,'core':self.code['scripts/statefree/core.py'],'model':self.model_files,'urdf':sha256(URDF)},self.prepare,[self.dataset])
        data=files_digest(self.dataset)
        self.stage('verify',{'data':data,'code':self.verify_code},self.verify,[self.work/'verification/policy_tests.json'])
        if not self.a.replay_only:
            self.stage('train',{'data':data,'code':self.train_code,'seconds':self.a.train_seconds,'host':self.a.ssh_host,'python':self.a.remote_python,'gpu':self.a.gpu},self.train,[self.checkpoint,self.work/'training.json',self.work/'checkpoint_sha256.json'])
        elif not (self.checkpoint/'model.safetensors').exists():raise FileNotFoundError('No retrieved checkpoint; run the complete pipeline first')
        inputs={'checkpoint':files_digest(self.checkpoint),'data':data,'code':self.replay_code,'model':self.common,'frames':self.a.frames,
                'episode':self.a.episode,'interpolation':self.a.interpolation,'control_fps':self.a.control_fps,'video_fps':self.a.video_fps}
        if self.a.gui:self.manifest['stages'].pop('replay',None)
        self.stage('replay',inputs,self.replay,[self.work/'replay'])
        summary={k:{'status':v['status'],'result':v.get('result')} for k,v in self.manifest['stages'].items()}
        write_json(self.work/'summary.json',summary);print('Results: '+str(self.work/'summary.json'),flush=True)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--ssh-host',default='gpuserver')
    p.add_argument('--train-seconds',type=float,default=300.0);p.add_argument('--sim',choices=['mujoco'],default='mujoco')
    p.add_argument('--episodes',default='0,1');p.add_argument('--frames',type=int,default=300)
    p.add_argument('--episode',type=int,default=0,help='Converted dataset episode index for replay')
    p.add_argument('--work-dir',default=str(ROOT.parent/'statefree_smoke'));p.add_argument('--scene',default='/home/son/Downloads/AI_Worker_Practice/third_party/robotis_mujoco_menagerie/robotis_ffw/scene_ffw_sg2.xml')
    p.add_argument('--remote-dir',default='/data/son_statefree_smoke');p.add_argument('--remote-python',default='/data/son_statefree_smoke/venv/bin/python')
    p.add_argument('--gpu',default='0');p.add_argument('--replay-only',action='store_true');p.add_argument('--gui',action='store_true')
    p.add_argument('--interpolation',choices=['none','cubic','quintic'],default='quintic')
    p.add_argument('--control-fps',type=int,default=100);p.add_argument('--video-fps',type=int,default=100)
    return p


if __name__=='__main__':Pipeline(parser().parse_args()).execute()
