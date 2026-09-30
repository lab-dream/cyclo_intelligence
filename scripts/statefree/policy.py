"""Configuration, real-data checks, and the existing LeRobot training entry point."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from core import BASE_COMMIT, CAMERAS, LAYOUT, read_json, sha256, write_json
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors, make_policy
from lerobot.utils.random_utils import set_seed

SEED = 42


def config(dataset, output, steps=20, device=None):
    info = read_json(Path(dataset)/'meta/info.json')
    inputs = {k:PolicyFeature(type=FeatureType.VISUAL, shape=(3, *info['features'][k]['shape'][:2])) for k in CAMERAS}
    policy = DiffusionConfig(input_features=inputs,
        output_features={'action':PolicyFeature(type=FeatureType.ACTION, shape=(len(LAYOUT),))},
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu'), push_to_hub=False,
        n_obs_steps=2, horizon=16, n_action_steps=8, drop_n_last_frames=0,
        resize_shape=(96,96), crop_ratio=1., pretrained_backbone_weights=None,
        use_group_norm=True, use_separate_rgb_encoder_per_camera=False,
        down_dims=(64,128,256), diffusion_step_embed_dim=64, spatial_softmax_num_keypoints=16,
        num_train_timesteps=100, num_inference_steps=10, noise_scheduler_type='DDIM',
        do_mask_loss_for_padding=True, scheduler_warmup_steps=5)
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/statefree-smoke', root=str(Path(dataset).resolve()),
                               video_backend='pyav', use_imagenet_stats=True),
        policy=policy, output_dir=Path(output), steps=steps, batch_size=4, num_workers=2,
        log_freq=5, save_freq=steps, seed=SEED, wandb=WandBConfig(enable=False), env_eval_freq=0,
        eval_steps=0, save_checkpoint=True, save_checkpoint_to_hub=False)


def weights_digest(model):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify(dataset, out):
    """Actual images, episode-edge masks, backward, queues and state invariance."""
    torch.set_num_threads(4)
    cfg = config(dataset, Path(out)/'unused')
    ds = make_dataset(cfg)
    assert 'observation.state' not in ds.meta.features
    sample = ds[0]
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
    assert all(batch[k].shape == (2,2,3,240,424) for k in CAMERAS)
    episode_end = int(ds.meta.episodes[0]['dataset_to_index'])-1
    last = ds[episode_end]
    following = ds[episode_end+1]
    assert not last['action_is_pad'][1] and last['action_is_pad'][2:].all()
    assert following['action_is_pad'][0] and not following['action_is_pad'][1]
    assert sample['action_is_pad'][0] and not sample['action_is_pad'][1]
    for k in CAMERAS:
        batch[k] = batch[k].float()/255 if batch[k].dtype == torch.uint8 else batch[k]
    set_seed(SEED)
    p = make_policy(cfg.policy, ds_meta=ds.meta)
    assert set(p.config.input_features) == set(CAMERAS)
    pre, post = make_pre_post_processors(p.config, dataset_stats=ds.meta.stats)
    batch = pre(batch)
    training_batch = {k:batch[k] for k in CAMERAS+['action','action_is_pad']}
    assert not any('state' in k for k in training_batch)
    restored = post(batch['action'])
    original = torch.stack([ds[i]['action'] for i in range(2)])
    normalization_error = float((restored.cpu()-original).abs().max())
    assert normalization_error < 2e-6
    before = weights_digest(p)
    optim = torch.optim.Adam(p.parameters(), lr=1e-4)
    loss, _ = p(training_batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
    optim.step()
    assert weights_digest(p) != before
    p.eval()
    noise = torch.randn(2, p.config.horizon, len(LAYOUT), device=p.config.device)
    images = {k:batch[k] for k in CAMERAS}
    with torch.inference_mode():
        a = p.predict_action_chunk(images, noise=noise.clone())
        b = p.predict_action_chunk({**images,'observation.state':torch.randn(2,2,22,device=p.config.device)*1000}, noise=noise.clone())
        c = p.predict_action_chunk({**images,'observation.state':torch.zeros(1,1,1,device=p.config.device)}, noise=noise.clone())
    assert torch.equal(a,b) and torch.equal(a,c)
    assert 'observation.state' not in p._queues
    p.reset()
    with torch.inference_mode():
        for _ in range(18):
            action = p.select_action({k:v[:, -1] for k,v in images.items()})
            assert action.shape == (2,len(LAYOUT)) and torch.isfinite(action).all()
    p.reset()
    checkpoint = Path(out)/'reload_test'
    p.save_pretrained(checkpoint); pre.save_pretrained(checkpoint); post.save_pretrained(checkpoint)
    shutil.copy2(Path(dataset)/'meta/action_representation.json', checkpoint/'action_representation.json')
    # A separate interpreter exercises Cyclo's true load/processor/prediction path.
    subprocess.run([sys.executable, __file__, 'reload', '--dataset', str(dataset), '--output',str(checkpoint)], check=True)
    # Existing state-based Diffusion remains supported (forward/backward + select_action).
    state_cfg = config(dataset, Path(out)/'unused_state').policy
    state_cfg.input_features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(22,))
    state_policy = DiffusionPolicy(state_cfg).to(state_cfg.device)
    state_batch = {**training_batch, 'observation.state':torch.randn(2,2,22,device=state_cfg.device)}
    state_loss,_ = state_policy(state_batch); state_loss.backward()
    assert torch.isfinite(state_loss)
    state_policy.eval()
    with torch.inference_mode():
        old_path = state_policy.select_action({**{k:v[:,-1] for k,v in images.items()},'observation.state':state_batch['observation.state'][:,-1]})
    assert old_path.shape == (2,len(LAYOUT))
    report = {'status':'PASS','loss':float(loss.detach()),'state_based_loss':float(state_loss.detach()),
        'normalization_max_error':normalization_error,'state_invariance_exact':True,
        'fresh_process_cyclo_reload':True,'episode_padding':True,'input_keys':list(p.config.input_features),
        'input_shapes':{k:list(batch[k].shape) for k in CAMERAS},'parameters':sum(v.numel() for v in p.parameters())}
    write_json(Path(out)/'policy_tests.json',report)
    print(json.dumps(report),flush=True)


def reload_test(dataset, checkpoint):
    from lerobot_engine import LeRobotEngine
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('local/statefree-smoke',root=dataset,video_backend='pyav')
    engine = LeRobotEngine(); engine.load_recorded_policy(checkpoint)
    engine.observe_recorded(ds[0]); engine.observe_recorded(ds[1])
    torch.manual_seed(42); a = engine.predict_recorded_chunk()
    assert a['action'].shape == (8,len(LAYOUT)) and np.isfinite(a['action']).all()
    engine.reset_recorded_episode()
    for i in range(2):
        engine.observe_recorded({**ds[i],'observation.state':torch.randn(22)*999})
    torch.manual_seed(42); b = engine.predict_recorded_chunk()
    assert np.array_equal(a['action'],b['action'])
    print('Fresh-process Cyclo load/preprocess/predict and state exclusion: PASS',flush=True)


class TrainingEvidence(logging.Filter):
    def __init__(self):
        super().__init__(); self.start=None; self.end=None; self.metrics=[]
    def filter(self,record):
        msg=record.getMessage()
        if msg.startswith('Start offline training'):
            self.start=time.perf_counter()
        if msg.startswith('Checkpoint policy after step'):
            self.end=time.perf_counter()
        if msg.startswith('step:'):
            self.metrics.append({'elapsed':time.perf_counter()-(self.start or time.perf_counter()),'message':msg})
        return True


def train_once(dataset, output, steps):
    from lerobot.scripts.lerobot_train import train
    from lerobot.utils.utils import init_logging
    torch.set_num_threads(4)
    cfg = config(dataset, output, steps)
    # Same seed/model constructor as LeRobot's trainer, recorded before updates.
    set_seed(SEED)
    initial = DiffusionPolicy(cfg.policy)
    initial_hash = weights_digest(initial)
    del initial
    init_logging()
    evidence=TrainingEvidence(); logging.getLogger().addFilter(evidence)
    process_start=time.perf_counter()
    train(cfg)
    checkpoints = [p for p in (Path(output)/'checkpoints').glob('*/pretrained_model') if p.parent.name != 'last']
    checkpoint=max(checkpoints,key=lambda p:int(p.parent.name))
    trained = DiffusionPolicy.from_pretrained(checkpoint)
    final_hash=weights_digest(trained)
    assert final_hash != initial_hash
    assert all(torch.isfinite(v).all() for v in trained.state_dict().values())
    metadata=Path(dataset)/'meta/action_representation.json'
    shutil.copy2(metadata,checkpoint/'action_representation.json')
    assert evidence.start is not None and evidence.end is not None
    losses=[float(re.search(r'loss:([^ ]+)', row['message']).group(1)) for row in evidence.metrics]
    assert losses and np.isfinite(losses).all()
    training_state=read_json(checkpoint.parent/'training_state/training_step.json')
    assert training_state['step']==steps
    from safetensors.torch import load_file
    optimizer_state=load_file(checkpoint.parent/'training_state/optimizer_state.safetensors')
    assert any('exp_avg' in k and torch.count_nonzero(v) > 0 for k,v in optimizer_state.items())
    report={'status':'PASS','steps':steps,'loss_first_logged':losses[0],'loss_last_logged':losses[-1],
            'finite_losses':True,'optimizer_updates_verified':True,'training_seconds':evidence.end-evidence.start,
            'process_seconds':time.perf_counter()-process_start,'initial_weights_sha256':initial_hash,
            'final_weights_sha256':final_hash,'weights_changed':True,'metrics':evidence.metrics,
            'checkpoint':str(checkpoint.resolve()),'seed':SEED,'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(), 'lerobot_import':__import__('lerobot').__file__,
            'diffusion_code_sha256':sha256(Path(__import__('lerobot.policies.diffusion.modeling_diffusion',fromlist=['x']).__file__)),
            'lerobot_base_commit':BASE_COMMIT}
    write_json(Path(output)/'training_evidence.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','train','reload'])
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True);parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
    if args.mode=='verify':verify(args.dataset,args.output)
    elif args.mode=='reload':reload_test(args.dataset,args.output)
    else:train_once(args.dataset,args.output,args.steps)
