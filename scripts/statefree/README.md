# State-free Diffusion / FFW-SG2 smoke test

This is a recorded-observation inference and MuJoCo **kinematic** replay. It
verifies a real-data training/checkpoint/inference/IK/rendering connection. It
is not a closed-loop rollout, a dynamics test, or a task-success evaluation.
No robot client, hardware bringup, or external model/dataset upload is used.

From the repository root on the tested PC:

```bash
./scripts/run_statefree_smoke.sh \
  --dataset-root /media/son/Remember/AX_Humanoid/dataset/merge_0616_to_0819_v3 \
  --ssh-host gpuserver --train-seconds 300 --sim mujoco
```

To open the trained replay in the MuJoCo GUI, without server training:

```bash
MUJOCO_GL=glfw ./scripts/run_statefree_smoke.sh \
  --dataset-root /media/son/Remember/AX_Humanoid/dataset/merge_0616_to_0819_v3 \
  --replay-only --gui
```

Default artifacts: `../statefree_smoke/`. The manifest checks the selected
source data/videos, relevant code, settings, and output SHA-256 before reuse.
Failed stages have no successful receipt. A recorded remote attempt can resume
checkpoint retrieval without repeating completed training. Changed conversion/checkpoint outputs
are retained as `.previous-*`; training attempts use separate server paths.
Use a new `--work-dir` for an entirely fresh run. `--episodes all` converts the
full dataset, episode by episode; `--episodes 0,1` is the default small subset.
The wrapper never commits or pushes automatically.

Conversion checks episode frame indices, timestamps, dataset spans and both
camera intervals before processing a row. A stale episode `length` is corrected
only in the output metadata when those independent counts agree; the correction
is recorded in `meta/action_representation.json`. Output global indices are
regenerated, with inconsistent source indices recorded. A mismatched video span
is rejected rather than reading images from a following episode.

The full `merge_0616_to_0819_v3` audit found stale lengths in source episodes
160–168 and inconsistent global indices in 125 and 170. Source episode 169 has
1,663 data rows but a 50-frame camera interval, so it must be repaired from the
original recordings or explicitly excluded. The extended 200,000-step run uses
the other 368 episodes (224,956 frames), retaining the original files. Therefore
`--episodes all` correctly rejects episode 169 in this source dataset.

## Environment and managed patch

The parent repository pins LeRobot to
`240b4a0314ae0879cdd928c7f4bdc1eee9a01b3b`. Run
`scripts/statefree/bootstrap.sh` before importing it. The script refuses another
base or unexpected Diffusion edits, and detects an already applied patch.
The submodule Git link stays at the pinned commit; its two modified files are
reproduced by the parent-owned patch, not by a new submodule commit.

`STATEFREE_PYTHON` overrides the local interpreter (default
`../.statefree-venv/bin/python`, otherwise `python3`). `--remote-python`,
`--remote-dir`, `--gpu`, and `--scene` override the tested server/model paths.
Dependencies are the pinned checkout's Diffusion/training extras, plus SciPy,
MuJoCo, pytest, and ffmpeg. The wrapper reuses prepared environments and does
not install or upgrade packages automatically.

The actual local environment uses Python 3.12.13, PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, MuJoCo 3.7.0, NumPy 2.2.6, datasets 4.8.5, and
diffusers 0.39.0. It inherits existing packages from
`/home/son/miniconda3/envs/lerobot` and the existing user installation; only
missing test/training helpers were installed into the separate venv.

The server environment `/data/son_statefree_smoke/venv` uses Python 3.12.13 and
an explicit `.pth` referencing the existing
`/data/dw_ws/lerobot/.venv/lib/python3.12/site-packages` (PyTorch 2.11.0+cu128).
`PYTHONPATH` selects the isolated, patched LeRobot source before that environment's
LeRobot. Other users' environments and processes are unchanged. The source
module SHA-256 matches locally, on the GPU server, and in the source volume
mounted into the running Cyclo orchestration container. No LeRobot inference
container was running; the orchestration container's hardware services were
not started or restarted.

## Data and action contract

- Original: 369 episodes, 226,619 frames, 30 FPS, LeRobot v3.
- Tested subset: source episodes 0 and 1, 490 + 616 = 1,106 frames (0.488% of
  frames). This limits video transfer and conversion work for the smoke test.
- Inputs: `observation.images.cam_wrist_left` and
  `observation.images.cam_wrist_right` only. Both retain their real 240×424 RGB
  metadata/videos. The shared ResNet18 encoder resizes each to 96×96 using the
  same policy transform in training and inference. No camera rotation is added
  to already recorded frames. Two image observations form the history.
- Horizon 16, executed chunk length 8, DDIM inference 10 steps, shared image
  encoder, U-Net widths 64/128/256, batch 4, seed 42, warmup 5 updates.
- The 22 source names are read and checked in their recorded order. Current
  robot configuration and recording converter identify measured JointState
  positions and JointTrajectory target positions. Mobile columns are odometry
  and command velocities, not joint angles or base poses. Same-row source
  alignment is preserved; original bag timestamps/converter provenance were
  unavailable, so historical sub-frame latency cannot be reconstructed.

The **17 output dimensions**, in order, are:

| Indices | Values | Units |
|---|---|---|
| 0–5 | left EEF body translation + SO(3) rotation vector | m, rad |
| 6 | left gripper absolute joint target | rad |
| 7–12 | right EEF body translation + SO(3) rotation vector | m, rad |
| 13 | right gripper absolute joint target | rad |
| 14–15 | head_joint1, head_joint2 absolute targets | rad |
| 16 | lift_joint absolute target | m |

`Delta_t = inverse(FK(state_t)) @ FK(action_t)`. Replay composes each action
once with the **current simulated** EEF pose. This is per-step/body-frame
relative control, not chunk-origin control. Rotation vectors are converted
through SO(3), never subtracted as Euler angles. There are no quaternion action
components. Each new policy chunk uses the two latest recorded images; no
future image/action enters the policy. The adapter owns image history and the
replay owns the execution index; policy queues are reset at episode boundaries.

EEF frames are `end_effector_l_link` and `end_effector_r_link`: arm link 7 plus
TCP `[0, 0, -0.215]` metres, with identical orientation. The lift is included in
both FK poses and output. The base is fixed; its three original velocity columns
are preserved separately. Both selected episodes have zero base commands.
Original state, joint commands, and reference/target FK poses are in
`relative_eef/reference/*.npz`, outside the LeRobot feature/input table.
New action feature definitions and episode/global statistics are generated.
Video files remain unchanged symlinks locally; rsync dereferences them on transfer.

The model is the existing local ROBOTIS menagerie practice copy based on
`d8344c0dbe7a00208d0301111523dde65efc174a`; recorded XML SHA-256 values are
its authoritative identity. The MuJoCo arm-base attachment X is corrected at
load from +0.0055 m to the Cyclo URDF's -0.0199 m. Source XML is unchanged.
Independent URDF FK and corrected MuJoCo FK agree to `6.67e-16` in tested poses.
SE(3) conversion/restoration error is below `7.44e-9`.

## Cyclo integration and validation

`LeRobotEngine.load_recorded_policy`, `observe_recorded`, and
`predict_recorded_chunk` reuse the existing `_load_policy_assets`, saved
pre/postprocessors, `_predict_chunk`, and `_to_numpy_chunk` implementation.
Only selected images cross the processor/network boundary. Normalized and
unnormalized outputs are both logged; inverse normalization occurs once.
`load_policy` rejects relative-EEF metadata before attaching a live RobotClient
or passing outputs to the joint-command runtime. Legacy joint models retain
the existing path.

`policy.py verify` uses actual LeRobot image/action batches and checks:
image-only forward/backward/optimizer update, finite gradients, inference and
queues, exact output invariance under added/removed/changed external state with
fixed noise, episode-start/end masks, normalization inversion, checkpoint
reload in a fresh process through Cyclo, and state-based Diffusion regression.
`test_core.py` and the existing Cyclo image/preprocessing/mapping tests add twelve
numerical/regression checks (including cache corruption and retry behavior).

The existing `lerobot.scripts.lerobot_train.train` performs all training. A
100-step calibration estimates the duration of a normal fixed-step run;
setup/loading and training durations are recorded separately. There is no
forced timeout. Saved optimizer step/moments, finite weights/losses, and changed
weight hashes verify updates. Config, processor graphs, normalization tensors,
action metadata, and model weights are all retrieved and checksum-verified.

The 2026-10-01 smoke run on A100 MIG 1g.10gb completed **6,850 updates in
329.791 seconds**, with logged loss 1.155 → 0.112. The first five updates
included approximately 25 seconds of CUDA autotuning; process setup/loading
outside the measured training loop was approximately 1.22 seconds.
Server checkpoint:
`/data/son_statefree_smoke/train_300s/checkpoints/006850/pretrained_model`.
Local checkpoint: `../statefree_smoke/checkpoint/`.

Local RTX 5060 Ti replay: **300 frames / 10 s / 38 chunks**, 1.234 seconds total
inference, 10.592 seconds wall time including IK/rendering/encoding. IK accepted
60 frames and held the entire preceding pose on 240 failures. All 300 frames
had at least one limited component. Maximum actual arm-joint displacement was
2.362 rad. Successful IK position residual was below 2 mm; failed IK never
substitutes recorded joints. This high failure/limiting rate is a limitation
of this short-trained recorded-observation replay, not a task-success claim.

The initial recorded head and right gripper exceed MJCF limits slightly. Initial
qpos is preserved exactly; the first accepted auxiliary command projects these
into limits (this initial projection can exceed the nominal per-step cap).
Thereafter the applied caps are 2 cm / 0.08 rad EEF delta norm, 0.12 rad arm
joint change, 0.05 rad gripper, 0.03 rad head, and 8 mm lift per step.

The video contains the robot mesh moving via predicted IK qpos, target/achieved/
recorded trajectories, XYZ axes, both input cameras, timestamps and legend:

- `../statefree_smoke/replay/replay.mp4`: H.264, 1280×720, 30 FPS, 300 frames.
- `frame_0000.png`, `frame_0150.png`, `frame_0299.png`: representative images.
- `replay.npz`: normalized/raw/applied actions, raw/applied targets, IK qpos,
  achieved/reference EEF, residuals, failures, clipping and inference latency.
- `replay_report.json`, `numerical_checks.json`, `training.json`,
  `checkpoint_sha256.json`, `manifest.json`, `summary.json`: evidence.

A separate fresh wrapper run with a 100-step calibration and 10-update training
also exercised conversion → SSH bootstrap/sync → trainer → retrieval → Cyclo
→ rendering. The main command's subsequent invocation verified and reused all
completed stages. A GLFW GUI replay was also executed for 24 frames.

References: [State-free Policy](https://statefreepolicy.github.io/),
[paper](https://arxiv.org/abs/2509.18644),
[ROBOTIS workflow](https://docs.robotis.com/docs/systems/aiworker/imitation_learning/),
[official MuJoCo model](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie).
The supplied Claude artifact could not be accessed; no behavior was inferred
from it. The small converter is based on inspected schemas, recording code,
URDF and verified MuJoCo kinematics.
