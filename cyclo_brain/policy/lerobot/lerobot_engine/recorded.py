"""Recorded-image adapter using Cyclo's real loading, processors and prediction.

No RobotClient is instantiated. Call observe_recorded once per recorded frame,
then predict_recorded_chunk only when the previously returned chunk is exhausted.
The caller owns the EEF -> IK boundary and the per-step simulation pose.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .image_preprocessing import prepare_policy_image


class RecordedObservationMixin:
    def load_recorded_policy(self, model_path, device=None):
        model_path = self._resolve_model_dir(str(model_path))
        metadata = json.loads((Path(model_path) / "action_representation.json").read_text())
        if metadata["representation"] != "per_step_body_se3_rotvec":
            raise ValueError("Unsupported recorded EEF action representation")
        mode = metadata.get('action_mode', 'eef')
        joints = metadata.get('erj_joints', [])
        if mode not in ('eef', 'erj') or len(metadata['action_layout']) != (19 if mode == 'erj' else 17):
            raise ValueError('Invalid recorded EEF/ERJ action mode or dimension')
        if (mode == 'eef' and joints) or (mode == 'erj' and (len(joints) != 2 or
                any(name not in [f'arm_{side}_joint{i}' for i in range(1, 8)]
                    for side, name in zip(('l', 'r'), joints)) or
                metadata['action_layout'][17:] != [f'{name}_absolute_rad' for name in joints])):
            raise ValueError('Invalid ERJ named joint action specification')
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._policy, self._preprocessor, self._postprocessor = self._load_policy_assets(model_path, self._device)
        cfg = self._policy.config
        if cfg.type != "diffusion" or set(cfg.input_features) != set(cfg.image_features):
            raise ValueError("Recorded state-free replay requires image-only Diffusion")
        if set(cfg.input_features) != set(metadata["policy_input_keys"]):
            raise ValueError("Checkpoint and action metadata input keys differ")
        if cfg.action_feature.shape[0] != len(metadata["action_layout"]):
            raise ValueError("Checkpoint and action metadata output dimensions differ")
        self._loaded_model_path = model_path
        self._image_resize = self._infer_image_resize(self._policy)
        self._action_representation = metadata
        self.reset_recorded_episode()
        return metadata

    def reset_recorded_episode(self):
        self._policy.reset()
        self._recorded_history = {k: deque(maxlen=self._policy.config.n_obs_steps)
                                  for k in self._policy.config.image_features}

    def observe_recorded(self, observation):
        # Explicit allowlist: proprioception, ground-truth action and future
        # observations never cross the processor/network boundary.
        for key, history in self._recorded_history.items():
            img = observation[key]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
            if img.dtype != np.uint8:
                if not np.isfinite(img).all() or img.min() < 0 or img.max() > 1:
                    raise ValueError("Float dataset images must be finite RGB in [0,1]")
                img = np.rint(img * 255).astype(np.uint8)
            img = prepare_policy_image(img, rotation_deg=0, target_size=self._image_resize.get(key))
            tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
            history.append(tensor)
            while len(history) < history.maxlen:
                history.append(tensor)

    def predict_recorded_chunk(self):
        if any(not h for h in self._recorded_history.values()):
            raise RuntimeError("Observe a recorded frame before inference")
        batch = {k: torch.stack(list(h)).unsqueeze(0) for k, h in self._recorded_history.items()}
        with torch.inference_mode():
            processed = self._preprocessor(batch)
            # Processor transitions may also emit empty task/reward fields.
            # Forward only the configured camera tensors to the network.
            images = {k: processed[k] for k in self._policy.config.image_features}
            normalized = self._predict_chunk(images)
            actions = self._postprocessor(normalized)  # exactly one unnormalization
        return {"normalized_action": self._to_numpy_chunk(normalized),
                "action": self._to_numpy_chunk(actions),
                "representation": self._action_representation["representation"]}
