#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
Enhanced PI0 Model with Attention Map Visualization
"""
import json
import sys
import logging
import jax
import numpy as np
import torch
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

from attn_visualizer import AttentionConfig, AttentionVideoRecorder

logger = logging.getLogger(__name__)

class PolicyWrapper:
    """
    Lightweight wrapper around Policy to extract attention maps.
    
    Registers forward hooks on the policy model to capture intermediate
    activations (attention maps) during inference.
    
    Uses layer name (string) instead of index to reliably locate target modules.
    """
    
    def __init__(self, policy, target_layer_name: str = "layers.10.self_attn"):
        """
        Args:
            policy: Original policy object from policy_config.create_trained_policy()
            target_layer_name: String to match the module name 
                             e.g., 'layers.10.self_attn', 'gemma_expert.model.layers.10', etc.
        """
        self.policy = policy
        self.target_layer_name = target_layer_name
        self._activations = {}
        self._hooks = []
        
        # Register hooks on the policy's model
        if hasattr(policy, '_model'):
            self._register_hooks(policy._model)
        else:
            print("⚠ Policy has no _model attribute")
    
    def _register_hooks(self, model):
        """Register forward hooks to capture layer activations by layer name."""
        found = False
        for name, module in model.named_modules():
            if self.target_layer_name in name:
                print(f"✓ Found target module: {name} ({type(module).__name__})")
                
                # 注册 Hook
                hook = module.register_forward_hook(self._get_hook(name))
                self._hooks.append(hook)
                found = True
        
        if not found:
            print(f"⚠ Layer '{self.target_layer_name}' not found in model.")
            print("Available top-level modules:", [n for n, _ in model.named_children()])
    
    def _get_hook(self, name):
        """Create a hook function for a specific named module."""
        def hook(module, input, output):
            extracted_data = None
            
            # 情况 A: 输出是 Tuple，寻找 Attention Weights (通常形状 [Batch, Heads, Seq, Seq])
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        # 检查形状特征：Attention Map 通常是 4D: [B, H, S, S]，且最后两个维度相等
                        if item.ndim == 4 and item.shape[-1] == item.shape[-2]:
                            extracted_data = item
                            print(f"  ✓ Extracted attention weights from {name}: shape {item.shape}")
                            break
            
            # 情况 B: 输出是 Tensor
            elif isinstance(output, torch.Tensor):
                if output.ndim == 4 and output.shape[-1] == output.shape[-2]:
                    extracted_data = output
                    print(f"  ✓ Extracted attention weights from {name}: shape {output.shape}")
            
            if extracted_data is not None:
                self._activations[name] = extracted_data.detach().cpu()
            
        return hook
    
    def infer(self, observation, **kwargs):
        """
        Call policy.infer() and extract per-view attention maps.
        
        Returns:
            dict with keys:
                - "actions": action array from policy
                - "attention_maps": dict with keys {"cam_high", "cam_left_wrist", "cam_right_wrist"}
                  each containing a [16, 16] numpy array of attention weights
        """
        # Clear previous activations
        self._activations.clear()
        
        # Call original policy.infer()
        results = self.policy.infer(observation, **kwargs)
        
        # Extract per-view attention maps from hooks
        # Based on PI0Pytorch.embed_prefix():
        # - Positions 0-255: cam_high (base_0)
        # - Positions 256-511: cam_left_wrist
        # - Positions 512-767: cam_right_wrist
        # - Positions 768+: language tokens
        
        attention_maps = {
            "cam_high": None,
            "cam_left_wrist": None,
            "cam_right_wrist": None,
        }
        
        if self._activations:
            # Get the last captured activation
            raw_attn = list(self._activations.values())[-1]
            
            # raw_attn should be [Batch, Heads, Seq, Seq]
            if isinstance(raw_attn, torch.Tensor):
                raw_attn = raw_attn.float()
            elif not isinstance(raw_attn, np.ndarray):
                raw_attn = np.array(raw_attn, dtype=np.float32)
            
            # Extract attention matrix: [Batch, Heads, Seq_Len, Seq_Len] -> [Seq, Seq]
            if isinstance(raw_attn, torch.Tensor) and raw_attn.ndim == 4:
                attn_matrix = raw_attn[0].mean(dim=0).detach().cpu().numpy()  # [Seq, Seq]
            elif isinstance(raw_attn, np.ndarray) and raw_attn.ndim == 4:
                attn_matrix = raw_attn[0].mean(axis=0)  # [Seq, Seq]
            elif isinstance(raw_attn, torch.Tensor) and raw_attn.ndim == 3:
                attn_matrix = raw_attn[0].detach().cpu().numpy() if raw_attn.shape[0] >= 32 else raw_attn.mean(dim=0).detach().cpu().numpy()
            elif isinstance(raw_attn, np.ndarray) and raw_attn.ndim == 3:
                attn_matrix = raw_attn[0] if raw_attn.shape[0] >= 32 else raw_attn.mean(axis=0)
            else:
                attn_matrix = None
            
            # Extract action tokens' attention to each view's image patches
            if attn_matrix is not None and attn_matrix.ndim == 2:
                # Determine action horizon (remaining positions after 768 image+language patches)
                # Assuming: 3*256=768 image patches + ~200 language tokens + action tokens
                # Find where action tokens start by looking for large seq dimension
                seq_len = attn_matrix.shape[0]
                
                # Image patch positions: 0-767 (3 views * 256 patches each)
                num_patches_per_view = 256
                num_views = 3
                total_image_patches = num_patches_per_view * num_views  # 768
                
                # Action tokens should be at the end
                if seq_len > total_image_patches:
                    action_horizon = min(50, seq_len - total_image_patches)
                    
                    # Extract attention from action tokens to each view's patches
                    view_ranges = {
                        "cam_high": (0, 256),
                        "cam_left_wrist": (256, 512),
                        "cam_right_wrist": (512, 768),
                    }
                    
                    for view_name, (patch_start, patch_end) in view_ranges.items():
                        # Get action tokens' attention to this view's patches
                        action_attn = attn_matrix[-action_horizon:, patch_start:patch_end]  # [action_horizon, 256]
                        
                        # Average across action tokens to get patch importance
                        view_attn = action_attn.mean(axis=0)  # [256]
                        
                        # Reshape to spatial heatmap (16x16)
                        if len(view_attn) == 256:
                            view_attn = view_attn.reshape(16, 16)
                        
                        # Normalize to [0, 1]
                        view_attn = view_attn.astype(np.float32)
                        attn_min, attn_max = view_attn.min(), view_attn.max()
                        if attn_max > attn_min:
                            view_attn = (view_attn - attn_min) / (attn_max - attn_min + 1e-8)
                        
                        attention_maps[view_name] = view_attn
        
        results["attention_maps"] = attention_maps
        return results
    
    def __getattr__(self, name):
        """Delegate all other attributes to wrapped policy."""
        return getattr(self.policy, name)


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
        extract_attention: bool = False,
        target_layer_name: str = "layers.10.self_attn",
        attn_save_dir: str = "eval_result"
    ):
        """
        Args:
            train_config_name: Model config name
            model_name: Model checkpoint name
            checkpoint_id: Checkpoint identifier
            pi0_step: Number of action steps
            extract_attention: Enable attention extraction
            target_layer_name: Layer name to hook (e.g., 'layers.10.self_attn', 'gemma_expert.model.layers.10')
            attn_save_dir: Output directory for attention maps
        """
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.extract_attention = extract_attention
        self.target_layer_name = target_layer_name

        config = _config.get_config(self.train_config_name)
        policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_id,
        )
        
        # Wrap policy to enable attention map extraction
        if extract_attention:
            self.policy = PolicyWrapper(policy, target_layer_name=target_layer_name)
            print(f"✓ Policy wrapped for attention extraction (target: {target_layer_name})")
        else:
            self.policy = policy
        
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        
        # Attention visualization setup
        if self.extract_attention:
            attn_config = AttentionConfig(save_dir=attn_save_dir)
            self.attn_recorder = AttentionVideoRecorder(attn_config)
        else:
            self.attn_recorder = None
        
        self.current_test_point = None
        self.frame_images = {}  # Store images for each view
        self._current_attention_maps = {}  # Store per-view attention maps

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        
        # Helper function to ensure HWC format for visualization
        def to_hwc(img):
            """Convert image to HWC format if it's CHW."""
            if img.ndim == 3:
                # Check if it looks like CHW (channels-first): C < H, C < W
                if img.shape[0] <= 4 and img.shape[1] > img.shape[0] and img.shape[2] > img.shape[0]:
                    # Likely CHW -> convert to HWC
                    return np.transpose(img, (1, 2, 0))
                else:
                    # Already HWC
                    return img
            return img
        
        # Store images for attention visualization in HWC format
        self.frame_images = {
            "cam_high": to_hwc(img_front),
            "cam_left_wrist": to_hwc(img_left),
            "cam_right_wrist": to_hwc(img_right),
        }
        
        # Prepare images for model input (ensure CHW format)
        def to_chw(img):
            """Convert image to CHW format if it's HWC."""
            if img.ndim == 3:
                # Check if it looks like HWC (channels-last): last dim is 3 or 4
                if img.shape[2] <= 4 and img.shape[0] > 4 and img.shape[1] > 4:
                    # Likely HWC -> convert to CHW
                    return np.transpose(img, (2, 0, 1))
                else:
                    # Already CHW
                    return img
            return img
        
        img_front = to_chw(img_front)
        img_right = to_chw(img_right)
        img_left = to_chw(img_left)

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        
        results = self.policy.infer(self.observation_window)
        actions = results["actions"]
        
        if self.extract_attention:
            attention_maps = results.get("attention_maps", {})
            # Store per-view attention maps for retrieval
            self._current_attention_maps = attention_maps
        
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.frame_images = {}
        print("successfully unset obs and language instruction")
    
    def start_test_point(self, test_point_id: str):
        """Start recording a new test point sequence."""
        self.current_test_point = test_point_id
        if self.attn_recorder:
            self.attn_recorder.start_test_point(test_point_id)
            print(f"Started recording test point: {test_point_id}")
    
    def end_test_point(self):
        """End recording current test point."""
        if self.attn_recorder and self.current_test_point:
            self.attn_recorder.end_test_point()
            print(f"Ended recording test point: {self.current_test_point}")
        self.current_test_point = None
