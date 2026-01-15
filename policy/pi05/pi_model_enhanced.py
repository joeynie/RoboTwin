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

from inference_attention_visualizer import InferenceAttentionVisualizer

logger = logging.getLogger(__name__)


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
        extract_attention: bool = False,
        layer_idx: Optional[int] = None,
        attn_save_dir: str = "eval_result"
    ):
        """
        Args:
            train_config_name: Model config name
            model_name: Model checkpoint name
            checkpoint_id: Checkpoint identifier
            pi0_step: Number of action steps
            extract_attention: Enable attention extraction
            layer_idx: Layer index for attention visualization (0-indexed, None means use the last layer)
            attn_save_dir: Output directory for attention maps
        """
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.extract_attention = extract_attention
        self.layer_idx = layer_idx

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_id,
        )
        
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        
        # Attention visualization setup with InferenceAttentionVisualizer
        if self.extract_attention:
            self.attn_visualizer = InferenceAttentionVisualizer(
                output_dir=attn_save_dir,
                save_every_n_calls=1,
                layer_idx=layer_idx,
                create_videos=True
            )
            # Set attention visualization layer index on the model
            self.policy._model.attention_viz_layer_idx = layer_idx
            # Set up original images attribute for visualization
            self.policy._model.original_images_for_viz = None
            # Wrap the model inside policy to enable automatic attention capture
            self.policy._model = self.attn_visualizer.wrap_model(self.policy._model)
            # Update policy's sample_actions reference to the wrapped version
            if hasattr(self.policy, '_sample_actions'):
                self.policy._sample_actions = self.policy._model.sample_actions
            layer_info = f"layer {layer_idx}" if layer_idx is not None else "last layer"
            print(f"✓ Model wrapped for attention visualization ({layer_info})")
        else:
            self.attn_visualizer = None
        
        self.current_test_point = None
        self.frame_images = {}  # Store images for each view (for backward compatibility)
        self._observation = None  # Store observation for InferenceAttentionVisualizer

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
        
        # Pass original images to model for visualization (before preprocessing)
        if self.extract_attention and self.attn_visualizer:
            self.policy._model.original_images_for_viz = self.frame_images.copy()
        
        results = self.policy.infer(self.observation_window)
        actions = results["actions"]
        
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.frame_images = {}
        print("successfully unset obs and language instruction")
    
    def start_test_point(self, test_point_id: str):
        """Start recording a new test point sequence."""
        self.current_test_point = test_point_id
        if self.attn_visualizer:
            self.attn_visualizer.frames = []
            self.attn_visualizer.tmp_frames = []
            print(f"Started recording test point: {test_point_id}")
    
    def end_test_point(self):
        """End recording current test point."""
        if self.attn_visualizer and self.current_test_point:
            self.attn_visualizer.save_summary_video()
            print(f"Ended recording test point: {self.current_test_point}")
        self.current_test_point = None
