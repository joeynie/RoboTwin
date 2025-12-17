#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
Enhanced PI0 Model with Attention Map Visualization
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image
from pathlib import Path

from attn_visualizer import AttentionConfig, AttentionVisualizer, TestPointRecorder


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step,
        extract_attention: bool = False,
        attention_layer_idx: int = 10,
        attn_save_dir: str = "eval_result"
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.extract_attention = extract_attention
        self.attention_layer_idx = attention_layer_idx

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
        )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        
        # Attention visualization setup
        if self.extract_attention:
            attn_config = AttentionConfig(
                layer_idx=attention_layer_idx,
                save_dir=attn_save_dir
            )
            self.attn_recorder = TestPointRecorder(attn_config)
        else:
            self.attn_recorder = None
        
        self.current_test_point = None
        self.frame_images = {}  # Store images for each view

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
        
        # Store images for attention visualization (convert CHW to HWC)
        self.frame_images = {
            "cam_high": np.transpose(img_front, (1, 2, 0)),
            "cam_left_wrist": np.transpose(img_left, (1, 2, 0)),
            "cam_right_wrist": np.transpose(img_right, (1, 2, 0)),
        }
        
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

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
        
        # Get inference results
        results = self.policy.infer(self.observation_window)
        
        # Process attention map if available
        if self.extract_attention and self.attn_recorder and "attention_map" in results:
            attention_map = results["attention_map"]
            self.attn_recorder.process_frame(self.frame_images, attention_map)
        
        return results["actions"]

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
