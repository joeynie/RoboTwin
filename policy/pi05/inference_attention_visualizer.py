#!/usr/bin/env python3
"""
Inference-Time Attention Visualizer

This module provides real-time attention visualization during model inference,
designed to work with serve_policy.py and the PI0 policy inference pipeline.

Usage:
    # Initialize the visualizer
    visualizer = InferenceAttentionVisualizer(
        output_dir="./inference_attention",
        save_every_n_calls=5
    )

    # Wrap your model with the visualizer
    model = visualizer.wrap_model(model)

    # Run inference normally - attention will be captured automatically
    outputs = model.sample_actions(rng, observation)

    # The visualizer will automatically save attention visualizations
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openpi.models.model import Observation


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Resize image to target height and width without distortion by padding with zeros.
    """
    cur_height, cur_width = image.shape[:2]
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize image
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Create zero-padded image
    if len(image.shape) == 3:
        zero_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    else:
        zero_image = np.zeros((height, width), dtype=image.dtype)
    
    # Calculate padding
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)

    # Paste resized image onto zero-padded image
    zero_image[pad_height:pad_height + resized_height, pad_width:pad_width + resized_width] = resized_image
    
    return zero_image


class InferenceAttentionHook:
    """Hook to capture attention weights during inference."""

    def __init__(self):
        self.attention_weights = {}
        self.enabled = True
        self.call_count = 0

    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        if not self.enabled:
            return

        self.call_count += 1

        # The output structure for PaliGemmaWithExpertModel:
        # ([prefix_output, suffix_output], past_key_values, all_qk_states)
        # all_qk_states is a list of (layer_idx, qk_dict) tuples
        if isinstance(output, tuple) and len(output) >= 3 and output[2] is not None:
            all_qk_states = output[2]
            if all_qk_states and isinstance(all_qk_states, list):
                attention_weights = {}
                for layer_idx, qk_dict in all_qk_states:
                    if not isinstance(qk_dict, dict): continue
                    
                    attn_probs = None
                    # Try to get pre-computed attention_probs first
                    if 'attention_probs' in qk_dict and qk_dict['attention_probs'] is not None:
                        attn_probs = qk_dict['attention_probs']
                    # If not available (e.g., when using depth attention), compute from Q/K
                    elif 'attention' in qk_dict:
                        q, k = qk_dict['attention']
                        if q is not None and k is not None:
                            # Compute attention: softmax(Q @ K^T / sqrt(d))
                            # q: [B, num_heads, seq_len, head_dim]
                            # k: [B, num_heads, seq_len, head_dim]
                            head_dim = q.shape[-1]
                            scaling = 1.0 / (head_dim ** 0.5)
                            
                            # Compute attention scores: [B, num_heads, seq_len, seq_len]
                            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
                            
                            # Apply softmax to get attention probabilities
                            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
                        
                        if attn_probs is not None:
                            attention_weights[layer_idx] = attn_probs.detach().cpu()
                
                if attention_weights:
                    self.attention_weights = attention_weights

    def clear(self):
        """Clear stored attention weights."""
        self.attention_weights = {}

    def disable(self):
        """Disable the hook."""
        self.enabled = False

    def enable(self):
        """Enable the hook."""
        self.enabled = True


class InferenceAttentionVisualizer:
    """Real-time attention visualizer for inference."""

    def __init__(
        self,
        output_dir: str = "./inference_attention",
        save_every_n_calls: int = 10,
        image_size: Tuple[int, int] = (224, 224),
        alpha: float = 0.5,
        layer_idx: Optional[int] = None,
        create_videos: bool = True,
    ):
        """
        Initialize the inference attention visualizer.

        Args:
            output_dir: Directory to save visualizations
            save_every_n_calls: Save visualization every N inference calls
            image_size: Image size (H, W)
            alpha: Transparency factor for attention overlays
            layer_idx: Specific layer index to visualize (0-indexed)
            create_videos: Whether to create videos from samples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_n_calls = save_every_n_calls
        self.image_size = image_size
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.create_videos = create_videos

        # Hook for attention extraction
        self.attention_hook = InferenceAttentionHook()
        self.hook_handles = []

        # Statistics
        self.total_calls = 0
        self.saved_count = 0
        
        # Frame storage for final video
        self.frames = []
        # Tmp video frames for periodic saving
        self.tmp_frames = []
        self.tmp_video_count = 0

        # Model configuration
        self.num_images = 3  # base, secondary, wrist
        self.num_patches_per_image = 256  # 16x16 patches

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap the model to automatically capture attention during inference."""
        # Find the PaligemmaWithGemmaExpert model
        target_model = None
        if hasattr(model, 'paligemma_with_expert'):
            target_model = model.paligemma_with_expert
        elif hasattr(model, 'model') and hasattr(model.model, 'paligemma_with_expert'):
            target_model = model.model.paligemma_with_expert
        else:
            # Try to find it recursively
            for module in model.modules():
                if hasattr(module, 'paligemma_with_expert'):
                    target_model = module.paligemma_with_expert
                    break

        if target_model is None:
            print("⚠ Could not find PaligemmaWithGemmaExpert model - attention visualization disabled")
            return model

        # Register forward hook
        hook_handle = target_model.register_forward_hook(self.attention_hook)
        self.hook_handles.append(hook_handle)

        print(f"✓ Registered attention hook on {type(target_model).__name__}")

        # Wrap the model's sample_actions method
        original_sample_actions = model.sample_actions

        def wrapped_sample_actions(rng, observation, **kwargs):
            """Wrapped sample_actions that captures attention."""
            # Clear previous attention weights
            self.attention_hook.clear()

            # Call original method with enable_attention_viz=True
            start_time = time.monotonic()
            actions = original_sample_actions(rng, observation, enable_attention_viz=True, **kwargs)
            inference_time = time.monotonic() - start_time

            # Process attention visualization
            self._process_inference_sample(observation, actions, inference_time)

            return actions

        # Replace the method
        model.sample_actions = wrapped_sample_actions
        
        self.model = model

        return model

    def _process_inference_sample(self, observation: Observation, actions: Any, inference_time: float):
        """Process a single inference sample for attention visualization."""
        self.total_calls += 1

        # Extract attention weights
        attention_weights = self.attention_hook.attention_weights

        if not attention_weights or len(attention_weights) == 0:
            return

        # Collect frame if sampling interval reached
        if self.total_calls % self.save_every_n_calls == 0:
            processed_attention = self._process_attention_weights(attention_weights)
            frame = self._create_visualization_frame(observation, processed_attention)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store frame for final video
            self.frames.append(frame_rgb)
            # Store for tmp video
            self.tmp_frames.append(frame_rgb)
            self.saved_count += 1
            
            # Save tmp video every 100 samples
            if self.saved_count % 100 == 0:
                self._save_tmp_video()
                self.tmp_video_count += 1

    def _convert_image_format(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert image to display format and resize with padding to square."""
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().numpy()
        else:
            img_np = np.array(image)

        # Remove batch dimension if present
        if img_np.ndim == 4 and img_np.shape[0] == 1:
            img_np = img_np[0]

        # Convert from channel-first to channel-last
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Convert to uint8
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:  # If normalized [0,1]
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Convert BGR to RGB for display (OpenCV uses BGR by default)
        if img_np.shape[-1] == 3: 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Resize with padding to target size (handles non-square images from RobotWin)
        target_h, target_w = self.image_size
        if img_np.shape[0] != target_h or img_np.shape[1] != target_w:
            img_np = resize_with_pad(img_np, target_h, target_w)

        return img_np

    def _process_attention_weights(self, attention_weights: Dict[int, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Process attention weights to extract image attention."""
        if not attention_weights:
            return {}

        # Determine which layer to visualize
        if self.layer_idx is not None:
            layer_idx = self.layer_idx
        else:
            # Use the highest layer by default
            available_layers = sorted(attention_weights.keys())
            layer_idx = available_layers[-1] if available_layers else None

        if layer_idx is None or layer_idx not in attention_weights:
            return {}

        processed_attention = {}
        
        layer_attention = attention_weights[layer_idx]  # [B, num_heads, seq_len, seq_len]

        if layer_attention.numel() == 0:
            return {}

        # Take first sample (batch_size=1 during inference)
        layer_attention = layer_attention[0]  # [num_heads, seq_len, seq_len]

        seq_len = layer_attention.shape[1]

        # Estimate token positions
        image_tokens_start = 0
        image_tokens_end = self.num_images * self.num_patches_per_image  # 3 * 256 = 768
        # Last few tokens are likely action tokens
        action_tokens_start = max(0, seq_len - 16)
        action_tokens_end = seq_len

        # Extract attention from last action token to all image patches
        if action_tokens_start < action_tokens_end and image_tokens_start < image_tokens_end:
            last_action_idx = action_tokens_end - 1
            action_to_image_attn = layer_attention[:, last_action_idx, image_tokens_start:image_tokens_end]  # [num_heads, num_image_patches]

            # Average over heads
            attention_map = action_to_image_attn.mean(dim=0)  # [num_image_patches]

            # Reshape to image grid format
            if attention_map.numel() == self.num_images * self.num_patches_per_image:
                attention_map = attention_map.view(self.num_images, self.num_patches_per_image)

                # Split into individual views
                view_names = ["base_0", "left_wrist_0", "right_wrist_0"]
                for i, view_name in enumerate(view_names):
                    if i < attention_map.shape[0]:
                        view_attention = attention_map[i].cpu().numpy()  # [256]
                        view_attention = view_attention.reshape(16, 16)  # [16, 16]
                        processed_attention[f"layer_{layer_idx}_{view_name}"] = view_attention

        return processed_attention

    def _create_visualization_frame(
        self,
        observation: Observation,
        processed_attention: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create a visualization frame for an inference sample."""
        # Try to use original images from model attribute first
        base_image = None
        left_wrist_image = None
        right_wrist_image = None
        
        # Check if model has original images set (provided externally)
        if hasattr(self.model, 'original_images_for_viz') and self.model.original_images_for_viz:
            orig_images = self.model.original_images_for_viz
            if 'cam_high' in orig_images:
                base_image = orig_images['cam_high']
            if 'cam_left_wrist' in orig_images:
                left_wrist_image = orig_images['cam_left_wrist']
            if 'cam_right_wrist' in orig_images:
                right_wrist_image = orig_images['cam_right_wrist']
        

        # Convert images to display format
        if base_image is not None:
            base_image = self._convert_image_format(base_image)
        if left_wrist_image is not None:
            left_wrist_image = self._convert_image_format(left_wrist_image)
        if right_wrist_image is not None:
            right_wrist_image = self._convert_image_format(right_wrist_image)

        # Handle missing images
        if base_image is None:
            base_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        if left_wrist_image is None:
            left_wrist_image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        if right_wrist_image is None:
            right_wrist_image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Create visualization layout: 2x3 grid
        # Top row: Original images (base, left_wrist, right_wrist)
        # Bottom row: Attention overlays
        height, width = base_image.shape[:2]
        
        if processed_attention:
            combined_frame = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)

            # Top row: Original images
            combined_frame[0:height, 0:width] = base_image
            combined_frame[0:height, width:width*2] = left_wrist_image
            combined_frame[0:height, width*2:width*3] = right_wrist_image

            # Bottom row: Attention overlays
            # Base camera
            base_attn_key = None
            for key in processed_attention.keys():
                if "base_0" in key:
                    base_attn_key = key
                    break

            if base_attn_key:
                base_overlay = self._create_attention_overlay(base_image, processed_attention[base_attn_key])
                combined_frame[height:height*2, 0:width] = base_overlay
            else:
                combined_frame[height:height*2, 0:width] = base_image

            # Left wrist camera
            left_wrist_attn_key = None
            for key in processed_attention.keys():
                if "left_wrist_0" in key:
                    left_wrist_attn_key = key
                    break

            if left_wrist_attn_key:
                left_wrist_overlay = self._create_attention_overlay(left_wrist_image, processed_attention[left_wrist_attn_key])
                combined_frame[height:height*2, width:width*2] = left_wrist_overlay
            else:
                combined_frame[height:height*2, width:width*2] = left_wrist_image

            # Right wrist camera
            right_wrist_attn_key = None
            for key in processed_attention.keys():
                if "right_wrist_0" in key:
                    right_wrist_attn_key = key
                    break

            if right_wrist_attn_key:
                right_wrist_overlay = self._create_attention_overlay(right_wrist_image, processed_attention[right_wrist_attn_key])
                combined_frame[height:height*2, width*2:width*3] = right_wrist_overlay
            else:
                combined_frame[height:height*2, width*2:width*3] = right_wrist_image

            # Add text labels for clarity
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (255, 255, 255)
            
            # Labels for top row
            cv2.putText(combined_frame, "Base View", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_frame, "Left Wrist", (width + 10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_frame, "Right Wrist", (width*2 + 10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
            
            # Labels for bottom row
            cv2.putText(combined_frame, "Attention", (10, height + 20), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_frame, "Attention", (width + 10, height + 20), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(combined_frame, "Attention", (width*2 + 10, height + 20), font, font_scale, color, thickness, cv2.LINE_AA)

        else:
            # Simple horizontal layout without attention
            combined_frame = np.zeros((height, width * 3, 3), dtype=np.uint8)
            combined_frame[0:height, 0:width] = base_image
            combined_frame[0:height, width:width*2] = left_wrist_image
            combined_frame[0:height, width*2:width*3] = right_wrist_image

        return combined_frame

    def _create_attention_overlay(self, image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Create attention overlay on image."""
        # Resize attention map to match image size
        attention_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 255]
        if attention_resized.max() > attention_resized.min():
            attention_normalized = ((attention_resized - attention_resized.min()) /
                                   (attention_resized.max() - attention_resized.min() + 1e-8) * 255).astype(np.uint8)
        else:
            attention_normalized = np.zeros_like(attention_resized, dtype=np.uint8)

        # Apply colormap
        attention_colored = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)

        # Create overlay
        overlay = cv2.addWeighted(image, 1 - self.alpha, attention_colored, self.alpha, 0)
        return overlay

    def _save_tmp_video(self):
        """Save a temporary video of collected frames."""
        if not self.tmp_frames:
            return
        
        tmp_video_path = self.output_dir / f"tmp_video_{self.tmp_video_count:03d}.mp4"
        
        import imageio
        # Dynamically adjust fps based on save_every_n_calls
        # If save_every_n_calls is small, we have more frames, so increase fps proportionally
        fps = max(2.0, 10.0 / max(1, self.save_every_n_calls))
        imageio.mimsave(str(tmp_video_path), self.tmp_frames, fps=fps)
        
        print(f"✓ Saved tmp video {self.tmp_video_count} ({len(self.tmp_frames)} frames) to {tmp_video_path}")
        self.tmp_frames = []

    def save_summary_video(self, output_path: Optional[str] = None, fps: Optional[float] = None):
        """Create a summary video of all collected frames.
        
        Args:
            output_path: Path to save video
            fps: Frames per second. If None, automatically adjusted based on save_every_n_calls
        """
        if not self.frames:
            print("No frames to create summary video")
            return

        if output_path is None:
            output_path = self.output_dir / f"attention_summary_{int(time.time())}.mp4"

        # Create video from collected frames
        import imageio
        # Dynamically adjust fps if not specified
        if fps is None:
            fps = max(2.0, 10.0 / max(1, self.save_every_n_calls))
        imageio.mimsave(str(output_path), self.frames, fps=fps)
        print(f"✓ Summary video saved to {output_path} ({len(self.frames)} frames @ {fps} fps)")

    def get_stats(self) -> Dict[str, Any]:
        """Get visualization statistics."""
        return {
            "total_calls": self.total_calls,
            "saved_count": self.saved_count,
            "frames_in_memory": len(self.frames),
            "tmp_videos_saved": self.tmp_video_count,
            "output_directory": str(self.output_dir),
        }

    def cleanup(self):
        """Cleanup resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.attention_hook.clear()