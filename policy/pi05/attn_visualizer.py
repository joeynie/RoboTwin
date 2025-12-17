"""
Attention Map Visualization Module

Handles extraction, processing, and visualization of attention maps from policy models.
Supports both JAX and PyTorch models.
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention map extraction and visualization."""
    layer_idx: int = 10  # Which layer to extract attention from
    save_dir: str = "attn_maps"  # Directory to save visualization
    save_video: bool = True  # Save video with attention overlay
    fps: int = 10  # Video FPS
    overlay_alpha: float = 0.5  # Transparency of attention overlay
    colormap: str = "jet"  # Colormap for visualization


class AttentionMapExtractor:
    """Extracts attention maps from models during inference."""
    
    def __init__(self, model, config: AttentionConfig):
        self.model = model
        self.config = config
        self.is_pytorch = hasattr(model, '__class__') and 'torch' in str(type(model).__module__)
        self.activations = {}
        self.hooks = []
        
        if self.is_pytorch:
            self._register_pytorch_hooks()
    
    def _register_pytorch_hooks(self):
        """Register forward hooks for PyTorch model."""
        import torch
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Find the layer at specified index
        layers = list(self.model.named_modules())
        if self.config.layer_idx < len(layers):
            layer_name, layer = layers[self.config.layer_idx]
            hook = layer.register_forward_hook(get_activation(layer_name))
            self.hooks.append(hook)
            logger.info(f"Registered hook for layer {self.config.layer_idx}: {layer_name}")
    
    def extract_attention(self) -> Optional[np.ndarray]:
        """Extract attention map from last forward pass."""
        if self.is_pytorch:
            for name, activation in self.activations.items():
                # Handle different activation shapes
                if activation.dim() >= 2:
                    # Convert to numpy and average across batch/channel dims if needed
                    attn = activation.cpu().numpy()
                    
                    # If multi-dimensional, average across channels
                    while attn.ndim > 2:
                        attn = attn.mean(axis=1)
                    
                    return attn
        
        return None
    
    def cleanup(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.activations.clear()


class AttentionVisualizer:
    """Visualizes attention maps and creates videos."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Video writers for each view
        self.video_writers: Dict[str, cv2.VideoWriter] = {}
        self.frames_buffer: Dict[str, List[np.ndarray]] = {}
        self.test_point_id = None
        self.colormap = cv2.COLORMAP_JET if config.colormap == "jet" else cv2.COLORMAP_VIRIDIS
    
    def set_test_point(self, test_point_id: str):
        """Set current test point ID for video naming."""
        self.test_point_id = test_point_id
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize frame buffers for different views."""
        self.frames_buffer = {
            "cam_high": [],
            "cam_left_wrist": [],
            "cam_right_wrist": [],
        }
    
    def visualize_and_save(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        view_name: str,
        frame_idx: int
    ) -> np.ndarray:
        """
        Visualize attention map on image and buffer for video.
        
        Args:
            image: RGB image (H, W, 3), values in [0, 255]
            attention_map: Attention map (H', W'), values in [0, 1]
            view_name: Camera view name (e.g., 'cam_high')
            frame_idx: Frame index in sequence
        
        Returns:
            Image with attention overlay
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        h, w = image.shape[:2]
        
        # Resize attention map to match image size if needed
        if attention_map.shape != (h, w):
            attention_map = cv2.resize(
                attention_map,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Normalize attention map to [0, 1]
        attn_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Convert to heatmap
        attn_heatmap = cv2.applyColorMap(
            (attn_normalized * 255).astype(np.uint8),
            self.colormap
        )
        
        # Convert BGR to RGB
        attn_heatmap = cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(
            image,
            1 - self.config.overlay_alpha,
            attn_heatmap,
            self.config.overlay_alpha,
            0
        )
        
        # Buffer frame for video
        if view_name in self.frames_buffer:
            self.frames_buffer[view_name].append(overlay)
        
        # Add frame index text
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return overlay
    
    def save_attention_map(
        self,
        attention_map: np.ndarray,
        view_name: str,
        frame_idx: int
    ):
        """Save attention map as image."""
        if not self.test_point_id:
            return
        
        test_dir = self.save_dir / self.test_point_id / view_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize and convert to uint8
        attn_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)
        
        # Apply colormap
        attn_colored = cv2.applyColorMap(attn_uint8, self.colormap)
        
        # Save
        output_path = test_dir / f"attn_frame_{frame_idx:04d}.png"
        cv2.imwrite(str(output_path), attn_colored)
    
    def finalize_video(self):
        """Create video files from buffered frames."""
        if not self.test_point_id:
            return
        
        test_dir = self.save_dir / self.test_point_id
        
        for view_name, frames in self.frames_buffer.items():
            if not frames:
                continue
            
            video_dir = test_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Get frame dimensions
            h, w = frames[0].shape[:2]
            
            # Create video writer
            video_path = video_dir / f"{view_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.config.fps,
                (w, h)
            )
            
            # Write frames (convert RGB to BGR for opencv)
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            logger.info(f"Saved video: {video_path}")
        
        # Save metadata
        metadata = {
            "test_point_id": self.test_point_id,
            "config": {
                "layer_idx": self.config.layer_idx,
                "fps": self.config.fps,
                "overlay_alpha": self.config.overlay_alpha,
            },
            "views": list(self.frames_buffer.keys()),
            "num_frames": {
                view: len(frames)
                for view, frames in self.frames_buffer.items()
            }
        }
        
        metadata_path = test_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_path}")
    
    def cleanup(self):
        """Cleanup video writers."""
        for writer in self.video_writers.values():
            if writer is not None:
                writer.release()
        self.video_writers.clear()


class TestPointRecorder:
    """Records test sequences with attention visualization."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.visualizer = AttentionVisualizer(config)
        self.current_test_point = None
        self.frame_count = 0
    
    def start_test_point(self, test_point_id: str):
        """Start recording a new test point."""
        self.current_test_point = test_point_id
        self.visualizer.set_test_point(test_point_id)
        self.frame_count = 0
        logger.info(f"Started recording test point: {test_point_id}")
    
    def process_frame(
        self,
        images: Dict[str, np.ndarray],
        attention_map: Optional[np.ndarray]
    ):
        """Process a single frame with attention visualization."""
        if not self.current_test_point or attention_map is None:
            return
        
        # Visualize for each view
        for view_name, image in images.items():
            self.visualizer.visualize_and_save(
                image,
                attention_map,
                view_name,
                self.frame_count
            )
            self.visualizer.save_attention_map(
                attention_map,
                view_name,
                self.frame_count
            )
        
        self.frame_count += 1
    
    def end_test_point(self):
        """Finalize recording for current test point."""
        if self.current_test_point:
            self.visualizer.finalize_video()
            logger.info(f"Ended recording test point: {self.current_test_point}")
            self.current_test_point = None
            self.frame_count = 0
