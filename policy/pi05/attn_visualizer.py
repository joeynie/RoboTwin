import numpy as np
import cv2
import imageio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention map extraction and visualization."""
    save_dir: str = "attn_maps"  # Directory to save visualization
    fps: int = 10  # Video FPS
    patch_size: int = 14  # Patch size for upsampling attention map (14 for ViT)
    colormap: str = "jet"  # Colormap for visualization


def create_attention_overlay(image: np.ndarray, attention_map: np.ndarray, patch_size: int = 14) -> np.ndarray:
    """Create an overlay image combining the original image with an attention map heatmap.
    
    Args:
        image: RGB image as numpy array (H, W, 3) in uint8
        attention_map: Attention map (H', W') with values in [0, 1]
        patch_size: Patch size for upsampling (e.g., 14 for ViT)
    
    Returns:
        RGB image with attention overlay as numpy array (H, W, 3) in uint8
    """
    # Upsample attention map to match image size
    img_h, img_w = image.shape[:2]
    
    # Resize attention map to match image dimensions
    attn_map_resized = cv2.resize(attention_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    if np.max(attn_map_resized) > np.min(attn_map_resized):
        norm_map = (attn_map_resized - np.min(attn_map_resized)) / (
            np.max(attn_map_resized) - np.min(attn_map_resized) + 1e-8
        )
    else:
        norm_map = np.zeros_like(attn_map_resized)
    
    norm_map_uint8 = (norm_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map_uint8, cv2.COLORMAP_JET)
    
    image_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb


class AttentionVideoRecorder:
    """Records videos with per-view attention map overlays."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames_buffer: Dict[str, List[np.ndarray]] = {}
        self.current_test_point = None
        self.frame_count = 0
    
    def start_test_point(self, test_point_id: str):
        """Start recording a new test point."""
        self.current_test_point = test_point_id
        self.frames_buffer = {
            "cam_high": [],
            "cam_left_wrist": [],
            "cam_right_wrist": [],
        }
        self.frame_count = 0
        logger.info(f"Started recording test point: {test_point_id}")
    def process_frame(
        self,
        images: Dict[str, np.ndarray],
        attention_maps: Dict[str, np.ndarray]
    ):
        """Process a single frame with per-view attention visualization.
        
        Args:
            images: Dict of images for each view (already HWC uint8 format)
            attention_maps: Dict[str, np.ndarray] with per-view attention maps (each [H, W] or [16, 16])
        """
        if not self.current_test_point:
            return
        
        for view_name, image in images.items():
            if image is None:
                continue
            
            attention_map = attention_maps.get(view_name, None)
            
            # Apply attention overlay if provided
            if attention_map is not None and attention_map.ndim == 2:
                try:
                    overlay = create_attention_overlay(image, attention_map, self.config.patch_size)
                except Exception as e:
                    logger.warning(f"Failed to create attention overlay for {view_name}: {e}")
                    overlay = image.astype(np.uint8) if image.dtype != np.uint8 else image
            else:
                overlay = image.astype(np.uint8) if image.dtype != np.uint8 else image
            
            # Buffer frame for video
            if view_name in self.frames_buffer:
                self.frames_buffer[view_name].append(overlay)
        
        self.frame_count += 1
    
    def end_test_point(self):
        """Finalize recording for current test point and save video."""
        if not self.current_test_point:
            return
        
        logger.info(f"Total frames captured: {self.frame_count}")
        
        # Create output directory and save videos
        test_dir = self.save_dir / self.current_test_point
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for view_name, frames in self.frames_buffer.items():
            if not frames:
                logger.warning(f"No frames captured for {view_name}")
                continue
            
            # Prepare frames for video (ensure RGB uint8)
            video_frames = []
            for frame in frames:
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                # Handle channel issues
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                
                if frame.ndim == 3 and frame.shape[2] == 3:
                    video_frames.append(frame)
            
            if not video_frames:
                logger.warning(f"No valid frames for {view_name}, skipping")
                continue
            
            # Save video using imageio
            video_path = test_dir / f"{view_name}.mp4"
            try:
                imageio.mimsave(
                    str(video_path),
                    video_frames,
                    fps=self.config.fps,
                    codec='libx264',
                    pixelformat='yuv420p'
                )
                logger.info(f"Saved video: {video_path}")
            except Exception as e:
                logger.error(f"Failed to save video {video_path}: {e}")
        
        self.current_test_point = None
        self.frame_count = 0
