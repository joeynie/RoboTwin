#!/usr/bin/env python3
"""
Simple inference-time visualizer for attention, depth, and skill in one video.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch


def _load_viz_helpers():
    if __package__:
        from . import viz_object_and_depth_attention as viz
    else:
        import os as _os
        import sys as _sys

        script_dir = _os.path.dirname(_os.path.abspath(__file__))
        if script_dir not in _sys.path:
            _sys.path.insert(0, script_dir)
        import viz_object_and_depth_attention as viz

    return viz


class _VideoWriter:
    def __init__(self, output_path: str, fps: int, frame_size: tuple[int, int]):
        import imageio
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (W, H)
        self.writer = None

        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        self.writer = imageio.get_writer(output_path, fps=fps)

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        h, w = frame_rgb.shape[:2]
        if (w, h) != self.frame_size:
            frame_rgb = cv2.resize(frame_rgb, self.frame_size, interpolation=cv2.INTER_AREA)
        self.writer.append_data(frame_rgb)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class _QKCaptureHook:
    def __init__(self):
        self.qk_states = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple) and len(output) >= 3:
            self.qk_states = output[2]


class _SkillPlotter:
    def __init__(self, skill_names: Optional[list[str]] = None, index_offset: int = 1):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots(figsize=(6, 1.7), dpi=300)
        self._series: list[int] = []
        self._index_offset = index_offset
        self._skill_names = skill_names or [
            "approach target",
            "interact at target",
            "transport with object",
        ]

    def reset(self) -> None:
        self._series = []

    def update(self, skill_idx: int) -> None:
        self._series.append(int(skill_idx))

    def _ensure_names(self, num_classes: int) -> list[str]:
        names = list(self._skill_names)
        while len(names) < num_classes:
            names.append(f"Skill {len(names)}")
        return names[:num_classes]

    def render(self, width: int, height: int, num_classes: int) -> np.ndarray:
        ax = self._ax
        fig = self._fig
        ax.clear()

        if not self._series:
            ax.set_axis_off()
            fig.tight_layout()
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        frames = np.arange(1, len(self._series) + 1, dtype=np.int32)
        skills = np.asarray(self._series, dtype=np.int32) + self._index_offset

        ax.plot(frames, skills, color="#2F2F2FB3", linewidth=2.2, zorder=1)
        ax.scatter(frames[-1], skills[-1], s=90, color="#7162FCFF", edgecolor="white", linewidth=0.8, zorder=3)

        names = self._ensure_names(num_classes)
        y_ticks = [self._index_offset + i for i in range(len(names))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(self._index_offset - 0.5, self._index_offset + len(names) - 0.5)
        ax.set_xlim(1, max(20, len(self._series)))
        ax.set_xlabel("Frame", fontsize=9, fontweight="bold")
        ax.set_ylabel("Skill", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class InferenceTripleVisualizer:
    def __init__(
        self,
        output_dir: str = "./inference_viz",
        save_every_n_calls: int = 1,
        image_size: tuple[int, int] = (224, 224),
        fps: int = 5,
        alpha: float = 0.4,
        layer_idx: Optional[int] = None,
        use_origin_branch: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_calls = max(1, int(save_every_n_calls))
        self.image_size = image_size
        self.fps = fps
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.use_origin_branch = use_origin_branch

        self.calls = 0
        self.writer: Optional[_VideoWriter] = None
        self.policy = None
        self.model = None
        self.skill_plotter = _SkillPlotter()
        self.test_point_id: Optional[str] = None
        self._qk_hook: Optional[_QKCaptureHook] = None
        self._hook_handles: list[Any] = []
        self._latest_depth_kv = None
        self._viz = None

    def wrap_policy(self, policy: Any) -> Any:
        self.policy = policy
        self.model = policy._model

        if hasattr(policy, "_sample_kwargs"):
            policy._sample_kwargs["enable_attention_viz"] = True
        if self.layer_idx is not None:
            self.model.attention_viz_layer_idx = self.layer_idx

        self._wrap_depth_cache(self.model)
        self._attach_hooks(self.model)
        policy._return_skill_logits = True

        original_infer = policy.infer

        def wrapped_infer(obs: dict):
            result = original_infer(obs)
            self.maybe_visualize(obs, result)
            return result

        policy.infer = wrapped_infer
        return policy

    def start_test_point(self, test_point_id: str) -> None:
        self.test_point_id = test_point_id
        self.calls = 0
        self.skill_plotter.reset()
        self._close_writer()

    def end_test_point(self) -> None:
        self._close_writer()
        self.test_point_id = None

    def _close_writer(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _wrap_depth_cache(self, model: Any) -> None:
        if not hasattr(model, "_apply_depth_ablation"):
            return
        if hasattr(model, "_depth_kv_wrapped"):
            return

        original = model._apply_depth_ablation

        def wrapped(depth_layer_features):
            depth_kv, falseness_cos, strength_ratio = original(depth_layer_features)
            model.last_depth_kv = depth_kv
            return depth_kv, falseness_cos, strength_ratio

        model._apply_depth_ablation = wrapped
        model._depth_kv_wrapped = True

    def _find_attention_target(self, model: Any):
        if hasattr(model, 'paligemma_with_expert'):
            return model.paligemma_with_expert
        if hasattr(model, 'model') and hasattr(model.model, 'paligemma_with_expert'):
            return model.model.paligemma_with_expert
        for module in model.modules():
            if hasattr(module, 'paligemma_with_expert'):
                return module.paligemma_with_expert
        return None

    def _capture_depth_kwargs(self, module, args, kwargs):
        if 'depth_kv' in kwargs:
            self._latest_depth_kv = kwargs['depth_kv']

    def _attach_hooks(self, model: Any) -> None:
        target = self._find_attention_target(model)
        if target is None:
            return
        self._qk_hook = _QKCaptureHook()
        self._hook_handles.append(target.register_forward_hook(self._qk_hook))
        self._hook_handles.append(
            target.register_forward_pre_hook(self._capture_depth_kwargs, with_kwargs=True)
        )

    def _ensure_viz(self):
        if self._viz is None:
            self._viz = _load_viz_helpers()
        return self._viz

    def _infer_device(self):
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _resolve_layer_idx(self, qk_states: Any) -> int:
        if self.layer_idx is not None:
            return int(self.layer_idx)
        if isinstance(qk_states, (list, tuple)) and qk_states:
            return max(int(li) for li, _ in qk_states)
        return 0

    def _build_viz_args(self, layer_idx: int):
        depth_model_name = None
        if self.model is not None and hasattr(self.model, "config"):
            depth_model_name = getattr(self.model.config, "depth_model_name", None)
        return SimpleNamespace(
            layer_idx=layer_idx,
            head_idx=None,
            head_agg="mean",
            action_agg="mean",
            action_step=-1,
            mapping="reshape_or_interpolate",
            colormap="magma",
            object_colormap="jet",
            alpha=self.alpha,
            depth_model_name=depth_model_name,
            depth_viz_mode="predicted",
            depth_feature_idx=-1,
            depth_colormap="spectral_r",
            norm="per_image",
            attn_gamma=1.0,
            attn_clip_percentile=None,
            object_head_idx=0,
            object_view_idx=0,
        )

    def _resize_rgb(self, image_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        if image_rgb.shape[:2] == target_hw:
            return image_rgb
        return cv2.resize(image_rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)

    def _get_image_raw(self, obs: dict, key: str):
        images = {}
        if self.model is not None and getattr(self.model, "original_images_for_viz", None):
            images = self.model.original_images_for_viz
        if key in images:
            return images[key]
        return obs.get("images", {}).get(key)

    def _as_bhwc_uint8(self, image: Any) -> np.ndarray:
        viz = self._ensure_viz()
        if image is None:
            h, w = self.image_size
            return np.zeros((1, h, w, 3), dtype=np.uint8)
        return viz._to_uint8(viz._ensure_bhwc(viz._as_numpy(image)))

    def _get_qk_states(self):
        if self.model is None:
            return None
        if hasattr(self.model, 'last_qk_states') and self.model.last_qk_states is not None:
            return self.model.last_qk_states
        if self._qk_hook is not None:
            return self._qk_hook.qk_states
        return None

    def _get_depth_kv(self):
        if self.model is None:
            return None
        if hasattr(self.model, 'last_depth_kv') and self.model.last_depth_kv is not None:
            return self.model.last_depth_kv
        return self._latest_depth_kv

    def _extract_depth_tensor(self, depth_kv: Any) -> Any:
        if depth_kv is None:
            return None
        if isinstance(depth_kv, list) and depth_kv:
            depth_kv = depth_kv[0]
        if isinstance(depth_kv, dict):
            if 'depth_token_k' in depth_kv:
                return depth_kv['depth_token_k']
            if 'depth_token_v' in depth_kv:
                return depth_kv['depth_token_v']
            for value in depth_kv.values():
                return value
            return None
        return depth_kv

    def _collect_attention(self, qk_states: Any) -> Dict[int, torch.Tensor]:
        if isinstance(qk_states, dict):
            return qk_states
        attn: Dict[int, torch.Tensor] = {}
        if not isinstance(qk_states, (list, tuple)):
            return attn
        for layer_idx, qk_dict in qk_states:
            if not isinstance(qk_dict, dict):
                continue
            attn_probs = None
            # 根据 use_origin_branch 选择使用哪个分支的 attention_probs
            if self.use_origin_branch:
                attn_probs = qk_dict.get("attention_probs_origin")
            if attn_probs is None:
                attn_probs = qk_dict.get("attention_probs")
            if attn_probs is None and "attention" in qk_dict:
                q, k = qk_dict["attention"]
                head_dim = q.shape[-1]
                scaling = 1.0 / math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
                attn_probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
            if attn_probs is not None:
                attn[int(layer_idx)] = attn_probs
        return attn

    def _compute_object_attention_3views(self, qk_states: Any, layer_idx: int) -> list[np.ndarray]:
        """Compute object attention for all 3 views (head 0,1 average, last action token)."""
        # Get attention for the specified layer
        qk_data = None
        for li, data in qk_states:
            if li == layer_idx:
                qk_data = data
                break
        if qk_data is None:
            return [np.zeros(256) for _ in range(3)]
        
        # Get attention probs
        attn_probs = None
        if self.use_origin_branch:
            attn_probs = qk_data.get("attention_probs_origin")
        if attn_probs is None:
            attn_probs = qk_data.get("attention_probs")
        
        # Fallback: compute from Q/K
        if attn_probs is None and "attention" in qk_data:
            q, k = qk_data["attention"]
            if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                head_dim = q.shape[-1]
                scaling = 1.0 / math.sqrt(head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
                attn_probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        
        if attn_probs is None:
            return [np.zeros(256) for _ in range(3)]
        
        # attn_probs: [B, H, seq_len, seq_len]
        attn = attn_probs[0]  # [H, seq_len, seq_len]
        
        # Select heads 0 and 1, average them
        num_heads = attn.shape[0]
        head_indices = [h for h in [0, 1] if h < num_heads]
        if head_indices:
            attn = attn[head_indices].mean(dim=0)  # [seq_len, seq_len]
        else:
            attn = attn.mean(dim=0)
        
        # Take last row (last action token → all keys), slice first 768 (image tokens)
        last_action_attn = attn[-1]  # [seq_len]
        total_image_tokens = 768  # 3 views × 256 patches
        
        if last_action_attn.shape[0] >= total_image_tokens:
            image_attn = last_action_attn[:total_image_tokens].cpu().numpy()
        else:
            image_attn = last_action_attn.cpu().numpy()
            image_attn = np.pad(image_attn, (0, total_image_tokens - len(image_attn)))
        
        # Split into 3 views
        return [image_attn[i * 256 : (i + 1) * 256] for i in range(3)]

    def _create_attention_overlay(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        """Create attention overlay on image (same logic as inference_attention_visualizer)."""
        # attention_map: [16, 16] or [256]
        if attention_map.ndim == 1:
            side = int(np.sqrt(attention_map.shape[0]))
            attention_map = attention_map.reshape(side, side)
        
        # Resize attention map to match image size
        attention_resized = cv2.resize(
            attention_map.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize using global vmin/vmax
        if vmax > vmin:
            attention_normalized = ((attention_resized - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        else:
            attention_normalized = np.zeros_like(attention_resized, dtype=np.uint8)
        
        # Apply JET colormap
        attention_colored = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
        # Convert BGR to RGB
        attention_colored = attention_colored[:, :, ::-1]
        
        # Blend with alpha
        overlay = cv2.addWeighted(image, 1 - self.alpha, attention_colored, self.alpha, 0)
        return overlay

    def _compose_frame(
        self,
        orig_views: list[np.ndarray],
        attn_views: list[np.ndarray],
        skill_img: np.ndarray,
        depth_color: np.ndarray,
        depth_heat: np.ndarray,
    ) -> np.ndarray:
        """Compose frame with 3x2 grid (orig/attn for 3 views) + bottom row (skill, depth_color, depth_heat)."""
        h, w = orig_views[0].shape[:2]
        
        # Top 2 rows: 3 columns
        top_grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        for i, (orig, attn) in enumerate(zip(orig_views, attn_views)):
            top_grid[0:h, i * w : (i + 1) * w] = orig
            top_grid[h : h * 2, i * w : (i + 1) * w] = attn
        
        # Bottom row: skill | depth_color | depth_heat
        bottom_row = np.zeros((h, w * 3, 3), dtype=np.uint8)
        skill_resized = cv2.resize(skill_img, (w, h), interpolation=cv2.INTER_AREA)
        bottom_row[0:h, 0:w] = skill_resized
        bottom_row[0:h, w : w * 2] = depth_color
        bottom_row[0:h, w * 2 : w * 3] = depth_heat
        
        # Combine
        final = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)
        final[0 : h * 2] = top_grid
        final[h * 2 : h * 3] = bottom_row
        return final

    def maybe_visualize(self, obs: dict, result: dict) -> None:
        self.calls += 1
        if self.calls % self.save_every_n_calls != 0:
            return

        if self.model is None:
            return

        viz = self._ensure_viz()
        qk_states = self._get_qk_states()
        if qk_states is None:
            return
        layer_idx = self._resolve_layer_idx(qk_states)
        viz_args = self._build_viz_args(layer_idx)

        base_raw = self._get_image_raw(obs, "cam_high")
        base_bhwc_uint8 = self._as_bhwc_uint8(base_raw)

        depth_viz = None
        depth_kv = self._get_depth_kv()
        if depth_kv is not None:
            depth_attn = viz._compute_depth_attention_from_qk(
                self.model, qk_states, viz_args.layer_idx, depth_kv=depth_kv
            )
            depth_viz = viz.compute_overlay_for_sample(
                attn=depth_attn,
                image_bhwc_uint8=base_bhwc_uint8,
                input_npz=None,
                meta={},
                args=viz_args,
                device=self._infer_device(),
                batch_idx=0,
            )

        # Compute object attention tokens for all 3 views using consistent logic
        view_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        view_tokens = self._compute_object_attention_3views(qk_states, layer_idx)

        # Global min/max for normalization (same as inference_attention_visualizer)
        all_tokens = np.concatenate([t.flatten() for t in view_tokens])
        global_min, global_max = float(all_tokens.min()), float(all_tokens.max())

        # Get all 3 view images and their overlays using our own overlay logic
        orig_views = []
        attn_views = []
        for view_idx, key in enumerate(view_keys):
            img_bhwc = self._as_bhwc_uint8(self._get_image_raw(obs, key))
            img_rgb = img_bhwc[0]  # [H, W, 3]
            overlay = self._create_attention_overlay(
                img_rgb, view_tokens[view_idx], global_min, global_max
            )
            orig_views.append(img_rgb)
            attn_views.append(overlay)

        cell_h, cell_w = orig_views[0].shape[:2]
        
        # Resize all views to same size
        orig_views = [self._resize_rgb(v, (cell_h, cell_w)) for v in orig_views]
        attn_views = [self._resize_rgb(v, (cell_h, cell_w)) for v in attn_views]

        # Depth visualization
        if depth_viz is not None:
            depth_color = self._resize_rgb(depth_viz["base_rgb"], (cell_h, cell_w))
            depth_heat = self._resize_rgb(depth_viz["heatmap_color"], (cell_h, cell_w))
        else:
            depth_color = orig_views[0]
            depth_heat = np.zeros_like(orig_views[0])

        num_classes = int(getattr(self.model, "skill_num_classes", 3))
        if "skill_logits" in result:
            logits = np.asarray(result["skill_logits"]).reshape(-1)
            if logits.size > 0:
                self.skill_plotter.update(int(np.argmax(logits)))

        skill_img = self.skill_plotter.render(cell_w, cell_h, num_classes)
        frame = self._compose_frame(orig_views, attn_views, skill_img, depth_color, depth_heat)

        if self.writer is None:
            stamp = self.test_point_id or time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"viz_{stamp}.mp4")
            self.writer = _VideoWriter(output_path, self.fps, (frame.shape[1], frame.shape[0]))

        self.writer.write(frame)
