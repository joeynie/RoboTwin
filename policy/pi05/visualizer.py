#!/usr/bin/env python3
"""
Simple inference-time visualizer for attention, depth, and skill in one video.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize to target size without distortion by padding."""
    cur_h, cur_w = image.shape[:2]
    if cur_h == height and cur_w == width:
        return image

    ratio = max(cur_w / width, cur_h / height)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 3:
        canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((height, width), dtype=image.dtype)

    pad_h = max(0, (height - resized_h) // 2)
    pad_w = max(0, (width - resized_w) // 2)
    canvas[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    return canvas


class _VideoWriter:
    def __init__(self, output_path: str, fps: int, frame_size: tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (W, H)
        self.backend = "cv2"
        self.writer = None

        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        h, w = frame_rgb.shape[:2]
        if (w, h) != self.frame_size:
            frame_rgb = cv2.resize(frame_rgb, self.frame_size, interpolation=cv2.INTER_AREA)
        frame_bgr = frame_rgb[:, :, ::-1]
        self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


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
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
        ax.set_xlabel("Frame", fontsize=9, fontweight="bold")
        ax.set_ylabel("Skill", fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class InferenceTripleVisualizer:
    def __init__(
        self,
        output_dir: str = "./inference_viz",
        save_every_n_calls: int = 1,
        image_size: tuple[int, int] = (224, 224),
        fps: int = 10,
        alpha: float = 0.35,
        layer_idx: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_calls = max(1, int(save_every_n_calls))
        self.image_size = image_size
        self.fps = fps
        self.alpha = alpha
        self.layer_idx = layer_idx

        self.calls = 0
        self.writer: Optional[_VideoWriter] = None
        self.policy = None
        self.model = None
        self.skill_plotter = _SkillPlotter()
        self.test_point_id: Optional[str] = None

        self.num_images = 3
        self.num_patches_per_image = 256

    def wrap_policy(self, policy: Any) -> Any:
        self.policy = policy
        self.model = policy._model

        if hasattr(policy, "_sample_kwargs"):
            policy._sample_kwargs["enable_attention_viz"] = True
        if self.layer_idx is not None and hasattr(self.model, "attention_viz_layer_idx"):
            self.model.attention_viz_layer_idx = self.layer_idx

        self._wrap_depth_cache(self.model)
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

    def _to_uint8_hwc(self, image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu().numpy()
        else:
            img = np.asarray(image)

        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            if img.min() < 0:
                img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
            elif img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        target_h, target_w = self.image_size
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = resize_with_pad(img, target_h, target_w)
        return img

    def _get_base_image(self, obs: dict) -> np.ndarray:
        images = {}
        if self.model is not None and getattr(self.model, "original_images_for_viz", None):
            images = self.model.original_images_for_viz
        base = images.get("cam_high")
        if base is None:
            base = obs.get("images", {}).get("cam_high")
        if base is None:
            base = np.zeros((*self.image_size, 3), dtype=np.uint8)
        return self._to_uint8_hwc(base)

    def _collect_attention(self, qk_states: Any) -> Dict[int, torch.Tensor]:
        if isinstance(qk_states, dict):
            return qk_states
        attn: Dict[int, torch.Tensor] = {}
        if not isinstance(qk_states, (list, tuple)):
            return attn
        for layer_idx, qk_dict in qk_states:
            if not isinstance(qk_dict, dict):
                continue
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

    def _attention_map(self, qk_states: Any) -> Optional[np.ndarray]:
        attn_by_layer = self._collect_attention(qk_states)
        if not attn_by_layer:
            return None
        if self.layer_idx is None:
            layer_idx = sorted(attn_by_layer.keys())[-1]
        else:
            layer_idx = self.layer_idx
        if layer_idx not in attn_by_layer:
            return None

        layer_attn = attn_by_layer[layer_idx]
        if isinstance(layer_attn, torch.Tensor):
            layer_attn = layer_attn.detach()
        layer_attn = layer_attn[0]  # [num_heads, seq_len, seq_len]
        seq_len = layer_attn.shape[-1]
        image_tokens_end = min(self.num_images * self.num_patches_per_image, seq_len)
        last_action_idx = seq_len - 1

        attn = layer_attn[:, last_action_idx, :image_tokens_end]
        attn = attn.mean(dim=0)
        attn = attn[: self.num_images * self.num_patches_per_image]
        if attn.numel() != self.num_images * self.num_patches_per_image:
            return None
        attn = attn.view(self.num_images, 16, 16)[0]
        return attn.cpu().numpy()

    def _overlay(self, image: np.ndarray, attn_map: np.ndarray, colormap: int) -> np.ndarray:
        attn_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        vmin, vmax = float(attn_resized.min()), float(attn_resized.max())
        if vmax > vmin:
            attn_norm = ((attn_resized - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            attn_norm = np.zeros_like(attn_resized, dtype=np.uint8)
        heat = cv2.applyColorMap(attn_norm, colormap)
        return cv2.addWeighted(image, 1.0 - self.alpha, heat, self.alpha, 0)

    def _depth_views(self, depth_kv: Any, base_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if depth_kv is None:
            blank = np.zeros_like(base_image)
            return blank, blank

        if isinstance(depth_kv, (tuple, list)):
            depth_kv = depth_kv[0]
        depth = torch.as_tensor(depth_kv).float()
        while depth.ndim > 2:
            depth = depth.mean(dim=0)
        if depth.ndim == 2:
            depth = depth.mean(dim=-1)
        depth = depth.flatten()
        n = depth.numel()
        side = int(math.sqrt(n))
        if side * side == n:
            depth_map = depth.view(side, side)
        else:
            depth_map = depth.view(1, -1)
        depth_map = depth_map.detach().cpu().numpy()
        depth_map = cv2.resize(depth_map, (base_image.shape[1], base_image.shape[0]), interpolation=cv2.INTER_LINEAR)

        vmin, vmax = float(depth_map.min()), float(depth_map.max())
        if vmax > vmin:
            depth_norm = ((depth_map - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        depth_base = cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE)
        depth_heat = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return depth_base, depth_heat

    def _compose_frame(
        self,
        base: np.ndarray,
        attn: np.ndarray,
        depth_base: np.ndarray,
        depth_heat: np.ndarray,
        skill_img: np.ndarray,
    ) -> np.ndarray:
        h, w = base.shape[:2]
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        grid[0:h, 0:w] = base
        grid[0:h, w : w * 2] = attn
        grid[h : h * 2, 0:w] = depth_base
        grid[h : h * 2, w : w * 2] = depth_heat

        if skill_img.shape[1] != w * 2:
            skill_img = cv2.resize(skill_img, (w * 2, skill_img.shape[0]), interpolation=cv2.INTER_AREA)
        final = np.zeros((grid.shape[0] + skill_img.shape[0], grid.shape[1], 3), dtype=np.uint8)
        final[: grid.shape[0]] = grid
        final[grid.shape[0] :] = skill_img
        return final

    def maybe_visualize(self, obs: dict, result: dict) -> None:
        self.calls += 1
        if self.calls % self.save_every_n_calls != 0:
            return

        if self.model is None:
            return

        base = self._get_base_image(obs)
        qk_states = getattr(self.model, "last_qk_states", None)
        attn_map = self._attention_map(qk_states)
        if attn_map is None:
            return

        attn_overlay = self._overlay(base, attn_map, cv2.COLORMAP_JET)
        depth_base, depth_heat = self._depth_views(getattr(self.model, "last_depth_kv", None), base)

        num_classes = int(getattr(self.model, "skill_num_classes", 3))
        if "skill_logits" in result:
            logits = np.asarray(result["skill_logits"]).reshape(-1)
            if logits.size > 0:
                self.skill_plotter.update(int(np.argmax(logits)))

        skill_img = self.skill_plotter.render(base.shape[1] * 2, base.shape[0] // 2, num_classes)
        frame = self._compose_frame(base, attn_overlay, depth_base, depth_heat, skill_img)

        if self.writer is None:
            stamp = self.test_point_id or time.strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"viz_{stamp}.mp4")
            self.writer = _VideoWriter(output_path, self.fps, (frame.shape[1], frame.shape[0]))

        self.writer.write(frame)
