#!/usr/bin/env python3

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

# Optional deps
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - required for saving
    raise RuntimeError("PIL is required for saving images. Please install pillow.") from exc

try:
    from matplotlib import cm as mpl_cm  # type: ignore
except Exception:  # pragma: no cover - optional
    mpl_cm = None


_DEPTH_ENCODER_CACHE: Dict[Tuple[str, str], Any] = {}
_DEPTH_FULL_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


# -----------------------------
# Utility helpers
# -----------------------------

def _print_shape(name: str, arr: Any) -> None:
    if isinstance(arr, torch.Tensor):
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
    elif isinstance(arr, np.ndarray):
        shape = arr.shape
        dtype = str(arr.dtype)
    else:
        shape = "?"
        dtype = type(arr)
    print(f"[viz] {name}: shape={shape}, dtype={dtype}")


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _ensure_bhwc(image: np.ndarray) -> np.ndarray:
    """Normalize image array into [B, H, W, C] RGB-like layout."""
    img = image
    if img.ndim == 2:
        # grayscale HxW -> HxW x1
        img = img[:, :, None]
    if img.ndim == 3:
        # HWC or CHW
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img[None, ...]
    elif img.ndim == 4:
        # BCHW or BHWC
        if img.shape[1] in (1, 3):
            img = np.transpose(img, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")

    # Ensure 3 channels
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _to_uint8(image_bhwc: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0,255] for visualization."""
    img = image_bhwc.astype(np.float32)
    if img.min() >= -1.0 and img.max() <= 1.0:
        img = (img + 1.0) * 127.5
    elif img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _move_observation_to_device(obs: Any, device: torch.device) -> Any:
    """Best-effort move of Observation-like object to device."""
    if obs is None:
        return obs

    def _to_dev(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    def _move_mapping(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _to_dev(v) for k, v in value.items()}
        return value

    # If obs is a frozen dataclass (e.g. flax.struct), use dataclasses.replace.
    if dataclasses.is_dataclass(obs):
        kwargs: Dict[str, Any] = {}
        if hasattr(obs, "images"):
            kwargs["images"] = _move_mapping(getattr(obs, "images"))
        if hasattr(obs, "image_masks"):
            kwargs["image_masks"] = _move_mapping(getattr(obs, "image_masks"))
        for attr in [
            "state",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "token_ar_mask",
            "token_loss_mask",
            "skill_id",
            "skill_soft",
        ]:
            if hasattr(obs, attr):
                kwargs[attr] = _to_dev(getattr(obs, attr))
        try:
            return dataclasses.replace(obs, **kwargs)
        except Exception:
            # Fall back to best-effort mutation if replace fails.
            pass

    if hasattr(obs, "images") and isinstance(obs.images, dict):
        obs.images = {k: _to_dev(v) for k, v in obs.images.items()}
    if hasattr(obs, "image_masks") and isinstance(obs.image_masks, dict):
        obs.image_masks = {k: _to_dev(v) for k, v in obs.image_masks.items()}
    for attr in [
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "skill_id",
        "skill_soft",
    ]:
        if hasattr(obs, attr):
            setattr(obs, attr, _to_dev(getattr(obs, attr)))
    return obs


def _resize_2d(map_2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to target (H, W) with cubic interpolation."""
    target_h, target_w = target_hw
    if cv2 is not None:
        return cv2.resize(map_2d, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    # PIL fallback
    pil = Image.fromarray(map_2d.astype(np.float32), mode="F")
    pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(pil)


def _apply_colormap(values_01: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply a colormap to a [H,W] heatmap in [0,1], returning RGB uint8."""
    cmap_name = cmap_name.lower()
    values_uint8 = np.clip(values_01 * 255.0, 0, 255).astype(np.uint8)

    # Prefer OpenCV colormaps if available (fast, and consistent)
    if cv2 is not None:
        cv2_cmaps = {
            "jet": getattr(cv2, "COLORMAP_JET", None),
            "magma": getattr(cv2, "COLORMAP_MAGMA", None),
            "viridis": getattr(cv2, "COLORMAP_VIRIDIS", None),
            "inferno": getattr(cv2, "COLORMAP_INFERNO", None),
            "turbo": getattr(cv2, "COLORMAP_TURBO", None),
            "plasma": getattr(cv2, "COLORMAP_PLASMA", None),
        }
        cv2_cmap = cv2_cmaps.get(cmap_name)
        if cv2_cmap is not None:
            colored_bgr = cv2.applyColorMap(values_uint8, cv2_cmap)
            # Convert BGR -> RGB
            return colored_bgr[:, :, ::-1]

    # Fallback to matplotlib
    if mpl_cm is None:
        raise RuntimeError("Neither OpenCV nor matplotlib colormap is available.")
    if cmap_name == "spectral":
        mpl_name = "Spectral"
    elif cmap_name == "spectral_r":
        mpl_name = "Spectral_r"
    else:
        mpl_name = cmap_name
    cmap = mpl_cm.get_cmap(mpl_name)
    colored = cmap(values_01)[:, :, :3]  # RGB in [0,1]
    return np.clip(colored * 255.0, 0, 255).astype(np.uint8)


def _load_depth_anything_weights(model: torch.nn.Module, model_path: str) -> None:
    """Load Depth Anything weights from a local path (safetensors preferred)."""
    path = Path(model_path)
    if path.is_dir():
        ckpt = None
        for fname in ["model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"]:
            candidate = path / fname
            if candidate.exists():
                ckpt = candidate
                break
        if ckpt is None:
            # Fallback: first .safetensors file
            for candidate in path.glob("*.safetensors"):
                ckpt = candidate
                break
        if ckpt is None:
            raise RuntimeError(f"No checkpoint file found under: {path}")
        ckpt_path = str(ckpt)
    else:
        if not path.exists():
            raise RuntimeError(f"Depth model path not found: {model_path}")
        ckpt_path = str(path)

    print(f"[viz] Loading DepthAnything weights from: {ckpt_path}")

    state = None
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required to load .safetensors checkpoints") from exc
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError("Unexpected checkpoint format; expected a state_dict-like dict.")

    model_keys = set(model.state_dict().keys())
    prefixes = [
        "depth_anything.",
        "model.",
        "module.",
        "depth_model.",
        "backbone.",
        "encoder.",
    ]

    stripped_state = {}
    for k in state.keys():
        stripped = k
        for pfx in prefixes:
            if stripped.startswith(pfx):
                stripped = stripped[len(pfx) :]
        if stripped not in stripped_state:
            stripped_state[stripped] = k

    filtered_state = {}
    for mk in model_keys:
        if mk in state:
            filtered_state[mk] = state[mk]
            continue
        stripped_mk = mk
        for pfx in prefixes:
            if stripped_mk.startswith(pfx):
                stripped_mk = stripped_mk[len(pfx) :]
        if stripped_mk in state:
            filtered_state[mk] = state[stripped_mk]
            continue
        if mk in stripped_state:
            filtered_state[mk] = state[stripped_state[mk]]
            continue
        if stripped_mk in stripped_state:
            filtered_state[mk] = state[stripped_state[stripped_mk]]

    if not filtered_state:
        print("[viz] Warning: no matching keys found when loading DepthAnything weights.")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"[viz] DepthAnything missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[viz] DepthAnything unexpected keys (showing up to 20): {unexpected[:20]}")


def _get_cached_depth_anything_full(
    args: argparse.Namespace, device: torch.device
) -> Tuple[Any, torch.nn.Module]:
    """Load (or reuse) full Depth Anything model + image processor."""
    key = (str(args.depth_model_name), str(device))
    if key in _DEPTH_FULL_MODEL_CACHE:
        return _DEPTH_FULL_MODEL_CACHE[key]

    from transformers import AutoImageProcessor
    from depth.configuration_depth_anything import DepthAnythingConfig
    from depth.modeling_depth_anything_full import DepthAnythingForDepthEstimationFull

    print("[viz] Loading DepthAnything full model (cached).")
    image_processor = AutoImageProcessor.from_pretrained(args.depth_model_name)
    config = DepthAnythingConfig.from_pretrained(args.depth_model_name)
    if getattr(config.backbone_config, "reshape_hidden_states", False):
        print("[viz] Forcing backbone_config.reshape_hidden_states=False for neck+head.")
        config.backbone_config.reshape_hidden_states = False
    depth_model = DepthAnythingForDepthEstimationFull(config=config).to(device)
    _load_depth_anything_weights(depth_model, args.depth_model_name)
    depth_model.eval()

    _DEPTH_FULL_MODEL_CACHE[key] = (image_processor, depth_model)
    return image_processor, depth_model


def _get_predicted_depth_from_full_model(
    image_rgb_uint8: np.ndarray, args: argparse.Namespace, device: torch.device
) -> np.ndarray:
    """Run full Depth Anything (backbone+neck+head) to obtain predicted depth."""
    print("[viz] Running DepthAnything full model for predicted depth.")
    image_processor, depth_model = _get_cached_depth_anything_full(args, device)

    pil = Image.fromarray(image_rgb_uint8)
    inputs = image_processor(images=pil, return_tensors="pt")
    model_dtype = next(depth_model.parameters()).dtype
    inputs = {k: v.to(device=device, dtype=model_dtype) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = depth_model(**inputs)

    if hasattr(outputs, "predicted_depth"):
        pred = outputs.predicted_depth
    elif isinstance(outputs, (tuple, list)):
        pred = outputs[0]
    else:
        pred = outputs

    if not isinstance(pred, torch.Tensor):
        raise RuntimeError("Depth model did not return a tensor predicted_depth.")

    depth_map = pred[0].float().cpu().numpy()
    depth_map = _resize_2d(depth_map, (image_rgb_uint8.shape[0], image_rgb_uint8.shape[1]))
    depth_color = _visualize_depth_map(
        depth_map,
        ret_minmax=False,
        ret_type=np.uint8,
        cmap=args.depth_colormap,
    )
    _print_shape("predicted_depth_color", depth_color)
    return depth_color


# -----------------------------
# Required functions
# -----------------------------

def load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Load input NPZ and extract the main view image plus optional metadata."""
    if not args.input_npz:
        raise ValueError("--input_npz is required.")

    data = np.load(args.input_npz, allow_pickle=True)
    keys = list(data.keys())
    print(f"[viz] Loaded input_npz keys: {keys}")

    # Try to locate the main image.
    image = None
    if args.image_key and args.image_key in data:
        image = data[args.image_key]
        print(f"[viz] Using image key from --image_key: {args.image_key}")
    else:
        candidate_keys = [
            "image",
            "images",
            "rgb",
            "obs_image",
            "agentview_image",
            "main_image",
            "image_0",
            "base_0_rgb",
        ]
        for k in candidate_keys:
            if k in data:
                image = data[k]
                print(f"[viz] Using image key: {k}")
                break

    # If image is stored as a dict-like object in NPZ, try to pull a value out.
    if image is not None and isinstance(image, np.ndarray) and image.dtype == object:
        try:
            image_obj = image.item()
            if isinstance(image_obj, dict):
                # Prefer common base camera keys.
                for k in ["base_0_rgb", "image", "rgb"]:
                    if k in image_obj:
                        image = image_obj[k]
                        print(f"[viz] Using image from dict key: {k}")
                        break
        except Exception:
            pass

    if image is None:
        raise ValueError(
            "Could not find a usable image in input_npz. "
            "Please provide --image_key or include one of: image/images/rgb/base_0_rgb."
        )

    image = _as_numpy(image)
    image_bhwc = _ensure_bhwc(image)
    image_bhwc_uint8 = _to_uint8(image_bhwc)

    _print_shape("input_image_bhwc", image_bhwc_uint8)

    # Optional sample_id for naming
    sample_id = None
    for k in ["sample_id", "frame_id", "id"]:
        if k in data:
            sample_id = str(_as_numpy(data[k]).reshape(-1)[0])
            print(f"[viz] Using sample_id from input_npz key: {k}={sample_id}")
            break
    if args.sample_id:
        sample_id = args.sample_id
        print(f"[viz] Using sample_id from --sample_id: {sample_id}")
    if sample_id is None:
        sample_id = f"idx{args.idx}"

    return {
        "npz": data,
        "image_bhwc_uint8": image_bhwc_uint8,
        "sample_id": sample_id,
    }


def _create_libero_loader(args: argparse.Namespace, num_batches: Optional[int]):
    """Create a LIBERO data loader with repo/root overrides applied."""
    # Lazy imports to avoid heavy deps when not needed.
    from openpi.training import config as train_config
    from openpi.training import data_loader as data_loader_mod

    cfg = _load_config(args.config)
    if cfg is None:
        raise RuntimeError("Failed to load config for LIBERO sample.")

    # Override repo_id / local_root_dir if provided.
    data_factory = cfg.data
    base_config = data_factory.base_config
    if args.libero_root:
        if base_config is None:
            base_config = train_config.DataConfig()
        base_config = dataclasses.replace(base_config, local_root_dir=args.libero_root)
    if args.libero_repo_id:
        data_factory = dataclasses.replace(data_factory, repo_id=args.libero_repo_id, base_config=base_config)
    elif args.libero_root:
        data_factory = dataclasses.replace(data_factory, base_config=base_config)

    cfg = dataclasses.replace(cfg, data=data_factory, batch_size=1, num_workers=args.libero_num_workers)

    # Build loader (skip norm stats to avoid asset dependency for visualization).
    loader = data_loader_mod.create_data_loader(
        cfg,
        framework="pytorch",
        shuffle=False,
        num_batches=num_batches,
        skip_norm_stats=True,
        split=args.libero_split,
    )
    return loader


def load_libero_sample(args: argparse.Namespace) -> Dict[str, Any]:
    """Load a single LIBERO sample via the configured data loader (no NPZ required)."""
    if not args.config:
        raise ValueError("--config is required to load LIBERO samples.")

    loader = _create_libero_loader(args, num_batches=args.libero_sample_idx + 1)

    observation = None
    actions = None
    for i, batch in enumerate(loader):
        observation, actions, _ = batch
        if i >= args.libero_sample_idx:
            break

    if observation is None:
        raise RuntimeError("Failed to load LIBERO sample. Check dataset path and index.")

    # Extract base image for visualization.
    base_img = observation.images.get("base_0_rgb")
    if base_img is None:
        raise RuntimeError("LIBERO observation missing base_0_rgb image.")

    base_np = _as_numpy(base_img)
    base_bhwc = _ensure_bhwc(base_np)
    base_bhwc_uint8 = _to_uint8(base_bhwc)

    _print_shape("libero_base_image_bhwc", base_bhwc_uint8)

    sample_id = f"libero_{args.libero_sample_idx}"
    return {
        "npz": None,
        "observation": observation,
        "actions": actions,
        "image_bhwc_uint8": base_bhwc_uint8,
        "sample_id": sample_id,
    }


def iter_libero_samples(
    args: argparse.Namespace,
    *,
    start_idx: int,
    num_frames: Optional[int],
    frame_skip: int,
) -> Iterator[Tuple[int, Any, Any, np.ndarray]]:
    """Iterate LIBERO samples, yielding (index, observation, actions, image_bhwc_uint8)."""
    if start_idx < 0:
        raise ValueError("--libero_sample_idx must be >= 0")
    if frame_skip <= 0:
        raise ValueError("--video_frame_skip must be >= 1")

    if num_frames is not None and num_frames > 0:
        num_batches = start_idx + (num_frames - 1) * frame_skip + 1
    else:
        num_batches = None

    loader = _create_libero_loader(args, num_batches=num_batches)

    yielded = 0
    for i, batch in enumerate(loader):
        if i < start_idx:
            continue
        if (i - start_idx) % frame_skip != 0:
            continue

        observation, actions, _ = batch
        base_img = observation.images.get("base_0_rgb")
        if base_img is None:
            raise RuntimeError("LIBERO observation missing base_0_rgb image.")

        base_np = _as_numpy(base_img)
        base_bhwc = _ensure_bhwc(base_np)
        base_bhwc_uint8 = _to_uint8(base_bhwc)

        yield i, observation, actions, base_bhwc_uint8

        yielded += 1
        if num_frames is not None and num_frames > 0 and yielded >= num_frames:
            break


def load_attention(args: argparse.Namespace) -> Dict[str, Any]:
    """Load attention from NPZ. Supports attn_probs_mod or attn_probs with depth slice."""
    if not args.attn_npz:
        return {"attn": None, "meta": {}}

    attn_data = np.load(args.attn_npz, allow_pickle=True)
    print(f"[viz] Loaded attn_npz keys: {list(attn_data.keys())}")

    attn = None
    meta: Dict[str, Any] = {}

    # Prefer explicit attn_probs_mod.
    if "attn_probs_mod" in attn_data:
        attn = attn_data["attn_probs_mod"]
        print("[viz] Using attn_probs_mod from attn_npz")
    else:
        # Try a layer-specific key (e.g., attn_probs_mod_L12)
        layer_key_candidates = [
            f"attn_probs_mod_L{args.layer_idx}",
            f"attn_probs_mod_layer{args.layer_idx}",
        ]
        for k in layer_key_candidates:
            if k in attn_data:
                attn = attn_data[k]
                print(f"[viz] Using layer-specific key from attn_npz: {k}")
                break

    if attn is None and "attn_probs" in attn_data:
        if args.depth_start is None or args.depth_end is None:
            raise ValueError(
                "attn_probs found but depth token slice not provided. "
                "Please set --depth_start and --depth_end."
            )
        full_attn = attn_data["attn_probs"]
        _print_shape("attn_probs", full_attn)
        attn = full_attn[..., args.depth_start : args.depth_end]
        print(
            f"[viz] Sliced attn_probs with depth range [{args.depth_start}, {args.depth_end}) "
            f"-> shape {attn.shape}"
        )

    if attn is None:
        raise ValueError(
            "No attention maps found in attn_npz. Expected attn_probs_mod or attn_probs."
        )

    # Collect optional metadata for token-to-patch mapping.
    for k in [
        "depth_token_coords",
        "token_coords",
        "depth_token_hw",
        "token_grid_hw",
        "depth_grid_hw",
        "depth_token_indices",
        "token_indices",
    ]:
        if k in attn_data:
            meta[k] = attn_data[k]

    return {"attn": attn, "meta": meta}


def _get_cached_depth_encoder(args: argparse.Namespace, device: torch.device):
    """Load (or reuse) DepthEncoder for proxy depth visualization."""
    key = (str(args.depth_model_name), str(device))
    if key in _DEPTH_ENCODER_CACHE:
        return _DEPTH_ENCODER_CACHE[key]

    from depth.model import DepthEncoder

    print(f"[viz] Loading DepthEncoder from: {args.depth_model_name}")
    depth_encoder = DepthEncoder(depth_model_name=args.depth_model_name, feature_dim=1024, freeze_depth_model=True)
    depth_encoder = depth_encoder.to(device)
    depth_encoder.eval()

    _DEPTH_ENCODER_CACHE[key] = depth_encoder
    return depth_encoder


def get_colored_depth(
    image_rgb_uint8: np.ndarray,
    args: argparse.Namespace,
    input_npz: Optional[Dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    """
    Obtain a colored depth visualization.

    Priority:
    1) Use precomputed colored depth if present in input_npz.
    2) Use raw depth map if present, and apply colormap.
    3) Run DepthEncoder / DepthAnything to produce a proxy depth map from features.
    """
    # 1) Precomputed colored depth
    if input_npz is not None:
        for k in ["colored_depth", "depth_colored", "depth_rgb"]:
            if k in input_npz:
                depth_img = _as_numpy(input_npz[k])
                depth_img = _ensure_bhwc(depth_img)[0]
                depth_img = _to_uint8(depth_img)
                print(f"[viz] Using precomputed colored depth from key: {k}")
                _print_shape("colored_depth", depth_img)
                return depth_img

    # 2) Raw depth map
    if input_npz is not None:
        for k in ["depth", "depth_map", "pred_depth", "depth_anything"]:
            if k in input_npz:
                depth = _as_numpy(input_npz[k])
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth[:, :, 0]
                if depth.ndim == 3:
                    # assume batch
                    depth = depth[0]
                print(f"[viz] Using raw depth map from key: {k}")
                depth_color = _visualize_depth_map(
                    depth,
                    ret_minmax=False,
                    ret_type=np.uint8,
                    cmap=args.depth_colormap,
                )
                _print_shape("depth_color", depth_color)
                return depth_color

    # 3) Run DepthEncoder / DepthAnything (backbone-only in this codebase)
    if args.depth_model_name is None:
        raise RuntimeError(
            "Depth model weights not provided. "
            "Please pass --depth_model_name pointing to a local Depth Anything checkpoint, "
            "or include depth/colored_depth in input_npz."
        )

    if not Path(args.depth_model_name).exists():
        raise RuntimeError(
            f"Depth model path not found: {args.depth_model_name}. "
            "To avoid internet downloads, please provide a local path."
        )

    if args.depth_viz_mode == "predicted":
        try:
            return _get_predicted_depth_from_full_model(image_rgb_uint8, args, device)
        except Exception as exc:
            print(f"[viz] Predicted depth failed ({exc}); falling back to backbone proxy.")

    depth_encoder = _get_cached_depth_encoder(args, device)

    # Convert image to torch, in [0,1] float range (DepthEncoder handles scaling).
    img = image_rgb_uint8.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Use DepthEncoder's image processor + depth model directly to keep access to pixel size.
    pil = depth_encoder._tensor_to_pil(img_t)
    inputs = depth_encoder.image_processor(images=pil, return_tensors="pt")
    model_dtype = next(depth_encoder.depth_model.parameters()).dtype
    inputs = {k: v.to(device=device, dtype=model_dtype) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = depth_encoder.depth_model(**inputs)

    # Backbone-only model returns feature maps. Support both tuple/list and objects with feature_maps.
    if hasattr(outputs, "feature_maps"):
        features = outputs.feature_maps
    else:
        features = outputs
    if not isinstance(features, (tuple, list)) or len(features) == 0:
        raise RuntimeError("Depth backbone did not return feature maps as expected.")

    feat_idx = args.depth_feature_idx
    print(f"[viz] DepthAnything backbone-only: using feature map index {feat_idx} as proxy depth.")
    feat = features[feat_idx]
    _print_shape("depth_feature", feat)

    # Infer token grid if needed.
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        _, _, proc_h, proc_w = pixel_values.shape
        patch = getattr(depth_encoder.depth_model.config.backbone_config, "patch_size", None)
    else:
        proc_h = proc_w = None
        patch = None

    # Convert backbone feature map to a pseudo depth map.
    # NOTE: The codebase drops the DepthAnything neck/head, so we use a
    # channel-mean proxy depth from backbone features for visualization only.
    if feat.ndim == 4:
        # [B, C, H, W] -> mean across channels
        depth_map = feat.mean(dim=1)[0].float().cpu().numpy()
        print("[viz] Depth feature is 4D; using channel-mean as pseudo depth.")
    elif feat.ndim == 3:
        # [B, N, C] tokens
        feat_tokens = feat[0]
        token_count = feat_tokens.shape[0]
        print(f"[viz] Depth feature is 3D tokens: N={token_count}")

        # Try to infer grid from patch size and processed image.
        grid_hw = None
        if patch is not None and proc_h is not None and proc_w is not None:
            grid_hw = (proc_h // patch, proc_w // patch)
            print(f"[viz] Inferred token grid from patch size: {grid_hw}")

        depth_vec = feat_tokens.mean(dim=-1).float().cpu().numpy()  # [N]

        # Remove CLS token if it matches the grid size + 1.
        if grid_hw is not None:
            h_t, w_t = grid_hw
            if token_count == h_t * w_t + 1:
                depth_vec = depth_vec[1:]
                token_count = depth_vec.shape[0]
                print("[viz] Dropped CLS token from depth tokens.")
            if token_count == h_t * w_t:
                depth_map = depth_vec.reshape(h_t, w_t)
            else:
                print("[viz] Token count does not match inferred grid; using reshape fallback.")
                depth_map = _reshape_tokens_to_grid(depth_vec, grid_hw=None, target_hw=None)
        else:
            depth_map = _reshape_tokens_to_grid(depth_vec, grid_hw=None, target_hw=None)
    else:
        raise RuntimeError(f"Unsupported depth feature shape: {feat.shape}")

    # Resize to original image size for visualization.
    depth_map = _resize_2d(depth_map, (image_rgb_uint8.shape[0], image_rgb_uint8.shape[1]))
    depth_color = _visualize_depth_map(
        depth_map,
        ret_minmax=False,
        ret_type=np.uint8,
        cmap=args.depth_colormap,
    )
    _print_shape("colored_depth", depth_color)
    return depth_color


def reduce_attention_to_1d(
    attn: Any,
    head_idx: Optional[int],
    head_agg: str,
    action_agg: str,
    action_step: Optional[int],
) -> np.ndarray:
    """Reduce [B, H, Q, S] -> [B, S] by selecting / aggregating heads and action steps."""
    attn_t = attn
    if isinstance(attn_t, np.ndarray):
        attn_t = torch.from_numpy(attn_t)

    if attn_t.ndim == 5:
        raise ValueError(
            "Attn has 5 dims. Please select a layer before calling reduce_attention_to_1d."
        )
    if attn_t.ndim != 4:
        raise ValueError(f"Expected attn dims [B,H,Q,S], got {attn_t.shape}")

    _print_shape("attn_input", attn_t)

    # Head selection / aggregation
    if head_idx is not None:
        if head_idx < 0 or head_idx >= attn_t.shape[1]:
            raise ValueError(f"head_idx {head_idx} out of range for H={attn_t.shape[1]}")
        attn_t = attn_t[:, head_idx]
        print(f"[viz] Using head_idx={head_idx}")
    else:
        if head_agg == "mean":
            attn_t = attn_t.mean(dim=1)
        elif head_agg == "sum":
            attn_t = attn_t.sum(dim=1)
        else:
            raise ValueError(f"Unsupported head_agg: {head_agg}")
        print(f"[viz] Aggregated heads with {head_agg}")

    # Action-step selection / aggregation
    if action_step is not None:
        # if action_step < 0 or action_step >= attn_t.shape[1]:
        #     raise ValueError(f"action_step {action_step} out of range for Q={attn_t.shape[1]}")
        attn_t = attn_t[:, action_step]
        print(f"[viz] Using action_step={action_step}")
    else:
        if action_agg == "mean":
            attn_t = attn_t.mean(dim=1)
        else:
            raise ValueError(f"Unsupported action_agg: {action_agg}")
        print(f"[viz] Aggregated actions with {action_agg}")

    _print_shape("attn_reduced", attn_t)
    return attn_t.detach().cpu().numpy()


def _reshape_tokens_to_grid(
    token_vec: np.ndarray,
    grid_hw: Optional[Tuple[int, int]] = None,
    target_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Reshape a 1D token vector to a 2D grid with best-effort sizing."""
    s = token_vec.shape[0]

    if grid_hw is not None and grid_hw[0] * grid_hw[1] == s:
        h_t, w_t = grid_hw
        print(f"[viz] Using provided grid size: {grid_hw}")
        grid = token_vec.reshape(h_t, w_t)
    else:
        # Try to find factor pairs for a clean grid.
        factors = []
        root = int(math.sqrt(s))
        for h in range(1, root + 1):
            if s % h == 0:
                w = s // h
                factors.append((h, w))
        if factors:
            if target_hw is not None:
                tgt_ratio = target_hw[1] / max(target_hw[0], 1)
                best = min(
                    factors,
                    key=lambda hw: abs(math.log((hw[1] / hw[0]) / max(tgt_ratio, 1e-6))),
                )
            else:
                best = min(factors, key=lambda hw: abs(hw[0] - hw[1]))
            h_t, w_t = best
            print(f"[viz] Reshaping tokens to exact grid: {h_t}x{w_t}")
            grid = token_vec.reshape(h_t, w_t)
        else:
            # Fallback: approximate grid by padding to nearest rectangle.
            h_t = int(math.floor(math.sqrt(s)))
            w_t = int(math.ceil(s / max(h_t, 1)))
            pad = h_t * w_t - s
            if pad < 0:
                h_t += 1
                w_t = int(math.ceil(s / h_t))
                pad = h_t * w_t - s
            print(f"[viz] Padding tokens to grid: {h_t}x{w_t} (pad={pad})")
            if pad > 0:
                token_vec = np.pad(token_vec, (0, pad), mode="constant")
            grid = token_vec.reshape(h_t, w_t)

    return grid


def map_tokens_to_grid_and_image(
    token_vec: np.ndarray,
    image_hw: Tuple[int, int],
    mapping: str,
    meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map a 1D depth-token vector to (grid_2d, heatmap_resized)."""
    token_vec = token_vec.astype(np.float32)
    s = token_vec.shape[0]
    print(f"[viz] Mapping {s} depth tokens to image {image_hw} using {mapping}")

    def _token_to_patch_map() -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Attempt to use token coordinates if available.
        coords = None
        if "depth_token_coords" in meta:
            coords = meta["depth_token_coords"]
        elif "token_coords" in meta:
            coords = meta["token_coords"]

        if coords is not None:
            coords = _as_numpy(coords)
            if coords.ndim == 3:
                coords = coords[0]
            if coords.shape[0] != s or coords.shape[1] != 2:
                print("[viz] token_coords shape mismatch; falling back.")
                return None

            # Heuristic: normalized coords in [0,1] if max <= 1.5
            if coords.max() <= 1.5:
                ys = (coords[:, 0] * (image_hw[0] - 1)).round().astype(int)
                xs = (coords[:, 1] * (image_hw[1] - 1)).round().astype(int)
                heatmap = np.zeros(image_hw, dtype=np.float32)
                counts = np.zeros(image_hw, dtype=np.float32)
                for v, y, x in zip(token_vec, ys, xs):
                    y = np.clip(y, 0, image_hw[0] - 1)
                    x = np.clip(x, 0, image_hw[1] - 1)
                    heatmap[y, x] += v
                    counts[y, x] += 1.0
                counts = np.maximum(counts, 1.0)
                heatmap = heatmap / counts
                print("[viz] Mapped tokens using normalized coords directly to image grid.")
                return heatmap, heatmap

            # Otherwise treat coords as grid indices.
            max_y = int(coords[:, 0].max())
            max_x = int(coords[:, 1].max())
            grid_h = max_y + 1
            grid_w = max_x + 1
            grid = np.zeros((grid_h, grid_w), dtype=np.float32)
            counts = np.zeros((grid_h, grid_w), dtype=np.float32)
            for v, y, x in zip(token_vec, coords[:, 0].astype(int), coords[:, 1].astype(int)):
                if 0 <= y < grid_h and 0 <= x < grid_w:
                    grid[y, x] += v
                    counts[y, x] += 1.0
            counts = np.maximum(counts, 1.0)
            grid = grid / counts
            print(f"[viz] Mapped tokens using integer coords to grid {grid_h}x{grid_w}.")
            return grid, _resize_2d(grid, image_hw)

        # Attempt to use flattened indices if available.
        indices = None
        if "depth_token_indices" in meta:
            indices = meta["depth_token_indices"]
        elif "token_indices" in meta:
            indices = meta["token_indices"]

        if indices is not None:
            indices = _as_numpy(indices).reshape(-1)
            if indices.shape[0] != s:
                print("[viz] token_indices shape mismatch; falling back.")
                return None
            grid_hw = None
            for k in ["depth_token_hw", "token_grid_hw", "depth_grid_hw"]:
                if k in meta:
                    grid_hw = tuple(int(x) for x in _as_numpy(meta[k]).reshape(-1)[:2])
                    print(f"[viz] Using grid size from meta: {grid_hw}")
                    break
            if grid_hw is None:
                # Best-effort grid from indices range.
                max_idx = int(indices.max())
                grid_hw = (int(math.floor(math.sqrt(max_idx + 1))), int(math.ceil((max_idx + 1) ** 0.5)))
                print(f"[viz] Inferred grid size from indices: {grid_hw}")
            grid = np.zeros(grid_hw, dtype=np.float32)
            counts = np.zeros(grid_hw, dtype=np.float32)
            for v, idx in zip(token_vec, indices):
                y = int(idx) // grid_hw[1]
                x = int(idx) % grid_hw[1]
                if 0 <= y < grid_hw[0] and 0 <= x < grid_hw[1]:
                    grid[y, x] += v
                    counts[y, x] += 1.0
            counts = np.maximum(counts, 1.0)
            grid = grid / counts
            print(f"[viz] Mapped tokens using indices to grid {grid_hw}.")
            return grid, _resize_2d(grid, image_hw)

        # If only grid size is provided and tokens match, reshape.
        for k in ["depth_token_hw", "token_grid_hw", "depth_grid_hw"]:
            if k in meta:
                grid_hw = tuple(int(x) for x in _as_numpy(meta[k]).reshape(-1)[:2])
                if grid_hw[0] * grid_hw[1] == s:
                    grid = token_vec.reshape(grid_hw)
                    print(f"[viz] Reshaped tokens using grid size meta: {grid_hw}")
                    return grid, _resize_2d(grid, image_hw)

        return None

    if mapping == "token_to_patch_map":
        out = _token_to_patch_map()
        if out is None:
            print("[viz] token_to_patch_map unavailable; falling back to reshape_or_interpolate.")
        else:
            return out

    # Fallback / default: reshape_or_interpolate
    grid = _reshape_tokens_to_grid(token_vec, grid_hw=None, target_hw=image_hw)
    heatmap = _resize_2d(grid, image_hw)
    return grid, heatmap


def map_tokens_to_image(
    token_vec: np.ndarray,
    image_hw: Tuple[int, int],
    mapping: str,
    meta: Dict[str, Any],
) -> np.ndarray:
    """Backward-compatible wrapper returning only the resized heatmap."""
    _, heatmap = map_tokens_to_grid_and_image(token_vec, image_hw, mapping, meta)
    return heatmap


def _normalize_map(map_2d: np.ndarray, mode: str = "per_image") -> np.ndarray:
    """Normalize heatmap to [0,1], either per-image or global (same here)."""
    if mode not in ("per_image", "global"):
        raise ValueError(f"Unsupported norm mode: {mode}")
    m = np.nan_to_num(map_2d.astype(np.float32))
    mn = float(m.min())
    mx = float(m.max())
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def _visualize_depth_map(
    depth: np.ndarray,
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    ret_minmax: bool = False,
    ret_type: Any = np.uint8,
    cmap: str = "Spectral_r",
):
    """Visualize a depth map using min/max normalization and a colormap."""
    depth = depth.copy().astype(np.float32)
    if depth_min is None:
        depth_min = float(depth.min())
    if depth_max is None:
        depth_max = float(depth.max())
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    depth_norm = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    if mpl_cm is None:
        raise RuntimeError("matplotlib is required for Spectral_r depth visualization.")

    cmap_name = cmap
    if isinstance(cmap_name, str) and cmap_name.lower() == "spectral":
        cmap_name = "Spectral"
    elif isinstance(cmap_name, str) and cmap_name.lower() == "spectral_r":
        cmap_name = "Spectral_r"

    cm = mpl_cm.get_cmap(cmap_name)
    img_colored = cm(depth_uint8)[:, :, :3]  # RGB in [0,1]
    if ret_type == np.uint8:
        img_colored = (img_colored * 255.0).astype(np.uint8)
    elif ret_type in (np.float32, np.float64):
        img_colored = img_colored.astype(ret_type)
    else:
        raise ValueError(f"Invalid return type: {ret_type}")

    if ret_minmax:
        return img_colored, depth_min, depth_max
    return img_colored


def _clip_percentile(map_2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip map values to [lo, hi] percentiles."""
    if lo is None or hi is None:
        return map_2d
    if hi <= lo:
        return map_2d
    lo_v = float(np.percentile(map_2d, lo))
    hi_v = float(np.percentile(map_2d, hi))
    if hi_v - lo_v < 1e-8:
        return map_2d
    return np.clip(map_2d, lo_v, hi_v)


def overlay_heatmap(
    base_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
    colormap: str,
    norm: str,
    *,
    gamma: Optional[float] = None,
    clip_percentile: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay heatmap on base image. Returns (heatmap_color, overlay)."""
    base = base_rgb.astype(np.float32)
    if clip_percentile is not None:
        heatmap = _clip_percentile(heatmap, clip_percentile[0], clip_percentile[1])
    heatmap_norm = _normalize_map(heatmap, mode=norm)
    if gamma is not None and gamma > 0:
        if abs(gamma - 1.0) > 1e-6:
            heatmap_norm = np.power(heatmap_norm, gamma)
    hm_color = _apply_colormap(heatmap_norm, colormap).astype(np.float32)
    overlay = (1.0 - alpha) * base + alpha * hm_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    hm_color = np.clip(hm_color, 0, 255).astype(np.uint8)
    return hm_color, overlay


def _draw_grid_overlay(image_rgb: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
    """Draw grid lines on a copy of the image (best-effort)."""
    grid_img = image_rgb.copy()
    grid_h, grid_w = grid_hw
    if grid_h <= 0 or grid_w <= 0:
        return grid_img

    img_h, img_w = grid_img.shape[:2]
    patch_h = max(img_h // grid_h, 1)
    patch_w = max(img_w // grid_w, 1)

    if cv2 is not None:
        for i in range(grid_h + 1):
            y = min(i * patch_h, img_h - 1)
            cv2.line(grid_img, (0, y), (img_w, y), (255, 255, 255), 2)
        for j in range(grid_w + 1):
            x = min(j * patch_w, img_w - 1)
            cv2.line(grid_img, (x, 0), (x, img_h), (255, 255, 255), 2)
        return grid_img

    # PIL fallback
    from PIL import ImageDraw

    pil_img = Image.fromarray(grid_img)
    draw = ImageDraw.Draw(pil_img)
    for i in range(grid_h + 1):
        y = min(i * patch_h, img_h - 1)
        draw.line([(0, y), (img_w, y)], fill=(255, 255, 255), width=2)
    for j in range(grid_w + 1):
        x = min(j * patch_w, img_w - 1)
        draw.line([(x, 0), (x, img_h)], fill=(255, 255, 255), width=2)
    return np.array(pil_img)


def save_montage(
    out_path: str,
    orig_rgb: np.ndarray,
    base_rgb: np.ndarray,
    attn_grid: np.ndarray,
    heatmap_color: np.ndarray,
    overlay: np.ndarray,
    *,
    show: bool = False,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """Save a multi-panel visualization (original + depth + attention panels)."""
    if mpl_cm is None:
        raise RuntimeError("matplotlib is required for montage visualization.")

    import matplotlib.pyplot as plt  # local import to keep startup light

    grid_norm = _normalize_map(attn_grid, mode="per_image")
    grid_h, grid_w = attn_grid.shape[:2]
    img_h, img_w = base_rgb.shape[:2]

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(_draw_grid_overlay(orig_rgb, (grid_h, grid_w)))
    axes[1, 0].set_title("Original Grid")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(base_rgb)
    axes[0, 1].set_title("Colored Depth")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Overlay")
    axes[1, 1].axis("off")

    im1 = axes[0, 2].imshow(grid_norm, cmap="hot", aspect="auto")
    axes[0, 2].set_title(f"Attention Grid ({grid_h}×{grid_w})")
    axes[0, 2].set_xlabel("Patch X")
    axes[0, 2].set_ylabel("Patch Y")
    fig.colorbar(im1, ax=axes[0, 2])

    im3 = axes[1, 2].imshow(
        _resize_2d(grid_norm, (img_h, img_w)),
        cmap="hot",
        extent=[0, img_w, img_h, 0],
    )
    axes[1, 2].set_title("Attention Distribution")
    axes[1, 2].axis("off")
    fig.colorbar(im3, ax=axes[1, 2])

    axes[0, 3].imshow(heatmap_color)
    axes[0, 3].set_title("Colored Heatmap")
    axes[0, 3].axis("off")

    axes[1, 3].imshow(_draw_grid_overlay(base_rgb, (grid_h, grid_w)))
    axes[1, 3].set_title("Depth Grid")
    axes[1, 3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved montage: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_outputs(
    out_dir: str,
    sample_id: str,
    layer_idx: int,
    head_tag: str,
    colored_depth: np.ndarray,
    heatmap_color: np.ndarray,
    overlay: np.ndarray,
    attn_grid: Optional[np.ndarray],
    save_components: bool,
    *,
    mode: Optional[str] = None,
) -> None:
    """Save overlay (and optional components) to disk."""
    os.makedirs(out_dir, exist_ok=True)

    if mode:
        prefix = f"{sample_id}_{mode}_L{layer_idx}_H{head_tag}"
    else:
        prefix = f"{sample_id}_L{layer_idx}_H{head_tag}"

    overlay_path = os.path.join(out_dir, f"{prefix}_overlay.png")
    Image.fromarray(overlay).save(overlay_path)
    print(f"[viz] Saved overlay: {overlay_path}")

    if save_components:
        depth_path = os.path.join(out_dir, f"{prefix}_colored_depth.png")
        heat_path = os.path.join(out_dir, f"{prefix}_heatmap.png")
        Image.fromarray(colored_depth).save(depth_path)
        Image.fromarray(heatmap_color).save(heat_path)
        print(f"[viz] Saved colored depth: {depth_path}")
        print(f"[viz] Saved heatmap: {heat_path}")
        if attn_grid is not None:
            grid_norm = _normalize_map(attn_grid, mode="per_image")
            grid_vis = _apply_colormap(grid_norm, "jet")
            grid_path = os.path.join(out_dir, f"{prefix}_attn_grid.png")
            Image.fromarray(grid_vis).save(grid_path)
            print(f"[viz] Saved attention grid: {grid_path}")


class _OverlayVideoWriter:
    """Lightweight video writer with cv2/imageio fallback."""

    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size  # (W, H)
        self.backend = None
        self.writer = None

        os.makedirs(str(Path(output_path).parent), exist_ok=True)

        if cv2 is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
            self.backend = "cv2"
            return

        try:
            import imageio
        except Exception as exc:
            raise RuntimeError("Video writing requires OpenCV or imageio.") from exc

        self.writer = imageio.get_writer(output_path, fps=fps)
        self.backend = "imageio"

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        h, w = frame_rgb.shape[:2]
        if (w, h) != self.frame_size:
            if cv2 is not None:
                frame_rgb = cv2.resize(frame_rgb, self.frame_size, interpolation=cv2.INTER_AREA)
            else:
                frame_rgb = np.array(Image.fromarray(frame_rgb).resize(self.frame_size, resample=Image.BILINEAR))

        if self.backend == "cv2":
            frame_bgr = frame_rgb[:, :, ::-1]
            self.writer.write(frame_bgr)
        else:
            self.writer.append_data(frame_rgb)

    def close(self) -> None:
        if self.backend == "cv2":
            self.writer.release()
        elif self.backend == "imageio":
            self.writer.close()


def _derive_video_path(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


# -----------------------------
# Model inference (optional)
# -----------------------------

def _load_config(config_arg: str):
    """Load a config by name (preferred) or from a python file if provided."""
    from openpi.training import config as train_config

    if config_arg is None:
        return None

    if Path(config_arg).exists():
        # Best-effort: load python file with CONFIG or get_config.
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_config", config_arg)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import config from path: {config_arg}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "CONFIG"):
            return getattr(module, "CONFIG")
        if hasattr(module, "get_config"):
            return module.get_config()
        raise RuntimeError(
            "Config path provided, but no CONFIG or get_config found. "
            "Please pass a config name known to openpi.training.config.get_config."
        )

    # Treat as a named config.
    return train_config.get_config(config_arg)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load a checkpoint into the PI0Pytorch model."""
    ckpt_path = str(ckpt_path)
    path = Path(ckpt_path)
    if path.is_dir():
        # Prefer safetensors if available.
        for fname in ["model.safetensors", "pytorch_model.bin", "model.pt", "model.pth"]:
            candidate = path / fname
            if candidate.exists():
                ckpt_path = str(candidate)
                break

    print(f"[viz] Loading checkpoint: {ckpt_path}")

    state = None
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required to load .safetensors checkpoints") from exc
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError("Unexpected checkpoint format; expected a state_dict-like dict.")

    # Clean up keys (remove _orig_mod. prefix if present)
    clean_state = {}
    for k, v in state.items():
        new_key = k.replace("_orig_mod.", "")
        clean_state[new_key] = v

    # Handle tied weights: embed_tokens and lm_head
    embed_tokens_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if embed_tokens_key not in clean_state and lm_head_key in clean_state:
        clean_state[embed_tokens_key] = clean_state[lm_head_key]

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[viz] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[viz] Unexpected keys (showing up to 20): {unexpected[:20]}")

    model.to(device)


def _build_observation_from_npz(npz_data: Dict[str, Any], device: torch.device):
    """Best-effort construction of an Observation-like object for PI0Pytorch."""
    # Try a dict stored under "observation" (pickled).
    if "observation" in npz_data:
        obs_obj = npz_data["observation"]
        if isinstance(obs_obj, np.ndarray) and obs_obj.dtype == object:
            obs_obj = obs_obj.item()
        if isinstance(obs_obj, dict):
            obs_dict = obs_obj
        else:
            obs_dict = None
    else:
        obs_dict = None

    # Otherwise build from flat keys: image_*, image_mask_*, state, tokenized_prompt, etc.
    if obs_dict is None:
        image_dict: Dict[str, Any] = {}
        mask_dict: Dict[str, Any] = {}
        for k in npz_data.keys():
            if k.startswith("image_"):
                image_dict[k[len("image_") :]] = npz_data[k]
            if k.startswith("image_mask_"):
                mask_dict[k[len("image_mask_") :]] = npz_data[k]
        if image_dict:
            obs_dict = {
                "image": image_dict,
                "image_mask": mask_dict if mask_dict else None,
                "state": npz_data.get("state"),
                "tokenized_prompt": npz_data.get("tokenized_prompt"),
                "tokenized_prompt_mask": npz_data.get("tokenized_prompt_mask"),
            }

    if obs_dict is None:
        return None

    # Validate required fields.
    if "image" not in obs_dict or obs_dict.get("image") is None:
        return None
    if "state" not in obs_dict or obs_dict.get("state") is None:
        return None
    if "tokenized_prompt" not in obs_dict or obs_dict.get("tokenized_prompt") is None:
        return None
    if "tokenized_prompt_mask" not in obs_dict or obs_dict.get("tokenized_prompt_mask") is None:
        return None

    # Convert arrays to torch tensors.
    images = {}
    for k, v in obs_dict["image"].items():
        arr = _as_numpy(v)
        arr = _ensure_bhwc(arr)
        # model expects [-1, 1] float
        arr = arr.astype(np.float32)
        if arr.max() > 1.0 or arr.min() < -1.0:
            arr = arr / 255.0 * 2.0 - 1.0
        images[k] = torch.from_numpy(arr).to(device)

    image_masks = {}
    if obs_dict.get("image_mask"):
        for k, v in obs_dict["image_mask"].items():
            image_masks[k] = torch.from_numpy(_as_numpy(v).astype(np.bool_)).to(device)

    state = torch.from_numpy(_as_numpy(obs_dict["state"]).astype(np.float32)).to(device)
    tokenized_prompt = torch.from_numpy(_as_numpy(obs_dict["tokenized_prompt"]).astype(np.int64)).to(device)
    tokenized_prompt_mask = torch.from_numpy(_as_numpy(obs_dict["tokenized_prompt_mask"]).astype(np.bool_)).to(device)

    # Create a simple object with attributes expected by preprocess_observation_pytorch.
    class SimpleObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
        skill_id=None,
        skill_soft=None,
    )


def _setup_model_for_attention(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load the PI0Pytorch model + checkpoint for attention extraction."""
    config = _load_config(args.config)
    if config is None:
        raise RuntimeError("Failed to load config.")

    # Use model config if TrainConfig is provided.
    model_cfg = getattr(config, "model", config)

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    model = PI0Pytorch(model_cfg).to(device)

    ckpt_path = args.ckpt or getattr(config, "pytorch_weight_path", None)
    if ckpt_path is None:
        raise RuntimeError("Checkpoint path not provided and config has no pytorch_weight_path.")
    _load_checkpoint(model, ckpt_path, device)

    if args.layer_idx is not None:
        model.attention_viz_layer_idx = args.layer_idx

    return model


def _capture_attention_from_model(
    model: torch.nn.Module, observation: Any, args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    """Run a single inference pass and extract attention."""
    model.eval()
    captured = {"qk_states": None}
    hook_handle = None
    try:
        def _capture_hook(_module, _inputs, outputs):
            # outputs: (outputs_embeds, past_key_values, all_qk_states)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                qk = outputs[2]
                if qk:
                    captured["qk_states"] = qk

        hook_handle = model.paligemma_with_expert.register_forward_hook(_capture_hook)
        with torch.no_grad():
            # Prefer the original (eager) method to avoid torch.compile side effects.
            try:
                sample_actions_fn = model.__class__.sample_actions.__get__(model, model.__class__)
            except Exception:
                sample_actions_fn = model.sample_actions
            output = sample_actions_fn(device, observation, num_steps=args.num_steps, enable_attention_viz=True)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # Heuristic: try to find attention tensors in output or model attributes.
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                print("[viz] Found attention tensor in model output tuple.")
                return item

    for attr in [
        "last_attn_probs_mod",
        "attn_probs_mod",
        "attn_viz_cache",
        "attention_viz_cache",
    ]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, torch.Tensor) and val.ndim == 4:
                print(f"[viz] Found attention tensor in model attribute: {attr}")
                return val
            if isinstance(val, dict) and "attn_probs_mod" in val:
                return val["attn_probs_mod"]

    if getattr(model, "last_qk_states", None) is None and captured["qk_states"] is not None:
        model.last_qk_states = captured["qk_states"]

    qk_states = getattr(model, "last_qk_states", None) or captured["qk_states"]
    if qk_states is not None:
        # Ensure depth_kv exists (compute if needed).
        depth_kv = getattr(model, "last_depth_kv", None)
        if depth_kv is None and getattr(model, "use_depth", False):
            try:
                images, _, _, _, _ = model._preprocess_observation(observation, train=False)
                depth_layer_features = model.depth_module(images[0])
                depth_kv, _, _ = model._apply_depth_ablation(depth_layer_features)
                depth_attend_all = getattr(model.config, "depth_action_attend_all", False)
                for kv in depth_kv:
                    kv["attend_all"] = depth_attend_all
            except Exception as exc:
                raise RuntimeError(f"Failed to compute depth_kv for attention viz: {exc}") from exc
        if depth_kv is not None:
            print("[viz] Computing depth attention from Q/K states.")
            return _compute_depth_attention_from_qk(model, qk_states, args.layer_idx, depth_kv=depth_kv)

    raise RuntimeError(
        "Model inference finished but attention was not found. "
        "Please export attn_probs_mod to NPZ and pass --attn_npz."
    )


def _run_model_for_attention(
    args: argparse.Namespace, input_npz: Optional[Dict[str, Any]], observation: Optional[Any] = None
) -> Any:
    """Attempt to run PI0Pytorch once and extract depth attention."""
    if not args.config:
        raise RuntimeError("Model inference requested but --config not provided.")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model_for_attention(args, device)

    if observation is None:
        if input_npz is None:
            raise RuntimeError("No observation provided and input_npz is missing.")
        observation = _build_observation_from_npz(input_npz, device)
        if observation is None:
            raise RuntimeError(
                "Could not build Observation from input_npz. "
                "Please provide an attn_npz or include full observation fields "
                "(image_*, state, tokenized_prompt, tokenized_prompt_mask)."
            )
    else:
        observation = _move_observation_to_device(observation, device)

    return _capture_attention_from_model(model, observation, args, device)


def _compute_depth_attention_from_qk(
    model: torch.nn.Module, qk_states: Any, layer_idx: int, depth_kv: Optional[Any] = None
) -> torch.Tensor:
    """Compute depth-token attention from cached Q/K states and depth KV."""
    if qk_states is None or len(qk_states) == 0:
        raise RuntimeError("No Q/K states found for depth attention computation.")
    if depth_kv is None:
        if not hasattr(model, "last_depth_kv") or model.last_depth_kv is None:
            raise RuntimeError("Model does not have cached depth_kv. Ensure enable_attention_viz=True and use_depth=True.")
        depth_kv = model.last_depth_kv

    try:
        from depth.attention import prepare_special_attention_config
    except Exception as exc:
        raise RuntimeError("Failed to import depth.attention.prepare_special_attention_config") from exc

    # Find the requested layer entry.
    qk_data = None
    for li, data in qk_states:
        if li == layer_idx:
            qk_data = data
            break
    if qk_data is None:
        raise RuntimeError(f"Requested layer {layer_idx} not found in Q/K states.")

    q_distill = qk_data.get("attention")
    if q_distill is None:
        raise RuntimeError("Q/K states missing 'attention' entry.")
    q = q_distill[0]  # [B, H, S, D]

    depth_conf = prepare_special_attention_config(layer_idx, model.depth_head_layer_idx, depth_kv)
    if depth_conf is None:
        raise RuntimeError(f"Layer {layer_idx} is not configured for depth attention.")

    depth_k = depth_conf["depth_token_k"]  # [B, H, S_depth, D]
    heads_to_modify = depth_conf.get("heads_to_modify", list(range(depth_k.shape[1])))
    attend_all = bool(depth_conf.get("attend_all", False))

    # Map heads_to_modify to a list for indexing.
    if isinstance(heads_to_modify, (tuple, list)):
        head_idx = list(heads_to_modify)
    else:
        head_idx = list(range(depth_k.shape[1]))

    # Slice action queries: last action_horizon positions.
    action_horizon = getattr(model.config, "action_horizon", None)
    if action_horizon is None:
        raise RuntimeError("Model config missing action_horizon.")
    q_len = q.shape[2]
    action_start = q_len - action_horizon
    if action_start < 0:
        raise RuntimeError("Action horizon exceeds sequence length.")

    q_mod = q[:, head_idx, action_start:, :]  # [B, H_mod, Q, D]
    k_mod = depth_k[:, head_idx, :, :]  # [B, H_mod, S_depth, D]

    # Use the same scaling as attention.
    scaling = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.scaling
    if q_mod.dtype != k_mod.dtype:
        attn_dtype = torch.promote_types(q_mod.dtype, k_mod.dtype)
        q_mod = q_mod.to(attn_dtype)
        k_mod = k_mod.to(attn_dtype)
        scaling = torch.tensor(scaling, dtype=attn_dtype, device=q_mod.device).item()
    attn_scores = torch.matmul(q_mod, k_mod.transpose(-2, -1)) * scaling

    if attend_all:
        print("[viz] depth_attend_all=True; computing depth-only attention as a visualization proxy.")

    attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)
    _print_shape("depth_attn_probs", attn_probs)
    return attn_probs


def _get_layer_qk_data(qk_states: Any, layer_idx: int) -> Dict[str, Any]:
    """Fetch Q/K cache dict for the requested layer index."""
    if qk_states is None:
        raise RuntimeError("Q/K states are missing; enable_attention_viz must be True.")
    for li, data in qk_states:
        if li == layer_idx:
            return data
    raise RuntimeError(f"Requested layer {layer_idx} not found in Q/K states.")


def _resolve_object_heads(
    model: torch.nn.Module,
    args: argparse.Namespace,
    num_heads: int,
) -> Tuple[Optional[int], Optional[List[int]], str]:
    """Resolve which heads to use for object attention and return a tag."""
    head_idx = args.object_head_idx if hasattr(args, "object_head_idx") else None
    head_indices: Optional[List[int]] = None

    if head_idx is None:
        obj_heads = getattr(model, "object_head_indices", None)
        if obj_heads is not None and len(obj_heads) > 0:
            head_indices = [int(h) for h in obj_heads if 0 <= int(h) < num_heads]
        elif args.head_idx is not None:
            head_idx = args.head_idx

    if head_idx is not None:
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"object_head_idx {head_idx} out of range for H={num_heads}")
        head_tag = f"obj{head_idx}"
        return head_idx, None, head_tag

    if head_indices:
        if len(head_indices) == 1:
            head_tag = f"obj{head_indices[0]}"
        else:
            head_tag = f"obj{args.head_agg}{len(head_indices)}"
        return None, head_indices, head_tag

    return None, None, f"obj{args.head_agg}"


def _get_view_patch_indices(
    model: torch.nn.Module, view_idx: int, seq_len: int
) -> torch.Tensor:
    """Get patch indices for the specified view, with a fallback."""
    if hasattr(model, "view_patch_indices"):
        view_indices = getattr(model, "view_patch_indices")
        if isinstance(view_indices, torch.Tensor):
            if view_idx < 0 or view_idx >= view_indices.shape[0]:
                raise ValueError(f"object_view_idx {view_idx} out of range for views={view_indices.shape[0]}")
            return view_indices[view_idx].view(-1)
    # Fallback: assume 3 views, each with 256 patches (tokens 0-767).
    # view_idx=0 → 0-255, view_idx=1 → 256-511, view_idx=2 → 512-767
    patches_per_view = 256
    start = view_idx * patches_per_view
    end = min(start + patches_per_view, seq_len)
    if start >= seq_len:
        start = 0
        end = min(patches_per_view, seq_len)
    return torch.arange(start, end, device="cpu", dtype=torch.long)


def _compute_object_attention_tokens(
    model: torch.nn.Module,
    qk_states: Any,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, str]:
    """Compute object attention over image patches for the selected layer."""
    qk_data = _get_layer_qk_data(qk_states, args.layer_idx)

    use_control = bool(getattr(model, "object_use_control", True))
    attn_key = "attention" if use_control else "attention_origin"
    probs_key = "attention_probs" if use_control else "attention_probs_origin"

    qk_pair = qk_data.get(attn_key)
    if qk_pair is None:
        qk_pair = qk_data.get("attention")
    if qk_pair is None:
        raise RuntimeError("Q/K pair for object attention is missing.")

    attn_probs = qk_data.get(probs_key)
    if attn_probs is None:
        attn_probs = qk_data.get("attention_probs")
    if attn_probs is None:
        # Fallback: compute attention probs from Q/K without mask.
        q, k = qk_pair
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise RuntimeError("Q/K tensors are required to compute object attention.")
        scaling = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.scaling
        if q.dtype != k.dtype:
            attn_dtype = torch.promote_types(q.dtype, k.dtype)
            q = q.to(attn_dtype)
            k = k.to(attn_dtype)
            scaling = torch.tensor(scaling, dtype=attn_dtype, device=q.device).item()
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

    if not isinstance(attn_probs, torch.Tensor):
        raise RuntimeError("attention_probs is not a torch.Tensor.")

    num_heads = 2   #attn_probs.shape[1]
    head_idx, head_indices, head_tag = _resolve_object_heads(model, args, num_heads)

    # Select heads if requested.
    if head_idx is not None:
        attn_sel = attn_probs[:, head_idx]  # [B, Q, S]
    elif head_indices is not None:
        if not head_indices:
            raise RuntimeError("No valid object heads available for attention.")
        attn_sel = attn_probs[:, head_indices]  # [B, H_sel, Q, S]
    else:
        attn_sel = attn_probs  # [B, H, Q, S]

    # Action selection / aggregation
    if args.action_step is not None:
        # if args.action_step < 0 or args.action_step >= attn_sel.shape[-2]:
        #     raise ValueError(f"action_step {args.action_step} out of range for Q={attn_sel.shape[-2]}")
        if attn_sel.ndim == 3:
            attn_sel = attn_sel[:, args.action_step]
        else:
            attn_sel = attn_sel[:, :, args.action_step, :]
    else:
        if attn_sel.ndim == 3:
            attn_sel = attn_sel.mean(dim=1)
        else:
            attn_sel = attn_sel.mean(dim=2)

    # Head aggregation if needed.
    if attn_sel.ndim == 3:
        if args.head_agg == "mean":
            attn_sel = attn_sel.mean(dim=1)
        elif args.head_agg == "sum":
            attn_sel = attn_sel.sum(dim=1)
        else:
            raise ValueError(f"Unsupported head_agg: {args.head_agg}")

    if attn_sel.ndim != 2:
        raise RuntimeError(f"Unexpected object attention shape after reduction: {attn_sel.shape}")

    # Select view patches.
    view_idx = getattr(args, "object_view_idx", 0)
    patch_indices = _get_view_patch_indices(model, int(view_idx), attn_sel.shape[-1]).to(attn_sel.device)
    if patch_indices.numel() == 0:
        raise RuntimeError("No patch indices found for object attention.")

    if int(patch_indices.max()) >= attn_sel.shape[-1]:
        max_len = attn_sel.shape[-1]
        patch_indices = patch_indices.clamp(max=max_len - 1)

    token_vec = attn_sel.index_select(-1, patch_indices)
    _print_shape("object_token_vec", token_vec)
    return token_vec.detach().cpu().numpy(), head_tag


def compute_object_attention_tokens(
    model: torch.nn.Module,
    qk_states: Any,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, str]:
    """Public wrapper for object attention token extraction."""
    return _compute_object_attention_tokens(model, qk_states, args)


def compute_object_overlay_for_sample(
    token_vec: np.ndarray,
    image_bhwc_uint8: np.ndarray,
    args: argparse.Namespace,
    batch_idx: int,
    *,
    head_tag: str,
) -> Dict[str, Any]:
    """Compute object attention overlay for a single sample."""
    b = batch_idx
    if b < 0 or b >= image_bhwc_uint8.shape[0]:
        raise ValueError(f"idx {b} out of range for batch {image_bhwc_uint8.shape[0]}")

    if token_vec.ndim == 2:
        token_vec = token_vec[b]

    base_rgb = image_bhwc_uint8[b]
    attn_grid, heatmap = map_tokens_to_grid_and_image(
        token_vec, (base_rgb.shape[0], base_rgb.shape[1]), args.mapping, meta={}
    )
    obj_cmap = getattr(args, "object_colormap", args.colormap)
    heatmap_color, overlay = overlay_heatmap(
        base_rgb,
        heatmap,
        args.alpha,
        obj_cmap,
        args.norm,
        gamma=None,
        clip_percentile=None,
    )

    return {
        "overlay": overlay,
        "heatmap_color": heatmap_color,
        "base_rgb": base_rgb,
        "attn_grid": attn_grid,
        "head_tag": head_tag,
    }


def compute_overlay_for_sample(
    attn: Any,
    image_bhwc_uint8: np.ndarray,
    input_npz: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    batch_idx: int,
) -> Dict[str, Any]:
    """Compute overlay visualization for a single sample."""
    # Convert attention to torch for slicing.
    attn_t = torch.from_numpy(attn) if isinstance(attn, np.ndarray) else attn

    # If attention has layers dimension, select layer_idx.
    if attn_t.ndim == 5:
        # Heuristic: [B, L, H, Q, S] or [L, B, H, Q, S]
        if attn_t.shape[0] == image_bhwc_uint8.shape[0] and attn_t.shape[1] < 128:
            attn_t = attn_t[:, args.layer_idx]
            print(f"[viz] Selected layer {args.layer_idx} from dim=1")
        elif attn_t.shape[1] == image_bhwc_uint8.shape[0] and attn_t.shape[0] < 128:
            attn_t = attn_t[args.layer_idx]
            print(f"[viz] Selected layer {args.layer_idx} from dim=0")
        else:
            attn_t = attn_t[args.layer_idx]
            print(f"[viz] Selected layer {args.layer_idx} from dim=0 (fallback)")

    _print_shape("attn_selected_layer", attn_t)

    # Reduce to [B, S]
    attn_reduced = reduce_attention_to_1d(
        attn_t,
        head_idx=args.head_idx,
        head_agg=args.head_agg,
        action_agg=args.action_agg,
        action_step=args.action_step,
    )

    # Select batch index
    b = batch_idx
    if b < 0 or b >= attn_reduced.shape[0]:
        raise ValueError(f"idx {b} out of range for batch {attn_reduced.shape[0]}")
    token_vec = attn_reduced[b]
    _print_shape("token_vec", token_vec)

    # Prepare colored depth (base image)
    base_rgb = get_colored_depth(image_bhwc_uint8[b], args, input_npz, device)

    # Map tokens to grid + heatmap
    attn_grid, heatmap = map_tokens_to_grid_and_image(
        token_vec, (base_rgb.shape[0], base_rgb.shape[1]), args.mapping, meta
    )
    _print_shape("attn_grid", attn_grid)
    _print_shape("heatmap", heatmap)

    # Overlay
    heatmap_color, overlay = overlay_heatmap(
        base_rgb,
        heatmap,
        args.alpha,
        args.colormap,
        args.norm,
        gamma=getattr(args, "attn_gamma", None),
        clip_percentile=getattr(args, "attn_clip_percentile", None),
    )

    # Naming tag for head
    head_tag = f"{args.head_idx}" if args.head_idx is not None else args.head_agg

    return {
        "overlay": overlay,
        "heatmap_color": heatmap_color,
        "base_rgb": base_rgb,
        "attn_grid": attn_grid,
        "head_tag": head_tag,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize object attention (on RGB) and depth-token attention (on depth)"
    )

    # Required / core args
    parser.add_argument("--ckpt", type=str, default=None, help="PI0Pytorch checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config name or path")
    parser.add_argument("--input_npz", type=str, default=None, help="Input NPZ containing images (and optional obs fields)")
    parser.add_argument("--attn_npz", type=str, default=None, help="Attention NPZ containing attn_probs_mod or attn_probs")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index to visualize")
    parser.add_argument("--head_idx", type=int, default=None, help="Head index to visualize")
    parser.add_argument("--head_agg", type=str, default="mean", choices=["mean", "sum"], help="Head aggregation")
    parser.add_argument("--action_agg", type=str, default="mean", choices=["mean"], help="Action aggregation")
    parser.add_argument("--action_step", type=int, default=None, help="Action step index to visualize")
    parser.add_argument(
        "--mapping",
        type=str,
        default="reshape_or_interpolate",
        choices=["reshape_or_interpolate", "token_to_patch_map"],
        help="Token-to-image mapping strategy",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        choices=["magma", "jet", "viridis", "inferno", "turbo", "plasma"],
        help="Heatmap colormap (depth attention)",
    )
    parser.add_argument(
        "--object_colormap",
        type=str,
        default="jet",
        choices=["magma", "jet", "viridis", "inferno", "turbo", "plasma"],
        help="Heatmap colormap for object attention",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay alpha")
    parser.add_argument(
        "--attn_gamma",
        type=float,
        default=1.0,
        help="Gamma for depth attention heatmap enhancement (smaller = higher contrast).",
    )
    parser.add_argument(
        "--attn_clip_percentile",
        type=str,
        default="",
        help="Percentile clip for depth attention heatmap, e.g. '1,99' (set empty to disable).",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_components", action="store_true", help="Save base depth & heatmap as separate images")
    parser.add_argument("--save_montage", action="store_true", help="Save a multi-panel montage (matplotlib)")
    parser.add_argument("--montage_path", type=str, default=None, help="Override montage output path")
    parser.add_argument("--show", action="store_true", help="Show montage interactively (requires display)")
    parser.add_argument("--montage_figsize", type=str, default="15,10", help="Montage figsize as W,H")

    # Optional extras
    parser.add_argument("--idx", type=int, default=0, help="Sample index (when batch > 1)")
    parser.add_argument("--image_key", type=str, default=None, help="Override image key in input_npz")
    parser.add_argument("--depth_model_name", type=str, default=None, help="Local Depth Anything model path")
    parser.add_argument(
        "--depth_viz_mode",
        type=str,
        default="predicted",
        choices=["predicted", "backbone"],
        help="Depth visualization mode: use full predicted depth or backbone proxy",
    )
    parser.add_argument(
        "--depth_feature_idx",
        type=int,
        default=-1,
        help="Which depth feature map to visualize (index into DepthAnything outputs)",
    )
    parser.add_argument(
        "--depth_colormap",
        type=str,
        default="spectral_r",
        choices=["magma", "jet", "viridis", "spectral", "spectral_r", "inferno", "turbo", "plasma"],
        help="Colormap for colored depth",
    )
    parser.add_argument("--norm", type=str, default="per_image", choices=["per_image", "global"], help="Normalization")
    parser.add_argument("--depth_start", type=int, default=None, help="Depth token start index (for attn_probs)")
    parser.add_argument("--depth_end", type=int, default=None, help="Depth token end index (for attn_probs)")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (cpu/cuda)")
    parser.add_argument("--num_steps", type=int, default=1, help="Num diffusion steps if running model")
    parser.add_argument("--sample_id", type=str, default=None, help="Override sample id for naming")
    parser.add_argument("--libero_repo_id", type=str, default=None, help="Override LIBERO repo_id (LeRobot)")
    parser.add_argument("--libero_root", type=str, default=None, help="Override LIBERO local_root_dir")
    parser.add_argument("--libero_sample_idx", type=int, default=0, help="LIBERO sample index to visualize")
    parser.add_argument(
        "--libero_split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="LIBERO split to sample from",
    )
    parser.add_argument("--libero_num_workers", type=int, default=0, help="Num workers for LIBERO data loader")
    parser.add_argument("--video_out", type=str, default=None, help="Optional depth-attention video output path (mp4)")
    parser.add_argument(
        "--video_out_object",
        type=str,
        default=None,
        help="Optional object-attention video output path (mp4). If omitted, derived from --video_out.",
    )
    parser.add_argument("--video_fps", type=int, default=10, help="Video FPS for overlay output")
    parser.add_argument(
        "--video_num_frames",
        type=int,
        default=200,
        help="Number of frames to render in video (<=0 for all)",
    )
    parser.add_argument(
        "--video_frame_skip",
        type=int,
        default=1,
        help="Use every Nth sample when generating video",
    )
    parser.add_argument(
        "--video_save_frames",
        action="store_true",
        help="Also save per-frame overlays when writing a video",
    )
    parser.add_argument(
        "--object_head_idx",
        type=int,
        default=None,
        help="Override object head index for object attention (default: use model object_head_indices)",
    )
    parser.add_argument(
        "--object_view_idx",
        type=int,
        default=0,
        help="View index for object attention visualization (0=base, 1=left_wrist, 2=right_wrist)",
    )

    args = parser.parse_args()
    # Parse percentile clip for attention map
    if args.attn_clip_percentile:
        try:
            parts = [p.strip() for p in str(args.attn_clip_percentile).split(",")]
            if len(parts) == 2:
                args.attn_clip_percentile = (float(parts[0]), float(parts[1]))
            else:
                args.attn_clip_percentile = None
        except Exception:
            args.attn_clip_percentile = None
    else:
        args.attn_clip_percentile = None

    if args.input_npz is None and args.config is None:
        raise ValueError("Please provide --input_npz or --config (for LIBERO sampling).")

    # If depth_model_name isn't provided, try to reuse it from config.
    if args.depth_model_name is None and args.config is not None:
        try:
            cfg_tmp = _load_config(args.config)
            model_cfg_tmp = getattr(cfg_tmp, "model", cfg_tmp)
            if hasattr(model_cfg_tmp, "depth_model_name"):
                args.depth_model_name = getattr(model_cfg_tmp, "depth_model_name")
                print(f"[viz] Using depth_model_name from config: {args.depth_model_name}")
        except Exception:
            pass

    # Video mode: iterate over LIBERO samples and write overlay videos during inference.
    video_out_depth = args.video_out
    video_out_object = args.video_out_object
    if video_out_depth or video_out_object:
        if args.attn_npz:
            raise ValueError("Video outputs currently support inference mode without --attn_npz.")
        if args.config is None:
            raise ValueError("Video outputs require --config (for LIBERO sampling and inference).")

        if video_out_depth is None and video_out_object is not None:
            video_out_depth = _derive_video_path(video_out_object, "depth")
        if video_out_object is None and video_out_depth is not None:
            video_out_object = _derive_video_path(video_out_depth, "object")

        device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _setup_model_for_attention(args, device)

        start_idx = args.libero_sample_idx
        frame_skip = args.video_frame_skip
        num_frames = args.video_num_frames

        depth_writer = None
        object_writer = None
        processed = 0

        for sample_idx, observation, _actions, image_bhwc_uint8 in iter_libero_samples(
            args,
            start_idx=start_idx,
            num_frames=num_frames,
            frame_skip=frame_skip,
        ):
            obj_image_bhwc_uint8 = image_bhwc_uint8
            if args.object_view_idx != 0:
                view_key_map = {1: "left_wrist_0_rgb", 2: "right_wrist_0_rgb"}
                view_key = view_key_map.get(args.object_view_idx)
                if view_key is None:
                    raise ValueError(f"Unsupported object_view_idx: {args.object_view_idx}")
                view_img = observation.images.get(view_key) if hasattr(observation, "images") else None
                if view_img is None:
                    raise RuntimeError(f"Observation missing view image: {view_key}")
                view_np = _as_numpy(view_img)
                obj_image_bhwc_uint8 = _to_uint8(_ensure_bhwc(view_np))

            # Keep a CPU copy of the base image before moving tensors to device.
            obs_on_device = _move_observation_to_device(observation, device)
            attn = _capture_attention_from_model(model, obs_on_device, args, device)

            qk_states = getattr(model, "last_qk_states", None)
            if qk_states is None:
                raise RuntimeError("Model did not cache Q/K states for object attention.")

            obj_tokens, obj_head_tag = _compute_object_attention_tokens(model, qk_states, args)

            depth_viz = compute_overlay_for_sample(
                attn=attn,
                image_bhwc_uint8=image_bhwc_uint8,
                input_npz=None,
                meta={},
                args=args,
                device=device,
                batch_idx=0,
            )
            obj_viz = compute_object_overlay_for_sample(
                token_vec=obj_tokens,
                image_bhwc_uint8=obj_image_bhwc_uint8,
                args=args,
                batch_idx=0,
                head_tag=obj_head_tag,
            )

            depth_overlay = depth_viz["overlay"]
            obj_overlay = obj_viz["overlay"]

            if depth_writer is None:
                h, w = depth_overlay.shape[:2]
                depth_writer = _OverlayVideoWriter(video_out_depth, args.video_fps, (w, h))
                print(f"[viz] Writing depth video to: {video_out_depth} ({w}x{h} @ {args.video_fps} fps)")
            if object_writer is None:
                h, w = obj_overlay.shape[:2]
                object_writer = _OverlayVideoWriter(video_out_object, args.video_fps, (w, h))
                print(f"[viz] Writing object video to: {video_out_object} ({w}x{h} @ {args.video_fps} fps)")

            depth_writer.write(depth_overlay)
            object_writer.write(obj_overlay)
            processed += 1

            if args.video_save_frames:
                sample_id = f"libero_{sample_idx}"
                save_outputs(
                    out_dir=args.out_dir,
                    sample_id=sample_id,
                    layer_idx=args.layer_idx,
                    head_tag=depth_viz["head_tag"],
                    colored_depth=depth_viz["base_rgb"],
                    heatmap_color=depth_viz["heatmap_color"],
                    overlay=depth_overlay,
                    attn_grid=depth_viz["attn_grid"],
                    save_components=args.save_components,
                )
                save_outputs(
                    out_dir=args.out_dir,
                    sample_id=sample_id,
                    layer_idx=args.layer_idx,
                    head_tag=obj_viz["head_tag"],
                    colored_depth=obj_viz["base_rgb"],
                    heatmap_color=obj_viz["heatmap_color"],
                    overlay=obj_overlay,
                    attn_grid=obj_viz["attn_grid"],
                    save_components=args.save_components,
                    mode="object",
                )

            if processed % 50 == 0:
                print(f"[viz] Processed {processed} frames for video...")

        if depth_writer is None or object_writer is None:
            raise RuntimeError("No frames were processed; video writers were never created.")

        depth_writer.close()
        object_writer.close()
        print(f"[viz] Depth video saved: {video_out_depth} (frames={processed})")
        print(f"[viz] Object video saved: {video_out_object} (frames={processed})")
        return

    # Load inputs (NPZ or LIBERO)
    observation = None
    if args.input_npz:
        inputs = load_inputs(args)
        input_npz = inputs["npz"]
        image_bhwc_uint8 = inputs["image_bhwc_uint8"]
        sample_id = inputs["sample_id"]
    else:
        inputs = load_libero_sample(args)
        input_npz = inputs["npz"]
        observation = inputs["observation"]
        image_bhwc_uint8 = inputs["image_bhwc_uint8"]
        sample_id = inputs["sample_id"]

    # Choose device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load attention
    attn_pack = load_attention(args) if args.attn_npz else {"attn": None, "meta": {}}
    attn = attn_pack["attn"]
    meta = attn_pack["meta"]

    # If no attn provided, try running model (best-effort).
    if attn is None:
        print("[viz] No --attn_npz provided. Attempting to run model for attention...")
        attn = _run_model_for_attention(args, input_npz, observation=observation)

    viz = compute_overlay_for_sample(
        attn=attn,
        image_bhwc_uint8=image_bhwc_uint8,
        input_npz=input_npz,
        meta=meta,
        args=args,
        device=device,
        batch_idx=args.idx,
    )
    head_tag = viz["head_tag"]

    save_outputs(
        out_dir=args.out_dir,
        sample_id=sample_id,
        layer_idx=args.layer_idx,
        head_tag=head_tag,
        colored_depth=viz["base_rgb"],
        heatmap_color=viz["heatmap_color"],
        overlay=viz["overlay"],
        attn_grid=viz["attn_grid"],
        save_components=args.save_components,
    )

    if args.save_montage:
        if args.montage_path:
            montage_path = args.montage_path
        else:
            montage_path = os.path.join(args.out_dir, f"{sample_id}_L{args.layer_idx}_H{head_tag}_montage.png")
        try:
            fig_parts = [p.strip() for p in args.montage_figsize.split(",")]
            figsize = (int(fig_parts[0]), int(fig_parts[1])) if len(fig_parts) == 2 else (15, 10)
        except Exception:
            figsize = (15, 10)
        save_montage(
            montage_path,
            orig_rgb=image_bhwc_uint8[args.idx],
            base_rgb=viz["base_rgb"],
            attn_grid=viz["attn_grid"],
            heatmap_color=viz["heatmap_color"],
            overlay=viz["overlay"],
            show=args.show,
            figsize=figsize,
        )


if __name__ == "__main__":
    main()
