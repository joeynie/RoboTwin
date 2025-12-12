"""
Process current_target_mask and gripper_mask in HDF5 files to 16x16 soft attention maps.
All processed data is saved to a single output HDF5 file.

Usage:
    python script/process_target_mask.py --data_path ./data/task_name/config_name --output attention_maps.h5
"""

import numpy as np
import h5py
import argparse
import re
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def _resize_with_pad_pil(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """Resize mask to target size with padding to preserve aspect ratio."""
    cur_height, cur_width = img.shape[:2]
    if cur_width == width and cur_height == height:
        return img

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    pil_img = Image.fromarray(img.astype(np.uint8))
    resized_image = pil_img.resize((resized_width, resized_height), resample=Image.NEAREST)

    zero_image = np.zeros((height, width), dtype=img.dtype)
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)

    zero_image[pad_height:pad_height + resized_height, pad_width:pad_width + resized_width] = np.array(resized_image)
    return zero_image


def make_soft_attention_map(mask: np.ndarray, patch_size: int = 14, sigma: float = 10.0) -> np.ndarray:
    """
    Generate 16x16 soft attention map from binary mask.
    
    1. Distance transform from mask boundary
    2. Gaussian decay: exp(-dist^2 / (2 * sigma^2))
    3. Average pool to 16x16
    """
    if mask.sum() == 0:
        return np.zeros((16, 16), dtype=np.float32)
    
    # Distance transform (distance to nearest foreground pixel)
    dist_map = distance_transform_edt(1 - mask)
    
    # Gaussian decay
    smooth_map = np.exp(-(dist_map ** 2) / (2 * sigma ** 2))
    
    # Pool to 16x16 (224 / 14 = 16)
    h, w = mask.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    smooth_map_cropped = smooth_map[:num_patches_h * patch_size, :num_patches_w * patch_size]
    reshaped = smooth_map_cropped.reshape(num_patches_h, patch_size, num_patches_w, patch_size)
    attention_map = reshaped.mean(axis=(1, 3)).astype(np.float32)
    
    return attention_map


def process_mask(mask: np.ndarray, patch_size: int = 14, sigma: float = 10.0) -> np.ndarray:
    """Process raw mask to 16x16 soft attention map."""
    # Resize to 224x224 with padding
    resized_mask = _resize_with_pad_pil(mask, height=224, width=224)
    # Generate soft attention map
    return make_soft_attention_map(resized_mask, patch_size=patch_size, sigma=sigma)


def extract_episode_idx(filename: str) -> int:
    """Extract episode index from filename like 'episode0.hdf5' or 'episode_10.hdf5'."""
    match = re.search(r'episode_?(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def process_and_save_to_single_file(
    data_path: str,
    output_path: str,
    patch_size: int = 14,
    sigma: float = 10.0
) -> None:
    """
    Process all HDF5 files in data_path and save to a single output HDF5 file.
    
    Output structure:
        episode_{ep_idx}/
            frame_{frame_idx}_target_attn_{camera_name}: (16, 16) float32
            frame_{frame_idx}_gripper_attn_{camera_name}: (16, 16) float32
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    hdf5_files = sorted(data_path.glob("*.hdf5"), key=lambda x: extract_episode_idx(x.name))
    if not hdf5_files:
        print(f"No HDF5 files found in {data_path}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    print(f"Output file: {output_path}")
    
    with h5py.File(output_path, 'w') as f_out:
        for hdf5_file in tqdm(hdf5_files, desc="Processing episodes"):
            ep_idx = extract_episode_idx(hdf5_file.name)
            if ep_idx < 0:
                print(f"Warning: Could not extract episode index from {hdf5_file.name}, skipping")
                continue
            
            group_name = f"episode_{ep_idx}"
            group = f_out.require_group(group_name)
            
            try:
                with h5py.File(str(hdf5_file), 'r') as f_in:
                    # Process current_target_mask
                    if 'current_target_mask' in f_in:
                        _process_and_save_masks(
                            f_in, group, 'current_target_mask', 'target_attn',
                            patch_size, sigma
                        )
                    
                    # Process gripper_mask
                    if 'gripper_mask' in f_in:
                        _process_and_save_masks(
                            f_in, group, 'gripper_mask', 'gripper_attn',
                            patch_size, sigma
                        )
                        
            except Exception as e:
                print(f"Error processing {hdf5_file.name}: {e}")
    
    print(f"Done! Saved to {output_path}")


def _process_and_save_masks(
    f_in: h5py.File,
    group: h5py.Group,
    mask_key: str,
    attn_prefix: str,
    patch_size: int,
    sigma: float
) -> None:
    """
    Process masks from input file and save attention maps to output group.
    
    Args:
        f_in: Input HDF5 file
        group: Output HDF5 group (episode group)
        mask_key: Key for masks in input file (e.g., 'current_target_mask')
        attn_prefix: Prefix for attention map dataset names (e.g., 'target_attn')
        patch_size: ViT patch size
        sigma: Gaussian sigma
    """
    masks_data = f_in[mask_key]
    
    # Check if it's a dict-like structure (camera_name: mask_array) or direct array
    if isinstance(masks_data, h5py.Group):
        # Dict structure: {camera_name: (num_frames, H, W)}
        for camera_name in masks_data.keys():
            masks = masks_data[camera_name][:]
            num_frames = masks.shape[0]
            
            for frame_idx in range(num_frames):
                attn_map = process_mask(masks[frame_idx], patch_size=patch_size, sigma=sigma)
                
                dset_name = f"frame_{frame_idx}_{attn_prefix}_{camera_name}"
                if dset_name in group:
                    del group[dset_name]
                group.create_dataset(dset_name, data=attn_map, compression="gzip")
    else:
        # Direct array: (num_frames, H, W) or (num_frames, num_cams, H, W)
        masks = masks_data[:]
        
        if len(masks.shape) == 3:
            # Single camera: (num_frames, H, W)
            num_frames = masks.shape[0]
            for frame_idx in range(num_frames):
                attn_map = process_mask(masks[frame_idx], patch_size=patch_size, sigma=sigma)
                
                dset_name = f"frame_{frame_idx}_{attn_prefix}"
                if dset_name in group:
                    del group[dset_name]
                group.create_dataset(dset_name, data=attn_map, compression="gzip")
        else:
            # Multiple cameras: (num_frames, num_cams, H, W)
            num_frames, num_cams = masks.shape[:2]
            for frame_idx in range(num_frames):
                for cam_idx in range(num_cams):
                    attn_map = process_mask(masks[frame_idx, cam_idx], patch_size=patch_size, sigma=sigma)
                    
                    dset_name = f"frame_{frame_idx}_{attn_prefix}_cam{cam_idx}"
                    if dset_name in group:
                        del group[dset_name]
                    group.create_dataset(dset_name, data=attn_map, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description='Process masks to 16x16 soft attention maps and save to single file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data folder containing HDF5 files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HDF5 file path (default: {data_path}/attention_maps.h5)')
    parser.add_argument('--patch_size', type=int, default=14,
                        help='ViT patch size (default: 14)')
    parser.add_argument('--sigma', type=float, default=10.0,
                        help='Gaussian sigma for soft attention (default: 10.0)')
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        args.output = str(Path(args.data_path) / "attention_maps.h5")
    
    process_and_save_to_single_file(
        args.data_path,
        args.output,
        patch_size=args.patch_size,
        sigma=args.sigma
    )


if __name__ == "__main__":
    main()
