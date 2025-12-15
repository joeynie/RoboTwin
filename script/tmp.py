import h5py
import numpy as np
from PIL import Image
import imageio.v3 as iio
import os
import cv2
import re

ws_path="/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/niexian/RoboTwin"
# input_path=f"data/adjust_bottle/demo_randomized_mask_10/data/episode0.hdf5"
input_path=f"/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/niexian/RoboTwin/data/place_burger_fries/demo_randomized_mask/data/episode0.hdf5"
# input_path=f"policy/pi05/processed_data/dump_bin_bigbin-demo_randomized_mask_3-3/episode_0/episode_0.hdf5"
os.makedirs("tmp", exist_ok=True)

def save_video(obj, output="tmp/video.mp4"):
    data = obj[:]
    if data.max() <= 1.0:
        data = (data * 255).astype(np.uint8)
    frames = np.stack([data] * 3, axis=-1)
    iio.imwrite(output,frames,fps=30)

# def save_video(obj, output="tmp/video.mp4"):
#     """obj: HDF5 dataset with padded JPEG bytes, shape=(N,), dtype='|Sxxx'"""

#     jpeg_padded_list = obj[:]                  # shape=(N,)
#     frames = []

#     for padded in jpeg_padded_list:
#         # 去掉 padding 的 b"\0"
#         jpeg_bytes = padded.rstrip(b"\0")

#         # 解码
#         img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
#         frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         if frame is None:
#             raise ValueError("解码 JPEG 失败，请检查存储格式")

#         frames.append(frame)

#     # 保存视频
#     iio.imwrite(output, frames, fps=30)
#     print(f"Video saved to {output}")

CAMERA_RENAME = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
def rename_camera_dataset(group):
    """
    递归遍历 group，查找 dataset 并做 rename。
    """
    to_rename = []

    # 先收集（不能边遍历边修改）
    for key in group.keys():
        obj = group[key]

        if isinstance(obj, h5py.Group):
            rename_camera_dataset(obj)

        elif isinstance(obj, h5py.Dataset):
            for old_cam, new_cam in CAMERA_RENAME.items():
                if key.endswith(old_cam):
                    new_name = key[: -len(old_cam)] + new_cam
                    to_rename.append((key, new_name))

    # 执行 rename
    for old, new in to_rename:
        # print(f"Renaming: {old}  →  {new}")
        group.move(old, new)

def print_tree(name, obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, h5py.Group):
        num_items = len(obj.keys())
        print(f"{prefix}📁 {name}/  ({num_items} items)")
        for key in obj:
            print_tree(key, obj[key], indent + 1)
    else:  # Dataset
        print(f"{prefix}📄 {name}  shape={obj.shape}, dtype={obj.dtype}")
        # if name.endswith("camera"):
        #     save_video(obj, f"tmp/{name}.mp4")

def show_h5_tree(path):
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print_tree("/", f, 0)

def collect_frames_and_save_video(episode_group, camera_suffix, output="tmp/output.mp4"):
    """
    从 episode_group 中收集所有名字形如:
    frame_9_xxx_<camera_suffix>
    的数据集，排序后写入视频
    """
    frame_regex = re.compile(r"frame_(\d+)_target_.*_" + re.escape(camera_suffix))

    frame_items = []

    for name, ds in episode_group.items():
        m = frame_regex.match(name)
        if m:
            frame_idx = int(m.group(1))
            frame_items.append((frame_idx, ds))

    if not frame_items:
        print(f"❌ No frames found for camera '{camera_suffix}'")
        return

    # 按帧号排序
    frame_items.sort(key=lambda x: x[0])

    frames = []
    for idx, ds in frame_items:
        arr = ds[...]  # (H, W) float32
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)

        arr = cv2.resize(arr, None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
        # 灰度 → RGB（重复三通道）
        frame = np.stack([arr] * 3, axis=-1)
        frames.append(frame)

    # 保存视频
    iio.imwrite(output, frames, fps=10)
    print(f"✅ Saved {len(frames)} frames to: {output}")

if __name__ == "__main__":
    # with h5py.File(input_path, "r") as f:
    #     episode = f["episode_0"]
    #     collect_frames_and_save_video(
    #         episode,
    #         camera_suffix="cam_high",
    #         output="tmp/episode_0_cam_high.mp4"
    #     )

    # with h5py.File(input_path, "r+") as f:
    #     rename_camera_dataset(f)
    show_h5_tree(input_path)
    
