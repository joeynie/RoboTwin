"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
# from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        # 为每一帧增加一个离散的 skill_id 标签，用于后续在 pi0 中做 skill-level 辅助训练
        # 注意：根据当前 lerobot 中 `get_hf_features_from_features` 的约定，
        # 标量特征需要使用 shape=(1,) 才能被映射为 datasets.Value；shape=() 会被判定为非法。
        "observation.skill_id": {
            "dtype": "int64",
            "shape": (1,),
        },
        # 额外保存原始的 skill 文本，方便后续可视化 / 调试。
        # 文本特征在 lerobot 中使用 dtype="string"，shape=(1,)。
        "observation.skill_text": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    # if Path(HF_LEROBOT_HOME / repo_id).exists():
    #     shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        np.ndarray,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )
        
        # ======== 读取 subtask_text 作为 skill 文本（长度与 state 对齐） ========
        num_frames = state.shape[0]
        if "subtask_text" in ep:
            raw_subtasks = ep["subtask_text"][()]  # 可能是 bytes 数组或 unicode
            # HDF5 中通常存为 bytes，需要显式 decode
            if isinstance(raw_subtasks, np.ndarray) and np.issubdtype(raw_subtasks.dtype, np.bytes_):
                subtasks_full = np.array([s.decode("utf-8").rstrip("\0") for s in raw_subtasks])
            else:
                subtasks_full = raw_subtasks.astype(str)
            # 对齐到 state 的时间长度
            if subtasks_full.shape[0] >= num_frames:
                subtasks = subtasks_full[:num_frames]
            else:
                # 极少数异常情况：subtask_text 比 state 短，做安全 pad
                pad_len = num_frames - subtasks_full.shape[0]
                subtasks = np.concatenate(
                    [subtasks_full, np.array([""] * pad_len, dtype=subtasks_full.dtype)]
                )
            print(f"[DEBUG convert_lerobot] Loaded subtask_text from {os.path.basename(ep_path)}: "
                  f"raw_length={len(subtasks_full)}, num_frames={num_frames}, "
                  f"final_length={len(subtasks)}, unique_count={len(set(subtasks))}, "
                  f"sample_texts={list(subtasks[:min(3, len(subtasks))])}")
        else:
            # 如果没有 subtask_text，就用空串占位，后续仍然会给一个 skill_id（通常是 0）
            subtasks = np.array([""] * num_frames)
            print(f"[DEBUG convert_lerobot] No subtask_text found in {os.path.basename(ep_path)}, using empty strings")

    return imgs_per_cam, state, action, velocity, effort, subtasks


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    # 在整个数据集范围内维护一个全局的 skill vocab：skill_text -> skill_id
    # 这样同一条 skill 文本在不同 episode / frame 中会得到一致的整数 id。
    skill_vocab: dict[str, int] = {}

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, subtasks = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        
        # 用于本 episode 的简单统计，方便后续在日志中做 sanity check
        episode_skill_texts: set[str] = set()
        
        # add prompt
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
        
        for i in range(num_frames):
            # 当前帧的 skill 文本（如果原始数据里没有，则为空串）
            skill_text = str(subtasks[i]) if subtasks is not None else ""
            episode_skill_texts.add(skill_text)
            
            # 为每一种 skill 文本分配一个稳定的整数 id
            if skill_text in skill_vocab:
                skill_id = skill_vocab[skill_text]
            else:
                skill_id = len(skill_vocab)
                skill_vocab[skill_text] = skill_id
            
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,  # episode-level 指令
                # hf_features 中将 shape=(1,) 的离散特征映射为 `datasets.Value`，
                # 因此这里按照长度为 1 的数组 / 序列进行存储，避免 shape=() 带来的不兼容问题。
                "observation.skill_id": np.array([skill_id], dtype=np.int64),  # 离散 skill id，供模型做分类等
                "observation.skill_text": skill_text,  # 原始文本，方便后处理 / 可视化
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            # Debug: 打印前3帧的 skill 信息，验证是否正确添加到 frame 中
            if i < 3:
                print(f"[DEBUG convert_lerobot] Frame {i} in episode {ep_idx}: "
                      f"skill_id={skill_id}, skill_text={skill_text!r}, "
                      f"frame_keys={list(frame.keys())}, "
                      f"has_skill_id={'observation.skill_id' in frame}, "
                      f"has_skill_text={'observation.skill_text' in frame}")
            
            dataset.add_frame(frame)
        
        dataset.save_episode()
        
        # ======== 每个 episode 转换完成后的验证信息打印 ========
        unique_skills = sorted(episode_skill_texts)
        max_print_skills = 10
        preview_skills = unique_skills[:max_print_skills]
        print(
            f"[MM-ACT -> LeRobot] Converted episode {ep_idx} ({os.path.basename(ep_path)}): "
            f"frames={num_frames}, "
            f"unique_skills={len(unique_skills)}, "
            f"example_skills={preview_skills}, "
            f"instruction={instruction[:80]!r}"
            + ("..." if len(instruction) > 80 else "")
        )
        print(f"[DEBUG convert_lerobot] Episode {ep_idx} saved. Global skill_vocab size: {len(skill_vocab)}, "
              f"skill_vocab sample: {dict(list(skill_vocab.items())[:min(5, len(skill_vocab))])}")

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # if (HF_LEROBOT_HOME / repo_id).exists():
    #     shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)
    
    # Sort by episode index to ensure consistent ordering with attention_map h5
    import re
    def extract_episode_idx(path):
        match = re.search(r'episode_?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1
    hdf5_files = sorted(hdf5_files, key=extract_episode_idx)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
