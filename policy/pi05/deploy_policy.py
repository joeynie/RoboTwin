import numpy as np
import torch
import dill
import os, sys
from pathlib import Path

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

try:
    from pi_model_enhanced import PI0
except ImportError:
    from pi_model import PI0


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    """Initialize PI0 model with optional attention visualization.
    
    usr_args keys:
        train_config_name: Model config name
        model_name: Model checkpoint name  
        checkpoint_id: Checkpoint identifier
        pi0_step: Number of action steps
        extract_attention: Enable attention extraction (default: False)
        layer_idx: Layer index for attention visualization (0-indexed, None means last layer)
        attn_save_dir: Output directory for attention maps (default: "eval_result")
    """
    train_config_name = usr_args["train_config_name"]
    model_name = usr_args["model_name"]
    checkpoint_id = usr_args["checkpoint_id"]
    pi0_step = usr_args["pi0_step"]
    
    # Attention visualization parameters (optional)
    extract_attention = usr_args.get("extract_attention", False)
    layer_idx = usr_args.get("layer_idx", None)
    attn_save_dir = usr_args.get("attn_save_dir", "eval_result")
    
    return PI0(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        extract_attention=extract_attention,
        layer_idx=layer_idx,
        attn_save_dir=attn_save_dir
    )


def eval(TASK_ENV, model, observation):
    """Run evaluation loop with optional attention map recording.
    
    Args:
        TASK_ENV: Environment interface
        model: PI0 model instance
        observation: Initial observation
    """
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]
    # Execute actions
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
