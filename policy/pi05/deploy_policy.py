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
        target_layer_name: Layer name for attention (default: "layers.10.self_attn")
                          Examples: "layers.10.self_attn", "gemma_expert.model.layers.10"
        attn_save_dir: Output directory for attention maps (default: "eval_result")
    """
    train_config_name = usr_args["train_config_name"]
    model_name = usr_args["model_name"]
    checkpoint_id = usr_args["checkpoint_id"]
    pi0_step = usr_args["pi0_step"]
    
    # Attention visualization parameters (optional)
    extract_attention = usr_args.get("extract_attention", False)
    target_layer_name = usr_args.get("target_layer_name", "layers.10.self_attn")
    attn_save_dir = usr_args.get("attn_save_dir", "eval_result")
    
    return PI0(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        extract_attention=extract_attention,
        target_layer_name=target_layer_name,
        attn_save_dir=attn_save_dir
    )


def eval(TASK_ENV, model, observation, test_point_id=None):
    """Run evaluation loop with optional attention map recording.
    
    Args:
        TASK_ENV: Environment interface
        model: PI0 model instance
        observation: Initial observation
        test_point_id: Deprecated, ignored. Use eval_policy.py to control start/end.
    """
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]
    
    # Get the attention maps from the latest inference
    current_attention_maps = getattr(model, '_current_attention_maps', {})
    
    # Record initial frame with per-view attention
    if hasattr(model, 'attn_recorder') and model.attn_recorder is not None and hasattr(model, 'frame_images') and model.frame_images:
        model.attn_recorder.process_frame(model.frame_images, current_attention_maps)

    # Execute actions and record frames with the same attention maps
    # (since we don't re-infer, we reuse the attention from this batch of actions)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        
        # Record frame with the same attention maps
        if hasattr(model, 'attn_recorder') and model.attn_recorder is not None and hasattr(model, 'frame_images') and model.frame_images:
            model.attn_recorder.process_frame(model.frame_images, current_attention_maps)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
