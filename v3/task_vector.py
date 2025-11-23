# task_vector.py

import os
import torch
import shutil
import argparse
from typing import Dict
from safetensors.torch import load_file, save_file
from transformers import CLIPModel

from utils.constant import CLIP_MODEL_NAME

def _get_file_path(path: str) -> str:
    """
    Helper to find the weight file (safetensors or bin) in a directory.
    Priority: model.safetensors > pytorch_model.bin
    """
    if os.path.isfile(path):
        return path
    
    # Check for common HF filenames
    safetensors_path = os.path.join(path, "model.safetensors")
    bin_path = os.path.join(path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        return safetensors_path
    elif os.path.exists(bin_path):
        return bin_path
    else:
        raise FileNotFoundError(f"No model weights found at {path}")

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from path, handling both safetensors and torch.load.
    Returns tensors on CPU.
    """
    file_path = _get_file_path(path)
    print(f"Loading weights from: {file_path}")
    
    if file_path.endswith(".safetensors"):
        state_dict = load_file(file_path, device="cpu")
    else:
        state_dict = torch.load(file_path, map_location="cpu")
        
    return state_dict

def copy_config_files(src_dir: str, dst_dir: str):
    """
    Copy all non-weight configuration files (json, txt) from src to dst.
    """
    if not os.path.isdir(src_dir):
        src_dir = os.path.dirname(src_dir)
        
    os.makedirs(dst_dir, exist_ok=True)
    
    # Files to ignore (weights)
    ignore_files = ["model.safetensors", "pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"]
    
    for filename in os.listdir(src_dir):
        if filename in ignore_files:
            continue
        if filename.endswith(".json") or filename.endswith(".txt") or filename.endswith(".model"):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copyfile(src_file, dst_file)

def task_vector(model_A_path: str, s: float, model_B_path: str, save_path: str):
    """
    Implements: Target = Model_A + s * (Model_B - Pretrained_Model)
    
    Args:
        model_A_path: Path to the first fine-tuned model directory.
        s: Scaling factor (float).
        model_B_path: Path to the second fine-tuned model directory.
        save_path: Directory to save the resulting model.
    """
    print(f"--- Starting Task Vector Arithmetic ---")
    print(f"Formula: Result = A + {s:.2f} * (B - Pretrained)")
    
    # 1. Load State Dicts
    print("Loading Model A...")
    state_a = _load_state_dict(model_A_path)
    
    print("Loading Model B...")
    state_b = _load_state_dict(model_B_path)
    
    print(f"Loading Pretrained Model ({CLIP_MODEL_NAME})...")
    # Load pretrained model from HF Hub or Cache
    base_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    state_pre = base_model.state_dict()
    del base_model # Free memory
    
    # 2. Check Key Consistency
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    keys_pre = set(state_pre.keys())
    
    common_keys = keys_a.intersection(keys_b).intersection(keys_pre)
    
    if len(common_keys) != len(keys_a):
        print(f"Warning: Key mismatch detected. Operating only on {len(common_keys)} common layers.")

    # 3. Perform Arithmetic
    new_state_dict = {}
    print("Computing task vector...")
    
    for key in common_keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        tensor_pre = state_pre[key]
        
        # Skip non-floating point tensors (e.g., position_ids, int64 buffers)
        # We assume structure is identical, so we just copy from A
        if not tensor_a.is_floating_point():
            new_state_dict[key] = tensor_a
            continue
            
        # Formula: A + s * (B - Pre)
        # Calculate task vector: (B - Pre)
        task_vec = tensor_b - tensor_pre
        
        # Apply to A
        new_tensor = tensor_a + (s * task_vec)
        
        new_state_dict[key] = new_tensor

    # 4. Save Result
    os.makedirs(save_path, exist_ok=True)
    
    # Determine save format (default to safetensors for efficiency)
    save_file_path = os.path.join(save_path, "model.safetensors")
    save_file(new_state_dict, save_file_path)
    print(f"Saved merged weights to: {save_file_path}")
    
    # 5. Copy Configs (Tokenizer, Model Config, etc.) from Model A
    # This ensures the folder is loadable by CLIPModel.from_pretrained
    print("Copying configuration files from Model A...")
    copy_config_files(model_A_path, save_path)
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Vector Arithmetic: A + s * (B - Pretrain)")
    parser.add_argument("--model_a", type=str, required=True, help="Path to fine-tuned model A")
    parser.add_argument("--s", type=float, required=True, help="Scaling factor (alpha)")
    parser.add_argument("--model_b", type=str, required=True, help="Path to fine-tuned model B")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the result")

    args = parser.parse_args()

    task_vector(
        model_A_path=args.model_a,
        s=args.s,
        model_B_path=args.model_b,
        save_path=args.save_path
    )