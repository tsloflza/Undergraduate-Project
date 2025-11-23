import os
import torch
import shutil
import argparse
from typing import Dict
from safetensors.torch import load_file, save_file

def _get_file_path(path: str) -> str:
    """
    Helper to find the weight file (safetensors or bin) in a directory.
    """
    if os.path.isfile(path):
        return path
    
    # Check for common PEFT filenames
    safetensors_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    
    if os.path.exists(safetensors_path):
        return safetensors_path
    elif os.path.exists(bin_path):
        return bin_path
    else:
        raise FileNotFoundError(f"No adapter weights found at {path}")

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from path, handling both safetensors and torch.load.
    Returns tensors on CPU.
    """
    file_path = _get_file_path(path)
    # print(f"Loading weights from: {file_path}")
    
    if file_path.endswith(".safetensors"):
        state_dict = load_file(file_path, device="cpu")
    else:
        state_dict = torch.load(file_path, map_location="cpu")
        
    return state_dict

def linear_arith(adapter_a_path: str, scalar: float, adapter_b_path: str, save_path: str):
    """
    Calculates: Result = Adapter_A + (scalar * Adapter_B)
    Saves the result and copies the configuration JSON.
    """
    # 1. Load weights
    state_a = _load_state_dict(adapter_a_path)
    state_b = _load_state_dict(adapter_b_path)

    # 2. Perform Arithmetic: A + scalar * B
    new_state_dict = {}
    
    # Intersection of keys to be safe, though typically they should match
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    common_keys = keys_a.intersection(keys_b)

    if len(common_keys) != len(keys_a):
        print(f"Warning: Mismatch in keys. Adapter A has {len(keys_a)}, Adapter B has {len(keys_b)}.")
        print("Proceeding with common keys only.")

    print(f"Computing: A + {scalar:.2f} * B ...")
    
    for key in common_keys:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        
        # Calculation
        new_tensor = tensor_a + (tensor_b * scalar)
        new_state_dict[key] = new_tensor

    # 3. Prepare Output Directory
    os.makedirs(save_path, exist_ok=True)

    # 4. Save Weights (Force safetensors format for output)
    out_weights_path = os.path.join(save_path, "adapter_model.safetensors")
    save_file(new_state_dict, out_weights_path, metadata={"format": "peft"})
    print(f"Saved mixed weights to: {out_weights_path}")

    # 5. Copy Config JSON
    # We assume Adapter A provides the base config structure
    config_filename = "adapter_config.json"
    src_config_path = os.path.join(adapter_a_path if os.path.isdir(adapter_a_path) else os.path.dirname(adapter_a_path), config_filename)
    
    # If not found in A, try B (fallback)
    if not os.path.exists(src_config_path):
        src_config_path = os.path.join(adapter_b_path if os.path.isdir(adapter_b_path) else os.path.dirname(adapter_b_path), config_filename)

    if os.path.exists(src_config_path):
        dst_config_path = os.path.join(save_path, config_filename)
        shutil.copyfile(src_config_path, dst_config_path)
        # print(f"Copied config to: {dst_config_path}")
    else:
        print("Warning: adapter_config.json not found in source paths. You may need to copy it manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapter Arithmetic: A + scalar * B")
    parser.add_argument("--adapter_a", type=str, required=True, help="Path to adapter A (directory or file)")
    parser.add_argument("--scalar", type=float, required=True, help="Scaling factor for adapter B")
    parser.add_argument("--adapter_b", type=str, required=True, help="Path to adapter B (directory or file)")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save the result")

    args = parser.parse_args()

    linear_arith(
        adapter_a_path=args.adapter_a,
        scalar=args.scalar,
        adapter_b_path=args.adapter_b,
        save_path=args.save_path
    )