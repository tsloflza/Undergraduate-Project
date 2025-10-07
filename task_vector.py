# task_vector.py
"""
Task vector utilities tailored for PEFT LoRA adapters.

Features:
- Construct a TaskVector by subtracting two LoRA adapter states (AdapterA - AdapterB).
- Addition, subtraction, scaling of TaskVector.
- Convenient save/load methods for the vector itself.
- Apply (add scaled task vector) to a base LoRA adapter state and write a new adapter file.

Note: This implementation assumes the input are lightweight PEFT adapter state_dicts 
      (i.e., the content of adapter_model.safetensors or .bin).

Example usage:
    tau_cat = TaskVector.from_adapters("path/to/cat_lora", "path/to/base_lora") # create task vector

    tau_cat.save(path)  # save the vector
    tau_cat.load(path)  # load the vector

    tau_sum = tau_cat + tau_dog # add two task vectors
    tau_diff = tau_cat - tau_dog # subtract two task vectors
    tau_half = tau_half = tau_cat * 0.5 # scale the task vector
    tau_half = tau_cat.scale(0.5) # scale the task vector (alias)

    tau_cat.apply_to_adapter("path/to/new_base_lora", "path/to/merged_adapter.pt", alpha=0.5)

    tau_cat.summary(n=10) # print first n keys in the vector
    len(tau_cat) # get number of parameters in the vector
    tau_cat.keys() # get all keys in the vector
"""

from typing import Dict, Optional
import torch
import os
import re
from safetensors.torch import save_file
import shutil

TensorDict = Dict[str, torch.Tensor]

# --- Adapter Loading Utility ---

def _load_adapter_state(adapter_path: str) -> TensorDict:
    """
    Load a PEFT adapter state dictionary from a path (e.g., adapter_model.safetensors or .bin).
    
    Args:
        adapter_path: Path to the adapter weight file (.pt, .pth, .bin, or .safetensors).
        
    Returns:
        TensorDict: A dictionary of adapter parameter names to tensors (on CPU).
    """
    if os.path.isdir(adapter_path):
        # Look for common PEFT files inside the directory
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_path, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            path_to_load = safetensors_path
        elif os.path.exists(bin_path):
            path_to_load = bin_path
        else:
            raise FileNotFoundError(f"No adapter file found in directory: {adapter_path}")
    else:
        path_to_load = adapter_path

    # Use safetensors for safety if available, otherwise torch.load
    if path_to_load.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path_to_load, device="cpu")
    else:
        # Load .pt, .pth, or .bin
        state = torch.load(path_to_load, map_location="cpu", weights_only=False)

    # Ensure tensors are on CPU and detached
    state_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) 
                 for k, v in state.items()}
    return state_cpu


# --- TaskVector Class ---

class TaskVector:
    """
    Represents a task vector as a mapping from parameter-name -> tensor (difference),
    specifically designed for LoRA adapter weights (A and B matrices).
    """

    def __init__(self, vector: Optional[TensorDict] = None):
        """Initializes the TaskVector with a dictionary of weights."""
        self.vector = {} if vector is None else {k: v.detach().cpu() for k, v in vector.items()}
        # Ensure only adapter weights (lora_A and lora_B) are included
        self.vector = {k: v for k, v in self.vector.items() 
                       if re.search(r"lora_(A|B)", k)}

    @classmethod
    def from_adapters(cls, adapter_a: str, adapter_b: str):
        """
        Create a TaskVector = state_a - state_b.
        
        Args:
            adapter_a: Path to the fine-tuned adapter (A) state.
            adapter_b: Path to the base adapter (B) state (often the zero-initialized one).
        """
        state_a = _load_adapter_state(adapter_a)
        state_b = _load_adapter_state(adapter_b)

        vec = {}
        for k, ta in state_a.items():
            # Check for name consistency and shape
            if k not in state_b:
                continue
            tb = state_b[k]
            if ta.shape != tb.shape:
                continue
            
            # The difference is the task vector
            vec[k] = (ta - tb).cpu().clone()
            
        return cls(vec)

    # --- Arithmetic Operations (Standard) ---

    def keys(self):
        """Returns a list of parameter names in the vector."""
        return list(self.vector.keys())

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Adds two TaskVectors element-wise."""
        new = {}
        keys = set(self.keys()) | set(other.keys())
        for k in keys:
            a = self.vector.get(k)
            b = other.vector.get(k)
            if a is None or b is None:
                # Handle keys present in only one vector by cloning the existing one
                new[k] = (a if a is not None else b).clone()
            else:
                new[k] = (a + b).clone()
        return TaskVector(new)

    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Subtracts two TaskVectors element-wise."""
        new = {}
        keys = set(self.keys()) | set(other.keys())
        for k in keys:
            a = self.vector.get(k)
            b = other.vector.get(k)
            if a is None:
                new[k] = (-b).clone()
            elif b is None:
                new[k] = a.clone()
            else:
                new[k] = (a - b).clone()
        return TaskVector(new)
    
    def __mul__(self, scalar: float) -> "TaskVector":
        """Scales the TaskVector by a scalar."""
        new = {k: (v * float(scalar)).clone() for k, v in self.vector.items()}
        return TaskVector(new)

    def __rmul__(self, scalar: float) -> "TaskVector":
        """Handles scalar multiplication from the left."""
        return self.__mul__(scalar)    
    def scale(self, scalar: float) -> "TaskVector":
        """Scales the vector (alias for *)."""
        return self * scalar

    # --- Save/Load (Convenience) ---

    def save(self, path: str):
        """
        Save the task vector (the difference weights) to a file using torch.save.
        Stored dict format: {"vector": state_dict}
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        to_save = {"vector": self.vector}
        torch.save(to_save, path)
        return path

    @classmethod
    def load(cls, path: str) -> "TaskVector":
        """
        Load a TaskVector from a file saved by TaskVector.save().
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        vec = payload.get("vector", payload)
        # Ensure tensors are loaded as CPU tensors
        vec_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in vec.items()}
        return cls(vec_cpu)

    # --- Application Logic ---

    def apply_to_adapter(self, base_adapter_path: str, out_dir_path: str, alpha: float = 1.0):
        """
        Apply (add) alpha * task_vector to the base adapter's state and save the new adapter state
        into a directory structure compatible with PEFT loading.
        
        Args:
            base_adapter_path: Path to the base adapter state (e.g., zero-initialized LoRA weights).
            out_dir_path: Path to write the NEW ADAPTER DIRECTORY.
            alpha: Scaling factor for the task vector.
        """
        base_state = _load_adapter_state(base_adapter_path)
        
        # Prepare new state dict from base
        new_state = dict(base_state)
        
        missing = []
        mismatched = []
        
        for k, v in self.vector.items():
            if k not in new_state:
                missing.append(k)
                continue
            if new_state[k].shape != v.shape:
                mismatched.append(k)
                continue
            
            # Apply the task vector: New_Weight = Base_Weight + alpha * Delta_Weight
            new_state[k] = (new_state[k] + v * float(alpha)).clone()

        os.makedirs(out_dir_path, exist_ok=True)
        out_weight_path = os.path.join(out_dir_path, "adapter_model.safetensors")
        save_file(new_state, out_weight_path, metadata={"format": "peft"})

        source_config_path = os.path.join(base_adapter_path, "adapter_config.json")
        target_config_path = os.path.join(out_dir_path, "adapter_config.json")
        
        if os.path.exists(source_config_path):
            shutil.copyfile(source_config_path, target_config_path) # 複製檔案
            config_msg = "adapter_config.json copied successfully."
        else:
            config_msg = "⚠️ WARNING: adapter_config.json not found at base path."

        return out_dir_path
    
    # --- Utility Methods ---

    def __len__(self):
        """Returns the number of parameters in the task vector."""
        return len(self.vector)

    def summary(self, n=10):
        """Prints a summary of the task vector's contents."""
        ks = list(self.vector.keys())
        print(f"TaskVector (LoRA-based) with {len(ks)} params. Showing first {min(n, len(ks))}:")
        for k in ks[:n]:
            print(f"  {k}: {tuple(self.vector[k].shape)}")