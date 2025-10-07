# addition.py

from train import train_clip_lora
from test import test_clip_lora
from task_vector import TaskVector
import os
import numpy as np

train_clip_lora(first_class=1, last_class=9)
train_clip_lora(first_class=1, last_class=10)
train_clip_lora(first_class=10, last_class=20)
train_clip_lora(first_class=11, last_class=20)

# tau_10 = adapter(10~20) - adapter(11~20)
tau_10 = TaskVector.from_adapters("output/class_10_to_20/lora_adapter", "output/class_11_to_20/lora_adapter")

base_adapter = "output/class_1_to_9/lora_adapter" # ➖ base_adapter = "output/class_1_to_10/lora_adapter"
base_save_root = "output/addition"

# Task Addition on different alphas
for alpha in np.arange(0, 1.01, 0.05): # ➖ for alpha in np.arange(0, -1.01, -0.05):
    save_root = os.path.join(base_save_root, f"alpha={alpha:.2f}")
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, "merged_adapter")
    
    # New_Adapter = base_adapter + alpha * tau_10
    tau_10.apply_to_adapter(base_adapter, save_path, alpha=alpha)
    print(f"Generated adapter with α={alpha:.2f}. Saved to: {save_path}")

    test_clip_lora(first_class=1, last_class=10, adapter_path=save_path)
    print(f"Testing alpha={alpha:.2f} finish!")