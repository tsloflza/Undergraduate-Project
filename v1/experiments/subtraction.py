# subtraction.py

import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from train_lora import train_clip_lora
from test_lora import test_clip_lora
from task_vector import TaskVector
from utils.merge_csv import merge_accuracy_data

if __name__ == "__main__":
    # === Train base adapters ===
    class_1_10_adapter = train_clip_lora(first_class=1, last_class=10)
    class_10_20_adapter = train_clip_lora(first_class=10, last_class=20)
    class_11_20_adapter = train_clip_lora(first_class=11, last_class=20)

    test_clip_lora(first_class=1, last_class=20, adapter_path=class_1_10_adapter)
    test_clip_lora(first_class=1, last_class=20, adapter_path=class_10_20_adapter)
    test_clip_lora(first_class=1, last_class=20, adapter_path=class_11_20_adapter)

    # tau_10 = adapter(10~20) - adapter(11~20)
    tau_10 = TaskVector.from_adapters(class_10_20_adapter, class_11_20_adapter)

    base_adapter = class_1_10_adapter
    base_save_root = "output/subtraction"

    # Task Subtraction on different alphas
    for alpha in np.arange(0, 2.01, 0.1):
        save_root = os.path.join(base_save_root, f"alpha={alpha:.2f}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "merged_adapter")
        
        # New_Adapter = base_adapter - alpha * tau_10
        tau_10.apply_to_adapter(base_adapter, save_path, -alpha)
        test_clip_lora(first_class=1, last_class=10, adapter_path=save_path)
        print(f"Testing alpha={alpha:.2f} finish!")
    
    csv_path = os.path.join(base_save_root, "merge.csv")
    merge_accuracy_data(base_path=base_save_root, output_filename=csv_path)