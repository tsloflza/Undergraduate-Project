# subtraction.py

import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from train_lora import train_clip_lora
from test_lora import test_clip_lora
from adapter_arith import linear_arith
from utils.merge_csv import merge_accuracy_data

if __name__ == "__main__":
    # === Train base adapters ===
    f10_p0_adapter = train_clip_lora(flowers_class=10, pets_class=0)
    f0_p10_adapter = train_clip_lora(flowers_class=0, pets_class=10)
    f10_p10_adapter = train_clip_lora(flowers_class=10, pets_class=10)

    test_clip_lora(adapter_path=f10_p0_adapter, flowers_class=10, pets_class=10)
    test_clip_lora(adapter_path=f0_p10_adapter, flowers_class=10, pets_class=10)
    test_clip_lora(adapter_path=f10_p10_adapter, flowers_class=10, pets_class=10)

    # === f10_p10 - s * f10_p0 ===
    base_adapter = f10_p10_adapter
    base_save_root = "output/subtraction/f10_p0"

    for s in np.arange(0, 1.01, 0.1):
        save_root = os.path.join(base_save_root, f"s={s:.2f}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "merged_adapter")
        
        # New_Adapter = f10_p10 - s * f10_p0
        linear_arith(f10_p10_adapter, -s, f10_p0_adapter, save_path)
        test_clip_lora(adapter_path=save_path, flowers_class=10, pets_class=10)
        print(f"Testing s={s:.2f} finish!")

    csv_path = os.path.join(base_save_root, "merge.csv")
    merge_accuracy_data(base_path=base_save_root, output_filename=csv_path)

    # === f10_p10 - s * f0_p10 ===
    base_adapter = f10_p10_adapter
    base_save_root = "output/subtraction/f0_p10"

    for s in np.arange(0, 1.01, 0.1):
        save_root = os.path.join(base_save_root, f"s={s:.2f}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "merged_adapter")
        
        # New_Adapter = f10_p10 - s * f0_p10
        linear_arith(f10_p10_adapter, -s, f0_p10_adapter, save_path)
        test_clip_lora(adapter_path=save_path, flowers_class=10, pets_class=10)
        print(f"Testing s={s:.2f} finish!")
    
    csv_path = os.path.join(base_save_root, "merge.csv")
    merge_accuracy_data(base_path=base_save_root, output_filename=csv_path)