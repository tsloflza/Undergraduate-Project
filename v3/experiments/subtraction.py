# subtraction.py

import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from train_clip import train_clip
from test_clip import test_clip
from task_vector import task_vector
from utils.merge_csv import merge_accuracy_data

if __name__ == "__main__":
    # === Train base models ===
    class_1_100_model = train_clip(first_class=1, last_class=100, num_per_class=500)
    class_1_10_model = train_clip(first_class=1, last_class=10, num_per_class=500, epoch=1, skip_first=True)

    test_clip(first_class=1, last_class=100, model_path=class_1_100_model)
    test_clip(first_class=1, last_class=100, model_path=class_1_10_model)

    base_model = class_1_100_model
    base_save_root = "output/subtraction"

    for s in np.arange(0, 2.01, 0.1):
        save_root = os.path.join(base_save_root, f"s={s:.2f}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, "merged_model")
        
        # New_model = class_1_100 - s * class_1_10
        task_vector(class_1_100_model, -s, class_1_10_model, save_path)
        test_clip(model_path=save_path, first_class=1, last_class=100)
        print(f"Testing s={s:.2f} finish!")

    csv_path = os.path.join(base_save_root, "merge.csv")
    merge_accuracy_data(base_path=base_save_root, output_filename=csv_path)
