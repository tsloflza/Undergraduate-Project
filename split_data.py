# split_data.py
'''
Build a subset of DF20 dataset.

DATASET_ROOT = "/tmp2/dataset/DF20/"
first_class = 1
last_class = 10         # pick classes from FIRST_CLASS to LAST_CLASS (inclusive, 1-based)
num_per_class = 100      # pick N training images per class

usage
    builder = DF20SubsetBuilder(DATASET_ROOT)
    subset = builder.build_subset(first_class, last_class, num_per_class)
'''

import pandas as pd
from pathlib import Path
from collections import Counter

class DF20SubsetBuilder:
    def __init__(self, root):
        self.root = Path(root)
        self.image_dir = self.root / "DF20-300"
        self.train_meta = self.root / "DF20-train_metadata_PROD-2.csv"
        self.valid_meta = self.root / "DF20-public_test_metadata_PROD-2.csv"

    def _load_metadata(self, path):
        df = pd.read_csv(path)
        df = df[["species", "image_path"]].dropna().reset_index(drop=True)
        df["image_path"] = df.image_path.apply(
            lambda p: self.image_dir / Path(p).name.replace(".JPG", ".jpg"))
        return df

    def build_subset(self, first_class=1, last_class=10, num_per_class=100):
        # 讀取 metadata
        train_df = self._load_metadata(self.train_meta)
        valid_df = self._load_metadata(self.valid_meta)

        # 根據 train 資料的數量排序 class
        counter = Counter(train_df["species"])
        sorted_classes = [cls for cls, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)]

        # 依照 FIRST_CLASS, LAST_CLASS 選取部分 class (1-based)
        selected_classes = sorted_classes[first_class - 1:last_class]

        subset = {"train": [], "valid": [], "classes": selected_classes}

        for cls in selected_classes:
            train_images = train_df[train_df["species"] == cls]
            valid_images = valid_df[valid_df["species"] == cls]

            # 若資料數不足，給警告並取全部
            actual_train = min(num_per_class, len(train_images))
            if actual_train < num_per_class:
                print(f"Warning: {cls} train images less than {num_per_class}, use {actual_train}")

            selected_train = train_images.sample(n=actual_train, random_state=42)
            selected_valid = valid_images

            subset["train"].extend(list(zip(selected_train["image_path"], selected_train["species"])))
            subset["valid"].extend(list(zip(selected_valid["image_path"], selected_valid["species"])))

        print()
        print("Subset summary:")
        print(f"  Total classes : {len(selected_classes)}")
        print(f"  First class   : {selected_classes[0] if selected_classes else 'N/A'}")
        print(f"  Last class    : {selected_classes[-1] if selected_classes else 'N/A'}")
        print(f"  actual_train  : {actual_train}")
        print()

        return subset

# TODO: merge dataset