# split_data.py

'''
Build a subset of the DF20 dataset and optionally prepare it for UAP application.

DATASET_ROOT = "/tmp2/dataset/DF20/"
first_class = 1
last_class = 10         # Pick classes from first_class to last_class (inclusive, 1-based)
num_per_class = 100     # Pick N training images per class

Usage:
    builder = DF20SubsetBuilder(DATASET_ROOT)
    subset = builder.build_subset(first_class, last_class, num_per_class)
'''

import pandas as pd
from pathlib import Path
from collections import Counter
import torch
from PIL import Image
from torchvision import transforms

from utils.constant import NORM_MEAN, NORM_STD


class DF20SubsetBuilder:
    """
    Builds a subset of the DF20 dataset by selecting a range of classes
    and a specific number of samples per class.
    """
    def __init__(self, root="/tmp2/dataset/DF20/"):
        self.root = Path(root)
        self.image_dir = self.root / "DF20-300"
        self.train_meta = self.root / "DF20-train_metadata_PROD-2.csv"
        self.valid_meta = self.root / "DF20-public_test_metadata_PROD-2.csv"

    def _load_metadata(self, path):
        """Loads and prepares the metadata CSV file."""
        df = pd.read_csv(path)
        df = df[["species", "image_path"]].dropna().reset_index(drop=True)
        df["image_path"] = df.image_path.apply(
            lambda p: self.image_dir / Path(p).name.replace(".JPG", ".jpg"))
        return df

    def build_subset(self, first_class=1, last_class=10, num_per_class=100):
        """
        Constructs the data subset.

        Args:
            first_class (int): 1-based index of the first class to include.
            last_class (int): 1-based index of the last class to include.
            num_per_class (int): Number of training images to sample per class.

        Returns:
            dict: A dictionary containing train/valid samples and class list.
        """
        # Load metadata
        train_df = self._load_metadata(self.train_meta)
        valid_df = self._load_metadata(self.valid_meta)

        # Sort classes by the number of training images
        counter = Counter(train_df["species"])
        sorted_classes = [cls for cls, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True)]

        # Select the desired range of classes
        selected_classes = sorted_classes[first_class - 1:last_class]

        subset = {"train": [], "valid": [], "classes": selected_classes}

        # Sample images for each selected class
        actual_train = num_per_class
        for cls in selected_classes:
            train_images = train_df[train_df["species"] == cls]
            valid_images = valid_df[valid_df["species"] == cls]

            # Warn if there are not enough training images
            if len(train_images) < num_per_class:
                actual_train = len(train_images)
                print(f"Warning: {cls} has only {actual_train} training images, using all of them.")

            selected_train = train_images.sample(n=actual_train, random_state=42)

            subset["train"].extend(list(zip(selected_train["image_path"], selected_train["species"])))
            subset["valid"].extend(list(zip(valid_images["image_path"], valid_images["species"])))

        print("\nSubset summary:")
        print(f"  Total classes : {len(selected_classes)}")
        print(f"  First class   : {selected_classes[0] if selected_classes else 'N/A'}")
        print(f"  Last class    : {selected_classes[-1] if selected_classes else 'N/A'}")
        print(f"  Train samples/class: {actual_train}\n")

        return subset


class DF20Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for DF20.

    Expects the provided `transform` to include `ToTensor()` (or else a fallback
    `ToTensor()` will be used). Normalization is applied internally as the final step.
    """
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.base_transform = transform
        self.normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        label_idx = self.class_to_idx[label_name]

        img = Image.open(path).convert("RGB")

        # Apply user-provided transforms (e.g., Resize, ToTensor)
        if self.base_transform:
            img_tensor = self.base_transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        # Apply the final normalization step
        img_tensor = self.normalize(img_tensor)

        return img_tensor, label_idx