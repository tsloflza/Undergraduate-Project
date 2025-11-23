from pathlib import Path
from collections import Counter
import pandas as pd

class DF20Dataset:
    def __init__(self, root, split, transform=None, target_transform=None):
        if split not in ("train", "valid"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'valid'.")

        self.root = Path(root)
        self.split = split

        # 選擇對應的 metadata 檔
        if split == "train":
            metadata_path = self.root / "DF20-train_metadata_PROD-2.csv"
        else:
            metadata_path = self.root / "DF20-public_test_metadata_PROD-2.csv"

        image_dir = self.root / "DF20-300"
        df = pd.read_csv(metadata_path)
        df = df[["species", "image_path"]].dropna().reset_index(drop=True)
        df["image_path"] = df.image_path.apply(
            lambda p: image_dir / Path(p).name.replace(".JPG", ".jpg"))

        self.samples = list(zip(df["image_path"], df["species"]))
        self.classes = sorted(df["species"].unique())
        self.dataframe = df
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.class_num = Counter(df["species"])

    @property
    def num_classes(self) -> int:
        return len(self.classes)


if __name__ == "__main__":
    dataset_root = "/tmp2/dataset/DF20/"

    # 讀取 train 與 valid
    train_set = DF20Dataset(root=dataset_root, split="train")
    valid_set = DF20Dataset(root=dataset_root, split="valid")

    # 依 train 數量排序
    sorted_classes = sorted(train_set.class_num.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Class':30s} {'Train':>10s} {'Valid':>10s}")
    print("-" * 55)

    for cls, train_count in sorted_classes:
        valid_count = valid_set.class_num.get(cls, 0)
        print(f"{cls:30s} {train_count:10d} {valid_count:10d}")