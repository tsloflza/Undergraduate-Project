from pathlib import Path
from collections import Counter
import json

class OxfordDataset:
    def __init__(self, metadata_path, image_dir, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', 'test'.")

        self.metadata_path = Path(metadata_path)
        self.image_dir = Path(image_dir)
        self.split = split

        # 讀取 JSON metadata
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        # 每筆資料格式：[ "image_00190.jpg", 76, "passion flower" ]
        split_data = metadata[split]
        image_paths = [self.image_dir / item[0] for item in split_data]
        class_names = [item[2] for item in split_data]

        self.samples = list(zip(image_paths, class_names))
        self.classes = sorted(set(class_names))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.class_num = Counter(class_names)

    @property
    def num_classes(self) -> int:
        return len(self.classes)


if __name__ == "__main__":
    # metadata_file = "/tmp2/dataset/oxford_flowers/split_zhou_OxfordFlowers.json"
    # image_dir = "/tmp2/dataset/oxford_flowers/jpg/"
    metadata_file = "/tmp2/dataset/oxford_pets/split_zhou_OxfordPets.json"
    image_dir = "/tmp2/dataset/oxford_pets/images/"

    # 讀取 train , val, test
    train_set = OxfordDataset(metadata_file, image_dir, "train")
    val_set = OxfordDataset(metadata_file, image_dir, "val")
    test_set = OxfordDataset(metadata_file, image_dir, "test")

    # 依 train 數量排序
    sorted_classes = sorted(train_set.class_num.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Class':30s} {'Train':>10s} {'Val':>10s} {'Test':>10s}")
    print("-" * 65)

    for cls, train_count in sorted_classes:
        val_count = val_set.class_num.get(cls, 0)
        test_count = test_set.class_num.get(cls, 0)
        print(f"{cls:30s} {train_count:10d} {val_count:10d} {test_count:10d}")
