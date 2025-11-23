from pathlib import Path
import json
from collections import Counter

class OxfordDatasetBuilder:
    def __init__(self, 
                 flowers_meta="/tmp2/dataset/oxford_flowers/split_zhou_OxfordFlowers.json",
                 flowers_img="/tmp2/dataset/oxford_flowers/jpg/",
                 pets_meta="/tmp2/dataset/oxford_pets/split_zhou_OxfordPets.json",
                 pets_img="/tmp2/dataset/oxford_pets/images/"):
        self.flowers_meta = Path(flowers_meta)
        self.flowers_img = Path(flowers_img)
        self.pets_meta = Path(pets_meta)
        self.pets_img = Path(pets_img)

    def _load_metadata(self, metadata_path, image_dir):
        """讀取 JSON 並合併 train / val 作為 train"""
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        train_data = metadata.get("train", []) + metadata.get("val", [])
        test_data = metadata.get("test", []) if "test" in metadata else []
        # 每筆資料格式：[ "image_xxx.jpg", class_id, class_name ]
        train_samples = [(image_dir / item[0], item[2]) for item in train_data]
        test_samples = [(image_dir / item[0], item[2]) for item in test_data]
        return train_samples, test_samples

    def _select_first_classes(self, samples, num_classes):
        """選取前 num_classes 個類別的樣本"""
        counter = Counter([cls for _, cls in samples])
        class_names = [cls for cls, _ in counter.most_common()]
        if num_classes > len(class_names):
            raise ValueError(f"Requested {num_classes} classes, but only {len(class_names)} available.")
        selected_classes = class_names[:num_classes]
        filtered = [(img, cls) for img, cls in samples if cls in selected_classes]
        return filtered, selected_classes

    def build_mixed_subset(self, flowers_class, pets_class):
        # 讀取兩個資料集
        flower_train, flower_test = self._load_metadata(self.flowers_meta, self.flowers_img)
        pet_train, pet_test = self._load_metadata(self.pets_meta, self.pets_img)

        # 各取前幾個 class
        flower_train_filtered, flower_classes = self._select_first_classes(flower_train, flowers_class)
        pet_train_filtered, pet_classes = self._select_first_classes(pet_train, pets_class)
        flower_test_filtered = [(img, cls) for img, cls in flower_test if cls in flower_classes]
        pet_test_filtered = [(img, cls) for img, cls in pet_test if cls in pet_classes]

        # 混合 train / test
        mixed_train = flower_train_filtered + pet_train_filtered
        mixed_test = flower_test_filtered + pet_test_filtered

        # 建立 class list 與索引
        all_classes = sorted(flower_classes + pet_classes)
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

        # 統計分佈
        train_counter = Counter([cls for _, cls in mixed_train])
        test_counter = Counter([cls for _, cls in mixed_test])
        # print(f"📊 Mixed Dataset Summary:")
        # for cls in all_classes:
        #     print(f"{cls:30s} train: {train_counter.get(cls, 0):5d}    test: {test_counter.get(cls, 0):5d}")

        return {
            "train": mixed_train,
            "test": mixed_test,
            "classes": all_classes,
            "class_to_idx": class_to_idx,
        }


if __name__ == "__main__":
    builder = OxfordDatasetBuilder()
    subset = builder.build_mixed_subset(flowers_class=10, pets_class=10)

    print("\nExample:")
    print(f"Total train samples: {len(subset['train'])}")
    print(f"Total test samples: {len(subset['test'])}")
    print(f"Num classes: {len(subset['classes'])}")
    print(f"All classes: {subset['classes']}")
