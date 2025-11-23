# test_clip.py

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.classification import MulticlassAccuracy
import pandas as pd
from transformers import CLIPModel, AutoTokenizer

from split_data import DF20SubsetBuilder
from train_clip import DF20SubsetDataset # 引用 train_clip 中的 Dataset class
from utils.constant import NORM_MEAN, NORM_STD, CLIP_MODEL_NAME


def test_clip(
    batch_size=32,
    first_class=1,
    last_class=10,
    num_per_class=100,
    model_path="output/class_1_to_10/finetuned_model", # 指向儲存完整模型的資料夾
    skip_first=False
):
    """
    Evaluate Fine-tuned CLIP classifier on DF20 subset validation data.
    Saves class-wise accuracy results to a CSV file.

    Args:
        batch_size (int): batch size for evaluation.
        first_class (int): first class index (inclusive).
        last_class (int): last class index (inclusive).
        num_per_class (int): number of training samples per class (for subset consistency).
        model_path (str): path to the fine-tuned model directory.

    Returns:
        pd.DataFrame: class-wise accuracy results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 準備資料 ---
    builder = DF20SubsetBuilder()
    subset = builder.build_subset(first_class, last_class, num_per_class, skip_first)
    class_names = subset["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    valid_dataset = DF20SubsetDataset(subset["valid"], class_to_idx, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 2. 載入已微調的 CLIP 模型 ---
    print(f"Loading fine-tuned CLIP model from: {model_path}")
    try:
        # 嘗試從本地路徑載入
        clip_model = CLIPModel.from_pretrained(model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}. Please check the path.")
        return None
        
    clip_model = clip_model.to(device)
    clip_model.eval()

    # --- 3. 計算 Text Embeddings (分類錨點) ---
    # 嘗試從模型目錄載入 tokenizer，如果沒有則從原始名稱載入
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        print("Tokenizer not found in model path, loading default.")
        tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)

    # 文字描述：使用與訓練時相同的模板
    prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        # 提取 Text Embeddings
        text_embeds_pooler = clip_model.text_model(**text_inputs).pooler_output
        text_embeds = clip_model.text_projection(text_embeds_pooler)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1) 

    # --- 4. 計算每個 class 的 accuracy ---
    acc_metric = MulticlassAccuracy(num_classes=len(class_names), average=None).to(device)

    all_preds, all_labels = [], []
    print("Start inference...")
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)

            pooler_output = clip_model.vision_model(pixel_values=x).pooler_output
            image_embeds = clip_model.visual_projection(pooler_output)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            
            logits = image_embeds @ text_embeds.T 
            
            preds = logits.argmax(dim=-1)

            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    class_acc = acc_metric(all_preds, all_labels).cpu().numpy()

    # --- 5. 輸出結果 ---
    # 為了避免汙染 model 資料夾，我們把結果存在 model 資料夾的上層
    output_dir = os.path.dirname(model_path.rstrip('/'))
    csv_path = os.path.join(output_dir, f"class_{first_class}_to_{last_class}.csv")

    df = pd.DataFrame({
        "class_name": class_names,
        "accuracy": class_acc
    })
    df.to_csv(csv_path, index=False)

    print(f"Test Results saved to: {csv_path}")
    return df


if __name__ == "__main__":
    test_clip()