# train_clip.py

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import MulticlassAccuracy

from transformers import CLIPModel, AutoTokenizer
import pandas as pd
from torchvision import transforms

from split_data import DF20SubsetBuilder
from utils.constant import NORM_MEAN, NORM_STD, CLIP_MODEL_NAME

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DF20SubsetDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


class LitClassifier(L.LightningModule):
    def __init__(self, model: CLIPModel, class_names, lr: float = 1e-5):
        """
        model: Full CLIPModel
        class_names: list of class labels (strings)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr

        # 準備 tokenizer
        # 注意：如果是載入已訓練過的模型，model 裡可能沒有 tokenizer，需要重新載入
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)

        # 預先計算 Text Embeddings (因為我們通常凍結 Text Encoder，所以只需計算一次)
        prompts = [f"a photo of a {name}" for name in class_names]
        text_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        # 確保 text model 在正確的 device 上以進行推論
        self.model.to(device)
        
        with torch.no_grad():
            text_embeds = self.model.text_model(**text_inputs).pooler_output
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
        # 註冊為 buffer，不參與訓練梯度更新
        self.register_buffer("text_embeds", text_embeds)
        self.num_classes = len(class_names)

        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average="macro")

    def forward(self, x):
        # 提取圖片嵌入
        pooler_output = self.model.vision_model(pixel_values=x).pooler_output
        image_embeds = self.model.visual_projection(pooler_output)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # 用點乘取 logits
        logits = image_embeds @ self.text_embeds.T
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        # 只優化 requires_grad 為 True 的參數
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)


class LossHistoryCallback(Callback):
    """紀錄每個 epoch 的 train_loss 與 val_loss"""
    def __init__(self):
        super().__init__()
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        entry = {
            "epoch": int(trainer.current_epoch),
            "train_loss": float(logs.get("train_loss")) if "train_loss" in logs else None,
            "val_loss": float(logs.get("val_loss")) if "val_loss" in logs else None,
            "val_acc": float(logs.get("val_acc")) if "val_acc" in logs else None,
        }
        self.history.append(entry)


def train_clip(
    batch_size: int = 32,
    epoch: int = 10,
    first_class: int = 1,
    last_class: int = 10,
    num_per_class: int = 100,
    save_dir: str = "./output",
    lr: float = 1e-5,
    skip_first: bool = False
):
    """
    Fine-tune CLIP image encoder (Standard Fine-tuning). 
    Saves the full model.

    Returns:
        model_save_path (str): path to saved model directory
    """
    save_dir = os.path.join(save_dir, f"class_{first_class}_to_{last_class}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型儲存路徑
    model_save_path = os.path.join(save_dir, "finetuned_model")
    os.makedirs(model_save_path, exist_ok=True)

    # --- 資料 ---
    builder = DF20SubsetBuilder()
    subset = builder.build_subset(first_class, last_class, num_per_class, skip_first)
    class_names = subset["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    train_dataset = DF20SubsetDataset(subset["train"], class_to_idx, transform=transform)
    valid_dataset = DF20SubsetDataset(subset["valid"], class_to_idx, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 模型載入 ---
    print(f"Loading base model: {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

    # --- 設定訓練策略：凍結 Text Encoder，只訓練 Vision Encoder ---
    # 1. 凍結所有參數
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 解凍 Vision Encoder 參數
    for param in model.vision_model.parameters():
        param.requires_grad = True
        
    # 3. 解凍 Visual Projection (將 vision output 轉到 embedding space 的層)
    for param in model.visual_projection.parameters():
        param.requires_grad = True

    # 移動到 GPU
    model.to(device)

    # --- Lightning Module ---
    lit_model = LitClassifier(model, class_names=class_names, lr=lr)

    # --- callback / trainer ---
    loss_callback = LossHistoryCallback()
    L.seed_everything(42, workers=True)

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=epoch,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=save_dir,
        callbacks=[loss_callback],
    )

    # --- 訓練 ---
    print("Starting full fine-tuning...")
    trainer.fit(lit_model, train_loader, valid_loader)

    # --- 儲存完整模型 ---
    print(f"Saving full model to: {model_save_path}")
    # 使用 Hugging Face 的 save_pretrained
    lit_model.model.save_pretrained(model_save_path)
    # 同時儲存 tokenizer，方便之後載入使用
    lit_model.tokenizer.save_pretrained(model_save_path)

    # --- 儲存 training params & curve ---
    params = {
        "batch_size": batch_size,
        "epoch": epoch,
        "first_class": first_class,
        "last_class": last_class,
        "num_per_class": num_per_class,
        "lr": lr,
        "training_type": "full_finetune_vision_only"
    }
    with open(os.path.join(save_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    df = pd.DataFrame(loss_callback.history)
    csv_path = os.path.join(save_dir, "train_loss.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved training curve: {csv_path}")

    return model_save_path


if __name__ == "__main__":
    train_clip()