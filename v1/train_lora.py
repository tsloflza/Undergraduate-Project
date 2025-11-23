# train_lora.py

import os
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import MulticlassAccuracy

from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

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
    def __init__(self, model: CLIPModel, class_names, lr: float = 1e-4):
        """
        model: CLIPModel (optionally with LoRA)
        class_names: list of class labels (strings)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr

        # 準備 text embeddings
        tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
        self.tokenizer = tokenizer

        # prompt 模板
        prompts = [f"a photo of a {name}" for name in class_names]
        text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds = self.model.text_model(**text_inputs).pooler_output
            text_embeds = self.model.text_projection(text_embeds)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        # 註冊為 buffer，不參與訓練
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


def train_clip_lora(
    batch_size: int = 32,
    epoch: int = 10,
    first_class: int = 1,
    last_class: int = 10,
    num_per_class: int = 100,
    save_dir: str = "./output",
    lr: float = 1e-4,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    """
    Fine-tune CLIP image encoder with LoRA adapters. Save only the adapter (PEFT save_pretrained).

    Returns:
        adapter_path (str): path to saved adapter directory
    """
    save_dir = os.path.join(save_dir, f"class_{first_class}_to_{last_class}")
    os.makedirs(save_dir, exist_ok=True)
    adapter_dir = os.path.join(save_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # --- 資料 ---
    builder = DF20SubsetBuilder()
    subset = builder.build_subset(first_class, last_class, num_per_class)
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
    base_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    # prepare_model_for_kbit_training is useful when using 4-bit quantization; for regular LoRA it's optional.
    # base_model = prepare_model_for_kbit_training(base_model)  # not necessary here

    # --- LoRA 設定 (只放在 vision/image encoder 的 target modules) ---
    # target modules list is broad to cover attention & MLP linear layers in CLIP-ViT
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # wrap the model with PEFT
    peft_model = get_peft_model(base_model, lora_config)

    # Move to device
    peft_model.to(device)

    # --- Lightning Module ---
    lit_model = LitClassifier(peft_model, class_names=class_names, lr=lr)

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
    trainer.fit(lit_model, train_loader, valid_loader)

    # --- 儲存 adapter (只存 PEFT adapter) ---
    # get the underlying PeftModel object (peft_model)
    try:
        # if lit_model.model is PeftModel, it has save_pretrained
        peft_object = lit_model.model
    except Exception:
        peft_object = peft_model

    # save adapter weights only
    try:
        peft_object.save_pretrained(adapter_dir)
    except Exception as e:
        # fallback: try base_model.save_pretrained (not desired) but raise for clarity
        raise RuntimeError(f"Failed to save adapter via peft_object.save_pretrained: {e}")

    # --- 儲存 training params & curve ---
    params = {
        "batch_size": batch_size,
        "epoch": epoch,
        "first_class": first_class,
        "last_class": last_class,
        "num_per_class": num_per_class,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lr": lr,
    }
    with open(os.path.join(save_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=4)

    df = pd.DataFrame(loss_callback.history)
    csv_path = os.path.join(save_dir, "train_loss.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved training curve: {csv_path}")

    print(f"LoRA adapter saved at: {adapter_dir}")
    return adapter_dir


if __name__ == "__main__":
    train_clip_lora()
