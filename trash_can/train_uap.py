# train_uap.py
"""
Bilevel optimization for Unlearnable UAP (universal adversarial perturbation).
Inner (theta update): train model on subset1 (first-class images are perturbed by δ).
Outer (delta update): use subset2 (perturbed) + subset3 (clean) to update δ.
Evaluate acc on subset1, subset2, subset3.
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel
from tqdm import tqdm
import numpy as np
import sys

from split_data import DF20SubsetBuilder

# ----------------------------
# Config
# ----------------------------
DATASET_ROOT = "/tmp2/dataset/DF20/"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
CKPT_PATH = "lightning_logs/version_2/checkpoints/epoch=99-step=7900.ckpt"
OUTPUT_DIR = "uap_outputs"
BATCH_SIZE = 32
INNER_EPOCHS = 5 # update model
THETA_LR = 1e-6
OUTER_ITERS = 20 # update noise
DELTA_LR = 1e-3
EPS = 8.0 / 255.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# ----------------------------
# Dataset wrapper
# ----------------------------
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

# ----------------------------
# Helper: apply UAP
# ----------------------------
def apply_uap_to_batch(images, delta_norm, mask):
    if mask is None:
        return images
    if mask.any():
        delta_batched = delta_norm.unsqueeze(0).expand(images.size(0), -1, -1, -1)
        delta_batched = delta_batched.to(images.device)
        images = images.clone()
        images[mask] = images[mask] + delta_batched[mask]
    return images

# ----------------------------
# Classifier
# ----------------------------
class SimpleClipClassifier(nn.Module):
    def __init__(self, clip_model: CLIPModel, num_classes, freeze_backbone=True):
        super().__init__()
        self.clip = clip_model
        self.fc = nn.Linear(self.clip.config.projection_dim, num_classes)
        if freeze_backbone:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = False

    def forward(self, x):
        pooler_output = self.clip.vision_model(pixel_values=x).pooler_output
        image_embeds = self.clip.visual_projection(pooler_output)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        return self.fc(image_embeds)

# ----------------------------
# Eval helper
# ----------------------------
def evaluate(model, loader, delta, apply_noise_all=False, apply_noise_mask=False, target_idx=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            if apply_noise_all:
                mask = torch.ones((lbls.size(0),), dtype=torch.bool, device=DEVICE)
                imgs = apply_uap_to_batch(imgs, delta, mask)
            elif apply_noise_mask:
                mask = (lbls == target_idx)
                imgs = apply_uap_to_batch(imgs, delta, mask)

            logits = model(imgs)
            preds = logits.argmax(dim=-1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
    return correct / total

# ----------------------------
# Main bilevel procedure
# ----------------------------
def main():
    if OUTPUT_DIR is not None:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        log_path = os.path.join(OUTPUT_DIR, "out.log")
        sys.stdout = open(log_path, "w")

    builder = DF20SubsetBuilder(DATASET_ROOT)

    subset1 = builder.build_subset(10, 250, True)   # inner
    subset2 = builder.build_subset(1, 250, True)    # outer perturbed
    subset3 = builder.build_subset(1, 250, False)   # outer clean

    target_class = subset2["classes"][0]
    print(f"Target class: {target_class}")

    class_names = subset1["classes"]
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    if target_class not in class_to_idx:
        class_to_idx[target_class] = len(class_to_idx)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    inner_loader = DataLoader(DF20SubsetDataset(list(subset1["train"]), class_to_idx, transform), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    outer1_loader = DataLoader(DF20SubsetDataset(list(subset1["valid"]), class_to_idx, transform),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    outer2_loader = DataLoader(DF20SubsetDataset(list(subset2["valid"]), class_to_idx, transform),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    outer3_loader = DataLoader(DF20SubsetDataset(list(subset3["valid"]), class_to_idx, transform),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    model = SimpleClipClassifier(copy.deepcopy(clip_model),
                                 num_classes=len(class_to_idx),
                                 freeze_backbone=False).to(DEVICE)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)
        print(f"Loaded checkpoint from {CKPT_PATH}")

    criterion = nn.CrossEntropyLoss()
    theta_optimizer = torch.optim.AdamW(model.parameters(), lr=THETA_LR)

    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(3,1,1)
    clamp_val = (EPS / std).to(DEVICE)
    delta = nn.Parameter(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE))
    delta_optimizer = torch.optim.SGD([delta], lr=DELTA_LR)

    # ---- zero shot ----
    print(f"\n=== Zero Shot ===")
    acc1 = evaluate(model, outer1_loader, delta, apply_noise_mask=True, target_idx=class_to_idx[target_class])
    acc2 = evaluate(model, outer2_loader, delta, apply_noise_all=True)
    acc3 = evaluate(model, outer3_loader, delta, apply_noise_all=False)

    print(f"  Acc subset1 : {acc1*100:.2f}%")
    print(f"  Acc subset2 : {acc2*100:.2f}%")
    print(f"  Acc subset3 : {acc3*100:.2f}%")

    for outer_iter in range(OUTER_ITERS):
        print(f"\n=== Outer iter {outer_iter+1}/{OUTER_ITERS} ===")

        # ---- inner loop ----
        model.train()

        for epoch in tqdm(range(INNER_EPOCHS), desc="Inner epochs"):
            for images, labels in inner_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                mask = (labels == class_to_idx[target_class])
                images_pert = apply_uap_to_batch(images, delta, mask)

                logits = model(images_pert)
                loss = criterion(logits, labels)

                theta_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                theta_optimizer.step()

        # ---- outer loop ----
        model.eval()
        delta_optimizer.zero_grad()
        it2, it3 = iter(outer2_loader), iter(outer3_loader)
        n_steps = min(len(outer2_loader), len(outer3_loader))
        total_loss = 0.0

        for step in range(n_steps):
            try: imgs2, lbl2 = next(it2)
            except StopIteration: it2 = iter(outer2_loader); imgs2, lbl2 = next(it2)
            try: imgs3, lbl3 = next(it3)
            except StopIteration: it3 = iter(outer3_loader); imgs3, lbl3 = next(it3)

            imgs2, lbl2 = imgs2.to(DEVICE), lbl2.to(DEVICE)
            imgs3, lbl3 = imgs3.to(DEVICE), lbl3.to(DEVICE)

            mask2 = torch.ones((imgs2.size(0),), dtype=torch.bool, device=DEVICE)
            imgs2_pert = apply_uap_to_batch(imgs2, delta, mask2)

            logits2, logits3 = model(imgs2_pert), model(imgs3)
            loss2, loss3 = criterion(logits2, lbl2), criterion(logits3, lbl3)
            loss_comb = -(loss2 + loss3) * 0.5
            loss_comb.backward()
            total_loss += (-(loss2.item() + loss3.item()) * 0.5)

        delta_optimizer.step()
        with torch.no_grad():
            for c in range(3):
                delta[c].clamp_(-clamp_val[c,0,0], clamp_val[c,0,0])

        print(f"Outer iter {outer_iter+1} avg (-loss) update: {total_loss / max(1, n_steps):.4f}")

        # ---- evaluate on subset1/2/3 ----
        acc1 = evaluate(model, outer1_loader, delta, apply_noise_mask=True, target_idx=class_to_idx[target_class])
        acc2 = evaluate(model, outer2_loader, delta, apply_noise_all=True)
        acc3 = evaluate(model, outer3_loader, delta, apply_noise_all=False)

        print(f"  Acc subset1 : {acc1*100:.2f}%")
        print(f"  Acc subset2 : {acc2*100:.2f}%")
        print(f"  Acc subset3 : {acc3*100:.2f}%")

    # After outer loops, save delta to file (in pixel-space approx)
    uap_delta_norm_path = os.path.join(OUTPUT_DIR, "uap_delta_norm.npy")
    uap_delta_pixel_path = os.path.join(OUTPUT_DIR, "uap_delta_pixel.npy")

    delta_cpu = delta.detach().cpu()
    np.save(uap_delta_norm_path, delta_cpu.numpy())
    print(f"\nSaved UAP to {uap_delta_norm_path}")

    # Covert back to pixel-space delta (0..255)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    delta_pixel = (delta_cpu * std + mean)  # not exact inverse but indicative
    # clip to [0,1]
    delta_pixel = torch.clamp(delta_pixel, 0.0, 1.0)
    np.save(uap_delta_pixel_path, delta_pixel.numpy())
    print(f"Saved approximate pixel-space UAP to {uap_delta_pixel_path}")

    print(f"\n=== Done! ===")

if __name__ == "__main__":
    main()