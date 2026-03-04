import copy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import timm
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------
# SETTINGS
# -------------------
DATA_DIR = Path("dataset")  # keep this as your current dataset
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

MODEL_NAME = "efficientnet_b0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
BEST_PATH = OUT_DIR / "best_model.pth"

# -------------------
# AUGMENTATIONS (anti-cheat)
# -------------------
train_aug = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(height=224, width=224),

    A.ImageCompression(quality_lower=40, quality_upper=95, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Downscale(scale_min=0.6, scale_max=0.9, p=0.3),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),

    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.7),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# -------------------
# DATASET WRAPPER
# -------------------
class AlbDataset(Dataset):
    def __init__(self, imagefolder_ds, aug):
        self.samples = imagefolder_ds.samples
        self.classes = imagefolder_ds.classes
        self.class_to_idx = imagefolder_ds.class_to_idx
        self.aug = aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.aug(image=img)["image"]
        return x, label

base_train = datasets.ImageFolder(TRAIN_DIR)
base_val = datasets.ImageFolder(VAL_DIR)

train_ds = AlbDataset(base_train, train_aug)
val_ds = AlbDataset(base_val, val_aug)

print("✅ Classes:", base_train.classes)  # should be ['ai', 'real']

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------
# MODEL
# -------------------
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def run_epoch(loader, training: bool):
    model.train() if training else model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="TRAIN" if training else "VAL")
    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=total_loss / total, acc=total_correct / total)

    return total_loss / total, total_correct / total

best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())

print("✅ Device:", DEVICE)

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    va_loss, va_acc = run_epoch(val_loader, training=False)

    print(f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
    print(f"Val:   loss={va_loss:.4f}, acc={va_acc:.4f}")

    if va_acc > best_acc:
        best_acc = va_acc
        best_wts = copy.deepcopy(model.state_dict())
        torch.save(best_wts, BEST_PATH)
        print(f"✅ Saved best model to: {BEST_PATH} (val_acc={best_acc:.4f})")

print(f"\nDone ✅ Best val acc: {best_acc:.4f}")
