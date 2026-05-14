"""
Rice Disease Detection using MobileNetV2 — Production-Grade Pipeline
=====================================================================
Recommended Dataset (best results in literature):
  Paddy Doctor - https://www.kaggle.com/c/paddy-disease-classification
  10,407 images | 10 classes | validated labels
  kaggle competitions download -c paddy-disease-classification

Second best option:
  https://www.kaggle.com/datasets/trumanrase/rice-leaf-diseases
  11,281 images | 12 classes

WHY YOUR PREVIOUS DATASET FAILED:
  The Mendeley dataset only has ~120 images. MobileNet has 3.4M parameters.
  Training on ~120 images causes severe overfitting → random predictions.
  Use 5,000+ images minimum.

Usage:
  pip install torch torchvision timm albumentations scikit-learn matplotlib seaborn
  python train.py --data_dir ./dataset --epochs 50 --batch_size 32
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib")
)

import argparse
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from PIL import Image
from tqdm.auto import tqdm

# ─────────────────────────── Reproducibility ─────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────── Dataset ─────────────────────────────────────────
class RiceDiseaseDataset(Dataset):
    """
    Expects folder structure:
        data_dir/
            train/
                Bacterial_Blight/  img1.jpg ...
                Blast/             img1.jpg ...
                Brown_Spot/        img1.jpg ...
                Healthy/           img1.jpg ...
                ...
            val/
                ...
            test/
                ...
    """
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        samples=None,
        classes=None,
        class_to_idx=None,
    ):
        self.root = Path(root)
        self.transform = transform

        if samples is not None:
            self.samples = list(samples)
            self.classes = list(classes)
            self.class_to_idx = dict(class_to_idx)
        else:
            self.root = self.root / split
            self.classes = sorted([
                d.name for d in self.root.iterdir() if d.is_dir()
            ])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.samples = []
            for cls in self.classes:
                for img_path in (self.root / cls).glob("*"):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}. "
                               "Check your folder structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_class_weights(self):
        """Compute inverse-frequency weights for balanced sampling."""
        counts = Counter([s[1] for s in self.samples])
        total = len(self.samples)
        weights = [total / counts[s[1]] for s in self.samples]
        return weights

    @classmethod
    def from_samples(cls, root, samples, classes, class_to_idx, transform=None):
        return cls(
            root=root,
            transform=transform,
            samples=samples,
            classes=classes,
            class_to_idx=class_to_idx,
        )


# ─────────────────────────── Augmentation ────────────────────────────────────
IMG_SIZE = 224

def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-20, 20),
            p=0.5,
        ),
        # Colour jitter — critical for field vs lab generalization
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.1, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=25,
            val_shift_limit=20, p=0.5
        ),
        # Disease-area robustness
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.11),
            hole_width_range=(0.05, 0.11),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────── Model ───────────────────────────────────────────
class RiceMobileNetV2(nn.Module):
    """
    MobileNetV2 with:
      - ImageNet pretrained weights (transfer learning)
      - Dropout before classifier (prevents overfitting)
      - Optional label smoothing via loss function
    """
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze early layers, unfreeze later ones
        for name, param in base.features.named_parameters():
            layer_idx = int(name.split(".")[0])
            param.requires_grad = (layer_idx >= 12)  # Fine-tune last 6 blocks

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def unfreeze_all(self):
        """Call after initial training to fine-tune the full network."""
        for param in self.parameters():
            param.requires_grad = True


# ─────────────────────────── Training Loop ───────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=8, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        improved = False
        if self.best_loss is None:
            self.best_loss = val_loss
            improved = True
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            improved = True
        return improved


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_datasets(data_dir, seed=42, val_ratio=0.15, test_ratio=0.15):
    data_path = Path(data_dir)
    split_dirs = all((data_path / split).is_dir() for split in ("train", "val", "test"))

    if split_dirs:
        print("Detected train/val/test directory structure.")
        train_ds = RiceDiseaseDataset(data_dir, "train", get_train_transform())
        val_ds = RiceDiseaseDataset(data_dir, "val", get_val_transform())
        test_ds = RiceDiseaseDataset(data_dir, "test", get_val_transform())
        return train_ds, val_ds, test_ds

    print("Detected raw class-folder dataset. Building stratified train/val/test splits.")
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    all_samples = []
    for cls_name in classes:
        for img_path in (data_path / cls_name).glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                all_samples.append((img_path, class_to_idx[cls_name]))

    if not all_samples:
        raise RuntimeError(f"No class images found under {data_path}")

    indices = np.arange(len(all_samples))
    labels = np.array([label for _, label in all_samples])
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]

    train_ds = RiceDiseaseDataset.from_samples(
        data_path, train_samples, classes, class_to_idx, get_train_transform()
    )
    val_ds = RiceDiseaseDataset.from_samples(
        data_path, val_samples, classes, class_to_idx, get_val_transform()
    )
    test_ds = RiceDiseaseDataset.from_samples(
        data_path, test_samples, classes, class_to_idx, get_val_transform()
    )
    return train_ds, val_ds, test_ds


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    use_cuda_amp = scaler is not None and scaler.is_enabled()
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
    )
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device.type, enabled=use_cuda_amp
        ):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_cuda_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name, epoch=None, total_epochs=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    desc = split_name
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} [{split_name}]"

    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{correct / total:.4f}",
        )
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────── Utilities ───────────────────────────────────────
def save_training_curves(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved training curves → {out_dir / 'training_curves.png'}")


def save_confusion_matrix(labels, preds, class_names, out_dir):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title("Normalized Confusion Matrix (Test Set)")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Saved confusion matrix  → {out_dir / 'confusion_matrix.png'}")


# ─────────────────────────── Main ────────────────────────────────────────────
def main(args):
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    pin_memory = device.type == "cuda"
    print(f"\n{'='*60}")
    print(f"  Rice Disease Detection — MobileNetV2")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data_dir}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = create_datasets(args.data_dir, seed=args.seed)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    # Save class mapping for inference
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Balanced sampling — critical when classes are imbalanced
    sample_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RiceMobileNetV2(num_classes=num_classes, dropout=0.35).to(device)

    # Label smoothing reduces overconfidence → better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    # Cosine annealing: gradually reduces LR for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    early_stop = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    # ── Phase 1: Train classifier only (frozen backbone) ─────────────────────
    print("Phase 1: Training classifier head (backbone partially frozen)...")
    for epoch in range(1, args.epochs + 1):
        # After half the epochs, unfreeze the full network for fine-tuning
        if epoch == args.unfreeze_epoch:
            print(f"\n  ↳ Epoch {epoch}: Unfreezing full network for fine-tuning")
            model.unfreeze_all()
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=1e-7
            )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, device, "val", epoch, args.epochs
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"| Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} "
            f"| Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f} "
            f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
            f"| {elapsed:.1f}s"
        )

        improved = early_stop(vl_loss)
        if improved:
            best_val_loss = vl_loss
            best_val_acc = vl_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "class_names": class_names,
            }, out_dir / "best_model.pth")
            print(
                f"  ✓ Saved best model (val_loss={vl_loss:.4f}, val_acc={vl_acc:.4f})"
            )
        else:
            print(
                f"  Early stopping patience: {early_stop.counter}/{early_stop.patience}"
            )

        if early_stop.should_stop:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(
        f"\nBest checkpoint: epoch {best_epoch} "
        f"| val_loss={best_val_loss:.4f} | val_acc={best_val_acc:.4f}"
    )

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, "test"
    )
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nPer-class Report:")
    print(classification_report(
        test_labels, test_preds, target_names=class_names, digits=4
    ))

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_training_curves(history, out_dir)
    save_confusion_matrix(test_labels, test_preds, class_names, out_dir)

    # Save results summary
    results = {
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "test_accuracy": round(test_acc, 4),
        "num_classes": num_classes,
        "class_names": class_names,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to: {out_dir}/")
    print("  best_model.pth       ← trained weights")
    print("  class_names.json     ← class index mapping")
    print("  training_curves.png  ← loss/accuracy plots")
    print("  confusion_matrix.png ← per-class performance")
    print("  results.json         ← final metrics")


# ─────────────────────────── Entry Point ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobileNetV2 Rice Disease Detector"
    )
    parser.add_argument("--data_dir",      type=str,
                        default="./dataset")
    parser.add_argument("--output_dir",    type=str,   default="./outputs")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--unfreeze_epoch",type=int,   default=15,
                        help="Epoch at which to unfreeze full network")
    args = parser.parse_args()
    main(args)
