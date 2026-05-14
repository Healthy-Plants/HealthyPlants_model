"""
Rice Disease Detection using MobileNetV3-Small
==============================================
Separate training entrypoint for the 4-class rice leaf dataset currently in use.

Usage:
  python train_mobilenetv3_small.py
  python train_mobilenetv3_small.py --epochs 30 --batch_size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from train import (
    EarlyStopping,
    create_datasets,
    evaluate,
    get_device,
    save_confusion_matrix,
    save_training_curves,
    seed_everything,
    train_one_epoch,
)


class RiceMobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small with ImageNet weights and a compact custom classifier head.
    This stays Android-friendly while trimming model capacity versus MobileNetV2.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        for name, param in base.features.named_parameters():
            layer_idx = int(name.split(".")[0])
            param.requires_grad = (layer_idx >= 9)

        in_features = base.classifier[0].in_features
        base.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self.model = base

    def forward(self, x):
        return self.model(x)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def main(args):
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    pin_memory = device.type == "cuda"

    print(f"\n{'=' * 60}")
    print(f"  {args.task_name} — MobileNetV3-Small")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data_dir}")
    print(f"{'=' * 60}\n")

    train_ds, val_ds, test_ds = create_datasets(args.data_dir, seed=args.seed)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    sample_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_ds), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = RiceMobileNetV3Small(
        num_classes=num_classes, dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    early_stop = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    print("Phase 1: Training MobileNetV3-Small classifier head...")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            print(f"\n  ↳ Epoch {epoch}: Unfreezing full network for fine-tuning")
            model.unfreeze_all()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr * args.finetune_lr_scale,
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1, eta_min=1e-7
            )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            args.epochs,
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vl_loss,
                    "val_acc": vl_acc,
                    "class_names": class_names,
                },
                out_dir / "best_model.pth",
            )
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

    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, "test"
    )
    print(f"\nTest Accuracy: {test_acc:.4f}")

    save_training_curves(history, out_dir)
    save_confusion_matrix(test_labels, test_preds, class_names, out_dir)

    results = {
        "task_name": args.task_name,
        "model": "mobilenet_v3_small",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small Rice Disease Detector"
    )
    parser.add_argument(
        "--task_name", type=str, default="Rice Disease Detection"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset_rice_leaf_grouped",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs_mobilenetv3_small"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze_epoch", type=int, default=10)
    parser.add_argument("--finetune_lr_scale", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
