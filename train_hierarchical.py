"""
Train the full hierarchical pipeline:
1. crop classifier
2. one disease classifier per crop
"""

import argparse
from argparse import Namespace
from pathlib import Path

from train import create_datasets
from train_mobilenetv3_small import main as train_mobilenetv3_small_main


def build_training_args(
    task_name: str,
    data_dir: Path,
    output_dir: Path,
    args,
):
    return Namespace(
        task_name=task_name,
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        patience=args.patience,
        seed=args.seed,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        unfreeze_epoch=args.unfreeze_epoch,
        finetune_lr_scale=args.finetune_lr_scale,
    )


def discover_crops(data_root: Path):
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])


def train_crop_classifier(args, crop_classifier_dir: Path, output_root: Path):
    train_ds, _, _ = create_datasets(crop_classifier_dir, seed=args.seed)
    if len(train_ds.classes) < 2:
        raise SystemExit(
            "Crop classifier needs at least 2 crop classes in the crop_classifier dataset."
        )

    train_args = build_training_args(
        task_name="Crop Classification",
        data_dir=crop_classifier_dir,
        output_dir=output_root / "crop_classifier",
        args=args,
    )
    train_mobilenetv3_small_main(train_args)


def train_disease_classifiers(args, disease_root: Path, output_root: Path, crops):
    for crop in crops:
        data_dir = disease_root / crop
        if not data_dir.is_dir():
            raise SystemExit(f"Disease dataset not found for crop '{crop}': {data_dir}")

        train_args = build_training_args(
            task_name=f"{crop.title()} Disease Classification",
            data_dir=data_dir,
            output_dir=output_root / "disease_classifiers" / crop,
            args=args,
        )
        train_mobilenetv3_small_main(train_args)


def main():
    parser = argparse.ArgumentParser(
        description="Train crop classifier + crop-specific disease classifiers"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./hierarchical_data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs_hierarchical",
    )
    parser.add_argument(
        "--crops",
        nargs="*",
        default=None,
        help="Optional list of crops to train disease models for. Defaults to all crops found.",
    )
    parser.add_argument("--skip_crop_classifier", action="store_true")
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

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    crop_classifier_dir = data_root / "crop_classifier"
    disease_root = data_root / "disease_classifiers"

    if not crop_classifier_dir.is_dir():
        raise SystemExit(f"Crop classifier dataset not found: {crop_classifier_dir}")
    if not disease_root.is_dir():
        raise SystemExit(f"Disease classifier dataset root not found: {disease_root}")

    crops = args.crops or discover_crops(disease_root)
    if not crops:
        raise SystemExit(f"No crop disease datasets found in {disease_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_crop_classifier:
        train_crop_classifier(args, crop_classifier_dir, output_root)

    train_disease_classifiers(args, disease_root, output_root, crops)


if __name__ == "__main__":
    main()
