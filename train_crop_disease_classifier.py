"""
Train the stage-2 disease classifier for a specific crop in the hierarchical pipeline.
"""

import argparse
from pathlib import Path

from train_mobilenetv3_small import main as train_mobilenetv3_small_main


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small crop-specific disease classifier"
    )
    parser.add_argument("--crop", type=str, default="rice")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./hierarchical_data/disease_classifiers",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs_hierarchical/disease_classifiers",
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

    crop_name = args.crop.strip()
    data_dir = Path(args.data_root) / crop_name
    if not data_dir.exists():
        raise SystemExit(f"Disease dataset not found for crop '{crop_name}': {data_dir}")

    args.data_dir = str(data_dir)
    args.output_dir = str(Path(args.output_root) / crop_name)
    args.task_name = f"{crop_name.title()} Disease Classification"
    del args.crop
    del args.data_root
    del args.output_root

    train_mobilenetv3_small_main(args)


if __name__ == "__main__":
    main()
