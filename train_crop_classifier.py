"""
Train the stage-1 crop classifier for the hierarchical pipeline.
"""

import argparse
from pathlib import Path

from train import create_datasets
from train_mobilenetv3_small import main as train_mobilenetv3_small_main


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small crop classifier"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./hierarchical_data/crop_classifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_hierarchical/crop_classifier",
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

    train_ds, _, _ = create_datasets(args.data_dir, seed=args.seed)
    if len(train_ds.classes) < 2:
        raise SystemExit(
            "Crop classifier needs at least 2 crop classes. "
            "Add another crop dataset to /hierarchical_data/crop_classifier first."
        )

    args.task_name = "Crop Classification"
    train_mobilenetv3_small_main(args)


if __name__ == "__main__":
    main()
