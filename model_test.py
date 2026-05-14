"""
Export test-set predictions and class probabilities for one trained classifier.

Example:
  python model_test.py \
    --model_dir ./outputs_hierarchical/disease_classifiers/corn \
    --data_dir ./hierarchical_data/disease_classifiers/corn
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import create_datasets, get_device
from train_mobilenetv3_small import RiceMobileNetV3Small


def export_predictions(model_dir: Path, data_dir: Path, out_csv: Path, batch_size: int):
    device = get_device()
    _, _, test_ds = create_datasets(str(data_dir), seed=42)

    with (model_dir / "class_names.json").open() as f:
        class_names = json.load(f)

    model = RiceMobileNetV3Small(num_classes=len(class_names), dropout=0.3).to(device)
    checkpoint = torch.load(model_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    rows = []
    sample_paths = [str(path) for path, _ in test_ds.samples]
    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1).cpu()
            predictions = probabilities.argmax(dim=1)

            batch_size_actual = len(labels)
            batch_paths = sample_paths[offset : offset + batch_size_actual]
            offset += batch_size_actual

            for i in range(batch_size_actual):
                row = {
                    "image_path": batch_paths[i],
                    "image_name": Path(batch_paths[i]).name,
                    "actual_label": class_names[labels[i].item()],
                    "predicted_label": class_names[predictions[i].item()],
                }
                for class_index, class_name in enumerate(class_names):
                    row[class_name] = round(float(probabilities[i, class_index].item()), 6)
                rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "image_name", "actual_label", "predicted_label"]
    fieldnames.extend(class_names)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Export model predictions and probabilities for a test split"
    )
    parser.add_argument(
        "--model_dir",
        default="./outputs_hierarchical/disease_classifiers/corn",
        help="Directory containing best_model.pth and class_names.json",
    )
    parser.add_argument(
        "--data_dir",
        default="./hierarchical_data/disease_classifiers/corn",
        help="Dataset directory containing train/val/test folders",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path. Defaults to <model_dir>/test_predictions_with_probs.csv",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv) if args.out_csv else model_dir / "test_predictions_with_probs.csv"
    export_predictions(model_dir, data_dir, out_csv, args.batch_size)


if __name__ == "__main__":
    main()
