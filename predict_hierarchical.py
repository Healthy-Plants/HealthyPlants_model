"""
Hierarchical prediction: crop classifier first, then crop-specific disease classifier.
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse
import json
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import models


IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class MobileNetV3SmallClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
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


def preprocess(image_path: str):
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )
    return transform(image=img_np)["image"].unsqueeze(0), img_pil


def load_model(model_path: Path, device):
    class_names_path = model_path.parent / "class_names.json"
    with class_names_path.open() as f:
        class_names = json.load(f)
    checkpoint = torch.load(model_path, map_location=device)
    model = MobileNetV3SmallClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_topk(model, img_tensor, class_names, device, top_k=3):
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu()
    topk = probs.topk(min(top_k, len(class_names)))
    return [
        {
            "class": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2),
        }
        for prob, idx in zip(topk.values, topk.indices)
    ]


def resolve_disease_model_dir(root: Path, crop_label: str):
    candidates = [root / crop_label, root / crop_label.lower(), root / crop_label.upper()]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for candidate in root.iterdir():
        if candidate.is_dir() and candidate.name.lower() == crop_label.lower():
            return candidate
    return None


def predict_one(image_path: Path, crop_model, crop_classes, disease_root: Path, device):
    img_tensor, _ = preprocess(str(image_path))
    crop_predictions = predict_topk(crop_model, img_tensor, crop_classes, device, top_k=3)
    crop_label = crop_predictions[0]["class"]

    disease_dir = resolve_disease_model_dir(disease_root, crop_label)
    if disease_dir is None:
        return {
            "image": str(image_path),
            "crop_predictions": crop_predictions,
            "disease_predictions": [],
            "warning": f"No disease model found for crop '{crop_label}'",
        }

    disease_model_path = disease_dir / "best_model.pth"
    if not disease_model_path.exists():
        return {
            "image": str(image_path),
            "crop_predictions": crop_predictions,
            "disease_predictions": [],
            "warning": f"No best_model.pth found for crop '{crop_label}' in {disease_dir}",
        }

    disease_model, disease_classes = load_model(disease_model_path, device)
    disease_predictions = predict_topk(
        disease_model, img_tensor, disease_classes, device, top_k=3
    )
    return {
        "image": str(image_path),
        "crop_predictions": crop_predictions,
        "disease_predictions": disease_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical crop + disease predictor")
    parser.add_argument(
        "--crop_model",
        type=str,
        default="./outputs_hierarchical/crop_classifier/best_model.pth",
    )
    parser.add_argument(
        "--disease_models_root",
        type=str,
        default="./outputs_hierarchical/disease_classifiers",
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--folder", default=None)
    args = parser.parse_args()

    if not args.image and not args.folder:
        raise SystemExit("Provide either --image or --folder")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    crop_model, crop_classes = load_model(Path(args.crop_model), device)
    disease_root = Path(args.disease_models_root)

    images = []
    if args.image:
        images.append(Path(args.image))
    if args.folder:
        folder = Path(args.folder)
        images.extend(sorted(folder.glob("**/*.jpg")))
        images.extend(sorted(folder.glob("**/*.jpeg")))
        images.extend(sorted(folder.glob("**/*.png")))

    all_results = []
    for image_path in images:
        result = predict_one(image_path, crop_model, crop_classes, disease_root, device)
        all_results.append(result)
        print(f"\nImage: {image_path}")
        print("  Crop:")
        for pred in result["crop_predictions"]:
            print(f"    {pred['class']:<20} {pred['confidence']:6.2f}%")
        if result.get("warning"):
            print(f"  Warning: {result['warning']}")
            continue
        print("  Disease:")
        for pred in result["disease_predictions"]:
            print(f"    {pred['class']:<20} {pred['confidence']:6.2f}%")

    if args.folder:
        out_json = Path(args.folder) / "hierarchical_predictions.json"
        with out_json.open("w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {out_json}")


if __name__ == "__main__":
    main()
