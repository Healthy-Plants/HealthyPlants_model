"""
Rice Disease Inference Script
==============================
Run a single image or a whole folder through the trained model.

Usage:
  # Single image
  python predict.py --model outputs/best_model.pth --image leaf.jpg

  # Folder of images
  python predict.py --model outputs/best_model.pth --folder ./test_images/

  # With Grad-CAM visualization
  python predict.py --model outputs/best_model.pth --image leaf.jpg --gradcam
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib")
)

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ── Model (must match train.py or train_mobilenetv3_small.py) ──────────────────
class RiceMobileNetV2(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.35):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
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


class RiceMobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        # Use weights=None since we are loading our own checkpoint
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


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        # Determine target layer based on architecture
        if hasattr(model, "features"): # MobileNetV2 structure
            target_layer = model.features[-1]
        elif hasattr(model, "model") and hasattr(model.model, "features"): # RiceMobileNetV3Small structure
            target_layer = model.model.features[-1]
        else:
            raise AttributeError("Model architecture not recognized for Grad-CAM target layer")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_path: str):
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    tensor = transform(image=img_np)["image"].unsqueeze(0)
    return tensor, img_pil


# ── Single prediction ─────────────────────────────────────────────────────────
def predict(model, img_tensor, class_names, device, top_k=3):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
    probs  = torch.softmax(logits, dim=1)[0].cpu()
    topk   = probs.topk(min(top_k, len(class_names)))
    results = []
    for prob, idx in zip(topk.values, topk.indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })
    return results


# ── Visualize with Grad-CAM ───────────────────────────────────────────────────
def visualize_gradcam(model, img_tensor, img_pil, predictions, save_path=None):
    cam_gen = GradCAM(model)
    img_tensor_grad = img_tensor.clone().requires_grad_(True)
    cam = cam_gen.generate(img_tensor_grad)

    orig = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cm.jet(cam)[:, :, :3]
    overlay = (0.55 * orig / 255 + 0.45 * heatmap).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig);        axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(heatmap);     axes[1].set_title("Grad-CAM");      axes[1].axis("off")
    axes[2].imshow(overlay);     axes[2].set_title("Overlay");       axes[2].axis("off")

    title = "\n".join([
        f"#{i+1} {r['class']} — {r['confidence']:.1f}%"
        for i, r in enumerate(predictions[:3])
    ])
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    class_names_path = Path(args.model).parent / "class_names.json"
    with open(class_names_path) as f:
        class_names = json.load(f)

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    num_classes = len(class_names)

    # Simple heuristic: MobileNetV3 small uses 'model.model.features.0.0.weight'
    # MobileNetV2 uses 'features.0.0.weight'
    is_v3 = any(k.startswith("model.features") for k in checkpoint["model_state_dict"].keys())

    if is_v3:
        print("Detected MobileNetV3-Small architecture")
        model = RiceMobileNetV3Small(num_classes=num_classes).to(device)
    else:
        print("Detected MobileNetV2 architecture")
        model = RiceMobileNetV2(num_classes=num_classes).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded | {len(class_names)} classes: {class_names}\n")

    # Single image
    if args.image:
        img_tensor, img_pil = preprocess(args.image)
        results = predict(model, img_tensor, class_names, device)
        print(f"Predictions for: {args.image}")
        for r in results:
            bar = "█" * int(r["confidence"] / 5)
            print(f"  {r['class']:<25} {r['confidence']:6.2f}% {bar}")

        if args.gradcam:
            save_path = Path(args.image).stem + "_gradcam.png"
            visualize_gradcam(model, img_tensor, img_pil, results, save_path)

    # Folder
    if args.folder:
        folder = Path(args.folder)
        image_files = list(folder.glob("**/*.jpg")) + \
                      list(folder.glob("**/*.jpeg")) + \
                      list(folder.glob("**/*.png"))
        print(f"Found {len(image_files)} images in {folder}\n")

        all_results = {}
        for img_path in sorted(image_files):
            img_tensor, img_pil = preprocess(str(img_path))
            preds = predict(model, img_tensor, class_names, device, top_k=1)
            label = preds[0]["class"]
            conf  = preds[0]["confidence"]
            all_results[img_path.name] = {"prediction": label, "confidence": conf}
            print(f"  {img_path.name:<40} → {label} ({conf:.1f}%)")

            if args.gradcam:
                save_path = img_path.parent / (img_path.stem + "_gradcam.png")
                visualize_gradcam(model, img_tensor, img_pil, preds, save_path)

        # Save results JSON
        out_json = folder / "predictions.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rice Disease Predictor")
    parser.add_argument("--model",   required=True, help="Path to best_model.pth")
    parser.add_argument("--image",   default=None,  help="Single image path")
    parser.add_argument("--folder",  default=None,  help="Folder of images")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate Grad-CAM visualizations")
    args = parser.parse_args()
    main(args)
