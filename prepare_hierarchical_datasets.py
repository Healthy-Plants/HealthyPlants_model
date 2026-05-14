"""
Prepare a two-stage hierarchical dataset layout for crop -> disease classification.

Expected raw layout when using --raw_root:
  raw_root/
    rice/
      Blast/
      Brownspot/
      ...
    cotton/
      Bacterial_Blight/
      Healthy/
      ...

You can also pass repeated --crop mappings like:
  --crop rice=/path/to/rice_dataset --crop cotton=/path/to/cotton_dataset

Outputs:
  hierarchical_data/
    crop_classifier/
      train/<crop>/*.jpg
      val/<crop>/*.jpg
      test/<crop>/*.jpg
    disease_classifiers/
      <crop>/
        train/<disease>/*.jpg
        val/<disease>/*.jpg
        test/<disease>/*.jpg
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from prepare_rice_leaf_grouped_split import prepare_grouped_split


DEFAULT_DATASET_ROOT = "./dataset"
DEFAULT_RICE_PATH = "./raw_data/rice_leaf_disease_images"


def canonicalize_crop_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    known_crops = ["rice", "cotton", "wheat", "maize", "corn", "tomato", "potato"]
    for crop in known_crops:
        if crop in normalized:
            return crop

    return normalized.replace(" dataset", "").replace(" images", "").replace(" ", "_")


def parse_crop_mapping(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Crop mapping must look like crop_name=/path/to/dataset"
        )
    crop_name, crop_path = value.split("=", 1)
    crop_name = crop_name.strip()
    crop_path = crop_path.strip()
    if not crop_name or not crop_path:
        raise argparse.ArgumentTypeError(
            "Crop mapping must include both crop name and dataset path"
        )
    return crop_name, Path(crop_path)


def discover_from_raw_root(raw_root: Path):
    crop_sources = {}
    for crop_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        disease_dirs = [p for p in crop_dir.iterdir() if p.is_dir()]
        if disease_dirs:
            crop_name = canonicalize_crop_name(crop_dir.name)
            crop_sources[crop_name] = crop_dir
    return crop_sources


def link_or_copy(src: Path, dest: Path, link_mode: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(src, dest)
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


def rebuild_crop_classifier_dataset(disease_root: Path, crop_classifier_root: Path, link_mode: str):
    if crop_classifier_root.exists():
        shutil.rmtree(crop_classifier_root)
    crop_classifier_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for crop_dir in sorted([p for p in disease_root.iterdir() if p.is_dir()]):
        crop_name = crop_dir.name
        summary[crop_name] = {}
        for split_dir in sorted([p for p in crop_dir.iterdir() if p.is_dir()]):
            split_name = split_dir.name
            count = 0
            for disease_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                for src in sorted([p for p in disease_dir.iterdir() if p.is_file()]):
                    dest = (
                        crop_classifier_root
                        / split_name
                        / crop_name
                        / f"{disease_dir.name}__{src.name}"
                    )
                    link_or_copy(src, dest, link_mode)
                    count += 1
            summary[crop_name][split_name] = count
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hierarchical crop/disease datasets"
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing crop subfolders, each with disease subfolders",
    )
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        help="Explicit crop mapping like rice=/path/to/dataset. Can be repeated.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./hierarchical_data",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--distance_threshold", type=int, default=6)
    parser.add_argument(
        "--link_mode",
        choices=["hardlink", "copy"],
        default="hardlink",
    )
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    crop_sources = {}
    if args.raw_root:
        crop_sources.update(discover_from_raw_root(Path(args.raw_root)))
    for item in args.crop:
        crop_name, crop_path = parse_crop_mapping(item)
        crop_sources[crop_name] = crop_path

    if not crop_sources:
        default_rice = Path(DEFAULT_RICE_PATH)
        if default_rice.is_dir():
            crop_sources["rice"] = default_rice
        else:
            raise RuntimeError(
                "No crop sources found. Pass --raw_root or one or more --crop mappings."
            )

    out_dir = Path(args.out_dir)
    disease_root = out_dir / "disease_classifiers"
    disease_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "crop_sources": {crop: str(path) for crop, path in crop_sources.items()},
        "distance_threshold": args.distance_threshold,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "disease_classifiers": {},
    }

    for crop_name, crop_path in sorted(crop_sources.items()):
        crop_out_dir = disease_root / crop_name
        print(f"\nPreparing disease splits for crop '{crop_name}' from {crop_path}")
        prepare_grouped_split(
            data_dir=Path(crop_path),
            out_dir=crop_out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            distance_threshold=args.distance_threshold,
            link_mode=args.link_mode,
        )

        split_manifest_path = crop_out_dir / "split_manifest.json"
        if split_manifest_path.exists():
            with split_manifest_path.open() as f:
                manifest["disease_classifiers"][crop_name] = json.load(f)

    crop_classifier_root = out_dir / "crop_classifier"
    crop_summary = rebuild_crop_classifier_dataset(
        disease_root, crop_classifier_root, args.link_mode
    )
    manifest["crop_classifier"] = crop_summary

    with (out_dir / "hierarchical_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nHierarchical datasets written to: {out_dir}")
    print(f"  Crop classifier data   → {crop_classifier_root}")
    print(f"  Disease classifier data → {disease_root}")
    if len(crop_sources) < 2:
        print(
            "  Note: only one crop is currently available, so the crop classifier "
            "stage is scaffolded but not meaningful to train yet."
        )


if __name__ == "__main__":
    main()
