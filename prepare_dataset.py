"""
Dataset Preparation Script
============================
Downloads the recommended dataset and structures it for training.

Recommended dataset: Paddy Doctor (Kaggle Competition)
  kaggle competitions download -c paddy-disease-classification

Steps this script does:
  1. Reads the raw dataset (train_images folder + train.csv from Paddy Doctor)
  2. Splits into train/val/test (70/15/15)
  3. Saves to ./dataset/train, ./dataset/val, ./dataset/test
  4. Prints a class distribution report

Usage:
  pip install kaggle pandas scikit-learn
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_api_key
  kaggle competitions download -c paddy-disease-classification
  unzip paddy-disease-classification.zip -d ./raw_paddy/
  python prepare_dataset.py --raw_dir ./raw_paddy --out_dir ./dataset
"""

import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter


def prepare_paddy_doctor(raw_dir: Path, out_dir: Path, seed: int = 42):
    """
    Paddy Doctor structure:
      raw_dir/
        train_images/
          bacterial_leaf_blight/ img.jpg ...
          blast/
          ...
        train.csv  (columns: image_id, label)
    """
    csv_path = raw_dir / "train.csv"
    img_root = raw_dir / "train_images"

    if csv_path.exists():
        print("Using train.csv for labels...")
        df = pd.read_csv(csv_path)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        id_col    = "image_id" if "image_id" in df.columns else df.columns[0]
        label_col = "label"    if "label"    in df.columns else df.columns[1]

        # Find actual image paths
        rows = []
        for _, row in df.iterrows():
            img_id = row[id_col]
            label  = row[label_col]
            # Search inside label subfolder first, then root
            candidate = img_root / label / img_id
            if not candidate.exists():
                candidate = img_root / img_id
            if candidate.exists():
                rows.append({"path": candidate, "label": label})
        df_clean = pd.DataFrame(rows)

    else:
        print("No CSV found — reading folder structure directly...")
        rows = []
        for label_dir in img_root.iterdir():
            if label_dir.is_dir():
                for img in label_dir.glob("*"):
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        rows.append({"path": img, "label": label_dir.name})
        df_clean = pd.DataFrame(rows)

    print(f"\nTotal images found: {len(df_clean)}")
    print("Class distribution:")
    counts = Counter(df_clean["label"])
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<35} {cnt:5d}")

    # Split
    train_df, temp_df = train_test_split(
        df_clean, test_size=0.30, stratify=df_clean["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,  test_size=0.50, stratify=temp_df["label"],  random_state=seed
    )
    print(f"\nSplit: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Copy files
    for split_name, split_df in [("train", train_df),
                                  ("val",   val_df),
                                  ("test",  test_df)]:
        for _, row in split_df.iterrows():
            dest = out_dir / split_name / row["label"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["path"], dest / Path(row["path"]).name)

    print(f"\nDataset written to: {out_dir}/")
    print("  train/  val/  test/")
    print("\nYou can now run:")
    print(f"  python train.py --data_dir {out_dir}")


def prepare_generic(raw_dir: Path, out_dir: Path, seed: int = 42):
    """
    Generic handler: expects raw_dir to already have class subfolders.
      raw_dir/
        Bacterial_Blight/  img.jpg ...
        Blast/
        ...
    """
    rows = []
    for label_dir in raw_dir.iterdir():
        if label_dir.is_dir():
            for img in label_dir.glob("*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    rows.append({"path": img, "label": label_dir.name})

    if not rows:
        print(f"ERROR: No images found in {raw_dir}")
        return

    df = pd.DataFrame(rows)
    print(f"Total images: {len(df)}")
    print("Classes:", df["label"].unique().tolist())

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )

    for split_name, split_df in [("train", train_df),
                                  ("val",   val_df),
                                  ("test",  test_df)]:
        for _, row in split_df.iterrows():
            dest = out_dir / split_name / row["label"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["path"], dest / Path(row["path"]).name)

    print(f"\nDataset ready at: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare rice disease dataset")
    parser.add_argument("--raw_dir", required=True,
                        help="Raw downloaded dataset folder")
    parser.add_argument("--out_dir", default="./dataset",
                        help="Output directory for train/val/test splits")
    parser.add_argument("--format",  default="paddy",
                        choices=["paddy", "generic"],
                        help="'paddy' = Paddy Doctor format; "
                             "'generic' = class subfolders")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.format == "paddy":
        prepare_paddy_doctor(raw_dir, out_dir, args.seed)
    else:
        prepare_generic(raw_dir, out_dir, args.seed)