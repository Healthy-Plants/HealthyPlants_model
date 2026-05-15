"""
Prepare a leakage-resistant train/val/test split for the Rice Leaf Disease Images dataset.

This script is designed for flat class-folder datasets where filename patterns are not
reliable. It groups visually similar images within each class using perceptual hashes,
then assigns whole groups to a single split so near-duplicates do not leak across
train/val/test.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def average_hash(img: Image.Image, hash_size: int = 8) -> int:
    img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    bits = arr >= arr.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def difference_hash(img: Image.Image, hash_size: int = 8) -> int:
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    bits = arr[:, 1:] >= arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_hashes(path: Path):
    if Image is None or np is None:
        return None, None
    with Image.open(path) as img:
        ahash = average_hash(img)
        dhash = difference_hash(img)
    return ahash, dhash


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def collect_class_images(class_dir: Path):
    return sorted([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def build_groups(paths, distance_threshold):
    hashes = []
    md5_buckets = defaultdict(list)
    has_perceptual_hash = Image is not None and np is not None

    for idx, path in enumerate(paths):
        ahash, dhash = compute_hashes(path)
        hashes.append((ahash, dhash))
        md5_buckets[file_md5(path)].append(idx)

    dsu = DisjointSet(len(paths))

    # Exact duplicate union by file bytes first.
    for indices in md5_buckets.values():
        if len(indices) > 1:
            head = indices[0]
            for idx in indices[1:]:
                dsu.union(head, idx)

    # Near-duplicate union by perceptual hash distance.
    if has_perceptual_hash:
        for i in range(len(paths)):
            a1, d1 = hashes[i]
            for j in range(i + 1, len(paths)):
                a2, d2 = hashes[j]
                if (
                    hamming_distance(a1, a2) <= distance_threshold
                    and hamming_distance(d1, d2) <= distance_threshold
                ):
                    dsu.union(i, j)

    groups = defaultdict(list)
    for idx, path in enumerate(paths):
        groups[dsu.find(idx)].append(path)

    return list(groups.values())


def assign_groups(groups, train_ratio, val_ratio, test_ratio):
    total = sum(len(group) for group in groups)
    target_counts = {
        "train": train_ratio * total,
        "val": val_ratio * total,
        "test": test_ratio * total,
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    assignments = {"train": [], "val": [], "test": []}

    for group in sorted(groups, key=len, reverse=True):
        split = min(
            split_counts,
            key=lambda name: (split_counts[name] / max(target_counts[name], 1), split_counts[name]),
        )
        assignments[split].append(group)
        split_counts[split] += len(group)

    return assignments


def reset_output_dir(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def materialize_split(assignments, out_dir: Path, class_name: str, link_mode: str):
    split_counts = {}
    for split_name, groups in assignments.items():
        split_dir = out_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for group_idx, group in enumerate(groups):
            for img_idx, src in enumerate(group):
                dest = split_dir / f"{group_idx:04d}_{img_idx:04d}_{src.name}"
                if link_mode == "hardlink":
                    try:
                        os.link(src, dest)
                    except OSError:
                        shutil.copy2(src, dest)
                else:
                    shutil.copy2(src, dest)
                count += 1
        split_counts[split_name] = count
    return split_counts


def prepare_grouped_split(
    data_dir: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    distance_threshold: int,
    link_mode: str,
):
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {data_dir}")

    reset_output_dir(out_dir)
    manifest = {
        "source_dir": str(data_dir),
        "grouping_mode": "perceptual_hash+md5" if Image is not None and np is not None else "md5_only",
        "distance_threshold": distance_threshold,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "classes": {},
    }

    for class_dir in class_dirs:
        image_paths = collect_class_images(class_dir)
        groups = build_groups(image_paths, distance_threshold=distance_threshold)
        assignments = assign_groups(groups, train_ratio, val_ratio, test_ratio)
        split_counts = materialize_split(assignments, out_dir, class_dir.name, link_mode)

        manifest["classes"][class_dir.name] = {
            "total_images": len(image_paths),
            "num_groups": len(groups),
            "largest_group": max(len(group) for group in groups),
            "split_counts": split_counts,
        }

        print(
            f"{class_dir.name:<18} images={len(image_paths):4d} "
            f"groups={len(groups):4d} largest_group={max(len(group) for group in groups):3d} "
            f"train={split_counts['train']:4d} val={split_counts['val']:4d} test={split_counts['test']:4d}"
        )

    with (out_dir / "split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGrouped split written to: {out_dir}")
    print("Manifest saved to split_manifest.json")
    if Image is None or np is None:
        print("Note: numpy/Pillow not available; used exact-duplicate grouping only.")


def main():
    parser = argparse.ArgumentParser(
        description="Create grouped train/val/test splits for Rice Leaf Disease Images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./raw_data/rice_leaf_disease_images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./dataset_rice_leaf_grouped",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--distance_threshold",
        type=int,
        default=6,
        help="Max Hamming distance for both aHash and dHash to group near-duplicates",
    )
    parser.add_argument(
        "--link_mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Use hardlinks when possible to avoid duplicating image bytes",
    )
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    prepare_grouped_split(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        distance_threshold=args.distance_threshold,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()
