#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

import numpy as np


DATASETS = {
    "blenderproc": Path("/home/yeo/Downloads/race_6d/dataset_blender"),
    "foundationpose": Path("/home/yeo/Downloads/race_6d/dataset_foundationpose"),
}

ROUND_DIGITS = 6


def load_k_values(path: Path):
    data = json.loads(path.read_text())
    values = []

    for ann in data.get("annotations", []):
        K = ann.get("cam_K")
        if K is None or len(K) != 9:
            continue

        K = tuple(round(float(v), ROUND_DIGITS) for v in K)
        values.append(K)

    return values


def print_stats(dataset_name: str, root: Path):
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Path   : {root}")
    print("=" * 80)

    all_groups = {}

    for split in ("train", "val"):
        json_path = root / "annotations" / f"instances_{split}.json"

        if not json_path.exists():
            print(f"\n[{split}] file not found: {json_path}")
            continue

        values = load_k_values(json_path)
        counter = Counter(values)
        all_groups[split] = counter

        print(f"\n[{split}]")
        print(f"Annotations : {len(values)}")
        print(f"Unique K    : {len(counter)}")

        if not counter:
            continue

        total = len(values)

        for index, (K, count) in enumerate(counter.most_common(), 1):
            fx, _, cx, _, fy, cy, _, _, _ = K

            print(f"\n  K Group {index}")
            print(f"  Count  : {count} ({count / total * 100:.2f}%)")
            print(f"  fx     : {fx:.6f}")
            print(f"  fy     : {fy:.6f}")
            print(f"  cx     : {cx:.6f}")
            print(f"  cy     : {cy:.6f}")
            print("  Matrix :")
            print(np.array(K).reshape(3, 3))

    return all_groups


def compare_datasets(stats):
    print("\n" + "=" * 80)
    print("Cross Dataset Camera Intrinsic Comparison")
    print("=" * 80)

    dataset_names = list(stats.keys())

    all_keys = set()

    for dataset_name in dataset_names:
        for split in ("train", "val"):
            all_keys.update(stats[dataset_name].get(split, {}).keys())

    print(f"\nTotal unique K across datasets: {len(all_keys)}")

    for index, K in enumerate(sorted(all_keys), 1):
        fx, _, cx, _, fy, cy, _, _, _ = K

        print(f"\nK Group {index}")
        print(f"fx = {fx:.6f}")
        print(f"fy = {fy:.6f}")
        print(f"cx = {cx:.6f}")
        print(f"cy = {cy:.6f}")

        for dataset_name in dataset_names:
            train_count = stats[dataset_name].get("train", {}).get(K, 0)
            val_count = stats[dataset_name].get("val", {}).get(K, 0)
            total = train_count + val_count

            print(
                f"  {dataset_name}: "
                f"train={train_count}, "
                f"val={val_count}, "
                f"total={total}"
            )


def main():
    stats = {}

    for dataset_name, root in DATASETS.items():
        stats[dataset_name] = print_stats(dataset_name, root)

    compare_datasets(stats)


if __name__ == "__main__":
    main()