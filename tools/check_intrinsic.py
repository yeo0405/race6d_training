#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read camera intrinsics from RACE-6D COCO annotations."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to instances_train.json",
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=None,
        help="Read intrinsics for a specific image ID.",
    )
    parser.add_argument(
        "--annotation-id",
        type=int,
        default=None,
        help="Read intrinsics for a specific annotation ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json.exists():
        raise FileNotFoundError(args.json)

    with args.json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])

    if not annotations:
        raise ValueError("No annotations found.")

    selected = annotations

    if args.annotation_id is not None:
        selected = [
            ann for ann in annotations
            if ann["id"] == args.annotation_id
        ]

    if args.image_id is not None:
        selected = [
            ann for ann in selected
            if ann["image_id"] == args.image_id
        ]

    if not selected:
        raise ValueError("No matching annotation found.")

    Ks = []

    for ann in selected:
        if "cam_K" not in ann:
            raise ValueError(
                f"Annotation {ann['id']} does not contain cam_K."
            )

        K = np.asarray(
            ann["cam_K"],
            dtype=np.float64,
        ).reshape(3, 3)

        Ks.append(K)

    Ks = np.stack(Ks)

    reference = Ks[0]
    same = np.allclose(
        Ks,
        reference[None, ...],
        rtol=0.0,
        atol=1e-6,
    )

    print("=" * 60)
    print("Camera Intrinsics")
    print("=" * 60)

    print(f"JSON          : {args.json}")
    print(f"Annotations   : {len(selected):,}")
    print(f"Same K        : {same}")

    print("\nK:")
    print(reference)

    print("\nParameters:")
    print(f"fx = {reference[0, 0]:.6f}")
    print(f"fy = {reference[1, 1]:.6f}")
    print(f"cx = {reference[0, 2]:.6f}")
    print(f"cy = {reference[1, 2]:.6f}")

    if not same:
        unique = np.unique(
            Ks.reshape(len(Ks), -1),
            axis=0,
        )

        print(f"\nUnique K matrices: {len(unique)}")

        for i, values in enumerate(unique):
            K = values.reshape(3, 3)
            print(f"\nK #{i + 1}:")
            print(K)

    if args.image_id is not None:
        print(f"\nImage ID: {args.image_id}")

        for ann in selected:
            print(
                f"  annotation={ann['id']}, "
                f"category={ann['category_id']}"
            )


if __name__ == "__main__":
    main()
