#!/usr/bin/env python3
"""Check the first image and annotation in a converted RACE-6D dataset."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
    )
    p.add_argument("--save-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.dataset.resolve()

    ann_path = root / "annotations" / f"instances_{args.split}.json"

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open() as f:
        data = json.load(f)

    if not data["images"]:
        raise ValueError("No images found")

    image_info = data["images"][0]
    image_id = image_info["id"]

    anns = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_id
    ]

    if not anns:
        raise ValueError(f"No annotation found for image_id={image_id}")

    rgb_path = root / "rgb" / image_info["file_name"]
    depth_path = root / "rgb_depth" / Path(image_info["file_name"]).with_suffix(".png").name

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise RuntimeError(f"Failed to read RGB: {rgb_path}")

    if depth is None:
        raise RuntimeError(f"Failed to read depth: {depth_path}")

    print("=" * 70)
    print("RACE-6D DATASET CHECK")
    print("=" * 70)

    print(f"\nDataset      : {root}")
    print(f"Split        : {args.split}")
    print(f"Annotation   : {ann_path}")
    print(f"Image ID     : {image_id}")
    print(f"RGB file     : {rgb_path}")
    print(f"Depth file   : {depth_path}")

    print("\n--- RGB ---")
    print(f"JSON size    : {image_info['width']} x {image_info['height']}")
    print(f"Actual size  : {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"Channels     : {rgb.shape[2] if rgb.ndim == 3 else 1}")
    print(f"Dtype        : {rgb.dtype}")

    print("\n--- Depth ---")
    print(f"Actual size  : {depth.shape[1]} x {depth.shape[0]}")
    print(f"Dtype        : {depth.dtype}")
    print(f"Min          : {depth.min()}")
    print(f"Max          : {depth.max()}")
    print(f"Mean         : {depth.mean():.3f}")
    print(f"Zero pixels  : {np.count_nonzero(depth == 0)}")
    print("Unit         : mm")

    if rgb.shape[:2] != depth.shape[:2]:
        print("\n[ERROR] RGB and Depth sizes do not match!")
    else:
        print("\n[OK] RGB and Depth sizes match.")

    print("\n--- Camera Intrinsic ---")

    for i, ann in enumerate(anns):
        K = np.asarray(ann["cam_K"], dtype=np.float64).reshape(3, 3)

        print(f"\nObject {i + 1}")
        print(f"Category ID  : {ann['category_id']}")
        print("K =")
        print(K)

        print(f"fx           : {K[0, 0]:.6f}")
        print(f"fy           : {K[1, 1]:.6f}")
        print(f"cx           : {K[0, 2]:.6f}")
        print(f"cy           : {K[1, 2]:.6f}")

        expected_w = image_info["width"]
        expected_h = image_info["height"]

        print(f"Image center : ({expected_w / 2:.3f}, {expected_h / 2:.3f})")
        print(
            f"Principal pt : ({K[0, 2]:.3f}, {K[1, 2]:.3f})"
        )

        print("\nBBox")
        bbox = np.asarray(ann["bbox"], dtype=np.float64)
        x, y, w, h = bbox
        print(f"xywh         : {bbox.tolist()}")
        print(f"x1,y1        : ({x:.2f}, {y:.2f})")
        print(f"x2,y2        : ({x + w:.2f}, {y + h:.2f})")
        print(f"center       : ({x + w / 2:.2f}, {y + h / 2:.2f})")
        print(f"area         : {ann['area']}")

        if (
            x < 0
            or y < 0
            or x + w > expected_w
            or y + h > expected_h
        ):
            print("[WARNING] BBox is outside image boundaries!")
        else:
            print("[OK] BBox is inside image.")

        pose = np.asarray(ann["pose"], dtype=np.float64)

        print("\nPose")
        print(f"Translation  : {pose[:3]} mm")
        print("Rotation     :")
        print(pose[3:].reshape(3, 3))

    print("\n--- Expected 640x480 check ---")

    if image_info["width"] == 640 and image_info["height"] == 480:
        print("[OK] Dataset image size is 640 x 480.")
    else:
        print(
            f"[WARNING] Dataset image size is "
            f"{image_info['width']} x {image_info['height']}"
        )

    K = np.asarray(anns[0]["cam_K"], dtype=np.float64).reshape(3, 3)

    expected_K = np.array([
        [359.0618, 0.0, 322.56],
        [0.0, 478.7491, 250.24],
        [0.0, 0.0, 1.0],
    ])

    print("\nExpected K for original 1920x1080 -> 640x480:")
    print(expected_K)

    print("\nActual K:")
    print(K)

    print("\nK difference:")
    print(K - expected_K)

    if np.allclose(K, expected_K, atol=1.0):
        print("[OK] K is consistent with 1920x1080 -> 640x480 resize.")
    else:
        print(
            "[INFO] K differs from the example K. "
            "This may be normal if the source camera has different intrinsics."
        )

    if args.save_dir is not None:
        save_dir = args.save_dir.resolve()
    else:
        save_dir = root / "debug_check"

    save_dir.mkdir(parents=True, exist_ok=True)

    # RGB + bbox
    vis = rgb.copy()

    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])

        cv2.rectangle(
            vis,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        text = f"obj={ann['category_id']}"
        cv2.putText(
            vis,
            text,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    rgb_debug = save_dir / f"{args.split}_first_rgb_bbox.jpg"
    cv2.imwrite(str(rgb_debug), vis)

    # Depth visualization
    depth_float = depth.astype(np.float32)

    if depth_float.max() > 0:
        depth_vis = np.clip(
            depth_float / depth_float.max() * 255.0,
            0,
            255,
        ).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)

    depth_debug = save_dir / f"{args.split}_first_depth.png"
    cv2.imwrite(str(depth_debug), depth_vis)

    print("\n--- Debug images ---")
    print(f"RGB + BBox   : {rgb_debug}")
    print(f"Depth        : {depth_debug}")

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()