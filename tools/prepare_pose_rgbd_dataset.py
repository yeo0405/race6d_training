#!/usr/bin/env python3
"""Convert POSE RGB-D dataset to a resized RACE-6D COCO/BOP-style dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_ply_vertices(path: Path):
    lines = []
    with path.open("r", encoding="ascii") as f:
        for line in f:
            lines.append(line.rstrip())
            if line.rstrip() == "end_header":
                break
        body = f.read()

    if "format ascii 1.0" not in lines:
        raise ValueError(f"Only ASCII PLY is supported: {path}")

    count = int(next(x.split()[2] for x in lines if x.startswith("element vertex ")))

    names = []
    in_vertex = False
    for line in lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
        elif in_vertex and line.startswith("property "):
            names.append(line.split()[-1])

    raw = np.loadtxt(body.splitlines()[:count], usecols=range(len(names)))
    raw = np.atleast_2d(raw)

    vertices = np.zeros(count, dtype=[(name, "<f8") for name in names])
    for i, name in enumerate(names):
        vertices[name] = raw[:, i]

    return None, vertices, vertices.dtype


def write_binary_ply(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = vertices.dtype.names or ()

    with path.open("wb") as f:
        f.write((
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            + "".join(f"property double {name}\n" for name in names)
            + "end_header\n"
        ).encode("ascii"))
        f.write(vertices.astype([(name, "<f8") for name in names]).tobytes())


def cuboid_keypoints(xyz: np.ndarray) -> tuple[list, list]:
    lo, hi = xyz.min(0), xyz.max(0)
    x0, y0, z0 = lo.tolist()
    x1, y1, z1 = hi.tolist()

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    points = list(corners)
    masks = []

    for a, b in edges:
        i = len(points)
        points += [
            (2 * corners[a] + corners[b]) / 3,
            (corners[a] + 2 * corners[b]) / 3,
        ]
        masks.append([a, i, i + 1, b])

    return np.asarray(points).tolist(), masks


def prepare_models(source: Path, output: Path, classes: dict[int, dict]) -> None:
    info, keypoints, edges = {}, {}, {}

    for class_id, cls in classes.items():
        ply = source / "objects" / cls["name"] / "points.ply"
        _, vertices, _ = read_ply_vertices(ply)

        scaled = vertices.copy()
        for axis in "xyz":
            scaled[axis] = scaled[axis].astype(np.float64) * 1000.0

        write_binary_ply(
            output / "models" / f"obj_{class_id:06d}.ply",
            scaled,
        )

        xyz = np.column_stack([scaled[a] for a in "xyz"]).astype(float)
        lo, hi = xyz.min(0), xyz.max(0)
        size = hi - lo

        info[str(class_id)] = {
            "diameter": float(np.linalg.norm(size)),
            "min_x": float(lo[0]),
            "min_y": float(lo[1]),
            "min_z": float(lo[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
        }

        keypoints[str(class_id)], edges[str(class_id)] = cuboid_keypoints(xyz)

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    (models_dir / "models_info.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )

    cache = {
        "metadata": {
            "units": "millimetres",
            "num_keypoints_per_class": 32,
        },
        "keypoints_3d": keypoints,
        "edges_mask": edges,
    }

    (output / "cached_keypoints_3d_edge.json").write_text(
        json.dumps(cache, indent=2) + "\n"
    )


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="POSE dataset root")
    p.add_argument("--output", type=Path, required=True, help="RACE-6D dataset root")
    p.add_argument("--category-yaml", type=Path, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--copy-rgb", action="store_true")
    p.add_argument("--max-depth-mm", type=float, default=2000.0)
    p.add_argument("--invalid-depth-value", type=int, default=65535)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def read_classes(source: Path) -> dict[int, dict]:
    with (source / "dataset_config/objectids.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

    result = {int(r["id"]): r for r in rows}

    if not result or sorted(result) != list(range(1, len(result) + 1)):
        raise ValueError("objectids.csv IDs must be contiguous and start at 1")

    return result


def read_split(source: Path, name: str) -> list[str]:
    path = source / "dataset_config" / f"{name}_data_list.txt"

    items = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if not items:
        raise ValueError(f"No records in {path}")

    return items


def unique_stem(record: str) -> str:
    parts = Path(record).parts

    if len(parts) < 3:
        raise ValueError(f"Unexpected list entry: {record}")

    return f"{parts[-3]}_{parts[-1]}"


def copy_or_link(source: Path, target: Path, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return

    if copy:
        shutil.copy2(source, target)
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def write_rgb(
    source: Path,
    target: Path,
    width: int,
    height: int,
    copy: bool,
) -> None:
    rgb = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if rgb is None:
        raise ValueError(f"Could not read RGB image: {source}")

    src_h, src_w = rgb.shape[:2]

    if src_w == width and src_h == height:
        copy_or_link(source, target, copy)
        return

    rgb = cv2.resize(
        rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    target.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(target), rgb):
        raise IOError(f"Could not write RGB: {target}")


def write_depth(
    source: Path,
    target: Path,
    width: int,
    height: int,
    invalid: int,
    max_depth_mm: float,
) -> None:
    depth = cv2.imread(
        str(source),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None or depth.ndim != 2:
        raise ValueError(
            f"Expected 16-bit single-channel depth PNG: {source}"
        )

    depth = depth.astype(np.uint16, copy=False)
    depth[depth == invalid] = 0

    src_h, src_w = depth.shape

    if src_w != width or src_h != height:
        depth = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    if max_depth_mm > 0:
        depth = depth.copy()
        depth[depth > max_depth_mm] = 0

    target.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(target), depth):
        raise IOError(f"Could not write depth: {target}")


def resize_object_mask(
    label: np.ndarray,
    object_id: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    src_mask = (label == object_id).astype(np.uint8)
    source_pixels = int(src_mask.sum())

    if source_pixels == 0:
        return np.zeros((height, width), dtype=np.uint8), 0, 0

    src_h, src_w = src_mask.shape

    if src_w == width and src_h == height:
        resized = src_mask
    else:
        resized = cv2.resize(
            src_mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        resized = (resized > 0.0).astype(np.uint8)

    resized_pixels = int(resized.sum())

    return resized, source_pixels, resized_pixels


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if label is None:
        raise ValueError(f"Could not read label: {path}")

    if label.ndim == 3:
        label = label[:, :, 0]

    return label


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue

        polygon = (
            contour.reshape(-1, 2)
            .astype(float)
            .ravel()
            .tolist()
        )

        if len(polygon) >= 6:
            polygons.append(polygon)

    return polygons


def bbox_polygon(
    xs: np.ndarray,
    ys: np.ndarray,
) -> list[list[float]]:
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())

    if x1 <= x0:
        x1 = x0 + 1.0

    if y1 <= y0:
        y1 = y0 + 1.0

    return [[
        x0, y0,
        x1, y0,
        x1, y1,
        x0, y1,
    ]]


def resize_intrinsic(
    K: np.ndarray,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    return K


def convert_records(
    source: Path,
    output: Path,
    records: list[str],
    image_start: int,
    ann_start: int,
    classes: dict[int, dict],
    copy_rgb: bool,
    invalid: int,
    max_depth_mm: float,
    width: int,
    height: int,
    split_name: str,
    progress_every: int,
) -> tuple[tuple[list, list], int, int]:

    images = []
    annotations = []
    image_id = image_start
    ann_id = ann_start
    total = len(records)

    missing_source_masks = 0
    resized_empty_masks = 0

    print(f"[{split_name}] converting {total:,} images...")

    for index, record in enumerate(records, start=1):
        base = source / "data" / record
        frame = Path(record).name
        stem = unique_stem(record)

        rgb_path = base.with_name(frame + "_color.jpg")
        depth_path = base.with_name(frame + "_depth.png")
        label_path = base.with_name(frame + "_label.png")
        meta_path = base.with_name(frame + "_meta.json")

        required = (
            rgb_path,
            depth_path,
            label_path,
            meta_path,
        )

        if not all(p.exists() for p in required):
            raise FileNotFoundError(
                f"Missing companion files for {record}"
            )

        with Image.open(rgb_path) as im:
            src_width, src_height = im.size

        rgb_out = output / "rgb" / f"{stem}.jpg"
        depth_out = output / "rgb_depth" / f"{stem}.png"

        write_rgb(
            rgb_path,
            rgb_out,
            width,
            height,
            copy_rgb,
        )

        write_depth(
            depth_path,
            depth_out,
            width,
            height,
            invalid,
            max_depth_mm,
        )

        label = load_label(label_path)
        meta = json.loads(meta_path.read_text())

        camera_original = np.asarray(
            meta["intrinsic"],
            dtype=np.float64,
        ).reshape(3, 3)

        camera = resize_intrinsic(
            camera_original,
            src_width,
            src_height,
            width,
            height,
        )

        images.append({
            "id": image_id,
            "file_name": rgb_out.name,
            "width": width,
            "height": height,
        })

        for object_id in (int(x) for x in meta["objects"]):
            if object_id not in classes:
                raise ValueError(
                    f"Unknown object ID {object_id} in {meta_path}"
                )

            mask, source_pixels, resized_pixels = resize_object_mask(
                label,
                object_id,
                width,
                height,
            )

            if source_pixels == 0:
                missing_source_masks += 1
                print(
                    f"[WARN] Object {object_id} exists in metadata "
                    f"but has no pixels in source label: "
                    f"{label_path}"
                )
                continue

            if resized_pixels == 0:
                resized_empty_masks += 1
                print(
                    f"[WARN] Object {object_id} disappeared after resize: "
                    f"{label_path} "
                    f"(source_pixels={source_pixels})"
                )
                continue

            ys, xs = np.nonzero(mask)

            polygons = mask_to_polygons(mask)

            if not polygons:
                polygons = bbox_polygon(xs, ys)
                print(
                    f"[WARN] Polygon fallback -> bbox: "
                    f"object={object_id}, image={stem}, "
                    f"pixels={len(xs)}"
                )

            T = np.asarray(
                meta["object_poses"][str(object_id)],
                dtype=np.float64,
            ).reshape(4, 4)

            pose = np.r_[
                T[:3, 3] * 1000.0,
                T[:3, :3].reshape(-1),
            ].tolist()

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": object_id,
                "bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ],
                "area": int(mask.sum()),
                "segmentation": polygons,
                "iscrowd": 0,
                "ignore": False,
                "visibility": 1.0,
                "cam_K": camera.reshape(-1).tolist(),
                "pose": pose,
                "depth_scale": 1.0,
            })

            ann_id += 1

        image_id += 1

        if progress_every and (
            index % progress_every == 0 or index == total
        ):
            print(
                f"[{split_name}] {index:,}/{total:,} "
                f"({index / total:.1%}), "
                f"{len(annotations):,} annotations"
            )

    print(
        f"[{split_name}] label diagnostics: "
        f"metadata-only={missing_source_masks:,}, "
        f"disappeared-after-resize={resized_empty_masks:,}"
    )

    return (images, annotations), image_id, ann_id


def write_coco(
    path: Path,
    data: tuple[list, list],
    classes: dict[int, dict],
) -> None:
    images, annotations = data

    payload = {
        "info": {
            "description": "POSE RGB-D converted for RACE-6D"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": i,
                "name": c["name"],
                "supercategory": "object",
            }
            for i, c in classes.items()
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n"
    )


def make_training_yaml(
    output: Path,
    category_yaml: Path,
    width: int,
    height: int,
) -> Path:
    output = output.resolve()
    category_yaml = category_yaml.resolve()

    try:
        dataset_path = f"./{output.relative_to(ROOT).as_posix()}"
    except ValueError:
        dataset_path = output.as_posix()

    try:
        category_path = f"./{category_yaml.relative_to(ROOT).as_posix()}"
    except ValueError:
        category_path = category_yaml.as_posix()

    config = {
        "__include__": [
            "../../dataset/coco_detection.yml",
            "../../runtime.yml",
            "../include/dataloader.yml",
            "../include/optimizer.yml",
            "../include/race6d_r50vd_best.yml",
        ],
        "output_dir": "./output/race6d_r50vd_pose_rgbd",
        "coco_path": dataset_path,
        "category_file": category_path,
        "num_classes": 2,
        "eval_spatial_size": [height, width],
        "RACE6D": {
            "backbone": "PResNet_depth",
        },
        "PResNet_depth": {
            "depth": 50,
            "variant": "d",
            "freeze_at": -1,
            "return_idx": [1, 2, 3],
            "num_stages": 4,
            "freeze_norm": False,
            "pretrained": True,
            "in_chans": 4,
        },
        "HybridEncoder": {
            "in_channels": [512, 1024, 2048],
            "feat_strides": [8, 16, 32],
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "hidden_dim": 256,
            "depth_mult": 0.67,
            "enc_act": "quick_gelu",
        },
        "RACE6DTransformer_DQE": {
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "num_levels": 3,
            "num_points": [3, 6, 3],
            "num_layers": 3,
            "vis_enc": True,
            "num_queries": 20,
            "eval_idx": -1,
            "activation": "quick_gelu",
            "mlp_act": "quick_gelu",
            "r_min": 0.5,
            "r_max": 2.0,
        },
        "RACE6DPostProcessor": {
            "num_top_queries": 20,
            "vis_enc": True,
            "remap_mscoco_category": False,
        },
        "RACE6DCriterion_addr": {
            "weight_dict": {
                "loss_vfl": 1,
                "loss_bbox": 5,
                "loss_giou": 2,
                "loss_keypoints": 1,
                "loss_oks": 1,
                "loss_cr": 0.1,
                "loss_pose": 5,
            },
            "losses": [
                "vfl",
                "boxes",
                "pose",
                "keypoint",
            ],
            "alpha": 0.75,
            "gamma_pos": 1.0,
            "gamma_neg": 2.0,
            "matcher": {
                "type": "HungarianMatcher",
                "weight_dict": {
                    "cost_class": 2,
                    "cost_bbox": 5,
                    "cost_giou": 2,
                },
                "alpha": 0.25,
                "gamma": 2.0,
            },
        },
        "epoches": 100,
        "checkpoint_freq": 5,
        "print_freq": 20,
        "progress_bar": True,
        "lr_scheduler": {
            "type": "FlatCosineAnnealingLR",
            "T_max": 50,
            "lr_gamma": 0.5,
            "warmup_iter": 1000,
            "flat_epochs": 5,
            "no_aug_epochs": 10,
        },
        "train_dataloader": {
            "dataset": {
                "img_folder": f"{dataset_path}/rgb",
                "ann_file": (
                    f"{dataset_path}/annotations/"
                    "instances_train.json"
                ),
                "transforms": {
                    "ops": [
                        {
                            "type": "FilterSmallBoxLowVis",
                            "min_size": 5,
                            "min_visib": 0.1,
                        },
                        {"type": "RandomCoarseDropout"},
                        {"type": "ColorJitter"},
                        {"type": "RandomHSVAdjust"},
                        {"type": "RandomSharpen"},
                        {"type": "RandomMotionBlur"},
                        {"type": "RandomGaussianBlur"},
                        {"type": "RandomGaussianNoise"},
                        {"type": "RandomAdditionalNoise"},
                        {
                            "type": "ConvertPILImage",
                            "dtype": "float32",
                            "scale": True,
                        },
                        {
                            "type": "ConvertBoxes",
                            "fmt": "cxcywh",
                            "normalize": True,
                        },
                    ],
                    "policy": {
                        "name": "stop_epoch",
                        "epoch": 40,
                        "ops": [
                            "RandomCoarseDropout",
                            "ColorJitter",
                            "RandomHSVAdjust",
                            "RandomSharpen",
                            "RandomMotionBlur",
                            "RandomGaussianBlur",
                            "RandomGaussianNoise",
                            "RandomAdditionalNoise",
                        ],
                    },
                },
                "return_masks": True,
                "return_full_masks": False,
                "remap_mscoco_category": True,
                "return_depth": True,
                "depth_z_max_mm": 2000.0,
                "category_file": category_path,
                "coco_path": dataset_path,
                "need_aligned": False,
            },
            "collate_fn": {
                "scales": None,
            },
            "total_batch_size": 8,
            "num_workers": 0,
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": f"{dataset_path}/rgb",
                "ann_file": (
                    f"{dataset_path}/annotations/"
                    "instances_val.json"
                ),
                "transforms": {
                    "ops": [
                        {
                            "type": "FilterSmallBoxLowVis",
                            "min_size": 5,
                            "min_visib": 0.1,
                        },
                        {
                            "type": "ConvertPILImage",
                            "dtype": "float32",
                            "scale": True,
                        },
                        {
                            "type": "ConvertBoxes",
                            "fmt": "cxcywh",
                            "normalize": True,
                        },
                    ],
                },
                "return_masks": True,
                "return_full_masks": False,
                "remap_mscoco_category": True,
                "return_depth": True,
                "depth_z_max_mm": 2000.0,
                "category_file": category_path,
                "coco_path": dataset_path,
                "need_aligned": False,
            },
            "total_batch_size": 1,
            "num_workers": 0,
        },
    }

    yaml_path = (
        ROOT
        / "configs"
        / "race6d"
        / "r50vd"
        / "race6d_r50vd_pose_rgbd.yml"
    )

    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(
        "[OK] Training YAML written to:\n"
        f"     {yaml_path}"
    )

    return yaml_path


def main() -> None:
    a = args()

    source = a.source.resolve()
    output = a.output.resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"Source does not exist: {source}"
        )

    if a.width <= 0 or a.height <= 0:
        raise ValueError(
            "width and height must be positive"
        )

    classes = read_classes(source)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output must be empty: {output}"
        )

    output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("POSE RGB-D -> RACE-6D RESIZED DATASET")
    print("=" * 70)
    print(f"Source     : {source}")
    print(f"Output     : {output}")
    print(f"Resolution : {a.width} x {a.height}")
    print(f"Depth max  : {a.max_depth_mm} mm")
    print("Label      : per-object INTER_AREA + threshold")
    print("=" * 70)

    train = read_split(source, "train")
    test = read_split(source, "test")

    if set(train) & set(test):
        raise ValueError("train/test lists overlap")

    train_data, next_img, next_ann = convert_records(
        source=source,
        output=output,
        records=train,
        image_start=1,
        ann_start=1,
        classes=classes,
        copy_rgb=a.copy_rgb,
        invalid=a.invalid_depth_value,
        max_depth_mm=a.max_depth_mm,
        width=a.width,
        height=a.height,
        split_name="train",
        progress_every=a.progress_every,
    )

    test_data, _, _ = convert_records(
        source=source,
        output=output,
        records=test,
        image_start=next_img,
        ann_start=next_ann,
        classes=classes,
        copy_rgb=a.copy_rgb,
        invalid=a.invalid_depth_value,
        max_depth_mm=a.max_depth_mm,
        width=a.width,
        height=a.height,
        split_name="val",
        progress_every=a.progress_every,
    )

    write_coco(
        output / "annotations/instances_train.json",
        train_data,
        classes,
    )

    write_coco(
        output / "annotations/instances_val.json",
        test_data,
        classes,
    )

    prepare_models(
        source,
        output,
        classes,
    )

    category_yaml = a.category_yaml or output / "category2name.yml"
    category_yaml = (
        category_yaml
        if category_yaml.is_absolute()
        else ROOT / category_yaml
    ).resolve()

    category_yaml.parent.mkdir(parents=True, exist_ok=True)

    category_yaml.write_text(
        yaml.safe_dump(
            {
                "category2name": {
                    i: c["name"]
                    for i, c in classes.items()
                }
            },
            sort_keys=False,
        )
    )

    def split_stats(data):
        counts = {str(i): 0 for i in classes}

        for ann in data[1]:
            counts[str(ann["category_id"])] += 1

        return {
            "images": len(data[0]),
            "annotations": len(data[1]),
            "instances_by_category": counts,
        }

    manifest = {
        "source": str(source),
        "output": str(output),
        "resolution": {
            "width": a.width,
            "height": a.height,
        },
        "resize": {
            "rgb": "INTER_LINEAR",
            "depth": "INTER_NEAREST",
            "label": "per-object INTER_AREA + threshold > 0",
        },
        "label_output": False,
        "camera_intrinsic": {
            "description": (
                "K scaled from original image coordinates "
                "to resized image coordinates"
            ),
            "formula": {
                "fx": "fx * sx",
                "fy": "fy * sy",
                "cx": "cx * sx",
                "cy": "cy * sy",
            },
        },
        "pose": {
            "translation_unit": "mm",
            "rotation": "3x3_matrix",
            "unchanged_by_resize": True,
        },
        "depth": {
            "unit": "mm",
            "invalid_value": a.invalid_depth_value,
            "z_max_mm": a.max_depth_mm,
        },
        "train": split_stats(train_data),
        "val": split_stats(test_data),
        "classes": {
            i: c["name"]
            for i, c in classes.items()
        },
        "category_yaml": str(category_yaml),
    }

    (output / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    training_yaml = make_training_yaml(
        output=output,
        category_yaml=category_yaml,
        width=a.width,
        height=a.height,
    )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Resolution : {a.width} x {a.height}")
    print(
        f"Train     : {len(train_data[0]):,} images, "
        f"{len(train_data[1]):,} annotations"
    )
    print(
        f"Val       : {len(test_data[0]):,} images, "
        f"{len(test_data[1]):,} annotations"
    )

    print("\nGenerated:")
    print(f"  {output / 'rgb'}")
    print(f"  {output / 'rgb_depth'}")
    print(f"  {output / 'annotations/instances_train.json'}")
    print(f"  {output / 'annotations/instances_val.json'}")
    print(f"  {output / 'models'}")
    print(f"  {output / 'cached_keypoints_3d_edge.json'}")
    print(f"  {output / 'prepare_manifest.json'}")
    print(f"  {category_yaml}")
    print(f"  {training_yaml}")
    print("=" * 70)


if __name__ == "__main__":
    main()
