#!/usr/bin/env python3
"""Randomly preview masks and 6D poses from a converted RACE-6D dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='converted dataset root')
    p.add_argument('--category-yaml', type=Path, default=None)
    p.add_argument('--split', choices=('train', 'val', 'all'), default='all')
    p.add_argument('--num-images', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('output/dataset_preview'))
    return p.parse_args()


def draw_pose(image: np.ndarray, pose: list[float], K: np.ndarray) -> None:
    t = np.asarray(pose[:3], dtype=np.float64)
    R = np.asarray(pose[3:], dtype=np.float64).reshape(3, 3)
    # Pose and K use the original pixel coordinates; a depth-relative axis is
    # readable for both near and far instances. Units are millimetres.
    length = max(30.0, float(t[2]) * 0.10)
    points = np.vstack((np.zeros(3), np.eye(3) * length))
    camera_points = points @ R.T + t
    valid = camera_points[:, 2] > 1e-6
    if not valid.all(): return
    pixels_h = camera_points @ K.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    origin = tuple(np.rint(pixels[0]).astype(int))
    for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.arrowedLine(image, origin, tuple(np.rint(endpoint).astype(int)), color, 2, cv2.LINE_AA, tipLength=.18)


def main() -> None:
    a = parse_args(); root = a.dataset.resolve()
    names = {}
    category_path = a.category_yaml or root / 'category2name.yml'
    if category_path.exists():
        names = {int(k): str(v) for k, v in (yaml.safe_load(category_path.read_text()) or {}).get('category2name', {}).items()}
    payloads = []
    splits = ('train', 'val') if a.split == 'all' else (a.split,)
    for split in splits:
        path = root / 'annotations' / f'instances_{split}.json'
        data = json.loads(path.read_text()); payloads.append((split, data))
        for c in data.get('categories', []): names.setdefault(int(c['id']), c['name'])
    candidates = [(split, image, [x for x in data['annotations'] if x['image_id'] == image['id']])
                  for split, data in payloads for image in data['images']]
    if not candidates: raise ValueError('No images found')
    selected = random.Random(a.seed).sample(candidates, min(a.num_images, len(candidates)))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (split, record, anns) in enumerate(selected, 1):
        image = cv2.imread(str(root / 'rgb' / record['file_name']), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(record['file_name'])
        overlay = image.copy()
        for ann in anns:
            color = ((37 * ann['category_id']) % 256, (113 * ann['category_id']) % 256, (211 * ann['category_id']) % 256)
            for polygon in ann.get('segmentation', []):
                cv2.fillPoly(overlay, [np.rint(np.asarray(polygon).reshape(-1, 2)).astype(np.int32)], color)
        image = cv2.addWeighted(image, .55, overlay, .45, 0)
        for ann in anns:
            x, y, w, h = np.rint(ann['bbox']).astype(int); category_id = int(ann['category_id'])
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{category_id}: {names.get(category_id, 'unknown')}", (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2, cv2.LINE_AA)
            draw_pose(image, ann['pose'], np.asarray(ann['cam_K'], dtype=np.float64).reshape(3,3))
        cv2.putText(image, 'mask=color  pose: X=red Y=green Z=blue', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image, 'mask=color  pose: X=red Y=green Z=blue', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1, cv2.LINE_AA)
        target = a.output_dir / f'{index:03d}_{split}_{Path(record["file_name"]).stem}.jpg'
        cv2.imwrite(str(target), image)
    print(json.dumps({'preview_images': len(selected), 'output_dir': str(a.output_dir), 'seed': a.seed, 'categories': names}, indent=2))


if __name__ == '__main__': main()
