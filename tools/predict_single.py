"""Run one RACE-6D RGB or RGB-D pose prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Importing these modules registers all YAML-configured components.
import src.zoo  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="model YAML configuration")
    parser.add_argument("-r", "--checkpoint", required=True, help=".pth checkpoint")
    parser.add_argument("--rgb", required=True, help="RGB image path")
    parser.add_argument("--depth", required=True, help="aligned depth image path")
    parser.add_argument(
        "--camera-k", type=float, nargs=9, metavar="K",
        help="3x3 camera intrinsic matrix in row-major order",
    )
    parser.add_argument(
        "--camera-json", help="pose JSON containing camera.intrinsics.matrix; alternative to --camera-k"
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-per-class", type=int, default=1,
                        help="highest-scoring predictions to retain for each class; default: 1")
    parser.add_argument("--max-detections", type=int, default=None,
                        help="optional total cap after per-class filtering")
    parser.add_argument("--depth-z-max-mm", type=float, default=None,
                        help="depth normalization cap; defaults to YAML value or 2000")
    parser.add_argument("--invalid-depth-value", type=int, default=65535)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--output", default="output/single_prediction.json")
    parser.add_argument("--debug-image", default="output/single_prediction.jpg")
    return parser.parse_args()


def load_camera(args: argparse.Namespace) -> np.ndarray:
    if args.camera_k is not None:
        return np.asarray(args.camera_k, dtype=np.float32).reshape(3, 3)
    if args.camera_json:
        with open(args.camera_json, encoding="utf-8") as handle:
            data = json.load(handle)
        try:
            return np.asarray(data["camera"]["intrinsics"]["matrix"], dtype=np.float32).reshape(3, 3)
        except (KeyError, ValueError) as error:
            raise ValueError("camera JSON must contain camera.intrinsics.matrix (3x3)") from error
    raise ValueError("translation needs camera intrinsics: provide --camera-k or --camera-json")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Training checkpoints contain model/EMA state; exported checkpoints may be state dicts.
    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        weights = state["ema"]["module"]
        source = "EMA"
    elif "model" in state:
        weights = state["model"]
        source = "model"
    else:
        weights = state
        source = "state dict"
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint ({source}) does not match the configured model; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def draw_debug(
    rgb: np.ndarray, depth: np.ndarray, detections: list[dict], camera: np.ndarray, path: Path
) -> None:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x0, y0, x1, y1 = np.rint(detection["bbox_xyxy"]).astype(int)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"score={detection['score']:.3f}", (x0, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Object-frame axes projected using the predicted [R|t], all in mm.
        rotation = np.asarray(detection["rotation"], dtype=np.float32)
        translation = np.asarray(detection["translation_mm"], dtype=np.float32)
        axis_length_mm = max(20.0, float(translation[2]) * 0.08)
        points_3d = np.vstack((np.zeros(3), np.eye(3) * axis_length_mm))
        camera_points = points_3d @ rotation.T + translation
        pixels_h = camera_points @ camera.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        origin = tuple(np.rint(pixels[0]).astype(int))
        for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.arrowedLine(canvas, origin, tuple(np.rint(endpoint).astype(int)), color, 2, cv2.LINE_AA, tipLength=0.15)

    # A compact depth preview makes alignment/input mistakes immediately visible.
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_preview = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
    if depth_preview.shape[:2] != canvas.shape[:2]:
        depth_preview = cv2.resize(depth_preview, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
    debug = np.hstack((canvas, depth_preview))
    cv2.putText(debug, "RGB prediction (left) | depth input (right)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), debug):
        raise IOError(f"Could not write debug image: {path}")


def main() -> None:
    args = parse_args()
    camera = load_camera(args)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"))
    depth = np.asarray(Image.open(args.depth))
    if depth.ndim != 2:
        raise ValueError(f"Depth input must be a single-channel image, got shape {depth.shape}")
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/depth must be aligned and the same size; got {rgb.shape[:2]} and {depth.shape[:2]}")
    height, width = rgb.shape[:2]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = YAMLConfig(args.config)
    # The checkpoint overwrites every backbone parameter.  Avoid an unnecessary
    # pretrained-backbone download while merely constructing the model.
    for backbone_name in ("PResNet", "PResNet_depth"):
        if backbone_name in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone_name]["pretrained"] = False
    model = cfg.model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    criterion = cfg.criterion.to(device).eval()
    # _c2t_pred is the canonical translation decoder used by validation.
    criterion.set_pose_source(model.decoder)
    load_checkpoint(model, args.checkpoint)

    # HybridEncoder caches positional embeddings at eval_spatial_size.  Its
    # inference input must therefore have exactly this resolution (480x640 for
    # the supplied iPhone configuration), even when the source camera is HD.
    input_height, input_width = cfg.yaml_cfg.get("eval_spatial_size", [height, width])
    model_rgb = cv2.resize(rgb, (int(input_width), int(input_height)), interpolation=cv2.INTER_LINEAR)
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(model_rgb)).permute(2, 0, 1).float().div_(255)
    expected_channels = model.backbone.conv1[0].conv.in_channels
    if expected_channels == 4:
        depth_model = cv2.resize(depth, (int(input_width), int(input_height)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        depth_model[depth_model == args.invalid_depth_value] = 0.0
        configured_cap = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("depth_z_max_mm", 2000.0)
        depth_cap = float(args.depth_z_max_mm or configured_cap)
        if depth_cap <= 0:
            raise ValueError("--depth-z-max-mm must be positive")
        depth_tensor = torch.from_numpy(np.clip(depth_model, 0.0, depth_cap) / depth_cap).unsqueeze(0)
        image = torch.cat((rgb_tensor, depth_tensor), dim=0)
    elif expected_channels == 3:
        image = rgb_tensor
    else:
        raise RuntimeError(f"Only 3- or 4-channel backbones are supported, got {expected_channels}")
    image = image.unsqueeze(0).to(device)
    original_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        result = postprocessor(model(image), original_size)[0]

    if args.max_per_class < 1 or (args.max_detections is not None and args.max_detections < 1):
        raise ValueError("--max-per-class and --max-detections must be at least 1")
    keep_indices = torch.nonzero(result["scores"] >= args.score_threshold, as_tuple=False).squeeze(1)
    if len(keep_indices):
        order = torch.argsort(result["scores"][keep_indices], descending=True)
        sorted_indices = keep_indices[order]
        # Detector queries can duplicate the same object.  For this dataset,
        # retain the best query for each class (one iPhone + one brown box).
        class_counts, selected = {}, []
        for index in sorted_indices.tolist():
            label = int(result["labels"][index])
            if class_counts.get(label, 0) >= args.max_per_class:
                continue
            selected.append(index)
            class_counts[label] = class_counts.get(label, 0) + 1
            if args.max_detections is not None and len(selected) >= args.max_detections:
                break
        keep_indices = torch.tensor(selected, dtype=torch.long, device=result["scores"].device)
    boxes = result["boxes"][keep_indices]
    scores = result["scores"][keep_indices]
    labels = result["labels"][keep_indices]
    rotations = result["rotations"][keep_indices]
    codes = result["translations"][keep_indices]
    boxes_normalized = boxes.clone()
    boxes_normalized[:, 0::2] /= width
    boxes_normalized[:, 1::2] /= height
    # xyxy -> normalized cxcywh, matching the validation path exactly.
    boxes_normalized = torch.stack(((boxes_normalized[:, 0] + boxes_normalized[:, 2]) / 2,
                                    (boxes_normalized[:, 1] + boxes_normalized[:, 3]) / 2,
                                    boxes_normalized[:, 2] - boxes_normalized[:, 0],
                                    boxes_normalized[:, 3] - boxes_normalized[:, 1]), dim=1)
    translations = criterion._c2t_pred(
        codes, torch.from_numpy(camera).to(device), boxes_normalized, width, height
    ) if len(codes) else codes.new_empty((0, 3))

    detections = []
    for box, score, label, rotation, translation in zip(boxes, scores, labels, rotations, translations):
        detections.append({
            "label": int(label), "score": float(score),
            "bbox_xyxy": box.detach().cpu().tolist(),
            "rotation": rotation.detach().cpu().tolist(),
            "translation_mm": translation.detach().cpu().tolist(),
        })
    debug_path = Path(args.debug_image)
    draw_debug(rgb, depth, detections, camera, debug_path)
    payload = {"rotation": detections[0]["rotation"] if detections else None,
               "translation_mm": detections[0]["translation_mm"] if detections else None,
               "debug_image": str(debug_path), "detections": detections}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
