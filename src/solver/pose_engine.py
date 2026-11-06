"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
https://github.com/facebookresearch/detr/blob/main/engine.py
----------------------------------------------------------------------
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
----------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import sys
import math
from typing import Iterable, Callable

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from .pose_reporting import PoseEvaluationReporter


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr",
        SmoothedValue(window_size=1, fmt="{value:.6f}"),
    )

    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)
    progress_bar: bool = kwargs.get("progress_bar", False)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get(
        "lr_warmup_scheduler",
        None,
    )
    lr_scheduler = kwargs.get("lr_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
            progress_bar=progress_bar,
        )
    ):
        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in target.items()
            }
            for target in targets
        ]

        global_step = epoch * len(data_loader) + i
        metas = {
            "epoch": epoch,
            "step": i,
            "global_step": global_step,
        }

        cam_K = None
        if targets and "cam_K" in targets[0] and len(targets[0]["cam_K"]) > 0:
            cam_K = torch.stack(
                [target["cam_K"][0].reshape(3, 3) for target in targets]
            )

        if scaler is not None:
            with torch.autocast(
                device_type=str(device),
                cache_enabled=True,
            ):
                outputs = model(
                    samples,
                    cam_K=cam_K,
                    targets=targets,
                )

            with torch.autocast(
                device_type=str(device),
                enabled=False,
            ):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(
                value
                for key, value in loss_dict.items()
                if not key.startswith("metric_")
            )

            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(
                samples,
                cam_K=cam_K,
                targets=targets,
            )

            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(
                value
                for key, value in loss_dict.items()
                if not key.startswith("metric_")
            )

            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm,
                )

            optimizer.step()
            optimizer.zero_grad()

        if ema is not None:
            ema.update(model)

        if lr_scheduler is not None and getattr(lr_scheduler, "_iter_based", False):
            lr_scheduler.step(global_step, optimizer)
        elif lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)

        metrics_reduced = {
            key: value
            for key, value in loss_dict_reduced.items()
            if key.startswith("metric_")
        }

        losses_reduced = {
            key: value
            for key, value in loss_dict_reduced.items()
            if not key.startswith("metric_")
        }

        loss_value = sum(losses_reduced.values())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(losses_reduced)
            sys.exit(1)

        metric_logger.update(**metrics_reduced)
        metric_logger.update(
            loss=loss_value,
            **losses_reduced,
        )
        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"],
        )

        if writer and dist_utils.is_main_process():
            writer.add_scalar(
                "Loss/total",
                loss_value.item(),
                global_step,
            )

            for j, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f"Lr/pg_{j}",
                    param_group["lr"],
                    global_step,
                )

            for key, value in losses_reduced.items():
                writer.add_scalar(
                    f"Loss/{key}",
                    value.item(),
                    global_step,
                )

            for key, value in metrics_reduced.items():
                writer.add_scalar(
                    f"Metric/{key}",
                    value.item(),
                    global_step,
                )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    report_pose: bool = False,
    overlay_dir: str | None = None,
    overlay_limit: int = 100,
    score_threshold: float = 0.25,
    pose_iou_threshold: float = 0.5,
    reporter_cls: Callable | None = None,
):
    """Run validation, COCO evaluation, and optional pose reporting."""
    model.eval()
    criterion.eval()

    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    mscoco_label2category = getattr(
        criterion,
        "mscoco_label2category",
        {},
    )

    if coco_evaluator is not None and hasattr(coco_evaluator, "set_label_mapping"):
        coco_evaluator.set_label_mapping(mscoco_label2category)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else []

    pose_reporter = None

    if report_pose and dist_utils.is_main_process():
        reporter_class = (
            reporter_cls if reporter_cls is not None else PoseEvaluationReporter
        )

        reporter_kwargs = {
            "overlay_dir": overlay_dir,
            "overlay_limit": overlay_limit,
            "score_threshold": score_threshold,
            "iou_threshold": pose_iou_threshold,
        }

        pose_reporter = reporter_class(
            criterion,
            data_loader,
            **reporter_kwargs,
        )

    for samples, targets in metric_logger.log_every(
        data_loader,
        100,
        header,
    ):
        samples = samples.to(device)

        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]

        cam_K = None

        if targets and "cam_K" in targets[0] and len(targets[0]["cam_K"]) > 0:
            cam_K = torch.stack(
                [target["cam_K"][0].reshape(3, 3) for target in targets]
            )

        outputs = model(
            samples,
            cam_K=cam_K,
        )

        loss_dict = criterion(
            outputs,
            targets,
        )

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)

        metrics_reduced_eval = {
            key: value
            for key, value in loss_dict_reduced.items()
            if key.startswith("metric_")
        }

        losses_reduced_eval = {
            key: value
            for key, value in loss_dict_reduced.items()
            if not key.startswith("metric_")
        }

        loss_value = sum(losses_reduced_eval.values())

        metric_logger.update(**metrics_reduced_eval)
        metric_logger.update(
            loss=loss_value,
            **losses_reduced_eval,
        )

        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in targets],
            dim=0,
        )

        results = postprocessor(
            outputs,
            orig_target_sizes,
        )

        if pose_reporter is not None:
            pose_reporter.update(
                outputs,
                results,
                targets,
            )

        if coco_evaluator is not None:
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, results)
            }
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}

    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    if pose_reporter is not None:
        pose_stats = pose_reporter.summarize()
        stats.update(pose_stats)
        print("Pose stats:", pose_stats)

    return stats, coco_evaluator
