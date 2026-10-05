#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skip invalid JSON at line {line_no}")
                continue

            if "epoch" not in data:
                continue

            records.append(data)

    records.sort(key=lambda x: x["epoch"])

    if not records:
        raise RuntimeError("No valid training records found.")

    return records


def values(records, key):
    result = []

    for r in records:
        value = r.get(key)

        if value is None:
            result.append(np.nan)
        else:
            result.append(float(value))

    return np.asarray(result)


def plot_train_test(records, train_key, test_key, title, ylabel, output):
    epochs = values(records, "epoch")
    train = values(records, train_key)
    test = values(records, test_key)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train,
        marker="o",
        label="Train",
    )

    plt.plot(
        epochs,
        test,
        marker="o",
        label="Test",
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output, dpi=150)
    plt.close()


def plot_metric_group(records, metrics, title, output):
    epochs = values(records, "epoch")

    plt.figure(figsize=(11, 7))

    for key, label in metrics:
        data = values(records, key)

        if np.all(np.isnan(data)):
            continue

        plt.plot(
            epochs,
            data,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output, dpi=150)
    plt.close()


def analyze_trend(records, key, window=3):
    data = values(records, key)

    valid = data[np.isfinite(data)]

    if len(valid) < 2:
        return None

    if len(valid) < window:
        window = len(valid)

    recent = valid[-window:]

    if len(recent) < 2:
        return None

    first = recent[0]
    last = recent[-1]

    if abs(first) < 1e-12:
        improvement = 0.0
    else:
        improvement = (first - last) / abs(first) * 100.0

    return {
        "first": float(first),
        "last": float(last),
        "improvement": float(improvement),
    }


def analyze_convergence(records):
    print("\n" + "=" * 70)
    print("Training Convergence Analysis")
    print("=" * 70)

    epochs = values(records, "epoch")

    print(
        f"Epoch range : "
        f"{int(np.nanmin(epochs))} -> "
        f"{int(np.nanmax(epochs))}"
    )

    print(f"Epoch count : {len(records)}")

    metrics = [
        ("train_loss", "Train loss"),
        ("test_loss", "Test loss"),
        ("train_metric_tx_error", "Train TX"),
        ("test_metric_tx_error", "Test TX"),
        ("train_metric_ty_error", "Train TY"),
        ("test_metric_ty_error", "Test TY"),
        ("train_metric_tz_error", "Train TZ"),
        ("test_metric_tz_error", "Test TZ"),
        ("train_metric_rot_error", "Train rotation"),
        ("test_metric_rot_error", "Test rotation"),
        ("train_metric_kpt_error", "Train keypoint"),
        ("test_metric_kpt_error", "Test keypoint"),
    ]

    print("\nRecent trends")
    print("-" * 70)

    for key, name in metrics:
        result = analyze_trend(records, key)

        if result is None:
            continue

        print(
            f"{name:20s} "
            f"{result['first']:10.4f} -> "
            f"{result['last']:10.4f} "
            f"({result['improvement']:+7.2f}%)"
        )


def detect_status(records):
    print("\n" + "=" * 70)
    print("Convergence Diagnosis")
    print("=" * 70)

    test_loss = values(records, "test_loss")
    train_loss = values(records, "train_loss")

    test_pose = values(
        records,
        "test_metric_rot_error",
    )

    test_tz = values(
        records,
        "test_metric_tz_error",
    )

    if len(records) < 4:
        print("Not enough epochs for reliable diagnosis.")
        return

    window = min(3, len(records))

    recent_test_loss = test_loss[-window:]
    recent_train_loss = train_loss[-window:]

    recent_pose = test_pose[-window:]
    recent_tz = test_tz[-window:]

    def slope(data):
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return np.nan

        x = np.arange(len(data))

        return np.polyfit(x, data, 1)[0]

    loss_slope = slope(recent_test_loss)
    train_slope = slope(recent_train_loss)
    pose_slope = slope(recent_pose)
    tz_slope = slope(recent_tz)

    print(f"\nRecent test loss slope : {loss_slope:+.6f}")
    print(f"Recent train loss slope: {train_slope:+.6f}")
    print(f"Recent rotation slope  : {pose_slope:+.6f}")
    print(f"Recent TZ slope        : {tz_slope:+.6f}")

    overfit = (
        np.isfinite(loss_slope)
        and np.isfinite(train_slope)
        and loss_slope > 0
        and train_slope < 0
    )

    test_improving = (
        np.isfinite(loss_slope)
        and loss_slope < 0
    )

    pose_improving = (
        np.isfinite(pose_slope)
        and pose_slope < 0
    )

    tz_improving = (
        np.isfinite(tz_slope)
        and tz_slope < 0
    )

    if overfit:
        print("\nSTATUS: OVERFITTING")
        print(
            "Train loss is decreasing while test loss is increasing."
        )
        print(
            "Continuing training may not improve generalization."
        )

    elif test_improving and (
        pose_improving or tz_improving
    ):
        print("\nSTATUS: STILL CONVERGING")
        print(
            "Test loss and pose-related metrics are still improving."
        )
        print(
            "Training can likely continue."
        )

    else:
        print("\nSTATUS: POSSIBLE PLATEAU")
        print(
            "Recent test metrics show little or inconsistent improvement."
        )
        print(
            "Consider checking the learning rate and validation trend."
        )


def plot_loss(records, output_dir):
    plot_train_test(
        records,
        "train_loss",
        "test_loss",
        "Training vs Test Loss",
        "Loss",
        output_dir / "loss.png",
    )


def plot_translation(records, output_dir):
    epochs = values(records, "epoch")

    metrics = [
        (
            "train_metric_tx_error",
            "Train TX",
        ),
        (
            "test_metric_tx_error",
            "Test TX",
        ),
        (
            "train_metric_ty_error",
            "Train TY",
        ),
        (
            "test_metric_ty_error",
            "Test TY",
        ),
        (
            "train_metric_tz_error",
            "Train TZ",
        ),
        (
            "test_metric_tz_error",
            "Test TZ",
        ),
    ]

    plt.figure(figsize=(11, 7))

    for key, label in metrics:
        data = values(records, key)

        plt.plot(
            epochs,
            data,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Translation Error")
    plt.title("Translation Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "translation_error.png",
        dpi=150,
    )

    plt.close()


def plot_pose(records, output_dir):
    epochs = values(records, "epoch")

    metrics = [
        (
            "train_metric_rot_error",
            "Train Rotation",
        ),
        (
            "test_metric_rot_error",
            "Test Rotation",
        ),
        (
            "train_metric_kpt_error",
            "Train Keypoint",
        ),
        (
            "test_metric_kpt_error",
            "Test Keypoint",
        ),
    ]

    plt.figure(figsize=(11, 7))

    for key, label in metrics:
        data = values(records, key)

        plt.plot(
            epochs,
            data,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Pose / Keypoint Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "pose_error.png",
        dpi=150,
    )

    plt.close()


def plot_bbox(records, output_dir):
    epochs = values(records, "epoch")

    metrics = [
        (
            "train_metric_bbox_cx_error",
            "Train CX",
        ),
        (
            "test_metric_bbox_cx_error",
            "Test CX",
        ),
        (
            "train_metric_bbox_cy_error",
            "Train CY",
        ),
        (
            "test_metric_bbox_cy_error",
            "Test CY",
        ),
        (
            "train_metric_bbox_w_error",
            "Train Width",
        ),
        (
            "test_metric_bbox_w_error",
            "Test Width",
        ),
        (
            "train_metric_bbox_h_error",
            "Train Height",
        ),
        (
            "test_metric_bbox_h_error",
            "Test Height",
        ),
    ]

    plt.figure(figsize=(11, 7))

    for key, label in metrics:
        data = values(records, key)

        plt.plot(
            epochs,
            data,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("BBox Error")
    plt.title("Bounding Box Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "bbox_error.png",
        dpi=150,
    )

    plt.close()


def plot_coco(records, output_dir):
    epochs = values(records, "epoch")

    metrics = {
        0: "AP",
        1: "AP50",
        2: "AP75",
        4: "AP_small",
        5: "AP_medium",
        6: "AP_large",
        7: "AR1",
        8: "AR10",
        9: "AR100",
        10: "AR_small",
        11: "AR_medium",
    }

    plt.figure(figsize=(11, 7))

    for index, label in metrics.items():
        data = []

        for record in records:
            values_list = record.get(
                "test_coco_eval_bbox"
            )

            if (
                not isinstance(values_list, list)
                or index >= len(values_list)
                or values_list[index] < 0
            ):
                data.append(np.nan)
            else:
                data.append(
                    float(values_list[index])
                )

        plt.plot(
            epochs,
            data,
            marker="o",
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("COCO Metric")
    plt.title("COCO Bounding Box Evaluation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "coco_eval.png",
        dpi=150,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RACE-6D training log."
    )

    parser.add_argument(
        "log",
        type=str,
        help="Path to log.txt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="training_analysis",
        help="Output directory.",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    output_dir = Path(args.output)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_log(log_path)

    print(f"Loaded {len(records)} epochs.")

    analyze_convergence(records)
    detect_status(records)

    plot_loss(
        records,
        output_dir,
    )

    plot_translation(
        records,
        output_dir,
    )

    plot_pose(
        records,
        output_dir,
    )

    plot_bbox(
        records,
        output_dir,
    )

    plot_coco(
        records,
        output_dir,
    )

    print("\n" + "=" * 70)
    print("Plots generated")
    print("=" * 70)
    print(f"Output directory: {output_dir.resolve()}")
    print()
    print("  loss.png")
    print("  translation_error.png")
    print("  pose_error.png")
    print("  bbox_error.png")
    print("  coco_eval.png")


if __name__ == "__main__":
    main()