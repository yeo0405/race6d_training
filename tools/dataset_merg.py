#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-a", type=Path, required=True)
    p.add_argument("--dataset-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_categories(dataset):
    path = dataset / "category2name.yml"

    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in data["category2name"].items()}

    path = dataset / "annotations" / "instances_train.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(x["id"]): x["name"] for x in data["categories"]}


def load_coco(dataset, split):
    path = dataset / "annotations" / f"instances_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def copy_images(dataset, output, prefix):
    for folder in ("rgb", "rgb_depth"):
        src = dataset / folder
        dst = output / folder
        dst.mkdir(parents=True, exist_ok=True)

        for file in src.iterdir():
            if file.is_file():
                shutil.copy2(file, dst / f"{prefix}_{file.name}")


def merge_split(
    dataset,
    split,
    prefix,
    category_map,
    output,
    next_image_id,
    next_annotation_id,
):
    coco = load_coco(dataset, split)

    image_id_map = {}
    images = []
    annotations = []

    for image in coco["images"]:
        old_id = int(image["id"])
        new_id = next_image_id
        next_image_id += 1

        image_id_map[old_id] = new_id

        images.append({
            **image,
            "id": new_id,
            "file_name": f"{prefix}_{Path(image['file_name']).name}",
        })

    for ann in coco["annotations"]:
        old_category = int(ann["category_id"])

        if old_category not in category_map:
            raise ValueError(
                f"Unknown category {old_category} in {dataset}"
            )

        annotations.append({
            **ann,
            "id": next_annotation_id,
            "image_id": image_id_map[int(ann["image_id"])],
            "category_id": category_map[old_category],
        })

        next_annotation_id += 1

    return (
        images,
        annotations,
        next_image_id,
        next_annotation_id,
    )


def merge_models(
    dataset_a,
    dataset_b,
    output,
    categories_a,
    categories_b,
    class_to_id,
):
    output_models = output / "models"
    output_models.mkdir(parents=True, exist_ok=True)

    info_a = json.loads(
        (dataset_a / "models" / "models_info.json").read_text()
    )
    info_b = json.loads(
        (dataset_b / "models" / "models_info.json").read_text()
    )

    key_a = json.loads(
        (dataset_a / "cached_keypoints_3d_edge.json").read_text()
    )
    key_b = json.loads(
        (dataset_b / "cached_keypoints_3d_edge.json").read_text()
    )

    merged_info = {}
    merged_keypoints = {}
    merged_edges = {}

    processed = set()

    for dataset, categories, info, key_data in (
        (dataset_a, categories_a, info_a, key_a),
        (dataset_b, categories_b, info_b, key_b),
    ):
        for old_id, name in categories.items():
            new_id = class_to_id[name]

            if name in processed:
                continue

            model = dataset / "models" / f"obj_{old_id:06d}.ply"

            if not model.exists():
                raise FileNotFoundError(model)

            shutil.copy2(
                model,
                output_models / f"obj_{new_id:06d}.ply",
            )

            merged_info[str(new_id)] = info[str(old_id)]
            merged_keypoints[str(new_id)] = (
                key_data["keypoints_3d"][str(old_id)]
            )
            merged_edges[str(new_id)] = (
                key_data["edges_mask"][str(old_id)]
            )

            processed.add(name)

    (output_models / "models_info.json").write_text(
        json.dumps(merged_info, indent=2) + "\n"
    )

    (output / "cached_keypoints_3d_edge.json").write_text(
        json.dumps({
            "metadata": {
                "units": "millimetres",
                "num_keypoints_per_class": 32,
            },
            "keypoints_3d": merged_keypoints,
            "edges_mask": merged_edges,
        }, indent=2) + "\n"
    )


def main():
    a = parse_args()

    dataset_a = a.dataset_a.resolve()
    dataset_b = a.dataset_b.resolve()
    output = a.output.resolve()

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    output.mkdir(parents=True)

    categories_a = load_categories(dataset_a)
    categories_b = load_categories(dataset_b)

    print("Dataset A:")
    for cid, name in sorted(categories_a.items()):
        print(f"  {cid}: {name}")

    print("\nDataset B:")
    for cid, name in sorted(categories_b.items()):
        print(f"  {cid}: {name}")

    class_to_id = {}

    for name in categories_a.values():
        if name not in class_to_id:
            class_to_id[name] = len(class_to_id) + 1

    for name in categories_b.values():
        if name not in class_to_id:
            class_to_id[name] = len(class_to_id) + 1

    category_map_a = {
        old_id: class_to_id[name]
        for old_id, name in categories_a.items()
    }

    category_map_b = {
        old_id: class_to_id[name]
        for old_id, name in categories_b.items()
    }

    print("\nMerged classes:")
    for name, cid in class_to_id.items():
        print(f"  {cid}: {name}")

    copy_images(dataset_a, output, "A")
    copy_images(dataset_b, output, "B")

    all_data = {}

    for split in ("train", "val"):
        images_a, anns_a, next_image, next_ann = merge_split(
            dataset_a,
            split,
            "A",
            category_map_a,
            output,
            1,
            1,
        )

        images_b, anns_b, next_image, next_ann = merge_split(
            dataset_b,
            split,
            "B",
            category_map_b,
            output,
            next_image,
            next_ann,
        )

        images = images_a + images_b
        annotations = anns_a + anns_b

        categories = [
            {
                "id": cid,
                "name": name,
                "supercategory": "object",
            }
            for name, cid in class_to_id.items()
        ]

        coco = {
            "info": {
                "description": "Merged RACE-6D dataset"
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        path = (
            output
            / "annotations"
            / f"instances_{split}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(coco, separators=(",", ":")) + "\n"
        )

        all_data[split] = (
            len(images),
            len(annotations),
        )

    merge_models(
        dataset_a,
        dataset_b,
        output,
        categories_a,
        categories_b,
        class_to_id,
    )

    category_yaml = {
        "category2name": {
            cid: name
            for name, cid in class_to_id.items()
        }
    }

    (output / "category2name.yml").write_text(
        yaml.safe_dump(
            category_yaml,
            sort_keys=False,
            allow_unicode=True,
        )
    )

    manifest = {
        "dataset_a": str(dataset_a),
        "dataset_b": str(dataset_b),
        "num_classes": len(class_to_id),
        "classes": {
            cid: name
            for name, cid in class_to_id.items()
        },
        "dataset_a_category_map": category_map_a,
        "dataset_b_category_map": category_map_b,
        "train": {
            "images": all_data["train"][0],
            "annotations": all_data["train"][1],
        },
        "val": {
            "images": all_data["val"][0],
            "annotations": all_data["val"][1],
        },
    }

    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print(f"Classes : {len(class_to_id)}")

    for cid, name in class_to_id.items():
        print(f"  {cid}: {name}")

    print(
        f"\nTrain: {all_data['train'][0]:,} images, "
        f"{all_data['train'][1]:,} annotations"
    )
    print(
        f"Val  : {all_data['val'][0]:,} images, "
        f"{all_data['val'][1]:,} annotations"
    )

    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()