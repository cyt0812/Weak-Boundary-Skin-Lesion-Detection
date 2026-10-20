"""Audit and prepare image-level ISIC2017 training folds for OOF masks.

This only writes a fold manifest and audit in the workspace. It neither trains
models nor reads/writes original checkpoints or predicted-mask directories.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
CLASSIFICATION_MANIFEST = Path(r"D:\Desktop\MB-DCNN\dataset\classification_crops\train\manifest.csv")
SEGMENTATION_ROOT = Path(r"D:\Desktop\MALUNet-ISIC2017\data\isic2017")
OUTPUT = ROOT / "oof_mask_plan"
FOLDS = 5
SEED = 42


def read_classification_ids():
    by_image = defaultdict(list)
    with CLASSIFICATION_MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            by_image[row["image_id"]].append(row)
    if len(by_image) != 2000:
        raise ValueError(f"Expected 2000 unique classification-train IDs, found {len(by_image)}")
    labels = {}
    for image_id, crops in by_image.items():
        label_set = {int(row["label"]) for row in crops}
        names = {row["file"] for row in crops}
        if len(label_set) != 1:
            raise ValueError(f"Bad crop grouping for {image_id}")
        label = label_set.pop()
        count = 9 if label == 0 else 15
        expected_names = {f"{image_id}.png"} | {
            f"{image_id}_{count}_{index}.png" for index in range(1, count)
        }
        if len(crops) != count or names != expected_names:
            raise ValueError(f"Bad crop grouping for {image_id}")
        labels[image_id] = label
    if set(labels.values()) != {0, 1, 2}:
        raise ValueError("Expected all three ISIC2017 classes")
    return labels


def image_ids(split, kind):
    directory = SEGMENTATION_ROOT / split / kind
    return {path.stem for path in directory.glob("ISIC_*.png")}


def main():
    labels = read_classification_ids()
    train_images = image_ids("train", "images")
    train_masks = image_ids("train", "masks")
    val_images = image_ids("val", "images")
    val_masks = image_ids("val", "masks")
    if set(labels) != train_images or train_images != train_masks:
        raise ValueError("Classification-train IDs and segmentation train image/mask IDs differ")
    if len(val_images) != 150 or val_images != val_masks:
        raise ValueError("Expected 150 paired segmentation validation images")
    if train_images & val_images:
        raise ValueError("Segmentation train and validation image IDs overlap")

    ids = np.asarray(sorted(labels))
    y = np.asarray([labels[image_id] for image_id in ids])
    assignments = {}
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_indices, held_out_indices) in enumerate(splitter.split(ids, y)):
        train_ids = set(ids[train_indices])
        held_out_ids = set(ids[held_out_indices])
        if train_ids & held_out_ids or train_ids | held_out_ids != set(labels):
            raise ValueError(f"Fold {fold} has overlapping or missing IDs")
        for index in held_out_indices:
            image_id = str(ids[index])
            if image_id in assignments:
                raise ValueError(f"Image {image_id} assigned twice")
            assignments[image_id] = fold
    if len(assignments) != 2000:
        raise ValueError("Fold assignment is incomplete")

    summary = {
        "purpose": "OOF training-mask plan only; no training has been run",
        "folds": FOLDS,
        "seed": SEED,
        "image_ids": len(assignments),
        "classification_crops": sum(9 if label == 0 else 15 for label in labels.values()),
        "validation_ids_reserved": len(val_images),
        "validation_id_overlap": len(set(assignments) & val_images),
        "total_by_class": dict(sorted(Counter(labels.values()).items())),
        "folds_detail": [],
        "selection_rule": "Each fold model trains on the other four folds; never use its own held-out fold for checkpoint selection. Existing 150-image official validation may be used for segmentation checkpoint selection, but has already been used in prior classification model selection; all downstream validation comparisons are exploratory.",
        "limitation": "Image-ID separation only; no lesion-group metadata is present in these local manifests.",
    }
    for fold in range(FOLDS):
        held_out = [image_id for image_id, assigned in assignments.items() if assigned == fold]
        train = [image_id for image_id, assigned in assignments.items() if assigned != fold]
        summary["folds_detail"].append({
            "fold": fold,
            "segmentation_train_images": len(train),
            "held_out_images_for_prediction": len(held_out),
            "held_out_class_counts": dict(sorted(Counter(labels[image_id] for image_id in held_out).items())),
            "held_out_classification_crops": sum(9 if labels[image_id] == 0 else 15 for image_id in held_out),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("image_id", "class_label", "held_out_fold"))
        writer.writeheader()
        for image_id in sorted(assignments):
            writer.writerow({
                "image_id": image_id,
                "class_label": labels[image_id],
                "held_out_fold": assignments[image_id],
            })
    (OUTPUT / "fold_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

