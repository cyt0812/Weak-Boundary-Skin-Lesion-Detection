"""Compare OOF and original in-sample predicted masks on train originals.

Uses official ISIC2017 segmentation masks only for this diagnostic. Ground
truth never becomes a classification input. No validation/test labels are read.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT, SEGMENTATION_ROOT
from train_classifier_ablation import MASKS


ROOT = Path(__file__).resolve().parent
OOF_MASKS = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks_oof5_train")
OUTPUT = ROOT / "oof_mask_plan"


def binary_mask(path, size=None):
    with Image.open(path) as source:
        mask = source.convert("L")
        if size is not None:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 127


def dice(prediction, truth):
    intersection = int(np.logical_and(prediction, truth).sum())
    denominator = int(prediction.sum()) + int(truth.sum())
    return 2 * intersection / denominator if denominator else 1.0


def summarize(rows):
    if not rows:
        raise ValueError("No rows to summarize")
    return {
        "images": len(rows),
        "oof_mean_dice": float(np.mean([row["oof_dice"] for row in rows])),
        "in_sample_mean_dice": float(np.mean([row["in_sample_dice"] for row in rows])),
        "mean_oof_minus_in_sample_dice": float(np.mean([row["oof_dice"] - row["in_sample_dice"] for row in rows])),
    }


def main():
    with (PLAN_ROOT / "fold_assignments.csv").open(newline="", encoding="utf-8") as file:
        assignments = list(csv.DictReader(file))
    if len(assignments) != 2000:
        raise ValueError("Expected 2000 fold assignments")
    rows = []
    for item in assignments:
        image_id = item["image_id"]
        predicted_path = OOF_MASKS / f"{image_id}.png"
        original_path = MASKS / "train" / f"{image_id}.png"
        truth_path = SEGMENTATION_ROOT / "train" / "masks" / f"{image_id}.png"
        if not predicted_path.is_file() or not original_path.is_file() or not truth_path.is_file():
            raise FileNotFoundError((predicted_path, original_path, truth_path))
        oof = binary_mask(predicted_path)
        original = binary_mask(original_path)
        if oof.shape != original.shape or oof.shape != (224, 224):
            raise ValueError(f"Wrong predicted mask dimensions for {image_id}")
        truth = binary_mask(truth_path, (224, 224))
        rows.append({
            "image_id": image_id,
            "class_label": int(item["class_label"]),
            "held_out_fold": int(item["held_out_fold"]),
            "oof_dice": dice(oof, truth),
            "in_sample_dice": dice(original, truth),
            "oof_coverage": float(oof.mean()),
            "in_sample_coverage": float(original.mean()),
        })
    by_class = defaultdict(list)
    by_fold = defaultdict(list)
    for row in rows:
        by_class[row["class_label"]].append(row)
        by_fold[row["held_out_fold"]].append(row)
    summary = {
        "overall": summarize(rows),
        "by_class": {str(label): summarize(group) for label, group in sorted(by_class.items())},
        "by_held_out_fold": {str(fold): summarize(group) for fold, group in sorted(by_fold.items())},
        "interpretation_limit": "OOF models train on 1600 instead of 2000 segmentation images. A Dice difference cannot be attributed solely to whether each image was seen during training.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "oof_full_image_quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "oof_full_image_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

