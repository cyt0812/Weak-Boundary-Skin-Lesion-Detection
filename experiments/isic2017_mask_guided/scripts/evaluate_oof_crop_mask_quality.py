"""Compare OOF and in-sample masks on the exact classification-train crops.

Official segmentation ground truth is used only for this quality diagnostic,
never as an input to classifier training or evaluation.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_oof_mask_folds import OUTPUT as PLAN_ROOT
from train_classifier_ablation import CROPS, MASKS


ROOT = Path(__file__).resolve().parent
RAW_GT = Path(r"D:\Desktop\MB-DCNN\dataset\raw\train\ISIC-2017_Training_Part1_GroundTruth")
OOF_MASKS = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks_oof5_train")
OUTPUT = ROOT / "oof_mask_plan"
FRACTIONS = {9: (0.8, 0.6, 0.4, 0.2), 15: (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)}


def index_for_name(image_id, filename, crop_count):
    if filename == f"{image_id}.png":
        return 0
    prefix = f"{image_id}_{crop_count}_"
    stem = Path(filename).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected crop name: {filename}")
    index = int(stem[len(prefix):])
    if index < 1 or index >= crop_count:
        raise ValueError(f"Unexpected crop index: {filename}")
    return index


def crop_ground_truth(mask, index, crop_count):
    if index == 0:
        return np.asarray(mask.resize((224, 224), Image.Resampling.NEAREST)) > 127
    fractions = FRACTIONS[crop_count]
    square = index > len(fractions)
    fraction = fractions[(index - 1) % len(fractions)]
    width, height = mask.size
    if square:
        crop_width = crop_height = int(fraction * min(width, height))
    else:
        crop_width = int(fraction * width)
        crop_height = int(fraction * height)
    center_x, center_y = int(width / 2), int(height / 2)
    rectangle = (
        center_x - crop_width / 2,
        center_y - crop_height / 2,
        center_x + crop_width / 2,
        center_y + crop_height / 2,
    )
    return np.asarray(mask.crop(rectangle).resize((224, 224), Image.Resampling.NEAREST)) > 127


def binary_mask(path):
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L")) > 127
    if mask.shape != (224, 224):
        raise ValueError(f"Expected 224x224 mask: {path}")
    return mask


def dice(prediction, truth):
    denominator = int(prediction.sum()) + int(truth.sum())
    return 2 * int(np.logical_and(prediction, truth).sum()) / denominator if denominator else 1.0


def summary(rows):
    return {
        "count": len(rows),
        "oof_mean_dice": float(np.mean([row["oof_dice"] for row in rows])),
        "in_sample_mean_dice": float(np.mean([row["in_sample_dice"] for row in rows])),
        "mean_oof_minus_in_sample_dice": float(np.mean([row["oof_dice"] - row["in_sample_dice"] for row in rows])),
    }


def main():
    with (CROPS / "train" / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        manifest = list(csv.DictReader(file))
    if len(manifest) != 21768:
        raise ValueError(f"Expected 21,768 train crops, found {len(manifest)}")
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["image_id"]].append(row)
    if len(grouped) != 2000:
        raise ValueError("Expected 2000 train image IDs")

    crop_rows = []
    image_rows = []
    for image_id, entries in grouped.items():
        crop_count = len(entries)
        if crop_count not in FRACTIONS:
            raise ValueError(f"Unexpected crop count for {image_id}: {crop_count}")
        gt_path = RAW_GT / f"{image_id}_segmentation.png"
        with Image.open(gt_path) as source:
            gt = source.convert("L")
            per_image = []
            for entry in entries:
                index = index_for_name(image_id, entry["file"], crop_count)
                truth = crop_ground_truth(gt, index, crop_count)
                oof = binary_mask(OOF_MASKS / entry["file"])
                in_sample = binary_mask(MASKS / "train" / entry["file"])
                row = {
                    "image_id": image_id,
                    "class_label": int(entry["label"]),
                    "crop_index": index,
                    "crop_count": crop_count,
                    "oof_dice": dice(oof, truth),
                    "in_sample_dice": dice(in_sample, truth),
                }
                per_image.append(row)
                crop_rows.append(row)
            if len({row["crop_index"] for row in per_image}) != crop_count:
                raise ValueError(f"Missing/duplicate crop index for {image_id}")
            image_rows.append({
                "image_id": image_id,
                "class_label": int(entries[0]["label"]),
                "crop_count": crop_count,
                "oof_dice": float(np.mean([row["oof_dice"] for row in per_image])),
                "in_sample_dice": float(np.mean([row["in_sample_dice"] for row in per_image])),
            })
    result = {
        "crop_weighted": summary(crop_rows),
        "image_weighted": summary(image_rows),
        "by_class_image_weighted": {
            str(label): summary([row for row in image_rows if row["class_label"] == label])
            for label in range(3)
        },
        "note": "Train has 9 crops per nevus image and 15 for each melanoma/SK image; validation has 9 per image. Compare OOF and in-sample on paired train crops, not raw aggregate train-vs-val crop means as if the crop distributions were identical.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "oof_crop_quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(crop_rows[0]))
        writer.writeheader()
        writer.writerows(crop_rows)
    with (OUTPUT / "oof_image_mean_crop_quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)
    (OUTPUT / "oof_crop_quality_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

