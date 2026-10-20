"""Measure MALUNet mask Dice on all nine ISIC2017 classification val crops.

Official segmentation ground truth is geometrically cropped using the same
centered rectangles as MB-DCNN's validation crop generator. No model inference
or training is performed.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST = Path(r"D:\Desktop\MB-DCNN\dataset\classification_crops\val\manifest.csv")
GROUND_TRUTH = Path(
    r"D:\Desktop\MB-DCNN\dataset\raw\val\ISIC-2017_Validation_Part1_GroundTruth"
)
PREDICTED = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017\predicted_masks\val")
OUTPUT = Path(__file__).resolve().parent / "diagnostics"
CLASS_NAMES = ("nevus", "melanoma", "seborrheic_keratosis")
FRACTIONS = (4 / 5, 3 / 5, 2 / 5, 1 / 5)


def ground_truth_crop(mask, index):
    width, height = mask.size
    if index == 0:
        return mask.resize((224, 224), Image.Resampling.NEAREST)
    fraction = FRACTIONS[(index - 1) % 4]
    if index <= 4:
        crop_width, crop_height = int(fraction * width), int(fraction * height)
    else:
        side = min(width, height)
        crop_width = crop_height = int(fraction * side)
    center_x, center_y = int(width / 2), int(height / 2)
    rectangle = (
        center_x - crop_width / 2,
        center_y - crop_height / 2,
        center_x + crop_width / 2,
        center_y + crop_height / 2,
    )
    return mask.crop(rectangle).resize((224, 224), Image.Resampling.NEAREST)


def mean_metrics(rows):
    return {
        "crops": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "median_dice": float(np.median([row["dice"] for row in rows])),
        "mean_predicted_coverage": float(np.mean([row["predicted_coverage"] for row in rows])),
        "mean_ground_truth_coverage": float(np.mean([row["ground_truth_coverage"] for row in rows])),
    }


def main():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest = list(csv.DictReader(file))
    if len(manifest) != 1350:
        raise ValueError(f"Expected 1350 validation crops, found {len(manifest)}")
    by_image = defaultdict(list)
    for entry in manifest:
        by_image[entry["image_id"]].append(entry)
    if len(by_image) != 150 or any(len(entries) != 9 for entries in by_image.values()):
        raise ValueError("Expected exactly nine crops for each of 150 images")

    rows = []
    for image_id, entries in by_image.items():
        with Image.open(GROUND_TRUTH / f"{image_id}_segmentation.png") as source:
            true_mask = source.convert("L")
            for entry in entries:
                filename = entry["file"]
                if filename == f"{image_id}.png":
                    crop_index = 0
                else:
                    stem = Path(filename).stem
                    prefix = f"{image_id}_9_"
                    if not stem.startswith(prefix):
                        raise ValueError(filename)
                    crop_index = int(stem[len(prefix):])
                with Image.open(PREDICTED / filename) as source_prediction:
                    prediction = np.asarray(source_prediction.convert("L")) > 0
                truth = np.asarray(ground_truth_crop(true_mask, crop_index)) > 127
                intersection = int(np.logical_and(prediction, truth).sum())
                p_area, t_area = int(prediction.sum()), int(truth.sum())
                rows.append({
                    "image_id": image_id,
                    "class": CLASS_NAMES[int(entry["label"])],
                    "crop": crop_index,
                    "dice": 2 * intersection / (p_area + t_area) if p_area + t_area else 1.0,
                    "predicted_coverage": p_area / prediction.size,
                    "ground_truth_coverage": t_area / truth.size,
                })

    summary = {"overall": mean_metrics(rows)}
    summary["by_crop"] = {
        str(crop): mean_metrics([row for row in rows if row["crop"] == crop])
        for crop in range(9)
    }
    summary["by_class"] = {
        name: mean_metrics([row for row in rows if row["class"] == name])
        for name in CLASS_NAMES
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "val_crop_mask_quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "val_crop_mask_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

