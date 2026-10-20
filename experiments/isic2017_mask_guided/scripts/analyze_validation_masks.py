"""Compare full-image ISIC2017 train/val predictions with official masks.

This is read-only with respect to the dataset and trained checkpoints. It uses
only the unaugmented classification crop for each official validation image.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"D:\Desktop\MALUNet-MB-DCNN-ISIC2017")
CROPS = Path(r"D:\Desktop\MB-DCNN\dataset\classification_crops")
GROUND_TRUTH = Path(r"D:\Desktop\MALUNet-ISIC2017\data\isic2017")
OUTPUT = Path(__file__).resolve().parent / "diagnostics"
CLASS_NAMES = ("nevus", "melanoma", "seborrheic_keratosis")


def summarize(rows):
    return {
        "images": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "median_dice": float(np.median([row["dice"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_predicted_coverage": float(np.mean([row["predicted_coverage"] for row in rows])),
        "mean_ground_truth_coverage": float(np.mean([row["ground_truth_coverage"] for row in rows])),
    }


def analyze_split(split, expected_images):
    with (CROPS / split / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        manifest = list(csv.DictReader(file))
    originals = [row for row in manifest if row["file"] == f'{row["image_id"]}.png']
    if len(originals) != expected_images or len({row["image_id"] for row in originals}) != expected_images:
        raise ValueError(f"Expected {expected_images} unique {split} images, got {len(originals)}")

    rows = []
    for item in originals:
        image_id = item["image_id"]
        prediction_path = ROOT / "predicted_masks" / split / f"{image_id}.png"
        ground_truth_path = GROUND_TRUTH / split / "masks" / f"{image_id}.png"
        if not prediction_path.is_file() or not ground_truth_path.is_file():
            raise FileNotFoundError((prediction_path, ground_truth_path))
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image.convert("L")) > 0
        with Image.open(ground_truth_path) as image:
            ground_truth = np.asarray(
                image.convert("L").resize(prediction.shape[::-1], Image.Resampling.NEAREST)
            ) > 127
        intersection = int(np.logical_and(prediction, ground_truth).sum())
        predicted_pixels = int(prediction.sum())
        ground_truth_pixels = int(ground_truth.sum())
        union = int(np.logical_or(prediction, ground_truth).sum())
        rows.append({
            "image_id": image_id,
            "class": CLASS_NAMES[int(item["label"])],
            "dice": 2 * intersection / (predicted_pixels + ground_truth_pixels)
            if predicted_pixels + ground_truth_pixels else 1.0,
            "iou": intersection / union if union else 1.0,
            "predicted_coverage": predicted_pixels / prediction.size,
            "ground_truth_coverage": ground_truth_pixels / ground_truth.size,
        })

    summary = {"overall": summarize(rows)}
    for class_name in CLASS_NAMES:
        summary[class_name] = summarize([row for row in rows if row["class"] == class_name])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / f"{split}_mask_quality.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / f"{split}_mask_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(split, json.dumps(summary, indent=2))
    print(split, "lowest Dice images:")
    for row in sorted(rows, key=lambda item: item["dice"])[:10]:
        print(row["image_id"], row["class"], f'Dice={row["dice"]:.3f}')
    return summary


def main():
    train = analyze_split("train", 2000)
    val = analyze_split("val", 150)
    print("Mean Dice train minus val:", train["overall"]["mean_dice"] - val["overall"]["mean_dice"])


if __name__ == "__main__":
    main()

